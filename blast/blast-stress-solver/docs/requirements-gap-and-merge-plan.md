# blast-stress-solver — requirements gap analysis & merge plan

Target spec: *"blast-stress-solver — destruction + the structure graph"*: the
authoritative source of structural semantics (chunks, bonds, supports) with a
wasm32 in-browser build, deterministic seeded runs, a programmatic damage API,
JS/TS structure-graph queries, a <5 ms counterfactual detachment query,
diff-carrying events (bond-broken / chunk-detached / settled), chunk-geometry
export for a navmesh rasterizer, and a single transform-synced Rapier world.

Snapshot date: 2026-07-05, `main` @ `0b4271aa`.

---

## 1. Where `main` stands against the requirements

| # | Requirement | Status on `main` | Evidence |
|---|---|---|---|
| 1 | wasm32 in-browser; solver step ≤4 ms @ ~200 chunks | **PARTIAL** | TS/Emscripten stack is real and deployed (`src/stress.ts`, demos). At 200 chunks the CGNR solve is sub-ms (0.1 ms at 1,344 nodes on `bench:large`), but there is **no committed 200-chunk/4 ms benchmark**, and frame cost is dominated by `rapierStep`, not the solver. The **Rust crate** builds `wasm32-unknown-unknown` first-class (CI `wasm-smoke`, zero `env.*`/WASI imports, prebuilt backend in the published crate) but ships **no JS bindings and no browser demo** — consumers must write their own `wasm-bindgen` glue. |
| 2 | Deterministic under seed → identical end-state | **PARTIAL** | Deterministic by construction (no RNG anywhere in `src/`); identical-run bond-count test (`rapier.headless-scenarios.test.ts:931`), `compareTrajectories` with `maxPosDelta === 0`, Rust↔JS cross-validation fixtures. **Missing: an end-state hash test** ("hash identical across 5 seeded runs"). Variable-dt per-rAF stepping also makes browser end-states display-rate-dependent until PR #59 lands. |
| 3 | Programmatic damage API (point / named bonds) | **PARTIAL** | `applyNodeDamage`, `applyExternalForce(nodeIndex, worldPoint, worldForce)`, `cutBond(bondIndex)`, `cutNodeBonds`, solver `addForce`/`addAllForces` — all JS-callable. PR **#26** upgrades `applyExternalForce` into a first-class primitive (solver + physics + resim in one tick). **Missing: world-point → node resolver** (caller must raycast + `colliderToNode` themselves) and any name-keyed addressing (bonds are numeric indices). |
| 4 | Structure-graph queries (chunks, bonds, bond health, supports) | **PARTIAL** | `chunks: ChunkData[]` (incl. `isSupport`, transforms, node health), `getNodeBonds() → BondRef[]`, `getActiveBondsCount`, actors, `colliderToNode`. **Missing: per-bond health/stress query** — bond health only appears transiently in fracture commands and debug-render lines. |
| 5 | Counterfactual query (<5 ms) | **MISSING** | No what-if API. Building blocks exist: `islandFind()`/`rebuildIslandComponents()` union-find over the live bond graph with supports as cut points (`destructible-core.ts:2960+`), and the non-mutating damage `previewTick`. |
| 6 | Events with diffs (bond-broken / chunk-detached / settled) | **PARTIAL** | `onImpact`, `onNodeDestroyed`, `onWorldReplaced`; `SplitEvent[]` exists internally but isn't surfaced. "Settled" exists only as island-sleep instrumentation (`getIslandSettledStats`), not an event. No diff payloads (changed chunks, final transforms, freed AABBs). |
| 7 | Chunk geometry export for navmesh | **PARTIAL** | Per-chunk geometry lives in `scenario.parameters.fragmentGeometries` (three.js `BufferGeometry`); no engine-neutral `{vertices, indices}` world-space export keyed by live chunk. |
| 8 | Same Rapier world, one WASM module ideally | **PARTIAL** | The core **creates and owns** its Rapier world (callers can't inject one) but is the single source of truth for chunk transforms — the render adapter reads its snapshots. Module topology today is **two** WASM modules: Emscripten `stress_solver.wasm` + `@dimforge/rapier3d-compat`'s wasm. The literal "one WASM module" reading is only reachable via the Rust crate (rapier as a cargo feature compiles into the same module). |

**"Done when" checklist:** wasm32 build passes ✅ (both stacks, CI-gated) · 100+ chunk collapse at 60 fps in Chrome ❓ (almost certainly true, never asserted) · end-state hash identical across 5 seeded runs ❌ (no hash test) · settled event with correct diff ❌ · counterfactual <5 ms ❌ · JS API documented ⚠️ (README + rich JSDoc on `types.ts`; no generated reference).

---

## 2. Un-merged work: what to land (and what to close)

29 open PRs analyzed; merge-cleanliness verified locally with `git merge-tree`
against `main` **and** stacked sequentially. Key mechanical fact: the perf PRs
all touch `src/rapier/destructible-core.ts` and **conflict with each other**,
so they must land in sequence with rebases between — order below.

### Tier A — requirement-critical, merge first

| PR | What | Why it matters here | Merge state |
|---|---|---|---|
| **#58** tier1-perf-defaults | Island-aware solve **default-on** (13.1 → 1.37 ms on the 36-tower bench), lazy-collider external movers, plus a **build fix** (`build:web` silently corrupted `dist/stress_solver.cjs`) | Req 1 solver budget; the build fix is a latent correctness bug for anyone consuming `dist/` | clean vs main |
| **#59** fixed-step-loop | Fixed-timestep driver + render interpolation | Prerequisite for req 2's cross-machine reproducibility (removes display-rate-dependent behaviour) | clean; stacks cleanly on #58 |
| **#48** city-stress harness | Measurement-only stress harness, A/B island parity, seeded deterministic city builders | The infrastructure to *prove* the "done when" criteria | clean |
| **#26** external-impact-api | `applyExternalForce` routed through solver + physics + resim in one tick | The core of req 3 (NPC Breach operator needs exactly this) | **needs rebase** (2 conflicting files; 128 commits behind) |
| **#72** GLB→CoACD pipeline → then **#74** render≠collision | Offline decomposition pipeline; per-node `colliderGeometries` channel separate from render geometry | Req 7 groundwork (geometry channels per chunk); both CI-green | both clean (#74 is based on #72's branch — merge #72 first) |

### Tier B — perf stack (roadmap Tiers 1–3); merge in this order, rebasing each onto the previous

`#60 splash-in-wasm → #61 alloc-free-hotpath → #63 bondstress-settled-skip →
#64 skip-irrelevant-contacts → #65 contact-hit-dedup (opt-in) →
#66 debris-settle-freeze → #57 resim-island-exact → #40 wasm-simd128 →
#62 worker-physics (optional, opt-in)`

All are bit-exact or opt-in per their PR bodies with parity suites. #66 also
introduces the settled-lifecycle notion that req 6's `settled` event should
build on. #62 has one trivial `package.json` conflict. Re-run CI after each
rebase — the 2026-06-12 runs have aged out.

### Tier C — demo/showcase PRs (clean, optional, merge as desired)

#75 structure presets (useful as ~200-chunk acceptance scenarios), #69 brick
castle, #68 keystone arch, #67 flywheel (Rust example), #70 mini-city building
types. All merge clean vs `main` individually; may need trivial rebases
against each other.

### Close as superseded / stale

| PR / branch | Reason |
|---|---|
| **#71** | Strict subset of #72 (its commits are contained in the `glb-coacd-destructible-pipeline` branch) |
| **#27, #37** | Superseded by #40 (roadmap Tier 0 says so explicitly; LTO/assertion flags measured ±1 % kernel) |
| **#23** | Self-described as superseded |
| **#1, #3, #4** | April-era, based on `feat/rapier-destruction` (249 behind). #4's wasm-source-build goal has since been achieved properly on `main` (build.rs wasm32 support + shims + CI smoke test). #3's invariant-test ideas are good but the code is stale — reimplement if wanted |
| **#47** | rapier.mjs→rapier.js import rewrite + `init?.()`; likely obsolete — verify against current demos, else close |
| **#43** | Debug-line-follow-fragments fix, 23 behind; review whether still applicable after subsequent refactors, rebase or close |
| branch `claude/resim-island-scoping` | Superseded by #57 |
| branch `claude/bond-reorder-locality` | Negative-result experiment, already recorded in the roadmap |
| branches `claude/nice-euler-I2Odj` (residual), `claude/bond-quality-fix`, `claude/jolly-dirac-gtZzA` | Redone/superseded by #72/#74 and #63/#66 respectively — verify then delete |

### Un-merged branch worth resurrecting

**`claude/web-rust-library-parity-1ihRV`** (no open PR, 98 behind, 3 Rust-file
conflicts): ports island-aware solving, production contact injection, and the
per-body splash grid from the web stack into the Rust crate. This is the seed
of the req-1/req-8 "Rust crate in-browser, one WASM module" track — rebase and
open a PR when Phase 2 (below) starts.

Also: `feat/rapier-destruction` carries two commits of local dev-setup scripts
(merged to the wrong base via #73) — cherry-pick onto `main` if wanted, then
retire the branch.

---

## 3. Plan for what remains

### Phase 0 — land the backlog (≈ days, mostly mechanical)

Execute §2. Outcome: island solve + fixed timestep + SIMD kernels by default,
damage primitive (#26), measurement harness (#48), geometry channels (#72/#74).
This alone resolves most of req 1 and the biggest risk to req 2.

### Phase 1 — close the semantic gaps (the real feature work)

All items are TS/Emscripten-stack first, but should be shaped FFI-friendly
(flat arrays, index-keyed) so they port to the Rust crate in Phase 2.

1. **Counterfactual query (req 5, MISSING).**
   `previewDetachment({ removeNodes?, removeBonds?, charge? }) →
   { detachedChunks, detachedMass, supportLost, wouldCollapse, freedAABBs, cost }`
   — union-find over the bond CSR adjacency *excluding* the hypothetical
   removals, with support (mass-0) nodes as anchors; components not reaching a
   support are "detached". `charge` maps to a removal predicate via the
   existing damage falloff (`damage.ts`). Non-mutating, no solver call.
   `islandFind()` already does this shape of work per-frame at 9k nodes, so
   <5 ms at breach scale is comfortable. Add a micro-bench asserting <5 ms and
   correctness tests against actual post-damage outcomes.
2. **Event stream with diffs (req 6).**
   Core emitter with `onBondBroken({bondIndex, node0, node1, cause})`,
   `onChunkDetached({nodeIndex, actorIndex, bodyHandle, transform})`,
   `onSettled({chunks, finalTransforms, freedAABBs})`. Wire bond-broken from
   `applyFractureCommands`'s fracture list, chunk-detached from
   `processSplitEvents` (the `SplitEvent[]` already exists internally), and
   settled from island sleep-edge transitions (`islandFind` + `isSleeping`,
   plus #66's freeze lifecycle once merged). Freed AABBs from chunk collider
   AABBs at settle time. Acceptance: headless test that a collapse fires
   `settled` exactly once with a correct diff.
3. **Graph-query completion (req 4).** Expose per-bond health/stress:
   add `getBondHealth(bondIndex)` / extend `BondRef` with `health` (+ optional
   `stress` from the solver's bond-stress buffer the debug-render path already
   reads). Optional string-name registry for chunks/bonds as a thin TS layer.
4. **Damage-API completion (req 3).** World-point resolver
   (`resolvePoint(worldPoint) → nodeIndex` via Rapier ray/proximity +
   `colliderToNode`) and a convenience `applyDamageAtPoint(worldPoint, radius,
   magnitude)` composing #26's primitive; `weakenBond(bondIndex, amount)`
   alongside the existing `cutBond`.
5. **Geometry export (req 7).** Engine-neutral
   `getChunkGeometry(nodeIndex) → { vertices: Float32Array, indices: Uint32Array }`
   (local space) + `chunk.worldPosition/worldQuaternion` for placement, plus a
   baked world-space batch variant for the navmesh rasterizer. Feed from
   `fragmentGeometries`/`colliderGeometries`; keep it three.js-free in the
   rapier layer.
6. **Determinism acceptance (req 2).** Canonical end-state serialization
   (alive-bond set + chunk states + float-bit-exact transforms) → SHA-256
   hash; CI test running one seeded damage script 5× and asserting identical
   hashes (build on #48's seeded city builders). Document the determinism
   envelope: same build + same flags; note that #65's dedup and any SIMD
   reassociation are opt-in for this reason.
7. **Perf acceptance (req 1).** A committed ~200-chunk scenario benchmark
   asserting solver step ≤4 ms (post-#58 this should pass with large margin),
   plus a Chrome acceptance test (Playwright, headed Chromium) asserting a
   100+ chunk collapse holds 60 fps.
8. **JS API docs.** Generate a typedoc reference from `src/rapier/types.ts` et
   al., and add a "structure graph & breach queries" README section covering
   the new APIs (events, counterfactual, geometry export).

### Phase 2 — architecture (req 8)

Two sequenced steps, not a fork:

- **2a (short-term, TS stack):** accept an optional caller-provided Rapier
  `world` in `buildDestructibleCore` so game sim and destruction share one
  world. Two WASM modules remain (Emscripten solver + rapier3d-compat) — the
  "one source of truth for chunk transforms" constraint is already satisfied.
- **2b (strategic, matches the requirement text literally):** a
  `blast-stress-solver-web` wasm-bindgen wrapper crate over
  `blast-stress-solver-rs` (features `rapier`, `scenarios`) exposing the
  Phase-1 API surface to JS/TS — one wasm module containing solver + Rapier.
  Seed with the rebased `claude/web-rust-library-parity-1ihRV` branch; reuse
  the JS↔Rust cross-validation fixtures as the parity gate; ship TS type
  definitions and run the same Chrome/determinism/counterfactual acceptance
  suite against it.

### Acceptance mapping

| "Done when" | Covered by |
|---|---|
| wasm32 build passes | already green (keep CI) |
| 100+ chunk collapse @60 fps Chrome | Phase 1.7 |
| end-state hash identical ×5 seeded runs | Phase 1.6 (+ #59 for cross-display) |
| settled event with correct diff | Phase 1.2 |
| counterfactual <5 ms | Phase 1.1 |
| JS API documented | Phase 1.8 |
