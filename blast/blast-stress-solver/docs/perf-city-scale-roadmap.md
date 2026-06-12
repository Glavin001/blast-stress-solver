# City-scale real-time performance — analysis & roadmap

Goal: **60–120 FPS on commodity web hardware** for everything from a single
high-detail destructible (car, house, bridge, high-rise) up to a mini city of
fractured buildings, **with zero quality degradation** — no reduced solver
iterations, no smaller worlds, no coarser fracture. The reference experiences
are Red Faction: Guerrilla and The Finals; this library is the building block
for fast-paced real-time destruction games.

Everything below is grounded in measurements taken on this branch (Node 22,
x64 container, default scalar WASM build unless noted), plus the committed
mini-city session recording analysed in
[`perf-resim-findings.md`](perf-resim-findings.md). Each proposal names its
**quality guard** — the test or harness that proves output is preserved.

---

## 1. Where the time goes today

### Full pipeline, large tower (`npm run bench:large` — 1 344 nodes / 8 112 bonds, 360 frames)

| phase | mean ms | p95 ms | notes |
|---|---:|---:|---|
| **total frame** | **19.8** | **27.0** | 290/360 frames blow the 16.6 ms budget |
| `rapierStepMs` | 8.9 | 13.6 | #1 cost — scales super-linearly with debris |
| `solverUpdateMs` | 5.2 | 7.9 | of which ↓ |
| — contact inject (JS) | 4.6 | 6.8 | **splash loop 3.4 ms** + resolve 1.1 ms |
| — WASM CGNR solve | 0.1 | 0.1 | small here; 3.7 ms on the real mini-city recording |
| `contactDrainMs` | 2.4 | 4.7 | per-event JS callbacks + object allocs |
| `snapshotCaptureMs` | 1.9 | 3.5 | **every frame**, all dynamic bodies (resim rollback) |
| `resimMs` | 0.6 | — | max 213 ms; ~2× frame cost whenever fracture fires |

The mini-city recording (709 frames, 1 355 bodies) agrees: `rapierStep` 45.6 %,
solver 28.2 % (CGNR 3.7 ms mean at 8 152 nodes / 21 769 bonds), resim on 40 %
of frames ≈ 28 % of wall time, 51 % of frames missing 60 FPS.

### Stress solve at city scale (`node scripts/bench-solver-city.mjs` — 36 towers, 9 216 nodes / 22 464 bonds)

| mode | steady (settled) | excited (1 impact/frame) | islands skipped |
|---|---:|---:|---:|
| whole-graph (today's default) | 1.41 ms | **13.10 ms** (p95 16.0) | 0/36 |
| island-aware + skip-settled | 1.07 ms | **1.37 ms** (p95 1.4) | 35/36 |

**One localized impact makes the whole-graph CGNR re-solve all 36 towers —
13 ms/frame. The island-aware path (already in the C++, default-off) solves
only the struck tower: ~10× cheaper, identical result per island.**

### Broadphase at city scale (`node scripts/lod-bench.mjs` — 8×8 city, 9 216 fragments)

| mode | idle rapierStep | post-hit rapierStep | peak active fragments |
|---|---:|---:|---:|
| eager colliders | 2.50 ms | 9.39 ms | 9 216 |
| lazy hierarchical (this branch) | 0.092 ms (**27×**) | 6.73 ms (1.39×) | 2 682 |

### Renderer instance sync (`node scripts/bench-batched-mesh.mjs` — 27 539 instances)

| BatchedMesh per-draw CPU work | ms/call |
|---|---:|
| sort + frustum cull (THREE default) | 6.13 |
| frustum cull only (this branch: `sortObjects=false`) | 1.74 |
| neither | 0.00 |
| all instances proxied/hidden | 0.06 |

### WASM build flags (measured, corrected for `prebuild-runtime.js` artifact caching)

`-sASSERTIONS=1` (today's default) vs assertions-off + `-flto`: **kernel time
unchanged within ±1 %** — the CGNR hot loop is memory-bound, not check-bound —
binary 330 436 → 312 798 B (−5.3 %). The flag cleanups (PR #27/#40) are worth
landing for size and hygiene, but they are not where the speed is.

### Structural facts that shape the roadmap

- Every demo steps physics **once per rAF with variable dt** (clamped to 1/30,
  e.g. `mini-city.ts:791`). On a 120 Hz display that is 120 physics steps/s —
  double the work for zero visual benefit — and frame-rate-dependent behaviour.
- `captureWorldSnapshot()` runs **every frame** when `resimulateOnFracture` is
  on (default), walking **all** dynamic bodies including long-asleep debris
  (`bodySnapshots.ts:28`).
- The island-aware solve, lazy intact colliders, and intact-proxy render LOD
  are **only wired up in mini-city**; high-rise, house, vehicle, etc. run with
  the whole-graph solve even though debris islands dominate after a collapse.
- Rendering and physics share the main thread; a 25 ms physics spike is a
  dropped render frame regardless of GPU headroom.

---

## 2. Already landed on this branch (keep)

- BatchedMesh **unchanged-instance skip** (`destructible-adapter.ts`,
  `BatchedSyncCache`) — settled chunks cost ~0 in render sync.
- **`sortObjects=false`** for the opaque chunk batch — −4.4 ms/frame at 27 k
  instances (table above).
- **Intact-building proxy LOD** (`three/intact-proxy.ts`) — distant un-hit
  buildings render as one box instance each, with hysteresis.
- **Hierarchical lazy intact colliders** + authored `collisionTree`
  (`rapier/collisionTree.ts`) — the 27× idle broadphase win.
- Chunk world-transform recompute skip for unchanged bodies
  (`destructible-core.ts:2843`).

---

## 3. Roadmap

Ordered by (impact × confidence) ÷ effort. "Bit-exact" means provably
identical output (guarded by `compareTrajectories` / parity tests); "physically
identical" means same algorithm and iteration counts but float-level
reassociation may differ.

### Tier 0 — land the open work that already exists

| item | impact | quality |
|---|---|---|
| **Merge this branch** (#53 + intact proxies + sort drop) | render sync ~0 idle; −4.4 ms/draw at 27 k instances | pinned by `three.adapter.test.ts`, `intact-proxy.test.ts` |
| **Land PR #40** (direct wasm-simd128 CGNR kernels + build flags; supersedes #37/#27) | **+24 % solver** measured on mini-city scale in that PR | `simd.parity.test.ts` gates scalar↔SIMD convergence |
| **Rebase & land PR #48** (city-scale stress harness) | measurement infra: p50/p95/p99 per phase on 4 city presets, A/B parity validation | measurement-only |

### Tier 1 — configuration & dependency wins (days, low risk)

1. **Swap to `@dimforge/rapier3d-simd-compat`** (same 0.19.x, drop-in).
   Measured here on `bench:large`: `rapierStep` mean 8.85 → 7.07 ms (**−20 %**),
   p95 13.13 → 9.92 ms (−24 %); whole frame −8.4 % mean, −12 % p95. Attacks the
   #1 phase in every scenario. *Physically identical* (same engine, same
   iterations; SIMD float reassociation) — re-baseline the determinism
   fixtures once. Ship as the default, keep the scalar package behind a flag.
2. **Island-aware solve + skip-settled ON by default** for every core
   (`destructible-core.ts:485` currently `false`; only mini-city flips it).
   Measured: 13.1 → 1.37 ms on the excited city; also wins on single
   structures the moment debris splits off (every detached cluster is an
   island, and settled clusters skip). Guard: the island A/B parity harness
   (PR #48) + `rapier.headless-scenarios` determinism suite. Expose the toggle
   in the shared `physics-controls.ts` so all demos get it.
3. **Fixed-timestep driver with render interpolation.** Add a
   `createFixedStepLoop(core, { hz: 60 })` helper: accumulator → N×`step(1/60)`,
   renderer lerps chunk poses between the last two physics states (the
   `BatchedSyncCache` already tracks per-instance poses — extend it to keep
   prev/cur). Benefits: 120 Hz displays stop paying 2× physics; identical
   simulation regardless of display; spiral-of-death protection in one place.
   *This improves physics quality* (removes frame-rate dependence).
4. **Lazy intact colliders default-on** for multi-building scenarios (now that
   the predictive-enable pass makes it output-identical — see
   [`lazy-intact-colliders.md`](lazy-intact-colliders.md)).

### Tier 2 — pipeline restructuring (output-identical, ~1–2 weeks each)

1. **Awake-only incremental rollback snapshot.** Today
   `captureDynamicBodySnapshots` copies every dynamic body every frame
   (1.9 ms at 1.3 k bodies; grows linearly — ≈7 ms at a 5 k-body city). A body
   asleep at capture time cannot move during the step, so its previous
   snapshot is still valid: keep a persistent per-body snapshot, refresh only
   awake bodies, and on restore also reset any body the failed step woke
   (wake-ups are detectable post-step). Capture becomes O(awake) instead of
   O(all). **Bit-exact**; guard: `rapier.resim-continuity.test.ts` +
   equivalence harness `maxPosDelta === 0`.
2. **Move splash-force expansion into WASM.** The JS splash loop
   (`destructible-core.ts:1763`) walks the precomputed adjacency and pushes
   per-neighbour forces into the batch buffer — 3.4 ms mean / 5.1 ms p95
   during impacts on `bench:large`, comparable to the entire WASM CGNR solve
   on the mini-city recording (3.7 ms).
   Ship the static splash adjacency (CSR + weights) to the WASM side once at
   build, then submit only the per-frame hit list (node, body-local force,
   body-membership epoch); C++ expands hits → neighbours in the same order JS
   does today. Est. `solverContactInjectMs` 4.6 → ~1 ms. **Bit-exact if the
   accumulation order is replicated**; guard: `bondStress.parity.test.ts`.
3. **Allocation-free contact drain.** `drainContactForces` allocates one
   object per contact per frame into `bufferedExternalContacts` (plus Vec3s);
   impact storms generate thousands → GC pauses on exactly the worst frames.
   Replace with pre-allocated growable SoA typed arrays (the force-batch
   buffers already follow this pattern). Same for the per-chunk
   `worldPosition`/`worldQuaternion` fresh objects (`destructible-core.ts:2880`)
   — mutate in place. **Bit-exact**; guard: existing integration suite.
   **Status: implemented — and re-scoped by measurement.** `--trace-gc` over
   bench:large shows ~2.1 GB of heap churn per 360-frame run on BOTH sides:
   the dominant allocator is **Rapier's JS binding temporaries** (every
   `translation()`/`rotation()`/`linvel()` read returns a fresh object —
   tens of thousands per frame across snapshot capture, contact drain, and
   transform readout), not our buffers. Mark-Compact counts dropped (11 → 6)
   but pause time is noise-level. The real Tier-2 GC lever is therefore
   **batched/raw body reads** (e.g. reading body poses straight from Rapier's
   WASM heap once per frame into a flat buffer) — tracked as follow-up.
4. **Scoped resim, island-exact (second attempt).** PR #41 froze non-affected
   bodies and diverged 12.8 m on cascades because contact-*loss* coupling was
   missed. The exact formulation: freeze only entire **Rapier islands** that
   contain no fracture-affected body, where affected = fracture seeds ∪
   contact closure *from the initial pass's contact graph* (resting contacts
   capture support-loss; contact-gain is covered by Rapier auto-wake + CCD).
   Constraint islands are independent within a step, so an untouched island
   re-integrates identically — freezing it at the initial-pass result is
   provably the same output. If cascades still defeat the proof, keep it
   opt-in. Prize: resim is ~28 % of wall time on the recording (fracture
   frames cost ~2×). Guard: equivalence harness `maxPosDelta === 0`
   **including multi-impact cascade scenarios** (the case that killed #41).

### Tier 3 — architecture for 120 FPS and bigger cities (weeks)

1. **Physics in a Web Worker (SharedArrayBuffer transform ring).** Move
   `DestructibleCore` (Rapier WASM + stress WASM) off the main thread; publish
   chunk poses into a SAB double-buffer the adapter reads; inputs (shots,
   explosions) go through a command queue. The main thread then renders at
   display rate and interpolates, completely insulated from physics spikes;
   pairs naturally with the Tier-1 fixed-timestep driver. Needs COOP/COEP
   headers (add to `serve-demo.mjs` + Vercel config). The adapter is already
   decoupled enough: it consumes `chunk.worldPosition/worldQuaternion`
   snapshots, not live Rapier handles.
2. **WASM threads for per-island solves** (after #40 and COOP/COEP): a city is
   hundreds of independent islands; solve them on a small thread pool inside
   one `update()`. Near-linear scaling on the "many things happening at once"
   case that single-island skipping doesn't cover. Guard: per-island results
   are independent, so parity is exact per island.
3. **Renderer, city-scale fidelity:**
   - **Shadow-map update gating**: `renderer.shadowMap.autoUpdate = false`;
     set `needsUpdate = true` only when any chunk pose changed (the
     `BatchedSyncCache` knows) or the light/camera moved. A settled city stops
     paying the full shadow depth pass every frame.
   - **Cascaded shadow maps** (three CSM or manual 2-cascade) so up-close
     fidelity survives city-sized shadow frustums.
   - **Damaged-building proxies**: extend intact proxies with a "scarred"
     impostor (the building's current silhouette baked to a low-poly shell +
     normal map once, refreshed on fracture) so even *hit* buildings collapse
     to one draw at distance. Today a single hit permanently un-proxies a
     building.
   - **Debris shadow policy**: small debris stops casting shadows beyond a
     distance threshold (screen-space imperceptible; render-only, no physics
     change).
4. **Startup at city scale:** extend `scenePackLoader` to carry fragment
   geometry + bonds + `collisionTree` (full offline bake), and run three-pinata
   fracturing in a worker pool at build time. The variant cache in mini-city
   already cut startup ~5–10×; baking removes it from the client entirely.

### Tier 4 — solver research (only if Tiers 0–3 leave a gap)

- SoA/blocked memory layout for the CGNR kernels — **note the negative
  result in PR #39** (bond/node reordering for cache locality didn't pay);
  any retry must be measured against `bench-solver-city.mjs` + PR #40's
  `bench-replay.mjs`.
- Preconditioning (block-Jacobi) would cut iterations but **changes the
  iteration trajectory** → not output-preserving; out of scope per the brief.

---

## 4. Quality-preservation protocol (applies to every item)

1. Implement behind a flag, default-off until proven.
2. **Faithfulness**: `compareTrajectories(before, after)` from
   `rapier.resim-perf.test.ts` — `maxPosDelta === 0` for bit-exact items;
   documented epsilon + visual A/B for "physically identical" items (Rapier
   SIMD swap, sleep tuning). Parity suites: `simd.parity.test.ts`,
   `bondStress.parity.test.ts`, island A/B (PR #48).
3. **Speed**: phase deltas via `scripts/bench.mjs`, `bench-solver-city.mjs`,
   `lod-bench.mjs`, `bench-batched-mesh.mjs`, and a re-recorded mini-city
   session through `recordingAnalysis.ts`.
4. No reductions in: solver iterations (`maxSolverIterationsPerFrame` stays
   24), Rapier solver iterations, resim passes, fracture resolution, world
   size, collision pairs. `graphReductionLevel` stays 0.

---

## 5. Projected frame budget (mini-city, action-heavy, commodity laptop)

| stage | action-frame estimate | basis |
|---|---:|---|
| today (recording) | ~19.5 ms mean, 51 % frames > 16.6 ms | `perf-resim-findings.md` |
| + Tier 1 (Rapier SIMD, island solve, fixed 60 Hz step) | ~11–13 ms | −1.8 ms rapierStep, −2…−11 ms solver, physics ≤ 60 Hz |
| + Tier 2 (snapshot, splash→WASM, SoA drain, scoped resim) | ~7–9 ms | −1.5, −2.5, −1, −3 (fracture frames) |
| + Tier 3 (worker + interpolation, shadow gating) | render thread ≈ GPU-bound only | physics off-thread |

That puts a mini city with active destruction inside the 60 FPS budget with
headroom on the main thread, and the worker + interpolation architecture is
what unlocks honest 120 FPS rendering on capable displays — without touching
a single physics iteration.

## 6. Reproduce

```bash
cd blast/blast-stress-solver
npm run build                                # WASM + TS (needs emsdk)
npm run bench:large                          # full pipeline, per-phase JSON
node scripts/bench-solver-city.mjs           # stress solve at city scale (this doc §1)
node scripts/lod-bench.mjs                   # eager vs lazy collider LOD
node scripts/bench-batched-mesh.mjs          # BatchedMesh sort/cull cost
# Rapier SIMD A/B: npm i @dimforge/rapier3d-simd-compat@0.19.1 --no-save, then
# symlink it over node_modules/@dimforge/rapier3d-compat and re-run bench:large.
```
