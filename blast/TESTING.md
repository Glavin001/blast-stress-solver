# Testing & observability for the Blast stress/destruction libraries

This document is the single source of truth for how we test the two destruction
libraries — `blast-stress-solver` (JS/TS) and `blast-stress-solver-rs` (Rust) — and,
just as importantly, **a living map of what we do _not_ yet cover** (see
[Coverage gaps](#coverage-gaps)).

## Philosophy

- **Invariants are physical laws, not magic numbers.** Mass conservation, no-NaN, and
  point-velocity continuity across a fracture are true regardless of implementation. A
  performance optimization that violates one is a real bug, so these tests can't be
  "cheated" — that is exactly why they're the backbone.
- **Tiered by cost, so iteration stays fast.** Pure kernels run in milliseconds on every
  change; tiny-scenario invariants run sub-second; full scenarios + determinism run in
  CI; browser/visual runs pre-release.
- **Smallest scenario that triggers the failure.** A 2-node split localizes a bug to one
  node/body with an exact magnitude — a root-cause pointer, not "the demo looks wrong".
- **No false confidence.** Tight invariants are blocking in CI. Genuinely flaky/stale
  tests are explicitly skipped _with a tracked reason_ and listed below — never masked
  with `|| true` or `continue-on-error`.
- **Zero release cost.** The Rust per-split continuity check is observability gated behind
  `DestructibleSet::set_record_split_continuity(true)`; it is off by default, so production
  behavior and cost are unchanged.

## Tiers and how to run them

| Tier | What | JS | Rust |
|---|---|---|---|
| **0 — Kernels** | Pure math / planner property tests (no physics runtime) | `src/tests/kernels.proptest.test.ts` (fast-check) | `src/rapier/motion_fit.rs` `#[cfg(test)]` (proptest) |
| **1 — Invariants** | Tiny-scenario physics invariants | `src/tests/rapier.invariants.test.ts`, `rapier.resim-continuity.test.ts` | `tests/kinematic_invariants_test.rs` |
| **2 — Parity/determinism** | Cross-language **count** parity + run-twice determinism | `scripts/generate-reference-data.mjs` → `js_reference_data.json` | `tests/cross_validation_test.rs`, determinism test in `kinematic_invariants_test.rs` |
| **3 — Browser** | Playwright smoke (non-blocking) | `js_stress_example/tests/stress.spec.js` | — |

```bash
# Rust (Tier 0 + 1 + 2) — blocking in CI; runs every test file
cargo test --manifest-path blast/blast-stress-solver-rs/Cargo.toml --features rapier,scenarios

# Rust — all known-bug repros (currently fail by design; gaps #1, #7, #9)
cargo test --manifest-path blast/blast-stress-solver-rs/Cargo.toml \
  --features rapier,scenarios --no-fail-fast -- --ignored

# JS — build the WASM runtime once, then run the blocking invariant + kernel suite.
# The set is defined once in the `test:invariants` package script (single source of truth).
cd blast/blast-stress-solver && npm install --ignore-scripts && npm run build
npm run test:invariants     # blocking, deterministic
npm run test:soak           # full suite, non-gating (environment-sensitive integration/perf)
```

> The JS `npm run build` step needs the Emscripten toolchain (emsdk `3.1.51`, as in CI).
> Pure-TS Tier-0 tests (`kernels.proptest.test.ts`) run without it; WASM-gated suites use
> `describe.skipIf(!runtimeAvailable)` and simply skip when `dist/stress_solver.wasm` is absent.

## Shared invariant spec

Both languages assert the same quantities with the same tolerances. The JS copy lives in
`blast-stress-solver/src/tests/invariants.shared.ts` (`TOL`); the Rust copy is the
`const`s at the top of `tests/kinematic_invariants_test.rs`.

| Quantity | Tolerance | Why |
|---|---|---|
| Point-velocity continuity across a split | `< 1e-3` m/s | A rigid fracture must preserve each chunk's world point velocity. **The headline invariant.** |
| Chunk world-position continuity | `< 1e-3` m | Chunks don't teleport across a split. |
| Body translation / rotation / angular-velocity continuity | `< 1e-6` | The child inherits the parent frame. |
| COM linear-velocity error | finite only | A child's COM legitimately differs from its parent's; only point-velocity is bounded. |
| Dynamic mass conservation | rel `< 1e-4` | A split repartitions mass; it never creates/destroys it. |
| No NaN/Inf | all finite | Guards the ill-conditioned angular fit on near-singular fragments. |

## The known bug (gap #1), reproduced and localized

The JS library transfers a fragment's velocity using Rapier's **real** centre of mass
(`syncBodyVelocityFromSource` in `destructible-core.ts`):
`v_child = v_parent + ω × (childCom − parentCom)`, and measures continuity in
`recordBodyContinuity` (`maxChunkPointVelocityError < 1e-3`).

The Rust split path (`body_tracker.rs::fit_child_motion`) instead fits the child's motion
about its **own node-centroid model** and never consults Rapier's collider-derived COM.
For fragments whose Rapier COM coincides with the node centroid (axis-aligned cuboids,
equal masses) the two agree and the split is continuous. For **offset-COM fragments**
(convex-hull / Voronoi pieces) combined with a **rotating parent**, the mismatch injects
`~|ω × comOffset|` of spurious velocity — the "sudden movement/rotation after destruction"
symptom.

`tests/kinematic_invariants_test.rs` makes this concrete and self-validating:

- `split_preserves_point_velocity_for_aligned_com_child` — cuboid fragment, same spin →
  continuity `< 1e-3`. **Passes** (positive control: the measurement is real).
- `split_preserves_point_velocity_for_offset_com_child` — convex-hull fragment, same spin →
  **fails today** (`#[ignore]`d, runs in the non-blocking repro lane), reporting e.g.
  `node 1 ... drifted 0.84 m/s (parent v {0.75,0,0} vs child v {1.5,-0.375,0})`.

The matched pair isolates the root cause to COM handling. **Detection only — this work does
not fix the bug.** Remove the `#[ignore]` once `fit_child_motion` is reconciled with Rapier's
COM, and the repro becomes a blocking regression guard automatically.

## Physics mechanisms: gravity orientation & momentum transfer

Two real-world behaviors were investigated and tested empirically in BOTH languages (don't
trust code inspection — assert the numbers). The *solver* implements both mechanisms in both
languages; the difference is which *pipeline* wires them up.

**Orientation-dependent gravity.** Gravity is global, but a chunk can be at any orientation;
a beam loaded perpendicular bends (and snaps), loaded axially only compresses. The solver
must apply gravity in each actor's *local* frame.
- Solver mechanism (proven both langs): `solver_mechanisms_test.rs` /
  `solver-mechanisms.test.ts` — `addGravity` and `addActorGravity` are direction-sensitive.
- JS pipeline: rotates gravity per actor into its local frame and calls `addActorGravity`
  (`destructible-core.ts`, on by default). Orientation-correct.
- Rust pipeline: calls the **global** `add_gravity` only — never `add_actor_gravity`, never
  syncs positions. **Orientation-blind** (gap #7). `gravity_orientation_test.rs` proves the
  control (gravity direction matters) passes while the rotation repro fails.

**Momentum transfer on fracture ("excess force").** When an impact breaks bonds, the load the
broken bonds carried should be released onto the freed fragments so they fly apart. NVIDIA
Blast computes this with `getExcessForces`.
- Solver mechanism (proven both langs): a released 10 kg / 100 m·s⁻² load reports ~1000 N of
  excess force (`solver_mechanisms_test.rs` / `solver-mechanisms.test.ts`). The pre-existing
  Rust test only checked finiteness, so this magnitude had never been asserted.
- Rust pipeline: applies it (`apply_excess_forces`, default on). `excess_force_integration_test.rs`
  shows fragments reach ~22 m/s with it on vs 0 with it off (physics must be integrated by the
  caller via `PhysicsPipeline`; `DestructibleSet::step` does not).
- JS pipeline: **never calls `getExcessForces`** — fragments get no kick without resimulation
  (gap #8).

## CI gating

`.github/workflows/ci.yml`:
- **`rust-tests`** (blocking): `cargo test --features rapier,scenarios`. A separate
  non-blocking step runs the `--ignored` repro so the bug stays visible.
- **`test`** (blocking): the JS invariant + kernel suites run without `continue-on-error`;
  a separate **non-blocking soak** step runs the full vitest suite for the
  environment-sensitive remainder.
- `deploy-production` now depends on `rust-tests` as well as `build` + `test`.

## Coverage gaps (the live list — what to address next)

1. **Rust split COM/velocity bug** — open. Repro: the `#[ignore]`d
   `split_preserves_point_velocity_for_offset_com_child`. Fix: make `fit_child_motion`
   express the fit about Rapier's actual centre of mass (read it back, as JS does), then
   un-ignore the test.
2. **Rust split-planner determinism** — unverified at scale. `fit_child_motion` /
   `handle_split` iterate `HashMap`s whose order is process-random; the small determinism
   test passes, but large scenarios aren't covered. Next: a multi-fracture determinism
   test, or switch the planner to ordered maps.
3. **No cross-language _state_ parity** — by deliberate decision. We assert the same
   invariants in each language and keep topological **count** parity
   (`cross_validation_test.rs`), but not bit-level JS↔Rust agreement on positions/velocities
   (infeasible across WASM/native f32 + Rapier internals, and flake-prone).
4. **Soak + UI not yet blocking** — the JS integration/perf suite and Playwright smoke run
   non-blocking. Promote them once they're proven stable across environments. The stale
   `organicSplit.spec.ts` assertion is quarantined (`it.skip`) pending re-baselining.
5. **No visual regression** — fractured/Voronoi demos (`fractured-tower.ts`, …) are
   browser-only; we have no screenshot/visual diff. A future Playwright visual-diff lane
   would catch rendering/positioning regressions the headless invariants can't see.
6. **JS mass/COM conservation** — only asserted Rust-side (where `RigidBody::mass()` is
   reachable in tests). The JS core would need a small per-body mass accessor to mirror it.
7. **Rust gravity is orientation-blind** — open, HIGH severity. The Rust pipeline applies
   global `add_gravity` on the authored geometry and ignores each actor's current rotation, so
   a chunk rotated into a stress-inducing orientation feels the wrong stress. JS handles it.
   Repro: `gravity_orientation_test.rs::actor_rotation_changes_fracture_behavior` (`#[ignore]`).
   Fix: in `DestructibleSet::step`, per actor, rotate gravity into the body's local frame and
   call `add_actor_gravity` (as `destructible-core.ts` does), then un-ignore the repro.
8. **JS does not transfer momentum on fracture** — open. The JS pipeline never calls
   `getExcessForces`, so fractured pieces get no outward kick unless resimulation is on; with
   resim off they just sit/inherit the rigid recoil. Rust applies excess forces and works.
   Fix: mirror Rust's `apply_excess_forces` in `destructible-core.ts`. (A JS integration test
   needs a per-body velocity/mass accessor on the core to assert fragment momentum.)
9. **Rust excess force is applied as a persistent force (CONFIRMED bug)** — `apply_excess_forces`
   uses Rapier `add_force` (continuous until reset) instead of a one-shot `apply_impulse`. Unless
   the consuming app resets forces every step, a single fracture re-accelerates fragments every
   physics step — fragment speed grows unbounded (repro: ~22 → ~1900 m/s over a few frames). A
   strong candidate for the "sudden movement" report — and, unlike gap #1, it needs no rotation.
   Repro: `excess_force_persistence_test.rs::excess_force_kick_should_be_one_shot` (`#[ignore]`).
   Fix: apply the excess as an impulse (or reset forces each step); then un-ignore.
10. **js_stress_example split specs don't run on CI** — open. `npm run test:split` fails at
    module resolution in the Test job (`blast-stress-solver/scenarios` export and
    `./stress_solver.cjs` are not built/linked there); it never actually ran — the old
    `|| true` hid it. Now reporting-only (non-gating). Fix: have the Test job build
    `blast-stress-solver` (not `--ignore-scripts`) and the js_stress_example WASM, or link the
    package, so the specs resolve, then promote back to blocking.

### Recently added unit/property coverage (all passing, blocking)

- **Bond stress** (`bond_stress_test.rs` + `bondStress.parity.test.ts`): hand-computed
  known values that double as a JS↔Rust parity lock, plus the previously-untested angular
  twist/bend paths and non-negativity properties.
- **Fracture policy** (`fracture_policy_test.rs`): per-frame budget and child-admission
  boundary cases (`>=` vs `>`, "0 = unlimited?").
- **Fatal threshold** (`stressLimits.boundary.test.ts`): `failureMode`'s strict-`>` boundary
  (at-limit must not fail) and channel priority.
- **Damage system** (`damage.invariants.test.ts`): health monotonic non-increasing, no
  healing, destruction irreversible, support chunks never destroyed, preview non-mutating.

> Notes from bug-hunting: JS and Rust `computeBondStress` are byte-identical (verified); the
> shipped wall/tower/bridge builders are structurally clean (unit normals, no out-of-range
> bonds); wall-collapse determinism holds across runs. The gap #1 split COM bug only manifests
> when a fragment is *rotating at the fracture instant* (stress-driven fractures usually fire
> before rotation develops), which matches the "sometimes" sudden-movement report.
