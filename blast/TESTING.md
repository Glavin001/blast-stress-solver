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

## The split COM/velocity bug (gap #1) — fixed

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
  continuity `< 1e-3`. Positive control (the measurement is real).
- `split_preserves_point_velocity_for_offset_com_child` — convex-hull fragment, same spin →
  also continuous now (drifted ~2.1 m/s before the fix).

**Fix (Rust):** `handle_split` runs a post-migration pass (`reconcile_child_velocity_with_com`)
that recomputes each dynamic child's mass properties and shifts its `linvel` by
`ω × (real_com − fit_center)`, so the body's velocity field matches the fitted
(parent-continuous) field at the node points regardless of collider-COM offset. Both tests in
the matched pair are now passing, blocking regression guards.

### JS reused-fragment velocity (gap #1b) — ✅ FIXED

The "JS already satisfies this" claim above was only true for the **created** child bodies. A
fracture partitions a parent into one **reused** body (the largest fragment — it keeps the
parent's Rapier handle and merely *loses* the colliders that migrated to the new children) and
N **created** bodies. `syncBodyVelocityFromSource` only ran for the *created* targets, so the
reused body was never re-derived for its shifted COM. Because Rapier stores `linvel` as the
velocity *of the centre of mass* and keeps it when `removeCollider` shifts the COM (the recompute
is even deferred to the next `step`), the reused fragment's velocity field jumped by `ω × ΔCOM`
on every fracture — the **"big fragment hovers while small debris falls normally"** report (the
JS high-rise demo). The split-continuity log stayed green (~1e-7) because it structurally never
observes the reused body.

**Fix:** `reconcileReusedBodyVelocity` in `destructible-core.ts` — after migration, recompute the
reused body's mass properties and shift its `linvel` by `ω × (newCom − oldCom)`. Applied both in
`flushColliderMigrations` (non-resim) and on `restoreWorldSnapshot` (resim re-applies the stale
COM-velocity; `BodySnapshot` now carries `com` so the rollback can re-correct). Guards:
- `rapier.splitVelocity.mechanism.test.ts` — pure-Rapier, no WASM. Pins the mechanism with exact
  magnitudes (removeCollider defers the recompute; ω×ΔCOM drift; symmetric-loss zero control;
  the `linvel += ω × ΔCOM` fix spec; the created-child transfer control).
- `rapier.splitVelocity.noHover.test.ts` — real pipeline (WASM): a controlled rotating split where
  the reused fragment's drift drops from ~1.5 m/s to <1e-2, plus cascade guards (no NaN, mass
  conserved, no gross hover).

### Rust reused-fragment resim teleport (gap #1c) — ✅ FIXED

Distinct symptom seen in the **Rust demo**: when a body fractures *under resimulation*, a large
fragment "yanks itself inwards" for a frame. The Rust split path RE-CENTERS each child body's
origin onto its fragment COM (so the velocity fit reference `fit_center` equals the COM, making
`reconcile_child_velocity_with_com` a no-op). For a **reused** body that moves its existing
origin and re-derives every retained collider's local offset relative to the *new* origin. The
demo's resimulation, however, snapshots each body's **origin** *before* the step and `restore()`s
it *after* a fracture (`BodySnapshots`); restoring the OLD origin onto the NEW offsets places the
colliders at `old_origin · new_offset` — teleported by the recentering distance, toward the old
origin (= "inward"). Velocity continuity can't see this; it is a pure position jump.

**Fix:** decouple the velocity-fit reference from the body pose. `ChildTargetState` now carries a
`fit_center` (the node-model COM, drives `fit_child_motion` + `reconcile_child_velocity_with_com`
— unchanged, so velocity is identical) separately from `pose`. **Reused** bodies keep their
inherited origin (no recentering → resim snapshots stay valid → no teleport); only newly
created/recycled bodies (absent from any snapshot) are recentered. This mirrors the JS pipeline,
which never recenters. Guards (Rust):
- `tests/resim_recenter_teleport_test.rs` — mimics the demo's capture→step→restore; the worst
  chunk teleport drops from ~1.5 m to <1e-3.
- `tests/split_position_continuity_test.rs` — asserts the new `SplitContinuityRecord.world_position_error`
  (Rust analogue of the JS `maxChunkWorldPositionError`) stays <1e-3 for recentered multi-node halves,
  alongside the existing velocity continuity.

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
- Rust pipeline: **fixed** (gap #7) — `apply_oriented_gravity` now rotates gravity into each
  actor's local frame and calls `add_actor_gravity`, like JS. `gravity_orientation_test.rs`
  (control + rotation guard) both pass.

**Momentum transfer on fracture ("excess force").** When an impact breaks bonds, the load the
broken bonds carried should be released onto the freed fragments so they fly apart. NVIDIA
Blast computes this with `getExcessForces`.
- Solver mechanism (proven both langs): a released 10 kg / 100 m·s⁻² load reports ~1000 N of
  excess force (`solver_mechanisms_test.rs` / `solver-mechanisms.test.ts`). The pre-existing
  Rust test only checked finiteness, so this magnitude had never been asserted.
- Rust pipeline: supports it as an **opt-in alternative** (`apply_excess_forces`, default
  **off**), applied as a **one-shot impulse** (`force × dt`, the real frame dt) about the body's real centre
  of mass. **Resimulation is the preferred, more sound source of fragment momentum** (re-resolve
  the actual contact against the fractured pieces); excess force is for consumers who don't
  resimulate. `excess_force_integration_test.rs` shows fragments reach ~22 m/s with it on vs 0
  with it off (physics integrated by the caller via `PhysicsPipeline`; `DestructibleSet::step`
  does not).
- JS pipeline: **never calls `getExcessForces`** — it gets fragment momentum from resimulation
  instead (gap #8 is therefore "by design" on the JS side).
- Resimulation validated in `resimulation_test.rs`: `BodySnapshots` capture/restore round-trips
  exactly, and with excess force OFF, the rollback + re-resolved contact gives fragments
  ~0.65× the ball's speed (sound recoil) — vs the excess-force estimate's transient ~8.7×, vs
  ~0.4× with neither mechanism. This is why resimulation is the preferred momentum source.

## CI gating

`.github/workflows/ci.yml`:
- **`rust-tests`** (blocking): `cargo test --features rapier,scenarios`. A separate
  non-blocking step runs the `--ignored` repro so the bug stays visible.
- **`test`** (blocking): the JS invariant + kernel suites run without `continue-on-error`;
  a separate **non-blocking soak** step runs the full vitest suite for the
  environment-sensitive remainder.
- `deploy-production` now depends on `rust-tests` as well as `build` + `test`.

## Coverage gaps (the live list — what to address next)

1. **Rust split COM/velocity bug** — ✅ FIXED. `handle_split` now reconciles each dynamic
   child's velocity with Rapier's actual centre of mass
   (`reconcile_child_velocity_with_com`: `linvel += ω × (real_com − fit_center)`). The former
   repro `kinematic_invariants_test.rs::split_preserves_point_velocity_for_offset_com_child`
   is now a passing, blocking regression guard.
   - **1b. JS reused-fragment COM/velocity — ✅ FIXED.** `reconcileReusedBodyVelocity` re-derives
     the reused fragment's velocity for its shifted COM (in `flushColliderMigrations` and on the
     resim `restoreWorldSnapshot`). Stops the "big fragment hovers" report. Guards:
     `rapier.splitVelocity.mechanism.test.ts` + `rapier.splitVelocity.noHover.test.ts`.
   - **1c. Rust reused-fragment resim teleport — ✅ FIXED.** Recentering a *reused* body's origin
     made the demo's resim rollback (which restores the pre-split origin) teleport its colliders
     inward ("fragment yanks itself inwards"). Reused bodies now keep their inherited origin (the
     velocity fit uses a decoupled `fit_center`, so velocity is unchanged). Guards:
     `resim_recenter_teleport_test.rs` + `split_position_continuity_test.rs` (new
     `world_position_error` metric). See the sections above.
2. **Rust split-planner determinism** — covered for wall + tower at scale
   (`multi_fracture_test.rs::tower_collapse_is_deterministic_at_scale`,
   `scenario_invariants_test.rs::wall_collapse_is_deterministic`); both pass, so the
   `HashMap` ordering doesn't affect results in practice. Bridge-scale still uncovered.
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
6. **JS mass/COM conservation** — ⚠️ PARTIAL. Dynamic *mass* conservation is now asserted
   JS-side too (`rapier.splitVelocity.noHover.test.ts` sums `RigidBody::mass()` over
   `world.forEachRigidBody`). *Momentum*/COM-velocity conservation across a rotating split is
   still not a blocking assertion on the reused fragment (blocked on gap #1b).
7. **Rust gravity orientation** — ✅ FIXED. `DestructibleSet::step` now applies gravity per
   actor, rotated into the body's current local frame via `add_actor_gravity`
   (`apply_oriented_gravity`), matching the JS pipeline. Guard:
   `gravity_orientation_test.rs::actor_rotation_changes_fracture_behavior` (now passing,
   blocking). No regression in the count-based wall/tower tests.
8. **JS does not transfer momentum on fracture** — open. The JS pipeline never calls
   `getExcessForces`, so fractured pieces get no outward kick unless resimulation is on; with
   resim off they just sit/inherit the rigid recoil. Rust applies excess forces and works.
   Fix: mirror Rust's `apply_excess_forces` in `destructible-core.ts`. (A JS integration test
   needs a per-body velocity/mass accessor on the core to assert fragment momentum.)
9. **Rust excess force persistence** — ✅ FIXED. The fracture kick is now a one-shot impulse
   (`apply_impulse(force × dt)`) instead of a persistent `add_force`, so fragment speed no longer
   grows unbounded. `dt` is the real frame timestep, threaded through
   `step_with_time(now_secs, dt, …)` (the demo passes its `IntegrationParameters::dt`); `step()`
   uses the `set_time_step` default. Guard:
   `excess_force_persistence_test.rs::excess_force_kick_should_be_one_shot`
   (now passing without any per-step force reset).
10. **js_stress_example split specs don't run on CI** — open. `npm run test:split` fails at
    module resolution in the Test job (`blast-stress-solver/scenarios` export and
    `./stress_solver.cjs` are not built/linked there); it never actually ran — the old
    `|| true` hid it. Now reporting-only (non-gating). Fix: have the Test job build
    `blast-stress-solver` (not `--ignore-scripts`) and the js_stress_example WASM, or link the
    package, so the specs resolve, then promote back to blocking.
11. **Excess-force one-frame spike on simultaneous mass-fracture (tuning, open)** — diagnosed
    precisely (projectile test, 2 kg ball @ 30 m/s → 6×5 wall):
    - NOT double-counting: with `apply_excess_forces` off, the anchored wall's fragments get
      ~0.4 m/s (Rapier's contact momentum drains into the anchor, not the fragments).
    - NOT a steady over-speed: the end-state fly-back is ~31.8 m/s ≈ **1× the ball** (reasonable
      recoil); momentum stays ~1.7× the ball.
    - It IS a **one-frame transient**: when many bonds break in a single frame (44 at once at
      unlimited rate), their per-bond excess forces SUM onto the impacted fragment that frame →
      a ~262 m/s (~8.7×) spike, which settles to ~31.8 the next frame as inter-fragment
      collisions redistribute it.
    Mitigation that already works: `max_fractures_per_frame` spreads the breaks across frames —
    at limit 4 the spike is gone and end-speed drops to ~3.7 m/s. Optional code safeguard: clamp
    the per-fragment excess Δv per frame so the spike can't occur regardless of fracture rate.
    Open decision: rely on the rate-limit knob vs add the per-fragment clamp. The
    `projectile_impact_test.rs` guard only catches runaway (regression to the old unbounded
    behavior), not this transient.

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

## Web → Rust parity port (island solving, batched gravity, contact splash)

The web library's recent performance/hardening work was ported to Rust. Most of it binds the
**same shared C++ core** the WASM build uses (`blast/source/shared/stress_solver/stress.cpp`,
exposed through `blast/rust_stress_example/ffi/ext_stress_bridge.cpp`), so parity is by
construction, not reimplementation.

- **Island-aware solving** (`island_solver_test.rs`): `ExtStressSolver::set_island_aware` /
  `set_skip_settled` + counters, wired into `DestructibleSet` (default **ON** for production
  sets; the raw solver keeps the C++ default OFF so `cross_validation_test.rs` fixtures stay
  bit-identical). Tests lock: single-island ⇒ bit-identical whole-graph fallback; multi-island
  fracture sequence/partition parity; skip-settled engages and is a *perfect no-op*.
- **Batched actor gravity** (`batched_actor_gravity_test.rs`): one FFI crossing rotates gravity
  into every actor's frame. The bridge forward-rotates by the supplied quaternion, so we pass each
  body rotation's **conjugate** to reproduce `R⁻¹·g` byte-identically to the per-actor path (which
  is retained as `apply_oriented_gravity_per_actor` for the equivalence test). Gravity orientation
  is `R⁻¹·g` (verified physically correct, gap #7); note the **web uses the forward `R·g`** there —
  a latent web/Rust convention divergence worth a focused follow-up.
- **Island quiescence stats** (`island_settled_stats_test.rs`): `DestructibleSet::island_settled_stats`
  (union-find with static nodes as cut points; settled = body asleep) — read-only instrumentation.
- **Contact injection + per-body splash grid** (`contact_injection_test.rs`):
  `DestructibleSet::inject_contacts` (rotate world→local, hit node + same-body splash neighbours
  with `(1−d/R)²` falloff, batched submit) with a per-body grid keyed on stable asset centroids.
  The grid is an acceleration structure only — tested **byte-for-byte against a full-scan reference**
  (intact and post-fracture).

> **Why island-aware solving can shift a large cascade slightly (e.g. weak wall 86→84 fractures).**
> Bonds in different islands share no nodes, so the CGNR system is *block-diagonal*. With a direct
> solve in exact arithmetic per-island ≡ whole-graph. But CGNR is iterative and computes its
> step-size scalars (α, β) from **global** dot-products; for a multi-block system those shared
> scalars are a compromise, so whole-graph CG stagnates at a higher residual floor while per-island
> CG gives each block its own optimal scalars and converges *tighter*. Empirically per-island
> reaches a **lower** residual (it is at least as accurate), and the gap is structural (stable from
> 200→1000 iterations, not under-convergence). With ≤1 island it falls back bit-identically. Guarded
> by `per_island_residual_is_no_worse_than_whole_graph` and the deterministic 84/30 golden.
