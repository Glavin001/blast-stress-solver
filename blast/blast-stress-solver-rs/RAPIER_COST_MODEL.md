# Rapier cost model & efficient-usage notes (real-time destruction)

Ground truth for how we mutate the Rapier world during fracture, so we keep our
per-frame world changes as cheap as Rapier allows. Derived from the `rapier3d` 0.30.1
source we compile against, plus the `rapier_cost_profile` raw-Rapier benchmark.

Reproduce the numbers: `cargo run --release --example rapier_cost_profile --features bench-support`

## The one thing to internalise: mutations are deferred

Every mutator (`insert`, `remove`, `set_parent`, `set_body_type`, `set_position`, …) is
cheap and synchronous: it edits the component and sets a dirty bit / pushes the handle to a
"modified" list. **The real work happens at the start of the next `PhysicsPipeline::step()`**,
which drains those lists (`pipeline/physics_pipeline.rs`, `pipeline/user_changes.rs`):
re-parent collider world-poses, recompute mass properties, re-add bodies to the active set,
update the broad-phase BVH, create/destroy contact pairs. Two consequences:

1. **Measure cost as `(mutate + one step) − (one step)`** — the call alone is misleading.
2. **Batch every edit for a frame, then step once.** The step has fixed per-call-batch
   sweeps (broad-phase pair retain, narrow-phase edge scan, BVH incremental optimize);
   `N` small mutation→step cycles pay them `N` times. We already do all fracture edits,
   then the caller steps once.

## Per-operation cost (source + measured)

| op | synchronous | deferred to next step | scales with |
|---|---|---|---|
| `RigidBodySet::insert` (+collider) | arena O(1), all dirty bits | BVH leaf insert, new contact pairs, mass, island insert | changed set |
| `RigidBodySet::remove` | arena O(1), island swap-remove O(1), **wakes every contacting body** | broad/narrow-phase leaf+node removal | changed set + woken island |
| `ColliderSet::set_parent` | detach/attach + 2 mass recomputes (immediate) | in-place AABB **update** (leaf kept), pos recompute | #colliders on the 2 bodies |
| `ColliderSet::remove`+`insert` | leaf removal, **wakes contacts**, new leaf | full re-pair in narrow-phase | changed set + woken |
| `set_body_type` | dirty TYPE, wake | active-set insert, dominance propagation to colliders | #its colliders |
| `recompute_mass_properties_from_colliders` | **fully immediate** O(#colliders) | (step recomputes again if colliders were also dirtied) | #colliders on body |
| `wake_up` | O(1) | re-activates the **whole connected island** via contact/joint traversal | island size |
| `step` solver + island stages | — | — | **awake bodies/islands only** (sleeping ≈ free) |
| `step` broad/narrow-phase | — | — | mostly the **changed** set (+ small full-pair sweeps) |

Measured marginal costs (release, raw Rapier, settled box stacks):

- **Insert**: ~0.4 µs/body synchronous, **~4 µs/body** in the deferred step (≈10× the call).
- **`set_parent` vs remove+reinsert** (per collider, positions preserved): call **2.2× cheaper**, post-step **~1.6× cheaper**. `set_parent` keeps the broad-phase leaf and contact-graph node; remove+reinsert churns both and wakes contacts.
- **Remove**: call ~0.1 µs/body, but removing 512 bodies **woke ~100 neighbours**, taking the next step 0.62 → 2.4 ms. Removal cost is dominated by the contacts it wakes, not the call.
- **Settled step, asleep vs awake** (THE lever): N=1024 → **0.23 ms asleep vs 3.39 ms awake (14.6×)**; N=2304 → **0.63 vs 8.0 ms (12.7×)**. Per-frame cost is dominated by *how many bodies are awake*, not by the topology edits.

## Efficient-usage checklist (and where we stand)

- ✅ **Batch all edits per frame, then one `step`.** (fracture applies all edits; one step/frame)
- ✅ **`set_parent` to re-parent colliders**, never remove+recreate. (`move_node_collider_to_body`)
- ✅ **Recycle a *live* body for a new fragment** instead of remove+insert. Handles are
  generational — a removed handle is dead forever, and a reinserted body pays full
  "all-dirty" cold-start cost — so reusing a live body is strictly better. (recycle pool)
- ✅ **Reuse the engine workspaces** (`PhysicsPipeline`/`BroadPhase`/`NarrowPhase`/`IslandManager`) across frames.
- ✅ **Sleep settled debris** (sleep thresholds) — the single biggest total-perf lever (≈13×).
- △ **Don't double-recompute mass.** Our velocity reconciliation calls
  `recompute_mass_properties_from_colliders` to read the COM, and the step recomputes it
  again because the colliders were dirtied. Small (sub-ms), correctness-required; left as-is.
- △ **Minimise net removals** (each wakes neighbours). The recycle pool already keeps
  `insert`/`remove` to the *net* body-count change; nothing further needed unless profiling says so.
- ☐ **Preallocate `RigidBodySet`/`ColliderSet` capacity** to expected peak fragment count to
  avoid mid-cascade reallocations (minor; not yet done).

## Takeaway

For real-time fracture of complex structures, the topology *edits themselves are cheap and
already use Rapier's efficient paths*. The dominant per-frame cost is **simulating awake
bodies**, so the performance frontier is keeping fragments asleep (and not waking more than
necessary), not micro-optimising individual Rapier calls further.
