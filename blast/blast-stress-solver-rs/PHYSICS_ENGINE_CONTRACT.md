# Physics-engine integration contract

This document describes the **contract** a rigid-body physics engine must satisfy
to drive the Blast stress solver's destruction pipeline. It exists so you can
answer one question precisely: *"what would it take to run this on engine X
instead of Rapier?"*

The crate ships one reference integration — `blast_stress_solver::rapier`
(feature `rapier`) — built on [Rapier3D](https://rapier.rs). Everything below is
expressed as the abstract capability the pipeline needs, followed by the concrete
Rapier call that satisfies it, so the same reasoning transfers to PhysX, Jolt,
Bullet, Havok, or a bespoke engine.

> TL;DR — the solver owns a **graph** of chunks (nodes) and bonds and tells you
> *when* and *how* a structure breaks apart. The physics engine owns **rigid
> bodies and colliders** and tells you *where* everything is and *what* it hit.
> The integration layer keeps a node↔collider / actor↔body mapping and, on each
> fracture, edits the engine's body/collider topology to match the new graph.
> Porting to another engine means re-implementing that edit layer against the new
> engine's body/collider API — the solver itself never changes.

---

## 1. The two halves and what crosses between them

```
        blast-stress-solver (engine-agnostic)          your physics engine
    ┌──────────────────────────────────────────┐   ┌────────────────────────┐
    │  ExtStressSolver                          │   │  RigidBodySet          │
    │   • nodes (centroid, mass, volume)        │   │  ColliderSet           │
    │   • bonds (centroid, normal, area, n0,n1) │   │  broad/narrow phase     │
    │   • actors  = connected components         │   │  island/joint solvers  │
    │   • stress solve → FractureCommand         │   │  integration + contacts│
    │   • apply fracture → SplitEvent            │   └────────────────────────┘
    └──────────────────────────────────────────┘                ▲
                     │  SplitEvent                                │
                     ▼                                            │
    ┌──────────────────────────────────────────────────────────┴──┐
    │  Integration layer (this is the part you port)               │
    │   • BodyTracker: node↔collider, actor↔body mapping           │
    │   • on split: reuse/recycle/create bodies, re-parent colliders│
    │   • collision groups, support-contact tracking, resim         │
    └──────────────────────────────────────────────────────────────┘
```

**Invariants the integration layer maintains:**

- **One actor ↔ one rigid body.** An "actor" is a connected component of the
  bond graph. When bonds break and a component splits, one body must become
  several.
- **One node ↔ one collider,** parented to that node's current body, positioned
  at the node's offset from the body origin. A node's collider is *moved between
  bodies* on a split — never destroyed and rebuilt, so contact state and the
  handle survive.
- **Support nodes (`mass == 0`) pin a body to the world.** A body containing any
  support node is *fixed/static*; otherwise it is *dynamic*. This can change for
  a body across a split (a chunk breaks free of its footing), so body type must
  be mutable in place.

Only three files touch the engine: `rapier/body_tracker.rs` (all topology
edits), `rapier/collision_groups.rs` (filtering), and the host's event wiring
(see the demo, `blast/blast-stress-demo-rs`). `rapier/split_migrator.rs`
(reuse/create planning) and `rapier/motion_fit.rs` (mass/velocity math) are pure
and engine-independent — they operate on opaque handles and plain vectors, so
they port unchanged.

---

## 2. Required engine capabilities

Each capability below is used by the pipeline. If your target engine cannot do
one of the **required** items, the pipeline cannot run faithfully on it. The
**optional** items gate specific features you can leave off.

### 2.1 Rigid-body lifecycle — *required*

| Capability | Why the pipeline needs it | Rapier reference |
|---|---|---|
| Create a **dynamic** body at a world pose, with initial linear/angular velocity | Each free-flying fragment becomes one dynamic body | `RigidBodyBuilder::dynamic().pose(..).linvel(..).angvel(..)` |
| Create a **fixed/static** body at a world pose | Support-anchored actors, ground | `RigidBodyBuilder::fixed().pose(..)` |
| **Remove** a body, cascading removal of its colliders and any joints, and updating island bookkeeping | Retiring a parent body after its nodes migrate to children; debris cleanup | `RigidBodySet::remove(handle, island_manager, colliders, impulse_joints, multibody_joints, true)` |
| Stable, comparable, hashable **handles** | The tracker keys maps on handles and **sorts them** for run-to-run determinism | `RigidBodyHandle` (`into_raw_parts()` gives a stable ordering) |

> **Determinism note.** `body_tracker.rs` sorts parent bodies by handle before
> recycling/creating (`collect_split_planning_data`), and sorts child cohorts,
> specifically so the engine allocates handles in a reproducible order. If your
> engine's handle allocation is nondeterministic, chaotic fracture will diverge
> run-to-run. A stable handle order is a hard requirement for reproducibility.

### 2.2 Body-type mutation (the "reparent"/flip) — *required*

| Capability | Why | Rapier reference |
|---|---|---|
| Flip a body between **dynamic and fixed in place**, keeping its handle | Body *reuse/recycle* on a split: a reused body whose node set gained/lost a support node must change type without being recreated | `RigidBody::set_body_type(ty, wake)` |
| **Wake** a body | After un-fixing, and after any velocity edit | `RigidBody::wake_up(true)` |

This is what makes body **reuse** possible instead of destroy-and-recreate. The
split planner (`split_migrator`) tries to hand each child actor an *existing*
parent body (minimal edit distance); reconciliation then flips its type if the
child's support status differs. See `reconcile_reused_body_type` /
`reinitialize_recycled_body`.

### 2.3 Collider lifecycle & re-parenting — *required (this is the crux)*

This is the capability the "split up / re-parent colliders" question is really
about. On every split, each migrating node's collider must be **moved to its new
body** and repositioned, without being destroyed:

| Capability | Why | Rapier reference |
|---|---|---|
| Create a collider (**cuboid** and **convex hull** minimum) parented to a body, at a **local pose** | Initial body build; inserting a collider for a node that never had one | `ColliderBuilder::cuboid/convex_hull(..).translation(..)`, `ColliderSet::insert_with_parent(collider, body, bodies)` |
| **Re-parent an existing collider to a different body** | The core split edit — move a chunk from the old (parent) body to the new (child) body, preserving the collider handle & its contact history | `ColliderSet::set_parent(collider, Some(new_body), bodies)` |
| Set a collider's **local pose relative to its parent** | After re-parenting, the node sits at a new offset from the new body's origin | `Collider::set_position_wrt_parent(Isometry)` |
| **Remove** a collider | Destroying a node (debris cull, sub-min-size child) | `ColliderSet::remove(collider, island_manager, bodies, true)` |
| Per-collider **mass contribution** | Node mass drives the body's aggregate mass/inertia; support nodes contribute mass 0 | `ColliderBuilder::mass(m)` |
| **Recompute a body's mass properties from its child colliders**, then read the resulting **center of mass** | After migrating colliders, the body's COM shifts; the velocity fit reconciles against the *real* COM (see §3) | `RigidBody::recompute_mass_properties_from_colliders(colliders)`, `RigidBody::center_of_mass()` |
| Look up a collider's **parent body** | Support-contact classification (is the other collider on a fixed body?) | `Collider::parent()` |

If your engine cannot re-parent a collider (some engines only allow
collider→body attachment at creation), you must emulate it: remove the old
collider and create a new one on the target body at the new local pose, and
accept that per-contact state (warm-starting, ongoing manifolds) is lost across
the split. The Rapier path deliberately re-parents to avoid that discontinuity.

### 2.4 Body pose & kinematics read/write — *required*

The tracker reads parent state at the instant of a split and writes fitted state
onto children.

| Capability | Rapier reference |
|---|---|
| Read/write body **pose** (translation + rotation) | `RigidBody::position()`, `set_position(iso, wake)`, `RigidBodyBuilder::pose` |
| Read/write **linear & angular velocity** | `linvel()/angvel()`, `set_linvel/set_angvel(v, wake)` |
| **Point velocity** at a world point (`v + ω × r` about the true COM) | `RigidBody::velocity_at_point(&p)` — used to sample each parent node's world velocity so children inherit continuous motion |
| **Sleep / wake / is-sleeping** | `sleep()`, `wake_up(bool)`, `is_sleeping()` |
| Query **body type / is-dynamic / is-fixed / is-kinematic** | `body_type()`, `is_dynamic()`, `is_fixed()`, `is_kinematic()` |

### 2.5 Contact & force feedback (engine → solver) — *required for interaction*

The solver is blind to the world; the engine must report contacts so the host
can feed them back in. Two event streams are consumed:

| Event | What the host does with it | Rapier reference |
|---|---|---|
| **Collision started** (two collider handles) | Map collider → node → body; if the other side is fixed/support, call `mark_body_support_contact` — this is how debris "knows" it landed, driving settle/sleep/cleanup and collision-group re-filtering | `CollisionEvent::Started`, drained from a `ChannelEventCollector` after the engine step |
| **Contact force** (two colliders, force magnitude & direction) | Map the struck collider → node, convert the impact into a stress-graph force via `DestructibleSet::add_force(node, world_pos, force)` so the hit propagates through the bond graph and can break it | `ContactForceEvent`, enabled per-collider via `ActiveEvents::CONTACT_FORCE_EVENTS` + `contact_force_event_threshold` |

Requirements this implies for the engine:

- Emit **collision-start** and **contact-force** events carrying the two
  collider identities and (for force events) a total force magnitude/vector.
- Let the host map a **collider handle → its parent body**, and the integration
  layer maps **collider handle → node** (`collider_node`) and **node → body**
  (`node_body`). Your engine only needs to provide the collider→body half and
  stable collider handles; the crate owns the node maps.

The reference colliders are created with
`ActiveEvents::CONTACT_FORCE_EVENTS | COLLISION_EVENTS` and
`ActiveHooks::FILTER_CONTACT_PAIRS | FILTER_INTERSECTION_PAIR` (see
`build_node_collider`).

### 2.6 Collision filtering — *optional (debris scaling & split cleanliness)*

Two independent filtering mechanisms are used:

| Mechanism | Purpose | Rapier reference |
|---|---|---|
| **Per-collider interaction groups** (membership + filter bitmasks), plus solver groups | Cheaply stop small debris from colliding with each other (`DebrisCollisionMode`) once counts explode; keep debris-vs-ground | `Collider::set_collision_groups/set_solver_groups(InteractionGroups)` — see `collision_groups.rs` |
| **Contact-pair filter hook** run *inside* the narrow phase | "Sibling grace": suppress contacts between fragments from the *same* split for a short window so freshly-separated pieces don't explosively push apart | `PhysicsHooks::filter_contact_pair` — see `SiblingGraceHooks` in the demo |

If your engine has no per-pair filter hook, you lose sibling-grace (fragments may
jitter apart on the split frame) but the pipeline still runs. If it has no
collision groups, debris-heavy scenes get more expensive but stay correct.

### 2.7 Tuning knobs on bodies — *optional (perf/quality)*

| Knob | Feature it powers | Rapier reference |
|---|---|---|
| Linear/angular **damping** | `SmallBodyDampingOptions` — settle tiny debris faster | `set_linear_damping/set_angular_damping` |
| **Sleep thresholds** (linear/angular activation) | `SleepThresholdOptions` — let debris sleep sooner | `RigidBody::activation_mut()` |
| **CCD** toggle per body | Fast projectiles / thin fragments not tunneling | `RigidBodyBuilder::ccd_enabled`, `is_ccd_enabled()` |
| **Apply impulse / torque-impulse** | `apply_excess_forces` — kick newly separated fragments with the solver's released load (`get_excess_forces`) as a one-shot `force × dt` | `apply_impulse`, `apply_torque_impulse` |

### 2.8 Full body-state snapshot & restore — *optional (resimulation)*

Resimulation rolls a frame back and re-steps it *after* a fracture, so contacts
resolve against the already-fractured pieces (more physically faithful than
kicking fragments manually). It requires snapshotting and restoring **every
non-fixed body**:

| Capability | Rapier reference |
|---|---|
| Read & write pose, linvel, angvel, damping, sleep flag, enabled flag; reset accumulated forces/torques | `BodySnapshots::capture_into/restore` in `resimulation.rs` (`set_enabled`, `reset_forces`, `reset_torques`, …) |
| **Deterministic step** so re-running the same frame is reproducible | engine-wide property |

Resimulation is off by default (`ResimulationOptions { enabled: false }`). Skip
this section entirely if you don't need it — but note it's the *preferred* source
of fragment momentum over `apply_excess_forces`.

---

## 3. Two correctness traps any port must handle

These are the non-obvious parts — where a naive integration produces the classic
"fragment jumps / spins on fracture" bug. Both are already solved in
`body_tracker.rs`; a port must reproduce them.

1. **COM shift after collider migration.** The velocity fit
   (`motion_fit::fit_rigid_motion`) expresses a child's motion about its
   node-model center of mass (`fit_center`). But the engine integrates rotation
   about the *collider-derived* COM, which differs for offset shapes. Left alone,
   a rotating fragment gains a spurious `ω × (engine_com − fit_center)`. After all
   colliders are migrated, the crate recomputes mass properties and shifts
   `linvel` by exactly that term (`reconcile_child_velocity_with_com`). This must
   happen in a **separate pass after every child's colliders are in place**, or a
   reused parent body still holding a sibling's collider reports the wrong COM.

2. **Recenter only new bodies, never reused ones.** A newly created/recycled body
   is recentered onto its COM. A **reused** body keeps its inherited origin — if
   you recenter it, a resimulation snapshot of its old origin becomes invalid and
   rollback teleports it ("fragment yanks inward"). See `ChildTargetState.pose`
   vs `fit_center` and the `reuse` flag through `compute_child_target_state`.

Point-velocity continuity (child world velocity at each node ≈ parent's, and
world-position drift ≈ 0 across the edit) is exactly the health metric captured
by `SplitContinuityRecord` when `set_record_split_continuity(true)` is on. Use it
to validate a new engine's split path.

---

## 4. The per-frame orchestration contract

The host owns the loop. `DestructibleSet::step_with_time` is the one integration
call; everything around it is engine work the host performs in this order:

```text
per frame:
  1. (resim only) capture_resimulation_snapshot(&bodies)      // §2.8
  loop (once, or until no new fracture and resim passes remain):
    2. step the physics engine                                 // integrate + contacts
    3. drain collision-start events  → mark_body_support_contact   // §2.5
       drain contact-force events    → add_force(node,pos,force)   // §2.5
    4. destructible.step_with_time(now, dt, bodies, colliders,
             island_manager, impulse_joints, multibody_joints):
         a. apply gravity per actor, rotated into its current frame
         b. (opt) apply centrifugal accel to spinning actors
         c. solver.update()  → overstressed bonds
         d. generate + rate-limit + apply fracture commands → SplitEvents
         e. for each SplitEvent: plan reuse/create, re-parent colliders,
            fit child kinematics, retire dead parents          // §2.1–2.4
         f. (opt) apply_excess_forces as one-shot impulses      // §2.7
    5. if fractured and resim passes remain:
         restore snapshot; recapture; repeat loop               // §2.8
  6. (opt) process_optimizations(now, ...) → debris TTL cleanup // §2.7
  7. read body poses back out to your renderer
```

Step 4 is entirely inside the crate and engine-agnostic *except* for the
`bodies`/`colliders`/`island_manager`/`impulse_joints`/`multibody_joints`
handles it edits — those are the Rapier types you'd swap. Steps 1–3, 5–7 are
host code you write against your engine, following the shapes in
`blast/blast-stress-demo-rs/src/main.rs`.

---

## 5. Porting checklist

To bring up the pipeline on a new engine, provide an equivalent of
`BodyTracker` + `collision_groups` + the event wiring that satisfies:

- [ ] Create dynamic & fixed bodies at a pose; remove bodies with collider/joint
      cascade (§2.1)
- [ ] Stable, sortable body & collider handles for determinism (§2.1)
- [ ] Flip body type dynamic↔fixed in place; wake bodies (§2.2)
- [ ] Create cuboid & convex-hull colliders parented at a local pose (§2.3)
- [ ] **Re-parent a collider to a different body** and set its local pose (§2.3)
- [ ] Per-collider mass; recompute body mass properties; read center of mass (§2.3)
- [ ] Read/write pose & velocities; point-velocity at a world point (§2.4)
- [ ] Collision-start + contact-force events carrying collider identities; map
      collider → parent body (§2.5)
- [ ] Reproduce the two correctness passes: COM-shift reconciliation, and
      recenter-new-but-not-reused (§3)
- [ ] *(optional)* Interaction groups + a contact-pair filter hook (§2.6)
- [ ] *(optional)* Damping, sleep thresholds, CCD, impulses (§2.7)
- [ ] *(optional)* Full body-state snapshot/restore + deterministic step for
      resimulation (§2.8)

Everything else — the stress solve, actor/island bookkeeping, fracture command
generation, split planning (`split_migrator`), and the rigid-motion fit
(`motion_fit`) — is already engine-independent and reused as-is.

---

## 6. PhysX 5 C++ implementation

The repository now contains a native implementation of this contract:

- public adapter:
  `blast/include/extensions/stressphysx/NvBlastExtStressPhysX.h`
- implementation:
  `blast/source/sdk/extensions/stressphysx/NvBlastExtStressPhysX.cpp`
- CPU/GPU host and benchmark:
  `demos/blast-stress-demo`

The implementation maps the contract to PhysX as follows:

- Every Blast actor owns one `PxRigidDynamic`; support-containing actors use
  `PxRigidBodyFlag::eKINEMATIC`, which can be toggled in place when a split
  changes support membership.
- Every node owns one stable `PxShape`. A split calls `detachShape(..., false)`,
  computes the new parent-local pose from the captured world pose, and calls
  `attachShape` on the assigned child body. The shape is not recooked or
  recreated during migration.
- Cuboids use `PxBoxGeometry`. Convexes are cooked with
  `PxCookingParams::buildGPUData = true` and reject more than 64 input points;
  the ScenePack loader performs deterministic point reduction before adapter
  creation.
- Child mass, COM, mass frame, and inertia are recomputed from node geometry
  with `PxMassProperties::sum`, then written with `setMass`,
  `setCMassLocalPose`, and `setMassSpaceInertiaTensor`.
- Parent shape world poses and point velocities are captured before topology
  edits. Child state is fitted from those samples, with explicit telemetry for
  maximum world-position and point-velocity drift.
- `PxSimulationEventCallback` contact points/impulses are routed back to the
  owning adapter shape, queued, and injected into the stress graph before the
  next solve.
- Stable adapter-owned 64-bit body and shape IDs provide deterministic mapping
  independent of PhysX pointer values. Snapshot APIs expose the resulting
  bodies/shapes for rendering and validation.
- §2.8 snapshot/restore is implemented per destructible
  (`captureResimulationSnapshot`/`restoreResimulationSnapshot`, keyed by stable
  body ID) and orchestrated by the library-owned `ExtStressPhysXFrameStepper`
  (`NvBlastExtStressPhysXResim.h`), which owns the full §4 loop:
  simulate/fetchResults → tick → on fracture, rollback + re-step up to
  `maxPasses` times, skipping the dead re-capture on the final pass. The
  scene-wide rollback restores every `PxRigidDynamic` that existed at capture
  (host projectiles included — the Rapier "every non-fixed body" semantics);
  restore order matches the Rust reference (pose → velocities → force/torque
  clear → sleep state), with two PhysX-specific rules: kinematic bodies are
  pose-only (velocity writes are rejected), and the stored linvel is
  re-expressed at the current COM (`linvel += ω × ΔCOM`) because a split moves
  a reused body's mass frame. Bodies created since the capture carry
  provenance (source parent + parent-relative pose) and are re-derived from
  their parent's restored state, the TS `restoreCreatedBodyFromSource`
  behavior. Excess forces and the separation impulse are disabled by the host
  when resimulation is on — the re-solved contact is the momentum source.
- GPU mode validates a `PxCudaContextManager`, enables
  `PxSceneFlag::eENABLE_GPU_DYNAMICS`, selects `PxBroadPhaseType::eGPU`, cooks
  GPU convex data, and sizes fixed GPU buffers from scene capacity.

### Implementation status against the checklist

Implemented: dynamic/kinematic body creation and removal, stable IDs, in-place
support mutation, cuboid/convex shape creation, shape migration and local-pose
updates, aggregate mass properties, pose/velocity/sleep access, contact impulse
feedback, body wake-up, continuity checks, §2.8 body-state snapshot/restore
with fracture-frame resimulation (`ExtStressPhysXFrameStepper`), CPU contract
tests including a behavioral penetrate-vs-deflect resimulation probe, and
strict GPU activation/capacity health checks.

Deferred: sibling contact grace, debris collision tiers/TTL cleanup, and body
pooling. Scoped (island-limited) resimulation is implemented in
`ExtStressPhysXFrameStepper` (`scopedResim`, default on) using the island-exact
algorithm: pre-fracture dynamic contact components seeded by this tick's
fracture bodies; settled outsiders skip restore and sleep through the re-step;
missing-seed graphs fall back to full-scene restore. Quiet-frame capture skip,
snapshot buffer reuse, and optional `baseStepSleep` are available. Direct GPU
batched motion state (`ExtStressPhysXDirectGpuMotionBuffer`) and contact-pair
drain (`ExtStressPhysXDirectGpuContactDrain`) are available behind
`useDirectGpuMotionState` / a Direct-GPU scene; topology edits still use the
CPU mutation window, and production city demos keep CPU `onContact` inject
unless the host opts into the Direct GPU path.

Determinism caveat: the restore itself is exact, but PhysX solver warm-start
and contact caches do not survive a rollback, so a re-stepped frame is
output-faithful rather than bit-identical (and GPU PhysX is not bit-reproducible
regardless). Tests assert behavioral outcomes, not bit equality — per the
product guidance, do not chase it.

The older `getBodySnapshots`/`getShapeSnapshots` APIs remain read-only
integration output for rendering/validation and are unrelated to §2.8.

For reproducible commands, recorder details, measured GPU/CPU results, and
known limits, see `demos/blast-stress-demo/README.md`.
