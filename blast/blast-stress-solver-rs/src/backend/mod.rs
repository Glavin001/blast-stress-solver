//! The physics-engine contract.
//!
//! A destruction pipeline needs very little from an engine, and that little is
//! the same on every engine worth targeting. Everything above it — split
//! planning, the rigid-motion fit, resimulation orchestration, debris policy,
//! island bookkeeping — is arithmetic over opaque handles and belongs in the
//! core, written once.
//!
//! # Implementing a new backend
//!
//! Implement [`PhysicsBackend`]: five batched reads, one batched mutation
//! entry point, a step, and a contact drain. Advertise what your engine can do
//! via [`Capabilities`]; the core supplies emulations for what it cannot, and
//! refuses — loudly, naming the missing bit — the features that genuinely
//! cannot work. Pass `Profile::minimal` in the conformance corpus and the rest
//! of the feature set unlocks as you add capability bits.
//!
//! # Two rules that are easy to get wrong
//!
//! **Writes must be elided when they change nothing.** Writing a rigid-body
//! property wakes a sleeping actor on both target engines. Re-stamping
//! unchanged values every frame previously held ~600 of ~735 chunk bodies
//! permanently awake, and separately kept 94% of a 14k-body debris field awake
//! indefinitely. [`Applied::writes_elided`] exists so a regression here is
//! visible rather than merely slow.
//!
//! **`read_center_of_mass` is only valid after `recompute_mass`.** Both target
//! engines defer mass updates: Rapier's `center_of_mass()` returns a cached
//! `world_com` that neither attaching nor detaching a collider refreshes. Read
//! it without the explicit recompute and the centre-of-mass correction reads
//! stale values and silently becomes a no-op — which is the "fragment lurches
//! after it fractures" bug.

pub mod caps;
pub mod conformance;
pub mod scene;
pub mod commands;
pub mod handle;
pub mod math;
pub mod soa;

pub use caps::{Capabilities, Unsupported};
pub use commands::{
    Applied, BodyKind, CommandBuffer, CommandResults, CreateBody, CreateShape, InteractionGroups,
    Phase, ReparentShape, ShapeGeom,
};
pub use handle::{BackendHandle, OpaqueId};
pub use math::{Pose, Quat};
pub use soa::{BodyFlags, BodyStateSoa, Contact, ContactBatch};

use crate::types::Vec3;

/// Something the backend could not do.
#[derive(Clone, Debug)]
pub enum BackendError {
    /// The engine rejected an edit (e.g. an invalid shape migration).
    Rejected(String),
    /// A handle referred to something that no longer exists.
    StaleHandle,
    /// An optional capability was required by this call.
    Unsupported(Unsupported),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Rejected(m) => write!(f, "backend rejected an edit: {m}"),
            BackendError::StaleHandle => write!(f, "stale handle"),
            BackendError::Unsupported(u) => write!(f, "{u}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// An opaque token for a captured motion snapshot.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SnapshotToken(pub u64);

/// The contract every engine adapter implements.
pub trait PhysicsBackend {
    type BodyId: BackendHandle;
    type ShapeId: BackendHandle;

    /// What this backend can do. Read once at attach.
    fn capabilities(&self) -> Capabilities;

    // ---- batched reads (sparse gather by handle list) ----

    /// Gather pose, velocities, flags and mass for `ids` into `out`.
    fn read_bodies(&self, ids: &[Self::BodyId], out: &mut BodyStateSoa);

    /// World-space centre of mass for `ids`.
    ///
    /// Only meaningful after a `recompute_mass` command has landed for those
    /// bodies — see the module note.
    fn read_center_of_mass(&self, ids: &[Self::BodyId], out: &mut Vec<Vec3>);

    /// Owning body of each shape, if any.
    fn shape_parent(&self, shapes: &[Self::ShapeId], out: &mut Vec<Option<Self::BodyId>>);

    /// Velocity at a world point on a body: `v + ω × (p − com)`.
    ///
    /// The default derivation is exact for a rigid body, so overriding it is
    /// only worthwhile when the engine has a native call (advertise
    /// [`Capabilities::NATIVE_POINT_VELOCITY`]).
    fn read_point_velocities(&self, queries: &[(Self::BodyId, Vec3)], out: &mut Vec<Vec3>) {
        out.clear();
        out.reserve(queries.len());
        let ids: Vec<Self::BodyId> = queries.iter().map(|(b, _)| *b).collect();
        let mut states = BodyStateSoa::default();
        let mut coms = Vec::new();
        self.read_bodies(&ids, &mut states);
        self.read_center_of_mass(&ids, &mut coms);
        for (i, (_, p)) in queries.iter().enumerate() {
            let r = *p - coms[i];
            out.push(states.linvel[i] + states.angvel[i].cross(r));
        }
    }

    // ---- the single mutation entry point ----

    /// Apply one phase of edits.
    ///
    /// Implementations must honour three things:
    /// 1. **Submission order** for body creation. The pipeline sorts parents
    ///    before creating so handles are allocated reproducibly; reordering,
    ///    deduplicating or parallelising creates breaks run-to-run determinism.
    /// 2. **Prefix application.** Budgets may cut a buffer short; report what
    ///    landed in [`Applied`] and the pipeline re-queues the rest.
    /// 3. **Write elision.** Skip writes whose value is already current, and
    ///    count them in [`Applied::writes_elided`].
    fn apply(
        &mut self,
        phase: Phase,
        cmds: &CommandBuffer<Self::BodyId, Self::ShapeId>,
        out: &mut CommandResults<Self::BodyId, Self::ShapeId>,
    ) -> Result<Applied, BackendError>;

    // ---- stepping and feedback ----

    /// Advance the engine by `dt`.
    fn step(&mut self, dt: f32);

    /// Drain contacts recorded during the last step.
    fn drain_contacts(&mut self, out: &mut ContactBatch<Self::ShapeId>) -> usize;

    /// Visit every dynamic body the engine knows about.
    ///
    /// Used only by whole-world operations (debris cleanup, unscoped
    /// rollback). Under bring-your-own-world these see host bodies too, which
    /// is exactly why the pipeline scopes them by default.
    fn for_each_dynamic_body(&self, f: &mut dyn FnMut(Self::BodyId));

    // ---- optional: pair exclusion (sibling grace) ----

    /// Suppress contacts between these body pairs for the next step.
    ///
    /// Freshly split siblings are born overlapping; without this the solver
    /// resolves a large interpenetration and blows them apart. Collision groups
    /// cannot substitute — they are per-shape bitmasks and cannot express an
    /// arbitrary pairwise exclusion between two bodies that must both still
    /// collide with everything else.
    fn set_excluded_pairs(&mut self, _pairs: &[(Self::BodyId, Self::BodyId)]) -> Result<(), Unsupported> {
        Err(Unsupported(Capabilities::PAIR_EXCLUSION))
    }

    // ---- optional: resimulation ----

    /// Capture motion for `scope` (empty means every non-fixed body).
    fn capture_motion(&mut self, _scope: &[Self::BodyId]) -> Result<SnapshotToken, Unsupported> {
        Err(Unsupported(Capabilities::MOTION_SNAPSHOT))
    }

    /// Restore a snapshot, optionally limited to `scope`.
    ///
    /// Bodies created since the capture must be **skipped, not errored** —
    /// that is what lets new fragments survive a rollback.
    fn restore_motion(&mut self, _t: SnapshotToken, _scope: &[Self::BodyId]) -> Result<(), Unsupported> {
        Err(Unsupported(Capabilities::MOTION_SNAPSHOT))
    }

    fn release_snapshot(&mut self, _t: SnapshotToken) {}
}

/// Check a backend against what the core requires.
///
/// Called once, at attach. The error names the missing bits so a half-built
/// adapter fails with a to-do list rather than a panic three layers down.
pub fn check_required(caps: Capabilities) -> Result<(), MissingCapabilities> {
    let missing = Capabilities::REQUIRED.difference(caps);
    if missing.is_empty() {
        Ok(())
    } else {
        Err(MissingCapabilities(missing))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissingCapabilities(pub Capabilities);

impl std::fmt::Display for MissingCapabilities {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "backend is missing required capabilities: {:?}", self.0.names())
    }
}

impl std::error::Error for MissingCapabilities {}
