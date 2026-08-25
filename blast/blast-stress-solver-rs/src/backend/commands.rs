//! Batched, phase-ordered mutation commands.
//!
//! Batching is not a micro-optimisation here. With GPU rigid bodies every
//! per-actor pose read is a device readback, so the engines this targets punish
//! per-body chatter; and PhysX's own Direct-GPU API is already shaped this way
//! (`capture(bodies, count)`), which is the strongest evidence the shape is
//! right. A per-body trait would make the fast paths impossible to express.

use super::math::Pose;
use crate::types::Vec3;

/// Which pass of a topology edit a buffer belongs to.
///
/// The split is not stylistic — it makes an ordering constraint structural.
/// `Motion` may only run after every `Topology` reparent has landed *and* mass
/// has been recomputed, because a reused parent body still holding a sibling's
/// shape reports the wrong centre of mass. That rule is currently a prose
/// comment repeated at three call sites; here the type enforces it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Create/recycle bodies, migrate shapes, flip body types, recompute mass.
    Topology,
    /// Child pose and velocity, then the COM-corrected linear velocity.
    Motion,
    /// Damping, sleep thresholds, CCD, collision groups.
    Tuning,
    /// Remove bodies emptied by the edit. Must follow every migration off them.
    Retire,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BodyKind {
    Dynamic,
    /// Immovable. A body owning any support (zero-mass) node is fixed.
    Fixed,
    /// Movable by the host but not by the solver. PhysX models supports this
    /// way so the flag can be flipped in place.
    Kinematic,
}

/// Collider geometry the pipeline can ask for.
#[derive(Clone, Debug)]
pub enum ShapeGeom {
    Cuboid { half_extents: Vec3 },
    ConvexHull { points: Vec<Vec3> },
}

#[derive(Clone, Debug)]
pub struct CreateBody {
    pub pose: Pose,
    pub kind: BodyKind,
    pub linvel: Vec3,
    pub angvel: Vec3,
    pub ccd: bool,
    /// Fragments of a sleeping structure are born asleep — a large saving on
    /// collapse-then-settle scenes.
    pub start_sleeping: bool,
}

#[derive(Clone, Debug)]
pub struct CreateShape<B> {
    pub body: B,
    pub local: Pose,
    pub geom: ShapeGeom,
    /// Support nodes contribute zero mass so the engine-derived centre of mass
    /// matches the node-mass model.
    pub mass: f32,
    /// Stress-graph node this shape represents.
    pub node: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ReparentShape<B, S> {
    pub shape: S,
    pub body: B,
    pub local: Pose,
}

#[derive(Clone, Copy, Debug)]
pub struct InteractionGroups {
    pub memberships: u32,
    pub filter: u32,
    /// Host entity id carried alongside the groups.
    ///
    /// Engines pack an application id into the same filter payload as the
    /// groups -- PhysX puts both in one `PxFilterData` -- and a host that
    /// raycasts its own scene needs the shape to answer "which entity am I".
    /// Zero means "the host did not say", which is what a backend writes when
    /// the library creates a shape the host never claimed.
    pub entity: u32,
}

/// One phase's worth of edits, in struct-of-arrays form.
///
/// Every write is expected to be **applied only if it changes something**.
/// Writing a rigid-body property wakes a sleeping actor on both target
/// engines, and re-stamping unchanged values every frame is what previously
/// held ~600 of ~735 chunk bodies permanently awake.
#[derive(Clone, Debug)]
pub struct CommandBuffer<B, S> {
    pub create_bodies: Vec<CreateBody>,
    pub set_body_kind: Vec<(B, BodyKind)>,
    pub set_pose: Vec<(B, Pose)>,
    pub set_velocity: Vec<(B, Vec3, Vec3)>,
    pub create_shapes: Vec<CreateShape<B>>,
    pub reparent_shapes: Vec<ReparentShape<B, S>>,
    pub set_shape_local: Vec<(S, Pose)>,
    pub remove_shapes: Vec<S>,
    /// Explicit because both engines defer mass updates: Rapier's
    /// `center_of_mass()` reads a cached `world_com` that no topology edit
    /// refreshes. Without this command the COM correction silently reads stale
    /// values and becomes a no-op.
    pub recompute_mass: Vec<B>,
    pub remove_bodies: Vec<B>,
    pub wake: Vec<B>,
    pub sleep: Vec<B>,
    pub set_damping: Vec<(B, f32, f32)>,
    pub set_sleep_thresholds: Vec<(B, f32, f32)>,
    pub set_ccd: Vec<(B, bool)>,
    pub set_groups: Vec<(S, InteractionGroups)>,
    pub set_shape_enabled: Vec<(S, bool)>,
    /// Linear and torque impulse. May target bodies the library does not own.
    pub apply_impulse: Vec<(B, Vec3, Vec3)>,
}

impl<B, S> Default for CommandBuffer<B, S> {
    fn default() -> Self {
        Self {
            create_bodies: Vec::new(),
            set_body_kind: Vec::new(),
            set_pose: Vec::new(),
            set_velocity: Vec::new(),
            create_shapes: Vec::new(),
            reparent_shapes: Vec::new(),
            set_shape_local: Vec::new(),
            remove_shapes: Vec::new(),
            recompute_mass: Vec::new(),
            remove_bodies: Vec::new(),
            wake: Vec::new(),
            sleep: Vec::new(),
            set_damping: Vec::new(),
            set_sleep_thresholds: Vec::new(),
            set_ccd: Vec::new(),
            set_groups: Vec::new(),
            set_shape_enabled: Vec::new(),
            apply_impulse: Vec::new(),
        }
    }
}

impl<B, S> CommandBuffer<B, S> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Clear every list but keep the allocations. The pipeline reuses one
    /// buffer across frames so a steady state allocates nothing.
    pub fn clear(&mut self) {
        self.create_bodies.clear();
        self.set_body_kind.clear();
        self.set_pose.clear();
        self.set_velocity.clear();
        self.create_shapes.clear();
        self.reparent_shapes.clear();
        self.set_shape_local.clear();
        self.remove_shapes.clear();
        self.recompute_mass.clear();
        self.remove_bodies.clear();
        self.wake.clear();
        self.sleep.clear();
        self.set_damping.clear();
        self.set_sleep_thresholds.clear();
        self.set_ccd.clear();
        self.set_groups.clear();
        self.set_shape_enabled.clear();
        self.apply_impulse.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.create_bodies.is_empty()
            && self.set_body_kind.is_empty()
            && self.set_pose.is_empty()
            && self.set_velocity.is_empty()
            && self.create_shapes.is_empty()
            && self.reparent_shapes.is_empty()
            && self.set_shape_local.is_empty()
            && self.remove_shapes.is_empty()
            && self.recompute_mass.is_empty()
            && self.remove_bodies.is_empty()
            && self.wake.is_empty()
            && self.sleep.is_empty()
            && self.set_damping.is_empty()
            && self.set_sleep_thresholds.is_empty()
            && self.set_ccd.is_empty()
            && self.set_groups.is_empty()
            && self.set_shape_enabled.is_empty()
            && self.apply_impulse.is_empty()
    }
}

/// Handles minted by an `apply` call, index-parallel to the create lists.
#[derive(Clone, Debug)]
pub struct CommandResults<B, S> {
    pub created_bodies: Vec<B>,
    pub created_shapes: Vec<S>,
}

impl<B, S> Default for CommandResults<B, S> {
    fn default() -> Self {
        Self { created_bodies: Vec::new(), created_shapes: Vec::new() }
    }
}

impl<B, S> CommandResults<B, S> {
    pub fn clear(&mut self) {
        self.created_bodies.clear();
        self.created_shapes.clear();
    }
}

/// How much of a buffer landed.
///
/// Topology edits are budgeted per frame (`max_new_bodies_per_frame`,
/// `max_collider_migrations_per_frame`), so a buffer is applied as a **prefix**
/// and the remainder is carried to the next frame. An all-or-nothing API cannot
/// express that.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    pub bodies_created: usize,
    pub shapes_created: usize,
    pub shapes_reparented: usize,
    pub bodies_removed: usize,
    /// Writes skipped because the value was already current. Tracked because a
    /// rising count is the early warning that a Tuning pass has started waking
    /// sleeping bodies for nothing.
    pub writes_elided: usize,
}
