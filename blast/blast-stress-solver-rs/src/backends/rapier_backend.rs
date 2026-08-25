//! Rapier 3D implementation of [`PhysicsBackend`].
//!
//! This is the reference adapter: it exists as much to prove the contract is
//! implementable thinly as to ship. Everything interesting about destruction —
//! split planning, the rigid-motion fit, resim orchestration, debris policy —
//! lives in the core and is not repeated here.
//!
//! Rapier specifics the contract has to accommodate, all handled at this
//! boundary and invisible above it:
//!
//! - `center_of_mass()` returns a **cached** `world_com` that no topology edit
//!   refreshes (attaching accumulates `local_mprops` eagerly, detaching never
//!   subtracts, and the fix-up runs during the step). So `recompute_mass` maps
//!   to a real `recompute_mass_properties_from_colliders` call.
//! - Contact force events are **level-triggered**: Rapier re-evaluates every
//!   pair every step against the threshold, so `persisting` is always true and
//!   no extra flag is needed. This is what PhysX needs `_PERSISTS` for.
//! - Rapier reports **force** (it has already divided the impulse by `dt`),
//!   unlike PhysX which reports the raw impulse.
//! - `set_enabled` couples simulation and scene-query visibility, so this
//!   backend advertises `SHAPE_SIMULATION_TOGGLE` but **not**
//!   `SHAPE_QUERY_TOGGLE`.

use std::sync::mpsc::{channel, Receiver};

use rapier3d::na::{Isometry3, Translation3, UnitQuaternion, Vector3};
use rapier3d::prelude::*;

use crate::backend::{
    check_required, Applied, BackendError, BackendHandle, BodyFlags, BodyKind, BodyStateSoa,
    Capabilities, CommandBuffer, CommandResults, Contact, ContactBatch, MissingCapabilities, Phase,
    PhysicsBackend, Pose, Quat, ShapeGeom, SnapshotToken, Unsupported,
};
use crate::types::Vec3;

impl BackendHandle for RigidBodyHandle {
    fn sort_key(self) -> u64 {
        let (i, g) = self.into_raw_parts();
        ((g as u64) << 32) | i as u64
    }
}

impl BackendHandle for ColliderHandle {
    fn sort_key(self) -> u64 {
        let (i, g) = self.into_raw_parts();
        ((g as u64) << 32) | i as u64
    }
}

// ---- conversions, confined to this file ----

fn to_iso(p: Pose) -> Isometry3<Real> {
    let q = UnitQuaternion::from_quaternion(rapier3d::na::Quaternion::new(
        p.rotation.w,
        p.rotation.x,
        p.rotation.y,
        p.rotation.z,
    ));
    Isometry3::from_parts(Translation3::new(p.translation.x, p.translation.y, p.translation.z), q)
}

fn from_iso(i: &Isometry3<Real>) -> Pose {
    let q = i.rotation.quaternion();
    Pose {
        translation: Vec3::new(i.translation.x, i.translation.y, i.translation.z),
        rotation: Quat::new(q.i, q.j, q.k, q.w),
    }
}

fn v(x: Vec3) -> Vector3<Real> {
    Vector3::new(x.x, x.y, x.z)
}

fn fv(x: &Vector3<Real>) -> Vec3 {
    Vec3::new(x.x, x.y, x.z)
}

/// A Rapier world the library drives end to end.
///
/// For bring-your-own-world, see [`RapierBackend`], which borrows a host's sets
/// instead of owning them.
pub struct RapierWorld {
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub islands: IslandManager,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pipeline: PhysicsPipeline,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    ccd: CCDSolver,
    pub integration: IntegrationParameters,
    pub gravity: Vector3<Real>,
    contact_force_rx: Receiver<ContactForceEvent>,
    collision_rx: Receiver<CollisionEvent>,
    collector: ChannelEventCollector,
    /// Bodies whose contacts are suppressed this step (sibling grace).
    excluded: std::collections::HashSet<(u64, u64)>,
    snapshots: Vec<(SnapshotToken, Vec<BodyMotion>)>,
    next_token: u64,
}

#[derive(Clone, Copy)]
struct BodyMotion {
    handle: RigidBodyHandle,
    pose: Isometry3<Real>,
    linvel: Vector3<Real>,
    angvel: Vector3<Real>,
    linear_damping: Real,
    angular_damping: Real,
    sleeping: bool,
    enabled: bool,
}

impl Default for RapierWorld {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, -9.81, 0.0))
    }
}

impl RapierWorld {
    pub fn new(gravity: Vec3) -> Self {
        let (cf_tx, contact_force_rx) = channel();
        let (col_tx, collision_rx) = channel();
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            islands: IslandManager::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            pipeline: PhysicsPipeline::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            ccd: CCDSolver::new(),
            integration: IntegrationParameters::default(),
            gravity: v(gravity),
            collector: ChannelEventCollector::new(col_tx, cf_tx),
            contact_force_rx,
            collision_rx,
            excluded: std::collections::HashSet::new(),
            snapshots: Vec::new(),
            next_token: 1,
        }
    }

    /// Fail fast if this backend cannot meet the core's requirements.
    pub fn check(&self) -> Result<(), MissingCapabilities> {
        check_required(self.capabilities())
    }

    fn kind_of(body: &RigidBody) -> BodyKind {
        if body.is_fixed() {
            BodyKind::Fixed
        } else if body.is_kinematic() {
            BodyKind::Kinematic
        } else {
            BodyKind::Dynamic
        }
    }

    fn rapier_kind(k: BodyKind) -> RigidBodyType {
        match k {
            BodyKind::Dynamic => RigidBodyType::Dynamic,
            BodyKind::Fixed => RigidBodyType::Fixed,
            BodyKind::Kinematic => RigidBodyType::KinematicPositionBased,
        }
    }

    fn build_shape(geom: &ShapeGeom) -> Option<ColliderBuilder> {
        match geom {
            ShapeGeom::Cuboid { half_extents } => Some(ColliderBuilder::cuboid(
                half_extents.x.max(1e-4),
                half_extents.y.max(1e-4),
                half_extents.z.max(1e-4),
            )),
            ShapeGeom::ConvexHull { points } => {
                let pts: Vec<Point<Real>> =
                    points.iter().map(|p| point![p.x, p.y, p.z]).collect();
                ColliderBuilder::convex_hull(&pts)
            }
        }
    }
}

impl PhysicsBackend for RapierWorld {
    type BodyId = RigidBodyHandle;
    type ShapeId = ColliderHandle;

    fn capabilities(&self) -> Capabilities {
        Capabilities::BODY_LIFECYCLE
            | Capabilities::BODY_TYPE_MUTATION
            | Capabilities::SHAPE_LIFECYCLE
            | Capabilities::MASS_PROPERTIES
            | Capabilities::POSE_VELOCITY_IO
            | Capabilities::CONTACT_EVENTS
            | Capabilities::DETERMINISTIC_HANDLES
            | Capabilities::REPARENT_SHAPE
            | Capabilities::SHAPE_SIMULATION_TOGGLE
            | Capabilities::PAIR_EXCLUSION
            | Capabilities::COLLISION_GROUPS
            | Capabilities::DAMPING
            | Capabilities::SLEEP_THRESHOLDS
            | Capabilities::CCD
            | Capabilities::IMPULSES
            | Capabilities::MOTION_SNAPSHOT
            | Capabilities::SCOPED_SNAPSHOT
            | Capabilities::NATIVE_POINT_VELOCITY
            | Capabilities::CONTACT_MANIFOLDS
    }

    fn read_bodies(&self, ids: &[Self::BodyId], out: &mut BodyStateSoa) {
        out.reset_for(ids.len());
        for id in ids {
            match self.bodies.get(*id) {
                Some(b) => {
                    let mut flags = BodyFlags::NONE;
                    flags.set(BodyFlags::DYNAMIC, b.is_dynamic());
                    flags.set(BodyFlags::KINEMATIC, b.is_kinematic());
                    flags.set(BodyFlags::SLEEPING, b.is_sleeping());
                    flags.set(BodyFlags::ENABLED, b.is_enabled());
                    out.push(from_iso(b.position()), fv(b.linvel()), fv(b.angvel()), flags, b.mass());
                }
                None => out.push(Pose::IDENTITY, Vec3::ZERO, Vec3::ZERO, BodyFlags::NONE, 0.0),
            }
        }
    }

    fn read_center_of_mass(&self, ids: &[Self::BodyId], out: &mut Vec<Vec3>) {
        out.clear();
        out.reserve(ids.len());
        for id in ids {
            out.push(match self.bodies.get(*id) {
                Some(b) => fv(&b.center_of_mass().coords),
                None => Vec3::ZERO,
            });
        }
    }

    fn shape_parent(&self, shapes: &[Self::ShapeId], out: &mut Vec<Option<Self::BodyId>>) {
        out.clear();
        out.reserve(shapes.len());
        for s in shapes {
            out.push(self.colliders.get(*s).and_then(|c| c.parent()));
        }
    }

    fn read_point_velocities(&self, queries: &[(Self::BodyId, Vec3)], out: &mut Vec<Vec3>) {
        out.clear();
        out.reserve(queries.len());
        for (b, p) in queries {
            out.push(match self.bodies.get(*b) {
                Some(body) => fv(&body.velocity_at_point(&point![p.x, p.y, p.z])),
                None => Vec3::ZERO,
            });
        }
    }

    fn apply(
        &mut self,
        phase: Phase,
        cmds: &CommandBuffer<Self::BodyId, Self::ShapeId>,
        out: &mut CommandResults<Self::BodyId, Self::ShapeId>,
    ) -> Result<Applied, BackendError> {
        let mut done = Applied::default();
        out.clear();

        // Creates run in strict submission order: the pipeline pre-sorts so
        // that Rapier allocates handles reproducibly.
        for cb in &cmds.create_bodies {
            let mut builder = match cb.kind {
                BodyKind::Dynamic => RigidBodyBuilder::dynamic(),
                BodyKind::Fixed => RigidBodyBuilder::fixed(),
                BodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
            }
            .position(to_iso(cb.pose))
            .ccd_enabled(cb.ccd);
            if matches!(cb.kind, BodyKind::Dynamic) {
                builder = builder.linvel(v(cb.linvel)).angvel(v(cb.angvel));
            }
            let h = self.bodies.insert(builder.build());
            if cb.start_sleeping {
                if let Some(b) = self.bodies.get_mut(h) {
                    b.sleep();
                }
            }
            out.created_bodies.push(h);
            done.bodies_created += 1;
        }

        for (h, kind) in &cmds.set_body_kind {
            if let Some(b) = self.bodies.get_mut(*h) {
                let want = Self::rapier_kind(*kind);
                if b.body_type() != want {
                    b.set_body_type(want, true);
                } else {
                    done.writes_elided += 1;
                }
            }
        }

        for (h, pose) in &cmds.set_pose {
            if let Some(b) = self.bodies.get_mut(*h) {
                b.set_position(to_iso(*pose), false);
            }
        }

        for (h, lin, ang) in &cmds.set_velocity {
            if let Some(b) = self.bodies.get_mut(*h) {
                // Wake only when the value actually moved: a wake re-opens the
                // whole contact island for at least the sleep timer.
                let changed = (fv(b.linvel()) - *lin).magnitude_squared() > 1e-12
                    || (fv(b.angvel()) - *ang).magnitude_squared() > 1e-12;
                if changed {
                    b.set_linvel(v(*lin), false);
                    b.set_angvel(v(*ang), false);
                } else {
                    done.writes_elided += 1;
                }
            }
        }

        for cs in &cmds.create_shapes {
            let Some(builder) = Self::build_shape(&cs.geom) else {
                return Err(BackendError::Rejected("degenerate convex hull".into()));
            };
            let col = builder
                .position(to_iso(cs.local))
                .mass(cs.mass.max(0.0))
                .friction(0.25)
                .restitution(0.0)
                .active_events(ActiveEvents::CONTACT_FORCE_EVENTS | ActiveEvents::COLLISION_EVENTS)
                .contact_force_event_threshold(0.0)
                .active_hooks(ActiveHooks::FILTER_CONTACT_PAIRS)
                .build();
            let h = self.colliders.insert_with_parent(col, cs.body, &mut self.bodies);
            out.created_shapes.push(h);
            done.shapes_created += 1;
        }

        // A true re-parent keeps the handle and its contact history. This is
        // the edit the whole split planner optimises for.
        for rp in &cmds.reparent_shapes {
            self.colliders.set_parent(rp.shape, Some(rp.body), &mut self.bodies);
            if let Some(c) = self.colliders.get_mut(rp.shape) {
                c.set_position_wrt_parent(to_iso(rp.local));
            }
            done.shapes_reparented += 1;
        }

        for (s, local) in &cmds.set_shape_local {
            if let Some(c) = self.colliders.get_mut(*s) {
                c.set_position_wrt_parent(to_iso(*local));
            }
        }

        for s in &cmds.remove_shapes {
            self.colliders.remove(*s, &mut self.islands, &mut self.bodies, true);
        }

        // Must be explicit: Rapier's cached `world_com` is not refreshed by
        // attach or detach, so reading the COM without this yields stale data.
        for h in &cmds.recompute_mass {
            let colliders = &self.colliders;
            if let Some(b) = self.bodies.get_mut(*h) {
                b.recompute_mass_properties_from_colliders(colliders);
            }
        }

        for (h, lin, ang) in &cmds.apply_impulse {
            if let Some(b) = self.bodies.get_mut(*h) {
                if b.is_dynamic() {
                    b.apply_impulse(v(*lin), true);
                    b.apply_torque_impulse(v(*ang), true);
                }
            }
        }

        for (h, l, a) in &cmds.set_damping {
            if let Some(b) = self.bodies.get_mut(*h) {
                if (b.linear_damping() - l).abs() > Real::EPSILON {
                    b.set_linear_damping(*l);
                } else {
                    done.writes_elided += 1;
                }
                if (b.angular_damping() - a).abs() > Real::EPSILON {
                    b.set_angular_damping(*a);
                } else {
                    done.writes_elided += 1;
                }
            }
        }

        for (h, lin, ang) in &cmds.set_sleep_thresholds {
            if let Some(b) = self.bodies.get_mut(*h) {
                let act = b.activation_mut();
                if (act.normalized_linear_threshold - lin).abs() > Real::EPSILON {
                    act.normalized_linear_threshold = lin.max(0.0);
                } else {
                    done.writes_elided += 1;
                }
                if (act.angular_threshold - ang).abs() > Real::EPSILON {
                    act.angular_threshold = ang.max(0.0);
                } else {
                    done.writes_elided += 1;
                }
            }
        }

        for (h, on) in &cmds.set_ccd {
            if let Some(b) = self.bodies.get_mut(*h) {
                if b.is_ccd_enabled() != *on {
                    b.enable_ccd(*on);
                } else {
                    done.writes_elided += 1;
                }
            }
        }

        for (s, g) in &cmds.set_groups {
            if let Some(c) = self.colliders.get_mut(*s) {
                let groups = InteractionGroups::new(
                    Group::from_bits_truncate(g.memberships),
                    Group::from_bits_truncate(g.filter),
                );
                if c.collision_groups() != groups {
                    c.set_collision_groups(groups);
                    c.set_solver_groups(groups);
                } else {
                    done.writes_elided += 1;
                }
            }
        }

        for (s, on) in &cmds.set_shape_enabled {
            if let Some(c) = self.colliders.get_mut(*s) {
                if c.is_enabled() != *on {
                    c.set_enabled(*on);
                } else {
                    done.writes_elided += 1;
                }
            }
        }

        for h in &cmds.wake {
            if let Some(b) = self.bodies.get_mut(*h) {
                if b.is_sleeping() {
                    b.wake_up(true);
                } else {
                    done.writes_elided += 1;
                }
            }
        }
        for h in &cmds.sleep {
            if let Some(b) = self.bodies.get_mut(*h) {
                if !b.is_sleeping() {
                    b.sleep();
                } else {
                    done.writes_elided += 1;
                }
            }
        }

        // Retirement last: removing a body cascades its colliders, so every
        // migration off it must already have landed.
        if matches!(phase, Phase::Retire) || !cmds.remove_bodies.is_empty() {
            for h in &cmds.remove_bodies {
                if self
                    .bodies
                    .remove(
                        *h,
                        &mut self.islands,
                        &mut self.colliders,
                        &mut self.impulse_joints,
                        &mut self.multibody_joints,
                        true,
                    )
                    .is_some()
                {
                    done.bodies_removed += 1;
                }
            }
        }

        Ok(done)
    }

    fn step(&mut self, dt: f32) {
        self.integration.dt = dt;
        let hooks = SiblingGrace { excluded: &self.excluded };
        self.pipeline.step(
            &self.gravity,
            &self.integration,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &hooks,
            &self.collector,
        );
    }

    fn drain_contacts(&mut self, out: &mut ContactBatch<Self::ShapeId>) -> usize {
        out.clear();
        // Collision events are drained so support-contact marking works; the
        // stress feed itself comes from the force events below.
        while self.collision_rx.try_recv().is_ok() {}
        while let Ok(ev) = self.contact_force_rx.try_recv() {
            let dir = ev.max_force_direction;
            out.contacts.push(Contact {
                shape_a: ev.collider1,
                shape_b: Some(ev.collider2),
                // Rapier's force event carries no contact point; the manifold
                // does. Callers that need the point (crush) read manifolds via
                // `narrow_phase`; this keeps the common path allocation-free.
                world_position: Vec3::ZERO,
                normal: Vec3::new(dir.x, dir.y, dir.z),
                force: ev.total_force_magnitude,
                relative_velocity: Vec3::ZERO,
                // Rapier re-evaluates every pair every step against the
                // threshold, so any event we see is by definition a contact
                // still carrying load.
                persisting: true,
            });
        }
        out.len()
    }

    fn for_each_dynamic_body(&self, f: &mut dyn FnMut(Self::BodyId)) {
        for (h, b) in self.bodies.iter() {
            if !b.is_fixed() {
                f(h);
            }
        }
    }

    fn set_excluded_pairs(&mut self, pairs: &[(Self::BodyId, Self::BodyId)]) -> Result<(), Unsupported> {
        self.excluded.clear();
        for (a, b) in pairs {
            let (x, y) = (a.sort_key(), b.sort_key());
            self.excluded.insert((x.min(y), x.max(y)));
        }
        Ok(())
    }

    fn capture_motion(&mut self, scope: &[Self::BodyId]) -> Result<SnapshotToken, Unsupported> {
        let mut motions = Vec::new();
        let mut push = |h: RigidBodyHandle, b: &RigidBody| {
            motions.push(BodyMotion {
                handle: h,
                pose: *b.position(),
                linvel: *b.linvel(),
                angvel: *b.angvel(),
                linear_damping: b.linear_damping(),
                angular_damping: b.angular_damping(),
                sleeping: b.is_sleeping(),
                enabled: b.is_enabled(),
            })
        };
        if scope.is_empty() {
            for (h, b) in self.bodies.iter() {
                if !b.is_fixed() {
                    push(h, b);
                }
            }
        } else {
            for h in scope {
                if let Some(b) = self.bodies.get(*h) {
                    if !b.is_fixed() {
                        push(*h, b);
                    }
                }
            }
        }
        let token = SnapshotToken(self.next_token);
        self.next_token += 1;
        self.snapshots.push((token, motions));
        Ok(token)
    }

    fn restore_motion(&mut self, t: SnapshotToken, scope: &[Self::BodyId]) -> Result<(), Unsupported> {
        let Some((_, motions)) = self.snapshots.iter().find(|(tok, _)| *tok == t) else {
            return Ok(());
        };
        let scoped: Option<std::collections::HashSet<RigidBodyHandle>> =
            (!scope.is_empty()).then(|| scope.iter().copied().collect());
        let motions = motions.clone();
        for m in motions {
            if let Some(set) = &scoped {
                if !set.contains(&m.handle) {
                    continue;
                }
            }
            // Bodies created since the capture are absent here and are simply
            // skipped — that is what lets new fragments survive a rollback.
            let Some(b) = self.bodies.get_mut(m.handle) else { continue };
            b.set_enabled(m.enabled);
            b.set_position(m.pose, false);
            b.set_linvel(m.linvel, false);
            b.set_angvel(m.angvel, false);
            b.set_linear_damping(m.linear_damping);
            b.set_angular_damping(m.angular_damping);
            b.reset_forces(false);
            b.reset_torques(false);
            // Sleep state last, so the setters above cannot clobber it.
            if m.sleeping {
                b.sleep();
            } else {
                b.wake_up(false);
            }
        }
        Ok(())
    }

    fn release_snapshot(&mut self, t: SnapshotToken) {
        self.snapshots.retain(|(tok, _)| *tok != t);
    }
}

/// Rejects contact pairs listed in the exclusion set before the narrow phase
/// produces a manifold — the `PreManifold` flavour of pair exclusion.
struct SiblingGrace<'a> {
    excluded: &'a std::collections::HashSet<(u64, u64)>,
}

impl PhysicsHooks for SiblingGrace<'_> {
    fn filter_contact_pair(&self, ctx: &PairFilterContext) -> Option<SolverFlags> {
        if self.excluded.is_empty() {
            return Some(SolverFlags::default());
        }
        let (a, b) = (ctx.rigid_body1?, ctx.rigid_body2?);
        let (x, y) = (a.sort_key(), b.sort_key());
        if self.excluded.contains(&(x.min(y), x.max(y))) {
            None
        } else {
            Some(SolverFlags::default())
        }
    }
}
