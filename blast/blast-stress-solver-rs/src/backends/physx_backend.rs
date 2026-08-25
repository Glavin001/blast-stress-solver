//! PhysX 5 implementation of [`PhysicsBackend`], CPU and GPU.
//!
//! The engine work lives in C++ (`blast/physx_backend/physx_backend.cpp`)
//! because that is where the proven convex cooking, mass composition and shape
//! migration already are; this file is the marshalling layer and nothing more.
//!
//! Bodies and shapes are addressed by adapter-minted `u64` ids that are
//! monotone and never reused. That is a deliberate contrast with Rapier, whose
//! generational handles reuse arena slots — with a never-reused id, a stale
//! reference is *detectable* rather than silently aliasing a recycled actor.

use std::ffi::c_void;

use crate::backend::{
    check_required, Applied, BackendError, BackendHandle, BodyFlags, BodyKind, BodyStateSoa,
    Capabilities, CommandBuffer, CommandResults, Contact, ContactBatch, MissingCapabilities, Phase,
    PhysicsBackend, Pose, Quat, ShapeGeom, SnapshotToken, Unsupported,
};
use crate::types::Vec3;

/// Stable PhysX-side body id. Never reused.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct PxBodyId(pub u64);

/// Stable PhysX-side shape id. Never reused.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct PxShapeId(pub u64);

impl BackendHandle for PxBodyId {
    fn sort_key(self) -> u64 {
        self.0
    }
}
impl BackendHandle for PxShapeId {
    fn sort_key(self) -> u64 {
        self.0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CVec3 {
    x: f32,
    y: f32,
    z: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct CQuat {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct CPose {
    p: CVec3,
    q: CQuat,
}

impl Default for CQuat {
    fn default() -> Self {
        CQuat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
    }
}
impl Default for CPose {
    fn default() -> Self {
        CPose { p: CVec3::default(), q: CQuat::default() }
    }
}

fn cv(v: Vec3) -> CVec3 {
    CVec3 { x: v.x, y: v.y, z: v.z }
}
fn rv(v: CVec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}
fn cp(p: Pose) -> CPose {
    CPose {
        p: cv(p.translation),
        q: CQuat { x: p.rotation.x, y: p.rotation.y, z: p.rotation.z, w: p.rotation.w },
    }
}
fn rp(p: CPose) -> Pose {
    Pose { translation: rv(p.p), rotation: Quat::new(p.q.x, p.q.y, p.q.z, p.q.w) }
}

#[repr(C)]
struct CCreateBody {
    pose: CPose,
    linvel: CVec3,
    angvel: CVec3,
    kind: u32,
    ccd: u8,
    start_sleeping: u8,
}

#[repr(C)]
struct CCreateShape {
    body: u64,
    local: CPose,
    half_extents: CVec3,
    points: *const CVec3,
    point_count: u32,
    geom: u32,
    mass: f32,
    node: u32,
}

#[repr(C)]
struct CReparent {
    shape: u64,
    body: u64,
    local: CPose,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CContact {
    shape_a: u64,
    shape_b: u64,
    world_position: CVec3,
    normal: CVec3,
    relative_velocity: CVec3,
    force: f32,
    persisting: u8,
}

#[repr(C)]
#[derive(Default)]
struct CCommands {
    create_bodies: *const CCreateBody,
    create_body_count: u32,
    set_kind_ids: *const u64,
    set_kind_values: *const u32,
    set_kind_count: u32,
    set_pose_ids: *const u64,
    set_pose_values: *const CPose,
    set_pose_count: u32,
    set_vel_ids: *const u64,
    set_vel_lin: *const CVec3,
    set_vel_ang: *const CVec3,
    set_vel_count: u32,
    create_shapes: *const CCreateShape,
    create_shape_count: u32,
    reparent: *const CReparent,
    reparent_count: u32,
    remove_shapes: *const u64,
    remove_shape_count: u32,
    recompute_mass: *const u64,
    recompute_mass_count: u32,
    remove_bodies: *const u64,
    remove_body_count: u32,
    wake: *const u64,
    wake_count: u32,
    sleep: *const u64,
    sleep_count: u32,
    damping_ids: *const u64,
    damping_lin: *const f32,
    damping_ang: *const f32,
    damping_count: u32,
    sleep_thr_ids: *const u64,
    sleep_thr_lin: *const f32,
    sleep_thr_ang: *const f32,
    sleep_thr_count: u32,
    ccd_ids: *const u64,
    ccd_values: *const u8,
    ccd_count: u32,
    // Order matches physx_backend.h exactly; this struct is passed by pointer,
    // so a field out of place is silently reinterpreted memory.
    group_shapes: *const u64,
    group_memberships: *const u32,
    group_filters: *const u32,
    group_entities: *const u32,
    group_count: u32,
    shape_enabled_ids: *const u64,
    shape_enabled_values: *const u8,
    shape_enabled_count: u32,
    impulse_ids: *const u64,
    impulse_lin: *const CVec3,
    impulse_ang: *const CVec3,
    impulse_count: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct CApplied {
    bodies_created: u32,
    shapes_created: u32,
    shapes_reparented: u32,
    bodies_removed: u32,
    writes_elided: u32,
}

extern "C" {
    fn pxb_world_create(gravity: CVec3, gpu: u8, cpu_threads: u32) -> *mut c_void;
    fn pxb_world_attach(scene: *mut c_void, physics: *mut c_void, material: *mut c_void) -> *mut c_void;
    fn pxb_world_is_attached(w: *const c_void) -> u8;
    fn pxb_world_scene(w: *const c_void) -> *mut c_void;
    fn pxb_world_physics(w: *const c_void) -> *mut c_void;
    fn pxb_note_dt(w: *mut c_void, dt: f32);
    fn pxb_inject_contact(
        w: *mut c_void,
        shape_a: *mut c_void,
        shape_b: *mut c_void,
        world_position: CVec3,
        normal: CVec3,
        relative_velocity: CVec3,
        impulse_magnitude: f32,
        persisting: u8,
    ) -> u8;
    fn pxb_world_destroy(w: *mut c_void);
    fn pxb_world_gpu_active(w: *const c_void) -> u8;
    fn pxb_capabilities(w: *const c_void) -> u32;
    fn pxb_read_bodies(
        w: *const c_void,
        ids: *const u64,
        count: u32,
        pose: *mut CPose,
        lin: *mut CVec3,
        ang: *mut CVec3,
        flags: *mut u8,
        mass: *mut f32,
    );
    fn pxb_read_center_of_mass(w: *const c_void, ids: *const u64, count: u32, out: *mut CVec3);
    fn pxb_shape_parent(w: *const c_void, shapes: *const u64, count: u32, out: *mut u64);
    fn pxb_read_point_velocities(
        w: *const c_void,
        ids: *const u64,
        pts: *const CVec3,
        count: u32,
        out: *mut CVec3,
    );
    fn pxb_apply(
        w: *mut c_void,
        phase: u32,
        cmds: *const CCommands,
        out_bodies: *mut u64,
        out_shapes: *mut u64,
        applied: *mut CApplied,
    ) -> u8;
    fn pxb_step(w: *mut c_void, dt: f32);
    fn pxb_drain_contacts(w: *mut c_void, out: *mut CContact, cap: u32) -> u32;
    fn pxb_dynamic_bodies(w: *const c_void, out: *mut u64, cap: u32) -> u32;
    fn pxb_set_excluded_pairs(w: *mut c_void, a: *const u64, b: *const u64, count: u32) -> u8;
    fn pxb_capture_motion(w: *mut c_void, scope: *const u64, count: u32) -> u64;
    fn pxb_restore_motion(w: *mut c_void, token: u64, scope: *const u64, count: u32) -> u8;
    fn pxb_release_snapshot(w: *mut c_void, token: u64);
}

/// A PhysX scene driven through the backend contract.
pub struct PhysXWorld {
    w: *mut c_void,
    gpu: bool,
    /// Reused marshalling scratch, so a steady-state frame allocates nothing.
    scratch: Scratch,
    contact_buf: Vec<CContact>,
}

#[derive(Default)]
struct Scratch {
    create_bodies: Vec<CCreateBody>,
    kind_ids: Vec<u64>,
    kind_vals: Vec<u32>,
    pose_ids: Vec<u64>,
    pose_vals: Vec<CPose>,
    vel_ids: Vec<u64>,
    vel_lin: Vec<CVec3>,
    vel_ang: Vec<CVec3>,
    create_shapes: Vec<CCreateShape>,
    hull_points: Vec<Vec<CVec3>>,
    reparent: Vec<CReparent>,
    remove_shapes: Vec<u64>,
    recompute: Vec<u64>,
    remove_bodies: Vec<u64>,
    wake: Vec<u64>,
    sleep: Vec<u64>,
    damp_ids: Vec<u64>,
    damp_lin: Vec<f32>,
    damp_ang: Vec<f32>,
    thr_ids: Vec<u64>,
    thr_lin: Vec<f32>,
    thr_ang: Vec<f32>,
    ccd_ids: Vec<u64>,
    ccd_vals: Vec<u8>,
    grp_shapes: Vec<u64>,
    grp_members: Vec<u32>,
    grp_filters: Vec<u32>,
    grp_entities: Vec<u32>,
    en_ids: Vec<u64>,
    en_vals: Vec<u8>,
    imp_ids: Vec<u64>,
    imp_lin: Vec<CVec3>,
    imp_ang: Vec<CVec3>,
    ids: Vec<u64>,
    pts: Vec<CVec3>,
}

unsafe impl Send for PhysXWorld {}

impl Drop for PhysXWorld {
    fn drop(&mut self) {
        if !self.w.is_null() {
            unsafe { pxb_world_destroy(self.w) };
            self.w = std::ptr::null_mut();
        }
    }
}

impl PhysXWorld {
    /// Create a CPU scene.
    pub fn new_cpu(gravity: Vec3, cpu_threads: u32) -> Option<Self> {
        Self::create(gravity, false, cpu_threads)
    }

    /// Create a GPU scene (GPU dynamics + GPU broadphase).
    ///
    /// Returns `None` rather than silently running on the CPU: a CPU scene
    /// reported as GPU misreports every measurement taken against it.
    pub fn new_gpu(gravity: Vec3, cpu_threads: u32) -> Option<Self> {
        Self::create(gravity, true, cpu_threads)
    }

    fn create(gravity: Vec3, gpu: bool, cpu_threads: u32) -> Option<Self> {
        let w = unsafe { pxb_world_create(cv(gravity), gpu as u8, cpu_threads) };
        if w.is_null() {
            return None;
        }
        let active = unsafe { pxb_world_gpu_active(w) } != 0;
        Some(Self { w, gpu: active, scratch: Scratch::default(), contact_buf: Vec::new() })
    }

    /// Wrap a `PxScene` the host already owns and drives.
    ///
    /// This is the bring-your-own-world path. The backend adds and removes only
    /// its own actors; it never steps or releases the scene, and it does not
    /// take the simulation-event callback slot — the host almost certainly has
    /// one already and PhysX allows only one. Feed contacts in with
    /// [`inject_contact`](Self::inject_contact) from the host's own `onContact`.
    ///
    /// # Safety
    /// `scene` and `physics` must be live `PxScene*` / `PxPhysics*` pointers
    /// that outlive the returned world.
    pub unsafe fn attach_scene(
        scene: *mut c_void,
        physics: *mut c_void,
        material: *mut c_void,
    ) -> Option<Self> {
        let w = pxb_world_attach(scene, physics, material);
        if w.is_null() {
            return None;
        }
        let gpu = pxb_world_gpu_active(w) != 0;
        Some(Self { w, gpu, scratch: Scratch::default(), contact_buf: Vec::new() })
    }

    /// Borrow the underlying `PxScene*`, for handing the same scene to another
    /// subsystem (which is what a host integration does).
    pub fn raw_scene(&self) -> *mut c_void {
        unsafe { pxb_world_scene(self.w) }
    }

    /// Borrow the underlying `PxPhysics*`.
    pub fn raw_physics(&self) -> *mut c_void {
        unsafe { pxb_world_physics(self.w) }
    }

    /// True when this world borrows a host scene rather than owning one.
    pub fn is_attached(&self) -> bool {
        unsafe { pxb_world_is_attached(self.w) != 0 }
    }

    /// Tell an attached world the timestep the host is about to use, so
    /// injected impulses convert to force correctly.
    pub fn note_dt(&mut self, dt: f32) {
        unsafe { pxb_note_dt(self.w, dt) }
    }

    /// Feed one contact from the host's simulation-event callback.
    ///
    /// `impulse_magnitude` is PhysX's raw impulse; only the magnitude is used,
    /// because PhysX contact impulse signs are ordering dependent and are never
    /// normalised. Direction comes from `normal`.
    ///
    /// # Safety
    /// The shape pointers must be live `PxShape*` values from the host scene.
    pub unsafe fn inject_contact(
        &mut self,
        shape_a: *mut c_void,
        shape_b: *mut c_void,
        world_position: Vec3,
        normal: Vec3,
        relative_velocity: Vec3,
        impulse_magnitude: f32,
        persisting: bool,
    ) -> bool {
        pxb_inject_contact(
            self.w,
            shape_a,
            shape_b,
            cv(world_position),
            cv(normal),
            cv(relative_velocity),
            impulse_magnitude,
            persisting as u8,
        ) != 0
    }

    pub fn gpu_active(&self) -> bool {
        self.gpu
    }

    pub fn check(&self) -> Result<(), MissingCapabilities> {
        check_required(self.capabilities())
    }
}

impl PhysicsBackend for PhysXWorld {
    type BodyId = PxBodyId;
    type ShapeId = PxShapeId;

    fn capabilities(&self) -> Capabilities {
        // The C++ side owns the authoritative list; mirroring the raw bits
        // keeps a single source of truth rather than two that can drift.
        let bits = unsafe { pxb_capabilities(self.w) };
        let mut c = Capabilities::NONE;
        for (bit, cap) in [
            (0, Capabilities::BODY_LIFECYCLE),
            (1, Capabilities::BODY_TYPE_MUTATION),
            (2, Capabilities::SHAPE_LIFECYCLE),
            (3, Capabilities::MASS_PROPERTIES),
            (4, Capabilities::POSE_VELOCITY_IO),
            (5, Capabilities::CONTACT_EVENTS),
            (6, Capabilities::DETERMINISTIC_HANDLES),
            (8, Capabilities::REPARENT_SHAPE),
            (9, Capabilities::SHAPE_SIMULATION_TOGGLE),
            (10, Capabilities::SHAPE_QUERY_TOGGLE),
            (11, Capabilities::PAIR_EXCLUSION),
            (12, Capabilities::COLLISION_GROUPS),
            (13, Capabilities::DAMPING),
            (14, Capabilities::SLEEP_THRESHOLDS),
            (15, Capabilities::CCD),
            (16, Capabilities::IMPULSES),
            (17, Capabilities::MOTION_SNAPSHOT),
            (18, Capabilities::SCOPED_SNAPSHOT),
            (19, Capabilities::BATCH_MOTION_IO),
            (20, Capabilities::NATIVE_POINT_VELOCITY),
            (21, Capabilities::CONTACT_MANIFOLDS),
        ] {
            if bits & (1 << bit) != 0 {
                c = c | cap;
            }
        }
        c
    }

    fn read_bodies(&self, ids: &[Self::BodyId], out: &mut BodyStateSoa) {
        let n = ids.len();
        out.reset_for(n);
        if n == 0 {
            return;
        }
        let raw: Vec<u64> = ids.iter().map(|b| b.0).collect();
        let mut pose = vec![CPose::default(); n];
        let mut lin = vec![CVec3::default(); n];
        let mut ang = vec![CVec3::default(); n];
        let mut flags = vec![0u8; n];
        let mut mass = vec![0f32; n];
        unsafe {
            pxb_read_bodies(
                self.w,
                raw.as_ptr(),
                n as u32,
                pose.as_mut_ptr(),
                lin.as_mut_ptr(),
                ang.as_mut_ptr(),
                flags.as_mut_ptr(),
                mass.as_mut_ptr(),
            )
        };
        for i in 0..n {
            let mut f = BodyFlags::NONE;
            f.set(BodyFlags::DYNAMIC, flags[i] & (1 << 0) != 0);
            f.set(BodyFlags::KINEMATIC, flags[i] & (1 << 1) != 0);
            f.set(BodyFlags::SLEEPING, flags[i] & (1 << 2) != 0);
            f.set(BodyFlags::ENABLED, flags[i] & (1 << 3) != 0);
            out.push(rp(pose[i]), rv(lin[i]), rv(ang[i]), f, mass[i]);
        }
    }

    fn read_center_of_mass(&self, ids: &[Self::BodyId], out: &mut Vec<Vec3>) {
        out.clear();
        if ids.is_empty() {
            return;
        }
        let raw: Vec<u64> = ids.iter().map(|b| b.0).collect();
        let mut coms = vec![CVec3::default(); ids.len()];
        unsafe { pxb_read_center_of_mass(self.w, raw.as_ptr(), ids.len() as u32, coms.as_mut_ptr()) };
        out.extend(coms.into_iter().map(rv));
    }

    fn shape_parent(&self, shapes: &[Self::ShapeId], out: &mut Vec<Option<Self::BodyId>>) {
        out.clear();
        if shapes.is_empty() {
            return;
        }
        let raw: Vec<u64> = shapes.iter().map(|s| s.0).collect();
        let mut parents = vec![0u64; shapes.len()];
        unsafe { pxb_shape_parent(self.w, raw.as_ptr(), shapes.len() as u32, parents.as_mut_ptr()) };
        out.extend(parents.into_iter().map(|b| (b != 0).then_some(PxBodyId(b))));
    }

    fn read_point_velocities(&self, queries: &[(Self::BodyId, Vec3)], out: &mut Vec<Vec3>) {
        out.clear();
        if queries.is_empty() {
            return;
        }
        let ids: Vec<u64> = queries.iter().map(|(b, _)| b.0).collect();
        let pts: Vec<CVec3> = queries.iter().map(|(_, p)| cv(*p)).collect();
        let mut vel = vec![CVec3::default(); queries.len()];
        unsafe {
            pxb_read_point_velocities(
                self.w,
                ids.as_ptr(),
                pts.as_ptr(),
                queries.len() as u32,
                vel.as_mut_ptr(),
            )
        };
        out.extend(vel.into_iter().map(rv));
    }

    fn apply(
        &mut self,
        phase: Phase,
        cmds: &CommandBuffer<Self::BodyId, Self::ShapeId>,
        out: &mut CommandResults<Self::BodyId, Self::ShapeId>,
    ) -> Result<Applied, BackendError> {
        out.clear();
        let s = &mut self.scratch;
        s.create_bodies.clear();
        s.kind_ids.clear();
        s.kind_vals.clear();
        s.pose_ids.clear();
        s.pose_vals.clear();
        s.vel_ids.clear();
        s.vel_lin.clear();
        s.vel_ang.clear();
        s.create_shapes.clear();
        s.hull_points.clear();
        s.reparent.clear();
        s.remove_shapes.clear();
        s.recompute.clear();
        s.remove_bodies.clear();
        s.wake.clear();
        s.sleep.clear();
        s.damp_ids.clear();
        s.damp_lin.clear();
        s.damp_ang.clear();
        s.thr_ids.clear();
        s.thr_lin.clear();
        s.thr_ang.clear();
        s.ccd_ids.clear();
        s.grp_shapes.clear();
        s.grp_members.clear();
        s.grp_filters.clear();
        s.grp_entities.clear();
        s.ccd_vals.clear();
        s.en_ids.clear();
        s.en_vals.clear();
        s.imp_ids.clear();
        s.imp_lin.clear();
        s.imp_ang.clear();

        for cb in &cmds.create_bodies {
            s.create_bodies.push(CCreateBody {
                pose: cp(cb.pose),
                linvel: cv(cb.linvel),
                angvel: cv(cb.angvel),
                kind: match cb.kind {
                    BodyKind::Dynamic => 0,
                    BodyKind::Fixed => 1,
                    BodyKind::Kinematic => 2,
                },
                ccd: cb.ccd as u8,
                start_sleeping: cb.start_sleeping as u8,
            });
        }
        for (b, k) in &cmds.set_body_kind {
            s.kind_ids.push(b.0);
            s.kind_vals.push(match k {
                BodyKind::Dynamic => 0,
                BodyKind::Fixed => 1,
                BodyKind::Kinematic => 2,
            });
        }
        for (b, p) in &cmds.set_pose {
            s.pose_ids.push(b.0);
            s.pose_vals.push(cp(*p));
        }
        for (b, l, a) in &cmds.set_velocity {
            s.vel_ids.push(b.0);
            s.vel_lin.push(cv(*l));
            s.vel_ang.push(cv(*a));
        }
        // Hull points must outlive the call, so they are staged first and the
        // pointers taken afterwards.
        for cs in &cmds.create_shapes {
            if let ShapeGeom::ConvexHull { points } = &cs.geom {
                s.hull_points.push(points.iter().map(|p| cv(*p)).collect());
            } else {
                s.hull_points.push(Vec::new());
            }
        }
        for (i, cs) in cmds.create_shapes.iter().enumerate() {
            let (geom, he) = match &cs.geom {
                ShapeGeom::Cuboid { half_extents } => (0u32, cv(*half_extents)),
                ShapeGeom::ConvexHull { .. } => (1u32, CVec3::default()),
            };
            s.create_shapes.push(CCreateShape {
                body: cs.body.0,
                local: cp(cs.local),
                half_extents: he,
                points: s.hull_points[i].as_ptr(),
                point_count: s.hull_points[i].len() as u32,
                geom,
                mass: cs.mass,
                node: cs.node,
            });
        }
        for r in &cmds.reparent_shapes {
            s.reparent.push(CReparent { shape: r.shape.0, body: r.body.0, local: cp(r.local) });
        }
        // `set_shape_local` with no reparent is a same-body move.
        for (sh, local) in &cmds.set_shape_local {
            s.reparent.push(CReparent { shape: sh.0, body: 0, local: cp(*local) });
        }
        s.remove_shapes.extend(cmds.remove_shapes.iter().map(|x| x.0));
        s.recompute.extend(cmds.recompute_mass.iter().map(|x| x.0));
        s.remove_bodies.extend(cmds.remove_bodies.iter().map(|x| x.0));
        s.wake.extend(cmds.wake.iter().map(|x| x.0));
        s.sleep.extend(cmds.sleep.iter().map(|x| x.0));
        for (b, l, a) in &cmds.set_damping {
            s.damp_ids.push(b.0);
            s.damp_lin.push(*l);
            s.damp_ang.push(*a);
        }
        for (b, l, a) in &cmds.set_sleep_thresholds {
            s.thr_ids.push(b.0);
            s.thr_lin.push(*l);
            s.thr_ang.push(*a);
        }
        for (sh, g) in &cmds.set_groups {
            s.grp_shapes.push(sh.0);
            s.grp_members.push(g.memberships);
            s.grp_filters.push(g.filter);
            s.grp_entities.push(g.entity);
        }
        for (b, on) in &cmds.set_ccd {
            s.ccd_ids.push(b.0);
            s.ccd_vals.push(*on as u8);
        }
        for (sh, on) in &cmds.set_shape_enabled {
            s.en_ids.push(sh.0);
            s.en_vals.push(*on as u8);
        }
        for (b, l, a) in &cmds.apply_impulse {
            s.imp_ids.push(b.0);
            s.imp_lin.push(cv(*l));
            s.imp_ang.push(cv(*a));
        }

        let c = CCommands {
            create_bodies: s.create_bodies.as_ptr(),
            create_body_count: s.create_bodies.len() as u32,
            set_kind_ids: s.kind_ids.as_ptr(),
            set_kind_values: s.kind_vals.as_ptr(),
            set_kind_count: s.kind_ids.len() as u32,
            set_pose_ids: s.pose_ids.as_ptr(),
            set_pose_values: s.pose_vals.as_ptr(),
            set_pose_count: s.pose_ids.len() as u32,
            set_vel_ids: s.vel_ids.as_ptr(),
            set_vel_lin: s.vel_lin.as_ptr(),
            set_vel_ang: s.vel_ang.as_ptr(),
            set_vel_count: s.vel_ids.len() as u32,
            create_shapes: s.create_shapes.as_ptr(),
            create_shape_count: s.create_shapes.len() as u32,
            reparent: s.reparent.as_ptr(),
            reparent_count: s.reparent.len() as u32,
            remove_shapes: s.remove_shapes.as_ptr(),
            remove_shape_count: s.remove_shapes.len() as u32,
            recompute_mass: s.recompute.as_ptr(),
            recompute_mass_count: s.recompute.len() as u32,
            remove_bodies: s.remove_bodies.as_ptr(),
            remove_body_count: s.remove_bodies.len() as u32,
            wake: s.wake.as_ptr(),
            wake_count: s.wake.len() as u32,
            sleep: s.sleep.as_ptr(),
            sleep_count: s.sleep.len() as u32,
            damping_ids: s.damp_ids.as_ptr(),
            damping_lin: s.damp_lin.as_ptr(),
            damping_ang: s.damp_ang.as_ptr(),
            damping_count: s.damp_ids.len() as u32,
            sleep_thr_ids: s.thr_ids.as_ptr(),
            sleep_thr_lin: s.thr_lin.as_ptr(),
            sleep_thr_ang: s.thr_ang.as_ptr(),
            sleep_thr_count: s.thr_ids.len() as u32,
            group_shapes: s.grp_shapes.as_ptr(),
            group_memberships: s.grp_members.as_ptr(),
            group_filters: s.grp_filters.as_ptr(),
            group_entities: s.grp_entities.as_ptr(),
            group_count: s.grp_shapes.len() as u32,
            ccd_ids: s.ccd_ids.as_ptr(),
            ccd_values: s.ccd_vals.as_ptr(),
            ccd_count: s.ccd_ids.len() as u32,
            shape_enabled_ids: s.en_ids.as_ptr(),
            shape_enabled_values: s.en_vals.as_ptr(),
            shape_enabled_count: s.en_ids.len() as u32,
            impulse_ids: s.imp_ids.as_ptr(),
            impulse_lin: s.imp_lin.as_ptr(),
            impulse_ang: s.imp_ang.as_ptr(),
            impulse_count: s.imp_ids.len() as u32,
        };

        let mut created_bodies = vec![0u64; s.create_bodies.len().max(1)];
        let mut created_shapes = vec![0u64; s.create_shapes.len().max(1)];
        let mut applied = CApplied::default();
        let ok = unsafe {
            pxb_apply(
                self.w,
                phase as u32,
                &c,
                created_bodies.as_mut_ptr(),
                created_shapes.as_mut_ptr(),
                &mut applied,
            )
        };
        if ok == 0 {
            return Err(BackendError::Rejected("PhysX rejected a command buffer".into()));
        }
        out.created_bodies.extend(
            created_bodies.iter().take(s.create_bodies.len()).map(|b| PxBodyId(*b)),
        );
        out.created_shapes.extend(
            created_shapes.iter().take(s.create_shapes.len()).map(|x| PxShapeId(*x)),
        );
        Ok(Applied {
            bodies_created: applied.bodies_created as usize,
            shapes_created: applied.shapes_created as usize,
            shapes_reparented: applied.shapes_reparented as usize,
            bodies_removed: applied.bodies_removed as usize,
            writes_elided: applied.writes_elided as usize,
        })
    }

    fn step(&mut self, dt: f32) {
        unsafe { pxb_step(self.w, dt) };
    }

    fn drain_contacts(&mut self, out: &mut ContactBatch<Self::ShapeId>) -> usize {
        out.clear();
        if self.contact_buf.len() < 4096 {
            self.contact_buf.resize(
                4096,
                CContact {
                    shape_a: 0,
                    shape_b: 0,
                    world_position: CVec3::default(),
                    normal: CVec3::default(),
                    relative_velocity: CVec3::default(),
                    force: 0.0,
                    persisting: 0,
                },
            );
        }
        let n = unsafe {
            pxb_drain_contacts(self.w, self.contact_buf.as_mut_ptr(), self.contact_buf.len() as u32)
        } as usize;
        for c in &self.contact_buf[..n] {
            out.contacts.push(Contact {
                shape_a: PxShapeId(c.shape_a),
                shape_b: (c.shape_b != 0).then_some(PxShapeId(c.shape_b)),
                world_position: rv(c.world_position),
                normal: rv(c.normal),
                force: c.force,
                relative_velocity: rv(c.relative_velocity),
                persisting: c.persisting != 0,
            });
        }
        n
    }

    fn for_each_dynamic_body(&self, f: &mut dyn FnMut(Self::BodyId)) {
        let n = unsafe { pxb_dynamic_bodies(self.w, std::ptr::null_mut(), 0) } as usize;
        if n == 0 {
            return;
        }
        let mut buf = vec![0u64; n];
        let got = unsafe { pxb_dynamic_bodies(self.w, buf.as_mut_ptr(), n as u32) } as usize;
        for id in buf.into_iter().take(got.min(n)) {
            f(PxBodyId(id));
        }
    }

    fn set_excluded_pairs(&mut self, pairs: &[(Self::BodyId, Self::BodyId)]) -> Result<(), Unsupported> {
        let a: Vec<u64> = pairs.iter().map(|(x, _)| x.0).collect();
        let b: Vec<u64> = pairs.iter().map(|(_, y)| y.0).collect();
        unsafe { pxb_set_excluded_pairs(self.w, a.as_ptr(), b.as_ptr(), pairs.len() as u32) };
        Ok(())
    }

    fn capture_motion(&mut self, scope: &[Self::BodyId]) -> Result<SnapshotToken, Unsupported> {
        let raw: Vec<u64> = scope.iter().map(|b| b.0).collect();
        let t = unsafe { pxb_capture_motion(self.w, raw.as_ptr(), raw.len() as u32) };
        Ok(SnapshotToken(t))
    }

    fn restore_motion(&mut self, t: SnapshotToken, scope: &[Self::BodyId]) -> Result<(), Unsupported> {
        let raw: Vec<u64> = scope.iter().map(|b| b.0).collect();
        unsafe { pxb_restore_motion(self.w, t.0, raw.as_ptr(), raw.len() as u32) };
        Ok(())
    }

    fn release_snapshot(&mut self, t: SnapshotToken) {
        unsafe { pxb_release_snapshot(self.w, t.0) };
    }
}
