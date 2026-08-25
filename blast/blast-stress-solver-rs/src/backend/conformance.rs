//! A reusable conformance suite every backend must pass.
//!
//! This is the mechanism behind the claim *"implement the contract correctly
//! and the features work"*. It is generic over [`PhysicsBackend`], so the same
//! assertions run against Rapier, PhysX-CPU, PhysX-GPU and anything added
//! later — and a new adapter gets a to-do list rather than a mystery.
//!
//! Everything here is a **structural invariant**: a truth that holds
//! regardless of what the physics computes. Nothing in this file compares two
//! engines' numbers to each other, because they will not match and should not
//! be expected to.

use super::*;
use crate::types::Vec3;

/// One check's outcome.
#[derive(Clone, Debug)]
pub struct CheckResult {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
    /// True when the backend does not advertise the capability under test, so
    /// the check was not applicable rather than failed.
    pub skipped: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub results: Vec<CheckResult>,
}

impl Report {
    pub fn failures(&self) -> Vec<&CheckResult> {
        self.results.iter().filter(|r| !r.passed && !r.skipped).collect()
    }
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }
    pub fn skipped(&self) -> usize {
        self.results.iter().filter(|r| r.skipped).count()
    }
    pub fn is_ok(&self) -> bool {
        self.failures().is_empty()
    }
    fn add(&mut self, name: &'static str, passed: bool, detail: impl Into<String>) {
        self.results.push(CheckResult { name, passed, detail: detail.into(), skipped: false });
    }
    fn skip(&mut self, name: &'static str, why: impl Into<String>) {
        self.results.push(CheckResult { name, passed: true, detail: why.into(), skipped: true });
    }
}

/// Run the full suite against `backend`.
pub fn run<B: PhysicsBackend>(backend: &mut B) -> Report {
    let mut r = Report::default();
    let caps = backend.capabilities();

    // ---- required capabilities ----
    match check_required(caps) {
        Ok(()) => r.add("required_capabilities", true, "all present"),
        Err(e) => r.add("required_capabilities", false, e.to_string()),
    }

    // ---- body creation returns handles index-parallel and in order ----
    let mut cmds: CommandBuffer<B::BodyId, B::ShapeId> = CommandBuffer::new();
    let mut out: CommandResults<B::BodyId, B::ShapeId> = CommandResults::default();
    for i in 0..4 {
        cmds.create_bodies.push(CreateBody {
            pose: Pose::from_translation(Vec3::new(i as f32 * 2.0, 5.0, 0.0)),
            kind: if i == 0 { BodyKind::Fixed } else { BodyKind::Dynamic },
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            ccd: false,
            start_sleeping: false,
        });
    }
    let applied = backend.apply(Phase::Topology, &cmds, &mut out);
    let bodies: Vec<B::BodyId> = out.created_bodies.clone();
    match applied {
        Ok(a) => {
            r.add(
                "create_bodies_returns_parallel_handles",
                a.bodies_created == 4 && bodies.len() == 4,
                format!("created {} handles {}", a.bodies_created, bodies.len()),
            );
        }
        Err(e) => r.add("create_bodies_returns_parallel_handles", false, e.to_string()),
    }

    // Handles must be distinct — an adapter that reuses a live id breaks every
    // map the pipeline keys on them.
    let mut keys: Vec<u64> = bodies.iter().map(|b| b.sort_key()).collect();
    let distinct = {
        let mut k = keys.clone();
        k.sort_unstable();
        k.dedup();
        k.len() == keys.len()
    };
    r.add("handles_are_distinct", distinct, format!("{:?}", keys));

    // Ordering must be a total order derived from sort_key alone.
    keys.sort_unstable();
    r.add("sort_key_totally_orders", keys.windows(2).all(|w| w[0] < w[1]), "strictly increasing");

    // ---- pose round-trips ----
    let mut soa = BodyStateSoa::default();
    backend.read_bodies(&bodies, &mut soa);
    let pose_ok = soa.len() == 4
        && (soa.pose[2].translation.x - 4.0).abs() < 1e-4
        && (soa.pose[2].translation.y - 5.0).abs() < 1e-4;
    r.add(
        "read_bodies_round_trips_pose",
        pose_ok,
        format!("body[2] at {:?}", soa.pose.get(2).map(|p| p.translation)),
    );
    r.add(
        "body_kind_round_trips",
        !soa.is_dynamic(0) && soa.is_dynamic(1),
        format!("fixed={} dynamic={}", !soa.is_dynamic(0), soa.is_dynamic(1)),
    );

    // ---- shapes ----
    cmds.clear();
    for (i, b) in bodies.iter().enumerate() {
        cmds.create_shapes.push(CreateShape {
            body: *b,
            local: Pose::IDENTITY,
            geom: ShapeGeom::Cuboid { half_extents: Vec3::new(0.5, 0.5, 0.5) },
            mass: if i == 0 { 0.0 } else { 1.0 },
            node: i as u32,
        });
    }
    let shapes = match backend.apply(Phase::Topology, &cmds, &mut out) {
        Ok(a) => {
            r.add("create_shapes", a.shapes_created == 4, format!("{} created", a.shapes_created));
            out.created_shapes.clone()
        }
        Err(e) => {
            r.add("create_shapes", false, e.to_string());
            Vec::new()
        }
    };

    if shapes.len() == 4 {
        let mut parents = Vec::new();
        backend.shape_parent(&shapes, &mut parents);
        let ok = parents.iter().zip(bodies.iter()).all(|(p, b)| *p == Some(*b));
        r.add("shape_parent_reports_owner", ok, format!("{:?}", parents.len()));
    }

    // ---- the COM staleness rule ----
    // Adding a shape offset from the body origin moves the centre of mass, but
    // only after an explicit recompute. This is the rule that, when violated,
    // silently turns the split COM correction into a no-op.
    if shapes.len() == 4 {
        cmds.clear();
        cmds.create_shapes.push(CreateShape {
            body: bodies[1],
            local: Pose::from_translation(Vec3::new(4.0, 0.0, 0.0)),
            geom: ShapeGeom::Cuboid { half_extents: Vec3::new(0.5, 0.5, 0.5) },
            mass: 1.0,
            node: 99,
        });
        let _ = backend.apply(Phase::Topology, &cmds, &mut out);

        cmds.clear();
        cmds.recompute_mass.push(bodies[1]);
        let _ = backend.apply(Phase::Topology, &cmds, &mut out);

        let mut coms = Vec::new();
        backend.read_center_of_mass(&bodies[1..2], &mut coms);
        // Two unit masses at local x=0 and x=4 put the COM at local x=2.
        let body_x = soa.pose[1].translation.x;
        let expected = body_x + 2.0;
        let ok = (coms[0].x - expected).abs() < 0.05;
        r.add(
            "com_reflects_shapes_after_recompute",
            ok,
            format!("com.x={} expected~{}", coms[0].x, expected),
        );
    }

    // ---- point velocity ----
    cmds.clear();
    cmds.set_velocity.push((bodies[1], Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 2.0)));
    let _ = backend.apply(Phase::Motion, &cmds, &mut out);
    let mut pv = Vec::new();
    let mut coms = Vec::new();
    backend.read_center_of_mass(&bodies[1..2], &mut coms);
    let probe = coms[0] + Vec3::new(0.0, 1.0, 0.0);
    backend.read_point_velocities(&[(bodies[1], probe)], &mut pv);
    // v + ω × r with ω = (0,0,2) and r = (0,1,0) gives (-2,0,0) added to v.
    let ok = (pv[0].x - (1.0 - 2.0)).abs() < 1e-3 && pv[0].y.abs() < 1e-3;
    r.add("point_velocity_matches_rigid_field", ok, format!("{:?}", pv[0]));

    // ---- write elision ----
    // Re-applying an identical Tuning buffer must change nothing and must say
    // so. A backend that rewrites unchanged values wakes sleeping bodies, which
    // is the single most expensive mistake an adapter can make at scale.
    if caps.contains(Capabilities::DAMPING) {
        cmds.clear();
        cmds.set_damping.push((bodies[1], 0.5, 0.5));
        let _ = backend.apply(Phase::Tuning, &cmds, &mut out);
        match backend.apply(Phase::Tuning, &cmds, &mut out) {
            Ok(a) => r.add(
                "repeated_writes_are_elided",
                a.writes_elided > 0,
                format!("{} elided on the second identical apply", a.writes_elided),
            ),
            Err(e) => r.add("repeated_writes_are_elided", false, e.to_string()),
        }
    } else {
        r.skip("repeated_writes_are_elided", "no DAMPING capability");
    }

    // ---- reparent preserves shape identity ----
    if caps.contains(Capabilities::REPARENT_SHAPE) && shapes.len() == 4 {
        cmds.clear();
        cmds.reparent_shapes.push(ReparentShape {
            shape: shapes[3],
            body: bodies[2],
            local: Pose::IDENTITY,
        });
        let _ = backend.apply(Phase::Topology, &cmds, &mut out);
        let mut parents = Vec::new();
        backend.shape_parent(&shapes[3..4], &mut parents);
        r.add(
            "reparent_preserves_handle_and_moves_owner",
            parents[0] == Some(bodies[2]),
            "shape kept its handle and changed body",
        );
    } else {
        r.skip("reparent_preserves_handle_and_moves_owner", "no REPARENT_SHAPE capability");
    }

    // ---- snapshot / restore ----
    if caps.contains(Capabilities::MOTION_SNAPSHOT) {
        cmds.clear();
        cmds.set_velocity.push((bodies[1], Vec3::new(7.0, 0.0, 0.0), Vec3::ZERO));
        let _ = backend.apply(Phase::Motion, &cmds, &mut out);
        let token = backend.capture_motion(&[]).expect("capture");

        // Perturb, then create a body *after* the capture.
        cmds.clear();
        cmds.set_velocity.push((bodies[1], Vec3::new(-3.0, 0.0, 0.0), Vec3::ZERO));
        cmds.create_bodies.push(CreateBody {
            pose: Pose::from_translation(Vec3::new(50.0, 50.0, 50.0)),
            kind: BodyKind::Dynamic,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            ccd: false,
            start_sleeping: false,
        });
        let _ = backend.apply(Phase::Topology, &cmds, &mut out);
        let post_capture = out.created_bodies[0];

        backend.restore_motion(token, &[]).expect("restore");
        let mut after = BodyStateSoa::default();
        backend.read_bodies(&[bodies[1], post_capture], &mut after);
        r.add(
            "restore_returns_captured_velocity",
            (after.linvel[0].x - 7.0).abs() < 1e-3,
            format!("linvel.x={}", after.linvel[0].x),
        );
        // The post-capture body must survive: skipping unknown handles is what
        // lets fragments created by a fracture live through the rollback.
        r.add(
            "restore_skips_bodies_created_after_capture",
            (after.pose[1].translation.x - 50.0).abs() < 1e-3,
            "new body untouched by rollback",
        );
        backend.release_snapshot(token);
    } else {
        r.skip("restore_returns_captured_velocity", "no MOTION_SNAPSHOT capability");
        r.skip("restore_skips_bodies_created_after_capture", "no MOTION_SNAPSHOT capability");
    }

    // ---- sleep state survives a rollback ----
    if caps.contains(Capabilities::MOTION_SNAPSHOT) {
        cmds.clear();
        cmds.sleep.push(bodies[2]);
        let _ = backend.apply(Phase::Tuning, &cmds, &mut out);
        let token = backend.capture_motion(&[]).expect("capture");
        cmds.clear();
        cmds.wake.push(bodies[2]);
        let _ = backend.apply(Phase::Tuning, &cmds, &mut out);
        backend.restore_motion(token, &[]).expect("restore");
        let mut after = BodyStateSoa::default();
        backend.read_bodies(&bodies[2..3], &mut after);
        r.add(
            "restore_preserves_sleep_state",
            after.is_sleeping(0),
            "a body asleep at capture is asleep after rollback",
        );
        backend.release_snapshot(token);
    } else {
        r.skip("restore_preserves_sleep_state", "no MOTION_SNAPSHOT capability");
    }

    // ---- pair exclusion is accepted when advertised ----
    if caps.contains(Capabilities::PAIR_EXCLUSION) {
        let ok = backend.set_excluded_pairs(&[(bodies[1], bodies[2])]).is_ok();
        r.add("pair_exclusion_accepted", ok, "set_excluded_pairs succeeded");
        let _ = backend.set_excluded_pairs(&[]);
    } else {
        r.skip("pair_exclusion_accepted", "no PAIR_EXCLUSION capability");
    }

    // ---- unsupported operations error rather than silently no-op ----
    if !caps.contains(Capabilities::PAIR_EXCLUSION) {
        r.add(
            "unsupported_ops_report_error",
            backend.set_excluded_pairs(&[]).is_err(),
            "returns Err(Unsupported)",
        );
    } else {
        r.skip("unsupported_ops_report_error", "backend supports every optional op under test");
    }

    // ---- body removal ----
    cmds.clear();
    cmds.remove_bodies.push(bodies[3]);
    match backend.apply(Phase::Retire, &cmds, &mut out) {
        Ok(a) => r.add("remove_bodies", a.bodies_removed == 1, format!("{} removed", a.bodies_removed)),
        Err(e) => r.add("remove_bodies", false, e.to_string()),
    }

    r
}
