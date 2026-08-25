//! Bring-your-own-world: destruction inside a scene the host owns and drives.
//!
//! This is the integration shape a real application needs — vibe-land's `/city`
//! runs players, vehicles and the destructible city in **one** PxScene — and it
//! is the case the library previously could not serve at all, because the
//! pipeline assumed it owned the world.
//!
//! The assertion that matters is negative: adding destruction must not perturb
//! bodies the library does not manage.

#![cfg(feature = "physx")]

use blast_stress_solver::backend::*;
use blast_stress_solver::backends::PhysXWorld;
use blast_stress_solver::types::Vec3;

const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Drop a few host-owned boxes and report where they end up.
fn run_host_only(frames: u32) -> Vec<Vec3> {
    let mut host = PhysXWorld::new_cpu(G, 2).expect("host scene");
    let ids = spawn_host_bodies(&mut host);
    for _ in 0..frames {
        host.step(1.0 / 60.0);
    }
    read_positions(&host, &ids)
}

fn spawn_host_bodies(w: &mut PhysXWorld) -> Vec<<PhysXWorld as PhysicsBackend>::BodyId> {
    let mut cmds: CommandBuffer<_, _> = CommandBuffer::new();
    let mut out: CommandResults<_, _> = CommandResults::default();
    // A floor plus three "player" boxes.
    cmds.create_bodies.push(CreateBody {
        pose: Pose::from_translation(Vec3::new(0.0, -1.0, 0.0)),
        kind: BodyKind::Fixed,
        linvel: Vec3::ZERO,
        angvel: Vec3::ZERO,
        ccd: false,
        start_sleeping: false,
    });
    for i in 0..3 {
        cmds.create_bodies.push(CreateBody {
            pose: Pose::from_translation(Vec3::new(-8.0 + i as f32 * 2.0, 3.0, 0.0)),
            kind: BodyKind::Dynamic,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            ccd: false,
            start_sleeping: false,
        });
    }
    w.apply(Phase::Topology, &cmds, &mut out).expect("host bodies");
    let created = out.created_bodies.clone();

    cmds.clear();
    cmds.create_shapes.push(CreateShape {
        body: created[0],
        local: Pose::IDENTITY,
        geom: ShapeGeom::Cuboid { half_extents: Vec3::new(50.0, 1.0, 50.0) },
        mass: 0.0,
        node: 0,
    });
    for b in &created[1..] {
        cmds.create_shapes.push(CreateShape {
            body: *b,
            local: Pose::IDENTITY,
            geom: ShapeGeom::Cuboid { half_extents: Vec3::new(0.5, 0.5, 0.5) },
            mass: 1.0,
            node: 1,
        });
    }
    w.apply(Phase::Topology, &cmds, &mut out).expect("host shapes");
    cmds.clear();
    cmds.recompute_mass.extend(created.iter().copied());
    w.apply(Phase::Topology, &cmds, &mut out).expect("mass");
    created[1..].to_vec()
}

fn read_positions(w: &PhysXWorld, ids: &[<PhysXWorld as PhysicsBackend>::BodyId]) -> Vec<Vec3> {
    let mut soa = BodyStateSoa::default();
    w.read_bodies(ids, &mut soa);
    (0..soa.len()).map(|i| soa.pose[i].translation).collect()
}

#[test]
fn an_attached_world_borrows_the_host_scene() {
    let host = PhysXWorld::new_cpu(G, 2).expect("host scene");
    let attached = unsafe {
        PhysXWorld::attach_scene(host.raw_scene(), host.raw_physics(), std::ptr::null_mut())
    }
    .expect("attach");
    assert!(attached.is_attached(), "attached world should report borrowing");
    assert!(!host.is_attached(), "owned world should not");

    // Pair exclusion needs our own contact-modify callback, which a borrowed
    // scene does not give us. The capability must say so rather than pretending.
    assert!(
        !attached.capabilities().contains(Capabilities::PAIR_EXCLUSION),
        "an attached world must not claim PAIR_EXCLUSION"
    );
    assert!(
        host.capabilities().contains(Capabilities::PAIR_EXCLUSION),
        "an owned world should offer PAIR_EXCLUSION"
    );
    // Required capabilities still hold, so the core will accept it.
    attached.check().expect("attached world must satisfy the required contract");
}

#[test]
fn library_bodies_do_not_perturb_host_bodies() {
    const FRAMES: u32 = 90;
    let control = run_host_only(FRAMES);

    // Same host scene, but now with a library-managed world attached to it.
    let mut host = PhysXWorld::new_cpu(G, 2).expect("host scene");
    let host_ids = spawn_host_bodies(&mut host);
    let mut attached = unsafe {
        PhysXWorld::attach_scene(host.raw_scene(), host.raw_physics(), std::ptr::null_mut())
    }
    .expect("attach");

    // The library adds its own bodies, well away from the host's.
    let mut cmds: CommandBuffer<_, _> = CommandBuffer::new();
    let mut out: CommandResults<_, _> = CommandResults::default();
    for i in 0..6 {
        cmds.create_bodies.push(CreateBody {
            pose: Pose::from_translation(Vec3::new(10.0, 1.0 + i as f32 * 1.1, 0.0)),
            kind: BodyKind::Dynamic,
            linvel: Vec3::ZERO,
            angvel: Vec3::ZERO,
            ccd: false,
            start_sleeping: false,
        });
    }
    attached.apply(Phase::Topology, &cmds, &mut out).expect("library bodies");
    let lib_ids = out.created_bodies.clone();
    cmds.clear();
    for b in &lib_ids {
        cmds.create_shapes.push(CreateShape {
            body: *b,
            local: Pose::IDENTITY,
            geom: ShapeGeom::Cuboid { half_extents: Vec3::new(0.5, 0.5, 0.5) },
            mass: 1.0,
            node: 0,
        });
    }
    attached.apply(Phase::Topology, &cmds, &mut out).expect("library shapes");
    cmds.clear();
    cmds.recompute_mass.extend(lib_ids.iter().copied());
    attached.apply(Phase::Topology, &cmds, &mut out).expect("mass");

    // The HOST drives the clock. The attached world must not step it — doing so
    // would double-advance the scene everything else shares.
    for _ in 0..FRAMES {
        attached.note_dt(1.0 / 60.0);
        host.step(1.0 / 60.0);
        attached.step(1.0 / 60.0); // deliberately a no-op for an attached world
    }

    let after = read_positions(&host, &host_ids);
    assert_eq!(control.len(), after.len());
    for (i, (c, a)) in control.iter().zip(after.iter()).enumerate() {
        let d = (*c - *a).magnitude();
        assert!(
            d < 1e-4,
            "host body {i} moved {d} m because destruction was attached \
             (control {c:?} vs {a:?}) — the library perturbed a body it does not own"
        );
    }

    // And the library's own bodies really did simulate.
    let lib_after = read_positions(&attached, &lib_ids);
    assert!(
        lib_after.iter().any(|p| p.y < 6.0),
        "library bodies never fell; the attached world did not simulate"
    );
}

#[test]
fn an_attached_world_does_not_double_step_the_scene() {
    // Stepping an attached world must be inert. If it were not, a host calling
    // both would advance its scene twice per frame and every velocity would be
    // wrong in a way that looks like a physics bug rather than an API misuse.
    let mut host = PhysXWorld::new_cpu(G, 2).expect("host");
    let ids = spawn_host_bodies(&mut host);
    let mut attached = unsafe {
        PhysXWorld::attach_scene(host.raw_scene(), host.raw_physics(), std::ptr::null_mut())
    }
    .expect("attach");

    for _ in 0..30 {
        host.step(1.0 / 60.0);
        for _ in 0..5 {
            attached.step(1.0 / 60.0); // must all be no-ops
        }
    }
    let with_extra = read_positions(&host, &ids);
    let control = run_host_only(30);
    for (c, a) in control.iter().zip(with_extra.iter()) {
        assert!(
            (*c - *a).magnitude() < 1e-4,
            "stepping the attached world advanced the host scene"
        );
    }
}
