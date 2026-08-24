//! The full destruction pipeline, running on two engines from one code path.
//!
//! This is the claim under test: stress solving, fracture, split planning, the
//! rigid-motion fit and the topology edit all live in the core, and the only
//! difference between engines is which adapter is constructed.

#![cfg(all(feature = "rapier", feature = "scenarios", feature = "physx"))]

use blast_stress_solver::backend::PhysicsBackend;
use blast_stress_solver::backends::{PhysXWorld, RapierWorld};
use blast_stress_solver::pipeline::{Destructible, DestructibleConfig};
use blast_stress_solver::scenarios::{build_wall_scenario, WallOptions};
use blast_stress_solver::types::{ScenarioDesc, SolverSettings, Vec3};

const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn scenario() -> ScenarioDesc {
    build_wall_scenario(&WallOptions::default())
}

/// Weak enough that gravity alone shatters it, so every engine exercises the
/// full split path rather than just standing still.
fn config() -> DestructibleConfig {
    DestructibleConfig {
        gravity: G,
        solver: SolverSettings {
            max_solver_iterations_per_frame: 24,
            compression_elastic_limit: 5.0,
            compression_fatal_limit: 10.0,
            tension_elastic_limit: 5.0,
            tension_fatal_limit: 10.0,
            shear_elastic_limit: 5.0,
            shear_fatal_limit: 10.0,
            ..SolverSettings::default()
        },
        min_child_nodes: 1,
        max_new_bodies_per_step: usize::MAX,
        ..Default::default()
    }
}

#[derive(Debug)]
struct Outcome {
    fractures: usize,
    splits: usize,
    created: usize,
    retired: usize,
    actors: u32,
    bodies: usize,
}

fn run<B: PhysicsBackend>(backend: &mut B, frames: u32) -> Outcome {
    let scn = scenario();
    let mut d = Destructible::attach(backend, &scn, config()).expect("attach");
    let mut o = Outcome { fractures: 0, splits: 0, created: 0, retired: 0, actors: 0, bodies: 0 };
    for _ in 0..frames {
        backend.step(1.0 / 60.0);
        let r = d.step(backend, 1.0 / 60.0);
        o.fractures += r.fractures;
        o.splits += r.split_events;
        o.created += r.bodies_created;
        o.retired += r.bodies_retired;
    }
    o.actors = d.solver().actor_count();
    o.bodies = d.bodies().len();
    o
}

#[test]
fn the_same_structure_fractures_on_rapier_and_physx() {
    let mut rapier = RapierWorld::new(G);
    let r = run(&mut rapier, 30);
    eprintln!("[rapier]    {r:?}");

    let mut physx = PhysXWorld::new_cpu(G, 2).expect("PhysX CPU scene");
    let p = run(&mut physx, 30);
    eprintln!("[physx-cpu] {p:?}");

    // Tier 1: structural truths that hold on any engine.
    for (name, o) in [("rapier", &r), ("physx-cpu", &p)] {
        assert!(o.fractures > 0, "[{name}] the structure never fractured");
        assert!(o.splits > 0, "[{name}] fractures produced no split events");
        assert!(o.created > 0, "[{name}] no fragment bodies were created");
        assert_eq!(
            o.bodies, o.actors as usize,
            "[{name}] tracked bodies ({}) must equal solver actors ({})",
            o.bodies, o.actors
        );
    }

    // The stress solve is identical code on both engines and is driven by the
    // authored geometry, not the rigid-body state, so the fracture count should
    // agree exactly. The engines diverge in where the debris ends up, not in
    // whether the structure breaks.
    assert_eq!(
        r.fractures, p.fractures,
        "the shared solver produced different fracture counts per engine"
    );
    assert_eq!(r.actors, p.actors, "actor topology diverged between engines");
}

#[test]
fn physx_gpu_runs_the_full_pipeline() {
    let Some(mut gpu) = PhysXWorld::new_gpu(G, 2) else {
        eprintln!("[physx-gpu] no usable GPU scene; skipping");
        return;
    };
    let o = run(&mut gpu, 30);
    eprintln!("[physx-gpu] {o:?}");
    assert!(o.fractures > 0 && o.splits > 0);
    assert_eq!(o.bodies, o.actors as usize);
}

#[test]
fn attach_leaves_one_body_per_actor() {
    // Before any fracture, the mapping the whole pipeline depends on must hold.
    let mut w = RapierWorld::new(G);
    let scn = scenario();
    let d = Destructible::attach(&mut w, &scn, config()).expect("attach");
    assert_eq!(
        d.bodies().len(),
        d.solver().actor_count() as usize,
        "attach must create exactly one body per solver actor"
    );
    for n in 0..scn.nodes.len() as u32 {
        assert!(d.node_body(n).is_some(), "node {n} has no body after attach");
    }
}
