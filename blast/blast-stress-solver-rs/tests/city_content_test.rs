//! Runs real `/city` scene packs through the standardized pipeline.
//!
//! Synthetic scenarios prove the plumbing; production content proves the
//! contract survives contact with authored geometry — convex hulls, per-bond
//! materials, realistic MPa limits, hundreds of nodes.
//!
//! The packs live in the consumer repo, so the test skips when they are absent
//! rather than pinning a path that only exists on one machine.

#![cfg(all(feature = "rapier", feature = "scenarios", feature = "physx"))]

use std::path::PathBuf;

use blast_stress_solver::backend::PhysicsBackend;
use blast_stress_solver::backends::{PhysXWorld, RapierWorld};
use blast_stress_solver::pipeline::{Destructible, DestructibleConfig};
use blast_stress_solver::scenarios::load_scenario_file;
use blast_stress_solver::types::Vec3;

const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn scene_path() -> Option<PathBuf> {
    for root in [
        "/root/workspace/vibe-land-3/destruction/assets/scenes",
        "/root/workspace/vibe-land-2/destruction/assets/scenes",
    ] {
        let p = PathBuf::from(root).join("high-rise-3f-local.json");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

#[derive(Debug, PartialEq)]
struct Outcome {
    fractures: usize,
    splits: usize,
    created: usize,
    actors: u32,
    bodies: usize,
}

fn run<B: PhysicsBackend>(backend: &mut B, path: &PathBuf, impact: f32) -> Outcome {
    let loaded = load_scenario_file(path).expect("scene pack must parse");
    let scn = loaded.scenario;
    let cfg = DestructibleConfig { gravity: G, solver: loaded.settings, ..Default::default() };
    let mut d = Destructible::attach(backend, &scn, cfg).expect("attach");

    // Nearest dynamic node to the impact point.
    let target = Vec3::new(0.0, 6.0, 0.0);
    let (hit, _) = scn
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.mass > 0.0)
        .map(|(i, n)| (i as u32, (n.centroid - target).magnitude_squared()))
        .fold((0u32, f32::MAX), |acc, x| if x.1 < acc.1 { x } else { acc });

    let mut o = Outcome { fractures: 0, splits: 0, created: 0, actors: 0, bodies: 0 };
    for f in 0..60 {
        if f == 5 && impact > 0.0 {
            let p = scn.nodes[hit as usize].centroid;
            d.add_force(hit, p, Vec3::new(0.0, 0.0, impact));
        }
        backend.step(1.0 / 60.0);
        let r = d.step(backend, 1.0 / 60.0);
        o.fractures += r.fractures;
        o.splits += r.split_events;
        o.created += r.bodies_created;
    }
    o.actors = d.solver().actor_count();
    o.bodies = d.bodies().len();
    o
}

#[test]
fn an_authored_highrise_stands_under_gravity_on_every_engine() {
    let Some(path) = scene_path() else {
        eprintln!("city scene packs not present; skipping");
        return;
    };
    // A high-rise authored with realistic limits (12 MPa elastic / 30 MPa
    // fatal) must NOT self-destruct under its own weight. This is the
    // gravity-stability invariant, run against production content.
    for (name, o) in [
        ("rapier", run(&mut RapierWorld::new(G), &path, 0.0)),
        ("physx-cpu", run(&mut PhysXWorld::new_cpu(G, 2).expect("scene"), &path, 0.0)),
    ] {
        assert_eq!(o.fractures, 0, "[{name}] the building collapsed under gravity alone: {o:?}");
        assert_eq!(o.actors, 1, "[{name}] structure fragmented without a load: {o:?}");
    }
}

#[test]
fn the_same_impact_breaks_the_building_identically_on_every_engine() {
    let Some(path) = scene_path() else {
        eprintln!("city scene packs not present; skipping");
        return;
    };
    let impact = 3.0e9;
    let r = run(&mut RapierWorld::new(G), &path, impact);
    let p = run(&mut PhysXWorld::new_cpu(G, 2).expect("scene"), &path, impact);
    eprintln!("[rapier]    {r:?}");
    eprintln!("[physx-cpu] {p:?}");

    assert!(r.fractures > 0 && r.splits > 0, "impact did not break the structure");

    // The stress solve is shared code driven by authored geometry, so the
    // fracture topology is engine-independent even though the debris motion is
    // not. Divergence here would mean an adapter is feeding the solver
    // different loads, which is the failure this test exists to catch.
    assert_eq!(r, p, "the same impact produced different destruction per engine");

    // And the bookkeeping invariant must hold on both.
    assert_eq!(r.bodies, r.actors as usize);
    assert_eq!(p.bodies, p.actors as usize);
}
