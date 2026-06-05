//! Read-only island quiescence instrumentation (`DestructibleSet::island_settled_stats`), ported
//! from the web's `getIslandSettledStats`. Partitions the live bond graph into islands with static
//! (support) nodes as cut points, and reports how much of the structure is "settled" (asleep) — the
//! skippable-cost ceiling for island-aware solving. Pure measurement, no behavior change.

#![cfg(feature = "rapier")]

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;
use rapier3d::prelude::*;

/// Two cantilever arms (along +X and -X) sharing one static ground node at the origin. The shared
/// static node is a solver cut point ⇒ TWO islands, even though they are one bonded actor/body.
fn two_arms_sharing_ground() -> ScenarioDesc {
    let mut nodes = vec![ScenarioNode { centroid: Vec3::new(0.0, 0.0, 0.0), mass: 0.0, volume: 1.0 }];
    let mut bonds = Vec::new();
    for &dir in &[1.0f32, -1.0] {
        let mut prev = 0u32;
        for s in 1..=4u32 {
            let idx = nodes.len() as u32;
            nodes.push(ScenarioNode { centroid: Vec3::new(dir * s as f32, 0.0, 0.0), mass: 1.0, volume: 1.0 });
            let (p, c) = (nodes[prev as usize].centroid, nodes[idx as usize].centroid);
            bonds.push(ScenarioBond {
                node0: prev,
                node1: idx,
                centroid: (p + c) * 0.5,
                normal: Vec3::new(dir, 0.0, 0.0),
                area: 0.5,
            });
            prev = idx;
        }
    }
    let n = nodes.len();
    ScenarioDesc { nodes, bonds, node_sizes: vec![Vec3::new(0.9, 0.3, 0.3); n], collider_shapes: Vec::new() }
}

fn strong() -> SolverSettings {
    SolverSettings {
        compression_elastic_limit: 1.0e6,
        compression_fatal_limit: 1.0e7,
        tension_elastic_limit: 1.0e6,
        tension_fatal_limit: 1.0e7,
        shear_elastic_limit: 1.0e6,
        shear_fatal_limit: 1.0e7,
        ..SolverSettings::default()
    }
}

#[test]
fn settled_stats_track_sleep_state() {
    let scenario = two_arms_sharing_ground();
    let policy = FracturePolicy { idle_skip: false, ..FracturePolicy::default() };
    let mut set =
        DestructibleSet::from_scenario(&scenario, strong(), Vec3::new(0.0, -9.81, 0.0), policy).unwrap();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    set.initialize(&mut bodies, &mut colliders);

    // Two islands, both awake right after creation: nothing settled yet.
    let awake = set.island_settled_stats(&bodies);
    assert_eq!(awake.islands_total, 2, "two arms sharing a static node ⇒ two islands");
    assert!(awake.total_nodes >= 8, "eight dynamic arm nodes counted, got {}", awake.total_nodes);
    assert_eq!(awake.islands_settled, 0, "no body is asleep yet");
    assert_eq!(awake.settled_node_fraction, 0.0);

    // Force every dynamic arm body to sleep — both islands become settled.
    let mut arm_bodies: Vec<RigidBodyHandle> = Vec::new();
    for n in 1..scenario.nodes.len() as u32 {
        if let Some(h) = set.node_body(n) {
            if !arm_bodies.contains(&h) {
                arm_bodies.push(h);
            }
        }
    }
    for h in &arm_bodies {
        if let Some(b) = bodies.get_mut(*h) {
            b.sleep();
        }
    }

    let settled = set.island_settled_stats(&bodies);
    assert_eq!(settled.islands_total, 2);
    assert_eq!(settled.islands_settled, 2, "both islands' bodies are asleep");
    assert_eq!(settled.total_nodes, awake.total_nodes, "topology unchanged");
    assert_eq!(settled.settled_nodes, settled.total_nodes);
    assert!((settled.settled_node_fraction - 1.0).abs() < 1e-6, "all nodes settled");
    assert!((settled.settled_bond_fraction - 1.0).abs() < 1e-6, "all bonds settled");

    // Wake one arm body again ⇒ the settled fraction drops below 1.
    if let Some(b) = bodies.get_mut(arm_bodies[0]) {
        b.wake_up(true);
    }
    let partial = set.island_settled_stats(&bodies);
    assert!(partial.islands_settled < 2, "waking a body un-settles its island(s)");
    assert!(partial.settled_node_fraction < 1.0, "settled fraction drops after wake");
}
