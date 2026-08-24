//! Reproduces the "fragment yanks itself inwards" bug seen in the Rust demo.
//!
//! The demo resimulates by capturing a snapshot of every dynamic body's ORIGIN pose before the
//! step, then — if the step fractured something — `restore()`-ing that snapshot and re-stepping
//! (see blast-stress-demo-rs `step_physics`). The split path, however, RE-CENTERS a reused
//! body's origin onto its fragment's centre of mass and re-derives every retained collider's
//! local offset relative to that NEW origin. Restoring the OLD origin with the NEW offsets
//! places the colliders at `old_origin * new_offset` — i.e. teleported by the recentering
//! distance (toward the old origin = "inward"). Velocity continuity can't see this; only a
//! world-position check does.

#![cfg(feature = "rapier")]

use rapier3d::prelude::*;

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

const WORLD_POSITION_CONTINUITY_TOL: f32 = 1.0e-3;

fn rapier_world() -> (
    RigidBodySet,
    ColliderSet,
    IslandManager,
    ImpulseJointSet,
    MultibodyJointSet,
) {
    (
        RigidBodySet::new(),
        ColliderSet::new(),
        IslandManager::new(),
        ImpulseJointSet::new(),
        MultibodyJointSet::new(),
    )
}

/// Horizontal beam of `n` unit cubes (no supports → one dynamic body), with a weak bond at
/// `weak_bond` so a stretching load fractures it into two multi-node halves.
fn beam_scenario(n: usize, weak_bond: usize) -> ScenarioDesc {
    let size = Vec3::new(0.5, 0.5, 0.5);
    ScenarioDesc {
        nodes: (0..n)
            .map(|i| ScenarioNode {
                centroid: Vec3::new(i as f32, 5.0, 0.0),
                mass: 1.0,
                volume: 0.125,
            })
            .collect(),
        bonds: (0..n - 1)
            .map(|i| ScenarioBond {
                node0: i as u32,
                node1: (i + 1) as u32,
                centroid: Vec3::new(i as f32 + 0.5, 5.0, 0.0),
                normal: Vec3::new(1.0, 0.0, 0.0),
                area: if i == weak_bond { 0.01 } else { 50.0 }, material: 0, })
            .collect(),
        node_sizes: vec![size; n],
        collider_shapes: vec![None; n],
    }
}

/// World position of a node = its body's pose applied to its (current) local collider offset —
/// exactly how the demo positions chunk meshes (`node_world_transform`).
fn node_world(set: &DestructibleSet, bodies: &RigidBodySet, node: u32) -> Option<Vec3> {
    let body = bodies.get(set.node_body(node)?)?;
    let local = set.node_local_offset(node)?;
    let w = body
        .position()
        .transform_point(&point![local.x, local.y, local.z]);
    Some(Vec3::new(w.x, w.y, w.z))
}

/// Drive a static beam (zero gravity) to fracture, mimicking the demo's resim: snapshot the
/// origins BEFORE the step, step (which splits + recenters the reused half), then restore the
/// snapshot. Returns the worst node world-position jump caused by the restore.
fn worst_resim_restore_teleport(n: usize, weak_bond: usize) -> f32 {
    let scenario = beam_scenario(n, weak_bond);
    let policy = FracturePolicy {
        idle_skip: false,
        ..FracturePolicy::default()
    };
    let mut set =
        DestructibleSet::from_scenario(&scenario, SolverSettings::default(), Vec3::ZERO, policy)
            .unwrap();
    let (mut bodies, mut colliders, mut islands, mut ij, mut mj) = rapier_world();
    set.initialize(&mut bodies, &mut colliders);

    for _ in 0..120 {
        let pre: Vec<Option<Vec3>> = (0..n)
            .map(|i| node_world(&set, &bodies, i as u32))
            .collect();
        let snapshot = BodySnapshots::capture(&bodies); // demo captures BEFORE the step
        set.add_force(0, scenario.nodes[0].centroid, Vec3::new(-4000.0, 0.0, 0.0));
        set.add_force(
            (n - 1) as u32,
            scenario.nodes[n - 1].centroid,
            Vec3::new(4000.0, 0.0, 0.0),
        );
        let result = set.step(&mut bodies, &mut colliders, &mut islands, &mut ij, &mut mj);
        if result.split_events > 0 {
            snapshot.restore(&mut bodies); // demo's resim rollback
            let mut worst = 0.0_f32;
            for i in 0..n {
                if let (Some(a), Some(b)) = (pre[i], node_world(&set, &bodies, i as u32)) {
                    worst = worst.max((a - b).magnitude());
                }
            }
            return worst;
        }
    }
    panic!("beam did not fracture");
}

/// REGRESSION GUARD (Rust "yank"). A resim rollback after a recentering split must leave every
/// chunk where it was — the body is static (zero gravity, no velocity), so a faithful restore
/// is a no-op on world positions. Before the fix the reused half teleports by ~its recentering
/// distance (~1 m for a 6-cube beam); the created half (absent from the snapshot) does not.
#[test]
fn resim_restore_does_not_teleport_recentered_reused_fragment() {
    let teleport = worst_resim_restore_teleport(6, 2);
    assert!(
        teleport < WORLD_POSITION_CONTINUITY_TOL,
        "resim restore teleported a reused-fragment chunk by {:.4} m (recentering vs snapshot \
         origin mismatch)",
        teleport,
    );
}
