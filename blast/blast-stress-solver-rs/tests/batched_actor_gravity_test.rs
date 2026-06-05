//! Equivalence test for batched actor gravity (`ext_stress_solver_add_all_actor_gravity`).
//!
//! Production `apply_oriented_gravity` builds one rotation buffer and rotates world gravity into
//! every actor's local frame in a single FFI crossing. The C++ bridge applies a *forward* rotation
//! by the supplied quaternion, so we feed each actor's rotation **conjugate** to reproduce the old
//! per-actor `body.rotation().inverse_transform_vector(g)` (`R⁻¹·g`). This test locks that the
//! batched path is byte-identical to the per-actor reference across identity and rotated actors.

#![cfg(feature = "rapier")]

use blast_stress_solver::*;
use rapier3d::na::{UnitQuaternion, Vector3};

/// Several arms from one shared static hub; heavy gravity shatters them into multiple actors.
fn arms(arms: u32, seg: u32) -> (Vec<NodeDesc>, Vec<BondDesc>) {
    let mut nodes = vec![NodeDesc { centroid: Vec3::new(0.0, 0.0, 0.0), mass: 0.0, volume: 1.0 }];
    let mut bonds = Vec::new();
    let dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];
    for a in 0..arms {
        let dir = dirs[a as usize % dirs.len()];
        let mut prev = 0u32;
        for s in 1..=seg {
            let idx = nodes.len() as u32;
            nodes.push(NodeDesc {
                centroid: Vec3::new(dir.x * s as f32, dir.y * s as f32, dir.z * s as f32),
                mass: 1.0,
                volume: 1.0,
            });
            let (p, c) = (nodes[prev as usize].centroid, nodes[idx as usize].centroid);
            bonds.push(BondDesc { centroid: (p + c) * 0.5, normal: dir, area: 0.5, node0: prev, node1: idx });
            prev = idx;
        }
    }
    (nodes, bonds)
}

fn weak() -> SolverSettings {
    SolverSettings {
        compression_elastic_limit: 0.02,
        compression_fatal_limit: 0.05,
        tension_elastic_limit: 0.02,
        tension_fatal_limit: 0.05,
        shear_elastic_limit: 0.02,
        shear_fatal_limit: 0.05,
        ..SolverSettings::default()
    }
}

/// Build a solver and fracture it into several actors with a heavy gravity burst.
fn fractured_solver(nodes: &[NodeDesc], bonds: &[BondDesc]) -> ExtStressSolver {
    let mut s = ExtStressSolver::new(nodes, bonds, &weak()).unwrap();
    for _ in 0..6 {
        s.add_gravity(Vec3::new(0.0, -400.0, 0.0));
        s.update();
        let cmds = s.generate_fracture_commands();
        if !cmds.is_empty() {
            s.apply_fracture_commands(&cmds);
        }
    }
    s.reset();
    s
}

/// A deterministic test rotation per actor index: identity for even indices, a non-trivial
/// rotation (varying axis/angle) for odd indices — so both the identity and rotated code paths run.
fn test_rotation(actor_index: u32) -> UnitQuaternion<f32> {
    if actor_index % 2 == 0 {
        UnitQuaternion::identity()
    } else {
        let angle = 0.3 + 0.17 * actor_index as f32;
        let axis = Vector3::new(
            0.2 + 0.1 * (actor_index as f32),
            1.0,
            -0.3 * actor_index as f32,
        )
        .normalize();
        UnitQuaternion::from_axis_angle(&rapier3d::na::Unit::new_normalize(axis), angle)
    }
}

#[test]
fn batched_actor_gravity_matches_per_actor() {
    let (nodes, bonds) = arms(4, 5);
    let g = Vec3::new(0.0, -30.0, 0.0);
    let world_g = Vector3::new(g.x, g.y, g.z);

    // Reference: per-actor R⁻¹·g via add_actor_gravity (the old apply_oriented_gravity math).
    let mut per_actor = fractured_solver(&nodes, &bonds);
    let actors: Vec<u32> = per_actor.actors().iter().map(|a| a.actor_index).collect();
    assert!(actors.len() >= 3, "scenario should fracture into ≥3 actors, got {}", actors.len());
    for &ai in &actors {
        let q = test_rotation(ai);
        let l = q.inverse_transform_vector(&world_g);
        per_actor.add_actor_gravity(ai, Vec3::new(l.x, l.y, l.z));
    }
    per_actor.update();

    // Batched: conjugate quaternion per actor index, single FFI crossing.
    let mut batched = fractured_solver(&nodes, &bonds);
    let max_actor = actors.iter().copied().max().unwrap() as usize;
    let mut rotations = vec![0.0f32; (max_actor + 1) * 4];
    for slot in rotations.chunks_exact_mut(4) {
        slot[3] = 1.0;
    }
    for &ai in &actors {
        let q = test_rotation(ai);
        let base = ai as usize * 4;
        rotations[base] = -q.i;
        rotations[base + 1] = -q.j;
        rotations[base + 2] = -q.k;
        rotations[base + 3] = q.w;
    }
    let applied = batched.add_all_actor_gravity(g, &rotations);
    assert_eq!(applied as usize, actors.len(), "every actor should receive gravity");
    batched.update();

    assert_eq!(
        batched.overstressed_bond_count(),
        per_actor.overstressed_bond_count(),
        "batched vs per-actor overstressed count"
    );
    assert_eq!(batched.linear_error(), per_actor.linear_error(), "batched vs per-actor linear error");
    assert_eq!(batched.angular_error(), per_actor.angular_error(), "batched vs per-actor angular error");
    // Per-actor released-load forces must match too.
    for &ai in &actors {
        let com = Vec3::new(0.0, 0.0, 0.0);
        assert_eq!(
            batched.get_excess_forces(ai, com),
            per_actor.get_excess_forces(ai, com),
            "excess forces for actor {ai}"
        );
    }
}

#[test]
fn empty_rotation_buffer_applies_unrotated_world_gravity() {
    let (nodes, bonds) = arms(4, 5);
    let g = Vec3::new(0.3, -30.0, 0.7);

    let mut per_actor = fractured_solver(&nodes, &bonds);
    let actors: Vec<u32> = per_actor.actors().iter().map(|a| a.actor_index).collect();
    for &ai in &actors {
        per_actor.add_actor_gravity(ai, g); // unrotated
    }
    per_actor.update();

    let mut batched = fractured_solver(&nodes, &bonds);
    batched.add_all_actor_gravity(g, &[]); // empty buffer ⇒ identity rotation for all actors
    batched.update();

    assert_eq!(batched.overstressed_bond_count(), per_actor.overstressed_bond_count());
    assert_eq!(batched.linear_error(), per_actor.linear_error());
}
