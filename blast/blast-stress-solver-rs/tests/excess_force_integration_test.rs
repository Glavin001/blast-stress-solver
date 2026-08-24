//! Integration test for momentum transfer on fracture ("excess force").
//!
//! When an impact breaks bonds, the load the broken bonds were carrying should be released
//! onto the freed fragments so they fly apart — otherwise a shattered wall's pieces just sit
//! there. NVIDIA Blast computes this via `getExcessForces`; the Rust pipeline applies it on
//! each separated actor when `FracturePolicy.apply_excess_forces` is set (default `true`).
//!
//! Note: `DestructibleSet::step` does NOT integrate Rapier physics — the caller runs the
//! `PhysicsPipeline`. So this test runs the pipeline (with zero Rapier gravity) to integrate
//! the excess force into actual fragment velocity, and resets accumulated forces each step so
//! the excess acts as a one-shot kick.
//!
//! The JS pipeline does NOT apply excess forces at all (it relies on resimulation); see gap
//! #8 in blast/TESTING.md.

#![cfg(feature = "rapier")]

use rapier3d::prelude::*;

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

fn wall(cols: u32, rows: u32) -> ScenarioDesc {
    let (bw, bh, bd) = (1.0f32, 0.5f32, 0.5f32);
    let volume = bw * bh * bd;
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    let idx = |c: u32, r: u32| -> u32 { r * cols + c };
    for r in 0..rows {
        for c in 0..cols {
            let x = c as f32 * bw + bw * 0.5 - (cols as f32 * bw) * 0.5;
            let y = bh * 0.5 + r as f32 * bh;
            let mass = if r == 0 { 0.0 } else { volume };
            nodes.push(ScenarioNode { centroid: Vec3::new(x, y, 0.0), mass, volume });
        }
    }
    for r in 0..rows {
        for c in 0..cols - 1 {
            let (n0, n1) = (idx(c, r), idx(c + 1, r));
            let (a, b) = (nodes[n0 as usize].centroid, nodes[n1 as usize].centroid);
            bonds.push(ScenarioBond { node0: n0, node1: n1, centroid: (a + b) * 0.5, normal: Vec3::new(1.0, 0.0, 0.0), area: bh * bd, material: 0, });
        }
    }
    for r in 0..rows - 1 {
        for c in 0..cols {
            let (n0, n1) = (idx(c, r), idx(c, r + 1));
            let (a, b) = (nodes[n0 as usize].centroid, nodes[n1 as usize].centroid);
            bonds.push(ScenarioBond { node0: n0, node1: n1, centroid: (a + b) * 0.5, normal: Vec3::new(0.0, 1.0, 0.0), area: bw * bd, material: 0, });
        }
    }
    ScenarioDesc { nodes, bonds, node_sizes: vec![Vec3::new(bw, bh, bd); (cols * rows) as usize], collider_shapes: Vec::new() }
}

/// Returns the maximum fragment speed after impacting + fracturing the wall, integrating
/// physics with zero Rapier gravity so motion can only come from the excess-force kick.
fn max_fragment_speed(apply_excess_forces: bool) -> f32 {
    let scenario = wall(6, 5);
    let settings = SolverSettings {
        compression_elastic_limit: 0.002, compression_fatal_limit: 0.004,
        tension_elastic_limit: 0.002, tension_fatal_limit: 0.004,
        shear_elastic_limit: 0.002, shear_fatal_limit: 0.004,
        ..SolverSettings::default()
    };
    let policy = FracturePolicy { idle_skip: false, apply_excess_forces, ..FracturePolicy::default() };
    let mut set = DestructibleSet::from_scenario(&scenario, settings, Vec3::ZERO, policy).unwrap();
    let (mut bodies, mut colliders, mut isl, mut ij, mut mj) = (
        RigidBodySet::new(), ColliderSet::new(), IslandManager::new(), ImpulseJointSet::new(), MultibodyJointSet::new(),
    );
    set.initialize(&mut bodies, &mut colliders);

    let ip = IntegrationParameters::default();
    let mut pipeline = PhysicsPipeline::new();
    let (mut bp, mut np, mut ccd) = (BroadPhaseBvh::new(), NarrowPhase::new(), CCDSolver::new());
    let zero_g = vector![0.0, 0.0, 0.0];

    let target = 14u32;
    let pos = scenario.nodes[target as usize].centroid;
    for _ in 0..25 {
        set.add_force(target, pos, Vec3::new(4000.0, 0.0, 0.0));
        set.step(&mut bodies, &mut colliders, &mut isl, &mut ij, &mut mj);
        pipeline.step(&zero_g, &ip, &mut isl, &mut bp, &mut np, &mut bodies, &mut colliders, &mut ij, &mut mj, &mut ccd, &(), &());
        for (_, b) in bodies.iter_mut() {
            b.reset_forces(true);
            b.reset_torques(true);
        }
    }

    bodies
        .iter()
        .filter(|(_, b)| b.is_dynamic())
        .map(|(_, b)| b.linvel().norm())
        .fold(0.0f32, f32::max)
}

/// With excess forces enabled, an impact that shatters the wall must throw the fragments
/// (significant speed). With them disabled, the fragments receive no kick (the impact force
/// only ever entered the stress solver, not the Rapier bodies) and stay put. The large gap
/// between the two is the momentum-transfer mechanism doing its job.
#[test]
fn excess_forces_impart_fragment_momentum() {
    let with = max_fragment_speed(true);
    let without = max_fragment_speed(false);
    assert!(
        without < 1.0e-3,
        "without excess forces, the impact should not move fragments, got {without:.4} m/s"
    );
    assert!(
        with > 1.0,
        "with excess forces, fragments should be thrown by the impact, got {with:.4} m/s"
    );
    assert!(
        with > without * 100.0 + 1.0,
        "excess forces should dominate fragment motion: with={with:.4} m/s vs without={without:.4} m/s"
    );
}
