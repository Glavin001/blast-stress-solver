//! Gap #9: the fracture "excess force" kick should be one-shot, but the Rust pipeline applies
//! it with Rapier's `add_force` (a PERSISTENT force) instead of `apply_impulse`. If the
//! consuming app doesn't reset forces every step, a single fracture keeps re-accelerating the
//! fragment every physics step — fragments fly off at ever-increasing speed. This is a strong
//! candidate for the Rust "sudden movement after destruction" report (it doesn't even require
//! rotation, unlike the COM bug in `kinematic_invariants_test.rs`).
//!
//! - `excess_force_kick_is_bounded_when_forces_are_reset` (control) PASSES: reset each step
//!   and the kick behaves one-shot — post-fracture speed is stable.
//! - `excess_force_kick_should_be_one_shot` (REPRO, `#[ignore]`) currently FAILS: without a
//!   reset, post-fracture speed grows unbounded. Remove `#[ignore]` once the pipeline applies
//!   the excess as an impulse (or resets forces).

#![cfg(feature = "rapier")]

use rapier3d::prelude::*;

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

fn wall(cols: u32, rows: u32) -> ScenarioDesc {
    let (bw, bh, bd) = (1.0f32, 0.5f32, 0.5f32);
    let vol = bw * bh * bd;
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    let idx = |c: u32, r: u32| r * cols + c;
    for r in 0..rows {
        for c in 0..cols {
            let x = c as f32 * bw + bw * 0.5 - (cols as f32 * bw) * 0.5;
            let y = bh * 0.5 + r as f32 * bh;
            nodes.push(ScenarioNode { centroid: Vec3::new(x, y, 0.0), mass: if r == 0 { 0.0 } else { vol }, volume: vol });
        }
    }
    for r in 0..rows {
        for c in 0..cols - 1 {
            let (n0, n1) = (idx(c, r), idx(c + 1, r));
            let (a, b) = (nodes[n0 as usize].centroid, nodes[n1 as usize].centroid);
            bonds.push(ScenarioBond { node0: n0, node1: n1, centroid: (a + b) * 0.5, normal: Vec3::new(1.0, 0.0, 0.0), area: bh * bd });
        }
    }
    for r in 0..rows - 1 {
        for c in 0..cols {
            let (n0, n1) = (idx(c, r), idx(c, r + 1));
            let (a, b) = (nodes[n0 as usize].centroid, nodes[n1 as usize].centroid);
            bonds.push(ScenarioBond { node0: n0, node1: n1, centroid: (a + b) * 0.5, normal: Vec3::new(0.0, 1.0, 0.0), area: bw * bd });
        }
    }
    ScenarioDesc { nodes, bonds, node_sizes: vec![Vec3::new(bw, bh, bd); (cols * rows) as usize], collider_shapes: Vec::new() }
}

/// Fracture the wall (impact on frames 0–1), then keep integrating physics with zero Rapier
/// gravity. Returns the max fragment speed early (frame 3) vs late (frame 7). For a one-shot
/// kick these are ~equal; for a persistent force the late value is far larger.
fn early_vs_late_speed(reset_forces_each_step: bool) -> (f32, f32) {
    let scenario = wall(6, 5);
    let settings = SolverSettings {
        compression_elastic_limit: 0.002, compression_fatal_limit: 0.004,
        tension_elastic_limit: 0.002, tension_fatal_limit: 0.004,
        shear_elastic_limit: 0.002, shear_fatal_limit: 0.004,
        ..SolverSettings::default()
    };
    let policy = FracturePolicy { idle_skip: false, ..FracturePolicy::default() };
    let mut set = DestructibleSet::from_scenario(&scenario, settings, Vec3::ZERO, policy).unwrap();
    let (mut b, mut c, mut isl, mut ij, mut mj) = (
        RigidBodySet::new(), ColliderSet::new(), IslandManager::new(), ImpulseJointSet::new(), MultibodyJointSet::new(),
    );
    set.initialize(&mut b, &mut c);
    let ip = IntegrationParameters::default();
    let mut pipe = PhysicsPipeline::new();
    let (mut bp, mut np, mut ccd) = (BroadPhaseBvh::new(), NarrowPhase::new(), CCDSolver::new());
    let zero = vector![0.0, 0.0, 0.0];
    let target = 14u32;
    let pos = scenario.nodes[target as usize].centroid;

    let max_speed = |bodies: &RigidBodySet| {
        bodies.iter().filter(|(_, x)| x.is_dynamic()).map(|(_, x)| x.linvel().norm()).fold(0.0f32, f32::max)
    };

    let (mut early, mut late) = (0.0f32, 0.0f32);
    for i in 0..8 {
        if i < 2 {
            set.add_force(target, pos, Vec3::new(4000.0, 0.0, 0.0));
        }
        set.step(&mut b, &mut c, &mut isl, &mut ij, &mut mj);
        pipe.step(&zero, &ip, &mut isl, &mut bp, &mut np, &mut b, &mut c, &mut ij, &mut mj, &mut ccd, &(), &());
        if reset_forces_each_step {
            for (_, body) in b.iter_mut() {
                body.reset_forces(true);
                body.reset_torques(true);
            }
        }
        if i == 3 {
            early = max_speed(&b);
        }
        if i == 7 {
            late = max_speed(&b);
        }
    }
    (early, late)
}

/// CONTROL — passes. Resetting forces each step makes the kick effectively one-shot: the
/// fragment speed after fracturing is stable, not ever-growing.
#[test]
fn excess_force_kick_is_bounded_when_forces_are_reset() {
    let (early, late) = early_vs_late_speed(true);
    assert!(early > 1.0, "the impact should have thrown fragments (early={early:.2})");
    assert!(
        late <= early * 1.5 + 1.0,
        "with forces reset, post-fracture speed should be stable: early={early:.2} late={late:.2}"
    );
}

/// REGRESSION GUARD (gap #9, fixed). The fracture kick is a one-shot impulse, so even
/// without a per-step force reset the fragment speed is stable after fracturing — it does NOT
/// keep accelerating. (Before the fix this grew unbounded, ~22 -> ~1900 m/s.)
#[test]
fn excess_force_kick_should_be_one_shot() {
    let (early, late) = early_vs_late_speed(false);
    assert!(early > 1.0, "the impact should have thrown fragments (early={early:.2})");
    assert!(
        late <= early * 1.5 + 1.0,
        "fracture kick should be one-shot, but fragment speed kept growing: early={early:.2} late={late:.2} \
         (persistent excess force re-accelerates every step)"
    );
}
