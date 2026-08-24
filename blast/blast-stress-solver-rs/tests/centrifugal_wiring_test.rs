//! Centrifugal acceleration wired into the Rapier step loop.
//!
//! A freely-spinning bar (no anchor, zero gravity) develops internal stress purely from its
//! rotation: every segment needs an inward centripetal force to stay on its circular path, so the
//! radial bonds load in compression. NVIDIA Blast models this by feeding each dynamic actor
//! `ω × (ω × r)` as a body acceleration each frame. `DestructibleSet` only does that when
//! `FracturePolicy.apply_centrifugal` is set, so these tests pin the behavior on both sides of the
//! flag.
//!
//! With gravity zero, a uniform field (gravity) would produce no internal stress; only the
//! position-dependent centrifugal field can fracture the bar, which makes this a clean isolation
//! of the new code path.

#![cfg(feature = "rapier")]

use rapier3d::prelude::*;

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

/// A free-floating 5-segment bar laid out along +X, every node dynamic (no support/anchor).
fn free_bar() -> ScenarioDesc {
    let n = 5u32;
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    for i in 0..n {
        nodes.push(ScenarioNode { centroid: Vec3::new(i as f32, 0.0, 0.0), mass: 1.0, volume: 1.0 });
    }
    for i in 0..n - 1 {
        let (a, b) = (nodes[i as usize].centroid, nodes[(i + 1) as usize].centroid);
        bonds.push(ScenarioBond {
            node0: i,
            node1: i + 1,
            centroid: (a + b) * 0.5,
            normal: Vec3::new(1.0, 0.0, 0.0),
            area: 0.5, material: 0, });
    }
    ScenarioDesc {
        nodes,
        bonds,
        node_sizes: vec![Vec3::new(0.9, 0.3, 0.3); n as usize],
        collider_shapes: Vec::new(),
    }
}

/// A bar spinning about its centre feels inward (centripetal) acceleration `ω × (ω × r)`, so its
/// radial bonds load in COMPRESSION. Make compression weak so the spin snaps it; keep tension and
/// shear strong so nothing else can.
fn settings() -> SolverSettings {
    SolverSettings {
        compression_elastic_limit: 0.5,
        compression_fatal_limit: 1.0,
        tension_elastic_limit: 1.0e6,
        tension_fatal_limit: 1.0e7,
        shear_elastic_limit: 1.0e6,
        shear_fatal_limit: 1.0e7,
        ..SolverSettings::default()
    }
}

/// Spin every dynamic body about +Z at `omega` rad/s each frame and count fractures over 15 steps.
/// Gravity is zero, so any fracture comes solely from the centrifugal path.
fn spin_fractures(omega: f32, centrifugal: bool) -> usize {
    let scenario = free_bar();
    let policy = FracturePolicy {
        idle_skip: false,
        apply_centrifugal: centrifugal,
        ..FracturePolicy::default()
    };
    let mut set =
        DestructibleSet::from_scenario(&scenario, settings(), Vec3::new(0.0, 0.0, 0.0), policy)
            .unwrap();
    let (mut bodies, mut colliders, mut isl, mut ij, mut mj) = (
        RigidBodySet::new(),
        ColliderSet::new(),
        IslandManager::new(),
        ImpulseJointSet::new(),
        MultibodyJointSet::new(),
    );
    set.initialize(&mut bodies, &mut colliders);

    let mut fractures = 0usize;
    for _ in 0..15 {
        // Re-assert the spin on every live dynamic body (root + any split children) so the bar
        // keeps tumbling at a known rate regardless of how the physics step integrated it.
        for (_, body) in bodies.iter_mut() {
            if body.is_dynamic() {
                body.set_angular_damping(0.0);
                body.set_angvel(vector![0.0, 0.0, omega], true);
            }
        }
        fractures += set
            .step(&mut bodies, &mut colliders, &mut isl, &mut ij, &mut mj)
            .fractures;
    }
    fractures
}

/// The flag is what enables the behavior: a fast spin fractures the bar only when
/// `apply_centrifugal` is on. With it off (and zero gravity) nothing stresses the bar.
#[test]
fn centrifugal_flag_enables_spin_fracture() {
    let with = spin_fractures(40.0, true);
    let without = spin_fractures(40.0, false);
    assert_eq!(
        without, 0,
        "zero gravity + centrifugal disabled must not fracture, got {without}"
    );
    assert!(
        with > 0,
        "a fast-spinning bar with centrifugal enabled should fracture, got {with}"
    );
}

/// Centrifugal stress scales with spin: no spin means no centrifugal force, hence no fracture
/// even with the flag enabled.
#[test]
fn centrifugal_no_spin_no_fracture() {
    let still = spin_fractures(0.0, true);
    assert_eq!(
        still, 0,
        "centrifugal enabled but no spin must not fracture, got {still}"
    );
}
