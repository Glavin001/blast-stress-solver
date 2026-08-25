//! Orientation-dependent gravity stress.
//!
//! Gravity is global ("down"), but a rigid chunk can be at any orientation. A beam loaded
//! perpendicular to its axis feels a bending moment that tries to snap it; the same beam
//! loaded along its axis feels only compression. A correct solver must therefore apply
//! gravity **in each actor's current local frame**.
//!
//! The JS library does this: `destructible-core.ts` reads each actor's body rotation and
//! rotates the gravity vector into the actor's local frame before calling
//! `addActorGravity`. The Rust pipeline does NOT — `DestructibleSet::step` calls the
//! global `add_gravity(self.gravity)`, never `add_actor_gravity`, and never syncs node
//! positions from Rapier. So Rust computes gravity stress on the *authored* geometry,
//! blind to how chunks have rotated during the simulation.
//!
//! - `gravity_direction_changes_fracture_behavior` (CONTROL) passes: the solver respects
//!   the gravity direction relative to the structure's geometry.
//! - `actor_rotation_changes_fracture_behavior` (REPRO, `#[ignore]`) currently FAILS:
//!   rotating a chunk's body during simulation does not change its gravity stress.
//!   See blast/TESTING.md gap #7.

#![cfg(feature = "rapier")]

use rapier3d::na::UnitQuaternion;
use rapier3d::prelude::*;

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

/// A 6-segment horizontal cantilever anchored at node 0, laid out along +X.
fn cantilever() -> ScenarioDesc {
    let n = 6u32;
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    for i in 0..n {
        let mass = if i == 0 { 0.0 } else { 1.0 };
        nodes.push(ScenarioNode { centroid: Vec3::new(i as f32, 0.0, 0.0), mass, volume: 1.0 });
    }
    for i in 0..n - 1 {
        let (a, b) = (nodes[i as usize].centroid, nodes[(i + 1) as usize].centroid);
        bonds.push(ScenarioBond { node0: i, node1: i + 1, centroid: (a + b) * 0.5, normal: Vec3::new(1.0, 0.0, 0.0), area: 0.5, material: 0, });
    }
    ScenarioDesc { nodes, bonds, node_sizes: vec![Vec3::new(0.9, 0.3, 0.3); n as usize], collider_shapes: Vec::new() }
}

fn settings() -> SolverSettings {
    // Weak in tension/shear (bending snaps it), strong in compression (axial load does not).
    SolverSettings {
        compression_elastic_limit: 1.0e5, compression_fatal_limit: 1.0e6,
        tension_elastic_limit: 0.05, tension_fatal_limit: 0.1,
        shear_elastic_limit: 0.05, shear_fatal_limit: 0.1,
        ..SolverSettings::default()
    }
}

fn run(gravity: Vec3, rotate_body_to: Option<UnitQuaternion<Real>>) -> usize {
    let scenario = cantilever();
    let policy = FracturePolicy { idle_skip: false, ..FracturePolicy::default() };
    let mut set = DestructibleSet::from_scenario(&scenario, settings(), gravity, policy).unwrap();
    let (mut bodies, mut colliders, mut isl, mut ij, mut mj) = (
        RigidBodySet::new(), ColliderSet::new(), IslandManager::new(), ImpulseJointSet::new(), MultibodyJointSet::new(),
    );
    set.initialize(&mut bodies, &mut colliders);
    let mut fractures = 0usize;
    for _ in 0..15 {
        if let Some(rot) = rotate_body_to {
            if let Some(h) = set.node_body(1) {
                if let Some(b) = bodies.get_mut(h) {
                    b.set_rotation(rot, true);
                }
            }
        }
        fractures += set.step(&mut bodies, &mut colliders, &mut isl, &mut ij, &mut mj).fractures;
    }
    fractures
}

/// CONTROL — passes. Same horizontal beam, two global gravity directions: perpendicular
/// (`-Y`, a bending load) snaps it; along the axis toward the anchor (`-X`, compression)
/// does not. This proves the solver computes orientation-dependent stress from geometry +
/// gravity direction.
#[test]
fn gravity_direction_changes_fracture_behavior() {
    let bending = run(Vec3::new(0.0, -30.0, 0.0), None);
    let axial = run(Vec3::new(-30.0, 0.0, 0.0), None);
    assert!(
        axial < bending,
        "gravity direction must matter: perpendicular(bending)={bending} fractures, \
         axial(compression)={axial} fractures (expected axial < bending)"
    );
}

/// REGRESSION GUARD (gap #7, fixed). Keep gravity world-`-Y`, but rotate the beam's body a
/// quarter turn about Z so the beam physically points +Y. In the body's local frame the load
/// is now axial (compression), so the solver — now fed per-actor rotated gravity — fractures
/// it far less than the un-rotated bending case. (Before the fix, rotation was ignored and
/// both fractured identically.)
#[test]
fn actor_rotation_changes_fracture_behavior() {
    let bending = run(Vec3::new(0.0, -30.0, 0.0), None);
    // +90° about Z maps the local +X beam axis onto world +Y, so world -Y gravity becomes
    // local -X — i.e. axial/compression in the beam's own frame.
    let quarter = UnitQuaternion::from_axis_angle(&Vector::z_axis(), std::f32::consts::FRAC_PI_2);
    let rotated = run(Vec3::new(0.0, -30.0, 0.0), Some(quarter));
    assert!(
        rotated < bending,
        "rotating the beam so gravity is axial in its local frame should reduce bending \
         fractures (orientation-aware), but got rotated={rotated} vs bending={bending} — the \
         solver is blind to actor orientation"
    );
}
