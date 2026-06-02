//! Structural + behavioral guards that currently PASS — they lock in confidence gained
//! during bug-hunting so future changes can't silently regress it.
//!
//! * shipped scenario builders produce well-formed graphs (unit normals, positive areas,
//!   in-range indices, no self-loops, no negative mass) — element-wise, not just counts;
//! * a collapsing wall is deterministic across runs (no HashMap-ordering nondeterminism in
//!   the split planner for this case — see gap #2);
//! * a realistic cuboid wall stays point-velocity continuous through fracture (the split
//!   COM bug is specific to offset-COM / convex-hull fragments + rotation, not cuboids).

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

fn weak() -> SolverSettings {
    SolverSettings {
        compression_elastic_limit: 0.001, compression_fatal_limit: 0.002,
        tension_elastic_limit: 0.001, tension_fatal_limit: 0.002,
        shear_elastic_limit: 0.001, shear_fatal_limit: 0.002,
        ..SolverSettings::default()
    }
}

fn fresh_world() -> (RigidBodySet, ColliderSet, IslandManager, ImpulseJointSet, MultibodyJointSet) {
    (RigidBodySet::new(), ColliderSet::new(), IslandManager::new(), ImpulseJointSet::new(), MultibodyJointSet::new())
}

#[cfg(feature = "scenarios")]
#[test]
fn shipped_scenario_builders_are_structurally_valid() {
    use blast_stress_solver::scenarios::*;
    let wall = build_wall_scenario(&WallOptions::default());
    let tower = build_tower_scenario(&TowerOptions::default());
    let bridge = build_bridge_scenario(&BridgeOptions::default());
    for (name, s) in [("wall", &wall), ("tower", &tower), ("bridge", &bridge)] {
        let n = s.nodes.len();
        for (bi, b) in s.bonds.iter().enumerate() {
            assert!((b.node0 as usize) < n && (b.node1 as usize) < n, "{name} bond {bi}: node index out of range");
            assert_ne!(b.node0, b.node1, "{name} bond {bi}: self-loop");
            assert!(b.area > 0.0, "{name} bond {bi}: non-positive area {}", b.area);
            let nm = b.normal.magnitude();
            assert!((nm - 1.0).abs() < 1.0e-3, "{name} bond {bi}: non-unit normal (‖n‖={nm})");
        }
        for (ni, node) in s.nodes.iter().enumerate() {
            assert!(node.mass >= 0.0, "{name} node {ni}: negative mass");
            let c = node.centroid;
            assert!(c.x.is_finite() && c.y.is_finite() && c.z.is_finite(), "{name} node {ni}: non-finite centroid");
        }
    }
}

fn run_wall(steps: usize) -> (usize, u32, Vec<(i64, i64, i64)>) {
    let scenario = wall(6, 5);
    let policy = FracturePolicy { idle_skip: false, ..FracturePolicy::default() };
    let mut set = DestructibleSet::from_scenario(&scenario, weak(), Vec3::new(0.0, -30.0, 0.0), policy).unwrap();
    let (mut bodies, mut colliders, mut isl, mut ij, mut mj) = fresh_world();
    set.initialize(&mut bodies, &mut colliders);
    let mut fractures = 0usize;
    for _ in 0..steps {
        fractures += set.step(&mut bodies, &mut colliders, &mut isl, &mut ij, &mut mj).fractures;
    }
    let mut positions: Vec<(i64, i64, i64)> = bodies
        .iter()
        .filter(|(_, b)| b.is_dynamic())
        .map(|(_, b)| {
            let c = b.center_of_mass();
            let q = |v: f32| (v as f64 * 1.0e4).round() as i64;
            (q(c.x), q(c.y), q(c.z))
        })
        .collect();
    positions.sort_unstable();
    (fractures, set.actor_count(), positions)
}

#[test]
fn wall_collapse_is_deterministic() {
    let a = run_wall(40);
    let b = run_wall(40);
    assert_eq!(a, b, "two identical wall collapses diverged (nondeterminism)");
}

#[test]
fn cuboid_wall_split_is_point_velocity_continuous() {
    // Lateral gravity makes the wall topple; cuboid fragments must keep point-velocity
    // continuity (worst error ~0). Hull/offset-COM fragments are covered by
    // kinematic_invariants_test.rs.
    let scenario = wall(6, 5);
    let policy = FracturePolicy { idle_skip: false, ..FracturePolicy::default() };
    let mut set = DestructibleSet::from_scenario(&scenario, weak(), Vec3::new(2.0, -30.0, 0.0), policy).unwrap();
    let (mut bodies, mut colliders, mut isl, mut ij, mut mj) = fresh_world();
    set.initialize(&mut bodies, &mut colliders);
    set.set_record_split_continuity(true);
    let mut worst = 0.0f32;
    let mut samples = 0usize;
    for _ in 0..40 {
        set.clear_split_continuity();
        set.step(&mut bodies, &mut colliders, &mut isl, &mut ij, &mut mj);
        for rec in set.split_continuity() {
            samples += 1;
            assert!(rec.finite, "non-finite continuity sample at node {}", rec.node_index);
            worst = worst.max(rec.point_velocity_error);
        }
    }
    assert!(samples > 0, "the wall should split and produce continuity samples");
    assert!(worst < 1.0e-3, "cuboid wall split should be continuous, worst error = {worst:.6} m/s");
}
