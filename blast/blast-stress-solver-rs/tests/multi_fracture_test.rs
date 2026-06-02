//! Multi-fracture integrity, determinism, and cascade continuity — driven through the
//! SHIPPED scenario builders (wall/tower) plus a cascading grid, so they exercise the real
//! pipeline under sustained, repeated fracturing rather than a single split.

#![cfg(all(feature = "rapier", feature = "scenarios"))]

use std::collections::HashSet;

use rapier3d::prelude::*;

use blast_stress_solver::rapier::*;
use blast_stress_solver::scenarios::*;
use blast_stress_solver::*;

fn weak() -> SolverSettings {
    SolverSettings {
        compression_elastic_limit: 0.001, compression_fatal_limit: 0.002,
        tension_elastic_limit: 0.001, tension_fatal_limit: 0.002,
        shear_elastic_limit: 0.001, shear_fatal_limit: 0.002,
        ..SolverSettings::default()
    }
}

fn world() -> (RigidBodySet, ColliderSet, IslandManager, ImpulseJointSet, MultibodyJointSet) {
    (RigidBodySet::new(), ColliderSet::new(), IslandManager::new(), ImpulseJointSet::new(), MultibodyJointSet::new())
}

/// Every node must map to exactly one body + one collider, with bidirectional consistency,
/// and no two nodes may share a collider — i.e. the split machinery never corrupts the
/// node↔collider↔body bookkeeping. This is the "no corruption" guard a future performance
/// optimization could trip.
fn assert_mapping_consistent(set: &DestructibleSet, node_count: u32, label: &str) {
    let mut seen_colliders = HashSet::new();
    let mut mapped = 0u32;
    for n in 0..node_count {
        let Some(body) = set.node_body(n) else { continue };
        mapped += 1;
        let col = set
            .node_collider(n)
            .unwrap_or_else(|| panic!("{label}: mapped node {n} has no collider"));
        assert_eq!(set.collider_node(col), Some(n), "{label}: collider→node mismatch for node {n}");
        assert!(set.body_nodes(body).contains(&n), "{label}: body_nodes({body:?}) missing node {n}");
        assert!(seen_colliders.insert(col), "{label}: collider {col:?} shared by multiple nodes");
    }
    assert_eq!(mapped, node_count, "{label}: {mapped}/{node_count} nodes mapped (default policy keeps all)");
}

fn drive(scenario: &ScenarioDesc, gravity: Vec3, steps: usize) -> DestructibleSet {
    let policy = FracturePolicy { idle_skip: false, ..FracturePolicy::default() };
    let mut set = DestructibleSet::from_scenario(scenario, weak(), gravity, policy).unwrap();
    let (mut b, mut c, mut isl, mut ij, mut mj) = world();
    set.initialize(&mut b, &mut c);
    for _ in 0..steps {
        set.step(&mut b, &mut c, &mut isl, &mut ij, &mut mj);
    }
    set
}

#[test]
fn node_mapping_stays_consistent_through_fractures() {
    let wall = build_wall_scenario(&WallOptions::default());
    let wall_nodes = wall.nodes.len() as u32;
    let set = drive(&wall, Vec3::new(2.0, -40.0, 0.0), 40);
    assert_mapping_consistent(&set, wall_nodes, "wall");

    let tower = build_tower_scenario(&TowerOptions::default());
    let tower_nodes = tower.nodes.len() as u32;
    let set = drive(&tower, Vec3::new(2.0, -40.0, 0.0), 40);
    assert_mapping_consistent(&set, tower_nodes, "tower");
}

#[test]
fn tower_collapse_is_deterministic_at_scale() {
    fn run() -> (u32, Vec<(i64, i64, i64)>) {
        let tower = build_tower_scenario(&TowerOptions::default());
        let set = drive(&tower, Vec3::new(0.0, -40.0, 0.0), 40);
        let (mut b, mut c, mut isl, mut ij, mut mj) = world();
        // Re-run from scratch to capture final state (drive consumed its own world).
        let policy = FracturePolicy { idle_skip: false, ..FracturePolicy::default() };
        let mut set2 = DestructibleSet::from_scenario(&tower, weak(), Vec3::new(0.0, -40.0, 0.0), policy).unwrap();
        set2.initialize(&mut b, &mut c);
        for _ in 0..40 {
            set2.step(&mut b, &mut c, &mut isl, &mut ij, &mut mj);
        }
        let _ = set;
        let mut pos: Vec<(i64, i64, i64)> = b
            .iter()
            .filter(|(_, body)| body.is_dynamic())
            .map(|(_, body)| {
                let m = body.center_of_mass();
                let q = |v: f32| (v as f64 * 1.0e4).round() as i64;
                (q(m.x), q(m.y), q(m.z))
            })
            .collect();
        pos.sort_unstable();
        (set2.actor_count(), pos)
    }
    let a = run();
    let b = run();
    assert_eq!(a, b, "tower collapse diverged across identical runs (nondeterminism at scale)");
}

/// A grid anchored along its bottom row with weak vertical bonds: rows detach progressively
/// over many steps (a true multi-LEVEL cascade, not one big split). Cuboid fragments must
/// stay point-velocity continuous through every generation.
fn cascading_grid() -> ScenarioDesc {
    let (rows, cols) = (5u32, 3u32);
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    let idx = |r: u32, c: u32| r * cols + c;
    for r in 0..rows {
        for c in 0..cols {
            let mass = if r == 0 { 0.0 } else { 1.0 };
            nodes.push(ScenarioNode { centroid: Vec3::new(c as f32, r as f32 + 0.5, 0.0), mass, volume: 1.0 });
        }
    }
    // Strong horizontal bonds (hold each row together), weak vertical bonds (snap row by row).
    for r in 0..rows {
        for c in 0..cols - 1 {
            let (n0, n1) = (idx(r, c), idx(r, c + 1));
            let (a, b) = (nodes[n0 as usize].centroid, nodes[n1 as usize].centroid);
            bonds.push(ScenarioBond { node0: n0, node1: n1, centroid: (a + b) * 0.5, normal: Vec3::new(1.0, 0.0, 0.0), area: 2.0 });
        }
    }
    for r in 0..rows - 1 {
        for c in 0..cols {
            let (n0, n1) = (idx(r, c), idx(r + 1, c));
            let (a, b) = (nodes[n0 as usize].centroid, nodes[n1 as usize].centroid);
            bonds.push(ScenarioBond { node0: n0, node1: n1, centroid: (a + b) * 0.5, normal: Vec3::new(0.0, 1.0, 0.0), area: 0.02 });
        }
    }
    let n = (rows * cols) as usize;
    ScenarioDesc { nodes, bonds, node_sizes: vec![Vec3::new(0.9, 0.9, 0.9); n], collider_shapes: Vec::new() }
}

#[test]
fn cascading_cuboid_split_stays_continuous() {
    let scenario = cascading_grid();
    // Rate-limit fractures so they spread across many steps — a true multi-LEVEL cascade
    // (and exercises the per-frame budget) instead of one big simultaneous shatter.
    let policy = FracturePolicy { idle_skip: false, max_fractures_per_frame: 2, ..FracturePolicy::default() };
    let mut set = DestructibleSet::from_scenario(&scenario, weak(), Vec3::new(0.0, -25.0, 0.0), policy).unwrap();
    let (mut b, mut c, mut isl, mut ij, mut mj) = world();
    set.initialize(&mut b, &mut c);
    set.set_record_split_continuity(true);
    let (mut split_steps, mut worst, mut samples) = (0usize, 0.0f32, 0usize);
    for _ in 0..120 {
        set.clear_split_continuity();
        let r = set.step(&mut b, &mut c, &mut isl, &mut ij, &mut mj);
        if r.split_events > 0 {
            split_steps += 1;
        }
        for rec in set.split_continuity() {
            samples += 1;
            assert!(rec.finite, "non-finite continuity sample at node {}", rec.node_index);
            worst = worst.max(rec.point_velocity_error);
        }
    }
    assert!(split_steps >= 2, "expected a multi-level cascade, got {split_steps} split step(s)");
    assert!(samples > 0, "cascade produced no continuity samples");
    assert!(worst < 1.0e-3, "cascading cuboid split should stay continuous, worst = {worst:.6} m/s");
}
