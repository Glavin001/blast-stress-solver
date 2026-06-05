//! Production contact-force injection + per-body splash grid
//! ([`DestructibleSet::inject_contacts`] / [`contact_force_batch`]), ported from the web runtime.
//!
//! The splash grid is just an acceleration structure: it must produce exactly the same per-node
//! force batch as a brute-force O(nodes) full scan. These tests lock that equivalence (intact and
//! after fracturing), determinism, and that a single contact's batch is order-independent.

#![cfg(all(feature = "rapier", feature = "scenarios"))]

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;
use rapier3d::prelude::*;

/// A `cols × rows` grid wall in the XY plane (1 m spacing) bonded into one intact body. Row 0 is
/// static; the rest are dynamic, so every dynamic node shares one body — ideal for splash.
fn grid_wall(cols: u32, rows: u32) -> ScenarioDesc {
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    let idx = |c: u32, r: u32| r * cols + c;
    for r in 0..rows {
        for c in 0..cols {
            nodes.push(ScenarioNode {
                centroid: Vec3::new(c as f32, r as f32, 0.0),
                mass: if r == 0 { 0.0 } else { 1.0 },
                volume: 1.0,
            });
        }
    }
    let mut bond = |a: u32, b: u32, n: Vec3| {
        let (pa, pb) = (nodes[a as usize].centroid, nodes[b as usize].centroid);
        bonds.push(ScenarioBond { node0: a, node1: b, centroid: (pa + pb) * 0.5, normal: n, area: 0.5 });
    };
    for r in 0..rows {
        for c in 0..cols {
            if c + 1 < cols {
                bond(idx(c, r), idx(c + 1, r), Vec3::new(1.0, 0.0, 0.0));
            }
            if r + 1 < rows {
                bond(idx(c, r), idx(c, r + 1), Vec3::new(0.0, 1.0, 0.0));
            }
        }
    }
    let n = nodes.len();
    ScenarioDesc { nodes, bonds, node_sizes: vec![Vec3::new(0.9, 0.9, 0.9); n], collider_shapes: Vec::new() }
}

/// Brute-force reference batch: the exact two-pass resolve→splash math from `contact_force_batch`,
/// but with an O(nodes) full scan instead of the grid. Returns (node, px,py,pz, fx,fy,fz) entries.
fn brute_force_batch(
    set: &DestructibleSet,
    bodies: &RigidBodySet,
    centroids: &[Vec3],
    contacts: &[ContactInjection],
    radius: f32,
    scale: f32,
) -> Vec<(u32, [f32; 3], [f32; 3])> {
    let mut out = Vec::new();
    for ct in contacts {
        let node = ct.node;
        let Some(bh) = set.node_body(node) else { continue };
        let Some(body) = bodies.get(bh) else { continue };
        let wf = vector![ct.world_force.x, ct.world_force.y, ct.world_force.z];
        let lf = body.rotation().inverse_transform_vector(&wf);
        let local = [lf.x * scale, lf.y * scale, lf.z * scale];
        let hit = centroids[node as usize];
        out.push((node, [hit.x, hit.y, hit.z], local));
        for ci in 0..centroids.len() as u32 {
            if ci == node || set.node_body(ci) != Some(bh) {
                continue;
            }
            let c = centroids[ci as usize];
            let d2 = (c.x - hit.x).powi(2) + (c.y - hit.y).powi(2) + (c.z - hit.z).powi(2);
            if d2 > radius * radius {
                continue;
            }
            let dist = d2.sqrt();
            let f2 = {
                let f = 1.0 - dist / radius;
                f * f
            };
            if f2 <= 0.0 {
                continue;
            }
            out.push((ci, [c.x, c.y, c.z], [local[0] * f2, local[1] * f2, local[2] * f2]));
        }
    }
    out
}

fn batch_to_entries(b: (Vec<u32>, Vec<f32>, Vec<f32>)) -> Vec<(u32, [f32; 3], [f32; 3])> {
    let (idx, pos, force) = b;
    (0..idx.len())
        .map(|i| {
            (
                idx[i],
                [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]],
                [force[i * 3], force[i * 3 + 1], force[i * 3 + 2]],
            )
        })
        .collect()
}

fn sorted(mut v: Vec<(u32, [f32; 3], [f32; 3])>) -> Vec<(u32, [f32; 3], [f32; 3])> {
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

fn make_set(scenario: &ScenarioDesc) -> (DestructibleSet, RigidBodySet, ColliderSet) {
    let policy = FracturePolicy { idle_skip: false, ..FracturePolicy::default() };
    let settings = SolverSettings {
        compression_elastic_limit: 0.01,
        compression_fatal_limit: 0.02,
        tension_elastic_limit: 0.01,
        tension_fatal_limit: 0.02,
        shear_elastic_limit: 0.01,
        shear_fatal_limit: 0.02,
        ..SolverSettings::default()
    };
    let mut set =
        DestructibleSet::from_scenario(scenario, settings, Vec3::new(0.0, -9.81, 0.0), policy).unwrap();
    let mut bodies = RigidBodySet::new();
    let mut colliders = ColliderSet::new();
    set.initialize(&mut bodies, &mut colliders);
    (set, bodies, colliders)
}

#[test]
fn grid_batch_matches_brute_force_intact() {
    let scenario = grid_wall(7, 7);
    let centroids: Vec<Vec3> = scenario.nodes.iter().map(|n| n.centroid).collect();
    let (mut set, bodies, _c) = make_set(&scenario);

    // A single contact in the middle of the wall (distinct nodes ⇒ order-independent batch).
    let center = 3 * 7 + 3;
    let contacts = vec![ContactInjection { node: center, world_force: Vec3::new(120.0, -40.0, 15.0) }];
    let r = DEFAULT_SPLASH_RADIUS;

    let mut timing = ContactInjectTiming::default();
    let grid = batch_to_entries(set.contact_force_batch(&contacts, r, 1.0, &bodies, &mut timing));
    let brute = brute_force_batch(&set, &bodies, &centroids, &contacts, r, 1.0);

    assert!(grid.len() > 1, "splash should reach neighbours, got {} entries", grid.len());
    assert_eq!(sorted(grid), sorted(brute), "grid batch must equal the full-scan reference");
}

#[test]
fn grid_batch_is_deterministic() {
    let scenario = grid_wall(6, 6);
    let (mut set, bodies, _c) = make_set(&scenario);
    let contacts = vec![ContactInjection { node: 2 * 6 + 2, world_force: Vec3::new(50.0, -10.0, 0.0) }];
    let mut t = ContactInjectTiming::default();
    let a = batch_to_entries(set.contact_force_batch(&contacts, DEFAULT_SPLASH_RADIUS, 1.0, &bodies, &mut t));
    let b = batch_to_entries(set.contact_force_batch(&contacts, DEFAULT_SPLASH_RADIUS, 1.0, &bodies, &mut t));
    assert_eq!(a, b, "same inputs ⇒ identical batch, byte for byte");
}

#[test]
fn grid_rebuilds_and_matches_brute_force_after_fracture() {
    let scenario = grid_wall(8, 8);
    let centroids: Vec<Vec3> = scenario.nodes.iter().map(|n| n.centroid).collect();
    let (mut set, mut bodies, mut colliders) = make_set(&scenario);
    let mut islands = IslandManager::new();
    let mut ij = ImpulseJointSet::new();
    let mut mj = MultibodyJointSet::new();

    // Inject once (builds the grid), then drive several heavy-gravity frames to fracture the wall
    // into multiple bodies (invalidating the grid), then compare again on the new topology.
    let early = vec![ContactInjection { node: 4 * 8 + 4, world_force: Vec3::new(10.0, 0.0, 0.0) }];
    set.inject_contacts(&early, DEFAULT_SPLASH_RADIUS, 1.0, &bodies);
    let mut total_fractures = 0usize;
    for _ in 0..40 {
        let r = set.step(&mut bodies, &mut colliders, &mut islands, &mut ij, &mut mj);
        total_fractures += r.fractures;
    }
    assert!(total_fractures > 0, "the wall should have fractured");

    // Pick a still-live dynamic node and compare the grid batch to a brute-force scan on the
    // post-fracture topology (the grid must have rebuilt and now respect the new body boundaries).
    let live = (0..scenario.nodes.len() as u32)
        .find(|&n| !set.is_support(n) && set.node_body(n).is_some())
        .expect("some node should still be live");
    let contacts = vec![ContactInjection { node: live, world_force: Vec3::new(30.0, -20.0, 5.0) }];
    let mut t = ContactInjectTiming::default();
    let grid = batch_to_entries(set.contact_force_batch(&contacts, DEFAULT_SPLASH_RADIUS, 1.0, &bodies, &mut t));
    let brute = brute_force_batch(&set, &bodies, &centroids, &contacts, DEFAULT_SPLASH_RADIUS, 1.0);
    assert_eq!(sorted(grid), sorted(brute), "post-fracture grid batch must match the full scan");
}
