//! Resimulation — the preferred, physically-sound source of fragment momentum.
//!
//! Instead of estimating a kick (NVIDIA Blast's opt-in excess force), resimulation rolls the
//! world back and re-runs the physics step against the ALREADY-fractured pieces, so the actual
//! contact is re-resolved against the fragments by the constraint solver. The library provides
//! the primitives (`BodySnapshots::capture`/`restore`, `capture_resimulation_snapshot`); the
//! caller orchestrates the rollback (as the demo does). These tests validate both the snapshot
//! primitive and the end-to-end sound behavior.

#![cfg(feature = "rapier")]

use rapier3d::prelude::*;

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

fn world() -> (RigidBodySet, ColliderSet, IslandManager, ImpulseJointSet, MultibodyJointSet) {
    (RigidBodySet::new(), ColliderSet::new(), IslandManager::new(), ImpulseJointSet::new(), MultibodyJointSet::new())
}

/// `BodySnapshots::capture` then `restore` must reproduce dynamic body state exactly — the
/// foundation of a transparent rollback (non-fractured bodies are untouched by a resim pass).
#[test]
fn snapshot_round_trip_is_exact() {
    let mut bodies = RigidBodySet::new();
    let h = bodies.insert(
        RigidBodyBuilder::dynamic()
            .translation(vector![1.0, 2.0, 3.0])
            .linvel(vector![4.0, -5.0, 6.0])
            .angvel(vector![0.5, -0.5, 0.25]),
    );
    let snap = BodySnapshots::capture(&bodies);

    // Perturb the body as a physics step would.
    {
        let b = bodies.get_mut(h).unwrap();
        b.set_translation(vector![9.0, 9.0, 9.0], true);
        b.set_linvel(vector![0.0, 0.0, 0.0], true);
        b.set_angvel(vector![0.0, 0.0, 0.0], true);
    }
    snap.restore(&mut bodies);

    let b = bodies.get(h).unwrap();
    let t = b.translation();
    let lv = b.linvel();
    let av = b.angvel();
    assert!((t - vector![1.0, 2.0, 3.0]).norm() < 1e-6, "translation not restored: {t:?}");
    assert!((lv - vector![4.0, -5.0, 6.0]).norm() < 1e-6, "linvel not restored: {lv:?}");
    assert!((av - vector![0.5, -0.5, 0.25]).norm() < 1e-6, "angvel not restored: {av:?}");
}

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

/// End-to-end sound path: a projectile fractures an anchored wall, and with EXCESS FORCE OFF,
/// resimulation (rollback + re-resolve the contact against the fragments) is what makes the
/// pieces move. Without resim and without excess force, the anchored wall sends the impact into
/// the ground and fragments barely move (~0.4 m/s). Resim must give them real, bounded momentum.
#[test]
fn resimulation_imparts_fragment_momentum_without_excess_force() {
    let scenario = wall(6, 5);
    let settings = SolverSettings {
        compression_elastic_limit: 0.01, compression_fatal_limit: 0.02,
        tension_elastic_limit: 0.01, tension_fatal_limit: 0.02,
        shear_elastic_limit: 0.01, shear_fatal_limit: 0.02,
        ..SolverSettings::default()
    };
    // Excess force OFF — momentum must come from re-resolved contact, not an estimated kick.
    let policy = FracturePolicy { idle_skip: false, apply_excess_forces: false, ..FracturePolicy::default() };
    let dt = 1.0 / 60.0;
    let mut set = DestructibleSet::from_scenario(&scenario, settings, Vec3::ZERO, policy).unwrap();
    set.set_resimulation_options(ResimulationOptions { enabled: true, max_passes: 3 });
    let (mut bodies, mut colliders, mut isl, mut ij, mut mj) = world();
    set.initialize(&mut bodies, &mut colliders);

    let (ball_mass, ball_speed) = (2.0f32, 30.0f32);
    let ball = bodies.insert(
        RigidBodyBuilder::dynamic().translation(vector![0.0, 1.5, 2.0]).linvel(vector![0.0, 0.0, -ball_speed]).ccd_enabled(true),
    );
    colliders.insert_with_parent(ColliderBuilder::ball(0.4).mass(ball_mass), ball, &mut bodies);

    let ip = IntegrationParameters { dt, ..IntegrationParameters::default() };
    let mut pipe = PhysicsPipeline::new();
    let (mut bp, mut np, mut ccd) = (BroadPhaseBvh::new(), NarrowPhase::new(), CCDSolver::new());
    let zero = vector![0.0, 0.0, 0.0];
    let support: Vec<bool> = (0..scenario.nodes.len() as u32).map(|n| set.is_support(n)).collect();
    let nearest = |x: f32, y: f32| -> u32 {
        (0..scenario.nodes.len() as u32).filter(|&n| !support[n as usize]).min_by(|&a, &b| {
            let ca = scenario.nodes[a as usize].centroid; let cb = scenario.nodes[b as usize].centroid;
            ((ca.x - x).powi(2) + (ca.y - y).powi(2)).partial_cmp(&((cb.x - x).powi(2) + (cb.y - y).powi(2))).unwrap()
        }).unwrap()
    };

    let mut max_frag_speed = 0.0f32;
    for _ in 0..40 {
        // One outer frame, with up to `max_passes` resim rollbacks (mirrors the demo loop).
        let mut snapshot = Some(BodySnapshots::capture(&bodies));
        let mut passes_left = set.resimulation_options().max_passes;
        loop {
            let vb = bodies.get(ball).map(|b| *b.linvel()).unwrap_or(zero);
            pipe.step(&zero, &ip, &mut isl, &mut bp, &mut np, &mut bodies, &mut colliders, &mut ij, &mut mj, &mut ccd, &(), &());
            if let Some(b) = bodies.get(ball) {
                let dp = (vb - *b.linvel()) * ball_mass;
                if dp.norm() > 1e-4 {
                    let p = b.translation();
                    let n = nearest(p.x, p.y);
                    let f = dp / dt;
                    set.add_force(n, Vec3::new(p.x, p.y, 0.0), Vec3::new(f.x, f.y, f.z));
                }
            }
            let r = set.step(&mut bodies, &mut colliders, &mut isl, &mut ij, &mut mj);
            let fractured = r.split_events > 0 || r.new_bodies > 0;
            if !fractured || passes_left == 0 {
                break;
            }
            // Roll the pre-existing bodies back; the new fragment bodies persist, so the next
            // pass re-resolves the contact against them.
            if let Some(snap) = snapshot.as_ref() {
                snap.restore(&mut bodies);
            }
            passes_left -= 1;
            snapshot = Some(BodySnapshots::capture(&bodies));
        }
        for (h, b) in bodies.iter() {
            if h == ball || !b.is_dynamic() { continue; }
            max_frag_speed = max_frag_speed.max(b.linvel().norm());
        }
    }
    println!("resim (excess off): max_frag_speed={max_frag_speed:.2} (ball {ball_speed})");
    // Resim must move the fragments (more than the ~0.4 m/s no-resim/no-excess baseline)...
    assert!(max_frag_speed > 2.0, "resim should impart fragment momentum, got {max_frag_speed:.2} m/s");
    // ...and the re-resolved contact keeps it bounded (no estimation runaway).
    assert!(max_frag_speed < 10.0 * ball_speed, "resim fragment speed unbounded: {max_frag_speed:.2}");
}
