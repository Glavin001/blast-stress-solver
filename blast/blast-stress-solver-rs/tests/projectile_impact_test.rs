//! Behavioral test: a real Rapier projectile hits a destructible wall, with the projectile's
//! contact impulse injected into the stress solver (as the demo/a consumer does). This checks
//! *actual* behavior — not just that the split math round-trips — specifically that
//! excess-force fragments don't fly back at runaway speeds.
//!
//! After the gap #9 fix (2 kg ball @ 30 m/s into a 6x5 wall): fragments are BOUNDED — max
//! fragment speed ~260 m/s (~8.7x the ball; a fragment is ~16x lighter than the ball, so a
//! high speed from the shared impulse is expected) and total fragment momentum ~1.7x the
//! ball's. Before the fix the kick was a persistent force and fragments ran away to ~1900 m/s
//! (>60x). This guards against regressing to that unbounded behavior. Whether ~8.7x is the
//! desired "feel" is a tuning question (TESTING.md gap #11).
#![cfg(feature = "rapier")]

use rapier3d::prelude::*;
use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

// Wall in the x-y plane (thin in z), anchored along the bottom row. Ball comes from +z.
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
    for r in 0..rows { for c in 0..cols - 1 {
        let (n0, n1) = (idx(c, r), idx(c + 1, r)); let (a, b) = (nodes[n0 as usize].centroid, nodes[n1 as usize].centroid);
        bonds.push(ScenarioBond { node0: n0, node1: n1, centroid: (a + b) * 0.5, normal: Vec3::new(1.0, 0.0, 0.0), area: bh * bd });
    }}
    for r in 0..rows - 1 { for c in 0..cols {
        let (n0, n1) = (idx(c, r), idx(c, r + 1)); let (a, b) = (nodes[n0 as usize].centroid, nodes[n1 as usize].centroid);
        bonds.push(ScenarioBond { node0: n0, node1: n1, centroid: (a + b) * 0.5, normal: Vec3::new(0.0, 1.0, 0.0), area: bw * bd });
    }}
    ScenarioDesc { nodes, bonds, node_sizes: vec![Vec3::new(bw, bh, bd); (cols * rows) as usize], collider_shapes: Vec::new() }
}

#[test]
fn projectile_fragments_stay_bounded() {
    let scenario = wall(6, 5);
    let settings = SolverSettings {
        compression_elastic_limit: 0.01, compression_fatal_limit: 0.02,
        tension_elastic_limit: 0.01, tension_fatal_limit: 0.02,
        shear_elastic_limit: 0.01, shear_fatal_limit: 0.02,
        ..SolverSettings::default()
    };
    let policy = FracturePolicy { idle_skip: false, ..FracturePolicy::default() };
    let dt = 1.0 / 60.0;
    let mut set = DestructibleSet::from_scenario(&scenario, settings, Vec3::ZERO, policy).unwrap();
    set.set_time_step(dt);
    let (mut bodies, mut colliders, mut isl, mut ij, mut mj) = (
        RigidBodySet::new(), ColliderSet::new(), IslandManager::new(), ImpulseJointSet::new(), MultibodyJointSet::new());
    set.initialize(&mut bodies, &mut colliders);

    // Projectile: a ball aimed at the wall centre, coming from +z.
    let ball_mass = 2.0f32;
    let ball_speed = 30.0f32;
    let ball_handle = bodies.insert(
        RigidBodyBuilder::dynamic().translation(vector![0.0, 1.5, 2.0]).linvel(vector![0.0, 0.0, -ball_speed]).ccd_enabled(true),
    );
    colliders.insert_with_parent(ColliderBuilder::ball(0.4).mass(ball_mass), ball_handle, &mut bodies);
    let ball_p0 = ball_mass * ball_speed;

    let ip = IntegrationParameters { dt, ..IntegrationParameters::default() };
    let mut pipe = PhysicsPipeline::new();
    let (mut bp, mut np, mut ccd) = (BroadPhaseBvh::new(), NarrowPhase::new(), CCDSolver::new());
    let zero = vector![0.0, 0.0, 0.0];

    // nearest dynamic wall node to a world point (x,y on the wall face)
    let support: Vec<bool> = (0..scenario.nodes.len() as u32).map(|n| set.is_support(n)).collect();
    let nearest_node = |x: f32, y: f32| -> u32 {
        (0..scenario.nodes.len() as u32).filter(|&n| !support[n as usize])
            .min_by(|&a, &b| {
                let ca = scenario.nodes[a as usize].centroid; let cb = scenario.nodes[b as usize].centroid;
                let da = (ca.x - x).powi(2) + (ca.y - y).powi(2); let db = (cb.x - x).powi(2) + (cb.y - y).powi(2);
                da.partial_cmp(&db).unwrap()
            }).unwrap()
    };

    let mut max_frag_speed = 0.0f32;
    for _ in 0..40 {
        let v_before = bodies.get(ball_handle).map(|b| *b.linvel()).unwrap_or(zero);
        pipe.step(&zero, &ip, &mut isl, &mut bp, &mut np, &mut bodies, &mut colliders, &mut ij, &mut mj, &mut ccd, &(), &());
        // Contact-force injection: the ball's momentum change this frame = impulse it delivered.
        if let Some(ball) = bodies.get(ball_handle) {
            let v_after = *ball.linvel();
            let dp = (v_before - v_after) * ball_mass; // impulse delivered to the wall
            if dp.norm() > 1.0e-4 {
                let pos = ball.translation();
                let node = nearest_node(pos.x, pos.y);
                let force = dp / dt; // contact force
                set.add_force(node, Vec3::new(pos.x, pos.y, 0.0), Vec3::new(force.x, force.y, force.z));
            }
        }
        set.step(&mut bodies, &mut colliders, &mut isl, &mut ij, &mut mj);
        // reset accumulated forces so nothing besides our injected impulse persists
        for (_, b) in bodies.iter_mut() { b.reset_forces(true); b.reset_torques(true); }
        // measure dynamic WALL fragment speeds (exclude the ball)
        for (h, b) in bodies.iter() {
            if h == ball_handle || !b.is_dynamic() { continue; }
            max_frag_speed = max_frag_speed.max(b.linvel().norm());
        }
    }
    let ball_v_final = bodies.get(ball_handle).map(|b| b.linvel().norm()).unwrap_or(0.0);
    // total wall-fragment momentum magnitude
    let mut frag_p = 0.0f32;
    let mut frag_bodies = 0;
    for (h, b) in bodies.iter() {
        if h == ball_handle || !b.is_dynamic() { continue; }
        frag_p += b.mass() * b.linvel().norm();
        frag_bodies += 1;
    }
    println!("BALL: m={ball_mass} v0={ball_speed} p0={ball_p0:.2} v_final={ball_v_final:.2}");
    println!("FRAGMENTS: bodies={frag_bodies} max_speed={max_frag_speed:.2} total_|p|={frag_p:.2}");
    println!("RATIOS: max_frag_speed/ball_speed={:.2}  total_frag_p/ball_p0={:.2}", max_frag_speed / ball_speed, frag_p / ball_p0);

    // The impact must actually shatter the wall and produce finite motion.
    assert!(frag_bodies > 0, "the projectile should have fractured the wall");
    assert!(max_frag_speed.is_finite() && frag_p.is_finite(), "non-finite fragment state");
    assert!(max_frag_speed > 1.0, "fragments should be thrown by the impact (got {max_frag_speed:.2})");
    // REGRESSION GUARD: no fragment may fly back at runaway speed. Bounded (~8.7x) after the
    // one-shot-impulse fix; the old persistent-force bug ran away past 60x the ball.
    assert!(
        max_frag_speed < 20.0 * ball_speed,
        "fragment fly-back is excessive ({:.1}x the projectile) — excess force may be unbounded again",
        max_frag_speed / ball_speed,
    );
    // Momentum budget: total fragment momentum stays within a small factor of the projectile's.
    assert!(
        frag_p < 5.0 * ball_p0,
        "total fragment momentum {:.1} >> projectile {:.1} (excess force injecting too much)",
        frag_p, ball_p0,
    );
}
