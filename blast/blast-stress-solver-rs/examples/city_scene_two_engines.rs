//! Runs a real vibe-land `/city` scene pack through the standardized pipeline
//! on both engines.
//!
//! The point is not the numbers — it is that the *production content* loads and
//! fractures through the engine-neutral core, with the only difference between
//! runs being which adapter is constructed.

use std::path::PathBuf;

use blast_stress_solver::backend::PhysicsBackend;
use blast_stress_solver::backends::{PhysXWorld, RapierWorld};
use blast_stress_solver::pipeline::{Destructible, DestructibleConfig};
use blast_stress_solver::scenarios::load_scenario_file;
use blast_stress_solver::types::Vec3;

const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn run<B: PhysicsBackend>(backend: &mut B, label: &str, path: &PathBuf, frames: u32) {
    let loaded = match load_scenario_file(path) {
        Ok(l) => l,
        Err(e) => {
            println!("[{label}] could not load {}: {e}", path.display());
            return;
        }
    };
    let scn = loaded.scenario;
    let mut cfg = DestructibleConfig { gravity: G, ..Default::default() };
    cfg.solver = loaded.settings;

    let Some(mut d) = Destructible::attach(backend, &scn, cfg) else {
        println!("[{label}] attach failed");
        return;
    };
    let bodies_at_attach = d.bodies().len();
    let actors_at_attach = d.solver().actor_count();

    // Standing under gravity is the correct answer for an authored high-rise;
    // to exercise the fracture path we have to actually hit it. Pick the node
    // nearest a chosen impact point and drive a large force through it.
    let impact_at = Vec3::new(0.0, 6.0, 0.0);
    let mut hit_node = 0u32;
    let mut best = f32::MAX;
    for (i, n) in scn.nodes.iter().enumerate() {
        if n.mass <= 0.0 {
            continue;
        }
        let dd = (n.centroid - impact_at).magnitude_squared();
        if dd < best {
            best = dd;
            hit_node = i as u32;
        }
    }
    let impact_force: f32 = std::env::var("IMPACT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);

    let mut fractures = 0usize;
    let mut splits = 0usize;
    let mut created = 0usize;
    for f in 0..frames {
        if f == 5 && impact_force > 0.0 {
            let p = scn.nodes[hit_node as usize].centroid;
            d.add_force(hit_node, p, Vec3::new(0.0, 0.0, impact_force));
        }
        backend.step(1.0 / 60.0);
        let r = d.step(backend, 1.0 / 60.0);
        fractures += r.fractures;
        splits += r.split_events;
        created += r.bodies_created;
    }

    println!(
        "[{label:<10}] nodes={:<5} bonds={:<5} attach: {bodies_at_attach} bodies / {actors_at_attach} actors \
         -> after {frames}f: fractures={fractures} splits={splits} created={created} \
         actors={} bodies={}",
        scn.nodes.len(),
        scn.bonds.len(),
        d.solver().actor_count(),
        d.bodies().len()
    );
}

fn main() {
    let scene = std::env::args().nth(1).unwrap_or_else(|| {
        "/root/workspace/vibe-land-3/destruction/assets/scenes/high-rise-3f-local.json".to_string()
    });
    let path = PathBuf::from(scene);
    let frames: u32 = std::env::var("FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    println!("scene: {}", path.display());

    let mut rapier = RapierWorld::new(G);
    run(&mut rapier, "rapier", &path, frames);

    match PhysXWorld::new_cpu(G, 2) {
        Some(mut w) => run(&mut w, "physx-cpu", &path, frames),
        None => println!("[physx-cpu ] scene unavailable"),
    }
    match PhysXWorld::new_gpu(G, 2) {
        Some(mut w) => run(&mut w, "physx-gpu", &path, frames),
        None => println!("[physx-gpu ] no usable GPU scene"),
    }
}
