//! Headless parameter sweep for the high-rise scene pack.
//!
//! Loads the committed shared scene pack and drives a scripted "wrecking ball"
//! (a per-frame impact force at a target node) across a grid of impact forces and
//! target types, reporting how the structure responds: bonds broken (total + by
//! skeleton/infill), peak/final actor count, and the broken-bond fraction (the
//! glass-vs-local signal). Emits CSV to stdout.
//!
//!   cargo run --example high_rise_sweep --features scenarios
//!   cargo run --example high_rise_sweep --features scenarios -- --frames 240
//!
//! This is the tuning tool: use it to pick realistic strengths/ball energy so the
//! ball punches a LOCAL hole instead of shattering the whole structure like glass.

#[cfg(not(feature = "scenarios"))]
fn main() {
    eprintln!("run with --features scenarios");
}

#[cfg(feature = "scenarios")]
fn main() {
    use blast_stress_solver::scenarios::*;
    use blast_stress_solver::*;

    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str, default: u32| -> u32 {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let frames = arg("--frames", 200);
    let impact_frames = arg("--impact-frames", 6);

    let loaded = match load_scenario_file(&high_rise_scene_path()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not load high-rise scene pack: {e}");
            eprintln!("generate it: (cd ../blast-stress-solver && npm run build && npm run generate:high-rise)");
            std::process::exit(1);
        }
    };

    let total_bonds = loaded.scenario.bonds.len();
    eprintln!(
        "loaded high-rise: nodes={} bonds={} mat_scale={:.2e} limits(comp/ten/shear fatal)={:.2e}/{:.2e}/{:.2e}",
        loaded.scenario.nodes.len(),
        total_bonds,
        loaded.material_scale,
        loaded.settings.compression_fatal_limit,
        loaded.settings.tension_fatal_limit,
        loaded.settings.shear_fatal_limit,
    );

    // Bounds for targeting.
    let (mut lo, mut hi) = (
        Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
        Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
    );
    for n in &loaded.scenario.nodes {
        lo.x = lo.x.min(n.centroid.x);
        lo.y = lo.y.min(n.centroid.y);
        lo.z = lo.z.min(n.centroid.z);
        hi.x = hi.x.max(n.centroid.x);
        hi.y = hi.y.max(n.centroid.y);
        hi.z = hi.z.max(n.centroid.z);
    }

    let nearest = |point: Vec3, want: &str| -> u32 {
        let mut best = 0u32;
        let mut best_d = f32::INFINITY;
        for (i, n) in loaded.scenario.nodes.iter().enumerate() {
            if loaded.node_types.get(i).map(String::as_str).unwrap_or("") != want {
                continue;
            }
            let d = (n.centroid.x - point.x).powi(2)
                + (n.centroid.y - point.y).powi(2)
                + (n.centroid.z - point.z).powi(2);
            if d < best_d {
                best_d = d;
                best = i as u32;
            }
        }
        best
    };

    let mid_y = (lo.y + hi.y) * 0.5;
    let targets = [
        ("infill_mid", nearest(Vec3::new(0.0, mid_y, lo.z), "infill")),
        ("column_low", nearest(Vec3::new(0.0, lo.y + 4.0, lo.z), "column")),
        ("slab_mid", nearest(Vec3::new(0.0, mid_y, 0.0), "slab")),
    ];

    // Impact force magnitudes (Newtons) to sweep.
    let forces = [
        1.0e4_f32, 3.0e4, 1.0e5, 3.0e5, 1.0e6, 3.0e6, 1.0e7, 3.0e7, 1.0e8, 3.0e8,
    ];

    println!("target,force_N,total_fractures,skeleton,infill,foundation,peak_actors,final_actors,frac_ratio");
    for (tname, node) in targets {
        let pos = loaded.scenario.nodes[node as usize].centroid;
        for &f in &forces {
            let (nodes, bonds) = loaded.scenario.to_solver_descs();
            let mut solver = ExtStressSolver::new(&nodes, &bonds, &loaded.settings).unwrap();
            let g = loaded.gravity_vec();
            let dir = Vec3::new(0.0, 0.0, 1.0); // into the building (+Z) for the -Z face
            let force = Vec3::new(dir.x * f, dir.y * f, dir.z * f);

            let mut total = 0u32;
            let mut skel = 0u32;
            let mut infill = 0u32;
            let mut found = 0u32;
            let mut peak = 0u32;
            for frame in 0..frames {
                solver.add_gravity(g);
                if frame < impact_frames {
                    solver.add_force(node, pos, force, ForceMode::Force);
                }
                solver.update();
                if solver.overstressed_bond_count() > 0 {
                    let cmds = solver.generate_fracture_commands();
                    for c in &cmds {
                        for bf in &c.bond_fractures {
                            total += 1;
                            let a = loaded.is_skeleton(bf.node_index0 as usize);
                            let b = loaded.is_skeleton(bf.node_index1 as usize);
                            let f0 = loaded.node_types.get(bf.node_index0 as usize).map(String::as_str);
                            let f1 = loaded.node_types.get(bf.node_index1 as usize).map(String::as_str);
                            if f0 == Some("foundation") || f1 == Some("foundation") {
                                found += 1;
                            }
                            if a && b {
                                skel += 1;
                            } else {
                                infill += 1;
                            }
                        }
                    }
                    if !cmds.is_empty() {
                        solver.apply_fracture_commands(&cmds);
                    }
                }
                peak = peak.max(solver.actor_count());
            }
            let final_actors = solver.actor_count();
            let ratio = total as f32 / total_bonds.max(1) as f32;
            println!(
                "{tname},{f:.0},{total},{skel},{infill},{found},{peak},{final_actors},{ratio:.4}"
            );
        }
    }
}
