//! Adjudicates the second pre-existing cross-validation divergence.
//!
//! `wall_strong_gravity_parity` expects 0 fractures for a strong wall under
//! gravity (the recorded JS number) but Rust produces 48. Union-find cannot
//! settle this one — it is not a bookkeeping question but a physics one: are
//! the bonds genuinely over their limit?
//!
//! So read the solver's own per-bond stresses and compare them against the
//! material limits that decide fracture. If the peak stress exceeds the fatal
//! limit, fracturing is the correct behaviour and the recorded 0 is stale.

use blast_stress_solver::scenarios::{build_wall_scenario, WallOptions};
use blast_stress_solver::{ExtStressSolver, SolverSettings, StressLimits, Vec3};

fn main() {
    let s = SolverSettings {
        max_solver_iterations_per_frame: 24,
        compression_elastic_limit: 90_000.0,
        compression_fatal_limit: 270_000.0,
        tension_elastic_limit: 90_000.0,
        tension_fatal_limit: 270_000.0,
        shear_elastic_limit: 120_000.0,
        shear_fatal_limit: 360_000.0,
        ..SolverSettings::default()
    };
    let limits = StressLimits {
        compression_elastic_limit: s.compression_elastic_limit,
        compression_fatal_limit: s.compression_fatal_limit,
        tension_elastic_limit: s.tension_elastic_limit,
        tension_fatal_limit: s.tension_fatal_limit,
        shear_elastic_limit: s.shear_elastic_limit,
        shear_fatal_limit: s.shear_fatal_limit,
    };

    let wall = build_wall_scenario(&WallOptions::default());
    let (nodes, bonds) = wall.to_solver_descs();
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &s).unwrap();

    let mut first_overstress_frame = None;
    let mut peak = (0.0f32, 0.0f32, 0.0f32);
    let mut total = 0u32;

    for frame in 0..60 {
        solver.add_gravity(Vec3::new(0.0, -9.81, 0.0));
        solver.update();

        for st in solver.bond_stresses() {
            peak.0 = peak.0.max(st.compression);
            peak.1 = peak.1.max(st.tension);
            peak.2 = peak.2.max(st.shear);
        }

        let over = solver.overstressed_bond_count();
        if over > 0 {
            if first_overstress_frame.is_none() {
                first_overstress_frame = Some(frame);
            }
            let cmds = solver.generate_fracture_commands();
            total += cmds.iter().map(|c| c.bond_fractures.len() as u32).sum::<u32>();
            if !cmds.is_empty() {
                solver.apply_fracture_commands(&cmds);
            }
        }
    }

    println!("peak bond stress over 60 frames (Pa):");
    println!("  compression {:>12.1}   fatal limit {:>12.1}", peak.0, limits.compression_fatal());
    println!("  tension     {:>12.1}   fatal limit {:>12.1}", peak.1, limits.tension_fatal());
    println!("  shear       {:>12.1}   fatal limit {:>12.1}", peak.2, limits.shear_fatal());
    println!("first overstressed frame: {first_overstress_frame:?}");
    println!("total fractures: {total}   (JS reference recorded 0)");

    let exceeded = peak.0 > limits.compression_fatal()
        || peak.1 > limits.tension_fatal()
        || peak.2 > limits.shear_fatal();
    println!(
        "\nverdict: {}",
        if exceeded {
            "a fatal limit IS exceeded — fracturing is correct, the recorded 0 is stale"
        } else {
            "no fatal limit exceeded — Rust is fracturing bonds it should not"
        }
    );
}
