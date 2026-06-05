//! Per-frame spike / percentile profiler over a full destruction timeline.
//!
//! Unlike criterion (which reports a stable mean/median of a repeated op), this drives
//! a *realistic* one-shot destruction run — gravity settle, then a projectile impact and
//! the fracture cascade + topology edits + resim that follow — and reports the per-frame
//! wall-clock distribution (mean / p50 / p90 / p99 / **max**), where the time goes
//! (physics vs. stress-step vs. resim vs. Rapier topology edits), and a behavioral
//! fingerprint (bonds broken, settled shape) so we can confirm an optimization did not
//! change *what* happens, only how fast.
//!
//!   cargo run --release --example frame_profile --features bench-support
//!   cargo run --release --example frame_profile --features bench-support -- --frames 360

use blast_stress_solver::bench_harness::*;
use blast_stress_solver::rapier::{FracturePolicy, ResimulationOptions};
use blast_stress_solver::*;

struct Case {
    name: &'static str,
    scenario: ScenarioDesc,
    cfg: SimConfig,
    /// Projectile fired at `impact_frame` (None = pure gravity timeline).
    projectile: Option<Projectile>,
    impact_frame: u32,
}

fn bounds(scn: &ScenarioDesc) -> (Vec3, Vec3) {
    let mut lo = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut hi = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for n in &scn.nodes {
        lo.x = lo.x.min(n.centroid.x);
        lo.y = lo.y.min(n.centroid.y);
        lo.z = lo.z.min(n.centroid.z);
        hi.x = hi.x.max(n.centroid.x);
        hi.y = hi.y.max(n.centroid.y);
        hi.z = hi.z.max(n.centroid.z);
    }
    (lo, hi)
}

/// A projectile aimed at the middle of the structure's `-z` face, travelling `+z`.
fn ball_at_face(scn: &ScenarioDesc, mass: f32, radius: f32, speed: f32) -> Projectile {
    let (lo, hi) = bounds(scn);
    let cx = 0.5 * (lo.x + hi.x);
    let cy = 0.5 * (lo.y + hi.y);
    Projectile {
        spawn: Vec3::new(cx, cy, lo.z - 3.0),
        velocity: Vec3::new(0.0, 0.0, speed),
        radius,
        mass,
    }
}

fn print_run(name: &str, reports: &[FrameReport], fp0: &QualityFingerprint, fp1: &QualityFingerprint) {
    let totals: Vec<f64> = reports.iter().map(|r| r.total_ms).collect();
    let physics: Vec<f64> = reports.iter().map(|r| r.physics_ms).collect();
    let solver: Vec<f64> = reports.iter().map(|r| r.solver_step_ms).collect();
    let resim: Vec<f64> = reports.iter().map(|r| r.resim_ms + r.resim_physics_ms).collect();
    let edits: Vec<f64> = reports.iter().map(|r| r.split_edit_ms).collect();

    let st = timing_stats(&totals);
    let frac: usize = reports.iter().map(|r| r.fractures).sum();
    let new_bodies: usize = reports.iter().map(|r| r.new_bodies).sum();
    let max_passes = reports.iter().map(|r| r.passes).max().unwrap_or(0);
    let peak_dyn = reports.iter().map(|r| r.dynamic_bodies).max().unwrap_or(0);

    // Find the worst (spike) frame.
    let (spike_i, spike) = reports
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_ms.partial_cmp(&b.1.total_ms).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, r)| (i, *r))
        .unwrap_or((0, FrameReport::default()));

    println!("\n══════════════════════════════════════════════════════════════════════");
    println!("  {name}");
    println!("──────────────────────────────────────────────────────────────────────");
    println!(
        "  frames={}  fractures={}  new_bodies={}  peak_dyn_bodies={}  max_resim_passes={}",
        st.frames, frac, new_bodies, peak_dyn, max_passes
    );
    println!(
        "  TOTAL ms/frame   mean={:.3}  p50={:.3}  p90={:.3}  p99={:.3}  MAX={:.3}",
        st.mean_ms, st.p50_ms, st.p90_ms, st.p99_ms, st.max_ms
    );
    println!(
        "  budget misses    >60fps(16.7ms)={}   >120fps(8.33ms)={}",
        st.over_60fps, st.over_120fps
    );
    println!(
        "  component means  physics={:.3}  stress_step={:.3}  resim(phys+snap)={:.3}  rapier_edits={:.3}",
        avg(&physics),
        avg(&solver),
        avg(&resim),
        avg(&edits)
    );
    // Solver-step sub-phase breakdown (mirrors the web profiler) + island-aware stats.
    let g: Vec<f64> = reports.iter().map(|r| r.solver_gravity_inject_ms).collect();
    let ci: Vec<f64> = reports.iter().map(|r| r.solver_contact_inject_ms).collect();
    let sv: Vec<f64> = reports.iter().map(|r| r.solver_solve_ms).collect();
    let max_islands = reports.iter().map(|r| r.islands_total).max().unwrap_or(0);
    let max_skipped = reports.iter().map(|r| r.islands_skipped).max().unwrap_or(0);
    println!(
        "  solver subphases gravity={:.3}  contact_inject={:.3}  solve={:.3}  (contact: resolve={:.3} grid={:.3} splash={:.3} submit={:.3})",
        avg(&g),
        avg(&ci),
        avg(&sv),
        avg(&reports.iter().map(|r| r.contact_inject_resolve_ms).collect::<Vec<_>>()),
        avg(&reports.iter().map(|r| r.contact_inject_grid_ms).collect::<Vec<_>>()),
        avg(&reports.iter().map(|r| r.contact_inject_splash_ms).collect::<Vec<_>>()),
        avg(&reports.iter().map(|r| r.contact_inject_submit_ms).collect::<Vec<_>>()),
    );
    println!("  islands          max_total={max_islands}  max_skipped={max_skipped}");
    println!(
        "  SPIKE @frame {}  total={:.3}ms  (physics={:.3} stress_step={:.3} resim_phys={:.3} snap={:.3} edits={:.3})  fractures={} new_bodies={} passes={}",
        spike_i, spike.total_ms, spike.physics_ms, spike.solver_step_ms, spike.resim_physics_ms,
        spike.resim_ms, spike.split_edit_ms, spike.fractures, spike.new_bodies, spike.passes
    );
    println!(
        "  fingerprint  before: bonds={} bodies={} dyn={} actors={}",
        fp0.active_bonds, fp0.bodies, fp0.dynamic_bodies, fp0.actors
    );
    println!(
        "               after:  bonds={} bodies={} dyn={} actors={}  com=({:.2},{:.2},{:.2})",
        fp1.active_bonds, fp1.bodies, fp1.dynamic_bodies, fp1.actors, fp1.com[0], fp1.com[1], fp1.com[2]
    );
}

fn avg(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn run_case(case: Case, frames: u32) {
    let mut sim = Sim::new(&case.scenario, case.cfg);
    let fp0 = sim.fingerprint();
    let mut reports = Vec::with_capacity(frames as usize);
    for f in 0..frames {
        if let Some(p) = case.projectile {
            if f == case.impact_frame {
                sim.spawn_projectile(&p);
            }
        }
        reports.push(sim.step_frame());
    }
    let fp1 = sim.fingerprint();
    print_run(case.name, &reports, &fp0, &fp1);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let frames = args
        .iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(300);

    println!("Per-frame spike profiler — {frames} frames/case (release build recommended)");

    // No-fracture knobs (idle_skip off so the solver runs every frame — worst case for steady cost).
    let always_solve = FracturePolicy {
        idle_skip: false,
        apply_excess_forces: false,
        ..FracturePolicy::default()
    };
    let resim = ResimulationOptions { enabled: true, max_passes: 2 };

    // 1) Programmatic wall — weak material, projectile shatter (lots of topology churn).
    {
        let scn = wall(20, 10, 1);
        run_case(
            Case {
                name: "wall 20x10x1 (weak, ball)",
                scenario: scn.clone(),
                cfg: SimConfig {
                    settings: Material::Weak.settings(),
                    policy: always_solve,
                    resim,
                    ..SimConfig::default()
                },
                projectile: Some(ball_at_face(&scn, 5.0, 0.4, 35.0)),
                impact_frame: 30,
            },
            frames,
        );
    }

    // 2) Large programmatic wall — steady solver cost at scale (strong, gravity only).
    {
        let scn = wall(36, 18, 2);
        run_case(
            Case {
                name: "wall 36x18x2 (strong, gravity only — steady solver cost @ scale)",
                scenario: scn.clone(),
                cfg: SimConfig {
                    settings: Material::Strong.settings(),
                    policy: always_solve,
                    resim,
                    ..SimConfig::default()
                },
                projectile: None,
                impact_frame: 0,
            },
            frames,
        );
    }

    // 3) Tower — weak, ball.
    {
        let scn = tower(4, 16);
        run_case(
            Case {
                name: "tower 4x16 (weak, ball)",
                scenario: scn.clone(),
                cfg: SimConfig {
                    settings: Material::Weak.settings(),
                    policy: always_solve,
                    resim,
                    ..SimConfig::default()
                },
                projectile: Some(ball_at_face(&scn, 6.0, 0.4, 30.0)),
                impact_frame: 30,
            },
            frames,
        );
    }

    // 4) Committed scene packs (real fractured geometry the user called out).
    for pack in ["fractured-tower", "fractured-bridge", "brick-building"] {
        if let Some(loaded) = try_load_scene(pack) {
            let scn = loaded.scenario.clone();
            // A hard hit (heavy, fast) to breach these tough, tuned real-geometry structures
            // so we exercise the fracture/topology/resim path on real meshes.
            let proj = ball_at_face(&scn, 150.0, 0.6, 60.0);
            run_case(
                Case {
                    name: pack,
                    scenario: scn,
                    cfg: config_for_loaded(
                        &loaded,
                        SimConfig {
                            policy: always_solve,
                            resim,
                            ..SimConfig::default()
                        },
                    ),
                    projectile: Some(proj),
                    impact_frame: 30,
                },
                frames,
            );
        }
    }

    // 5) High-rise — the headline large structure (930 nodes / 3261 bonds), documented
    //    wrecking ball (0.6 m, 2500 kg, 18 m/s).
    if let Some(loaded) = try_load_scene("high-rise") {
        let scn = loaded.scenario.clone();
        run_case(
            Case {
                name: "high-rise (930 nodes / 3261 bonds, wrecking ball)",
                scenario: scn.clone(),
                cfg: config_for_loaded(
                    &loaded,
                    SimConfig {
                        policy: always_solve,
                        resim,
                        ..SimConfig::default()
                    },
                ),
                projectile: Some(ball_at_face(&scn, 2500.0, 0.6, 18.0)),
                impact_frame: 60,
            },
            frames,
        );
    }
}
