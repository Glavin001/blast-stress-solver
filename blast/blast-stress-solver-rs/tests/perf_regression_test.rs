//! Behavioral-equivalence guard for performance optimizations ("no cheating").
//!
//! A performance change is only legitimate if it does the *same physics* faster — it
//! must not quietly reduce fracture work, clamp breaks, or otherwise trade quality for
//! speed. These tests pin the *outcome* of fixed scenarios via [`QualityFingerprint`]
//! and assert:
//!   1. **Determinism** — the same scenario run twice produces an identical fingerprint.
//!   2. **Genuine destruction** — a shatter actually fragments the structure (an
//!      optimization that secretly limited fractures would fall below the floor).
//!   3. **Gravity stability** — a strongly-anchored structure under gravity does not
//!      fracture (an optimization that broke the solver would spuriously fracture).
//!   4. **Golden bands** — headline counts stay within a tolerance window of the
//!      recorded baseline, so a regression that materially changes the result fails.
//!
//! Run: `cargo test --features bench-support --test perf_regression_test -- --nocapture`

#![cfg(feature = "bench-support")]

use blast_stress_solver::bench_harness::*;
use blast_stress_solver::rapier::{FracturePolicy, ResimulationOptions};
use blast_stress_solver::*;

fn weak_ball_wall() -> (ScenarioDesc, SimConfig, Projectile) {
    let scn = wall(16, 8, 1);
    let cfg = SimConfig {
        settings: Material::Weak.settings(),
        policy: FracturePolicy {
            idle_skip: false,
            apply_excess_forces: false,
            ..FracturePolicy::default()
        },
        resim: ResimulationOptions { enabled: true, max_passes: 2 },
        with_ground: true,
        ..SimConfig::default()
    };
    let proj = Projectile {
        spawn: Vec3::new(0.0, 1.5, -3.0),
        velocity: Vec3::new(0.0, 0.0, 35.0),
        radius: 0.4,
        mass: 5.0,
    };
    (scn, cfg, proj)
}

/// Drive the fixed shatter scenario for `frames`, firing the ball at frame 20.
fn run_shatter(frames: u32) -> QualityFingerprint {
    let (scn, cfg, proj) = weak_ball_wall();
    let mut sim = Sim::new(&scn, cfg);
    for f in 0..frames {
        if f == 20 {
            sim.spawn_projectile(&proj);
        }
        sim.step_frame();
    }
    sim.fingerprint()
}

#[test]
fn shatter_is_deterministic() {
    let a = run_shatter(180);
    let b = run_shatter(180);
    // The discrete outcome (bonds broken, body/actor counts) must be identical run-to-run;
    // positions may differ by a sub-ULP (NvBlast's actor set iterates in pointer order, which
    // perturbs only float accumulation order, not the structural result). `approx_eq` checks
    // counts exactly and COM/spread to within 1mm.
    assert!(
        a.approx_eq(&b, 1.0e-2),
        "identical scenario must be deterministic run-to-run\n a={a:?}\n b={b:?}"
    );
}

#[test]
fn shatter_genuinely_fragments() {
    let fp = run_shatter(180);
    eprintln!("[shatter] fingerprint = {fp:?}");
    // The wall has 16*8=128 nodes, 16*7 + 15*8 = 232 bonds, 16 supports (row 0).
    // A 5 kg ball at 35 m/s into weak infill MUST genuinely break it apart.
    assert!(
        fp.actors >= 8,
        "shatter should fragment into many actors (got {}). A fracture-limiting \
         'optimization' would fail here.",
        fp.actors
    );
    assert!(
        fp.dynamic_bodies >= 8,
        "shatter should create many dynamic fragments (got {})",
        fp.dynamic_bodies
    );
    // ...and it must be bounded — not every bond gone (that would be its own bug).
    assert!(
        fp.active_bonds > 0,
        "not literally every bond should break in a localized shatter"
    );
}

#[test]
fn gravity_stable_structure_does_not_fracture() {
    // Strong, anchored wall under gravity: a correct solver fractures nothing.
    let scn = wall(16, 8, 1);
    let cfg = SimConfig {
        settings: Material::Strong.settings(),
        policy: FracturePolicy {
            idle_skip: false,
            ..FracturePolicy::default()
        },
        resim: ResimulationOptions { enabled: true, max_passes: 2 },
        with_ground: true,
        ..SimConfig::default()
    };
    let mut sim = Sim::new(&scn, cfg);
    let total_bonds = sim.fingerprint().active_bonds;
    for _ in 0..240 {
        sim.step_frame();
    }
    let fp = sim.fingerprint();
    eprintln!("[gravity] fingerprint = {fp:?}");
    assert_eq!(
        fp.active_bonds, total_bonds,
        "a strong anchored wall must not lose any bonds under gravity"
    );
    assert_eq!(fp.actors, 1, "structure must remain one connected actor");
}

// ── Golden bands ────────────────────────────────────────────────────────────
// Recorded from the baseline build. An optimization may perturb exact float
// ordering (and thus a borderline bond), so these are *bands*, not equalities —
// but a band wide enough to catch a real regression (e.g. fractures vanishing).
// Update deliberately (with justification) if a legitimate change shifts them.
#[test]
fn shatter_matches_golden_band() {
    let fp = run_shatter(180);
    eprintln!("[golden] fingerprint = {fp:?}");
    // Filled from the recorded baseline; see the printed value above.
    const GOLDEN_ACTORS: f32 = GOLDEN_ACTORS_BASELINE;
    const GOLDEN_DYN: f32 = GOLDEN_DYN_BASELINE;
    let within = |got: f32, want: f32, frac: f32| (got - want).abs() <= want * frac + 2.0;
    assert!(
        within(fp.actors as f32, GOLDEN_ACTORS, 0.20),
        "actor count {} drifted >20% from golden {}",
        fp.actors,
        GOLDEN_ACTORS
    );
    assert!(
        within(fp.dynamic_bodies as f32, GOLDEN_DYN, 0.20),
        "dynamic-body count {} drifted >20% from golden {}",
        fp.dynamic_bodies,
        GOLDEN_DYN
    );
}

// Baselines recorded from the deterministic build (wall 16x8, weak, ball @ frame 20,
// 180 frames): the shatter fragments the wall into ~97 actors / ~96 dynamic bodies.
const GOLDEN_ACTORS_BASELINE: f32 = 97.0;
const GOLDEN_DYN_BASELINE: f32 = 96.0;
