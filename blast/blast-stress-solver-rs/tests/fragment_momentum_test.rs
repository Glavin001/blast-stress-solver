//! Fragments must get their momentum from physics, and only from physics.
//!
//! When a loaded bond breaks, the strain energy it was holding has to go
//! somewhere. There are exactly two mechanisms that put it there: resimulation,
//! which re-solves the impact contact against the fractured pieces, and the
//! solver's released-load impulse. With neither, a fragment inherits only the
//! parent's velocity field from the rigid-motion fit -- continuity is correct,
//! the energy is silently gone, and a building separates and slides apart
//! instead of bursting.
//!
//! That is exactly the state the shipped PhysX integration was in: the
//! resimulation translation unit was compiled and never called, and excess
//! forces were switched off. Neither shows up in any counter, which is why this
//! file asserts on energy rather than on fracture counts.
//!
//! The other half matters just as much. Excess forces were switched off because
//! they produced a 946 m/s body at -5605 m, and that was recorded as a note
//! rather than a test. A note does not fail CI.
#![cfg(all(feature = "rapier", feature = "scenarios", feature = "physx"))]

use blast_stress_solver::backend::PhysicsBackend;
use blast_stress_solver::backends::{PhysXWorld, RapierWorld};
use blast_stress_solver::pipeline::{Destructible, DestructibleConfig};
use blast_stress_solver::scenarios::{build_wall_scenario, WallOptions};
use blast_stress_solver::types::{ScenarioDesc, SolverSettings, Vec3};

const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);
const DT: f32 = 1.0 / 60.0;
/// Short enough that gravity cannot itself account for a large speed: after
/// 30 ticks of free fall a body is doing 4.9 m/s and has dropped 1.2 m.
const STEPS: usize = 30;

fn scenario() -> ScenarioDesc {
    build_wall_scenario(&WallOptions::default())
}

fn config(excess: bool) -> DestructibleConfig {
    DestructibleConfig {
        gravity: G,
        solver: SolverSettings {
            max_solver_iterations_per_frame: 24,
            compression_elastic_limit: 5.0,
            compression_fatal_limit: 10.0,
            tension_elastic_limit: 5.0,
            tension_fatal_limit: 10.0,
            shear_elastic_limit: 5.0,
            shear_fatal_limit: 10.0,
            ..SolverSettings::default()
        },
        apply_excess_forces: excess,
        min_child_nodes: 1,
        max_new_bodies_per_step: usize::MAX,
        ..Default::default()
    }
}

struct Outcome {
    peak_speed: f32,
    kinetic_energy: f32,
    lowest_y: f32,
    fractures: usize,
}

fn run<B: PhysicsBackend>(backend: &mut B, excess: bool) -> Outcome {
    let mut d = Destructible::attach(backend, &scenario(), config(excess)).expect("attach");
    let mut fractures = 0;
    for _ in 0..STEPS {
        backend.step(DT);
        fractures += d.step(backend, DT).fractures;
    }
    let mut peak_speed = 0.0f32;
    let mut kinetic_energy = 0.0f32;
    let mut lowest_y = f32::MAX;
    for m in d.island_poses(backend) {
        let speed = m.linvel.magnitude();
        peak_speed = peak_speed.max(speed);
        // Mass is not needed for the comparison and the islands here are of
        // similar size, so this is a speed-weighted proxy, named honestly.
        kinetic_energy += speed * speed;
        lowest_y = lowest_y.min(m.pose.translation.y);
    }
    Outcome { peak_speed, kinetic_energy, lowest_y, fractures }
}

/// The gap, demonstrated: turning the released load off measurably removes
/// energy from the scene. If this ever stops holding, either resim has landed
/// (and this file needs rewriting around it) or the impulse is a no-op.
fn assert_released_load_is_the_momentum_source(engine: &str, off: &Outcome, on: &Outcome) {
    assert!(
        on.fractures > 0 && off.fractures > 0,
        "[{engine}] nothing fractured, so neither run exercised the path"
    );
    assert!(
        on.kinetic_energy > off.kinetic_energy,
        "[{engine}] the released load added no energy ({:.3} on vs {:.3} off) -- \
         with resim absent this leaves the pipeline with no source of fragment \
         momentum at all",
        on.kinetic_energy,
        off.kinetic_energy
    );
}

/// The 946 m/s gate.
///
/// The bound is physical, not a tuned constant: over `STEPS` ticks gravity
/// alone reaches 4.9 m/s, and the released load in this scene is bounded by the
/// elastic energy the wall's own weight could store. 60 m/s is far above
/// anything legitimate here and far below the runaway it exists to catch, and
/// the failure message reports the measurement so a real change is legible
/// rather than just red.
fn assert_no_runaway(engine: &str, o: &Outcome) {
    assert!(
        o.peak_speed < 60.0,
        "[{engine}] peak fragment speed {:.1} m/s -- a released-load impulse \
         producing this is a wrong magnitude or a wrong point of application, \
         not a reason to switch the mechanism off",
        o.peak_speed
    );
    assert!(
        o.lowest_y > -50.0,
        "[{engine}] an island reached y={:.1} m in {STEPS} ticks; free fall \
         alone would be about -1.2 m",
        o.lowest_y
    );
    assert!(o.peak_speed.is_finite(), "[{engine}] non-finite velocity");
}

#[test]
fn released_load_is_the_momentum_source_on_rapier() {
    let off = run(&mut RapierWorld::new(G), false);
    let on = run(&mut RapierWorld::new(G), true);
    assert_released_load_is_the_momentum_source("rapier", &off, &on);
    assert_no_runaway("rapier", &on);
    println!(
        "[rapier] energy proxy off={:.3} on={:.3}, peak {:.2} m/s, lowest y {:.2} m",
        off.kinetic_energy, on.kinetic_energy, on.peak_speed, on.lowest_y
    );
}

#[test]
fn released_load_is_the_momentum_source_on_physx() {
    let off = run(&mut PhysXWorld::new_cpu(G, 2).expect("physx"), false);
    let on = run(&mut PhysXWorld::new_cpu(G, 2).expect("physx"), true);
    assert_released_load_is_the_momentum_source("physx-cpu", &off, &on);
    assert_no_runaway("physx-cpu", &on);
    println!(
        "[physx-cpu] energy proxy off={:.3} on={:.3}, peak {:.2} m/s, lowest y {:.2} m",
        off.kinetic_energy, on.kinetic_energy, on.peak_speed, on.lowest_y
    );
}

/// The released load is a one-shot impulse, not a standing force.
///
/// A persistent force keeps re-accelerating a fragment every step after the
/// break, so a bond that broke once feeds it energy forever. Running longer
/// must not therefore produce an ever-growing speed: with a one-shot impulse
/// the fragment's speed after settling is governed by gravity and contact, so
/// a long run stays in the same regime as a short one rather than diverging.
fn assert_no_runaway_over_a_long_run<B: PhysicsBackend>(engine: &str, brief: &Outcome, long: &mut B) {
    let mut d = Destructible::attach(long, &scenario(), config(true)).expect("attach");
    for _ in 0..(STEPS * 8) {
        long.step(DT);
        d.step(long, DT);
    }
    let peak = d
        .island_poses(long)
        .into_iter()
        .map(|m| m.linvel.magnitude())
        .fold(0.0f32, f32::max);
    // 8x the ticks is 8x the free-fall speed, so the bound scales with gravity
    // rather than being a second magic number.
    let gravity_bound = 9.81 * DT * (STEPS * 8) as f32 * 1.5;
    let bound = gravity_bound.max(brief.peak_speed * 3.0);
    assert!(
        peak < bound,
        "[{engine}] peak speed grew to {peak:.1} m/s over a longer run (bound \
         {bound:.1}); a released load applied as a standing force rather than a \
         one-shot impulse looks exactly like this"
    );
    println!("[{engine}] long-run peak {peak:.2} m/s (bound {bound:.1})");
}

#[test]
fn the_impulse_does_not_keep_accelerating_fragments_on_rapier() {
    let brief = run(&mut RapierWorld::new(G), true);
    assert_no_runaway_over_a_long_run("rapier", &brief, &mut RapierWorld::new(G));
}

#[test]
fn the_impulse_does_not_keep_accelerating_fragments_on_physx() {
    let brief = run(&mut PhysXWorld::new_cpu(G, 2).expect("physx"), true);
    assert_no_runaway_over_a_long_run(
        "physx-cpu",
        &brief,
        &mut PhysXWorld::new_cpu(G, 2).expect("physx"),
    );
}
