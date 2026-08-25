//! One scene, three backends: Rapier, PhysX CPU, PhysX GPU.
//!
//! The driver in `backend::scene` is generic over `PhysicsBackend` and has no
//! engine-specific code, so this file only decides *which* backends to build.
//! That is the whole claim of the adapter pattern, stated as a test.

#![cfg(all(feature = "rapier", feature = "physx"))]

use blast_stress_solver::backend::scene::{
    check_invariants, cross_engine_within_band, run_scene, SceneOutcome, SceneSpec,
};
use blast_stress_solver::backend::PhysicsBackend;
use blast_stress_solver::backends::{PhysXWorld, RapierWorld};
use blast_stress_solver::types::Vec3;

const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn assert_engine_ok(name: &str, spec: &SceneSpec, o: &SceneOutcome) {
    eprintln!(
        "[{name}] bodies={} mass={:.2} centroid.y={:.4} sleeping={:.0}% peak_speed={:.2} min_y={:.3}",
        o.bodies, o.total_mass, o.centroid.y, o.sleeping_fraction * 100.0, o.peak_speed, o.min_y
    );
    let v = check_invariants(spec, o);
    assert!(v.is_empty(), "[{name}] invariant violations:\n  {}", v.join("\n  "));
}

/// Run the scene twice on one backend and report the centroid spread. This is
/// the engine's own noise floor, and it is what the cross-engine band is
/// calibrated against.
fn spread<B: PhysicsBackend>(mut make: impl FnMut() -> B, spec: &SceneSpec) -> (SceneOutcome, f32) {
    let mut a = make();
    let first = run_scene(&mut a, spec);
    let mut b = make();
    let second = run_scene(&mut b, spec);
    let s = (first.centroid.y - second.centroid.y).abs();
    (first, s)
}

#[test]
fn the_same_scene_runs_on_rapier_and_physx() {
    let spec = SceneSpec::default();

    let (rapier, rapier_spread) = spread(|| RapierWorld::new(G), &spec);
    assert_engine_ok("rapier", &spec, &rapier);

    let (physx, physx_spread) = spread(
        || PhysXWorld::new_cpu(G, 2).expect("PhysX CPU scene"),
        &spec,
    );
    assert_engine_ok("physx-cpu", &spec, &physx);

    eprintln!(
        "[variance] rapier run-to-run {:.5} m, physx run-to-run {:.5} m",
        rapier_spread, physx_spread
    );

    // Not an equality check: a settled box stack should land in the same
    // regime on both engines, and "same regime" is defined by their own
    // reproducibility plus a physical slack of one box half-extent.
    cross_engine_within_band(&rapier, rapier_spread, &physx, physx_spread, spec.box_half)
        .expect("engines disagreed beyond their own variance");
}

#[test]
fn physx_gpu_runs_the_same_scene() {
    let spec = SceneSpec::default();
    let Some(mut gpu) = PhysXWorld::new_gpu(G, 2) else {
        eprintln!("[physx-gpu] no usable GPU scene; skipping");
        return;
    };
    assert!(gpu.gpu_active());
    let o = run_scene(&mut gpu, &spec);
    assert_engine_ok("physx-gpu", &spec, &o);
}

#[test]
fn each_engine_is_reproducible_run_to_run() {
    // The band in the cross-engine test is only meaningful if each engine's own
    // spread is small. If an engine were wildly non-reproducible the band would
    // widen until it asserted nothing — so pin the noise floor directly.
    let spec = SceneSpec { frames: 120, ..SceneSpec::default() };

    let (_, r) = spread(|| RapierWorld::new(G), &spec);
    assert!(r < 0.05, "Rapier run-to-run spread {r} is too large to calibrate against");

    let (_, p) = spread(|| PhysXWorld::new_cpu(G, 2).expect("scene"), &spec);
    assert!(p < 0.05, "PhysX run-to-run spread {p} is too large to calibrate against");
    eprintln!("[noise floor] rapier {r:.6} m, physx {p:.6} m");
}
