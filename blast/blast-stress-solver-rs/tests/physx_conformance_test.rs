//! The PhysX backend against the same contract suite Rapier passes.
//!
//! This is the payoff of the adapter pattern: not one line of the suite is
//! PhysX-specific. If these pass, the core's pipeline can drive PhysX exactly
//! as it drives Rapier.

#![cfg(feature = "physx")]

use blast_stress_solver::backend::conformance;
use blast_stress_solver::backends::PhysXWorld;
use blast_stress_solver::types::Vec3;

fn report(name: &str, r: &conformance::Report) {
    eprintln!("[{name}] {} passed, {} skipped, {} failed", r.passed(), r.skipped(), r.failures().len());
    for c in &r.results {
        let tag = if c.skipped { "skip" } else if c.passed { "ok  " } else { "FAIL" };
        eprintln!("  {tag} {:<46} {}", c.name, c.detail);
    }
}

fn assert_clean(name: &str, r: &conformance::Report) {
    let failures: Vec<String> =
        r.failures().iter().map(|f| format!("{}: {}", f.name, f.detail)).collect();
    assert!(failures.is_empty(), "[{name}] contract violations:\n  {}", failures.join("\n  "));
}

#[test]
fn physx_cpu_satisfies_the_contract() {
    let mut w = PhysXWorld::new_cpu(Vec3::new(0.0, -9.81, 0.0), 2)
        .expect("PhysX CPU scene must construct");
    w.check().expect("PhysX must meet the required capabilities");
    assert!(!w.gpu_active(), "asked for CPU, got GPU");

    let r = conformance::run(&mut w);
    report("physx-cpu", &r);
    assert_clean("physx-cpu", &r);
}

#[test]
fn physx_gpu_satisfies_the_contract() {
    // GPU construction returns None rather than silently falling back, so a
    // machine without a usable GPU skips instead of quietly testing the CPU
    // path twice and reporting it as GPU coverage.
    let Some(mut w) = PhysXWorld::new_gpu(Vec3::new(0.0, -9.81, 0.0), 2) else {
        eprintln!("[physx-gpu] no usable GPU scene; skipping");
        return;
    };
    assert!(w.gpu_active(), "gpu scene reported inactive");
    w.check().expect("PhysX GPU must meet the required capabilities");

    let r = conformance::run(&mut w);
    report("physx-gpu", &r);
    assert_clean("physx-gpu", &r);
}

#[test]
fn physx_ids_are_never_reused() {
    // PhysX mints its own monotone ids precisely so a stale reference is
    // detectable rather than aliasing a recycled actor -- the failure mode
    // generational arena handles are vulnerable to.
    use blast_stress_solver::backend::*;
    let mut w = PhysXWorld::new_cpu(Vec3::new(0.0, -9.81, 0.0), 2).expect("scene");
    let mut cmds: CommandBuffer<_, _> = CommandBuffer::new();
    let mut out: CommandResults<_, _> = CommandResults::default();

    cmds.create_bodies.push(CreateBody {
        pose: Pose::IDENTITY,
        kind: BodyKind::Dynamic,
        linvel: Vec3::ZERO,
        angvel: Vec3::ZERO,
        ccd: false,
        start_sleeping: false,
    });
    w.apply(Phase::Topology, &cmds, &mut out).unwrap();
    let first = out.created_bodies[0];

    let mut rm: CommandBuffer<_, _> = CommandBuffer::new();
    rm.remove_bodies.push(first);
    w.apply(Phase::Retire, &rm, &mut out).unwrap();

    w.apply(Phase::Topology, &cmds, &mut out).unwrap();
    let second = out.created_bodies[0];

    assert_ne!(
        first.sort_key(),
        second.sort_key(),
        "an id was reused after removal; a stale handle would now alias a live body"
    );
}
