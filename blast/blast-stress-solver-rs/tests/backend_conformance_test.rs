//! Runs the reusable backend conformance suite against every shipped adapter.
//!
//! Adding an engine means adding one `run_suite` call here. If it passes, the
//! core's features are known to be driveable on that engine; if it fails, the
//! report names exactly which contract clause was broken.

#![cfg(feature = "rapier")]

use blast_stress_solver::backend::conformance;
use blast_stress_solver::backends::RapierWorld;
use blast_stress_solver::types::Vec3;

fn report(name: &str, r: &conformance::Report) {
    eprintln!(
        "[{name}] {} passed, {} skipped, {} failed",
        r.passed(),
        r.skipped(),
        r.failures().len()
    );
    for c in &r.results {
        let tag = if c.skipped {
            "skip"
        } else if c.passed {
            "ok  "
        } else {
            "FAIL"
        };
        eprintln!("  {tag} {:<46} {}", c.name, c.detail);
    }
}

#[test]
fn rapier_backend_satisfies_the_contract() {
    let mut world = RapierWorld::new(Vec3::new(0.0, -9.81, 0.0));
    world.check().expect("Rapier must meet the required capabilities");

    let r = conformance::run(&mut world);
    report("rapier", &r);

    let failures: Vec<String> = r
        .failures()
        .iter()
        .map(|f| format!("{}: {}", f.name, f.detail))
        .collect();
    assert!(failures.is_empty(), "contract violations:\n  {}", failures.join("\n  "));
}

#[test]
fn the_suite_actually_exercises_the_optional_paths() {
    // A conformance suite that skipped everything would pass vacuously. Rapier
    // advertises most optional capabilities, so most checks must really run.
    let mut world = RapierWorld::new(Vec3::new(0.0, -9.81, 0.0));
    let r = conformance::run(&mut world);
    assert!(
        r.skipped() <= 1,
        "expected Rapier to exercise nearly every check, but {} were skipped",
        r.skipped()
    );
    assert!(r.passed() >= 14, "only {} checks ran", r.passed());
}
