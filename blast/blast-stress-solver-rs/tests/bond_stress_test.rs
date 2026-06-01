//! `compute_bond_stress` coverage: hand-computed known values (which also LOCK JS↔Rust
//! parity — the JS `computeBondStress` is byte-identical, and `bondStress.parity.test.ts`
//! asserts the same numbers), the angular twist/bend paths (the existing tests only used
//! `ang = 0`), and structural properties.

use blast_stress_solver::stress_processor::{StressBondDesc, StressImpulse, StressNodeDesc};
use blast_stress_solver::{compute_bond_stress, Vec3};

/// Two nodes 2 m apart along +X => bond normal = +X, distance = 2.
fn axis_nodes() -> Vec<StressNodeDesc> {
    vec![
        StressNodeDesc { com: Vec3::new(0.0, 0.0, 0.0), mass: 1.0, inertia: 1.0 },
        StressNodeDesc { com: Vec3::new(2.0, 0.0, 0.0), mass: 1.0, inertia: 1.0 },
    ]
}

fn bond() -> StressBondDesc {
    StressBondDesc { centroid: Vec3::new(1.0, 0.0, 0.0), node0: 0, node1: 1 }
}

fn approx(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-4
}

// ---- Known values (shared with the JS parity test) ----

#[test]
fn pure_tension_along_normal() {
    // lin along +normal => tension = lin·n / area.
    let s = compute_bond_stress(&bond(), &StressImpulse { lin: Vec3::new(6.0, 0.0, 0.0), ang: Vec3::ZERO }, &axis_nodes(), 2.0);
    assert!(approx(s.tension, 3.0) && approx(s.compression, 0.0) && approx(s.shear, 0.0), "{s:?}");
}

#[test]
fn pure_compression_against_normal() {
    let s = compute_bond_stress(&bond(), &StressImpulse { lin: Vec3::new(-6.0, 0.0, 0.0), ang: Vec3::ZERO }, &axis_nodes(), 2.0);
    assert!(approx(s.compression, 3.0) && approx(s.tension, 0.0) && approx(s.shear, 0.0), "{s:?}");
}

#[test]
fn pure_linear_shear() {
    // lin perpendicular to normal => shear = |lin| / area.
    let s = compute_bond_stress(&bond(), &StressImpulse { lin: Vec3::new(0.0, 5.0, 0.0), ang: Vec3::ZERO }, &axis_nodes(), 1.0);
    assert!(approx(s.shear, 5.0) && approx(s.tension, 0.0) && approx(s.compression, 0.0), "{s:?}");
}

#[test]
fn pure_twist_adds_to_shear() {
    // ang along normal (twist): shear += |ang·n|/area * 2/distance = 3/1 * 2/2 = 3.
    let s = compute_bond_stress(&bond(), &StressImpulse { lin: Vec3::ZERO, ang: Vec3::new(3.0, 0.0, 0.0) }, &axis_nodes(), 1.0);
    assert!(approx(s.shear, 3.0) && approx(s.tension, 0.0) && approx(s.compression, 0.0), "{s:?}");
}

#[test]
fn pure_bend_adds_to_normal() {
    // ang perpendicular to normal (bend): normal += |ang_perp|/area * 2/distance = 4/1 * 2/2 = 4,
    // signed toward tension when normal is non-negative.
    let s = compute_bond_stress(&bond(), &StressImpulse { lin: Vec3::ZERO, ang: Vec3::new(0.0, 4.0, 0.0) }, &axis_nodes(), 1.0);
    assert!(approx(s.tension, 4.0) && approx(s.compression, 0.0) && approx(s.shear, 0.0), "{s:?}");
}

#[test]
fn mixed_linear_shear_plus_twist() {
    // lin shear 5 + twist 2*2/2=2 => shear 7.
    let s = compute_bond_stress(&bond(), &StressImpulse { lin: Vec3::new(0.0, 5.0, 0.0), ang: Vec3::new(2.0, 0.0, 0.0) }, &axis_nodes(), 1.0);
    assert!(approx(s.shear, 7.0) && approx(s.tension, 0.0) && approx(s.compression, 0.0), "{s:?}");
}

// ---- Degenerate guards ----

#[test]
fn non_positive_area_is_zero() {
    let s = compute_bond_stress(&bond(), &StressImpulse { lin: Vec3::new(6.0, 0.0, 0.0), ang: Vec3::ZERO }, &axis_nodes(), 0.0);
    assert!(approx(s.compression, 0.0) && approx(s.tension, 0.0) && approx(s.shear, 0.0));
}

#[test]
fn out_of_range_nodes_are_zero() {
    let bad = StressBondDesc { centroid: Vec3::ZERO, node0: 0, node1: 9 };
    let s = compute_bond_stress(&bad, &StressImpulse { lin: Vec3::new(6.0, 0.0, 0.0), ang: Vec3::ZERO }, &axis_nodes(), 1.0);
    assert!(approx(s.compression, 0.0) && approx(s.tension, 0.0) && approx(s.shear, 0.0));
}

// ---- Property: stress components are never negative ----

#[test]
fn stress_components_are_non_negative() {
    use proptest::prelude::*;
    proptest!(|(l in prop::array::uniform3(-50.0f32..50.0), a in prop::array::uniform3(-50.0f32..50.0), area in 0.1f32..10.0)| {
        let s = compute_bond_stress(
            &bond(),
            &StressImpulse { lin: Vec3::new(l[0], l[1], l[2]), ang: Vec3::new(a[0], a[1], a[2]) },
            &axis_nodes(),
            area,
        );
        prop_assert!(s.compression >= 0.0 && s.tension >= 0.0 && s.shear >= 0.0, "{s:?}");
        // compression and tension are mutually exclusive.
        prop_assert!(s.compression == 0.0 || s.tension == 0.0, "{s:?}");
    });
}
