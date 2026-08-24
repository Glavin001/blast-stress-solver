//! What a severed island's bonds actually carry.
//!
//! A building is destructible because its own weight already loads its bonds:
//! gravity accelerates every node, the mass-0 foundation nodes do not move, and
//! that difference is the load path. Cut the building free of its foundation and
//! the load path goes with it — gravity is now a uniform acceleration over every
//! node, which is a rigid translation, so no bond sees relative motion and the
//! stress field is exactly zero. The island becomes indestructible: each shot has
//! to fund the whole fracture itself, with nothing stored to cascade.
//!
//! These tests pin all three halves of that story, because the fix depends on the
//! middle one being true and the outer two being the way back:
//!   1. anchored + gravity              => loaded   (the baseline that works)
//!   2. free + gravity                  => zero     (the bug, and correct physics)
//!   3. free + contact reaction         => loaded   (resting on the ground)
//!   4. free + spin                     => loaded   (tumbling)
//!
//! (3) and (4) are the signals the PhysX adapter now feeds. Without them a landed
//! or tumbling island is stress-free no matter what happens to it.

use blast_stress_solver::*;

/// A vertical stack of `n` nodes 1 m apart along +Y, bonded in series.
///
/// With `anchored`, node 0 has mass 0 and is a support — the classic building.
/// Without it every node is dynamic: a severed island in free fall.
fn stack(n: u32, anchored: bool) -> (Vec<NodeDesc>, Vec<BondDesc>) {
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    for i in 0..n {
        nodes.push(NodeDesc {
            centroid: Vec3::new(0.0, i as f32, 0.0),
            mass: if anchored && i == 0 { 0.0 } else { 100.0 },
            volume: 1.0,
        });
    }
    for i in 0..n - 1 {
        bonds.push(BondDesc {
            centroid: Vec3::new(0.0, i as f32 + 0.5, 0.0),
            normal: Vec3::new(0.0, 1.0, 0.0),
            area: 1.0,
            node0: i,
            node1: i + 1, material: 0 });
    }
    (nodes, bonds)
}

/// Limits low enough that ordinary self-weight registers as overstress, so the
/// tests can assert on a count rather than on a tuned magnitude.
fn sensitive_settings() -> SolverSettings {
    SolverSettings {
        compression_elastic_limit: 0.001,
        compression_fatal_limit: 0.002,
        tension_elastic_limit: 0.001,
        tension_fatal_limit: 0.002,
        shear_elastic_limit: 0.001,
        shear_fatal_limit: 0.002,
        ..SolverSettings::default()
    }
}

const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

#[test]
fn anchored_stack_carries_its_own_weight() {
    let (nodes, bonds) = stack(6, true);
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &sensitive_settings()).unwrap();

    solver.add_gravity(GRAVITY);
    solver.update();

    assert!(
        solver.overstressed_bond_count() > 0,
        "an anchored stack must load its bonds under its own weight; \
         this is the load path every fracture cascade depends on"
    );
}

#[test]
fn free_island_under_gravity_alone_carries_nothing() {
    let (nodes, bonds) = stack(6, false);
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &sensitive_settings()).unwrap();

    solver.add_gravity(GRAVITY);
    solver.update();

    // Correct physics — free fall has no internal stress — and simultaneously
    // the reason a severed building half stops being destructible. Anything
    // that loads such an island has to come from contact or from spin.
    assert_eq!(
        solver.overstressed_bond_count(),
        0,
        "gravity alone is a uniform acceleration on an unanchored island, so it \
         must produce no bond stress at all"
    );
}

#[test]
fn free_island_resting_on_the_ground_carries_its_weight_again() {
    let (nodes, bonds) = stack(6, false);
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &sensitive_settings()).unwrap();

    // Weight pulls every node down...
    solver.add_gravity(GRAVITY);
    // ...and the ground pushes back on the one node touching it. That pair is
    // what the mass-0 foundation used to provide for free, and it is what the
    // adapter now reproduces by pairing persistent contact reports with gravity
    // for resting bodies.
    let total_mass = 100.0 * 6.0;
    let reaction = Vec3::new(0.0, 9.81 * total_mass, 0.0);
    let contact_point = Vec3::new(0.0, 0.0, 0.0);
    solver.add_force(0, contact_point, reaction, ForceMode::Force);

    solver.update();

    assert!(
        solver.overstressed_bond_count() > 0,
        "an island resting on the ground must carry load through its contact, \
         the same way it did through its foundation"
    );
}

#[test]
fn free_island_that_is_spinning_carries_load() {
    let (nodes, bonds) = stack(6, false);
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &sensitive_settings()).unwrap();

    // Tumbling debris is genuinely under omega-squared-r tension; the centre is
    // the island's own centre of mass, in the frame the solver states node
    // positions in.
    let centre = Vec3::new(0.0, 2.5, 0.0);
    let applied = solver.add_centrifugal_acceleration(0, centre, Vec3::new(0.0, 0.0, 12.0));
    assert!(applied, "centrifugal load must reach the live actor");

    solver.update();

    assert!(
        solver.overstressed_bond_count() > 0,
        "a fast-spinning island must load its bonds even with no contact and no anchor"
    );
}
