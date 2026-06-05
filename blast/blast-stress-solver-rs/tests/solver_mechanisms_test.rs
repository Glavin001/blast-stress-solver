//! Empirical tests of two physics mechanisms at the SOLVER level (no Rapier, no trust in
//! code inspection): orientation-dependent gravity and excess-force-on-fracture. The JS
//! suite has a mirror of these (`src/tests/solver-mechanisms.test.ts`) so we can be
//! confident BOTH libraries' solvers actually implement them.
//!
//! These verify the *mechanisms exist and respond correctly*. Whether each library's
//! high-level destruction pipeline actually FEEDS them the right inputs is a separate,
//! integration-level question (see gap #7 / gap #8 in blast/TESTING.md and
//! `gravity_orientation_test.rs`).

use blast_stress_solver::*;

/// 6-node horizontal cantilever: node 0 fixed (mass 0), nodes 1..5 along +X.
fn cantilever() -> (Vec<NodeDesc>, Vec<BondDesc>) {
    let n = 6u32;
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    for i in 0..n {
        let mass = if i == 0 { 0.0 } else { 1.0 };
        nodes.push(NodeDesc { centroid: Vec3::new(i as f32, 0.0, 0.0), mass, volume: 1.0 });
    }
    for i in 0..n - 1 {
        let (a, b) = (nodes[i as usize].centroid, nodes[(i + 1) as usize].centroid);
        bonds.push(BondDesc { node0: i, node1: i + 1, centroid: (a + b) * 0.5, normal: Vec3::new(1.0, 0.0, 0.0), area: 0.5 });
    }
    (nodes, bonds)
}

/// Weak tension/shear, very strong compression: a bending (perpendicular) load snaps the
/// beam; an axial load only compresses it.
fn bend_sensitive_settings() -> SolverSettings {
    SolverSettings {
        compression_elastic_limit: 1.0e5, compression_fatal_limit: 1.0e6,
        tension_elastic_limit: 0.05, tension_fatal_limit: 0.1,
        shear_elastic_limit: 0.05, shear_fatal_limit: 0.1,
        ..SolverSettings::default()
    }
}

fn overstress_for_gravity(gravity: Vec3, per_actor: bool) -> u32 {
    let (nodes, bonds) = cantilever();
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &bend_sensitive_settings()).unwrap();
    if per_actor {
        let actor = solver.actors()[0].actor_index;
        assert!(solver.add_actor_gravity(actor, gravity), "add_actor_gravity should succeed");
    } else {
        solver.add_gravity(gravity);
    }
    solver.update();
    solver.overstressed_bond_count()
}

/// Mechanism 1a: the GLOBAL gravity vector's direction changes the stress. A perpendicular
/// (bending) load overstresses the beam; an axial load (along the beam, into the anchor)
/// does not.
#[test]
fn add_gravity_is_direction_sensitive() {
    let bending = overstress_for_gravity(Vec3::new(0.0, -50.0, 0.0), false);
    let axial = overstress_for_gravity(Vec3::new(-50.0, 0.0, 0.0), false);
    assert!(
        bending > axial,
        "gravity direction must affect stress: perpendicular bending overstressed {bending} bonds, \
         axial overstressed {axial} (expected bending > axial)"
    );
}

/// Mechanism 1b: the PER-ACTOR gravity API (`add_actor_gravity`) — the exact call the JS
/// pipeline uses with gravity rotated into each actor's local frame — is direction
/// sensitive. This is what makes orientation-correct gravity possible.
#[test]
fn add_actor_gravity_is_direction_sensitive() {
    let bending = overstress_for_gravity(Vec3::new(0.0, -50.0, 0.0), true);
    let axial = overstress_for_gravity(Vec3::new(-50.0, 0.0, 0.0), true);
    assert!(
        bending > axial,
        "add_actor_gravity must be direction sensitive: bending overstressed {bending} bonds, \
         axial overstressed {axial} (expected bending > axial)"
    );
}

/// Mechanism 2: when a loaded bond breaks, `getExcessForces` reports the released load (so a
/// pipeline can throw the freed fragment). A 10 kg mass hung from a fixed support by one
/// bond under g = 100 carries ~1000 N; after the bond fractures the freed mass's actor must
/// report a non-trivial excess force. (The pre-existing solver test only checked finiteness,
/// so this magnitude was never actually asserted.)
#[test]
fn excess_force_reports_released_load() {
    let nodes = vec![
        NodeDesc { centroid: Vec3::new(0.0, 1.0, 0.0), mass: 0.0, volume: 1.0 },
        NodeDesc { centroid: Vec3::new(0.0, 0.0, 0.0), mass: 10.0, volume: 1.0 },
    ];
    let bonds = vec![BondDesc {
        centroid: Vec3::new(0.0, 0.5, 0.0), normal: Vec3::new(0.0, 1.0, 0.0), area: 1.0, node0: 0, node1: 1,
    }];
    let settings = SolverSettings {
        compression_elastic_limit: 1.0, compression_fatal_limit: 2.0,
        tension_elastic_limit: 1.0, tension_fatal_limit: 2.0,
        shear_elastic_limit: 1.0, shear_fatal_limit: 2.0,
        ..SolverSettings::default()
    };
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &settings).unwrap();
    solver.add_gravity(Vec3::new(0.0, -100.0, 0.0));
    solver.update();
    assert_eq!(solver.overstressed_bond_count(), 1, "the loaded bond should overstress");

    let cmds = solver.generate_fracture_commands();
    let events = solver.apply_fracture_commands(&cmds);
    assert!(!events.is_empty(), "the bond should fracture and split the actor");

    // The freed mass's actor must report a meaningful released load (~weight = 1000 N).
    let freed = solver
        .actors()
        .into_iter()
        .find(|a| a.nodes.contains(&1))
        .expect("freed-mass actor exists");
    let (force, _torque) = solver
        .get_excess_forces(freed.actor_index, Vec3::new(0.0, 0.0, 0.0))
        .expect("excess force available right after fracture");
    let mag = force.magnitude();
    assert!(
        mag > 100.0,
        "excess force should report the released load (~1000 N), got |f| = {mag:.1} N \
         ({force:?}) — near-zero would mean the momentum-transfer mechanism is dormant"
    );
}
