//! Equivalence tests for the batched external-force injection entry point
//! (`ext_stress_solver_add_all_forces`). Submitting a set of forces in one
//! batched FFI crossing must leave the solver in exactly the same state as
//! submitting them one at a time via `add_force` — this guards the
//! contact-injection performance optimisation against any behavioural drift.

use blast_stress_solver::*;

/// Simple 3-node triangle: two support nodes (mass 0) and a top mass node,
/// connected by three bonds. Mirrors the topology used elsewhere in the suite.
fn triangle() -> (Vec<NodeDesc>, Vec<BondDesc>) {
    let nodes = vec![
        NodeDesc { centroid: Vec3::new(-1.0, 0.0, 0.0), mass: 0.0, volume: 1.0 },
        NodeDesc { centroid: Vec3::new(1.0, 0.0, 0.0), mass: 0.0, volume: 1.0 },
        NodeDesc { centroid: Vec3::new(0.0, 1.5, 0.0), mass: 15.0, volume: 1.0 },
    ];
    let bonds = vec![
        BondDesc { centroid: Vec3::new(-0.5, 0.75, 0.0), normal: Vec3::new(0.55, 0.83, 0.0), area: 0.6, node0: 0, node1: 2 },
        BondDesc { centroid: Vec3::new(0.5, 0.75, 0.0), normal: Vec3::new(-0.55, 0.83, 0.0), area: 0.6, node0: 1, node1: 2 },
        BondDesc { centroid: Vec3::new(0.0, 0.0, 0.0), normal: Vec3::new(1.0, 0.0, 0.0), area: 0.9, node0: 0, node1: 1 },
    ];
    (nodes, bonds)
}

fn low_limit_settings() -> SolverSettings {
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

/// A force set with a repeated node index (node 2 appears twice) to mimic the
/// contact-injection pattern where a hit node plus its splash neighbours — and
/// multiple contacts landing on the same node — accumulate within one frame.
/// Returns parallel (indices, positions xyz, forces xyz) buffers.
fn force_set() -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let entries: &[(u32, [f32; 3], [f32; 3])] = &[
        (2, [0.0, 1.5, 0.0], [1000.0, 0.0, 0.0]),
        (0, [-1.0, 0.0, 0.0], [0.0, -250.0, 30.0]),
        (1, [1.0, 0.0, 0.0], [-120.0, 80.0, -10.0]),
        (2, [0.0, 1.5, 0.0], [200.0, -400.0, 15.0]), // same node again: must accumulate
    ];
    let mut idx = Vec::new();
    let mut pos = Vec::new();
    let mut force = Vec::new();
    for (n, p, f) in entries {
        idx.push(*n);
        pos.extend_from_slice(p);
        force.extend_from_slice(f);
    }
    (idx, pos, force)
}

#[test]
fn batched_forces_match_per_call_injection() {
    let (nodes, bonds) = triangle();
    let cfg = low_limit_settings();
    let (idx, pos, force) = force_set();

    // Reference: apply each force with its own FFI crossing.
    let mut per_call = ExtStressSolver::new(&nodes, &bonds, &cfg).unwrap();
    for i in 0..idx.len() {
        per_call.add_force(
            idx[i],
            Vec3::new(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]),
            Vec3::new(force[i * 3], force[i * 3 + 1], force[i * 3 + 2]),
            ForceMode::Force,
        );
    }
    per_call.update();

    // Optimised: apply the identical set in a single batched crossing.
    let mut batched = ExtStressSolver::new(&nodes, &bonds, &cfg).unwrap();
    let applied = batched.add_all_forces(&idx, &pos, &force, ForceMode::Force);
    assert_eq!(applied, idx.len() as u32, "every entry should be processed");
    batched.update();

    assert_eq!(
        per_call.overstressed_bond_count(),
        batched.overstressed_bond_count(),
        "overstressed bond count must match"
    );
    assert_eq!(
        per_call.converged(),
        batched.converged(),
        "convergence must match"
    );
    assert!(
        (per_call.linear_error() - batched.linear_error()).abs() <= 1e-6,
        "linear error must match: per-call={} batched={}",
        per_call.linear_error(),
        batched.linear_error(),
    );
    assert!(
        (per_call.angular_error() - batched.angular_error()).abs() <= 1e-6,
        "angular error must match: per-call={} batched={}",
        per_call.angular_error(),
        batched.angular_error(),
    );

    // Per-actor excess forces drive fracture decisions, so they must agree too.
    let com = Vec3::new(0.0, 0.75, 0.0);
    match (
        per_call.get_excess_forces(0, com),
        batched.get_excess_forces(0, com),
    ) {
        (Some((al, aa)), Some((bl, ba))) => {
            let pairs = [
                (al.x, bl.x), (al.y, bl.y), (al.z, bl.z),
                (aa.x, ba.x), (aa.y, ba.y), (aa.z, ba.z),
            ];
            for (x, y) in pairs {
                assert!((x - y).abs() <= 1e-4, "excess-force component mismatch: {x} vs {y}");
            }
        }
        (None, None) => {}
        (a, b) => panic!(
            "excess-force presence mismatch: per-call={} batched={}",
            a.is_some(),
            b.is_some()
        ),
    }
}

#[test]
fn batched_forces_empty_is_noop() {
    let (nodes, bonds) = triangle();
    let cfg = low_limit_settings();

    let mut untouched = ExtStressSolver::new(&nodes, &bonds, &cfg).unwrap();
    untouched.update();

    let mut empty = ExtStressSolver::new(&nodes, &bonds, &cfg).unwrap();
    let applied = empty.add_all_forces(&[], &[], &[], ForceMode::Force);
    assert_eq!(applied, 0, "an empty batch applies nothing");
    empty.update();

    assert_eq!(
        untouched.overstressed_bond_count(),
        empty.overstressed_bond_count()
    );
    assert!((untouched.linear_error() - empty.linear_error()).abs() <= 1e-6);
}

/// Exercises the O(1) input-node -> actor lookup specifically across a topology
/// change: fracture the structure into multiple actors (which invalidates the
/// lookup index), then inject forces by input-node index. The index must rebuild
/// and map each node to its *new* owning actor — so batched injection must still
/// match per-call injection after the split.
#[test]
fn batched_forces_match_per_call_after_fracture() {
    let (nodes, bonds) = triangle();
    let cfg = SolverSettings {
        compression_elastic_limit: 0.001,
        compression_fatal_limit: 0.002,
        tension_elastic_limit: 0.001,
        tension_fatal_limit: 0.002,
        shear_elastic_limit: 0.001,
        shear_fatal_limit: 0.002,
        ..SolverSettings::default()
    };

    // Fracture into multiple actors, then clear accumulated state so the test
    // injection below is what drives stress. apply_fracture_commands + reset both
    // mark the actor index dirty, so the first force lookup rebuilds it.
    fn fracture(s: &mut ExtStressSolver) {
        s.add_gravity(Vec3::new(0.0, -500.0, 0.0));
        s.update();
        let cmds = s.generate_fracture_commands();
        if !cmds.is_empty() {
            s.apply_fracture_commands(&cmds);
        }
        s.reset();
    }

    let mut per_call = ExtStressSolver::new(&nodes, &bonds, &cfg).unwrap();
    fracture(&mut per_call);
    let mut batched = ExtStressSolver::new(&nodes, &bonds, &cfg).unwrap();
    fracture(&mut batched);
    assert_eq!(
        per_call.actor_count(),
        batched.actor_count(),
        "both solvers must reach the same post-fracture topology"
    );

    let (idx, pos, force) = force_set();
    for i in 0..idx.len() {
        per_call.add_force(
            idx[i],
            Vec3::new(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]),
            Vec3::new(force[i * 3], force[i * 3 + 1], force[i * 3 + 2]),
            ForceMode::Force,
        );
    }
    per_call.update();
    batched.add_all_forces(&idx, &pos, &force, ForceMode::Force);
    batched.update();

    assert_eq!(
        per_call.overstressed_bond_count(),
        batched.overstressed_bond_count(),
        "post-fracture overstressed bond count must match"
    );
    assert!(
        (per_call.linear_error() - batched.linear_error()).abs() <= 1e-6,
        "post-fracture linear error must match: per-call={} batched={}",
        per_call.linear_error(),
        batched.linear_error(),
    );
}
