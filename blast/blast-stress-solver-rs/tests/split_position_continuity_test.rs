//! Position-continuity invariant for the split path (companion to the velocity-continuity
//! checks in `kinematic_invariants_test.rs`).
//!
//! A faithful rigid fracture must not *teleport* a chunk: each migrated/retained node's world
//! position is the same an instant after the split as an instant before. The split path
//! re-centers each child body's origin on its fragment's centre of mass and re-derives every
//! node's collider offset; if those two edits disagree, the chunk jumps — the "half the bridge
//! yanked itself inwards" symptom. `point_velocity_error` cannot see a pure position jump, so
//! this asserts the dedicated `world_position_error` metric.
//!
//! Unlike the JS pipeline (everything starts on a fixed root), a support-free Rust scenario is
//! a single dynamic body from `initialize`, so we can spin it and fracture it directly.

#![cfg(feature = "rapier")]

use rapier3d::prelude::*;

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

const POINT_VELOCITY_CONTINUITY_TOL: f32 = 1.0e-3;
const WORLD_POSITION_CONTINUITY_TOL: f32 = 1.0e-3;

fn rapier_world() -> (
    RigidBodySet,
    ColliderSet,
    IslandManager,
    ImpulseJointSet,
    MultibodyJointSet,
) {
    (
        RigidBodySet::new(),
        ColliderSet::new(),
        IslandManager::new(),
        ImpulseJointSet::new(),
        MultibodyJointSet::new(),
    )
}

/// A horizontal beam of `n` unit-cube dynamic nodes along x at height 5 (no supports → one
/// dynamic body). Every bond is strong except the one at `weak_bond`, which is ~5000× weaker,
/// so a stretching load fractures the beam there into two multi-node halves (a large recentre).
fn beam_scenario(n: usize, weak_bond: usize) -> ScenarioDesc {
    let size = Vec3::new(0.5, 0.5, 0.5);
    let nodes = (0..n)
        .map(|i| ScenarioNode {
            centroid: Vec3::new(i as f32, 5.0, 0.0),
            mass: 1.0,
            volume: 0.125,
        })
        .collect();
    let bonds = (0..n - 1)
        .map(|i| ScenarioBond {
            node0: i as u32,
            node1: (i + 1) as u32,
            centroid: Vec3::new(i as f32 + 0.5, 5.0, 0.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            area: if i == weak_bond { 0.01 } else { 50.0 }, material: 0, })
        .collect();
    ScenarioDesc {
        nodes,
        bonds,
        node_sizes: vec![size; n],
        collider_shapes: vec![None; n],
    }
}

/// Spin the beam at a fixed angular velocity (zero gravity, re-imposed each step) and stretch it
/// with a solver-only force until the weak bond fractures. Returns the per-node continuity
/// records captured at the split.
fn drive_rotating_beam_split(n: usize, weak_bond: usize) -> Vec<SplitContinuityRecord> {
    let scenario = beam_scenario(n, weak_bond);
    let policy = FracturePolicy {
        idle_skip: false,
        ..FracturePolicy::default()
    };
    let mut set =
        DestructibleSet::from_scenario(&scenario, SolverSettings::default(), Vec3::ZERO, policy)
            .unwrap();
    let (mut bodies, mut colliders, mut islands, mut ij, mut mj) = rapier_world();
    set.initialize(&mut bodies, &mut colliders);
    set.set_record_split_continuity(true);

    let parent = set.node_body(0).expect("beam starts on one dynamic body");
    let omega = vector![0.0, 0.0, 2.0];
    let mut records = Vec::new();
    for _ in 0..120 {
        if let Some(body) = bodies.get_mut(parent) {
            body.set_linvel(vector![0.0, 0.0, 0.0], true);
            body.set_angvel(omega, true);
        }
        // Stretch the beam: pull the two ends apart (solver-only forces — they create bond
        // tension without perturbing the Rapier body, so the spin we set is the true motion).
        set.add_force(0, scenario.nodes[0].centroid, Vec3::new(-4000.0, 0.0, 0.0));
        set.add_force(
            (n - 1) as u32,
            scenario.nodes[n - 1].centroid,
            Vec3::new(4000.0, 0.0, 0.0),
        );
        set.clear_split_continuity();
        let result = set.step(&mut bodies, &mut colliders, &mut islands, &mut ij, &mut mj);
        if result.split_events > 0 {
            records = set.split_continuity().to_vec();
            break;
        }
    }
    assert!(
        !records.is_empty(),
        "the weak bond should fracture the beam"
    );
    records
}

fn worst_position(records: &[SplitContinuityRecord]) -> &SplitContinuityRecord {
    records
        .iter()
        .max_by(|a, b| {
            a.world_position_error
                .partial_cmp(&b.world_position_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("a split must produce continuity records")
}

/// REGRESSION GUARD. A spinning beam that fractures into two multi-node halves must keep every
/// chunk's world position continuous across the split — the reused half (which keeps the parent
/// handle and is re-centered onto its own COM) must not teleport. Also re-checks velocity so the
/// position guard can't pass by trading one discontinuity for another.
#[test]
fn split_preserves_chunk_world_position_for_recentered_halves() {
    let records = drive_rotating_beam_split(6, 2); // halves {0,1,2} and {3,4,5}
    let worst_pos = worst_position(&records);
    assert!(
        worst_pos.world_position_error < WORLD_POSITION_CONTINUITY_TOL,
        "split teleported node {} on body {:?} by {:.5} m (recentering vs collider offset disagree)",
        worst_pos.node_index,
        worst_pos.target_body,
        worst_pos.world_position_error,
    );
    let worst_vel = records
        .iter()
        .map(|r| r.point_velocity_error)
        .fold(0.0_f32, f32::max);
    assert!(
        worst_vel < POINT_VELOCITY_CONTINUITY_TOL,
        "split injected spurious velocity: worst point-velocity drift {:.5} m/s",
        worst_vel,
    );
}

/// Smaller matched case (single-node children) — also must not teleport.
#[test]
fn split_preserves_chunk_world_position_for_small_beam() {
    let records = drive_rotating_beam_split(3, 1);
    assert!(
        worst_position(&records).world_position_error < WORLD_POSITION_CONTINUITY_TOL,
        "small-beam split teleported a chunk by {:.5} m",
        worst_position(&records).world_position_error,
    );
}
