//! Tier-1 small-scenario kinematic invariants for the fracture/split path.
//!
//! These are the "workhorse" physics invariants from `blast/TESTING.md`. They run
//! on the smallest scenarios that exercise a split, so a failure localizes the
//! offending body/node with an exact magnitude (root-cause pointer) rather than a
//! vague "the demo looks wrong".
//!
//! Shared invariant spec (mirrored in the JS suite — see
//! `blast/blast-stress-solver/src/tests/invariants.shared.ts`):
//!   * point-velocity continuity across a split: `< 1e-3` m/s
//!   * no NaN/Inf in any post-split body state
//!   * dynamic mass conserved across a split: rel `< 1e-4`
//!
//! The split path is *correct* for fragments whose Rapier centre of mass coincides
//! with the node-centroid model (e.g. axis-aligned cuboids). It diverges when they
//! differ (e.g. convex-hull fragments) and the parent is rotating — that is the
//! "sudden movement/rotation after destruction" bug. The two continuity tests below
//! form a matched pair that isolates exactly that variable.

#![cfg(feature = "rapier")]

use rapier3d::prelude::*;

use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

// Shared invariant spec (keep in sync with invariants.shared.ts).
const POINT_VELOCITY_CONTINUITY_TOL: f32 = 1.0e-3;
const MASS_REL_TOL: f32 = 1.0e-4;

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

fn weak_settings() -> SolverSettings {
    SolverSettings {
        compression_elastic_limit: 0.01,
        compression_fatal_limit: 0.05,
        tension_elastic_limit: 0.01,
        tension_fatal_limit: 0.05,
        shear_elastic_limit: 0.01,
        shear_fatal_limit: 0.05,
        ..SolverSettings::default()
    }
}

/// Two dynamic nodes joined by one bond. `node1_collider` selects node 1's collider
/// shape so we can compare an origin-aligned COM (cuboid) against an offset COM
/// (a convex hull whose volume centroid is shifted from the node centroid).
fn pair_scenario(node1_collider: Option<ScenarioCollider>) -> ScenarioDesc {
    let size = Vec3::new(0.5, 0.5, 0.5);
    ScenarioDesc {
        nodes: vec![
            ScenarioNode {
                centroid: Vec3::new(-0.25, 0.5, 0.0),
                mass: 1.0,
                volume: 0.125,
            },
            ScenarioNode {
                centroid: Vec3::new(0.25, 0.5, 0.0),
                mass: 1.0,
                volume: 0.125,
            },
        ],
        bonds: vec![ScenarioBond {
            node0: 0,
            node1: 1,
            centroid: Vec3::new(0.0, 0.5, 0.0),
            normal: Vec3::new(1.0, 0.0, 0.0),
            area: 0.25, material: 0, }],
        node_sizes: vec![size, size],
        collider_shapes: vec![None, node1_collider],
    }
}

/// A convex hull occupying the unit cube `[0,1]^3`: its volume centroid is at
/// `(0.5, 0.5, 0.5)` in collider-local space, so Rapier's body centre of mass ends
/// up offset from the node centroid by that amount — the COM divergence the split
/// path ignores.
fn offset_hull() -> ScenarioCollider {
    ScenarioCollider::ConvexHull {
        points: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
        ],
    }
}

#[derive(Clone, Copy, Debug)]
struct BodySummary {
    mass: f32,
    dynamic: bool,
    finite: bool,
}

/// Drive the pair to fracture while holding the parent body in a known rigid motion
/// (zero gravity + a fixed angular velocity re-imposed each step). Returns the
/// per-node continuity records captured at the split, plus a summary of every body
/// afterwards. The angular velocity is what turns a COM offset into a velocity
/// discontinuity, so every continuity test uses the same spin.
fn drive_rotating_pair_split(
    scenario: &ScenarioDesc,
) -> (Vec<SplitContinuityRecord>, Vec<BodySummary>) {
    let policy = FracturePolicy {
        idle_skip: false,
        ..FracturePolicy::default()
    };
    let mut set =
        DestructibleSet::from_scenario(scenario, weak_settings(), Vec3::ZERO, policy).unwrap();
    let (mut bodies, mut colliders, mut islands, mut ij, mut mj) = rapier_world();
    set.initialize(&mut bodies, &mut colliders);
    set.set_record_split_continuity(true);

    let parent = set.node_body(0).expect("node 0 has a body");
    assert_eq!(
        set.node_body(1),
        Some(parent),
        "both nodes start bonded on a single parent body"
    );

    let omega = vector![0.0, 0.0, 3.0];
    let mut records = Vec::new();
    let mut split = false;
    for _ in 0..60 {
        if let Some(body) = bodies.get_mut(parent) {
            // Pure rotation about the world origin's z axis; zero gravity keeps it
            // torque-free so this is the body's actual motion at the split instant.
            body.set_linvel(vector![0.0, 0.0, 0.0], true);
            body.set_angvel(omega, true);
        }
        // Overstress the bond. `add_force` feeds the stress solver only; it does not
        // perturb the Rapier body, so the parent's rigid motion stays the one we set.
        set.add_force(1, scenario.nodes[1].centroid, Vec3::new(2000.0, 0.0, 0.0));
        set.clear_split_continuity();
        let result = set.step(&mut bodies, &mut colliders, &mut islands, &mut ij, &mut mj);
        if result.split_events > 0 {
            records = set.split_continuity().to_vec();
            split = true;
            break;
        }
    }
    assert!(split, "the single bond should fracture under the applied force");

    let summaries = bodies
        .iter()
        .map(|(_, body)| {
            let t = body.translation();
            let lv = body.linvel();
            let av = body.angvel();
            let finite = [t.x, t.y, t.z, lv.x, lv.y, lv.z, av.x, av.y, av.z]
                .iter()
                .all(|v| v.is_finite());
            BodySummary {
                mass: body.mass(),
                dynamic: body.is_dynamic(),
                finite,
            }
        })
        .collect();

    (records, summaries)
}

fn worst_continuity(records: &[SplitContinuityRecord]) -> &SplitContinuityRecord {
    assert!(!records.is_empty(), "a split must produce continuity records");
    records
        .iter()
        .max_by(|a, b| {
            a.point_velocity_error
                .partial_cmp(&b.point_velocity_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}

// ---------------------------------------------------------------------------
// Point-velocity continuity — the matched pair that isolates the COM bug.
// ---------------------------------------------------------------------------

/// Positive control: a fragment whose Rapier COM coincides with the node centroid
/// (axis-aligned cuboid) must preserve point velocity across the split even while
/// the parent spins. This passes on current code and proves the measurement is real
/// (not vacuously failing).
#[test]
fn split_preserves_point_velocity_for_aligned_com_child() {
    let (records, _) = drive_rotating_pair_split(&pair_scenario(None));
    let worst = worst_continuity(&records);
    assert!(
        worst.point_velocity_error < POINT_VELOCITY_CONTINUITY_TOL,
        "aligned-COM split should be continuous, but node {} on body {:?} drifted {:.5} m/s \
         (parent v {:?} vs child v {:?})",
        worst.node_index,
        worst.target_body,
        worst.point_velocity_error,
        worst.parent_point_velocity,
        worst.child_point_velocity,
    );
}

/// REGRESSION GUARD (gap #1, fixed). Identical to the positive control except node 1 carries
/// a convex-hull collider whose volume centre of mass is offset from the node centroid. The
/// split path now reconciles each child's velocity with Rapier's ACTUAL centre of mass, so a
/// rotating parent no longer injects `~|ω × offset|` of spurious velocity — the fragment stays
/// point-velocity continuous. (Before the fix this drifted ~2.1 m/s.)
#[test]
fn split_preserves_point_velocity_for_offset_com_child() {
    let (records, _) = drive_rotating_pair_split(&pair_scenario(Some(offset_hull())));
    let worst = worst_continuity(&records);
    assert!(
        worst.point_velocity_error < POINT_VELOCITY_CONTINUITY_TOL,
        "offset-COM split injected spurious motion: node {} on body {:?} drifted {:.5} m/s \
         (parent v {:?} vs child v {:?})",
        worst.node_index,
        worst.target_body,
        worst.point_velocity_error,
        worst.parent_point_velocity,
        worst.child_point_velocity,
    );
}

// ---------------------------------------------------------------------------
// Always-true invariants (blocking) — hold regardless of the COM bug.
// ---------------------------------------------------------------------------

/// No post-split body or continuity sample may contain NaN/Inf. The COM bug yields
/// finite-but-wrong velocities, so this guards specifically against the
/// ill-conditioned angular fit (a near-singular fragment) producing garbage.
#[test]
fn split_produces_finite_state() {
    let (records, summaries) = drive_rotating_pair_split(&pair_scenario(Some(offset_hull())));
    for record in &records {
        assert!(
            record.finite,
            "non-finite child point velocity at node {}: {:?}",
            record.node_index, record.child_point_velocity
        );
    }
    for summary in &summaries {
        assert!(summary.finite, "non-finite body state after split: {summary:?}");
    }
}

/// Total dynamic mass is conserved across a fracture: a split repartitions mass, it
/// never creates or destroys it.
#[test]
fn dynamic_mass_conserved_across_split() {
    let scenario = pair_scenario(None);
    let initial_dynamic_mass: f32 = scenario.nodes.iter().map(|n| n.mass).sum();
    let (_, summaries) = drive_rotating_pair_split(&scenario);
    let total: f32 = summaries
        .iter()
        .filter(|s| s.dynamic)
        .map(|s| s.mass)
        .sum();
    let rel = (total - initial_dynamic_mass).abs() / initial_dynamic_mass.max(f32::EPSILON);
    assert!(
        rel < MASS_REL_TOL,
        "dynamic mass not conserved: got {total}, expected {initial_dynamic_mass} (rel {rel:e})"
    );
}

// ---------------------------------------------------------------------------
// Determinism harness — detects nondeterminism (e.g. HashMap iteration order in
// the split planner). See blast/TESTING.md gap #2.
// ---------------------------------------------------------------------------

/// Run the identical fracture twice in fresh worlds and require identical outcomes
/// (fracture count + the multiset of post-split body centre-of-mass positions). We
/// compare a sorted set of positions rather than per-handle values so legitimate
/// body-handle relabeling does not flake the test — only true physical divergence
/// fails it.
#[test]
fn simulation_is_deterministic_across_runs() {
    fn run() -> (usize, Vec<(i64, i64, i64)>) {
        let scenario = pair_scenario(None);
        let policy = FracturePolicy {
            idle_skip: false,
            ..FracturePolicy::default()
        };
        let mut set =
            DestructibleSet::from_scenario(&scenario, weak_settings(), Vec3::ZERO, policy).unwrap();
        let (mut bodies, mut colliders, mut islands, mut ij, mut mj) = rapier_world();
        set.initialize(&mut bodies, &mut colliders);
        let mut fractures = 0usize;
        for _ in 0..60 {
            set.add_force(1, scenario.nodes[1].centroid, Vec3::new(2000.0, 0.0, 0.0));
            let r = set.step(&mut bodies, &mut colliders, &mut islands, &mut ij, &mut mj);
            fractures += r.fractures;
        }
        // Quantize positions so the comparison is exact and order-independent.
        let mut positions: Vec<(i64, i64, i64)> = bodies
            .iter()
            .map(|(_, b)| {
                let c = b.center_of_mass();
                let q = |v: f32| (v as f64 * 1.0e6).round() as i64;
                (q(c.x), q(c.y), q(c.z))
            })
            .collect();
        positions.sort_unstable();
        (fractures, positions)
    }

    let first = run();
    let second = run();
    assert_eq!(
        first, second,
        "two identical runs diverged (nondeterminism): {first:?} vs {second:?}"
    );
}
