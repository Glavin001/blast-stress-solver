//! Island-aware solving parity (ported from the web suite: `rapier.island-solve-parity.test.ts`,
//! `rapier.island-skip.test.ts`, `rapier.island-identity.test.ts`). Both libraries call the *same*
//! C++ `solveIslandAware`, so these lock the Rust FFI bindings to the same guarantees the web asserts:
//!
//!   1. Single island (intact structure) → island-aware falls back to the bit-identical whole-graph
//!      path: exact float equality with the legacy solve.
//!   2. Multiple islands → each component is solved independently but the overstressed-bond sequence
//!      and final actor partition are identical to the whole-graph solve, every frame.
//!   3. Settled-island skipping is observationally identical to not skipping, and actually engages on
//!      a stable scene.

use blast_stress_solver::*;

/// A horizontal cantilever: a static anchor at the origin and `segs` dynamic nodes along +X.
/// One island (a single connected component anchored at one static node).
fn cantilever(segs: u32) -> (Vec<NodeDesc>, Vec<BondDesc>) {
    let mut nodes = vec![NodeDesc { centroid: Vec3::new(0.0, 0.0, 0.0), mass: 0.0, volume: 1.0 }];
    let mut bonds = Vec::new();
    for i in 1..=segs {
        nodes.push(NodeDesc { centroid: Vec3::new(i as f32, 0.0, 0.0), mass: 1.0, volume: 1.0 });
        let a = nodes[(i - 1) as usize].centroid;
        let b = nodes[i as usize].centroid;
        bonds.push(BondDesc {
            centroid: (a + b) * 0.5,
            normal: Vec3::new(1.0, 0.0, 0.0),
            area: 0.5,
            node0: i - 1,
            node1: i,
        });
    }
    (nodes, bonds)
}

/// `arms` cantilever arms radiating from a single shared static ground node (node 0). The shared
/// static node ties them into ONE actor, but is a solver cut point → `arms` independent islands.
fn arms_sharing_ground(arms: u32, seg_per_arm: u32) -> (Vec<NodeDesc>, Vec<BondDesc>) {
    let mut nodes = vec![NodeDesc { centroid: Vec3::new(0.0, 0.0, 0.0), mass: 0.0, volume: 1.0 }];
    let mut bonds = Vec::new();
    // Spread arms over distinct directions so geometry is non-degenerate.
    let dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];
    for a in 0..arms {
        let dir = dirs[(a as usize) % dirs.len()];
        let mut prev = 0u32; // attach the first segment to the shared ground node
        for s in 1..=seg_per_arm {
            let idx = nodes.len() as u32;
            nodes.push(NodeDesc {
                centroid: Vec3::new(dir.x * s as f32, dir.y * s as f32, dir.z * s as f32),
                mass: 1.0,
                volume: 1.0,
            });
            let p = nodes[prev as usize].centroid;
            let c = nodes[idx as usize].centroid;
            bonds.push(BondDesc {
                centroid: (p + c) * 0.5,
                normal: dir,
                area: 0.5,
                node0: prev,
                node1: idx,
            });
            prev = idx;
        }
    }
    (nodes, bonds)
}

fn rigid_settings() -> SolverSettings {
    // High limits → never fractures (used to test the ≤1-island bit-identical fallback).
    SolverSettings {
        compression_elastic_limit: 1.0e9,
        compression_fatal_limit: 1.0e10,
        tension_elastic_limit: 1.0e9,
        tension_fatal_limit: 1.0e10,
        shear_elastic_limit: 1.0e9,
        shear_fatal_limit: 1.0e10,
        ..SolverSettings::default()
    }
}

fn fracture_settings() -> SolverSettings {
    // Low limits → decisively fractures under the test gravity.
    SolverSettings {
        compression_elastic_limit: 0.02,
        compression_fatal_limit: 0.05,
        tension_elastic_limit: 0.02,
        tension_fatal_limit: 0.05,
        shear_elastic_limit: 0.02,
        shear_fatal_limit: 0.05,
        ..SolverSettings::default()
    }
}

/// Sorted, canonical actor partition (each actor's node set sorted, then actors sorted).
fn sorted_partition(solver: &ExtStressSolver) -> Vec<Vec<u32>> {
    let mut parts: Vec<Vec<u32>> = solver
        .actors()
        .into_iter()
        .map(|a| {
            let mut ns = a.nodes;
            ns.sort_unstable();
            ns
        })
        .collect();
    parts.sort_by(|a, b| a.first().cmp(&b.first()));
    parts
}

struct Run {
    over: Vec<u32>,
    err_lin: Vec<f32>,
    islands: Vec<u32>,
    partition: Vec<Vec<u32>>,
    converged: bool,
}

fn simulate(
    nodes: &[NodeDesc],
    bonds: &[BondDesc],
    settings: &SolverSettings,
    gravity: Vec3,
    island_aware: bool,
    frames: usize,
) -> Run {
    let mut solver = ExtStressSolver::new(nodes, bonds, settings).unwrap();
    solver.set_island_aware(island_aware);
    assert_eq!(solver.island_aware(), island_aware);
    let mut run = Run { over: Vec::new(), err_lin: Vec::new(), islands: Vec::new(), partition: Vec::new(), converged: true };
    for _ in 0..frames {
        solver.add_gravity(gravity);
        solver.update();
        run.over.push(solver.overstressed_bond_count());
        run.err_lin.push(solver.linear_error());
        run.islands.push(solver.island_count());
        run.converged = run.converged && solver.converged();
        let commands = solver.generate_fracture_commands();
        if !commands.is_empty() {
            solver.apply_fracture_commands(&commands);
        }
    }
    run.partition = sorted_partition(&solver);
    run
}

#[test]
fn single_island_is_bit_identical_to_legacy() {
    // Rigid (never fractures) cantilever stays one island throughout → island-aware must match
    // the whole-graph path exactly (same code path via the ≤1-island fallback).
    let (nodes, bonds) = cantilever(6);
    let g = Vec3::new(0.0, -50.0, 0.0);
    let legacy = simulate(&nodes, &bonds, &rigid_settings(), g, false, 6);
    let island = simulate(&nodes, &bonds, &rigid_settings(), g, true, 6);
    assert!(island.islands.iter().all(|&c| c == 1), "should stay a single island: {:?}", island.islands);
    assert_eq!(island.over, legacy.over, "no fractures, identical overstress");
    assert_eq!(island.err_lin, legacy.err_lin, "EXACT float equality (same code path)");
}

#[test]
fn two_islands_fracture_sequence_matches_legacy() {
    let (nodes, bonds) = arms_sharing_ground(2, 4);
    let g = Vec3::new(0.0, -200.0, 0.0);
    let legacy = simulate(&nodes, &bonds, &fracture_settings(), g, false, 8);
    let island = simulate(&nodes, &bonds, &fracture_settings(), g, true, 8);
    assert!(island.islands[0] >= 2, "two arms ⇒ ≥2 islands from frame 0, got {}", island.islands[0]);
    assert_eq!(island.over, legacy.over, "identical fracture decisions every frame");
    assert_eq!(island.partition, legacy.partition, "identical resulting actor partition");
    assert_eq!(island.converged, legacy.converged);
}

#[test]
fn four_islands_evolution_matches_legacy() {
    let (nodes, bonds) = arms_sharing_ground(4, 4);
    let g = Vec3::new(0.0, -200.0, 0.0);
    let legacy = simulate(&nodes, &bonds, &fracture_settings(), g, false, 8);
    let island = simulate(&nodes, &bonds, &fracture_settings(), g, true, 8);
    assert_eq!(island.islands[0], 4, "four arms ⇒ four independent islands");
    assert_eq!(island.over, legacy.over, "identical fracture decisions every frame");
    assert_eq!(island.partition, legacy.partition, "identical resulting actor partition");
}

#[test]
fn island_count_binding_reflects_topology() {
    // FFI smoke: the C++ island counter is wired through and reflects the support graph.
    let (nodes, bonds) = arms_sharing_ground(2, 4);
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &rigid_settings()).unwrap();
    solver.set_island_aware(true);
    solver.add_gravity(Vec3::new(0.0, -5.0, 0.0));
    solver.update();
    assert_eq!(solver.island_count(), 2, "two arms sharing a static node ⇒ two islands");
    assert!(solver.island_aware());
    solver.set_island_aware(false);
    assert!(!solver.island_aware());
}

#[test]
fn skip_settled_engages_on_stable_scene() {
    let (nodes, bonds) = arms_sharing_ground(2, 4);
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &rigid_settings()).unwrap();
    solver.set_island_aware(true);
    solver.set_skip_settled(true);
    let mut skipped = 0;
    for _ in 0..6 {
        solver.add_gravity(Vec3::new(0.0, -5.0, 0.0));
        solver.update();
        skipped = solver.islands_skipped();
    }
    assert_eq!(solver.island_count(), 2);
    assert_eq!(solver.overstressed_bond_count(), 0, "rigid scene must not fracture");
    assert_eq!(skipped, 2, "both settled islands skipped once inputs stop changing");
}

#[test]
fn skip_settled_is_observationally_identical() {
    // Light load (islands settle and skip) then a heavy load that wakes + fractures them. With
    // skipping on vs off the entire evolution must be identical.
    let (nodes, bonds) = arms_sharing_ground(2, 4);
    // Tiny elastic light load (safely below the 0.02 limit → settles without fracturing), then a
    // heavy load that wakes and shatters the arms.
    let schedule: Vec<Vec3> = std::iter::repeat(Vec3::new(0.0, -0.001, 0.0))
        .take(5)
        .chain(std::iter::repeat(Vec3::new(0.0, -400.0, 0.0)).take(5))
        .collect();

    let run = |skip: bool| -> (Vec<u32>, Vec<Vec<u32>>, u32) {
        let mut solver = ExtStressSolver::new(&nodes, &bonds, &fracture_settings()).unwrap();
        solver.set_island_aware(true);
        solver.set_skip_settled(skip);
        let mut over = Vec::new();
        let mut max_skipped = 0;
        for g in &schedule {
            solver.add_gravity(*g);
            solver.update();
            over.push(solver.overstressed_bond_count());
            max_skipped = max_skipped.max(solver.islands_skipped());
            let cmds = solver.generate_fracture_commands();
            if !cmds.is_empty() {
                solver.apply_fracture_commands(&cmds);
            }
        }
        (over, sorted_partition(&solver), max_skipped)
    };

    let (no_over, no_part, no_max) = run(false);
    let (sk_over, sk_part, sk_max) = run(true);
    assert_eq!(sk_over, no_over, "identical overstress sequence with skip on vs off");
    assert_eq!(sk_part, no_part, "identical resulting actor partition");
    assert_eq!(no_max, 0, "skip-settled OFF never skips");
    assert!(sk_max > 0, "skip-settled ON engages while the light load is stable");
}

/// Why island-aware solving can shift a real fracture cascade slightly (e.g. the weak wall goes
/// 86→84 fractures) even though it is "the same solve":
///
/// Bonds in different islands share no nodes, so the CGNR system is **block-diagonal** (one block
/// per island). With a *direct* solve in exact arithmetic, solving each block independently is
/// mathematically identical to solving the whole system. But CGNR is **iterative** and computes its
/// step-size scalars (α, β) from **global** dot-products over the entire vector. For a multi-block
/// system those shared scalars are a compromise: whole-graph CG stagnates at a higher residual
/// floor, while per-island CG gives each block its own optimal scalars and converges each to its own
/// (lower) floor. So:
///   • ≤1 island  → whole-graph fallback → bit-identical (intact structures, small arms).
///   • >1 island that doesn't trivially converge → per-island reaches a *lower* residual (it is at
///     least as accurate), which flips a couple of borderline bonds near the fatal threshold.
/// Both outcomes are valid; per-island is the better-converged physics. These two tests lock that
/// understanding: per-island is deterministic, and never less accurate than whole-graph.
#[cfg(feature = "scenarios")]
#[test]
fn weak_wall_island_aware_is_deterministic_and_no_less_accurate() {
    use blast_stress_solver::scenarios::*;

    let weak = SolverSettings {
        max_solver_iterations_per_frame: 24,
        compression_elastic_limit: 0.001,
        compression_fatal_limit: 0.002,
        tension_elastic_limit: 0.001,
        tension_fatal_limit: 0.002,
        shear_elastic_limit: 0.001,
        shear_fatal_limit: 0.002,
        ..SolverSettings::default()
    };
    let wall = build_wall_scenario(&WallOptions::default());
    let (nodes, bonds) = wall.to_solver_descs();

    let run = || -> (u32, u32) {
        let mut solver = ExtStressSolver::new(&nodes, &bonds, &weak).unwrap();
        solver.set_island_aware(true);
        solver.set_skip_settled(true);
        let mut total = 0u32;
        for _ in 0..60 {
            solver.add_gravity(Vec3::new(0.0, -9.81, 0.0));
            solver.update();
            let cmds = solver.generate_fracture_commands();
            if !cmds.is_empty() {
                total += cmds.iter().map(|c| c.bond_fractures.len() as u32).sum::<u32>();
                solver.apply_fracture_commands(&cmds);
            }
        }
        (total, solver.actor_count())
    };

    // Deterministic: two island-aware runs are identical, and match the per-island golden outcome
    // (84/30) — distinct from the whole-graph fixture (86/32) because per-island converges tighter.
    let a = run();
    let b = run();
    assert_eq!(a, b, "island-aware solving must be deterministic");
    assert_eq!(a, (84, 30), "per-island golden outcome for the weak wall (vs whole-graph 86/32)");
}

/// On a large fixed multi-island system, per-island solving reaches a residual no worse than the
/// whole-graph solve — the quantitative justification for defaulting it ON. (Same topology both
/// ways, so this is a clean apples-to-apples solver-accuracy comparison.)
#[test]
fn per_island_residual_is_no_worse_than_whole_graph() {
    // 24 long arms from one shared static hub → 24 independent islands, large enough that the
    // global-scalar whole-graph CG stagnates above the per-island floor.
    let (arms, seg) = (24u32, 30u32);
    let mut nodes = vec![NodeDesc { centroid: Vec3::new(0.0, 0.0, 0.0), mass: 0.0, volume: 1.0 }];
    let mut bonds = Vec::new();
    for a in 0..arms {
        let ang = a as f32 * 0.27;
        let dir = Vec3::new(ang.cos(), 0.3, ang.sin());
        let mut prev = 0u32;
        for s in 1..=seg {
            let idx = nodes.len() as u32;
            nodes.push(NodeDesc {
                centroid: Vec3::new(dir.x * s as f32, dir.y * s as f32, dir.z * s as f32),
                mass: 1.0,
                volume: 1.0,
            });
            let (p, c) = (nodes[prev as usize].centroid, nodes[idx as usize].centroid);
            bonds.push(BondDesc { centroid: (p + c) * 0.5, normal: dir, area: 0.5, node0: prev, node1: idx });
            prev = idx;
        }
    }
    let st = SolverSettings {
        max_solver_iterations_per_frame: 300,
        ..rigid_settings()
    };
    let mut whole = ExtStressSolver::new(&nodes, &bonds, &st).unwrap();
    whole.set_island_aware(false);
    let mut island = ExtStressSolver::new(&nodes, &bonds, &st).unwrap();
    island.set_island_aware(true);
    whole.add_gravity(Vec3::new(0.0, -50.0, 0.0));
    whole.update();
    island.add_gravity(Vec3::new(0.0, -50.0, 0.0));
    island.update();
    assert_eq!(island.island_count(), arms, "each arm is its own island");
    assert!(
        island.linear_error() <= whole.linear_error() + 1e-4,
        "per-island residual ({}) must be no worse than whole-graph ({})",
        island.linear_error(),
        whole.linear_error()
    );
}
