//! Criterion micro/throughput benchmarks for the destructible pipeline hot paths.
//!
//! These give statistically-stable numbers (warm-up, many samples, outlier detection,
//! `--save-baseline`/compare) for the specific operations an optimization touches:
//!   * `solver_update`      — the C++ CGNR stress solve per frame, vs structure size.
//!   * `full_step_steady`   — a whole `DestructibleSet::step` with no fracture (gravity hold).
//!   * `split_planning`     — the topology-diff (body reuse vs create) planner.
//!   * `snapshot`           — resim `BodySnapshots` capture/restore, vs body count.
//!
//! Per-frame *spike* behavior over a destruction cascade is profiled separately by the
//! `frame_profile` example (max/p99 over a realistic timeline).
//!
//!   cargo bench --features bench-support
//!   cargo bench --features bench-support -- --save-baseline before
//!   # (after a change) cargo bench --features bench-support -- --baseline before

use std::collections::HashSet;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rapier3d::prelude::*;

use blast_stress_solver::bench_harness::*;
use blast_stress_solver::rapier::{
    plan_split_migration_reference, plan_split_migration_with_support, BodySnapshots,
    ExistingBodyState, FracturePolicy, PlannerChildSupport, ResimulationOptions,
};
use blast_stress_solver::*;

/// Steady-state cost of one stress solve (`add_gravity` + `update`) on a settled graph.
fn bench_solver_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_update");

    let mut cases: Vec<(String, ScenarioDesc, SolverSettings, Vec3)> = vec![
        ("wall_12x6".into(), wall(12, 6, 1), Material::Strong.settings(), Vec3::new(0.0, -9.81, 0.0)),
        ("wall_24x12".into(), wall(24, 12, 1), Material::Strong.settings(), Vec3::new(0.0, -9.81, 0.0)),
        ("wall_36x18x2".into(), wall(36, 18, 2), Material::Strong.settings(), Vec3::new(0.0, -9.81, 0.0)),
        ("tower_4x16".into(), tower(4, 16), Material::Strong.settings(), Vec3::new(0.0, -9.81, 0.0)),
    ];
    for pack in ["fractured-tower", "fractured-bridge", "high-rise"] {
        if let Some(l) = try_load_scene(pack) {
            cases.push((pack.into(), l.scenario.clone(), l.settings, l.gravity_vec()));
        }
    }

    for (name, scn, settings, gravity) in &cases {
        let (nodes, bonds) = scn.to_solver_descs();
        let mut solver = ExtStressSolver::new(&nodes, &bonds, settings).expect("solver");
        // Warm to steady state (warm-start converges in a couple iterations thereafter).
        for _ in 0..8 {
            solver.add_gravity(*gravity);
            solver.update();
        }
        group.throughput(Throughput::Elements(bonds.len() as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                solver.add_gravity(black_box(*gravity));
                solver.update();
                black_box(solver.overstressed_bond_count())
            })
        });
    }
    group.finish();
}

/// Whole-frame cost with no fracture (the common 120 FPS case: gravity hold, solver runs).
fn bench_full_step_steady(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_step_steady");
    let always_solve = FracturePolicy {
        idle_skip: false,
        ..FracturePolicy::default()
    };

    let cases: Vec<(&str, ScenarioDesc)> = vec![
        ("wall_12x6", wall(12, 6, 1)),
        ("wall_36x18x2", wall(36, 18, 2)),
        ("tower_4x16", tower(4, 16)),
    ];

    for (name, scn) in &cases {
        let mut sim = Sim::new(
            scn,
            SimConfig {
                settings: Material::Strong.settings(),
                policy: always_solve,
                resim: ResimulationOptions { enabled: false, max_passes: 0 },
                with_ground: true,
                ..SimConfig::default()
            },
        );
        sim.set_record_timing(false);
        // Let it settle.
        for _ in 0..30 {
            sim.step_frame();
        }
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| black_box(sim.step_frame()))
        });
    }
    group.finish();
}

/// Topology-diff planner: map child node-sets onto existing parent bodies.
fn bench_split_planning(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_planning");
    let mut set = RigidBodySet::new();

    // Case A: clean 2-way split of a 64-node body.
    {
        let parent = set.insert(RigidBodyBuilder::dynamic());
        let existing = vec![ExistingBodyState {
            handle: parent,
            node_indices: (0u32..64).collect::<HashSet<_>>(),
            is_fixed: false,
        }];
        let children = vec![
            SplitChild { actor_index: 0, nodes: (0u32..32).collect() },
            SplitChild { actor_index: 1, nodes: (32u32..64).collect() },
        ];
        let support = vec![PlannerChildSupport::default(); 2];
        group.bench_function("clean_2way_64", |b| {
            b.iter(|| {
                black_box(plan_split_migration_with_support(
                    black_box(&existing),
                    black_box(&children),
                    black_box(&support),
                ))
            })
        });
    }

    // Case B: shatter — one 64-node body into 16 children of 4 (overlap matching / Hungarian).
    {
        let mut existing = Vec::new();
        for k in 0..4u32 {
            let h = set.insert(RigidBodyBuilder::dynamic());
            existing.push(ExistingBodyState {
                handle: h,
                node_indices: (k * 16..k * 16 + 16).collect::<HashSet<_>>(),
                is_fixed: false,
            });
        }
        let children: Vec<SplitChild> = (0..16u32)
            .map(|k| SplitChild { actor_index: k, nodes: (k * 4..k * 4 + 4).collect() })
            .collect();
        let support = vec![PlannerChildSupport::default(); 16];
        group.bench_function("shatter_4bodies_16children", |b| {
            b.iter(|| {
                black_box(plan_split_migration_with_support(
                    black_box(&existing),
                    black_box(&children),
                    black_box(&support),
                ))
            })
        });
    }

    // Case C: the catastrophic cascade — ONE parent body shatters into 256 children.
    // This is the common "whole structure lets go in one frame" case; the assignment
    // problem is 1×256 but a naive Hungarian pads it to 256×256 (O(256^3)).
    {
        let parent = set.insert(RigidBodyBuilder::dynamic());
        let existing = vec![ExistingBodyState {
            handle: parent,
            node_indices: (0u32..256).collect::<HashSet<_>>(),
            is_fixed: false,
        }];
        let children: Vec<SplitChild> = (0..256u32)
            .map(|k| SplitChild { actor_index: k, nodes: vec![k] })
            .collect();
        let support = vec![PlannerChildSupport::default(); 256];
        group.bench_function("cascade_1body_256children", |b| {
            b.iter(|| {
                black_box(plan_split_migration_with_support(
                    black_box(&existing),
                    black_box(&children),
                    black_box(&support),
                ))
            })
        });
        // Fast path vs. the reference (forced) Hungarian on the *same* 1×256 input — the
        // A/B that isolates the planner optimization (O(N) argmax vs O(N^3) padded solve).
        group.bench_function("cascade_1body_256children_reference_hungarian", |b| {
            b.iter(|| {
                black_box(plan_split_migration_reference(
                    black_box(&existing),
                    black_box(&children),
                    black_box(&support),
                ))
            })
        });
    }

    // Case D: complex partial reparenting — the genuine worst case. Many surviving
    // multi-node bodies competing for many reparented children (no single-parent shortcut;
    // this always runs the full overlap-matrix + Hungarian). Scaled to show O(max(R,C)^3).
    for &m in &[16usize, 32, 64] {
        // m bodies own 4 nodes each; the shatter regroups into m children that each
        // straddle two adjacent bodies (overlap 2 vs 2 — dense, non-trivial assignment).
        let mut existing = Vec::new();
        for k in 0..m as u32 {
            let h = set.insert(RigidBodyBuilder::dynamic());
            existing.push(ExistingBodyState {
                handle: h,
                node_indices: (k * 4..k * 4 + 4).collect::<HashSet<_>>(),
                is_fixed: false,
            });
        }
        let children: Vec<SplitChild> = (0..m as u32)
            .map(|k| {
                let a = k * 4 + 2; // last 2 nodes of body k
                let b = ((k + 1) % m as u32) * 4; // first 2 nodes of body k+1
                SplitChild { actor_index: k, nodes: vec![a, a + 1, b, b + 1] }
            })
            .collect();
        let support = vec![PlannerChildSupport::default(); m];
        group.bench_function(BenchmarkId::new("complex_reparent_MxM", m), |b| {
            b.iter(|| {
                black_box(plan_split_migration_with_support(
                    black_box(&existing),
                    black_box(&children),
                    black_box(&support),
                ))
            })
        });
    }
    group.finish();
}

/// Resim snapshot primitives vs. number of dynamic bodies.
fn bench_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot");
    for &n in &[64usize, 256, 1024] {
        let mut bodies = RigidBodySet::new();
        for i in 0..n {
            bodies.insert(
                RigidBodyBuilder::dynamic()
                    .translation(vector![i as f32, 1.0, 0.5])
                    .linvel(vector![0.1, -0.2, 0.3])
                    .angvel(vector![0.01, 0.02, 0.03]),
            );
        }
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("capture", n), &bodies, |b, bodies| {
            b.iter(|| black_box(BodySnapshots::capture(black_box(bodies))))
        });
        // capture_into reuses its buffer — no per-call allocation once warm (the resim path).
        let mut reused = BodySnapshots::capture(&bodies);
        group.bench_with_input(BenchmarkId::new("capture_into_reused", n), &bodies, |b, bodies| {
            b.iter(|| reused.capture_into(black_box(bodies)))
        });
        let snap = BodySnapshots::capture(&bodies);
        group.bench_with_input(BenchmarkId::new("restore", n), &snap, |b, snap| {
            b.iter(|| snap.restore(black_box(&mut bodies)))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_solver_update,
    bench_full_step_steady,
    bench_split_planning,
    bench_snapshot
);
criterion_main!(benches);
