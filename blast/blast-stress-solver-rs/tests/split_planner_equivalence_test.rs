//! Proves the generalised split planner is identical to the Rapier-bound one
//! it replaces.
//!
//! Moving an algorithm out of an adapter and into the core is only safe if it
//! demonstrably computes the same thing. The original
//! `rapier::split_migrator` is still in tree, so the two can be run against
//! each other on randomised inputs rather than trusted by inspection.

#![cfg(all(feature = "rapier", feature = "scenarios"))]

use std::collections::HashSet;

use blast_stress_solver::pipeline::split_planner as newp;
use blast_stress_solver::rapier::{plan_split_migration as old_plan, ExistingBodyState as OldState};
use blast_stress_solver::types::SplitChild;
use rapier3d::prelude::{RigidBodyBuilder, RigidBodyHandle, RigidBodySet};

/// SplitMix64 — deterministic, so a failure is reproducible from the seed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        if n == 0 { 0 } else { self.next() % n }
    }
}

fn handles(bodies: &mut RigidBodySet, n: usize) -> Vec<RigidBodyHandle> {
    (0..n).map(|_| bodies.insert(RigidBodyBuilder::dynamic().build())).collect()
}

#[test]
fn generalised_planner_matches_the_rapier_bound_original() {
    let mut rng = Rng(0xC0FFEE);
    let mut mismatches = Vec::new();

    for case in 0..400u32 {
        let mut set = RigidBodySet::new();
        let body_count = 1 + rng.below(5) as usize;
        let child_count = 1 + rng.below(6) as usize;
        let node_pool = 4 + rng.below(12) as u32;
        let hs = handles(&mut set, body_count);

        // Existing bodies own disjoint node sets, which is the planner's
        // documented precondition (a node lives on exactly one body).
        let mut owner: Vec<Option<usize>> = vec![None; node_pool as usize];
        for n in 0..node_pool as usize {
            if rng.below(4) != 0 {
                owner[n] = Some(rng.below(body_count as u64) as usize);
            }
        }

        let mk_old: Vec<OldState> = (0..body_count)
            .map(|i| OldState {
                handle: hs[i],
                node_indices: owner
                    .iter()
                    .enumerate()
                    .filter_map(|(n, o)| (*o == Some(i)).then_some(n as u32))
                    .collect::<HashSet<u32>>(),
                is_fixed: false,
            })
            .collect();
        let mk_new: Vec<newp::ExistingBodyState<RigidBodyHandle>> = (0..body_count)
            .map(|i| newp::ExistingBodyState {
                handle: hs[i],
                node_indices: owner
                    .iter()
                    .enumerate()
                    .filter_map(|(n, o)| (*o == Some(i)).then_some(n as u32))
                    .collect::<HashSet<u32>>(),
                is_fixed: false,
            })
            .collect();

        let children: Vec<SplitChild> = (0..child_count)
            .map(|c| SplitChild {
                actor_index: c as u32,
                nodes: (0..node_pool).filter(|_| rng.below(3) == 0).collect(),
            })
            .filter(|c: &SplitChild| !c.nodes.is_empty())
            .collect();
        if children.is_empty() {
            continue;
        }

        let a = old_plan(&mk_old, &children);
        let b = newp::plan_split_migration(&mk_new, &children);

        let a_reuse: Vec<(usize, RigidBodyHandle)> =
            a.reuse.iter().map(|r| (r.child_index, r.body_handle)).collect();
        let b_reuse: Vec<(usize, RigidBodyHandle)> =
            b.reuse.iter().map(|r| (r.child_index, r.body_handle)).collect();
        let a_create: Vec<usize> = a.create.iter().map(|c| c.child_index).collect();
        let b_create: Vec<usize> = b.create.iter().map(|c| c.child_index).collect();

        if a_reuse != b_reuse || a_create != b_create {
            mismatches.push(format!(
                "case {case}: bodies={body_count} children={} old_reuse={a_reuse:?} \
                 new_reuse={b_reuse:?} old_create={a_create:?} new_create={b_create:?}",
                children.len()
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "generalised planner diverged on {} of 400 cases:\n  {}",
        mismatches.len(),
        mismatches.iter().take(5).cloned().collect::<Vec<_>>().join("\n  ")
    );
}

#[test]
fn plan_order_is_deterministic_across_runs() {
    // The plan's ordering is load-bearing: the pipeline creates bodies in this
    // order so the engine allocates handles reproducibly.
    let mut set = RigidBodySet::new();
    let hs = handles(&mut set, 3);
    let bodies: Vec<newp::ExistingBodyState<RigidBodyHandle>> = (0..3)
        .map(|i| newp::ExistingBodyState {
            handle: hs[i],
            node_indices: (0..4).map(|n| n + i as u32 * 4).collect(),
            is_fixed: false,
        })
        .collect();
    let children: Vec<SplitChild> = (0..4)
        .map(|c| SplitChild { actor_index: c, nodes: (0..3).map(|n| n + c * 3).collect() })
        .collect();

    let first = newp::plan_split_migration(&bodies, &children);
    for _ in 0..20 {
        let again = newp::plan_split_migration(&bodies, &children);
        let a: Vec<_> = first.reuse.iter().map(|r| (r.child_index, r.body_handle)).collect();
        let b: Vec<_> = again.reuse.iter().map(|r| (r.child_index, r.body_handle)).collect();
        assert_eq!(a, b, "plan order varied between identical runs");
    }
}
