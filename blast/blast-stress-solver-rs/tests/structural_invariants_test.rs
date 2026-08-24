//! Tier-1 structural invariants: truths that hold regardless of the physics.
//!
//! These are the assertions that transfer unchanged to any physics engine, so
//! they are the backbone of the cross-engine conformance corpus. A violation is
//! an integration bug, never a tuning difference.
//!
//! They are also self-adjudicating in a way parity fixtures are not. A recorded
//! "engine A got 21, engine B got 32" comparison cannot say which is right;
//! recomputing the quantity from first principles can.

#![cfg(feature = "scenarios")]

use std::collections::HashSet;

use blast_stress_solver::scenarios::{build_tower_scenario, build_wall_scenario, TowerOptions, WallOptions};
use blast_stress_solver::{BondDesc, ExtStressSolver, SolverSettings, Vec3};

/// Union-find with a lowest-root rule, so component roots are stable.
struct Uf(Vec<u32>);
impl Uf {
    fn new(n: usize) -> Self { Uf((0..n as u32).collect()) }
    fn find(&mut self, a: u32) -> u32 {
        let mut a = a;
        while self.0[a as usize] != a {
            let g = self.0[self.0[a as usize] as usize];
            self.0[a as usize] = g;
            a = g;
        }
        a
    }
    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.0[hi as usize] = lo;
        }
    }
}

struct RunResult {
    fractures: u32,
    broken: HashSet<(u32, u32)>,
    actor_count: u32,
    node_count: usize,
}

fn run(nodes: &[blast_stress_solver::NodeDesc], bonds: &[BondDesc], s: &SolverSettings, frames: u32) -> RunResult {
    let mut solver = ExtStressSolver::new(nodes, bonds, s).unwrap();
    let mut broken = HashSet::new();
    let mut fractures = 0u32;
    for _ in 0..frames {
        solver.add_gravity(Vec3::new(0.0, -9.81, 0.0));
        solver.update();
        if solver.overstressed_bond_count() > 0 {
            let cmds = solver.generate_fracture_commands();
            for c in &cmds {
                for bf in &c.bond_fractures {
                    let a = bf.node_index0.min(bf.node_index1);
                    let b = bf.node_index0.max(bf.node_index1);
                    broken.insert((a, b));
                    fractures += 1;
                }
            }
            if !cmds.is_empty() {
                solver.apply_fracture_commands(&cmds);
            }
        }
    }
    RunResult { fractures, broken, actor_count: solver.actor_count(), node_count: nodes.len() }
}

/// Count connected components of the surviving bond graph, independently of
/// the solver's own bookkeeping.
fn independent_components(r: &RunResult, bonds: &[BondDesc]) -> (usize, usize) {
    let mut uf = Uf::new(r.node_count);
    let mut surviving = 0usize;
    for b in bonds {
        let key = (b.node0.min(b.node1), b.node0.max(b.node1));
        if !r.broken.contains(&key) {
            uf.union(b.node0, b.node1);
            surviving += 1;
        }
    }
    let mut roots = HashSet::new();
    for n in 0..r.node_count as u32 {
        let root = uf.find(n);
        roots.insert(root);
    }
    (roots.len(), surviving)
}

fn weak() -> SolverSettings {
    SolverSettings {
        max_solver_iterations_per_frame: 24,
        compression_elastic_limit: 5.0,
        compression_fatal_limit: 10.0,
        tension_elastic_limit: 5.0,
        tension_fatal_limit: 10.0,
        shear_elastic_limit: 5.0,
        shear_fatal_limit: 10.0,
        ..SolverSettings::default()
    }
}

/// The invariant: the solver's actor count IS the number of connected
/// components of the live bond graph. Recomputing it from the fracture record
/// is what lets us say a disputed count is right or wrong on its own terms.
#[test]
fn actor_count_equals_independent_component_count() {
    for (name, (nodes, bonds)) in [
        ("wall", build_wall_scenario(&WallOptions::default()).to_solver_descs()),
        ("tower", build_tower_scenario(&TowerOptions::default()).to_solver_descs()),
    ] {
        let r = run(&nodes, &bonds, &weak(), 30);
        let (components, surviving) = independent_components(&r, &bonds);
        assert_eq!(
            r.actor_count as usize, components,
            "[{name}] solver reported {} actors but the surviving bond graph has {components} \
             connected components ({} nodes, {surviving} surviving bonds, {} broken)",
            r.actor_count, r.node_count, r.broken.len()
        );
    }
}

/// Fractures must be monotone: a bond is broken once, never resurrected.
#[test]
fn every_fracture_breaks_a_distinct_bond() {
    let (nodes, bonds) = build_wall_scenario(&WallOptions::default()).to_solver_descs();
    let r = run(&nodes, &bonds, &weak(), 30);
    assert_eq!(
        r.fractures as usize,
        r.broken.len(),
        "{} fracture commands but only {} distinct bonds — a bond was broken twice",
        r.fractures,
        r.broken.len()
    );
    assert!(r.broken.len() <= bonds.len(), "broke more bonds than exist");
}

/// A component count can never exceed the node count, nor drop below the count
/// implied by the surviving edges. For a forest the two coincide exactly.
#[test]
fn component_count_is_bounded_by_graph_size() {
    let (nodes, bonds) = build_wall_scenario(&WallOptions::default()).to_solver_descs();
    let r = run(&nodes, &bonds, &weak(), 30);
    let (components, surviving) = independent_components(&r, &bonds);
    assert!(components <= r.node_count, "more components than nodes");
    assert!(
        components >= r.node_count.saturating_sub(surviving),
        "{components} components is below the floor implied by {surviving} surviving edges"
    );
}
