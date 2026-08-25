//! Adjudicates a pre-existing cross-validation divergence.
//!
//! `cross_validation_test::determinism_parity_with_js` records the same number
//! of fractures as the JS reference (86) but a different actor count (Rust 32,
//! JS 21). Since both bind the same C++ solver, one of those counts is wrong —
//! but the parity fixture cannot say which, because it only compares two
//! recorded numbers to each other.
//!
//! This computes the answer independently: replay the identical simulation,
//! track which bonds were actually fractured, then count connected components
//! of the surviving bond graph with a union-find that knows nothing about the
//! solver. Whichever recorded number that matches is the correct one.

use std::collections::HashSet;

use blast_stress_solver::scenarios::{build_wall_scenario, WallOptions};
use blast_stress_solver::{ExtStressSolver, SolverSettings, Vec3};

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
        if ra != rb { let (lo, hi) = (ra.min(rb), ra.max(rb)); self.0[hi as usize] = lo; }
    }
}

fn main() {
    let settings = SolverSettings {
        max_solver_iterations_per_frame: 24,
        compression_elastic_limit: 5.0,
        compression_fatal_limit: 10.0,
        tension_elastic_limit: 5.0,
        tension_fatal_limit: 10.0,
        shear_elastic_limit: 5.0,
        shear_fatal_limit: 10.0,
        ..SolverSettings::default()
    };

    let wall = build_wall_scenario(&WallOptions::default());
    let (nodes, bonds) = wall.to_solver_descs();
    let node_count = nodes.len();
    let bond_count = bonds.len();
    let supports = nodes.iter().filter(|n| n.mass == 0.0).count();
    let mut solver = ExtStressSolver::new(&nodes, &bonds, &settings).unwrap();

    // Replay the fixture's loop, recording every (n0,n1) pair we fracture.
    let mut broken: HashSet<(u32, u32)> = HashSet::new();
    let mut total_fractures = 0u32;
    for _ in 0..30 {
        solver.add_gravity(Vec3::new(0.0, -9.81, 0.0));
        solver.update();
        if solver.overstressed_bond_count() > 0 {
            let cmds = solver.generate_fracture_commands();
            for c in &cmds {
                for bf in &c.bond_fractures {
                    let (a, b) = (bf.node_index0.min(bf.node_index1), bf.node_index0.max(bf.node_index1));
                    broken.insert((a, b));
                    total_fractures += 1;
                }
            }
            if !cmds.is_empty() { solver.apply_fracture_commands(&cmds); }
        }
    }

    // Independent component count over the surviving bond graph.
    let mut uf = Uf::new(node_count);
    let mut surviving = 0usize;
    for b in &bonds {
        let (a, c) = (b.node0.min(b.node1), b.node0.max(b.node1));
        if !broken.contains(&(a, c)) { uf.union(b.node0, b.node1); surviving += 1; }
    }
    let mut roots = HashSet::new();
    for n in 0..node_count as u32 { let r = uf.find(n); roots.insert(r); }

    println!("scenario:  {node_count} nodes ({supports} support), {bond_count} bonds");
    println!("fractures: {total_fractures} (unique bonds broken: {})", broken.len());
    println!("surviving bonds: {surviving}");
    println!("independent connected components: {}", roots.len());
    println!("solver.actor_count():             {}", solver.actor_count());
    println!("JS reference actor count:         21");
}
