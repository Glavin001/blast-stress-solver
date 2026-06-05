#![cfg(feature = "rapier")]
use rapier3d::prelude::*;
use blast_stress_solver::rapier::*;
use blast_stress_solver::*;

fn beam(segments: u32, seg: f32, height: f32) -> ScenarioDesc {
    let mut nodes = Vec::new();
    let mut bonds = Vec::new();
    let c = (segments - 1) as f32 / 2.0;
    for i in 0..segments {
        let x = (i as f32 - c) * seg;
        nodes.push(ScenarioNode { centroid: Vec3::new(x, height, 0.0), mass: 1.0, volume: seg*seg*seg });
    }
    for i in 0..segments-1 {
        let a = nodes[i as usize].centroid; let b = nodes[(i+1) as usize].centroid;
        bonds.push(ScenarioBond { node0: i, node1: i+1, centroid: (a+b)*0.5, normal: Vec3::new(1.0,0.0,0.0), area: seg*seg });
    }
    ScenarioDesc { nodes, bonds, node_sizes: vec![Vec3::new(seg,seg,seg); segments as usize], collider_shapes: Vec::new() }
}

fn run(omega: f32, fatal: f32, centrifugal: bool) -> (usize, i64) {
    let scenario = beam(9, 0.6, 6.0);
    let settings = SolverSettings {
        compression_elastic_limit: fatal*0.5, compression_fatal_limit: fatal,
        tension_elastic_limit: 1.0e6, tension_fatal_limit: 1.0e7,
        shear_elastic_limit: 1.0e6, shear_fatal_limit: 1.0e7, ..SolverSettings::default()
    };
    let policy = FracturePolicy { idle_skip: false, apply_centrifugal: centrifugal, ..FracturePolicy::default() };
    let mut set = DestructibleSet::from_scenario(&scenario, settings, Vec3::new(0.0,0.0,0.0), policy).unwrap();
    let (mut b, mut c, mut isl, mut ij, mut mj) = (RigidBodySet::new(), ColliderSet::new(), IslandManager::new(), ImpulseJointSet::new(), MultibodyJointSet::new());
    set.initialize(&mut b, &mut c);
    let bonds0 = set.active_bond_count() as i64;
    let mut fr = 0usize;
    for _ in 0..150 {
        for (_, body) in b.iter_mut() { if body.is_dynamic() { body.set_angular_damping(0.0); body.set_angvel(vector![0.0, omega, 0.0], true); } }
        fr += set.step(&mut b, &mut c, &mut isl, &mut ij, &mut mj).fractures;
    }
    (fr, bonds0 - set.active_bond_count() as i64)
}

#[test]
fn sweep() {
    eprintln!("baseline OFF (spin 120, fatal 1): {:?}", run(120.0, 1.0, false));
    for &omega in &[1.0f32, 5.0, 20.0, 60.0, 120.0] {
        for &fatal in &[0.1f32, 1.0, 10.0, 100.0, 1000.0] {
            let (fr, broke) = run(omega, fatal, true);
            eprintln!("omega={omega:>5} fatal={fatal:>8} -> fractures={fr:>3} bondsBroken={broke}");
        }
    }
}
