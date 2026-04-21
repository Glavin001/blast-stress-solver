#![cfg(all(feature = "rapier", feature = "scenarios"))]

use blast_stress_solver::rapier::*;
use blast_stress_solver::scenarios::*;
use blast_stress_solver::*;
use rapier3d::prelude::*;

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

fn node_world_position(
    set: &DestructibleSet,
    node: u32,
    bodies: &RigidBodySet,
    colliders: &ColliderSet,
) -> Vec3 {
    if let Some(collider) = set.node_collider(node) {
        if let Some(c) = colliders.get(collider) {
            let t = c.position().translation.vector;
            return Vec3::new(t.x, t.y, t.z);
        }
    }

    let body = bodies
        .get(set.node_body(node).expect("node should map to body"))
        .expect("body should exist");
    let t = body.position().translation.vector;
    Vec3::new(t.x, t.y, t.z)
}

fn node_world_velocity(
    set: &DestructibleSet,
    node: u32,
    bodies: &RigidBodySet,
    colliders: &ColliderSet,
) -> Vec3 {
    let body_handle = set.node_body(node).expect("node should map to body");
    let body = bodies.get(body_handle).expect("body should exist");
    let p = node_world_position(set, node, bodies, colliders);
    let com = body.position().translation.vector;
    let r = vector![p.x - com.x, p.y - com.y, p.z - com.z];
    let v = *body.linvel() + body.angvel().cross(&r);
    Vec3::new(v.x, v.y, v.z)
}

fn step_until_first_split(
    set: &mut DestructibleSet,
    bodies: &mut RigidBodySet,
    colliders: &mut ColliderSet,
    island_manager: &mut IslandManager,
    impulse_joints: &mut ImpulseJointSet,
    multibody_joints: &mut MultibodyJointSet,
) -> StepResult {
    for _ in 0..8 {
        let step = set.step(
            bodies,
            colliders,
            island_manager,
            impulse_joints,
            multibody_joints,
        );
        if step.split_events > 0 {
            return step;
        }
    }
    panic!("expected split event within budgeted steps");
}

fn max_node_displacement(before: &[Vec3], after: &[Vec3]) -> f32 {
    before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (*b - *a).magnitude())
        .fold(0.0f32, f32::max)
}

fn wall_split_handoff_fixture() -> (ScenarioDesc, DestructibleSet) {
    let scenario = build_wall_scenario(&WallOptions {
        span_segments: 3,
        height_segments: 2,
        layers: 1,
        ..WallOptions::default()
    });

    let set = DestructibleSet::from_scenario(
        &scenario,
        weak_settings(),
        Vec3::ZERO,
        FracturePolicy {
            idle_skip: false,
            apply_excess_forces: false,
            ..FracturePolicy::default()
        },
    )
    .expect("scenario should initialize");

    (scenario, set)
}

#[test]
fn split_with_recentering_disabled_keeps_node_world_positions_continuous() {
    let (scenario, mut set) = wall_split_handoff_fixture();
    set.set_split_child_recentering_enabled(false);
    set.set_split_child_velocity_fit_enabled(false);

    let (mut bodies, mut colliders, mut island_manager, mut impulse_joints, mut multibody_joints) =
        rapier_world();
    set.initialize(&mut bodies, &mut colliders);

    let target = (scenario.nodes.len() / 2) as u32;
    let pos = scenario.nodes[target as usize].centroid;
    set.add_force(target, pos, Vec3::new(20_000.0, 0.0, 0.0));

    let before: Vec<Vec3> = (0..scenario.nodes.len() as u32)
        .map(|n| node_world_position(&set, n, &bodies, &colliders))
        .collect();

    let split_step = step_until_first_split(
        &mut set,
        &mut bodies,
        &mut colliders,
        &mut island_manager,
        &mut impulse_joints,
        &mut multibody_joints,
    );
    assert!(split_step.fractures > 0);

    let after: Vec<Vec3> = (0..scenario.nodes.len() as u32)
        .map(|n| node_world_position(&set, n, &bodies, &colliders))
        .collect();

    let max_delta = max_node_displacement(&before, &after);
    assert!(
        max_delta < 1.0e-3,
        "node teleported during split handoff in no-recenter mode: max |Δx|={max_delta}"
    );
}

#[test]
fn split_with_default_recentering_should_keep_node_world_positions_continuous() {
    let (scenario, mut set) = wall_split_handoff_fixture();

    let (mut bodies, mut colliders, mut island_manager, mut impulse_joints, mut multibody_joints) =
        rapier_world();
    set.initialize(&mut bodies, &mut colliders);

    let target = (scenario.nodes.len() / 2) as u32;
    let pos = scenario.nodes[target as usize].centroid;
    set.add_force(target, pos, Vec3::new(20_000.0, 0.0, 0.0));

    let before: Vec<Vec3> = (0..scenario.nodes.len() as u32)
        .map(|n| node_world_position(&set, n, &bodies, &colliders))
        .collect();

    step_until_first_split(
        &mut set,
        &mut bodies,
        &mut colliders,
        &mut island_manager,
        &mut impulse_joints,
        &mut multibody_joints,
    );

    let after: Vec<Vec3> = (0..scenario.nodes.len() as u32)
        .map(|n| node_world_position(&set, n, &bodies, &colliders))
        .collect();

    let max_delta = max_node_displacement(&before, &after);
    assert!(
        max_delta < 1.0e-3,
        "default split handoff should be continuous: max |Δx|={max_delta}"
    );
}

#[test]
fn split_preserves_total_linear_momentum() {
    let scenario = build_wall_scenario(&WallOptions {
        span_segments: 3,
        height_segments: 2,
        layers: 1,
        ..WallOptions::default()
    });

    let mut set = DestructibleSet::from_scenario(
        &scenario,
        weak_settings(),
        Vec3::ZERO,
        FracturePolicy {
            idle_skip: false,
            apply_excess_forces: false,
            ..FracturePolicy::default()
        },
    )
    .expect("scenario should initialize");

    let (mut bodies, mut colliders, mut island_manager, mut impulse_joints, mut multibody_joints) =
        rapier_world();
    set.initialize(&mut bodies, &mut colliders);

    let parent = set
        .node_body((scenario.nodes.len() / 2) as u32)
        .expect("dynamic wall should have body");
    {
        let body = bodies.get_mut(parent).unwrap();
        body.set_linvel(vector![2.0, -1.0, 0.25], true);
        body.set_angvel(vector![0.0, 0.0, 4.0], true);
    }

    let target = (scenario.nodes.len() / 2) as u32;
    let pos = scenario.nodes[target as usize].centroid;
    set.add_force(target, pos, Vec3::new(15_000.0, 0.0, 0.0));

    let momentum_before = scenario
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.mass > 0.0)
        .fold(Vec3::ZERO, |acc, (i, node)| {
            acc + node_world_velocity(&set, i as u32, &bodies, &colliders) * node.mass
        });

    step_until_first_split(
        &mut set,
        &mut bodies,
        &mut colliders,
        &mut island_manager,
        &mut impulse_joints,
        &mut multibody_joints,
    );

    let momentum_after = scenario
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.mass > 0.0)
        .fold(Vec3::ZERO, |acc, (i, node)| {
            acc + node_world_velocity(&set, i as u32, &bodies, &colliders) * node.mass
        });

    let diff = (momentum_after - momentum_before).magnitude();
    assert!(
        diff < 1.0e-2,
        "split should conserve aggregate linear momentum: |Δp|={diff}"
    );
}

fn run_gravity_breakage(scenario: &ScenarioDesc, limit: f32) -> usize {
    let settings = SolverSettings {
        max_solver_iterations_per_frame: 48,
        compression_elastic_limit: limit,
        compression_fatal_limit: limit * 2.0,
        tension_elastic_limit: limit,
        tension_fatal_limit: limit * 2.0,
        shear_elastic_limit: limit,
        shear_fatal_limit: limit * 2.0,
        ..SolverSettings::default()
    };

    let mut set = DestructibleSet::from_scenario(
        scenario,
        settings,
        Vec3::new(0.0, -9.81, 0.0),
        FracturePolicy {
            idle_skip: false,
            ..FracturePolicy::default()
        },
    )
    .expect("scenario should initialize");

    let (mut bodies, mut colliders, mut island_manager, mut impulse_joints, mut multibody_joints) =
        rapier_world();
    set.initialize(&mut bodies, &mut colliders);

    let mut total_fractures = 0;
    for _ in 0..40 {
        let step = set.step(
            &mut bodies,
            &mut colliders,
            &mut island_manager,
            &mut impulse_joints,
            &mut multibody_joints,
        );
        total_fractures += step.fractures;
    }

    total_fractures
}

#[test]
fn wall_tower_bridge_strength_sweep_distinguishes_weak_vs_strong_material() {
    let scenarios = [
        ("wall", build_wall_scenario(&WallOptions::default())),
        (
            "tower",
            build_tower_scenario(&TowerOptions {
                side: 3,
                stories: 4,
                ..TowerOptions::default()
            }),
        ),
        (
            "bridge",
            build_bridge_scenario(&BridgeOptions {
                span_segments: 10,
                width_segments: 2,
                thickness_layers: 1,
                ..BridgeOptions::default()
            }),
        ),
    ];

    for (name, scenario) in scenarios {
        let weak = run_gravity_breakage(&scenario, 0.05);
        let strong = run_gravity_breakage(&scenario, 5_000.0);
        assert!(
            weak > strong,
            "{name}: weak material should fracture more than very strong material ({weak} vs {strong})"
        );
    }
}
