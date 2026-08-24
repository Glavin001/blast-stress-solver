//! Settle and wake edges come from the engine's own sleep state.
//!
//! The alternative that was on offer -- force-sleeping a body a fixed number of
//! ticks after promotion -- is not tested here because it is not implemented.
//! It declares "at rest" about a body that is still moving, so the consumer
//! stops updating something the physics is still integrating. What is asserted
//! instead is the property that makes the engine edge trustworthy: a body only
//! settles once it has actually stopped, and an impulse un-settles it.
#![cfg(all(feature = "rapier", feature = "scenarios", feature = "physx"))]

use std::collections::HashMap;

use blast_stress_solver::backend::{
    BodyKind, CommandBuffer, CommandResults, CreateBody, CreateShape, PhysicsBackend, Phase, Pose,
    ShapeGeom,
};
use blast_stress_solver::backends::{PhysXWorld, RapierWorld};
use blast_stress_solver::pipeline::{
    DestructionEvent, Destructible, DestructibleConfig, IslandSerial,
};
use blast_stress_solver::scenarios::{build_wall_scenario, WallOptions};
use blast_stress_solver::types::{ScenarioDesc, SolverSettings, Vec3};

const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);
const DT: f32 = 1.0 / 60.0;

fn scenario() -> ScenarioDesc {
    build_wall_scenario(&WallOptions::default())
}

/// A floor, without which nothing can settle.
///
/// The first version of this test had no ground and asserted that rubble comes
/// to rest. Rapier reported a single settle -- the anchored island, which never
/// moved -- and PhysX reported none, so the test looked like a PhysX adapter
/// bug. It was not: with nothing to land on, the fragments were still
/// accelerating downward at tick 600 and neither engine was wrong to keep them
/// awake. Worth recording, because "the backend never sleeps" and "the scene
/// has no floor" produce the same symptom.
fn add_ground<B: PhysicsBackend>(backend: &mut B) {
    let mut cmds: CommandBuffer<B::BodyId, B::ShapeId> = CommandBuffer::new();
    let mut out: CommandResults<B::BodyId, B::ShapeId> = CommandResults::default();
    cmds.create_bodies.push(CreateBody {
        pose: Pose::from_translation(Vec3::new(0.0, -1.0, 0.0)),
        kind: BodyKind::Fixed,
        linvel: Vec3::ZERO,
        angvel: Vec3::ZERO,
        ccd: false,
        start_sleeping: false,
    });
    backend.apply(Phase::Topology, &cmds, &mut out).expect("ground body");
    let ground = out.created_bodies[0];
    cmds.clear();
    cmds.create_shapes.push(CreateShape {
        body: ground,
        local: Pose::IDENTITY,
        geom: ShapeGeom::Cuboid { half_extents: Vec3::new(200.0, 1.0, 200.0) },
        mass: 0.0,
        node: 0,
    });
    backend.apply(Phase::Topology, &cmds, &mut out).expect("ground shape");
}

fn config() -> DestructibleConfig {
    DestructibleConfig {
        gravity: G,
        solver: SolverSettings {
            max_solver_iterations_per_frame: 24,
            compression_elastic_limit: 5.0,
            compression_fatal_limit: 10.0,
            tension_elastic_limit: 5.0,
            tension_fatal_limit: 10.0,
            shear_elastic_limit: 5.0,
            shear_fatal_limit: 10.0,
            ..SolverSettings::default()
        },
        min_child_nodes: 1,
        max_new_bodies_per_step: usize::MAX,
        ..Default::default()
    }
}

struct Trace {
    settled: Vec<IslandSerial>,
    woke: Vec<IslandSerial>,
    /// Sleep level sampled per tick, for cross-checking the edges against it.
    level: HashMap<IslandSerial, bool>,
    /// Edges that disagreed with the sampled level.
    inconsistencies: Vec<String>,
}

fn run<B: PhysicsBackend>(backend: &mut B, steps: usize) -> Trace {
    add_ground(backend);
    let mut d = Destructible::attach(backend, &scenario(), config()).expect("attach");
    let _ = d.drain_events();
    let mut t = Trace {
        settled: Vec::new(),
        woke: Vec::new(),
        level: HashMap::new(),
        inconsistencies: Vec::new(),
    };
    for _ in 0..steps {
        backend.step(DT);
        d.step(backend, DT);
        for e in d.drain_events() {
            match e {
                DestructionEvent::IslandSettled { serial } => t.settled.push(serial),
                DestructionEvent::IslandWoke { serial } => t.woke.push(serial),
                _ => {}
            }
        }
        // The edge must agree with the level the pipeline itself reports.
        for m in d.island_poses(backend) {
            t.level.insert(m.serial, m.sleeping);
        }
    }
    // Anything that reported settled and is now awake must have reported a wake.
    for s in &t.settled {
        if t.level.get(s) == Some(&false) && !t.woke.contains(s) {
            t.inconsistencies
                .push(format!("{s:?} settled, is awake, and never reported waking"));
        }
    }
    t
}

fn check(engine: &str, t: &Trace) {
    assert!(
        t.inconsistencies.is_empty(),
        "[{engine}] edges disagree with the sampled level: {:#?}",
        t.inconsistencies
    );
    assert!(
        !t.settled.is_empty(),
        "[{engine}] nothing ever settled, so this run never exercised the edge"
    );
    // Edges, not levels: an island cannot report settling twice without waking
    // in between, which is exactly the bug a level-triggered implementation has.
    let mut open: HashMap<IslandSerial, bool> = HashMap::new();
    for s in &t.settled {
        assert!(
            open.insert(*s, true).is_none() || t.woke.contains(s),
            "[{engine}] island {s:?} settled twice without waking -- that is a \
             level being reported as an edge"
        );
    }
    println!("[{engine}] {} settle edges, {} wake edges", t.settled.len(), t.woke.len());
}

#[test]
fn rubble_settles_and_reports_the_edge_on_rapier() {
    let mut w = RapierWorld::new(G);
    check("rapier", &run(&mut w, 600));
}

#[test]
fn rubble_settles_and_reports_the_edge_on_physx() {
    let mut w = PhysXWorld::new_cpu(G, 2).expect("physx cpu world");
    check("physx-cpu", &run(&mut w, 600));
}

/// The first observation of an island is a level, not an edge.
///
/// An anchored island is asleep from the moment it exists. Reporting that as a
/// settle would fire a spurious edge for every support actor on tick one, and a
/// consumer would record a settle for something that never moved.
#[test]
fn a_first_observation_is_never_reported_as_an_edge() {
    let mut w = RapierWorld::new(G);
    add_ground(&mut w);
    let mut d = Destructible::attach(&mut w, &scenario(), config()).expect("attach");
    let _ = d.drain_events();
    w.step(DT);
    d.step(&mut w, DT);
    let first: Vec<_> = d
        .drain_events()
        .into_iter()
        .filter(|e| {
            matches!(
                e,
                DestructionEvent::IslandSettled { .. } | DestructionEvent::IslandWoke { .. }
            )
        })
        .collect();
    assert!(
        first.is_empty(),
        "the first tick reported sleep edges for islands never previously observed: {first:#?}"
    );
}

/// Settling is not terminal: something that gets hit must report waking.
///
/// Without this the settle edge is only half a contract. A consumer that stops
/// sending updates on settle and is never told about the wake leaves the island
/// frozen on screen while the server keeps simulating it -- and no aggregate
/// counter would show it, because the totals stay right.
fn disturb_and_expect_wake<B: PhysicsBackend>(engine: &str, backend: &mut B) {
    add_ground(backend);
    let mut d = Destructible::attach(backend, &scenario(), config()).expect("attach");

    // Anchored islands are excluded by construction, not by luck: a fixed body
    // reports sleeping forever and can never wake, so kicking one would fail
    // for a reason that has nothing to do with the edge under test.
    let mut anchored: Vec<IslandSerial> = Vec::new();
    let mut note = |events: Vec<DestructionEvent>, settled: &mut Vec<IslandSerial>| {
        for e in events {
            match e {
                DestructionEvent::IslandPromoted { serial, anchored: a, .. } if a => {
                    anchored.push(serial)
                }
                DestructionEvent::IslandSettled { serial } => settled.push(serial),
                _ => {}
            }
        }
    };
    let mut settled: Vec<IslandSerial> = Vec::new();
    note(d.drain_events(), &mut settled);
    settled.clear();

    for _ in 0..600 {
        backend.step(DT);
        d.step(backend, DT);
        note(d.drain_events(), &mut settled);
    }
    assert!(!settled.is_empty(), "[{engine}] nothing settled to disturb");

    let target = d
        .island_poses(backend)
        .into_iter()
        .find(|m| m.sleeping && settled.contains(&m.serial) && !anchored.contains(&m.serial))
        .unwrap_or_else(|| panic!("[{engine}] no sleeping unanchored settled island to kick"));
    let body = d
        .bodies()
        .into_iter()
        .find(|b| d.island_of(b) == Some(target.serial))
        .expect("body for island");

    // Velocity *and* an explicit wake. Writing a property deliberately does not
    // wake a sleeping body -- re-stamping every body each tick is what once
    // held ~600 of ~735 bodies permanently awake -- so a host disturbing an
    // island asks for the wake rather than getting it as a side effect.
    let mut cmds: CommandBuffer<B::BodyId, B::ShapeId> = CommandBuffer::new();
    let mut out: CommandResults<B::BodyId, B::ShapeId> = CommandResults::default();
    cmds.set_velocity.push((body, Vec3::new(0.0, 6.0, 0.0), Vec3::ZERO));
    cmds.wake.push(body);
    backend.apply(Phase::Motion, &cmds, &mut out).expect("kick");

    let mut woke = false;
    for _ in 0..120 {
        backend.step(DT);
        d.step(backend, DT);
        for e in d.drain_events() {
            if e == (DestructionEvent::IslandWoke { serial: target.serial }) {
                woke = true;
            }
        }
        if woke {
            break;
        }
    }
    assert!(
        woke,
        "[{engine}] island {:?} was kicked and never reported waking",
        target.serial
    );
    println!("[{engine}] wake edge reported for {:?}", target.serial);
}

#[test]
fn a_settled_island_that_is_disturbed_reports_waking_on_rapier() {
    disturb_and_expect_wake("rapier", &mut RapierWorld::new(G));
}

#[test]
fn a_settled_island_that_is_disturbed_reports_waking_on_physx() {
    disturb_and_expect_wake("physx-cpu", &mut PhysXWorld::new_cpu(G, 2).expect("physx cpu world"));
}
