//! The native event stream must be enough, on its own, to track the structure.
//!
//! The bar these tests set is deliberately the one the old snapshot-diff
//! approach could not meet: a consumer that sees *only* the events -- never the
//! physics scene -- must end up with the same island membership and the same
//! chunk world positions the pipeline has. If that holds, the diff is not
//! compensating for a gap and can be deleted rather than kept as an oracle.
#![cfg(all(feature = "rapier", feature = "scenarios", feature = "physx"))]

use std::collections::{HashMap, HashSet};

use blast_stress_solver::backend::{PhysicsBackend, Pose};
use blast_stress_solver::backends::{PhysXWorld, RapierWorld};
use blast_stress_solver::pipeline::{
    DestructionEvent, Destructible, DestructibleConfig, IslandMotion, IslandSerial,
};
use blast_stress_solver::scenarios::{build_wall_scenario, WallOptions};
use blast_stress_solver::types::{ScenarioDesc, SolverSettings, Vec3};

const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn scenario() -> ScenarioDesc {
    build_wall_scenario(&WallOptions::default())
}

/// Weak enough that gravity alone shatters it, so the split path really runs.
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

/// A consumer that knows nothing but the event stream.
///
/// This is the whole point: it never touches the backend, so anything it gets
/// right is something the stream genuinely carried.
#[derive(Default)]
struct Ledger {
    island_of: HashMap<u32, IslandSerial>,
    members: HashMap<IslandSerial, HashSet<u32>>,
    pose: HashMap<IslandSerial, Pose>,
    offset: HashMap<(IslandSerial, u32), Vec3>,
    live: HashSet<IslandSerial>,
    retired: HashSet<IslandSerial>,
    broken_bonds: usize,
    /// Every violation of the stream's own ordering contract.
    order_violations: Vec<String>,
}

impl Ledger {
    fn apply(&mut self, events: &[DestructionEvent]) {
        for e in events {
            match e {
                DestructionEvent::BondBroken { .. } => self.broken_bonds += 1,
                DestructionEvent::IslandPromoted {
                    serial,
                    pose,
                    members,
                    provenance,
                    ..
                } => {
                    if self.retired.contains(serial) {
                        self.order_violations
                            .push(format!("serial {serial:?} promoted after retirement"));
                    }
                    if !self.live.insert(*serial) {
                        self.order_violations
                            .push(format!("serial {serial:?} promoted twice"));
                    }
                    if *provenance != IslandSerial::NONE
                        && !self.live.contains(provenance)
                        && !self.retired.contains(provenance)
                    {
                        self.order_violations
                            .push(format!("promotion cites unknown provenance {provenance:?}"));
                    }
                    self.pose.insert(*serial, *pose);
                    self.members.entry(*serial).or_default();
                    for m in members {
                        self.offset.insert((*serial, m.chunk), m.offset);
                    }
                }
                DestructionEvent::ChunkMigrated { chunk, from, to } => {
                    if !self.live.contains(to) {
                        self.order_violations
                            .push(format!("chunk {chunk} migrated onto unknown island {to:?}"));
                    }
                    if *from != IslandSerial::NONE {
                        if let Some(set) = self.members.get_mut(from) {
                            set.remove(chunk);
                        }
                        if self.island_of.get(chunk) != Some(from) {
                            self.order_violations.push(format!(
                                "chunk {chunk} said to leave {from:?} but ledger had {:?}",
                                self.island_of.get(chunk)
                            ));
                        }
                    }
                    self.members.entry(*to).or_default().insert(*chunk);
                    self.island_of.insert(*chunk, *to);
                }
                DestructionEvent::IslandRecomposed { serial, members, .. } => {
                    if !self.live.contains(serial) {
                        self.order_violations
                            .push(format!("island {serial:?} recomposed while not live"));
                    }
                    // Authoritative: replace rather than patch, so a consumer
                    // that joins mid-stream converges instead of accumulating
                    // whatever it happened to miss.
                    let fresh: HashSet<u32> = members.iter().map(|m| m.chunk).collect();
                    self.members.insert(*serial, fresh);
                    for m in members {
                        self.offset.insert((*serial, m.chunk), m.offset);
                        self.island_of.insert(m.chunk, *serial);
                    }
                }
                DestructionEvent::IslandRetired { serial } => {
                    if let Some(set) = self.members.get(serial) {
                        if !set.is_empty() {
                            self.order_violations.push(format!(
                                "island {serial:?} retired still holding {} chunks",
                                set.len()
                            ));
                        }
                    }
                    self.live.remove(serial);
                    self.retired.insert(*serial);
                }
                DestructionEvent::IslandSettled { .. }
                | DestructionEvent::ChunkDestroyed { .. } => {}
            }
        }
    }

    /// Take the per-tick motion sample, exactly as a netcode layer would.
    ///
    /// Events carry topology; motion is sampled. A consumer that only ever
    /// replayed promotion poses would draw every island frozen where it broke.
    fn observe(&mut self, motions: &[IslandMotion]) {
        for m in motions {
            assert!(
                self.live.contains(&m.serial),
                "sampled island {:?} was never promoted",
                m.serial
            );
            self.pose.insert(m.serial, m.pose);
        }
    }

    /// Where the ledger believes a chunk is, from the COM-frame rule alone.
    fn chunk_world(&self, chunk: u32) -> Option<Vec3> {
        let island = *self.island_of.get(&chunk)?;
        let pose = *self.pose.get(&island)?;
        let offset = *self.offset.get(&(island, chunk))?;
        Some(pose.translation + pose.rotation.rotate(offset))
    }
}

fn run<B: PhysicsBackend>(backend: &mut B, steps: usize) -> (Ledger, Destructible<B>) {
    let mut d = Destructible::attach(backend, &scenario(), config()).expect("attach");
    let mut ledger = Ledger::default();
    ledger.apply(&d.drain_events());
    ledger.observe(&d.island_poses(backend));
    for _ in 0..steps {
        backend.step(1.0 / 60.0);
        d.step(backend, 1.0 / 60.0);
        // Topology first, then motion -- the order a consumer must use, since a
        // sample naming an island it has not been told about is unusable.
        ledger.apply(&d.drain_events());
        ledger.observe(&d.island_poses(backend));
    }
    (ledger, d)
}

fn check(engine: &str, ledger: &Ledger, expected_nodes: usize) {
    assert!(
        ledger.order_violations.is_empty(),
        "[{engine}] the stream broke its own ordering contract: {:#?}",
        ledger.order_violations
    );
    assert!(
        ledger.broken_bonds > 0,
        "[{engine}] no bonds broke, so this case never exercised the split path"
    );

    // Every chunk is on exactly one live island, and the islands partition the
    // chunks -- the invariant a diff-based reconstruction is worst at, because
    // it can only ever see the endpoints of a change.
    assert_eq!(
        ledger.island_of.len(),
        expected_nodes,
        "[{engine}] every node must be placed"
    );
    let mut seen = HashSet::new();
    for (serial, members) in &ledger.members {
        if members.is_empty() {
            continue;
        }
        assert!(
            ledger.live.contains(serial),
            "[{engine}] retired island {serial:?} still holds {} chunks",
            members.len()
        );
        for chunk in members {
            assert!(
                seen.insert(*chunk),
                "[{engine}] chunk {chunk} is on two islands at once"
            );
            assert_eq!(
                ledger.island_of.get(chunk),
                Some(serial),
                "[{engine}] membership and island_of disagree for chunk {chunk}"
            );
        }
    }
    assert_eq!(seen.len(), expected_nodes, "[{engine}] islands must partition the chunks");
    assert!(
        ledger.live.is_disjoint(&ledger.retired),
        "[{engine}] a serial is both live and retired -- serials were reused"
    );
}

#[test]
fn the_event_stream_alone_tracks_every_chunk_on_rapier() {
    let mut w = RapierWorld::new(G);
    let (ledger, d) = run(&mut w, 90);
    check("rapier", &ledger, d.node_count());
}

#[test]
fn the_event_stream_alone_tracks_every_chunk_on_physx() {
    let mut w = PhysXWorld::new_cpu(G, 2).expect("physx cpu world");
    let (ledger, d) = run(&mut w, 90);
    check("physx-cpu", &ledger, d.node_count());
}

/// The reason poses are COM-frame rather than actor-frame.
///
/// A reused body keeps its parent's origin while its COM moves into
/// `centerOfMassLocalPose`, so a consumer handed the raw actor frame draws that
/// island's chunks displaced by exactly the COM offset. Reconstructing from the
/// stream and comparing against the pipeline's own node positions is what
/// catches that -- an aggregate counter never can.
#[test]
fn reconstructed_chunk_positions_match_the_pipeline() {
    for (engine, positions, ledger) in [
        {
            let mut w = RapierWorld::new(G);
            let (l, d) = run(&mut w, 90);
            ("rapier", d.node_world_positions(&w), l)
        },
        {
            let mut w = PhysXWorld::new_cpu(G, 2).expect("physx cpu world");
            let (l, d) = run(&mut w, 90);
            ("physx-cpu", d.node_world_positions(&w), l)
        },
    ] {
        let mut worst = 0.0f32;
        let mut worst_chunk = u32::MAX;
        for (node, truth) in positions.iter().enumerate() {
            let Some(truth) = truth else { continue };
            let got = ledger
                .chunk_world(node as u32)
                .unwrap_or_else(|| panic!("[{engine}] chunk {node} never placed by the stream"));
            let err = (got - *truth).magnitude();
            if err > worst {
                worst = err;
                worst_chunk = node as u32;
            }
        }
        // 1 mm, the same band `QualityFingerprint` uses for positions. This is
        // a pure algebraic reconstruction, so the only error is f32 rounding in
        // composing the COM offset -- not integration drift.
        assert!(
            worst < 1.0e-3,
            "[{engine}] reconstruction drifted {worst} m at chunk {worst_chunk}; \
             a COM-frame error shows up here as a constant per-island offset"
        );
        println!("[{engine}] worst reconstruction error {worst:.3e} m");
    }
}
