//! Rapier-call cost profiler — measures the *true* marginal cost of a world topology
//! change on a raw Rapier world (no DestructibleSet, no stress solver — pure Rapier).
//!
//! The point micro-benchmarks miss: mutations like `insert` / `set_parent` /
//! `set_body_type` do little work synchronously — Rapier *defers* the real cost (collider
//! AABB / broad-phase update, island re-eval, mass recompute, contact-pair creation) to
//! the next `step()`. So the honest marginal cost is `(mutate + one step) - (one step)`.
//!
//! We time the step with `Instant` (Rapier's own `counters.*_time_ms()` are no-ops unless
//! the crate is built with the `profiler` feature) and report load metrics — awake bodies
//! and contact/constraint counts — that explain where the time goes (cross-referenced with
//! the engine's stage model: solver/island scale with AWAKE bodies; broad/narrow with the
//! CHANGED set; sleeping islands cost ~nothing).
//!
//! Run: `cargo run --release --example rapier_cost_profile --features bench-support`

use std::time::Instant;

use rapier3d::prelude::*;

struct World {
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad: BroadPhaseBvh,
    narrow: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse: ImpulseJointSet,
    multibody: MultibodyJointSet,
    ccd: CCDSolver,
    gravity: Vector<Real>,
    ip: IntegrationParameters,
}

impl World {
    fn new() -> Self {
        World {
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad: BroadPhaseBvh::new(),
            narrow: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse: ImpulseJointSet::new(),
            multibody: MultibodyJointSet::new(),
            ccd: CCDSolver::new(),
            gravity: vector![0.0, -9.81, 0.0],
            ip: IntegrationParameters::default(),
        }
    }
    fn step(&mut self) {
        self.pipeline.step(
            &self.gravity, &self.ip, &mut self.islands, &mut self.broad, &mut self.narrow,
            &mut self.bodies, &mut self.colliders, &mut self.impulse, &mut self.multibody,
            &mut self.ccd, &(), &(),
        );
    }
    /// One step, wall-clock-timed.
    fn timed_step(&mut self) -> f64 {
        let t = Instant::now();
        self.step();
        t.elapsed().as_secs_f64() * 1000.0
    }
    /// Median total-step ms over `n` consecutive steps (for repeatable, no-mutation cases).
    fn median_step_ms(&mut self, n: usize) -> f64 {
        let mut v: Vec<f64> = (0..n).map(|_| self.timed_step()).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
    fn awake(&self) -> usize {
        self.bodies.iter().filter(|(_, b)| b.is_dynamic() && !b.is_sleeping()).count()
    }
    fn contact_pairs(&self) -> usize {
        self.narrow.contact_pairs().count()
    }
}

/// `side`×`side` columns × `layers` cuboids on a fixed floor, stepped until (mostly) asleep.
/// Returns the world, the floor handle, and the dynamic body handles.
fn settled_stack(side: u32, layers: u32) -> (World, RigidBodyHandle, Vec<RigidBodyHandle>) {
    let mut w = World::new();
    let floor = w.bodies.insert(RigidBodyBuilder::fixed().translation(vector![0.0, -0.5, 0.0]));
    w.colliders.insert_with_parent(
        ColliderBuilder::cuboid(side as f32 + 4.0, 0.5, side as f32 + 4.0),
        floor,
        &mut w.bodies,
    );
    let mut handles = Vec::new();
    for x in 0..side {
        for z in 0..side {
            for y in 0..layers {
                let h = w.bodies.insert(RigidBodyBuilder::dynamic().translation(vector![
                    x as f32 * 1.02, 0.5 + y as f32 * 1.02, z as f32 * 1.02
                ]));
                w.colliders.insert_with_parent(
                    ColliderBuilder::cuboid(0.5, 0.5, 0.5).density(1.0), h, &mut w.bodies);
                handles.push(h);
            }
        }
    }
    for _ in 0..250 {
        w.step();
    }
    (w, floor, handles)
}

fn main() {
    println!("Rapier-call cost profiler (raw Rapier, Instant-timed) — times in ms\n");

    // ── 1. Settled-step cost: ASLEEP vs ALL-AWAKE (the total-performance lever) ──────
    println!("[1] Per-step cost of a settled world — asleep vs all-awake");
    println!("    (solver + island stages scale with AWAKE bodies; sleeping islands ~free)");
    for (side, layers) in [(16u32, 4u32), (24, 4)] {
        let (mut w, _floor, handles) = settled_stack(side, layers);
        let n = handles.len();
        let asleep_ms = w.median_step_ms(15);
        println!("  N={n:<5} asleep: step={asleep_ms:.3} ms  (awake={}, contact_pairs={})",
            w.awake(), w.contact_pairs());
        for &h in &handles {
            if let Some(b) = w.bodies.get_mut(h) { b.wake_up(true); }
        }
        let awake_ms = w.median_step_ms(15);
        println!("  N={n:<5} awake:  step={awake_ms:.3} ms  (awake={}, contact_pairs={})  -> {:.1}× costlier\n",
            w.awake(), w.contact_pairs(), awake_ms / asleep_ms.max(1e-6));
    }

    // ── 2. Marginal cost of INSERTING bodies (batched: K inserts, then ONE step) ─────
    println!("[2] Insert K=512 dynamic bodies+colliders into settled N=1024, then 1 step");
    {
        let (mut w, _floor, _) = settled_stack(16, 4);
        let baseline = w.median_step_ms(15);
        let k = 512u32;
        let t = Instant::now();
        for i in 0..k {
            let h = w.bodies.insert(RigidBodyBuilder::dynamic().translation(vector![
                (i % 32) as f32, 40.0 + (i / 32) as f32, (i % 7) as f32]));
            w.colliders.insert_with_parent(
                ColliderBuilder::cuboid(0.5, 0.5, 0.5).density(1.0), h, &mut w.bodies);
        }
        let call_ms = t.elapsed().as_secs_f64() * 1000.0;
        let step_after = w.timed_step();
        println!("  insert calls: {call_ms:.3} ms ({:.4} ms/body)", call_ms / k as f64);
        println!("  baseline step: {baseline:.3} ms  ->  step after inserts: {step_after:.3} ms");
        println!("  marginal step cost of the inserts: {:.3} ms ({:.4} ms/body)\n",
            step_after - baseline, (step_after - baseline) / k as f64);
    }

    // ── 3. Re-parent K colliders onto a fresh body at the SAME spot (the fracture
    //       pattern: a node's collider moves to its new fragment body) — set_parent vs
    //       remove+reinsert. Positions preserved, so no contact explosion. ────────────
    println!("[3] Move K=512 colliders to fresh per-collider bodies: set_parent vs remove+reinsert");
    {
        // (a) set_parent — keeps the same broad-phase leaf + contact-graph node
        let (mut w, _floor, handles) = settled_stack(16, 4);
        let base_a = w.median_step_ms(10);
        let cols: Vec<ColliderHandle> = handles.iter().take(512)
            .filter_map(|&h| w.bodies.get(h).and_then(|b| b.colliders().first().copied())).collect();
        let t = Instant::now();
        for &c in &cols {
            let p = *w.colliders[c].position();
            let nb = w.bodies.insert(RigidBodyBuilder::dynamic().position(p));
            w.colliders.set_parent(c, Some(nb), &mut w.bodies);
            w.colliders[c].set_position_wrt_parent(Isometry::identity());
        }
        let call_a = t.elapsed().as_secs_f64() * 1000.0;
        let step_a = w.timed_step();
        println!("  set_parent:      call={call_a:.3} ms  step_after={step_a:.3} ms  (Δstep={:.3})",
            step_a - base_a);

        // (b) remove + reinsert — leaf removal + new leaf + contact-graph churn + wakes
        let (mut w, _floor, handles) = settled_stack(16, 4);
        let base_b = w.median_step_ms(10);
        let cols: Vec<ColliderHandle> = handles.iter().take(512)
            .filter_map(|&h| w.bodies.get(h).and_then(|b| b.colliders().first().copied())).collect();
        let t = Instant::now();
        for &c in &cols {
            let p = *w.colliders[c].position();
            let nb = w.bodies.insert(RigidBodyBuilder::dynamic().position(p));
            w.colliders.remove(c, &mut w.islands, &mut w.bodies, true);
            w.colliders.insert_with_parent(
                ColliderBuilder::cuboid(0.5, 0.5, 0.5).density(1.0), nb, &mut w.bodies);
        }
        let call_b = t.elapsed().as_secs_f64() * 1000.0;
        let step_b = w.timed_step();
        println!("  remove+reinsert: call={call_b:.3} ms  step_after={step_b:.3} ms  (Δstep={:.3})",
            step_b - base_b);
        println!("  -> set_parent call {:.1}× cheaper; post-step Δ {:.1}× cheaper\n",
            call_b / call_a.max(1e-6), (step_b - base_b) / (step_a - base_a).max(1e-6));
    }

    // ── 4. Removing bodies wakes their contact neighbours -> the next step inflates.
    //       (RigidBodySet::remove has no opt-out: collider removal always wakes contacts.) ─
    println!("[4] Remove K=512 bodies from settled N=2304, then step (removal wakes neighbours)");
    {
        let (mut w, _floor, handles) = settled_stack(24, 4);
        let baseline = w.median_step_ms(10);
        let t = Instant::now();
        for &h in handles.iter().take(512) {
            w.bodies.remove(h, &mut w.islands, &mut w.colliders, &mut w.impulse, &mut w.multibody, true);
        }
        let call_ms = t.elapsed().as_secs_f64() * 1000.0;
        let step_after = w.timed_step();
        println!("  remove call={call_ms:.3} ms ({:.4} ms/body)   baseline_step={baseline:.3} ms  ->  step_after={step_after:.3} ms",
            call_ms / 512.0);
        println!("  neighbours woken by the removals (awake after): {}  (these now pay full solver/island cost next step)",
            w.awake());
    }
}
