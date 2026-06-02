//! Reusable benchmarking / profiling harness for the full destructible pipeline.
//!
//! This module is gated behind the `bench-support` feature (off by default, never
//! shipped in a release) so that benches (`benches/destruction.rs`), the
//! `frame_profile` example, and the perf-regression test can all drive the
//! **same** realistic scenario the demos do — full Rapier physics, contact-force
//! injection from projectiles, fracture, topology edits, and resimulation — and
//! measure it consistently.
//!
//! Design goals:
//! - **Confidence**: one code path that everything shares, so a criterion number
//!   and the spike report and the regression guard all describe the same work.
//! - **Realism / no cheating**: the harness never limits fractures or clamps
//!   solver work. It exposes a [`QualityFingerprint`] so any optimization can be
//!   checked for behavioral equivalence (same bonds break, same settled shape).
//! - **Spike visibility**: [`FrameReport`] carries per-frame wall-clock timings
//!   plus the library's own split-edit breakdown, so we can attribute a spike to
//!   the stress solve vs. the Rapier topology edits vs. resim rollback.

use std::time::Instant;

use rapier3d::prelude::*;

use crate::rapier::{
    BodySnapshots, DestructibleConfig, DestructibleSet, FracturePolicy, ResimulationOptions,
    SleepThresholdOptions, SmallBodyDampingOptions,
};
use crate::rapier::{DebrisCleanupOptions, OptimizationMode};
use crate::scenarios::{
    build_bridge_scenario, build_tower_scenario, build_wall_scenario, load_scenario_file,
    BridgeOptions, LoadedScenario, TowerOptions, WallOptions,
};
use crate::types::*;

/// Material strength preset for a scenario.
#[derive(Clone, Copy, Debug)]
pub enum Material {
    /// Strong: only a hard impact breaks bonds (gravity alone holds).
    Strong,
    /// Weak: a light impact shatters; gravity may progressively collapse.
    Weak,
    /// Custom limits.
    Custom(SolverSettings),
}

impl Material {
    pub fn settings(self) -> SolverSettings {
        match self {
            // "Strong" is meant to hold under gravity (and a normal impact): the
            // steady-state, nothing-breaking benchmark — the common pre-impact 120 FPS
            // case where the solver still runs every frame. Limits are high enough that a
            // heavy deck does not overstress any bond.
            Material::Strong => SolverSettings {
                max_solver_iterations_per_frame: 32,
                graph_reduction_level: 0,
                compression_elastic_limit: 5.0e5,
                compression_fatal_limit: 1.0e6,
                tension_elastic_limit: 5.0e5,
                tension_fatal_limit: 1.0e6,
                shear_elastic_limit: 5.0e5,
                shear_fatal_limit: 1.0e6,
            },
            Material::Weak => SolverSettings {
                max_solver_iterations_per_frame: 32,
                graph_reduction_level: 0,
                compression_elastic_limit: 0.02,
                compression_fatal_limit: 0.05,
                tension_elastic_limit: 0.01,
                tension_fatal_limit: 0.03,
                shear_elastic_limit: 0.01,
                shear_fatal_limit: 0.03,
            },
            Material::Custom(s) => s,
        }
    }
}

/// Build a wall scenario of a given grid resolution (mirrors the wall-demolition demo).
pub fn wall(span_segments: u32, height_segments: u32, layers: u32) -> ScenarioDesc {
    build_wall_scenario(&WallOptions {
        span_segments,
        height_segments,
        layers,
        ..WallOptions::default()
    })
}

/// Build a tower scenario (mirrors the tower-collapse demo).
pub fn tower(side: u32, stories: u32) -> ScenarioDesc {
    build_tower_scenario(&TowerOptions {
        side,
        stories,
        ..TowerOptions::default()
    })
}

/// Build a bridge scenario (mirrors the bridge-split demo).
pub fn bridge(span_segments: u32, width_segments: u32, thickness_layers: u32) -> ScenarioDesc {
    build_bridge_scenario(&BridgeOptions {
        span_segments,
        width_segments,
        thickness_layers,
        ..BridgeOptions::default()
    })
}

/// Absolute path to a committed scene pack under the Bevy demo's `assets/scenes/`,
/// resolved relative to this crate (works regardless of CWD).
pub fn scene_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../blast-stress-demo-rs/assets/scenes")
        .join(format!("{name}.json"))
}

/// Try to load a scene pack by name (e.g. `"high-rise"`, `"fractured-tower"`).
/// Returns `None` (with a logged reason) if the pack isn't present — the generated
/// `high-rise.json` is git-ignored, so benches/tests skip gracefully without it.
pub fn try_load_scene(name: &str) -> Option<LoadedScenario> {
    match load_scenario_file(&scene_path(name)) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("SKIP scene pack '{name}': {e}");
            None
        }
    }
}

/// Build a [`SimConfig`] for a loaded scene pack, taking its gravity + tuned solver
/// limits from the pack and the rest (resim, dt, ground) from `base`.
pub fn config_for_loaded(loaded: &LoadedScenario, base: SimConfig) -> SimConfig {
    SimConfig {
        gravity: loaded.gravity_vec(),
        settings: loaded.settings,
        ..base
    }
}

/// A projectile to fire at the structure.
#[derive(Clone, Copy, Debug)]
pub struct Projectile {
    pub spawn: Vec3,
    pub velocity: Vec3,
    pub radius: f32,
    pub mass: f32,
}

/// Configuration for a [`Sim`].
#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    /// Gravity applied to BOTH the stress solver and the Rapier world.
    pub gravity: Vec3,
    pub settings: SolverSettings,
    pub policy: FracturePolicy,
    pub resim: ResimulationOptions,
    pub dt: f32,
    /// Add a large fixed ground plane (so debris settles and generates contacts).
    pub with_ground: bool,
    /// Enable debris sleep thresholds + cleanup, matching the demos' "lots of bodies" tuning.
    pub debris_optimizations: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
            settings: Material::Strong.settings(),
            policy: FracturePolicy {
                idle_skip: true,
                apply_excess_forces: false,
                ..FracturePolicy::default()
            },
            resim: ResimulationOptions {
                enabled: true,
                max_passes: 2,
            },
            dt: 1.0 / 60.0,
            with_ground: true,
            debris_optimizations: false,
        }
    }
}

/// Per-frame measurement. Times are wall-clock milliseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameReport {
    pub frame: u32,
    /// Total wall-clock for the whole frame (all resim passes included).
    pub total_ms: f64,
    /// Time inside Rapier `PhysicsPipeline::step` for the FIRST pass (the base
    /// physics cost that resim does not add to).
    pub physics_ms: f64,
    /// Extra Rapier `PhysicsPipeline::step` time from resim re-runs (passes 2+).
    /// This is the physics cost resim *adds* — the lever for resim optimization.
    pub resim_physics_ms: f64,
    /// Time inside `DestructibleSet::step` (stress solve + fracture + topology edits),
    /// summed over passes.
    pub solver_step_ms: f64,
    /// Time spent capturing/restoring resim snapshots.
    pub resim_ms: f64,
    /// Library-reported time applying split topology edits to Rapier.
    pub split_edit_ms: f64,
    pub passes: u32,
    pub fractures: usize,
    pub new_bodies: usize,
    pub split_events: usize,
    pub dynamic_bodies: usize,
    pub converged: bool,
}

/// A fingerprint of simulation *outcome* — used to guard against "cheating"
/// optimizations that trade quality for speed. Two runs that look the same to a
/// player should produce near-identical fingerprints.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct QualityFingerprint {
    /// Bonds still intact (Rust-side active bond tracking).
    pub active_bonds: usize,
    /// Total tracked rigid bodies.
    pub bodies: usize,
    /// Dynamic (movable) bodies.
    pub dynamic_bodies: usize,
    /// Number of distinct stress-graph actors (connected components).
    pub actors: u32,
    /// Center of mass of all dynamic bodies (settled shape signature).
    pub com: [f32; 3],
    /// Sum of |position| of all dynamic bodies (spread signature).
    pub position_l1: f32,
}

impl QualityFingerprint {
    /// True if `self` and `other` agree within tolerances suitable for a
    /// behavioral-equivalence check (small float drift allowed; counts exact).
    pub fn approx_eq(&self, other: &Self, pos_tol: f32) -> bool {
        self.active_bonds == other.active_bonds
            && self.bodies == other.bodies
            && self.dynamic_bodies == other.dynamic_bodies
            && self.actors == other.actors
            && (self.com[0] - other.com[0]).abs() <= pos_tol
            && (self.com[1] - other.com[1]).abs() <= pos_tol
            && (self.com[2] - other.com[2]).abs() <= pos_tol
            && (self.position_l1 - other.position_l1).abs()
                <= pos_tol * (self.dynamic_bodies.max(1) as f32)
    }
}

/// A full destructible simulation: stress solver + Rapier world + physics pipeline,
/// with the demo's resim loop and contact-force injection built in.
pub struct Sim {
    pub set: DestructibleSet,
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub islands: IslandManager,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pipeline: PhysicsPipeline,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    ccd: CCDSolver,
    gravity_vec: Vector<Real>,
    ip: IntegrationParameters,
    node_count: u32,
    /// Tracked projectiles: (body, mass) — used to inject contact forces into the solver.
    projectiles: Vec<(RigidBodyHandle, f32)>,
    now_secs: f32,
    frame: u32,
    record_timing: bool,
}

impl Sim {
    pub fn new(scenario: &ScenarioDesc, cfg: SimConfig) -> Self {
        let (nodes, bonds) = scenario.to_solver_descs();
        let node_count = nodes.len() as u32;

        let node_sizes: Vec<Vec3> = scenario
            .node_sizes
            .iter()
            .copied()
            .chain(
                scenario
                    .nodes
                    .iter()
                    .skip(scenario.node_sizes.len())
                    .map(|n| {
                        let side = n.volume.cbrt().max(0.01);
                        Vec3::new(side, side, side)
                    }),
            )
            .collect();
        let node_colliders: Vec<Option<ScenarioCollider>> = scenario
            .collider_shapes
            .iter()
            .cloned()
            .chain(
                scenario
                    .nodes
                    .iter()
                    .skip(scenario.collider_shapes.len())
                    .map(|_| None),
            )
            .collect();

        let (sleep_thresholds, debris_cleanup, small_body_damping) = if cfg.debris_optimizations {
            (
                SleepThresholdOptions {
                    mode: OptimizationMode::Always,
                    linear_threshold: 0.2,
                    angular_threshold: 0.2,
                },
                DebrisCleanupOptions::default(),
                SmallBodyDampingOptions::default(),
            )
        } else {
            (
                SleepThresholdOptions::default(),
                DebrisCleanupOptions::default(),
                SmallBodyDampingOptions::default(),
            )
        };

        let mut set = DestructibleSet::new(DestructibleConfig {
            nodes,
            bonds,
            node_sizes,
            node_colliders,
            solver_settings: cfg.settings,
            gravity: cfg.gravity,
            fracture_policy: cfg.policy,
            resimulation: cfg.resim,
            skip_single_bodies: false,
            sleep_thresholds,
            small_body_damping,
            debris_cleanup,
            dynamic_body_ccd_enabled: false,
        })
        .expect("failed to build DestructibleSet");
        set.set_time_step(cfg.dt);

        let mut bodies = RigidBodySet::new();
        let mut colliders = ColliderSet::new();
        let islands = IslandManager::new();
        let impulse_joints = ImpulseJointSet::new();
        let multibody_joints = MultibodyJointSet::new();

        set.initialize(&mut bodies, &mut colliders);

        if cfg.with_ground {
            let ground = bodies.insert(RigidBodyBuilder::fixed().translation(vector![0.0, 0.0, 0.0]));
            colliders.insert_with_parent(
                ColliderBuilder::cuboid(200.0, 0.5, 200.0)
                    .translation(vector![0.0, -0.5, 0.0])
                    .friction(0.6),
                ground,
                &mut bodies,
            );
            set.set_ground_body_handle(Some(ground));
        }

        let ip = IntegrationParameters {
            dt: cfg.dt,
            ..IntegrationParameters::default()
        };

        Sim {
            set,
            bodies,
            colliders,
            islands,
            impulse_joints,
            multibody_joints,
            pipeline: PhysicsPipeline::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            ccd: CCDSolver::new(),
            gravity_vec: vector![cfg.gravity.x, cfg.gravity.y, cfg.gravity.z],
            ip,
            node_count,
            projectiles: Vec::new(),
            now_secs: 0.0,
            frame: 0,
            record_timing: true,
        }
    }

    /// Disable per-frame wall-clock timing capture (used by criterion, which times
    /// the closure itself — avoids paying for `Instant::now` twice).
    pub fn set_record_timing(&mut self, enabled: bool) {
        self.record_timing = enabled;
    }

    /// Spawn a projectile and track it for contact-force injection.
    pub fn spawn_projectile(&mut self, p: &Projectile) -> RigidBodyHandle {
        let handle = self.bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(vector![p.spawn.x, p.spawn.y, p.spawn.z])
                .linvel(vector![p.velocity.x, p.velocity.y, p.velocity.z])
                .ccd_enabled(true),
        );
        self.colliders.insert_with_parent(
            ColliderBuilder::ball(p.radius).mass(p.mass),
            handle,
            &mut self.bodies,
        );
        self.projectiles.push((handle, p.mass));
        handle
    }

    fn physics_step(&mut self) {
        self.pipeline.step(
            &self.gravity_vec,
            &self.ip,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &(),
            &(),
        );
    }

    /// Find the nearest non-support node (with a live body) to a world point,
    /// returning its index and current world position.
    fn nearest_node_world(&self, x: f32, y: f32, z: f32) -> Option<(u32, Vec3)> {
        let mut best: Option<(u32, Vec3)> = None;
        let mut best_d = f32::MAX;
        for n in 0..self.node_count {
            if self.set.is_support(n) {
                continue;
            }
            let Some(bh) = self.set.node_body(n) else {
                continue;
            };
            let Some(off) = self.set.node_local_offset(n) else {
                continue;
            };
            let Some(body) = self.bodies.get(bh) else {
                continue;
            };
            let p = body.position() * point![off.x, off.y, off.z];
            let d = (p.x - x).powi(2) + (p.y - y).powi(2) + (p.z - z).powi(2);
            if d < best_d {
                best_d = d;
                best = Some((n, Vec3::new(p.x, p.y, p.z)));
            }
        }
        best
    }

    /// Inject contact forces from each projectile's momentum change into the solver,
    /// rotated into the hit body's local frame (mirrors the demo's contact-force path).
    fn inject_contact_forces(&mut self, pre_vel: &[Vector<Real>]) {
        let dt = self.ip.dt;
        for (i, (handle, mass)) in self.projectiles.clone().iter().enumerate() {
            let Some(body) = self.bodies.get(*handle) else {
                continue;
            };
            let v_now = *body.linvel();
            let dp = (pre_vel[i] - v_now) * *mass;
            if dp.norm() <= 1e-3 {
                continue;
            }
            let pos = body.translation();
            if let Some((node, world_p)) = self.nearest_node_world(pos.x, pos.y, pos.z) {
                let world_force = dp / dt;
                // Rotate into the hit node's body-local frame, as the solver expects.
                let local_force = self
                    .set
                    .node_body(node)
                    .and_then(|h| self.bodies.get(h))
                    .map(|b| b.rotation().inverse_transform_vector(&world_force))
                    .unwrap_or(world_force);
                self.set.add_force(
                    node,
                    world_p,
                    Vec3::new(local_force.x, local_force.y, local_force.z),
                );
            }
        }
    }

    fn capture_pre_vel(&self) -> Vec<Vector<Real>> {
        self.projectiles
            .iter()
            .map(|(h, _)| self.bodies.get(*h).map(|b| *b.linvel()).unwrap_or_else(Vector::zeros))
            .collect()
    }

    /// Advance one full frame, including the resim rollback loop (mirrors the demo).
    pub fn step_frame(&mut self) -> FrameReport {
        let frame_start = if self.record_timing {
            Some(Instant::now())
        } else {
            None
        };
        let mut report = FrameReport {
            frame: self.frame,
            ..FrameReport::default()
        };

        let resim = self.set.resimulation_options();
        let want_snapshot = resim.enabled && self.set.needs_resimulation_snapshot();

        let mut resim_ms = 0.0f64;
        let mut snapshot = if want_snapshot {
            let t = Instant::now();
            let snap = BodySnapshots::capture(&self.bodies);
            resim_ms += t.elapsed().as_secs_f64() * 1e3;
            Some(snap)
        } else {
            None
        };
        let mut passes_left = resim.max_passes;

        loop {
            report.passes += 1;

            // 1. Physics. Attribute pass 1 to base cost; passes 2+ are resim re-runs.
            let pre_vel = self.capture_pre_vel();
            let t_phys = Instant::now();
            self.physics_step();
            let phys_ms = t_phys.elapsed().as_secs_f64() * 1e3;
            if report.passes == 1 {
                report.physics_ms += phys_ms;
            } else {
                report.resim_physics_ms += phys_ms;
            }

            // 2. Inject contact forces from projectile impacts.
            self.inject_contact_forces(&pre_vel);

            // 3. Stress solve + fracture + topology edits.
            let t_step = Instant::now();
            let r = self.set.step_with_time(
                self.now_secs,
                self.ip.dt,
                &mut self.bodies,
                &mut self.colliders,
                &mut self.islands,
                &mut self.impulse_joints,
                &mut self.multibody_joints,
            );
            report.solver_step_ms += t_step.elapsed().as_secs_f64() * 1e3;

            report.fractures += r.fractures;
            report.new_bodies += r.new_bodies;
            report.split_events += r.split_events;
            report.split_edit_ms += split_edit_total_ms(&r);
            report.converged = r.converged;

            let fractured = r.split_events > 0 || r.new_bodies > 0;
            if !fractured || passes_left == 0 {
                break;
            }

            // 4. Roll the pre-existing bodies back; new fragment bodies persist, so the
            //    next pass re-resolves the real contact against the fractured pieces.
            if let Some(snap) = snapshot.as_ref() {
                let t = Instant::now();
                snap.restore(&mut self.bodies);
                resim_ms += t.elapsed().as_secs_f64() * 1e3;
            }
            passes_left -= 1;
            let t = Instant::now();
            snapshot = Some(BodySnapshots::capture(&self.bodies));
            resim_ms += t.elapsed().as_secs_f64() * 1e3;
        }

        report.resim_ms = resim_ms;
        report.dynamic_bodies = self.set.dynamic_body_count(&self.bodies);
        if let Some(start) = frame_start {
            report.total_ms = start.elapsed().as_secs_f64() * 1e3;
        }

        self.now_secs += self.ip.dt;
        self.frame += 1;
        report
    }

    /// Run `frames` frames, returning per-frame reports.
    pub fn run(&mut self, frames: u32) -> Vec<FrameReport> {
        (0..frames).map(|_| self.step_frame()).collect()
    }

    /// Capture a behavioral-outcome fingerprint of the current state.
    pub fn fingerprint(&self) -> QualityFingerprint {
        let mut com = [0.0f32; 3];
        let mut position_l1 = 0.0f32;
        let mut dynamic_bodies = 0usize;
        let mut total_bodies = 0usize;
        for (_, body) in self.bodies.iter() {
            total_bodies += 1;
            if !body.is_dynamic() {
                continue;
            }
            dynamic_bodies += 1;
            let t = body.translation();
            com[0] += t.x;
            com[1] += t.y;
            com[2] += t.z;
            position_l1 += t.x.abs() + t.y.abs() + t.z.abs();
        }
        if dynamic_bodies > 0 {
            let inv = 1.0 / dynamic_bodies as f32;
            com[0] *= inv;
            com[1] *= inv;
            com[2] *= inv;
        }
        QualityFingerprint {
            active_bonds: self.set.active_bond_count(),
            bodies: total_bodies,
            dynamic_bodies,
            actors: self.set.actor_count(),
            com,
            position_l1,
        }
    }
}

fn split_edit_total_ms(r: &crate::rapier::StepResult) -> f64 {
    (r.split_sanitize_ms
        + r.split_estimate_ms
        + r.split_edits.plan_ms
        + r.split_edits.apply_ms) as f64
}

/// Aggregate timing statistics over a run.
#[derive(Clone, Copy, Debug, Default)]
pub struct TimingStats {
    pub frames: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub total_ms: f64,
    /// Frames whose total exceeded a 120 FPS (8.33 ms) budget.
    pub over_120fps: usize,
    /// Frames whose total exceeded a 60 FPS (16.67 ms) budget.
    pub over_60fps: usize,
}

/// Compute timing statistics from a slice of per-frame totals (ms).
pub fn timing_stats(frame_ms: &[f64]) -> TimingStats {
    if frame_ms.is_empty() {
        return TimingStats::default();
    }
    let mut sorted: Vec<f64> = frame_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let pct = |p: f64| -> f64 {
        let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n - 1);
        sorted[idx]
    };
    let total: f64 = sorted.iter().sum();
    TimingStats {
        frames: n,
        mean_ms: total / n as f64,
        p50_ms: pct(0.50),
        p90_ms: pct(0.90),
        p99_ms: pct(0.99),
        max_ms: *sorted.last().unwrap(),
        total_ms: total,
        over_120fps: frame_ms.iter().filter(|&&t| t > 1000.0 / 120.0).count(),
        over_60fps: frame_ms.iter().filter(|&&t| t > 1000.0 / 60.0).count(),
    }
}
