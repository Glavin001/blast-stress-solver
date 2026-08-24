//! The engine-independent destruction pipeline.
//!
//! This is the whole point of the contract: one implementation of stress
//! solving, fracture, split planning and the topology edit, driving any backend
//! that satisfies [`PhysicsBackend`]. Nothing here names an engine.
//!
//! # The centre-of-mass law
//!
//! One rule accounts for what used to be three divergent implementations. A
//! stored linear velocity is the velocity **of a specific point** — the body's
//! centre of mass. Whenever that reference point moves, the value must be
//! re-expressed:
//!
//! ```text
//! linvel_at(B) = linvel_at(A) + ω × (B − A)
//! ```
//!
//! The rigid fit produces a velocity at the child's *node-model* centre
//! (`fit_center`), while the engine stores velocity at its *collider-derived*
//! centre of mass. Skipping the correction leaves a rotating fragment with a
//! spurious `ω × (engine_com − fit_center)` — the "fragment lurches after it
//! fractures" bug. The adapter reports both centres; core applies the law once.
//!
//! Crucially the COM may only be read **after** an explicit `recompute_mass`,
//! because both target engines defer mass updates. Read it early and the lever
//! is zero and the correction silently does nothing.

use std::collections::{HashMap, HashSet};

use crate::backend::{
    Applied, BackendHandle, BodyKind, BodyStateSoa, CommandBuffer, CommandResults, CreateBody,
    CreateShape, Phase, PhysicsBackend, Pose, ReparentShape, ShapeGeom,
};
use crate::ext_stress_solver::ExtStressSolver;
use crate::pipeline::motion_fit::{fit_rigid_motion, weighted_center_of_mass};
use crate::pipeline::split_planner::{plan_split_migration, ExistingBodyState};
use crate::types::{ScenarioCollider, ScenarioDesc, SolverSettings, SplitEvent, Vec3};

/// Tuning that is genuinely engine-independent.
#[derive(Clone, Debug)]
pub struct DestructibleConfig {
    /// Where this structure sits in the world.
    ///
    /// Pack node centroids are **structure-local**; a city is N instances of
    /// one pack at N poses, so placement belongs here rather than being baked
    /// into the geometry at export time.
    pub world_pose: Pose,
    pub gravity: Vec3,
    pub solver: SolverSettings,
    /// Children below this node count are not given a body.
    pub min_child_nodes: usize,
    /// Cap on bodies created in one step; the remainder carries to the next.
    pub max_new_bodies_per_step: usize,
}

impl Default for DestructibleConfig {
    fn default() -> Self {
        Self {
            world_pose: Pose::IDENTITY,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            solver: SolverSettings::default(),
            min_child_nodes: 1,
            max_new_bodies_per_step: usize::MAX,
        }
    }
}

/// What one step did.
#[derive(Clone, Debug, Default)]
pub struct StepReport {
    pub fractures: usize,
    pub split_events: usize,
    pub bodies_created: usize,
    pub shapes_reparented: usize,
    pub bodies_retired: usize,
    pub writes_elided: usize,
    pub converged: bool,
}

/// A destructible structure driven through a backend.
pub struct Destructible<B: PhysicsBackend> {
    solver: ExtStressSolver,
    cfg: DestructibleConfig,
    node_body: Vec<Option<B::BodyId>>,
    node_shape: Vec<Option<B::ShapeId>>,
    node_local: Vec<Vec3>,
    node_centroid: Vec<Vec3>,
    node_mass: Vec<f32>,
    support: HashSet<u32>,
    body_nodes: HashMap<B::BodyId, Vec<u32>>,
    /// Reused across steps so a steady state allocates nothing.
    cmds: CommandBuffer<B::BodyId, B::ShapeId>,
    out: CommandResults<B::BodyId, B::ShapeId>,
    soa: BodyStateSoa,
    scratch_ids: Vec<B::BodyId>,
    scratch_com: Vec<Vec3>,
    pending: Vec<SplitEvent>,
}

impl<B: PhysicsBackend> Destructible<B> {
    /// Build the structure and instantiate it in `backend`.
    pub fn attach(backend: &mut B, scenario: &ScenarioDesc, cfg: DestructibleConfig) -> Option<Self> {
        let (nodes, bonds) = scenario.to_solver_descs();
        let solver = ExtStressSolver::new(&nodes, &bonds, &cfg.solver)?;
        let n = nodes.len();

        let mut d = Self {
            solver,
            cfg,
            node_body: vec![None; n],
            node_shape: vec![None; n],
            node_local: vec![Vec3::ZERO; n],
            node_centroid: nodes.iter().map(|x| x.centroid).collect(),
            node_mass: nodes.iter().map(|x| x.mass).collect(),
            support: nodes
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (x.mass == 0.0).then_some(i as u32))
                .collect(),
            body_nodes: HashMap::new(),
            cmds: CommandBuffer::new(),
            out: CommandResults::default(),
            soa: BodyStateSoa::default(),
            scratch_ids: Vec::new(),
            scratch_com: Vec::new(),
            pending: Vec::new(),
        };

        // One body per actor (connected component), one shape per node.
        let actors = d.solver.actors();
        d.cmds.clear();
        let mut actor_nodes: Vec<Vec<u32>> = Vec::with_capacity(actors.len());
        for a in &actors {
            let has_support = a.nodes.iter().any(|x| d.support.contains(x));
            let centre = d.cfg.world_pose.transform_point(d.centre_of(&a.nodes));
            d.cmds.create_bodies.push(CreateBody {
                pose: Pose::new(centre, d.cfg.world_pose.rotation),
                kind: if has_support { BodyKind::Fixed } else { BodyKind::Dynamic },
                linvel: Vec3::ZERO,
                angvel: Vec3::ZERO,
                ccd: false,
                start_sleeping: false,
            });
            actor_nodes.push(a.nodes.clone());
        }
        backend.apply(Phase::Topology, &d.cmds, &mut d.out).ok()?;
        let bodies = d.out.created_bodies.clone();

        d.cmds.clear();
        let mut shape_owner: Vec<u32> = Vec::new();
        for (ai, ns) in actor_nodes.iter().enumerate() {
            let body = bodies[ai];
            let centre = d.centre_of(ns);
            for &node in ns {
                // Shape offsets stay in the structure frame: the body carries
                // the world placement, so a rotated instance needs no per-shape
                // rotation of its own.
                let local = d.node_centroid[node as usize] - centre;
                d.node_local[node as usize] = local;
                d.node_body[node as usize] = Some(body);
                d.cmds.create_shapes.push(CreateShape {
                    body,
                    local: Pose::from_translation(local),
                    geom: geom_for(scenario, node as usize),
                    mass: d.node_mass[node as usize].max(0.0),
                    node,
                });
                shape_owner.push(node);
            }
            d.body_nodes.insert(body, ns.clone());
        }
        backend.apply(Phase::Topology, &d.cmds, &mut d.out).ok()?;
        for (i, node) in shape_owner.iter().enumerate() {
            d.node_shape[*node as usize] = Some(d.out.created_shapes[i]);
        }

        d.cmds.clear();
        d.cmds.recompute_mass.extend(bodies.iter().copied());
        backend.apply(Phase::Topology, &d.cmds, &mut d.out).ok()?;
        Some(d)
    }

    fn centre_of(&self, nodes: &[u32]) -> Vec3 {
        let pts: Vec<(Vec3, f32)> = nodes
            .iter()
            .map(|n| (self.node_centroid[*n as usize], self.node_mass[*n as usize].max(0.0)))
            .collect();
        weighted_center_of_mass(&pts).unwrap_or_else(|| {
            let mut c = Vec3::ZERO;
            for (p, _) in &pts {
                c += *p;
            }
            c / pts.len().max(1) as f32
        })
    }

    pub fn solver(&self) -> &ExtStressSolver {
        &self.solver
    }
    pub fn solver_mut(&mut self) -> &mut ExtStressSolver {
        &mut self.solver
    }

    /// Bodies currently owned by this structure, in a stable order.
    pub fn bodies(&self) -> Vec<B::BodyId> {
        let mut v: Vec<B::BodyId> = self.body_nodes.keys().copied().collect();
        v.sort_by_key(|h| h.sort_key());
        v
    }

    /// Nearest load-bearing node to a world point.
    ///
    /// This is the hitscan/blast entry point: a host raycasts, gets a world
    /// position, and needs the stress-graph node to drive the load through.
    /// Support nodes are skipped because they are world-anchored and absorb
    /// force without ever failing, so aiming at one is silently a no-op.
    pub fn nearest_dynamic_node(&self, world_point: Vec3) -> Option<u32> {
        let mut best = None;
        let mut best_d = f32::MAX;
        for node in 0..self.node_centroid.len() as u32 {
            if self.support.contains(&node) {
                continue;
            }
            // Track the node's live position: after a fracture a chunk has
            // moved, and its authored centroid no longer says where it is.
            let p = match self.node_body(node) {
                Some(_) => self.node_centroid[node as usize],
                None => continue,
            };
            let d = (p - world_point).magnitude_squared();
            if d < best_d {
                best_d = d;
                best = Some(node);
            }
        }
        best
    }

    /// Authored centroid of a node, in the structure's own frame.
    pub fn node_centroid(&self, node: u32) -> Option<Vec3> {
        self.node_centroid.get(node as usize).copied()
    }

    pub fn node_body(&self, node: u32) -> Option<B::BodyId> {
        self.node_body.get(node as usize).copied().flatten()
    }

    /// Apply an external force to a node, in the structure's authored frame.
    pub fn add_force(&mut self, node: u32, position: Vec3, force: Vec3) {
        self.solver.add_force(node, position, force, crate::types::ForceMode::Force);
    }

    /// Advance the stress solve and apply any resulting topology change.
    pub fn step(&mut self, backend: &mut B, _dt: f32) -> StepReport {
        let mut report = StepReport::default();

        // Gravity in each actor's current frame, so a rotated chunk feels it
        // from the right direction.
        self.apply_oriented_gravity(backend);

        self.solver.update();
        report.converged = self.solver.converged();

        if self.solver.overstressed_bond_count() > 0 {
            let cmds = self.solver.generate_fracture_commands();
            report.fractures = cmds.iter().map(|c| c.bond_fractures.len()).sum();
            if !cmds.is_empty() {
                self.pending.extend(self.solver.apply_fracture_commands(&cmds));
            }
        }

        let mut budget = self.cfg.max_new_bodies_per_step;
        while let Some(event) = self.pending.first().cloned() {
            if budget == 0 {
                break;
            }
            self.pending.remove(0);
            report.split_events += 1;
            let applied = self.apply_split(backend, &event, &mut budget);
            report.bodies_created += applied.bodies_created;
            report.shapes_reparented += applied.shapes_reparented;
            report.bodies_retired += applied.bodies_removed;
            report.writes_elided += applied.writes_elided;
        }
        report
    }

    fn apply_oriented_gravity(&mut self, backend: &B) {
        let reps = self.solver.collect_actor_reps();
        if reps.is_empty() {
            return;
        }
        self.scratch_ids.clear();
        let mut actor_of = Vec::with_capacity(reps.len());
        for (actor, node) in &reps {
            if let Some(b) = self.node_body(*node) {
                self.scratch_ids.push(b);
                actor_of.push(*actor);
            }
        }
        if self.scratch_ids.is_empty() {
            return;
        }
        backend.read_bodies(&self.scratch_ids, &mut self.soa);
        for (i, actor) in actor_of.iter().enumerate() {
            let local_g = self.soa.pose[i].rotation.rotate_inverse(self.cfg.gravity);
            self.solver.add_actor_gravity(*actor, local_g);
        }
    }

    fn apply_split(&mut self, backend: &mut B, event: &SplitEvent, budget: &mut usize) -> Applied {
        let mut total = Applied::default();

        // Children too small to embody are dropped rather than paying a full
        // split for something about to be culled.
        let children: Vec<&crate::types::SplitChild> = event
            .children
            .iter()
            .filter(|c| c.nodes.len() >= self.cfg.min_child_nodes)
            .collect();
        if children.is_empty() {
            return total;
        }

        // --- Capture: every read happens before any edit ---
        let parent_bodies = self.parents_of(event);
        if parent_bodies.is_empty() {
            return total;
        }
        backend.read_bodies(&parent_bodies, &mut self.soa);
        let parent_state: HashMap<B::BodyId, (Pose, Vec3, Vec3, bool)> = parent_bodies
            .iter()
            .enumerate()
            .map(|(i, b)| {
                (*b, (self.soa.pose[i], self.soa.linvel[i], self.soa.angvel[i], self.soa.is_sleeping(i)))
            })
            .collect();

        // Node world positions and velocities, sampled on the pre-split parent.
        let mut queries = Vec::new();
        let mut query_nodes = Vec::new();
        for c in &children {
            for &n in &c.nodes {
                if let Some(b) = self.node_body(n) {
                    let pose = parent_state[&b].0;
                    queries.push((b, pose.transform_point(self.node_local[n as usize])));
                    query_nodes.push(n);
                }
            }
        }
        let mut point_vels = Vec::new();
        backend.read_point_velocities(&queries, &mut point_vels);
        let world_pos: HashMap<u32, Vec3> =
            query_nodes.iter().zip(queries.iter()).map(|(n, (_, p))| (*n, *p)).collect();
        let world_vel: HashMap<u32, Vec3> =
            query_nodes.iter().zip(point_vels.iter()).map(|(n, v)| (*n, *v)).collect();

        // --- Plan (pure) ---
        let existing: Vec<ExistingBodyState<B::BodyId>> = parent_bodies
            .iter()
            .map(|b| ExistingBodyState {
                handle: *b,
                node_indices: self.body_nodes.get(b).cloned().unwrap_or_default().into_iter().collect(),
                is_fixed: false,
            })
            .collect();
        let owned: Vec<crate::types::SplitChild> = children.iter().map(|c| (*c).clone()).collect();
        let plan = plan_split_migration(&existing, &owned);

        let mut target: Vec<Option<B::BodyId>> = vec![None; owned.len()];
        for r in &plan.reuse {
            target[r.child_index] = Some(r.body_handle);
        }

        // --- Topology: create bodies for children with no reusable parent ---
        self.cmds.clear();
        let mut create_order = Vec::new();
        for c in &plan.create {
            if *budget == 0 {
                break;
            }
            *budget -= 1;
            let ns = &owned[c.child_index].nodes;
            let fit_center = self.world_centre(ns, &world_pos);
            let has_support = ns.iter().any(|n| self.support.contains(n));
            let motion = self.fit_child(ns, fit_center, &world_pos, &world_vel);
            self.cmds.create_bodies.push(CreateBody {
                pose: Pose::from_translation(fit_center),
                kind: if has_support { BodyKind::Fixed } else { BodyKind::Dynamic },
                linvel: motion.0,
                angvel: motion.1,
                ccd: false,
                start_sleeping: false,
            });
            create_order.push(c.child_index);
        }
        let a = backend.apply(Phase::Topology, &self.cmds, &mut self.out).unwrap_or_default();
        total.bodies_created += a.bodies_created;
        for (i, ci) in create_order.iter().enumerate() {
            target[*ci] = Some(self.out.created_bodies[i]);
        }

        // --- Topology: migrate shapes, then force a mass recompute ---
        self.cmds.clear();
        let mut fit_centers: Vec<(B::BodyId, Vec3, Vec3, Vec3)> = Vec::new();
        for (ci, child) in owned.iter().enumerate() {
            let Some(body) = target[ci] else { continue };
            let fit_center = self.world_centre(&child.nodes, &world_pos);
            let (lin, ang) = self.fit_child(&child.nodes, fit_center, &world_pos, &world_vel);
            fit_centers.push((body, fit_center, lin, ang));
            for &n in &child.nodes {
                let Some(shape) = self.node_shape[n as usize] else { continue };
                let local = world_pos[&n] - fit_center;
                self.node_local[n as usize] = local;
                self.cmds.reparent_shapes.push(ReparentShape {
                    shape,
                    body,
                    local: Pose::from_translation(local),
                });
                self.node_body[n as usize] = Some(body);
            }
            self.body_nodes.insert(body, child.nodes.clone());
        }
        for (b, _, _, _) in &fit_centers {
            self.cmds.recompute_mass.push(*b);
        }
        let a = backend.apply(Phase::Topology, &self.cmds, &mut self.out).unwrap_or_default();
        total.shapes_reparented += a.shapes_reparented;

        // --- Read COM (only valid now), then Motion: apply the law ---
        self.scratch_ids.clear();
        self.scratch_ids.extend(fit_centers.iter().map(|(b, _, _, _)| *b));
        backend.read_center_of_mass(&self.scratch_ids, &mut self.scratch_com);

        self.cmds.clear();
        for (i, (body, fit_center, lin, ang)) in fit_centers.iter().enumerate() {
            let engine_com = self.scratch_com[i];
            // linvel_at(engine_com) = linvel_at(fit_center) + ω × (engine_com − fit_center)
            let corrected = *lin + ang.cross(engine_com - *fit_center);
            self.cmds.set_velocity.push((*body, corrected, *ang));
        }
        let a = backend.apply(Phase::Motion, &self.cmds, &mut self.out).unwrap_or_default();
        total.writes_elided += a.writes_elided;

        // --- Retire: parents that kept no nodes ---
        let kept: HashSet<B::BodyId> = target.iter().flatten().copied().collect();
        self.cmds.clear();
        let mut retire: Vec<B::BodyId> =
            parent_bodies.iter().copied().filter(|b| !kept.contains(b)).collect();
        retire.sort_by_key(|h| h.sort_key());
        for b in &retire {
            self.body_nodes.remove(b);
        }
        self.cmds.remove_bodies.extend(retire);
        let a = backend.apply(Phase::Retire, &self.cmds, &mut self.out).unwrap_or_default();
        total.bodies_removed += a.bodies_removed;

        let _ = parent_state;
        total
    }

    /// Parent bodies touched by an event, sorted so that body creation order —
    /// and therefore engine handle allocation — is reproducible.
    fn parents_of(&self, event: &SplitEvent) -> Vec<B::BodyId> {
        let mut v: Vec<B::BodyId> = event
            .children
            .iter()
            .flat_map(|c| c.nodes.iter())
            .filter_map(|n| self.node_body(*n))
            .collect();
        v.sort_by_key(|h| h.sort_key());
        v.dedup();
        v
    }

    fn world_centre(&self, nodes: &[u32], world_pos: &HashMap<u32, Vec3>) -> Vec3 {
        let pts: Vec<(Vec3, f32)> = nodes
            .iter()
            .filter_map(|n| world_pos.get(n).map(|p| (*p, self.node_mass[*n as usize].max(0.0))))
            .collect();
        if pts.is_empty() {
            return Vec3::ZERO;
        }
        weighted_center_of_mass(&pts).unwrap_or_else(|| {
            let mut c = Vec3::ZERO;
            for (p, _) in &pts {
                c += *p;
            }
            c / pts.len() as f32
        })
    }

    fn fit_child(
        &self,
        nodes: &[u32],
        centre: Vec3,
        world_pos: &HashMap<u32, Vec3>,
        world_vel: &HashMap<u32, Vec3>,
    ) -> (Vec3, Vec3) {
        let samples: Vec<(Vec3, Vec3, f32)> = nodes
            .iter()
            .filter_map(|n| {
                Some((
                    *world_pos.get(n)?,
                    *world_vel.get(n)?,
                    self.node_mass[*n as usize].max(1.0e-4),
                ))
            })
            .collect();
        match fit_rigid_motion(&samples, centre) {
            // A singular fit is the common case for single-node and collinear
            // fragments -- the most frequent shatter products -- so falling back
            // to zero spin rather than inverting an ill-conditioned matrix is
            // the whole reason that path exists.
            Some(f) => (f.linvel, f.angvel.unwrap_or(Vec3::ZERO)),
            None => (Vec3::ZERO, Vec3::ZERO),
        }
    }
}

fn geom_for(scenario: &ScenarioDesc, node: usize) -> ShapeGeom {
    if let Some(Some(c)) = scenario.collider_shapes.get(node) {
        return match c {
            ScenarioCollider::Cuboid { half_extents } => {
                ShapeGeom::Cuboid { half_extents: *half_extents }
            }
            ScenarioCollider::ConvexHull { points } => {
                ShapeGeom::ConvexHull { points: points.clone() }
            }
        };
    }
    let size = scenario.node_sizes.get(node).copied().unwrap_or(Vec3::new(1.0, 1.0, 1.0));
    ShapeGeom::Cuboid {
        half_extents: Vec3::new(
            (size.x * 0.5).max(1.0e-3),
            (size.y * 0.5).max(1.0e-3),
            (size.z * 0.5).max(1.0e-3),
        ),
    }
}
