use std::collections::{HashMap, HashSet};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

// `std::time::Instant::now()` panics on `wasm32-unknown-unknown` because the
// target has no monotonic clock. We only use it for split-edit telemetry, so a
// zero-duration shim keeps those stats at 0.0 ms on wasm without trapping.
#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct Instant;

#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self {
        Self
    }

    fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::ZERO
    }
}

use rapier3d::na::UnitQuaternion;
use rapier3d::prelude::*;

use super::collision_groups::{apply_collision_groups_for_body, DebrisCollisionMode};
use super::optimization::{
    DebrisCleanupOptions, OptimizationMode, OptimizationResult, SleepThresholdOptions,
    SmallBodyDampingOptions,
};
use crate::types::*;

#[derive(Clone, Copy)]
struct BodyState {
    position: Isometry<Real>,
    linvel: Vector<Real>,
    angvel: Vector<Real>,
    was_sleeping: bool,
}

#[derive(Clone, Copy, Debug)]
struct BodyMetadata {
    created_at: f32,
    first_support_contact_at: Option<f32>,
    damping_promoted: bool,
    sleep_threshold_configured: bool,
    has_support: bool,
}

#[derive(Clone, Copy)]
struct NodeSourceState {
    body_handle: RigidBodyHandle,
    collider_handle: Option<ColliderHandle>,
    world_centroid: Vec3,
    world_velocity: Vec3,
}

struct SplitPlanningData {
    existing: Vec<super::split_migrator::ExistingBodyState>,
    parent_bodies: Vec<RigidBodyHandle>,
    parent_states: HashMap<RigidBodyHandle, BodyState>,
    node_sources: HashMap<u32, NodeSourceState>,
}

#[derive(Clone, Copy)]
enum PlannedBodyTarget {
    Existing(RigidBodyHandle),
    Recycled(RigidBodyHandle),
    Created(RigidBodyHandle),
}

impl PlannedBodyTarget {
    fn handle(self) -> RigidBodyHandle {
        match self {
            Self::Existing(handle) | Self::Recycled(handle) | Self::Created(handle) => handle,
        }
    }
}

#[derive(Clone, Copy)]
struct ChildTargetState {
    /// The pose to actually set on the body. For a REUSED body this is its inherited (current)
    /// pose — it is deliberately NOT recentered, so a resim snapshot of its origin stays valid
    /// and a rollback cannot teleport it ("fragment yanks inwards"). For created bodies it is the
    /// recentered pose.
    pose: Isometry<Real>,
    /// The fragment's node-model centre of mass — the reference point the velocity fit and the
    /// COM reconciliation use. Kept separate from `pose` so the reused body's velocity stays
    /// correct (identical to the recentered behavior) even though its origin is left in place.
    fit_center: Vec3,
    linvel: Vector<Real>,
    angvel: Vector<Real>,
    should_sleep: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SplitCost {
    pub create_bodies: usize,
    pub collider_migrations: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SplitEditStats {
    pub plan_ms: f32,
    pub apply_ms: f32,
    pub child_pose_ms: f32,
    pub velocity_fit_ms: f32,
    pub sleep_init_ms: f32,
    pub body_create_ms: f32,
    pub collider_move_ms: f32,
    pub collider_insert_ms: f32,
    pub body_retire_ms: f32,
    pub reused_bodies: usize,
    pub recycled_bodies: usize,
    pub created_bodies: usize,
    pub retired_bodies: usize,
    pub body_type_flips: usize,
    pub moved_colliders: usize,
    pub inserted_colliders: usize,
    pub removed_colliders: usize,
}

#[derive(Debug, Default)]
pub struct SplitApplyResult {
    pub new_handles: Vec<RigidBodyHandle>,
    pub cohort_handles: Vec<RigidBodyHandle>,
    pub source_handles: Vec<RigidBodyHandle>,
    pub stats: SplitEditStats,
}

/// Per-node observability record captured at the instant of a split when
/// continuity recording is enabled (see [`BodyTracker::set_record_split_continuity`]).
///
/// This is the Rust analog of the JS library's `recordBodyContinuity`
/// (`destructible-core.ts`): for a node migrating from a parent body to a child
/// body, a faithful rigid split must preserve the node's world-space point
/// velocity, i.e. `point_velocity_error ≈ 0`. A large value is exactly the
/// "sudden movement/rotation after destruction" symptom.
#[derive(Clone, Copy, Debug)]
pub struct SplitContinuityRecord {
    /// The child body the node was migrated to.
    pub target_body: RigidBodyHandle,
    /// The migrated node.
    pub node_index: u32,
    /// `‖child.velocity_at_point(p) − parent.velocity_at_point(p)‖` at the node
    /// world point `p`, measured against Rapier's freshly-recomputed centre of
    /// mass on the child body. This is the workhorse continuity metric.
    pub point_velocity_error: f32,
    /// Parent body's velocity at the node point (captured pre-split).
    pub parent_point_velocity: Vec3,
    /// Child body's velocity at the same world point, post-split.
    pub child_point_velocity: Vec3,
    /// `‖child_world_centroid − parent_world_centroid‖`: how far the node's world position moved
    /// across the split edit itself (recentering + collider re-offset), measured at the split
    /// instant with no integration in between. Non-zero = a teleport — the "fragment jumped on
    /// fracture" symptom — which the velocity metric cannot see.
    pub world_position_error: f32,
    /// Whether all measured child velocity components are finite (no NaN/Inf).
    pub finite: bool,
}

/// Tracks the mapping between stress graph nodes and Rapier rigid bodies.
pub struct BodyTracker {
    /// For each node index, which body it belongs to (if any).
    node_to_body: Vec<Option<RigidBodyHandle>>,
    /// For each node index, which collider it belongs to (if any).
    node_to_collider: Vec<Option<ColliderHandle>>,
    /// For each collider handle, which node index it belongs to.
    collider_to_node: HashMap<ColliderHandle, u32>,
    /// For each body handle, which node indices it contains.
    body_to_nodes: HashMap<RigidBodyHandle, Vec<u32>>,
    /// Set of support (fixed, mass=0) node indices.
    support_nodes: HashSet<u32>,
    /// Per-node size used for creating colliders.
    node_sizes: Vec<Vec3>,
    /// Optional exact per-node collider shapes.
    node_colliders: Vec<Option<ScenarioCollider>>,
    /// Per-node centroid.
    node_centroids: Vec<Vec3>,
    /// Per-node mass.
    node_masses: Vec<f32>,
    /// Per-node local offset relative to its owning body.
    node_local_offsets: Vec<Vec3>,
    /// Per-body lifecycle metadata used for replay stabilization.
    body_metadata: HashMap<RigidBodyHandle, BodyMetadata>,
    /// Whether newly created dynamic bodies should enable CCD.
    dynamic_body_ccd_enabled: bool,
    /// Collision filtering mode for small debris bodies.
    debris_collision_mode: DebrisCollisionMode,
    /// Optional ground body used by debris-ground-only filtering.
    ground_body_handle: Option<RigidBodyHandle>,
    /// Whether split children should be recentered to their own pose on handoff.
    split_child_recentering_enabled: bool,
    /// Whether split children should get fitted kinematics on handoff.
    split_child_velocity_fit_enabled: bool,
    /// Observability: when enabled, per-split point-velocity continuity is recorded.
    /// Off by default so production behavior/cost is unchanged.
    record_split_continuity: bool,
    /// Continuity records accumulated since the last `clear_split_continuity`.
    split_continuity: Vec<SplitContinuityRecord>,
}

impl BodyTracker {
    /// Create a new tracker from a scenario description.
    pub fn new(
        nodes: &[ScenarioNode],
        node_sizes: Vec<Vec3>,
        node_colliders: Vec<Option<ScenarioCollider>>,
    ) -> Self {
        let support_nodes: HashSet<u32> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.mass == 0.0)
            .map(|(i, _)| i as u32)
            .collect();

        let node_centroids = nodes.iter().map(|n| n.centroid).collect();
        let node_masses = nodes.iter().map(|n| n.mass).collect();

        Self {
            node_to_body: vec![None; nodes.len()],
            node_to_collider: vec![None; nodes.len()],
            collider_to_node: HashMap::new(),
            body_to_nodes: HashMap::new(),
            support_nodes,
            node_sizes,
            node_colliders,
            node_centroids,
            node_masses,
            node_local_offsets: vec![Vec3::ZERO; nodes.len()],
            body_metadata: HashMap::new(),
            dynamic_body_ccd_enabled: false,
            debris_collision_mode: DebrisCollisionMode::All,
            ground_body_handle: None,
            split_child_recentering_enabled: true,
            split_child_velocity_fit_enabled: true,
            record_split_continuity: false,
            split_continuity: Vec::new(),
        }
    }

    pub fn set_dynamic_body_ccd_enabled(&mut self, enabled: bool) {
        self.dynamic_body_ccd_enabled = enabled;
    }

    /// Enable/disable per-split point-velocity continuity recording (observability
    /// only; does not change simulation). Off by default.
    pub fn set_record_split_continuity(&mut self, enabled: bool) {
        self.record_split_continuity = enabled;
    }

    /// Clear accumulated continuity records.
    pub fn clear_split_continuity(&mut self) {
        self.split_continuity.clear();
    }

    /// Continuity records accumulated since the last `clear_split_continuity`.
    pub fn split_continuity(&self) -> &[SplitContinuityRecord] {
        &self.split_continuity
    }

    pub fn debris_collision_mode(&self) -> DebrisCollisionMode {
        self.debris_collision_mode
    }

    pub fn set_debris_collision_mode(&mut self, mode: DebrisCollisionMode) {
        self.debris_collision_mode = mode;
    }

    pub fn ground_body_handle(&self) -> Option<RigidBodyHandle> {
        self.ground_body_handle
    }

    pub fn set_ground_body_handle(&mut self, handle: Option<RigidBodyHandle>) {
        self.ground_body_handle = handle;
    }

    pub fn set_split_child_recentering_enabled(&mut self, enabled: bool) {
        self.split_child_recentering_enabled = enabled;
    }

    pub fn set_split_child_velocity_fit_enabled(&mut self, enabled: bool) {
        self.split_child_velocity_fit_enabled = enabled;
    }

    pub fn refresh_collision_groups(
        &self,
        bodies: &RigidBodySet,
        colliders: &mut ColliderSet,
        max_colliders_for_debris: usize,
    ) {
        if let Some(ground_body_handle) = self.ground_body_handle {
            apply_collision_groups_for_body(
                ground_body_handle,
                bodies,
                colliders,
                self.ground_body_handle,
                self.debris_collision_mode,
                true,
                max_colliders_for_debris,
            );
        }

        for &body_handle in self.body_to_nodes.keys() {
            apply_collision_groups_for_body(
                body_handle,
                bodies,
                colliders,
                self.ground_body_handle,
                self.debris_collision_mode,
                self.debris_collision_active(body_handle),
                max_colliders_for_debris,
            );
        }
    }

    /// Create initial Rapier bodies from the actor table.
    ///
    /// Each actor becomes one rigid body. Support-only actors become fixed bodies.
    /// Returns the list of created body handles.
    pub fn create_initial_bodies(
        &mut self,
        actors: &[Actor],
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        created_at: f32,
        small_body_damping: SmallBodyDampingOptions,
        sleep_thresholds: SleepThresholdOptions,
        max_colliders_for_debris: usize,
    ) -> Vec<RigidBodyHandle> {
        let mut handles = Vec::new();

        for actor in actors {
            if actor.nodes.is_empty() {
                continue;
            }

            let has_support = actor.nodes.iter().any(|n| self.support_nodes.contains(n));

            // Compute center of mass for the actor
            let com = self.compute_actor_com(&actor.nodes);

            let body = if has_support {
                bodies.insert(RigidBodyBuilder::fixed().translation(vector![com.x, com.y, com.z]))
            } else {
                bodies.insert(
                    RigidBodyBuilder::dynamic()
                        .translation(vector![com.x, com.y, com.z])
                        .ccd_enabled(self.dynamic_body_ccd_enabled),
                )
            };

            // Create colliders for each node
            for &node_idx in &actor.nodes {
                let ni = node_idx as usize;
                let centroid = self.node_centroids[ni];

                // Collider position is relative to the body
                let local_x = centroid.x - com.x;
                let local_y = centroid.y - com.y;
                let local_z = centroid.z - com.z;

                let collider =
                    self.build_node_collider(node_idx, Vec3::new(local_x, local_y, local_z));
                let collider_handle = colliders.insert_with_parent(collider, body, bodies);
                self.attach_node(
                    node_idx,
                    body,
                    collider_handle,
                    Vec3::new(local_x, local_y, local_z),
                );
            }

            self.body_to_nodes.insert(body, actor.nodes.clone());
            self.update_body_metadata(body, &actor.nodes, created_at);
            self.promote_small_body_damping(body, bodies, &small_body_damping);
            self.configure_sleep_thresholds(body, bodies, &sleep_thresholds);
            self.apply_collision_groups(body, bodies, colliders, max_colliders_for_debris);
            handles.push(body);
        }

        handles
    }

    /// Handle a split event: create new bodies for child actors.
    ///
    /// Returns newly created body handles.
    pub fn handle_split(
        &mut self,
        event: &SplitEvent,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        island_manager: &mut IslandManager,
        impulse_joints: &mut ImpulseJointSet,
        multibody_joints: &mut MultibodyJointSet,
        created_at: f32,
        small_body_damping: SmallBodyDampingOptions,
        sleep_thresholds: SleepThresholdOptions,
        max_colliders_for_debris: usize,
    ) -> SplitApplyResult {
        let mut result = SplitApplyResult::default();
        let planning = self.collect_split_planning_data(event, bodies);
        result.source_handles = planning.parent_bodies.clone();
        let plan_started_at = Instant::now();
        let child_support: Vec<super::split_migrator::PlannerChildSupport> = event
            .children
            .iter()
            .map(|child| super::split_migrator::PlannerChildSupport {
                is_support: child.nodes.iter().any(|n| self.support_nodes.contains(n)),
            })
            .collect();
        let plan = super::split_migrator::plan_split_migration_with_support(
            &planning.existing,
            &event.children,
            &child_support,
        );
        result.stats.plan_ms = plan_started_at.elapsed().as_secs_f64() as f32 * 1_000.0;
        result.stats.reused_bodies = plan.reuse.len();
        let reused_by_body: HashMap<RigidBodyHandle, usize> = plan
            .reuse
            .iter()
            .map(|entry| (entry.body_handle, entry.child_index))
            .collect();
        let mut recycled_fixed = Vec::new();
        let mut recycled_dynamic = Vec::new();
        for &body_handle in &planning.parent_bodies {
            if reused_by_body.contains_key(&body_handle) {
                continue;
            }
            if bodies
                .get(body_handle)
                .map(|body| body.is_dynamic())
                .unwrap_or(false)
            {
                recycled_dynamic.push(body_handle);
            } else {
                recycled_fixed.push(body_handle);
            }
        }

        let mut child_targets: HashMap<usize, PlannedBodyTarget> = HashMap::new();
        let mut child_target_states: HashMap<usize, ChildTargetState> = HashMap::new();
        let apply_started_at = Instant::now();

        for entry in &plan.reuse {
            let child = &event.children[entry.child_index];
            let child_has_support = child.nodes.iter().any(|n| self.support_nodes.contains(n));
            // reuse = true => keep the inherited origin (no recentering); see ChildTargetState.
            let (target_state, child_pose_ms, velocity_fit_ms) =
                self.compute_child_target_state(child, &planning, child_has_support, true);
            result.stats.child_pose_ms += child_pose_ms;
            result.stats.velocity_fit_ms += velocity_fit_ms;
            let flipped =
                self.reconcile_reused_body_type(entry.body_handle, child_has_support, bodies);
            result.stats.body_type_flips += usize::from(flipped);
            if let Some(body) = bodies.get_mut(entry.body_handle) {
                body.set_position(target_state.pose, false);
            }
            child_targets.insert(
                entry.child_index,
                PlannedBodyTarget::Existing(entry.body_handle),
            );
            child_target_states.insert(entry.child_index, target_state);
        }

        for entry in &plan.create {
            let child = &event.children[entry.child_index];
            let child_has_support = child.nodes.iter().any(|n| self.support_nodes.contains(n));
            // reuse = false => a fresh body, safe to recenter onto its COM.
            let (target_state, child_pose_ms, velocity_fit_ms) =
                self.compute_child_target_state(child, &planning, child_has_support, false);
            result.stats.child_pose_ms += child_pose_ms;
            result.stats.velocity_fit_ms += velocity_fit_ms;

            let recycled_handle = if child_has_support {
                recycled_fixed.pop().or_else(|| recycled_dynamic.pop())
            } else {
                recycled_dynamic.pop().or_else(|| recycled_fixed.pop())
            };

            if let Some(body_handle) = recycled_handle {
                let flipped = self.reinitialize_recycled_body(
                    body_handle,
                    child_has_support,
                    target_state,
                    bodies,
                );
                result.stats.body_type_flips += usize::from(flipped);
                result.stats.recycled_bodies += 1;
                child_targets.insert(entry.child_index, PlannedBodyTarget::Recycled(body_handle));
                child_target_states.insert(entry.child_index, target_state);
                continue;
            }

            let body_create_started_at = Instant::now();
            let body_handle = self.create_body_for_child(child, target_state, bodies);
            result.stats.body_create_ms +=
                body_create_started_at.elapsed().as_secs_f64() as f32 * 1_000.0;
            result.stats.created_bodies += 1;
            result.new_handles.push(body_handle);
            child_targets.insert(entry.child_index, PlannedBodyTarget::Created(body_handle));
            child_target_states.insert(entry.child_index, target_state);
        }

        // Process children in a deterministic (child-index) order so collider migration
        // and metadata edits are reproducible, independent of `HashMap` iteration order.
        let mut ordered_children: Vec<(usize, PlannedBodyTarget)> =
            child_targets.iter().map(|(k, v)| (*k, *v)).collect();
        ordered_children.sort_unstable_by_key(|(child_index, _)| *child_index);

        for &(child_index, target) in &ordered_children {
            let child = &event.children[child_index];
            let target_body = target.handle();
            let target_state = *child_target_states
                .get(&child_index)
                .unwrap_or_else(|| panic!("missing target state for child {}", child_index));
            let target_pose = target_state.pose;

            for &n in &child.nodes {
                let Some(source) = planning.node_sources.get(&n) else {
                    continue;
                };
                let local_offset = self.local_offset_for_pose(source, &target_pose);
                let already_on_target = self.node_body(n) == Some(target_body);
                if already_on_target
                    && self.node_local_offset(n).is_some_and(|current| {
                        (current - local_offset).magnitude_squared() <= 1.0e-8
                    })
                {
                    continue;
                }
                let move_started_at = Instant::now();
                let collider_handle = if already_on_target {
                    self.update_node_collider_local_pose(
                        source.collider_handle,
                        local_offset,
                        colliders,
                    )
                } else {
                    self.move_node_collider_to_body(
                        n,
                        target_body,
                        source.collider_handle,
                        local_offset,
                        bodies,
                        colliders,
                    )
                };
                let move_ms = move_started_at.elapsed().as_secs_f64() as f32 * 1_000.0;
                if source.collider_handle.is_some() {
                    result.stats.moved_colliders += 1;
                    result.stats.collider_move_ms += move_ms;
                } else {
                    result.stats.inserted_colliders += 1;
                    result.stats.collider_insert_ms += move_ms;
                }
                self.attach_node(n, target_body, collider_handle, local_offset);
            }

            self.body_to_nodes.insert(target_body, child.nodes.clone());
            let reset_metadata = !matches!(target, PlannedBodyTarget::Existing(_));
            self.refresh_body_metadata(target_body, &child.nodes, created_at, reset_metadata);
            self.promote_small_body_damping(target_body, bodies, &small_body_damping);
            self.configure_sleep_thresholds(target_body, bodies, &sleep_thresholds);
            self.apply_collision_groups(target_body, bodies, colliders, max_colliders_for_debris);
            let sleep_started_at = Instant::now();
            self.apply_child_kinematics(
                target_body,
                child.nodes.iter().any(|n| self.support_nodes.contains(n)),
                target_state,
                bodies,
            );
            result.stats.sleep_init_ms += sleep_started_at.elapsed().as_secs_f64() as f32 * 1_000.0;
        }

        // Reconcile each dynamic child's velocity with Rapier's real centre of mass, in a
        // SEPARATE pass after every child's colliders have been migrated (so the COM is final).
        for &(child_index, target) in &ordered_children {
            let fc = child_target_states[&child_index].fit_center;
            let fit_center = vector![fc.x, fc.y, fc.z];
            self.reconcile_child_velocity_with_com(target.handle(), fit_center, bodies, colliders);
        }

        // Record split continuity in a SEPARATE pass, after every child's colliders have
        // been migrated — otherwise a reused parent body would still hold a sibling's
        // collider and report a spurious (and nondeterministic) centre of mass.
        if self.record_split_continuity {
            for &(child_index, target) in &ordered_children {
                let child = &event.children[child_index];
                self.record_split_child_continuity(target.handle(), child, &planning, bodies, colliders);
            }
        }

        let mut cohort_handles: Vec<RigidBodyHandle> =
            ordered_children.iter().map(|(_, target)| target.handle()).collect();
        cohort_handles.sort_unstable_by_key(|h| h.into_raw_parts());
        cohort_handles.dedup();
        result.cohort_handles = cohort_handles;

        let reused_handles: HashSet<RigidBodyHandle> = reused_by_body.keys().copied().collect();
        let recycled_handles: HashSet<RigidBodyHandle> = child_targets
            .values()
            .filter_map(|target| match target {
                PlannedBodyTarget::Recycled(handle) => Some(*handle),
                _ => None,
            })
            .collect();
        for &body_handle in &planning.parent_bodies {
            if reused_handles.contains(&body_handle) || recycled_handles.contains(&body_handle) {
                continue;
            }
            let retire_started_at = Instant::now();
            self.body_to_nodes.remove(&body_handle);
            self.body_metadata.remove(&body_handle);
            if bodies.get(body_handle).is_some() {
                bodies.remove(
                    body_handle,
                    island_manager,
                    colliders,
                    impulse_joints,
                    multibody_joints,
                    true,
                );
                result.stats.retired_bodies += 1;
            }
            result.stats.body_retire_ms +=
                retire_started_at.elapsed().as_secs_f64() as f32 * 1_000.0;
        }

        result.stats.apply_ms = apply_started_at.elapsed().as_secs_f64() as f32 * 1_000.0;
        result
    }

    /// Get the body handle for a node.
    pub fn node_body(&self, node_index: u32) -> Option<RigidBodyHandle> {
        self.node_to_body
            .get(node_index as usize)
            .copied()
            .flatten()
    }

    /// Get the collider handle for a node.
    pub fn node_collider(&self, node_index: u32) -> Option<ColliderHandle> {
        self.node_to_collider
            .get(node_index as usize)
            .copied()
            .flatten()
    }

    /// Get the node owning a collider.
    pub fn collider_node(&self, collider: ColliderHandle) -> Option<u32> {
        self.collider_to_node.get(&collider).copied()
    }

    /// Get the node local offset relative to its owning body.
    pub fn node_local_offset(&self, node_index: u32) -> Option<Vec3> {
        let idx = node_index as usize;
        if self.node_to_body.get(idx).copied().flatten().is_some() {
            self.node_local_offsets.get(idx).copied()
        } else {
            None
        }
    }

    /// Get the node indices attached to a body.
    pub fn body_nodes(&self, body_handle: RigidBodyHandle) -> Vec<u32> {
        self.body_to_nodes
            .get(&body_handle)
            .cloned()
            .unwrap_or_default()
    }

    pub fn body_nodes_slice(&self, body_handle: RigidBodyHandle) -> &[u32] {
        self.body_to_nodes
            .get(&body_handle)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn body_node_count(&self, body_handle: RigidBodyHandle) -> usize {
        self.body_to_nodes
            .get(&body_handle)
            .map(Vec::len)
            .unwrap_or(0)
    }

    pub fn body_has_support(&self, body_handle: RigidBodyHandle) -> bool {
        self.body_metadata
            .get(&body_handle)
            .map(|metadata| metadata.has_support)
            .unwrap_or(false)
    }

    /// Whether a node is a support (fixed) node.
    pub fn is_support(&self, node_index: u32) -> bool {
        self.support_nodes.contains(&node_index)
    }

    /// Number of tracked bodies.
    pub fn body_count(&self) -> usize {
        self.body_to_nodes.len()
    }

    /// Number of tracked colliders owned by destructible nodes.
    pub fn collider_count(&self) -> usize {
        self.node_to_collider
            .iter()
            .filter(|handle| handle.is_some())
            .count()
    }

    /// Number of dynamic bodies.
    pub fn dynamic_body_count(&self, bodies: &RigidBodySet) -> usize {
        self.body_to_nodes
            .keys()
            .filter(|h| bodies.get(**h).map(|b| b.is_dynamic()).unwrap_or(false))
            .count()
    }

    pub fn awake_dynamic_body_count(&self, bodies: &RigidBodySet) -> usize {
        self.body_to_nodes
            .keys()
            .filter(|h| {
                bodies
                    .get(**h)
                    .map(|b| b.is_dynamic() && !b.is_sleeping())
                    .unwrap_or(false)
            })
            .count()
    }

    pub fn sleeping_dynamic_body_count(&self, bodies: &RigidBodySet) -> usize {
        self.body_to_nodes
            .keys()
            .filter(|h| {
                bodies
                    .get(**h)
                    .map(|b| b.is_dynamic() && b.is_sleeping())
                    .unwrap_or(false)
            })
            .count()
    }

    pub fn support_body_count(&self) -> usize {
        self.body_metadata
            .values()
            .filter(|metadata| metadata.has_support)
            .count()
    }

    pub fn ccd_enabled_body_count(&self, bodies: &RigidBodySet) -> usize {
        self.body_to_nodes
            .keys()
            .filter(|h| bodies.get(**h).map(|b| b.is_ccd_enabled()).unwrap_or(false))
            .count()
    }

    pub fn estimate_split_cost(&self, event: &SplitEvent, bodies: &RigidBodySet) -> SplitCost {
        let existing = self.collect_existing_body_states(event, bodies);
        let child_support: Vec<super::split_migrator::PlannerChildSupport> = event
            .children
            .iter()
            .map(|child| super::split_migrator::PlannerChildSupport {
                is_support: child.nodes.iter().any(|n| self.support_nodes.contains(n)),
            })
            .collect();
        let plan = super::split_migrator::plan_split_migration_with_support(
            &existing,
            &event.children,
            &child_support,
        );
        let recyclable_bodies = existing.len().saturating_sub(plan.reuse.len());
        SplitCost {
            create_bodies: plan.create.len().saturating_sub(recyclable_bodies),
            collider_migrations: event.children.iter().map(|child| child.nodes.len()).sum(),
        }
    }

    pub fn mark_support_contact(
        &mut self,
        body_handle: RigidBodyHandle,
        now_secs: f32,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        damping: SmallBodyDampingOptions,
        sleep_thresholds: SleepThresholdOptions,
        max_colliders_for_debris: usize,
    ) -> bool {
        let collider_count = self.body_node_count(body_handle);
        let Some(metadata) = self.body_metadata.get_mut(&body_handle) else {
            return false;
        };
        if metadata.has_support {
            return false;
        }
        metadata.first_support_contact_at.get_or_insert(now_secs);
        let mut changed = false;
        if damping.mode == OptimizationMode::AfterGroundCollision
            && !metadata.damping_promoted
            && collider_count <= damping.collider_count_threshold
        {
            if Self::apply_small_body_damping(bodies, body_handle, &damping) {
                metadata.damping_promoted = true;
                changed = true;
            }
        }
        if sleep_thresholds.mode == OptimizationMode::AfterGroundCollision
            && !metadata.sleep_threshold_configured
        {
            if Self::apply_sleep_thresholds(bodies, body_handle, &sleep_thresholds) {
                metadata.sleep_threshold_configured = true;
                changed = true;
            }
        }
        if metadata.first_support_contact_at == Some(now_secs)
            && self.debris_collision_mode != DebrisCollisionMode::All
        {
            self.apply_collision_groups(body_handle, bodies, colliders, max_colliders_for_debris);
            changed = true;
        }
        changed
    }

    pub fn cleanup_expired_bodies(
        &mut self,
        now_secs: f32,
        cleanup: DebrisCleanupOptions,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        island_manager: &mut IslandManager,
        impulse_joints: &mut ImpulseJointSet,
        multibody_joints: &mut MultibodyJointSet,
    ) -> OptimizationResult {
        let mut result = OptimizationResult::default();
        if cleanup.mode == OptimizationMode::Off {
            return result;
        }

        let mut expired = Vec::new();
        for (&body_handle, metadata) in &self.body_metadata {
            if metadata.has_support {
                continue;
            }
            let Some(body) = bodies.get(body_handle) else {
                continue;
            };
            if !body.is_dynamic() {
                continue;
            }
            if self.body_node_count(body_handle) > cleanup.max_colliders_for_debris {
                continue;
            }

            let anchor = match cleanup.mode {
                OptimizationMode::Off => None,
                OptimizationMode::Always => Some(metadata.created_at),
                OptimizationMode::AfterGroundCollision => metadata.first_support_contact_at,
            };
            let Some(anchor) = anchor else {
                continue;
            };
            if cleanup.mode == OptimizationMode::AfterGroundCollision && !body.is_sleeping() {
                continue;
            }
            if now_secs - anchor >= cleanup.debris_ttl_secs {
                expired.push(body_handle);
            }
        }

        for body_handle in expired {
            let removed_nodes = self.remove_body(
                body_handle,
                bodies,
                colliders,
                island_manager,
                impulse_joints,
                multibody_joints,
            );
            if !removed_nodes.is_empty() {
                result.removed_bodies.push(body_handle);
                result.removed_nodes.extend(removed_nodes);
            }
        }

        result
    }

    pub fn destroy_nodes(
        &mut self,
        nodes: &[u32],
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        island_manager: &mut IslandManager,
        impulse_joints: &mut ImpulseJointSet,
        multibody_joints: &mut MultibodyJointSet,
    ) -> Vec<u32> {
        let mut removed = Vec::new();
        let mut affected_bodies = HashSet::new();

        for &node_index in nodes {
            let idx = node_index as usize;
            let Some(body_handle) = self.node_to_body[idx] else {
                continue;
            };
            if let Some(collider_handle) = self.node_to_collider[idx] {
                colliders.remove(collider_handle, island_manager, bodies, true);
            }
            self.detach_node(node_index);
            if let Some(body_nodes) = self.body_to_nodes.get_mut(&body_handle) {
                body_nodes.retain(|node| *node != node_index);
            }
            affected_bodies.insert(body_handle);
            removed.push(node_index);
        }

        for body_handle in affected_bodies {
            let should_remove = self
                .body_to_nodes
                .get(&body_handle)
                .map(|nodes| nodes.is_empty())
                .unwrap_or(true);
            if should_remove {
                self.body_to_nodes.remove(&body_handle);
                self.body_metadata.remove(&body_handle);
                if bodies.get(body_handle).is_some() {
                    bodies.remove(
                        body_handle,
                        island_manager,
                        colliders,
                        impulse_joints,
                        multibody_joints,
                        true,
                    );
                }
            } else if let Some(metadata) = self.body_metadata.get_mut(&body_handle) {
                metadata.has_support = self
                    .body_to_nodes
                    .get(&body_handle)
                    .map(|nodes| nodes.iter().any(|node| self.support_nodes.contains(node)))
                    .unwrap_or(false);
            }
        }

        removed
    }

    fn compute_actor_com(&self, nodes: &[u32]) -> Vec3 {
        if nodes.is_empty() {
            return Vec3::ZERO;
        }
        let mut sum = Vec3::ZERO;
        for &n in nodes {
            sum += self.node_centroids[n as usize];
        }
        sum / nodes.len() as f32
    }

    fn compute_child_target_state(
        &self,
        child: &SplitChild,
        planning: &SplitPlanningData,
        has_support: bool,
        reuse: bool,
    ) -> (ChildTargetState, f32, f32) {
        let pose_started_at = Instant::now();
        // `fit_center` is the fragment's node-model COM — the reference the velocity fit and the
        // COM reconciliation use. It is computed the same way regardless of reuse, so the velocity
        // result is identical to the recentered behavior.
        let fit_center = if has_support {
            self.child_world_centroid(child, planning)
        } else {
            self.child_world_center_of_mass(child, planning)
        };
        // The body pose: recenter only NEWLY created/recycled bodies (they are absent from any
        // resim snapshot). A REUSED body keeps its inherited origin so resim rollback can't
        // teleport it; its velocity is still reconciled to the real COM via `fit_center` below.
        let pose = if self.split_child_recentering_enabled && !reuse {
            self.compute_child_target_pose(child, planning, has_support)
        } else {
            self.inherited_child_target_pose(child, planning)
        };
        let child_pose_ms = pose_started_at.elapsed().as_secs_f64() as f32 * 1_000.0;
        let velocity_fit_started_at = Instant::now();
        let (linvel, angvel, should_sleep) = if self.split_child_velocity_fit_enabled {
            self.fit_child_motion(child, planning, has_support, fit_center)
        } else {
            self.inherited_child_motion(child, planning, has_support)
        };
        let velocity_fit_ms = velocity_fit_started_at.elapsed().as_secs_f64() as f32 * 1_000.0;
        (
            ChildTargetState {
                pose,
                fit_center,
                linvel,
                angvel,
                should_sleep,
            },
            child_pose_ms,
            velocity_fit_ms,
        )
    }

    fn inherited_child_target_pose(
        &self,
        child: &SplitChild,
        planning: &SplitPlanningData,
    ) -> Isometry<Real> {
        let Some(body_handle) = self.dominant_child_body_handle(child, planning) else {
            return Isometry::identity();
        };
        planning
            .parent_states
            .get(&body_handle)
            .map(|state| state.position)
            .unwrap_or_else(Isometry::identity)
    }

    fn compute_child_target_pose(
        &self,
        child: &SplitChild,
        planning: &SplitPlanningData,
        has_support: bool,
    ) -> Isometry<Real> {
        let translation = if has_support {
            self.child_world_centroid(child, planning)
        } else {
            self.child_world_center_of_mass(child, planning)
        };
        let rotation = self.dominant_child_rotation(child, planning);
        Isometry::from_parts(
            Translation::new(translation.x, translation.y, translation.z),
            rotation,
        )
    }

    fn child_world_centroid(&self, child: &SplitChild, planning: &SplitPlanningData) -> Vec3 {
        let mut sum = Vec3::ZERO;
        let mut count = 0usize;
        for &node in &child.nodes {
            if let Some(source) = planning.node_sources.get(&node) {
                sum += source.world_centroid;
                count += 1;
            }
        }
        if count == 0 {
            self.compute_actor_com(&child.nodes)
        } else {
            sum / count as f32
        }
    }

    fn child_world_center_of_mass(&self, child: &SplitChild, planning: &SplitPlanningData) -> Vec3 {
        let samples: Vec<(Vec3, f32)> = child
            .nodes
            .iter()
            .filter_map(|&node| {
                planning.node_sources.get(&node).map(|source| {
                    (
                        source.world_centroid,
                        self.node_masses[node as usize].max(0.0),
                    )
                })
            })
            .collect();
        super::motion_fit::weighted_center_of_mass(&samples)
            .unwrap_or_else(|| self.child_world_centroid(child, planning))
    }

    fn dominant_child_rotation(
        &self,
        child: &SplitChild,
        planning: &SplitPlanningData,
    ) -> UnitQuaternion<Real> {
        let Some(body_handle) = self.dominant_child_body_handle(child, planning) else {
            return UnitQuaternion::identity();
        };
        planning
            .parent_states
            .get(&body_handle)
            .map(|state| state.position.rotation)
            .unwrap_or_else(UnitQuaternion::identity)
    }

    fn dominant_child_body_handle(
        &self,
        child: &SplitChild,
        planning: &SplitPlanningData,
    ) -> Option<RigidBodyHandle> {
        let mut body_weights: HashMap<RigidBodyHandle, f32> = HashMap::new();
        for &node in &child.nodes {
            let Some(source) = planning.node_sources.get(&node) else {
                continue;
            };
            *body_weights.entry(source.body_handle).or_insert(0.0) +=
                self.node_masses[node as usize].max(1.0);
        }
        // Pick the heaviest source body, breaking weight ties by lowest handle so the
        // dominant-body choice (which sets the child's inherited rotation/velocity) is
        // deterministic rather than dependent on `HashMap` iteration order.
        let best = body_weights.iter().max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.into_raw_parts().cmp(&a.0.into_raw_parts()))
        });
        best.map(|(&handle, _)| handle)
    }

    fn fit_child_motion(
        &self,
        child: &SplitChild,
        planning: &SplitPlanningData,
        has_support: bool,
        fit_center: Vec3,
    ) -> (Vector<Real>, Vector<Real>, bool) {
        if has_support {
            return (Vector::zeros(), Vector::zeros(), false);
        }

        let child_com = fit_center;
        let mut all_sleeping = true;
        let mut samples: Vec<(Vec3, Vec3, f32)> = Vec::new();
        let mut source_weights: HashMap<RigidBodyHandle, f32> = HashMap::new();

        for &node in &child.nodes {
            let Some(source) = planning.node_sources.get(&node) else {
                continue;
            };
            let Some(parent_state) = planning.parent_states.get(&source.body_handle) else {
                continue;
            };
            let mass = self.node_masses[node as usize].max(1.0e-4);
            all_sleeping &= parent_state.was_sleeping;
            *source_weights.entry(source.body_handle).or_insert(0.0) += mass;
            samples.push((source.world_centroid, source.world_velocity, mass));
        }

        // Mass/COM/inertia fit is delegated to the pure kernel so it can be
        // property-tested in isolation (see `super::motion_fit`).
        let Some(fit) = super::motion_fit::fit_rigid_motion(&samples, child_com) else {
            return (Vector::zeros(), Vector::zeros(), false);
        };
        let linvel = vector![fit.linvel.x, fit.linvel.y, fit.linvel.z];

        let dominant_source = source_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .and_then(|(handle, _)| planning.parent_states.get(handle));

        let angvel = fit
            .angvel
            .map(|omega| vector![omega.x, omega.y, omega.z])
            .or_else(|| dominant_source.map(|state| state.angvel))
            .unwrap_or_else(Vector::zeros);

        let should_sleep =
            all_sleeping && linvel.norm_squared() <= 1.0e-4 && angvel.norm_squared() <= 1.0e-4;

        (linvel, angvel, should_sleep)
    }

    fn inherited_child_motion(
        &self,
        child: &SplitChild,
        planning: &SplitPlanningData,
        has_support: bool,
    ) -> (Vector<Real>, Vector<Real>, bool) {
        if has_support {
            return (Vector::zeros(), Vector::zeros(), false);
        }
        let Some(body_handle) = self.dominant_child_body_handle(child, planning) else {
            return (Vector::zeros(), Vector::zeros(), false);
        };
        let Some(state) = planning.parent_states.get(&body_handle) else {
            return (Vector::zeros(), Vector::zeros(), false);
        };
        let should_sleep = state.was_sleeping
            && state.linvel.norm_squared() <= 1.0e-4
            && state.angvel.norm_squared() <= 1.0e-4;
        (state.linvel, state.angvel, should_sleep)
    }

    fn desired_body_type(has_support: bool) -> RigidBodyType {
        if has_support {
            RigidBodyType::Fixed
        } else {
            RigidBodyType::Dynamic
        }
    }

    fn reconcile_reused_body_type(
        &self,
        body_handle: RigidBodyHandle,
        has_support: bool,
        bodies: &mut RigidBodySet,
    ) -> bool {
        let desired = Self::desired_body_type(has_support);
        let Some(body) = bodies.get_mut(body_handle) else {
            return false;
        };
        if body.body_type() == desired {
            return false;
        }
        body.set_body_type(desired, true);
        if desired == RigidBodyType::Dynamic {
            body.wake_up(true);
        }
        true
    }

    fn reinitialize_recycled_body(
        &self,
        body_handle: RigidBodyHandle,
        has_support: bool,
        target: ChildTargetState,
        bodies: &mut RigidBodySet,
    ) -> bool {
        let desired = Self::desired_body_type(has_support);
        let Some(body) = bodies.get_mut(body_handle) else {
            return false;
        };
        let flipped = body.body_type() != desired;
        if flipped {
            body.set_body_type(desired, true);
        }
        body.set_position(target.pose, false);
        if desired == RigidBodyType::Dynamic {
            body.set_linvel(target.linvel, false);
            body.set_angvel(target.angvel, false);
            if target.should_sleep {
                body.sleep();
            } else {
                body.wake_up(true);
            }
        }
        flipped
    }

    fn create_body_for_child(
        &self,
        child: &SplitChild,
        target: ChildTargetState,
        bodies: &mut RigidBodySet,
    ) -> RigidBodyHandle {
        let has_support = child.nodes.iter().any(|n| self.support_nodes.contains(n));
        if has_support {
            bodies.insert(RigidBodyBuilder::fixed().pose(target.pose))
        } else {
            let handle = bodies.insert(
                RigidBodyBuilder::dynamic()
                    .pose(target.pose)
                    .linvel(target.linvel)
                    .angvel(target.angvel)
                    .ccd_enabled(self.dynamic_body_ccd_enabled),
            );
            if target.should_sleep {
                if let Some(body) = bodies.get_mut(handle) {
                    body.sleep();
                }
            }
            handle
        }
    }

    fn apply_child_kinematics(
        &self,
        body_handle: RigidBodyHandle,
        has_support: bool,
        target: ChildTargetState,
        bodies: &mut RigidBodySet,
    ) {
        let Some(body) = bodies.get_mut(body_handle) else {
            return;
        };
        body.set_position(target.pose, false);
        if has_support || !body.is_dynamic() {
            return;
        }
        let current_linvel = *body.linvel();
        let current_angvel = *body.angvel();
        let velocity_changed = (current_linvel - target.linvel).norm_squared() > 1.0e-6
            || (current_angvel - target.angvel).norm_squared() > 1.0e-6;
        body.set_linvel(target.linvel, false);
        body.set_angvel(target.angvel, false);
        if target.should_sleep {
            body.sleep();
        } else if velocity_changed || body.is_sleeping() {
            body.wake_up(true);
        }
    }

    /// Reconcile a freshly-split dynamic child's velocity with Rapier's ACTUAL centre of mass.
    ///
    /// `fit_child_motion` expresses the child's motion about `fit_center` (the body origin =
    /// mass-weighted node-centroid model). Rapier, however, integrates rotation about the
    /// collider-derived centre of mass, which differs for offset shapes (e.g. convex-hull
    /// fragments). Left uncorrected, a rotating fragment gains spurious velocity
    /// `omega x (rapier_com - fit_center)` — the "sudden movement after destruction" bug
    /// (gap #1). Shifting `linvel` by exactly that term makes the body's velocity field match
    /// the fitted (parent-continuous) field at the node points.
    fn reconcile_child_velocity_with_com(
        &self,
        body_handle: RigidBodyHandle,
        fit_center: Vector<Real>,
        bodies: &mut RigidBodySet,
        colliders: &ColliderSet,
    ) {
        let Some(body) = bodies.get_mut(body_handle) else {
            return;
        };
        if !body.is_dynamic() {
            return;
        }
        // Collider migration updates mass properties lazily; force a recompute so the COM is
        // current before we read it.
        body.recompute_mass_properties_from_colliders(colliders);
        let real_com = body.center_of_mass().coords;
        let lever = real_com - fit_center;
        if lever.norm_squared() <= f32::EPSILON {
            return;
        }
        let angvel = *body.angvel();
        let linvel = *body.linvel();
        body.set_linvel(linvel + angvel.cross(&lever), false);
    }

    /// Observability hook (only invoked when `record_split_continuity` is on):
    /// for each node migrated to `target_body`, measure how far the child body's
    /// world point velocity drifts from the parent's at the same point. Mass
    /// properties are recomputed first so `velocity_at_point` uses Rapier's true
    /// centre of mass — which is precisely where the centroid-model split diverges.
    fn record_split_child_continuity(
        &mut self,
        target_body: RigidBodyHandle,
        child: &SplitChild,
        planning: &SplitPlanningData,
        bodies: &mut RigidBodySet,
        colliders: &ColliderSet,
    ) {
        if let Some(body) = bodies.get_mut(target_body) {
            body.recompute_mass_properties_from_colliders(colliders);
        }
        let Some(body) = bodies.get(target_body) else {
            return;
        };
        if !body.is_dynamic() {
            return;
        }
        for &node in &child.nodes {
            let Some(source) = planning.node_sources.get(&node) else {
                continue;
            };
            let p = point![
                source.world_centroid.x,
                source.world_centroid.y,
                source.world_centroid.z
            ];
            let cv = body.velocity_at_point(&p);
            let child_v = Vec3::new(cv.x, cv.y, cv.z);
            let parent_v = source.world_velocity;
            // Where the node actually sits now, on its (possibly recentered) child body.
            let local = self.node_local_offsets[node as usize];
            let cw = body
                .position()
                .transform_point(&point![local.x, local.y, local.z]);
            let child_world = Vec3::new(cw.x, cw.y, cw.z);
            let world_position_error = (child_world - source.world_centroid).magnitude();
            let finite = child_v.x.is_finite()
                && child_v.y.is_finite()
                && child_v.z.is_finite()
                && child_world.x.is_finite()
                && child_world.y.is_finite()
                && child_world.z.is_finite();
            self.split_continuity.push(SplitContinuityRecord {
                target_body,
                node_index: node,
                point_velocity_error: (child_v - parent_v).magnitude(),
                parent_point_velocity: parent_v,
                child_point_velocity: child_v,
                world_position_error,
                finite,
            });
        }
    }

    fn attach_node(
        &mut self,
        node_index: u32,
        body: RigidBodyHandle,
        collider: ColliderHandle,
        local_offset: Vec3,
    ) {
        let idx = node_index as usize;
        self.node_to_body[idx] = Some(body);
        self.node_to_collider[idx] = Some(collider);
        self.collider_to_node.insert(collider, node_index);
        self.node_local_offsets[idx] = local_offset;
    }

    fn detach_node(&mut self, node_index: u32) {
        let idx = node_index as usize;
        if let Some(collider) = self.node_to_collider[idx].take() {
            self.collider_to_node.remove(&collider);
        }
        self.node_to_body[idx] = None;
        self.node_local_offsets[idx] = Vec3::ZERO;
    }

    fn update_body_metadata(
        &mut self,
        body_handle: RigidBodyHandle,
        nodes: &[u32],
        created_at: f32,
    ) {
        let has_support = nodes.iter().any(|node| self.support_nodes.contains(node));
        self.body_metadata.insert(
            body_handle,
            BodyMetadata {
                created_at,
                first_support_contact_at: None,
                damping_promoted: false,
                sleep_threshold_configured: false,
                has_support,
            },
        );
    }

    fn refresh_body_metadata(
        &mut self,
        body_handle: RigidBodyHandle,
        nodes: &[u32],
        created_at: f32,
        reset: bool,
    ) {
        if reset || !self.body_metadata.contains_key(&body_handle) {
            self.update_body_metadata(body_handle, nodes, created_at);
            return;
        }

        let has_support = nodes.iter().any(|node| self.support_nodes.contains(node));
        if let Some(metadata) = self.body_metadata.get_mut(&body_handle) {
            let support_changed = metadata.has_support != has_support;
            metadata.has_support = has_support;
            if support_changed && !has_support {
                metadata.first_support_contact_at = None;
                metadata.damping_promoted = false;
                metadata.sleep_threshold_configured = false;
            }
        }
    }

    fn promote_small_body_damping(
        &mut self,
        body_handle: RigidBodyHandle,
        bodies: &mut RigidBodySet,
        damping: &SmallBodyDampingOptions,
    ) {
        if damping.mode != OptimizationMode::Always {
            return;
        }
        if self.body_node_count(body_handle) > damping.collider_count_threshold {
            return;
        }
        if Self::apply_small_body_damping(bodies, body_handle, damping) {
            if let Some(metadata) = self.body_metadata.get_mut(&body_handle) {
                metadata.damping_promoted = true;
            }
        }
    }

    fn configure_sleep_thresholds(
        &mut self,
        body_handle: RigidBodyHandle,
        bodies: &mut RigidBodySet,
        sleep_thresholds: &SleepThresholdOptions,
    ) {
        if sleep_thresholds.mode != OptimizationMode::Always {
            return;
        }
        if Self::apply_sleep_thresholds(bodies, body_handle, sleep_thresholds) {
            if let Some(metadata) = self.body_metadata.get_mut(&body_handle) {
                metadata.sleep_threshold_configured = true;
            }
        }
    }

    fn apply_small_body_damping(
        bodies: &mut RigidBodySet,
        body_handle: RigidBodyHandle,
        damping: &SmallBodyDampingOptions,
    ) -> bool {
        let Some(body) = bodies.get_mut(body_handle) else {
            return false;
        };
        if !body.is_dynamic() {
            return false;
        }
        let mut changed = false;
        if body.linear_damping() < damping.min_linear_damping {
            body.set_linear_damping(damping.min_linear_damping);
            changed = true;
        }
        if body.angular_damping() < damping.min_angular_damping {
            body.set_angular_damping(damping.min_angular_damping);
            changed = true;
        }
        if changed {
            body.wake_up(true);
        }
        changed
    }

    fn apply_sleep_thresholds(
        bodies: &mut RigidBodySet,
        body_handle: RigidBodyHandle,
        sleep_thresholds: &SleepThresholdOptions,
    ) -> bool {
        let Some(body) = bodies.get_mut(body_handle) else {
            return false;
        };
        if !body.is_dynamic() {
            return false;
        }
        let activation = body.activation_mut();
        let desired_linear = sleep_thresholds.linear_threshold.max(0.0);
        let desired_angular = sleep_thresholds.angular_threshold.max(0.0);
        let changed = (activation.normalized_linear_threshold - desired_linear).abs()
            > Real::EPSILON
            || (activation.angular_threshold - desired_angular).abs() > Real::EPSILON;
        activation.normalized_linear_threshold = desired_linear;
        activation.angular_threshold = desired_angular;
        if changed {
            body.wake_up(true);
        }
        changed
    }

    fn remove_body(
        &mut self,
        body_handle: RigidBodyHandle,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
        island_manager: &mut IslandManager,
        impulse_joints: &mut ImpulseJointSet,
        multibody_joints: &mut MultibodyJointSet,
    ) -> Vec<u32> {
        let nodes = self.body_to_nodes.remove(&body_handle).unwrap_or_default();
        for node in &nodes {
            self.detach_node(*node);
        }
        self.body_metadata.remove(&body_handle);
        if bodies.get(body_handle).is_some() {
            bodies.remove(
                body_handle,
                island_manager,
                colliders,
                impulse_joints,
                multibody_joints,
                true,
            );
        }
        nodes
    }

    fn collect_existing_body_states(
        &self,
        event: &SplitEvent,
        bodies: &RigidBodySet,
    ) -> Vec<super::split_migrator::ExistingBodyState> {
        let mut parent_bodies = HashSet::new();
        for child in &event.children {
            for &node in &child.nodes {
                if let Some(handle) = self.node_body(node) {
                    parent_bodies.insert(handle);
                }
            }
        }

        parent_bodies
            .into_iter()
            .map(|handle| super::split_migrator::ExistingBodyState {
                handle,
                node_indices: self
                    .body_to_nodes
                    .get(&handle)
                    .map(|nodes| nodes.iter().copied().collect())
                    .unwrap_or_default(),
                is_fixed: bodies.get(handle).map(|b| !b.is_dynamic()).unwrap_or(false),
            })
            .collect()
    }

    fn collect_split_planning_data(
        &self,
        event: &SplitEvent,
        bodies: &RigidBodySet,
    ) -> SplitPlanningData {
        let mut parent_bodies = HashSet::new();
        let mut node_sources = HashMap::new();

        for child in &event.children {
            for &node in &child.nodes {
                let idx = node as usize;
                let Some(body_handle) = self.node_to_body.get(idx).copied().flatten() else {
                    continue;
                };
                let Some(body) = bodies.get(body_handle) else {
                    continue;
                };
                let local_offset = self.node_local_offsets[idx];
                let world = body.position().transform_point(&point![
                    local_offset.x,
                    local_offset.y,
                    local_offset.z
                ]);
                let world_velocity = body.velocity_at_point(&world);
                parent_bodies.insert(body_handle);
                node_sources.entry(node).or_insert(NodeSourceState {
                    body_handle,
                    collider_handle: self.node_to_collider[idx],
                    world_centroid: Vec3::new(world.x, world.y, world.z),
                    world_velocity: Vec3::new(world_velocity.x, world_velocity.y, world_velocity.z),
                });
            }
        }

        let mut parent_bodies: Vec<RigidBodyHandle> = parent_bodies.into_iter().collect();
        // Sort for determinism: body recycling/creation order (below) must not depend on
        // `HashSet` iteration order, otherwise Rapier allocates handles in a nondeterministic
        // order and its island/contact solver diverges run-to-run (amplified by chaotic
        // fracture). A stable order makes the whole pipeline reproducible.
        parent_bodies.sort_unstable_by_key(|h| h.into_raw_parts());
        let parent_states = parent_bodies
            .iter()
            .filter_map(|&handle| {
                let body = bodies.get(handle)?;
                Some((
                    handle,
                    BodyState {
                        position: *body.position(),
                        linvel: *body.linvel(),
                        angvel: *body.angvel(),
                        was_sleeping: body.is_sleeping(),
                    },
                ))
            })
            .collect();

        SplitPlanningData {
            existing: self.collect_existing_body_states(event, bodies),
            parent_bodies,
            parent_states,
            node_sources,
        }
    }

    fn local_offset_for_pose(
        &self,
        source: &NodeSourceState,
        target_pose: &Isometry<Real>,
    ) -> Vec3 {
        let local = target_pose.inverse_transform_point(&point![
            source.world_centroid.x,
            source.world_centroid.y,
            source.world_centroid.z
        ]);
        Vec3::new(local.x, local.y, local.z)
    }

    fn insert_node_collider(
        &self,
        node_index: u32,
        body_handle: RigidBodyHandle,
        local_offset: Vec3,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
    ) -> ColliderHandle {
        let collider = self.build_node_collider(node_index, local_offset);
        colliders.insert_with_parent(collider, body_handle, bodies)
    }

    fn build_node_collider(&self, node_index: u32, local_offset: Vec3) -> ColliderBuilder {
        let ni = node_index as usize;
        let fallback_size = self.node_sizes[ni];
        let fallback_half = vector![
            fallback_size.x * 0.5,
            fallback_size.y * 0.5,
            fallback_size.z * 0.5
        ];
        let base = match self.node_colliders.get(ni).and_then(Option::as_ref) {
            Some(ScenarioCollider::Cuboid { half_extents }) => {
                ColliderBuilder::cuboid(half_extents.x, half_extents.y, half_extents.z)
            }
            Some(ScenarioCollider::ConvexHull { points }) => {
                let hull_points: Vec<Point<Real>> = points
                    .iter()
                    .map(|point| point![point.x, point.y, point.z])
                    .collect();
                ColliderBuilder::convex_hull(&hull_points).unwrap_or_else(|| {
                    ColliderBuilder::cuboid(fallback_half.x, fallback_half.y, fallback_half.z)
                })
            }
            None => ColliderBuilder::cuboid(fallback_half.x, fallback_half.y, fallback_half.z),
        };

        let collider = base
            .translation(vector![local_offset.x, local_offset.y, local_offset.z])
            .active_events(ActiveEvents::CONTACT_FORCE_EVENTS | ActiveEvents::COLLISION_EVENTS)
            .active_hooks(ActiveHooks::FILTER_CONTACT_PAIRS | ActiveHooks::FILTER_INTERSECTION_PAIR)
            .contact_force_event_threshold(0.0)
            .friction(0.25)
            .restitution(0.0);

        if self.support_nodes.contains(&node_index) {
            collider.mass(0.0)
        } else {
            collider.mass(self.node_masses[ni].max(0.0))
        }
    }

    fn move_node_collider_to_body(
        &self,
        node_index: u32,
        body_handle: RigidBodyHandle,
        collider_handle: Option<ColliderHandle>,
        local_offset: Vec3,
        bodies: &mut RigidBodySet,
        colliders: &mut ColliderSet,
    ) -> ColliderHandle {
        if let Some(collider_handle) = collider_handle {
            colliders.set_parent(collider_handle, Some(body_handle), bodies);
            if let Some(collider) = colliders.get_mut(collider_handle) {
                collider.set_position_wrt_parent(Isometry::translation(
                    local_offset.x,
                    local_offset.y,
                    local_offset.z,
                ));
            }
            collider_handle
        } else {
            self.insert_node_collider(node_index, body_handle, local_offset, bodies, colliders)
        }
    }

    fn update_node_collider_local_pose(
        &self,
        collider_handle: Option<ColliderHandle>,
        local_offset: Vec3,
        colliders: &mut ColliderSet,
    ) -> ColliderHandle {
        let collider_handle = collider_handle.expect("existing node collider should exist");
        if let Some(collider) = colliders.get_mut(collider_handle) {
            collider.set_position_wrt_parent(Isometry::translation(
                local_offset.x,
                local_offset.y,
                local_offset.z,
            ));
        }
        collider_handle
    }

    fn apply_collision_groups(
        &self,
        body_handle: RigidBodyHandle,
        bodies: &RigidBodySet,
        colliders: &mut ColliderSet,
        max_colliders_for_debris: usize,
    ) {
        apply_collision_groups_for_body(
            body_handle,
            bodies,
            colliders,
            self.ground_body_handle,
            self.debris_collision_mode,
            self.debris_collision_active(body_handle),
            max_colliders_for_debris,
        );
    }

    fn debris_collision_active(&self, body_handle: RigidBodyHandle) -> bool {
        self.body_metadata
            .get(&body_handle)
            .and_then(|metadata| metadata.first_support_contact_at)
            .is_some()
    }
}
