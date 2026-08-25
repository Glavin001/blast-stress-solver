//! Many structures, one backend.
//!
//! A city is **N instances of one pack at grid offsets**, not a single
//! multi-structure pack — every shipped pack carries one `scenario` and no
//! `structures` array. So this is a placement layer, not a new format.
//!
//! Each structure keeps its own solver and its own body set; they interact only
//! through the physics engine, which is exactly right — two buildings share no
//! bonds, and a fracture in one must not touch the other's graph.

use crate::backend::PhysicsBackend;
use crate::pipeline::destructible::{Destructible, DestructibleConfig, IslandMotion, StepReport};
use crate::pipeline::events::DestructionEvent;
use crate::types::ScenarioDesc;

/// A structure's stable identity within the set.
///
/// The host owns the numbering; the library only requires that it is stable,
/// because every chunk and bond id is derived from it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct StructureId(pub u32);

/// Aggregate of one step across every structure.
#[derive(Clone, Debug, Default)]
pub struct SetStepReport {
    pub per_structure: Vec<(StructureId, StepReport)>,
    pub fractures: usize,
    pub bond_damage_events: usize,
    pub split_events: usize,
    pub bodies_created: usize,
    pub shapes_reparented: usize,
    pub bodies_retired: usize,
    pub writes_elided: usize,
}

pub struct DestructibleSet<B: PhysicsBackend> {
    structures: Vec<(StructureId, Destructible<B>)>,
}

impl<B: PhysicsBackend> Default for DestructibleSet<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: PhysicsBackend> DestructibleSet<B> {
    pub fn new() -> Self {
        Self { structures: Vec::new() }
    }

    /// Instantiate one structure at the pose in `cfg`.
    pub fn attach(
        &mut self,
        backend: &mut B,
        id: StructureId,
        scenario: &ScenarioDesc,
        cfg: DestructibleConfig,
    ) -> Result<(), AttachError> {
        if self.structures.iter().any(|(existing, _)| *existing == id) {
            return Err(AttachError::DuplicateId(id));
        }
        let d = Destructible::attach(backend, scenario, cfg).ok_or(AttachError::Failed(id))?;
        self.structures.push((id, d));
        // Kept sorted so iteration order — and therefore body-creation order —
        // is reproducible regardless of attach order.
        self.structures.sort_by_key(|(sid, _)| sid.0);
        Ok(())
    }

    /// Step every structure. Returns per-structure reports plus the totals.
    pub fn step(&mut self, backend: &mut B, dt: f32) -> SetStepReport {
        let mut out = SetStepReport::default();
        for (id, d) in self.structures.iter_mut() {
            let r = d.step(backend, dt);
            out.fractures += r.fractures;
            out.bond_damage_events += r.bond_damage_events;
            out.split_events += r.split_events;
            out.bodies_created += r.bodies_created;
            out.shapes_reparented += r.shapes_reparented;
            out.bodies_retired += r.bodies_retired;
            out.writes_elided += r.writes_elided;
            out.per_structure.push((*id, r));
        }
        out
    }

    /// Drain every structure's events, tagged with which structure they came
    /// from.
    ///
    /// Structure order is the sorted attach order, and each structure's events
    /// stay contiguous and in their own causal order. That matters: the
    /// ordering contract (promoted before migrated-onto, retired only when
    /// empty) is per-structure, and interleaving would break it for a consumer
    /// that applies the stream linearly.
    ///
    /// Island serials are per-structure, so a consumer must key on the pair.
    /// That is deliberate -- a global serial space would make every structure's
    /// numbering depend on every other structure's damage history, which is
    /// exactly what makes a late joiner unable to reconstruct state.
    pub fn drain_events(&mut self) -> Vec<(StructureId, DestructionEvent)> {
        let mut out = Vec::new();
        for (id, d) in self.structures.iter_mut() {
            out.extend(d.drain_events().into_iter().map(|e| (*id, e)));
        }
        out
    }

    /// Live COM-frame motion of every island in every structure.
    pub fn island_poses(&self, backend: &B) -> Vec<(StructureId, IslandMotion)> {
        let mut out = Vec::new();
        for (id, d) in self.structures.iter() {
            out.extend(d.island_poses(backend).into_iter().map(|m| (*id, m)));
        }
        out
    }

    pub fn len(&self) -> usize {
        self.structures.len()
    }
    pub fn is_empty(&self) -> bool {
        self.structures.is_empty()
    }
    pub fn ids(&self) -> Vec<StructureId> {
        self.structures.iter().map(|(id, _)| *id).collect()
    }
    pub fn get(&self, id: StructureId) -> Option<&Destructible<B>> {
        self.structures.iter().find(|(s, _)| *s == id).map(|(_, d)| d)
    }
    pub fn get_mut(&mut self, id: StructureId) -> Option<&mut Destructible<B>> {
        self.structures.iter_mut().find(|(s, _)| *s == id).map(|(_, d)| d)
    }
    pub fn iter(&self) -> impl Iterator<Item = (StructureId, &Destructible<B>)> {
        self.structures.iter().map(|(id, d)| (*id, d))
    }

    /// Total bodies across every structure. Must equal the summed actor count.
    pub fn body_count(&self) -> usize {
        self.structures.iter().map(|(_, d)| d.bodies().len()).sum()
    }

    pub fn actor_count(&self) -> u32 {
        self.structures.iter().map(|(_, d)| d.solver().actor_count()).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttachError {
    DuplicateId(StructureId),
    Failed(StructureId),
}

impl std::fmt::Display for AttachError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "structure {} is already attached", id.0),
            Self::Failed(id) => write!(f, "structure {} failed to attach", id.0),
        }
    }
}

impl std::error::Error for AttachError {}
