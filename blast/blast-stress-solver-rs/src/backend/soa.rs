//! Struct-of-arrays read buffers.
//!
//! Reads are **sparse gathers keyed by a handle list**, never dense sweeps over
//! the whole world. That matters twice: the hot paths (oriented gravity,
//! centrifugal, excess forces) want one body per *actor*, and an idle frame
//! must be able to issue zero engine reads at all — a dense per-frame gather
//! would destroy the idle-skip saving.

use super::math::Pose;
use crate::types::Vec3;

bitflags_lite! {
    /// Per-body state flags read back in a batch.
    pub struct BodyFlags: u8 {
        const DYNAMIC   = 1 << 0;
        const KINEMATIC = 1 << 1;
        const SLEEPING  = 1 << 2;
        const ENABLED   = 1 << 3;
    }
}

/// Parallel arrays of body state. Index `i` corresponds to the caller's
/// `ids[i]`.
#[derive(Clone, Debug, Default)]
pub struct BodyStateSoa {
    pub pose: Vec<Pose>,
    pub linvel: Vec<Vec3>,
    pub angvel: Vec<Vec3>,
    pub flags: Vec<BodyFlags>,
    pub mass: Vec<f32>,
}

impl BodyStateSoa {
    pub fn len(&self) -> usize {
        self.pose.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pose.is_empty()
    }
    /// Clear while keeping capacity, then reserve for `n` entries.
    pub fn reset_for(&mut self, n: usize) {
        self.pose.clear();
        self.linvel.clear();
        self.angvel.clear();
        self.flags.clear();
        self.mass.clear();
        self.pose.reserve(n);
        self.linvel.reserve(n);
        self.angvel.reserve(n);
        self.flags.reserve(n);
        self.mass.reserve(n);
    }
    pub fn push(&mut self, pose: Pose, linvel: Vec3, angvel: Vec3, flags: BodyFlags, mass: f32) {
        self.pose.push(pose);
        self.linvel.push(linvel);
        self.angvel.push(angvel);
        self.flags.push(flags);
        self.mass.push(mass);
    }
    pub fn is_dynamic(&self, i: usize) -> bool {
        self.flags[i].contains(BodyFlags::DYNAMIC)
    }
    pub fn is_sleeping(&self, i: usize) -> bool {
        self.flags[i].contains(BodyFlags::SLEEPING)
    }
}

/// One contact, normalised across engines.
///
/// Two normalisations happen at the adapter boundary and must not leak:
/// PhysX reports an **impulse** (N·s) while Rapier reports a **force** (N), so
/// adapters divide by `dt`; and PhysX contact-impulse *signs* are ordering
/// dependent (`eINTERNAL_CONTACTS_ARE_FLIPPED` is never corrected), so only
/// magnitudes are meaningful and direction must come from `normal`.
#[derive(Clone, Copy, Debug)]
pub struct Contact<S> {
    pub shape_a: S,
    pub shape_b: Option<S>,
    pub world_position: Vec3,
    /// Unit contact normal, oriented from `shape_a` toward `shape_b`.
    pub normal: Vec3,
    /// Force magnitude in newtons, already converted from impulse if needed.
    pub force: f32,
    /// Relative velocity at the contact point, `b - a`. Crush needs the closing
    /// component of this along `normal`.
    pub relative_velocity: Vec3,
    /// True while a contact keeps carrying load, not only on first touch.
    /// Load-bearing: a severed island gets no bond stress from gravity, so the
    /// ground's continuous reaction is the only thing reproducing the load path
    /// its foundation used to provide.
    pub persisting: bool,
}

/// A drained batch of contacts, reused across frames.
#[derive(Clone, Debug)]
pub struct ContactBatch<S> {
    pub contacts: Vec<Contact<S>>,
}

impl<S> Default for ContactBatch<S> {
    fn default() -> Self {
        Self { contacts: Vec::new() }
    }
}

impl<S> ContactBatch<S> {
    pub fn clear(&mut self) {
        self.contacts.clear();
    }
    pub fn len(&self) -> usize {
        self.contacts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.contacts.is_empty()
    }
}
