//! Destruction events, emitted where the pipeline decides them.
//!
//! Every consumer in this codebase's history *reconstructed* this stream by
//! diffing snapshots of the physics scene -- roughly 500 lines and an O(bonds)
//! scan per tick over 74k bonds, to rediscover facts the split path already
//! held in local variables. That is strictly more work and strictly less
//! information: a diff can see that a chunk changed body, but not which split
//! moved it or what it was part of before.
//!
//! So the pipeline emits directly. The rule for this module is that an event is
//! only emitted where the fact is *known*, never where it could be inferred --
//! an inferred event is a guess with a confident type.
//!
//! # Not yet produced
//!
//! Two variants are declared and never emitted, and are called out here rather
//! than left for a consumer to discover by waiting for one:
//!
//! - [`DestructionEvent::ChunkDestroyed`] needs crush/comminution, which is not
//!   in the core pipeline yet.
//! - [`DestructionEvent::IslandSettled`] needs the settle edge.
//!
//! They are declared now because their shape is settled and consumers can match
//! exhaustively today, but nothing synthesises them. A fabricated event would be
//! worse than a missing one -- the same reason `DestructionStats` refuses to
//! report a figure it cannot produce.
//!
//! # Poses are COM-frame, always
//!
//! `IslandPromoted::pose` is the island's centre of mass, not the engine's
//! actor origin, and member offsets are relative to that COM:
//!
//! ```text
//! chunk_world = pose.translation + pose.rotation * member.offset
//! ```
//!
//! This is not a convenience. Exactly one child per split reuses its parent's
//! body, and a reused body deliberately keeps the parent's frame so that a
//! resimulation snapshot of its origin stays valid. Its COM therefore moves
//! into `centerOfMassLocalPose` while its origin stays put. A consumer handed
//! the raw actor frame draws every chunk of that island one COM-height off and
//! watches it orbit the origin as the body tumbles -- which is precisely the
//! bug `com_world_position()` was written at three separate call sites to work
//! around. Normalising here means no consumer has to discover it a fourth time.

use crate::backend::{BodyKind, Pose};
use crate::types::Vec3;

/// Monotone island identity. Never reused, so a stale reference is detectably
/// stale rather than silently aliasing a different island.
///
/// Rapier's own handles are generational and its slots *are* recycled, which is
/// exactly the aliasing hazard this exists to avoid; PhysX actor pointers have
/// the same problem for the same reason. A serial outlives the body it names.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct IslandSerial(pub u64);

impl IslandSerial {
    pub const NONE: IslandSerial = IslandSerial(u64::MAX);
}

/// Where a chunk sits within its island, in the island's COM frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkPlacement {
    /// Node index in the scenario -- the app maps this to its own entity id.
    pub chunk: u32,
    /// Body-local offset from the island COM. See the module docs for the
    /// composition rule.
    pub offset: Vec3,
}

/// A topology change the pipeline made, or a bond it broke.
#[derive(Clone, Debug, PartialEq)]
pub enum DestructionEvent {
    /// The solver broke a bond. Emitted before the splits it causes, and
    /// emitted even when it causes none -- most broken bonds only weaken a
    /// structure, and a consumer showing damage needs those too.
    BondBroken {
        node0: u32,
        node1: u32,
        /// Health at the moment of breaking. Negative means the bond was
        /// overstressed past zero within one solve rather than worn down.
        health: f32,
    },
    /// A chunk changed island. `from` is `IslandSerial::NONE` only at attach.
    ChunkMigrated {
        chunk: u32,
        from: IslandSerial,
        to: IslandSerial,
    },
    /// A new island exists and here is everything needed to place it.
    IslandPromoted {
        serial: IslandSerial,
        /// COM-frame pose. See the module docs.
        pose: Pose,
        linvel: Vec3,
        angvel: Vec3,
        mass: f32,
        /// True when the island contains a support node, so it is anchored and
        /// will not move.
        ///
        /// Consumers that do not transmit anchored geometry filter on this
        /// rather than on a magic serial. The C++ convention that "serial 0 is
        /// the support actor" happens to hold here -- attach walks actors in
        /// order and the anchored one is built first -- but nothing depends on
        /// it, and a scene with two foundations would break anything that did.
        anchored: bool,
        /// The island this one was severed from, or `NONE` at attach.
        ///
        /// Load-bearing for a ledger consumer: a promotion drains these chunks
        /// out of their previous island, so without provenance the receiver has
        /// to search every island it knows to find where they went.
        provenance: IslandSerial,
        members: Vec<ChunkPlacement>,
    },
    /// An existing island's membership changed, so its COM moved and every
    /// member offset it was last given is stale.
    ///
    /// This is not redundant with `ChunkMigrated`. Offsets are measured from
    /// the island's centre of mass, so a chunk that never moved still needs a
    /// new offset once its island sheds mass -- the frame it is measured in
    /// shifted underneath it. Exactly one child per split reuses its parent
    /// body, so without this event that island draws every one of its chunks
    /// displaced by the COM shift, and the error compounds with each split.
    ///
    /// The member list is authoritative: a consumer replaces what it holds
    /// rather than patching it, which makes the event idempotent and means a
    /// consumer that joins late converges.
    IslandRecomposed {
        serial: IslandSerial,
        mass: f32,
        members: Vec<ChunkPlacement>,
    },
    /// An island no longer exists -- every chunk migrated off it.
    IslandRetired { serial: IslandSerial },
    /// A chunk was comminuted and is gone. Distinct from migration: nothing
    /// receives it.
    ChunkDestroyed { chunk: u32 },
    /// The island is definitively at rest. This is the edge every networked
    /// consumer needs in order to stop sending updates for it.
    IslandSettled { serial: IslandSerial },
}

/// Ordered event sink, drained by the host once per step.
///
/// Order is part of the contract: bonds break before the splits they cause,
/// an island is promoted before chunks migrate onto it, and an island is
/// retired only after the last chunk has left. A consumer applying the stream
/// in order never observes a chunk on an island it has not been told about.
#[derive(Clone, Debug, Default)]
pub struct EventSink {
    events: Vec<DestructionEvent>,
    /// Counts what the sink dropped, if a cap is ever added. Currently
    /// unbounded and always zero -- reported rather than silently absent so
    /// that adding a cap later cannot quietly lose events.
    dropped: usize,
}

impl EventSink {
    pub fn push(&mut self, event: DestructionEvent) {
        self.events.push(event);
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn dropped(&self) -> usize {
        self.dropped
    }

    pub fn as_slice(&self) -> &[DestructionEvent] {
        &self.events
    }

    /// Take everything queued, leaving the sink empty and its capacity intact
    /// so a steady state does not reallocate.
    pub fn drain(&mut self) -> Vec<DestructionEvent> {
        std::mem::take(&mut self.events)
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }
}

/// Hands out island serials.
///
/// Monotone and never reused, including across retirement. The width is u64
/// deliberately: the inherited 22-bit field tops out at 4.19M cumulative
/// islands, which a long-running server genuinely reaches, and wrapping it
/// would alias a live island onto a dead one.
#[derive(Clone, Debug)]
pub struct SerialAllocator {
    next: u64,
}

impl Default for SerialAllocator {
    fn default() -> Self {
        Self { next: 0 }
    }
}

impl SerialAllocator {
    pub fn next(&mut self) -> IslandSerial {
        let s = IslandSerial(self.next);
        self.next += 1;
        s
    }

    /// How many serials have ever been issued.
    pub fn issued(&self) -> u64 {
        self.next
    }
}

/// Convenience for consumers that only care about one island's shape.
impl DestructionEvent {
    /// The island this event concerns, when it concerns exactly one.
    pub fn island(&self) -> Option<IslandSerial> {
        match self {
            DestructionEvent::IslandPromoted { serial, .. }
            | DestructionEvent::IslandRecomposed { serial, .. }
            | DestructionEvent::IslandRetired { serial }
            | DestructionEvent::IslandSettled { serial } => Some(*serial),
            DestructionEvent::ChunkMigrated { to, .. } => Some(*to),
            DestructionEvent::BondBroken { .. } | DestructionEvent::ChunkDestroyed { .. } => None,
        }
    }
}

/// Body kind carried alongside a promotion, for backends that distinguish.
pub fn kind_for(anchored: bool) -> BodyKind {
    if anchored {
        BodyKind::Fixed
    } else {
        BodyKind::Dynamic
    }
}
