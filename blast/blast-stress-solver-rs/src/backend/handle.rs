//! Opaque, ordered body/shape handles.

use std::fmt::Debug;
use std::hash::Hash;

/// A backend's handle type.
///
/// `sort_key` exists because **determinism depends on it**. The pipeline sorts
/// parent bodies before recycling or creating, so that the engine allocates new
/// handles in a reproducible order; without a stable total order, a chaotic
/// fracture diverges run-to-run and every behavioural test becomes flaky.
///
/// The key need not be meaningful — only stable for a given allocation
/// sequence, and never derived from a pointer address or hash-map iteration.
/// Note the supertraits deliberately stop at `Eq + Hash`. Requiring `Ord`
/// would demand a trait real engines do not provide — Rapier's
/// `RigidBodyHandle` is not `Ord` — and it would be redundant besides, because
/// `sort_key` already defines the only ordering the pipeline is allowed to
/// use. Sorting is always `sort_by_key(|h| h.sort_key())`.
pub trait BackendHandle: Copy + Eq + Hash + Debug + 'static {
    fn sort_key(self) -> u64;
}

/// Default handle for backends that mint their own stable ids (PhysX does this
/// with a monotone counter that is never reused).
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct OpaqueId(pub u64);

impl BackendHandle for OpaqueId {
    fn sort_key(self) -> u64 {
        self.0
    }
}
