//! Capability advertisement.
//!
//! Three rules keep this from rotting into a bag of optional methods (the
//! failure mode the TypeScript `DestructibleCore` reached, with 25+ optional
//! members used as ad-hoc versioning):
//!
//! 1. A bit exists only if the pipeline has **exactly one** named degradation
//!    path for its absence, documented on the bit.
//! 2. An unsupported operation returns `Err(Unsupported)`. It never silently
//!    no-ops — a silent no-op is how a missing capability becomes a physics bug
//!    three layers away.
//! 3. Required bits are checked **once**, at attach. Nothing downstream
//!    re-checks.

/// Bitset of what a backend can do.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Capabilities(u32);

macro_rules! caps {
    ($($(#[$m:meta])* $name:ident = $bit:expr;)*) => {
        impl Capabilities {
            $($(#[$m])* pub const $name: Capabilities = Capabilities(1 << $bit);)*
            /// Human-readable names of the set bits, for error messages.
            pub fn names(self) -> Vec<&'static str> {
                let mut v = Vec::new();
                $(if self.contains(Capabilities::$name) { v.push(stringify!($name)); })*
                v
            }
        }
    };
}

caps! {
    // ---- required: attach() fails without these ----
    /// Create and remove bodies.
    BODY_LIFECYCLE = 0;
    /// Flip a body dynamic<->fixed/kinematic in place, keeping its handle.
    /// Required because a split can change whether a fragment owns a support.
    BODY_TYPE_MUTATION = 1;
    /// Create and remove shapes parented to a body.
    SHAPE_LIFECYCLE = 2;
    /// Recompute a body's mass properties from its shapes and read the COM.
    MASS_PROPERTIES = 3;
    /// Read and write pose, linear and angular velocity.
    POSE_VELOCITY_IO = 4;
    /// Report contacts with enough fidelity to drive the stress graph.
    CONTACT_EVENTS = 5;
    /// Handles are allocated in a reproducible order. Without this, chaotic
    /// fracture diverges run-to-run and the determinism assertions are skipped.
    DETERMINISTIC_HANDLES = 6;

    // ---- optional: each gates exactly one degradation ----
    /// Move a shape between bodies preserving its identity and contact state.
    /// Absent -> core emulates with remove+recreate, losing warm-start across
    /// every split. Rapier-JS lacks it; Rapier-Rust and PhysX have it.
    REPARENT_SHAPE = 8;
    /// Toggle a shape out of simulation without destroying it.
    /// Absent -> `CollisionLod` dormancy refuses to enable.
    SHAPE_SIMULATION_TOGGLE = 9;
    /// Toggle scene-query visibility independently of simulation.
    /// Absent -> dormant geometry is also invisible to host raycasts, and core
    /// exposes `materialize_region` instead. PhysX separates these bits;
    /// Rapier couples them.
    SHAPE_QUERY_TOGGLE = 10;
    /// Reject an arbitrary body pair for one step (sibling grace).
    /// Absent -> freshly split fragments may jitter apart on the split frame.
    PAIR_EXCLUSION = 11;
    /// Per-shape collision/solver group masks.
    COLLISION_GROUPS = 12;
    /// Per-body linear/angular damping.
    DAMPING = 13;
    /// Per-body sleep thresholds.
    SLEEP_THRESHOLDS = 14;
    /// Per-body continuous collision detection toggle.
    CCD = 15;
    /// Apply impulses, including to bodies the library does not own (the crush
    /// resistance ledger charges the crusher, which is usually a host body).
    IMPULSES = 16;
    /// Capture and restore body motion for resimulation rollback.
    /// Absent -> `Resim` is refused.
    MOTION_SNAPSHOT = 17;
    /// Restore a chosen subset rather than every body. Absent -> rollback is
    /// whole-world, which is correct but far more expensive.
    SCOPED_SNAPSHOT = 18;
    /// Batched motion read/write (PhysX Direct-GPU). Absent -> core loops.
    BATCH_MOTION_IO = 19;
    /// Native velocity-at-point. Absent -> core derives it from pose+vel+COM.
    NATIVE_POINT_VELOCITY = 20;
    /// Per-contact manifold data (position, normal, impulse). Required by
    /// crush, which needs the closing speed along the contact normal.
    CONTACT_MANIFOLDS = 21;
}

impl Capabilities {
    pub const NONE: Capabilities = Capabilities(0);

    /// Everything `attach()` insists on.
    pub const REQUIRED: Capabilities = Capabilities(
        Self::BODY_LIFECYCLE.0
            | Self::BODY_TYPE_MUTATION.0
            | Self::SHAPE_LIFECYCLE.0
            | Self::MASS_PROPERTIES.0
            | Self::POSE_VELOCITY_IO.0
            | Self::CONTACT_EVENTS.0
            | Self::DETERMINISTIC_HANDLES.0,
    );

    pub const fn bits(self) -> u32 {
        self.0
    }
    pub const fn contains(self, other: Capabilities) -> bool {
        (self.0 & other.0) == other.0
    }
    pub const fn union(self, other: Capabilities) -> Capabilities {
        Capabilities(self.0 | other.0)
    }
    /// Bits present in `self` but not `other`.
    pub const fn difference(self, other: Capabilities) -> Capabilities {
        Capabilities(self.0 & !other.0)
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for Capabilities {
    type Output = Capabilities;
    fn bitor(self, rhs: Capabilities) -> Capabilities {
        self.union(rhs)
    }
}

/// Returned when an optional operation is called on a backend that does not
/// advertise it. Deliberately an error rather than a no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unsupported(pub Capabilities);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "backend does not support {:?}", self.0.names())
    }
}

impl std::error::Error for Unsupported {}
