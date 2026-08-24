//! Engine-independent destruction pipeline.
//!
//! Everything here operates on opaque [`BackendHandle`](crate::backend::BackendHandle)
//! values and plain vectors, so it is written once and inherited by every
//! adapter. This is where the library's value lives; an adapter should never
//! need to reimplement any of it.

pub mod destructible;
pub mod destructible_set;
pub mod motion_fit;
pub mod split_planner;

pub use destructible_set::{AttachError, DestructibleSet, SetStepReport, StructureId};
pub use destructible::{Destructible, DestructibleConfig, StepReport};
pub use motion_fit::{fit_rigid_motion, weighted_center_of_mass, RigidMotionFit};
pub use split_planner::{
    plan_split_migration, plan_split_migration_with_support, CreateEntry, ExistingBodyState,
    PlannerChildSupport, ReuseEntry, SplitMigrationPlan,
};
