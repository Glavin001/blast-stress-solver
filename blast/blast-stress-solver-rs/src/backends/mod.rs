//! Engine adapters shipped with the library.
//!
//! Each is deliberately thin: it converts types, honours the write-elision and
//! ordering rules, and advertises capabilities. No destruction logic lives here.

#[cfg(feature = "rapier")]
pub mod rapier_backend;

#[cfg(feature = "rapier")]
pub use rapier_backend::RapierWorld;

#[cfg(feature = "physx")]
pub mod physx_backend;

#[cfg(feature = "physx")]
pub use physx_backend::PhysXWorld;
