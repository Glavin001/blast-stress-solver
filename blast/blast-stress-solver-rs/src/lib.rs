//! # Blast Stress Solver
//!
//! Rust library wrapping the NVIDIA Blast stress solver for destructible structures.
//!
//! ## Features
//!
//! - **Core solver**: `ExtStressSolver` — manages nodes, bonds, actors, fracture detection and splitting
//! - **Low-level solver**: `StressProcessor` — direct conjugate-gradient solver access
//! - **Bond stress analysis**: `compute_bond_stress` — decompose impulses into compression/tension/shear
//! - **Scenarios** (feature `scenarios`): Pre-built wall, tower, and bridge scenario builders
//! - **Rapier integration** (feature `rapier`): `DestructibleSet` — full pipeline with Rapier3D physics
//!
//! ## Quick Start (without Rapier)
//!
//! ```no_run
//! use blast_stress_solver::*;
//!
//! let nodes = vec![
//!     NodeDesc { centroid: Vec3::new(0.0, 0.0, 0.0), mass: 0.0, volume: 1.0 },  // support
//!     NodeDesc { centroid: Vec3::new(0.0, 1.0, 0.0), mass: 10.0, volume: 1.0 },  // dynamic
//! ];
//! let bonds = vec![
//!     BondDesc {
//!         centroid: Vec3::new(0.0, 0.5, 0.0),
//!         normal: Vec3::new(0.0, 1.0, 0.0),
//!         area: 1.0,
//!         node0: 0,
//!         node1: 1,
//!         // Index into the material table. `new` puts every bond on a
//!         // one-entry table built from `settings`; use
//!         // `new_with_materials` for a pack that authors several.
//!         material: 0,
//!     },
//! ];
//! let settings = SolverSettings::default();
//! let mut solver = ExtStressSolver::new(&nodes, &bonds, &settings).unwrap();
//!
//! solver.add_gravity(Vec3::new(0.0, -9.81, 0.0));
//! solver.update();
//!
//! let overstressed = solver.overstressed_bond_count();
//! ```

mod ffi;

// On `wasm32-unknown-unknown` the Blast C++ backend references dozens
// of libc symbols (malloc, fwrite, abort, …) through libc++'s STL
// helpers.  We provide pure-Rust stubs for all of them in
// `wasm_runtime_shims`, so the final wasm module imports neither
// `env.*` libc functions nor `wasi_snapshot_preview1.*` wasi calls
// — it is a pure library module.
#[cfg(target_arch = "wasm32")]
mod wasm_runtime_shims;

// `-fno-exceptions` is not enough to strip every mention of
// `__cxa_allocate_exception` / `__cxa_throw` from libc++ — STL
// containers still emit them behind `throw_bad_alloc`-style helpers.
// Provide trapping stubs so the wasm module stays self-contained.
#[cfg(target_arch = "wasm32")]
mod wasm_cxa_stubs;

/// C-ABI struct sizes, as reported by the C bridge itself.
///
/// Exposed so `tests/ffi_abi_test.rs` can pin the Rust `#[repr(C)]` mirrors
/// against the authoritative layout. These structs cross the boundary by
/// pointer, so a field-count drift is a runtime out-of-bounds read rather than
/// a compile error — see the note on `FfiExtStressMaterialDesc`.
#[doc(hidden)]
pub mod abi {
    /// Byte sizes the C bridge reports for the descriptors Rust mirrors.
    #[derive(Clone, Copy, Debug)]
    pub struct CAbiSizes {
        pub material_desc: usize,
        pub settings_desc: usize,
        pub bond_fracture: usize,
        pub node_desc: usize,
        pub bond_desc: usize,
    }

    /// ABI revision this crate was built against. Must equal the C bridge's
    /// `EXT_STRESS_ABI_VERSION`; a mismatch means the crate and the native
    /// sources come from different checkouts.
    pub const EXPECTED_ABI_VERSION: u32 = 1;

    /// The ABI revision the linked C bridge reports.
    pub fn c_abi_version() -> u32 {
        unsafe { crate::ffi::ext_stress_abi_version() }
    }

    /// Query the C bridge for its own struct sizes.
    pub fn c_abi_sizes() -> CAbiSizes {
        unsafe {
            CAbiSizes {
                material_desc: crate::ffi::ext_stress_sizeof_material_desc() as usize,
                settings_desc: crate::ffi::ext_stress_sizeof_ext_settings() as usize,
                bond_fracture: crate::ffi::ext_stress_sizeof_ext_bond_fracture() as usize,
                node_desc: crate::ffi::ext_stress_sizeof_ext_node_desc() as usize,
                bond_desc: crate::ffi::ext_stress_sizeof_ext_bond_desc() as usize,
            }
        }
    }

    /// Sizes of the Rust mirrors, for comparison against [`c_abi_sizes`].
    pub fn rust_mirror_sizes() -> CAbiSizes {
        use std::mem::size_of;
        CAbiSizes {
            material_desc: size_of::<crate::ffi::FfiExtStressMaterialDesc>(),
            settings_desc: size_of::<crate::ffi::FfiExtStressSolverSettingsDesc>(),
            bond_fracture: size_of::<crate::ffi::FfiExtStressBondFracture>(),
            node_desc: size_of::<crate::ffi::FfiExtStressNodeDesc>(),
            bond_desc: size_of::<crate::ffi::FfiExtStressBondDesc>(),
        }
    }
}

#[macro_use]
mod bitflags_lite;

pub mod backend;
pub mod ids;
pub mod backends;
pub mod pipeline;
pub mod scene_pack;

pub mod bond_stress;
pub mod ext_stress_solver;
pub mod stress_processor;
pub mod types;

#[cfg(feature = "scenarios")]
pub mod scenarios;

#[cfg(feature = "rapier")]
pub mod rapier;

// Reusable benchmarking/profiling harness, gated behind `bench-support` (off by
// default, never shipped). Shared by the criterion benches, the `frame_profile`
// example, and the perf-regression test so every measurement describes the same work.
#[cfg(feature = "bench-support")]
pub mod bench_harness;

// Re-export primary types at crate root for convenience
pub use bond_stress::compute_bond_stress;
pub use ext_stress_solver::ExtStressSolver;
pub use stress_processor::{
    StressBondDesc, StressDataParams, StressErrorSq, StressImpulse, StressNodeDesc,
    StressProcessor, StressSolverParams, StressVelocity,
};
pub use types::*;
