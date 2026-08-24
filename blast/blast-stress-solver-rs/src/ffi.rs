//! Raw FFI declarations for the C stress solver bridges.
//!
//! These map directly to `stress_bridge.h` and `ext_stress_bridge.h`.
//! All types are `repr(C)` to match the C layout.

#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_int;

use crate::types::Vec3;

// ---- Low-level StressProcessor FFI ----

#[repr(C)]
pub(crate) struct StressProcessorHandle {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FfiStressNodeDesc {
    pub com: Vec3,
    pub mass: f32,
    pub inertia: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FfiStressBondDesc {
    pub centroid: Vec3,
    pub node0: u32,
    pub node1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FfiStressVelocity {
    pub ang: Vec3,
    pub lin: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FfiStressImpulse {
    pub ang: Vec3,
    pub lin: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FfiStressDataParams {
    pub equalize_masses: u8,
    pub center_bonds: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FfiStressSolverParams {
    pub max_iterations: u32,
    pub tolerance: f32,
    pub warm_start: u8,
    pub _pad: [u8; 3],
}

impl Default for FfiStressSolverParams {
    fn default() -> Self {
        Self {
            max_iterations: 32,
            tolerance: 1.0e-6,
            warm_start: 0,
            _pad: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FfiStressErrorSq {
    pub ang: f32,
    pub lin: f32,
}

extern "C" {
    pub(crate) fn stress_processor_create(
        nodes: *const FfiStressNodeDesc,
        node_count: u32,
        bonds: *const FfiStressBondDesc,
        bond_count: u32,
        params: FfiStressDataParams,
    ) -> *mut StressProcessorHandle;

    pub(crate) fn stress_processor_destroy(handle: *mut StressProcessorHandle);

    pub(crate) fn stress_processor_node_count(handle: *const StressProcessorHandle) -> u32;

    pub(crate) fn stress_processor_bond_count(handle: *const StressProcessorHandle) -> u32;

    pub(crate) fn stress_processor_solve(
        handle: *mut StressProcessorHandle,
        impulses: *mut FfiStressImpulse,
        velocities: *const FfiStressVelocity,
        params: FfiStressSolverParams,
        out_error: *mut FfiStressErrorSq,
        resume_solver: u8,
    ) -> c_int;

    pub(crate) fn stress_processor_remove_bond(
        handle: *mut StressProcessorHandle,
        bond_index: u32,
    ) -> u8;

    pub(crate) fn stress_processor_get_node_desc(
        handle: *const StressProcessorHandle,
        index: u32,
        out_desc: *mut FfiStressNodeDesc,
    ) -> u8;

    pub(crate) fn stress_processor_get_bond_desc(
        handle: *const StressProcessorHandle,
        index: u32,
        out_desc: *mut FfiStressBondDesc,
    ) -> u8;

    pub(crate) fn stress_processor_using_simd() -> u8;
}

// ---- High-level ExtStressSolver FFI ----

#[repr(C)]
pub(crate) struct ExtStressSolverHandle {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FfiExtStressNodeDesc {
    pub centroid: Vec3,
    pub mass: f32,
    pub volume: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FfiExtStressBondDesc {
    pub centroid: Vec3,
    pub normal: Vec3,
    /// Geometry only: the real contact patch (m^2), which is also the damage
    /// pool. Strength is authored via `material`, never by scaling area.
    pub area: f32,
    pub node0: u32,
    pub node1: u32,
    /// Index into the material table passed to `ext_stress_solver_create`.
    pub material: u32,
}

/// Per-material stress limits (Pa). Negative tension/shear inherit compression.
///
/// **Layout is load-bearing.** This must mirror `ExtStressMaterialDesc` in
/// `ext_stress_bridge.h` field-for-field. It previously declared only the six
/// stress limits while the C struct carried nine further crush fields, so
/// `ext_stress_solver_create` read 60 bytes out of a 24-byte Rust allocation —
/// a 36-byte out-of-bounds read per material, with the crush parameters coming
/// from whatever happened to follow in memory. `material_desc_matches_c_abi`
/// in `tests/ffi_abi_test.rs` pins the size against
/// `ext_stress_sizeof_material_desc()`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FfiExtStressMaterialDesc {
    pub compression_elastic_limit: f32,
    pub compression_fatal_limit: f32,
    pub tension_elastic_limit: f32,
    pub tension_fatal_limit: f32,
    pub shear_elastic_limit: f32,
    pub shear_fatal_limit: f32,
    /// Pa. `<= 0` disables crushing for this material.
    pub crush_cap_pressure: f32,
    /// Pa. Drucker-Prager intercept at p = 0.
    pub crush_cohesion: f32,
    /// dq/dp of the cone, dimensionless.
    pub crush_friction_slope: f32,
    /// J/m^3. Plastic work to fully comminute.
    pub crush_energy: f32,
    /// Pa*s. Perzyna viscosity; > 0 when enabled.
    pub crush_viscosity: f32,
    /// CEB DIF exponent. 0 disables.
    pub crush_strain_rate_exponent: f32,
    /// 1/s.
    pub crush_reference_strain_rate: f32,
    /// [0,1] respawned as rigid fragments.
    pub crush_debris_mass_fraction: f32,
    pub crush_debris_fragment_count: u32,
}

impl Default for FfiExtStressMaterialDesc {
    /// All-zero crush fields, which leaves `crush_cap_pressure == 0.0` and so
    /// keeps crushing disabled unless a caller opts in explicitly.
    fn default() -> Self {
        Self {
            compression_elastic_limit: 1.0,
            compression_fatal_limit: 2.0,
            tension_elastic_limit: -1.0,
            tension_fatal_limit: -1.0,
            shear_elastic_limit: -1.0,
            shear_fatal_limit: -1.0,
            crush_cap_pressure: 0.0,
            crush_cohesion: 0.0,
            crush_friction_slope: 0.0,
            crush_energy: 0.0,
            crush_viscosity: 0.0,
            crush_strain_rate_exponent: 0.0,
            crush_reference_strain_rate: 0.0,
            crush_debris_mass_fraction: 0.0,
            crush_debris_fragment_count: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FfiExtStressSolverSettingsDesc {
    pub max_solver_iterations_per_frame: u32,
    pub graph_reduction_level: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FfiExtStressBondFracture {
    pub userdata: u32,
    pub node_index0: u32,
    pub node_index1: u32,
    pub health: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FfiExtStressFractureCommands {
    pub actor_index: u32,
    pub bond_fractures: *mut FfiExtStressBondFracture,
    pub bond_fracture_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FfiExtStressActor {
    pub actor_index: u32,
    pub nodes: *const u32,
    pub node_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FfiExtStressSplitEvent {
    pub parent_actor_index: u32,
    pub children: *mut FfiExtStressActor,
    pub child_count: u32,
}

extern "C" {
    /// ABI revision of the C bridge these declarations mirror.
    pub(crate) fn ext_stress_abi_version() -> u32;

    // ---- ABI size probes (used by tests/ffi_abi_test.rs to pin struct layout) ----
    pub(crate) fn ext_stress_sizeof_material_desc() -> u32;
    pub(crate) fn ext_stress_sizeof_ext_node_desc() -> u32;
    pub(crate) fn ext_stress_sizeof_ext_bond_desc() -> u32;
    pub(crate) fn ext_stress_sizeof_ext_settings() -> u32;
    pub(crate) fn ext_stress_sizeof_ext_bond_fracture() -> u32;
    pub(crate) fn ext_stress_sizeof_ext_fracture_commands() -> u32;
    pub(crate) fn ext_stress_sizeof_actor() -> u32;
    pub(crate) fn ext_stress_sizeof_ext_split_event() -> u32;

    pub(crate) fn ext_stress_solver_create(
        nodes: *const FfiExtStressNodeDesc,
        node_count: u32,
        bonds: *const FfiExtStressBondDesc,
        bond_count: u32,
        materials: *const FfiExtStressMaterialDesc,
        material_count: u32,
        settings: *const FfiExtStressSolverSettingsDesc,
    ) -> *mut ExtStressSolverHandle;

    // ---- Per-bond material table (replaces the settings-borne limits) ----
    pub(crate) fn ext_stress_solver_set_materials(
        handle: *mut ExtStressSolverHandle,
        materials: *const FfiExtStressMaterialDesc,
        material_count: u32,
    ) -> u8;

    pub(crate) fn ext_stress_solver_set_node_materials(
        handle: *mut ExtStressSolverHandle,
        material_indices: *const u32,
        node_count: u32,
    ) -> u8;

    // ---- Island-aware solving (Stage 4) ----
    pub(crate) fn ext_stress_solver_set_island_aware(handle: *mut ExtStressSolverHandle, enabled: u8);
    pub(crate) fn ext_stress_solver_get_island_aware(handle: *const ExtStressSolverHandle) -> u8;
    pub(crate) fn ext_stress_solver_set_skip_settled(handle: *mut ExtStressSolverHandle, enabled: u8);
    pub(crate) fn ext_stress_solver_get_skip_settled(handle: *const ExtStressSolverHandle) -> u8;
    pub(crate) fn ext_stress_solver_island_count(handle: *const ExtStressSolverHandle) -> u32;
    pub(crate) fn ext_stress_solver_islands_skipped(handle: *const ExtStressSolverHandle) -> u32;
    pub(crate) fn ext_stress_solver_islands_total(handle: *const ExtStressSolverHandle) -> u32;

    // ---- Batched per-actor gravity (one crossing instead of one per actor) ----
    pub(crate) fn ext_stress_solver_add_all_actor_gravity(
        handle: *mut ExtStressSolverHandle,
        world_gravity_x: f32,
        world_gravity_y: f32,
        world_gravity_z: f32,
        actor_rotations: *const f32,
        rotation_count: u32,
    ) -> u32;

    // ---- Crush / comminution ----
    pub(crate) fn ext_stress_solver_set_node_strain_rates(
        handle: *mut ExtStressSolverHandle,
        strain_rates: *const f32,
        node_count: u32,
        delta_time: f32,
    ) -> u8;
    pub(crate) fn ext_stress_solver_get_node_crush_damage(
        handle: *const ExtStressSolverHandle,
        out_damage: *mut f32,
        capacity: u32,
    ) -> u32;
    pub(crate) fn ext_stress_solver_get_node_stress_invariants(
        handle: *const ExtStressSolverHandle,
        out_pressure: *mut f32,
        out_deviator: *mut f32,
        capacity: u32,
    ) -> u32;
    pub(crate) fn ext_stress_solver_get_node_crush_utilisation(
        handle: *const ExtStressSolverHandle,
        out_utilisation: *mut f32,
        capacity: u32,
    ) -> u32;
    pub(crate) fn ext_stress_solver_get_crushed_nodes(
        handle: *mut ExtStressSolverHandle,
        out_node_indices: *mut u32,
        capacity: u32,
    ) -> u32;
    pub(crate) fn ext_stress_solver_retire_crushed_node(
        handle: *mut ExtStressSolverHandle,
        node_index: u32,
    ) -> u8;
    pub(crate) fn ext_stress_solver_is_crush_enabled(handle: *const ExtStressSolverHandle) -> u8;

    // ---- Bond stress readback ----
    pub(crate) fn ext_stress_solver_get_bond_stresses(
        handle: *const ExtStressSolverHandle,
        out_compression: *mut f32,
        out_tension: *mut f32,
        out_shear: *mut f32,
        capacity: u32,
    ) -> u32;

    // ---- CUDA stress-solve backend (orthogonal to the physics engine) ----
    pub(crate) fn ext_stress_solver_set_gpu_accelerated(handle: *mut ExtStressSolverHandle, enabled: u8) -> u8;
    pub(crate) fn ext_stress_solver_get_gpu_accelerated(handle: *const ExtStressSolverHandle) -> u8;
    pub(crate) fn ext_stress_solver_set_gpu_cuda_context(handle: *mut ExtStressSolverHandle, cuda_context: *mut std::ffi::c_void);
    pub(crate) fn ext_stress_solver_set_gpu_minimum_bond_count(handle: *mut ExtStressSolverHandle, bond_count: u32);
    pub(crate) fn ext_stress_solver_gpu_solve_milliseconds(handle: *const ExtStressSolverHandle) -> f32;
    pub(crate) fn ext_stress_solver_gpu_host_to_device_bytes(handle: *const ExtStressSolverHandle) -> u64;
    pub(crate) fn ext_stress_solver_gpu_device_to_host_bytes(handle: *const ExtStressSolverHandle) -> u64;

    pub(crate) fn ext_stress_solver_get_bond_utilisations(
        handle: *const ExtStressSolverHandle,
        out_utilisation: *mut f32,
        capacity: u32,
    ) -> u32;

    pub(crate) fn ext_stress_solver_destroy(handle: *mut ExtStressSolverHandle);

    pub(crate) fn ext_stress_solver_set_settings(
        handle: *mut ExtStressSolverHandle,
        settings: *const FfiExtStressSolverSettingsDesc,
    );

    pub(crate) fn ext_stress_solver_graph_node_count(handle: *const ExtStressSolverHandle) -> u32;

    pub(crate) fn ext_stress_solver_bond_count(handle: *const ExtStressSolverHandle) -> u32;

    pub(crate) fn ext_stress_solver_reset(handle: *mut ExtStressSolverHandle);

    pub(crate) fn ext_stress_solver_add_force(
        handle: *mut ExtStressSolverHandle,
        node_index: u32,
        local_position: *const Vec3,
        local_force: *const Vec3,
        mode: u32,
    );

    pub(crate) fn ext_stress_solver_add_all_forces(
        handle: *mut ExtStressSolverHandle,
        node_indices: *const u32,
        local_positions: *const f32,
        local_forces: *const f32,
        count: u32,
        mode: u32,
    ) -> u32;

    pub(crate) fn ext_stress_solver_add_gravity(
        handle: *mut ExtStressSolverHandle,
        local_gravity: *const Vec3,
    );

    pub(crate) fn ext_stress_solver_add_actor_gravity(
        handle: *mut ExtStressSolverHandle,
        actor_index: u32,
        local_gravity: *const Vec3,
    ) -> u8;

    pub(crate) fn ext_stress_solver_add_centrifugal_acceleration(
        handle: *mut ExtStressSolverHandle,
        actor_index: u32,
        local_center_mass: *const Vec3,
        local_angular_velocity: *const Vec3,
    ) -> u8;

    pub(crate) fn ext_stress_solver_update(handle: *mut ExtStressSolverHandle);

    pub(crate) fn ext_stress_solver_overstressed_bond_count(
        handle: *const ExtStressSolverHandle,
    ) -> u32;

    pub(crate) fn ext_stress_solver_generate_fracture_commands_per_actor(
        handle: *const ExtStressSolverHandle,
        command_buffer: *mut FfiExtStressFractureCommands,
        command_capacity: u32,
        bond_buffer: *mut FfiExtStressBondFracture,
        bond_capacity: u32,
        out_command_count: *mut u32,
        out_bond_count: *mut u32,
    ) -> u8;

    pub(crate) fn ext_stress_solver_apply_fracture_commands(
        handle: *mut ExtStressSolverHandle,
        command_buffer: *const FfiExtStressFractureCommands,
        command_count: u32,
        events_buffer: *mut FfiExtStressSplitEvent,
        event_capacity: u32,
        child_buffer: *mut FfiExtStressActor,
        child_capacity: u32,
        out_event_count: *mut u32,
        out_child_count: *mut u32,
        nodes_buffer: *mut u32,
        nodes_capacity: u32,
        out_node_count: *mut u32,
    ) -> u8;

    pub(crate) fn ext_stress_solver_actor_count(handle: *const ExtStressSolverHandle) -> u32;

    pub(crate) fn ext_stress_solver_collect_actors(
        handle: *const ExtStressSolverHandle,
        actor_buffer: *mut FfiExtStressActor,
        actor_capacity: u32,
        nodes_buffer: *mut u32,
        nodes_capacity: u32,
        out_actor_count: *mut u32,
        out_node_count: *mut u32,
    ) -> u8;

    pub(crate) fn ext_stress_solver_get_excess_forces(
        handle: *const ExtStressSolverHandle,
        actor_index: u32,
        center_of_mass: *const Vec3,
        out_force: *mut Vec3,
        out_torque: *mut Vec3,
    ) -> u8;

    pub(crate) fn ext_stress_solver_get_linear_error(handle: *const ExtStressSolverHandle) -> f32;

    pub(crate) fn ext_stress_solver_get_angular_error(handle: *const ExtStressSolverHandle) -> f32;

    pub(crate) fn ext_stress_solver_converged(handle: *const ExtStressSolverHandle) -> u8;
}
