#pragma once

#include <stdint.h>

#include "stress_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ExtStressSolverHandle;

typedef struct ExtStressNodeDesc {
    StressVec3 centroid;
    float mass;
    float volume;
    /* Rotational inertia (kg m^2) from the chunk's real shape, or 0 to let the
       solver fall back to its sphere-of-equal-volume approximation. */
    float inertia;
} ExtStressNodeDesc;

typedef struct ExtStressBondDesc {
    StressVec3 centroid;
    StressVec3 normal;
    /* Geometry only: the bond's real contact patch in m^2. It is also the
       damage pool. Strength is authored via `material`, never by scaling
       area. */
    float area;
    uint32_t node0;
    uint32_t node1;
    /* Index into the material table passed to ext_stress_solver_create. */
    uint32_t material;
} ExtStressBondDesc;

/* Per-material stress limits (Pa). Negative tension/shear limits inherit the
   corresponding compression limit (resolved inside the solver).

   The crush_* fields are the optional CHUNK comminution model: bond limits
   decide whether a JOINT fails, crush decides whether the CHUNK ITSELF is
   ground up and leaves the simulation. Disabled unless crush_cap_pressure > 0,
   so a caller that zero-initializes this struct gets the historical bond-only
   behavior exactly. See ExtStressCrushProperties in NvBlastExtStressSolver.h
   for the model. */
typedef struct ExtStressMaterialDesc {
    float compression_elastic_limit;
    float compression_fatal_limit;
    float tension_elastic_limit;
    float tension_fatal_limit;
    float shear_elastic_limit;
    float shear_fatal_limit;
    /* Young's modulus, Pa. Stiffness, not strength: decides how parallel load
       paths SHARE load (k = EA/L). 0 = unknown, treated as 30 GPa concrete. */
    float elastic_modulus_pa;
    float crush_cap_pressure;         /* Pa. <= 0 disables crushing. */
    float crush_cohesion;             /* Pa. Drucker-Prager intercept at p = 0. */
    float crush_friction_slope;       /* dq/dp of the cone, dimensionless. */
    float crush_energy;               /* J/m^3. Plastic work to fully comminute. */
    float crush_viscosity;            /* Pa*s. Perzyna viscosity; > 0 when enabled. */
    float crush_strain_rate_exponent; /* CEB DIF exponent. 0 disables. */
    float crush_reference_strain_rate;/* 1/s. */
    float crush_debris_mass_fraction; /* [0,1] respawned as rigid fragments. */
    uint32_t crush_debris_fragment_count;
} ExtStressMaterialDesc;

typedef struct ExtStressSolverSettingsDesc {
    uint32_t max_solver_iterations_per_frame;
    uint32_t graph_reduction_level;
} ExtStressSolverSettingsDesc;

typedef struct ExtStressDebugLine {
    StressVec3 p0;
    StressVec3 p1;
    uint32_t color0;
    uint32_t color1;
} ExtStressDebugLine;

typedef struct ExtStressBondFracture {
    uint32_t userdata;
    uint32_t nodeIndex0;
    uint32_t nodeIndex1;
    float health;
} ExtStressBondFracture;

typedef struct ExtStressFractureCommands {
    uint32_t actorIndex;
    ExtStressBondFracture* bondFractures;
    uint32_t bondFractureCount;
    /* Number of fully-crushed chunks severed on this actor. The command payload
       itself stays inside the bridge (it is addressed by asset chunk index,
       which is an internal detail) and is re-attached automatically by
       ext_stress_solver_apply_fracture_commands. A command with zero bond
       fractures and a non-zero count here is still real work and must be
       applied -- a chunk can pulverize while every joint around it holds.
       Drain ext_stress_solver_get_crushed_nodes for the node-level events. */
    uint32_t chunkFractureCount;
} ExtStressFractureCommands;

typedef struct ExtStressActor {
    uint32_t actorIndex;
    const uint32_t* nodes;
    uint32_t nodeCount;
} ExtStressActor;

typedef struct ExtStressSplitEvent {
    uint32_t parentActorIndex;
    ExtStressActor* children;
    uint32_t childCount;
} ExtStressSplitEvent;

/* ABI revision of this bridge. Bump whenever a struct crossing the boundary
   changes layout or an existing function changes signature. Consumers that
   compile these sources out-of-tree (the Rust crate, the CMake demo, and
   applications that point a BLAST_ROOT at this checkout) assert against it, so
   a layout drift fails loudly at startup instead of becoming an out-of-bounds
   read. Additive changes -- new functions, new trailing struct fields that the
   producer zero-fills -- do not require a bump. */
#define EXT_STRESS_ABI_VERSION 1u

uint32_t ext_stress_abi_version(void);

/* materials may be null (with material_count 0) to get a single default
   material — every bond must then use index 0. Any bond whose material index
   is out of range of the effective table is an authoring error and creation
   returns null. */
ExtStressSolverHandle* ext_stress_solver_create(const ExtStressNodeDesc* nodes,
                                                uint32_t node_count,
                                                const ExtStressBondDesc* bonds,
                                                uint32_t bond_count,
                                                const ExtStressMaterialDesc* materials,
                                                uint32_t material_count,
                                                const ExtStressSolverSettingsDesc* settings);

/* Replace the material table (e.g. sweep a global strength scale). Does not
   rebuild the graph. Count must be >= 1 and must cover every bond's index. */
uint8_t ext_stress_solver_set_materials(ExtStressSolverHandle* handle,
                                        const ExtStressMaterialDesc* materials,
                                        uint32_t material_count);

/* Per-bond utilisation from the last update(), asset-bond-indexed: max over
   modes of stress / that bond's OWN material elastic limit. 1/utilisation is
   the joint's safety factor. Broken bonds read 0. Returns entries written. */
uint32_t ext_stress_solver_get_bond_utilisations(const ExtStressSolverHandle* handle,
                                                 float* out_utilisation,
                                                 uint32_t capacity);

uint32_t ext_stress_sizeof_material_desc(void);

/* --- chunk crushing ------------------------------------------------------
   All node indices below are INPUT node indices (the order nodes were supplied
   to ext_stress_solver_create), matching every other node-indexed call. */

/* Assign each node the material whose crush properties govern it. Out-of-range
   indices clamp to 0; null resets every node to material 0. */
uint8_t ext_stress_solver_set_node_materials(ExtStressSolverHandle* handle,
                                             const uint32_t* material_indices,
                                             uint32_t node_count);

/* Supply each node's compaction strain rate (1/s) and the timestep the next
   update() advances by. The solver has no strain measure of its own; with all
   rates zero no crush damage ever accumulates. Null zeroes every rate. */
uint8_t ext_stress_solver_set_node_strain_rates(ExtStressSolverHandle* handle,
                                                const float* strain_rates,
                                                uint32_t node_count,
                                                float delta_time);

/* Per-node accumulated crush damage in [0,1]. Returns entries written. */
uint32_t ext_stress_solver_get_node_crush_damage(const ExtStressSolverHandle* handle,
                                                 float* out_damage,
                                                 uint32_t capacity);

/* Per-node stress invariants from the last update(): pressure p (Pa, positive
   in compression) and von Mises deviator q (Pa). Either pointer may be null.
   Populated only for nodes on a crush-enabled material. */
uint32_t ext_stress_solver_get_node_stress_invariants(const ExtStressSolverHandle* handle,
                                                      float* out_pressure,
                                                      float* out_deviator,
                                                      uint32_t capacity);

/* Per-node crush utilisation: max of q/(cohesion + frictionSlope*p) and
   p/capPressure. 1 means at yield. The crush analogue of bond utilisation, and
   readable whether or not anything is currently moving. Returns entries written. */
uint32_t ext_stress_solver_get_node_crush_utilisation(const ExtStressSolverHandle* handle,
                                                      float* out_utilisation,
                                                      uint32_t capacity);

/* Drain nodes that reached full crush damage. Each is reported exactly once.
   The caller owns removing the corresponding body/shape from its scene. */
uint32_t ext_stress_solver_get_crushed_nodes(ExtStressSolverHandle* handle,
                                             uint32_t* out_node_indices,
                                             uint32_t capacity);

/* Retire a pulverized chunk's actor.

   A chunk fracture severs the chunk structurally, but NvBlast never removes
   anything: a health-exhausted leaf chunk stays alive as an inert actor
   forever (NvBlastActor.cpp, partitionSingleLowerSupportChunk returns 0 before
   reaching release()). This calls NvBlastActorDeactivate, the SDK's only
   removal primitive, so the solver stops reporting an actor the caller has
   already removed from its scene.

   Only valid for an actor that consists solely of this node, which is what a
   crushed chunk always becomes once its bonds are zeroed. Returns 0 if the
   node is unknown or its actor still holds other nodes. */
uint8_t ext_stress_solver_retire_crushed_node(ExtStressSolverHandle* handle,
                                              uint32_t node_index);

/* Whether crushing is active at all: some material enables it AND the graph is
   unreduced. Crush requires graph_reduction_level 0. */
uint8_t ext_stress_solver_is_crush_enabled(const ExtStressSolverHandle* handle);

void ext_stress_solver_destroy(ExtStressSolverHandle* handle);

void ext_stress_solver_set_settings(ExtStressSolverHandle* handle,
                                    const ExtStressSolverSettingsDesc* settings);

uint32_t ext_stress_solver_graph_node_count(const ExtStressSolverHandle* handle);

uint32_t ext_stress_solver_bond_count(const ExtStressSolverHandle* handle);

void ext_stress_solver_reset(ExtStressSolverHandle* handle);

void ext_stress_solver_add_force(ExtStressSolverHandle* handle,
                                 uint32_t node_index,
                                 const StressVec3* local_position,
                                 const StressVec3* local_force,
                                 uint32_t mode);

// Batched external-force injection. Applies `count` forces in a single FFI
// crossing, mirroring ext_stress_solver_add_force for each entry. The parallel
// arrays are laid out as: node_indices[i] -> input node index; local_positions
// and local_forces -> flat [x, y, z] triples (offset 3*i). `mode` is shared by
// all entries. Null position/force arrays are treated as all-zero. Returns the
// number of entries processed.
uint32_t ext_stress_solver_add_all_forces(ExtStressSolverHandle* handle,
                                          const uint32_t* node_indices,
                                          const float* local_positions,
                                          const float* local_forces,
                                          uint32_t count,
                                          uint32_t mode);

void ext_stress_solver_add_gravity(ExtStressSolverHandle* handle,
                                   const StressVec3* local_gravity);

uint8_t ext_stress_solver_add_actor_gravity(ExtStressSolverHandle* handle,
                                            uint32_t actor_index,
                                            const StressVec3* local_gravity);

uint8_t ext_stress_solver_add_centrifugal_acceleration(ExtStressSolverHandle* handle,
                                                       uint32_t actor_index,
                                                       const StressVec3* local_center_mass,
                                                       const StressVec3* local_angular_velocity);

// Batched per-actor gravity. Applies `world_gravity` to every actor the solver
// currently tracks, rotating it into each actor's body-local frame on the C++
// side so the caller does not need to materialise the actor list or cross the
// FFI boundary once per actor.
//
// `actor_rotations` is an optional flat buffer of unit quaternions laid out as
// [x, y, z, w] per slot and indexed by actor index (slot N starts at offset
// 4*N). `rotation_count` is the number of slots available. Actors whose index
// falls outside [0, rotation_count) — or all actors when `actor_rotations` is
// null — receive the unrotated world gravity (identity rotation).
//
// Returns the number of actors gravity was applied to.
uint32_t ext_stress_solver_add_all_actor_gravity(ExtStressSolverHandle* handle,
                                                 float world_gravity_x,
                                                 float world_gravity_y,
                                                 float world_gravity_z,
                                                 const float* actor_rotations,
                                                 uint32_t rotation_count);

void ext_stress_solver_update(ExtStressSolverHandle* handle);

uint32_t ext_stress_solver_overstressed_bond_count(const ExtStressSolverHandle* handle);

// Per-bond stress from the last update(), indexed by ASSET bond index (same order as the
// bonds passed to ext_stress_solver_create). Compression and tension are mutually exclusive;
// shear is independent. All values are pressures comparable to the settings' elastic/fatal
// limits, so limit/stress is the joint's safety factor. Broken or graph-absent bonds read 0.
// Any output pointer may be null. Returns the number of entries written.
/// Live per-bond health, indexed by ASSET bond index. Health crossing zero is
/// the break; the fracture command stream reports damage, not breaks.
uint32_t ext_stress_solver_get_bond_healths(const ExtStressSolverHandle* handle,
                                            float* out_health,
                                            uint32_t capacity);

uint32_t ext_stress_solver_get_bond_stresses(const ExtStressSolverHandle* handle,
                                             float* out_compression,
                                             float* out_tension,
                                             float* out_shear,
                                             uint32_t capacity);

uint32_t ext_stress_solver_fill_debug_render(const ExtStressSolverHandle* handle,
                                             uint32_t mode,
                                             float scale,
                                             ExtStressDebugLine* out_lines,
                                             uint32_t max_lines);

uint8_t ext_stress_solver_generate_fracture_commands(const ExtStressSolverHandle* handle,
                                                     ExtStressFractureCommands* out_commands,
                                                     ExtStressBondFracture* bond_buffer,
                                                     uint32_t max_bonds);

uint32_t ext_stress_solver_actor_count(const ExtStressSolverHandle* handle);

uint8_t ext_stress_solver_collect_actors(const ExtStressSolverHandle* handle,
                                         ExtStressActor* actor_buffer,
                                         uint32_t actor_capacity,
                                         uint32_t* nodes_buffer,
                                         uint32_t nodes_capacity,
                                         uint32_t* out_actor_count,
                                         uint32_t* out_node_count);

uint8_t ext_stress_solver_generate_fracture_commands_per_actor(const ExtStressSolverHandle* handle,
                                                               ExtStressFractureCommands* command_buffer,
                                                               uint32_t command_capacity,
                                                               ExtStressBondFracture* bond_buffer,
                                                               uint32_t bond_capacity,
                                                               uint32_t* out_command_count,
                                                               uint32_t* out_bond_count);

uint8_t ext_stress_solver_apply_fracture_commands(ExtStressSolverHandle* handle,
                                                  const ExtStressFractureCommands* command_buffer,
                                                  uint32_t command_count,
                                                  ExtStressSplitEvent* events_buffer,
                                                  uint32_t event_capacity,
                                                  ExtStressActor* child_buffer,
                                                  uint32_t child_capacity,
                                                  uint32_t* out_event_count,
                                                  uint32_t* out_child_count,
                                                  uint32_t* nodes_buffer,
                                                  uint32_t nodes_capacity,
                                                  uint32_t* out_node_count);

uint8_t ext_stress_solver_get_excess_forces(const ExtStressSolverHandle* handle,
                                            uint32_t actor_index,
                                            const StressVec3* center_of_mass,
                                            StressVec3* out_force,
                                            StressVec3* out_torque);

float ext_stress_solver_get_linear_error(const ExtStressSolverHandle* handle);

float ext_stress_solver_get_angular_error(const ExtStressSolverHandle* handle);

uint8_t ext_stress_solver_converged(const ExtStressSolverHandle* handle);

// Number of connected components (islands) in the solver graph after the last update.
uint32_t ext_stress_solver_island_count(const ExtStressSolverHandle* handle);

// Enable/disable island-aware solving (solve each disconnected component independently).
void ext_stress_solver_set_island_aware(ExtStressSolverHandle* handle, uint8_t enabled);
uint8_t ext_stress_solver_get_island_aware(const ExtStressSolverHandle* handle);

// Enable/disable skipping of settled islands (requires island-aware). islands_skipped reports the
// number skipped in the last update.
// Timestep (s) the next update() advances by. Bond damage is a rate, so without
// this the solver charges damage per tick and ductility becomes a property of
// the frame rate rather than of the material. 0 keeps the legacy per-tick path.
void ext_stress_solver_set_delta_time(ExtStressSolverHandle* handle, float delta_time);

void ext_stress_solver_set_skip_settled(ExtStressSolverHandle* handle, uint8_t enabled);
void ext_stress_solver_set_skip_stable_unconverged(ExtStressSolverHandle* handle, uint8_t enabled);
uint8_t ext_stress_solver_get_skip_settled(const ExtStressSolverHandle* handle);
uint32_t ext_stress_solver_islands_skipped(const ExtStressSolverHandle* handle);
// Islands the last update partitioned the graph into for the per-island solve (islands_skipped <= this).
uint32_t ext_stress_solver_islands_total(const ExtStressSolverHandle* handle);

uint8_t ext_stress_solver_set_gpu_accelerated(ExtStressSolverHandle* handle, uint8_t enabled);
void ext_stress_solver_set_gpu_cuda_context(ExtStressSolverHandle* handle, void* cuda_context);
void ext_stress_solver_set_gpu_minimum_bond_count(ExtStressSolverHandle* handle, uint32_t bond_count);
uint8_t ext_stress_solver_get_gpu_accelerated(const ExtStressSolverHandle* handle);
float ext_stress_solver_gpu_solve_milliseconds(const ExtStressSolverHandle* handle);
float ext_stress_solver_gpu_host_work_milliseconds(const ExtStressSolverHandle* handle);
float ext_stress_solver_gpu_host_blocked_milliseconds(const ExtStressSolverHandle* handle);
uint64_t ext_stress_solver_gpu_host_to_device_bytes(const ExtStressSolverHandle* handle);
uint64_t ext_stress_solver_gpu_device_to_host_bytes(const ExtStressSolverHandle* handle);

uint32_t ext_stress_sizeof_ext_node_desc();
uint32_t ext_stress_sizeof_ext_bond_desc();
uint32_t ext_stress_sizeof_ext_settings();
uint32_t ext_stress_sizeof_ext_debug_line();
uint32_t ext_stress_sizeof_ext_bond_fracture();
uint32_t ext_stress_sizeof_ext_fracture_commands();
uint32_t ext_stress_sizeof_actor();
uint32_t ext_stress_sizeof_actor_buffer();
uint32_t ext_stress_sizeof_ext_split_event();

#ifdef __cplusplus
}
#endif

