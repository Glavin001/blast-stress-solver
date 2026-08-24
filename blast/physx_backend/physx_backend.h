/*
 * C ABI for the PhysX implementation of the core's PhysicsBackend contract.
 *
 * The Rust core owns the destruction pipeline; this exposes only the engine
 * operations it needs, in the same batched, struct-of-arrays shape the trait
 * uses. Batching is not cosmetic here: with GPU rigid bodies every per-actor
 * pose read is a device readback, so a per-body ABI would make the fast path
 * impossible to express.
 *
 * Bodies and shapes are addressed by adapter-minted uint64 ids that are
 * monotone and NEVER reused, so a stale id can be detected rather than
 * silently aliasing a recycled actor.
 */
#ifndef BLAST_PHYSX_BACKEND_H
#define BLAST_PHYSX_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PxbWorld PxbWorld;

typedef struct PxbVec3 { float x, y, z; } PxbVec3;
typedef struct PxbQuat { float x, y, z, w; } PxbQuat;
typedef struct PxbPose { PxbVec3 p; PxbQuat q; } PxbPose;

/* Body kinds. Supports are KINEMATIC rather than static so the flag can be
   flipped in place when a split changes support membership. */
enum PxbBodyKind { PXB_DYNAMIC = 0, PXB_FIXED = 1, PXB_KINEMATIC = 2 };

enum PxbBodyFlagBits {
    PXB_FLAG_DYNAMIC = 1 << 0,
    PXB_FLAG_KINEMATIC = 1 << 1,
    PXB_FLAG_SLEEPING = 1 << 2,
    PXB_FLAG_ENABLED = 1 << 3
};

typedef struct PxbCreateBody {
    PxbPose pose;
    PxbVec3 linvel;
    PxbVec3 angvel;
    uint32_t kind;
    uint8_t ccd;
    uint8_t start_sleeping;
} PxbCreateBody;

/* geom: 0 = cuboid (half_extents), 1 = convex hull (points/point_count). */
typedef struct PxbCreateShape {
    uint64_t body;
    PxbPose local;
    PxbVec3 half_extents;
    const PxbVec3* points;
    uint32_t point_count;
    uint32_t geom;
    float mass;
    uint32_t node;
} PxbCreateShape;

typedef struct PxbReparent {
    uint64_t shape;
    uint64_t body;
    PxbPose local;
} PxbReparent;

typedef struct PxbContact {
    uint64_t shape_a;
    uint64_t shape_b;
    PxbVec3 world_position;
    PxbVec3 normal;
    PxbVec3 relative_velocity;
    float force;
    uint8_t persisting;
} PxbContact;

/* Whole-buffer command submission. Any list may be null with count 0.
   Applied counts are written back so the caller can re-queue a remainder. */
typedef struct PxbCommands {
    const PxbCreateBody* create_bodies;      uint32_t create_body_count;
    const uint64_t* set_kind_ids;            const uint32_t* set_kind_values; uint32_t set_kind_count;
    const uint64_t* set_pose_ids;            const PxbPose* set_pose_values;  uint32_t set_pose_count;
    const uint64_t* set_vel_ids;             const PxbVec3* set_vel_lin; const PxbVec3* set_vel_ang; uint32_t set_vel_count;
    const PxbCreateShape* create_shapes;     uint32_t create_shape_count;
    const PxbReparent* reparent;             uint32_t reparent_count;
    const uint64_t* remove_shapes;           uint32_t remove_shape_count;
    const uint64_t* recompute_mass;          uint32_t recompute_mass_count;
    const uint64_t* remove_bodies;           uint32_t remove_body_count;
    const uint64_t* wake;                    uint32_t wake_count;
    const uint64_t* sleep;                   uint32_t sleep_count;
    const uint64_t* damping_ids;             const float* damping_lin; const float* damping_ang; uint32_t damping_count;
    const uint64_t* sleep_thr_ids;           const float* sleep_thr_lin; const float* sleep_thr_ang; uint32_t sleep_thr_count;
    const uint64_t* ccd_ids;                 const uint8_t* ccd_values; uint32_t ccd_count;
    /* Collision groups on shapes the library created.
     *
     * Required for bring-your-own-world: without filter data the host's own
     * raycasts and collision filtering cannot see library shapes at all, so a
     * hitscan into a destructible reports a miss.
     *
     * `entity` rides along because the host's query convention pairs a group
     * with an entity id in the same PxFilterData. */
    const uint64_t* group_shapes;            const uint32_t* group_memberships;
    const uint32_t* group_filters;           const uint32_t* group_entities;
    uint32_t group_count;
    const uint64_t* shape_enabled_ids;       const uint8_t* shape_enabled_values; uint32_t shape_enabled_count;
    const uint64_t* impulse_ids;             const PxbVec3* impulse_lin; const PxbVec3* impulse_ang; uint32_t impulse_count;
} PxbCommands;

typedef struct PxbApplied {
    uint32_t bodies_created;
    uint32_t shapes_created;
    uint32_t shapes_reparented;
    uint32_t bodies_removed;
    uint32_t writes_elided;
} PxbApplied;

/* gpu != 0 requests GPU dynamics + GPU broadphase. Returns null if that was
   requested and could not be established -- it never silently falls back,
   because a silent CPU fallback misreports every performance measurement. */
/* Bring-your-own-world: wrap a PxScene the host already owns and drives.
 *
 * The backend does NOT take ownership, does NOT step the scene, and does NOT
 * install a simulation-event callback -- the host almost certainly has one
 * already, and PhysX allows only one. Contacts therefore arrive by explicit
 * injection (pxb_inject_contact) from the host's own onContact, which is the
 * shape hosts are already in.
 *
 * `physics` is the host's PxPhysics and `material` its default material; both
 * are borrowed. Pass null for material to have one created.
 */
PxbWorld* pxb_world_attach(void* scene, void* physics, void* material);

/* Borrow the underlying PhysX objects. Intended for hosts that need to hand
   the same scene to another subsystem, and for tests that exercise the
   attach path. Returns null for a destroyed world. */
void* pxb_world_scene(const PxbWorld* w);
void* pxb_world_physics(const PxbWorld* w);

/* Feed one contact from the host's own simulation-event callback.
 * `impulse_magnitude` is PhysX's raw impulse (N*s); the backend converts to
 * force using the dt handed to pxb_note_dt. Only the magnitude is used: PhysX
 * contact impulse SIGNS are ordering dependent and never normalised. */
uint8_t pxb_inject_contact(PxbWorld* w, void* shape_a, void* shape_b,
                           PxbVec3 world_position, PxbVec3 normal,
                           PxbVec3 relative_velocity, float impulse_magnitude,
                           uint8_t persisting);

/* Tell an attached world the timestep the host is about to use, so injected
 * impulses convert to force correctly. Unnecessary for owned worlds. */
void pxb_note_dt(PxbWorld* w, float dt);

/* True when this world borrows a host scene rather than owning one. */
uint8_t pxb_world_is_attached(const PxbWorld* w);

PxbWorld* pxb_world_create(PxbVec3 gravity, uint8_t gpu, uint32_t cpu_threads);
void pxb_world_destroy(PxbWorld* w);
uint8_t pxb_world_gpu_active(const PxbWorld* w);

uint32_t pxb_capabilities(const PxbWorld* w);

/* Batched reads. Each writes exactly `count` entries. */
void pxb_read_bodies(const PxbWorld* w, const uint64_t* ids, uint32_t count,
                     PxbPose* out_pose, PxbVec3* out_linvel, PxbVec3* out_angvel,
                     uint8_t* out_flags, float* out_mass);
void pxb_read_center_of_mass(const PxbWorld* w, const uint64_t* ids, uint32_t count, PxbVec3* out_com);
void pxb_shape_parent(const PxbWorld* w, const uint64_t* shapes, uint32_t count, uint64_t* out_body);
void pxb_read_point_velocities(const PxbWorld* w, const uint64_t* ids, const PxbVec3* points,
                               uint32_t count, PxbVec3* out_vel);

/* out_created_* receive the ids minted by this call, index-parallel to the
   corresponding create list. Creation order is submission order. */
uint8_t pxb_apply(PxbWorld* w, uint32_t phase, const PxbCommands* cmds,
                  uint64_t* out_created_bodies, uint64_t* out_created_shapes,
                  PxbApplied* out_applied);

void pxb_step(PxbWorld* w, float dt);
uint32_t pxb_drain_contacts(PxbWorld* w, PxbContact* out, uint32_t capacity);
uint32_t pxb_dynamic_bodies(const PxbWorld* w, uint64_t* out, uint32_t capacity);

/* Sibling grace. PhysX cannot cheaply re-filter existing pairs
   (`resetFiltering` is expensive and wakes the actor), so this is applied as
   contact modification: the manifold is generated and its impulses zeroed. */
uint8_t pxb_set_excluded_pairs(PxbWorld* w, const uint64_t* a, const uint64_t* b, uint32_t count);

uint64_t pxb_capture_motion(PxbWorld* w, const uint64_t* scope, uint32_t scope_count);
uint8_t pxb_restore_motion(PxbWorld* w, uint64_t token, const uint64_t* scope, uint32_t scope_count);
void pxb_release_snapshot(PxbWorld* w, uint64_t token);

#ifdef __cplusplus
}
#endif
#endif
