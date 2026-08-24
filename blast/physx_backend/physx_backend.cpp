// PhysX implementation of the core's PhysicsBackend contract.
//
// Deliberately thin: it converts types, mints stable ids, honours the
// write-elision and ordering rules, and nothing else. No destruction logic
// lives here -- split planning, the motion fit, resim orchestration and debris
// policy are all in the Rust core, written once for every engine.
//
// PhysX specifics handled here so the core never sees them:
//   * `getGlobalPose()` is the ACTOR ORIGIN, not the centre of mass. The COM
//     is a separate local pose and must be composed in.
//   * Contacts report an IMPULSE (N*s); the core wants force, so divide by dt.
//   * Contact impulse SIGNS are ordering-dependent
//     (eINTERNAL_CONTACTS_ARE_FLIPPED is never corrected), so only magnitudes
//     are read and direction comes from the patch normal.
//   * `PxShape::setFlag` does NOT wake the owning actor, so shape toggles are
//     cheap -- unlike rigid-body property writes, which do wake and are
//     therefore elided when unchanged.
//   * Supports are kinematic PxRigidDynamic rather than PxRigidStatic, so the
//     flag can be flipped in place when a split changes support membership.

#include "physx_backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "PxPhysicsAPI.h"

using namespace physx;

namespace {

inline PxVec3 to_px(const PxbVec3& v) { return PxVec3(v.x, v.y, v.z); }
inline PxbVec3 from_px(const PxVec3& v) { PxbVec3 r{v.x, v.y, v.z}; return r; }
inline PxTransform to_px(const PxbPose& p) {
    return PxTransform(PxVec3(p.p.x, p.p.y, p.p.z), PxQuat(p.q.x, p.q.y, p.q.z, p.q.w));
}
inline PxbPose from_px(const PxTransform& t) {
    PxbPose r;
    r.p = from_px(t.p);
    r.q.x = t.q.x; r.q.y = t.q.y; r.q.z = t.q.z; r.q.w = t.q.w;
    return r;
}

struct ShapeRec {
    PxShape* shape = nullptr;
    uint64_t body = 0;
    uint32_t node = 0;
    float mass = 0.0f;
};

struct BodyRec {
    PxRigidDynamic* actor = nullptr;
    uint32_t kind = PXB_DYNAMIC;
    std::vector<uint64_t> shapes;
};

struct MotionRec {
    uint64_t body;
    PxTransform pose;
    PxVec3 linvel, angvel;
    PxVec3 world_com;
    float lin_damp, ang_damp;
    uint8_t sleeping;
    uint8_t kinematic;
};

class ErrCb : public PxErrorCallback {
public:
    void reportError(PxErrorCode::Enum code, const char* msg, const char* file, int line) override {
        if (msg && (strstr(msg, "capacity") || strstr(msg, "buffer overflow"))) capacity_warning = true;
        // Surface errors rather than swallowing them: a silently failed scene
        // is indistinguishable from a machine without a GPU, which is exactly
        // the misdiagnosis this cost once already.
        if (code & (PxErrorCode::eABORT | PxErrorCode::eINTERNAL_ERROR | PxErrorCode::eOUT_OF_MEMORY
                    | PxErrorCode::eINVALID_OPERATION | PxErrorCode::eINVALID_PARAMETER)) {
            fprintf(stderr, "[physx] error %d: %s (%s:%d)\n", (int)code, msg ? msg : "?",
                    file ? file : "?", line);
        }
    }
    bool capacity_warning = false;
};

} // namespace

// PhysX permits exactly one PxFoundation per process, so foundation, physics
// and the CPU dispatcher are shared and reference counted. Each PxbWorld then
// owns only its scene. Without this, constructing a second world in one
// process fails -- which is precisely what running two engines side by side in
// one test binary does.
namespace {

struct PxRuntime {
    PxDefaultAllocator alloc;
    ErrCb err;
    PxFoundation* foundation = nullptr;
    PxPhysics* physics = nullptr;
    PxMaterial* material = nullptr;
    PxCudaContextManager* cuda = nullptr;
    bool cuda_tried = false;
    int refs = 0;
};

PxRuntime& runtime() { static PxRuntime r; return r; }

/// Serialises every touch of the process-wide runtime.
///
/// Without this, two threads constructing worlds concurrently both observe
/// `!r.foundation`, both call `PxCreateFoundation`, and the process dies with
/// SIGSEGV -- observed as an intermittent crash in `byo_world_test`, which
/// builds three PhysX worlds and which cargo runs on three threads. It is not
/// a test artifact: any host that spins up destructibles from a thread pool
/// hits the same race.
///
/// Recursive because the failure paths in `pxb_world_create` call
/// `pxb_world_destroy`, which takes the same lock to run `runtime_release`.
std::recursive_mutex& runtime_mutex() { static std::recursive_mutex m; return m; }

bool runtime_acquire() {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    PxRuntime& r = runtime();
    // Key off the pointer, not the refcount. The runtime is never torn down
    // (PhysX cannot re-create a foundation in a process), so once it exists it
    // must be reused even after every world has been dropped -- otherwise the
    // next world tries to build a second foundation and PhysX refuses.
    if (!r.foundation) {
        r.foundation = PxCreateFoundation(PX_PHYSICS_VERSION, r.alloc, r.err);
        if (!r.foundation) return false;
        r.physics = PxCreatePhysics(PX_PHYSICS_VERSION, *r.foundation, PxTolerancesScale(), true, nullptr);
        if (!r.physics) { r.foundation->release(); r.foundation = nullptr; return false; }
        r.material = r.physics->createMaterial(0.25f, 0.25f, 0.0f);
    }
    ++r.refs;
    return true;
}

void runtime_release() {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    PxRuntime& r = runtime();
    if (r.refs > 0) --r.refs;
    // The runtime is deliberately NOT torn down at zero: PhysX does not
    // support re-creating a foundation after release within a process, and a
    // test binary constructs and drops many worlds.
}

PxCudaContextManager* runtime_cuda() {
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    PxRuntime& r = runtime();
    if (!r.cuda_tried) {
        r.cuda_tried = true;
        PxCudaContextManagerDesc d;
        r.cuda = PxCreateCudaContextManager(*r.foundation, d, PxGetProfilerCallback());
        if (r.cuda && !r.cuda->contextIsValid()) { r.cuda->release(); r.cuda = nullptr; }
    }
    return r.cuda;
}

} // namespace

struct PxbWorld : public PxSimulationEventCallback, public PxContactModifyCallback {
    PxDefaultCpuDispatcher* dispatcher = nullptr;
    PxScene* scene = nullptr;
    PxPhysics* physics = nullptr;   // borrowed from the shared runtime
    PxMaterial* material = nullptr; // borrowed from the shared runtime
    bool gpu = false;
    /// True when the scene belongs to the host: we neither step it nor release
    /// it, and contacts arrive by injection rather than from our own callback.
    bool attached = false;
    /// Material we created ourselves (attached worlds may not be handed one).
    PxMaterial* owned_material = nullptr;
    float last_dt = 1.0f / 60.0f;

    std::unordered_map<uint64_t, BodyRec> bodies;
    std::unordered_map<uint64_t, ShapeRec> shapes;
    std::unordered_map<const PxRigidActor*, uint64_t> actor_to_id;
    std::unordered_map<const PxShape*, uint64_t> shape_to_id;

    // Monotone and never reused, so a stale id is detectable rather than
    // silently aliasing a recycled actor.
    uint64_t next_body_id = 1;
    uint64_t next_shape_id = 1;

    std::vector<PxbContact> contacts;
    std::unordered_set<uint64_t> excluded;  // packed pair keys
    std::unordered_map<uint64_t, std::vector<MotionRec>> snapshots;
    uint64_t next_token = 1;

    // Retired actors are parked rather than released: a resim capture may
    // still hold the pointer, and a snapshot's restore checks getScene() on it,
    // so freeing here would make that check itself the crash.
    std::vector<PxRigidDynamic*> retired;

    static uint64_t pair_key(uint64_t a, uint64_t b) {
        uint64_t lo = a < b ? a : b, hi = a < b ? b : a;
        return lo * 1000003ull ^ (hi + 0x9e3779b97f4a7c15ull);
    }

    // ---- PxSimulationEventCallback ----
    void onContact(const PxContactPairHeader& header, const PxContactPair* pairs, PxU32 count) override {
        PxContactPairPoint pts[64];
        for (PxU32 i = 0; i < count; ++i) {
            const PxContactPair& p = pairs[i];
            if (p.flags & (PxContactPairFlag::eREMOVED_SHAPE_0 | PxContactPairFlag::eREMOVED_SHAPE_1)) continue;
            const PxU32 n = p.extractContacts(pts, 64);
            if (n == 0) continue;
            auto ita = shape_to_id.find(p.shapes[0]);
            auto itb = shape_to_id.find(p.shapes[1]);
            if (ita == shape_to_id.end() && itb == shape_to_id.end()) continue;

            // Relative velocity at the contact, from the header actors: a
            // shared shape reports a null getActor(), so the header is the only
            // place the actors are reachable.
            const PxRigidBody* a0 = header.actors[0] ? header.actors[0]->is<PxRigidBody>() : nullptr;
            const PxRigidBody* a1 = header.actors[1] ? header.actors[1]->is<PxRigidBody>() : nullptr;

            for (PxU32 k = 0; k < n; ++k) {
                PxVec3 v0(0.0f), v1(0.0f);
                if (a0) v0 = PxRigidBodyExt::getVelocityAtPos(*a0, pts[k].position);
                if (a1) v1 = PxRigidBodyExt::getVelocityAtPos(*a1, pts[k].position);
                PxbContact c{};
                c.shape_a = ita != shape_to_id.end() ? ita->second : 0;
                c.shape_b = itb != shape_to_id.end() ? itb->second : 0;
                c.world_position = from_px(pts[k].position);
                c.normal = from_px(pts[k].normal);
                // Impulse -> force. Magnitude only: the sign is ordering
                // dependent and PhysX never normalises it.
                c.force = pts[k].impulse.magnitude() / (last_dt > 0.0f ? last_dt : 1.0f);
                c.relative_velocity = from_px(v1 - v0);
                c.persisting = (p.events & PxPairFlag::eNOTIFY_TOUCH_PERSISTS) ? 1 : 0;
                contacts.push_back(c);
            }
        }
    }
    void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
    void onWake(PxActor**, PxU32) override {}
    void onSleep(PxActor**, PxU32) override {}
    void onTrigger(PxTriggerPair*, PxU32) override {}
    void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) override {}

    // ---- PxContactModifyCallback: the ZeroImpulse flavour of pair exclusion ----
    void onContactModify(PxContactModifyPair* const pairs, PxU32 count) override {
        if (excluded.empty()) return;
        for (PxU32 i = 0; i < count; ++i) {
            auto ia = actor_to_id.find(pairs[i].actor[0]);
            auto ib = actor_to_id.find(pairs[i].actor[1]);
            if (ia == actor_to_id.end() || ib == actor_to_id.end()) continue;
            if (!excluded.count(pair_key(ia->second, ib->second))) continue;
            // Zeroing max impulse suppresses the contact without needing a
            // refilter, which would be expensive and would wake the actors.
            for (PxU32 k = 0; k < pairs[i].contacts.size(); ++k) pairs[i].contacts.ignore(k);
        }
    }
};

static PxFilterFlags pxb_filter(PxFilterObjectAttributes a0, PxFilterData,
                                PxFilterObjectAttributes a1, PxFilterData,
                                PxPairFlags& pairFlags, const void*, PxU32) {
    PX_UNUSED(a0); PX_UNUSED(a1);
    pairFlags = PxPairFlag::eCONTACT_DEFAULT
              | PxPairFlag::eNOTIFY_TOUCH_FOUND
              // PERSISTS is the standing-load channel and is required for
              // correctness: a severed island gets no bond stress from gravity,
              // so the ground's continuous reaction is the only thing that
              // reproduces the load path its foundation used to provide.
              | PxPairFlag::eNOTIFY_TOUCH_PERSISTS
              | PxPairFlag::eNOTIFY_CONTACT_POINTS
              | PxPairFlag::eMODIFY_CONTACTS;
    return PxFilterFlag::eDEFAULT;
}

extern "C" {

PxbWorld* pxb_world_create(PxbVec3 gravity, uint8_t gpu, uint32_t cpu_threads) {
    // Held for the whole body, not just the acquire: `createScene` and the
    // CUDA context manager both allocate out of the shared PxPhysics.
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    if (!runtime_acquire()) return nullptr;
    PxRuntime& rt = runtime();
    PxbWorld* w = new PxbWorld();
    w->physics = rt.physics;
    w->material = rt.material;

    PxSceneDesc desc(rt.physics->getTolerancesScale());
    desc.gravity = to_px(gravity);
    w->dispatcher = PxDefaultCpuDispatcherCreate(cpu_threads ? cpu_threads : 2);
    desc.cpuDispatcher = w->dispatcher;
    desc.filterShader = pxb_filter;
    desc.simulationEventCallback = w;
    desc.contactModifyCallback = w;
    desc.solverType = PxSolverType::eTGS;
    desc.flags |= PxSceneFlag::eENABLE_PCM;
    desc.flags |= PxSceneFlag::eENABLE_STABILIZATION;

    if (gpu) {
        PxCudaContextManager* cuda = runtime_cuda();
        if (!cuda) { pxb_world_destroy(w); return nullptr; }
        desc.cudaContextManager = cuda;
        desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
        desc.broadPhaseType = PxBroadPhaseType::eGPU;
    }

    w->scene = rt.physics->createScene(desc);
    if (!w->scene) { pxb_world_destroy(w); return nullptr; }
    if (gpu) {
        // Never silently fall back: a CPU scene reported as GPU misreports
        // every measurement taken against it.
        const bool ok = w->scene->getFlags().isSet(PxSceneFlag::eENABLE_GPU_DYNAMICS)
                     && w->scene->getBroadPhaseType() == PxBroadPhaseType::eGPU;
        if (!ok) { pxb_world_destroy(w); return nullptr; }
        w->gpu = true;
    }
    return w;
}

PxbWorld* pxb_world_attach(void* scene, void* physics, void* material) {
    if (!scene || !physics) return nullptr;
    PxbWorld* w = new PxbWorld();
    w->scene = reinterpret_cast<PxScene*>(scene);
    w->physics = reinterpret_cast<PxPhysics*>(physics);
    w->attached = true;
    if (material) {
        w->material = reinterpret_cast<PxMaterial*>(material);
    } else {
        w->owned_material = w->physics->createMaterial(0.25f, 0.25f, 0.0f);
        w->material = w->owned_material;
    }
    // Deliberately no simulationEventCallback and no contactModifyCallback:
    // the host owns those slots. Pair exclusion is therefore unavailable on an
    // attached world, which capabilities() reports honestly.
    w->gpu = w->scene->getFlags().isSet(PxSceneFlag::eENABLE_GPU_DYNAMICS);
    return w;
}

void* pxb_world_scene(const PxbWorld* w) { return w ? (void*)w->scene : nullptr; }
void* pxb_world_physics(const PxbWorld* w) { return w ? (void*)w->physics : nullptr; }

uint8_t pxb_world_is_attached(const PxbWorld* w) { return (w && w->attached) ? 1 : 0; }

void pxb_note_dt(PxbWorld* w, float dt) { if (w && dt > 0.0f) w->last_dt = dt; }

uint8_t pxb_inject_contact(PxbWorld* w, void* shape_a, void* shape_b,
                           PxbVec3 world_position, PxbVec3 normal,
                           PxbVec3 relative_velocity, float impulse_magnitude,
                           uint8_t persisting) {
    if (!w) return 0;
    auto ia = w->shape_to_id.find(reinterpret_cast<const PxShape*>(shape_a));
    auto ib = w->shape_to_id.find(reinterpret_cast<const PxShape*>(shape_b));
    // A contact touching none of our shapes is the host's business.
    if (ia == w->shape_to_id.end() && ib == w->shape_to_id.end()) return 0;
    PxbContact c{};
    c.shape_a = ia != w->shape_to_id.end() ? ia->second : 0;
    c.shape_b = ib != w->shape_to_id.end() ? ib->second : 0;
    c.world_position = world_position;
    c.normal = normal;
    c.relative_velocity = relative_velocity;
    c.force = impulse_magnitude / (w->last_dt > 0.0f ? w->last_dt : 1.0f);
    c.persisting = persisting;
    w->contacts.push_back(c);
    return 1;
}

void pxb_world_destroy(PxbWorld* w) {
    if (!w) return;
    // Releasing a scene, its actors and the dispatcher all return memory to
    // the shared PxPhysics, so teardown races creation just as creation races
    // itself. Recursive, because `runtime_release` below re-enters.
    std::lock_guard<std::recursive_mutex> lock(runtime_mutex());
    if (w->attached) {
        // Borrowed scene: remove only what we added, then let the host keep it.
        for (auto& kv : w->bodies) {
            if (kv.second.actor && kv.second.actor->getScene()) w->scene->removeActor(*kv.second.actor);
            if (kv.second.actor) kv.second.actor->release();
        }
        for (auto* a : w->retired) if (a) a->release();
        if (w->owned_material) w->owned_material->release();
        delete w;
        return;
    }
    if (w->scene) w->scene->release();
    for (auto* a : w->retired) if (a) a->release();
    if (w->dispatcher) w->dispatcher->release();
    delete w;
    runtime_release();
}

uint8_t pxb_world_gpu_active(const PxbWorld* w) { return w && w->gpu ? 1 : 0; }

uint32_t pxb_capabilities(const PxbWorld* w) {
    // Mirrors Capabilities in the Rust core. PhysX uniquely offers a separate
    // scene-query toggle, so dormant geometry can stay raycastable.
    return (1u << 0) | (1u << 1) | (1u << 2) | (1u << 3) | (1u << 4) | (1u << 5) | (1u << 6)
         | (1u << 8)   // REPARENT_SHAPE
         | (1u << 9)   // SHAPE_SIMULATION_TOGGLE
         | (1u << 10)  // SHAPE_QUERY_TOGGLE
         | ((w && w->attached) ? 0u : (1u << 11))  // PAIR_EXCLUSION: needs our contact-modify callback
         | (1u << 13)  // DAMPING
         | (1u << 14)  // SLEEP_THRESHOLDS
         | (1u << 15)  // CCD
         | (1u << 16)  // IMPULSES
         | (1u << 17)  // MOTION_SNAPSHOT
         | (1u << 18)  // SCOPED_SNAPSHOT
         | (1u << 20)  // NATIVE_POINT_VELOCITY
         | (1u << 21); // CONTACT_MANIFOLDS
}

void pxb_read_bodies(const PxbWorld* w, const uint64_t* ids, uint32_t count,
                     PxbPose* out_pose, PxbVec3* out_linvel, PxbVec3* out_angvel,
                     uint8_t* out_flags, float* out_mass) {
    for (uint32_t i = 0; i < count; ++i) {
        auto it = w->bodies.find(ids[i]);
        if (it == w->bodies.end() || !it->second.actor) {
            out_pose[i] = from_px(PxTransform(PxIdentity));
            out_linvel[i] = out_angvel[i] = PxbVec3{0, 0, 0};
            out_flags[i] = 0;
            out_mass[i] = 0.0f;
            continue;
        }
        PxRigidDynamic* a = it->second.actor;
        const bool kin = a->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC);
        out_pose[i] = from_px(a->getGlobalPose());
        out_linvel[i] = from_px(a->getLinearVelocity());
        out_angvel[i] = from_px(a->getAngularVelocity());
        uint8_t f = PXB_FLAG_ENABLED;
        if (!kin && it->second.kind == PXB_DYNAMIC) f |= PXB_FLAG_DYNAMIC;
        if (kin) f |= PXB_FLAG_KINEMATIC;
        if (a->getScene() && a->isSleeping()) f |= PXB_FLAG_SLEEPING;
        out_flags[i] = f;
        out_mass[i] = a->getMass();
    }
}

void pxb_read_center_of_mass(const PxbWorld* w, const uint64_t* ids, uint32_t count, PxbVec3* out) {
    for (uint32_t i = 0; i < count; ++i) {
        auto it = w->bodies.find(ids[i]);
        if (it == w->bodies.end() || !it->second.actor) { out[i] = PxbVec3{0, 0, 0}; continue; }
        PxRigidDynamic* a = it->second.actor;
        // getGlobalPose() is the actor origin; compose the COM local pose in.
        out[i] = from_px(a->getGlobalPose().transform(a->getCMassLocalPose().p));
    }
}

void pxb_shape_parent(const PxbWorld* w, const uint64_t* shapes, uint32_t count, uint64_t* out) {
    for (uint32_t i = 0; i < count; ++i) {
        auto it = w->shapes.find(shapes[i]);
        out[i] = (it == w->shapes.end()) ? 0 : it->second.body;
    }
}

void pxb_read_point_velocities(const PxbWorld* w, const uint64_t* ids, const PxbVec3* pts,
                               uint32_t count, PxbVec3* out) {
    for (uint32_t i = 0; i < count; ++i) {
        auto it = w->bodies.find(ids[i]);
        if (it == w->bodies.end() || !it->second.actor) { out[i] = PxbVec3{0, 0, 0}; continue; }
        out[i] = from_px(PxRigidBodyExt::getVelocityAtPos(*it->second.actor, to_px(pts[i])));
    }
}

static void pxb_recompute_mass(PxbWorld* w, BodyRec& rec) {
    if (!rec.actor) return;
    std::vector<PxMassProperties> props;
    std::vector<PxTransform> xf;
    props.reserve(rec.shapes.size());
    xf.reserve(rec.shapes.size());
    for (uint64_t sid : rec.shapes) {
        auto si = w->shapes.find(sid);
        if (si == w->shapes.end() || !si->second.shape || si->second.mass <= 0.0f) continue;
        PxMassProperties p(si->second.shape->getGeometry());
        if (p.mass <= 0.0f) continue;
        p = p * (si->second.mass / p.mass);
        props.push_back(p);
        xf.push_back(si->second.shape->getLocalPose());
    }
    if (props.empty()) {
        rec.actor->setMass(1.0f);
        rec.actor->setCMassLocalPose(PxTransform(PxIdentity));
        rec.actor->setMassSpaceInertiaTensor(PxVec3(1.0f));
        return;
    }
    PxMassProperties combined = PxMassProperties::sum(props.data(), xf.data(), (PxU32)props.size());
    PxQuat frame;
    PxVec3 diag = PxMassProperties::getMassSpaceInertia(combined.inertiaTensor, frame);
    for (int k = 0; k < 3; ++k) diag[k] = PxMax(diag[k], 1e-6f);
    rec.actor->setMass(PxMax(combined.mass, 1e-6f));
    rec.actor->setCMassLocalPose(PxTransform(combined.centerOfMass, frame));
    rec.actor->setMassSpaceInertiaTensor(diag);
}

uint8_t pxb_apply(PxbWorld* w, uint32_t phase, const PxbCommands* c,
                  uint64_t* out_bodies, uint64_t* out_shapes, PxbApplied* applied) {
    PX_UNUSED(phase);
    PxbApplied done{};
    if (!w || !c) return 0;

    for (uint32_t i = 0; i < c->create_body_count; ++i) {
        const PxbCreateBody& cb = c->create_bodies[i];
        PxRigidDynamic* a = w->physics->createRigidDynamic(to_px(cb.pose));
        if (!a) return 0;
        const bool kinematic = (cb.kind != PXB_DYNAMIC);
        a->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, kinematic);
        if (!kinematic) {
            a->setLinearVelocity(to_px(cb.linvel));
            a->setAngularVelocity(to_px(cb.angvel));
            if (cb.ccd) a->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
        }
        a->setMass(1.0f);
        a->setMassSpaceInertiaTensor(PxVec3(1.0f));
        w->scene->addActor(*a);
        if (cb.start_sleeping && !kinematic) a->putToSleep();
        const uint64_t id = w->next_body_id++;
        BodyRec rec; rec.actor = a; rec.kind = cb.kind;
        w->bodies.emplace(id, std::move(rec));
        w->actor_to_id[a] = id;
        out_bodies[i] = id;
        ++done.bodies_created;
    }

    for (uint32_t i = 0; i < c->set_kind_count; ++i) {
        auto it = w->bodies.find(c->set_kind_ids[i]);
        if (it == w->bodies.end() || !it->second.actor) continue;
        const uint32_t want = c->set_kind_values[i];
        if (it->second.kind == want) { ++done.writes_elided; continue; }
        const bool kinematic = (want != PXB_DYNAMIC);
        it->second.actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, kinematic);
        it->second.kind = want;
    }

    for (uint32_t i = 0; i < c->set_pose_count; ++i) {
        auto it = w->bodies.find(c->set_pose_ids[i]);
        if (it != w->bodies.end() && it->second.actor)
            it->second.actor->setGlobalPose(to_px(c->set_pose_values[i]), false);
    }

    for (uint32_t i = 0; i < c->set_vel_count; ++i) {
        auto it = w->bodies.find(c->set_vel_ids[i]);
        if (it == w->bodies.end() || !it->second.actor) continue;
        // Velocity writes are rejected on kinematics; skip rather than error.
        if (it->second.actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC)) { ++done.writes_elided; continue; }
        const PxVec3 lin = to_px(c->set_vel_lin[i]), ang = to_px(c->set_vel_ang[i]);
        if ((it->second.actor->getLinearVelocity() - lin).magnitudeSquared() < 1e-12f &&
            (it->second.actor->getAngularVelocity() - ang).magnitudeSquared() < 1e-12f) {
            ++done.writes_elided; continue;
        }
        it->second.actor->setLinearVelocity(lin, false);
        it->second.actor->setAngularVelocity(ang, false);
    }

    for (uint32_t i = 0; i < c->create_shape_count; ++i) {
        const PxbCreateShape& cs = c->create_shapes[i];
        auto bi = w->bodies.find(cs.body);
        if (bi == w->bodies.end() || !bi->second.actor) return 0;
        PxShape* s = nullptr;
        if (cs.geom == 0) {
            PxBoxGeometry g(PxMax(cs.half_extents.x, 1e-4f), PxMax(cs.half_extents.y, 1e-4f), PxMax(cs.half_extents.z, 1e-4f));
            // Shared (non-exclusive) so identity survives migration between
            // bodies -- the edit the split planner exists to maximise.
            s = w->physics->createShape(g, *w->material, false);
        } else {
            std::vector<PxVec3> pts(cs.point_count);
            for (uint32_t k = 0; k < cs.point_count; ++k) pts[k] = to_px(cs.points[k]);
            PxConvexMeshDesc d;
            d.points.count = cs.point_count;
            d.points.stride = sizeof(PxVec3);
            d.points.data = pts.data();
            d.flags = PxConvexFlag::eCOMPUTE_CONVEX;
            // GPU rigid bodies cap hulls at 64 vertices. Asking the cooker to
            // honour that yields the optimal bounded hull; pre-thinning upstream
            // can only do worse.
            d.vertexLimit = 64;
            PxCookingParams cp(w->physics->getTolerancesScale());
            cp.buildGPUData = true;
            PxConvexMesh* mesh = PxCreateConvexMesh(cp, d, w->physics->getPhysicsInsertionCallback());
            if (!mesh) return 0;
            s = w->physics->createShape(PxConvexMeshGeometry(mesh), *w->material, false);
        }
        if (!s) return 0;
        s->setLocalPose(to_px(cs.local));
        bi->second.actor->attachShape(*s);
        const uint64_t sid = w->next_shape_id++;
        ShapeRec sr; sr.shape = s; sr.body = cs.body; sr.node = cs.node; sr.mass = cs.mass;
        w->shapes.emplace(sid, sr);
        w->shape_to_id[s] = sid;
        bi->second.shapes.push_back(sid);
        out_shapes[i] = sid;
        ++done.shapes_created;
    }

    for (uint32_t i = 0; i < c->reparent_count; ++i) {
        const PxbReparent& rp = c->reparent[i];
        auto si = w->shapes.find(rp.shape);
        auto to = w->bodies.find(rp.body);
        if (si == w->shapes.end() || to == w->bodies.end() || !si->second.shape) continue;
        auto from = w->bodies.find(si->second.body);
        if (from != w->bodies.end() && from->second.actor) {
            // false = do not wake on lost touch: waking every neighbour of every
            // migrating shape is the whole scene on a city collapse.
            from->second.actor->detachShape(*si->second.shape, false);
            auto& v = from->second.shapes;
            v.erase(std::remove(v.begin(), v.end(), rp.shape), v.end());
        }
        si->second.shape->setLocalPose(to_px(rp.local));
        to->second.actor->attachShape(*si->second.shape);
        si->second.body = rp.body;
        to->second.shapes.push_back(rp.shape);
        ++done.shapes_reparented;
    }

    for (uint32_t i = 0; i < c->remove_shape_count; ++i) {
        auto si = w->shapes.find(c->remove_shapes[i]);
        if (si == w->shapes.end() || !si->second.shape) continue;
        auto bi = w->bodies.find(si->second.body);
        if (bi != w->bodies.end() && bi->second.actor) {
            bi->second.actor->detachShape(*si->second.shape, false);
            auto& v = bi->second.shapes;
            v.erase(std::remove(v.begin(), v.end(), c->remove_shapes[i]), v.end());
        }
        w->shape_to_id.erase(si->second.shape);
        si->second.shape->release();
        w->shapes.erase(si);
    }

    for (uint32_t i = 0; i < c->recompute_mass_count; ++i) {
        auto bi = w->bodies.find(c->recompute_mass[i]);
        if (bi != w->bodies.end()) pxb_recompute_mass(w, bi->second);
    }

    for (uint32_t i = 0; i < c->impulse_count; ++i) {
        auto bi = w->bodies.find(c->impulse_ids[i]);
        if (bi == w->bodies.end() || !bi->second.actor) continue;
        if (bi->second.actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC)) continue;
        bi->second.actor->addForce(to_px(c->impulse_lin[i]), PxForceMode::eIMPULSE);
        bi->second.actor->addTorque(to_px(c->impulse_ang[i]), PxForceMode::eIMPULSE);
    }

    for (uint32_t i = 0; i < c->damping_count; ++i) {
        auto bi = w->bodies.find(c->damping_ids[i]);
        if (bi == w->bodies.end() || !bi->second.actor) continue;
        if (PxAbs(bi->second.actor->getLinearDamping() - c->damping_lin[i]) > 1e-9f)
            bi->second.actor->setLinearDamping(c->damping_lin[i]);
        else ++done.writes_elided;
        if (PxAbs(bi->second.actor->getAngularDamping() - c->damping_ang[i]) > 1e-9f)
            bi->second.actor->setAngularDamping(c->damping_ang[i]);
        else ++done.writes_elided;
    }

    for (uint32_t i = 0; i < c->sleep_thr_count; ++i) {
        auto bi = w->bodies.find(c->sleep_thr_ids[i]);
        if (bi == w->bodies.end() || !bi->second.actor) continue;
        const float t = PxMax(c->sleep_thr_lin[i], 0.0f);
        if (PxAbs(bi->second.actor->getSleepThreshold() - t) > 1e-9f)
            bi->second.actor->setSleepThreshold(t);
        else ++done.writes_elided;
    }

    for (uint32_t i = 0; i < c->ccd_count; ++i) {
        auto bi = w->bodies.find(c->ccd_ids[i]);
        if (bi == w->bodies.end() || !bi->second.actor) continue;
        const bool on = c->ccd_values[i] != 0;
        if (bi->second.actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD) == on) { ++done.writes_elided; continue; }
        bi->second.actor->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, on);
    }

    for (uint32_t i = 0; i < c->group_count; ++i) {
        auto si = w->shapes.find(c->group_shapes[i]);
        if (si == w->shapes.end() || !si->second.shape) continue;
        PxShape* s = si->second.shape;
        const PxU32 group  = c->group_memberships[i];
        const PxU32 mask   = c->group_filters[i];
        const PxU32 entity = c->group_entities ? c->group_entities[i] : 0U;
        // Layout mirrors the host's exactly, and deliberately so: simulation
        // data is (group, mask, entity), query data is (group, entity). A host
        // raycast tests `queryFilterData.word0 & mask`, so a shape with no
        // filter data is invisible to every query the host makes -- which reads
        // as "the shot missed" rather than as a configuration problem.
        const PxFilterData sim(group, mask, entity, 0);
        const PxFilterData query(group, entity, 0, 0);
        const PxFilterData cur_sim = s->getSimulationFilterData();
        const PxFilterData cur_q   = s->getQueryFilterData();
        const bool sim_same = cur_sim.word0 == sim.word0 && cur_sim.word1 == sim.word1
                           && cur_sim.word2 == sim.word2 && cur_sim.word3 == sim.word3;
        const bool q_same   = cur_q.word0 == query.word0 && cur_q.word1 == query.word1
                           && cur_q.word2 == query.word2 && cur_q.word3 == query.word3;
        if (sim_same && q_same) { ++done.writes_elided; continue; }
        // Writing shape filter data does NOT wake the owning actor, unlike a
        // rigid-body property write. Re-stamping every shape each tick would
        // otherwise be the mechanism that once held ~600 of ~735 bodies awake.
        if (!sim_same) s->setSimulationFilterData(sim);
        if (!q_same)   s->setQueryFilterData(query);
    }

    for (uint32_t i = 0; i < c->shape_enabled_count; ++i) {
        auto si = w->shapes.find(c->shape_enabled_ids[i]);
        if (si == w->shapes.end() || !si->second.shape) continue;
        const bool on = c->shape_enabled_values[i] != 0;
        PxShape* s = si->second.shape;
        if (s->getFlags().isSet(PxShapeFlag::eSIMULATION_SHAPE) == on) { ++done.writes_elided; continue; }
        // Does NOT wake the owning actor -- unlike a rigid-body property write.
        s->setFlag(PxShapeFlag::eSIMULATION_SHAPE, on);
    }

    for (uint32_t i = 0; i < c->wake_count; ++i) {
        auto bi = w->bodies.find(c->wake[i]);
        if (bi == w->bodies.end() || !bi->second.actor) continue;
        PxRigidDynamic* a = bi->second.actor;
        if (a->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC) || !a->getScene()) continue;
        if (a->isSleeping()) a->wakeUp(); else ++done.writes_elided;
    }
    for (uint32_t i = 0; i < c->sleep_count; ++i) {
        auto bi = w->bodies.find(c->sleep[i]);
        if (bi == w->bodies.end() || !bi->second.actor) continue;
        PxRigidDynamic* a = bi->second.actor;
        if (a->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC) || !a->getScene()) continue;
        if (!a->isSleeping()) a->putToSleep(); else ++done.writes_elided;
    }

    // Retirement last: every migration off these bodies has landed by now.
    for (uint32_t i = 0; i < c->remove_body_count; ++i) {
        auto bi = w->bodies.find(c->remove_bodies[i]);
        if (bi == w->bodies.end() || !bi->second.actor) continue;
        PxRigidDynamic* a = bi->second.actor;
        for (uint64_t sid : bi->second.shapes) {
            auto si = w->shapes.find(sid);
            if (si == w->shapes.end() || !si->second.shape) continue;
            a->detachShape(*si->second.shape, false);
            w->shape_to_id.erase(si->second.shape);
            si->second.shape->release();
            w->shapes.erase(si);
        }
        w->scene->removeActor(*a);
        w->actor_to_id.erase(a);
        w->retired.push_back(a);   // parked, not released -- see the note above
        w->bodies.erase(bi);
        ++done.bodies_removed;
    }

    if (applied) *applied = done;
    return 1;
}

void pxb_step(PxbWorld* w, float dt) {
    if (!w || !w->scene) return;
    w->last_dt = dt;
    // An attached world is driven by the host: it already called simulate and
    // fetchResults, and stepping again here would double-advance the scene the
    // rest of the game shares.
    if (w->attached) return;
    w->contacts.clear();
    w->scene->simulate(dt);
    w->scene->fetchResults(true);
}

uint32_t pxb_drain_contacts(PxbWorld* w, PxbContact* out, uint32_t cap) {
    if (!w) return 0;
    const uint32_t n = (uint32_t)w->contacts.size() < cap ? (uint32_t)w->contacts.size() : cap;
    for (uint32_t i = 0; i < n; ++i) out[i] = w->contacts[i];
    w->contacts.clear();
    return n;
}

uint32_t pxb_dynamic_bodies(const PxbWorld* w, uint64_t* out, uint32_t cap) {
    uint32_t n = 0;
    for (const auto& kv : w->bodies) {
        if (kv.second.kind != PXB_DYNAMIC) continue;
        if (n < cap) out[n] = kv.first;
        ++n;
    }
    return n;
}

uint8_t pxb_set_excluded_pairs(PxbWorld* w, const uint64_t* a, const uint64_t* b, uint32_t count) {
    if (!w) return 0;
    w->excluded.clear();
    for (uint32_t i = 0; i < count; ++i) w->excluded.insert(PxbWorld::pair_key(a[i], b[i]));
    return 1;
}

uint64_t pxb_capture_motion(PxbWorld* w, const uint64_t* scope, uint32_t scope_count) {
    if (!w) return 0;
    std::vector<MotionRec> recs;
    auto push = [&](uint64_t id, const BodyRec& r) {
        if (!r.actor) return;
        MotionRec m{};
        m.body = id;
        m.pose = r.actor->getGlobalPose();
        m.kinematic = r.actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC) ? 1 : 0;
        // Kinematics report meaningless velocities; store zeros so a restore
        // after a kind flip cannot inject garbage.
        m.linvel = m.kinematic ? PxVec3(0.0f) : r.actor->getLinearVelocity();
        m.angvel = m.kinematic ? PxVec3(0.0f) : r.actor->getAngularVelocity();
        m.world_com = r.actor->getGlobalPose().transform(r.actor->getCMassLocalPose().p);
        m.lin_damp = r.actor->getLinearDamping();
        m.ang_damp = r.actor->getAngularDamping();
        m.sleeping = (r.actor->getScene() && r.actor->isSleeping()) ? 1 : 0;
        recs.push_back(m);
    };
    if (scope_count == 0) { for (auto& kv : w->bodies) push(kv.first, kv.second); }
    else { for (uint32_t i = 0; i < scope_count; ++i) { auto it = w->bodies.find(scope[i]); if (it != w->bodies.end()) push(it->first, it->second); } }
    const uint64_t token = w->next_token++;
    w->snapshots.emplace(token, std::move(recs));
    return token;
}

uint8_t pxb_restore_motion(PxbWorld* w, uint64_t token, const uint64_t* scope, uint32_t scope_count) {
    if (!w) return 0;
    auto sit = w->snapshots.find(token);
    if (sit == w->snapshots.end()) return 1;
    std::unordered_set<uint64_t> scoped;
    for (uint32_t i = 0; i < scope_count; ++i) scoped.insert(scope[i]);
    for (const MotionRec& m : sit->second) {
        if (scope_count && !scoped.count(m.body)) continue;
        auto it = w->bodies.find(m.body);
        // Bodies created since the capture are absent and are simply skipped:
        // that is what lets new fragments survive a rollback.
        if (it == w->bodies.end() || !it->second.actor || !it->second.actor->getScene()) continue;
        PxRigidDynamic* a = it->second.actor;
        a->setGlobalPose(m.pose, false);
        if (a->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC)) continue;  // pose-only
        // The stored linvel is the velocity at the captured centre of mass; a
        // split may have moved this body's mass frame, so re-express it.
        const PxVec3 restored_com = m.pose.transform(a->getCMassLocalPose().p);
        a->setLinearVelocity(m.linvel + m.angvel.cross(restored_com - m.world_com), false);
        a->setAngularVelocity(m.angvel, false);
        a->clearForce(PxForceMode::eFORCE);
        a->clearForce(PxForceMode::eIMPULSE);
        a->clearTorque(PxForceMode::eFORCE);
        a->clearTorque(PxForceMode::eIMPULSE);
        a->setLinearDamping(m.lin_damp);
        a->setAngularDamping(m.ang_damp);
        if (m.sleeping) a->putToSleep(); else a->wakeUp();
    }
    return 1;
}

void pxb_release_snapshot(PxbWorld* w, uint64_t token) { if (w) w->snapshots.erase(token); }

} // extern "C"
