// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#ifndef NVBLASTEXTSTRESSPHYSX_H
#define NVBLASTEXTSTRESSPHYSX_H

#include "NvBlastTypes.h"
#include "PxPhysicsAPI.h"

#include <cstdint>

namespace physx
{
// PhysX 4 exposes this interface. PhysX 5 moved runtime cooking to the public
// PxCreate* functions, but retaining the forward declaration keeps one host
// descriptor source-compatible with either SDK generation.
class PxCooking;
}

namespace Nv
{
namespace Blast
{

using ExtStressPhysXId = uint64_t;

enum class ExtStressPhysXGeometryType : uint32_t
{
    Cuboid,
    Convex
};

enum class ExtStressPhysXError : uint32_t
{
    None,
    InvalidDescriptor,
    AllocationFailed,
    SolverCreationFailed,
    BodyCreationFailed,
    ShapeCreationFailed,
    ConvexPointLimitExceeded,
    ConvexCookingFailed,
    SceneMutationFailed,
    FractureBufferOverflow,
    MappingInvalid
};

struct ExtStressPhysXNodeGeometry
{
    ExtStressPhysXGeometryType type;
    physx::PxTransform localPose;
    physx::PxVec3 halfExtents;
    const physx::PxVec3* convexPoints;
    uint32_t convexPointCount;

    ExtStressPhysXNodeGeometry()
        : type(ExtStressPhysXGeometryType::Cuboid)
        , localPose(physx::PxIdentity)
        , halfExtents(0.5f)
        , convexPoints(nullptr)
        , convexPointCount(0)
    {
    }
};

/**
Node coordinates are in structure space. A mass of zero marks a support node;
every actor containing such a node is represented by a kinematic
PxRigidDynamic. Geometry point storage only needs to remain valid for create().
*/
struct ExtStressPhysXNodeDesc
{
    physx::PxVec3 centroid;
    float mass;
    float volume;
    ExtStressPhysXNodeGeometry geometry;

    ExtStressPhysXNodeDesc()
        : centroid(0.0f)
        , mass(1.0f)
        , volume(1.0f)
    {
    }
};

struct ExtStressPhysXBondDesc
{
    physx::PxVec3 centroid;
    physx::PxVec3 normal;
    float area;
    uint32_t node0;
    uint32_t node1;

    ExtStressPhysXBondDesc()
        : centroid(0.0f)
        , normal(0.0f, 1.0f, 0.0f)
        , area(1.0f)
        , node0(0)
        , node1(0)
    {
    }
};

struct ExtStressPhysXSettings
{
    uint32_t maxSolverIterationsPerFrame;
    uint32_t graphReductionLevel;
    float compressionElasticLimit;
    float compressionFatalLimit;
    float tensionElasticLimit;
    float tensionFatalLimit;
    float shearElasticLimit;
    float shearFatalLimit;
    bool islandAware;
    bool skipSettledIslands;
    bool gpuStressSolver;
    uint32_t gpuStressMinimumBondCount;
    bool recordSplitContinuity;
    bool applyExcessForces;
    float excessForceScale;
    // Opt-in hard stop only: when >0 and the structure already has this many
    // bodies, skip further fracture. Default 0 = unlimited. Do not use this as
    // a perf "budget" — capping fracture falsifies impact/resim behavior.
    uint32_t maximumBodies;
    // Opt-in per-actor bond-break cap for a single tick. Default 0 = unlimited.
    // Same warning as maximumBodies: artificial limits degrade simulation quality.
    uint32_t maximumFracturesPerActorPerTick;
    // NOTE: this adapter never overrides a stress-solver fracture verdict and
    // never applies persistent velocity caps to fracture bodies. Joint strength
    // is authored (bond area = geometry, material limits = strength); anything
    // the solver says breaks, breaks. Earlier revisions carried support-bond
    // protection, impact-bond fatalization, and per-body velocity clamps as
    // compensations for authoring the solver could not yet express — see
    // PHYSICS_ENGINE_CONTRACT.md for the invariant that replaced them.
    bool idleSkip;
    bool baseStepSleep;
    float settledLinearSpeed;
    float settledAngularSpeed;
    float minimumSeparationVelocity;
    physx::PxShapeFlags shapeFlags;

    ExtStressPhysXSettings()
        : maxSolverIterationsPerFrame(25)
        , graphReductionLevel(0)
        , compressionElasticLimit(1.0f)
        , compressionFatalLimit(2.0f)
        , tensionElasticLimit(-1.0f)
        , tensionFatalLimit(-1.0f)
        , shearElasticLimit(-1.0f)
        , shearFatalLimit(-1.0f)
        , islandAware(true)
        , skipSettledIslands(true)
        , gpuStressSolver(false)
        , gpuStressMinimumBondCount(4096)
        , recordSplitContinuity(true)
        , applyExcessForces(true)
        , excessForceScale(1.0f)
        , maximumBodies(0)
        , maximumFracturesPerActorPerTick(0)
        , idleSkip(true)
        , baseStepSleep(false)
        , settledLinearSpeed(0.15f)
        , settledAngularSpeed(0.15f)
        , minimumSeparationVelocity(0.0f)
        , shapeFlags(physx::PxShapeFlag::eVISUALIZATION |
                     physx::PxShapeFlag::eSCENE_QUERY_SHAPE |
                     physx::PxShapeFlag::eSIMULATION_SHAPE)
    {
    }
};

struct ExtStressPhysXTelemetry
{
    uint64_t ticks;
    uint64_t contactsQueued;
    uint64_t contactsProcessed;
    uint64_t contactsDropped;
    uint64_t sleepingActorsSkipped;
    uint64_t splits;
    uint64_t bodiesCreated;
    uint64_t bodiesReused;
    uint64_t bodiesRecycled;
    uint64_t shapesMigrated;
    uint64_t convexPointLimitRejections;
    uint64_t convexCookingFailures;
    uint64_t mappingValidationFailures;
    uint32_t bodyCount;
    uint32_t awakeDynamicBodyCount;
    uint32_t overstressedBondCount;
    uint32_t solverIslandCount;
    uint32_t solverIslandsSkipped;
    float maxSplitWorldPositionDrift;
    float maxSplitPointVelocityDrift;
    double contactProcessingMilliseconds;
    double gravityMilliseconds;
    double stressSolveMilliseconds;
    double fractureTopologyMilliseconds;
    double mappingValidationMilliseconds;
    double gpuStressSolveMilliseconds;
    uint64_t gpuStressHostToDeviceBytes;
    uint64_t gpuStressDeviceToHostBytes;
    uint64_t resimulationCaptures;
    uint64_t resimulationRestores;
    uint64_t resimulationBodiesRestored;
    uint64_t resimulationBodiesRederived;
    double resimulationCaptureMilliseconds;
    double resimulationRestoreMilliseconds;
    float resimulationMaxRederivedDriftMeters;
    ExtStressPhysXError lastError;
    uint32_t lastErrorNode;

    ExtStressPhysXTelemetry();
};

struct ExtStressPhysXBodySnapshot
{
    ExtStressPhysXId bodyId;
    uint32_t actorIndex;
    physx::PxRigidDynamic* body;
    physx::PxTransform globalPose;
    physx::PxTransform centerOfMassLocalPose;
    physx::PxVec3 linearVelocity;
    physx::PxVec3 angularVelocity;
    uint32_t nodeCount;
    bool kinematic;
    bool sleeping;
};

struct ExtStressPhysXShapeSnapshot
{
    ExtStressPhysXId shapeId;
    ExtStressPhysXId bodyId;
    uint32_t nodeIndex;
    physx::PxShape* shape;
    physx::PxTransform worldPose;
    bool bodyKinematic;
    bool bodySleeping;
};

/**
Queue one contact for the adapter-owned shape. impulse is the world-space
impulse applied to that shape during the preceding simulation step. shapeId may
be used instead of shape; if both are supplied, shapeId takes precedence.
*/
struct ExtStressPhysXContact
{
    ExtStressPhysXId shapeId;
    const physx::PxShape* shape;
    physx::PxVec3 worldPosition;
    physx::PxVec3 worldImpulse;

    ExtStressPhysXContact()
        : shapeId(0)
        , shape(nullptr)
        , worldPosition(0.0f)
        , worldImpulse(0.0f)
    {
    }
};

struct ExtStressPhysXSplitContinuity
{
    uint64_t splitSequence;
    ExtStressPhysXId parentBodyId;
    uint32_t parentActorIndex;
    uint32_t reusedChildActorIndex;
    float maxWorldPositionDrift;
    float maxPointVelocityDrift;
};

using ExtStressPhysXErrorCallback =
    void (*)(ExtStressPhysXError error, uint32_t nodeIndex, const char* message, void* userData);

struct ExtStressPhysXDesc
{
    physx::PxPhysics* physics;
    physx::PxCooking* cooking;
    physx::PxScene* scene;
    physx::PxMaterial* material;
    const ExtStressPhysXNodeDesc* nodes;
    uint32_t nodeCount;
    const ExtStressPhysXBondDesc* bonds;
    uint32_t bondCount;
    physx::PxTransform worldTransform;
    ExtStressPhysXSettings settings;
    ExtStressPhysXErrorCallback errorCallback;
    void* errorUserData;

    ExtStressPhysXDesc()
        : physics(nullptr)
        , cooking(nullptr)
        , scene(nullptr)
        , material(nullptr)
        , nodes(nullptr)
        , nodeCount(0)
        , bonds(nullptr)
        , bondCount(0)
        , worldTransform(physx::PxIdentity)
        , errorCallback(nullptr)
        , errorUserData(nullptr)
    {
    }
};

/**
Owns the PhysX representation and C stress-bridge handle for one Blast family.
Call tick only after PxScene::fetchResults, so topology edits occur outside
simulation. The host retains ownership of all objects supplied in the descriptor.
*/
class NV_DLL_EXPORT ExtStressPhysXDestructible
{
public:
    static ExtStressPhysXDestructible* create(
        const ExtStressPhysXDesc& desc,
        ExtStressPhysXTelemetry* failureTelemetry = nullptr);

    virtual void release() = 0;
    virtual bool queueContact(const ExtStressPhysXContact& contact) = 0;
    virtual bool queueContact(
        const physx::PxShape& shape,
        const physx::PxVec3& worldPosition,
        const physx::PxVec3& worldImpulse) = 0;
    /**
     * Three-phase tick API for batching independent destructibles. beginTick()
     * and endTick() access PhysX and must run serially after fetchResults().
     * solveTick() only updates this destructible's stress solver and may run
     * concurrently with solveTick() on other destructibles.
     */
    virtual bool beginTick(float dt, const physx::PxVec3& worldGravity) = 0;
    virtual bool solveTick() = 0;
    virtual bool endTick() = 0;
    virtual bool tick(float dt, const physx::PxVec3& worldGravity) = 0;

    virtual bool validateMappings() = 0;
    virtual const ExtStressPhysXTelemetry& getTelemetry() const = 0;
    virtual bool usesGpuStressSolver() const = 0;

    virtual uint32_t getBodySnapshots(
        ExtStressPhysXBodySnapshot* snapshots,
        uint32_t capacity) const = 0;
    virtual uint32_t getShapeSnapshots(
        ExtStressPhysXShapeSnapshot* snapshots,
        uint32_t capacity) const = 0;
    /**
     * Returns shapes owned by awake dynamic bodies plus one final snapshot
     * when a body transitions to sleeping. Intended for delta state export.
     */
    virtual uint32_t getActiveShapeSnapshots(
        ExtStressPhysXShapeSnapshot* snapshots,
        uint32_t capacity) const = 0;
    virtual uint32_t getSplitContinuity(
        ExtStressPhysXSplitContinuity* records,
        uint32_t capacity) const = 0;

    /**
     * Per-bond stress from the last solve, indexed by the authored bond index
     * (the order bonds were supplied in ExtStressPhysXDescriptor::bonds), so a
     * host can relate stress back to the joint it authored.
     *
     * Compression and tension are mutually exclusive; shear is independent.
     * Values are pressures directly comparable to the descriptor's stress
     * limits: limit / stress is that joint's safety factor. Broken bonds read 0.
     * Any output pointer may be null. Returns the number of entries written.
     *
     * Sampling this after gravity settle and before any impact is the load-path
     * check: it says how much of each joint's capacity the structure's own
     * weight already consumes.
     */
    virtual uint32_t getBondStresses(
        float* compression,
        float* tension,
        float* shear,
        uint32_t capacity) const = 0;

    /**
     * Fracture-frame resimulation (engine contract §2.8). Capture before
     * PxScene::simulate; if the following tick fractured, restore between
     * fetchResults and the re-run simulate, then step and tick again so
     * contacts resolve against the already-split pieces.
     *
     * Restore rewinds motion state only — fracture topology, masses, shapes,
     * and kinematic flags are kept. Bodies that existed at capture get their
     * pose, velocities, and sleep state back (with a COM-shift velocity
     * correction for bodies whose mass frame moved during a split); bodies
     * created since the capture are re-placed relative to their source
     * parent's restored state. Kinematic bodies are pose-only.
     *
     * Both calls require the Idle tick phase and must run outside
     * simulate/fetchResults. capture returns the number of bodies recorded;
     * restore returns false if no capture is held or the phase is wrong.
     */
    virtual uint32_t captureResimulationSnapshot() = 0;
    /**
     * Restore motion for captured bodies. When activeBodies is null / activeCount
     * is 0, every captured body is restored (full-scene §2.8). Otherwise only
     * bodies present in the active set are rewound; provenance children of
     * restored parents are still re-derived.
     */
    virtual bool restoreResimulationSnapshot(
        physx::PxRigidDynamic* const* activeBodies = nullptr,
        uint32_t activeCount = 0) = 0;

    virtual bool needsResimulationSnapshot() const = 0;
    virtual uint32_t getResimulationSeedBodies(
        physx::PxRigidDynamic** bodies,
        uint32_t capacity) const = 0;
    virtual uint32_t applyBaseStepSleep() = 0;

    virtual ExtStressPhysXId getBodyId(const physx::PxRigidDynamic* body) const = 0;
    virtual ExtStressPhysXId getShapeId(const physx::PxShape* shape) const = 0;

protected:
    virtual ~ExtStressPhysXDestructible() {}
};

} // namespace Blast
} // namespace Nv

#endif // NVBLASTEXTSTRESSPHYSX_H
