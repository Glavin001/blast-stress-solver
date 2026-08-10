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
    uint32_t maximumBodies;
    uint32_t maximumFracturesPerActorPerTick;
    float maximumLinearVelocity;
    float maximumAngularVelocity;
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
        , maximumLinearVelocity(0.0f)
        , maximumAngularVelocity(0.0f)
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
    virtual bool tick(float dt, const physx::PxVec3& worldGravity) = 0;

    virtual bool validateMappings() = 0;
    virtual const ExtStressPhysXTelemetry& getTelemetry() const = 0;

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

    virtual ExtStressPhysXId getBodyId(const physx::PxRigidDynamic* body) const = 0;
    virtual ExtStressPhysXId getShapeId(const physx::PxShape* shape) const = 0;

protected:
    virtual ~ExtStressPhysXDestructible() {}
};

} // namespace Blast
} // namespace Nv

#endif // NVBLASTEXTSTRESSPHYSX_H
