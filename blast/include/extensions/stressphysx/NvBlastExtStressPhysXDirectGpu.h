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

#ifndef NVBLASTEXTSTRESSPHYSXDIRECTGPU_H
#define NVBLASTEXTSTRESSPHYSXDIRECTGPU_H

#include "NvBlastExtStressPhysX.h"

#include <cstdint>

namespace Nv
{
namespace Blast
{

/**
Batched Direct GPU motion-state access for an active body set. Requires a scene
created with eENABLE_DIRECT_GPU_API + GPU dynamics + GPU broadphase, and at least
one completed simulate/fetchResults so GPU indices are valid.

Topology edits (actor create / shape migrate) stay on the CPU mutation window.
Contact inject can optionally use copyContactData via
ExtStressPhysXDirectGpuContactDrain instead of CPU onContact callbacks.
*/
class NV_DLL_EXPORT ExtStressPhysXDirectGpuMotionBuffer
{
public:
    static ExtStressPhysXDirectGpuMotionBuffer* create(physx::PxScene& scene);

    virtual void release() = 0;

    /**
     * Capture global poses + linear/angular velocities for the given bodies
     * into device buffers (then mirrored to a host cache for restore). Returns
     * false if Direct GPU API is unavailable or any body lacks a GPU index.
     */
    virtual bool capture(
        physx::PxRigidDynamic* const* bodies,
        uint32_t count) = 0;

    /**
     * Write previously captured poses/velocities back to the GPU buffers for
     * the same body set (order must match the last capture).
     */
    virtual bool restore() = 0;

    virtual uint32_t bodyCount() const = 0;
    virtual bool available() const = 0;

protected:
    virtual ~ExtStressPhysXDirectGpuMotionBuffer() {}
};

/**
Drain GPU narrow-phase contacts into ExtStressPhysXDestructible::queueContact.
Host must map shape GPU indices / actor pointers to adapter shapes. This is the
Phase E contact path companion to Direct GPU motion state.
*/
struct ExtStressPhysXDirectGpuContact
{
    physx::PxRigidActor* actor0;
    physx::PxRigidActor* actor1;
    physx::PxVec3 worldPosition;
    physx::PxVec3 impulseOnActor0;
};

class NV_DLL_EXPORT ExtStressPhysXDirectGpuContactDrain
{
public:
    static ExtStressPhysXDirectGpuContactDrain* create(
        physx::PxScene& scene,
        uint32_t maxPairs = 65536);

    virtual void release() = 0;

    /**
     * Copies contact pairs from the Direct GPU API into a host scratch buffer.
     * Returns the number of contacts written (capped by capacity).
     */
    virtual uint32_t copyContacts(
        ExtStressPhysXDirectGpuContact* out,
        uint32_t capacity) = 0;

    virtual bool available() const = 0;

protected:
    virtual ~ExtStressPhysXDirectGpuContactDrain() {}
};

} // namespace Blast
} // namespace Nv

#endif // NVBLASTEXTSTRESSPHYSXDIRECTGPU_H
