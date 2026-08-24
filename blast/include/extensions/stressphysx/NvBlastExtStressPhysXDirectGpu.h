#ifndef NVBLASTEXTSTRESSPHYSXDIRECTGPU_H
#define NVBLASTEXTSTRESSPHYSXDIRECTGPU_H

#include "NvBlastExtStressPhysX.h"
#include <cstdint>

namespace Nv
{
namespace Blast
{

class NV_DLL_EXPORT ExtStressPhysXDirectGpuMotionBuffer
{
public:
    static ExtStressPhysXDirectGpuMotionBuffer* create(physx::PxScene& scene);
    virtual void release() = 0;
    virtual bool capture(physx::PxRigidDynamic* const* bodies, uint32_t count) = 0;
    virtual bool restore() = 0;
    virtual uint32_t bodyCount() const = 0;
    virtual bool available() const = 0;
protected:
    virtual ~ExtStressPhysXDirectGpuMotionBuffer() {}
};

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
        physx::PxScene& scene, uint32_t maxPairs = 65536);
    virtual void release() = 0;
    virtual uint32_t copyContacts(
        ExtStressPhysXDirectGpuContact* out, uint32_t capacity) = 0;
    virtual bool available() const = 0;
protected:
    virtual ~ExtStressPhysXDirectGpuContactDrain() {}
};

} // namespace Blast
} // namespace Nv

#endif
