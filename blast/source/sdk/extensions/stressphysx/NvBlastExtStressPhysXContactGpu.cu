#include "NvBlastExtStressPhysXDirectGpu.h"
#include <PxContact.h>
#include <cuda_runtime.h>

namespace Nv { namespace Blast {
namespace {
using namespace physx;
__device__ void emitContact(const PxGpuContactPair& pair, const PxVec3& point,
    const PxVec3& impulse, ExtStressPhysXDirectGpuContact* output,
    ExtStressPhysXDirectGpuContactStatus* status, PxU32 capacity)
{
    if (impulse.isZero()) { return; }
    const PxU32 slot = atomicAdd(&status->count, 1u);
    if (slot >= capacity) { atomicExch(&status->overflow, 1u); return; }
    output[slot] = {static_cast<PxRigidActor*>(pair.actor0),
        static_cast<PxRigidActor*>(pair.actor1), point, impulse,
        pair.transformCacheRef0, pair.transformCacheRef1};
}
__global__ void decodeContacts(const PxGpuContactPair* pairs, const PxU32* pairCount,
    PxU32 pairCapacity, ExtStressPhysXDirectGpuContact* output,
    ExtStressPhysXDirectGpuContactStatus* status, PxU32 capacity)
{
    const PxU32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && *pairCount > pairCapacity) { atomicExch(&status->overflow, 1u); }
    if (i >= *pairCount || i >= pairCapacity) { return; }
    const auto& p = pairs[i];
    if (p.nbContacts && p.contactPatches && p.contactPoints && p.contactForces)
    {
        PxContactStreamIterator contacts(p.contactPatches, p.contactPoints,
            nullptr, p.nbPatches, p.nbContacts);
        PxU32 pointIndex = 0;
        while (contacts.hasNextPatch())
        {
            contacts.nextPatch();
            while (contacts.hasNextContact())
            {
                contacts.nextContact();
                emitContact(p, contacts.getContactPoint(),
                    contacts.getContactNormal() * p.contactForces[pointIndex++],
                    output, status, capacity);
            }
        }
    }
    if (p.frictionPatches && p.contactPatches)
    {
        PxFrictionAnchorStreamIterator friction(p.contactPatches, p.frictionPatches, p.nbPatches);
        while (friction.hasNextPatch())
        {
            friction.nextPatch();
            while (friction.hasNextFrictionAnchor())
            {
                friction.nextFrictionAnchor();
                emitContact(p, friction.getPosition(), friction.getImpulse(), output, status, capacity);
            }
        }
    }
}
}
bool launchDirectGpuContactDecode(const physx::PxGpuContactPair* pairs,
    const physx::PxU32* pairCount, physx::PxU32 pairCapacity,
    ExtStressPhysXDirectGpuContact* output, ExtStressPhysXDirectGpuContactStatus* status,
    physx::PxU32 capacity, CUstream stream)
{
    decodeContacts<<<(pairCapacity + 127u) / 128u, 128, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(pairs, pairCount, pairCapacity,
            output, status, capacity);
    return cudaGetLastError() == cudaSuccess;
}
}}
