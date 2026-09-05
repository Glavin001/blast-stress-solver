#ifndef NVBLASTEXTSTRESSPHYSXDIRECTGPU_H
#define NVBLASTEXTSTRESSPHYSXDIRECTGPU_H

#include "NvBlastExtStressPhysX.h"
#include <cstdint>

namespace Nv
{
namespace Blast
{

/** Motion-only checkpoint, not a complete scene/topology rollback.
 * Capture and restore must run outside simulate/fetchResults. Supplied actors
 * are borrowed: keep them alive until the next capture or buffer release.
 * A failed capture invalidates the checkpoint; GPU scenes never fall back to
 * stale CPU motion APIs. GPU checkpoints retain motion on the device.
 */
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
    // PhysX shape transform-cache identifiers for GPU shape-to-chunk routing.
    uint32_t transformCacheRef0;
    uint32_t transformCacheRef1;
};

// Every nonzero normal contact and friction anchor is a separate record.
// Keeping each point of application preserves the contact couple as well as
// total impulse. Actor pointers are opaque identifiers in device code.
// Record order is unspecified; consumers needing reproducible accumulation
// must order by stable ownership/contact keys rather than emission order.
struct ExtStressPhysXDirectGpuContactStatus
{
    uint32_t count;
    uint32_t overflow;
};

struct ExtStressPhysXDirectGpuContactView
{
    const ExtStressPhysXDirectGpuContact* contacts{nullptr};
    const ExtStressPhysXDirectGpuContactStatus* status{nullptr};
    uint32_t capacity{0};
    void* readyEvent{nullptr};
};

class NV_DLL_EXPORT ExtStressPhysXDirectGpuContactDrain
{
public:
    static ExtStressPhysXDirectGpuContactDrain* create(
        physx::PxScene& scene, uint32_t maxPairs = 65536);
    virtual void release() = 0;
    /** Decode directly into device records, with no host contact readback.
     * Call after fetchResults. Wait on readyEvent before consuming the view
     * or simulating again: decoding reads PhysX's transient contact buffers.
     * The view is borrowed until the next copy/release; finish consuming it
     * before either. status->overflow invalidates the entire result.
     */
    virtual bool copyContactsDevice(
        ExtStressPhysXDirectGpuContactView& view, uint32_t capacity) = 0;
    /// Synchronous reference/debug readback. Returns zero on any overflow.
    virtual uint32_t copyContacts(
        ExtStressPhysXDirectGpuContact* out, uint32_t capacity) = 0;
    /// Distinguishes an empty successful host copy from failure/truncation.
    virtual bool lastCopyComplete() const = 0;
    virtual bool available() const = 0;
protected:
    virtual ~ExtStressPhysXDirectGpuContactDrain() {}
};

} // namespace Blast
} // namespace Nv

#endif
