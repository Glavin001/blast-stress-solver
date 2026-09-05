#include "extensions/stressphysx/NvBlastExtStressPhysXDirectGpu.h"

#include "PxContact.h"
#include "PxDirectGPUAPI.h"
#include "extensions/PxCudaHelpersExt.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace Nv
{
namespace Blast
{

using namespace physx;
using physx::Ext::PxCudaHelpersExt;

#if defined(NVBLAST_ENABLE_DIRECT_GPU_CONTACT_DRAIN)
bool launchDirectGpuContactDecode(const PxGpuContactPair*, const PxU32*, PxU32,
    ExtStressPhysXDirectGpuContact*, ExtStressPhysXDirectGpuContactStatus*, PxU32, CUstream);
#endif

namespace
{

constexpr PxRigidDynamicGPUIndex kInvalidRigidDynamicGpuIndex =
    std::numeric_limits<PxRigidDynamicGPUIndex>::max();

bool directGpuApiAvailable(PxScene& scene)
{
#if PX_SUPPORT_GPU_PHYSX
    return scene.getCudaContextManager() != nullptr
        && (scene.getFlags() & PxSceneFlag::eENABLE_DIRECT_GPU_API);
#else
    (void)scene;
    return false;
#endif
}

class ExtStressPhysXDirectGpuMotionBufferImpl final
    : public ExtStressPhysXDirectGpuMotionBuffer
{
public:
    explicit ExtStressPhysXDirectGpuMotionBufferImpl(PxScene& scene)
        : m_scene(scene), m_available(directGpuApiAvailable(scene))
#if PX_SUPPORT_GPU_PHYSX
        , m_cuda(scene.getCudaContextManager())
#endif
    {
    }

    void release() override
    {
        releaseGpuBuffers();
#if PX_SUPPORT_GPU_PHYSX
        if (m_complete)
        {
            PxScopedCudaLock lock(*m_cuda);
            m_cuda->getCudaContext()->eventDestroy(m_complete);
        }
#endif
        delete this;
    }

    bool capture(PxRigidDynamic* const* bodies, uint32_t count) override
    {
        // A failed capture invalidates the previous checkpoint. Replaying an
        // older tick is worse than reporting that no checkpoint is available.
        m_captured = false;
        m_bodies.clear();
        if ((m_scene.getFlags() & PxSceneFlag::eENABLE_DIRECT_GPU_API) && !m_available)
        {
            return false;
        }
        if (!bodies || count == 0)
        {
            return false;
        }
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!bodies[i] || bodies[i]->getScene() != &m_scene)
            {
                return false;
            }
        }
        m_bodies.assign(bodies, bodies + count);
        if (m_available)
        {
            // CPU motion getters are stale on a Direct GPU scene. The
            // checkpoint stays on the device through capture AND restore.
            m_captured = captureViaGpu(count);
        }
        else
        {
            m_hostPoses.resize(count);
            m_hostLinearVelocities.resize(count);
            m_hostAngularVelocities.resize(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                PxRigidDynamic& body = *bodies[i];
                const bool kinematic = body.getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC);
                m_hostPoses[i] = body.getGlobalPose();
                m_hostLinearVelocities[i] = kinematic ? PxVec3(0.0f) : body.getLinearVelocity();
                m_hostAngularVelocities[i] = kinematic ? PxVec3(0.0f) : body.getAngularVelocity();
            }
            m_captured = true;
        }
        return m_captured;
    }

    bool restore() override
    {
        if (!m_captured || m_bodies.empty())
        {
            return false;
        }
        // Validate the complete list before mutating any actor. Actors are
        // borrowed and must remain alive until the next capture or release.
        for (PxRigidDynamic* body : m_bodies)
        {
            if (body->getScene() != &m_scene)
            {
                m_captured = false;
                return false;
            }
        }
        if (m_available)
        {
            const bool restored = restoreViaGpu();
            if (!restored) { m_captured = false; }
            // Never fall back to CPU setters on a Direct GPU scene.
            return restored;
        }
        for (uint32_t i = 0; i < m_bodies.size(); ++i)
        {
            PxRigidDynamic& body = *m_bodies[i];
            body.setGlobalPose(m_hostPoses[i], false);
            if (!body.getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
            {
                body.setLinearVelocity(m_hostLinearVelocities[i], false);
                body.setAngularVelocity(m_hostAngularVelocities[i], false);
            }
        }
        return true;
    }

    uint32_t bodyCount() const override
    {
        return m_captured ? static_cast<uint32_t>(m_bodies.size()) : 0u;
    }

    bool available() const override { return m_available; }

private:
    void releaseGpuBuffers()
    {
#if PX_SUPPORT_GPU_PHYSX
        if (!m_cuda) { return; }
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_devicePoses);
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_deviceLinearVelocities);
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_deviceAngularVelocities);
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_deviceIndices);
        m_deviceCapacity = 0;
#endif
    }

#if PX_SUPPORT_GPU_PHYSX
    bool ensureGpuBuffers(uint32_t count)
    {
        if (!m_cuda || count == 0) { return false; }
        if (!m_complete)
        {
            PxScopedCudaLock lock(*m_cuda);
            // CU_EVENT_DISABLE_TIMING: dependency/completion only.
            if (m_cuda->getCudaContext()->eventCreate(&m_complete, 2u) != 0)
            {
                return false;
            }
        }
        if (m_deviceCapacity >= count) { return true; }
        releaseGpuBuffers();
        m_devicePoses = PxCudaHelpersExt::allocDeviceBuffer<PxTransform>(*m_cuda, count);
        m_deviceLinearVelocities = PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(*m_cuda, count);
        m_deviceAngularVelocities = PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(*m_cuda, count);
        m_deviceIndices = PxCudaHelpersExt::allocDeviceBuffer<PxRigidDynamicGPUIndex>(*m_cuda, count);
        if (!m_devicePoses || !m_deviceLinearVelocities || !m_deviceAngularVelocities || !m_deviceIndices)
        {
            releaseGpuBuffers();
            return false;
        }
        m_deviceCapacity = count;
        return true;
    }

    bool uploadIndices()
    {
        m_gpuIndices.resize(m_bodies.size());
        for (uint32_t i = 0; i < m_bodies.size(); ++i)
        {
            m_gpuIndices[i] = m_bodies[i]->getGPUIndex();
            if (m_gpuIndices[i] == kInvalidRigidDynamicGpuIndex) { return false; }
        }
        PxScopedCudaLock lock(*m_cuda);
        return m_cuda->getCudaContext()->memcpyHtoD(
            reinterpret_cast<CUdeviceptr>(m_deviceIndices), m_gpuIndices.data(),
            m_gpuIndices.size() * sizeof(PxRigidDynamicGPUIndex)) == 0;
    }

    bool waitForCopies()
    {
        PxScopedCudaLock lock(*m_cuda);
        return m_cuda->getCudaContext()->eventSynchronize(m_complete) == 0;
    }
#endif

    bool captureViaGpu(uint32_t count)
    {
#if PX_SUPPORT_GPU_PHYSX
        if (!ensureGpuBuffers(count) || !uploadIndices()) { return false; }
        PxDirectGPUAPI& api = m_scene.getDirectGPUAPI();
        // PhysX orders these on its stream. Passing a completion event avoids
        // three implicit host synchronizations; wait once for the whole batch.
        // Do not short-circuit: drain all dispatched operations even on error.
        const bool pose = api.getRigidDynamicData(m_devicePoses, m_deviceIndices,
            PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE, count, nullptr, m_complete);
        const bool linear = api.getRigidDynamicData(m_deviceLinearVelocities, m_deviceIndices,
            PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, count, nullptr, m_complete);
        const bool angular = api.getRigidDynamicData(m_deviceAngularVelocities, m_deviceIndices,
            PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY, count, nullptr, m_complete);
        const bool complete = waitForCopies();
        return pose && linear && angular && complete;
#else
        (void)count;
        return false;
#endif
    }

    bool restoreViaGpu()
    {
#if PX_SUPPORT_GPU_PHYSX
        if (!uploadIndices()) { return false; }
        const uint32_t count = static_cast<uint32_t>(m_bodies.size());
        PxDirectGPUAPI& api = m_scene.getDirectGPUAPI();
        const bool pose = api.setRigidDynamicData(m_devicePoses, m_deviceIndices,
            PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE, count, nullptr, m_complete);
        const bool linear = api.setRigidDynamicData(m_deviceLinearVelocities, m_deviceIndices,
            PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY, count, nullptr, m_complete);
        const bool angular = api.setRigidDynamicData(m_deviceAngularVelocities, m_deviceIndices,
            PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY, count, nullptr, m_complete);
        const bool complete = waitForCopies();
        return pose && linear && angular && complete;
#else
        return false;
#endif
    }

    PxScene& m_scene;
    bool m_available{false};
    bool m_captured{false};
#if PX_SUPPORT_GPU_PHYSX
    PxCudaContextManager* m_cuda{nullptr};
    PxTransform* m_devicePoses{nullptr};
    PxVec3* m_deviceLinearVelocities{nullptr};
    PxVec3* m_deviceAngularVelocities{nullptr};
    PxRigidDynamicGPUIndex* m_deviceIndices{nullptr};
    CUevent m_complete{nullptr};
    uint32_t m_deviceCapacity{0};
#endif
    std::vector<PxRigidDynamic*> m_bodies;
    std::vector<PxTransform> m_hostPoses;
    std::vector<PxVec3> m_hostLinearVelocities;
    std::vector<PxVec3> m_hostAngularVelocities;
    std::vector<PxRigidDynamicGPUIndex> m_gpuIndices;
};

class ExtStressPhysXDirectGpuContactDrainImpl final
    : public ExtStressPhysXDirectGpuContactDrain
{
public:
    ExtStressPhysXDirectGpuContactDrainImpl(PxScene& scene, uint32_t maxPairs)
        : m_scene(scene), m_maxPairs(std::max(1u, maxPairs))
    {
#if PX_SUPPORT_GPU_PHYSX && defined(NVBLAST_ENABLE_DIRECT_GPU_CONTACT_DRAIN)
        m_cuda = scene.getCudaContextManager();
        if (!directGpuApiAvailable(scene)) { return; }
        m_pairs = PxCudaHelpersExt::allocDeviceBuffer<PxGpuContactPair>(*m_cuda, m_maxPairs);
        m_pairCount = PxCudaHelpersExt::allocDeviceBuffer<PxU32>(*m_cuda, 1);
        m_status = PxCudaHelpersExt::allocDeviceBuffer<ExtStressPhysXDirectGpuContactStatus>(*m_cuda, 1);
        PxScopedCudaLock lock(*m_cuda);
        auto* cuda = m_cuda->getCudaContext();
        m_available = m_pairs && m_pairCount && m_status
            && cuda->streamCreate(&m_stream, 1u) == 0
            && cuda->eventCreate(&m_pairsReady, 2u) == 0
            && cuda->eventCreate(&m_ready, 2u) == 0;
#endif
    }

    void release() override
    {
#if PX_SUPPORT_GPU_PHYSX
        if (m_cuda)
        {
            // Drain outstanding decoding before releasing its source buffers.
            {
                PxScopedCudaLock lock(*m_cuda);
                auto* cuda = m_cuda->getCudaContext();
                if (m_pairsReady) { cuda->eventSynchronize(m_pairsReady); }
                if (m_stream) { cuda->streamSynchronize(m_stream); cuda->streamDestroy(m_stream); }
                if (m_pairsReady) { cuda->eventDestroy(m_pairsReady); }
                if (m_ready) { cuda->eventDestroy(m_ready); }
            }
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_contacts);
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_status);
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_pairCount);
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_pairs);
        }
#endif
        delete this;
    }

    bool copyContactsDevice(ExtStressPhysXDirectGpuContactView& view, uint32_t capacity) override
    {
        view = {};
        m_lastCopyComplete = false;
#if PX_SUPPORT_GPU_PHYSX && defined(NVBLAST_ENABLE_DIRECT_GPU_CONTACT_DRAIN)
        if (!m_available || !capacity) { return false; }
        if (capacity > m_capacity)
        {
            {
                PxScopedCudaLock lock(*m_cuda);
                if (m_cuda->getCudaContext()->streamSynchronize(m_stream) != 0) { return false; }
            }
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_contacts);
            m_capacity = 0;
            m_contacts = PxCudaHelpersExt::allocDeviceBuffer<ExtStressPhysXDirectGpuContact>(*m_cuda, capacity);
            if (!m_contacts) { return false; }
            m_capacity = capacity;
        }
        {
            PxScopedCudaLock lock(*m_cuda);
            auto* cuda = m_cuda->getCudaContext();
            // PhysX 5.10's empty-pair path neither writes the count nor
            // records finishEvent. Seed both on our stream, ordered after
            // any previous decode. On a nonempty call PhysX waits on this
            // recording, then records the same event after its own copy.
            // CUDA stream waits capture the event recording current at enqueue.
            if (cuda->memsetD32Async(reinterpret_cast<CUdeviceptr>(m_pairCount), 0, 1, m_stream) != 0
                || cuda->memsetD32Async(reinterpret_cast<CUdeviceptr>(m_status), 0,
                    sizeof(*m_status) / sizeof(PxU32), m_stream) != 0
                || cuda->eventRecord(m_pairsReady, m_stream) != 0) { return false; }
        }
        if (!m_scene.getDirectGPUAPI().copyContactData(m_pairs, m_pairCount,
            m_maxPairs, m_pairsReady, m_pairsReady)) { return false; }
        PxScopedCudaLock lock(*m_cuda);
        auto* cuda = m_cuda->getCudaContext();
        if (cuda->streamWaitEvent(m_stream, m_pairsReady, 0) != 0
            || !launchDirectGpuContactDecode(m_pairs, m_pairCount, m_maxPairs,
                m_contacts, m_status, capacity, m_stream)
            || cuda->eventRecord(m_ready, m_stream) != 0) { return false; }
        view.contacts = m_contacts;
        view.status = m_status;
        view.capacity = capacity;
        view.readyEvent = m_ready;
        return true;
#else
        (void)capacity;
        return false;
#endif
    }

    uint32_t copyContacts(ExtStressPhysXDirectGpuContact* out, uint32_t capacity) override
    {
        m_lastCopyComplete = false;
        if (!out) { return 0; }
        ExtStressPhysXDirectGpuContactView view;
        if (!copyContactsDevice(view, capacity)) { return 0; }
#if PX_SUPPORT_GPU_PHYSX
        PxScopedCudaLock lock(*m_cuda);
        auto* cuda = m_cuda->getCudaContext();
        ExtStressPhysXDirectGpuContactStatus status{};
        if (cuda->eventSynchronize(m_ready) != 0
            || cuda->memcpyDtoH(&status, reinterpret_cast<CUdeviceptr>(m_status), sizeof(status)) != 0
            || status.overflow || status.count > capacity) { return 0; }
        if (status.count && cuda->memcpyDtoH(out, reinterpret_cast<CUdeviceptr>(m_contacts),
            sizeof(*out) * status.count) != 0) { return 0; }
        m_lastCopyComplete = true;
        return status.count;
#else
        return 0;
#endif
    }
    bool available() const override { return m_available; }
    bool lastCopyComplete() const override { return m_lastCopyComplete; }

private:
    PxScene& m_scene;
    uint32_t m_maxPairs;
    bool m_available{false};
    bool m_lastCopyComplete{false};
#if PX_SUPPORT_GPU_PHYSX
    PxCudaContextManager* m_cuda{nullptr};
    PxGpuContactPair* m_pairs{nullptr};
    PxU32* m_pairCount{nullptr};
    ExtStressPhysXDirectGpuContactStatus* m_status{nullptr};
    ExtStressPhysXDirectGpuContact* m_contacts{nullptr};
    uint32_t m_capacity{0};
    CUstream m_stream{nullptr};
    CUevent m_pairsReady{nullptr}, m_ready{nullptr};
#endif
};

} // namespace

ExtStressPhysXDirectGpuMotionBuffer* ExtStressPhysXDirectGpuMotionBuffer::create(
    PxScene& scene)
{
    return new (std::nothrow) ExtStressPhysXDirectGpuMotionBufferImpl(scene);
}

ExtStressPhysXDirectGpuContactDrain* ExtStressPhysXDirectGpuContactDrain::create(
    PxScene& scene,
    uint32_t maxPairs)
{
    return new (std::nothrow) ExtStressPhysXDirectGpuContactDrainImpl(scene, maxPairs);
}

} // namespace Blast
} // namespace Nv
