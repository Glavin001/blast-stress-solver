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
        : m_scene(scene)
        , m_available(directGpuApiAvailable(scene))
#if PX_SUPPORT_GPU_PHYSX
        , m_cuda(scene.getCudaContextManager())
#endif
    {
    }

    void release() override
    {
        releaseGpuBuffers();
        delete this;
    }

    bool capture(PxRigidDynamic* const* bodies, uint32_t count) override
    {
        if (!bodies || count == 0)
        {
            m_bodies.clear();
            return false;
        }

        m_bodies.assign(bodies, bodies + count);
        m_hostPoses.resize(count);
        m_hostLinearVelocities.resize(count);
        m_hostAngularVelocities.resize(count);
        m_gpuIndices.resize(count);

        for (uint32_t i = 0; i < count; ++i)
        {
            PxRigidDynamic* body = bodies[i];
            if (!body)
            {
                m_hostPoses[i] = PxTransform(PxIdentity);
                m_hostLinearVelocities[i] = PxVec3(0.0f);
                m_hostAngularVelocities[i] = PxVec3(0.0f);
                m_gpuIndices[i] = kInvalidRigidDynamicGpuIndex;
                continue;
            }

            m_hostPoses[i] = body->getGlobalPose();
            if (body->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
            {
                m_hostLinearVelocities[i] = PxVec3(0.0f);
                m_hostAngularVelocities[i] = PxVec3(0.0f);
            }
            else
            {
                m_hostLinearVelocities[i] = body->getLinearVelocity();
                m_hostAngularVelocities[i] = body->getAngularVelocity();
            }
            m_gpuIndices[i] = body->getGPUIndex();
        }

        if (!tryCaptureGpuState(count))
        {
            return true;
        }
        return true;
    }

    bool restore() override
    {
        if (m_bodies.empty())
        {
            return false;
        }

        refreshGpuIndices();
        if (!m_available)
        {
            return restoreViaCpu();
        }
        if (restoreViaGpu())
        {
            return true;
        }
        return restoreViaCpu();
    }

    uint32_t bodyCount() const override
    {
        return static_cast<uint32_t>(m_bodies.size());
    }

    bool available() const override
    {
        return m_available;
    }

private:
    void releaseGpuBuffers()
    {
#if PX_SUPPORT_GPU_PHYSX
        if (!m_cuda)
        {
            return;
        }
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_devicePoses);
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_deviceLinearVelocities);
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_deviceAngularVelocities);
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_deviceIndices);
#endif
    }

    bool ensureGpuBuffers(uint32_t count)
    {
#if PX_SUPPORT_GPU_PHYSX
        if (!m_cuda || count == 0)
        {
            return false;
        }
        if (m_deviceCapacity >= count)
        {
            return true;
        }

        releaseGpuBuffers();
        m_devicePoses =
            PxCudaHelpersExt::allocDeviceBuffer<PxTransform>(*m_cuda, count);
        m_deviceLinearVelocities =
            PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(*m_cuda, count);
        m_deviceAngularVelocities =
            PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(*m_cuda, count);
        m_deviceIndices = PxCudaHelpersExt::allocDeviceBuffer<PxRigidDynamicGPUIndex>(
            *m_cuda, count);
        if (!m_devicePoses || !m_deviceLinearVelocities || !m_deviceAngularVelocities
            || !m_deviceIndices)
        {
            releaseGpuBuffers();
            m_deviceCapacity = 0;
            return false;
        }
        m_deviceCapacity = count;
        return true;
#else
        (void)count;
        return false;
#endif
    }

    void refreshGpuIndices()
    {
        for (uint32_t i = 0; i < m_bodies.size(); ++i)
        {
            PxRigidDynamic* body = m_bodies[i];
            m_gpuIndices[i] =
                body ? body->getGPUIndex() : kInvalidRigidDynamicGpuIndex;
        }
    }

    bool tryCaptureGpuState(uint32_t count)
    {
#if PX_SUPPORT_GPU_PHYSX
        if (!m_available || !m_cuda || count == 0)
        {
            return false;
        }
        if (!ensureGpuBuffers(count))
        {
            return false;
        }

        bool copiedAny = false;
        PxDirectGPUAPI& api = m_scene.getDirectGPUAPI();
        PxCudaHelpersExt::copyHToD(*m_cuda, m_deviceIndices, m_gpuIndices.data(), count);

        if (api.getRigidDynamicData(
                m_devicePoses,
                m_deviceIndices,
                PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE,
                count))
        {
            copiedAny = true;
        }
        if (api.getRigidDynamicData(
                m_deviceLinearVelocities,
                m_deviceIndices,
                PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY,
                count))
        {
            copiedAny = true;
        }
        if (api.getRigidDynamicData(
                m_deviceAngularVelocities,
                m_deviceIndices,
                PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY,
                count))
        {
            copiedAny = true;
        }
        return copiedAny;
#else
        (void)count;
        return false;
#endif
    }

    bool restoreViaCpu()
    {
        for (uint32_t i = 0; i < m_bodies.size(); ++i)
        {
            PxRigidDynamic* body = m_bodies[i];
            if (!body || body->getScene() != &m_scene)
            {
                continue;
            }
            body->setGlobalPose(m_hostPoses[i], false);
            if (!body->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
            {
                body->setLinearVelocity(m_hostLinearVelocities[i], false);
                body->setAngularVelocity(m_hostAngularVelocities[i], false);
            }
        }
        return true;
    }

    bool restoreViaGpu()
    {
#if PX_SUPPORT_GPU_PHYSX
        if (!m_cuda || m_bodies.empty())
        {
            return false;
        }
        const uint32_t count = static_cast<uint32_t>(m_bodies.size());
        if (!ensureGpuBuffers(count))
        {
            return false;
        }

        PxDirectGPUAPI& api = m_scene.getDirectGPUAPI();
        PxCudaHelpersExt::copyHToD(*m_cuda, m_deviceIndices, m_gpuIndices.data(), count);
        PxCudaHelpersExt::copyHToD(*m_cuda, m_devicePoses, m_hostPoses.data(), count);
        PxCudaHelpersExt::copyHToD(
            *m_cuda, m_deviceLinearVelocities, m_hostLinearVelocities.data(), count);
        PxCudaHelpersExt::copyHToD(
            *m_cuda, m_deviceAngularVelocities, m_hostAngularVelocities.data(), count);

        bool restoredAny = false;
        if (api.setRigidDynamicData(
                m_devicePoses,
                m_deviceIndices,
                PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE,
                count))
        {
            restoredAny = true;
        }
        if (api.setRigidDynamicData(
                m_deviceLinearVelocities,
                m_deviceIndices,
                PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY,
                count))
        {
            restoredAny = true;
        }
        if (api.setRigidDynamicData(
                m_deviceAngularVelocities,
                m_deviceIndices,
                PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY,
                count))
        {
            restoredAny = true;
        }
        return restoredAny;
#else
        return false;
#endif
    }

    PxScene& m_scene;
    bool m_available{false};
#if PX_SUPPORT_GPU_PHYSX
    PxCudaContextManager* m_cuda{nullptr};
    PxTransform* m_devicePoses{nullptr};
    PxVec3* m_deviceLinearVelocities{nullptr};
    PxVec3* m_deviceAngularVelocities{nullptr};
    PxRigidDynamicGPUIndex* m_deviceIndices{nullptr};
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
        : m_scene(scene)
        , m_maxPairs(maxPairs > 0 ? maxPairs : 1)
        , m_available(directGpuApiAvailable(scene))
#if PX_SUPPORT_GPU_PHYSX
        , m_cuda(scene.getCudaContextManager())
#endif
    {
#if PX_SUPPORT_GPU_PHYSX
        if (m_available && m_cuda)
        {
            m_devicePairs = PxCudaHelpersExt::allocDeviceBuffer<PxGpuContactPair>(
                *m_cuda, m_maxPairs);
            m_deviceCount = PxCudaHelpersExt::allocDeviceBuffer<PxU32>(*m_cuda, 1);
            if (!m_devicePairs || !m_deviceCount)
            {
                releaseGpuBuffers();
                m_available = false;
            }
        }
#endif
    }

    void release() override
    {
        releaseGpuBuffers();
        delete this;
    }

    uint32_t copyContacts(
        ExtStressPhysXDirectGpuContact* out,
        uint32_t capacity) override
    {
        if (!out || capacity == 0 || !m_available)
        {
            return 0;
        }

#if PX_SUPPORT_GPU_PHYSX
        if (!m_cuda || !m_devicePairs || !m_deviceCount)
        {
            return 0;
        }

        PxDirectGPUAPI& api = m_scene.getDirectGPUAPI();
        if (!api.copyContactData(m_devicePairs, m_deviceCount, m_maxPairs))
        {
            return 0;
        }

        PxU32 hostCount = 0;
        PxCudaHelpersExt::copyDToH(*m_cuda, &hostCount, m_deviceCount, 1);
        if (hostCount > m_maxPairs)
        {
            hostCount = m_maxPairs;
        }

        std::vector<PxGpuContactPair> hostPairs(hostCount);
        if (hostCount > 0)
        {
            PxCudaHelpersExt::copyDToH(
                *m_cuda,
                hostPairs.data(),
                m_devicePairs,
                hostCount);
        }

        uint32_t written = 0;
        for (PxU32 i = 0; i < hostCount; ++i)
        {
            if (written >= capacity)
            {
                break;
            }
            out[written].actor0 = static_cast<PxRigidActor*>(hostPairs[i].actor0);
            out[written].actor1 = static_cast<PxRigidActor*>(hostPairs[i].actor1);
            out[written].worldPosition = PxVec3(0.0f);
            out[written].impulseOnActor0 = PxVec3(0.0f);
            ++written;
        }
        return written;
#else
        (void)out;
        (void)capacity;
        return 0;
#endif
    }

    bool available() const override
    {
        return m_available;
    }

private:
    void releaseGpuBuffers()
    {
#if PX_SUPPORT_GPU_PHYSX
        if (!m_cuda)
        {
            return;
        }
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_deviceCount);
        PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_devicePairs);
#endif
    }

    PxScene& m_scene;
    uint32_t m_maxPairs{0};
    bool m_available{false};
#if PX_SUPPORT_GPU_PHYSX
    PxCudaContextManager* m_cuda{nullptr};
    PxGpuContactPair* m_devicePairs{nullptr};
    PxU32* m_deviceCount{nullptr};
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
