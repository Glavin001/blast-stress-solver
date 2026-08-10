#include "extensions/stressphysx/NvBlastExtStressPhysXDirectGpu.h"

#include "extensions/PxCudaHelpersExt.h"

#include <limits>
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

constexpr PxRigidDynamicGPUIndex kInvalidGpuIndex =
    std::numeric_limits<PxRigidDynamicGPUIndex>::max();

bool sceneSupportsDirectGpu(PxScene& scene)
{
    const PxSceneFlags flags = scene.getFlags();
    return (flags & PxSceneFlag::eENABLE_DIRECT_GPU_API)
        && (flags & PxSceneFlag::eENABLE_GPU_DYNAMICS)
        && scene.getCudaContextManager() != nullptr;
}

} // namespace

class ExtStressPhysXDirectGpuMotionBufferImpl final
    : public ExtStressPhysXDirectGpuMotionBuffer
{
public:
    explicit ExtStressPhysXDirectGpuMotionBufferImpl(PxScene& scene)
        : m_scene(scene)
        , m_cuda(scene.getCudaContextManager())
    {
        m_available = sceneSupportsDirectGpu(scene) && m_cuda != nullptr;
    }

    ~ExtStressPhysXDirectGpuMotionBufferImpl() override
    {
        freeDevice();
    }

    void release() override
    {
        delete this;
    }

    bool available() const override
    {
        return m_available;
    }

    uint32_t bodyCount() const override
    {
        return static_cast<uint32_t>(m_bodies.size());
    }

    bool capture(PxRigidDynamic* const* bodies, uint32_t count) override
    {
        if (!m_available || !bodies)
        {
            return false;
        }
        m_bodies.assign(bodies, bodies + count);
        m_indicesHost.resize(count);
        m_posesHost.resize(count);
        m_linvelHost.resize(count);
        m_angvelHost.resize(count);
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!bodies[i])
            {
                return false;
            }
            const PxRigidDynamicGPUIndex index = bodies[i]->getGPUIndex();
            if (index == kInvalidGpuIndex)
            {
                return false;
            }
            m_indicesHost[i] = index;
            m_posesHost[i] = bodies[i]->getGlobalPose();
            m_linvelHost[i] = bodies[i]->getLinearVelocity();
            m_angvelHost[i] = bodies[i]->getAngularVelocity();
        }
        if (!ensureDevice(count))
        {
            return false;
        }
        PxCudaHelpersExt::copyHToD(
            *m_cuda, m_indicesDevice, m_indicesHost.data(), count);
        // Prefer Direct GPU reads when indices are warm; fall back to the CPU
        // mirror captured above if the API rejects the query (e.g. pre-first-step).
        PxDirectGPUAPI& api = m_scene.getDirectGPUAPI();
        if (api.getRigidDynamicData(
                m_posesDevice,
                m_indicesDevice,
                PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE,
                count))
        {
            PxCudaHelpersExt::copyDToH(
                *m_cuda, m_posesHost.data(), m_posesDevice, count);
        }
        if (api.getRigidDynamicData(
                m_linvelDevice,
                m_indicesDevice,
                PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY,
                count))
        {
            PxCudaHelpersExt::copyDToH(
                *m_cuda, m_linvelHost.data(), m_linvelDevice, count);
        }
        if (api.getRigidDynamicData(
                m_angvelDevice,
                m_indicesDevice,
                PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY,
                count))
        {
            PxCudaHelpersExt::copyDToH(
                *m_cuda, m_angvelHost.data(), m_angvelDevice, count);
        }
        m_captured = count > 0;
        return true;
    }

    bool restore() override
    {
        if (!m_available || !m_captured || m_bodies.empty())
        {
            return false;
        }
        const uint32_t count = static_cast<uint32_t>(m_bodies.size());
        if (!ensureDevice(count))
        {
            return false;
        }
        // Refresh GPU indices — actor add/remove during fracture can invalidate
        // the previous capture's indices even when body pointers survive.
        for (uint32_t i = 0; i < count; ++i)
        {
            const PxRigidDynamicGPUIndex index = m_bodies[i]->getGPUIndex();
            if (index == kInvalidGpuIndex)
            {
                return false;
            }
            m_indicesHost[i] = index;
        }
        PxCudaHelpersExt::copyHToD(
            *m_cuda, m_indicesDevice, m_indicesHost.data(), count);
        PxCudaHelpersExt::copyHToD(
            *m_cuda, m_posesDevice, m_posesHost.data(), count);
        PxCudaHelpersExt::copyHToD(
            *m_cuda, m_linvelDevice, m_linvelHost.data(), count);
        PxCudaHelpersExt::copyHToD(
            *m_cuda, m_angvelDevice, m_angvelHost.data(), count);

        PxDirectGPUAPI& api = m_scene.getDirectGPUAPI();
        if (!api.setRigidDynamicData(
                m_posesDevice,
                m_indicesDevice,
                PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE,
                count))
        {
            return false;
        }
        if (!api.setRigidDynamicData(
                m_linvelDevice,
                m_indicesDevice,
                PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY,
                count))
        {
            return false;
        }
        if (!api.setRigidDynamicData(
                m_angvelDevice,
                m_indicesDevice,
                PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY,
                count))
        {
            return false;
        }
        return true;
    }

private:
    bool ensureDevice(uint32_t count)
    {
        if (count <= m_deviceCapacity)
        {
            return true;
        }
        freeDevice();
        m_indicesDevice =
            PxCudaHelpersExt::allocDeviceBuffer<PxRigidDynamicGPUIndex>(
                *m_cuda, count);
        m_posesDevice =
            PxCudaHelpersExt::allocDeviceBuffer<PxTransform>(*m_cuda, count);
        m_linvelDevice =
            PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(*m_cuda, count);
        m_angvelDevice =
            PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(*m_cuda, count);
        if (!m_indicesDevice || !m_posesDevice || !m_linvelDevice || !m_angvelDevice)
        {
            freeDevice();
            return false;
        }
        m_deviceCapacity = count;
        return true;
    }

    void freeDevice()
    {
        if (!m_cuda)
        {
            return;
        }
        if (m_angvelDevice)
        {
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_angvelDevice);
            m_angvelDevice = nullptr;
        }
        if (m_linvelDevice)
        {
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_linvelDevice);
            m_linvelDevice = nullptr;
        }
        if (m_posesDevice)
        {
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_posesDevice);
            m_posesDevice = nullptr;
        }
        if (m_indicesDevice)
        {
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_indicesDevice);
            m_indicesDevice = nullptr;
        }
        m_deviceCapacity = 0;
    }

    PxScene& m_scene;
    PxCudaContextManager* m_cuda{nullptr};
    bool m_available{false};
    bool m_captured{false};
    uint32_t m_deviceCapacity{0};
    std::vector<PxRigidDynamic*> m_bodies;
    std::vector<PxRigidDynamicGPUIndex> m_indicesHost;
    std::vector<PxTransform> m_posesHost;
    std::vector<PxVec3> m_linvelHost;
    std::vector<PxVec3> m_angvelHost;
    PxRigidDynamicGPUIndex* m_indicesDevice{nullptr};
    PxTransform* m_posesDevice{nullptr};
    PxVec3* m_linvelDevice{nullptr};
    PxVec3* m_angvelDevice{nullptr};
};

class ExtStressPhysXDirectGpuContactDrainImpl final
    : public ExtStressPhysXDirectGpuContactDrain
{
public:
    ExtStressPhysXDirectGpuContactDrainImpl(PxScene& scene, uint32_t maxPairs)
        : m_scene(scene)
        , m_cuda(scene.getCudaContextManager())
        , m_maxPairs(maxPairs)
    {
        m_available = sceneSupportsDirectGpu(scene) && m_cuda != nullptr && maxPairs > 0;
        if (!m_available)
        {
            return;
        }
        m_pairsDevice =
            PxCudaHelpersExt::allocDeviceBuffer<PxGpuContactPair>(*m_cuda, maxPairs);
        m_countDevice = PxCudaHelpersExt::allocDeviceBuffer<PxU32>(*m_cuda, 1);
        m_pairsHost.resize(maxPairs);
        if (!m_pairsDevice || !m_countDevice)
        {
            releaseBuffers();
            m_available = false;
        }
    }

    ~ExtStressPhysXDirectGpuContactDrainImpl() override
    {
        releaseBuffers();
    }

    void release() override
    {
        delete this;
    }

    bool available() const override
    {
        return m_available;
    }

    uint32_t copyContacts(
        ExtStressPhysXDirectGpuContact* out,
        uint32_t capacity) override
    {
        if (!m_available || !out || capacity == 0)
        {
            return 0;
        }
        PxDirectGPUAPI& api = m_scene.getDirectGPUAPI();
        if (!api.copyContactData(m_pairsDevice, m_countDevice, m_maxPairs))
        {
            return 0;
        }
        PxU32 hostCount = 0;
        PxCudaHelpersExt::copyDToH(*m_cuda, &hostCount, m_countDevice, 1);
        if (hostCount > m_maxPairs)
        {
            hostCount = m_maxPairs;
        }
        if (hostCount == 0)
        {
            return 0;
        }
        PxCudaHelpersExt::copyDToH(*m_cuda, m_pairsHost.data(), m_pairsDevice, hostCount);
        const uint32_t written =
            capacity < hostCount ? capacity : static_cast<uint32_t>(hostCount);
        for (uint32_t i = 0; i < written; ++i)
        {
            const PxGpuContactPair& pair = m_pairsHost[i];
            ExtStressPhysXDirectGpuContact& dst = out[i];
            dst.actor0 = pair.actor0;
            dst.actor1 = pair.actor1;
            // Contact point/impulse live in device patch buffers; hosts that
            // need forces continue to use CPU onContact, while this drain is
            // enough to feed island-exact contact-pair recording.
            dst.worldPosition = PxVec3(0.0f);
            dst.impulseOnActor0 = PxVec3(0.0f);
        }
        return written;
    }

private:
    void releaseBuffers()
    {
        if (!m_cuda)
        {
            return;
        }
        if (m_countDevice)
        {
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_countDevice);
            m_countDevice = nullptr;
        }
        if (m_pairsDevice)
        {
            PxCudaHelpersExt::freeDeviceBuffer(*m_cuda, m_pairsDevice);
            m_pairsDevice = nullptr;
        }
    }

    PxScene& m_scene;
    PxCudaContextManager* m_cuda{nullptr};
    uint32_t m_maxPairs{0};
    bool m_available{false};
    PxGpuContactPair* m_pairsDevice{nullptr};
    PxU32* m_countDevice{nullptr};
    std::vector<PxGpuContactPair> m_pairsHost;
};

ExtStressPhysXDirectGpuMotionBuffer* ExtStressPhysXDirectGpuMotionBuffer::create(
    PxScene& scene)
{
    return new (std::nothrow) ExtStressPhysXDirectGpuMotionBufferImpl(scene);
}

ExtStressPhysXDirectGpuContactDrain* ExtStressPhysXDirectGpuContactDrain::create(
    PxScene& scene,
    uint32_t maxPairs)
{
    return new (std::nothrow)
        ExtStressPhysXDirectGpuContactDrainImpl(scene, maxPairs);
}

} // namespace Blast
} // namespace Nv
