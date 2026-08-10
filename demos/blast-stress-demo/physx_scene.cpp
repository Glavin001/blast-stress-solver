// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "physx_scene.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <stdexcept>

#include <cudamanager/PxCudaContext.h>
#include <extensions/PxDefaultCpuDispatcher.h>
#include <extensions/PxDefaultSimulationFilterShader.h>
#include <extensions/PxExtensionsAPI.h>
#include <extensions/PxRigidActorExt.h>

namespace blast_demo
{
namespace
{

bool containsInsensitive(const std::string& value, const char* needle)
{
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    std::string target(needle);
    std::transform(target.begin(), target.end(), target.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return lower.find(target) != std::string::npos;
}

physx::PxFilterFlags contactFilter(
    physx::PxFilterObjectAttributes attributes0,
    physx::PxFilterData,
    physx::PxFilterObjectAttributes attributes1,
    physx::PxFilterData,
    physx::PxPairFlags& pairFlags,
    const void*,
    physx::PxU32)
{
    if (physx::PxFilterObjectIsTrigger(attributes0) || physx::PxFilterObjectIsTrigger(attributes1))
    {
        pairFlags = physx::PxPairFlag::eTRIGGER_DEFAULT;
        return physx::PxFilterFlag::eDEFAULT;
    }
    pairFlags = physx::PxPairFlag::eCONTACT_DEFAULT
        | physx::PxPairFlag::eNOTIFY_TOUCH_FOUND
        | physx::PxPairFlag::eNOTIFY_TOUCH_PERSISTS
        | physx::PxPairFlag::eNOTIFY_CONTACT_POINTS;
    return physx::PxFilterFlag::eDEFAULT;
}

std::uint32_t capacityScale(const SceneCapacity& capacity)
{
    std::uint64_t requested = std::max<std::uint64_t>(
        capacity.maxBodies,
        std::max<std::uint64_t>(capacity.maxShapes, capacity.maxContactPairs / 8));
    std::uint32_t scale = 1;
    while (requested > 8000 && scale < 16)
    {
        requested = (requested + 1) / 2;
        scale *= 2;
    }
    return scale;
}

} // namespace

void TrackingErrorCallback::reportError(
    physx::PxErrorCode::Enum code,
    const char* message,
    const char* file,
    int line)
{
    m_lastMessage = message ? message : "";
    if (code == physx::PxErrorCode::eDEBUG_WARNING || code == physx::PxErrorCode::ePERF_WARNING)
    {
        ++m_warningCount;
    }
    m_gpuFailure = m_gpuFailure
        || code == physx::PxErrorCode::eOUT_OF_MEMORY
        || containsInsensitive(m_lastMessage, "cuda error")
        || containsInsensitive(m_lastMessage, "gpu simulation fallback")
        || containsInsensitive(m_lastMessage, "failed to create cuda");
    m_capacityWarning = m_capacityWarning
        || containsInsensitive(m_lastMessage, "capacity")
        || containsInsensitive(m_lastMessage, "buffer overflow")
        || containsInsensitive(m_lastMessage, "discarding");
    std::fprintf(
        stderr,
        "[PhysX:%d] %s (%s:%d)\n",
        static_cast<int>(code),
        m_lastMessage.c_str(),
        file ? file : "",
        line);
}

PhysXScene::PhysXScene(
    PhysicsMode mode,
    bool requireGpu,
    const SceneCapacity& capacity,
    physx::PxSimulationEventCallback* events,
    bool enableDirectGpuApi)
    : m_mode(mode)
    , m_requireGpu(requireGpu)
    , m_directGpuApiRequested(enableDirectGpuApi)
{
    if (m_directGpuApiRequested && m_mode != PhysicsMode::Gpu)
    {
        throw std::runtime_error("Direct GPU API requires GPU physics mode");
    }
    m_foundation = PxCreateFoundation(
        PX_PHYSICS_VERSION,
        m_allocator,
        m_errorCallback);
    if (!m_foundation)
    {
        throw std::runtime_error("PxCreateFoundation failed");
    }

    m_physics = PxCreatePhysics(
        PX_PHYSICS_VERSION,
        *m_foundation,
        physx::PxTolerancesScale(),
        true,
        nullptr);
    if (!m_physics)
    {
        throw std::runtime_error("PxCreatePhysics failed");
    }
    if (!PxInitExtensions(*m_physics, nullptr))
    {
        throw std::runtime_error("PxInitExtensions failed");
    }

    m_cookingParams = std::make_unique<physx::PxCookingParams>(m_physics->getTolerancesScale());
    m_cookingParams->buildGPUData = true;
    m_cookingParams->convexMeshCookingType = physx::PxConvexMeshCookingType::eQUICKHULL;

    if (m_mode == PhysicsMode::Gpu)
    {
        physx::PxCudaContextManagerDesc cudaDesc;
        m_cuda = PxCreateCudaContextManager(*m_foundation, cudaDesc, PxGetProfilerCallback());
        if (!m_cuda || !m_cuda->contextIsValid())
        {
            if (m_cuda)
            {
                m_cuda->release();
                m_cuda = nullptr;
            }
            if (m_requireGpu)
            {
                throw std::runtime_error("a valid CUDA context is required but was not created");
            }
            std::fprintf(stderr, "CUDA unavailable; falling back to CPU PhysX\n");
            m_mode = PhysicsMode::Cpu;
        }
    }

    m_dispatcher = physx::PxDefaultCpuDispatcherCreate(4);
    if (!m_dispatcher)
    {
        throw std::runtime_error("PxDefaultCpuDispatcherCreate failed");
    }

    physx::PxSceneDesc desc(m_physics->getTolerancesScale());
    desc.gravity = physx::PxVec3(0.0f, -9.81f, 0.0f);
    desc.cpuDispatcher = m_dispatcher;
    desc.filterShader = contactFilter;
    desc.simulationEventCallback = events;
    desc.solverType = physx::PxSolverType::eTGS;
    desc.flags |= physx::PxSceneFlag::eENABLE_PCM;
    desc.flags |= physx::PxSceneFlag::eENABLE_STABILIZATION;

    if (m_mode == PhysicsMode::Gpu)
    {
        desc.cudaContextManager = m_cuda;
        desc.flags |= physx::PxSceneFlag::eENABLE_GPU_DYNAMICS;
        desc.broadPhaseType = physx::PxBroadPhaseType::eGPU;
        desc.gpuMaxNumPartitions = 8;
        if (m_directGpuApiRequested)
        {
            desc.flags |= physx::PxSceneFlag::eENABLE_DIRECT_GPU_API;
            desc.flags |= physx::PxSceneFlag::eDISABLE_SLEEPING;
        }

        const std::uint32_t scale = capacityScale(capacity);
        auto& gpu = desc.gpuDynamicsConfig;
        gpu.tempBufferCapacity *= scale;
        gpu.maxRigidContactCount = std::max(
            gpu.maxRigidContactCount * scale,
            capacity.maxContactPairs * 8u);
        gpu.maxRigidPatchCount = std::max(
            gpu.maxRigidPatchCount * scale,
            capacity.maxContactPairs * 2u);
        gpu.heapCapacity *= scale;
        gpu.foundLostPairsCapacity = std::max(
            gpu.foundLostPairsCapacity * scale,
            capacity.maxContactPairs * 2u);
        gpu.foundLostAggregatePairsCapacity *= scale;
        gpu.totalAggregatePairsCapacity *= scale;
        gpu.collisionStackSize *= scale;
    }

    if (!desc.isValid())
    {
        throw std::runtime_error("PhysX scene descriptor is invalid");
    }
    m_scene = m_physics->createScene(desc);
    if (!m_scene)
    {
        throw std::runtime_error("PhysX scene creation failed");
    }

    m_material = m_physics->createMaterial(0.6f, 0.6f, 0.05f);
    if (!m_material)
    {
        throw std::runtime_error("PhysX material creation failed");
    }
    m_ground = PxCreatePlane(
        *m_physics,
        physx::PxPlane(0.0f, 1.0f, 0.0f, 0.0f),
        *m_material);
    if (!m_ground)
    {
        throw std::runtime_error("PhysX ground creation failed");
    }
    m_scene->addActor(*m_ground);
}

PhysXScene::~PhysXScene()
{
    if (m_ground) m_ground->release();
    if (m_material) m_material->release();
    if (m_scene) m_scene->release();
    if (m_dispatcher) m_dispatcher->release();
    m_cookingParams.reset();
    if (m_physics)
    {
        PxCloseExtensions();
        m_physics->release();
    }
    if (m_cuda) m_cuda->release();
    if (m_foundation) m_foundation->release();
}

bool PhysXScene::gpuActive() const
{
    return m_mode == PhysicsMode::Gpu
        && m_cuda
        && m_cuda->contextIsValid()
        && m_scene
        && (m_scene->getFlags() & physx::PxSceneFlag::eENABLE_GPU_DYNAMICS);
}

bool PhysXScene::directGpuApiActive() const
{
    return gpuActive()
        && m_scene
        && (m_scene->getFlags() & physx::PxSceneFlag::eENABLE_DIRECT_GPU_API);
}

bool PhysXScene::healthy() const
{
    if (m_errorCallback.hadGpuFailure() || m_errorCallback.hadCapacityWarning())
    {
        return false;
    }
    if (m_requireGpu && !gpuActive())
    {
        return false;
    }
    if (m_directGpuApiRequested && !directGpuApiActive())
    {
        return false;
    }
    return !m_cuda || !m_cuda->getCudaContext()->isInAbortMode();
}

physx::PxSimulationStatistics PhysXScene::statistics() const
{
    physx::PxSimulationStatistics stats;
    if (m_scene)
    {
        m_scene->getSimulationStatistics(stats);
    }
    return stats;
}

} // namespace blast_demo
