// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <PxPhysicsAPI.h>
#include <cooking/PxCooking.h>
#include <cudamanager/PxCudaContextManager.h>

namespace blast_demo
{

enum class PhysicsMode
{
    Cpu,
    Gpu
};

struct SceneCapacity
{
    std::uint32_t maxBodies{10000};
    std::uint32_t maxShapes{10000};
    std::uint32_t maxContactPairs{65536};
};

class TrackingErrorCallback final : public physx::PxErrorCallback
{
public:
    void reportError(
        physx::PxErrorCode::Enum code,
        const char* message,
        const char* file,
        int line) override;

    bool hadGpuFailure() const { return m_gpuFailure; }
    bool hadCapacityWarning() const { return m_capacityWarning; }
    std::uint32_t warningCount() const { return m_warningCount; }
    const std::string& lastMessage() const { return m_lastMessage; }

private:
    bool m_gpuFailure{false};
    bool m_capacityWarning{false};
    std::uint32_t m_warningCount{0};
    std::string m_lastMessage;
};

class PhysXScene
{
public:
    PhysXScene(
        PhysicsMode mode,
        bool requireGpu,
        const SceneCapacity& capacity,
        physx::PxSimulationEventCallback* events,
        bool enableDirectGpuApi = false);
    ~PhysXScene();

    PhysXScene(const PhysXScene&) = delete;
    PhysXScene& operator=(const PhysXScene&) = delete;

    physx::PxPhysics& physics() const { return *m_physics; }
    const physx::PxCookingParams& cookingParams() const { return *m_cookingParams; }
    physx::PxScene& scene() const { return *m_scene; }
    physx::PxMaterial& material() const { return *m_material; }
    physx::PxCudaContextManager* cudaContextManager() const { return m_cuda; }
    PhysicsMode mode() const { return m_mode; }
    bool gpuActive() const;
    bool directGpuApiActive() const;
    bool healthy() const;
    const TrackingErrorCallback& errors() const { return m_errorCallback; }
    physx::PxSimulationStatistics statistics() const;

private:
    PhysicsMode m_mode;
    bool m_requireGpu;
    bool m_directGpuApiRequested;
    physx::PxDefaultAllocator m_allocator;
    TrackingErrorCallback m_errorCallback;
    physx::PxFoundation* m_foundation{nullptr};
    physx::PxPhysics* m_physics{nullptr};
    std::unique_ptr<physx::PxCookingParams> m_cookingParams;
    physx::PxDefaultCpuDispatcher* m_dispatcher{nullptr};
    physx::PxCudaContextManager* m_cuda{nullptr};
    physx::PxScene* m_scene{nullptr};
    physx::PxMaterial* m_material{nullptr};
    physx::PxRigidStatic* m_ground{nullptr};
};

} // namespace blast_demo
