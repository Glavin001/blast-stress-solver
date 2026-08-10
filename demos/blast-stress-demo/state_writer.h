// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <PxPhysicsAPI.h>

namespace blast_demo
{

struct Camera
{
    physx::PxVec3 eye{0.0f};
    physx::PxVec3 direction{0.0f, 0.0f, -1.0f};
    float fovDegrees{55.0f};
};

struct VisualActor
{
    enum class Shape : std::uint8_t
    {
        Box = 1,
        Sphere = 2
    };

    Shape shape{Shape::Box};
    std::uint8_t part{0};
    physx::PxVec3 parameters{0.5f};
    physx::PxTransform localPose{physx::PxIdentity};
};

struct VisualPose
{
    std::uint32_t actorId{0};
    physx::PxTransform pose{physx::PxIdentity};
    bool sleeping{false};
};

class StateWriter
{
public:
    StateWriter() = default;
    ~StateWriter();

    StateWriter(const StateWriter&) = delete;
    StateWriter& operator=(const StateWriter&) = delete;

    bool open(
        const std::string& path,
        std::uint32_t fps,
        std::uint32_t frameCount,
        std::uint32_t paneWidth,
        std::uint32_t paneHeight,
        std::uint32_t buildingCount,
        float durationSeconds,
        float settleSeconds,
        const std::array<Camera, 4>& cameras);
    bool defineActor(std::uint32_t id, const VisualActor& actor);
    bool writeFrame(std::uint32_t frameIndex, const std::vector<VisualPose>& poses);
    bool finish();
    const std::string& error() const { return m_error; }

private:
    bool writeTransform(const physx::PxTransform& value);
    bool writeVec3(const physx::PxVec3& value);
    bool writeU8(std::uint8_t value);
    bool writeU32(std::uint32_t value);
    bool writeF32(float value);
    bool writeBytes(const void* data, std::size_t size);
    bool fail(const std::string& message);

    std::FILE* m_file{nullptr};
    std::uint32_t m_nextActorId{0};
    std::vector<VisualPose> m_previousPoses;
    std::vector<bool> m_poseInitialized;
    std::string m_error;
};

} // namespace blast_demo
