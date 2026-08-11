// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "state_writer.h"

#include <cerrno>
#include <cstring>

namespace blast_demo
{
namespace
{

constexpr std::uint8_t kRecordActor = 1;
constexpr std::uint8_t kRecordFrame = 2;
constexpr std::uint8_t kRecordEnd = 255;

bool samePose(const VisualPose& a, const VisualPose& b)
{
    return a.sleeping == b.sleeping
        && a.pose.p.x == b.pose.p.x
        && a.pose.p.y == b.pose.p.y
        && a.pose.p.z == b.pose.p.z
        && a.pose.q.x == b.pose.q.x
        && a.pose.q.y == b.pose.q.y
        && a.pose.q.z == b.pose.q.z
        && a.pose.q.w == b.pose.q.w;
}

} // namespace

StateWriter::~StateWriter()
{
    if (m_file)
    {
        std::fclose(m_file);
    }
}

bool StateWriter::open(
    const std::string& path,
    std::uint32_t fps,
    std::uint32_t frameCount,
    std::uint32_t paneWidth,
    std::uint32_t paneHeight,
    std::uint32_t buildingCount,
    float durationSeconds,
    float settleSeconds,
    const std::array<Camera, 4>& cameras)
{
    if (m_file)
    {
        return fail("state writer is already open");
    }
    m_file = std::fopen(path.c_str(), "wb");
    if (!m_file)
    {
        return fail("cannot open state file '" + path + "': " + std::strerror(errno));
    }
    const char magic[8] = {'T', 'W', 'S', 'T', 'A', 'T', 'E', '1'};
    if (!writeBytes(magic, sizeof(magic))
        || !writeU32(2) // format version 2: adds VisualActor::Shape::Mesh
        || !writeU32(fps)
        || !writeU32(frameCount)
        || !writeU32(paneWidth)
        || !writeU32(paneHeight)
        || !writeU32(buildingCount)
        || !writeU32(static_cast<std::uint32_t>(cameras.size()))
        || !writeF32(durationSeconds)
        || !writeF32(settleSeconds))
    {
        return false;
    }
    for (const Camera& camera : cameras)
    {
        if (!writeVec3(camera.eye)
            || !writeVec3(camera.direction)
            || !writeF32(camera.fovDegrees))
        {
            return false;
        }
    }
    return true;
}

bool StateWriter::defineActor(std::uint32_t id, const VisualActor& actor)
{
    if (id != m_nextActorId)
    {
        return fail(
            "TWSTATE1 actor IDs must be contiguous; expected " + std::to_string(m_nextActorId)
            + ", got " + std::to_string(id));
    }
    const std::uint32_t shapeCount = 1;
    if (!writeU8(kRecordActor)
        || !writeU32(id)
        || !writeU8(actor.part)
        || !writeU32(shapeCount)
        || !writeU8(static_cast<std::uint8_t>(actor.shape))
        || !writeVec3(actor.parameters)
        || !writeTransform(actor.localPose))
    {
        return false;
    }
    if (actor.shape == VisualActor::Shape::Mesh)
    {
        if (actor.meshPositions.size() != actor.meshNormals.size())
        {
            return fail("mesh actor has mismatched position/normal counts");
        }
        if (!writeU32(static_cast<std::uint32_t>(actor.meshPositions.size())))
        {
            return false;
        }
        for (std::size_t i = 0; i < actor.meshPositions.size(); ++i)
        {
            if (!writeVec3(actor.meshPositions[i]) || !writeVec3(actor.meshNormals[i]))
            {
                return false;
            }
        }
        if (!writeU32(static_cast<std::uint32_t>(actor.meshIndices.size())))
        {
            return false;
        }
        for (std::uint32_t index : actor.meshIndices)
        {
            if (!writeU32(index))
            {
                return false;
            }
        }
    }
    ++m_nextActorId;
    m_previousPoses.emplace_back();
    m_poseInitialized.push_back(false);
    return true;
}

bool StateWriter::writeFrame(std::uint32_t frameIndex, const std::vector<VisualPose>& poses)
{
    std::vector<const VisualPose*> updates;
    updates.reserve(poses.size());
    for (const VisualPose& pose : poses)
    {
        if (pose.actorId >= m_nextActorId)
        {
            return fail("frame references an undefined visual actor");
        }
        if (!m_poseInitialized[pose.actorId]
            || !samePose(pose, m_previousPoses[pose.actorId]))
        {
            updates.push_back(&pose);
        }
    }
    if (!writeU8(kRecordFrame)
        || !writeU32(frameIndex)
        || !writeU32(static_cast<std::uint32_t>(updates.size())))
    {
        return false;
    }
    for (const VisualPose* pose : updates)
    {
        if (!writeU32(pose->actorId)
            || !writeTransform(pose->pose)
            || !writeU8(pose->sleeping ? 1 : 0))
        {
            return false;
        }
        m_previousPoses[pose->actorId] = *pose;
        m_poseInitialized[pose->actorId] = true;
    }
    return true;
}

bool StateWriter::finish()
{
    if (!writeU8(kRecordEnd))
    {
        return false;
    }
    if (std::fflush(m_file) != 0)
    {
        return fail(std::string("could not flush state file: ") + std::strerror(errno));
    }
    return true;
}

bool StateWriter::writeTransform(const physx::PxTransform& value)
{
    return writeVec3(value.p)
        && writeF32(value.q.x)
        && writeF32(value.q.y)
        && writeF32(value.q.z)
        && writeF32(value.q.w);
}

bool StateWriter::writeVec3(const physx::PxVec3& value)
{
    return writeF32(value.x) && writeF32(value.y) && writeF32(value.z);
}

bool StateWriter::writeU8(std::uint8_t value)
{
    return writeBytes(&value, sizeof(value));
}

bool StateWriter::writeU32(std::uint32_t value)
{
    return writeBytes(&value, sizeof(value));
}

bool StateWriter::writeF32(float value)
{
    return writeBytes(&value, sizeof(value));
}

bool StateWriter::writeBytes(const void* data, std::size_t size)
{
    if (!m_file || std::fwrite(data, 1, size, m_file) != size)
    {
        return fail(std::string("could not write state file: ") + std::strerror(errno));
    }
    return true;
}

bool StateWriter::fail(const std::string& message)
{
    if (m_error.empty())
    {
        m_error = message;
    }
    return false;
}

} // namespace blast_demo
