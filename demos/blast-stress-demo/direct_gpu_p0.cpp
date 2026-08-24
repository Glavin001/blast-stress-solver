#include "physx_scene.h"

#include "PxContact.h"
#include "extensions/PxCudaHelpersExt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

using namespace physx;
using physx::Ext::PxCudaHelpersExt;
using namespace blast_demo;

constexpr PxRigidDynamicGPUIndex kInvalidRigidDynamicGpuIndex =
    std::numeric_limits<PxRigidDynamicGPUIndex>::max();

struct DirectBuffers
{
    PxCudaContextManager& cuda;
    PxRigidDynamicGPUIndex* indices{nullptr};
    PxTransform* poses{nullptr};
    PxVec3* velocities{nullptr};
    PxGpuContactPair* contactPairs{nullptr};
    PxU32* contactCount{nullptr};

    explicit DirectBuffers(PxCudaContextManager& manager)
        : cuda(manager)
    {
        indices = PxCudaHelpersExt::allocDeviceBuffer<PxRigidDynamicGPUIndex>(cuda, 2);
        poses = PxCudaHelpersExt::allocDeviceBuffer<PxTransform>(cuda, 2);
        velocities = PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(cuda, 2);
        contactPairs = PxCudaHelpersExt::allocDeviceBuffer<PxGpuContactPair>(cuda, 4096);
        contactCount = PxCudaHelpersExt::allocDeviceBuffer<PxU32>(cuda, 1);
        if (!indices || !poses || !velocities || !contactPairs || !contactCount)
        {
            throw std::runtime_error("failed to allocate Direct GPU API buffers");
        }
    }

    ~DirectBuffers()
    {
        PxCudaHelpersExt::freeDeviceBuffer(cuda, contactCount);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, contactPairs);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, velocities);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, poses);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, indices);
    }
};

struct Body
{
    PxRigidDynamic* actor{nullptr};
    PxShape* shape{nullptr};

    void release()
    {
        if (actor)
        {
            actor->release();
            actor = nullptr;
        }
        if (shape)
        {
            shape->release();
            shape = nullptr;
        }
    }
};

Body createBody(PhysXScene& context, const PxTransform& pose, const PxBoxGeometry& geometry)
{
    Body body;
    body.actor = context.physics().createRigidDynamic(pose);
    body.shape = context.physics().createShape(geometry, context.material(), false);
    if (!body.actor || !body.shape || !body.actor->attachShape(*body.shape))
    {
        body.release();
        throw std::runtime_error("failed to create P0 rigid body");
    }
    body.actor->setMass(1.0f);
    body.actor->setMassSpaceInertiaTensor(PxVec3(0.25f));
    body.actor->setLinearDamping(0.05f);
    body.actor->setAngularDamping(0.05f);
    return body;
}

bool finite(const PxTransform& transform)
{
    return std::isfinite(transform.p.x)
        && std::isfinite(transform.p.y)
        && std::isfinite(transform.p.z)
        && std::isfinite(transform.q.x)
        && std::isfinite(transform.q.y)
        && std::isfinite(transform.q.z)
        && std::isfinite(transform.q.w);
}

void step(PxScene& scene)
{
    scene.simulate(1.0f / 60.0f);
    if (!scene.fetchResults(true))
    {
        throw std::runtime_error("PhysX fetchResults failed");
    }
}

void uploadIndices(
    DirectBuffers& buffers,
    PxRigidDynamicGPUIndex first,
    PxRigidDynamicGPUIndex second,
    PxU32 count)
{
    const PxRigidDynamicGPUIndex host[2] = {first, second};
    PxCudaHelpersExt::copyHToD(buffers.cuda, buffers.indices, host, count);
}

void readPoses(
    PxDirectGPUAPI& api,
    DirectBuffers& buffers,
    PxTransform* host,
    PxU32 count)
{
    if (!api.getRigidDynamicData(
            buffers.poses,
            buffers.indices,
            PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE,
            count))
    {
        throw std::runtime_error("getRigidDynamicData(eGLOBAL_POSE) failed");
    }
    PxCudaHelpersExt::copyDToH(buffers.cuda, host, buffers.poses, count);
    for (PxU32 i = 0; i < count; ++i)
    {
        if (!finite(host[i]))
        {
            throw std::runtime_error("Direct GPU API returned a non-finite pose");
        }
    }
}

PxU32 readContactCount(PxDirectGPUAPI& api, DirectBuffers& buffers)
{
    if (!api.copyContactData(buffers.contactPairs, buffers.contactCount, 4096))
    {
        throw std::runtime_error("copyContactData failed");
    }
    PxU32 hostCount = 0;
    PxCudaHelpersExt::copyDToH(buffers.cuda, &hostCount, buffers.contactCount, 1);
    if (hostCount > 4096)
    {
        throw std::runtime_error("Direct GPU contact pair count exceeded buffer capacity");
    }
    return hostCount;
}

std::uint32_t parseIterations(int argc, char** argv)
{
    std::uint32_t iterations = 64;
    for (int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];
        if (option == "--iterations" && i + 1 < argc)
        {
            const unsigned long value = std::stoul(argv[++i]);
            if (value == 0 || value > 100000)
            {
                throw std::runtime_error("--iterations must be between 1 and 100000");
            }
            iterations = static_cast<std::uint32_t>(value);
        }
        else
        {
            throw std::runtime_error("usage: blast_direct_gpu_p0 [--iterations N]");
        }
    }
    return iterations;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::uint32_t iterations = parseIterations(argc, argv);
        SceneCapacity capacity;
        capacity.maxBodies = 256;
        capacity.maxShapes = 512;
        capacity.maxContactPairs = 16384;

        PhysXScene context(
            PhysicsMode::Gpu,
            true,
            capacity,
            nullptr,
            true);
        if (!context.directGpuApiActive() || !context.cudaContextManager())
        {
            throw std::runtime_error("Direct GPU API scene initialization failed");
        }

        PxScene& scene = context.scene();
        PxDirectGPUAPI& api = scene.getDirectGPUAPI();
        DirectBuffers buffers(*context.cudaContextManager());

        Body persistent = createBody(
            context,
            PxTransform(PxVec3(0.0f, 1.5f, 0.0f)),
            PxBoxGeometry(0.5f, 0.5f, 0.5f));
        PxShape* migratingShape =
            context.physics().createShape(PxSphereGeometry(0.25f), context.material(), false);
        if (!migratingShape || !persistent.actor->attachShape(*migratingShape))
        {
            if (migratingShape)
            {
                migratingShape->release();
            }
            persistent.release();
            throw std::runtime_error("failed to create migrating shape");
        }
        scene.addActor(*persistent.actor);

        step(scene);
        PxRigidDynamicGPUIndex persistentIndex = persistent.actor->getGPUIndex();
        if (persistentIndex == kInvalidRigidDynamicGpuIndex)
        {
            throw std::runtime_error("persistent actor has no GPU index after first step");
        }
        uploadIndices(buffers, persistentIndex, 0, 1);
        PxTransform initialPose[2];
        readPoses(api, buffers, initialPose, 1);

        std::uint64_t indexChanges = 0;
        std::uint64_t actorAdds = 0;
        std::uint64_t actorRemoves = 0;
        std::uint64_t shapeMigrations = 0;
        PxU32 peakContactPairs = 0;

        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration)
        {
            Body transient = createBody(
                context,
                PxTransform(PxVec3(2.0f + 0.01f * iteration, 2.0f, 0.0f)),
                PxBoxGeometry(0.35f, 0.35f, 0.35f));

            persistent.actor->detachShape(*migratingShape);
            if (!transient.actor->attachShape(*migratingShape))
            {
                transient.release();
                throw std::runtime_error("shape migration to transient actor failed");
            }
            ++shapeMigrations;
            scene.addActor(*transient.actor);
            ++actorAdds;

            step(scene);

            const PxRigidDynamicGPUIndex currentPersistentIndex =
                persistent.actor->getGPUIndex();
            const PxRigidDynamicGPUIndex transientIndex = transient.actor->getGPUIndex();
            if (currentPersistentIndex == kInvalidRigidDynamicGpuIndex
                || transientIndex == kInvalidRigidDynamicGpuIndex)
            {
                throw std::runtime_error("actor has no GPU index after topology mutation");
            }
            if (currentPersistentIndex != persistentIndex)
            {
                ++indexChanges;
                persistentIndex = currentPersistentIndex;
            }

            uploadIndices(buffers, persistentIndex, transientIndex, 2);
            PxTransform poses[2];
            readPoses(api, buffers, poses, 2);
            peakContactPairs = std::max(peakContactPairs, readContactCount(api, buffers));

            const PxVec3 hostVelocity[2] = {
                PxVec3(0.0f, -1.0f, 0.0f),
                PxVec3(-0.25f, 0.0f, 0.0f)};
            PxCudaHelpersExt::copyHToD(
                buffers.cuda,
                buffers.velocities,
                hostVelocity,
                2);
            if (!api.setRigidDynamicData(
                    buffers.velocities,
                    buffers.indices,
                    PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY,
                    2))
            {
                throw std::runtime_error("setRigidDynamicData(eLINEAR_VELOCITY) failed");
            }

            step(scene);
            uploadIndices(
                buffers,
                persistent.actor->getGPUIndex(),
                transient.actor->getGPUIndex(),
                2);
            readPoses(api, buffers, poses, 2);
            peakContactPairs = std::max(peakContactPairs, readContactCount(api, buffers));

            transient.actor->detachShape(*migratingShape);
            if (!persistent.actor->attachShape(*migratingShape))
            {
                throw std::runtime_error("shape migration back to persistent actor failed");
            }
            ++shapeMigrations;
            scene.removeActor(*transient.actor);
            ++actorRemoves;
            transient.release();
        }

        step(scene);
        const PxRigidDynamicGPUIndex finalIndex = persistent.actor->getGPUIndex();
        if (finalIndex == kInvalidRigidDynamicGpuIndex)
        {
            throw std::runtime_error("persistent actor lost its GPU index");
        }
        if (finalIndex != persistentIndex)
        {
            ++indexChanges;
        }
        uploadIndices(buffers, finalIndex, 0, 1);
        PxTransform finalPose[2];
        readPoses(api, buffers, finalPose, 1);
        peakContactPairs = std::max(peakContactPairs, readContactCount(api, buffers));

        scene.removeActor(*persistent.actor);
        persistent.release();
        migratingShape->release();

        if (!context.healthy())
        {
            throw std::runtime_error("PhysX reported an error during Direct GPU P0");
        }
        if (iterations >= 64 && peakContactPairs == 0)
        {
            throw std::runtime_error("Direct GPU contact API reported no ground contacts");
        }

        std::cout
            << "{\n"
            << "  \"schema\": \"blast-direct-gpu-p0-v1\",\n"
            << "  \"physxVersion\": \"" << PX_PHYSICS_VERSION_MAJOR << "."
            << PX_PHYSICS_VERSION_MINOR << "." << PX_PHYSICS_VERSION_BUGFIX << "\",\n"
            << "  \"iterations\": " << iterations << ",\n"
            << "  \"actorAdds\": " << actorAdds << ",\n"
            << "  \"actorRemoves\": " << actorRemoves << ",\n"
            << "  \"shapeMigrations\": " << shapeMigrations << ",\n"
            << "  \"persistentIndexChanges\": " << indexChanges << ",\n"
            << "  \"peakContactPairs\": " << peakContactPairs << ",\n"
            << "  \"passed\": true\n"
            << "}\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Direct GPU P0 failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
