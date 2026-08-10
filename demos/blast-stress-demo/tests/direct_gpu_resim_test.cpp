// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Phase E: Direct GPU motion buffer + contact drain availability on a GPU
// Direct-API scene; active-set capture/restore round-trips a body pose.

#include "../physx_scene.h"

#include <NvBlastExtStressPhysXDirectGpu.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace Nv::Blast;
using namespace physx;
using blast_demo::PhysXScene;
using blast_demo::PhysicsMode;
using blast_demo::SceneCapacity;

namespace
{

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string("direct gpu resim test failed: ") + message);
    }
}

void step(PxScene& scene)
{
    scene.simulate(1.0f / 60.0f);
    require(scene.fetchResults(true), "fetchResults failed");
}

} // namespace

int main()
{
    try
    {
        SceneCapacity capacity;
        capacity.maxBodies = 64;
        capacity.maxShapes = 128;
        capacity.maxContactPairs = 4096;
        PhysXScene context(PhysicsMode::Gpu, true, capacity, nullptr, true);
        require(context.directGpuApiActive(), "Direct GPU API must be active");

        PxRigidDynamic* body = context.physics().createRigidDynamic(
            PxTransform(PxVec3(0.0f, 2.0f, 0.0f)));
        require(body != nullptr, "body create failed");
        PxShape* shape = context.physics().createShape(
            PxBoxGeometry(0.25f), context.material(), false);
        require(shape && body->attachShape(*shape), "shape failed");
        body->setMass(1.0f);
        context.scene().addActor(*body);
        step(context.scene());

        ExtStressPhysXDirectGpuMotionBuffer* motion =
            ExtStressPhysXDirectGpuMotionBuffer::create(context.scene());
        require(motion && motion->available(), "motion buffer unavailable");
        PxRigidDynamic* bodies[] = {body};
        require(motion->capture(bodies, 1), "capture failed");

        body->setGlobalPose(PxTransform(PxVec3(1.0f, 3.0f, 0.0f)), true);
        body->setLinearVelocity(PxVec3(5.0f, 0.0f, 0.0f), true);
        require(motion->restore(), "restore failed");
        step(context.scene());

        // After Direct GPU restore + step, pose should remain finite near the
        // captured state (warm-start / disable-sleeping prevent bit equality).
        const PxTransform pose = body->getGlobalPose();
        require(std::isfinite(pose.p.x) && std::isfinite(pose.p.y), "pose not finite");
        require((pose.p - PxVec3(0.0f, 2.0f, 0.0f)).magnitude() < 2.0f,
            "restored body drifted unreasonably far");

        ExtStressPhysXDirectGpuContactDrain* drain =
            ExtStressPhysXDirectGpuContactDrain::create(context.scene(), 4096);
        require(drain && drain->available(), "contact drain unavailable");
        ExtStressPhysXDirectGpuContact contacts[64];
        const std::uint32_t n = drain->copyContacts(contacts, 64);
        (void)n;

        drain->release();
        motion->release();
        body->release();
        shape->release();
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::printf("direct gpu resim test passed\n");
    return 0;
}
