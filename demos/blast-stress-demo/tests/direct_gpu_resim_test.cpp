// GPU checkpoints must restore device motion, including non-zero angular
// velocity. CPU getters/setters are deliberately never used after insertion.
#include "../physx_scene.h"
#include <NvBlastExtStressPhysXDirectGpu.h>
#include <extensions/PxCudaHelpersExt.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Nv::Blast;
using namespace physx;
using physx::Ext::PxCudaHelpersExt;
using blast_demo::PhysXScene;
using blast_demo::PhysicsMode;
using blast_demo::SceneCapacity;

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}
void step(PxScene& scene)
{
    scene.simulate(1.0f / 60.0f);
    require(scene.fetchResults(true), "fetchResults failed");
}
struct State
{
    PxTransform pose;
    PxVec3 linear, angular;
};
struct Buffers
{
    PxCudaContextManager& cuda;
    PxRigidDynamicGPUIndex* indices;
    PxTransform* poses;
    PxVec3* velocities;
    Buffers(PxCudaContextManager& manager, uint32_t count) : cuda(manager)
    {
        indices = PxCudaHelpersExt::allocDeviceBuffer<PxRigidDynamicGPUIndex>(cuda, count);
        poses = PxCudaHelpersExt::allocDeviceBuffer<PxTransform>(cuda, count);
        velocities = PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(cuda, count);
        require(indices && poses && velocities, "test buffer allocation failed");
    }
    ~Buffers()
    {
        PxCudaHelpersExt::freeDeviceBuffer(cuda, velocities);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, poses);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, indices);
    }
    void bind(const std::vector<PxRigidDynamic*>& bodies)
    {
        std::vector<PxRigidDynamicGPUIndex> host;
        for (auto* body : bodies) { host.push_back(body->getGPUIndex()); }
        PxCudaHelpersExt::copyHToD(cuda, indices, host.data(), host.size());
    }
    std::vector<State> read(PxScene& scene, uint32_t count)
    {
        auto& api = scene.getDirectGPUAPI();
        std::vector<State> states(count);
        std::vector<PxTransform> hostPoses(count);
        std::vector<PxVec3> hostVelocities(count);
        require(api.getRigidDynamicData(poses, indices, PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE, count), "device pose read failed");
        PxCudaHelpersExt::copyDToH(cuda, hostPoses.data(), poses, count);
        require(api.getRigidDynamicData(velocities, indices, PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, count), "device linear read failed");
        PxCudaHelpersExt::copyDToH(cuda, hostVelocities.data(), velocities, count);
        for (uint32_t i = 0; i < count; ++i) { states[i].pose = hostPoses[i]; states[i].linear = hostVelocities[i]; }
        require(api.getRigidDynamicData(velocities, indices, PxRigidDynamicGPUAPIReadType::eANGULAR_VELOCITY, count), "device angular read failed");
        PxCudaHelpersExt::copyDToH(cuda, hostVelocities.data(), velocities, count);
        for (uint32_t i = 0; i < count; ++i) { states[i].angular = hostVelocities[i]; }
        return states;
    }
    void overwrite(PxScene& scene, const std::vector<State>& states)
    {
        auto& api = scene.getDirectGPUAPI();
        const auto count = static_cast<uint32_t>(states.size());
        std::vector<PxTransform> hostPoses;
        std::vector<PxVec3> linear, angular;
        for (const auto& state : states)
        {
            hostPoses.push_back(PxTransform(state.pose.p + PxVec3(100.0f, 50.0f, 0.0f), PxQuat(0.9f, PxVec3(1,0,0))));
            linear.push_back(PxVec3(8.0f, 4.0f, -3.0f));
            angular.push_back(PxVec3(-1.0f, 3.0f, 2.0f));
        }
        PxCudaHelpersExt::copyHToD(cuda, poses, hostPoses.data(), count);
        require(api.setRigidDynamicData(poses, indices, PxRigidDynamicGPUAPIWriteType::eGLOBAL_POSE, count), "device pose write failed");
        PxCudaHelpersExt::copyHToD(cuda, velocities, linear.data(), count);
        require(api.setRigidDynamicData(velocities, indices, PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY, count), "device linear write failed");
        PxCudaHelpersExt::copyHToD(cuda, velocities, angular.data(), count);
        require(api.setRigidDynamicData(velocities, indices, PxRigidDynamicGPUAPIWriteType::eANGULAR_VELOCITY, count), "device angular write failed");
    }
};
void compare(const std::vector<State>& expected, const std::vector<State>& actual)
{
    for (uint32_t i = 0; i < expected.size(); ++i)
    {
        const auto& a = expected[i]; const auto& b = actual[i];
        require(b.pose.isFinite() && b.linear.isFinite() && b.angular.isFinite(), "non-finite restored state");
        require((a.pose.p - b.pose.p).magnitude() < 1e-4f, "checkpoint restored incorrect position");
        require(std::abs(a.pose.q.dot(b.pose.q)) > 1.0f - 1e-6f, "checkpoint restored incorrect orientation");
        require((a.linear - b.linear).magnitude() < 1e-5f, "checkpoint restored incorrect linear velocity");
        require((a.angular - b.angular).magnitude() < 1e-5f, "checkpoint restored incorrect angular velocity");
    }
}
}
int main(int argc, char** argv)
{
    try
    {
        const uint32_t count = argc > 1 ? static_cast<uint32_t>(std::stoul(argv[1])) : 32u;
        require(count > 0 && count <= 20000, "body count must be 1..20000");
        SceneCapacity capacity;
        capacity.maxBodies = count + 64;
        capacity.maxShapes = count + 128;
        capacity.maxContactPairs = 4096;
        PhysXScene context(PhysicsMode::Gpu, true, capacity, nullptr, true);
        require(context.directGpuApiActive(), "Direct GPU API must be active");
        std::vector<PxRigidDynamic*> bodies;
        for (uint32_t i = 0; i < count; ++i)
        {
            auto* body = context.physics().createRigidDynamic(PxTransform(PxVec3(float(i % 128)*2, 100.0f, float(i/128)*2)));
            require(body != nullptr, "body create failed");
            const PxBoxGeometry geometry(PxVec3(0.25f));
            require(geometry.isValid(), "invalid fixture geometry");
            auto* shape = context.physics().createShape(geometry, context.material(), true);
            require(shape && body->attachShape(*shape), "shape create failed");
            shape->release();
            body->setMass(1.0f);
            body->setMassSpaceInertiaTensor(PxVec3(0.1f));
            body->setLinearVelocity(PxVec3(0.5f, 0.25f, -0.2f));
            body->setAngularVelocity(PxVec3(0.2f, 0.3f, -0.1f));
            context.scene().addActor(*body);
            bodies.push_back(body);
        }
        // Move away from CPU initialization values before any checkpoint.
        for (int i = 0; i < 12; ++i) { step(context.scene()); }
        Buffers buffers(*context.cudaContextManager(), count);
        buffers.bind(bodies);
        auto* motion = ExtStressPhysXDirectGpuMotionBuffer::create(context.scene());
        require(motion && motion->available(), "motion buffer unavailable");
        require(!motion->restore(), "restored without a capture");
        require(motion->capture(bodies.data(), 1), "initial capture failed");
        require(motion->capture(bodies.data(), count), "growing capture failed");
        require(motion->bodyCount() == count, "checkpoint body count wrong");
        const auto captured = buffers.read(context.scene(), count);
        step(context.scene());
        const auto expected = buffers.read(context.scene(), count);
        buffers.overwrite(context.scene(), captured);
        step(context.scene());
        const auto changed = buffers.read(context.scene(), count);
        require((changed[0].pose.p - captured[0].pose.p).magnitude() > 50.0f, "test failed to alter device state");
        require(motion->restore(), "restore failed");
        step(context.scene());
        compare(expected, buffers.read(context.scene(), count));
        // A checkpoint remains reusable; restoring must not overwrite it.
        require(motion->restore(), "second restore failed");
        step(context.scene());
        compare(expected, buffers.read(context.scene(), count));
        // Re-capture must replace it with this tick, not the initial state.
        require(motion->capture(bodies.data(), count), "re-capture failed");
        step(context.scene());
        const auto nextExpected = buffers.read(context.scene(), count);
        step(context.scene());
        require(motion->restore(), "re-captured restore failed");
        step(context.scene());
        compare(nextExpected, buffers.read(context.scene(), count));
        require(!motion->capture(nullptr, 0), "empty capture succeeded");
        require(motion->bodyCount() == 0 && !motion->restore(), "failed capture retained an old checkpoint");
        require(motion->capture(bodies.data(), count), "capture before removal failed");
        context.scene().removeActor(*bodies.back());
        require(!motion->restore(), "removed actor allowed a partial restore");
        context.scene().addActor(*bodies.back());
        step(context.scene());
        double captureMs = 0.0, restoreMs = 0.0;
        constexpr int repetitions = 100;
        for (int i = 0; i < repetitions; ++i)
        {
            const auto start = std::chrono::steady_clock::now();
            require(motion->capture(bodies.data(), count), "timed capture failed");
            const auto mid = std::chrono::steady_clock::now();
            require(motion->restore(), "timed restore failed");
            const auto end = std::chrono::steady_clock::now();
            captureMs += std::chrono::duration<double, std::milli>(mid-start).count();
            restoreMs += std::chrono::duration<double, std::milli>(end-mid).count();
        }
        motion->release();
        for (auto* body : bodies) { body->release(); }
        require(context.errors().warningCount() == 0 && context.healthy(), "PhysX reported errors or warnings");
        std::printf("direct GPU checkpoint passed: bodies=%u capture_ms=%.4f restore_ms=%.4f (mean of %d; motion remains on device)\n",
                    count, captureMs/repetitions, restoreMs/repetitions, repetitions);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "direct GPU checkpoint failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
