// Compare the cost of stock GPU sleeping, GPU without sleep, and Direct GPU.
// Independent boxes deliberately isolate activity/readback from fracture work.
#include "../physx_scene.h"
#include <extensions/PxCudaHelpersExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace physx;
using namespace blast_demo;
using physx::Ext::PxCudaHelpersExt;

namespace
{
constexpr float dt = 1.0f / 60.0f;
const char* modes[] = {"native_sleep", "gpu_forced_awake", "direct_gpu", "gpu_no_sleep_flag"};
constexpr unsigned modeCount = sizeof(modes) / sizeof(modes[0]);
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
void step(PhysXScene& context)
{
    context.scene().simulate(dt);
    require(context.scene().fetchResults(true), "fetchResults failed");
}
struct DeviceBuffers
{
    PxCudaContextManager& cuda;
    PxRigidDynamicGPUIndex* indices = nullptr;
    PxTransform* poses = nullptr;
    PxVec3* velocities = nullptr;
    explicit DeviceBuffers(PxCudaContextManager& manager,
                           const std::vector<PxRigidDynamic*>& bodies) : cuda(manager)
    {
        indices = PxCudaHelpersExt::allocDeviceBuffer<PxRigidDynamicGPUIndex>(cuda, bodies.size());
        poses = PxCudaHelpersExt::allocDeviceBuffer<PxTransform>(cuda, bodies.size());
        velocities = PxCudaHelpersExt::allocDeviceBuffer<PxVec3>(cuda, bodies.size());
        require(indices && poses && velocities, "device allocation failed");
        std::vector<PxRigidDynamicGPUIndex> host;
        for (auto* body : bodies) host.push_back(body->getGPUIndex());
        PxCudaHelpersExt::copyHToD(cuda, indices, host.data(), host.size());
    }
    ~DeviceBuffers()
    {
        PxCudaHelpersExt::freeDeviceBuffer(cuda, velocities);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, poses);
        PxCudaHelpersExt::freeDeviceBuffer(cuda, indices);
    }
};
// Validation reads are outside all measured intervals. CPU motion getters are
// used only in ordinary GPU scenes, never in Direct GPU scenes.
void validate(PhysXScene& context, const std::vector<PxRigidDynamic*>& bodies,
              DeviceBuffers* device, const std::string& phase, unsigned mode)
{
    std::vector<PxTransform> poses(bodies.size());
    std::vector<PxVec3> velocities(bodies.size());
    if (device)
    {
        auto& api = context.scene().getDirectGPUAPI();
        require(api.getRigidDynamicData(device->poses, device->indices,
            PxRigidDynamicGPUAPIReadType::eGLOBAL_POSE, bodies.size()), "pose read failed");
        require(api.getRigidDynamicData(device->velocities, device->indices,
            PxRigidDynamicGPUAPIReadType::eLINEAR_VELOCITY, bodies.size()), "velocity read failed");
        PxCudaHelpersExt::copyDToH(device->cuda, poses.data(), device->poses, bodies.size());
        PxCudaHelpersExt::copyDToH(device->cuda, velocities.data(), device->velocities, bodies.size());
    }
    else
    {
        for (size_t i = 0; i < bodies.size(); ++i)
        {
            poses[i] = bodies[i]->getGlobalPose();
            velocities[i] = bodies[i]->getLinearVelocity();
        }
    }
    for (size_t i = 0; i < bodies.size(); ++i)
    {
        require(poses[i].isFinite() && velocities[i].isFinite(), "non-finite motion");
        if (phase == "rest")
        {
            const bool settled = std::abs(poses[i].p.y - 0.5f) < 0.03f && velocities[i].magnitude() < 0.05f;
            if (!settled)
                std::fprintf(stderr, "unsettled body=%zu p=(%g,%g,%g) v=(%g,%g,%g)\n", i,
                    poses[i].p.x, poses[i].p.y, poses[i].p.z,
                    velocities[i].x, velocities[i].y, velocities[i].z);
            require(settled, "resting fixture did not settle on the floor");
        }
        else if (phase == "mass_wake")
            require(poses[i].p.y > 5.0f, "a body did not respond to the wake command");
        else
            require(poses[i].p.y > 700.0f && velocities[i].y < -50.0f,
                    "airborne fixture did not remain in free fall");
    }
    const auto stats = context.statistics();
    if (mode == 0 && phase == "rest")
    {
        for (auto* body : bodies) require(body->isSleeping(), "native body did not sleep");
        require(stats.nbActiveDynamicBodies == 0, "sleeping bodies remained active");
    }
    else if (mode == 3 && phase == "rest")
    {
        // Diagnostic arm. In the installed 5.10 engine, the flag alone leaves
        // all CPU island nodes inactive although isSleeping() returns false.
        // Do not treat this arm as an all-awake performance reference.
        for (auto* body : bodies) require(!body->isSleeping(), "flag-only body reports sleeping");
    }
    else require(stats.nbActiveDynamicBodies == bodies.size(), "expected all dynamics active");
    require(context.healthy(), "GPU error, fallback, or capacity overflow");
    std::fprintf(stderr, "validation %s %s: %u/%zu active, all body checks passed\n",
                 modes[mode], phase.c_str(), stats.nbActiveDynamicBodies, bodies.size());
}
void measure(PhysXScene& context, unsigned trial, unsigned mode, unsigned count,
             const char* phase, unsigned ticks, double commandMs = 0.0)
{
    std::vector<double> times;
    times.reserve(ticks);
    for (unsigned tick = 0; tick < ticks; ++tick)
    {
        const auto start = std::chrono::steady_clock::now();
        step(context);
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end-start).count());
    }
    // No logging, CPU body reads, statistics calls, or extra GPU copies inside
    // the step intervals or between measured steps.
    for (unsigned tick = 0; tick < ticks; ++tick)
        std::printf("%u,%s,%u,%s,%u,%.9f,%.9f\n", trial, modes[mode], count, phase, tick,
                    times[tick], tick == 0 ? commandMs : 0.0);
    const double first = times.front();
    const double mean = std::accumulate(times.begin(), times.end(), 0.0) / ticks;
    std::sort(times.begin(), times.end());
    std::fprintf(stderr, "trial=%u %s %s: mean=%.4f p99=%.4f first=%.4f command=%.4f ms\n",
                 trial, modes[mode], phase, mean, times[static_cast<size_t>(std::ceil(0.99*ticks))-1], first, commandMs);
}
void run(unsigned trial, unsigned mode, unsigned count, bool airborne)
{
    SceneCapacity capacity;
    capacity.maxBodies = count + 64;
    capacity.maxShapes = count + 128;
    capacity.maxContactPairs = std::max(4096u, count * 2);
    PhysXScene context(PhysicsMode::Gpu, true, capacity, nullptr, mode == 2, mode != 0);
    require(context.gpuActive() && context.directGpuApiActive() == (mode == 2), "wrong GPU mode");
    require(context.scene().getFlags().isSet(PxSceneFlag::eDISABLE_SLEEPING) == (mode != 0),
            "wrong sleeping configuration");
    std::vector<PxRigidDynamic*> bodies;
    const unsigned width = static_cast<unsigned>(std::ceil(std::sqrt(count)));
    for (unsigned i = 0; i < count; ++i)
    {
        auto* body = context.physics().createRigidDynamic(PxTransform(
            PxVec3(float(i % width)*2, airborne ? 1000.0f : 0.5f, float(i/width)*2)));
        require(body != nullptr, "body allocation failed");
        const PxBoxGeometry geometry(PxVec3(0.5f));
        require(geometry.isValid(), "invalid fixture geometry");
        auto* shape = context.physics().createShape(geometry, context.material(), true);
        require(shape && body->attachShape(*shape), "shape allocation failed");
        shape->release();
        body->setMass(1.0f);
        body->setMassSpaceInertiaTensor(PxVec3(1.0f/6.0f));
        // No aerodynamic damping: the airborne arm stays awake and exercises
        // the same dynamics in all modes.
        body->setLinearDamping(0.0f);
        body->setAngularDamping(0.0f);
        // eDISABLE_SLEEPING alone does not keep the installed engine's
        // ordinary GPU island nodes active. A long counter makes the control
        // truly awake throughout this finite experiment (at most 12 seconds).
        context.scene().addActor(*body);
        if (mode == 1)
        {
            // Insertion into an eDISABLE_SLEEPING scene first resets the
            // counter to the scene default, so override it after insertion.
            body->setWakeCounter(3600.0f);
            require(body->getWakeCounter() == 3600.0f, "awake control counter was reset");
        }
        bodies.push_back(body);
    }
    for (unsigned tick = 0; tick < (airborne ? 120u : 360u); ++tick) step(context);
    std::unique_ptr<DeviceBuffers> device;
    if (mode == 2) device = std::make_unique<DeviceBuffers>(*context.cudaContextManager(), bodies);
    const char* phase = airborne ? "airborne" : "rest";
    measure(context, trial, mode, count, phase, 300);
    validate(context, bodies, device.get(), phase, mode);
    if (!airborne)
    {
        // Identical commanded velocities. Keep submission separately visible
        // so a bulk device write is not compared with hidden CPU wake work.
        const std::vector<PxVec3> velocities(device ? count : 0, PxVec3(0, 10, 0));
        const auto commandStart = std::chrono::steady_clock::now();
        if (device)
        {
            PxCudaHelpersExt::copyHToD(device->cuda, device->velocities, velocities.data(), count);
            require(context.scene().getDirectGPUAPI().setRigidDynamicData(device->velocities,
                device->indices, PxRigidDynamicGPUAPIWriteType::eLINEAR_VELOCITY, count), "wake write failed");
        }
        else for (auto* body : bodies) body->setLinearVelocity(PxVec3(0, 10, 0), true);
        const double commandMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-commandStart).count();
        measure(context, trial, mode, count, "mass_wake", 60, commandMs);
        validate(context, bodies, device.get(), "mass_wake", mode);
    }
    device.reset();
    for (auto* body : bodies) body->release();
    require(context.healthy(), "GPU failure during cleanup");
}
}
int main(int argc, char** argv)
{
    try
    {
        const unsigned count = argc > 1 ? std::stoul(argv[1]) : 4096;
        const unsigned trials = argc > 2 ? std::stoul(argv[2]) : 3;
        require(argc <= 3 && count > 0 && count <= 20000 && trials > 0 && trials <= 20,
                "usage: gpu_activity_bench [bodies=4096 (1..20000)] [trials=3 (1..20)]");
        std::puts("trial,mode,bodies,phase,tick,step_ms,command_ms");
        for (unsigned trial = 0; trial < trials; ++trial)
            for (unsigned offset = 0; offset < modeCount; ++offset)
                for (bool airborne : {true, false})
                    run(trial, (trial + offset) % modeCount, count, airborne);
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "gpu_activity_bench FAILED: %s\n", error.what());
        return 1;
    }
}
