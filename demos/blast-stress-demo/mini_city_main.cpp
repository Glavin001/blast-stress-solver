// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "physx_scene.h"
#include "scene_pack.h"
#include "state_writer.h"

#include <NvBlastExtStressPhysX.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace Nv::Blast;
using namespace physx;

namespace blast_demo
{
namespace
{

struct Options
{
    PhysicsMode physics{PhysicsMode::Gpu};
    bool requireGpu{false};
    bool gpuStress{false};
    std::uint32_t gpuStressMinimumBonds{4096};
    bool requirePartialDestruction{false};
    bool requireRealtime{false};
    std::uint32_t requireMinimumAuthoredChunks{0};
    bool requireVariedBuildingHeights{false};
    bool selfTest{false};
    std::uint32_t grid{3};
    std::uint32_t stressWorkers{0};
    bool serializeGpuStress{false};
    std::uint32_t maximumBodiesPerStructure{48};
    std::uint32_t maximumFracturesPerActorPerTick{0};
    std::uint32_t tallBuildingStride{0};
    bool variedBuildingHeights{true};
    float durationSeconds{12.0f};
    float settleSeconds{1.5f};
    float projectileMassScale{1.5f};
    float projectileSpeedScale{1.0f};
    float projectileRadiusScale{1.0f};
    float projectileTtlScale{0.4f};
    float contactForceScale{1.5f};
    float minimumStressContactImpulse{0.0f};
    float stressLimitScale{0.8f};
    float excessForceScale{0.017f};
    std::uint32_t snapshotFps{30};
    std::uint32_t paneWidth{960};
    std::uint32_t paneHeight{540};
    std::string scenePath;
    std::string statePath{"blast-physx-mini-city.towerstate"};
    std::string metadataPath{"blast-physx-mini-city.json"};
    std::string telemetryPath;
    std::string frameTelemetryPath;
};

struct BuildingVariant
{
    ScenePack pack;
    std::vector<ExtStressPhysXNodeDesc> nodes;
    std::vector<ExtStressPhysXBondDesc> bonds;
    std::uint32_t floors{0};
    float height{0.0f};
};

struct BuildingInstance
{
    std::size_t variantIndex{0};
    PxVec3 offset{0.0f};
    std::uint32_t visualBase{0};
};

struct DestructibleDeleter
{
    void operator()(ExtStressPhysXDestructible* value) const
    {
        if (value) value->release();
    }
};

using DestructiblePtr = std::unique_ptr<ExtStressPhysXDestructible, DestructibleDeleter>;

std::uint32_t resolveStressWorkerCount(std::uint32_t requested)
{
    if (requested > 0)
    {
        return requested;
    }
    return std::min<std::uint32_t>(
        8,
        std::max<std::uint32_t>(1, std::thread::hardware_concurrency()));
}

class StressExecutor
{
public:
    StressExecutor(std::uint32_t workerCount, bool serializeGpuStress)
        : m_workerCount(workerCount)
        , m_serializeGpuStress(serializeGpuStress)
    {
        if (workerCount <= 1)
        {
            return;
        }
        m_workers.reserve(workerCount);
        for (std::uint32_t worker = 0; worker < workerCount; ++worker)
        {
            m_workers.emplace_back([this]() { workerLoop(); });
        }
    }

    ~StressExecutor()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stopping = true;
        }
        m_workAvailable.notify_all();
        for (std::thread& worker : m_workers)
        {
            worker.join();
        }
    }

    std::uint32_t workerCount() const
    {
        return m_workerCount;
    }

    bool solve(std::vector<DestructiblePtr>& destructibles)
    {
        if (m_workers.empty())
        {
            for (const DestructiblePtr& destructible : destructibles)
            {
                if (!destructible->solveTick())
                {
                    return false;
                }
            }
            return true;
        }

        std::atomic<bool> succeeded{true};
        run(destructibles.size(), [&](std::size_t index) {
            bool solved = false;
            if (m_serializeGpuStress && destructibles[index]->usesGpuStressSolver())
            {
                std::lock_guard<std::mutex> lock(m_gpuMutex);
                solved = destructibles[index]->solveTick();
            }
            else
            {
                solved = destructibles[index]->solveTick();
            }
            if (!solved)
            {
                succeeded.store(false, std::memory_order_relaxed);
            }
        });
        return succeeded.load(std::memory_order_relaxed);
    }

private:
    void run(std::size_t count, std::function<void(std::size_t)> task)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_task = std::move(task);
            m_taskCount = count;
            m_nextTask.store(0, std::memory_order_relaxed);
            m_finishedWorkers = 0;
            m_exception = nullptr;
            ++m_generation;
        }
        m_workAvailable.notify_all();

        std::unique_lock<std::mutex> lock(m_mutex);
        m_workFinished.wait(lock, [this]() {
            return m_finishedWorkers == m_workers.size();
        });
        if (m_exception)
        {
            std::rethrow_exception(m_exception);
        }
        m_task = {};
    }

    void workerLoop()
    {
        std::uint64_t observedGeneration = 0;
        while (true)
        {
            std::function<void(std::size_t)> task;
            std::size_t taskCount = 0;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_workAvailable.wait(lock, [&]() {
                    return m_stopping || m_generation != observedGeneration;
                });
                if (m_stopping)
                {
                    return;
                }
                observedGeneration = m_generation;
                task = m_task;
                taskCount = m_taskCount;
            }

            try
            {
                while (true)
                {
                    const std::size_t index =
                        m_nextTask.fetch_add(1, std::memory_order_relaxed);
                    if (index >= taskCount)
                    {
                        break;
                    }
                    task(index);
                }
            }
            catch (...)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_exception)
                {
                    m_exception = std::current_exception();
                }
            }

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                ++m_finishedWorkers;
                if (m_finishedWorkers == m_workers.size())
                {
                    m_workFinished.notify_one();
                }
            }
        }
    }

    std::uint32_t m_workerCount{1};
    bool m_serializeGpuStress{false};
    std::vector<std::thread> m_workers;
    std::mutex m_gpuMutex;
    std::mutex m_mutex;
    std::condition_variable m_workAvailable;
    std::condition_variable m_workFinished;
    std::function<void(std::size_t)> m_task;
    std::size_t m_taskCount{0};
    std::atomic<std::size_t> m_nextTask{0};
    std::size_t m_finishedWorkers{0};
    std::uint64_t m_generation{0};
    std::exception_ptr m_exception;
    bool m_stopping{false};
};

void tickDestructibles(
    std::vector<DestructiblePtr>& destructibles,
    StressExecutor& executor,
    float dt,
    const PxVec3& gravity)
{
    for (const DestructiblePtr& destructible : destructibles)
    {
        if (!destructible->beginTick(dt, gravity))
        {
            throw std::runtime_error("PhysX destruction adapter load phase failed");
        }
    }
    if (!executor.solve(destructibles))
    {
        throw std::runtime_error("PhysX destruction adapter stress phase failed");
    }
    for (const DestructiblePtr& destructible : destructibles)
    {
        if (!destructible->endTick())
        {
            throw std::runtime_error("PhysX destruction adapter topology phase failed");
        }
    }
}

struct Projectile
{
    PxRigidDynamic* body{nullptr};
    PxShape* shape{nullptr};
    std::uint32_t visualId{0};
    float launchAt{0.0f};
    float retireAt{0.0f};
    PxVec3 launchPosition{0.0f};
    PxVec3 launchVelocity{0.0f};
    bool launched{false};
    bool retired{false};
};

struct AggregateTelemetry
{
    std::uint64_t contactsQueued{0};
    std::uint64_t contactsProcessed{0};
    std::uint64_t contactsDropped{0};
    std::uint64_t splits{0};
    std::uint64_t bodiesCreated{0};
    std::uint64_t bodiesReused{0};
    std::uint64_t bodiesRecycled{0};
    std::uint64_t shapesMigrated{0};
    std::uint64_t sleepingActorsSkipped{0};
    std::uint32_t peakBodies{0};
    std::uint32_t peakAwakeBodies{0};
    std::uint32_t overstressedBonds{0};
    std::uint32_t solverIslands{0};
    std::uint32_t solverIslandsSkipped{0};
    float maxPositionDrift{0.0f};
    float maxVelocityDrift{0.0f};
    double contactProcessingMilliseconds{0.0};
    double gravityMilliseconds{0.0};
    double stressSolveMilliseconds{0.0};
    double gpuStressSolveMilliseconds{0.0};
    std::uint64_t gpuStressHostToDeviceBytes{0};
    std::uint64_t gpuStressDeviceToHostBytes{0};
    double fractureTopologyMilliseconds{0.0};
    double mappingValidationMilliseconds{0.0};
};

struct FrameMetrics
{
    std::uint32_t step{0};
    double simulationSeconds{0.0};
    double physicsStepMilliseconds{0.0};
    double contactCallbackMilliseconds{0.0};
    double contactProcessingMilliseconds{0.0};
    double gravityMilliseconds{0.0};
    double stressSolveMilliseconds{0.0};
    double gpuStressSolveMilliseconds{0.0};
    std::uint64_t gpuStressHostToDeviceBytes{0};
    std::uint64_t gpuStressDeviceToHostBytes{0};
    double fractureTopologyMilliseconds{0.0};
    double adapterTickMilliseconds{0.0};
    double mappingValidationMilliseconds{0.0};
    double stateExportMilliseconds{0.0};
    double frameHostMilliseconds{0.0};
    AggregateTelemetry telemetry;
    std::uint64_t frameContacts{0};
    std::uint64_t frameSplits{0};
    std::uint64_t frameShapesMigrated{0};
    std::uint64_t projectileImpactContacts{0};
    std::uint64_t frameProjectileImpactContacts{0};
    double projectileImpactImpulse{0.0};
    double frameProjectileImpactImpulse{0.0};
};

struct RuntimeTimings
{
    std::vector<double> frameHostMilliseconds;
    std::vector<double> physicsStepMilliseconds;
    std::vector<double> adapterTickMilliseconds;
    std::vector<double> gpuStressSolveMilliseconds;
    std::vector<double> stateExportMilliseconds;
    std::uint32_t budgetMissFrames{0};
    std::uint32_t destructionBudgetMissFrames{0};
    std::uint32_t destructionFrameSamples{0};
    double maximumDestructionFrameMilliseconds{0.0};
};

struct DestructionDistribution
{
    std::uint32_t intactStructures{0};
    std::uint32_t partiallyFracturedStructures{0};
    std::uint32_t heavilyFracturedStructures{0};
    std::uint32_t shatteredStructures{0};
    std::uint32_t minimumBodiesPerStructure{0};
    std::uint32_t maximumBodiesPerStructure{0};
    double meanBodiesPerStructure{0.0};
};

struct DestructionMotion
{
    std::uint32_t structuresWithMovedChunks{0};
    std::uint32_t structuresWithFallenChunks{0};
    std::uint32_t movedChunks{0};
    std::uint32_t fallenChunks{0};
    std::uint32_t farTravelingChunks{0};
    std::uint32_t dynamicChunks{0};
    std::uint32_t movingChunks{0};
    std::uint32_t supportedRemainderChunks{0};
    float maximumDisplacement{0.0f};
    float maximumDownwardDisplacement{0.0f};
};

struct ContractBaseline
{
    ExtStressPhysXId bodyId{0};
    std::vector<ExtStressPhysXId> shapeIds;
    std::vector<const PxShape*> shapes;
};

void requireContract(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string("PhysX contract self-test failed: ") + message);
    }
}

ContractBaseline captureContractBaseline(
    ExtStressPhysXDestructible& destructible,
    std::uint32_t nodeCount)
{
    ContractBaseline result;
    std::vector<ExtStressPhysXBodySnapshot> bodies(nodeCount);
    const std::uint32_t bodyCount =
        destructible.getBodySnapshots(bodies.data(), static_cast<std::uint32_t>(bodies.size()));
    requireContract(bodyCount == 1, "initial structure must own exactly one body");
    requireContract(bodies[0].kinematic, "initial support structure must be kinematic");
    result.bodyId = bodies[0].bodyId;

    std::vector<ExtStressPhysXShapeSnapshot> shapes(nodeCount);
    const std::uint32_t shapeCount =
        destructible.getShapeSnapshots(shapes.data(), static_cast<std::uint32_t>(shapes.size()));
    requireContract(shapeCount == nodeCount, "initial structure must expose one shape per node");
    result.shapeIds.resize(nodeCount);
    result.shapes.resize(nodeCount);
    for (const ExtStressPhysXShapeSnapshot& shape : shapes)
    {
        requireContract(shape.nodeIndex < nodeCount, "shape has an invalid node index");
        requireContract(shape.bodyId == result.bodyId, "initial shapes must share the initial body");
        result.shapeIds[shape.nodeIndex] = shape.shapeId;
        result.shapes[shape.nodeIndex] = shape.shape;
    }
    return result;
}

void validateContractResult(
    const ScenePack& pack,
    ExtStressPhysXDestructible& destructible,
    const ContractBaseline& baseline)
{
    const std::uint32_t nodeCount = static_cast<std::uint32_t>(pack.nodes.size());
    std::vector<ExtStressPhysXShapeSnapshot> shapes(nodeCount);
    const std::uint32_t shapeCount =
        destructible.getShapeSnapshots(shapes.data(), static_cast<std::uint32_t>(shapes.size()));
    requireContract(shapeCount == nodeCount, "fracture lost a node shape");

    struct ExpectedBody
    {
        float mass{0.0f};
        bool hasSupport{false};
        std::uint32_t nodeCount{0};
    };
    std::unordered_map<ExtStressPhysXId, ExpectedBody> expectedBodies;
    for (const ExtStressPhysXShapeSnapshot& shape : shapes)
    {
        requireContract(shape.nodeIndex < nodeCount, "fractured shape has an invalid node index");
        requireContract(
            shape.shapeId == baseline.shapeIds[shape.nodeIndex],
            "shape ID changed while migrating between bodies");
        requireContract(
            shape.shape == baseline.shapes[shape.nodeIndex],
            "PxShape identity changed while migrating between bodies");
        ExpectedBody& expected = expectedBodies[shape.bodyId];
        expected.mass += pack.nodes[shape.nodeIndex].mass;
        expected.hasSupport = expected.hasSupport || pack.nodes[shape.nodeIndex].mass == 0.0f;
        ++expected.nodeCount;
    }

    std::vector<ExtStressPhysXBodySnapshot> bodies(nodeCount);
    const std::uint32_t bodyCount =
        destructible.getBodySnapshots(bodies.data(), static_cast<std::uint32_t>(bodies.size()));
    requireContract(bodyCount > 1, "forced fracture did not create child bodies");
    requireContract(bodyCount == expectedBodies.size(), "shape-to-body cohorts are inconsistent");

    bool reusedInitialBody = false;
    bool foundKinematic = false;
    bool foundDynamic = false;
    std::uint32_t assignedNodes = 0;
    ExtStressPhysXId previousBodyId = 0;
    for (std::uint32_t i = 0; i < bodyCount; ++i)
    {
        const ExtStressPhysXBodySnapshot& body = bodies[i];
        requireContract(i == 0 || body.bodyId > previousBodyId, "body snapshots are not stable-ID sorted");
        previousBodyId = body.bodyId;
        reusedInitialBody = reusedInitialBody || body.bodyId == baseline.bodyId;

        const auto expected = expectedBodies.find(body.bodyId);
        requireContract(expected != expectedBodies.end(), "body has no node cohort");
        requireContract(body.nodeCount == expected->second.nodeCount, "body node count is inconsistent");
        requireContract(body.kinematic == expected->second.hasSupport, "support state was not reconciled");
        foundKinematic = foundKinematic || body.kinematic;
        foundDynamic = foundDynamic || !body.kinematic;
        assignedNodes += body.nodeCount;

        const float expectedMass = expected->second.mass > 0.0f ? expected->second.mass : 1.0f;
        const float tolerance = std::max(1.0e-4f, expectedMass * 1.0e-4f);
        requireContract(
            std::abs(body.body->getMass() - expectedMass) <= tolerance,
            "aggregate PhysX body mass does not match node mass");
    }
    requireContract(assignedNodes == nodeCount, "fracture did not assign every node exactly once");
    requireContract(reusedInitialBody, "split did not deterministically reuse the parent body");
    requireContract(foundKinematic && foundDynamic, "split did not exercise support-to-dynamic separation");

    const ExtStressPhysXTelemetry telemetry = destructible.getTelemetry();
    requireContract(telemetry.bodiesReused > 0, "body reuse was not recorded");
    requireContract(telemetry.shapesMigrated > 0, "shape migration was not recorded");
    requireContract(telemetry.mappingValidationFailures == 0, "mapping validation failed");
    requireContract(telemetry.maxSplitWorldPositionDrift <= 1.0e-3f, "world-position continuity exceeded tolerance");
    requireContract(telemetry.maxSplitPointVelocityDrift <= 1.0e-3f, "point-velocity continuity exceeded tolerance");
}

void usage(const char* executable)
{
    std::printf(
        "Usage: %s [options]\n"
        "  --physics cpu|gpu       PhysX pipeline (default gpu)\n"
        "  --require-gpu           Fail instead of allowing CPU fallback\n"
        "  --gpu-stress            Use the resident CUDA CGNR stress backend\n"
        "  --gpu-stress-min-bonds N  CPU fallback below N solver bonds (default 4096)\n"
        "  --require-partial-destruction  Require visible movement, falling debris, and a stable remainder\n"
        "  --require-dynamic-destruction  Alias for --require-partial-destruction\n"
        "  --require-realtime      Fail if any post-settle frame exceeds 16.67 ms\n"
        "  --require-min-authored-chunks N  Fail unless the latent scene has at least N chunks\n"
        "  --require-varied-building-heights  Fail unless all three skyline heights are present\n"
        "  --scene PATH            ScenePack v1 JSON\n"
        "  --grid N                N by N city (default 3)\n"
        "  --stress-workers N      Parallel per-building stress solves (default auto, max 64)\n"
        "  --serialize-gpu-stress  Serialize CUDA stress solves; CPU solves remain parallel\n"
        "  --max-bodies-per-structure N  Fracture body budget per building (default 48)\n"
        "  --max-fractures-per-actor-per-tick N  Spread bond breaks across steps (0 unlimited)\n"
        "  --tall-building-stride N  Place one full-height tower every N buildings\n"
        "  --uniform-building-heights  Disable the default 1/2/3-floor skyline\n"
        "  --duration SECONDS      Destruction duration (default 12)\n"
        "  --settle SECONDS        Initial settle time (default 1.5)\n"
        "  --projectile-mass-scale X    Multiply ScenePack projectile mass (default 1.5)\n"
        "  --projectile-speed-scale X   Multiply ScenePack projectile speed\n"
        "  --projectile-radius-scale X  Multiply ScenePack projectile radius\n"
        "  --projectile-ttl-scale X     Multiply ScenePack projectile lifetime (default 0.4)\n"
        "  --contact-force-scale X      Multiply stress contact impulse transfer\n"
        "  --min-stress-contact-impulse X  Ignore weaker non-projectile stress feedback\n"
        "  --stress-limit-scale X       Multiply all elastic/fatal stress limits\n"
        "  --excess-force-scale X       Scale released bond load applied to new debris\n"
        "  --impact-transfer-scale X    Alias for --excess-force-scale\n"
        "  --snapshot-fps FPS      60 Hz divisor (default 30)\n"
        "  --output-state PATH     TWSTATE1 output (`--state` is an alias)\n"
        "  --metadata PATH         JSON telemetry output\n"
        "  --telemetry PATH        Optional second JSON telemetry sidecar\n"
        "  --frame-telemetry PATH  Per-physics-step CSV diagnostics\n"
        "  --pane-width PIXELS     Per-camera width (default 960)\n"
        "  --pane-height PIXELS    Per-camera height (default 540)\n"
        "  --self-test             Short CPU contract/fracture test\n",
        executable);
}

std::string defaultScenePath(const char* executable)
{
    std::string path(executable ? executable : "");
    const std::size_t slash = path.find_last_of('/');
    if (slash != std::string::npos)
    {
        path.resize(slash);
    }
    else
    {
        path = ".";
    }
    return path + "/../../../blast/blast-stress-demo-rs/assets/scenes/fractured-tower.json";
}

std::uint32_t parseU32(const char* value, const char* option)
{
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (!value[0] || !end || *end != '\0' || parsed > UINT32_MAX)
    {
        throw std::runtime_error(std::string("invalid value for ") + option);
    }
    return static_cast<std::uint32_t>(parsed);
}

float parseFloat(const char* value, const char* option)
{
    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (!value[0] || !end || *end != '\0' || !std::isfinite(parsed))
    {
        throw std::runtime_error(std::string("invalid value for ") + option);
    }
    return parsed;
}

Options parseOptions(int argc, char** argv)
{
    Options options;
    options.scenePath = defaultScenePath(argv[0]);
    for (int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];
        auto argument = [&]() -> const char* {
            if (++i >= argc)
            {
                throw std::runtime_error("missing argument for " + option);
            }
            return argv[i];
        };
        if (option == "--help" || option == "-h")
        {
            usage(argv[0]);
            std::exit(0);
        }
        else if (option == "--physics")
        {
            const std::string mode = argument();
            if (mode == "cpu") options.physics = PhysicsMode::Cpu;
            else if (mode == "gpu") options.physics = PhysicsMode::Gpu;
            else throw std::runtime_error("--physics must be cpu or gpu");
        }
        else if (option == "--require-gpu") options.requireGpu = true;
        else if (option == "--gpu-stress") options.gpuStress = true;
        else if (option == "--gpu-stress-min-bonds")
            options.gpuStressMinimumBonds = parseU32(argument(), "--gpu-stress-min-bonds");
        else if (option == "--require-partial-destruction"
            || option == "--require-dynamic-destruction")
            options.requirePartialDestruction = true;
        else if (option == "--require-realtime") options.requireRealtime = true;
        else if (option == "--require-min-authored-chunks")
            options.requireMinimumAuthoredChunks =
                parseU32(argument(), "--require-min-authored-chunks");
        else if (option == "--require-varied-building-heights")
            options.requireVariedBuildingHeights = true;
        else if (option == "--self-test") options.selfTest = true;
        else if (option == "--scene") options.scenePath = argument();
        else if (option == "--grid") options.grid = parseU32(argument(), "--grid");
        else if (option == "--stress-workers")
            options.stressWorkers = parseU32(argument(), "--stress-workers");
        else if (option == "--serialize-gpu-stress")
            options.serializeGpuStress = true;
        else if (option == "--max-bodies-per-structure")
            options.maximumBodiesPerStructure =
                parseU32(argument(), "--max-bodies-per-structure");
        else if (option == "--max-fractures-per-actor-per-tick")
            options.maximumFracturesPerActorPerTick =
                parseU32(argument(), "--max-fractures-per-actor-per-tick");
        else if (option == "--tall-building-stride")
            options.tallBuildingStride =
                parseU32(argument(), "--tall-building-stride");
        else if (option == "--uniform-building-heights")
            options.variedBuildingHeights = false;
        else if (option == "--duration") options.durationSeconds = parseFloat(argument(), "--duration");
        else if (option == "--settle") options.settleSeconds = parseFloat(argument(), "--settle");
        else if (option == "--projectile-mass-scale")
            options.projectileMassScale = parseFloat(argument(), "--projectile-mass-scale");
        else if (option == "--projectile-speed-scale")
            options.projectileSpeedScale = parseFloat(argument(), "--projectile-speed-scale");
        else if (option == "--projectile-radius-scale")
            options.projectileRadiusScale = parseFloat(argument(), "--projectile-radius-scale");
        else if (option == "--projectile-ttl-scale")
            options.projectileTtlScale = parseFloat(argument(), "--projectile-ttl-scale");
        else if (option == "--contact-force-scale")
            options.contactForceScale = parseFloat(argument(), "--contact-force-scale");
        else if (option == "--min-stress-contact-impulse")
            options.minimumStressContactImpulse =
                parseFloat(argument(), "--min-stress-contact-impulse");
        else if (option == "--stress-limit-scale")
            options.stressLimitScale = parseFloat(argument(), "--stress-limit-scale");
        else if (option == "--excess-force-scale" || option == "--impact-transfer-scale")
            options.excessForceScale = parseFloat(argument(), option.c_str());
        else if (option == "--snapshot-fps") options.snapshotFps = parseU32(argument(), "--snapshot-fps");
        else if (option == "--output-state" || option == "--state") options.statePath = argument();
        else if (option == "--metadata") options.metadataPath = argument();
        else if (option == "--telemetry") options.telemetryPath = argument();
        else if (option == "--frame-telemetry") options.frameTelemetryPath = argument();
        else if (option == "--pane-width") options.paneWidth = parseU32(argument(), "--pane-width");
        else if (option == "--pane-height") options.paneHeight = parseU32(argument(), "--pane-height");
        else throw std::runtime_error("unknown option: " + option);
    }
    if (options.grid == 0 || options.grid > 20)
    {
        throw std::runtime_error("--grid must be between 1 and 20");
    }
    if (options.stressWorkers > 64)
    {
        throw std::runtime_error("--stress-workers must be between 0 and 64");
    }
    if (options.maximumBodiesPerStructure == 0
        || options.maximumBodiesPerStructure > 1024)
    {
        throw std::runtime_error("--max-bodies-per-structure must be between 1 and 1024");
    }
    if (options.tallBuildingStride > 1024)
    {
        throw std::runtime_error("--tall-building-stride must be between 0 and 1024");
    }
    if (options.durationSeconds <= 0.0f || options.settleSeconds < 0.0f)
    {
        throw std::runtime_error("duration must be positive and settle must be non-negative");
    }
    if (options.projectileMassScale <= 0.0f
        || options.projectileSpeedScale <= 0.0f
        || options.projectileRadiusScale <= 0.0f
        || options.projectileTtlScale <= 0.0f
        || options.contactForceScale <= 0.0f
        || options.minimumStressContactImpulse < 0.0f
        || options.stressLimitScale <= 0.0f
        || options.excessForceScale < 0.0f)
    {
        throw std::runtime_error("destruction tuning scales must be positive");
    }
    if (options.snapshotFps == 0 || options.snapshotFps > 60 || 60 % options.snapshotFps != 0)
    {
        throw std::runtime_error("--snapshot-fps must be a positive divisor of 60");
    }
    if (options.selfTest)
    {
        options.physics = PhysicsMode::Cpu;
        options.requireGpu = false;
        options.requirePartialDestruction = false;
        options.requireVariedBuildingHeights = false;
        options.grid = 1;
        options.stressWorkers = 1;
        options.variedBuildingHeights = false;
        options.durationSeconds = 0.5f;
        options.settleSeconds = 0.0f;
        options.statePath.clear();
        options.metadataPath.clear();
        options.telemetryPath.clear();
        options.frameTelemetryPath.clear();
    }
    return options;
}

class ContactRouter final : public PxSimulationEventCallback
{
public:
    ContactRouter(float forceScale, float minimumStressImpulse)
        : m_forceScale(forceScale)
        , m_minimumStressImpulse(minimumStressImpulse)
    {
    }

    void registerDestructible(ExtStressPhysXDestructible& destructible, std::uint32_t nodeCount)
    {
        std::vector<ExtStressPhysXShapeSnapshot> snapshots(nodeCount);
        const std::uint32_t count = destructible.getShapeSnapshots(snapshots.data(), nodeCount);
        if (count != nodeCount)
        {
            throw std::runtime_error("adapter did not expose every node shape");
        }
        for (const ExtStressPhysXShapeSnapshot& snapshot : snapshots)
        {
            snapshot.shape->userData = &destructible;
        }
    }

    void registerProjectile(PxShape& shape)
    {
        shape.userData = this;
    }

    void onContact(
        const PxContactPairHeader&,
        const PxContactPair* pairs,
        PxU32 pairCount) override
    {
        const auto start = std::chrono::steady_clock::now();
        std::array<PxContactPairPoint, 64> points;
        for (PxU32 pairIndex = 0; pairIndex < pairCount; ++pairIndex)
        {
            const PxContactPair& pair = pairs[pairIndex];
            if (pair.flags & (PxContactPairFlag::eREMOVED_SHAPE_0 | PxContactPairFlag::eREMOVED_SHAPE_1))
            {
                continue;
            }
            const PxU32 count = pair.extractContacts(points.data(), static_cast<PxU32>(points.size()));
            ExtStressPhysXDestructible* owner0 = owner(pair.shapes[0]);
            ExtStressPhysXDestructible* owner1 = owner(pair.shapes[1]);
            const bool projectileImpact =
                (pair.shapes[0]->userData == this && owner1 != nullptr)
                || (pair.shapes[1]->userData == this
                    && owner0 != nullptr);
            for (PxU32 i = 0; i < count; ++i)
            {
                const float impulseMagnitude = points[i].impulse.magnitude();
                if (projectileImpact)
                {
                    ++m_projectileImpactContacts;
                    m_projectileImpactImpulse += impulseMagnitude;
                }
                if (!projectileImpact && impulseMagnitude < m_minimumStressImpulse)
                {
                    continue;
                }
                queue(owner0, pair.shapes[0], points[i].position, points[i].impulse);
                queue(owner1, pair.shapes[1], points[i].position, -points[i].impulse);
            }
        }
        m_callbackMilliseconds +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
    }

    double callbackMilliseconds() const { return m_callbackMilliseconds; }
    std::uint64_t projectileImpactContacts() const { return m_projectileImpactContacts; }
    double projectileImpactImpulse() const { return m_projectileImpactImpulse; }

    void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
    void onWake(PxActor**, PxU32) override {}
    void onSleep(PxActor**, PxU32) override {}
    void onTrigger(PxTriggerPair*, PxU32) override {}
    void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) override {}

private:
    ExtStressPhysXDestructible* owner(const PxShape* shape) const
    {
        return shape && shape->userData && shape->userData != this
            ? static_cast<ExtStressPhysXDestructible*>(shape->userData)
            : nullptr;
    }

    void queue(
        ExtStressPhysXDestructible* destructible,
        const PxShape* shape,
        const PxVec3& position,
        const PxVec3& impulse)
    {
        if (destructible)
        {
            destructible->queueContact(*shape, position, impulse * m_forceScale);
        }
    }

    float m_forceScale{1.0f};
    float m_minimumStressImpulse{0.0f};
    double m_callbackMilliseconds{0.0};
    std::uint64_t m_projectileImpactContacts{0};
    double m_projectileImpactImpulse{0.0};
};

void adapterError(
    ExtStressPhysXError error,
    std::uint32_t node,
    const char* message,
    void*)
{
    std::fprintf(
        stderr,
        "[StressPhysX:%u node=%u] %s\n",
        static_cast<unsigned>(error),
        node,
        message ? message : "");
}

std::vector<ExtStressPhysXNodeDesc> makeNodeDescs(const ScenePack& pack)
{
    std::vector<ExtStressPhysXNodeDesc> result(pack.nodes.size());
    for (std::size_t i = 0; i < pack.nodes.size(); ++i)
    {
        const SceneNode& source = pack.nodes[i];
        ExtStressPhysXNodeDesc& target = result[i];
        target.centroid = source.centroid;
        target.mass = source.mass;
        target.volume = source.volume;
        target.geometry.localPose = PxTransform(source.centroid);
        target.geometry.halfExtents = source.collider.halfExtents;
        if (source.collider.kind == SceneColliderKind::ConvexHull)
        {
            target.geometry.type = ExtStressPhysXGeometryType::Convex;
            target.geometry.convexPoints = source.collider.points.data();
            target.geometry.convexPointCount =
                static_cast<std::uint32_t>(source.collider.points.size());
        }
    }
    return result;
}

std::vector<ExtStressPhysXBondDesc> makeBondDescs(const ScenePack& pack)
{
    std::vector<ExtStressPhysXBondDesc> result(pack.bonds.size());
    for (std::size_t i = 0; i < pack.bonds.size(); ++i)
    {
        const SceneBond& source = pack.bonds[i];
        ExtStressPhysXBondDesc& target = result[i];
        target.node0 = source.node0;
        target.node1 = source.node1;
        target.centroid = source.centroid;
        target.normal = source.normal;
        target.area = source.area;
    }
    return result;
}

ScenePack truncateToFloors(
    const ScenePack& source,
    std::uint32_t floors,
    std::uint32_t maximumFloors)
{
    if (floors == 0 || floors > maximumFloors || source.nodes.empty())
    {
        throw std::runtime_error("invalid building floor count");
    }
    ScenePack result = source;
    result.title = source.title + " " + std::to_string(floors) + "-floor";
    result.nodes.clear();
    result.bonds.clear();

    float minimumY = source.nodes.front().centroid.y;
    float maximumY = minimumY;
    for (const SceneNode& node : source.nodes)
    {
        minimumY = std::min(minimumY, node.centroid.y);
        maximumY = std::max(maximumY, node.centroid.y);
    }
    const float cutoff =
        floors == maximumFloors
        ? maximumY + 1.0f
        : minimumY + (maximumY - minimumY) * static_cast<float>(floors)
            / static_cast<float>(maximumFloors);
    std::vector<std::uint32_t> remap(source.nodes.size(), UINT32_MAX);
    for (std::uint32_t nodeIndex = 0; nodeIndex < source.nodes.size(); ++nodeIndex)
    {
        if (source.nodes[nodeIndex].centroid.y <= cutoff)
        {
            remap[nodeIndex] = static_cast<std::uint32_t>(result.nodes.size());
            result.nodes.push_back(source.nodes[nodeIndex]);
        }
    }
    for (const SceneBond& sourceBond : source.bonds)
    {
        if (sourceBond.node0 >= remap.size() || sourceBond.node1 >= remap.size()
            || remap[sourceBond.node0] == UINT32_MAX
            || remap[sourceBond.node1] == UINT32_MAX)
        {
            continue;
        }
        SceneBond bond = sourceBond;
        bond.node0 = remap[sourceBond.node0];
        bond.node1 = remap[sourceBond.node1];
        result.bonds.push_back(bond);
    }
    if (result.nodes.empty() || result.bonds.empty()
        || std::none_of(
            result.nodes.begin(),
            result.nodes.end(),
            [](const SceneNode& node) { return node.mass == 0.0f; }))
    {
        throw std::runtime_error("floor truncation produced an invalid supported structure");
    }
    return result;
}

std::vector<BuildingVariant> makeBuildingVariants(
    const ScenePack& source,
    bool variedBuildingHeights)
{
    constexpr std::uint32_t maximumFloors = 3;
    const std::uint32_t firstFloor = variedBuildingHeights ? 1 : maximumFloors;
    std::vector<BuildingVariant> result;
    for (std::uint32_t floors = firstFloor; floors <= maximumFloors; ++floors)
    {
        BuildingVariant variant;
        variant.pack = truncateToFloors(source, floors, maximumFloors);
        variant.nodes = makeNodeDescs(variant.pack);
        variant.bonds = makeBondDescs(variant.pack);
        variant.floors = floors;
        for (const SceneNode& node : variant.pack.nodes)
        {
            variant.height = std::max(
                variant.height,
                node.centroid.y + node.visualHalfExtents.y);
        }
        result.push_back(std::move(variant));
    }
    return result;
}

std::array<Camera, 4> makeCameras(std::uint32_t grid, float pitch)
{
    const float radius = std::max(28.0f, grid * pitch * 0.85f);
    const PxVec3 target(0.0f, 8.0f, 0.0f);
    std::array<Camera, 4> cameras{{
        {PxVec3(radius, radius * 0.65f, radius), PxVec3(0.0f), 55.0f},
        {PxVec3(-radius, radius * 0.45f, radius * 0.8f), PxVec3(0.0f), 55.0f},
        {PxVec3(0.0f, radius * 1.5f, 0.1f), PxVec3(0.0f), 60.0f},
        {PxVec3(0.0f, 14.0f, -radius), PxVec3(0.0f), 70.0f},
    }};
    for (Camera& camera : cameras)
    {
        camera.direction = (target - camera.eye).getNormalized();
    }
    return cameras;
}

std::vector<PxVec3> buildingOffsets(std::uint32_t grid, float pitch)
{
    std::vector<PxVec3> offsets;
    offsets.reserve(grid * grid);
    const float half = static_cast<float>(grid - 1) * pitch * 0.5f;
    for (std::uint32_t row = 0; row < grid; ++row)
    {
        for (std::uint32_t column = 0; column < grid; ++column)
        {
            offsets.emplace_back(
                -half + column * pitch,
                0.0f,
                -half + row * pitch);
        }
    }
    return offsets;
}

std::vector<Projectile> createProjectiles(
    PhysXScene& context,
    const ScenePack& pack,
    const Options& options,
    const std::vector<PxVec3>& offsets,
    const std::vector<float>& buildingHeights,
    std::uint32_t firstVisualId,
    float settleSeconds)
{
    const std::size_t projectileCount =
        offsets.size() + (offsets.size() + 7) / 8;
    const float projectileLifetime =
        pack.projectileTtlSeconds * options.projectileTtlScale;
    const float launchWindow =
        options.durationSeconds - projectileLifetime - 1.0f;
    if (projectileCount > 1 && launchWindow <= 0.0f)
    {
        throw std::runtime_error(
            "destruction duration must exceed projectile lifetime by at least one second");
    }
    const float launchSpacing = projectileCount > 1
        ? std::min(0.15f, launchWindow / static_cast<float>(projectileCount - 1))
        : 0.0f;
    std::vector<Projectile> projectiles;
    projectiles.reserve(projectileCount);
    for (std::size_t round = 0; round < 2; ++round)
    {
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            if (round == 1 && i % 8 != 0)
            {
                continue;
            }
            const float targetHeight = std::max(
                1.5f,
                buildingHeights[i] * (round == 0 ? 0.58f : 0.34f));
            const PxVec3 target = offsets[i] + PxVec3(0.0f, targetHeight, 0.0f);
            PxVec3 start;
            const float side = -1.0f;
            if (round == 0)
            {
                start = target + PxVec3(side * 11.0f, 1.5f, -3.0f);
            }
            else
            {
                // A lower, opposing oblique hit peels wall/floor sections away
                // instead of punching vertically through the roof.
                start = target + PxVec3(-side * 11.0f, 2.0f, 3.0f);
            }
            const PxVec3 velocity = (target - start).getNormalized()
                * pack.projectileSpeed
                * options.projectileSpeedScale
                * (round == 0 ? 1.0f : 0.8f);

            PxRigidDynamic* body = context.physics().createRigidDynamic(
                PxTransform(PxVec3(0.0f, -1000.0f, 0.0f)));
            if (!body)
            {
                throw std::runtime_error("could not create projectile body");
            }
            PxShape* shape = context.physics().createShape(
                PxSphereGeometry(
                    pack.projectileRadius
                    * options.projectileRadiusScale
                    * (round == 0 ? 1.0f : 1.15f)),
                context.material(),
                false);
            if (!shape || !body->attachShape(*shape))
            {
                if (shape) shape->release();
                body->release();
                throw std::runtime_error("could not create projectile shape");
            }
            body->setMass(
                pack.projectileMass
                * options.projectileMassScale
                * (round == 0 ? 1.0f : 1.25f));
            body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
            context.scene().addActor(*body);

            Projectile projectile;
            projectile.body = body;
            projectile.shape = shape;
            projectile.visualId = firstVisualId + static_cast<std::uint32_t>(projectiles.size());
            // Strike every building once, then every eighth building from the
            // opposite side. Spread the work over the available destruction
            // window so this remains a useful frame-budget stress test.
            projectile.launchAt =
                settleSeconds
                + static_cast<float>(projectiles.size()) * launchSpacing;
            projectile.retireAt = projectile.launchAt + projectileLifetime;
            projectile.launchPosition = start;
            projectile.launchVelocity = velocity;
            projectiles.push_back(projectile);
        }
    }
    return projectiles;
}

void releaseProjectiles(std::vector<Projectile>& projectiles)
{
    for (Projectile& projectile : projectiles)
    {
        if (projectile.body)
        {
            projectile.body->release();
            projectile.body = nullptr;
        }
        if (projectile.shape)
        {
            projectile.shape->release();
            projectile.shape = nullptr;
        }
    }
}

void launchProjectiles(std::vector<Projectile>& projectiles, float simulationTime)
{
    for (Projectile& projectile : projectiles)
    {
        if (projectile.launched
            && !projectile.retired
            && simulationTime >= projectile.retireAt)
        {
            projectile.body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
            projectile.body->setGlobalPose(PxTransform(PxVec3(0.0f, -1000.0f, 0.0f)));
            projectile.retired = true;
            continue;
        }
        if (projectile.launched || simulationTime < projectile.launchAt)
        {
            continue;
        }
        projectile.body->setGlobalPose(PxTransform(projectile.launchPosition));
        projectile.body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, false);
        projectile.body->setLinearVelocity(projectile.launchVelocity);
        projectile.body->setAngularVelocity(PxVec3(1.0f, 2.0f, 0.5f));
        projectile.body->wakeUp();
        projectile.launched = true;
    }
}

AggregateTelemetry aggregate(const std::vector<DestructiblePtr>& destructibles)
{
    AggregateTelemetry result;
    for (const DestructiblePtr& destructible : destructibles)
    {
        const ExtStressPhysXTelemetry& source = destructible->getTelemetry();
        result.contactsQueued += source.contactsQueued;
        result.contactsProcessed += source.contactsProcessed;
        result.contactsDropped += source.contactsDropped;
        result.splits += source.splits;
        result.bodiesCreated += source.bodiesCreated;
        result.bodiesReused += source.bodiesReused;
        result.bodiesRecycled += source.bodiesRecycled;
        result.shapesMigrated += source.shapesMigrated;
        result.sleepingActorsSkipped += source.sleepingActorsSkipped;
        result.peakBodies += source.bodyCount;
        result.peakAwakeBodies += source.awakeDynamicBodyCount;
        result.overstressedBonds += source.overstressedBondCount;
        result.solverIslands += source.solverIslandCount;
        result.solverIslandsSkipped += source.solverIslandsSkipped;
        result.maxPositionDrift =
            std::max(result.maxPositionDrift, source.maxSplitWorldPositionDrift);
        result.maxVelocityDrift =
            std::max(result.maxVelocityDrift, source.maxSplitPointVelocityDrift);
        result.contactProcessingMilliseconds += source.contactProcessingMilliseconds;
        result.gravityMilliseconds += source.gravityMilliseconds;
        result.stressSolveMilliseconds += source.stressSolveMilliseconds;
        result.gpuStressSolveMilliseconds += source.gpuStressSolveMilliseconds;
        result.gpuStressHostToDeviceBytes += source.gpuStressHostToDeviceBytes;
        result.gpuStressDeviceToHostBytes += source.gpuStressDeviceToHostBytes;
        result.fractureTopologyMilliseconds += source.fractureTopologyMilliseconds;
        result.mappingValidationMilliseconds += source.mappingValidationMilliseconds;
    }
    return result;
}

DestructionDistribution destructionDistribution(
    const std::vector<DestructiblePtr>& destructibles,
    const std::vector<std::uint32_t>& chunksPerStructure)
{
    DestructionDistribution result;
    if (destructibles.empty())
    {
        return result;
    }
    result.minimumBodiesPerStructure = UINT32_MAX;
    std::uint64_t totalBodies = 0;
    for (std::size_t structure = 0; structure < destructibles.size(); ++structure)
    {
        const DestructiblePtr& destructible = destructibles[structure];
        const std::uint32_t chunks = chunksPerStructure[structure];
        const std::uint32_t heavyThreshold =
            std::max<std::uint32_t>(2, (chunks + 1) / 2);
        const std::uint32_t shatteredThreshold =
            std::max<std::uint32_t>(2, static_cast<std::uint32_t>(
                std::ceil(chunks * 0.75)));
        const std::uint32_t bodies = destructible->getTelemetry().bodyCount;
        totalBodies += bodies;
        result.minimumBodiesPerStructure =
            std::min(result.minimumBodiesPerStructure, bodies);
        result.maximumBodiesPerStructure =
            std::max(result.maximumBodiesPerStructure, bodies);
        if (bodies <= 1)
        {
            ++result.intactStructures;
        }
        else if (bodies < heavyThreshold)
        {
            ++result.partiallyFracturedStructures;
        }
        else if (bodies < shatteredThreshold)
        {
            ++result.heavilyFracturedStructures;
        }
        else
        {
            ++result.shatteredStructures;
        }
    }
    result.meanBodiesPerStructure =
        static_cast<double>(totalBodies) / destructibles.size();
    return result;
}

DestructionMotion destructionMotion(
    const std::vector<DestructiblePtr>& destructibles,
    const std::vector<const ScenePack*>& packs,
    const std::vector<PxVec3>& offsets)
{
    DestructionMotion result;
    for (std::size_t structure = 0; structure < destructibles.size(); ++structure)
    {
        const DestructiblePtr& destructible = destructibles[structure];
        const ScenePack& pack = *packs[structure];
        std::vector<ExtStressPhysXBodySnapshot> bodies(pack.nodes.size());
        const std::uint32_t bodyCount = destructible->getBodySnapshots(
            bodies.data(),
            static_cast<std::uint32_t>(bodies.size()));
        std::unordered_map<ExtStressPhysXId, const ExtStressPhysXBodySnapshot*> bodyById;
        bodyById.reserve(bodyCount);
        for (std::uint32_t bodyIndex = 0; bodyIndex < bodyCount; ++bodyIndex)
        {
            const ExtStressPhysXBodySnapshot& body = bodies[bodyIndex];
            bodyById.emplace(body.bodyId, &body);
            if (body.kinematic)
            {
                result.supportedRemainderChunks += body.nodeCount;
            }
            else
            {
                result.dynamicChunks += body.nodeCount;
                if (body.linearVelocity.magnitude() >= 0.5f
                    || body.angularVelocity.magnitude() >= 0.25f)
                {
                    result.movingChunks += body.nodeCount;
                }
            }
        }

        std::vector<ExtStressPhysXShapeSnapshot> shapes(pack.nodes.size());
        const std::uint32_t shapeCount = destructible->getShapeSnapshots(
            shapes.data(),
            static_cast<std::uint32_t>(shapes.size()));
        std::uint32_t structureMovedChunks = 0;
        std::uint32_t structureFallenChunks = 0;
        for (std::uint32_t shapeIndex = 0; shapeIndex < shapeCount; ++shapeIndex)
        {
            const ExtStressPhysXShapeSnapshot& shape = shapes[shapeIndex];
            if (shape.nodeIndex >= pack.nodes.size())
            {
                continue;
            }
            const PxVec3 initialPosition =
                offsets[structure] + pack.nodes[shape.nodeIndex].centroid;
            const PxVec3 delta = shape.worldPose.p - initialPosition;
            const float displacement = delta.magnitude();
            const float downwardDisplacement = std::max(0.0f, -delta.y);
            result.maximumDisplacement = std::max(result.maximumDisplacement, displacement);
            result.maximumDownwardDisplacement =
                std::max(result.maximumDownwardDisplacement, downwardDisplacement);
            if (displacement >= 0.5f)
            {
                ++result.movedChunks;
                ++structureMovedChunks;
            }
            if (downwardDisplacement >= 0.5f)
            {
                ++result.fallenChunks;
                ++structureFallenChunks;
            }
            if (displacement >= 2.0f)
            {
                ++result.farTravelingChunks;
            }
        }
        if (structureMovedChunks >= 1)
        {
            ++result.structuresWithMovedChunks;
        }
        if (structureFallenChunks >= 1)
        {
            ++result.structuresWithFallenChunks;
        }
    }
    return result;
}

std::vector<VisualPose> collectVisualPoses(
    const std::vector<DestructiblePtr>& destructibles,
    const std::vector<Projectile>& projectiles,
    const std::vector<std::uint32_t>& nodesPerBuilding,
    const std::vector<std::uint32_t>& visualBases,
    bool fullSnapshot)
{
    std::vector<VisualPose> result;
    result.reserve(
        (visualBases.empty() ? 0 : visualBases.back() + nodesPerBuilding.back())
        + projectiles.size());
    const std::uint32_t maximumNodes = nodesPerBuilding.empty()
        ? 0
        : *std::max_element(nodesPerBuilding.begin(), nodesPerBuilding.end());
    std::vector<ExtStressPhysXShapeSnapshot> shapes(maximumNodes);
    for (std::size_t building = 0; building < destructibles.size(); ++building)
    {
        const std::uint32_t nodeCount = nodesPerBuilding[building];
        const std::uint32_t shapeCount = fullSnapshot
            ? destructibles[building]->getShapeSnapshots(shapes.data(), nodeCount)
            : destructibles[building]->getActiveShapeSnapshots(shapes.data(), nodeCount);
        for (std::uint32_t i = 0; i < shapeCount; ++i)
        {
            VisualPose pose;
            pose.actorId = visualBases[building] + shapes[i].nodeIndex;
            pose.pose = shapes[i].worldPose;
            pose.sleeping = shapes[i].bodySleeping;
            result.push_back(pose);
        }
    }
    for (const Projectile& projectile : projectiles)
    {
        VisualPose pose;
        pose.actorId = projectile.visualId;
        pose.pose = projectile.body->getGlobalPose();
        pose.sleeping =
            !projectile.launched || projectile.retired || projectile.body->isSleeping();
        result.push_back(pose);
    }
    return result;
}

class FrameTelemetryWriter
{
public:
    explicit FrameTelemetryWriter(const std::string& path)
    {
        if (path.empty())
        {
            return;
        }
        m_output.open(path);
        if (!m_output)
        {
            throw std::runtime_error("could not write frame telemetry: " + path);
        }
        m_output
            << "step,simulation_seconds,physics_step_ms,contact_callback_ms,"
               "contact_processing_ms,gravity_ms,stress_solve_ms,gpu_stress_solve_ms,"
               "gpu_stress_h2d_bytes,gpu_stress_d2h_bytes,fracture_topology_ms,"
               "adapter_tick_ms,mapping_validation_ms,state_export_ms,frame_host_ms,"
               "realtime_factor,bodies,awake_bodies,solver_islands,solver_islands_skipped,"
               "overstressed_bonds,contacts_frame,contacts_total,contacts_dropped_total,"
               "projectile_impacts_frame,projectile_impacts_total,"
               "projectile_impulse_frame,projectile_impulse_total,"
               "splits_frame,splits_total,shapes_migrated_frame,shapes_migrated_total,"
               "sleeping_actors_skipped,max_position_drift,max_point_velocity_drift\n";
    }

    void write(const FrameMetrics& frame)
    {
        if (!m_output)
        {
            return;
        }
        const double realtimeFactor =
            (1000.0 / 60.0) / std::max(frame.frameHostMilliseconds, 1.0e-9);
        m_output
            << frame.step << ','
            << frame.simulationSeconds << ','
            << frame.physicsStepMilliseconds << ','
            << frame.contactCallbackMilliseconds << ','
            << frame.contactProcessingMilliseconds << ','
            << frame.gravityMilliseconds << ','
            << frame.stressSolveMilliseconds << ','
            << frame.gpuStressSolveMilliseconds << ','
            << frame.gpuStressHostToDeviceBytes << ','
            << frame.gpuStressDeviceToHostBytes << ','
            << frame.fractureTopologyMilliseconds << ','
            << frame.adapterTickMilliseconds << ','
            << frame.mappingValidationMilliseconds << ','
            << frame.stateExportMilliseconds << ','
            << frame.frameHostMilliseconds << ','
            << realtimeFactor << ','
            << frame.telemetry.peakBodies << ','
            << frame.telemetry.peakAwakeBodies << ','
            << frame.telemetry.solverIslands << ','
            << frame.telemetry.solverIslandsSkipped << ','
            << frame.telemetry.overstressedBonds << ','
            << frame.frameContacts << ','
            << frame.telemetry.contactsProcessed << ','
            << frame.telemetry.contactsDropped << ','
            << frame.frameProjectileImpactContacts << ','
            << frame.projectileImpactContacts << ','
            << frame.frameProjectileImpactImpulse << ','
            << frame.projectileImpactImpulse << ','
            << frame.frameSplits << ','
            << frame.telemetry.splits << ','
            << frame.frameShapesMigrated << ','
            << frame.telemetry.shapesMigrated << ','
            << frame.telemetry.sleepingActorsSkipped << ','
            << frame.telemetry.maxPositionDrift << ','
            << frame.telemetry.maxVelocityDrift << '\n';
    }

private:
    std::ofstream m_output;
};

struct PhaseStats
{
    double mean{0.0};
    double p95{0.0};
    double maximum{0.0};
};

PhaseStats summarizePhase(const std::vector<double>& samples)
{
    PhaseStats result;
    if (samples.empty())
    {
        return result;
    }
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double sample : samples)
    {
        sum += sample;
    }
    result.mean = sum / samples.size();
    const std::size_t p95Index =
        std::min(sorted.size() - 1, static_cast<std::size_t>(std::ceil(sorted.size() * 0.95)) - 1);
    result.p95 = sorted[p95Index];
    result.maximum = sorted.back();
    return result;
}

void writeMetadata(
    const std::string& path,
    const Options& options,
    const ScenePack& pack,
    const PhysXScene& context,
    const AggregateTelemetry& telemetry,
    const RuntimeTimings& timings,
    const DestructionDistribution& destruction,
    const DestructionMotion& motion,
    std::uint64_t projectileImpactContacts,
    double projectileImpactImpulse,
    double wallSeconds,
    std::uint32_t frameCount,
    const std::vector<std::uint32_t>& nodesPerBuilding,
    const std::vector<std::uint32_t>& floorsPerBuilding)
{
    if (path.empty())
    {
        return;
    }
    const PxSimulationStatistics stats = context.statistics();
    const PxGpuDynamicsMemoryConfigStatistics& gpu = stats.gpuDynamicsMemoryConfigStatistics;
    const PhaseStats frameStats = summarizePhase(timings.frameHostMilliseconds);
    const PhaseStats physicsStats = summarizePhase(timings.physicsStepMilliseconds);
    const PhaseStats adapterStats = summarizePhase(timings.adapterTickMilliseconds);
    const PhaseStats gpuStressStats = summarizePhase(timings.gpuStressSolveMilliseconds);
    const PhaseStats exportStats = summarizePhase(timings.stateExportMilliseconds);
    std::uint64_t totalAuthoredChunks = 0;
    std::uint32_t minimumNodesPerBuilding = UINT32_MAX;
    std::uint32_t maximumNodesPerBuilding = 0;
    std::array<std::uint32_t, 4> buildingFloorCounts{};
    for (std::size_t building = 0; building < nodesPerBuilding.size(); ++building)
    {
        totalAuthoredChunks += nodesPerBuilding[building];
        minimumNodesPerBuilding =
            std::min(minimumNodesPerBuilding, nodesPerBuilding[building]);
        maximumNodesPerBuilding =
            std::max(maximumNodesPerBuilding, nodesPerBuilding[building]);
        if (floorsPerBuilding[building] < buildingFloorCounts.size())
        {
            ++buildingFloorCounts[floorsPerBuilding[building]];
        }
    }
    const double meanNodesPerBuilding =
        static_cast<double>(totalAuthoredChunks)
        / std::max<std::size_t>(1, nodesPerBuilding.size());
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("could not write metadata: " + path);
    }
    output
        << "{\n"
        << "  \"format\": \"blast-physx-mini-city-v1\",\n"
        << "  \"scene\": \"" << pack.title << "\",\n"
        << "  \"physicsMode\": \"" << (context.mode() == PhysicsMode::Gpu ? "gpu" : "cpu") << "\",\n"
        << "  \"gpuActive\": " << (context.gpuActive() ? "true" : "false") << ",\n"
        << "  \"gpuStressRequested\": " << (options.gpuStress ? "true" : "false") << ",\n"
        << "  \"gpuStressMinimumBondCount\": " << options.gpuStressMinimumBonds << ",\n"
        << "  \"gpuStressSerialized\": "
        << (options.serializeGpuStress ? "true" : "false") << ",\n"
        << "  \"realtimeRequired\": " << (options.requireRealtime ? "true" : "false") << ",\n"
        << "  \"minimumAuthoredChunksRequired\": "
        << options.requireMinimumAuthoredChunks << ",\n"
        << "  \"variedBuildingHeightsRequired\": "
        << (options.requireVariedBuildingHeights ? "true" : "false") << ",\n"
        << "  \"tuning\": {\n"
        << "    \"projectileMass\": " << pack.projectileMass * options.projectileMassScale << ",\n"
        << "    \"projectileSpeed\": " << pack.projectileSpeed * options.projectileSpeedScale << ",\n"
        << "    \"projectileRadius\": " << pack.projectileRadius * options.projectileRadiusScale << ",\n"
        << "    \"projectileLifetimeSeconds\": "
        << pack.projectileTtlSeconds * options.projectileTtlScale << ",\n"
        << "    \"contactForceScale\": "
        << pack.contactForceScale * options.contactForceScale << ",\n"
        << "    \"minimumStressContactImpulse\": "
        << options.minimumStressContactImpulse << ",\n"
        << "    \"stressLimitScale\": " << options.stressLimitScale << ",\n"
        << "    \"excessForceScale\": " << options.excessForceScale << ",\n"
        << "    \"maximumBodiesPerStructure\": "
        << options.maximumBodiesPerStructure << ",\n"
        << "    \"maximumFracturesPerActorPerTick\": "
        << options.maximumFracturesPerActorPerTick << ",\n"
        << "    \"tallBuildingStride\": " << options.tallBuildingStride << "\n"
        << "  },\n"
        << "  \"grid\": " << options.grid << ",\n"
        << "  \"buildingCount\": " << options.grid * options.grid << ",\n"
        << "  \"variedBuildingHeights\": "
        << (options.variedBuildingHeights ? "true" : "false") << ",\n"
        << "  \"authoredChunkCount\": " << totalAuthoredChunks << ",\n"
        << "  \"minimumNodesPerBuilding\": " << minimumNodesPerBuilding << ",\n"
        << "  \"maximumNodesPerBuilding\": " << maximumNodesPerBuilding << ",\n"
        << "  \"meanNodesPerBuilding\": " << meanNodesPerBuilding << ",\n"
        << "  \"buildingFloorCounts\": {\"1\": " << buildingFloorCounts[1]
        << ", \"2\": " << buildingFloorCounts[2]
        << ", \"3\": " << buildingFloorCounts[3] << "},\n"
        << "  \"frames\": " << frameCount << ",\n"
        << "  \"fixedHz\": 60,\n"
        << "  \"wallSeconds\": " << wallSeconds << ",\n"
        << "  \"simulationRate\": "
        << ((options.durationSeconds + options.settleSeconds) / std::max(wallSeconds, 1.0e-9))
        << ",\n"
        << "  \"contactsQueued\": " << telemetry.contactsQueued << ",\n"
        << "  \"contactsProcessed\": " << telemetry.contactsProcessed << ",\n"
        << "  \"contactsDropped\": " << telemetry.contactsDropped << ",\n"
        << "  \"projectileImpactContacts\": " << projectileImpactContacts << ",\n"
        << "  \"projectileImpactImpulse\": " << projectileImpactImpulse << ",\n"
        << "  \"splits\": " << telemetry.splits << ",\n"
        << "  \"bodyCount\": " << telemetry.peakBodies << ",\n"
        << "  \"awakeBodyCount\": " << telemetry.peakAwakeBodies << ",\n"
        << "  \"peakBodyCount\": " << telemetry.peakBodies << ",\n"
        << "  \"peakAwakeBodyCount\": " << telemetry.peakAwakeBodies << ",\n"
        << "  \"bodiesCreated\": " << telemetry.bodiesCreated << ",\n"
        << "  \"bodiesReused\": " << telemetry.bodiesReused << ",\n"
        << "  \"bodiesRecycled\": " << telemetry.bodiesRecycled << ",\n"
        << "  \"shapesMigrated\": " << telemetry.shapesMigrated << ",\n"
        << "  \"maxSplitWorldPositionDrift\": " << telemetry.maxPositionDrift << ",\n"
        << "  \"maxSplitPointVelocityDrift\": " << telemetry.maxVelocityDrift << ",\n"
        << "  \"gpuStressSolveMilliseconds\": "
        << telemetry.gpuStressSolveMilliseconds << ",\n"
        << "  \"gpuStressHostToDeviceBytes\": "
        << telemetry.gpuStressHostToDeviceBytes << ",\n"
        << "  \"gpuStressDeviceToHostBytes\": "
        << telemetry.gpuStressDeviceToHostBytes << ",\n"
        << "  \"destructionDistribution\": {\n"
        << "    \"intactStructures\": " << destruction.intactStructures << ",\n"
        << "    \"partiallyFracturedStructures\": "
        << destruction.partiallyFracturedStructures << ",\n"
        << "    \"heavilyFracturedStructures\": "
        << destruction.heavilyFracturedStructures << ",\n"
        << "    \"shatteredStructures\": " << destruction.shatteredStructures << ",\n"
        << "    \"minimumBodiesPerStructure\": "
        << destruction.minimumBodiesPerStructure << ",\n"
        << "    \"maximumBodiesPerStructure\": "
        << destruction.maximumBodiesPerStructure << ",\n"
        << "    \"meanBodiesPerStructure\": "
        << destruction.meanBodiesPerStructure << "\n"
        << "  },\n"
        << "  \"destructionMotion\": {\n"
        << "    \"structuresWithMovedChunks\": " << motion.structuresWithMovedChunks << ",\n"
        << "    \"structuresWithFallenChunks\": " << motion.structuresWithFallenChunks << ",\n"
        << "    \"movedChunks\": " << motion.movedChunks << ",\n"
        << "    \"fallenChunks\": " << motion.fallenChunks << ",\n"
        << "    \"farTravelingChunks\": " << motion.farTravelingChunks << ",\n"
        << "    \"dynamicChunks\": " << motion.dynamicChunks << ",\n"
        << "    \"movingChunks\": " << motion.movingChunks << ",\n"
        << "    \"supportedRemainderChunks\": " << motion.supportedRemainderChunks << ",\n"
        << "    \"maximumDisplacement\": " << motion.maximumDisplacement << ",\n"
        << "    \"maximumDownwardDisplacement\": "
        << motion.maximumDownwardDisplacement << "\n"
        << "  },\n"
        << "  \"frameTelemetry\": {\n"
        << "    \"samples\": " << timings.frameHostMilliseconds.size() << ",\n"
        << "    \"budgetMilliseconds\": " << (1000.0 / 60.0) << ",\n"
        << "    \"budgetMissFrames\": " << timings.budgetMissFrames << ",\n"
        << "    \"destructionFrameSamples\": " << timings.destructionFrameSamples << ",\n"
        << "    \"destructionBudgetMissFrames\": "
        << timings.destructionBudgetMissFrames << ",\n"
        << "    \"maximumDestructionFrameMilliseconds\": "
        << timings.maximumDestructionFrameMilliseconds << ",\n"
        << "    \"budgetMissPercent\": "
        << (100.0 * timings.budgetMissFrames
            / std::max<std::size_t>(1, timings.frameHostMilliseconds.size()))
        << ",\n"
        << "    \"meanHostFramesPerSecond\": "
        << (1000.0 / std::max(frameStats.mean, 1.0e-9)) << ",\n"
        << "    \"meanFrameHostMilliseconds\": " << frameStats.mean << ",\n"
        << "    \"p95FrameHostMilliseconds\": " << frameStats.p95 << ",\n"
        << "    \"maxFrameHostMilliseconds\": " << frameStats.maximum << ",\n"
        << "    \"meanPhysicsStepMilliseconds\": " << physicsStats.mean << ",\n"
        << "    \"p95PhysicsStepMilliseconds\": " << physicsStats.p95 << ",\n"
        << "    \"maxPhysicsStepMilliseconds\": " << physicsStats.maximum << ",\n"
        << "    \"meanAdapterTickMilliseconds\": " << adapterStats.mean << ",\n"
        << "    \"p95AdapterTickMilliseconds\": " << adapterStats.p95 << ",\n"
        << "    \"maxAdapterTickMilliseconds\": " << adapterStats.maximum << ",\n"
        << "    \"meanGpuStressSolveMilliseconds\": " << gpuStressStats.mean << ",\n"
        << "    \"p95GpuStressSolveMilliseconds\": " << gpuStressStats.p95 << ",\n"
        << "    \"maxGpuStressSolveMilliseconds\": " << gpuStressStats.maximum << ",\n"
        << "    \"meanStateExportMilliseconds\": " << exportStats.mean << ",\n"
        << "    \"p95StateExportMilliseconds\": " << exportStats.p95 << ",\n"
        << "    \"maxStateExportMilliseconds\": " << exportStats.maximum << "\n"
        << "  },\n"
        << "  \"gpuRequiredTempBytes\": " << gpu.tempBufferCapacity << ",\n"
        << "  \"gpuRequiredRigidContacts\": " << gpu.rigidContactCount << ",\n"
        << "  \"gpuRequiredRigidPatches\": " << gpu.rigidPatchCount << ",\n"
        << "  \"gpuRequiredFoundLostPairs\": " << gpu.foundLostPairs << ",\n"
        << "  \"gpuRequiredCollisionStackBytes\": " << gpu.collisionStackSize << "\n"
        << "}\n";
}

int run(const Options& options)
{
    constexpr float physicsDt = 1.0f / 60.0f;
    constexpr float cityPitch = 18.0f;

    const ScenePack pack = loadScenePack(options.scenePath);
    const std::vector<BuildingVariant> variants =
        makeBuildingVariants(pack, options.variedBuildingHeights);
    if (options.selfTest)
    {
        const std::vector<BuildingVariant> testVariants =
            makeBuildingVariants(pack, true);
        requireContract(testVariants.size() == 3, "skyline must expose three floor variants");
        for (std::size_t i = 0; i < testVariants.size(); ++i)
        {
            constexpr std::array<std::size_t, 3> expectedNodes{83, 148, 204};
            constexpr std::array<std::size_t, 3> expectedBonds{209, 373, 546};
            requireContract(
                testVariants[i].floors == i + 1,
                "skyline floor metadata is not monotonic");
            requireContract(
                testVariants[i].pack.nodes.size() == expectedNodes[i],
                "skyline variant authored-chunk count changed");
            requireContract(
                testVariants[i].pack.bonds.size() == expectedBonds[i],
                "skyline variant bond count changed");
            std::size_t supportCount = 0;
            std::vector<std::vector<std::uint32_t>> adjacency(
                testVariants[i].pack.nodes.size());
            for (const SceneBond& bond : testVariants[i].pack.bonds)
            {
                requireContract(
                    bond.node0 < testVariants[i].pack.nodes.size()
                        && bond.node1 < testVariants[i].pack.nodes.size(),
                    "truncated skyline bond remap is invalid");
                adjacency[bond.node0].push_back(bond.node1);
                adjacency[bond.node1].push_back(bond.node0);
            }
            for (std::size_t node = 0; node < testVariants[i].pack.nodes.size(); ++node)
            {
                supportCount += testVariants[i].pack.nodes[node].mass == 0.0f ? 1 : 0;
                requireContract(
                    testVariants[i].pack.nodes[node].mass == 0.0f
                        || !adjacency[node].empty(),
                    "truncated skyline contains an orphan dynamic chunk");
            }
            requireContract(supportCount == 36, "skyline support-node count changed");
            std::vector<bool> visited(testVariants[i].pack.nodes.size(), false);
            std::vector<std::uint32_t> pending{0};
            visited[0] = true;
            for (std::size_t cursor = 0; cursor < pending.size(); ++cursor)
            {
                for (std::uint32_t neighbor : adjacency[pending[cursor]])
                {
                    if (!visited[neighbor])
                    {
                        visited[neighbor] = true;
                        pending.push_back(neighbor);
                    }
                }
            }
            requireContract(
                std::all_of(visited.begin(), visited.end(), [](bool value) { return value; }),
                "truncated skyline graph is disconnected");
        }
    }
    const std::vector<PxVec3> offsets = buildingOffsets(options.grid, cityPitch);
    std::vector<BuildingInstance> buildings;
    std::vector<const ScenePack*> buildingPacks;
    std::vector<std::uint32_t> nodesPerBuilding;
    std::vector<std::uint32_t> floorsPerBuilding;
    std::vector<std::uint32_t> visualBases;
    std::vector<float> buildingHeights;
    buildings.reserve(offsets.size());
    buildingPacks.reserve(offsets.size());
    nodesPerBuilding.reserve(offsets.size());
    floorsPerBuilding.reserve(offsets.size());
    visualBases.reserve(offsets.size());
    buildingHeights.reserve(offsets.size());
    std::uint32_t totalAuthoredChunks = 0;
    for (std::size_t buildingIndex = 0; buildingIndex < offsets.size(); ++buildingIndex)
    {
        std::size_t variantIndex = 0;
        if (options.variedBuildingHeights)
        {
            if (options.tallBuildingStride > 0)
            {
                variantIndex =
                    buildingIndex % options.tallBuildingStride == 0
                    ? 2
                    : buildingIndex % 2;
            }
            else
            {
                variantIndex = 2 - buildingIndex % 3;
            }
        }
        const BuildingVariant& variant = variants[variantIndex];
        buildings.push_back({variantIndex, offsets[buildingIndex], totalAuthoredChunks});
        buildingPacks.push_back(&variant.pack);
        nodesPerBuilding.push_back(static_cast<std::uint32_t>(variant.pack.nodes.size()));
        floorsPerBuilding.push_back(variant.floors);
        visualBases.push_back(totalAuthoredChunks);
        buildingHeights.push_back(variant.height);
        totalAuthoredChunks += static_cast<std::uint32_t>(variant.pack.nodes.size());
    }
    if (options.requireVariedBuildingHeights)
    {
        std::array<std::uint32_t, 4> floorCounts{};
        for (std::uint32_t floors : floorsPerBuilding)
        {
            if (floors < floorCounts.size())
            {
                ++floorCounts[floors];
            }
        }
        if (!options.variedBuildingHeights
            || variants.size() != 3
            || floorCounts[1] == 0
            || floorCounts[2] == 0
            || floorCounts[3] == 0)
        {
            throw std::runtime_error(
                "--require-varied-building-heights needs populated 1/2/3-floor variants");
        }
    }

    ContactRouter contacts(
        pack.contactForceScale * options.contactForceScale,
        options.minimumStressContactImpulse);
    SceneCapacity capacity;
    capacity.maxBodies = totalAuthoredChunks * 2;
    capacity.maxShapes = capacity.maxBodies + static_cast<std::uint32_t>(offsets.size() * 2);
    capacity.maxContactPairs = std::max<std::uint32_t>(65536, capacity.maxBodies * 64);
    PhysXScene context(options.physics, options.requireGpu, capacity, &contacts);
    context.scene().setGravity(PxVec3(0.0f, pack.gravity, 0.0f));

    std::vector<DestructiblePtr> destructibles;
    destructibles.reserve(offsets.size());
    for (const BuildingInstance& building : buildings)
    {
        const BuildingVariant& variant = variants[building.variantIndex];
        ExtStressPhysXDesc desc;
        desc.physics = &context.physics();
        desc.scene = &context.scene();
        desc.material = &context.material();
        desc.nodes = variant.nodes.data();
        desc.nodeCount = static_cast<std::uint32_t>(variant.nodes.size());
        desc.bonds = variant.bonds.data();
        desc.bondCount = static_cast<std::uint32_t>(variant.bonds.size());
        desc.worldTransform = PxTransform(building.offset);
        desc.settings.compressionElasticLimit =
            pack.stressLimits.compressionElastic * options.stressLimitScale;
        desc.settings.compressionFatalLimit =
            pack.stressLimits.compressionFatal * options.stressLimitScale;
        desc.settings.tensionElasticLimit =
            pack.stressLimits.tensionElastic * options.stressLimitScale;
        desc.settings.tensionFatalLimit =
            pack.stressLimits.tensionFatal * options.stressLimitScale;
        desc.settings.shearElasticLimit =
            pack.stressLimits.shearElastic * options.stressLimitScale;
        desc.settings.shearFatalLimit =
            pack.stressLimits.shearFatal * options.stressLimitScale;
        desc.settings.islandAware = true;
        desc.settings.skipSettledIslands = true;
        desc.settings.gpuStressSolver = options.gpuStress;
        desc.settings.gpuStressMinimumBondCount = options.gpuStressMinimumBonds;
        desc.settings.recordSplitContinuity = true;
        desc.settings.applyExcessForces = true;
        desc.settings.excessForceScale = options.excessForceScale;
        desc.settings.maximumBodies = options.maximumBodiesPerStructure;
        desc.settings.maximumFracturesPerActorPerTick =
            options.maximumFracturesPerActorPerTick;
        desc.settings.maximumLinearVelocity = 6.0f;
        desc.settings.maximumAngularVelocity = 8.0f;
        desc.settings.minimumSeparationVelocity = 4.0f;
        desc.errorCallback = adapterError;

        ExtStressPhysXTelemetry failure;
        DestructiblePtr destructible(ExtStressPhysXDestructible::create(desc, &failure));
        if (!destructible)
        {
            throw std::runtime_error(
                "could not create PhysX destructible; adapter error "
                + std::to_string(static_cast<unsigned>(failure.lastError))
                + " at node " + std::to_string(failure.lastErrorNode));
        }
        contacts.registerDestructible(*destructible, desc.nodeCount);
        destructibles.push_back(std::move(destructible));
    }
    ContractBaseline contractBaseline;
    if (options.selfTest)
    {
        requireContract(destructibles.size() == 1, "self-test must create exactly one structure");
        contractBaseline = captureContractBaseline(
            *destructibles.front(),
            nodesPerBuilding.front());
    }

    std::vector<Projectile> projectiles;
    if (!options.selfTest)
    {
        projectiles = createProjectiles(
            context,
            pack,
            options,
            offsets,
            buildingHeights,
            totalAuthoredChunks,
            options.settleSeconds);
    }
    for (const Projectile& projectile : projectiles)
    {
        contacts.registerProjectile(*projectile.shape);
    }
    StressExecutor stressExecutor(
        resolveStressWorkerCount(options.stressWorkers),
        options.serializeGpuStress);

    // Warm lazy PhysX/CUDA allocations and solver scratch before the measured
    // fixed-step interval. The structures and projectiles are still kinematic,
    // so this cannot consume an impact or alter the authored initial pose.
    for (std::uint32_t warmupStep = 0; warmupStep < 8; ++warmupStep)
    {
        context.scene().simulate(physicsDt);
        if (!context.scene().fetchResults(true))
        {
            throw std::runtime_error("PxScene::fetchResults failed during warmup");
        }
        tickDestructibles(
            destructibles,
            stressExecutor,
            physicsDt,
            PxVec3(0.0f, pack.gravity, 0.0f));
    }
    if (aggregate(destructibles).splits != 0)
    {
        throw std::runtime_error("structures fractured under self-weight during warmup");
    }

    StateWriter writer;
    const std::uint32_t snapshotStride = 60 / options.snapshotFps;
    const std::uint32_t physicsSteps = static_cast<std::uint32_t>(
        std::ceil((options.durationSeconds + options.settleSeconds) * 60.0f));
    const std::uint32_t frameCount = physicsSteps / snapshotStride + 1;
    if (!options.statePath.empty())
    {
        const auto cameras = makeCameras(options.grid, cityPitch);
        if (!writer.open(
                options.statePath,
                options.snapshotFps,
                frameCount,
                options.paneWidth,
                options.paneHeight,
                static_cast<std::uint32_t>(offsets.size()),
                options.durationSeconds,
                options.settleSeconds,
                cameras))
        {
            throw std::runtime_error(writer.error());
        }
        for (std::size_t building = 0; building < destructibles.size(); ++building)
        {
            const ScenePack& buildingPack = *buildingPacks[building];
            for (std::size_t node = 0; node < buildingPack.nodes.size(); ++node)
            {
                VisualActor actor;
                actor.shape = VisualActor::Shape::Box;
                actor.part = buildingPack.nodes[node].mass == 0.0f
                    ? 7
                    : static_cast<std::uint8_t>(building % 5);
                actor.parameters = buildingPack.nodes[node].visualHalfExtents;
                const std::uint32_t id =
                    visualBases[building] + static_cast<std::uint32_t>(node);
                if (!writer.defineActor(id, actor))
                {
                    throw std::runtime_error(writer.error());
                }
            }
        }
        for (const Projectile& projectile : projectiles)
        {
            VisualActor actor;
            actor.shape = VisualActor::Shape::Sphere;
            actor.part = 5;
            const PxSphereGeometry& geometry =
                static_cast<const PxSphereGeometry&>(projectile.shape->getGeometry());
            actor.parameters = PxVec3(geometry.radius, 0.0f, 0.0f);
            if (!writer.defineActor(projectile.visualId, actor))
            {
                throw std::runtime_error(writer.error());
            }
        }
        if (!writer.writeFrame(
                0,
                collectVisualPoses(
                    destructibles,
                    projectiles,
                    nodesPerBuilding,
                    visualBases,
                    true)))
        {
            throw std::runtime_error(writer.error());
        }
    }

    AggregateTelemetry peaks;
    AggregateTelemetry previousFrameTelemetry = aggregate(destructibles);
    double previousContactCallbackMilliseconds = contacts.callbackMilliseconds();
    std::uint64_t previousProjectileImpactContacts = contacts.projectileImpactContacts();
    double previousProjectileImpactImpulse = contacts.projectileImpactImpulse();
    FrameTelemetryWriter frameTelemetry(options.frameTelemetryPath);
    RuntimeTimings runtimeTimings;
    runtimeTimings.frameHostMilliseconds.reserve(physicsSteps);
    runtimeTimings.physicsStepMilliseconds.reserve(physicsSteps);
    runtimeTimings.adapterTickMilliseconds.reserve(physicsSteps);
    runtimeTimings.stateExportMilliseconds.reserve(physicsSteps);
    const auto start = std::chrono::steady_clock::now();
    std::uint32_t writtenFrames = 1;
    std::uint64_t splitsBeforeFirstImpact = 0;
    for (std::uint32_t step = 0; step < physicsSteps; ++step)
    {
        const auto frameStart = std::chrono::steady_clock::now();
        const float simulationTime = step * physicsDt;
        launchProjectiles(projectiles, simulationTime);

        const auto physicsStart = std::chrono::steady_clock::now();
        context.scene().simulate(physicsDt);
        if (!context.scene().fetchResults(true))
        {
            throw std::runtime_error("PxScene::fetchResults failed");
        }
        const double physicsStepMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - physicsStart).count();
        const double contactCallbackMilliseconds =
            contacts.callbackMilliseconds() - previousContactCallbackMilliseconds;

        if (options.selfTest && step == 0)
        {
            for (std::size_t structure = 0; structure < destructibles.size(); ++structure)
            {
                const DestructiblePtr& destructible = destructibles[structure];
                const ScenePack& buildingPack = *buildingPacks[structure];
                std::vector<ExtStressPhysXShapeSnapshot> shapes(buildingPack.nodes.size());
                destructible->getShapeSnapshots(
                    shapes.data(),
                    static_cast<std::uint32_t>(shapes.size()));
                for (const ExtStressPhysXShapeSnapshot& shape : shapes)
                {
                    if (buildingPack.nodes[shape.nodeIndex].mass > 0.0f)
                    {
                        destructible->queueContact(
                            *shape.shape,
                            shape.worldPose.p,
                            PxVec3(2.0e8f, 5.0e8f, -3.0e8f));
                        if (shape.nodeIndex > 8)
                        {
                            break;
                        }
                    }
                }
            }
        }

        const auto adapterStart = std::chrono::steady_clock::now();
        tickDestructibles(
            destructibles,
            stressExecutor,
            physicsDt,
            PxVec3(0.0f, pack.gravity, 0.0f));
        const double adapterTickMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - adapterStart).count();

        // Fracture validates immediately inside the adapter. This periodic full
        // audit catches latent mapping drift without adding an O(nodes) scan to
        // every 60 Hz frame.
        if (step % 60 == 0)
        {
            for (const DestructiblePtr& destructible : destructibles)
            {
                if (!destructible->validateMappings())
                {
                    throw std::runtime_error("PhysX destruction adapter mapping validation failed");
                }
            }
        }
        if (!context.healthy())
        {
            throw std::runtime_error(
                "PhysX GPU scene reported fallback, capacity overflow, CUDA abort, or OOM: "
                + context.errors().lastMessage());
        }

        const AggregateTelemetry current = aggregate(destructibles);
        if (simulationTime < options.settleSeconds)
        {
            splitsBeforeFirstImpact = current.splits;
        }
        const std::uint32_t previousPeakBodies = peaks.peakBodies;
        const std::uint32_t previousPeakAwake = peaks.peakAwakeBodies;
        peaks = current;
        peaks.peakBodies = std::max(previousPeakBodies, current.peakBodies);
        peaks.peakAwakeBodies = std::max(previousPeakAwake, current.peakAwakeBodies);

        double stateExportMilliseconds = 0.0;
        if (!options.statePath.empty() && (step + 1) % snapshotStride == 0)
        {
            const auto exportStart = std::chrono::steady_clock::now();
            if (!writer.writeFrame(
                    writtenFrames++,
                    collectVisualPoses(
                        destructibles,
                        projectiles,
                        nodesPerBuilding,
                        visualBases,
                        false)))
            {
                throw std::runtime_error(writer.error());
            }
            stateExportMilliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - exportStart).count();
        }

        FrameMetrics frame;
        frame.step = step;
        frame.simulationSeconds = simulationTime + physicsDt;
        frame.physicsStepMilliseconds = physicsStepMilliseconds;
        frame.contactCallbackMilliseconds = contactCallbackMilliseconds;
        frame.contactProcessingMilliseconds =
            current.contactProcessingMilliseconds -
            previousFrameTelemetry.contactProcessingMilliseconds;
        frame.gravityMilliseconds =
            current.gravityMilliseconds - previousFrameTelemetry.gravityMilliseconds;
        frame.stressSolveMilliseconds =
            current.stressSolveMilliseconds - previousFrameTelemetry.stressSolveMilliseconds;
        frame.gpuStressSolveMilliseconds =
            current.gpuStressSolveMilliseconds -
            previousFrameTelemetry.gpuStressSolveMilliseconds;
        frame.gpuStressHostToDeviceBytes =
            current.gpuStressHostToDeviceBytes -
            previousFrameTelemetry.gpuStressHostToDeviceBytes;
        frame.gpuStressDeviceToHostBytes =
            current.gpuStressDeviceToHostBytes -
            previousFrameTelemetry.gpuStressDeviceToHostBytes;
        frame.fractureTopologyMilliseconds =
            current.fractureTopologyMilliseconds -
            previousFrameTelemetry.fractureTopologyMilliseconds;
        frame.adapterTickMilliseconds = adapterTickMilliseconds;
        frame.mappingValidationMilliseconds =
            current.mappingValidationMilliseconds -
            previousFrameTelemetry.mappingValidationMilliseconds;
        frame.stateExportMilliseconds = stateExportMilliseconds;
        frame.telemetry = current;
        frame.frameContacts =
            current.contactsProcessed - previousFrameTelemetry.contactsProcessed;
        frame.frameSplits = current.splits - previousFrameTelemetry.splits;
        frame.frameShapesMigrated =
            current.shapesMigrated - previousFrameTelemetry.shapesMigrated;
        frame.projectileImpactContacts = contacts.projectileImpactContacts();
        frame.frameProjectileImpactContacts =
            frame.projectileImpactContacts - previousProjectileImpactContacts;
        frame.projectileImpactImpulse = contacts.projectileImpactImpulse();
        frame.frameProjectileImpactImpulse =
            frame.projectileImpactImpulse - previousProjectileImpactImpulse;
        frame.frameHostMilliseconds =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - frameStart).count();

        runtimeTimings.frameHostMilliseconds.push_back(frame.frameHostMilliseconds);
        runtimeTimings.physicsStepMilliseconds.push_back(frame.physicsStepMilliseconds);
        runtimeTimings.adapterTickMilliseconds.push_back(frame.adapterTickMilliseconds);
        runtimeTimings.gpuStressSolveMilliseconds.push_back(
            frame.gpuStressSolveMilliseconds);
        runtimeTimings.stateExportMilliseconds.push_back(frame.stateExportMilliseconds);
        if (frame.frameHostMilliseconds > 1000.0 / 60.0)
        {
            ++runtimeTimings.budgetMissFrames;
        }
        if (simulationTime >= options.settleSeconds)
        {
            ++runtimeTimings.destructionFrameSamples;
            runtimeTimings.maximumDestructionFrameMilliseconds = std::max(
                runtimeTimings.maximumDestructionFrameMilliseconds,
                frame.frameHostMilliseconds);
            if (frame.frameHostMilliseconds > 1000.0 / 60.0)
            {
                ++runtimeTimings.destructionBudgetMissFrames;
            }
        }
        frameTelemetry.write(frame);
        previousFrameTelemetry = current;
        previousContactCallbackMilliseconds = contacts.callbackMilliseconds();
        previousProjectileImpactContacts = contacts.projectileImpactContacts();
        previousProjectileImpactImpulse = contacts.projectileImpactImpulse();
    }
    const double wallSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    if (!options.statePath.empty() && !writer.finish())
    {
        throw std::runtime_error(writer.error());
    }
    {
        const AggregateTelemetry finalTelemetry = aggregate(destructibles);
        const std::uint32_t peakBodies = peaks.peakBodies;
        const std::uint32_t peakAwake = peaks.peakAwakeBodies;
        peaks = finalTelemetry;
        peaks.peakBodies = std::max(peakBodies, finalTelemetry.peakBodies);
        peaks.peakAwakeBodies = std::max(peakAwake, finalTelemetry.peakAwakeBodies);
    }
    const DestructionDistribution destruction = destructionDistribution(
        destructibles,
        nodesPerBuilding);
    const DestructionMotion motion = destructionMotion(
        destructibles,
        buildingPacks,
        offsets);
    writeMetadata(
        options.metadataPath,
        options,
        pack,
        context,
        peaks,
        runtimeTimings,
        destruction,
        motion,
        contacts.projectileImpactContacts(),
        contacts.projectileImpactImpulse(),
        wallSeconds,
        writtenFrames,
        nodesPerBuilding,
        floorsPerBuilding);
    if (!options.telemetryPath.empty() && options.telemetryPath != options.metadataPath)
    {
        writeMetadata(
            options.telemetryPath,
            options,
            pack,
            context,
            peaks,
            runtimeTimings,
            destruction,
            motion,
            contacts.projectileImpactContacts(),
            contacts.projectileImpactImpulse(),
            wallSeconds,
            writtenFrames,
            nodesPerBuilding,
            floorsPerBuilding);
    }

    bool mappingsValid = true;
    for (const DestructiblePtr& destructible : destructibles)
    {
        mappingsValid = destructible->validateMappings() && mappingsValid;
    }
    const bool selfTestFractured = !options.selfTest || peaks.splits > 0;
    const bool continuityValid =
        peaks.maxPositionDrift <= 1.0e-3f && peaks.maxVelocityDrift <= 1.0e-3f;
    const std::uint32_t locallyDamagedStructures =
        destruction.partiallyFracturedStructures + destruction.heavilyFracturedStructures;
    const std::uint32_t requiredDamagedStructures = std::max<std::uint32_t>(
        1,
        static_cast<std::uint32_t>(std::ceil(destructibles.size() * 0.6)));
    const std::uint32_t requiredFallingStructures = std::max<std::uint32_t>(
        1,
        static_cast<std::uint32_t>(std::ceil(destructibles.size() / 6.0)));
    const std::uint32_t totalChunks = totalAuthoredChunks;
    const bool partialDestructionValid =
        !options.requirePartialDestruction
        || (contacts.projectileImpactContacts() > 0
            && peaks.splits > 0
            && splitsBeforeFirstImpact == 0
            && locallyDamagedStructures >= requiredDamagedStructures
            && destruction.shatteredStructures == 0
            && motion.structuresWithMovedChunks >= requiredDamagedStructures
            && motion.structuresWithFallenChunks >= requiredFallingStructures
            && motion.movedChunks >= requiredDamagedStructures * 2
            && motion.fallenChunks >= requiredFallingStructures
            && motion.farTravelingChunks >= requiredFallingStructures
            && motion.dynamicChunks >= requiredDamagedStructures * 2
            && motion.supportedRemainderChunks >= totalChunks / 2
            && motion.maximumDisplacement >= 2.0f
            && motion.maximumDisplacement
                <= std::max(20.0f, options.durationSeconds * 5.0f)
            && motion.maximumDownwardDisplacement >= 0.5f);
    const bool authoredChunkScaleValid =
        options.requireMinimumAuthoredChunks == 0
        || totalChunks >= options.requireMinimumAuthoredChunks;
    const bool realtimeValid =
        !options.requireRealtime || runtimeTimings.budgetMissFrames == 0;
    std::printf(
        "Blast PhysX mini-city finished: mode=%s gpu=%s buildings=%zu nodes=%zu "
        "bodies=%u awake=%u splits=%llu contacts=%llu drift=(%.3g,%.3g) "
        "damage=(intact:%u partial:%u heavy:%u shattered:%u) "
        "motion=(structures:%u falling:%u chunks:%u fallen:%u far:%u max:%.2f down:%.2f) "
        "impacts=%llu "
        "wall=%.3fs rate=%.2fx\n",
        context.mode() == PhysicsMode::Gpu ? "gpu" : "cpu",
        context.gpuActive() ? "active" : "inactive",
        destructibles.size(),
        static_cast<std::size_t>(totalAuthoredChunks),
        peaks.peakBodies,
        peaks.peakAwakeBodies,
        static_cast<unsigned long long>(peaks.splits),
        static_cast<unsigned long long>(peaks.contactsProcessed),
        peaks.maxPositionDrift,
        peaks.maxVelocityDrift,
        destruction.intactStructures,
        destruction.partiallyFracturedStructures,
        destruction.heavilyFracturedStructures,
        destruction.shatteredStructures,
        motion.structuresWithMovedChunks,
        motion.structuresWithFallenChunks,
        motion.movedChunks,
        motion.fallenChunks,
        motion.farTravelingChunks,
        motion.maximumDisplacement,
        motion.maximumDownwardDisplacement,
        static_cast<unsigned long long>(contacts.projectileImpactContacts()),
        wallSeconds,
        (options.durationSeconds + options.settleSeconds) / std::max(wallSeconds, 1.0e-9));

    if (options.selfTest)
    {
        validateContractResult(
            *buildingPacks.front(),
            *destructibles.front(),
            contractBaseline);
        std::vector<ExtStressPhysXShapeSnapshot> activeShapes(
            nodesPerBuilding.front());
        const std::uint32_t activeShapeCount =
            destructibles.front()->getActiveShapeSnapshots(
                activeShapes.data(),
                static_cast<std::uint32_t>(activeShapes.size()));
        requireContract(
            activeShapeCount > 0,
            "active-shape delta export omitted moving fracture bodies");
        for (std::uint32_t i = 0; i < activeShapeCount; ++i)
        {
            requireContract(
                !activeShapes[i].bodyKinematic,
                "active-shape delta export included a supported body");
        }
    }
    releaseProjectiles(projectiles);
    if (!mappingsValid
        || !selfTestFractured
        || !continuityValid
        || !partialDestructionValid
        || !authoredChunkScaleValid
        || !realtimeValid)
    {
        std::fprintf(
            stderr,
            "contract validation failed: mappings=%s self-test-fractured=%s "
            "continuity=%s dynamic-destruction=%s stable-before-impact=%s "
            "authored-chunk-scale=%s (%u/%u) realtime=%s "
            "(all-misses=%u destruction-misses=%u/%u max=%.3fms) "
            "damaged=%u/%u moved-structures=%u/%u falling-structures=%u/%u "
            "moved=%u fallen=%u far=%u dynamic=%u supported=%u/%u "
            "max-displacement=%.3f max-down=%.3f\n",
            mappingsValid ? "true" : "false",
            selfTestFractured ? "true" : "false",
            continuityValid ? "true" : "false",
            partialDestructionValid ? "true" : "false",
            splitsBeforeFirstImpact == 0 ? "true" : "false",
            authoredChunkScaleValid ? "true" : "false",
            totalChunks,
            options.requireMinimumAuthoredChunks,
            realtimeValid ? "true" : "false",
            runtimeTimings.budgetMissFrames,
            runtimeTimings.destructionBudgetMissFrames,
            runtimeTimings.destructionFrameSamples,
            runtimeTimings.maximumDestructionFrameMilliseconds,
            locallyDamagedStructures,
            requiredDamagedStructures,
            motion.structuresWithMovedChunks,
            requiredDamagedStructures,
            motion.structuresWithFallenChunks,
            requiredFallingStructures,
            motion.movedChunks,
            motion.fallenChunks,
            motion.farTravelingChunks,
            motion.dynamicChunks,
            motion.supportedRemainderChunks,
            totalChunks / 2,
            motion.maximumDisplacement,
            motion.maximumDownwardDisplacement);
        return 1;
    }
    return 0;
}

} // namespace
} // namespace blast_demo

int main(int argc, char** argv)
{
    try
    {
        return blast_demo::run(blast_demo::parseOptions(argc, argv));
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "blast_stress_demo: %s\n", error.what());
        return 1;
    }
}
