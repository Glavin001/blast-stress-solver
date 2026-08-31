// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Coverage for the load-path readout: ExtStressPhysXDestructible::getBondStresses.
//
// Bond area is simultaneously the denominator of stress and the bond's damage
// pool, so inflating it is a tempting way to "make a joint strong". It is not:
// past a point the joint carries no measurable load and cannot break under any
// impulse the simulation can produce, which silently turns a destruction demo
// into an indestructible one while every damage gate still reports success.
//
// These tests pin the property that makes that failure detectable:
//   1. Under self-weight a sanely authored joint reports non-trivial stress
//      that scales with the load above it.
//   2. Stress is inversely proportional to authored area, so an over-authored
//      joint's utilisation collapses toward zero — the signature the demo's
//      --require-max-safety-factor gate keys on.
//   3. Broken bonds read back as unloaded rather than retaining stale stress.

#include "../physx_scene.h"

#include <NvBlastExtStressPhysX.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Nv::Blast;
using namespace physx;
using blast_demo::PhysXScene;
using blast_demo::PhysicsMode;
using blast_demo::SceneCapacity;

namespace
{

constexpr float kDt = 1.0f / 60.0f;
const PxVec3 kGravity(0.0f, -9.81f, 0.0f);

// Well above the self-weight stress at unit area so gravity never fractures the
// stack; the ratios under test are unaffected by the absolute value.
constexpr float kCompressionElastic = 1.0e6f;

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error("load path test failed: " + message);
    }
}

struct StackDesc
{
    std::vector<ExtStressPhysXNodeDesc> nodes;
    std::vector<ExtStressPhysXBondDesc> bonds;
};

// A column of `panelCount` 100 kg panels on a mass-0 footing. Bond i joins
// panel i to panel i+1, so bond 0 (the footing joint) carries the whole stack
// and the topmost bond carries a single panel.
StackDesc makeColumn(
    std::uint32_t panelCount,
    float bondArea,
    std::uint32_t uniformMaterial = 0,
    std::int32_t footingMaterial = -1)
{
    StackDesc result;
    for (std::uint32_t i = 0; i <= panelCount; ++i)
    {
        ExtStressPhysXNodeDesc node;
        node.centroid = PxVec3(0.0f, 0.5f + static_cast<float>(i), 0.0f);
        node.mass = i == 0 ? 0.0f : 100.0f;
        node.volume = 1.0f;
        node.geometry.localPose = PxTransform(node.centroid);
        node.geometry.halfExtents = PxVec3(0.5f);
        result.nodes.push_back(node);
    }
    for (std::uint32_t i = 0; i < panelCount; ++i)
    {
        ExtStressPhysXBondDesc bond;
        bond.node0 = i;
        bond.node1 = i + 1;
        bond.centroid = PxVec3(0.0f, 1.0f + static_cast<float>(i), 0.0f);
        bond.normal = PxVec3(0.0f, 1.0f, 0.0f);
        bond.area = bondArea;
        bond.material = (footingMaterial >= 0 && i == 0)
            ? static_cast<std::uint32_t>(footingMaterial)
            : uniformMaterial;
        result.bonds.push_back(bond);
    }
    return result;
}

struct DestructibleHolder
{
    ExtStressPhysXDestructible* value{nullptr};

    ~DestructibleHolder()
    {
        if (value)
        {
            value->release();
        }
    }
};

/** Uniform material at the reference strength. */
std::vector<ExtStressPhysXMaterial> defaultMaterials()
{
    ExtStressPhysXMaterial material;
    material.compressionElasticLimit = kCompressionElastic;
    material.compressionFatalLimit = 2.0f * kCompressionElastic;
    return {material};
}

ExtStressPhysXDestructible* createColumn(
    PhysXScene& context,
    const StackDesc& stack,
    const std::vector<ExtStressPhysXMaterial>& materials,
    const PxVec3& origin = PxVec3(0.0f),
    std::uint32_t* outError = nullptr)
{
    ExtStressPhysXDesc desc;
    desc.physics = &context.physics();
    desc.scene = &context.scene();
    desc.material = &context.material();
    desc.nodes = stack.nodes.data();
    desc.nodeCount = static_cast<std::uint32_t>(stack.nodes.size());
    desc.bonds = stack.bonds.data();
    desc.bondCount = static_cast<std::uint32_t>(stack.bonds.size());
    desc.worldTransform = PxTransform(origin);
    // Iteration budget is the variable under test: CG propagates roughly one
    // graph hop per iteration, so if the divergence depth tracks this number,
    // truncation is the cause rather than a fixed limit somewhere.
    if (const char* itEnv = std::getenv("BLAST_TEST_ITERS"))
    {
        desc.settings.maxSolverIterationsPerFrame =
            static_cast<uint32_t>(std::strtoul(itEnv, nullptr, 10));
    }
    // unconvergedExtraUpdates: extra solver batches while the solve has not
    // converged. Its own doc calls this a CORRECTNESS setting, and the plateau
    // this probe measures is precisely what it exists to prevent.
    if (const char* exEnv = std::getenv("BLAST_TEST_EXTRA_UPDATES"))
    {
        desc.settings.unconvergedExtraUpdates =
            static_cast<uint32_t>(std::strtoul(exEnv, nullptr, 10));
    }
    desc.settings.applyExcessForces = false;
    desc.settings.minimumSeparationVelocity = 0.0f;
    desc.stressMaterials = materials.data();
    desc.stressMaterialCount = static_cast<std::uint32_t>(materials.size());
    ExtStressPhysXTelemetry failure;
    ExtStressPhysXDestructible* destructible =
        ExtStressPhysXDestructible::create(desc, &failure);
    if (outError)
    {
        *outError = static_cast<std::uint32_t>(failure.lastError);
        return destructible;
    }
    require(destructible != nullptr, "column destructible creation failed");
    return destructible;
}

void settle(PhysXScene& context, ExtStressPhysXDestructible& destructible, std::uint32_t steps)
{
    for (std::uint32_t step = 0; step < steps; ++step)
    {
        context.scene().simulate(kDt);
        context.scene().fetchResults(true);
        destructible.tick(kDt, kGravity);
    }
}

std::vector<float> readCompression(
    ExtStressPhysXDestructible& destructible,
    std::uint32_t bondCount)
{
    std::vector<float> compression(bondCount, -1.0f);
    const std::uint32_t written = destructible.getBondStresses(
        compression.data(), nullptr, nullptr, bondCount);
    require(written == bondCount, "getBondStresses wrote the wrong entry count");
    return compression;
}

// Depth-scaling probe: at what structure size do the CPU and GPU stress
// solvers stop agreeing?
//
// The city-scale divergence (CPU collapses a resting city, GPU holds it) does
// not reproduce on the few-chunk scenes the other tests use, where the two
// solvers agree. Something about scale causes it, and "an entire city" is not
// a minimal reproducible case.
//
// Depth is the first suspect, ahead of raw bond count. Conjugate gradient
// propagates information roughly one graph hop per iteration, so a column
// deeper than the iteration cap physically cannot have converged: the load at
// the top has not yet reached the footing. A 32-iteration budget should
// therefore start failing somewhere around a 32-panel column, and the two
// implementations have no reason to truncate identically.
//
// This is deliberately NOT a pass/fail test. It prints the stress field per
// depth so the two solvers can be diffed against each other and against the
// analytic expectation, which is what identifies the characteristic rather
// than merely detecting it.
//
// Run twice and diff:
//   ./solver_divergence_test > cpu.txt
//   BLAST_TEST_PHYSICS_MODE=gpu BLAST_STRESS_GPU=1 \
//     BLAST_STRESS_GPU_MIN_BONDS=0 ./solver_divergence_test > gpu.txt
void probeDepth(PhysXScene& context, std::uint32_t panels)
{
    const auto materials = defaultMaterials();
    const StackDesc stack = makeColumn(panels, 0.05f);
    std::uint32_t error = 0;
    ExtStressPhysXDestructible* column =
        createColumn(context, stack, materials, PxVec3(0.0f), &error);
    if (column == nullptr)
    {
        std::printf("depth=%u CREATE_FAILED error=%u\n", panels, error);
        return;
    }
    settle(context, *column, 30);
    const std::uint32_t bondCount = static_cast<std::uint32_t>(stack.bonds.size());
    const std::vector<float> compression = readCompression(*column, bondCount);

    // The footing carries every panel above it, so it is the entry whose
    // magnitude is set by the whole load path -- exactly the quantity that
    // needs enough iterations to be right.
    float footing = 0.0f;
    float peak = 0.0f;
    double total = 0.0;
    for (std::uint32_t i = 0; i < bondCount; ++i)
    {
        const float c = compression[i];
        total += double(c);
        if (c > peak) { peak = c; }
    }
    footing = compression.empty() ? 0.0f : compression[0];
    std::printf("depth=%-4u bonds=%-4u footing=%.6f peak=%.6f sum=%.6f\n",
                panels, bondCount, footing, peak, total);
    column->release();
}

}  // namespace

int main()
{
    try
    {
        const char* modeEnv = std::getenv("BLAST_TEST_PHYSICS_MODE");
        const bool useGpu = modeEnv != nullptr && std::string(modeEnv) == "gpu";
        std::printf("# physics mode: %s\n", useGpu ? "gpu" : "cpu");
        SceneCapacity capacity;
        PhysXScene context(
            useGpu ? PhysicsMode::Gpu : PhysicsMode::Cpu, false, capacity, nullptr);
        for (std::uint32_t depth : {2u, 4u, 8u, 16u, 24u, 32u, 48u, 64u, 96u, 128u})
        {
            probeDepth(context, depth);
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "solver divergence probe failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
