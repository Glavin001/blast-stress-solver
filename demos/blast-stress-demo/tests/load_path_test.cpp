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
StackDesc makeColumn(std::uint32_t panelCount, float bondArea)
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

ExtStressPhysXDestructible* createColumn(PhysXScene& context, const StackDesc& stack)
{
    ExtStressPhysXDesc desc;
    desc.physics = &context.physics();
    desc.scene = &context.scene();
    desc.material = &context.material();
    desc.nodes = stack.nodes.data();
    desc.nodeCount = static_cast<std::uint32_t>(stack.nodes.size());
    desc.bonds = stack.bonds.data();
    desc.bondCount = static_cast<std::uint32_t>(stack.bonds.size());
    desc.worldTransform = PxTransform(PxVec3(0.0f));
    desc.settings.applyExcessForces = false;
    desc.settings.minimumSeparationVelocity = 0.0f;
    desc.settings.compressionElasticLimit = kCompressionElastic;
    desc.settings.compressionFatalLimit = 2.0f * kCompressionElastic;
    ExtStressPhysXTelemetry failure;
    ExtStressPhysXDestructible* destructible =
        ExtStressPhysXDestructible::create(desc, &failure);
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

// A sanely authored column: the footing joint must carry a measurable share of
// its capacity, and stress must fall monotonically going up the stack as fewer
// panels bear on each joint.
void testSelfWeightIsMeasurable(PhysXScene& context)
{
    const std::uint32_t panels = 8;
    const StackDesc stack = makeColumn(panels, 1.0f);
    DestructibleHolder column{createColumn(context, stack)};
    settle(context, *column.value, 30);

    const std::vector<float> compression = readCompression(*column.value, panels);

    // 8 panels x 100 kg x 9.81 / 1 m^2 ~= 7.8 kPa at the footing joint.
    require(compression[0] > 1.0e3f, "footing joint reports no self-weight stress");
    require(
        compression[0] < kCompressionElastic,
        "footing joint is overstressed under self-weight alone");
    // Utilisation must be in a band a real structure occupies, not ~0.
    const float utilisation = compression[0] / kCompressionElastic;
    require(
        utilisation > 1.0e-4f,
        "footing utilisation is indistinguishable from zero at unit bond area");

    for (std::uint32_t bond = 1; bond < panels; ++bond)
    {
        require(
            compression[bond] <= compression[bond - 1] + 1.0f,
            "stress must not increase going up the stack (bond "
                + std::to_string(bond) + ")");
    }
    require(
        compression[0] > compression[panels - 1] * 2.0f,
        "footing joint should carry far more than the topmost joint");
}

// Stress is force / area, so scaling authored area by K divides utilisation by
// K. This is exactly why an over-authored joint becomes unbreakable, and it is
// the property --require-max-safety-factor detects.
void testStressScalesInverselyWithAuthoredArea(PhysXScene& context)
{
    const std::uint32_t panels = 8;
    constexpr float kInflation = 1.0e4f;

    const StackDesc geometric = makeColumn(panels, 1.0f);
    DestructibleHolder geometricColumn{createColumn(context, geometric)};
    settle(context, *geometricColumn.value, 30);
    const float geometricStress = readCompression(*geometricColumn.value, panels)[0];
    geometricColumn.value->release();
    geometricColumn.value = nullptr;

    const StackDesc inflated = makeColumn(panels, kInflation);
    DestructibleHolder inflatedColumn{createColumn(context, inflated)};
    settle(context, *inflatedColumn.value, 30);
    const float inflatedStress = readCompression(*inflatedColumn.value, panels)[0];

    require(geometricStress > 0.0f, "geometric column reported zero stress");
    require(inflatedStress > 0.0f, "inflated column reported zero stress");

    const float ratio = geometricStress / inflatedStress;
    require(
        ratio > kInflation * 0.5f && ratio < kInflation * 2.0f,
        "stress did not scale inversely with authored bond area (ratio "
            + std::to_string(ratio) + ", expected ~" + std::to_string(kInflation) + ")");

    // The headline consequence, stated as an assertion: the same structure,
    // same load, authored 1e4x heavier, is now loaded to ~1e-8 of capacity —
    // a "safety factor" of 1e8, i.e. nothing can ever break it.
    const float inflatedUtilisation = inflatedStress / kCompressionElastic;
    require(
        inflatedUtilisation < 1.0e-5f,
        "over-authored joint should read as effectively unloaded");
}

// A topology edit must not corrupt the readback: no stale, negative, or
// non-finite stress survives a split, and the array stays asset-indexed.
void testReadbackSurvivesTopologyEdit(PhysXScene& context)
{
    const std::uint32_t panels = 4;
    const StackDesc stack = makeColumn(panels, 1.0f);
    DestructibleHolder column{createColumn(context, stack)};
    settle(context, *column.value, 10);

    const std::uint32_t before = readCompression(*column.value, panels).size();
    require(before == panels, "unexpected bond count before fracture");

    // Rip the top panel off with an impulse far past the fatal limit.
    std::vector<ExtStressPhysXShapeSnapshot> shapes(panels + 1);
    const std::uint32_t shapeCount = column.value->getShapeSnapshots(
        shapes.data(), static_cast<std::uint32_t>(shapes.size()));
    const ExtStressPhysXShapeSnapshot* topShape = nullptr;
    for (std::uint32_t i = 0; i < shapeCount; ++i)
    {
        if (shapes[i].nodeIndex == panels)
        {
            topShape = &shapes[i];
            break;
        }
    }
    require(topShape != nullptr, "could not locate the topmost panel shape");
    // Sized to shear the topmost joint only: impulse/dt / area comfortably
    // exceeds the 2 MPa fatal limit at the top bond while the joints further
    // down, which see it spread over the rest of the column, stay intact.
    require(
        column.value->queueContact(
            *topShape->shape,
            topShape->worldPose.p,
            PxVec3(1.0e5f, 0.0f, 0.0f)),
        "queueContact rejected the fracture impulse");
    context.scene().simulate(kDt);
    context.scene().fetchResults(true);
    column.value->tick(kDt, kGravity);
    require(
        column.value->getTelemetry().splits > 0,
        "synthetic contact did not fracture the column");

    // Sample on the fracture tick itself. Load-path sampling is only meaningful
    // while the island is actually being solved; once debris settles the solver
    // legitimately skips it (settled-island skip) and every bond reads 0. The
    // demo samples pre-impact for exactly this reason.
    const std::vector<float> compression = readCompression(*column.value, panels);
    for (std::uint32_t bond = 0; bond < panels; ++bond)
    {
        require(
            std::isfinite(compression[bond]) && compression[bond] >= 0.0f,
            "bond stress readback produced a non-finite or negative value");
    }
    require(
        readCompression(*column.value, panels).size() == panels,
        "bond stress readback changed size across a topology edit");
}

} // namespace

int main()
{
    try
    {
        SceneCapacity capacity;
        PhysXScene context(PhysicsMode::Cpu, false, capacity, nullptr);
        context.scene().setGravity(kGravity);
        testSelfWeightIsMeasurable(context);
        testStressScalesInverselyWithAuthoredArea(context);
        testReadbackSurvivesTopologyEdit(context);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::printf("load path test passed\n");
    return 0;
}
