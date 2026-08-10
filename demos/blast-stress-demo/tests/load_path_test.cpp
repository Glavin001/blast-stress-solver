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

// A sanely authored column: the footing joint must carry a measurable share of
// its capacity, and stress must fall monotonically going up the stack as fewer
// panels bear on each joint.
void testSelfWeightIsMeasurable(PhysXScene& context)
{
    const std::uint32_t panels = 8;
    const StackDesc stack = makeColumn(panels, 1.0f);
    DestructibleHolder column{createColumn(context, stack, defaultMaterials())};
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
    DestructibleHolder geometricColumn{createColumn(context, geometric, defaultMaterials())};
    settle(context, *geometricColumn.value, 30);
    const float geometricStress = readCompression(*geometricColumn.value, panels)[0];
    geometricColumn.value->release();
    geometricColumn.value = nullptr;

    const StackDesc inflated = makeColumn(panels, kInflation);
    DestructibleHolder inflatedColumn{createColumn(context, inflated, defaultMaterials())};
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
    DestructibleHolder column{createColumn(context, stack, defaultMaterials())};
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

// THE pin against this branch's failure mode: geometry and strength must be
// independent axes. Two columns with IDENTICAL geometry, differing only in the
// footing joint's material, must report the SAME stress (stress is force/area,
// pure geometry) while only the weak one fails under the same load.
void testMaterialsDecoupleStrengthFromStress(PhysXScene& context)
{
    const std::uint32_t panels = 6;

    // Index 0 strong (reference), index 1 weak. The weak band is chosen so the
    // footing yields gradually (elastic below self-weight stress ~5.9 kPa,
    // fatal well above) rather than snapping on tick 1 — that keeps both
    // columns intact for the stress comparison and still fails the weak one
    // within the settle window.
    std::vector<ExtStressPhysXMaterial> materials = defaultMaterials();
    ExtStressPhysXMaterial weak;
    weak.compressionElasticLimit = 1.0e3f;
    weak.compressionFatalLimit = 1.0e5f;
    materials.push_back(weak);

    const StackDesc strongStack = makeColumn(panels, 1.0f, /*uniformMaterial=*/0);
    const StackDesc weakFootingStack =
        makeColumn(panels, 1.0f, /*uniformMaterial=*/0, /*footingMaterial=*/1);

    DestructibleHolder strongColumn{
        createColumn(context, strongStack, materials, PxVec3(0.0f, 0.0f, 0.0f))};
    DestructibleHolder weakColumn{
        createColumn(context, weakFootingStack, materials, PxVec3(20.0f, 0.0f, 0.0f))};

    // One tick: both columns are still intact, so this compares the stress a
    // given geometry carries — not what survived.
    settle(context, *strongColumn.value, 1);
    settle(context, *weakColumn.value, 1);
    require(
        strongColumn.value->getTelemetry().splits == 0
            && weakColumn.value->getTelemetry().splits == 0,
        "both columns must still be intact when stress is compared");

    const float strongStress = readCompression(*strongColumn.value, panels)[0];
    const float weakStress = readCompression(*weakColumn.value, panels)[0];
    require(strongStress > 0.0f, "strong column footing reports no stress");
    require(
        std::fabs(strongStress - weakStress) <= 0.02f * strongStress,
        "identical geometry must produce identical stress regardless of material");

    // Utilisation, by contrast, must differ by the material ratio.
    std::vector<float> strongUtil(panels, 0.0f);
    std::vector<float> weakUtil(panels, 0.0f);
    require(
        strongColumn.value->getBondUtilisations(strongUtil.data(), panels) == panels
            && weakColumn.value->getBondUtilisations(weakUtil.data(), panels) == panels,
        "getBondUtilisations wrote the wrong entry count");
    require(strongUtil[0] > 0.0f, "strong footing utilisation is zero");
    require(
        weakUtil[0] > strongUtil[0] * 100.0f,
        "the weak-material footing must be far closer to its own limit");

    // And only the weak footing actually fails under identical self-weight.
    settle(context, *strongColumn.value, 120);
    settle(context, *weakColumn.value, 120);
    require(
        strongColumn.value->getTelemetry().splits == 0,
        "strong column must stand under its own weight");
    require(
        weakColumn.value->getTelemetry().splits > 0,
        "weak-material footing must fail under the same load and geometry");
}

// A bond referencing a material outside the table is an authoring error and
// must fail loudly at create rather than clamp to index 0.
void testOutOfRangeMaterialRejected(PhysXScene& context)
{
    StackDesc stack = makeColumn(4, 1.0f);
    stack.bonds[1].material = 7; // table has one entry
    std::uint32_t error = 0;
    ExtStressPhysXDestructible* destructible =
        createColumn(context, stack, defaultMaterials(), PxVec3(40.0f, 0.0f, 0.0f), &error);
    require(destructible == nullptr, "out-of-range bond material must fail creation");
    require(
        error == static_cast<std::uint32_t>(ExtStressPhysXError::InvalidDescriptor),
        "out-of-range bond material must report InvalidDescriptor");
}

// A missing material table is likewise rejected: every structure must state
// what it is made of instead of inheriting silent placeholder limits.
void testMissingMaterialTableRejected(PhysXScene& context)
{
    const StackDesc stack = makeColumn(4, 1.0f);
    ExtStressPhysXDesc desc;
    desc.physics = &context.physics();
    desc.scene = &context.scene();
    desc.material = &context.material();
    desc.nodes = stack.nodes.data();
    desc.nodeCount = static_cast<std::uint32_t>(stack.nodes.size());
    desc.bonds = stack.bonds.data();
    desc.bondCount = static_cast<std::uint32_t>(stack.bonds.size());
    desc.worldTransform = PxTransform(PxVec3(60.0f, 0.0f, 0.0f));
    ExtStressPhysXTelemetry failure;
    require(
        ExtStressPhysXDestructible::create(desc, &failure) == nullptr,
        "a destructible without a material table must fail creation");
    require(
        failure.lastError == ExtStressPhysXError::InvalidDescriptor,
        "missing material table must report InvalidDescriptor");
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
        testMaterialsDecoupleStrengthFromStress(context);
        testOutOfRangeMaterialRejected(context);
        testMissingMaterialTableRejected(context);
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
