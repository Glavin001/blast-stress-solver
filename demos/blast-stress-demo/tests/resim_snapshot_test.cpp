// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Unit coverage for the fracture-frame resimulation primitives (engine
// contract §2.8) and the library frame stepper: exact snapshot round-trips,
// provenance re-derivation of fracture-created children, tick-phase guards,
// and single-effective-step semantics for non-adapter scene bodies across a
// rollback.

#include "../physx_scene.h"

#include <NvBlastExtStressPhysX.h>
#include <NvBlastExtStressPhysXResim.h>

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

void require(bool condition, const char* message)
{
    if (!condition)
    {
        throw std::runtime_error(std::string("resim snapshot test failed: ") + message);
    }
}

bool nearlyEqual(const PxVec3& a, const PxVec3& b, float tolerance)
{
    return (a - b).magnitude() <= tolerance;
}

bool nearlyEqual(const PxTransform& a, const PxTransform& b, float tolerance)
{
    const float dot = std::fabs(a.q.dot(b.q));
    return nearlyEqual(a.p, b.p, tolerance) && dot >= 1.0f - tolerance;
}

// A vertical stack of panels on one mass-0 support base: a single kinematic
// body until fractured, with peel-allowed panel masses.
struct StackDesc
{
    std::vector<ExtStressPhysXNodeDesc> nodes;
    std::vector<ExtStressPhysXBondDesc> bonds;
};

StackDesc makeStack(std::uint32_t panelCount)
{
    StackDesc result;
    for (std::uint32_t i = 0; i <= panelCount; ++i)
    {
        ExtStressPhysXNodeDesc node;
        node.centroid = PxVec3(0.0f, 0.5f + static_cast<float>(i), 0.0f);
        node.mass = i == 0 ? 0.0f : 10.0f;
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
        bond.area = 1.0f;
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

ExtStressPhysXDestructible* createStack(
    PhysXScene& context,
    const StackDesc& stack,
    const PxTransform& worldTransform)
{
    ExtStressPhysXDesc desc;
    desc.physics = &context.physics();
    desc.scene = &context.scene();
    desc.material = &context.material();
    desc.nodes = stack.nodes.data();
    desc.nodeCount = static_cast<std::uint32_t>(stack.nodes.size());
    desc.bonds = stack.bonds.data();
    desc.bondCount = static_cast<std::uint32_t>(stack.bonds.size());
    desc.worldTransform = worldTransform;
    desc.settings.applyExcessForces = false;
    desc.settings.minimumSeparationVelocity = 0.0f;
    // Far above the ~500 N self-weight load, far below the synthetic
    // fracture impulse: gravity settles, the queued contact splits.
    ExtStressPhysXMaterial material;
    material.compressionElasticLimit = 1.0e6f;
    material.compressionFatalLimit = 2.0e6f;
    desc.stressMaterials = &material;
    desc.stressMaterialCount = 1;
    ExtStressPhysXTelemetry failure;
    ExtStressPhysXDestructible* destructible =
        ExtStressPhysXDestructible::create(desc, &failure);
    require(destructible != nullptr, "stack destructible creation failed");
    return destructible;
}

ExtStressPhysXShapeSnapshot findNodeShape(
    ExtStressPhysXDestructible& destructible,
    std::uint32_t nodeIndex)
{
    std::vector<ExtStressPhysXShapeSnapshot> shapes(64);
    const std::uint32_t count = destructible.getShapeSnapshots(
        shapes.data(),
        static_cast<std::uint32_t>(shapes.size()));
    for (std::uint32_t i = 0; i < count; ++i)
    {
        if (shapes[i].nodeIndex == nodeIndex)
        {
            return shapes[i];
        }
    }
    return ExtStressPhysXShapeSnapshot{};
}

std::uint32_t bodyCount(ExtStressPhysXDestructible& destructible)
{
    std::vector<ExtStressPhysXBodySnapshot> bodies(64);
    return destructible.getBodySnapshots(
        bodies.data(),
        static_cast<std::uint32_t>(bodies.size()));
}

// Queue an impulse large enough to overstress the top bonds and split panels
// off the kinematic base, mirroring the demo's synthetic contract contact.
void queueFracturingContact(ExtStressPhysXDestructible& destructible, std::uint32_t nodeIndex)
{
    const ExtStressPhysXShapeSnapshot shape = findNodeShape(destructible, nodeIndex);
    require(shape.shape != nullptr, "node shape lookup failed");
    require(
        destructible.queueContact(
            *shape.shape,
            shape.worldPose.p,
            PxVec3(2.0e8f, 5.0e8f, -3.0e8f)),
        "queueContact rejected the fracture impulse");
}

void testRoundTripAndProvenance(PhysXScene& context)
{
    const StackDesc stack = makeStack(5);
    DestructibleHolder holder;
    holder.value = createStack(context, stack, PxTransform(PxVec3(0.0f)));
    ExtStressPhysXDestructible& destructible = *holder.value;

    require(bodyCount(destructible) == 1, "stack must start as one body");
    require(
        destructible.captureResimulationSnapshot() == 1,
        "initial capture must record one body");

    // Fracture against the held capture: children must be recorded with
    // provenance and re-derived from the parent's restored state.
    std::vector<ExtStressPhysXBodySnapshot> before(64);
    const std::uint32_t beforeCount = destructible.getBodySnapshots(before.data(), 64);
    require(beforeCount == 1, "pre-fracture snapshot export mismatch");
    const PxTransform parentPoseAtCapture = before[0].globalPose;

    queueFracturingContact(destructible, 5);
    require(destructible.tick(kDt, kGravity), "fracture tick failed");
    const std::uint32_t fracturedBodies = bodyCount(destructible);
    require(fracturedBodies > 1, "the contact must split the stack");
    require(
        destructible.getTelemetry().splits > 0,
        "split telemetry must record the fracture");

    // Let the children drift for a step so restore has real work to undo.
    context.scene().simulate(kDt);
    require(context.scene().fetchResults(true), "post-fracture step failed");

    require(destructible.restoreResimulationSnapshot(), "restore after fracture failed");
    require(
        bodyCount(destructible) == fracturedBodies,
        "restore must keep the fractured topology");

    std::vector<ExtStressPhysXBodySnapshot> after(64);
    const std::uint32_t afterCount = destructible.getBodySnapshots(after.data(), 64);
    const ExtStressPhysXTelemetry& telemetry = destructible.getTelemetry();
    require(
        telemetry.resimulationBodiesRederived + telemetry.resimulationBodiesRestored > 0,
        "restore telemetry must record work");
    for (std::uint32_t i = 0; i < afterCount; ++i)
    {
        if (after[i].bodyId == before[0].bodyId)
        {
            require(
                nearlyEqual(after[i].globalPose, parentPoseAtCapture, 1.0e-5f),
                "reused parent must return to its captured pose");
        }
        else
        {
            // Children of the kinematic parent re-derive to zero velocity at
            // their creation pose relative to the (unmoved) parent.
            require(
                nearlyEqual(after[i].linearVelocity, PxVec3(0.0f), 1.0e-5f),
                "kinematic-parent children must re-derive to zero velocity");
        }
    }

    // Exact round-trip on the now-mixed kinematic/dynamic body set.
    require(
        destructible.captureResimulationSnapshot() == fracturedBodies,
        "post-fracture capture must record every body");
    std::vector<ExtStressPhysXBodySnapshot> captured(64);
    const std::uint32_t capturedCount = destructible.getBodySnapshots(captured.data(), 64);
    for (std::uint32_t i = 0; i < capturedCount; ++i)
    {
        PxRigidDynamic& body = *captured[i].body;
        body.setGlobalPose(
            PxTransform(
                captured[i].globalPose.p + PxVec3(1.0f, 2.0f, 3.0f),
                captured[i].globalPose.q),
            false);
        if (!captured[i].kinematic)
        {
            body.setLinearVelocity(PxVec3(4.0f, 5.0f, 6.0f), false);
            body.setAngularVelocity(PxVec3(0.5f, 0.25f, 0.125f), false);
        }
    }
    require(destructible.restoreResimulationSnapshot(), "round-trip restore failed");
    std::vector<ExtStressPhysXBodySnapshot> restored(64);
    const std::uint32_t restoredCount = destructible.getBodySnapshots(restored.data(), 64);
    require(restoredCount == capturedCount, "round-trip body count changed");
    for (std::uint32_t i = 0; i < restoredCount; ++i)
    {
        require(
            nearlyEqual(restored[i].globalPose, captured[i].globalPose, 1.0e-5f),
            "round-trip pose mismatch");
        if (!captured[i].kinematic)
        {
            require(
                nearlyEqual(restored[i].linearVelocity, captured[i].linearVelocity, 1.0e-5f),
                "round-trip linear velocity mismatch");
            require(
                nearlyEqual(restored[i].angularVelocity, captured[i].angularVelocity, 1.0e-5f),
                "round-trip angular velocity mismatch");
        }
    }

    // Phase guard: capture and restore are Idle-phase-only.
    require(destructible.beginTick(kDt, kGravity), "guard beginTick failed");
    require(
        destructible.captureResimulationSnapshot() == 0,
        "capture must reject the Prepared phase");
    require(
        !destructible.restoreResimulationSnapshot(),
        "restore must reject the Prepared phase");
    require(destructible.solveTick(), "guard solveTick failed");
    require(destructible.endTick(), "guard endTick failed");
}

void testFrameStepper(PhysXScene& context)
{
    const StackDesc stack = makeStack(5);
    DestructibleHolder holder;
    holder.value = createStack(context, stack, PxTransform(PxVec3(20.0f, 0.0f, 0.0f)));
    ExtStressPhysXDestructible& destructible = *holder.value;
    ExtStressPhysXDestructible* destructibles[] = {&destructible};

    // A free falling sphere, far from everything: across a rollback it must
    // advance exactly one effective step, never two.
    PxRigidDynamic* sphere =
        context.physics().createRigidDynamic(PxTransform(PxVec3(40.0f, 30.0f, 0.0f)));
    require(sphere != nullptr, "sphere creation failed");
    PxShape* sphereShape = context.physics().createShape(
        PxSphereGeometry(0.5f),
        context.material(),
        false);
    require(
        sphereShape != nullptr && sphere->attachShape(*sphereShape),
        "sphere shape creation failed");
    sphere->setMass(10.0f);
    context.scene().addActor(*sphere);

    ExtStressPhysXFrameStepper* stepper =
        ExtStressPhysXFrameStepper::create(context.scene());
    require(stepper != nullptr, "frame stepper creation failed");

    ExtStressPhysXResimOptions options;
    ExtStressPhysXFrameStats stats;

    // Quiet frame with rollback armed: no fracture, no resim pass.
    options.maxPasses = 1;
    require(
        stepper->stepFrame(kDt, kGravity, destructibles, 1, options, nullptr, &stats),
        "quiet stepFrame failed");
    require(stats.resimPasses == 0, "quiet frame must not resimulate");
    require(stats.splits == 0, "quiet frame must not split");

    // Fracture frame: exactly one rollback + re-step, and the sphere still
    // advances a single effective step.
    const float sphereVyBefore = sphere->getLinearVelocity().y;
    queueFracturingContact(destructible, 5);
    require(
        stepper->stepFrame(kDt, kGravity, destructibles, 1, options, nullptr, &stats),
        "fracture stepFrame failed");
    require(stats.resimPasses == 1, "fracture frame must run one resim pass");
    require(stats.splits > 0, "fracture frame must record splits");
    require(bodyCount(destructible) > 1, "fracture frame must split the stack");
    const float stepDeltaVy = kGravity.y * kDt;
    const float sphereVyAfter = sphere->getLinearVelocity().y;
    require(
        std::fabs(sphereVyAfter - (sphereVyBefore + stepDeltaVy))
            < 0.25f * std::fabs(stepDeltaVy),
        "rollback must rewind non-adapter bodies to one effective step");

    // maxPasses = 0 must behave like the raw simulate/tick sequence.
    options.maxPasses = 0;
    queueFracturingContact(destructible, 4);
    require(
        stepper->stepFrame(kDt, kGravity, destructibles, 1, options, nullptr, &stats),
        "resim-disabled stepFrame failed");
    require(stats.resimPasses == 0, "maxPasses=0 must never resimulate");

    stepper->release();
    sphere->release();
    sphereShape->release();
}

} // namespace

int main()
{
    try
    {
        SceneCapacity capacity;
        PhysXScene context(PhysicsMode::Cpu, false, capacity, nullptr);
        context.scene().setGravity(kGravity);
        testRoundTripAndProvenance(context);
        testFrameStepper(context);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::printf("resim snapshot test passed\n");
    return 0;
}
