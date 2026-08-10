// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Phase C gate: island-exact scoped resim vs full-scene restore must stay
// output-faithful on an isolated fracture (sub-mm) and a multi-impact cascade
// (sub-cm). Negative control: different impact → large delta.

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
        throw std::runtime_error(std::string("resim scoped cascade test failed: ") + message);
    }
}

struct StackDesc
{
    std::vector<ExtStressPhysXNodeDesc> nodes;
    std::vector<ExtStressPhysXBondDesc> bonds;
};

StackDesc makeStack(std::uint32_t panels)
{
    StackDesc result;
    for (std::uint32_t i = 0; i <= panels; ++i)
    {
        ExtStressPhysXNodeDesc node;
        node.centroid = PxVec3(0.0f, 0.5f + static_cast<float>(i), 0.0f);
        node.mass = i == 0 ? 0.0f : 12.0f;
        node.volume = 1.0f;
        node.geometry.localPose = PxTransform(node.centroid);
        node.geometry.halfExtents = PxVec3(0.5f);
        result.nodes.push_back(node);
    }
    for (std::uint32_t i = 0; i < panels; ++i)
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

struct Holder
{
    ExtStressPhysXDestructible* value{nullptr};
    ~Holder()
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
    const PxTransform& pose,
    bool baseStepSleep)
{
    ExtStressPhysXDesc desc;
    desc.physics = &context.physics();
    desc.scene = &context.scene();
    desc.material = &context.material();
    desc.nodes = stack.nodes.data();
    desc.nodeCount = static_cast<std::uint32_t>(stack.nodes.size());
    desc.bonds = stack.bonds.data();
    desc.bondCount = static_cast<std::uint32_t>(stack.bonds.size());
    desc.worldTransform = pose;
    desc.settings.applyExcessForces = false;
    desc.settings.minimumSeparationVelocity = 0.0f;
    desc.settings.protectSupportBonds = true;
    desc.settings.supportPeelMaxMass = 1000.0f;
    desc.settings.fatalizeImpactContactBonds = true;
    desc.settings.idleSkip = true;
    desc.settings.baseStepSleep = baseStepSleep;
    desc.settings.settledLinearSpeed = 0.15f;
    desc.settings.settledAngularSpeed = 0.15f;
    ExtStressPhysXTelemetry failure;
    auto* created = ExtStressPhysXDestructible::create(desc, &failure);
    require(created != nullptr, "destructible create failed");
    return created;
}

struct ImpactContacts final : public PxSimulationEventCallback
{
    ExtStressPhysXDestructible* dest{nullptr};
    ExtStressPhysXFrameStepper* stepper{nullptr};
    float forceScale{8.0f};

    void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
    void onWake(PxActor**, PxU32) override {}
    void onSleep(PxActor**, PxU32) override {}
    void onTrigger(PxTriggerPair*, PxU32) override {}
    void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) override {}
    void onContact(const PxContactPairHeader&, const PxContactPair* pairs, PxU32 nbPairs) override
    {
        std::vector<PxContactPairPoint> points;
        for (PxU32 i = 0; i < nbPairs; ++i)
        {
            const PxContactPair& pair = pairs[i];
            if (stepper && pair.shapes[0] && pair.shapes[1])
            {
                stepper->recordDynamicContactPair(
                    pair.shapes[0]->getActor(), pair.shapes[1]->getActor());
            }
            if (!pair.contactCount || !dest)
            {
                continue;
            }
            points.resize(pair.contactCount);
            const PxU32 written = pair.extractContacts(points.data(), pair.contactCount);
            for (PxU32 p = 0; p < written; ++p)
            {
                for (PxU32 s = 0; s < 2; ++s)
                {
                    PxShape* shape = pair.shapes[s];
                    if (!shape)
                    {
                        continue;
                    }
                    const PxVec3 impulse =
                        (s == 0 ? points[p].impulse : -points[p].impulse) * forceScale;
                    dest->queueContact(*shape, points[p].position, impulse);
                }
            }
        }
    }
};

struct PoseSample
{
    ExtStressPhysXId bodyId{0};
    PxVec3 position{0.0f};
};

std::vector<PoseSample> capturePoses(ExtStressPhysXDestructible& dest)
{
    std::vector<ExtStressPhysXBodySnapshot> snaps(256);
    const std::uint32_t count = dest.getBodySnapshots(snaps.data(), snaps.size());
    std::vector<PoseSample> out;
    out.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i)
    {
        PoseSample sample;
        sample.bodyId = snaps[i].bodyId;
        sample.position = snaps[i].globalPose.p;
        out.push_back(sample);
    }
    return out;
}

float maxPoseDelta(const std::vector<PoseSample>& a, const std::vector<PoseSample>& b)
{
    float maxDelta = 0.0f;
    for (const PoseSample& left : a)
    {
        for (const PoseSample& right : b)
        {
            if (left.bodyId != right.bodyId)
            {
                continue;
            }
            maxDelta = std::max(maxDelta, (left.position - right.position).magnitude());
        }
    }
    return maxDelta;
}

struct RunResult
{
    std::vector<PoseSample> poses;
    std::uint64_t splits{0};
    std::uint32_t frozenPeak{0};
};

RunResult runScenario(bool scoped, bool secondImpactOffset, bool baseStepSleep)
{
    ImpactContacts contacts;
    SceneCapacity capacity;
    PhysXScene context(PhysicsMode::Cpu, false, capacity, &contacts);
    context.scene().setGravity(kGravity);

    const StackDesc stack = makeStack(8);
    Holder holder;
    holder.value = createStack(
        context, stack, PxTransform(PxVec3(0.0f)), baseStepSleep);
    ExtStressPhysXDestructible& dest = *holder.value;
    contacts.dest = &dest;

    ExtStressPhysXFrameStepper* stepper = ExtStressPhysXFrameStepper::create(context.scene());
    require(stepper != nullptr, "stepper create failed");
    contacts.stepper = stepper;

    ExtStressPhysXDestructible* destructibles[] = {&dest};
    ExtStressPhysXResimOptions options;
    options.maxPasses = 1;
    options.scopedResim = scoped;
    options.quietCaptureSkip = false;

    auto launch = [&](const PxVec3& start, const PxVec3& vel) {
        PxRigidDynamic* ball = context.physics().createRigidDynamic(PxTransform(start));
        require(ball != nullptr, "ball create failed");
        PxShape* shape =
            context.physics().createShape(PxSphereGeometry(0.3f), context.material(), false);
        require(shape && ball->attachShape(*shape), "ball shape failed");
        ball->setMass(80.0f);
        ball->setLinearVelocity(vel);
        context.scene().addActor(*ball);
        return ball;
    };

    PxRigidDynamic* ball0 = launch(PxVec3(0.0f, 4.5f, -4.0f), PxVec3(0.0f, 0.0f, 28.0f));
    PxRigidDynamic* ball1 = nullptr;

    RunResult result;
    for (std::uint32_t step = 0; step < 180; ++step)
    {
        if (step == 45)
        {
            const float z = secondImpactOffset ? -4.0f : -4.0f;
            const float x = secondImpactOffset ? 1.5f : 0.0f;
            ball1 = launch(PxVec3(x, 3.5f, z), PxVec3(0.0f, 0.0f, 26.0f));
        }
        ExtStressPhysXFrameStats stats;
        require(
            stepper->stepFrame(kDt, kGravity, destructibles, 1, options, nullptr, &stats),
            "stepFrame failed");
        result.frozenPeak = std::max(result.frozenPeak, stats.sceneBodiesFrozen);
    }
    result.splits = dest.getTelemetry().splits;
    result.poses = capturePoses(dest);
    require(result.splits > 0, "scenario must fracture");

    stepper->release();
    ball0->release();
    if (ball1)
    {
        ball1->release();
    }
    return result;
}

void testScopedVsFullCascade()
{
    const RunResult full = runScenario(false, false, false);
    const RunResult scoped = runScenario(true, false, false);
    const float delta = maxPoseDelta(full.poses, scoped.poses);
    require(delta < 0.01f, "cascade scoped vs full pose delta must be < 1cm");
    std::printf("cascade scoped-vs-full maxDelta=%.6f frozenPeak=%u\n", delta, scoped.frozenPeak);
}

void testIsolatedTight()
{
    // Same launcher path with scoped on/off; agreement must stay sub-cm.
    const RunResult full = runScenario(false, false, false);
    const RunResult scoped = runScenario(true, false, false);
    const float delta = maxPoseDelta(full.poses, scoped.poses);
    require(delta < 0.01f, "isolated/cascade agreement must stay sub-cm");
    std::printf("isolated-style agreement maxDelta=%.6f\n", delta);
}

void testNegativeControl()
{
    const RunResult a = runScenario(true, false, false);
    const RunResult b = runScenario(true, true, false);
    const float delta = maxPoseDelta(a.poses, b.poses);
    require(delta > 0.05f, "different impact must diverge");
    std::printf("negative-control maxDelta=%.6f\n", delta);
}

void testBaseStepSleepDoesNotExplode()
{
    // Stepper no longer auto-sleeps mid-cascade (support-loss hazard). The
    // setting alone must be output-neutral; hosts call applyBaseStepSleep()
    // after long quiet windows.
    const RunResult awake = runScenario(true, false, false);
    const RunResult flagged = runScenario(true, false, true);
    const float delta = maxPoseDelta(awake.poses, flagged.poses);
    std::printf("base-step-sleep-flag maxDelta=%.6f\n", delta);
    require(delta < 1.0e-4f, "baseStepSleep flag without host calls must be neutral");
    require(flagged.splits > 0, "must still fracture");

    // Explicit API call after the run is safe (may sleep resting debris).
    ImpactContacts contacts;
    SceneCapacity capacity;
    PhysXScene context(PhysicsMode::Cpu, false, capacity, &contacts);
    context.scene().setGravity(kGravity);
    const StackDesc stack = makeStack(4);
    Holder holder;
    holder.value = createStack(context, stack, PxTransform(PxIdentity), true);
    const std::uint32_t slept = holder.value->applyBaseStepSleep();
    std::printf("applyBaseStepSleep returned %u\n", slept);
}

} // namespace

int main()
{
    try
    {
        testScopedVsFullCascade();
        testIsolatedTight();
        testNegativeControl();
        testBaseStepSleepDoesNotExplode();
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::printf("resim scoped cascade test passed\n");
    return 0;
}
