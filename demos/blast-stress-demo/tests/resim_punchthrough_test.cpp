// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Regression for fracture-frame punch-through: a heavy projectile hitting a
// still-monolithic, densely bonded facade must split on the impact frame so
// ExtStressPhysXFrameStepper can re-solve the contact against the opened hole.
//
// Root cause this guards: maximumBodies used to truncate bond-fracture commands
// per tick by remaining body slots (1 bond ≈ 1 body). On a large support graph
// one impact overstresses far more bonds than that leftover budget; applying an
// arbitrary prefix often failed to cut around the contact, so the projectile
// rebounded from the monolith and resimulation had no hole on that frame.
// Bond breaks must not be artificially budgeted against body caps.

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
        throw std::runtime_error(std::string("resim punchthrough test failed: ") + message);
    }
}

struct GridDesc
{
    std::vector<ExtStressPhysXNodeDesc> nodes;
    std::vector<ExtStressPhysXBondDesc> bonds;
    std::uint32_t hitNode{0};
    PxVec3 hitWorld{0.0f};
};

// Tall, densely bonded facade: two mass-0 side columns plus an MxN panel grid
// with 4-neighbor bonds. Redundant connectivity means a random subset of bond
// breaks often leaves the struck panel attached — matching the high-rise
// support graph failure mode under a tight body budget.
GridDesc makeFacadeGrid(std::uint32_t rows, std::uint32_t cols)
{
    require(rows >= 3 && cols >= 3, "facade grid too small");
    GridDesc result;
    const float panel = 1.0f;
    auto addNode = [&](const PxVec3& c, float mass) {
        ExtStressPhysXNodeDesc node;
        node.centroid = c;
        node.mass = mass;
        node.volume = 1.0f;
        node.geometry.localPose = PxTransform(c);
        node.geometry.halfExtents = PxVec3(panel * 0.45f);
        result.nodes.push_back(node);
        return static_cast<std::uint32_t>(result.nodes.size() - 1);
    };
    auto addBond = [&](std::uint32_t a, std::uint32_t b) {
        ExtStressPhysXBondDesc bond;
        bond.node0 = a;
        bond.node1 = b;
        bond.centroid = 0.5f * (result.nodes[a].centroid + result.nodes[b].centroid);
        bond.normal = (result.nodes[b].centroid - result.nodes[a].centroid).getNormalized();
        bond.area = 0.25f;
        result.bonds.push_back(bond);
    };

    const std::uint32_t leftCol = addNode(PxVec3(-1.0f, rows * 0.5f, 0.0f), 0.0f);
    const std::uint32_t rightCol = addNode(PxVec3(cols * panel, rows * 0.5f, 0.0f), 0.0f);
    std::vector<std::uint32_t> panelIndex(rows * cols, 0);
    for (std::uint32_t r = 0; r < rows; ++r)
    {
        for (std::uint32_t c = 0; c < cols; ++c)
        {
            const PxVec3 centroid(
                panel * static_cast<float>(c),
                panel * (0.5f + static_cast<float>(r)),
                0.0f);
            panelIndex[r * cols + c] = addNode(centroid, 40.0f);
        }
    }
    for (std::uint32_t r = 0; r < rows; ++r)
    {
        addBond(leftCol, panelIndex[r * cols + 0]);
        addBond(rightCol, panelIndex[r * cols + (cols - 1)]);
        for (std::uint32_t c = 0; c < cols; ++c)
        {
            const std::uint32_t i = panelIndex[r * cols + c];
            if (c + 1 < cols)
            {
                addBond(i, panelIndex[r * cols + c + 1]);
            }
            if (r + 1 < rows)
            {
                addBond(i, panelIndex[(r + 1) * cols + c]);
            }
        }
    }

    const std::uint32_t hitRow = rows / 2;
    const std::uint32_t hitCol = cols / 2;
    result.hitNode = panelIndex[hitRow * cols + hitCol];
    result.hitWorld = result.nodes[result.hitNode].centroid;
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

ExtStressPhysXDestructible* createFacade(
    PhysXScene& context,
    const GridDesc& grid,
    std::uint32_t maximumBodies)
{
    ExtStressPhysXDesc desc;
    desc.physics = &context.physics();
    desc.scene = &context.scene();
    desc.material = &context.material();
    desc.nodes = grid.nodes.data();
    desc.nodeCount = static_cast<std::uint32_t>(grid.nodes.size());
    desc.bonds = grid.bonds.data();
    desc.bondCount = static_cast<std::uint32_t>(grid.bonds.size());
    desc.worldTransform = PxTransform(PxVec3(0.0f));
    desc.settings.applyExcessForces = false;
    desc.settings.minimumSeparationVelocity = 0.0f;
    desc.settings.protectSupportBonds = true;
    desc.settings.supportPeelMaxMass = 1000.0f;
    desc.settings.fatalizeImpactContactBonds = true;
    desc.settings.maximumBodies = maximumBodies;
    desc.settings.compressionElasticLimit = 5.0e5f;
    desc.settings.compressionFatalLimit = 1.5e6f;
    desc.settings.tensionElasticLimit = 5.0e5f;
    desc.settings.tensionFatalLimit = 1.5e6f;
    desc.settings.shearElasticLimit = 5.0e5f;
    desc.settings.shearFatalLimit = 1.5e6f;
    ExtStressPhysXTelemetry failure;
    ExtStressPhysXDestructible* destructible =
        ExtStressPhysXDestructible::create(desc, &failure);
    require(destructible != nullptr, "facade destructible creation failed");
    return destructible;
}

std::uint32_t bodyCount(ExtStressPhysXDestructible& destructible)
{
    std::vector<ExtStressPhysXBodySnapshot> bodies(256);
    return destructible.getBodySnapshots(
        bodies.data(),
        static_cast<std::uint32_t>(bodies.size()));
}

struct ImpactContacts : public PxSimulationEventCallback
{
    ExtStressPhysXDestructible* dest{nullptr};
    float forceScale{80.0f};

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

void testImpactFramePunchthrough()
{
    ImpactContacts contacts;
    SceneCapacity capacity;
    PhysXScene context(PhysicsMode::Cpu, false, capacity, &contacts);
    context.scene().setGravity(kGravity);

    // 8x8 panels + 2 columns. Pass an explicit low maximumBodies to prove the
    // old "truncate bond breaks by remaining body slots" regression stays fixed:
    // under the cap we must still apply the full overstressed cut on impact.
    const GridDesc grid = makeFacadeGrid(8, 8);
    DestructibleHolder holder;
    holder.value = createFacade(context, grid, /*maximumBodies=*/24);
    ExtStressPhysXDestructible& destructible = *holder.value;
    contacts.dest = &destructible;
    require(bodyCount(destructible) == 1, "facade must start monolithic");

    const PxVec3 launchDir(0.0f, 0.0f, 1.0f);
    const float launchSpeed = 30.0f;
    const PxVec3 start = grid.hitWorld - launchDir * 6.0f;
    PxRigidDynamic* ball =
        context.physics().createRigidDynamic(PxTransform(start));
    require(ball != nullptr, "projectile body creation failed");
    PxShape* ballShape =
        context.physics().createShape(PxSphereGeometry(0.35f), context.material(), false);
    require(ballShape != nullptr && ball->attachShape(*ballShape), "projectile shape failed");
    ball->setMass(2500.0f);
    ball->setLinearVelocity(launchDir * launchSpeed);
    context.scene().addActor(*ball);

    ExtStressPhysXFrameStepper* stepper =
        ExtStressPhysXFrameStepper::create(context.scene());
    require(stepper != nullptr, "frame stepper creation failed");
    ExtStressPhysXDestructible* destructibles[] = {&destructible};
    ExtStressPhysXResimOptions options;
    options.maxPasses = 1;

    bool sawImpact = false;
    for (std::uint32_t step = 0; step < 120; ++step)
    {
        ExtStressPhysXFrameStats stats;
        const std::uint64_t contactsBefore = destructible.getTelemetry().contactsProcessed;
        const std::uint64_t splitsBefore = destructible.getTelemetry().splits;
        require(
            stepper->stepFrame(kDt, kGravity, destructibles, 1, options, nullptr, &stats),
            "stepFrame failed");
        const std::uint64_t contactsAfter = destructible.getTelemetry().contactsProcessed;
        const std::uint64_t splitsAfter = destructible.getTelemetry().splits;
        if (contactsAfter == contactsBefore)
        {
            continue;
        }
        sawImpact = true;
        const float forwardAfter = ball->getLinearVelocity().dot(launchDir);
        require(
            splitsAfter > splitsBefore,
            "first projectile-contact frame must split the facade");
        require(stats.resimPasses >= 1, "first impact split must trigger a resim pass");
        require(
            forwardAfter > 0.25f * launchSpeed,
            "projectile must keep forward speed through the opened hole");
        require(bodyCount(destructible) > 1, "impact must leave more than one body");
        break;
    }
    require(sawImpact, "projectile never contacted the facade");

    stepper->release();
    ball->release();
    ballShape->release();
}

} // namespace

int main()
{
    try
    {
        testImpactFramePunchthrough();
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::printf("resim punchthrough test passed\n");
    return 0;
}
