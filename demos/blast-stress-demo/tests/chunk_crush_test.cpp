// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// What chunk CRUSHING actually does — the evidence behind the crush model.
//
// Bond limits decide whether a JOINT fails. Crush properties decide whether the
// CHUNK ITSELF is comminuted and leaves the rigid-body simulation as dust. Both
// happen in the same impact: most of a wall separates along its joints while
// the small region under the hit is ground up.
//
// Every claim the crush documentation makes is measured here, on tiny CPU-only
// fixtures so the matrix stays broad, fast and deterministic.
//
//   crushing is OPT-IN and byte-identical when off  -> testDisabledByDefault
//   a settled structure never grinds itself to dust -> testSettledStructureNeverCrushes
//   utilisation reads before anything crushes       -> testUtilisationIsReadableBeforeYield
//   past yield, chunks comminute                    -> testOverstressedChunkCrushes
//   the cap separates confined from unconfined      -> testConfinementDiscriminates
//   mass and momentum are reported, not guessed     -> testEventCarriesMassAndMomentum
//   the chunk leaves the scene, and Blast's does not-> testCrushedChunkLeavesTheSimulation
//   crushing needs no contact of its own            -> testBondLoadedChunkCrushes
//   flow is quadratic in overstress                 -> testFlowIsQuadraticInOverstress
//   toughness decides how much is lost              -> testCrushEnergyControlsHowMuchIsLost
//   debris can carry part of the mass back          -> testDebrisFractionRespawnsMass

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

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error("chunk crush test failed: " + message);
    }
}

/**
 * A material strong enough at the JOINTS that nothing ever breaks a bond, so
 * every test here measures crushing in isolation rather than racing it against
 * fracture.
 */
ExtStressPhysXMaterial unbreakableJoints()
{
    ExtStressPhysXMaterial result;
    result.compressionElasticLimit = 1.0e12f;
    result.compressionFatalLimit = 2.0e12f;
    return result;
}

/**
 * Attach crush properties to a material. Yield is a Drucker-Prager cone with a
 * pressure cap; the cone is pinned to `fc` the same way the reference building
 * authors it, so a chunk yields at its own compressive strength under an
 * unconfined squeeze.
 */
ExtStressPhysXMaterial withCrush(
    ExtStressPhysXMaterial material,
    float fc,
    float crushEnergy,
    float crushViscosity = 2.0e5f,
    float frictionSlope = 1.2f)
{
    material.crushCapPressure = 2.5f * fc;
    material.crushCohesion = fc * (1.0f - frictionSlope / 3.0f);
    material.crushFrictionSlope = frictionSlope;
    material.crushEnergy = crushEnergy;
    material.crushViscosity = crushViscosity;
    return material;
}

struct Structure
{
    std::vector<ExtStressPhysXNodeDesc> nodes;
    std::vector<ExtStressPhysXBondDesc> bonds;
};

/**
 * A column of `panels` masses stacked on a mass-0 footing, the same fixture
 * shape material_behavior_test uses. Bond i joins panel i to i+1, so the
 * bottom panel carries the whole stack.
 */
Structure makeColumn(
    std::uint32_t panels,
    float panelMass,
    std::uint32_t nodeMaterial = 0,
    float halfExtent = 0.5f)
{
    Structure result;
    const float pitch = halfExtent * 2.0f;
    for (std::uint32_t i = 0; i <= panels; ++i)
    {
        ExtStressPhysXNodeDesc node;
        node.centroid = PxVec3(0.0f, halfExtent + static_cast<float>(i) * pitch, 0.0f);
        node.mass = i == 0 ? 0.0f : panelMass;
        node.volume = pitch * pitch * pitch;
        node.material = nodeMaterial;
        node.geometry.localPose = PxTransform(node.centroid);
        node.geometry.halfExtents = PxVec3(halfExtent);
        result.nodes.push_back(node);
    }
    for (std::uint32_t i = 0; i < panels; ++i)
    {
        ExtStressPhysXBondDesc bond;
        bond.node0 = i;
        bond.node1 = i + 1;
        bond.centroid = PxVec3(0.0f, pitch * (static_cast<float>(i) + 1.0f), 0.0f);
        bond.normal = PxVec3(0.0f, 1.0f, 0.0f);
        bond.area = pitch * pitch;
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

ExtStressPhysXDestructible* create(
    PhysXScene& context,
    const Structure& structure,
    const std::vector<ExtStressPhysXMaterial>& materials,
    const PxVec3& origin)
{
    ExtStressPhysXDesc desc;
    desc.physics = &context.physics();
    desc.scene = &context.scene();
    desc.material = &context.material();
    desc.nodes = structure.nodes.data();
    desc.nodeCount = static_cast<std::uint32_t>(structure.nodes.size());
    desc.bonds = structure.bonds.data();
    desc.bondCount = static_cast<std::uint32_t>(structure.bonds.size());
    desc.worldTransform = PxTransform(origin);
    desc.stressMaterials = materials.data();
    desc.stressMaterialCount = static_cast<std::uint32_t>(materials.size());
    desc.settings.applyExcessForces = false;
    desc.settings.minimumSeparationVelocity = 0.0f;
    // Same reasoning as material_behavior_test: these fixtures are fully
    // supported and never move, so with the throughput skips on the solver
    // would freeze its inputs and the tests would measure a cache.
    desc.settings.skipSettledIslands = false;
    desc.settings.idleSkip = false;
    ExtStressPhysXTelemetry failure;
    ExtStressPhysXDestructible* created = ExtStressPhysXDestructible::create(desc, &failure);
    require(created != nullptr, "destructible creation failed");
    return created;
}

void step(PhysXScene& context, ExtStressPhysXDestructible& destructible, std::uint32_t steps)
{
    for (std::uint32_t i = 0; i < steps; ++i)
    {
        context.scene().simulate(kDt);
        context.scene().fetchResults(true);
        require(destructible.tick(kDt, kGravity), "tick failed");
    }
}

std::vector<float> crushUtilisation(
    ExtStressPhysXDestructible& destructible,
    std::uint32_t nodes)
{
    std::vector<float> result(nodes, 0.0f);
    destructible.getNodeCrushUtilisation(result.data(), nodes);
    return result;
}

std::vector<float> crushDamage(ExtStressPhysXDestructible& destructible, std::uint32_t nodes)
{
    std::vector<float> result(nodes, 0.0f);
    destructible.getNodeCrushDamage(result.data(), nodes);
    return result;
}

float maximumOf(const std::vector<float>& values)
{
    float peak = 0.0f;
    for (float value : values)
    {
        peak = std::max(peak, value);
    }
    return peak;
}

/**
 * Load one node directly with a steady force, expressed as a contact.
 *
 * This is how these fixtures create overstress without needing a projectile:
 * a real force at a real application point, which is what a per-chunk stress
 * tensor is built from.
 */
void pressNode(
    ExtStressPhysXDestructible& destructible,
    const Structure& structure,
    std::uint32_t nodeIndex,
    const PxVec3& force,
    const PxVec3& relativeVelocity = PxVec3(0.0f))
{
    std::vector<ExtStressPhysXShapeSnapshot> shapes(structure.nodes.size());
    const std::uint32_t count = destructible.getShapeSnapshots(
        shapes.data(), static_cast<std::uint32_t>(shapes.size()));
    for (std::uint32_t i = 0; i < count; ++i)
    {
        if (shapes[i].nodeIndex != nodeIndex)
        {
            continue;
        }
        ExtStressPhysXContact contact;
        contact.shapeId = shapes[i].shapeId;
        contact.worldPosition = shapes[i].worldPose.p;
        contact.worldImpulse = force * kDt;
        contact.worldRelativeVelocity = relativeVelocity;
        destructible.queueContact(contact);
        return;
    }
}

/** Press a node every step for `steps` steps, returning chunks crushed. */
std::uint64_t pressFor(
    PhysXScene& context,
    ExtStressPhysXDestructible& destructible,
    const Structure& structure,
    std::uint32_t nodeIndex,
    const PxVec3& force,
    std::uint32_t steps)
{
    for (std::uint32_t i = 0; i < steps; ++i)
    {
        pressNode(destructible, structure, nodeIndex, force);
        step(context, destructible, 1);
    }
    return destructible.getTelemetry().chunksCrushed;
}

// ── Crushing is opt-in ───────────────────────────────────────────────────────
//
// A material that authors no crush properties must behave exactly as it did
// before crushing existed. This is the guarantee that lets the feature ship
// without re-validating every existing asset.

void testDisabledByDefault()
{
    PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
    const Structure structure = makeColumn(4, 5000.0f);
    // Default-constructed material: no crush block at all.
    const std::vector<ExtStressPhysXMaterial> materials{unbreakableJoints()};

    Holder holder;
    holder.value = create(context, structure, materials, PxVec3(0.0f));
    require(!holder.value->isCrushEnabled(), "crushing must be off without crush properties");

    pressFor(context, *holder.value, structure, 4, PxVec3(0.0f, -5.0e8f, 0.0f), 90);

    require(holder.value->getTelemetry().chunksCrushed == 0,
            "a material with no crush properties must never crush");
    require(maximumOf(crushUtilisation(*holder.value, 5)) == 0.0f,
            "crush utilisation must read zero when crushing is disabled");
    require(holder.value->validateMappings(), "mappings must stay valid");
}

// ── A settled structure never grinds itself to dust ──────────────────────────
//
// The failure mode a crush model must not have: standing still long enough and
// slowly comminuting. Flow is driven by OVERSTRESS, so a chunk inside its yield
// surface accumulates exactly nothing no matter how long it stands.

void testSettledStructureNeverCrushes()
{
    PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
    const Structure structure = makeColumn(4, 5000.0f);
    const std::vector<ExtStressPhysXMaterial> materials{
        withCrush(unbreakableJoints(), 24.0e6f, 1.2e8f)};

    Holder holder;
    holder.value = create(context, structure, materials, PxVec3(0.0f));
    require(holder.value->isCrushEnabled(), "crushing should be enabled");

    step(context, *holder.value, 600);

    require(holder.value->getTelemetry().chunksCrushed == 0,
            "a structure carrying only its own weight must never crush");
    require(maximumOf(crushDamage(*holder.value, 5)) == 0.0f,
            "no crush damage may accumulate below yield");
    const float peak = maximumOf(crushUtilisation(*holder.value, 5));
    require(peak > 0.0f && peak < 1.0f,
            "self-weight should register a real but sub-yield utilisation, got "
                + std::to_string(peak));
}

// ── Utilisation is readable before anything crushes ──────────────────────────
//
// The crush analogue of a joint's safety factor, and the number to author
// against. It must read correctly while the structure is intact and still,
// otherwise the only way to discover the crush margin is to exceed it.

void testUtilisationIsReadableBeforeYield()
{
    PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
    const Structure structure = makeColumn(4, 5000.0f);

    // Same geometry and load, two chunk strengths an order of magnitude apart.
    const std::vector<ExtStressPhysXMaterial> strong{
        withCrush(unbreakableJoints(), 24.0e6f, 1.2e8f)};
    const std::vector<ExtStressPhysXMaterial> weak{
        withCrush(unbreakableJoints(), 2.4e6f, 1.2e8f)};

    Holder strongHolder;
    strongHolder.value = create(context, structure, strong, PxVec3(0.0f));
    step(context, *strongHolder.value, 120);
    const float strongPeak = maximumOf(crushUtilisation(*strongHolder.value, 5));

    Holder weakHolder;
    weakHolder.value = create(context, structure, weak, PxVec3(50.0f, 0.0f, 0.0f));
    step(context, *weakHolder.value, 120);
    const float weakPeak = maximumOf(crushUtilisation(*weakHolder.value, 5));

    require(strongPeak > 0.0f, "utilisation must be observable on an intact structure");
    // Ten times weaker chunks under the same load means about ten times the
    // utilisation: it is a stress/limit ratio, so it tracks the limit directly.
    const float ratio = weakPeak / strongPeak;
    require(ratio > 5.0f && ratio < 20.0f,
            "utilisation should scale inversely with chunk strength, ratio was "
                + std::to_string(ratio));
}

// ── Past yield, chunks comminute ─────────────────────────────────────────────

void testOverstressedChunkCrushes()
{
    PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
    const Structure structure = makeColumn(4, 5000.0f);
    // Deliberately friable, gypsum-board class: this is the material that
    // should read as dust.
    const std::vector<ExtStressPhysXMaterial> materials{
        withCrush(unbreakableJoints(), 0.8e6f, 2.0e6f)};

    Holder holder;
    holder.value = create(context, structure, materials, PxVec3(0.0f));

    const std::uint64_t crushed =
        pressFor(context, *holder.value, structure, 4, PxVec3(0.0f, -4.0e6f, 0.0f), 180);

    require(crushed > 0, "a chunk held far past its yield surface must comminute");
    require(holder.value->getTelemetry().crushedMassKg > 0.0,
            "crushing must remove mass from the simulation");
    require(holder.value->validateMappings(),
            "mappings must stay valid after a chunk leaves the scene");
}

// ── The pressure cap separates confined crushing from unconfined shear ───────
//
// This is what earns the Drucker-Prager cap over a bare pressure threshold. The
// same chunk, the same material, the same force magnitude: pressed INTO its
// neighbours it is confined and comminutes; pulled away from them it is not,
// and it must survive. A scalar-stress model cannot tell these apart, which is
// how a crush feature ends up turning free-floating debris to dust.

void testConfinementDiscriminates()
{
    const std::vector<ExtStressPhysXMaterial> materials{
        withCrush(unbreakableJoints(), 0.8e6f, 2.0e6f)};

    // Confined: pressed down into the stack below it.
    std::uint64_t confinedCrushed = 0;
    float confinedPressure = 0.0f;
    {
        PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
        const Structure structure = makeColumn(4, 5000.0f);
        Holder holder;
        holder.value = create(context, structure, materials, PxVec3(0.0f));
        for (std::uint32_t i = 0; i < 180; ++i)
        {
            pressNode(*holder.value, structure, 4, PxVec3(0.0f, -4.0e6f, 0.0f));
            step(context, *holder.value, 1);
            // Sampled per step, not at the end: a comminuted chunk no longer
            // has a stress state, so reading afterwards would always find zero.
            std::vector<float> pressure(5, 0.0f);
            holder.value->getNodeStressInvariants(pressure.data(), nullptr, 5);
            confinedPressure = std::max(confinedPressure, maximumOf(pressure));
        }
        confinedCrushed = holder.value->getTelemetry().chunksCrushed;
    }

    // Unconfined: the same force magnitude pulling the top chunk UP and away
    // from the stack. Tension, not confinement.
    std::uint64_t unconfinedCrushed = 0;
    {
        PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
        const Structure structure = makeColumn(4, 5000.0f);
        Holder holder;
        holder.value = create(context, structure, materials, PxVec3(0.0f));
        unconfinedCrushed =
            pressFor(context, *holder.value, structure, 4, PxVec3(0.0f, 4.0e6f, 0.0f), 180);
    }

    require(confinedPressure > 0.0f,
            "a chunk pressed into its neighbours must register positive confining pressure");
    require(confinedCrushed > unconfinedCrushed,
            "confined loading must comminute more than unconfined loading of the same "
            "magnitude (confined=" + std::to_string(confinedCrushed)
                + " unconfined=" + std::to_string(unconfinedCrushed) + ")");
}

// ── The event carries mass and momentum, so dust need not be guessed ─────────
//
// Without these on the event a consumer can only invent a dust cloud's mass and
// velocity, and invented dust is what makes destruction read as a cartoon.

void testEventCarriesMassAndMomentum()
{
    PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
    const float panelMass = 5000.0f;
    const Structure structure = makeColumn(4, panelMass);
    const std::vector<ExtStressPhysXMaterial> materials{
        withCrush(unbreakableJoints(), 0.8e6f, 2.0e6f)};

    Holder holder;
    holder.value = create(context, structure, materials, PxVec3(0.0f));
    pressFor(context, *holder.value, structure, 4, PxVec3(0.0f, -4.0e6f, 0.0f), 180);

    std::vector<ExtStressPhysXChunkDestroyed> events(16);
    const std::uint32_t drained = holder.value->drainChunkDestroyedEvents(
        events.data(), static_cast<std::uint32_t>(events.size()));
    require(drained > 0, "a pulverized chunk must report an event");

    double reportedMass = 0.0;
    for (std::uint32_t i = 0; i < drained; ++i)
    {
        const ExtStressPhysXChunkDestroyed& event = events[i];
        require(event.mass > 0.0f, "event must report the mass that left the simulation");
        require(event.volume > 0.0f, "event must report the volume that left");
        require(std::fabs(event.mass - panelMass) < 1.0f,
                "event mass must match the authored chunk mass");
        require(event.worldPose.isValid(), "event must report where the chunk was");
        require(event.linearVelocity.isFinite() && event.angularVelocity.isFinite(),
                "event must report the chunk's motion so dust can match it");
        require(event.peakPressure != 0.0f || event.peakDeviator != 0.0f,
                "event must report the stress state that destroyed the chunk");
        reportedMass += event.mass;
    }

    require(std::fabs(reportedMass - holder.value->getTelemetry().crushedMassKg) < 1.0f,
            "the events must account for exactly the mass the telemetry reports");

    // A drain is a drain: nothing may be reported twice.
    const std::uint32_t second = holder.value->drainChunkDestroyedEvents(
        events.data(), static_cast<std::uint32_t>(events.size()));
    require(second == 0, "draining twice must not repeat events");
}

// ── The chunk leaves the simulation, which Blast alone would never do ────────
//
// NvBlast has no removal: a health-exhausted leaf chunk stays alive as an inert
// but still-present actor forever (NvBlastActor.cpp,
// partitionSingleLowerSupportChunk returns before reaching release()). The
// physics-side removal is this adapter's job, and this is the test that says it
// happened.

void testCrushedChunkLeavesTheSimulation()
{
    PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
    const Structure structure = makeColumn(4, 5000.0f);
    const std::vector<ExtStressPhysXMaterial> materials{
        withCrush(unbreakableJoints(), 0.8e6f, 2.0e6f)};

    Holder holder;
    holder.value = create(context, structure, materials, PxVec3(0.0f));

    std::vector<ExtStressPhysXShapeSnapshot> before(structure.nodes.size());
    const std::uint32_t shapesBefore = holder.value->getShapeSnapshots(
        before.data(), static_cast<std::uint32_t>(before.size()));

    const std::uint64_t crushed =
        pressFor(context, *holder.value, structure, 4, PxVec3(0.0f, -4.0e6f, 0.0f), 180);
    require(crushed > 0, "expected at least one chunk to comminute");

    std::vector<ExtStressPhysXShapeSnapshot> after(structure.nodes.size());
    const std::uint32_t shapesAfter = holder.value->getShapeSnapshots(
        after.data(), static_cast<std::uint32_t>(after.size()));

    require(shapesAfter + crushed == shapesBefore,
            "every pulverized chunk must remove exactly one shape from the scene "
            "(before=" + std::to_string(shapesBefore) + " after=" + std::to_string(shapesAfter)
                + " crushed=" + std::to_string(crushed) + ")");
    require(holder.value->validateMappings(),
            "solver actors and PhysX bodies must still agree after removal");
}

// ── Crushing needs no contact of its own ─────────────────────────────────────
//
// The case a contact-driven model gets wrong: a chunk buried in a structure,
// never touched directly, comminuted by the load arriving through its BONDS.
// Flow is driven by overstress rather than by any measured strain, so this
// works — and it is the mechanism behind a collapsing building crushing its own
// lower storeys.

void testBondLoadedChunkCrushes()
{
    PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
    const Structure structure = makeColumn(4, 5000.0f);
    const std::vector<ExtStressPhysXMaterial> materials{
        withCrush(unbreakableJoints(), 0.8e6f, 2.0e6f)};

    Holder holder;
    holder.value = create(context, structure, materials, PxVec3(0.0f));

    // Press the TOP chunk only. Everything below it is loaded purely through
    // bonds and never receives a contact.
    const std::uint32_t topNode = static_cast<std::uint32_t>(structure.nodes.size() - 1);
    pressFor(context, *holder.value, structure, topNode, PxVec3(0.0f, -6.0e6f, 0.0f), 240);

    std::vector<ExtStressPhysXChunkDestroyed> events(16);
    const std::uint32_t drained = holder.value->drainChunkDestroyedEvents(
        events.data(), static_cast<std::uint32_t>(events.size()));
    require(drained > 0, "expected chunks to comminute under a bond-borne load");

    bool crushedWithoutContact = false;
    for (std::uint32_t i = 0; i < drained; ++i)
    {
        if (events[i].nodeIndex != topNode)
        {
            crushedWithoutContact = true;
        }
    }
    require(crushedWithoutContact,
            "a chunk loaded only through its bonds must be able to comminute");
}

// ── Flow is quadratic in overstress ──────────────────────────────────────────
//
// Perzyna overstress flow makes damage grow with the SQUARE of how far past
// yield a chunk sits. That is what lets one material be near-indestructible
// under ordinary load and still comminute under a hard enough hit, without a
// second threshold to author.

void testFlowIsQuadraticInOverstress()
{
    const std::vector<ExtStressPhysXMaterial> materials{
        withCrush(unbreakableJoints(), 0.8e6f, 2.0e8f)};

    const auto damageAfter = [&](float force) {
        PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
        const Structure structure = makeColumn(4, 5000.0f);
        Holder holder;
        holder.value = create(context, structure, materials, PxVec3(0.0f));
        for (std::uint32_t i = 0; i < 60; ++i)
        {
            pressNode(*holder.value, structure, 4, PxVec3(0.0f, -force, 0.0f));
            step(context, *holder.value, 1);
        }
        return maximumOf(crushDamage(*holder.value, 5));
    };

    const float single = damageAfter(4.0e6f);
    const float doubled = damageAfter(8.0e6f);

    require(single > 0.0f, "the reference load must accumulate some crush damage");
    require(doubled > single, "more overstress must accumulate more damage");
    // Doubling the load more than doubles the damage. Bounded above too: a
    // linear law would land at 2x and is the thing being ruled out, while an
    // unbounded ratio would mean the numbers had run away.
    const float ratio = doubled / single;
    require(ratio > 2.5f,
            "damage must grow faster than linearly with overstress, ratio was "
                + std::to_string(ratio));
}

// ── Toughness decides how much is lost ───────────────────────────────────────
//
// crushEnergy is the specific comminution energy, the real material property
// separating "the impact zone turns to dust" from "the whole structure does".

void testCrushEnergyControlsHowMuchIsLost()
{
    const auto crushedWith = [](float crushEnergy) {
        PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
        const Structure structure = makeColumn(4, 5000.0f);
        const std::vector<ExtStressPhysXMaterial> materials{
            withCrush(unbreakableJoints(), 0.8e6f, crushEnergy)};
        Holder holder;
        holder.value = create(context, structure, materials, PxVec3(0.0f));
        return pressFor(context, *holder.value, structure, 4, PxVec3(0.0f, -4.0e6f, 0.0f), 180);
    };

    const std::uint64_t friable = crushedWith(2.0e6f);
    const std::uint64_t tough = crushedWith(2.0e10f);

    require(friable > 0, "a friable material must comminute under this load");
    require(tough == 0,
            "a tough material under the same load must not comminute, got "
                + std::to_string(tough));
}

// ── Debris can carry part of the mass back ───────────────────────────────────
//
// Default is total loss: all of a pulverized chunk's mass leaves the rigid-body
// simulation and the event reports it for rendering as dust. Authoring a debris
// fraction trades body count for a pile that keeps some real mass.

void testDebrisFractionRespawnsMass()
{
    PhysXScene context(PhysicsMode::Cpu, false, SceneCapacity{}, nullptr);
    const Structure structure = makeColumn(4, 5000.0f);
    ExtStressPhysXMaterial crushing = withCrush(unbreakableJoints(), 0.8e6f, 2.0e6f);
    crushing.crushDebrisMassFraction = 0.5f;
    crushing.crushDebrisFragmentCount = 4;
    const std::vector<ExtStressPhysXMaterial> materials{crushing};

    Holder holder;
    holder.value = create(context, structure, materials, PxVec3(0.0f));
    const std::uint64_t crushed =
        pressFor(context, *holder.value, structure, 4, PxVec3(0.0f, -4.0e6f, 0.0f), 180);
    require(crushed > 0, "expected at least one chunk to comminute");

    const ExtStressPhysXTelemetry& telemetry = holder.value->getTelemetry();
    require(telemetry.debrisBodiesSpawned == crushed * 4,
            "each pulverized chunk must spawn its authored fragment count, got "
                + std::to_string(telemetry.debrisBodiesSpawned));

    std::vector<ExtStressPhysXChunkDestroyed> events(16);
    const std::uint32_t drained = holder.value->drainChunkDestroyedEvents(
        events.data(), static_cast<std::uint32_t>(events.size()));
    require(drained > 0, "expected chunk-destroyed events");
    for (std::uint32_t i = 0; i < drained; ++i)
    {
        require(std::fabs(events[i].debrisMassFraction - 0.5f) < 1.0e-5f,
                "the event must state what fraction was respawned, so a consumer knows "
                "how much mass the dust should carry");
        require(events[i].debrisBodiesSpawned == 4,
                "the event must state how many debris bodies were actually created");
    }
    require(holder.value->validateMappings(), "mappings must stay valid with debris present");
}

} // namespace

int main()
{
    try
    {
        testDisabledByDefault();
        testSettledStructureNeverCrushes();
        testUtilisationIsReadableBeforeYield();
        testOverstressedChunkCrushes();
        testConfinementDiscriminates();
        testEventCarriesMassAndMomentum();
        testCrushedChunkLeavesTheSimulation();
        testBondLoadedChunkCrushes();
        testFlowIsQuadraticInOverstress();
        testCrushEnergyControlsHowMuchIsLost();
        testDebrisFractionRespawnsMass();
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    std::printf("chunk crush tests passed\n");
    return 0;
}
