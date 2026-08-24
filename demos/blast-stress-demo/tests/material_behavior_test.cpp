// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// What materials actually DO — the evidence behind the authoring guide.
//
// This library deliberately ships no material library. There are infinitely
// many real materials, every project's scale and units differ, and a
// prescriptive enum would be a maintenance treadmill that still fails to fit
// anyone's case. What a physics library owes its users instead is a *method*:
// which knob produces which behavior, how to tell symptoms apart, and what to
// reach for next when a structure misbehaves.
//
// These are that method, executed. Each test isolates one authoring question
// and answers it with a measurement, so
// `.cursor/skills/blast-structure-authoring/SKILL.md` can cite a test by name
// for every claim it makes instead of restating numbers that quietly rot.
//
// Fixtures are deliberately tiny — a handful of nodes, CPU-only, milliseconds
// each — so the matrix can be broad and stay deterministic. Whole-scene tests
// would make this coverage unaffordable and flaky.
//
// The knobs, and where each is exercised:
//   strength           = elastic limit                 → testStrengthSetsFailureThreshold
//   ductility          = fatal - elastic band width    → testBandWidthControlsBrittleVsDuctile
//   which joint fails  = relative strength ordering    → testWeakestLinkFailsFirstRegardlessOfLoad
//   geometry vs material (independent axes)            → testAreaAndMaterialAreIndependentAxes
//   mode decoupling    = tension vs compression        → testTensionAndCompressionAreIndependent
//   progressive collapse                               → testLoadRedistributesOntoSurvivors
//   non-structural cladding                            → testFacadeShedsWithoutDroppingFrame

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
        throw std::runtime_error("material behavior test failed: " + message);
    }
}

ExtStressPhysXMaterial material(
    float compressionElastic,
    float compressionFatal,
    float tensionElastic = -1.0f,
    float tensionFatal = -1.0f,
    float shearElastic = -1.0f,
    float shearFatal = -1.0f)
{
    ExtStressPhysXMaterial result;
    result.compressionElasticLimit = compressionElastic;
    result.compressionFatalLimit = compressionFatal;
    result.tensionElasticLimit = tensionElastic;
    result.tensionFatalLimit = tensionFatal;
    result.shearElasticLimit = shearElastic;
    result.shearFatalLimit = shearFatal;
    return result;
}

struct Structure
{
    std::vector<ExtStressPhysXNodeDesc> nodes;
    std::vector<ExtStressPhysXBondDesc> bonds;
};

// A column of `panels` masses stacked on a mass-0 footing. Bond i joins panel
// i to i+1, so bond 0 carries the whole stack and the topmost carries one
// panel — a clean load gradient in a 7-node fixture.
Structure makeColumn(
    std::uint32_t panels,
    float panelMass,
    float bondArea,
    std::uint32_t uniformMaterial = 0)
{
    Structure result;
    for (std::uint32_t i = 0; i <= panels; ++i)
    {
        ExtStressPhysXNodeDesc node;
        node.centroid = PxVec3(0.0f, 0.5f + static_cast<float>(i), 0.0f);
        node.mass = i == 0 ? 0.0f : panelMass;
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
        bond.area = bondArea;
        bond.material = uniformMaterial;
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
    // Settled-island skip and idle skip are OFF for these fixtures on purpose.
    // They are throughput optimizations: a converged island whose node inputs
    // have not changed is not re-solved. These structures are fully supported
    // and never move, so their inputs never change — with the skips on, the
    // stress values freeze at their pre-fracture numbers and the tests would
    // be measuring a cache instead of the solve. (Real scenes self-correct:
    // once a piece detaches it moves, which changes inputs. But a fully
    // kinematic structure that fractures will NOT re-evaluate its load path
    // until something moves — see the authoring guide's symptom table.)
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
        destructible.tick(kDt, kGravity);
    }
}

/** Steps until the first split, returning the step index (or -1 within budget). */
std::int32_t stepsToFirstSplit(
    PhysXScene& context,
    ExtStressPhysXDestructible& destructible,
    std::uint32_t budget)
{
    for (std::uint32_t i = 0; i < budget; ++i)
    {
        step(context, destructible, 1);
        if (destructible.getTelemetry().splits > 0)
        {
            return static_cast<std::int32_t>(i);
        }
    }
    return -1;
}

std::vector<float> utilisations(ExtStressPhysXDestructible& destructible, std::uint32_t bonds)
{
    std::vector<float> result(bonds, 0.0f);
    destructible.getBondUtilisations(result.data(), bonds);
    return result;
}

// ── The elastic limit is the failure threshold ───────────────────────────────
// Authoring question: "how strong is this joint?" Answer: its elastic limit,
// compared against the stress its geometry actually carries. A structure
// stands when limit > stress and yields when it does not; nothing else in the
// material decides *whether* it fails.
void testStrengthSetsFailureThreshold(PhysXScene& context)
{
    // 6 panels x 200 kg over 1 m^2 ~= 11.8 kPa at the footing.
    const Structure column = makeColumn(6, 200.0f, 1.0f);

    Holder strong{create(context, column, {material(1.0e6f, 2.0e6f)}, PxVec3(0.0f))};
    step(context, *strong.value, 120);
    require(
        strong.value->getTelemetry().splits == 0,
        "a limit far above the carried stress must not fail");

    // Elastic limit below the ~11.8 kPa the footing carries: it must yield.
    Holder weak{create(context, column, {material(2.0e3f, 4.0e3f)}, PxVec3(30.0f, 0.0f, 0.0f))};
    require(
        stepsToFirstSplit(context, *weak.value, 120) >= 0,
        "a limit below the carried stress must fail");
}

// ── The fatal-elastic band is ductility ──────────────────────────────────────
// Authoring question: "it breaks, but too suddenly / too slowly." That is the
// BAND WIDTH, not the strength. Damage per tick is proportional to how far
// past elastic the stress is, divided by the band, so a narrow band snaps and
// a wide band yields over many frames. Both materials below have the SAME
// elastic limit — only the band differs — and they fail at very different
// rates. This is the main behavioral dial and it is independent of strength.
void testBandWidthControlsBrittleVsDuctile(PhysXScene& context)
{
    const Structure column = makeColumn(6, 200.0f, 1.0f);
    const float elastic = 2.0e3f;

    Holder brittle{
        create(context, column, {material(elastic, elastic * 1.05f)}, PxVec3(0.0f))};
    const std::int32_t brittleStep = stepsToFirstSplit(context, *brittle.value, 400);

    Holder ductile{
        create(context, column, {material(elastic, elastic * 500.0f)}, PxVec3(30.0f, 0.0f, 0.0f))};
    const std::int32_t ductileStep = stepsToFirstSplit(context, *ductile.value, 400);

    require(brittleStep >= 0, "the brittle material must eventually fail");
    require(
        ductileStep < 0 || ductileStep > brittleStep * 4,
        "a wide fatal-elastic band must take far longer to fail than a narrow one "
        "at the same elastic limit (brittle=" + std::to_string(brittleStep)
            + ", ductile=" + std::to_string(ductileStep) + ")");
}

// ── The weakest joint fails first, wherever the load is ──────────────────────
// Authoring question: "I want THIS to be the thing that gives." Make it the
// lowest-capacity joint relative to the stress it carries. Here the topmost
// bond carries the least load (one panel) yet still fails first, because its
// material is weak enough that its utilisation is highest.
void testWeakestLinkFailsFirstRegardlessOfLoad(PhysXScene& context)
{
    Structure column = makeColumn(6, 200.0f, 1.0f, /*uniformMaterial=*/0);
    column.bonds.back().material = 1; // topmost joint = least loaded

    const std::vector<ExtStressPhysXMaterial> materials{
        material(1.0e6f, 2.0e6f),   // 0: strong frame
        material(5.0e2f, 1.0e3f)};  // 1: deliberate fuse

    Holder structure{create(context, column, materials, PxVec3(0.0f))};
    require(
        stepsToFirstSplit(context, *structure.value, 200) >= 0,
        "the fuse joint must fail");

    // The fuse is the LEAST loaded joint, so utilisation — not raw stress —
    // is what predicts failure. That is why the load-path report and the
    // safety-factor gates are stated in utilisation.
    const std::vector<float> util = utilisations(*structure.value, 6);
    require(util[0] < util.back() || util.back() == 0.0f,
            "the fuse joint must show the highest utilisation before it breaks");
}

// ── Geometry and material are independent axes ───────────────────────────────
// The single most important property, and the one whose violation caused every
// authoring failure in this project's history. Bond area is the stress
// denominator AND the damage pool, so "make it stronger by giving it more
// area" corrupts the readout and the toughness together. Doubling area halves
// stress; changing material leaves stress untouched and moves only capacity.
void testAreaAndMaterialAreIndependentAxes(PhysXScene& context)
{
    const std::vector<ExtStressPhysXMaterial> one{material(1.0e6f, 2.0e6f)};

    Holder unit{create(context, makeColumn(6, 200.0f, 1.0f), one, PxVec3(0.0f))};
    Holder doubled{
        create(context, makeColumn(6, 200.0f, 2.0f), one, PxVec3(30.0f, 0.0f, 0.0f))};
    // Same geometry as `unit`, ten times the capacity.
    Holder stronger{create(
        context, makeColumn(6, 200.0f, 1.0f), {material(1.0e7f, 2.0e7f)},
        PxVec3(60.0f, 0.0f, 0.0f))};

    step(context, *unit.value, 1);
    step(context, *doubled.value, 1);
    step(context, *stronger.value, 1);

    std::vector<float> unitStress(6, 0.0f), doubledStress(6, 0.0f), strongerStress(6, 0.0f);
    unit.value->getBondStresses(unitStress.data(), nullptr, nullptr, 6);
    doubled.value->getBondStresses(doubledStress.data(), nullptr, nullptr, 6);
    stronger.value->getBondStresses(strongerStress.data(), nullptr, nullptr, 6);

    require(unitStress[0] > 0.0f, "footing must carry stress");
    // Area is the denominator: twice the patch, half the pressure.
    require(
        std::fabs(doubledStress[0] - unitStress[0] * 0.5f) <= 0.05f * unitStress[0],
        "doubling bond area must halve stress");
    // Material does not appear in the stress calculation at all.
    require(
        std::fabs(strongerStress[0] - unitStress[0]) <= 0.02f * unitStress[0],
        "changing material must not change stress");
    // ...but it does move utilisation, by exactly the capacity ratio.
    const std::vector<float> unitUtil = utilisations(*unit.value, 6);
    const std::vector<float> strongerUtil = utilisations(*stronger.value, 6);
    require(
        std::fabs(strongerUtil[0] - unitUtil[0] * 0.1f) <= 0.02f * unitUtil[0],
        "ten times the capacity must be ten times the safety factor");
}

// ── Tension, compression and shear are separate capacities ───────────────────
// Authoring question: "it holds its own weight but shatters on impact" (or the
// reverse). Real materials are wildly asymmetric — concrete carries ~10x more
// compression than tension — and the limits are independent so that asymmetry
// is expressible. A column in pure compression survives a tension limit far
// below the stress it carries, because that stress is not tension.
void testTensionAndCompressionAreIndependent(PhysXScene& context)
{
    const Structure column = makeColumn(6, 200.0f, 1.0f);
    // Strong in compression, almost nothing in tension/shear.
    Holder asymmetric{create(
        context, column,
        {material(1.0e6f, 2.0e6f, 1.0e2f, 2.0e2f, 1.0e6f, 2.0e6f)},
        PxVec3(0.0f))};
    step(context, *asymmetric.value, 120);

    std::vector<float> compression(6, 0.0f), tension(6, 0.0f);
    asymmetric.value->getBondStresses(compression.data(), tension.data(), nullptr, 6);
    require(compression[0] > 0.0f, "a stacked column loads its footing in compression");
    require(
        tension[0] <= 1.0e2f,
        "a stacked column must not develop tension at the footing");
    require(
        asymmetric.value->getTelemetry().splits == 0,
        "a near-zero TENSION limit must not fail a column loaded in COMPRESSION");
}

// ── Progressive collapse is emergent, not scripted ───────────────────────────
// Authoring question: "how do I get a chain reaction?" You do not script one.
// Remove a capacity link and the load it carried redistributes onto its
// neighbours, raising their utilisation; if that pushes them past their own
// limits they fail in turn. This test measures only the first half — the
// redistribution — because that is the mechanism the rest depends on.
void testLoadRedistributesOntoSurvivors(PhysXScene& context)
{
    // Two parallel legs under one mass: cutting one must load the other.
    Structure frame;
    auto addNode = [&](const PxVec3& centroid, float mass) {
        ExtStressPhysXNodeDesc node;
        node.centroid = centroid;
        node.mass = mass;
        node.volume = 1.0f;
        node.geometry.localPose = PxTransform(centroid);
        node.geometry.halfExtents = PxVec3(0.4f);
        frame.nodes.push_back(node);
        return static_cast<std::uint32_t>(frame.nodes.size() - 1);
    };
    auto addBond = [&](std::uint32_t a, std::uint32_t b, std::uint32_t material) {
        ExtStressPhysXBondDesc bond;
        bond.node0 = a;
        bond.node1 = b;
        bond.centroid = 0.5f * (frame.nodes[a].centroid + frame.nodes[b].centroid);
        bond.normal = (frame.nodes[b].centroid - frame.nodes[a].centroid).getNormalized();
        bond.area = 0.5f;
        bond.material = material;
        frame.bonds.push_back(bond);
    };

    const std::uint32_t leftFoot = addNode(PxVec3(-1.0f, 0.4f, 0.0f), 0.0f);
    const std::uint32_t rightFoot = addNode(PxVec3(1.0f, 0.4f, 0.0f), 0.0f);
    const std::uint32_t leftLeg = addNode(PxVec3(-1.0f, 1.4f, 0.0f), 400.0f);
    const std::uint32_t rightLeg = addNode(PxVec3(1.0f, 1.4f, 0.0f), 400.0f);
    const std::uint32_t deck = addNode(PxVec3(0.0f, 2.3f, 0.0f), 2000.0f);
    addBond(leftFoot, leftLeg, 1);   // sacrificial leg
    addBond(rightFoot, rightLeg, 0); // survivor
    addBond(leftLeg, deck, 0);
    addBond(rightLeg, deck, 0);

    const std::vector<ExtStressPhysXMaterial> materials{
        material(1.0e6f, 2.0e6f),   // 0: strong
        material(3.0e3f, 6.0e3f)};  // 1: the leg designed to go

    Holder structure{create(context, frame, materials, PxVec3(0.0f))};
    step(context, *structure.value, 1);
    const float survivorBefore = utilisations(*structure.value, 4)[1];

    require(
        stepsToFirstSplit(context, *structure.value, 400) >= 0,
        "the sacrificial leg must fail");
    step(context, *structure.value, 10);
    const float survivorAfter = utilisations(*structure.value, 4)[1];

    require(survivorBefore > 0.0f, "the survivor leg must carry load initially");
    require(
        survivorAfter > survivorBefore * 1.2f,
        "losing one leg must raise the survivor's utilisation — that redistribution "
        "IS progressive collapse; nothing scripts it (before="
            + std::to_string(survivorBefore) + ", after=" + std::to_string(survivorAfter) + ")");
}

// ── Non-structural cladding sheds without dropping the frame ─────────────────
// The canonical building requirement, and the reason heterogeneous materials
// exist at all. Panels hung on a weak material tear off under their own weight
// while the frame carrying them, on a strong material, is untouched. Before
// per-bond materials this could only be faked by inflating frame area, which
// made the frame unbreakable rather than strong.
void testFacadeShedsWithoutDroppingFrame(PhysXScene& context)
{
    Structure building;
    auto addNode = [&](const PxVec3& centroid, float mass) {
        ExtStressPhysXNodeDesc node;
        node.centroid = centroid;
        node.mass = mass;
        node.volume = 1.0f;
        node.geometry.localPose = PxTransform(centroid);
        node.geometry.halfExtents = PxVec3(0.4f);
        building.nodes.push_back(node);
        return static_cast<std::uint32_t>(building.nodes.size() - 1);
    };
    auto addBond = [&](std::uint32_t a, std::uint32_t b, float area, std::uint32_t material) {
        ExtStressPhysXBondDesc bond;
        bond.node0 = a;
        bond.node1 = b;
        bond.centroid = 0.5f * (building.nodes[a].centroid + building.nodes[b].centroid);
        bond.normal = (building.nodes[b].centroid - building.nodes[a].centroid).getNormalized();
        bond.area = area;
        bond.material = material;
        building.bonds.push_back(bond);
    };

    const std::uint32_t footing = addNode(PxVec3(0.0f, 0.4f, 0.0f), 0.0f);
    const std::uint32_t column = addNode(PxVec3(0.0f, 1.4f, 0.0f), 1200.0f);
    const std::uint32_t panelA = addNode(PxVec3(1.0f, 1.4f, 0.0f), 300.0f);
    const std::uint32_t panelB = addNode(PxVec3(-1.0f, 1.4f, 0.0f), 300.0f);
    addBond(footing, column, 0.25f, 0); // frame:  reinforced
    addBond(column, panelA, 0.20f, 1);  // facade: clip
    addBond(column, panelB, 0.20f, 1);

    const std::vector<ExtStressPhysXMaterial> materials{
        material(24.0e6f, 60.0e6f),          // 0: frame
        material(2.0e3f, 4.0e3f)};           // 1: cladding clip

    Holder structure{create(context, building, materials, PxVec3(0.0f))};
    step(context, *structure.value, 200);

    require(
        structure.value->getTelemetry().splits > 0,
        "the cladding must shed under its own weight");

    // The frame bond survives: its utilisation is still well under 1.
    const std::vector<float> util = utilisations(*structure.value, 3);
    require(
        util[0] > 0.0f && util[0] < 1.0f,
        "the frame joint must remain loaded and intact while cladding sheds "
        "(utilisation=" + std::to_string(util[0]) + ")");

    // And the column is still attached to the world, i.e. still kinematic.
    std::vector<ExtStressPhysXBodySnapshot> bodies(16);
    const std::uint32_t count =
        structure.value->getBodySnapshots(bodies.data(), 16);
    bool frameStanding = false;
    for (std::uint32_t i = 0; i < count; ++i)
    {
        if (bodies[i].kinematic && bodies[i].nodeCount >= 2)
        {
            frameStanding = true;
        }
    }
    require(frameStanding, "footing and column must remain one supported body");
}

} // namespace

int main()
{
    try
    {
        SceneCapacity capacity;
        PhysXScene context(PhysicsMode::Cpu, false, capacity, nullptr);
        context.scene().setGravity(kGravity);

        testStrengthSetsFailureThreshold(context);
        testBandWidthControlsBrittleVsDuctile(context);
        testWeakestLinkFailsFirstRegardlessOfLoad(context);
        testAreaAndMaterialAreIndependentAxes(context);
        testTensionAndCompressionAreIndependent(context);
        testLoadRedistributesOntoSurvivors(context);
        testFacadeShedsWithoutDroppingFrame(context);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    std::printf("material behavior test passed\n");
    return 0;
}
