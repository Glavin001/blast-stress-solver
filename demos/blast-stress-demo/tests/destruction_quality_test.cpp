// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Destruction QUALITY on a real multi-material ScenePack, driven through the
// path users actually run: ExtStressPhysXFrameStepper with resimulation on.
//
// The other suites each cover one layer — material_behavior_test isolates one
// knob at a time on synthetic fixtures, scene_pack_conformance_test checks the
// loader, resim_* tests check rollback mechanics on single-material stacks.
// None of them answer the question that decides whether the library is usable:
//
//   Does a real building, authored with heterogeneous materials and simulated
//   with resim=1, break WELL?
//
// "Well" is a band with a failure mode on each side, and both are easy to hit:
//
//   too brittle  — the structure atomizes into individual chunks the moment
//                  anything touches it (shattered glass). Every chunk its own
//                  body, no recognisable pieces, no standing remainder.
//   too rigid    — nothing breaks no matter what you throw. Usually means bond
//                  areas were inflated until joints carry no load, which is
//                  invisible to a damage gate because "0 shattered" reads as
//                  success. This is the failure this project actually shipped
//                  once.
//
// A good structure sits between them and moves through the band monotonically
// as impact energy rises: untouched -> cladding sheds with the frame standing
// -> frame comes down -> total collapse.
//
// Fixture: assets/reference/reference-building.json (ScenePack v2, 76 nodes,
// 4 materials, 3 floors). Runs on CPU in a couple of seconds.
//
// NOTE on what "partial" means for this fixture. The building now has a beam
// ring, and its standing fraction plateaus at ~0.66 across a wide band of
// energies. 0.66 is very close to "every frame chunk standing, every facade
// panel shed" (52 frame chunks of 76), so a passing `moderate` here is mostly
// a cladding result — `moved(column)` is the number that says whether the
// FRAME took part, and it stays 0 until ~1.8 t. See the experiment notes in
// export-reference-building.mjs; the band being wide is not the same as the
// damage being graded.

#include "../physx_scene.h"
#include "../scene_pack.h"

#include <NvBlastExtStressPhysX.h>
#include <NvBlastExtStressPhysXResim.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Nv::Blast;
using namespace physx;
using blast_demo::PhysXScene;
using blast_demo::PhysicsMode;
using blast_demo::SceneCapacity;
using blast_demo::ScenePack;
using blast_demo::SceneColliderKind;
using blast_demo::loadScenePack;

namespace
{

constexpr float kDt = 1.0f / 60.0f;
const PxVec3 kGravity(0.0f, -9.81f, 0.0f);

void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error("destruction quality test failed: " + message);
    }
}

// ── ScenePack -> adapter descriptors ────────────────────────────────────────
// Mirrors what the demo does, minus the parts a test does not need. Keeping it
// here (rather than sharing demo internals) also proves the pack carries
// everything an integration needs: geometry, roles, materials, indices.

std::vector<ExtStressPhysXNodeDesc> makeNodes(const ScenePack& pack)
{
    std::vector<ExtStressPhysXNodeDesc> result(pack.nodes.size());
    for (std::size_t i = 0; i < pack.nodes.size(); ++i)
    {
        const auto& source = pack.nodes[i];
        ExtStressPhysXNodeDesc& target = result[i];
        target.centroid = source.centroid;
        target.mass = source.mass;
        target.volume = source.volume;
        target.geometry.localPose = PxTransform(source.centroid);
        require(
            source.collider.kind == SceneColliderKind::Cuboid,
            "this fixture is expected to be all-cuboid");
        target.geometry.halfExtents = source.collider.halfExtents;
    }
    return result;
}

std::vector<ExtStressPhysXBondDesc> makeBonds(const ScenePack& pack)
{
    std::vector<ExtStressPhysXBondDesc> result(pack.bonds.size());
    for (std::size_t i = 0; i < pack.bonds.size(); ++i)
    {
        const auto& source = pack.bonds[i];
        ExtStressPhysXBondDesc& target = result[i];
        target.node0 = source.node0;
        target.node1 = source.node1;
        target.centroid = source.centroid;
        target.normal = source.normal;
        target.area = source.area;
        target.material = source.material;
    }
    return result;
}

std::vector<ExtStressPhysXMaterial> makeMaterials(const ScenePack& pack)
{
    std::vector<ExtStressPhysXMaterial> result;
    result.reserve(pack.materials.size());
    for (const auto& source : pack.materials)
    {
        ExtStressPhysXMaterial material;
        material.compressionElasticLimit = source.limits.compressionElastic;
        material.compressionFatalLimit = source.limits.compressionFatal;
        material.tensionElasticLimit = source.limits.tensionElastic;
        material.tensionFatalLimit = source.limits.tensionFatal;
        material.shearElasticLimit = source.limits.shearElastic;
        material.shearFatalLimit = source.limits.shearFatal;
        result.push_back(material);
    }
    return result;
}

// ── What a run produced ─────────────────────────────────────────────────────

struct Outcome
{
    std::uint64_t splits{0};
    std::uint32_t bodies{0};
    std::uint32_t largestBodyChunks{0};   // biggest surviving connected piece
    std::uint32_t supportedChunks{0};     // still attached to the world
    std::uint64_t resimPasses{0};
    std::map<std::string, std::uint32_t> movedByRole;
    std::uint32_t totalChunks{0};

    /** Fraction of the structure still in one connected supported piece. */
    float standingFraction() const
    {
        return totalChunks ? static_cast<float>(supportedChunks) / totalChunks : 0.0f;
    }
    /** 1.0 means every chunk became its own body — i.e. dust. */
    float fragmentation() const
    {
        return totalChunks ? static_cast<float>(bodies) / totalChunks : 0.0f;
    }
    std::uint32_t moved(const std::string& role) const
    {
        const auto found = movedByRole.find(role);
        return found == movedByRole.end() ? 0 : found->second;
    }
};

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

/**
 * Drop a projectile of `mass` kg at `speed` m/s into the building's mid-height
 * facade, simulate through the frame stepper with resim, and report what is
 * left. `mass == 0` runs gravity only.
 */
Outcome run(const ScenePack& pack, float mass, float speed)
{
    const std::vector<ExtStressPhysXNodeDesc> nodes = makeNodes(pack);
    const std::vector<ExtStressPhysXBondDesc> bonds = makeBonds(pack);
    const std::vector<ExtStressPhysXMaterial> materials = makeMaterials(pack);

    SceneCapacity capacity;
    capacity.maxBodies = static_cast<std::uint32_t>(nodes.size()) * 4 + 256;
    capacity.maxShapes = capacity.maxBodies * 2;

    // Contact routing: unity gain, exactly as the contract specifies. Anything
    // else here would let the test pass with capacities the impact could not
    // really break.
    struct Router : public PxSimulationEventCallback
    {
        ExtStressPhysXDestructible* destructible{nullptr};
        ExtStressPhysXFrameStepper* stepper{nullptr};
        void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
        void onWake(PxActor**, PxU32) override {}
        void onSleep(PxActor**, PxU32) override {}
        void onTrigger(PxTriggerPair*, PxU32) override {}
        void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) override {}
        void onContact(
            const PxContactPairHeader&, const PxContactPair* pairs, PxU32 pairCount) override
        {
            std::vector<PxContactPairPoint> points;
            for (PxU32 i = 0; i < pairCount; ++i)
            {
                const PxContactPair& pair = pairs[i];
                if (stepper && pair.shapes[0] && pair.shapes[1])
                {
                    stepper->recordDynamicContactPair(
                        pair.shapes[0]->getActor(), pair.shapes[1]->getActor());
                }
                if (!pair.contactCount || !destructible)
                {
                    continue;
                }
                points.resize(pair.contactCount);
                const PxU32 written = pair.extractContacts(points.data(), pair.contactCount);
                for (PxU32 p = 0; p < written; ++p)
                {
                    for (PxU32 s = 0; s < 2; ++s)
                    {
                        if (PxShape* shape = pair.shapes[s])
                        {
                            destructible->queueContact(
                                *shape,
                                points[p].position,
                                s == 0 ? points[p].impulse : -points[p].impulse);
                        }
                    }
                }
            }
        }
    } router;

    PhysXScene context(PhysicsMode::Cpu, false, capacity, &router);
    context.scene().setGravity(PxVec3(0.0f, pack.gravity, 0.0f));

    ExtStressPhysXDesc desc;
    desc.physics = &context.physics();
    desc.scene = &context.scene();
    desc.material = &context.material();
    desc.nodes = nodes.data();
    desc.nodeCount = static_cast<std::uint32_t>(nodes.size());
    desc.bonds = bonds.data();
    desc.bondCount = static_cast<std::uint32_t>(bonds.size());
    desc.stressMaterials = materials.data();
    desc.stressMaterialCount = static_cast<std::uint32_t>(materials.size());
    desc.worldTransform = PxTransform(PxVec3(0.0f));
    // Resim replaces the synthetic momentum paths, so both are off — the
    // re-solved contact IS the momentum source.
    desc.settings.applyExcessForces = false;
    desc.settings.minimumSeparationVelocity = 0.0f;

    ExtStressPhysXTelemetry failure;
    Holder building{ExtStressPhysXDestructible::create(desc, &failure)};
    require(building.value != nullptr, "destructible creation failed");
    router.destructible = building.value;

    ExtStressPhysXFrameStepper* stepper = ExtStressPhysXFrameStepper::create(context.scene());
    require(stepper != nullptr, "frame stepper creation failed");
    router.stepper = stepper;
    ExtStressPhysXDestructible* destructibles[] = {building.value};

    // resim=1 is the default users run and the primary path under test.
    ExtStressPhysXResimOptions options;
    options.maxPasses = 1;

    // Settle first so the impact lands on a structure at rest.
    std::uint64_t resimPasses = 0;
    for (std::uint32_t step = 0; step < 60; ++step)
    {
        ExtStressPhysXFrameStats stats;
        stepper->stepFrame(kDt, kGravity, destructibles, 1, options, nullptr, &stats);
        resimPasses += stats.resimPasses;
    }
    require(
        building.value->getTelemetry().splits == 0,
        "the reference building must settle under gravity without fracturing");

    PxRigidDynamic* ball = nullptr;
    PxShape* ballShape = nullptr;
    if (mass > 0.0f)
    {
        // Aim at mid-height on the -z facade, straight down the +z axis.
        const PxVec3 target(0.0f, 5.1f, -2.2f);
        const PxVec3 launchDir(0.0f, 0.0f, 1.0f);
        const PxVec3 start = target - launchDir * 5.0f;
        const float radius = 0.5f;
        ball = context.physics().createRigidDynamic(PxTransform(start));
        require(ball != nullptr, "projectile body creation failed");
        ballShape =
            context.physics().createShape(PxSphereGeometry(radius), context.material(), false);
        require(ballShape != nullptr && ball->attachShape(*ballShape), "projectile shape failed");
        ball->setMass(mass);
        PxRigidBodyExt::updateMassAndInertia(*ball, mass / (4.19f * radius * radius * radius));
        ball->setMass(mass);
        ball->setLinearVelocity(launchDir * speed);
        // Without CCD a fast heavy ball can tunnel a thin facade panel.
        ball->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, true);
        context.scene().addActor(*ball);
    }

    for (std::uint32_t step = 0; step < 240; ++step)
    {
        ExtStressPhysXFrameStats stats;
        stepper->stepFrame(kDt, kGravity, destructibles, 1, options, nullptr, &stats);
        resimPasses += stats.resimPasses;
    }

    Outcome outcome;
    outcome.totalChunks = static_cast<std::uint32_t>(pack.nodes.size());
    outcome.splits = building.value->getTelemetry().splits;
    outcome.resimPasses = resimPasses;

    std::vector<ExtStressPhysXBodySnapshot> bodies(pack.nodes.size() * 2 + 16);
    const std::uint32_t bodyCount =
        building.value->getBodySnapshots(bodies.data(), static_cast<std::uint32_t>(bodies.size()));
    outcome.bodies = bodyCount;
    for (std::uint32_t i = 0; i < bodyCount; ++i)
    {
        outcome.largestBodyChunks = std::max(outcome.largestBodyChunks, bodies[i].nodeCount);
        if (bodies[i].kinematic)
        {
            outcome.supportedChunks += bodies[i].nodeCount;
        }
    }

    std::vector<ExtStressPhysXShapeSnapshot> shapes(pack.nodes.size() * 2 + 16);
    const std::uint32_t shapeCount =
        building.value->getShapeSnapshots(shapes.data(), static_cast<std::uint32_t>(shapes.size()));
    for (std::uint32_t i = 0; i < shapeCount; ++i)
    {
        const auto& shape = shapes[i];
        if (shape.nodeIndex >= pack.nodes.size() || pack.nodeTypes.empty())
        {
            continue;
        }
        const PxVec3 delta = shape.worldPose.p - pack.nodes[shape.nodeIndex].centroid;
        if (delta.magnitude() >= 0.5f)
        {
            ++outcome.movedByRole[pack.nodeTypes[shape.nodeIndex]];
        }
    }

    stepper->release();
    if (ball) ball->release();
    if (ballShape) ballShape->release();
    return outcome;
}

void report(const char* label, const Outcome& outcome)
{
    std::printf(
        "  %-22s splits=%-4llu bodies=%-3u largest=%-3u standing=%.2f frag=%.2f "
        "resim=%-4llu moved(infill/column/slab/foundation)=%u/%u/%u/%u\n",
        label,
        static_cast<unsigned long long>(outcome.splits),
        outcome.bodies,
        outcome.largestBodyChunks,
        static_cast<double>(outcome.standingFraction()),
        static_cast<double>(outcome.fragmentation()),
        static_cast<unsigned long long>(outcome.resimPasses),
        outcome.moved("infill"),
        outcome.moved("column"),
        outcome.moved("slab"),
        outcome.moved("foundation"));
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: destruction_quality_test <reference-building.json>\n");
        return 2;
    }

    try
    {
        const ScenePack pack = loadScenePack(argv[1]);
        require(pack.version == 2, "fixture must be a v2 pack");
        require(pack.materials.size() >= 3, "fixture must be genuinely multi-material");
        require(!pack.nodeTypes.empty(), "fixture must carry node roles");

        std::printf("destruction quality (resim=1, unity contact gain):\n");

        // `--sweep` characterizes the response curve instead of asserting a
        // band — how the test levels below were chosen rather than guessed.
        //
        // An optional third argument replaces the default probes with a custom
        // list, "mass@speed,mass@speed,..." (kg and m/s). The default eight are
        // coarse — deliberately, they are the shape of the curve — so measuring
        // how WIDE the partial band actually is needs a finer ramp than they
        // provide. Passing probes beats editing and rebuilding this file, and
        // keeps a characterization run out of the asserted path entirely.
        if (argc >= 3 && std::string(argv[2]) == "--sweep")
        {
            std::vector<std::pair<float, float>> probes{
                {0.0f, 0.0f}, {500.0f, 12.0f}, {1000.0f, 14.0f}, {1500.0f, 16.0f},
                {2500.0f, 18.0f}, {4000.0f, 20.0f}, {8000.0f, 25.0f},
                {40000.0f, 45.0f}};
            if (argc >= 4)
            {
                probes.clear();
                const std::string spec(argv[3]);
                std::size_t at = 0;
                while (at < spec.size())
                {
                    const std::size_t comma = std::min(spec.find(',', at), spec.size());
                    const std::string item = spec.substr(at, comma - at);
                    const std::size_t split = item.find('@');
                    require(split != std::string::npos, "probe must be mass@speed: " + item);
                    probes.emplace_back(
                        std::stof(item.substr(0, split)), std::stof(item.substr(split + 1)));
                    at = comma + 1;
                }
                require(!probes.empty(), "custom sweep needs at least one probe");
            }
            for (const auto& probe : probes)
            {
                char label[64];
                std::snprintf(label, sizeof(label), "%.0fkg @ %.0fm/s", probe.first, probe.second);
                report(label, run(pack, probe.first, probe.second));
            }
            return 0;
        }

        // ── Gravity only: the structure must simply stand ───────────────────
        const Outcome rest = run(pack, 0.0f, 0.0f);
        report("gravity only", rest);
        require(rest.splits == 0, "must not fracture under self-weight");
        require(
            rest.standingFraction() > 0.99f,
            "an undamaged structure must be entirely supported");
        require(rest.bodies == 1, "an undamaged structure is one body");

        // ── Glancing hit: NOT made of glass ─────────────────────────────────
        // A structure that comes apart the moment anything touches it is the
        // first failure mode. This impact carries real energy (29 kJ) and must
        // still leave the building whole.
        //
        // 400 kg, down from 500: the beam ring lowered the energy at which the
        // cladding lets go, even though the facade is calibrated to the SAME
        // gravity safety factor (3.03) as the beamless building. The ring ties
        // all four faces into one stiff loop, so a hit on one face is carried
        // round it and pops clips on faces the projectile never touched. The
        // shed threshold is genuinely between 450 and 500 kg here, so this
        // level is the honest one rather than the one that used to pass.
        const Outcome glancing = run(pack, 400.0f, 12.0f);
        report("glancing (400kg@12)", glancing);
        require(
            glancing.splits == 0,
            "a 400 kg glancing hit must not break a 72 t concrete frame — a "
            "structure that shatters on contact is the 'glass' failure mode");

        // ── Moderate hit: THE interesting band ──────────────────────────────
        // Partial destruction: a real hole, a large connected remainder still
        // standing, and debris in recognisable pieces rather than dust. This
        // is the result the library exists to produce and the one both failure
        // modes destroy.
        const Outcome moderate = run(pack, 1500.0f, 16.0f);
        report("moderate (1.5t@16)", moderate);
        require(moderate.splits > 0, "a moderate hit must do real damage");
        require(
            moderate.moved("foundation") == 0,
            "foundations are world-fixed and must never move");
        require(
            moderate.standingFraction() > 0.25f && moderate.standingFraction() < 0.85f,
            "a moderate hit must leave a PARTIAL structure — neither untouched nor "
            "flattened (standing=" + std::to_string(moderate.standingFraction()) + ")");
        require(
            moderate.fragmentation() < 0.7f,
            "must not atomize into individual chunks — that is shattered glass, not "
            "destruction (fragmentation=" + std::to_string(moderate.fragmentation()) + ")");
        require(
            moderate.largestBodyChunks >= pack.nodes.size() / 4,
            "a large connected piece must survive, not dust "
            "(largest=" + std::to_string(moderate.largestBodyChunks) + " of "
                + std::to_string(pack.nodes.size()) + ")");
        require(
            moderate.resimPasses > 0,
            "resimulation must actually run on fracture frames — resim=1 is the "
            "path users run by default and the primary case under test");

        // ── Heavy hit: damage keeps escalating ──────────────────────────────
        const Outcome heavy = run(pack, 4000.0f, 20.0f);
        report("heavy (4t@20)", heavy);
        require(
            heavy.standingFraction() < moderate.standingFraction(),
            "more energy must leave less standing");
        require(
            heavy.moved("column") > moderate.moved("column"),
            "the frame must participate more as energy rises");

        // ── Extreme hit: everything CAN break ───────────────────────────────
        // The opposite failure mode. A structure nothing can break passes every
        // damage gate while being useless — this project shipped exactly that.
        const Outcome extreme = run(pack, 40000.0f, 45.0f);
        report("extreme (40t@45)", extreme);
        require(
            extreme.fragmentation() > 0.9f,
            "with enough energy EVERYTHING must be breakable — a structure that "
            "survives a 40 t impact is over-authored, not strong "
            "(fragmentation=" + std::to_string(extreme.fragmentation()) + ")");
        require(
            extreme.moved("column") > 0 && extreme.moved("slab") > 0,
            "at extreme energy the frame itself must fail, not just the cladding");

        // ── Monotonicity across the sweep ───────────────────────────────────
        // One ordering check catches non-physical tuning that happens to
        // satisfy each level in isolation.
        require(
            rest.standingFraction() >= glancing.standingFraction()
                && glancing.standingFraction() > moderate.standingFraction()
                && moderate.standingFraction() >= heavy.standingFraction()
                && heavy.standingFraction() >= extreme.standingFraction(),
            "the standing remainder must decrease monotonically with impact energy");
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    std::printf("destruction quality test passed\n");
    return 0;
}
