#include "extensions/stressphysx/NvBlastExtStressPhysX.h"

#include "../../../../rust_stress_example/ffi/ext_stress_bridge.h"

#include "cooking/PxCooking.h"
#include "extensions/PxMassProperties.h"
#include "extensions/PxRigidBodyExt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Nv
{
namespace Blast
{

using namespace physx;

namespace
{

constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();

/// E2: how often fracture() runs the full mapping audit (validateMappings).
/// The audit is a pure bug DETECTOR — it changes nothing, it only reports —
/// and it is O(all nodes + all bodies) with a heap allocation and a sort per
/// body, measured at 1.1–1.7 ms per fracturing tick at city scale. Running
/// it every Nth fracturing tick still catches a corrupted mapping loudly
/// within N fracture ticks (~a quarter second), for ~1/N of the cost.
/// Default 16; 1 = every tick (old behaviour); 0 = never. Create-time
/// validation is unconditional regardless — a bad build must not survive
/// construction.
static uint32_t validateInterval()
{
    static const uint32_t interval = [] {
        const char* raw = std::getenv("BLAST_VALIDATE_INTERVAL");
        if (raw == nullptr)
        {
            return 16u;
        }
        const long parsed = std::atol(raw);
        return parsed < 0 ? 16u : static_cast<uint32_t>(parsed);
    }();
    return interval;
}

/// A/B for dropping contacts aimed at bodies that have no bonds (default
/// ON). Not an approximation -- see the comment at the drop site -- but
/// switchable so one binary produces both arms of the measurement.
/// A/B for the flat per-node bondless flags (default ON). One binary, two
/// arms: the alternative is comparing two builds, which reintroduces build
/// identity as a confounder -- the mistake that cost a day on this tree.
/// DEFAULT ON, audited.
///
/// queueContact's hot path used to do m_nodes[nodeIndex].body followed by
/// owner->nodes.size() -- a random index into an 87k-entry array then a
/// dependent deref into a separately allocated BodyState. Two cache misses
/// per queued contact, ~72k a tick at grid 2, paid purely to DECIDE, with
/// "skip" the answer 65% of the time. This is one byte from a flat array
/// that fits in L2.
///
/// It shipped default-off for two rounds because the audit caught a real
/// bug: refreshNodeBondless() ran in beginTick, while body composition
/// changes in endTick, and the contacts that consult the flags arrive in
/// the NEXT step's callback -- before that beginTick. So on every tick
/// after a fracture the flags described the previous composition:
///
///   101,135,704 checks, 98,633 mismatches (0.0975%, the fracture rate)
///
/// and a contact wrongly classed as bondless is DROPPED before the solver
/// sees it, i.e. lost load on freshly fractured chunks. Rebuilding after
/// endTick instead, so the flags describe the composition the contacts will
/// actually be resolved against:
///
///   152,193,650 checks, 0 mismatches
///
/// Measured win, matched buckets, n=5242 per arm: cb_queue -39.7% p50,
/// cb_tick -26.7%, physx_step -15.8%. That last one is the interesting
/// number -- the contact callback runs on the host thread INSIDE
/// fetchResults, so a figure labelled "PhysX" has always contained our code.
///
/// BLAST_NODE_BONDLESS_FLAT=0 restores the two derefs;
/// BLAST_NODE_BONDLESS_VERIFY=1 re-runs the audit.
static bool flatBondlessFlags()
{
    static const bool enabled = [] {
        const char* raw = std::getenv("BLAST_NODE_BONDLESS_FLAT");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return enabled;
}

static bool skipBondlessContacts()
{
    static const bool enabled = [] {
        const char* raw = std::getenv("BLAST_SKIP_BONDLESS_CONTACTS");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return enabled;
}

/// A/B for the per-body contacted-actor hoist (default ON). One binary, two
/// arms: separate builds would reintroduce build identity as a confounder,
/// which is how a whole afternoon of measurements went wrong on this tree.
static bool contactedActorHoist()
{
    static const bool enabled = [] {
        const char* raw = std::getenv("BLAST_CONTACTED_ACTOR_HOIST");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return enabled;
}
constexpr uint32_t MAX_GPU_CONVEX_POINTS = 64;
constexpr float MIN_MASS = 1.0e-6f;
using TelemetryClock = std::chrono::steady_clock;

double elapsedMilliseconds(TelemetryClock::time_point start)
{
    return std::chrono::duration<double, std::milli>(TelemetryClock::now() - start).count();
}

StressVec3 toStress(const PxVec3& value)
{
    StressVec3 result{};
    result.x = value.x;
    result.y = value.y;
    result.z = value.z;
    return result;
}

PxVec3 fromStress(const StressVec3& value)
{
    return PxVec3(value.x, value.y, value.z);
}

float vectorLength(const PxVec3& value)
{
    return std::sqrt(value.magnitudeSquared());
}

bool finitePositive(const PxVec3& value)
{
    return value.isFinite() && value.x > 0.0f && value.y > 0.0f && value.z > 0.0f;
}

struct SceneWriteLock
{
    explicit SceneWriteLock(PxScene& scene)
        : m_scene(scene)
    {
        m_scene.lockWrite(__FILE__, __LINE__);
    }

    ~SceneWriteLock()
    {
        m_scene.unlockWrite();
    }

    PxScene& m_scene;
};

struct NodeSnapshot
{
    PxTransform shapeWorldPose{PxIdentity};
    PxVec3 position{0.0f};
    PxVec3 pointVelocity{0.0f};
    bool valid{false};
};

struct ParentMotion
{
    ExtStressPhysXId bodyId{0};
    uint32_t actorIndex{INVALID_INDEX};
    PxTransform pose{PxIdentity};
    PxVec3 centerOfMassWorld{0.0f};
    PxVec3 linearVelocity{0.0f};
    PxVec3 angularVelocity{0.0f};
};

struct ChildPlan
{
    uint32_t actorIndex{INVALID_INDEX};
    std::vector<uint32_t> nodes;
    PxVec3 fitCenter{0.0f};
    PxVec3 fitVelocity{0.0f};
    PxVec3 fitAngularVelocity{0.0f};
    bool reuse{false};
};

} // namespace

ExtStressPhysXTelemetry::ExtStressPhysXTelemetry()
    : ticks(0)
    , contactsQueued(0)
    , contactsProcessed(0)
    , contactsDropped(0)
    , sleepingActorsSkipped(0)
    , gravityQuietSkips(0)
    , splits(0)
    , bodiesCreated(0)
    , bodiesReused(0)
    , bodiesRecycled(0)
    , shapesMigrated(0)
    , convexPointLimitRejections(0)
    , convexCookingFailures(0)
    , mappingValidationFailures(0)
    , lookupTableDrifts(0)
    , bodyCount(0)
    , awakeDynamicBodyCount(0)
    , overstressedBondCount(0)
    , solverIslandCount(0)
    , solverIslandsSkipped(0)
    , maxSplitWorldPositionDrift(0.0f)
    , maxSplitPointVelocityDrift(0.0f)
    , contactProcessingMilliseconds(0.0)
    , gravityMilliseconds(0.0)
    , stressSolveMilliseconds(0.0)
    , fractureTopologyMilliseconds(0.0)
    , fractureGenerateMilliseconds(0.0)
    , fracturePrepMilliseconds(0.0)
    , fractureApplyMilliseconds(0.0)
    , fractureSceneMilliseconds(0.0)
    , fractureRebuildMilliseconds(0.0)
    , mappingValidationMilliseconds(0.0)
    , gpuStressSolveMilliseconds(0.0)
    , gpuStressHostToDeviceBytes(0)
    , gpuStressDeviceToHostBytes(0)
    , chunksCrushed(0)
    , crushedMassKg(0.0)
    , crushedVolumeM3(0.0)
    , nodesAtCrushYield(0)
    , peakCrushUtilisation(0.0f)
    , debrisBodiesSpawned(0)
    , crushResistanceJoules(0.0)
    , crushResistanceImpulses(0)
    , extraSolveUpdates(0)
    , unconvergedTicks(0)
    , resimulationCaptures(0)
    , resimulationRestores(0)
    , resimulationBodiesRestored(0)
    , resimulationBodiesRederived(0)
    , resimulationBodiesSkipped(0)
    , resimulationCaptureMilliseconds(0.0)
    , resimulationRestoreMilliseconds(0.0)
    , resimulationMaxRederivedDriftMeters(0.0f)
    , lastError(ExtStressPhysXError::None)
    , lastErrorNode(INVALID_INDEX)
{
}

class ExtStressPhysXDestructibleImpl final : public ExtStressPhysXDestructible
{
public:
    explicit ExtStressPhysXDestructibleImpl(const ExtStressPhysXDesc& desc)
        : m_physics(*desc.physics)
        , m_scene(*desc.scene)
        , m_material(*desc.material)
        , m_worldTransform(desc.worldTransform)
        , m_settings(desc.settings)
        , m_errorCallback(desc.errorCallback)
        , m_errorUserData(desc.errorUserData)
    {
    }

    ~ExtStressPhysXDestructibleImpl() override
    {
        destroy();
    }

    bool initialize(const ExtStressPhysXDesc& desc)
    {
        if (!desc.nodes || desc.nodeCount == 0 || !desc.bonds || desc.bondCount == 0 ||
            !desc.worldTransform.isValid())
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "A destructible needs valid nodes, bonds, and a world transform.");
        }

        // The material table is the ONLY strength authoring surface: bond area
        // is geometry. Requiring the table forces every structure to state
        // what it is made of instead of inheriting silent placeholder limits.
        if (!desc.stressMaterials || desc.stressMaterialCount == 0)
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "A destructible requires a stress material table (>= 1 entry).");
        }
        m_materialCount = desc.stressMaterialCount;
        m_materials.assign(desc.stressMaterials, desc.stressMaterials + desc.stressMaterialCount);
        m_materialDescs.resize(desc.stressMaterialCount);
        for (uint32_t i = 0; i < desc.stressMaterialCount; ++i)
        {
            const ExtStressPhysXMaterial& source = desc.stressMaterials[i];
            if (!std::isfinite(source.compressionElasticLimit)
                || !std::isfinite(source.compressionFatalLimit)
                || source.compressionElasticLimit < 0.0f
                || source.compressionFatalLimit < source.compressionElasticLimit)
            {
                return fail(
                    ExtStressPhysXError::InvalidDescriptor,
                    i,
                    "Material compression limits must be finite, non-negative, "
                    "and fatal >= elastic.");
            }
            ExtStressMaterialDesc& target = m_materialDescs[i];
            target.compression_elastic_limit = source.compressionElasticLimit;
            target.compression_fatal_limit = source.compressionFatalLimit;
            target.tension_elastic_limit = source.tensionElasticLimit;
            target.tension_fatal_limit = source.tensionFatalLimit;
            target.shear_elastic_limit = source.shearElasticLimit;
            target.shear_fatal_limit = source.shearFatalLimit;
            target.elastic_modulus_pa = source.elasticModulusPa;
            target.residual_area_fraction = source.residualAreaFraction;

            if (source.crushCapPressure > 0.0f)
            {
                if (!std::isfinite(source.crushCapPressure)
                    || !std::isfinite(source.crushCohesion) || source.crushCohesion < 0.0f
                    || !std::isfinite(source.crushFrictionSlope) || source.crushFrictionSlope < 0.0f
                    || !std::isfinite(source.crushEnergy) || source.crushEnergy <= 0.0f
                    || !std::isfinite(source.crushViscosity) || source.crushViscosity <= 0.0f)
                {
                    return fail(
                        ExtStressPhysXError::InvalidDescriptor,
                        i,
                        "Crush properties must be finite, non-negative, with crushEnergy > 0 "
                        "and crushViscosity > 0. "
                        "Set crushCapPressure <= 0 to disable crushing for this material.");
                }
                if (source.crushDebrisMassFraction < 0.0f || source.crushDebrisMassFraction > 1.0f)
                {
                    return fail(
                        ExtStressPhysXError::InvalidDescriptor,
                        i,
                        "crushDebrisMassFraction must be in [0, 1].");
                }
                m_crushEnabled = true;
            }
            target.crush_cap_pressure = source.crushCapPressure;
            target.crush_cohesion = source.crushCohesion;
            target.crush_friction_slope = source.crushFrictionSlope;
            target.crush_energy = source.crushEnergy;
            target.crush_viscosity = source.crushViscosity;
            target.crush_strain_rate_exponent = source.crushStrainRateExponent;
            target.crush_reference_strain_rate = source.crushReferenceStrainRate;
            target.crush_debris_mass_fraction = source.crushDebrisMassFraction;
            target.crush_debris_fragment_count = source.crushDebrisFragmentCount;
        }

        // Crushing needs an unreduced graph: reduction merges chunks into
        // aggregate solver nodes, so a per-chunk stress tensor would describe
        // the aggregate rather than the chunk. Report it rather than silently
        // producing a plausible wrong number.
        if (m_crushEnabled && desc.settings.graphReductionLevel > 0)
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "Chunk crushing requires graphReductionLevel 0.");
        }

        m_nodes.resize(desc.nodeCount);
        m_bonds.assign(desc.bonds, desc.bonds + desc.bondCount);
        m_nodeMaterials.assign(desc.nodeCount, 0);
        m_nodeStrainRates.assign(desc.nodeCount, 0.0f);
        m_nodeCharacteristicSize.assign(desc.nodeCount, 0.0f);
        m_nodeCrusher.assign(desc.nodeCount, CrusherTrack{});
        m_prevCrushDamage.assign(desc.nodeCount, 0.0f);

        std::vector<ExtStressNodeDesc> solverNodes(desc.nodeCount);
        for (uint32_t i = 0; i < desc.nodeCount; ++i)
        {
            const ExtStressPhysXNodeDesc& source = desc.nodes[i];
            if (!source.centroid.isFinite() || !std::isfinite(source.mass) || source.mass < 0.0f ||
                !std::isfinite(source.volume) || source.volume <= 0.0f ||
                !source.geometry.localPose.isValid())
            {
                return fail(
                    ExtStressPhysXError::InvalidDescriptor,
                    i,
                    "Node coordinates, mass, volume, and geometry pose must be valid.");
            }

            if (source.material >= m_materialCount)
            {
                return fail(
                    ExtStressPhysXError::InvalidDescriptor,
                    i,
                    "Node material index is outside the stress material table.");
            }

            NodeState& node = m_nodes[i];
            node.mass = source.mass;
            node.volume = source.volume;
            node.centroid = source.centroid;
            node.shapeId = m_nextShapeId++;
            node.material = source.material;
            m_nodeMaterials[i] = source.material;
            // Characteristic length for turning a contact closing SPEED into a
            // strain RATE. Cube root of volume, so it is geometry-derived
            // rather than another authored knob.
            m_nodeCharacteristicSize[i] = std::cbrt(source.volume);

            ExtStressNodeDesc& solverNode = solverNodes[i];
            solverNode.centroid = toStress(source.centroid);
            solverNode.mass = source.mass;
            solverNode.volume = source.volume;
            // Rotational inertia from the chunk's REAL shape. The solver's own
            // fallback is a sphere of equal volume, which mis-weights the
            // moment balance for the flat, wide pieces buildings are made of:
            // a slab's inertia about its flat axis and about its edge differ
            // by an order of magnitude, and the sphere splits the difference.
            //
            // Box: average of the three principal inertias,
            // (Ix+Iy+Iz)/3 = m(X^2+Y^2+Z^2)/18 for full extents X,Y,Z -- an
            // isotropic scalar because that is what the solver stores. Hulls
            // use their bounding box, which errs slightly large, in the same
            // direction as the hull's own convexity.
            {
                physx::PxVec3 halfExtents = source.geometry.halfExtents;
                if (source.geometry.type == ExtStressPhysXGeometryType::Convex
                    && source.geometry.convexPoints != nullptr
                    && source.geometry.convexPointCount > 0)
                {
                    physx::PxVec3 lo = source.geometry.convexPoints[0];
                    physx::PxVec3 hi = lo;
                    for (uint32_t pt = 1; pt < source.geometry.convexPointCount; ++pt)
                    {
                        lo = lo.minimum(source.geometry.convexPoints[pt]);
                        hi = hi.maximum(source.geometry.convexPoints[pt]);
                    }
                    halfExtents = (hi - lo) * 0.5f;
                }
                const physx::PxVec3 full = halfExtents * 2.0f;
                solverNode.inertia = source.mass > 0.0f
                    ? source.mass * (full.x*full.x + full.y*full.y + full.z*full.z) / 18.0f
                    : 0.0f;
            }
        }

        std::vector<ExtStressBondDesc> solverBonds(desc.bondCount);
        for (uint32_t i = 0; i < desc.bondCount; ++i)
        {
            const ExtStressPhysXBondDesc& source = desc.bonds[i];
            if (source.node0 >= desc.nodeCount || source.node1 >= desc.nodeCount ||
                source.node0 == source.node1 || !source.centroid.isFinite() ||
                !source.normal.isFinite() || !std::isfinite(source.area) || source.area <= 0.0f)
            {
                return fail(
                    ExtStressPhysXError::InvalidDescriptor,
                    INVALID_INDEX,
                    "Bond endpoints and geometry must be valid.");
            }
            if (source.material >= m_materialCount)
            {
                return fail(
                    ExtStressPhysXError::InvalidDescriptor,
                    i,
                    "Bond material index is outside the stress material table.");
            }
            ExtStressBondDesc& bond = solverBonds[i];
            bond.centroid = toStress(source.centroid);
            bond.normal = toStress(source.normal);
            bond.area = source.area;
            bond.node0 = source.node0;
            bond.node1 = source.node1;
            bond.material = source.material;
        }

        ExtStressSolverSettingsDesc solverSettings{};
        solverSettings.max_solver_iterations_per_frame = m_settings.maxSolverIterationsPerFrame;
        solverSettings.graph_reduction_level = m_settings.graphReductionLevel;

        m_solver = ext_stress_solver_create(
            solverNodes.data(),
            static_cast<uint32_t>(solverNodes.size()),
            solverBonds.data(),
            static_cast<uint32_t>(solverBonds.size()),
            m_materialDescs.data(),
            static_cast<uint32_t>(m_materialDescs.size()),
            &solverSettings);
        if (m_solver && m_crushEnabled)
        {
            ext_stress_solver_set_node_materials(
                m_solver,
                m_nodeMaterials.data(),
                static_cast<uint32_t>(m_nodeMaterials.size()));
        }
        if (!m_solver)
        {
            return fail(
                ExtStressPhysXError::SolverCreationFailed,
                INVALID_INDEX,
                "The C stress-solver bridge could not create its family.");
        }
        ext_stress_solver_set_island_aware(m_solver, m_settings.islandAware ? 1 : 0);
        ext_stress_solver_set_skip_settled(m_solver, m_settings.skipSettledIslands ? 1 : 0);
        ext_stress_solver_set_skip_stable_unconverged(
            m_solver, m_settings.skipStableUnconverged ? 1 : 0);
        // Env override so the GPU stress solver -- the one that actually
        // ships -- can be exercised by tests that build their own descriptor.
        // It must be applied HERE, before the block below reads it: the
        // solver's GPU backend is chosen from the descriptor at creation, so
        // calling setGpuAccelerated afterwards reports success and changes
        // nothing (measured: set=1, active=0).
        //
        // Requires PhysicsMode::Gpu as well, because the CUDA context comes
        // from the PhysX scene's context manager.
        if (const char* gpuEnv = std::getenv("BLAST_STRESS_GPU"))
        {
            m_settings.gpuStressSolver = (gpuEnv[0] != '0');
        }
        // Separate knob on purpose. gpuStressMinimumBondCount defaults to 4096,
        // which is right for production and above any test-scene size, so a
        // test that asks for the GPU and nothing else still gets the CPU. That
        // combination is now WARNED about rather than silent, so the two are
        // kept independent: you can reproduce the trap deliberately.
        if (const char* minEnv = std::getenv("BLAST_STRESS_GPU_MIN_BONDS"))
        {
            m_settings.gpuStressMinimumBondCount =
                static_cast<uint32_t>(std::strtoul(minEnv, nullptr, 10));
        }
        if (m_settings.gpuStressSolver)
        {
            physx::PxCudaContextManager* cudaManager = m_scene.getCudaContextManager();
            if (!cudaManager)
            {
                return fail(
                    ExtStressPhysXError::SolverCreationFailed,
                    INVALID_INDEX,
                    "CUDA stress solver requires the PhysX CUDA context manager.");
            }
            ext_stress_solver_set_gpu_cuda_context(
                m_solver,
                reinterpret_cast<void*>(cudaManager->getContext()));
        }
        ext_stress_solver_set_gpu_minimum_bond_count(
            m_solver,
            m_settings.gpuStressMinimumBondCount);
        if (!ext_stress_solver_set_gpu_accelerated(
                m_solver,
                m_settings.gpuStressSolver ? 1 : 0))
        {
            return fail(
                ExtStressPhysXError::SolverCreationFailed,
                INVALID_INDEX,
                "CUDA stress solver requested but unavailable.");
        }

        for (uint32_t i = 0; i < desc.nodeCount; ++i)
        {
            if (!createNodeShape(i, desc.nodes[i]))
            {
                return false;
            }
        }

        std::vector<ExtStressActor> actors(1);
        std::vector<uint32_t> actorNodes(desc.nodeCount);
        uint32_t actorCount = 0;
        uint32_t actorNodeCount = 0;
        if (ext_stress_solver_collect_actors(
                m_solver,
                actors.data(),
                static_cast<uint32_t>(actors.size()),
                actorNodes.data(),
                static_cast<uint32_t>(actorNodes.size()),
                &actorCount,
                &actorNodeCount) == 0 ||
            actorCount != 1 || !actors[0].nodes || actors[0].nodeCount != desc.nodeCount)
        {
            return fail(
                ExtStressPhysXError::SolverCreationFailed,
                INVALID_INDEX,
                "The stress bridge did not produce one initial actor.");
        }

        {
            SceneWriteLock lock(m_scene);
            auto body = createBody(m_worldTransform, actors[0].actorIndex);
            if (!body)
            {
                return false;
            }

            body->nodes.assign(actors[0].nodes, actors[0].nodes + actors[0].nodeCount);
            std::sort(body->nodes.begin(), body->nodes.end());
            for (uint32_t nodeIndex : body->nodes)
            {
                NodeState& node = m_nodes[nodeIndex];
                node.shape->setLocalPose(desc.nodes[nodeIndex].geometry.localPose);
                if (!body->body->attachShape(*node.shape))
                {
                    return fail(
                        ExtStressPhysXError::ShapeCreationFailed,
                        nodeIndex,
                        "PhysX rejected a node shape attachment.");
                }
                node.body = body.get();
            }

            setBodyKinematic(*body, containsSupport(body->nodes));
            updateMassProperties(*body);
            m_scene.addActor(*body->body);
            // E3: m_bodyToId is maintained here, at the mutation site, not by
            // the whole-map rebuild in rebuildLookupTables (see there).
            m_bodyToId.emplace(body->body, body->bodyId);
            m_actorBodies.emplace(body->actorIndex, std::move(body));
            m_bondlessDirty = true;
        m_sortedBodiesValid = false;
        }

        rebuildLookupTables();
        m_telemetry.bodyCount = 1;
        return validateMappings();
    }

    void release() override
    {
        delete this;
    }

    bool queueContact(const ExtStressPhysXContact& contact) override
    {
        uint32_t nodeIndex = INVALID_INDEX;
        if (contact.nodeIndex != INVALID_INDEX)
        {
            // Host pre-resolved via nodeForShape this tick; the map lookup
            // below is exactly what produced it, so skipping is value-
            // identical.
            nodeIndex = contact.nodeIndex;
        }
        else if (contact.shapeId != 0)
        {
            const auto found = m_shapeIdToNode.find(contact.shapeId);
            if (found != m_shapeIdToNode.end())
            {
                nodeIndex = found->second;
            }
        }
        else if (contact.shape)
        {
            const auto found = m_shapeToNode.find(contact.shape);
            if (found != m_shapeToNode.end())
            {
                nodeIndex = found->second;
            }
        }

        if (nodeIndex == INVALID_INDEX || !contact.worldPosition.isFinite() ||
            !contact.worldImpulse.isFinite())
        {
            ++m_telemetry.contactsDropped;
            return false;
        }

        // A contact on a body with no bonds cannot do anything.
        //
        // Islands are built by walking BONDS (stress.cpp): a node with no
        // bond never joins one, so it is not in any island, the CG solve
        // never visits it, and no bond of its can be overstressed because it
        // has none. Every stage this contact would pass through -- the queue
        // push, the pose transform, the three array appends, the solver's
        // per-force accumulation -- is work whose only possible outcome is
        // zero. Dropping it is not an approximation; it removes arithmetic
        // that provably cannot change a result.
        //
        // Measured on a grid-2 collapse: 92% of AWAKE bodies are single-node
        // debris, and 65% of queued contacts are aimed at them.
        //
        // Wake is handled before the skip: a single-node body still needs to
        // be woken by an impact, and that is the one effect a contact on it
        // legitimately has.
        // One byte read, not two pointer chases.
        //
        // This test used to be m_nodes[nodeIndex].body followed by
        // owner->nodes.size(), which is a random index into an 87k-entry node
        // array and then a dependent deref into a separately allocated
        // BodyState -- two cache misses, paid for EVERY queued contact
        // (~72k a tick at grid 2) purely to decide, and 65% of the time the
        // answer is "skip and return".
        //
        // m_nodeBondless is a flat byte per node, rebuilt sequentially once
        // per tick in refreshNodeBondless(). 87 KB fits in L2, the rebuild is
        // a linear walk over bodies rather than random access, and the
        // decision it encodes is bit-identical to the two derefs it replaces.
        if (skipBondlessContacts() && !contact.wake)
        {
            bool bondless = false;
            if (flatBondlessFlags())
            {
                bondless = nodeIndex < m_nodeBondless.size()
                    && m_nodeBondless[nodeIndex] != 0;
                // The flat array is only an optimisation if it encodes the
                // SAME predicate. BLAST_NODE_BONDLESS_VERIFY=1 evaluates both
                // and counts disagreements; it must stay 0. Same discipline as
                // the node cache, which ran 75M checks clean before shipping.
                static const bool verify = [] {
                    const char* raw = std::getenv("BLAST_NODE_BONDLESS_VERIFY");
                    return raw != nullptr && std::string(raw) != "0";
                }();
                if (verify && nodeIndex < m_nodes.size())
                {
                    const BodyState* owner = m_nodes[nodeIndex].body;
                    const bool truth = owner != nullptr && owner->nodes.size() == 1;
                    ++m_telemetry.bondlessVerifyChecks;
                    if (truth != bondless)
                    {
                        ++m_telemetry.bondlessVerifyMismatches;
                    }
                }
            }
            else if (nodeIndex < m_nodes.size())
            {
                const BodyState* owner = m_nodes[nodeIndex].body;
                bondless = owner != nullptr && owner->nodes.size() == 1;
            }
            if (bondless)
            {
                ++m_telemetry.bondlessContactsSkipped;
                return false;
            }
        }

        QueuedContact queued;
        queued.nodeIndex = nodeIndex;
        queued.position = contact.worldPosition;
        queued.impulse = contact.worldImpulse;
        queued.relativeVelocity =
            contact.worldRelativeVelocity.isFinite() ? contact.worldRelativeVelocity : PxVec3(0.0f);
        queued.otherActor = contact.otherActor;
        queued.wake = contact.wake;
        m_contacts.push_back(queued);
        ++m_telemetry.contactsQueued;
        return true;
    }

    bool isNodeBondless(uint32_t nodeIndex) const override
    {
        // Must mirror queueContact's skip EXACTLY, including its gates: if
        // skipping is disabled, or the flat array is disabled, the answer has
        // to be "not bondless" so the caller takes the normal path and
        // queueContact makes the real decision.
        if (!skipBondlessContacts() || !flatBondlessFlags())
        {
            return false;
        }
        return nodeIndex < m_nodeBondless.size() && m_nodeBondless[nodeIndex] != 0;
    }

    /// Contacts the HOST skipped on our behalf, so the published
    /// bondlessContactsSkipped stays comparable across the change.
    void noteBondlessSkipped(uint32_t count) override
    {
        m_telemetry.bondlessContactsSkipped += count;
    }

    uint32_t nodeForShape(const PxShape* shape) const override
    {
        if (shape == nullptr)
        {
            return INVALID_INDEX;
        }
        const auto found = m_shapeToNode.find(shape);
        return found == m_shapeToNode.end() ? INVALID_INDEX : found->second;
    }

    bool queueContact(
        const PxShape& shape,
        const PxVec3& worldPosition,
        const PxVec3& worldImpulse) override
    {
        ExtStressPhysXContact contact;
        contact.shape = &shape;
        contact.worldPosition = worldPosition;
        contact.worldImpulse = worldImpulse;
        return queueContact(contact);
    }

    /// Gravity injection from a caller-supplied snapshot. No PhysX calls.
    /// Default ON. Off-switch for the quiet-tick gravity skip below.
    static bool gravityQuietSkipEnabled()
    {
        static const bool enabled = [] {
            const char* raw = std::getenv("BLAST_GRAVITY_QUIET_SKIP");
            return raw == nullptr || std::string(raw)[0] != '0';
        }();
        return enabled;
    }

    /// True when this tick's stress inputs are provably bit-identical to last
    /// tick's, so recomputing them cannot change a single bit.
    ///
    /// This is memoization on an exact key, not an approximation and not a
    /// tolerance. The argument is a closed enumeration of what can write the
    /// per-node load array `localVel`, which is the solver's entire input:
    ///
    ///   1. contact forces  -- submitted only when m_contactNodeIndices is
    ///                         non-empty, checked below;
    ///   2. gravity         -- a pure function of worldGravity (constant) and
    ///                         snapshot.globalPose.q, both in the key;
    ///   3. centrifugal     -- a pure function of the pose and
    ///                         snapshot.angularVelocity, both in the key.
    ///
    /// There is no fourth writer: addNodeForce is the only function in the
    /// solver that assigns localVel, and these three are the only callers the
    /// adapter has. Everything else that acts on the city -- a shot, a player
    /// impulse -- goes through PhysX, which moves bodies, which changes the
    /// pose in the key and disengages the skip on the very next tick.
    ///
    /// So if the key matches, the loads this tick WOULD BE the same bits the
    /// solver is already holding. Nothing is being approximated away and no
    /// force is being dropped: the elastic response, the stress state and the
    /// fracture threshold are all evaluated exactly as they would have been.
    /// The only thing skipped is recomputing a value that is already correct.
    ///
    /// Comparing 108 poses to decide this costs nothing; what it buys is
    /// skipping gravity application across ~87,000 nodes. On an idle city that
    /// is ~1.0 ms/tick directly, and it unlocks another ~0.9 ms indirectly:
    /// addNodeForce is the only writer of localVel, so once gravity stops
    /// touching it the solver's walk-in skips itself too.
    ///
    /// Safety rests on the loads persisting. The solver keeps its velocity
    /// array between ticks and the walk-in is what overwrites it, so if both
    /// gravity and the walk-in sit out a tick, the solver simply re-solves the
    /// inputs it already had -- which are the correct ones for a scene that
    /// has not moved.
    bool snapshotUnchanged(const ExtStressPhysXBodySnapshot* bodies, uint32_t bodyCount)
    {
        // Both contact sets, because they are not the same set and only one
        // of them gates force submission. m_contactNodeIndices is what
        // actually guards add_all_forces; m_contactedActors is the woken-body
        // set. Requiring both empty is the exact condition plus a margin,
        // rather than a proxy that happens to correlate.
        if (!m_contactNodeIndices.empty() || !m_contactedActors.empty() ||
            bodyCount != m_lastSnapshotCount)
        {
            return false;
        }
        if (m_lastSnapshot.size() != bodyCount)
        {
            return false;
        }
        for (uint32_t i = 0; i < bodyCount; ++i)
        {
            const ExtStressPhysXBodySnapshot& now = bodies[i];
            const ExtStressPhysXBodySnapshot& was = m_lastSnapshot[i];
            if (now.bodyId != was.bodyId || now.actorIndex != was.actorIndex ||
                now.nodeCount != was.nodeCount || now.kinematic != was.kinematic ||
                now.sleeping != was.sleeping)
            {
                return false;
            }
            // Bit-exact pose comparison. An approximate one would let a slow
            // drift accumulate silently under the skip.
            if (now.globalPose.p.x != was.globalPose.p.x ||
                now.globalPose.p.y != was.globalPose.p.y ||
                now.globalPose.p.z != was.globalPose.p.z ||
                now.globalPose.q.x != was.globalPose.q.x ||
                now.globalPose.q.y != was.globalPose.q.y ||
                now.globalPose.q.z != was.globalPose.q.z ||
                now.globalPose.q.w != was.globalPose.q.w ||
                now.angularVelocity.x != was.angularVelocity.x ||
                now.angularVelocity.y != was.angularVelocity.y ||
                now.angularVelocity.z != was.angularVelocity.z)
            {
                return false;
            }
        }
        return true;
    }

    void addGravityFromSnapshot(
        const PxVec3& worldGravity,
        const ExtStressPhysXBodySnapshot* bodies,
        uint32_t bodyCount)
    {
        static const bool skipSingleton = [] {
            const char* raw = std::getenv("BLAST_GRAVITY_SKIP_SINGLETON");
            return raw == nullptr || std::string(raw)[0] != '0';
        }();
        // m_loadsValid means "the solver already holds gravity loads computed
        // for exactly this snapshot". It goes true when we apply and STAYS
        // true while the snapshot is unchanged, because the solver keeps its
        // velocity array between ticks -- reapplying identical loads to
        // identical poses cannot change anything.
        //
        // It previously flipped -- `m_lastSnapshotApplied = !skipLoads` --
        // which quietly built an ALTERNATING skip: apply, skip, apply, skip.
        // It looked like it worked (idle did get faster) and it left exactly
        // half the benefit on the floor. The tell was that 50.0% of idle ticks
        // were expensive while nothing physical correlated with which ones --
        // no contact, no wake, no pose change, no body count change. A 50%
        // split with no physical correlate is a state machine, not a scene.
        const bool unchanged = gravityQuietSkipEnabled() && snapshotUnchanged(bodies, bodyCount);
        const bool skipLoads = unchanged && m_loadsValid;
        m_lastSnapshot.assign(bodies, bodies + bodyCount);
        m_lastSnapshotCount = bodyCount;
        if (!skipLoads)
        {
            m_loadsValid = true;   // about to apply, for these exact poses
        }
        if (skipLoads)
        {
            ++m_telemetry.gravityQuietSkips;
        }
        for (uint32_t i = 0; i < bodyCount; ++i)
        {
            const ExtStressPhysXBodySnapshot& snapshot = bodies[i];
            // A single-node body has no bond to load: addGravity and
            // addCentrifugalAcceleration both return false untouched for
            // graphNodeCount <= 1 (provably -- validateMappings asserts the
            // node sets match). Skipping here saves the red-black-tree lookup
            // below for the majority population of a demolished city, which is
            // single-chunk debris. nodeCount is already on the row.
            if (skipSingleton && snapshot.nodeCount <= 1)
            {
                // The gravity walk is ALSO where awakeDynamicBodyCount and
                // sleepingActorsSkipped are tallied. Skipping without
                // preserving them silently un-counted every awake single-chunk
                // debris body: damage held while "awake" read 8x low, and a
                // whole bench comparison bucketed on that metric was garbage.
                // Replicate the original counting exactly; the row carries
                // everything the counters need.
                if (!snapshot.kinematic)
                {
                    if (snapshot.sleeping)
                    {
                        if (m_contactedActors.find(snapshot.actorIndex) ==
                            m_contactedActors.end())
                        {
                            ++m_telemetry.sleepingActorsSkipped;
                        }
                    }
                    else
                    {
                        ++m_telemetry.awakeDynamicBodyCount;
                    }
                }
                continue;
            }
            // Keyed by actorIndex: m_actorBodies is a map<actorIndex, BodyState>,
            // NOT by bodyId. Looking up by bodyId silently missed nearly every
            // body, so most of the structure got no gravity and it fractured
            // about half as much -- caught by the severed-tower bond count.
            const auto found = m_actorBodies.find(snapshot.actorIndex);
            if (found == m_actorBodies.end())
            {
                continue;  // belongs to another destructible
            }
            BodyState& body = *found->second;
            if (body.bodyId != snapshot.bodyId)
            {
                continue;  // stale snapshot row for a recycled actor slot
            }
            if (!snapshot.kinematic)
            {
                if (snapshot.sleeping)
                {
                    // A sleeping body that took contact load this tick still
                    // needs its weight: consumeContactsFromSnapshot has
                    // already pushed the support reaction into the solver, and
                    // gravity is the other half of that pair. Skipping it here
                    // would leave a resting pile loaded upward only.
                    if (m_contactedActors.find(body.actorIndex) ==
                        m_contactedActors.end())
                    {
                        ++m_telemetry.sleepingActorsSkipped;
                        continue;
                    }
                }
                else
                {
                    ++m_telemetry.awakeDynamicBodyCount;
                }
            }
            // Counters above are still tallied on a skipped tick -- they are
            // what "awake" and "sleeping skipped" report, and un-counting them
            // once already made a whole bench comparison garbage.
            if (skipLoads)
            {
                continue;
            }
            const PxVec3 localGravity = snapshot.globalPose.q.rotateInv(worldGravity);
            const StressVec3 bridgeGravity = toStress(localGravity);
            ext_stress_solver_add_actor_gravity(
                m_solver,
                body.actorIndex,
                &bridgeGravity);
            if (!snapshot.kinematic)
            {
                addCentrifugal(body, snapshot.globalPose, snapshot.angularVelocity);
            }
        }
    }

    /// consumeContacts() against snapshot poses, deferring wakeUp() to the
    /// caller because it is a scene write.
    struct QueuedContact
    {
        uint32_t nodeIndex{INVALID_INDEX};
        PxVec3 position{0.0f};
        PxVec3 impulse{0.0f};
        // Other body's velocity minus this one's at the contact point, world
        // space. Only its closing component is used, and only for crushing.
        PxVec3 relativeVelocity{0.0f};
        // The body pressing on us, when the host supplied it. Crush
        // resistance charges comminution work to this body.
        PxRigidActor* otherActor{nullptr};
        bool wake{true};
    };

    /**
    Record this contact's compaction strain rate on its node.

    The contact impulse direction is the contact normal up to sign, so the
    closing speed is the relative velocity projected onto it. Only CLOSING
    counts: a contact that is separating does no work on the chunk, and a
    resting contact that is neither does none either -- which is why a settled
    rubble pile accumulates no crush damage. Dividing by the chunk's
    characteristic size V^(1/3) turns a speed into a strain rate.

    The max (not the sum) over a node's contacts: several contact points from
    the same impact describe one closing event, and summing them would make
    crush damage scale with contact-point count rather than with the physics.
    */
    void accumulateCrushStrainRate(const QueuedContact& contact)
    {
        if (!m_crushEnabled || contact.nodeIndex >= m_nodeCharacteristicSize.size())
        {
            return;
        }
        const float impulseMagnitude = contact.impulse.magnitude();
        if (impulseMagnitude <= 0.0f)
        {
            return;
        }
        const PxVec3 normal = contact.impulse / impulseMagnitude;
        // Sign: `normal` is the impulse THIS body receives, which points the
        // same way the approaching body is travelling, and relativeVelocity is
        // "other minus this". So a closing contact gives a POSITIVE dot; a
        // separating one gives a negative dot and does no work on the chunk.
        const float closingSpeed = contact.relativeVelocity.dot(normal);
        if (closingSpeed > 0.0f)
        {
            const float size = m_nodeCharacteristicSize[contact.nodeIndex];
            if (size > 0.0f)
            {
                float& rate = m_nodeStrainRates[contact.nodeIndex];
                rate = std::max(rate, closingSpeed / size);
            }
        }

        // Remember the strongest external body pressing on this node this
        // tick: it is who pays the comminution bill. The strongest, not the
        // sum -- several contact points from one impactor describe one press,
        // and the payer is a body, not a contact.
        //
        // Deliberately NOT gated on the closing speed: relative velocities are
        // sampled after fetchResults, and a hard impact reads near zero there
        // because PhysX already stopped the impactor within the step. The
        // impulse direction still names the approach axis, and every path that
        // CHARGES the payer re-reads its live velocity at charge time -- so a
        // stale track can never over-extract, only fail to identify.
        if (m_settings.applyCrushResistance && contact.otherActor)
        {
            PxRigidBody* other = contact.otherActor->is<PxRigidBody>();
            // Our OWN bodies never pay: once a structure splits, its debris
            // pieces trade enormous internal contact impulses, and without
            // this exclusion the "dominant crusher" is invariably one of them
            // -- a sideways wall-on-wall press outbidding the actual impactor,
            // billing a flying plug along an axis it never travels. Internal
            // energy exchange is already inside the solve; only EXTERNAL
            // bodies owe comminution work.
            const bool internal = other != nullptr
                && m_bodyToId.find(static_cast<const PxRigidDynamic*>(
                       static_cast<const PxRigidBody*>(other)))
                       != m_bodyToId.end();
            if (other && !internal
                && !(other->getRigidBodyFlags() & PxRigidBodyFlag::eKINEMATIC)
                && impulseMagnitude > m_nodeCrusher[contact.nodeIndex].impulseMagnitude)
            {
                CrusherTrack& track = m_nodeCrusher[contact.nodeIndex];
                track.body = other;
                track.direction = normal;
                track.closingSpeed = std::max(closingSpeed, 0.0f);
                track.impulseMagnitude = impulseMagnitude;
                // The tick's dominant impactor. An impact's stress spreads
                // through bonds, so most chunks it comminutes never touch the
                // impactor -- but the energy still came from it, and it is who
                // pays when a crushed node has no contact of its own.
                if (impulseMagnitude > m_tickDominantCrusher.impulseMagnitude)
                {
                    m_tickDominantCrusher = track;
                }
            }
        }
    }

    /// Zero every node's strain rate. Rates describe ONE tick's contacts: a
    /// chunk that stops being pressed must stop accumulating crush damage.
    void resetCrushStrainRates()
    {
        if (m_crushEnabled)
        {
            std::fill(m_nodeStrainRates.begin(), m_nodeStrainRates.end(), 0.0f);
            std::fill(m_nodeCrusher.begin(), m_nodeCrusher.end(), CrusherTrack{});
            m_tickDominantCrusher = CrusherTrack{};
        }
    }

    /// Hand this tick's strain rates and timestep to the solver.
    void pushCrushStrainRates(float dt)
    {
        if (m_crushEnabled)
        {
            ext_stress_solver_set_node_strain_rates(
                m_solver,
                m_nodeStrainRates.data(),
                static_cast<uint32_t>(m_nodeStrainRates.size()),
                dt);
        }
    }

    void consumeContactsFromSnapshot(
        float dt,
        const ExtStressPhysXBodySnapshot* bodies,
        uint32_t bodyCount,
        ExtStressPhysXId* outWakeBodies,
        uint32_t wakeCapacity,
        uint32_t* outWakeCount)
    {
        uint32_t wakeCount = 0;
        // Written before the empty-queue early return, not after the loop:
        // this struct is not zero-initialised, so a tick with no contacts
        // left the field holding whatever was in that memory. It read ~1.1e9
        // in the first live reports, which is exactly the kind of number that
        // is obviously wrong -- the dangerous version of this bug is the one
        // that returns a plausible value.
        m_telemetry.singleNodeContacts = 0;
        resetCrushStrainRates();
        if (m_contacts.empty())
        {
            // Nothing to look anything up for. Building the index anyway cost
            // a hash insert per body per tick -- ten thousand of them on a
            // collapsing city -- to answer zero queries.
            m_contactNodeIndices.clear();
            m_contactLocalPositions.clear();
            m_contactLocalForces.clear();
            m_contactedActors.clear();
            pushCrushStrainRates(dt);
            if (outWakeCount != nullptr)
            {
                *outWakeCount = 0;
            }
            return;
        }
        m_contactedActors.clear();

        // Row hints replace the wholesale index: building m_snapshotByBodyId
        // was a hash insert per BODY per tick (eleven thousand on a collapsing
        // city) to answer per-CONTACT queries. The hint is stamped by
        // getBodySnapshots itself, so it is fresh the tick the snapshot was
        // produced; the bodyId verify makes a stale hint cost one map build,
        // never a wrong answer. BLAST_SNAPSHOT_ROW_HINT=0 restores the index.
        static const bool rowHint = [] {
            const char* raw = std::getenv("BLAST_SNAPSHOT_ROW_HINT");
            return raw == nullptr || std::string(raw)[0] != '0';
        }();
        bool indexBuilt = false;
        auto snapshotFor =
            [&](const BodyState& body) -> const ExtStressPhysXBodySnapshot* {
            if (rowHint && body.snapshotRow < bodyCount
                && bodies[body.snapshotRow].bodyId == body.bodyId)
            {
                return &bodies[body.snapshotRow];
            }
            if (!indexBuilt)
            {
                m_snapshotByBodyId.clear();
                for (uint32_t i = 0; i < bodyCount; ++i)
                {
                    m_snapshotByBodyId[bodies[i].bodyId] = &bodies[i];
                }
                indexBuilt = true;
            }
            const auto found = m_snapshotByBodyId.find(body.bodyId);
            return found == m_snapshotByBodyId.end() ? nullptr : found->second;
        };

        uint32_t singleNodeContacts = 0;
        m_contactNodeIndices.clear();
        m_contactLocalPositions.clear();
        m_contactLocalForces.clear();
        m_contactNodeIndices.reserve(m_contacts.size());
        m_contactLocalPositions.reserve(m_contacts.size() * 3);
        m_contactLocalForces.reserve(m_contacts.size() * 3);
        const uint64_t generation = ++m_contactGeneration;
        for (const QueuedContact& contact : m_contacts)
        {
            if (contact.nodeIndex >= m_nodes.size())
            {
                ++m_telemetry.contactsDropped;
                continue;
            }
            NodeState& node = m_nodes[contact.nodeIndex];
            if (!node.body || !node.body->body)
            {
                ++m_telemetry.contactsDropped;
                continue;
            }
            BodyState& body = *node.body;
            // A single-node body has no internal bonds, so no contact routed
            // into it can ever produce a fracture. Counted to size that.
            if (body.nodes.size() == 1)
            {
                ++singleNodeContacts;
            }
            // A body carrying contact load must also carry its weight this
            // tick, even if PhysX has it asleep: the contact pushes its
            // support nodes up, and without the matching gravity pulling the
            // rest down the solver sees a net upward load and reports
            // wrong-signed stress. addGravityFromSnapshot consults this.
            // Once per body per tick, not once per CONTACT. Measured live:
            // 170,781 queued contacts against at most a few thousand distinct
            // bodies, so this unordered_set was being hashed ~7x more often
            // than it could possibly learn anything. Idempotent, so the set
            // contents are identical -- including for a body whose snapshot
            // is missing below, which is why this uses its own generation
            // rather than the pose cache's.
            if (!contactedActorHoist())
            {
                m_contactedActors.insert(body.actorIndex);
            }
            else if (body.contactedActorGeneration != generation)
            {
                m_contactedActors.insert(body.actorIndex);
                body.contactedActorGeneration = generation;
            }
            const ExtStressPhysXBodySnapshot* found = snapshotFor(body);
            if (found == nullptr)
            {
                // Counted, not silent: a contact whose body has no snapshot row
                // would be lost damage, and contactsDropped is how that shows up.
                ++m_telemetry.contactsDropped;
                continue;
            }
            const ExtStressPhysXBodySnapshot& snapshot = *found;

            if (contact.wake && !snapshot.kinematic && snapshot.sleeping)
            {
                if (wakeCount < wakeCapacity && outWakeBodies != nullptr)
                {
                    outWakeBodies[wakeCount] = body.bodyId;
                }
                ++wakeCount;
            }

            if (body.contactGeneration != generation)
            {
                body.contactGlobalPose = snapshot.globalPose;
                body.contactGeneration = generation;
            }
            const PxTransform& pose = body.contactGlobalPose;
            const PxVec3 localPosition = pose.transformInv(contact.position);
            const PxVec3 localForce = pose.q.rotateInv(contact.impulse / dt);

            accumulateCrushStrainRate(contact);

            m_contactNodeIndices.push_back(contact.nodeIndex);
            m_contactLocalPositions.insert(
                m_contactLocalPositions.end(),
                {localPosition.x, localPosition.y, localPosition.z});
            m_contactLocalForces.insert(
                m_contactLocalForces.end(),
                {localForce.x, localForce.y, localForce.z});
        }
        m_telemetry.singleNodeContacts = singleNodeContacts;
        if (!m_contactNodeIndices.empty())
        {
            m_telemetry.contactsProcessed += ext_stress_solver_add_all_forces(
                m_solver,
                m_contactNodeIndices.data(),
                m_contactLocalPositions.data(),
                m_contactLocalForces.data(),
                static_cast<uint32_t>(m_contactNodeIndices.size()),
                0);
        }
        m_contacts.clear();
        if (outWakeCount != nullptr)
        {
            *outWakeCount = wakeCount;
        }
        pushCrushStrainRates(dt);
    }

    bool beginTickFromSnapshot(
        float dt,
        const PxVec3& worldGravity,
        const ExtStressPhysXBodySnapshot* bodies,
        uint32_t bodyCount,
        ExtStressPhysXId* outWakeBodies,
        uint32_t wakeCapacity,
        uint32_t* outWakeCount) override
    {
        ++m_telemetry.ticks;
        m_telemetry.awakeDynamicBodyCount = 0;
        if (outWakeCount != nullptr)
        {
            *outWakeCount = 0;
        }
        if (m_tickPhase != TickPhase::Idle)
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "beginTickFromSnapshot called before the previous tick completed.");
        }
        if (!m_solver || !std::isfinite(dt) || dt <= 0.0f || !worldGravity.isFinite())
        {
            m_contacts.clear();
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "beginTickFromSnapshot requires a finite positive dt and finite gravity.");
        }
        if (bodies == nullptr && bodyCount > 0)
        {
            m_contacts.clear();
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "beginTickFromSnapshot requires a body snapshot.");
        }

        TelemetryClock::time_point phaseStart = TelemetryClock::now();
        consumeContactsFromSnapshot(
            dt, bodies, bodyCount, outWakeBodies, wakeCapacity, outWakeCount);
        m_telemetry.contactProcessingMilliseconds += elapsedMilliseconds(phaseStart);

        phaseStart = TelemetryClock::now();
        addGravityFromSnapshot(worldGravity, bodies, bodyCount);
        m_telemetry.gravityMilliseconds += elapsedMilliseconds(phaseStart);
        m_tickDt = dt;
        // The solver charges bond damage per second of overload, so it needs
        // to know how long this tick is.
        ext_stress_solver_set_delta_time(m_solver, dt);
        m_tickPhase = TickPhase::Prepared;
        return true;
    }


    /// Flat per-node "the body owning me has exactly one node" flags.
    ///
    /// Rebuilt once per tick by walking bodies in order and stamping their
    /// nodes, which is sequential. The alternative -- asking per contact --
    /// is two dependent random derefs times the contact count, and the
    /// contact count is two orders of magnitude larger than the body count.
    static bool verifyBondlessFlags()
    {
        static const bool on = [] {
            const char* raw = std::getenv("BLAST_VERIFY_BONDLESS_FLAGS");
            return raw != nullptr && std::string(raw) == "1";
        }();
        return on;
    }

    void refreshNodeBondless()
    {
        if (!skipBondlessContacts() || !flatBondlessFlags())
        {
            return;
        }
        m_nodeBondless.assign(m_nodes.size(), 0u);
        for (const auto& entry : m_actorBodies)
        {
            const BodyState* body = entry.second.get();
            if (body == nullptr || body->nodes.size() != 1)
            {
                continue;
            }
            for (const uint32_t node : body->nodes)
            {
                if (node < m_nodeBondless.size())
                {
                    m_nodeBondless[node] = 1u;
                }
            }
        }
    }

    bool beginTick(float dt, const PxVec3& worldGravity) override
    {
        ++m_telemetry.ticks;
        m_telemetry.awakeDynamicBodyCount = 0;
        if (m_tickPhase != TickPhase::Idle)
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "beginTick called before the previous tick completed.");
        }
        if (!m_solver || !std::isfinite(dt) || dt <= 0.0f || !worldGravity.isFinite())
        {
            m_contacts.clear();
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "beginTick requires a finite positive dt and finite gravity.");
        }

        TelemetryClock::time_point phaseStart = TelemetryClock::now();
        consumeContacts(dt);
        m_telemetry.contactProcessingMilliseconds += elapsedMilliseconds(phaseStart);

        phaseStart = TelemetryClock::now();
        addGravity(worldGravity);
        m_telemetry.gravityMilliseconds += elapsedMilliseconds(phaseStart);
        m_tickDt = dt;
        // The solver charges bond damage per second of overload, so it needs
        // to know how long this tick is.
        ext_stress_solver_set_delta_time(m_solver, dt);
        m_tickPhase = TickPhase::Prepared;
        return true;
    }

    /// Telemetry pulled after a solve completes. Shared by the monolithic and
    /// split paths so the two cannot report different things.
    void collectSolveTelemetry()
    {
        // A GPU request that quietly ran on the CPU is the most expensive
        // failure mode here: every timing, every A/B, and every "the invariant
        // test passes on GPU" claim is then about the wrong solver. The
        // backend only exists after the first solve, so this is the earliest
        // point the question can be answered honestly.
        if (!m_gpuVerified)
        {
            m_gpuVerified = true;
            if (m_settings.gpuStressSolver
                && !ext_stress_solver_get_gpu_accelerated(m_solver))
            {
                const char* why = ext_stress_solver_gpu_inactive_reason(m_solver);
                std::fprintf(stderr,
                             "[NvBlastExtStressPhysX] WARNING: gpuStressSolver was "
                             "requested but the CUDA backend is NOT running -- this is "
                             "the CPU solver. Reason: %s\n",
                             why ? why : "unknown");
            }
        }
        m_telemetry.overstressedBondCount =
            ext_stress_solver_overstressed_bond_count(m_solver);
        m_telemetry.solverIslandCount = ext_stress_solver_island_count(m_solver);
        m_telemetry.solverIslandsSkipped = ext_stress_solver_islands_skipped(m_solver);
        m_telemetry.gpuStressSolveMilliseconds +=
            ext_stress_solver_gpu_solve_milliseconds(m_solver);
        m_telemetry.gpuStressHostWorkMilliseconds +=
            ext_stress_solver_gpu_host_work_milliseconds(m_solver);
        m_telemetry.stressImpulseCopyMilliseconds +=
            ext_stress_solver_impulse_copy_milliseconds(m_solver);
        m_telemetry.bondStressGroupsSkipped =
            ext_stress_solver_bond_stress_groups_skipped(m_solver);
        m_telemetry.bondStressGpuSkipped =
            ext_stress_solver_bond_stress_gpu_skipped(m_solver);
        m_telemetry.bondStressGpuRuns =
            ext_stress_solver_bond_stress_gpu_runs(m_solver);
        m_telemetry.bondStressParallelChecks =
            ext_stress_solver_bond_stress_parallel_checks(m_solver);
        m_telemetry.bondStressParallelMismatches =
            ext_stress_solver_bond_stress_parallel_mismatches(m_solver);
        m_telemetry.stressHostWalkInMilliseconds +=
            ext_stress_solver_host_walk_in_milliseconds(m_solver);
        m_telemetry.stressHostResetMilliseconds +=
            ext_stress_solver_host_reset_milliseconds(m_solver);
        m_telemetry.stressHostBondStressMilliseconds +=
            ext_stress_solver_host_bond_stress_milliseconds(m_solver);
        m_telemetry.stressHostNodeStressMilliseconds +=
            ext_stress_solver_host_node_stress_milliseconds(m_solver);
        m_telemetry.stressGraphSolveMilliseconds +=
            ext_stress_solver_graph_solve_milliseconds(m_solver);
        m_telemetry.stressInitializeMilliseconds +=
            ext_stress_solver_initialize_milliseconds(m_solver);
        m_telemetry.stressCalcErrorMilliseconds +=
            ext_stress_solver_calc_error_milliseconds(m_solver);
        m_telemetry.gpuStressHostBlockedMilliseconds +=
            ext_stress_solver_gpu_host_blocked_milliseconds(m_solver);
        m_telemetry.gpuStressHostToDeviceBytes +=
            ext_stress_solver_gpu_host_to_device_bytes(m_solver);
        m_telemetry.gpuStressDeviceToHostBytes +=
            ext_stress_solver_gpu_device_to_host_bytes(m_solver);
    }

    /// Split solveTick, for callers that want to fan the bond-stress strips
    /// of EVERY structure out in one flat dispatch instead of one per
    /// structure. solveTickBeginSplit runs the CG solve and stops; the caller
    /// then drives bondStressStrip(i) for i in [0, stripCount) and calls
    /// solveTickFinishSplit.
    ///
    /// Refuses when unconvergedExtraUpdates > 0. That loop re-runs the whole
    /// update -- CG solve AND bond stress -- until convergence, so a split
    /// that hoists bond stress out of it would silently change how many times
    /// each half runs. Production sets 0 and takes the split; anything else
    /// falls back to the monolithic path rather than quietly meaning
    /// something different.
    bool supportsSplitSolve() const override
    {
        return m_settings.unconvergedExtraUpdates == 0;
    }

    uint32_t bondStressStripCount() const override
    {
        return ext_stress_solver_bond_stress_strip_count(m_solver);
    }

    bool solveTickBeginSplit() override
    {
        if (!supportsSplitSolve())
        {
            return false;
        }
        if (m_tickPhase != TickPhase::Prepared)
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "solveTickBeginSplit requires a successful beginTick.");
        }
        m_splitPhaseStart = TelemetryClock::now();
        ext_stress_solver_set_defer_bond_stress(m_solver, 1);
        ext_stress_solver_update(m_solver);
        return true;
    }

    void bondStressStrip(uint32_t stripIdx) override
    {
        ext_stress_solver_bond_stress_strip(m_solver, stripIdx);
    }

    bool solveTickFinishSplit() override
    {
        ext_stress_solver_bond_stress_complete(m_solver);
        ext_stress_solver_set_defer_bond_stress(m_solver, 0);
        collectSolveTelemetry();
        drainCrushedNodes();
        m_telemetry.stressSolveMilliseconds += elapsedMilliseconds(m_splitPhaseStart);
        m_tickPhase = TickPhase::Solved;
        return true;
    }

    bool solveTick() override
    {
        if (m_tickPhase != TickPhase::Prepared)
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "solveTick requires a successful beginTick.");
        }
        const TelemetryClock::time_point phaseStart = TelemetryClock::now();
        ext_stress_solver_update(m_solver);
        // Pursue equilibrium. See unconvergedExtraUpdates: an unconverged
        // island cannot earn the settled skip AND its residual reads as bond
        // stress, so under-solving costs every tick and breaks phantom bonds.
        for (uint32_t extra = 0;
             extra < m_settings.unconvergedExtraUpdates
             && ext_stress_solver_converged(m_solver) == 0;
             ++extra)
        {
            ext_stress_solver_update(m_solver);
            ++m_telemetry.extraSolveUpdates;
            m_telemetry.gpuStressHostWorkMilliseconds +=
                ext_stress_solver_gpu_host_work_milliseconds(m_solver);
            m_telemetry.gpuStressHostBlockedMilliseconds +=
                ext_stress_solver_gpu_host_blocked_milliseconds(m_solver);
        }
        if (ext_stress_solver_converged(m_solver) == 0)
        {
            ++m_telemetry.unconvergedTicks;
        }

        collectSolveTelemetry();
        // The last untimed block inside solveTick. Named because solve's
        // remainder was 48.4% and every named candidate so far came back
        // negligible; a remainder that large has to be SOMETHING.
        const TelemetryClock::time_point drainStart = TelemetryClock::now();
        drainCrushedNodes();
        m_telemetry.stressDrainMilliseconds += elapsedMilliseconds(drainStart);
        m_telemetry.stressSolveMilliseconds += elapsedMilliseconds(phaseStart);
        m_tickPhase = TickPhase::Solved;
        return true;
    }

    /**
    Move newly pulverized nodes out of the solver and into the adapter's own
    pending list. Draining here rather than in endTick keeps solveTick free of
    PhysX calls, which is what lets it run concurrently across destructibles.
    */
    void drainCrushedNodes()
    {
        if (!m_crushEnabled)
        {
            return;
        }
        m_crushScratch.resize(m_nodes.size());
        uint32_t drained = 0;
        while ((drained = ext_stress_solver_get_crushed_nodes(
                    m_solver,
                    m_crushScratch.data(),
                    static_cast<uint32_t>(m_crushScratch.size()))) > 0)
        {
            m_pendingCrushedNodes.insert(
                m_pendingCrushedNodes.end(),
                m_crushScratch.begin(),
                m_crushScratch.begin() + drained);
            if (drained < m_crushScratch.size())
            {
                break;
            }
        }

        // Report how many chunks are currently past their yield surface but not
        // yet fully comminuted: the crush analogue of the overstressed bond
        // count, and the number to watch when tuning.
        m_nodeDamageScratch.resize(m_nodes.size());
        const uint32_t written = ext_stress_solver_get_node_crush_damage(
            m_solver,
            m_nodeDamageScratch.data(),
            static_cast<uint32_t>(m_nodeDamageScratch.size()));
        uint32_t yielding = 0;
        for (uint32_t i = 0; i < written; ++i)
        {
            if (m_nodeDamageScratch[i] > 0.0f && m_nodeDamageScratch[i] < 1.0f)
            {
                ++yielding;
            }
        }
        m_telemetry.nodesAtCrushYield = yielding;

        // Running peak utilisation. Without this a run that crushes nothing is
        // indistinguishable from a run that came nowhere near crushing, and
        // there is no way to author toward the behaviour you want.
        m_nodeUtilisationScratch.resize(m_nodes.size());
        const uint32_t utilisationCount = ext_stress_solver_get_node_crush_utilisation(
            m_solver,
            m_nodeUtilisationScratch.data(),
            static_cast<uint32_t>(m_nodeUtilisationScratch.size()));
        for (uint32_t i = 0; i < utilisationCount; ++i)
        {
            m_telemetry.peakCrushUtilisation =
                std::max(m_telemetry.peakCrushUtilisation, m_nodeUtilisationScratch[i]);
        }
    }

    /**
    Charge this tick's comminution work to the bodies that did the crushing.

    Each node's damage increment dD dissipated dD * crushEnergy * volume of
    work grinding the material. That work is extracted from the strongest
    external contactor's kinetic energy along the closing axis: the impulse J
    on a payer of mass M closing at speed v that removes energy dE is

        J = M * (v - sqrt(v^2 - 2*dE/M)),   capped at M*v

    -- the cap is what makes this a resistance and not a spring: it can bring
    the crusher to a stop but can never bounce it. The momentum-conserving
    reaction goes to the chunk's own body (a no-op while the structure is
    kinematic, real once it is debris).

    Runs BEFORE fracture applies, while every damaged chunk still has its
    shape and pose. Bond-borne crushes with no external contactor charge
    nobody: their load path is the structure itself and the reaction is
    already inside the solve. Must run in endTick -- impulses are scene
    writes, and solveTick is the phase that must stay free of them.
    */
    /**
    Levy the deferred comminution charges after a resim restore.

    Runs under the scene write lock. Velocities are re-read fresh, but impulses
    applied here only land at the next simulate, so a per-payer running
    velocity estimate is kept locally -- several chunks charged against one
    ball must share ONE kinetic-energy budget, or the sum of individually
    clamped impulses would reverse it (the exact bug the per-tick path fixed,
    one level up).
    */
    void levyPendingResistance()
    {
        if (!m_pendingResistanceArmed || m_pendingResistance.empty())
        {
            return;
        }
        m_pendingResistanceArmed = false;

        std::unordered_map<PxRigidBody*, PxVec3> velocityEstimate;
        for (const PendingResistance& pending : m_pendingResistance)
        {
            if (!pending.payer || pending.payer->getScene() != &m_scene
                || pending.work <= 0.0f)
            {
                continue;
            }
            const float mass = pending.payer->getMass();
            if (mass <= 0.0f)
            {
                continue;
            }

            auto estimate = velocityEstimate.find(pending.payer);
            if (estimate == velocityEstimate.end())
            {
                estimate = velocityEstimate
                               .emplace(pending.payer, pending.payer->getLinearVelocity())
                               .first;
            }

            // Charge OPPOSING THE PAYER'S MOTION, not along the recorded
            // contact axis. The recorded axis is whichever contact impulse
            // happened to be strongest -- late in a penetration that is a
            // grazing, friction-dominated side contact, and projecting onto
            // it voids the bill on exactly the impacts that owe the most.
            // Comminution resistance is drag-like: it acts against wherever
            // the penetrator is actually going, and charging that way also
            // makes reversal impossible by construction.
            const PxVec3 velocity = estimate->second;
            const float speed = velocity.magnitude();
            if (speed <= 1.0e-3f)
            {
                continue;
            }
            const PxVec3 motion = velocity / speed;
            const float kineticEnergy = 0.5f * mass * speed * speed;
            const float charged = std::min(pending.work, kineticEnergy);
            const float impulse = pending.work >= kineticEnergy
                ? mass * speed
                : mass
                      * (speed
                         - std::sqrt(std::max(
                               0.0f, speed * speed - 2.0f * pending.work / mass)));

            pending.payer->addForce(-motion * impulse, PxForceMode::eIMPULSE);
            estimate->second -= motion * (impulse / mass);

            // No reaction impulse: the chunk this bill is for no longer
            // exists, and its share of the momentum leaves with the dust.
            m_telemetry.crushResistanceJoules += charged;
            ++m_telemetry.crushResistanceImpulses;
        }
        m_pendingResistance.clear();
    }

    void applyCrushResistanceImpulses()
    {
        if (!m_crushEnabled || !m_settings.applyCrushResistance)
        {
            return;
        }

        m_resistanceDamageScratch.resize(m_nodes.size());
        const uint32_t count = ext_stress_solver_get_node_crush_damage(
            m_solver,
            m_resistanceDamageScratch.data(),
            static_cast<uint32_t>(m_resistanceDamageScratch.size()));

        SceneWriteLock lock(m_scene);

        levyPendingResistance();

        for (uint32_t node = 0; node < count; ++node)
        {
            const float delta = m_resistanceDamageScratch[node] - m_prevCrushDamage[node];
            m_prevCrushDamage[node] = m_resistanceDamageScratch[node];
            if (delta <= 1.0e-6f)
            {
                continue;
            }
            if (m_nodes[node].crushed)
            {
                // A crushed node's whole bill went through the pending path;
                // charging its damage increment here too would double-bill the
                // resim pass.
                continue;
            }

            const CrusherTrack& crusher = m_nodeCrusher[node];
            if (!crusher.body)
            {
                continue;
            }

            const NodeState& state = m_nodes[node];
            const ExtStressPhysXMaterial& material = m_materials[state.material];
            const float work = delta * material.crushEnergy * state.volume;
            const float mass = crusher.body->getMass();
            if (work <= 0.0f || mass <= 0.0f)
            {
                continue;
            }

            // Charge against the payer's CURRENT closing speed, not the one
            // recorded at contact time. An impulse applied here only takes
            // effect at the next simulate, so the recorded speed can be one
            // tick stale -- and charging the full axis energy twice off the
            // same stale reading would REVERSE the crusher instead of
            // stopping it. Recomputing at charge time makes the ledger
            // self-limiting: once the payer has been stopped along this axis,
            // its current closing speed is zero and nothing more can be
            // extracted, whatever the host fed us.
            const PxVec3 position = state.body && state.body->body
                ? (state.body->body->getGlobalPose()
                       * (state.shape ? state.shape->getLocalPose() : PxTransform(PxIdentity))).p
                : m_worldTransform.transform(state.centroid);

            const PxVec3 payerCom = crusher.body->getGlobalPose().transform(
                crusher.body->getCMassLocalPose().p);
            const PxVec3 payerVelocity = crusher.body->getLinearVelocity()
                + crusher.body->getAngularVelocity().cross(position - payerCom);
            // Live closing speed only: the recorded one is post-fetchResults
            // and reads zero exactly when the press is hardest, and the live
            // value is what the clamp needs anyway -- the energy actually
            // still available along this axis.
            const float closing = payerVelocity.dot(crusher.direction);
            if (closing <= 1.0e-4f)
            {
                continue;
            }
            const float axisEnergy = 0.5f * mass * closing * closing;
            const float charged = std::min(work, axisEnergy);
            const float impulse = work >= axisEnergy
                ? mass * closing
                : mass * (closing - std::sqrt(std::max(0.0f, closing * closing - 2.0f * work / mass)));

            PxRigidBodyExt::addForceAtPos(
                *crusher.body, -crusher.direction * impulse, position, PxForceMode::eIMPULSE);
            if (state.body && state.body->body && !isKinematic(*state.body))
            {
                PxRigidBodyExt::addForceAtPos(
                    *state.body->body, crusher.direction * impulse, position, PxForceMode::eIMPULSE);
            }

            m_telemetry.crushResistanceJoules += charged;
            ++m_telemetry.crushResistanceImpulses;
        }
    }

    bool endTick() override
    {
        if (m_tickPhase != TickPhase::Solved)
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "endTick requires a successful solveTick.");
        }
        applyCrushResistanceImpulses();

        // A pulverized chunk is topology work even when no bond is overstressed:
        // a chunk can be ground up while every joint around it still holds.
        if (m_telemetry.overstressedBondCount > 0 || !m_pendingCrushedNodes.empty())
        {
            const TelemetryClock::time_point phaseStart = TelemetryClock::now();
            const bool fractured = fracture(m_tickDt);
            m_telemetry.fractureTopologyMilliseconds += elapsedMilliseconds(phaseStart);
            if (!fractured)
            {
        // Rebuild AFTER topology settles, not before the next beginTick.
        //
        // Fracture creates, splits and recycles bodies here in endTick. The
        // contacts that consult these flags arrive in the NEXT step's contact
        // callback -- which happens before the next beginTick. Rebuilding in
        // beginTick therefore left the flags describing the previous
        // composition on every tick after a topology change, and the audit
        // measured exactly that: 98,633 mismatches in 101,135,704 checks,
        // 0.0975%, which is the fracture rate.
        //
        // A contact wrongly classed as bondless is DROPPED before the solver
        // sees it, so the failure was lost load on freshly fractured chunks.
        refreshNodeBondless();
                m_tickPhase = TickPhase::Idle;
                return false;
            }
        }

        m_telemetry.bodyCount = static_cast<uint32_t>(m_actorBodies.size());
        // Bodies with no internal bonds: nothing in them can break, so the
        // contacts routed into them and their share of the solve cannot
        // change anything.
        //
        // E4: sampled, not per-tick. The first version of this census walked
        // every body EVERY tick calling getRigidBodyFlags()+isSleeping() —
        // PhysX scene reads, ~15k of them, inside the serial endTick, even
        // when nothing fractured. That is the same class of per-body scene
        // read that forced beginTick into beginTickFromSnapshot, and it was
        // added as instrumentation: the thermometer was heating the patient.
        // It is telemetry — ≤16 ticks of staleness changes nothing — so it
        // now runs one tick in BLAST_SINGLE_NODE_CENSUS_INTERVAL (default
        // 16, 1 = every tick, 0 = never) and holds the last values between.
        static const uint32_t censusInterval = [] {
            const char* raw = std::getenv("BLAST_SINGLE_NODE_CENSUS_INTERVAL");
            if (raw == nullptr)
            {
                return 16u;
            }
            const long parsed = std::atol(raw);
            return parsed < 0 ? 16u : static_cast<uint32_t>(parsed);
        }();
        if (censusInterval != 0 && (m_censusTick++ % censusInterval) == 0)
        {
            uint32_t singleNode = 0;
            uint32_t singleNodeAwake = 0;
            for (const auto& entry : m_actorBodies)
            {
                if (entry.second && entry.second->nodes.size() == 1)
                {
                    ++singleNode;
                    const PxRigidDynamic* actor = entry.second->body;
                    if (actor != nullptr
                        && !actor->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC)
                        && !actor->isSleeping())
                    {
                        ++singleNodeAwake;
                    }
                }
            }
            m_telemetry.singleNodeBodyCount = singleNode;
            m_telemetry.singleNodeAwakeBodies = singleNodeAwake;
        }
        // Rebuild AFTER topology settles, not before the next beginTick.
        //
        // Fracture creates, splits and recycles bodies here in endTick. The
        // contacts that consult these flags arrive in the NEXT step's contact
        // callback -- which happens before the next beginTick. Rebuilding in
        // beginTick therefore left the flags describing the previous
        // composition on every tick after a topology change, and the audit
        // measured exactly that: 98,633 mismatches in 101,135,704 checks,
        // 0.0975%, which is the fracture rate.
        //
        // A contact wrongly classed as bondless is DROPPED before the solver
        // sees it, so the failure was lost load on freshly fractured chunks.
        if (m_bondlessDirty)
        {
            refreshNodeBondless();
            m_bondlessDirty = false;
        }
        else if (verifyBondlessFlags())
        {
            const std::vector<uint8_t> lazy = m_nodeBondless;
            refreshNodeBondless();
            if (lazy != m_nodeBondless)
            {
                std::fprintf(stderr,
                             "[blast] bondless flags drifted: composition "
                             "changed without marking m_bondlessDirty\n");
            }
        }
        m_tickPhase = TickPhase::Idle;
        return true;
    }

    bool tick(float dt, const PxVec3& worldGravity) override
    {
        return beginTick(dt, worldGravity) && solveTick() && endTick();
    }

    bool validateMappings() override
    {
        const TelemetryClock::time_point validationStart = TelemetryClock::now();
        bool valid = m_solver != nullptr;
        std::unordered_set<const PxRigidDynamic*> bodies;
        std::vector<uint32_t> nodeOwners(m_nodes.size(), 0);

        for (const auto& actorBody : m_actorBodies)
        {
            const BodyState& body = *actorBody.second;
            valid = valid && actorBody.first == body.actorIndex && body.body != nullptr;
            valid = valid && bodies.insert(body.body).second;
            for (uint32_t nodeIndex : body.nodes)
            {
                if (nodeIndex >= m_nodes.size())
                {
                    valid = false;
                    continue;
                }
                ++nodeOwners[nodeIndex];
                valid = valid && m_nodes[nodeIndex].body == &body &&
                        m_nodes[nodeIndex].shape != nullptr;
                // NOTE: a physical-attachment check (shape->getActor() ==
                // body.body) was tried here and reverted: it fails during
                // creation, where validateMappings runs before every shape's
                // final attachment, and a validation failure aborts create().
                // If reintroduced it must be a separate post-attachment
                // check, never part of the create-time gate.
            }
        }

        for (uint32_t nodeIndex = 0; nodeIndex < nodeOwners.size(); ++nodeIndex)
        {
            // A pulverized chunk is owned by nothing, by design: its shape was
            // released and its actor retired. Everything else must have
            // exactly one owner.
            const uint32_t expected = m_nodes[nodeIndex].crushed ? 0u : 1u;
            valid = valid && nodeOwners[nodeIndex] == expected;
        }

        const uint32_t solverActorCount = m_solver ? ext_stress_solver_actor_count(m_solver) : 0;
        std::vector<ExtStressActor> actors(solverActorCount);
        std::vector<uint32_t> nodes(m_nodes.size());
        uint32_t actorCount = 0;
        uint32_t nodeCount = 0;
        if (solverActorCount == 0 ||
            ext_stress_solver_collect_actors(
                m_solver,
                actors.data(),
                solverActorCount,
                nodes.data(),
                static_cast<uint32_t>(nodes.size()),
                &actorCount,
                &nodeCount) != 1)
        {
            valid = false;
        }
        else
        {
            valid = valid && actorCount == m_actorBodies.size()
                && nodeCount == m_nodes.size() - m_crushedNodeCount;
            for (uint32_t i = 0; i < actorCount; ++i)
            {
                const auto bodyFound = m_actorBodies.find(actors[i].actorIndex);
                if (bodyFound == m_actorBodies.end() || !actors[i].nodes)
                {
                    valid = false;
                    continue;
                }
                std::vector<uint32_t> solverNodes(
                    actors[i].nodes,
                    actors[i].nodes + actors[i].nodeCount);
                std::sort(solverNodes.begin(), solverNodes.end());
                valid = valid && solverNodes == bodyFound->second->nodes;
            }
        }

        if (!valid)
        {
            ++m_telemetry.mappingValidationFailures;
            fail(
                ExtStressPhysXError::MappingInvalid,
                INVALID_INDEX,
                "Blast actor, PhysX body, node, or shape mappings disagree.");
        }
        m_telemetry.mappingValidationMilliseconds += elapsedMilliseconds(validationStart);
        return valid;
    }

    const ExtStressPhysXTelemetry& getTelemetry() const override
    {
        return m_telemetry;
    }

    bool usesGpuStressSolver() const override
    {
        return m_solver && ext_stress_solver_get_gpu_accelerated(m_solver) != 0;
    }

    uint32_t getBodySnapshots(
        ExtStressPhysXBodySnapshot* snapshots,
        uint32_t capacity) const override
    {
        if (!snapshots || capacity == 0)
        {
            return 0;
        }
        // Rebuilt only when the body set changes. Per call this was a heap
        // allocation, a full red-black-tree walk and ~185k pointer-chasing
        // comparisons at city scale, every tick, producing the identical order
        // as the tick before. The size assertion turns a missed invalidation
        // into a rebuild instead of a use-after-free.
        static const bool sortCache = [] {
            const char* raw = std::getenv("BLAST_SNAPSHOT_SORT_CACHE");
            return raw == nullptr || std::string(raw)[0] != '0';
        }();
        if (!sortCache || !m_sortedBodiesValid
            || m_sortedBodies.size() != m_actorBodies.size())
        {
            m_sortedBodies.clear();
            m_sortedBodies.reserve(m_actorBodies.size());
            for (const auto& entry : m_actorBodies)
            {
                m_sortedBodies.push_back(entry.second.get());
            }
            std::sort(
                m_sortedBodies.begin(),
                m_sortedBodies.end(),
                [](const BodyState* a, const BodyState* b) {
                    return a->bodyId < b->bodyId;
                });
            m_sortedBodiesValid = true;
        }
        const std::vector<const BodyState*>& sorted = m_sortedBodies;

        const uint32_t count =
            std::min(capacity, static_cast<uint32_t>(sorted.size()));
        for (uint32_t i = 0; i < count; ++i)
        {
            const BodyState& source = *sorted[i];
            ExtStressPhysXBodySnapshot& target = snapshots[i];
            target.bodyId = source.bodyId;
            target.actorIndex = source.actorIndex;
            target.body = source.body;
            target.globalPose = source.body->getGlobalPose();
            target.centerOfMassLocalPose = source.body->getCMassLocalPose();
            target.linearVelocity = source.body->getLinearVelocity();
            target.angularVelocity = source.body->getAngularVelocity();
            target.nodeCount = static_cast<uint32_t>(source.nodes.size());
            target.mass = source.cachedMass;
            // Same-tick row hint for consumeContactsFromSnapshot: stamped at
            // the one site that writes rows FROM body states, so it is fresh
            // by construction and costs nothing.
            source.snapshotRow = i;
            target.kinematic = isKinematic(source);
            target.sleeping = !target.kinematic && source.body->isSleeping();
        }
        return count;
    }

    uint32_t getShapeSnapshots(
        ExtStressPhysXShapeSnapshot* snapshots,
        uint32_t capacity) const override
    {
        if (!snapshots || capacity == 0)
        {
            return 0;
        }
        const uint32_t limit =
            std::min(capacity, static_cast<uint32_t>(m_nodes.size()));
        const uint64_t generation = ++m_shapeSnapshotGeneration;
        uint32_t count = 0;
        for (uint32_t i = 0; i < limit; ++i)
        {
            const NodeState& source = m_nodes[i];
            // A pulverized chunk has no shape and no body: it is gone. Emitting
            // a row for it would hand consumers a null shape and an identity
            // pose, and a renderer driven by this stream would keep drawing the
            // chunk after it turned to dust. Rows are therefore COMPACTED --
            // read nodeIndex from the row rather than assuming row i is node i.
            if (source.crushed)
            {
                continue;
            }
            ExtStressPhysXShapeSnapshot& target = snapshots[count++];
            target.shapeId = source.shapeId;
            target.bodyId = source.body ? source.body->bodyId : 0;
            target.nodeIndex = i;
            target.shape = source.shape;
            if (source.body)
            {
                target.bodyKinematic = isKinematic(*source.body);
                target.bodySleeping =
                    !target.bodyKinematic && source.body->body->isSleeping();
                if (source.body->snapshotGeneration != generation)
                {
                    source.body->snapshotGlobalPose = source.body->body->getGlobalPose();
                    source.body->snapshotGeneration = generation;
                }
                target.worldPose =
                    source.body->snapshotGlobalPose * source.shape->getLocalPose();
            }
            else
            {
                target.worldPose = PxTransform(PxIdentity);
                target.bodyKinematic = false;
                target.bodySleeping = true;
            }
        }
        return count;
    }

    uint32_t getActiveShapeSnapshots(
        ExtStressPhysXShapeSnapshot* snapshots,
        uint32_t capacity) const override
    {
        if (!snapshots || capacity == 0)
        {
            return 0;
        }
        uint32_t count = 0;
        for (const auto& entry : m_actorBodies)
        {
            BodyState& body = *entry.second;
            const bool kinematic = isKinematic(body);
            const bool sleeping = !kinematic && body.body->isSleeping();
            const bool active = !kinematic && !sleeping;
            if (!active && !body.snapshotWasActive)
            {
                continue;
            }
            body.snapshotWasActive = active;
            const PxTransform globalPose = body.body->getGlobalPose();
            for (uint32_t nodeIndex : body.nodes)
            {
                if (count >= capacity)
                {
                    return count;
                }
                const NodeState& source = m_nodes[nodeIndex];
                ExtStressPhysXShapeSnapshot& target = snapshots[count++];
                target.shapeId = source.shapeId;
                target.bodyId = body.bodyId;
                target.nodeIndex = nodeIndex;
                target.shape = source.shape;
                target.worldPose = globalPose * source.shape->getLocalPose();
                target.bodyKinematic = kinematic;
                target.bodySleeping = sleeping;
            }
        }
        return count;
    }

    uint32_t getSplitContinuity(
        ExtStressPhysXSplitContinuity* records,
        uint32_t capacity) const override
    {
        if (!records || capacity == 0)
        {
            return 0;
        }
        const uint32_t count =
            std::min(capacity, static_cast<uint32_t>(m_continuity.size()));
        std::copy_n(m_continuity.begin(), count, records);
        return count;
    }

    uint32_t drainChunkDestroyedEvents(
        ExtStressPhysXChunkDestroyed* records,
        uint32_t capacity) override
    {
        if (!records || capacity == 0 || m_chunkDestroyed.empty())
        {
            return 0;
        }
        const uint32_t count =
            std::min(capacity, static_cast<uint32_t>(m_chunkDestroyed.size()));
        std::copy_n(m_chunkDestroyed.begin(), count, records);
        m_chunkDestroyed.erase(
            m_chunkDestroyed.begin(), m_chunkDestroyed.begin() + count);
        return count;
    }

    uint32_t getNodeCrushDamage(float* damage, uint32_t capacity) const override
    {
        if (!damage || capacity == 0)
        {
            return 0;
        }
        if (!m_crushEnabled)
        {
            const uint32_t count = std::min(capacity, static_cast<uint32_t>(m_nodes.size()));
            std::fill_n(damage, count, 0.0f);
            return count;
        }
        return ext_stress_solver_get_node_crush_damage(m_solver, damage, capacity);
    }

    uint32_t getNodeStressInvariants(
        float* pressure,
        float* deviator,
        uint32_t capacity) const override
    {
        if ((!pressure && !deviator) || capacity == 0)
        {
            return 0;
        }
        if (!m_crushEnabled)
        {
            const uint32_t count = std::min(capacity, static_cast<uint32_t>(m_nodes.size()));
            if (pressure) std::fill_n(pressure, count, 0.0f);
            if (deviator) std::fill_n(deviator, count, 0.0f);
            return count;
        }
        return ext_stress_solver_get_node_stress_invariants(
            m_solver, pressure, deviator, capacity);
    }

    uint32_t getNodeCrushUtilisation(float* utilisation, uint32_t capacity) const override
    {
        if (!utilisation || capacity == 0)
        {
            return 0;
        }
        if (!m_crushEnabled)
        {
            const uint32_t count = std::min(capacity, static_cast<uint32_t>(m_nodes.size()));
            std::fill_n(utilisation, count, 0.0f);
            return count;
        }
        return ext_stress_solver_get_node_crush_utilisation(m_solver, utilisation, capacity);
    }

    bool isCrushEnabled() const override
    {
        return m_crushEnabled;
    }

    uint32_t getBondStresses(
        float* compression,
        float* tension,
        float* shear,
        uint32_t capacity) const override
    {
        if (!m_solver || capacity == 0)
        {
            return 0;
        }
        return ext_stress_solver_get_bond_stresses(
            m_solver, compression, tension, shear, capacity);
    }

    uint32_t getBondUtilisations(
        float* utilisation,
        uint32_t capacity) const override
    {
        if (!m_solver || capacity == 0)
        {
            return 0;
        }
        return ext_stress_solver_get_bond_utilisations(m_solver, utilisation, capacity);
    }

    uint32_t captureResimulationSnapshot() override
    {
        if (m_tickPhase != TickPhase::Idle)
        {
            fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "captureResimulationSnapshot requires the Idle tick phase.");
            return 0;
        }
        const TelemetryClock::time_point captureStart = TelemetryClock::now();
        m_resimIndexByBodyId.clear();
        m_resimProvenance.clear();
        m_resimSeeds.clear();
        // The crush-damage baseline travels with the motion snapshot: a
        // rewound frame rewinds the crusher's velocity, so re-running it must
        // re-charge the same comminution work against that restored motion.
        // Without this the resim pass would see almost no damage increment and
        // fracture frames would systematically under-charge resistance.
        m_resimPrevCrushDamage = m_prevCrushDamage;
        // Charges never levied belong to a frame that was never refunded:
        // without a restore, PhysX's own contact kept the momentum exchange
        // and there is nothing to reclaim. Drop them -- UNLESS they are armed:
        // the frame stepper re-captures immediately after a restore so the
        // resim pass can itself be rewound, and that capture arrives between
        // arming and the levy. Clearing armed charges there silently voided
        // every bill the mechanism exists to collect.
        if (!m_pendingResistanceArmed)
        {
            m_pendingResistance.clear();
        }

        SceneWriteLock lock(m_scene);
        const uint32_t bodyCount = static_cast<uint32_t>(m_actorBodies.size());
        if (m_resimSnapshot.size() < bodyCount)
        {
            m_resimSnapshot.resize(bodyCount);
        }
        uint32_t written = 0;
        for (const auto& entry : m_actorBodies)
        {
            const BodyState& body = *entry.second;
            ResimBodySnapshot& snapshot = m_resimSnapshot[written];
            snapshot.bodyId = body.bodyId;
            snapshot.globalPose = body.body->getGlobalPose();
            snapshot.kinematic = isKinematic(body);
            if (!snapshot.kinematic)
            {
                snapshot.linearVelocity = body.body->getLinearVelocity();
                snapshot.angularVelocity = body.body->getAngularVelocity();
                snapshot.sleeping = body.body->isSleeping();
                snapshot.wakeCounter = body.body->getWakeCounter();
            }
            else
            {
                snapshot.linearVelocity = PxVec3(0.0f);
                snapshot.angularVelocity = PxVec3(0.0f);
                snapshot.sleeping = false;
                snapshot.wakeCounter = 0.0f;
            }
            snapshot.worldCenterOfMass =
                snapshot.globalPose.transform(body.body->getCMassLocalPose().p);
            m_resimIndexByBodyId.emplace(snapshot.bodyId, written);
            ++written;
        }
        m_resimSnapshot.resize(written);
        m_resimSnapshotValid = true;
        ++m_telemetry.resimulationCaptures;
        m_telemetry.resimulationCaptureMilliseconds +=
            elapsedMilliseconds(captureStart);
        return written;
    }

    bool restoreResimulationSnapshot(
        PxRigidDynamic* const* activeBodies,
        uint32_t activeCount) override
    {
        // Arm the comminution charges on the restore REQUEST, not on its
        // success. The frame stepper rewinds scene motion itself before
        // calling here, so the refund the charges exist to reclaim has already
        // happened even when this adapter holds no snapshot of its own --
        // which is precisely the case on an impact frame that arrives out of a
        // quiet spell, when needsResimulationSnapshot() had said no at frame
        // start and the early-outs below fire.
        m_pendingResistanceArmed = !m_pendingResistance.empty();

        if (m_tickPhase != TickPhase::Idle)
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "restoreResimulationSnapshot requires the Idle tick phase.");
        }
        if (!m_resimSnapshotValid)
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "restoreResimulationSnapshot requires a prior capture.");
        }
        const TelemetryClock::time_point restoreStart = TelemetryClock::now();
        // Rewind the crush-damage baseline with the motion; see the capture.
        if (m_resimPrevCrushDamage.size() == m_prevCrushDamage.size())
        {
            m_prevCrushDamage = m_resimPrevCrushDamage;
        }
        SceneWriteLock lock(m_scene);

        std::unordered_set<PxRigidDynamic*> activeSet;
        const bool scoped = activeBodies != nullptr && activeCount > 0;
        if (scoped)
        {
            activeSet.reserve(activeCount);
            for (uint32_t i = 0; i < activeCount; ++i)
            {
                if (activeBodies[i])
                {
                    activeSet.insert(activeBodies[i]);
                }
            }
        }

        // Only the provenance loop below needs bodies by id, and it is empty
        // on most restores; building an O(bodies) map for it every time was
        // paid for nothing.
        std::unordered_map<ExtStressPhysXId, BodyState*> bodiesById;
        if (!m_resimProvenance.empty())
        {
            bodiesById.reserve(m_actorBodies.size());
            for (auto& entry : m_actorBodies)
            {
                bodiesById.emplace(entry.second->bodyId, entry.second.get());
            }
        }

        // A body that was asleep (or kinematic) when the snapshot was taken
        // and still is now was never integrated by the step in between: its
        // pose and velocity are the captured ones already, it accumulated no
        // force (a force write wakes an actor), and PhysX cannot sleep a body
        // it woke within the same step. Restoring it is eight PhysX setters
        // that write back what is there. On a settled city that is most of
        // the population -- the restore cost ~11 ms per split tick at grid 2
        // with ~2k of ~13k bodies awake. BLAST_RESIM_RESTORE_SCOPED=0
        // restores every captured body, for A/B.
        static const bool restoreScoped = [] {
            const char* raw = std::getenv("BLAST_RESIM_RESTORE_SCOPED");
            return raw == nullptr || raw[0] != '0';
        }();
        // BLAST_RESIM_RESTORE_VERIFY=1: for every body the scoped path skips,
        // check that PhysX still holds exactly the captured pose, and count
        // the ones it does not. Same tick, so cascade chaos cannot hide it.
        static const bool restoreVerify = [] {
            const char* raw = std::getenv("BLAST_RESIM_RESTORE_VERIFY");
            return raw != nullptr && raw[0] == '1';
        }();
        uint64_t verifyChecked = 0;
        uint64_t verifyMoved = 0;

        for (auto& entry : m_actorBodies)
        {
            BodyState& body = *entry.second;
            if (scoped && activeSet.count(body.body) == 0)
            {
                continue;
            }
            const auto found = m_resimIndexByBodyId.find(body.bodyId);
            if (found == m_resimIndexByBodyId.end())
            {
                continue;
            }
            const ResimBodySnapshot& snapshot = m_resimSnapshot[found->second];
            if (restoreScoped)
            {
                // Exact, not inferred: the state alone is not enough, because
                // the adapter itself repositions kinematic and sleeping bodies
                // between capture and restore (freeze parking, the kill
                // floor, a test's own setGlobalPose) -- measured as 31 of
                // 5,877 state-matched bodies with a different pose. So the
                // skip also requires the pose PhysX holds to be bit-identical
                // to the captured one: one read instead of eight writes, and
                // every write restoreBodyMotion would have made is then a
                // no-op by construction (a sleeping body has zero velocity and
                // no accumulated force; a kinematic one gets only the pose).
                const bool kinematicNow = isKinematic(body);
                const bool stateMatches =
                    (snapshot.kinematic && kinematicNow)
                    || (snapshot.sleeping && !kinematicNow && body.body->isSleeping());
                if (stateMatches)
                {
                    const PxTransform now = body.body->getGlobalPose();
                    const bool samePose = now.p == snapshot.globalPose.p
                        && now.q.x == snapshot.globalPose.q.x
                        && now.q.y == snapshot.globalPose.q.y
                        && now.q.z == snapshot.globalPose.q.z
                        && now.q.w == snapshot.globalPose.q.w;
                    if (restoreVerify)
                    {
                        ++verifyChecked;
                        if (!samePose)
                        {
                            ++verifyMoved;
                        }
                    }
                    if (samePose)
                    {
                        ++m_telemetry.resimulationBodiesSkipped;
                        continue;
                    }
                }
            }
            restoreBodyMotion(body, snapshot);
            ++m_telemetry.resimulationBodiesRestored;
        }
        if (restoreVerify)
        {
            static uint64_t totalChecked = 0;
            static uint64_t totalMoved = 0;
            totalChecked += verifyChecked;
            totalMoved += verifyMoved;
            std::fprintf(stderr,
                         "[resim-restore] scoped skip: checked=%llu moved=%llu (cumulative %llu / %llu)\n",
                         static_cast<unsigned long long>(verifyChecked),
                         static_cast<unsigned long long>(verifyMoved),
                         static_cast<unsigned long long>(totalChecked),
                         static_cast<unsigned long long>(totalMoved));
        }

        for (const ResimBodyProvenance& provenance : m_resimProvenance)
        {
            const auto childFound = bodiesById.find(provenance.bodyId);
            const auto parentFound = bodiesById.find(provenance.sourceParentBodyId);
            if (childFound == bodiesById.end() || parentFound == bodiesById.end())
            {
                continue;
            }
            BodyState& child = *childFound->second;
            const BodyState& parent = *parentFound->second;
            if (scoped && activeSet.count(parent.body) == 0 && activeSet.count(child.body) == 0)
            {
                continue;
            }
            const PxTransform previousPose = child.body->getGlobalPose();
            const PxTransform childPose =
                parent.body->getGlobalPose() * provenance.parentRelativePose;
            child.body->setGlobalPose(childPose, false);
            if (!isKinematic(child))
            {
                PxVec3 linearVelocity(0.0f);
                PxVec3 angularVelocity(0.0f);
                if (!isKinematic(parent))
                {
                    const PxVec3 parentCenter = parent.body->getGlobalPose().transform(
                        parent.body->getCMassLocalPose().p);
                    const PxVec3 childCenter =
                        childPose.transform(child.body->getCMassLocalPose().p);
                    angularVelocity = parent.body->getAngularVelocity();
                    linearVelocity = parent.body->getLinearVelocity() +
                        angularVelocity.cross(childCenter - parentCenter);
                }
                child.body->setLinearVelocity(linearVelocity, false);
                child.body->setAngularVelocity(angularVelocity, false);
                child.body->clearForce(PxForceMode::eFORCE);
                child.body->clearForce(PxForceMode::eIMPULSE);
                child.body->clearTorque(PxForceMode::eFORCE);
                child.body->clearTorque(PxForceMode::eIMPULSE);
                child.body->wakeUp();
            }
            m_telemetry.resimulationMaxRederivedDriftMeters = std::max(
                m_telemetry.resimulationMaxRederivedDriftMeters,
                vectorLength(childPose.p - previousPose.p));
            ++m_telemetry.resimulationBodiesRederived;
        }

        ++m_telemetry.resimulationRestores;
        m_telemetry.resimulationRestoreMilliseconds +=
            elapsedMilliseconds(restoreStart);
        return true;
    }

    bool needsResimulationSnapshot() const override
    {
        if (!m_settings.idleSkip)
        {
            return true;
        }
        return m_framesSinceFracture <= 2 || !m_contacts.empty() || m_hadForcesLastTick;
    }

    uint32_t getResimulationSeedBodies(
        PxRigidDynamic** bodies,
        uint32_t capacity) const override
    {
        if (!bodies || capacity == 0)
        {
            return static_cast<uint32_t>(m_resimSeeds.size());
        }
        const uint32_t count = std::min(
            capacity, static_cast<uint32_t>(m_resimSeeds.size()));
        std::copy_n(m_resimSeeds.begin(), count, bodies);
        return count;
    }

    uint32_t applyBaseStepSleep() override
    {
        if (!m_settings.baseStepSleep)
        {
            return 0;
        }
        // Do not sleep through support-loss windows: recently fractured or
        // contacted bodies may look settled for a frame then free-fall when a
        // neighbor splits (PR #41 class of failure).
        if (m_framesSinceFracture <= 60 || m_hadForcesLastTick || !m_contacts.empty())
        {
            return 0;
        }
        SceneWriteLock lock(m_scene);
        uint32_t slept = 0;
        const float linThr2 =
            m_settings.settledLinearSpeed * m_settings.settledLinearSpeed;
        const float angThr2 =
            m_settings.settledAngularSpeed * m_settings.settledAngularSpeed;
        for (auto& entry : m_actorBodies)
        {
            BodyState& body = *entry.second;
            if (isKinematic(body) || body.body->isSleeping())
            {
                continue;
            }
            const float lin2 = body.body->getLinearVelocity().magnitudeSquared();
            const float ang2 = body.body->getAngularVelocity().magnitudeSquared();
            if (lin2 <= linThr2 && ang2 <= angThr2)
            {
                body.body->putToSleep();
                ++slept;
            }
        }
        return slept;
    }

    ExtStressPhysXId getBodyId(const PxRigidDynamic* body) const override
    {
        const auto found = m_bodyToId.find(body);
        return found == m_bodyToId.end() ? 0 : found->second;
    }

    ExtStressPhysXId getShapeId(const PxShape* shape) const override
    {
        const auto found = m_shapeToNode.find(shape);
        return found == m_shapeToNode.end() ? 0 : m_nodes[found->second].shapeId;
    }

private:
    enum class TickPhase
    {
        Idle,
        Prepared,
        Solved
    };

    struct BodyState;

    struct NodeState
    {
        float mass{1.0f};
        float volume{1.0f};
        PxVec3 centroid{0.0f};
        ExtStressPhysXId shapeId{0};
        PxShape* shape{nullptr};
        PxConvexMesh* convexMesh{nullptr};
        BodyState* body{nullptr};
        uint32_t material{0};
        // Pulverized: shape detached and released, no longer part of any body.
        // Latched, so a crushed node is never resurrected by a later split.
        bool crushed{false};
    };

    struct BodyState
    {
        /// See ExtStressPhysXBodySnapshot::mass.
        float cachedMass = 0.0f;
        /// Row this body occupied in the last getBodySnapshots output.
        /// Verified against bodyId before use; a stale hint is a map lookup,
        /// never a wrong answer.
        mutable uint32_t snapshotRow = 0xffffffffu;
        ExtStressPhysXId bodyId{0};
        uint32_t actorIndex{INVALID_INDEX};
        PxRigidDynamic* body{nullptr};
        std::vector<uint32_t> nodes;
        /**
        Mass-weighted mean of the member nodes' centroids, in STRUCTURE space.

        That is the frame the solver states node positions in, so it is the
        frame addCentrifugalAcceleration expects. Deliberately NOT the PhysX
        body centre of mass, which is body-local: passing that would offset
        every centrifugal radius by the body origin. Cached because it only
        moves when membership does; invalidated wherever `nodes` is assigned.
        */
        mutable PxVec3 localCentreOfMass{0.0f};
        mutable bool localComValid{false};
        mutable uint64_t snapshotGeneration{0};
        mutable PxTransform snapshotGlobalPose{PxIdentity};
        mutable bool snapshotWasActive{false};
        uint64_t contactGeneration{0};
        PxTransform contactGlobalPose{PxIdentity};
        /// Separate from contactGeneration on purpose: the contacted-actor
        /// record is taken BEFORE the snapshot lookup can reject a contact,
        /// so a body whose snapshot row is missing must still be recorded as
        /// contacted. Sharing the pose generation would silently change which
        /// actors carry gravity.
        uint64_t contactedActorGeneration{0};

        ~BodyState()
        {
            if (body)
            {
                body->release();
            }
        }
    };

    /**
    Structure-space centre of mass of a body's member nodes.

    The solver states node positions in structure space, so the centrifugal
    radius must be measured from a centre in that same frame.
    */
    const PxVec3& bodyLocalCentreOfMass(const BodyState& body) const
    {
        if (!body.localComValid)
        {
            PxVec3 weighted(0.0f);
            float totalMass = 0.0f;
            for (uint32_t nodeIndex : body.nodes)
            {
                if (nodeIndex >= m_nodes.size())
                {
                    continue;
                }
                const NodeState& node = m_nodes[nodeIndex];
                // Support nodes carry mass 0; weight them uniformly so an
                // all-support body still resolves to its geometric centre
                // rather than dividing by zero.
                const float mass = node.mass > 0.0f ? node.mass : 1.0f;
                weighted += node.centroid * mass;
                totalMass += mass;
            }
            body.localCentreOfMass =
                totalMass > 0.0f ? weighted / totalMass : PxVec3(0.0f);
            body.localComValid = true;
        }
        return body.localCentreOfMass;
    }

    /**
    Feed the spin of a tumbling body to the solver as centrifugal load.

    Without this a free island carries no internal load at all: gravity enters
    as a uniform per-node acceleration, which for an unanchored body is a rigid
    translation and so produces zero relative velocity across every bond. A
    spinning slab genuinely is under omega-squared-r tension, and this is the
    only term that expresses it.
    */
    void addCentrifugal(
        const BodyState& body,
        const PxTransform& globalPose,
        const PxVec3& worldAngularVelocity)
    {
        if (!m_settings.applyCentrifugal)
        {
            return;
        }
        // Below this the term is numerically irrelevant and only costs a call.
        constexpr float kMinAngularSpeed = 1.0e-3f;
        if (worldAngularVelocity.magnitudeSquared() <=
            kMinAngularSpeed * kMinAngularSpeed)
        {
            return;
        }
        // Same frame convention as gravity above.
        const PxVec3 localAngular = globalPose.q.rotateInv(worldAngularVelocity);
        const StressVec3 bridgeCom = toStress(bodyLocalCentreOfMass(body));
        const StressVec3 bridgeAngular = toStress(localAngular);
        ext_stress_solver_add_centrifugal_acceleration(
            m_solver,
            body.actorIndex,
            &bridgeCom,
            &bridgeAngular);
    }


    // Motion state of one body at capture time, keyed by the stable bodyId
    // (actorIndex is reassigned across splits). worldCenterOfMass feeds the
    // COM-shift velocity correction: a reused body's mass frame moves when
    // shapes migrate away, so the stored linvel is re-expressed at the new COM.
    struct ResimBodySnapshot
    {
        ExtStressPhysXId bodyId{0};
        PxTransform globalPose{PxIdentity};
        PxVec3 linearVelocity{0.0f};
        PxVec3 angularVelocity{0.0f};
        PxVec3 worldCenterOfMass{0.0f};
        float wakeCounter{0.0f};
        bool kinematic{false};
        bool sleeping{false};
    };

    // One entry per body created since the last capture, in creation order so
    // a chain (child of a same-frame child) re-derives parents first. The
    // source parent always survives a split: the largest-overlap child reuses
    // the parent PxRigidDynamic.
    struct ResimBodyProvenance
    {
        ExtStressPhysXId bodyId{0};
        ExtStressPhysXId sourceParentBodyId{0};
        PxTransform parentRelativePose{PxIdentity};
    };

    bool fail(ExtStressPhysXError error, uint32_t nodeIndex, const char* message)
    {
        m_telemetry.lastError = error;
        m_telemetry.lastErrorNode = nodeIndex;
        if (m_errorCallback)
        {
            m_errorCallback(error, nodeIndex, message, m_errorUserData);
        }
        return false;
    }

    bool createNodeShape(uint32_t nodeIndex, const ExtStressPhysXNodeDesc& desc)
    {
        PxShape* shape = nullptr;
        PxConvexMesh* convexMesh = nullptr;

        if (desc.geometry.type == ExtStressPhysXGeometryType::Cuboid)
        {
            if (!finitePositive(desc.geometry.halfExtents))
            {
                return fail(
                    ExtStressPhysXError::InvalidDescriptor,
                    nodeIndex,
                    "Cuboid half extents must be finite and positive.");
            }
            const PxBoxGeometry geometry(desc.geometry.halfExtents);
            shape = m_physics.createShape(
                geometry,
                m_material,
                false,
                m_settings.shapeFlags);
        }
        else if (desc.geometry.type == ExtStressPhysXGeometryType::Convex)
        {
            if (!desc.geometry.convexPoints || desc.geometry.convexPointCount < 4)
            {
                return fail(
                    ExtStressPhysXError::InvalidDescriptor,
                    nodeIndex,
                    "A convex node needs at least four points.");
            }
            // Assets are expected to author hulls within the GPU vertex cap;
            // an oversized cloud is clamped by the cooker's own vertex limit
            // below, which computes the best bounded hull rather than
            // discarding points arbitrarily. The sanity bound only guards
            // cooking cost against a degenerate asset.
            if (desc.geometry.convexPointCount > 2048)
            {
                ++m_telemetry.convexPointLimitRejections;
                return fail(
                    ExtStressPhysXError::ConvexPointLimitExceeded,
                    nodeIndex,
                    "Convex node point cloud is unreasonably large (> 2048 points).");
            }
            for (uint32_t i = 0; i < desc.geometry.convexPointCount; ++i)
            {
                if (!desc.geometry.convexPoints[i].isFinite())
                {
                    return fail(
                        ExtStressPhysXError::InvalidDescriptor,
                        nodeIndex,
                        "Convex points must be finite.");
                }
            }

            PxConvexMeshDesc convexDesc;
            convexDesc.points.count = desc.geometry.convexPointCount;
            convexDesc.points.stride = sizeof(PxVec3);
            convexDesc.points.data = desc.geometry.convexPoints;
            convexDesc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
            // GPU rigid bodies require hulls of at most 64 vertices. Asking
            // the cooker to honour that directly produces the optimal bounded
            // hull; anything upstream that pre-thins the cloud can only do
            // worse.
            convexDesc.vertexLimit = static_cast<PxU16>(MAX_GPU_CONVEX_POINTS);

            PxCookingParams cookingParams(m_physics.getTolerancesScale());
            cookingParams.buildGPUData = true;
            PxConvexMeshCookingResult::Enum cookingResult =
                PxConvexMeshCookingResult::eFAILURE;
            convexMesh = PxCreateConvexMesh(
                cookingParams,
                convexDesc,
                m_physics.getPhysicsInsertionCallback(),
                &cookingResult);
            if (!convexMesh)
            {
                ++m_telemetry.convexCookingFailures;
                return fail(
                    ExtStressPhysXError::ConvexCookingFailed,
                    nodeIndex,
                    "PhysX could not cook a GPU-data convex mesh.");
            }

            const PxConvexMeshGeometry geometry(convexMesh);
            shape = m_physics.createShape(
                geometry,
                m_material,
                false,
                m_settings.shapeFlags);
        }
        else
        {
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                nodeIndex,
                "Unknown node geometry type.");
        }

        if (!shape)
        {
            if (convexMesh)
            {
                convexMesh->release();
            }
            return fail(
                ExtStressPhysXError::ShapeCreationFailed,
                nodeIndex,
                "PhysX could not create a shared node shape.");
        }

        NodeState& node = m_nodes[nodeIndex];
        // Maintain the node lookups here, at the one place a node's shape is
        // ever assigned. m_nodes is resized once at build and shapeId is handed
        // out once per node, so both maps are invariant apart from this site and
        // the crush/teardown paths -- which already erase from them. Rebuilding
        // them wholesale on every fracture was 48,210 of the ~51,000 hash
        // inserts in rebuildLookupTables, to update a body map of ~2,700.
        if (node.shape != nullptr && node.shape != shape)
        {
            m_shapeToNode.erase(node.shape);
        }
        node.shape = shape;
        node.convexMesh = convexMesh;
        m_shapeToNode[shape] = nodeIndex;
        m_shapeIdToNode[node.shapeId] = nodeIndex;
        return true;
    }

    std::unique_ptr<BodyState> createBody(
        const PxTransform& pose,
        uint32_t actorIndex)
    {
        PxRigidDynamic* rigid = m_physics.createRigidDynamic(pose);
        if (!rigid)
        {
            fail(
                ExtStressPhysXError::BodyCreationFailed,
                INVALID_INDEX,
                "PhysX could not create a rigid dynamic.");
            return nullptr;
        }

        std::unique_ptr<BodyState> result(new (std::nothrow) BodyState());
        if (!result)
        {
            rigid->release();
            fail(
                ExtStressPhysXError::AllocationFailed,
                INVALID_INDEX,
                "Could not allocate an adapter body record.");
            return nullptr;
        }
        rigid->setLinearDamping(m_settings.linearDamping);
        rigid->setAngularDamping(m_settings.angularDamping);
        // See the settings comment: creation is the only site that provably
        // covers a body's first simulation step.
        if (m_settings.bodyPositionIterations > 0)
        {
            rigid->setSolverIterationCounts(
                m_settings.bodyPositionIterations,
                std::max(1u, m_settings.bodyVelocityIterations));
        }
        if (m_settings.enableSpeculativeCcd)
        {
            rigid->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD, true);
        }
        if (m_settings.maxDepenetrationVelocity > 0.0f)
        {
            rigid->setMaxDepenetrationVelocity(m_settings.maxDepenetrationVelocity);
        }
        result->bodyId = m_nextBodyId++;
        result->cachedMass = rigid->getMass();
        result->actorIndex = actorIndex;
        result->body = rigid;
        ++m_telemetry.bodiesCreated;
        return result;
    }

    bool containsSupport(const std::vector<uint32_t>& nodes) const
    {
        for (uint32_t nodeIndex : nodes)
        {
            if (m_nodes[nodeIndex].mass == 0.0f)
            {
                return true;
            }
        }
        return false;
    }

    static bool isKinematic(const BodyState& body)
    {
        return body.body->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC);
    }

    void setBodyKinematic(BodyState& body, bool kinematic)
    {
        const bool wasKinematic = isKinematic(body);
        if (wasKinematic != kinematic)
        {
            body.body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, kinematic);
            if (!kinematic)
            {
                body.body->wakeUp();
            }
        }
    }

    void restoreBodyMotion(BodyState& body, const ResimBodySnapshot& snapshot)
    {
        body.body->setGlobalPose(snapshot.globalPose, false);
        // Kinematic status derives from the kept (post-fracture) topology, so
        // test the body's current flag, not the captured one: velocity writes
        // are rejected on kinematic actors.
        if (isKinematic(body))
        {
            return;
        }
        // The stored linvel is the velocity at the captured center of mass. A
        // split may have moved this body's mass frame, so re-express it at the
        // current COM under the restored pose; the term is zero when unchanged.
        const PxVec3 restoredCenter =
            snapshot.globalPose.transform(body.body->getCMassLocalPose().p);
        const PxVec3 linearVelocity = snapshot.linearVelocity +
            snapshot.angularVelocity.cross(restoredCenter - snapshot.worldCenterOfMass);
        body.body->setLinearVelocity(linearVelocity, false);
        body.body->setAngularVelocity(snapshot.angularVelocity, false);
        body.body->clearForce(PxForceMode::eFORCE);
        body.body->clearForce(PxForceMode::eIMPULSE);
        body.body->clearTorque(PxForceMode::eFORCE);
        body.body->clearTorque(PxForceMode::eIMPULSE);
        if (snapshot.sleeping)
        {
            body.body->putToSleep();
        }
        else
        {
            if (body.body->isSleeping())
            {
                body.body->wakeUp();
            }
            body.body->setWakeCounter(snapshot.wakeCounter);
        }
    }

    void updateMassProperties(BodyState& body)
    {
        std::vector<PxMassProperties> properties;
        std::vector<PxTransform> transforms;
        properties.reserve(body.nodes.size());
        transforms.reserve(body.nodes.size());

        for (uint32_t nodeIndex : body.nodes)
        {
            const NodeState& node = m_nodes[nodeIndex];
            if (node.mass <= 0.0f)
            {
                continue;
            }
            PxMassProperties property(node.shape->getGeometry());
            if (property.mass <= MIN_MASS || !std::isfinite(property.mass))
            {
                continue;
            }
            property = property * (node.mass / property.mass);
            properties.push_back(property);
            transforms.push_back(node.shape->getLocalPose());
        }

        if (properties.empty())
        {
            body.body->setMass(1.0f);
            body.body->setCMassLocalPose(PxTransform(PxIdentity));
            body.body->setMassSpaceInertiaTensor(PxVec3(1.0f));
            return;
        }

        const PxMassProperties combined = PxMassProperties::sum(
            properties.data(),
            transforms.data(),
            static_cast<PxU32>(properties.size()));

        // The centre of mass is declared on the shared convention -- the
        // mass-weighted mean of the nodes' authored centroids (each shape's
        // local pose) -- NOT the volume centroid PxMassProperties computes
        // from the hull geometry.
        //
        // Every layer that reconstructs a chunk from a streamed body pose
        // (server ledger, bootstrap, client topology) subtracts the
        // centroid-convention COM from the rest position. For boxes the two
        // definitions coincide, which is how the volume-centroid COM shipped
        // unnoticed; for authored convex hulls they can differ by tens of
        // centimetres, and every reconstructing layer then draws each chunk
        // displaced by exactly that difference the instant a body splits.
        //
        // A well-authored asset states its centroid AT the volume centroid,
        // making this physically exact as well as consistent. For assets that
        // do not, consistency wins: the declared COM is the contract the rest
        // of the pipeline reconstructs against, and the inertia tensor is
        // translated to remain correct about the declared point.
        PxVec3 conventionCom(0.0f);
        float conventionMass = 0.0f;
        for (uint32_t nodeIndex : body.nodes)
        {
            const NodeState& node = m_nodes[nodeIndex];
            if (node.mass <= 0.0f)
            {
                continue;
            }
            conventionCom += node.shape->getLocalPose().p * node.mass;
            conventionMass += node.mass;
        }
        const PxVec3 centerOfMass = conventionMass > 0.0f
            ? conventionCom * (1.0f / conventionMass)
            : combined.centerOfMass;
        const PxMat33 aboutConvention = PxMassProperties::translateInertia(
            combined.inertiaTensor,
            combined.mass,
            centerOfMass - combined.centerOfMass);

        PxQuat massFrame(PxIdentity);
        PxVec3 diagonal =
            PxMassProperties::getMassSpaceInertia(aboutConvention, massFrame);
        diagonal.x = std::max(diagonal.x, MIN_MASS);
        diagonal.y = std::max(diagonal.y, MIN_MASS);
        diagonal.z = std::max(diagonal.z, MIN_MASS);

        body.body->setMass(std::max(combined.mass, MIN_MASS));
        body.body->setCMassLocalPose(PxTransform(centerOfMass, massFrame));
        body.body->setMassSpaceInertiaTensor(diagonal);
        body.cachedMass = body.body != nullptr ? body.body->getMass() : 0.0f;
}

    PxVec3 pointVelocity(const BodyState& body, const PxVec3& worldPoint) const
    {
        const PxTransform pose = body.body->getGlobalPose();
        const PxVec3 center = pose.transform(body.body->getCMassLocalPose().p);
        return body.body->getLinearVelocity() +
            body.body->getAngularVelocity().cross(worldPoint - center);
    }

    void consumeContacts(float dt)
    {
        resetCrushStrainRates();
        m_contactNodeIndices.clear();
        m_contactLocalPositions.clear();
        m_contactLocalForces.clear();
        m_contactNodeIndices.reserve(m_contacts.size());
        m_contactLocalPositions.reserve(m_contacts.size() * 3);
        m_contactLocalForces.reserve(m_contacts.size() * 3);
        const uint64_t generation = ++m_contactGeneration;
        for (const QueuedContact& contact : m_contacts)
        {
            if (contact.nodeIndex >= m_nodes.size())
            {
                ++m_telemetry.contactsDropped;
                continue;
            }
            NodeState& node = m_nodes[contact.nodeIndex];
            if (!node.body || !node.body->body)
            {
                ++m_telemetry.contactsDropped;
                continue;
            }

            BodyState& body = *node.body;
            if (!isKinematic(body) && body.body->isSleeping())
            {
                body.body->wakeUp();
            }

            if (body.contactGeneration != generation)
            {
                body.contactGlobalPose = body.body->getGlobalPose();
                body.contactGeneration = generation;
            }
            const PxTransform& pose = body.contactGlobalPose;
            const PxVec3 localPosition = pose.transformInv(contact.position);
            const PxVec3 localForce = pose.q.rotateInv(contact.impulse / dt);

            accumulateCrushStrainRate(contact);

            m_contactNodeIndices.push_back(contact.nodeIndex);
            m_contactLocalPositions.insert(
                m_contactLocalPositions.end(),
                {localPosition.x, localPosition.y, localPosition.z});
            m_contactLocalForces.insert(
                m_contactLocalForces.end(),
                {localForce.x, localForce.y, localForce.z});
        }
        if (!m_contactNodeIndices.empty())
        {
            m_telemetry.contactsProcessed += ext_stress_solver_add_all_forces(
                m_solver,
                m_contactNodeIndices.data(),
                m_contactLocalPositions.data(),
                m_contactLocalForces.data(),
                static_cast<uint32_t>(m_contactNodeIndices.size()),
                0);
        }
        pushCrushStrainRates(dt);
        m_contacts.clear();
    }

    void addGravity(const PxVec3& worldGravity)
    {
        for (auto& actorBody : m_actorBodies)
        {
            BodyState& body = *actorBody.second;
            if (!isKinematic(body))
            {
                if (body.body->isSleeping())
                {
                    ++m_telemetry.sleepingActorsSkipped;
                    continue;
                }
                ++m_telemetry.awakeDynamicBodyCount;
            }
            const PxTransform globalPose = body.body->getGlobalPose();
            const PxVec3 localGravity = globalPose.q.rotateInv(worldGravity);
            const StressVec3 bridgeGravity = toStress(localGravity);
            ext_stress_solver_add_actor_gravity(
                m_solver,
                body.actorIndex,
                &bridgeGravity);
            if (!isKinematic(body))
            {
                addCentrifugal(body, globalPose, body.body->getAngularVelocity());
            }
        }
    }

    /// Snapshot the motion of every parent that SPLIT this tick, keyed by the
    /// split events rather than by the commands.
    ///
    /// It used to run over every actor that received a command, reading a
    /// PhysX pose per node. Under the sub-fatal damage band a standing city
    /// issues damage-only commands to its four big structures on every tick,
    /// so that was ~87k pose reads a tick -- 3.8 ms at rest -- for snapshots
    /// nobody read, because only a parent that actually splits is consumed by
    /// applySplit / makeChildPlan. The solver apply that precedes this call
    /// touches the Blast family only; the PhysX bodies are untouched until
    /// applySplit, so the values are the same as a pre-apply snapshot.
    std::map<uint32_t, ParentMotion> snapshotParents(
        const std::vector<ExtStressSplitEvent>& events,
        std::vector<NodeSnapshot>& nodeSnapshots) const
    {
        std::map<uint32_t, ParentMotion> parents;
        for (const ExtStressSplitEvent& event : events)
        {
            if (parents.find(event.parentActorIndex) != parents.end())
            {
                continue;
            }
            const auto found = m_actorBodies.find(event.parentActorIndex);
            if (found == m_actorBodies.end())
            {
                continue;
            }
            const BodyState& body = *found->second;
            ParentMotion motion;
            motion.bodyId = body.bodyId;
            motion.actorIndex = body.actorIndex;
            motion.pose = body.body->getGlobalPose();
            motion.centerOfMassWorld =
                motion.pose.transform(body.body->getCMassLocalPose().p);
            motion.linearVelocity = body.body->getLinearVelocity();
            motion.angularVelocity = body.body->getAngularVelocity();
            parents.emplace(event.parentActorIndex, motion);

            for (uint32_t nodeIndex : body.nodes)
            {
                NodeSnapshot& snapshot = nodeSnapshots[nodeIndex];
                snapshot.shapeWorldPose =
                    motion.pose * m_nodes[nodeIndex].shape->getLocalPose();
                snapshot.position = snapshot.shapeWorldPose.p;
                snapshot.pointVelocity = pointVelocity(body, snapshot.position);
                snapshot.valid = true;
            }
        }
        return parents;
    }

    ChildPlan makeChildPlan(
        const ExtStressActor& child,
        const ParentMotion& parent,
        const std::vector<NodeSnapshot>& nodeSnapshots) const
    {
        ChildPlan plan;
        plan.actorIndex = child.actorIndex;
        plan.nodes.assign(child.nodes, child.nodes + child.nodeCount);
        std::sort(plan.nodes.begin(), plan.nodes.end());

        float totalMass = 0.0f;
        for (uint32_t nodeIndex : plan.nodes)
        {
            const float mass = m_nodes[nodeIndex].mass;
            if (mass > 0.0f)
            {
                plan.fitCenter += nodeSnapshots[nodeIndex].position * mass;
                totalMass += mass;
            }
        }
        if (totalMass > 0.0f)
        {
            plan.fitCenter /= totalMass;
        }
        else
        {
            for (uint32_t nodeIndex : plan.nodes)
            {
                plan.fitCenter += nodeSnapshots[nodeIndex].position;
            }
            if (!plan.nodes.empty())
            {
                plan.fitCenter /= static_cast<float>(plan.nodes.size());
            }
        }

        plan.fitAngularVelocity = parent.angularVelocity;
        plan.fitVelocity = parent.linearVelocity +
            parent.angularVelocity.cross(plan.fitCenter - parent.centerOfMassWorld);
        return plan;
    }

    /**
    Exit fracture() having still retired any pulverized chunks.

    Every early return in fracture() predates crushing and means "no SPLIT
    work this tick". A tick can pulverize chunks without splitting anything --
    a chunk comminuted while its joints hold produces no bond fractures and no
    split event -- so none of those returns may skip this.
    */
    bool finishCrushOnly()
    {
        if (m_pendingCrushedNodes.empty())
        {
            return true;
        }
        {
            SceneWriteLock lock(m_scene);
            applyCrushedNodes();
        }
        rebuildLookupTables();
        return validateMappingsSampled();
    }

    /// E2: the per-fracture-tick audit, sampled. See validateInterval().
    /// A skipped audit returns true — no audit, no verdict — which matches
    /// what every non-fracturing tick has always done.
    bool validateMappingsSampled()
    {
        const uint32_t interval = validateInterval();
        if (interval == 0)
        {
            return true;
        }
        if ((m_validateTick++ % interval) != 0)
        {
            return true;
        }
        return validateMappings();
    }

    bool fracture(float dt)
    {
        const uint32_t actorCapacity = ext_stress_solver_actor_count(m_solver);
        const uint32_t bondCapacity =
            std::max(1u, ext_stress_solver_bond_count(m_solver));
        if (actorCapacity == 0)
        {
            return true;
        }

        TelemetryClock::time_point phase = TelemetryClock::now();
        // A/B on one binary: BLAST_FRACTURE_REUSE_BUFFERS=0 restores the
        // allocate-every-tick path. Value-checked, not presence-checked --
        // presence checks made `=0` mean "on" elsewhere in this tree.
        static const bool reuseBuffers = [] {
            const char* raw = std::getenv("BLAST_FRACTURE_REUSE_BUFFERS");
            return raw == nullptr || std::string(raw) != "0";
        }();
        if (reuseBuffers)
        {
            // resize() only grows here, and every element the callee reads it
            // writes first, so there is nothing to clear.
            m_fractureCommands.resize(
                std::max<std::size_t>(m_fractureCommands.size(), actorCapacity));
            m_fractureBonds.resize(
                std::max<std::size_t>(m_fractureBonds.size(), bondCapacity));
        }
        else
        {
            m_fractureCommands.assign(actorCapacity, ExtStressFractureCommands{});
            m_fractureBonds.assign(bondCapacity, ExtStressBondFracture{});
            m_fractureCommands.shrink_to_fit();
            m_fractureBonds.shrink_to_fit();
        }
        std::vector<ExtStressFractureCommands>& commands = m_fractureCommands;
        std::vector<ExtStressBondFracture>& fractures = m_fractureBonds;
        uint32_t commandCount = 0;
        uint32_t fractureCount = 0;
        const uint8_t generated = ext_stress_solver_generate_fracture_commands_per_actor(
            m_solver,
            commands.data(),
            actorCapacity,
            fractures.data(),
            bondCapacity,
            &commandCount,
            &fractureCount);
        m_telemetry.fractureGenerateMilliseconds += elapsedMilliseconds(phase);
        phase = TelemetryClock::now();
        if (generated == 0)
        {
            return fail(
                ExtStressPhysXError::SceneMutationFailed,
                INVALID_INDEX,
                "The stress bridge could not generate fracture commands.");
        }
        if (generated == 2)
        {
            return fail(
                ExtStressPhysXError::FractureBufferOverflow,
                INVALID_INDEX,
                "Fracture command buffers were unexpectedly truncated.");
        }
        if (commandCount == 0)
        {
            return finishCrushOnly();
        }


        commands.resize(commandCount);
        std::sort(commands.begin(), commands.end(), [](const ExtStressFractureCommands& a,
                                                        const ExtStressFractureCommands& b) {
            return a.actorIndex < b.actorIndex;
        });
        // maximumBodies / maximumFracturesPerActorPerTick are opt-in quality
        // degradations (default 0 = unlimited). Never use the remaining body
        // slot count to truncate how many bonds may break this tick: that
        // applied an arbitrary prefix of the overstressed list and often left
        // the impacted facade connected, so projectiles rebounded from a
        // monolith and fracture-frame resimulation had no hole to push through.
        if (m_settings.maximumBodies > 0
            && static_cast<uint32_t>(m_actorBodies.size()) >= m_settings.maximumBodies)
        {
            // Still retire pulverized chunks: they REDUCE the body count, so
            // skipping them here would deadlock the structure at the cap.
            return finishCrushOnly();
        }
        // Bounded by commandCount, not by the buffer's size. That was harmless
        // when the buffer was freshly value-initialised every tick (the tail
        // read as zero counts and got skipped); with the buffer reused it is
        // load-bearing, because the tail now holds last tick's commands.
        std::vector<ExtStressFractureCommands>& limitedCommands = m_fractureLimited;
        limitedCommands.clear();
        if (!reuseBuffers)
        {
            limitedCommands.shrink_to_fit();
        }
        limitedCommands.reserve(commandCount);
        for (uint32_t commandIndex = 0; commandIndex < commandCount; ++commandIndex)
        {
            ExtStressFractureCommands command = commands[commandIndex];
            if (m_settings.maximumFracturesPerActorPerTick > 0)
            {
                command.bondFractureCount = std::min(
                    command.bondFractureCount,
                    m_settings.maximumFracturesPerActorPerTick);
            }
            // A command with no bond fractures but a pulverized chunk is real
            // work: a chunk can be comminuted while every joint around it
            // still holds, and dropping it here left the chunk severed in the
            // solver but still present in the scene.
            if (command.bondFractureCount == 0 && command.chunkFractureCount == 0)
            {
                continue;
            }
            limitedCommands.push_back(command);
        }
        commands.swap(limitedCommands);
        commandCount = static_cast<uint32_t>(commands.size());
        if (commandCount == 0)
        {
            return finishCrushOnly();
        }

        // E7: the node-sized scratch, reused. These were fresh allocations —
        // zero-filled, ~1 MB each at 24k nodes — on EVERY fracturing tick,
        // the same disease BLAST_FRACTURE_REUSE_BUFFERS cured for the
        // bond-sized buffers; it never reached these. resize() to the same
        // size touches nothing after the first tick, and every element a
        // reader visits was written this tick first: snapshotParents fills
        // every node of every fracturing parent, and the only readers
        // (makeChildPlan, the continuity loop) visit child nodes, which are
        // subsets of their parent's nodes. The apply buffers are written by
        // the callee up to the returned counts, which is all anyone reads.
        if (reuseBuffers)
        {
            m_nodeSnapshotScratch.resize(m_nodes.size());
            m_splitEventScratch.resize(commandCount);
            m_splitChildScratch.resize(m_nodes.size());
            m_splitChildNodeScratch.resize(m_nodes.size());
        }
        else
        {
            m_nodeSnapshotScratch.assign(m_nodes.size(), NodeSnapshot{});
            m_splitEventScratch.assign(commandCount, ExtStressSplitEvent{});
            m_splitChildScratch.assign(m_nodes.size(), ExtStressActor{});
            m_splitChildNodeScratch.assign(m_nodes.size(), 0u);
        }
        std::vector<NodeSnapshot>& nodeSnapshots = m_nodeSnapshotScratch;
        std::vector<ExtStressSplitEvent>& events = m_splitEventScratch;
        std::vector<ExtStressActor>& children = m_splitChildScratch;
        std::vector<uint32_t>& childNodes = m_splitChildNodeScratch;
        // Parent snapshots are taken AFTER the solver apply, from the split
        // events, not here from the commands. See snapshotParents.
        m_telemetry.fracturePrepMilliseconds += elapsedMilliseconds(phase);
        phase = TelemetryClock::now();
        uint32_t eventCount = 0;
        uint32_t childCount = 0;
        uint32_t childNodeCount = 0;
        const uint8_t applied = ext_stress_solver_apply_fracture_commands(
            m_solver,
            commands.data(),
            commandCount,
            events.data(),
            static_cast<uint32_t>(events.size()),
            children.data(),
            static_cast<uint32_t>(children.size()),
            &eventCount,
            &childCount,
            childNodes.data(),
            static_cast<uint32_t>(childNodes.size()),
            &childNodeCount);
        if (applied == 0)
        {
            return fail(
                ExtStressPhysXError::SceneMutationFailed,
                INVALID_INDEX,
                "The stress bridge could not apply fracture commands.");
        }
        if (applied == 2)
        {
            return fail(
                ExtStressPhysXError::FractureBufferOverflow,
                INVALID_INDEX,
                "Split event buffers were unexpectedly truncated.");
        }
        if (eventCount == 0)
        {
            return finishCrushOnly();
        }

        m_telemetry.fractureApplyMilliseconds += elapsedMilliseconds(phase);
        phase = TelemetryClock::now();

        events.resize(eventCount);
        // Only now is it known WHICH parents split, and the PhysX bodies are
        // still exactly where they were before the solver apply (nothing
        // between here and applySplit moves them), so the snapshot can be
        // taken here for just those parents.
        const std::map<uint32_t, ParentMotion> parentMotions =
            snapshotParents(events, nodeSnapshots);
        m_telemetry.fracturePrepMilliseconds += elapsedMilliseconds(phase);
        phase = TelemetryClock::now();
        std::sort(events.begin(), events.end(), [&](const ExtStressSplitEvent& a,
                                                    const ExtStressSplitEvent& b) {
            const auto aParent = parentMotions.find(a.parentActorIndex);
            const auto bParent = parentMotions.find(b.parentActorIndex);
            const ExtStressPhysXId aId =
                aParent == parentMotions.end() ? 0 : aParent->second.bodyId;
            const ExtStressPhysXId bId =
                bParent == parentMotions.end() ? 0 : bParent->second.bodyId;
            return aId < bId;
        });

        {
            SceneWriteLock lock(m_scene);
            for (const ExtStressSplitEvent& event : events)
            {
                if (!applySplit(event, parentMotions, nodeSnapshots, dt))
                {
                    return false;
                }
            }
            // After the splits, so a crushed chunk is removed from whichever
            // body it ended up on rather than the one it started the tick on.
            applyCrushedNodes();
        }

        m_telemetry.fractureSceneMilliseconds += elapsedMilliseconds(phase);

        phase = TelemetryClock::now();
        rebuildLookupTables();
        m_telemetry.fractureRebuildMilliseconds += elapsedMilliseconds(phase);
        return validateMappingsSampled();
    }

    /**
    Remove every pulverized chunk from the scene and record its event.

    Blast has already severed the chunk structurally (the chunk fracture
    command zeroed its bonds and dropped its node from the island graph), but
    Blast never removes anything: a health-exhausted leaf chunk stays alive as
    an inert, still-visible actor forever. The physics-side removal is ours.

    Must be called under the scene write lock.
    */
    void applyCrushedNodes()
    {
        if (m_pendingCrushedNodes.empty())
        {
            return;
        }

        m_nodePressureScratch.resize(m_nodes.size());
        m_nodeDeviatorScratch.resize(m_nodes.size());
        ext_stress_solver_get_node_stress_invariants(
            m_solver,
            m_nodePressureScratch.data(),
            m_nodeDeviatorScratch.data(),
            static_cast<uint32_t>(m_nodes.size()));

        for (const uint32_t nodeIndex : m_pendingCrushedNodes)
        {
            if (nodeIndex >= m_nodes.size())
            {
                continue;
            }
            NodeState& node = m_nodes[nodeIndex];
            if (node.crushed)
            {
                continue;   // already removed; never resurrect a crushed chunk
            }

            ExtStressPhysXChunkDestroyed record;
            record.sequence = ++m_crushSequence;
            record.nodeIndex = nodeIndex;
            record.materialIndex = node.material;
            record.shapeId = node.shapeId;
            record.mass = node.mass;
            record.volume = node.volume;
            record.peakPressure =
                nodeIndex < m_nodePressureScratch.size() ? m_nodePressureScratch[nodeIndex] : 0.0f;
            record.peakDeviator =
                nodeIndex < m_nodeDeviatorScratch.size() ? m_nodeDeviatorScratch[nodeIndex] : 0.0f;

            const ExtStressPhysXMaterial& material = m_materials[node.material];
            record.debrisMassFraction = material.crushDebrisMassFraction;

            BodyState* body = node.body;
            if (body && body->body)
            {
                record.bodyId = body->bodyId;
                const PxTransform bodyPose = body->body->getGlobalPose();
                record.worldPose =
                    node.shape ? bodyPose * node.shape->getLocalPose() : bodyPose;
                record.angularVelocity = body->body->getAngularVelocity();
                // Velocity AT THE CHUNK, not at the body's centre of mass:
                // a dust cloud spawned with the body's COM velocity drifts
                // visibly wrong on a large tumbling piece.
                const PxVec3 comWorld = bodyPose.transform(
                    body->body->getCMassLocalPose().p);
                record.linearVelocity = body->body->getLinearVelocity() +
                    record.angularVelocity.cross(record.worldPose.p - comWorld);
            }
            else
            {
                record.worldPose = PxTransform(m_worldTransform.transform(node.centroid));
            }

            // The comminution bill for this chunk, owed by whoever pressed it.
            // Deferred to the resim pass; see PendingResistance.
            if (m_settings.applyCrushResistance)
            {
                // A node the impactor touched pays along its own contact
                // axis; every other comminuted node bills the tick's dominant
                // impactor, whose energy drove the stress that crushed it.
                const CrusherTrack& crusher = m_nodeCrusher[nodeIndex].body
                    ? m_nodeCrusher[nodeIndex]
                    : m_tickDominantCrusher;
                if (crusher.body)
                {
                    const float baseline =
                        m_resimPrevCrushDamage.size() == m_prevCrushDamage.size()
                            ? m_resimPrevCrushDamage[nodeIndex]
                            : 0.0f;
                    PendingResistance pending;
                    pending.payer = crusher.body;
                    pending.direction = crusher.direction;
                    pending.work =
                        material.crushEnergy * node.volume * std::max(0.0f, 1.0f - baseline);
                    if (pending.work > 0.0f)
                    {
                        m_pendingResistance.push_back(pending);
                    }
                }
            }

            detachCrushedShape(node);
            node.crushed = true;
            ++m_crushedNodeCount;

            // Blast keeps a health-exhausted leaf chunk alive as an inert
            // actor forever, so the solver would go on reporting an actor with
            // no body behind it. Retire it explicitly.
            if (ext_stress_solver_retire_crushed_node(m_solver, nodeIndex) == 0)
            {
                fail(
                    ExtStressPhysXError::MappingInvalid,
                    nodeIndex,
                    "A pulverized chunk's solver actor could not be retired.");
            }

            record.debrisBodiesSpawned = spawnCrushDebris(record, material);

            ++m_telemetry.chunksCrushed;
            m_telemetry.crushedMassKg += record.mass;
            m_telemetry.crushedVolumeM3 += record.volume;
            m_telemetry.debrisBodiesSpawned += record.debrisBodiesSpawned;
            m_chunkDestroyed.push_back(record);
        }

        m_pendingCrushedNodes.clear();
    }

    /**
    Detach and release a crushed chunk's shape, and retire its body if that was
    the last chunk on it.
    */
    void detachCrushedShape(NodeState& node)
    {
        BodyState* body = node.body;
        if (node.shape)
        {
            if (body && body->body)
            {
                body->body->detachShape(*node.shape, false);
                ++m_telemetry.shapesMigrated;
            }
            m_shapeToNode.erase(node.shape);
            node.shape->release();
            node.shape = nullptr;
        }
        m_shapeIdToNode.erase(node.shapeId);

        if (!body)
        {
            return;
        }

        body->nodes.erase(
            std::remove(body->nodes.begin(), body->nodes.end(),
                        static_cast<uint32_t>(&node - m_nodes.data())),
            body->nodes.end());
        m_bondlessDirty = true;
        node.body = nullptr;

        if (body->nodes.empty())
        {
            // The whole island was pulverized. Retire the body rather than
            // leaving an empty rigid actor in the scene.
            const auto found = m_actorBodies.find(body->actorIndex);
            if (found != m_actorBodies.end())
            {
                if (found->second->body)
                {
                    PxRigidDynamic* retiring = found->second->body;

                    // The resimulation seed list holds RAW body pointers and is
                    // only cleared at the next capture, so this body must not
                    // be handed to the frame stepper again.
                    m_resimSeeds.erase(
                        std::remove(m_resimSeeds.begin(), m_resimSeeds.end(), retiring),
                        m_resimSeeds.end());

                    // Remove from the scene, but do NOT release yet.
                    //
                    // The frame stepper captures every scene body BEFORE
                    // simulate and restores them after the ticks, and its
                    // restore already skips anything that has left the scene --
                    // but it checks that by calling getScene() on the pointer.
                    // Freeing the body here makes that check itself the crash.
                    // Deferring is not enough either: a fracture frame runs
                    // several restore passes against the same capture, so there
                    // is no point inside the frame where the pointer is
                    // provably unheld.
                    //
                    // So retired bodies are parked, empty and out of the scene,
                    // until release(). They cost a few hundred bytes each and
                    // are bounded by the number of fully pulverized islands,
                    // which is exactly the quantity crushing is authored to
                    // keep small.
                    m_scene.removeActor(*retiring);
                    m_retiredBodies.push_back(retiring);
                    // E3: the pointer key leaves the map when the body leaves
                    // the world — erase BEFORE nulling, while we still have it.
                    m_bodyToId.erase(retiring);
                    found->second->body = nullptr;
                }
                m_actorBodies.erase(found);
                m_bondlessDirty = true;
        m_sortedBodiesValid = false;
                ++m_telemetry.bodiesRecycled;
            }
        }
        else if (body->body)
        {
            updateMassProperties(*body);
        }
    }

    /**
    Optionally respawn part of a pulverized chunk's mass as rigid fragments.

    Default is zero fragments: all the mass leaves the rigid-body simulation
    and the event reports it so a consumer can render it as dust. Authoring a
    non-zero fraction trades body count for a pile that keeps some real mass.
    */
    uint32_t spawnCrushDebris(
        const ExtStressPhysXChunkDestroyed& record,
        const ExtStressPhysXMaterial& material)
    {
        const uint32_t fragmentCount = material.crushDebrisFragmentCount;
        if (material.crushDebrisMassFraction <= 0.0f || fragmentCount == 0 ||
            record.mass <= 0.0f || record.volume <= 0.0f)
        {
            return 0;
        }

        const float debrisMass = record.mass * material.crushDebrisMassFraction;
        const float debrisVolume = record.volume * material.crushDebrisMassFraction;
        const float fragmentMass = debrisMass / static_cast<float>(fragmentCount);
        const float fragmentVolume = debrisVolume / static_cast<float>(fragmentCount);
        const float halfExtent = 0.5f * std::cbrt(fragmentVolume);
        if (!(halfExtent > 0.0f))
        {
            return 0;
        }

        uint32_t spawned = 0;
        for (uint32_t i = 0; i < fragmentCount; ++i)
        {
            // Deterministic placement on a lattice around the chunk centre:
            // no RNG, so a replay of the same run produces the same debris.
            const float spread = std::cbrt(record.volume) * 0.25f;
            const PxVec3 offset(
                spread * ((i & 1u) ? 1.0f : -1.0f),
                spread * ((i & 2u) ? 1.0f : -1.0f),
                spread * ((i & 4u) ? 1.0f : -1.0f));

            PxRigidDynamic* fragment =
                m_physics.createRigidDynamic(PxTransform(record.worldPose.p + offset, record.worldPose.q));
            if (!fragment)
            {
                break;
            }
            PxShape* shape = m_physics.createShape(
                PxBoxGeometry(halfExtent, halfExtent, halfExtent), m_material, true, m_settings.shapeFlags);
            if (!shape)
            {
                fragment->release();
                break;
            }
            fragment->attachShape(*shape);
            shape->release();
            PxRigidBodyExt::setMassAndUpdateInertia(*fragment, fragmentMass);
            fragment->setLinearDamping(m_settings.linearDamping);
            fragment->setAngularDamping(m_settings.angularDamping);
            // Momentum-consistent: the fragments carry the chunk's motion, plus
            // a small outward component from the released elastic energy.
            fragment->setLinearVelocity(record.linearVelocity + offset.getNormalized() * 0.5f);
            fragment->setAngularVelocity(record.angularVelocity);
            m_scene.addActor(*fragment);
            m_crushDebris.push_back(fragment);
            ++spawned;
        }
        return spawned;
    }

    bool applySplit(
        const ExtStressSplitEvent& event,
        const std::map<uint32_t, ParentMotion>& parentMotions,
        const std::vector<NodeSnapshot>& nodeSnapshots,
        float dt)
    {
        auto parentBodyFound = m_actorBodies.find(event.parentActorIndex);
        const auto parentMotionFound = parentMotions.find(event.parentActorIndex);
        if (parentBodyFound == m_actorBodies.end() ||
            parentMotionFound == parentMotions.end() || !event.children ||
            event.childCount == 0)
        {
            return fail(
                ExtStressPhysXError::MappingInvalid,
                INVALID_INDEX,
                "A split event has no matching parent body or children.");
        }

        const ParentMotion& parent = parentMotionFound->second;
        std::unique_ptr<BodyState> parentBody = std::move(parentBodyFound->second);
        m_actorBodies.erase(parentBodyFound);
        m_bondlessDirty = true;
        // E3: the parent's PxRigidDynamic may be reused by a child below with
        // a different bodyId, so its stale entry must go now; every child —
        // reused pointer included — re-registers at the emplace site.
        if (parentBody && parentBody->body)
        {
            m_bodyToId.erase(parentBody->body);
        }
        m_sortedBodiesValid = false;

        std::unordered_set<uint32_t> parentNodes(
            parentBody->nodes.begin(),
            parentBody->nodes.end());
        std::vector<ChildPlan> plans;
        plans.reserve(event.childCount);
        for (uint32_t i = 0; i < event.childCount; ++i)
        {
            if (!event.children[i].nodes || event.children[i].nodeCount == 0)
            {
                return fail(
                    ExtStressPhysXError::MappingInvalid,
                    INVALID_INDEX,
                    "A split child has no nodes.");
            }
            plans.push_back(makeChildPlan(event.children[i], parent, nodeSnapshots));
        }
        std::sort(plans.begin(), plans.end(), [](const ChildPlan& a, const ChildPlan& b) {
            return a.actorIndex < b.actorIndex;
        });

        size_t reuseIndex = 0;
        uint32_t bestOverlap = 0;
        for (size_t i = 0; i < plans.size(); ++i)
        {
            uint32_t overlap = 0;
            for (uint32_t nodeIndex : plans[i].nodes)
            {
                overlap += parentNodes.count(nodeIndex) ? 1u : 0u;
            }
            if (overlap > bestOverlap ||
                (overlap == bestOverlap &&
                 plans[i].actorIndex < plans[reuseIndex].actorIndex))
            {
                bestOverlap = overlap;
                reuseIndex = i;
            }
        }
        plans[reuseIndex].reuse = true;

        struct AssignedChild
        {
            ChildPlan* plan{nullptr};
            std::unique_ptr<BodyState> body;
        };
        std::vector<AssignedChild> assigned;
        assigned.reserve(plans.size());

        for (ChildPlan& plan : plans)
        {
            AssignedChild child;
            child.plan = &plan;
            if (plan.reuse)
            {
                child.body = std::move(parentBody);
                child.body->actorIndex = plan.actorIndex;
                ++m_telemetry.bodiesReused;
                // Island-exact seeds must be bodies that existed at capture /
                // contact time. The reused parent keeps the same PxRigidDynamic*
                // and is the only safe graph seed; brand-new children are absent
                // from the pre-fracture contact graph and would force fallback.
                m_resimSeeds.push_back(child.body->body);
            }
            else
            {
                const PxTransform pose(plan.fitCenter, parent.pose.q);
                child.body = createBody(pose, plan.actorIndex);
                if (!child.body)
                {
                    return false;
                }
                if (m_resimSnapshotValid)
                {
                    // parent.pose is the pre-mutation pose of the surviving
                    // (reused) parent body, so a later restore can re-place
                    // this child from the parent's rewound state.
                    ResimBodyProvenance provenance;
                    provenance.bodyId = child.body->bodyId;
                    provenance.sourceParentBodyId = parent.bodyId;
                    provenance.parentRelativePose =
                        parent.pose.getInverse() * pose;
                    m_resimProvenance.push_back(provenance);
                }
            }
            child.body->nodes = plan.nodes;
            m_bondlessDirty = true;
            child.body->localComValid = false;  // membership changed
            assigned.push_back(std::move(child));
        }

        // One topology mutation pass. Shared shapes remain adapter-owned while
        // detached, so their identity survives migration.
        for (AssignedChild& child : assigned)
        {
            BodyState& target = *child.body;
            if (child.plan->reuse)
            {
                continue;
            }
            for (uint32_t nodeIndex : child.plan->nodes)
            {
                NodeState& node = m_nodes[nodeIndex];
                if (!nodeSnapshots[nodeIndex].valid || !node.body)
                {
                    return fail(
                        ExtStressPhysXError::MappingInvalid,
                        nodeIndex,
                        "A migrating node has no pre-split snapshot or parent.");
                }
                node.body->body->detachShape(*node.shape, false);
                const PxTransform localPose =
                    target.body->getGlobalPose().getInverse() *
                    nodeSnapshots[nodeIndex].shapeWorldPose;
                node.shape->setLocalPose(localPose);
                if (!target.body->attachShape(*node.shape))
                {
                    return fail(
                        ExtStressPhysXError::SceneMutationFailed,
                        nodeIndex,
                        "PhysX rejected shape migration to a child body.");
                }
                node.body = &target;
                ++m_telemetry.shapesMigrated;
            }
            m_scene.addActor(*target.body);
        }

        // The reused child kept its shapes attached. Every sibling shape has now
        // been removed, so all children can safely derive final mass properties.
        for (AssignedChild& child : assigned)
        {
            BodyState& target = *child.body;
            for (uint32_t nodeIndex : target.nodes)
            {
                m_nodes[nodeIndex].body = &target;
            }
            setBodyKinematic(target, containsSupport(target.nodes));
            updateMassProperties(target);
        }

        // Velocity reconciliation is intentionally separate from mass updates:
        // the engine COM is only final after every shape has moved.
        for (AssignedChild& child : assigned)
        {
            BodyState& target = *child.body;
            const ChildPlan& plan = *child.plan;
            if (!isKinematic(target))
            {
                const PxVec3 engineCenter = target.body->getGlobalPose().transform(
                    target.body->getCMassLocalPose().p);
                const PxVec3 reconciledVelocity = plan.fitVelocity +
                    plan.fitAngularVelocity.cross(engineCenter - plan.fitCenter);
                target.body->setAngularVelocity(plan.fitAngularVelocity, false);
                target.body->setLinearVelocity(reconciledVelocity, true);
            }
        }

        ExtStressPhysXSplitContinuity continuity{};
        continuity.splitSequence = ++m_splitSequence;
        continuity.parentBodyId = parent.bodyId;
        continuity.parentActorIndex = event.parentActorIndex;
        continuity.reusedChildActorIndex = plans[reuseIndex].actorIndex;
        for (AssignedChild& child : assigned)
        {
            for (uint32_t nodeIndex : child.body->nodes)
            {
                const NodeSnapshot& before = nodeSnapshots[nodeIndex];
                const PxVec3 afterPosition =
                    child.body->body->getGlobalPose()
                        .transform(m_nodes[nodeIndex].shape->getLocalPose().p);
                const PxVec3 afterVelocity =
                    pointVelocity(*child.body, afterPosition);
                continuity.maxWorldPositionDrift = std::max(
                    continuity.maxWorldPositionDrift,
                    vectorLength(afterPosition - before.position));
                continuity.maxPointVelocityDrift = std::max(
                    continuity.maxPointVelocityDrift,
                    vectorLength(afterVelocity - before.pointVelocity));
            }
        }
        m_telemetry.maxSplitWorldPositionDrift = std::max(
            m_telemetry.maxSplitWorldPositionDrift,
            continuity.maxWorldPositionDrift);
        m_telemetry.maxSplitPointVelocityDrift = std::max(
            m_telemetry.maxSplitPointVelocityDrift,
            continuity.maxPointVelocityDrift);
        if (m_settings.recordSplitContinuity)
        {
            m_continuity.push_back(continuity);
        }

        // Breaking a bond releases the load it carried. The parent was
        // kinematic at impact, so preserving only its pre-split velocity leaves
        // detached chunks motionless in their cavities. Convert Blast's excess
        // force/torque into one-shot impulses before the next simulation step.
        if (m_settings.applyExcessForces
            && (m_settings.excessForceScale > 0.0f
                || m_settings.minimumSeparationVelocity > 0.0f))
        {
            for (AssignedChild& child : assigned)
            {
                BodyState& target = *child.body;
                if (isKinematic(target))
                {
                    continue;
                }
                const PxVec3 worldCenter = target.body->getGlobalPose().transform(
                    target.body->getCMassLocalPose().p);
                const PxVec3 structureCenter = m_worldTransform.transformInv(worldCenter);
                const StressVec3 stressCenter = toStress(structureCenter);
                StressVec3 excessForce{};
                StressVec3 excessTorque{};
                if (m_settings.excessForceScale > 0.0f
                    && ext_stress_solver_get_excess_forces(
                        m_solver,
                        target.actorIndex,
                        &stressCenter,
                        &excessForce,
                        &excessTorque))
                {
                    const float impulseScale = dt * m_settings.excessForceScale;
                    const PxVec3 worldForce =
                        m_worldTransform.q.rotate(fromStress(excessForce)) * impulseScale;
                    const PxVec3 worldTorque =
                        m_worldTransform.q.rotate(fromStress(excessTorque)) * impulseScale;
                    target.body->addForce(worldForce, PxForceMode::eIMPULSE, true);
                    target.body->addTorque(worldTorque, PxForceMode::eIMPULSE, true);
                }
                if (m_settings.minimumSeparationVelocity > 0.0f)
                {
                    PxVec3 outward(structureCenter.x, -0.25f, structureCenter.z);
                    if (outward.magnitudeSquared() < 0.01f)
                    {
                        outward = PxVec3(1.0f, -0.25f, 0.0f);
                    }
                    outward.normalize();
                    const PxVec3 separationImpulse =
                        m_worldTransform.q.rotate(outward)
                        * target.body->getMass()
                        * m_settings.minimumSeparationVelocity;
                    target.body->addForce(
                        separationImpulse,
                        PxForceMode::eIMPULSE,
                        true);
                }
            }
        }

        for (AssignedChild& child : assigned)
        {
            // E3: see the create site — mutation-site maintenance of bodyToId.
            if (child.body->body)
            {
                m_bodyToId[child.body->body] = child.body->bodyId;
            }
            m_actorBodies.emplace(child.body->actorIndex, std::move(child.body));
            m_bondlessDirty = true;
        m_sortedBodiesValid = false;
        }
        ++m_telemetry.splits;
        return true;
    }

    /// Rebuilds only the map that actually changes when topology moves.
    ///
    /// This used to clear and refill all three on every fracturing tick:
    /// 24,105 nodes x 2 plus ~2,700 bodies, so ~51,000 hash inserts -- each one
    /// a node allocation, because clear() destroys the nodes even though it
    /// keeps the bucket array. Measured at 2.79-3.95 ms, the largest single
    /// item in the topology phase whenever that phase does anything.
    ///
    /// Only m_bodyToId genuinely churns: bodies are created and retired by
    /// splits. The two node maps are keyed on things assigned once per node at
    /// build and maintained at their mutation sites instead.
    ///
    /// BLAST_INCREMENTAL_LOOKUP=0 restores the full rebuild.
    /// BLAST_LOOKUP_VALIDATE=1 rebuilds into scratch maps and compares, which
    /// is how a missed mutation site would announce itself rather than becoming
    /// a silent stale lookup.
    void rebuildLookupTables()
    {
        static const bool incremental = [] {
            const char* raw = std::getenv("BLAST_INCREMENTAL_LOOKUP");
            return raw == nullptr || std::string(raw) != "0";
        }();
        static const bool validate =
            std::getenv("BLAST_LOOKUP_VALIDATE") != nullptr
            && std::string(std::getenv("BLAST_LOOKUP_VALIDATE")) != "0";

        if (!incremental || validate)
        {
            std::unordered_map<const PxShape*, uint32_t> shapeToNode;
            std::unordered_map<ExtStressPhysXId, uint32_t> shapeIdToNode;
            for (uint32_t i = 0; i < m_nodes.size(); ++i)
            {
                if (m_nodes[i].shape != nullptr)
                {
                    shapeToNode.emplace(m_nodes[i].shape, i);
                }
                shapeIdToNode.emplace(m_nodes[i].shapeId, i);
            }
            if (validate && incremental)
            {
                if (shapeToNode != m_shapeToNode || shapeIdToNode != m_shapeIdToNode)
                {
                    ++m_telemetry.lookupTableDrifts;
                    // Loud on purpose. This path is opt-in diagnostic only, and
                    // a stale lookup is otherwise silent -- it surfaces much
                    // later as a contact routed to the wrong node.
                    std::fprintf(
                        stderr,
                        "[blast] lookup drift #%llu: shapeToNode %zu vs %zu, "
                        "shapeIdToNode %zu vs %zu\n",
                        static_cast<unsigned long long>(m_telemetry.lookupTableDrifts),
                        m_shapeToNode.size(), shapeToNode.size(),
                        m_shapeIdToNode.size(), shapeIdToNode.size());
                }
            }
            if (!incremental)
            {
                m_shapeToNode.swap(shapeToNode);
                m_shapeIdToNode.swap(shapeIdToNode);
            }
        }

        // E3: m_bodyToId is now maintained at its four mutation sites (create,
        // crush-retire, split-parent erase, split-children emplace), like the
        // node maps above. The full O(bodies) clear+refill — ~18k hash
        // emplaces per fracturing tick, each a node allocation, for a churn
        // of tens — runs only on the non-incremental path, and as a scratch
        // comparison under BLAST_LOOKUP_VALIDATE so a missed mutation site
        // announces itself instead of going silently stale.
        if (!incremental || validate)
        {
            std::unordered_map<const PxRigidDynamic*, ExtStressPhysXId> bodyToId;
            for (const auto& actorBody : m_actorBodies)
            {
                bodyToId.emplace(actorBody.second->body, actorBody.second->bodyId);
            }
            if (validate && incremental && bodyToId != m_bodyToId)
            {
                ++m_telemetry.lookupTableDrifts;
                std::fprintf(
                    stderr,
                    "[blast] lookup drift #%llu: bodyToId %zu vs %zu\n",
                    static_cast<unsigned long long>(m_telemetry.lookupTableDrifts),
                    m_bodyToId.size(), bodyToId.size());
            }
            if (!incremental)
            {
                m_bodyToId.swap(bodyToId);
            }
        }
    }

    void destroy()
    {
        m_contacts.clear();
        m_pendingCrushedNodes.clear();
        m_chunkDestroyed.clear();
        for (PxRigidDynamic* body : m_retiredBodies)
        {
            if (body)
            {
                body->release();
            }
        }
        m_retiredBodies.clear();
        if (!m_crushDebris.empty())
        {
            // Crush debris are adapter-created bodies with no NodeState, so the
            // node loops below would leak them.
            SceneWriteLock lock(m_scene);
            for (PxRigidDynamic* body : m_crushDebris)
            {
                if (body)
                {
                    body->release();
                }
            }
            m_crushDebris.clear();
        }
        if (!m_actorBodies.empty())
        {
            SceneWriteLock lock(m_scene);
            // Releasing each rigid actor removes it from the scene and drops
            // its attachment references. The adapter's own shape references
            // keep shared node shapes alive until the explicit pass below.
            for (auto& entry : m_actorBodies)
            {
                if (entry.second && entry.second->body)
                {
                    entry.second->body->release();
                    entry.second->body = nullptr;
                }
            }
            m_actorBodies.clear();
            m_bondlessDirty = true;
        m_sortedBodiesValid = false;
        }

        for (NodeState& node : m_nodes)
        {
            if (node.shape)
            {
                node.shape->release();
                node.shape = nullptr;
            }
        }
        for (NodeState& node : m_nodes)
        {
            if (node.convexMesh)
            {
                node.convexMesh->release();
                node.convexMesh = nullptr;
            }
        }
        if (m_solver)
        {
            ext_stress_solver_destroy(m_solver);
            m_solver = nullptr;
        }
    }

    PxPhysics& m_physics;
    PxScene& m_scene;
    PxMaterial& m_material;
    PxTransform m_worldTransform;
    ExtStressPhysXSettings m_settings;
    ExtStressPhysXErrorCallback m_errorCallback;
    void* m_errorUserData;

    ExtStressSolverHandle* m_solver{nullptr};
    std::vector<NodeState> m_nodes;
    /// See refreshNodeBondless(): one byte per node, replacing two dependent
    /// pointer chases on the per-contact hot path.
    std::vector<uint8_t> m_nodeBondless;
    /// m_nodeBondless is a function of BODY COMPOSITION only (which bodies
    /// exist, and which of them hold exactly one node). It was rebuilt at the
    /// end of EVERY endTick -- an O(nodes) assign (96,420 at grid 2) plus a
    /// full O(bodies) hash-map walk -- whether or not anything fractured.
    ///
    /// That is why `end` cost the same on quiet ticks as on fracture ticks:
    /// measured 3.81 ms quiet vs 3.69 ms fracture across 1,053/1,045 ticks of
    /// `demolition`, and 1.05 ms on a completely static city. It is not a
    /// fracture walk that sometimes costs a lot; it was an every-tick walk.
    ///
    /// Set wherever composition changes (body created, retired, or a body's
    /// node set resized). Starts true so the first tick always builds.
    /// BLAST_VERIFY_BONDLESS_FLAGS=1 rebuilds unconditionally and reports any
    /// tick where the lazy flags differ from a full rebuild.
    bool m_bondlessDirty = true;
    std::vector<ExtStressPhysXBondDesc> m_bonds;
    /// Reused across ticks. These are sized by BOND count, and fracture() ran
    /// on every tick with an overstressed bond -- so a downtown-scale graph
    /// allocated and zero-filled ~1.2 MB (74k bonds x 16 B) per tick, then
    /// usually found commandCount == 0 and returned without using any of it.
    /// Measured before this: `generate` was ~75% of the whole topology phase,
    /// while apply/scene/rebuild sat at 0.00 because they were never reached.
    ///
    /// Two command buffers, not one, because the limit pass swaps: with a local
    /// on one side of the swap the big allocation would be freed every tick and
    /// the reuse would buy nothing.
    std::vector<ExtStressFractureCommands> m_fractureCommands;
    std::vector<ExtStressFractureCommands> m_fractureLimited;
    std::vector<ExtStressBondFracture> m_fractureBonds;
    /// E7: node-sized fracture scratch, grow-only under
    /// BLAST_FRACTURE_REUSE_BUFFERS (see the bond-sized pair above).
    std::vector<NodeSnapshot> m_nodeSnapshotScratch;
    std::vector<ExtStressSplitEvent> m_splitEventScratch;
    std::vector<ExtStressActor> m_splitChildScratch;
    std::vector<uint32_t> m_splitChildNodeScratch;
    std::map<uint32_t, std::unique_ptr<BodyState>> m_actorBodies;
    /// getBodySnapshots' sorted view, invalidated on every m_actorBodies
    /// mutation. Mutable: the snapshot read is const.
    mutable std::vector<const BodyState*> m_sortedBodies;
    mutable bool m_sortedBodiesValid = false;
    std::unordered_map<const PxShape*, uint32_t> m_shapeToNode;
    std::unordered_map<ExtStressPhysXId, uint32_t> m_shapeIdToNode;
    std::unordered_map<const PxRigidDynamic*, ExtStressPhysXId> m_bodyToId;
    std::vector<QueuedContact> m_contacts;
    /// bodyId -> snapshot, rebuilt per tick by the snapshot-fed begin phase.
    /// A member so that phase allocates nothing while running concurrently.
    std::unordered_map<ExtStressPhysXId, const ExtStressPhysXBodySnapshot*>
        m_snapshotByBodyId;
    std::vector<uint32_t> m_contactNodeIndices;
    std::vector<float> m_contactLocalPositions;
    std::vector<float> m_contactLocalForces;
    /// Actors that received contact load this tick, so the gravity phase can
    /// pair their weight even while PhysX has them asleep. Member (not local)
    /// so the begin phase allocates nothing while running concurrently.
    std::unordered_set<uint32_t> m_contactedActors;
    // Previous tick's body snapshot, for the quiet-tick gravity skip. Sized by
    // body count (~108 for the downtown city), not by node count, so holding
    // and comparing it is free next to the ~87,000 node writes it avoids.
    bool m_gpuVerified{false};
    std::vector<ExtStressPhysXBodySnapshot> m_lastSnapshot;
    uint32_t m_lastSnapshotCount{0};
    bool m_loadsValid{false};
    std::vector<ExtStressPhysXSplitContinuity> m_continuity;
    std::vector<ResimBodySnapshot> m_resimSnapshot;
    std::unordered_map<ExtStressPhysXId, uint32_t> m_resimIndexByBodyId;
    std::vector<ResimBodyProvenance> m_resimProvenance;
    std::vector<PxRigidDynamic*> m_resimSeeds;
    bool m_resimSnapshotValid{false};
    bool m_hadForcesLastTick{false};
    uint32_t m_framesSinceFracture{1000};
    // Converted material table forwarded to the bridge at create; kept for
    // re-creation and validation of bond material indices.
    std::vector<ExtStressMaterialDesc> m_materialDescs;
    // The unconverted table, kept because crush properties (debris fraction and
    // fragment count) are consumed on the PhysX side, not by the solver.
    std::vector<ExtStressPhysXMaterial> m_materials;
    uint32_t m_materialCount{0};

    // --- chunk crushing ---
    bool m_crushEnabled{false};
    std::vector<uint32_t> m_nodeMaterials;
    std::vector<float> m_nodeStrainRates;
    // Cube root of each node's authored volume: the length that turns a contact
    // closing speed into a strain rate.
    std::vector<float> m_nodeCharacteristicSize;
    // The strongest external body pressing on each node this tick: who pays
    // the comminution bill when crush resistance is on.
    struct CrusherTrack
    {
        PxRigidBody* body{nullptr};
        PxVec3 direction{0.0f};
        float closingSpeed{0.0f};
        float impulseMagnitude{0.0f};
    };
    std::vector<CrusherTrack> m_nodeCrusher;
    CrusherTrack m_tickDominantCrusher;
    /**
    Comminution charges owed to crushing bodies, levied on the resim pass.

    Why deferred: contact relative velocities are sampled after fetchResults,
    when PhysX has already done its rigid momentum exchange -- by then the
    payer's closing speed is near zero and an immediate charge extracts almost
    nothing. The energy the payer wrongly keeps only comes into existence on
    the RESIMULATION pass: motion is rewound to full speed, the crushed chunks
    are gone, and the contact that stopped the payer is never re-issued. That
    refunded pass is where the bill lands, against the restored velocity.

    Armed by restoreResimulationSnapshot, applied by the following endTick,
    cleared at the next capture. Without resim there is no refund -- PhysX's
    own contact kept the momentum exchange -- so unapplied charges are simply
    dropped, which is the correct ledger.
    */
    struct PendingResistance
    {
        PxRigidBody* payer{nullptr};
        PxVec3 direction{0.0f};
        float work{0.0f};
    };
    std::vector<PendingResistance> m_pendingResistance;
    bool m_pendingResistanceArmed{false};
    // Last tick's crush damage, so endTick can charge only the increment.
    // Snapshotted and restored with the resim capture: a rewound frame rewinds
    // the crusher's motion, so re-running it must re-charge the same work.
    std::vector<float> m_prevCrushDamage;
    std::vector<float> m_resimPrevCrushDamage;
    std::vector<float> m_resistanceDamageScratch;
    // Nodes pulverized this tick, awaiting scene removal in endTick.
    std::vector<uint32_t> m_pendingCrushedNodes;
    std::vector<uint32_t> m_crushScratch;
    std::vector<float> m_nodeDamageScratch;
    std::vector<float> m_nodeUtilisationScratch;
    std::vector<float> m_nodePressureScratch;
    std::vector<float> m_nodeDeviatorScratch;
    // Drained by the host; unlike m_continuity this never grows without bound.
    std::vector<ExtStressPhysXChunkDestroyed> m_chunkDestroyed;
    std::vector<PxRigidDynamic*> m_crushDebris;
    // Bodies whose every chunk was pulverized: out of the scene, kept alive
    // until release() so the frame stepper's restore can safely test them.
    std::vector<PxRigidDynamic*> m_retiredBodies;
    uint64_t m_crushSequence{0};
    // Chunks that have left the simulation. Subtracted from the expected node
    // total in validateMappings.
    std::size_t m_crushedNodeCount{0};
    ExtStressPhysXTelemetry m_telemetry;
    ExtStressPhysXId m_nextBodyId{1};
    ExtStressPhysXId m_nextShapeId{1};
    uint64_t m_splitSequence{0};
    TelemetryClock::time_point m_splitPhaseStart{};
    TickPhase m_tickPhase{TickPhase::Idle};
    float m_tickDt{0.0f};
    mutable uint64_t m_shapeSnapshotGeneration{0};
    uint64_t m_contactGeneration{0};
    /// E4: which tick the single-node census last ran (sampled telemetry).
    uint64_t m_censusTick{0};
    /// E2: fracturing ticks since creation, for the sampled audit.
    uint64_t m_validateTick{0};
};

ExtStressPhysXDestructible* ExtStressPhysXDestructible::create(
    const ExtStressPhysXDesc& desc,
    ExtStressPhysXTelemetry* failureTelemetry)
{
    if (!desc.physics || !desc.scene || !desc.material)
    {
        ExtStressPhysXTelemetry telemetry;
        telemetry.lastError = ExtStressPhysXError::InvalidDescriptor;
        if (failureTelemetry)
        {
            *failureTelemetry = telemetry;
        }
        if (desc.errorCallback)
        {
            desc.errorCallback(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "PxPhysics, PxScene, and PxMaterial are required.",
                desc.errorUserData);
        }
        return nullptr;
    }

    auto* destructible =
        new (std::nothrow) ExtStressPhysXDestructibleImpl(desc);
    if (!destructible)
    {
        ExtStressPhysXTelemetry telemetry;
        telemetry.lastError = ExtStressPhysXError::AllocationFailed;
        if (failureTelemetry)
        {
            *failureTelemetry = telemetry;
        }
        return nullptr;
    }
    if (!destructible->initialize(desc))
    {
        if (failureTelemetry)
        {
            *failureTelemetry = destructible->getTelemetry();
        }
        delete destructible;
        return nullptr;
    }
    return destructible;
}

} // namespace Blast
} // namespace Nv
