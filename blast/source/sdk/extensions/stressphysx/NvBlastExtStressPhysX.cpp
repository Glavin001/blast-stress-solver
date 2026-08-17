#include "extensions/stressphysx/NvBlastExtStressPhysX.h"

#include "../../../../rust_stress_example/ffi/ext_stress_bridge.h"

#include "cooking/PxCooking.h"
#include "extensions/PxMassProperties.h"
#include "extensions/PxRigidBodyExt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <new>
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
    , splits(0)
    , bodiesCreated(0)
    , bodiesReused(0)
    , bodiesRecycled(0)
    , shapesMigrated(0)
    , convexPointLimitRejections(0)
    , convexCookingFailures(0)
    , mappingValidationFailures(0)
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
    , mappingValidationMilliseconds(0.0)
    , gpuStressSolveMilliseconds(0.0)
    , gpuStressHostToDeviceBytes(0)
    , gpuStressDeviceToHostBytes(0)
    , resimulationCaptures(0)
    , resimulationRestores(0)
    , resimulationBodiesRestored(0)
    , resimulationBodiesRederived(0)
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
        }

        m_nodes.resize(desc.nodeCount);
        m_bonds.assign(desc.bonds, desc.bonds + desc.bondCount);

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

            NodeState& node = m_nodes[i];
            node.mass = source.mass;
            node.centroid = source.centroid;
            node.shapeId = m_nextShapeId++;

            ExtStressNodeDesc& solverNode = solverNodes[i];
            solverNode.centroid = toStress(source.centroid);
            solverNode.mass = source.mass;
            solverNode.volume = source.volume;
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
        if (!m_solver)
        {
            return fail(
                ExtStressPhysXError::SolverCreationFailed,
                INVALID_INDEX,
                "The C stress-solver bridge could not create its family.");
        }
        ext_stress_solver_set_island_aware(m_solver, m_settings.islandAware ? 1 : 0);
        ext_stress_solver_set_skip_settled(m_solver, m_settings.skipSettledIslands ? 1 : 0);
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
            m_actorBodies.emplace(body->actorIndex, std::move(body));
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
        if (contact.shapeId != 0)
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

        QueuedContact queued;
        queued.nodeIndex = nodeIndex;
        queued.position = contact.worldPosition;
        queued.impulse = contact.worldImpulse;
        queued.wake = contact.wake;
        m_contacts.push_back(queued);
        ++m_telemetry.contactsQueued;
        return true;
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
    void addGravityFromSnapshot(
        const PxVec3& worldGravity,
        const ExtStressPhysXBodySnapshot* bodies,
        uint32_t bodyCount)
    {
        for (uint32_t i = 0; i < bodyCount; ++i)
        {
            const ExtStressPhysXBodySnapshot& snapshot = bodies[i];
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
    void consumeContactsFromSnapshot(
        float dt,
        const ExtStressPhysXBodySnapshot* bodies,
        uint32_t bodyCount,
        ExtStressPhysXId* outWakeBodies,
        uint32_t wakeCapacity,
        uint32_t* outWakeCount)
    {
        uint32_t wakeCount = 0;
        if (m_contacts.empty())
        {
            // Nothing to look anything up for. Building the index anyway cost
            // a hash insert per body per tick -- ten thousand of them on a
            // collapsing city -- to answer zero queries.
            m_contactNodeIndices.clear();
            m_contactLocalPositions.clear();
            m_contactLocalForces.clear();
            m_contactedActors.clear();
            if (outWakeCount != nullptr)
            {
                *outWakeCount = 0;
            }
            return;
        }
        m_contactedActors.clear();

        m_snapshotByBodyId.clear();
        for (uint32_t i = 0; i < bodyCount; ++i)
        {
            m_snapshotByBodyId[bodies[i].bodyId] = &bodies[i];
        }

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
            // A body carrying contact load must also carry its weight this
            // tick, even if PhysX has it asleep: the contact pushes its
            // support nodes up, and without the matching gravity pulling the
            // rest down the solver sees a net upward load and reports
            // wrong-signed stress. addGravityFromSnapshot consults this.
            m_contactedActors.insert(body.actorIndex);
            const auto found = m_snapshotByBodyId.find(body.bodyId);
            if (found == m_snapshotByBodyId.end())
            {
                // Counted, not silent: a contact whose body has no snapshot row
                // would be lost damage, and contactsDropped is how that shows up.
                ++m_telemetry.contactsDropped;
                continue;
            }
            const ExtStressPhysXBodySnapshot& snapshot = *found->second;

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
        m_contacts.clear();
        if (outWakeCount != nullptr)
        {
            *outWakeCount = wakeCount;
        }
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
        m_tickPhase = TickPhase::Prepared;
        return true;
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
        m_tickPhase = TickPhase::Prepared;
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

        m_telemetry.overstressedBondCount =
            ext_stress_solver_overstressed_bond_count(m_solver);
        m_telemetry.solverIslandCount = ext_stress_solver_island_count(m_solver);
        m_telemetry.solverIslandsSkipped = ext_stress_solver_islands_skipped(m_solver);
        m_telemetry.gpuStressSolveMilliseconds +=
            ext_stress_solver_gpu_solve_milliseconds(m_solver);
        m_telemetry.gpuStressHostToDeviceBytes +=
            ext_stress_solver_gpu_host_to_device_bytes(m_solver);
        m_telemetry.gpuStressDeviceToHostBytes +=
            ext_stress_solver_gpu_device_to_host_bytes(m_solver);
        m_telemetry.stressSolveMilliseconds += elapsedMilliseconds(phaseStart);
        m_tickPhase = TickPhase::Solved;
        return true;
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
        if (m_telemetry.overstressedBondCount > 0)
        {
            const TelemetryClock::time_point phaseStart = TelemetryClock::now();
            const bool fractured = fracture(m_tickDt);
            m_telemetry.fractureTopologyMilliseconds += elapsedMilliseconds(phaseStart);
            if (!fractured)
            {
                m_tickPhase = TickPhase::Idle;
                return false;
            }
        }

        m_telemetry.bodyCount = static_cast<uint32_t>(m_actorBodies.size());
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

        for (uint32_t owners : nodeOwners)
        {
            valid = valid && owners == 1;
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
            valid = valid && actorCount == m_actorBodies.size() && nodeCount == m_nodes.size();
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
        std::vector<const BodyState*> sorted;
        sorted.reserve(m_actorBodies.size());
        for (const auto& entry : m_actorBodies)
        {
            sorted.push_back(entry.second.get());
        }
        std::sort(sorted.begin(), sorted.end(), [](const BodyState* a, const BodyState* b) {
            return a->bodyId < b->bodyId;
        });

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
        const uint32_t count =
            std::min(capacity, static_cast<uint32_t>(m_nodes.size()));
        const uint64_t generation = ++m_shapeSnapshotGeneration;
        for (uint32_t i = 0; i < count; ++i)
        {
            const NodeState& source = m_nodes[i];
            ExtStressPhysXShapeSnapshot& target = snapshots[i];
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

        std::unordered_map<ExtStressPhysXId, BodyState*> bodiesById;
        bodiesById.reserve(m_actorBodies.size());
        for (auto& entry : m_actorBodies)
        {
            bodiesById.emplace(entry.second->bodyId, entry.second.get());
        }

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
            restoreBodyMotion(body, m_resimSnapshot[found->second]);
            ++m_telemetry.resimulationBodiesRestored;
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
        PxVec3 centroid{0.0f};
        ExtStressPhysXId shapeId{0};
        PxShape* shape{nullptr};
        PxConvexMesh* convexMesh{nullptr};
        BodyState* body{nullptr};
    };

    struct BodyState
    {
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

    struct QueuedContact
    {
        uint32_t nodeIndex{INVALID_INDEX};
        PxVec3 position{0.0f};
        PxVec3 impulse{0.0f};
        bool wake{true};
    };

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
        node.shape = shape;
        node.convexMesh = convexMesh;
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
        result->bodyId = m_nextBodyId++;
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

    std::map<uint32_t, ParentMotion> snapshotParents(
        const std::vector<ExtStressFractureCommands>& commands,
        std::vector<NodeSnapshot>& nodeSnapshots) const
    {
        std::map<uint32_t, ParentMotion> parents;
        for (const ExtStressFractureCommands& command : commands)
        {
            const auto found = m_actorBodies.find(command.actorIndex);
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
            parents.emplace(command.actorIndex, motion);

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

    bool fracture(float dt)
    {
        const uint32_t actorCapacity = ext_stress_solver_actor_count(m_solver);
        const uint32_t bondCapacity =
            std::max(1u, ext_stress_solver_bond_count(m_solver));
        if (actorCapacity == 0)
        {
            return true;
        }

        std::vector<ExtStressFractureCommands> commands(actorCapacity);
        std::vector<ExtStressBondFracture> fractures(bondCapacity);
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
            return true;
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
            return true;
        }
        std::vector<ExtStressFractureCommands> limitedCommands;
        limitedCommands.reserve(commands.size());
        for (ExtStressFractureCommands command : commands)
        {
            if (m_settings.maximumFracturesPerActorPerTick > 0)
            {
                command.bondFractureCount = std::min(
                    command.bondFractureCount,
                    m_settings.maximumFracturesPerActorPerTick);
            }
            if (command.bondFractureCount == 0)
            {
                continue;
            }
            limitedCommands.push_back(command);
        }
        commands.swap(limitedCommands);
        commandCount = static_cast<uint32_t>(commands.size());
        if (commandCount == 0)
        {
            return true;
        }

        std::vector<NodeSnapshot> nodeSnapshots(m_nodes.size());
        const std::map<uint32_t, ParentMotion> parentMotions =
            snapshotParents(commands, nodeSnapshots);

        std::vector<ExtStressSplitEvent> events(commandCount);
        std::vector<ExtStressActor> children(m_nodes.size());
        std::vector<uint32_t> childNodes(m_nodes.size());
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
            return true;
        }

        events.resize(eventCount);
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
        }

        rebuildLookupTables();
        return validateMappings();
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
            m_actorBodies.emplace(child.body->actorIndex, std::move(child.body));
        }
        ++m_telemetry.splits;
        return true;
    }

    void rebuildLookupTables()
    {
        m_shapeToNode.clear();
        m_shapeIdToNode.clear();
        m_bodyToId.clear();
        for (uint32_t i = 0; i < m_nodes.size(); ++i)
        {
            m_shapeToNode.emplace(m_nodes[i].shape, i);
            m_shapeIdToNode.emplace(m_nodes[i].shapeId, i);
        }
        for (const auto& actorBody : m_actorBodies)
        {
            m_bodyToId.emplace(actorBody.second->body, actorBody.second->bodyId);
        }
    }

    void destroy()
    {
        m_contacts.clear();
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
    std::vector<ExtStressPhysXBondDesc> m_bonds;
    std::map<uint32_t, std::unique_ptr<BodyState>> m_actorBodies;
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
    uint32_t m_materialCount{0};
    ExtStressPhysXTelemetry m_telemetry;
    ExtStressPhysXId m_nextBodyId{1};
    ExtStressPhysXId m_nextShapeId{1};
    uint64_t m_splitSequence{0};
    TickPhase m_tickPhase{TickPhase::Idle};
    float m_tickDt{0.0f};
    mutable uint64_t m_shapeSnapshotGeneration{0};
    uint64_t m_contactGeneration{0};
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
