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
            ExtStressBondDesc& bond = solverBonds[i];
            bond.centroid = toStress(source.centroid);
            bond.normal = toStress(source.normal);
            bond.area = source.area;
            bond.node0 = source.node0;
            bond.node1 = source.node1;
        }

        ExtStressSolverSettingsDesc solverSettings{};
        solverSettings.max_solver_iterations_per_frame = m_settings.maxSolverIterationsPerFrame;
        solverSettings.graph_reduction_level = m_settings.graphReductionLevel;
        solverSettings.compression_elastic_limit = m_settings.compressionElasticLimit;
        solverSettings.compression_fatal_limit = m_settings.compressionFatalLimit;
        solverSettings.tension_elastic_limit = m_settings.tensionElasticLimit;
        solverSettings.tension_fatal_limit = m_settings.tensionFatalLimit;
        solverSettings.shear_elastic_limit = m_settings.shearElasticLimit;
        solverSettings.shear_fatal_limit = m_settings.shearFatalLimit;

        m_solver = ext_stress_solver_create(
            solverNodes.data(),
            static_cast<uint32_t>(solverNodes.size()),
            solverBonds.data(),
            static_cast<uint32_t>(solverBonds.size()),
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

    bool tick(float dt, const PxVec3& worldGravity) override
    {
        ++m_telemetry.ticks;
        m_telemetry.awakeDynamicBodyCount = 0;
        if (!m_solver || !std::isfinite(dt) || dt <= 0.0f || !worldGravity.isFinite())
        {
            m_contacts.clear();
            return fail(
                ExtStressPhysXError::InvalidDescriptor,
                INVALID_INDEX,
                "tick requires a finite positive dt and finite gravity.");
        }

        TelemetryClock::time_point phaseStart = TelemetryClock::now();
        consumeContacts(dt);
        m_telemetry.contactProcessingMilliseconds += elapsedMilliseconds(phaseStart);

        phaseStart = TelemetryClock::now();
        addGravity(worldGravity);
        m_telemetry.gravityMilliseconds += elapsedMilliseconds(phaseStart);

        phaseStart = TelemetryClock::now();
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

        if (m_telemetry.overstressedBondCount > 0)
        {
            phaseStart = TelemetryClock::now();
            const bool fractured = fracture(dt);
            m_telemetry.fractureTopologyMilliseconds += elapsedMilliseconds(phaseStart);
            if (!fractured)
            {
                return false;
            }
        }

        m_telemetry.bodyCount = static_cast<uint32_t>(m_actorBodies.size());
        return true;
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
        for (uint32_t i = 0; i < count; ++i)
        {
            const NodeState& source = m_nodes[i];
            ExtStressPhysXShapeSnapshot& target = snapshots[i];
            target.shapeId = source.shapeId;
            target.bodyId = source.body ? source.body->bodyId : 0;
            target.nodeIndex = i;
            target.shape = source.shape;
            target.worldPose = source.body
                ? source.body->body->getGlobalPose() * source.shape->getLocalPose()
                : PxTransform(PxIdentity);
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

        ~BodyState()
        {
            if (body)
            {
                body->release();
            }
        }
    };

    struct QueuedContact
    {
        uint32_t nodeIndex{INVALID_INDEX};
        PxVec3 position{0.0f};
        PxVec3 impulse{0.0f};
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
            if (desc.geometry.convexPointCount > MAX_GPU_CONVEX_POINTS)
            {
                ++m_telemetry.convexPointLimitRejections;
                return fail(
                    ExtStressPhysXError::ConvexPointLimitExceeded,
                    nodeIndex,
                    "GPU-compatible convex hulls are limited to 64 input points.");
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
        if (m_settings.maximumLinearVelocity > 0.0f)
        {
            rigid->setMaxLinearVelocity(m_settings.maximumLinearVelocity);
        }
        if (m_settings.maximumAngularVelocity > 0.0f)
        {
            rigid->setMaxAngularVelocity(m_settings.maximumAngularVelocity);
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
        PxQuat massFrame(PxIdentity);
        PxVec3 diagonal =
            PxMassProperties::getMassSpaceInertia(combined.inertiaTensor, massFrame);
        diagonal.x = std::max(diagonal.x, MIN_MASS);
        diagonal.y = std::max(diagonal.y, MIN_MASS);
        diagonal.z = std::max(diagonal.z, MIN_MASS);

        body.body->setMass(std::max(combined.mass, MIN_MASS));
        body.body->setCMassLocalPose(PxTransform(combined.centerOfMass, massFrame));
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

            const PxTransform pose = body.body->getGlobalPose();
            const PxVec3 localPosition = pose.transformInv(contact.position);
            const PxVec3 localForce = pose.q.rotateInv(contact.impulse / dt);
            const StressVec3 bridgePosition = toStress(localPosition);
            const StressVec3 bridgeForce = toStress(localForce);
            ext_stress_solver_add_force(
                m_solver,
                contact.nodeIndex,
                &bridgePosition,
                &bridgeForce,
                0);
            ++m_telemetry.contactsProcessed;
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
            const PxVec3 localGravity =
                body.body->getGlobalPose().q.rotateInv(worldGravity);
            const StressVec3 bridgeGravity = toStress(localGravity);
            ext_stress_solver_add_actor_gravity(
                m_solver,
                body.actorIndex,
                &bridgeGravity);
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
        uint32_t remainingBodyBudget = std::numeric_limits<uint32_t>::max();
        if (m_settings.maximumBodies > 0)
        {
            const uint32_t currentBodies = static_cast<uint32_t>(m_actorBodies.size());
            if (currentBodies >= m_settings.maximumBodies)
            {
                return true;
            }
            remainingBodyBudget = m_settings.maximumBodies - currentBodies;
        }
        std::vector<ExtStressFractureCommands> limitedCommands;
        limitedCommands.reserve(commands.size());
        for (ExtStressFractureCommands command : commands)
        {
            uint32_t allowed = command.bondFractureCount;
            if (m_settings.maximumFracturesPerActorPerTick > 0)
            {
                allowed = std::min(
                    allowed,
                    m_settings.maximumFracturesPerActorPerTick);
            }
            allowed = std::min(allowed, remainingBodyBudget);
            if (allowed == 0)
            {
                continue;
            }
            command.bondFractureCount = allowed;
            limitedCommands.push_back(command);
            remainingBodyBudget -= allowed;
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
            }
            else
            {
                const PxTransform pose(plan.fitCenter, parent.pose.q);
                child.body = createBody(pose, plan.actorIndex);
                if (!child.body)
                {
                    return false;
                }
            }
            child.body->nodes = plan.nodes;
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
                if (m_settings.maximumLinearVelocity > 0.0f)
                {
                    const PxVec3 velocity = target.body->getLinearVelocity();
                    const float maximum = m_settings.maximumLinearVelocity;
                    if (velocity.magnitudeSquared() > maximum * maximum)
                    {
                        target.body->setLinearVelocity(
                            velocity.getNormalized() * maximum,
                            true);
                    }
                }
                if (m_settings.maximumAngularVelocity > 0.0f)
                {
                    const PxVec3 velocity = target.body->getAngularVelocity();
                    const float maximum = m_settings.maximumAngularVelocity;
                    if (velocity.magnitudeSquared() > maximum * maximum)
                    {
                        target.body->setAngularVelocity(
                            velocity.getNormalized() * maximum,
                            true);
                    }
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
    std::vector<ExtStressPhysXSplitContinuity> m_continuity;
    ExtStressPhysXTelemetry m_telemetry;
    ExtStressPhysXId m_nextBodyId{1};
    ExtStressPhysXId m_nextShapeId{1};
    uint64_t m_splitSequence{0};
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
