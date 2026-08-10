#include "extensions/stressphysx/NvBlastExtStressPhysXResim.h"
#include "extensions/stressphysx/NvBlastExtStressPhysXDirectGpu.h"

#include <chrono>
#include <cmath>
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

using StepperClock = std::chrono::steady_clock;

double stepperElapsedMilliseconds(StepperClock::time_point start)
{
    return std::chrono::duration<double, std::milli>(StepperClock::now() - start)
        .count();
}

struct StepperSceneWriteLock
{
    explicit StepperSceneWriteLock(PxScene& scene)
        : m_scene(scene)
    {
        m_scene.lockWrite(__FILE__, __LINE__);
    }

    ~StepperSceneWriteLock()
    {
        m_scene.unlockWrite();
    }

    PxScene& m_scene;
};

struct UnionFind
{
    std::unordered_map<PxRigidActor*, PxRigidActor*> parent;

    PxRigidActor* find(PxRigidActor* actor)
    {
        if (!actor)
        {
            return nullptr;
        }
        const auto inserted = parent.emplace(actor, actor);
        PxRigidActor*& root = inserted.first->second;
        if (root != actor)
        {
            root = find(root);
        }
        return root;
    }

    void unite(PxRigidActor* a, PxRigidActor* b)
    {
        a = find(a);
        b = find(b);
        if (a && b && a != b)
        {
            parent[b] = a;
        }
    }
};

struct SceneBodySnapshot
{
    PxRigidDynamic* body{nullptr};
    PxTransform globalPose{PxIdentity};
    PxVec3 linearVelocity{0.0f};
    PxVec3 angularVelocity{0.0f};
    PxVec3 worldCenterOfMass{0.0f};
    float wakeCounter{0.0f};
    bool kinematic{false};
    bool sleeping{false};
};

struct ScopedBuildResult
{
    bool useScoped{true};
    std::vector<PxRigidDynamic*> activeBodies;
    std::unordered_set<PxRigidDynamic*> frozenBodies;
};

bool shouldCapture(
    const ExtStressPhysXResimOptions& options,
    ExtStressPhysXFrameHooks& hooks,
    ExtStressPhysXDestructible* const* destructibles,
    uint32_t destructibleCount)
{
    if (!options.quietCaptureSkip)
    {
        return true;
    }
    if (hooks.forceResimulationCapture())
    {
        return true;
    }
    for (uint32_t i = 0; i < destructibleCount; ++i)
    {
        if (destructibles[i]->needsResimulationSnapshot())
        {
            return true;
        }
    }
    return false;
}

bool isSettledBody(
    const PxRigidDynamic& body,
    float settledLinearSpeed,
    float settledAngularSpeed)
{
    // Kinematics do not integrate on the re-step; treat them as settled so
    // outsiders can skip restore unless they join a seeded contact component.
    if (body.getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
    {
        return true;
    }
    const float linThr2 = settledLinearSpeed * settledLinearSpeed;
    const float angThr2 = settledAngularSpeed * settledAngularSpeed;
    return body.getLinearVelocity().magnitudeSquared() <= linThr2
        && body.getAngularVelocity().magnitudeSquared() <= angThr2;
}

ScopedBuildResult buildScopedSets(
    const std::vector<std::pair<PxRigidActor*, PxRigidActor*>>& contactPairs,
    const std::vector<SceneBodySnapshot>& snapshot,
    ExtStressPhysXDestructible* const* destructibles,
    uint32_t destructibleCount,
    const ExtStressPhysXResimOptions& options,
    ExtStressPhysXFrameStats& frame)
{
    ScopedBuildResult result;

    std::vector<PxRigidDynamic*> seeds;
    for (uint32_t i = 0; i < destructibleCount; ++i)
    {
        const uint32_t seedCount =
            destructibles[i]->getResimulationSeedBodies(nullptr, 0);
        const std::size_t previousSize = seeds.size();
        seeds.resize(previousSize + seedCount);
        destructibles[i]->getResimulationSeedBodies(
            seeds.data() + previousSize,
            seedCount);
    }

    UnionFind components;
    for (const std::pair<PxRigidActor*, PxRigidActor*>& pair : contactPairs)
    {
        components.unite(pair.first, pair.second);
    }

    std::unordered_set<PxRigidActor*> seededComponents;
    seededComponents.reserve(seeds.size());
    uint32_t singletonSeeds = 0;
    for (PxRigidDynamic* seed : seeds)
    {
        if (!seed)
        {
            continue;
        }
        // A seed missing from this pass's contact pairs is still active, but
        // must not poison the whole scene into full restore (city cascades often
        // split bodies that had no narrow-phase pair this exact fetch). Insert
        // it as its own component and continue scoping everyone else.
        if (components.parent.find(seed) == components.parent.end())
        {
            ++singletonSeeds;
        }
        seededComponents.insert(components.find(seed));
    }
    if (seededComponents.empty())
    {
        frame.scopedFallbackFullRestore = 1;
        result.useScoped = false;
        return result;
    }
    (void)singletonSeeds;

    std::unordered_set<PxRigidDynamic*> activeSet;
    activeSet.reserve(snapshot.size());
    for (const SceneBodySnapshot& bodySnapshot : snapshot)
    {
        PxRigidDynamic* body = bodySnapshot.body;
        if (!body || body->getScene() == nullptr)
        {
            continue;
        }

        bool active = false;
        {
            const PxRigidActor* componentRoot = components.find(body);
            if (componentRoot
                && seededComponents.find(const_cast<PxRigidActor*>(componentRoot))
                    != seededComponents.end())
            {
                active = true;
            }
            if (!active
                && !isSettledBody(
                    *body,
                    options.settledLinearSpeed,
                    options.settledAngularSpeed))
            {
                // Still-moving disjoint dynamics must advance time.
                active = true;
            }
        }

        if (active)
        {
            activeSet.insert(body);
        }
        else
        {
            result.frozenBodies.insert(body);
        }
    }

    result.activeBodies.reserve(activeSet.size());
    for (PxRigidDynamic* body : activeSet)
    {
        result.activeBodies.push_back(body);
    }

    frame.sceneBodiesActive = static_cast<uint32_t>(result.activeBodies.size());
    frame.sceneBodiesFrozen = static_cast<uint32_t>(result.frozenBodies.size());
    return result;
}

struct DirectGpuMotionBufferDeleter
{
    void operator()(ExtStressPhysXDirectGpuMotionBuffer* buffer) const
    {
        if (buffer)
        {
            buffer->release();
        }
    }
};

} // namespace

class ExtStressPhysXFrameStepperImpl final : public ExtStressPhysXFrameStepper
{
public:
    explicit ExtStressPhysXFrameStepperImpl(PxScene& scene)
        : m_scene(scene)
        , m_directGpuMotion(ExtStressPhysXDirectGpuMotionBuffer::create(scene))
    {
    }

    void release() override
    {
        delete this;
    }

    void recordDynamicContactPair(
        PxRigidActor* actor0,
        PxRigidActor* actor1) override
    {
        if (!actor0 || !actor1 || actor0 == actor1)
        {
            return;
        }

        // Include kinematic rigid dynamics: support-bearing structures are
        // kinematic until they peel, and projectile→facade contacts are the
        // seeds' only link into the pre-fracture contact graph. Static actors
        // (ground) stay excluded.
        PxRigidDynamic* dynamic0 = actor0->is<PxRigidDynamic>();
        PxRigidDynamic* dynamic1 = actor1->is<PxRigidDynamic>();
        if (!dynamic0 || !dynamic1)
        {
            return;
        }

        if (actor0 > actor1)
        {
            std::swap(actor0, actor1);
        }
        m_contactPairs.emplace_back(actor0, actor1);
    }

    bool stepFrame(
        float dt,
        const PxVec3& worldGravity,
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount,
        const ExtStressPhysXResimOptions& options,
        ExtStressPhysXFrameHooks* hooks,
        ExtStressPhysXFrameStats* stats) override
    {
        ExtStressPhysXFrameStats localStats;
        ExtStressPhysXFrameStats& frame = stats ? *stats : localStats;
        frame = ExtStressPhysXFrameStats();

        ExtStressPhysXFrameHooks defaultHooks;
        ExtStressPhysXFrameHooks& frameHooks = hooks ? *hooks : defaultHooks;

        const uint64_t splitsAtFrameStart = sumSplits(destructibles, destructibleCount);
        const bool canResim = options.maxPasses > 0;
        const bool wantCapture =
            canResim
            && shouldCapture(options, frameHooks, destructibles, destructibleCount);
        bool haveSnapshot = false;
        if (wantCapture)
        {
            captureAll(destructibles, destructibleCount, frameHooks, frame);
            haveSnapshot = true;
        }

        m_frozenBodies.clear();
        uint32_t passesRemaining = options.maxPasses;

        for (;;)
        {
            if (frame.resimPasses == 0)
            {
                m_contactPairs.clear();
            }

            const bool resimPass = frame.resimPasses > 0;

            const StepperClock::time_point simulateStart = StepperClock::now();
            m_scene.simulate(dt);
            const double submitMs = stepperElapsedMilliseconds(simulateStart);

            const StepperClock::time_point fetchStart = StepperClock::now();
            const bool fetched = m_scene.fetchResults(true);
            const double fetchMs = stepperElapsedMilliseconds(fetchStart);

            frame.simulateSubmitMilliseconds += submitMs;
            frame.fetchResultsMilliseconds += fetchMs;
            frame.simulateMilliseconds += submitMs + fetchMs;
            if (resimPass)
            {
                frame.resimSimulateSubmitMilliseconds += submitMs;
                frame.resimFetchResultsMilliseconds += fetchMs;
            }
            else
            {
                frame.baseSimulateSubmitMilliseconds += submitMs;
                frame.baseFetchResultsMilliseconds += fetchMs;
            }
            if (!fetched)
            {
                wakeFrozenBodies();
                return false;
            }
            frameHooks.onPostFetchResults(frame.resimPasses);

            const uint64_t splitsBefore = sumSplits(destructibles, destructibleCount);
            double beginMs = 0.0;
            double solveMs = 0.0;
            double endMs = 0.0;
            const StepperClock::time_point tickStart = StepperClock::now();
            const bool ticked = tickAll(
                dt,
                worldGravity,
                destructibles,
                destructibleCount,
                frameHooks,
                beginMs,
                solveMs,
                endMs);
            const double tickMs = stepperElapsedMilliseconds(tickStart);
            frame.tickMilliseconds += tickMs;
            frame.beginTickMilliseconds += beginMs;
            frame.solveTickMilliseconds += solveMs;
            frame.endTickMilliseconds += endMs;
            if (resimPass)
            {
                frame.resimTickMilliseconds += tickMs;
            }
            else
            {
                frame.baseTickMilliseconds += tickMs;
            }
            if (!ticked)
            {
                wakeFrozenBodies();
                return false;
            }

            const uint64_t splitsAfter = sumSplits(destructibles, destructibleCount);
            frame.splits = splitsAfter - splitsAtFrameStart;
            const bool fractured = splitsAfter > splitsBefore;
            if (!fractured || passesRemaining == 0)
            {
                break;
            }
            if (!haveSnapshot)
            {
                wakeFrozenBodies();
                /* Phase D: host calls applyBaseStepSleep after long quiet */
                return true;
            }

            --passesRemaining;
            ++frame.resimPasses;

            frameHooks.onRestore();

            bool useScopedRestore = options.scopedResim;
            std::vector<PxRigidDynamic*> activeBodies;
            if (useScopedRestore)
            {
                const ScopedBuildResult scoped = buildScopedSets(
                    m_contactPairs,
                    m_snapshot,
                    destructibles,
                    destructibleCount,
                    options,
                    frame);
                if (!scoped.useScoped)
                {
                    useScopedRestore = false;
                }
                else
                {
                    activeBodies = scoped.activeBodies;
                    freezeOutsiders(scoped.frozenBodies);
                }
            }

            if (useScopedRestore)
            {
                restoreScene(frame, &activeBodies);
                const StepperClock::time_point adapterRestoreStart = StepperClock::now();
                for (uint32_t i = 0; i < destructibleCount; ++i)
                {
                    destructibles[i]->restoreResimulationSnapshot(
                        activeBodies.data(),
                        static_cast<uint32_t>(activeBodies.size()));
                }
                frame.adapterRestoreMilliseconds +=
                    stepperElapsedMilliseconds(adapterRestoreStart);

                if (options.useDirectGpuMotionState
                    && m_directGpuMotion
                    && m_directGpuMotion->available()
                    && !activeBodies.empty())
                {
                    if (m_directGpuMotion->capture(
                            activeBodies.data(),
                            static_cast<uint32_t>(activeBodies.size())))
                    {
                        m_directGpuMotion->restore();
                    }
                }
            }
            else
            {
                restoreScene(frame, nullptr);
                const StepperClock::time_point adapterRestoreStart = StepperClock::now();
                for (uint32_t i = 0; i < destructibleCount; ++i)
                {
                    destructibles[i]->restoreResimulationSnapshot();
                }
                frame.adapterRestoreMilliseconds +=
                    stepperElapsedMilliseconds(adapterRestoreStart);

                if (options.useDirectGpuMotionState
                    && m_directGpuMotion
                    && m_directGpuMotion->available())
                {
                    if (m_directGpuMotion->restore())
                    {
                        // Best-effort GPU sync after full CPU restore.
                    }
                }
            }

            if (passesRemaining > 0)
            {
                captureAll(destructibles, destructibleCount, frameHooks, frame);
            }
        }

        wakeFrozenBodies();
        /* Phase D: host calls applyBaseStepSleep after long quiet */
        return true;
    }

private:
    static uint64_t sumSplits(
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount)
    {
        uint64_t splits = 0;
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            splits += destructibles[i]->getTelemetry().splits;
        }
        return splits;
    }

    static bool tickAll(
        float dt,
        const PxVec3& worldGravity,
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount,
        ExtStressPhysXFrameHooks& hooks,
        double& beginMilliseconds,
        double& solveMilliseconds,
        double& endMilliseconds)
    {
        const StepperClock::time_point beginStart = StepperClock::now();
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            if (!destructibles[i]->beginTick(dt, worldGravity))
            {
                beginMilliseconds = stepperElapsedMilliseconds(beginStart);
                return false;
            }
        }
        beginMilliseconds = stepperElapsedMilliseconds(beginStart);

        const StepperClock::time_point solveStart = StepperClock::now();
        if (!hooks.solveAll(destructibles, destructibleCount))
        {
            solveMilliseconds = stepperElapsedMilliseconds(solveStart);
            return false;
        }
        solveMilliseconds = stepperElapsedMilliseconds(solveStart);

        const StepperClock::time_point endStart = StepperClock::now();
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            if (!destructibles[i]->endTick())
            {
                endMilliseconds = stepperElapsedMilliseconds(endStart);
                return false;
            }
        }
        endMilliseconds = stepperElapsedMilliseconds(endStart);
        return true;
    }

    void captureSceneInto(ExtStressPhysXFrameStats& frame)
    {
        const StepperClock::time_point sceneCaptureStart = StepperClock::now();
        {
            StepperSceneWriteLock lock(m_scene);
            const PxU32 actorCount =
                m_scene.getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC);
            m_actorScratch.resize(actorCount);
            const PxU32 fetched = m_scene.getActors(
                PxActorTypeFlag::eRIGID_DYNAMIC,
                m_actorScratch.data(),
                actorCount);

            if (m_snapshot.size() < fetched)
            {
                m_snapshot.resize(fetched);
            }

            uint32_t written = 0;
            for (PxU32 i = 0; i < fetched; ++i)
            {
                PxRigidDynamic& body =
                    *static_cast<PxRigidDynamic*>(m_actorScratch[i]);
                SceneBodySnapshot& snapshot = m_snapshot[written++];
                snapshot.body = &body;
                snapshot.globalPose = body.getGlobalPose();
                snapshot.kinematic =
                    body.getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC);
                if (!snapshot.kinematic)
                {
                    snapshot.linearVelocity = body.getLinearVelocity();
                    snapshot.angularVelocity = body.getAngularVelocity();
                    snapshot.sleeping = body.isSleeping();
                    snapshot.wakeCounter = body.getWakeCounter();
                }
                else
                {
                    snapshot.linearVelocity = PxVec3(0.0f);
                    snapshot.angularVelocity = PxVec3(0.0f);
                    snapshot.sleeping = false;
                    snapshot.wakeCounter = 0.0f;
                }
                snapshot.worldCenterOfMass =
                    snapshot.globalPose.transform(body.getCMassLocalPose().p);
            }
            m_snapshot.resize(written);
        }
        frame.sceneBodiesCaptured = static_cast<uint32_t>(m_snapshot.size());
        frame.sceneCaptureMilliseconds +=
            stepperElapsedMilliseconds(sceneCaptureStart);
    }

    void captureAll(
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount,
        ExtStressPhysXFrameHooks& hooks,
        ExtStressPhysXFrameStats& frame)
    {
        captureSceneInto(frame);

        const StepperClock::time_point adapterCaptureStart = StepperClock::now();
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            destructibles[i]->captureResimulationSnapshot();
        }
        hooks.onCapture();
        frame.adapterCaptureMilliseconds +=
            stepperElapsedMilliseconds(adapterCaptureStart);

        if (m_directGpuMotion && m_directGpuMotion->available())
        {
            std::vector<PxRigidDynamic*> bodies;
            bodies.reserve(m_snapshot.size());
            for (const SceneBodySnapshot& snapshot : m_snapshot)
            {
                if (snapshot.body)
                {
                    bodies.push_back(snapshot.body);
                }
            }
            if (!bodies.empty())
            {
                m_directGpuMotion->capture(
                    bodies.data(),
                    static_cast<uint32_t>(bodies.size()));
            }
        }
    }

    void restoreBodyFromSnapshot(
        PxRigidDynamic& body,
        const SceneBodySnapshot& snapshot)
    {
        body.setGlobalPose(snapshot.globalPose, false);
        if (body.getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
        {
            return;
        }

        const PxVec3 restoredCenter =
            snapshot.globalPose.transform(body.getCMassLocalPose().p);
        const PxVec3 linearVelocity = snapshot.linearVelocity
            + snapshot.angularVelocity.cross(restoredCenter - snapshot.worldCenterOfMass);
        body.setLinearVelocity(linearVelocity, false);
        body.setAngularVelocity(snapshot.angularVelocity, false);
        body.clearForce(PxForceMode::eFORCE);
        body.clearForce(PxForceMode::eIMPULSE);
        body.clearTorque(PxForceMode::eFORCE);
        body.clearTorque(PxForceMode::eIMPULSE);
        if (snapshot.sleeping)
        {
            body.putToSleep();
        }
        else
        {
            if (body.isSleeping())
            {
                body.wakeUp();
            }
            body.setWakeCounter(snapshot.wakeCounter);
        }
    }

    void restoreScene(
        ExtStressPhysXFrameStats& frame,
        const std::vector<PxRigidDynamic*>* activeBodies)
    {
        const StepperClock::time_point restoreStart = StepperClock::now();
        StepperSceneWriteLock lock(m_scene);

        std::unordered_set<PxRigidDynamic*> activeSet;
        const bool scoped = activeBodies != nullptr;
        if (scoped)
        {
            activeSet.reserve(activeBodies->size());
            for (PxRigidDynamic* body : *activeBodies)
            {
                if (body)
                {
                    activeSet.insert(body);
                }
            }
        }

        uint32_t restored = 0;
        for (const SceneBodySnapshot& snapshot : m_snapshot)
        {
            if (!snapshot.body || snapshot.body->getScene() != &m_scene)
            {
                continue;
            }
            if (scoped && activeSet.count(snapshot.body) == 0)
            {
                continue;
            }

            restoreBodyFromSnapshot(*snapshot.body, snapshot);
            ++restored;
        }

        frame.sceneBodiesRestored += restored;
        frame.sceneRestoreMilliseconds += stepperElapsedMilliseconds(restoreStart);
    }

    void freezeOutsiders(const std::unordered_set<PxRigidDynamic*>& frozenBodies)
    {
        StepperSceneWriteLock lock(m_scene);
        for (PxRigidDynamic* body : frozenBodies)
        {
            if (!body || body->getScene() != &m_scene)
            {
                continue;
            }
            // Kinematics do not integrate; putToSleep/wakeUp are illegal on them.
            if (body->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
            {
                continue;
            }
            if (body->isSleeping())
            {
                continue;
            }
            // Settled outsiders stay asleep after P2. Re-waking them every
            // fracture frame was reactivating ~7k debris and dominating the
            // next base-step fetch/contact cost (balls10x rate regression).
            // Later contacts still wake bodies normally.
            body->putToSleep();
        }
    }

    void wakeFrozenBodies()
    {
        // Kept as a no-op clear for early-exit call sites; freeze no longer
        // schedules a post-P2 wake restore (see freezeOutsiders).
        m_frozenBodies.clear();
    }

    static void applyBaseStepSleepAll(
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount)
    {
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            destructibles[i]->applyBaseStepSleep();
        }
    }

    PxScene& m_scene;
    std::unique_ptr<ExtStressPhysXDirectGpuMotionBuffer, DirectGpuMotionBufferDeleter>
        m_directGpuMotion;
    std::vector<PxActor*> m_actorScratch;
    std::vector<SceneBodySnapshot> m_snapshot;
    std::vector<std::pair<PxRigidActor*, PxRigidActor*>> m_contactPairs;
    std::vector<PxRigidDynamic*> m_frozenBodies;
};

ExtStressPhysXFrameStepper* ExtStressPhysXFrameStepper::create(PxScene& scene)
{
    return new (std::nothrow) ExtStressPhysXFrameStepperImpl(scene);
}

} // namespace Blast
} // namespace Nv
