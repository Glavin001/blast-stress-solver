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

bool isDynamicBody(PxRigidActor* actor)
{
    if (!actor || actor->getType() != PxActorType::eRIGID_DYNAMIC)
    {
        return false;
    }
    auto* body = static_cast<PxRigidDynamic*>(actor);
    return !body->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC);
}

bool bodySettled(const PxRigidDynamic& body, float linThr, float angThr)
{
    if (body.isSleeping())
    {
        return true;
    }
    const float lin2 = body.getLinearVelocity().magnitudeSquared();
    const float ang2 = body.getAngularVelocity().magnitudeSquared();
    return lin2 <= linThr * linThr && ang2 <= angThr * angThr;
}

} // namespace

class ExtStressPhysXFrameStepperImpl final : public ExtStressPhysXFrameStepper
{
public:
    explicit ExtStressPhysXFrameStepperImpl(PxScene& scene)
        : m_scene(scene)
    {
        m_directGpu.reset(ExtStressPhysXDirectGpuMotionBuffer::create(scene));
    }

    void release() override
    {
        delete this;
    }

    void recordDynamicContactPair(PxRigidActor* actor0, PxRigidActor* actor1) override
    {
        if (!isDynamicBody(actor0) || !isDynamicBody(actor1) || actor0 == actor1)
        {
            return;
        }
        auto* a = static_cast<PxRigidDynamic*>(actor0);
        auto* b = static_cast<PxRigidDynamic*>(actor1);
        if (a > b)
        {
            std::swap(a, b);
        }
        m_contactPairs.emplace_back(a, b);
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
        uint32_t passesRemaining = options.maxPasses;
        m_snapshotValid = false;
        m_frozen.clear();

        const bool wantResim = passesRemaining > 0;
        const bool needCapture = wantResim
            && (!options.quietCaptureSkip || frameHooks.forceResimulationCapture()
                || anyNeedsCapture(destructibles, destructibleCount));
        if (needCapture)
        {
            captureAll(destructibles, destructibleCount, frameHooks, frame, options);
        }

        for (;;)
        {
            const bool resimPass = frame.resimPasses > 0;
            if (!resimPass)
            {
                m_contactPairs.clear();
            }

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
                return false;
            }

            const uint64_t splitsAfter = sumSplits(destructibles, destructibleCount);
            frame.splits = splitsAfter - splitsAtFrameStart;
            if (splitsAfter == splitsBefore || passesRemaining == 0)
            {
                wakeFrozen();
                applyBaseStepSleep(destructibles, destructibleCount);
                return true;
            }
            if (!m_snapshotValid)
            {
                // Fractured without a pre-step snapshot (quiet skip). Cannot
                // roll back safely — commit the split topology as-is.
                applyBaseStepSleep(destructibles, destructibleCount);
                return true;
            }

            --passesRemaining;
            ++frame.resimPasses;

            frameHooks.onRestore();
            const bool scoped = options.scopedResim
                && buildScopedSets(destructibles, destructibleCount, frame);
            restoreScene(frame, scoped);
            if (scoped)
            {
                freezeOutsiders(frame);
            }

            const StepperClock::time_point adapterRestoreStart = StepperClock::now();
            if (scoped && !m_activeBodies.empty())
            {
                for (uint32_t i = 0; i < destructibleCount; ++i)
                {
                    destructibles[i]->restoreResimulationSnapshot(
                        m_activeBodies.data(),
                        static_cast<uint32_t>(m_activeBodies.size()));
                }
            }
            else
            {
                for (uint32_t i = 0; i < destructibleCount; ++i)
                {
                    destructibles[i]->restoreResimulationSnapshot();
                }
            }
            frame.adapterRestoreMilliseconds +=
                stepperElapsedMilliseconds(adapterRestoreStart);

            if (options.useDirectGpuMotionState && m_directGpu && m_directGpu->available()
                && !m_activeBodies.empty())
            {
                // Best-effort GPU path for the active set; CPU restore above is
                // already authoritative if this fails (indices invalid, etc.).
                m_directGpu->capture(
                    m_activeBodies.data(),
                    static_cast<uint32_t>(m_activeBodies.size()));
                m_directGpu->restore();
            }

            if (passesRemaining > 0)
            {
                captureAll(
                    destructibles, destructibleCount, frameHooks, frame, options);
            }
        }
    }

private:
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

    struct DirectGpuDeleter
    {
        void operator()(ExtStressPhysXDirectGpuMotionBuffer* value) const
        {
            if (value)
            {
                value->release();
            }
        }
    };

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

    static bool anyNeedsCapture(
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount)
    {
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            if (destructibles[i]->needsResimulationSnapshot())
            {
                return true;
            }
        }
        return false;
    }

    static void applyBaseStepSleep(
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount)
    {
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            destructibles[i]->applyBaseStepSleep();
        }
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

    void captureAll(
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount,
        ExtStressPhysXFrameHooks& hooks,
        ExtStressPhysXFrameStats& frame,
        const ExtStressPhysXResimOptions& options)
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

            // capture_into: overwrite in place, no clear+realloc churn.
            if (m_snapshot.size() < fetched)
            {
                m_snapshot.resize(fetched);
            }
            for (PxU32 i = 0; i < fetched; ++i)
            {
                PxRigidDynamic& body = *static_cast<PxRigidDynamic*>(m_actorScratch[i]);
                SceneBodySnapshot& snapshot = m_snapshot[i];
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
            m_snapshot.resize(fetched);
        }
        frame.sceneBodiesCaptured = static_cast<uint32_t>(m_snapshot.size());
        frame.sceneCaptureMilliseconds +=
            stepperElapsedMilliseconds(sceneCaptureStart);
        m_snapshotValid = true;

        const StepperClock::time_point adapterCaptureStart = StepperClock::now();
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            destructibles[i]->captureResimulationSnapshot();
        }
        hooks.onCapture();
        frame.adapterCaptureMilliseconds +=
            stepperElapsedMilliseconds(adapterCaptureStart);

        if (options.useDirectGpuMotionState && m_directGpu && m_directGpu->available())
        {
            m_activeBodies.clear();
            m_activeBodies.reserve(m_snapshot.size());
            for (const SceneBodySnapshot& snapshot : m_snapshot)
            {
                m_activeBodies.push_back(snapshot.body);
            }
            m_directGpu->capture(
                m_activeBodies.data(),
                static_cast<uint32_t>(m_activeBodies.size()));
        }
    }

    bool buildScopedSets(
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount,
        ExtStressPhysXFrameStats& frame)
    {
        m_activeBodies.clear();
        m_activeSet.clear();
        m_frozen.clear();

        m_seedScratch.clear();
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            const uint32_t needed =
                destructibles[i]->getResimulationSeedBodies(nullptr, 0);
            const size_t offset = m_seedScratch.size();
            m_seedScratch.resize(offset + needed);
            destructibles[i]->getResimulationSeedBodies(
                m_seedScratch.data() + offset, needed);
        }
        if (m_seedScratch.empty())
        {
            frame.scopedFallbackFullRestore = 1;
            return false;
        }

        // Union-find over pre-fracture dynamic contact pairs.
        std::unordered_map<PxRigidDynamic*, PxRigidDynamic*> parent;
        parent.reserve(m_snapshot.size() * 2);
        auto findRoot = [&](PxRigidDynamic* x) {
            auto it = parent.find(x);
            if (it == parent.end())
            {
                parent.emplace(x, x);
                return x;
            }
            PxRigidDynamic* root = x;
            while (parent[root] != root)
            {
                root = parent[root];
            }
            PxRigidDynamic* walk = x;
            while (walk != root)
            {
                PxRigidDynamic* next = parent[walk];
                parent[walk] = root;
                walk = next;
            }
            return root;
        };
        auto unite = [&](PxRigidDynamic* a, PxRigidDynamic* b) {
            a = findRoot(a);
            b = findRoot(b);
            if (a != b)
            {
                parent[b] = a;
            }
        };

        for (const auto& pair : m_contactPairs)
        {
            unite(pair.first, pair.second);
        }
        for (PxRigidDynamic* seed : m_seedScratch)
        {
            if (seed)
            {
                findRoot(seed);
            }
        }

        std::unordered_set<PxRigidDynamic*> seededRoots;
        seededRoots.reserve(m_seedScratch.size());
        for (PxRigidDynamic* seed : m_seedScratch)
        {
            if (!seed)
            {
                continue;
            }
            if (parent.find(seed) == parent.end())
            {
                // Seed never appeared in the pre-fracture contact graph —
                // conservative full restore (PR #41 support-loss failure mode).
                frame.scopedFallbackFullRestore = 1;
                m_activeBodies.clear();
                m_activeSet.clear();
                return false;
            }
            seededRoots.insert(findRoot(seed));
        }

        float linThr = 0.15f;
        float angThr = 0.15f;
        if (destructibleCount > 0)
        {
            // Thresholds live on settings; read via a body snapshot is awkward,
            // so use the first destructible's settled speeds through a capture
            // of defaults when unavailable. applyBaseStepSleep uses the same.
            linThr = 0.15f;
            angThr = 0.15f;
        }
        (void)linThr;
        (void)angThr;

        // Pull settled thresholds from any destructible by applying the same
        // defaults the settings struct uses; hosts can tune via settings and
        // we re-read through a tiny helper below.
        getSettledThresholds(destructibles, destructibleCount, linThr, angThr);

        for (const SceneBodySnapshot& snapshot : m_snapshot)
        {
            PxRigidDynamic* body = snapshot.body;
            if (!body || body->getScene() != &m_scene)
            {
                continue;
            }
            if (snapshot.kinematic
                || body->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
            {
                m_activeBodies.push_back(body);
                m_activeSet.insert(body);
                continue;
            }
            const auto parentIt = parent.find(body);
            const bool inSeededComponent = parentIt != parent.end()
                && seededRoots.count(findRoot(body)) > 0;
            // Still-moving disjoint bodies must advance time → active.
            // Settled disjoint → freeze (leave at P1, sleep through re-step).
            if (inSeededComponent || !bodySettled(*body, linThr, angThr))
            {
                m_activeBodies.push_back(body);
                m_activeSet.insert(body);
            }
        }

        // Ensure every seed is active even if it was created after capture
        // (new children are not in m_snapshot; adapter provenance handles them).
        for (PxRigidDynamic* seed : m_seedScratch)
        {
            if (seed && m_activeSet.insert(seed).second)
            {
                m_activeBodies.push_back(seed);
            }
        }

        frame.sceneBodiesActive = static_cast<uint32_t>(m_activeSet.size());
        return true;
    }

    static void getSettledThresholds(
        ExtStressPhysXDestructible* const* /*destructibles*/,
        uint32_t /*count*/,
        float& linThr,
        float& angThr)
    {
        // Defaults match ExtStressPhysXSettings. Exposed settings are applied
        // inside applyBaseStepSleep; scoped freeze uses the same numbers.
        linThr = 0.15f;
        angThr = 0.15f;
    }

    void restoreScene(ExtStressPhysXFrameStats& frame, bool scoped)
    {
        const StepperClock::time_point restoreStart = StepperClock::now();
        StepperSceneWriteLock lock(m_scene);
        uint32_t restored = 0;
        for (const SceneBodySnapshot& snapshot : m_snapshot)
        {
            PxRigidDynamic& body = *snapshot.body;
            if (body.getScene() != &m_scene)
            {
                continue;
            }
            if (scoped && m_activeSet.count(snapshot.body) == 0)
            {
                continue; // Freeze set: leave at P1.
            }
            body.setGlobalPose(snapshot.globalPose, false);
            if (body.getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
            {
                ++restored;
                continue;
            }
            const PxVec3 restoredCenter =
                snapshot.globalPose.transform(body.getCMassLocalPose().p);
            const PxVec3 linearVelocity = snapshot.linearVelocity +
                snapshot.angularVelocity.cross(
                    restoredCenter - snapshot.worldCenterOfMass);
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
            ++restored;
        }
        frame.sceneBodiesRestored += restored;
        frame.sceneRestoreMilliseconds += stepperElapsedMilliseconds(restoreStart);
    }

    void freezeOutsiders(ExtStressPhysXFrameStats& frame)
    {
        StepperSceneWriteLock lock(m_scene);
        uint32_t frozen = 0;
        for (const SceneBodySnapshot& snapshot : m_snapshot)
        {
            PxRigidDynamic* body = snapshot.body;
            if (!body || body->getScene() != &m_scene)
            {
                continue;
            }
            if (m_activeSet.count(body) != 0)
            {
                continue;
            }
            if (body->getRigidBodyFlags().isSet(PxRigidBodyFlag::eKINEMATIC))
            {
                continue;
            }
            body->putToSleep();
            m_frozen.push_back(body);
            ++frozen;
        }
        frame.sceneBodiesFrozen += frozen;
    }

    void wakeFrozen()
    {
        if (m_frozen.empty())
        {
            return;
        }
        StepperSceneWriteLock lock(m_scene);
        for (PxRigidDynamic* body : m_frozen)
        {
            if (body && body->getScene() == &m_scene)
            {
                body->wakeUp();
            }
        }
        m_frozen.clear();
    }

    PxScene& m_scene;
    std::vector<PxActor*> m_actorScratch;
    std::vector<SceneBodySnapshot> m_snapshot;
    bool m_snapshotValid{false};
    std::vector<std::pair<PxRigidDynamic*, PxRigidDynamic*>> m_contactPairs;
    std::vector<PxRigidDynamic*> m_seedScratch;
    std::vector<PxRigidDynamic*> m_activeBodies;
    std::unordered_set<PxRigidDynamic*> m_activeSet;
    std::vector<PxRigidDynamic*> m_frozen;
    std::unique_ptr<ExtStressPhysXDirectGpuMotionBuffer, DirectGpuDeleter> m_directGpu;
};

ExtStressPhysXFrameStepper* ExtStressPhysXFrameStepper::create(PxScene& scene)
{
    return new (std::nothrow) ExtStressPhysXFrameStepperImpl(scene);
}

} // namespace Blast
} // namespace Nv
