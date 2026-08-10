#include "extensions/stressphysx/NvBlastExtStressPhysXResim.h"

#include <chrono>
#include <new>
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

} // namespace

class ExtStressPhysXFrameStepperImpl final : public ExtStressPhysXFrameStepper
{
public:
    explicit ExtStressPhysXFrameStepperImpl(PxScene& scene)
        : m_scene(scene)
    {
    }

    void release() override
    {
        delete this;
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
        if (passesRemaining > 0)
        {
            captureAll(destructibles, destructibleCount, frameHooks, frame);
        }

        for (;;)
        {
            // resimPasses counts completed restores; the upcoming simulate is
            // the base step while it is still zero.
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
                return true;
            }
            --passesRemaining;
            ++frame.resimPasses;

            frameHooks.onRestore();
            restoreScene(frame);

            const StepperClock::time_point adapterRestoreStart = StepperClock::now();
            for (uint32_t i = 0; i < destructibleCount; ++i)
            {
                destructibles[i]->restoreResimulationSnapshot();
            }
            frame.adapterRestoreMilliseconds +=
                stepperElapsedMilliseconds(adapterRestoreStart);

            if (passesRemaining > 0)
            {
                // Re-capture so the next rollback rewinds this pass's fracture
                // children to their creation state. The final allowed pass can
                // never be rolled back, so its re-capture would be dead work.
                captureAll(destructibles, destructibleCount, frameHooks, frame);
            }
        }
    }

private:
    // Motion state of one scene body at capture time. Mirrors the adapter's
    // per-body snapshot: the stored linvel is re-expressed at the current COM
    // on restore in case a split moved the body's mass frame.
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

    void captureAll(
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount,
        ExtStressPhysXFrameHooks& hooks,
        ExtStressPhysXFrameStats& frame)
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

            m_snapshot.clear();
            m_snapshot.reserve(fetched);
            for (PxU32 i = 0; i < fetched; ++i)
            {
                PxRigidDynamic& body = *static_cast<PxRigidDynamic*>(m_actorScratch[i]);
                SceneBodySnapshot snapshot;
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
                snapshot.worldCenterOfMass =
                    snapshot.globalPose.transform(body.getCMassLocalPose().p);
                m_snapshot.push_back(snapshot);
            }
        }
        frame.sceneBodiesCaptured = static_cast<uint32_t>(m_snapshot.size());
        frame.sceneCaptureMilliseconds +=
            stepperElapsedMilliseconds(sceneCaptureStart);

        // The adapter capture (re)starts each destructible's fracture-child
        // provenance window; its restore below re-derives those children after
        // the generic scene rewind.
        const StepperClock::time_point adapterCaptureStart = StepperClock::now();
        for (uint32_t i = 0; i < destructibleCount; ++i)
        {
            destructibles[i]->captureResimulationSnapshot();
        }
        hooks.onCapture();
        frame.adapterCaptureMilliseconds +=
            stepperElapsedMilliseconds(adapterCaptureStart);
    }

    void restoreScene(ExtStressPhysXFrameStats& frame)
    {
        const StepperClock::time_point restoreStart = StepperClock::now();
        StepperSceneWriteLock lock(m_scene);
        uint32_t restored = 0;
        for (const SceneBodySnapshot& snapshot : m_snapshot)
        {
            PxRigidDynamic& body = *snapshot.body;
            if (body.getScene() != &m_scene)
            {
                continue; // Removed since capture; nothing to rewind.
            }
            body.setGlobalPose(snapshot.globalPose, false);
            // Kinematic status can change during a fracture tick and velocity
            // writes are rejected on kinematic actors, so test the current flag.
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

    PxScene& m_scene;
    std::vector<PxActor*> m_actorScratch;
    std::vector<SceneBodySnapshot> m_snapshot;
};

ExtStressPhysXFrameStepper* ExtStressPhysXFrameStepper::create(PxScene& scene)
{
    return new (std::nothrow) ExtStressPhysXFrameStepperImpl(scene);
}

} // namespace Blast
} // namespace Nv
