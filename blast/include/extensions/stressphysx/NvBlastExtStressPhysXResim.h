// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES.

#ifndef NVBLASTEXTSTRESSPHYSXRESIM_H
#define NVBLASTEXTSTRESSPHYSXRESIM_H

#include "NvBlastExtStressPhysX.h"

namespace Nv
{
namespace Blast
{

struct ExtStressPhysXResimOptions
{
    // Maximum rollback + re-step iterations per frame after a fracture.
    // 0 disables resimulation; 1 matches the reference Rapier defaults.
    uint32_t maxPasses;

    ExtStressPhysXResimOptions()
        : maxPasses(1)
    {
    }
};

struct ExtStressPhysXFrameStats
{
    uint32_t resimPasses;
    uint32_t sceneBodiesCaptured;
    uint32_t sceneBodiesRestored;
    uint64_t splits;

    // Totals across the base step and every re-step. simulateMilliseconds is
    // submit + fetchResults (matches the demo's physics_step_ms); the split
    // fields isolate GPU work kickoff from the host sync wait.
    double simulateMilliseconds;
    double simulateSubmitMilliseconds;
    double fetchResultsMilliseconds;
    double tickMilliseconds;
    double beginTickMilliseconds;
    double solveTickMilliseconds;
    double endTickMilliseconds;

    // Capture/restore are split so GPU↔CPU sync cost (PhysX body get/set) can
    // be separated from Blast adapter provenance bookkeeping.
    double sceneCaptureMilliseconds;
    double adapterCaptureMilliseconds;
    double sceneRestoreMilliseconds;
    double adapterRestoreMilliseconds;

    // Base step (pass 0) vs all rollback re-steps combined. Quiet frames only
    // fill the base_* fields; fracture frames fill both.
    double baseSimulateSubmitMilliseconds;
    double baseFetchResultsMilliseconds;
    double baseTickMilliseconds;
    double resimSimulateSubmitMilliseconds;
    double resimFetchResultsMilliseconds;
    double resimTickMilliseconds;

    ExtStressPhysXFrameStats()
        : resimPasses(0)
        , sceneBodiesCaptured(0)
        , sceneBodiesRestored(0)
        , splits(0)
        , simulateMilliseconds(0.0)
        , simulateSubmitMilliseconds(0.0)
        , fetchResultsMilliseconds(0.0)
        , tickMilliseconds(0.0)
        , beginTickMilliseconds(0.0)
        , solveTickMilliseconds(0.0)
        , endTickMilliseconds(0.0)
        , sceneCaptureMilliseconds(0.0)
        , adapterCaptureMilliseconds(0.0)
        , sceneRestoreMilliseconds(0.0)
        , adapterRestoreMilliseconds(0.0)
        , baseSimulateSubmitMilliseconds(0.0)
        , baseFetchResultsMilliseconds(0.0)
        , baseTickMilliseconds(0.0)
        , resimSimulateSubmitMilliseconds(0.0)
        , resimFetchResultsMilliseconds(0.0)
        , resimTickMilliseconds(0.0)
    {
    }
};

/**
Optional host extension points for the frame stepper. Override onCapture /
onRestore to snapshot host state that PhysX cannot see (frame counters,
gameplay bookkeeping) so it reflects only the final pass. Override solveAll to
run the destructibles' solveTick phase on a host thread pool; the default runs
them serially. onPostFetchResults runs after every fetchResults with the pass
index (0 = base step, 1+ = re-steps), before the destructible tick.
*/
class ExtStressPhysXFrameHooks
{
public:
    virtual ~ExtStressPhysXFrameHooks() {}
    virtual void onCapture() {}
    virtual void onRestore() {}
    virtual void onPostFetchResults(uint32_t pass) { (void)pass; }
    virtual bool solveAll(
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t count)
    {
        bool solved = true;
        for (uint32_t i = 0; i < count; ++i)
        {
            solved = destructibles[i]->solveTick() && solved;
        }
        return solved;
    }
};

/**
Library-owned fracture-frame resimulation loop (engine contract §4/§2.8):
step the scene, let stress decide fracture, roll the step back against the kept
fractured topology, and re-step so contacts resolve against the split pieces.

stepFrame runs simulate/fetchResults plus the destructibles' three-phase tick,
then up to options.maxPasses rollback + re-step iterations while new fractures
occur. The rollback restores every PxRigidDynamic in the scene that existed at
capture (host projectiles included — Rapier's "every non-fixed body"
semantics), then each destructible re-derives its fracture-created children.
The final allowed pass skips its dead re-capture.

Contract: run all host per-frame scene mutations (spawns, kinematic toggles,
actor removal) before stepFrame, never inside hooks; bodies must not be
released between capture and restore. Rollback restores motion state, not
solver/contact warm-start caches, so a re-stepped frame is output-faithful
rather than bit-identical — by design.
*/
class NV_DLL_EXPORT ExtStressPhysXFrameStepper
{
public:
    static ExtStressPhysXFrameStepper* create(physx::PxScene& scene);

    virtual void release() = 0;

    /**
     * Returns false when simulate/fetchResults fails or any destructible tick
     * phase reports an error; stats (optional) always reflects the work done.
     */
    virtual bool stepFrame(
        float dt,
        const physx::PxVec3& worldGravity,
        ExtStressPhysXDestructible* const* destructibles,
        uint32_t destructibleCount,
        const ExtStressPhysXResimOptions& options,
        ExtStressPhysXFrameHooks* hooks = nullptr,
        ExtStressPhysXFrameStats* stats = nullptr) = 0;

protected:
    virtual ~ExtStressPhysXFrameStepper() {}
};

} // namespace Blast
} // namespace Nv

#endif // NVBLASTEXTSTRESSPHYSXRESIM_H
