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
    uint32_t maxPasses;
    bool quietCaptureSkip;
    bool scopedResim;
    bool useDirectGpuMotionState;
    float settledLinearSpeed;
    float settledAngularSpeed;

    ExtStressPhysXResimOptions()
        : maxPasses(1)
        , quietCaptureSkip(false)
        , scopedResim(true)
        , useDirectGpuMotionState(false)
        , settledLinearSpeed(0.15f)
        , settledAngularSpeed(0.15f)
    {
    }
};

struct ExtStressPhysXFrameStats
{
    uint32_t resimPasses;
    uint32_t sceneBodiesCaptured;
    uint32_t sceneBodiesRestored;
    uint32_t sceneBodiesActive;
    uint32_t sceneBodiesFrozen;
    uint32_t scopedFallbackFullRestore;
    uint64_t splits;

    double simulateMilliseconds;
    double simulateSubmitMilliseconds;
    double fetchResultsMilliseconds;
    double tickMilliseconds;
    double beginTickMilliseconds;
    double solveTickMilliseconds;
    double endTickMilliseconds;

    double sceneCaptureMilliseconds;
    double adapterCaptureMilliseconds;
    double sceneRestoreMilliseconds;
    double adapterRestoreMilliseconds;

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
        , sceneBodiesActive(0)
        , sceneBodiesFrozen(0)
        , scopedFallbackFullRestore(0)
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

class ExtStressPhysXFrameHooks
{
public:
    virtual ~ExtStressPhysXFrameHooks() {}
    virtual void onCapture() {}
    virtual void onRestore() {}
    virtual void onPostFetchResults(uint32_t pass) { (void)pass; }

    /**
    Chunks were pulverized during this tick. Fires once per chunk, as it
    happens, before the frame's resimulation passes are decided.

    Resimulation restores MOTION only -- fracture topology, masses and shapes
    are kept (see ExtStressPhysXDestructible::restoreResimulationSnapshot), so
    a chunk crushed in pass 0 stays crushed in pass 1 and does not re-fire
    here. A host counting these does NOT need to rewind them in onRestore(),
    unlike per-frame impact counters.
    */
    virtual void onChunkDestroyed(
        ExtStressPhysXDestructible& destructible,
        const ExtStressPhysXChunkDestroyed* events,
        uint32_t count)
    {
        (void)destructible;
        (void)events;
        (void)count;
    }
    virtual bool forceResimulationCapture() const { return false; }
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

class NV_DLL_EXPORT ExtStressPhysXFrameStepper
{
public:
    static ExtStressPhysXFrameStepper* create(physx::PxScene& scene);

    virtual void release() = 0;

    virtual void recordDynamicContactPair(
        physx::PxRigidActor* actor0,
        physx::PxRigidActor* actor1) = 0;

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
