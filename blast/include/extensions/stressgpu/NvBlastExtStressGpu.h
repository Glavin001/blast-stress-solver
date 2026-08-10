// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#pragma once

#include <cstdint>

namespace Nv
{
namespace Blast
{

struct ExtStressGpuNode
{
    float position[3];
    float mass;
    float inertia;
};

struct ExtStressGpuBond
{
    std::uint32_t node0;
    std::uint32_t node1;
    float centroid[3];
    float normal[3];
    float area{1.0f};
    float health{1.0f};
};

struct ExtStressGpuVec3
{
    float x;
    float y;
    float z;
};

struct ExtStressGpuImpulse
{
    ExtStressGpuVec3 angular;
    ExtStressGpuVec3 linear;
};

struct ExtStressGpuSolveParams
{
    std::uint32_t maxIterations{25};
    float tolerance{0.001f};
    bool warmStart{true};
    bool applyDamage{false};
    float compressionElasticLimit{1.0f};
    float compressionFatalLimit{2.0f};
    float tensionElasticLimit{1.0f};
    float tensionFatalLimit{2.0f};
    float shearElasticLimit{1.0f};
    float shearFatalLimit{2.0f};
};

struct ExtStressGpuTelemetry
{
    float uploadMilliseconds{0.0f};
    float solveMilliseconds{0.0f};
    float downloadMilliseconds{0.0f};
    std::uint32_t iterations{0};
    std::uint64_t hostToDeviceBytes{0};
    std::uint64_t deviceToHostBytes{0};
    bool converged{false};
};

/**
 * Persistent CUDA implementation of Blast's low-level CGNR stress solve.
 *
 * Graph topology and warm-start impulses remain resident on the device. Per
 * solve, callers upload only node velocity/load inputs and read impulses only
 * when validation, fracture evaluation, or diagnostics require them.
 */
class ExtStressGpuSolver
{
public:
    static ExtStressGpuSolver* create(
        const ExtStressGpuNode* nodes,
        std::uint32_t nodeCount,
        const ExtStressGpuBond* bonds,
        std::uint32_t bondCount,
        void* cudaContext = nullptr);

    virtual void release() = 0;

    virtual bool solve(
        const ExtStressGpuImpulse* nodeVelocities,
        const ExtStressGpuSolveParams& params) = 0;

    virtual bool readbackImpulses(
        ExtStressGpuImpulse* bondImpulses,
        std::uint32_t capacity) = 0;

    /**
     * Executes a solve and reads its impulses with one host synchronization.
     * Prefer this when the caller always consumes impulses immediately.
     */
    virtual bool solveAndReadbackImpulses(
        const ExtStressGpuImpulse* nodeVelocities,
        const ExtStressGpuSolveParams& params,
        ExtStressGpuImpulse* bondImpulses,
        std::uint32_t capacity) = 0;

    virtual bool readbackBrokenBonds(
        std::uint32_t* bondIndices,
        std::uint32_t capacity,
        std::uint32_t& count) = 0;

    virtual bool readbackBondHealth(float* health, std::uint32_t capacity) = 0;

    virtual void resetWarmStart() = 0;

    virtual std::uint32_t nodeCount() const = 0;
    virtual std::uint32_t bondCount() const = 0;
    virtual const ExtStressGpuTelemetry& telemetry() const = 0;

protected:
    virtual ~ExtStressGpuSolver() = default;
};

} // namespace Blast
} // namespace Nv
