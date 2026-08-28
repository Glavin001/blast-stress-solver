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
    // Geometry: the bond's contact patch (m^2). Strength lives in the material.
    float area{1.0f};
    // Damage pool. Seed = area; a uniform seed makes authored strength
    // meaningless (see the CPU-path history).
    float health{1.0f};
    // Per-column compliance weight sqrt((E/E_ref)*A/L), computed by the CPU
    // StressProcessor and uploaded verbatim so both backends solve the same
    // weighted system. 1.0 when compliance weighting is disabled.
    float colScale{1.0f};
    // Index into the material table handed to create().
    std::uint32_t material{0};
};

/** Per-material stress limits (Pa), pre-resolved (no negative inheritance). */
struct ExtStressGpuMaterial
{
    float compressionElasticLimit{1.0f};
    float compressionFatalLimit{2.0f};
    float tensionElasticLimit{1.0f};
    float tensionFatalLimit{2.0f};
    float shearElasticLimit{1.0f};
    float shearFatalLimit{2.0f};
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
    /// Skip an input-stable island even when its last solve did NOT converge.
    ///
    /// A pristine downtown-scale island (~24k nodes) does not converge at any
    /// practical per-tick budget -- measured: 350k+ cumulative CG iterations
    /// without the converged flag latching, the iterate oscillating at the
    /// tolerance floor. Requiring convergence to skip therefore re-solved
    /// ~296k bonds every tick at rest, for an answer identical to the one
    /// already held. Reusing a frozen non-converged iterate for unchanged
    /// inputs is physics-equal to recomputing it; any bond damage its residual
    /// implies accrues identically either way.
    bool skipStableUnconverged{false};
    // The damage kernel reads each bond's own material from the table given
    // to create(); there are no global limits.
    bool applyDamage{false};
    /**
     * Skip islands whose inputs are unchanged since their last solve.
     *
     * An island is a disconnected component: its solution depends only on its
     * own nodes' velocities and its own warm-start impulses. If every dynamic
     * node's velocity is identical to the last solve and that solve reached
     * tolerance, re-solving reproduces the impulses it already holds, so the
     * work is waste. The device keeps those impulses resident and untouched.
     *
     * This mirrors StressProcessor::solveIslandAware on the CPU path, down to
     * the bit-exact velocity comparison -- see stress.cpp's angLin6Equal.
     */
    bool skipSettledIslands{false};
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
    /// Disconnected components the solver partitions its bonds into. Fixed for
    /// the solver's lifetime: topology changes go through a new solver.
    std::uint32_t islandCount{0};
    /// Of those, how many were settled and therefore not solved this call.
    /// Always 0 unless ExtStressGpuSolveParams::skipSettledIslands is set.
    std::uint32_t islandsSkipped{0};
    /// Host-side wall time inside solve(), split into work and waiting.
    ///
    /// solve_ms minus solveMilliseconds ran 3.84 ms against a 1.24 ms kernel,
    /// and "the host wrapper is expensive" and "the host is blocked on the
    /// GPU" are different problems with different fixes. Only the first is
    /// reclaimable by making host code faster, so the split is the ceiling on
    /// every host optimization and is measured rather than assumed.
    float hostPlanMilliseconds{0.0f};
    float hostSyncMilliseconds{0.0f};
    float hostFinishMilliseconds{0.0f};
    /// Extra full solve passes spent chasing convergence. Each is another
    /// enqueue/sync/finish cycle, so it multiplies everything above.
    std::uint32_t extraPasses{0};
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
        const ExtStressGpuMaterial* materials = nullptr,
        std::uint32_t materialCount = 0,
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

    /**
     * Bonds whose impulses the last solve could have changed, valid until the
     * next solve or release().
     *
     * With skipSettledIslands off this is every bond, in index order. With it
     * on it is exactly the bonds of the islands that were solved -- the rest
     * were not read back at all, so a caller mirroring impulses host-side must
     * consult this list rather than copying the whole array, or its per-frame
     * cost stays proportional to total bonds and the saving is given straight
     * back on the host side.
     */
    virtual const std::uint32_t* lastChangedBonds(std::uint32_t& count) const = 0;

    /**
     * Drop one bond, swap-with-last, mirroring the host solver's own
     * replaceWithLast so the two arrays stay index-for-index identical.
     *
     * Exists because the alternative is destroying and rebuilding the solver,
     * which is what the caller used to do on every fracture: at city scale a
     * bond breaks on most ticks, so the device state -- topology, island
     * partition, warm-start impulses and the settled baseline -- was thrown
     * away and rebuilt every tick, and nothing could ever be carried forward.
     *
     * Topology is re-uploaded lazily on the next solve, together with a
     * repartition and a warm-start reset. That reset is not a compromise: a
     * rebuilt solver started cold too, so this is the same physics, arrived at
     * without the allocation churn.
     */
    virtual bool removeBond(std::uint32_t bondIndex) = 0;

    virtual void resetWarmStart() = 0;

    virtual std::uint32_t nodeCount() const = 0;
    virtual std::uint32_t bondCount() const = 0;
    virtual const ExtStressGpuTelemetry& telemetry() const = 0;

protected:
    virtual ~ExtStressGpuSolver() = default;
};

} // namespace Blast
} // namespace Nv
