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
    /// Cap on the section-modulus bending gain, or <= 0 for the legacy fold of
    /// bend into axial stress.
    ///
    /// Passed rather than read, because the formula body compiles for the
    /// device and cannot reach an environment variable -- and because the host
    /// and device solving different bending is precisely the divergence the
    /// shared formula header exists to prevent.
    float bendGainMax{0.0f};
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

    /// Bond-stress walk, split so the next optimisation is chosen from
    /// evidence rather than from which part looks expensive.
    float bondStressUploadMs{0.0f};
    float bondStressKernelMs{0.0f};
    float bondStressReadbackMs{0.0f};
    float bondStressSyncMs{0.0f};
    float bondStressHostMs{0.0f};
    float bondStressPrepMs{0.0f};
    float bondStressProbeEmptyMs{0.0f};
    float bondStressProbeKernelMs{0.0f};
    float bondStressProbeKernel2Ms{0.0f};
    float bondStressProbeCopyMs{0.0f};
    float bondStressEnqueueMs{0.0f};
    std::uint64_t bondStressBytesUp{0};
    std::uint64_t bondStressBytesDown{0};
};

/**
 * Persistent CUDA implementation of Blast's low-level CGNR stress solve.
 *
 * Graph topology and warm-start impulses remain resident on the device. Per
 * solve, callers upload only node velocity/load inputs and read impulses only
 * when validation, fracture evaluation, or diagnostics require them.
 */
/**
Flattened solver-bond-group topology for the device bond-stress walk.

Mirrors SupportGraphProcessor's own flattened copy of m_solverBondsData. The
payload arrays are indexed by BLAST BOND, which is an asset-level index that is
never permuted, so they are uploaded once per graph resync. Only the three CSR
arrays change when a bond breaks, and they are small enough to refresh whole.

Member SLOT order inside a group is significant: it is the order the serial
host walk emits bond removals in, and it is not sorted.
*/
struct ExtStressGpuBondStressTopology
{
    std::uint32_t groupCount{0};
    std::uint32_t memberSlotCount{0};
    std::uint32_t graphNodeCount{0};
    std::uint32_t blastBondCount{0};

    /// CSR over member slots. [groupCount] each.
    const std::uint32_t* groupBegin{nullptr};
    const std::uint32_t* groupSize{nullptr};
    /// [memberSlotCount] -- blast bond index living in each slot.
    const std::uint32_t* memberBlastBond{nullptr};

    /// Static per blast bond. [blastBondCount], or 3x that for the vectors.
    const std::uint32_t* bondNode0{nullptr};
    const std::uint32_t* bondNode1{nullptr};
    const std::uint32_t* bondMaterial{nullptr};
    const float* bondNormal{nullptr};    ///< already sign-aligned to nodeDisp
    const float* bondCentroid{nullptr};
    const float* bondNodeDisp{nullptr};

    /// The three ELASTIC limits per material, in the caller's OWN resolved
    /// table: compression, tension, shear. Carried here rather than reusing
    /// the solver's copy, which is uploaded once at create() and goes stale
    /// the moment materials are pushed again -- the host walk resolves each
    /// bond against the live table, so the device must too.
    const float* materialElasticLimits{nullptr};
    std::uint32_t materialCount{0};

    /// Whether the three CSR arrays changed since the last call. They only
    /// change when a bond breaks, so re-uploading them every tick spends
    /// hundreds of KB per structure per tick to send identical bytes.
    bool csrDirty{true};

    /// Group slots the host has reassigned by replaceWithLast since the last
    /// call, as (destination, source) pairs in the order they were applied.
    ///
    /// The device keeps its per-group stress in the slot, and a slot that has
    /// been handed to a different group still holds the OLD group's values
    /// until the new one is next processed. Everything with live members is
    /// processed every tick so the window is narrow, but it is real: a longer
    /// run with more removals showed 132 of 653M values diverging where a
    /// shorter one showed none.
    const std::uint32_t* groupSwapDst{nullptr};
    const std::uint32_t* groupSwapSrc{nullptr};
    std::uint32_t groupSwapCount{0};
};

/**
What the device walk hands back. All pointers are host-side staging owned by
the solver and valid until the next updateBondStress call.
*/
struct ExtStressGpuBondStressResult
{
    /// Blast bond indices, in exactly the order the serial walk emits them:
    /// group index ascending, then member slot ascending inside each group.
    const std::uint32_t* bondIndicesToRemove{nullptr};
    std::uint32_t removeCount{0};
    std::uint32_t overstressedBondCount{0};
    /// [graphNodeCount], set-to-1. Feeds the E1 fracture-walk node skip.
    const std::uint8_t* nodeOverstressed{nullptr};
    /// Per solver bond group. A group the walk skipped keeps its previous
    /// values, exactly as the host's BondData does.
    const float* groupStressNormal{nullptr};
    const float* groupStressShear{nullptr};
    const float* groupNormal{nullptr};    ///< 3 per group
    const float* groupCentroid{nullptr};  ///< 3 per group
};

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


    /**
     * Hand over the flattened group topology. Cheap to call: it re-uploads the
     * static payload, so it belongs on a graph resync, not on a tick. The three
     * CSR arrays are refreshed on every updateBondStress instead.
     */
    virtual bool setBondStressTopology(const ExtStressGpuBondStressTopology& topology) = 0;

    /**
     * The bond-stress walk, on the device.
     *
     * Consumes the impulses already resident from the last solve -- which is
     * what makes this worth doing, since those impulses are currently copied
     * back to the host for no other reason than to feed this walk.
     *
     * One thread per group, with the member loop run serially in slot order.
     * That is deliberate: it reproduces the host's sequential float
     * accumulation operation for operation, so the dual-run audit can demand
     * bit-equality rather than a tolerance.
     */
    /**
     * Fetch the per-group normal and centroid, which updateBondStress leaves
     * on the device.
     *
     * They are 6 of the 8 floats per group and almost nothing reads them on a
     * given tick: excess forces wants the normal only for bonds that actually
     * broke, and the virial wants the centroid only when chunk crushing is on.
     * Pulling them across every tick cost more than the kernel did.
     */
    /// Per-group stresses, also left on the device. Nothing reads a bond's
    /// stress unless its node came back flagged, which on a settled city is
    /// never, so pulling them across every tick was pure loss.
    virtual bool readbackGroupStresses(
        const float*& stressNormal, const float*& stressShear) = 0;

    virtual bool readbackGroupVectors(
        const float*& groupNormal, const float*& groupCentroid) = 0;

    virtual bool updateBondStress(
        const ExtStressGpuBondStressTopology& csr,
        const float* blastBondHealth,
        float unbreakableLimit,
        ExtStressGpuBondStressResult& result) = 0;

    /**
     * Whether a topology edit is queued but not yet applied to the device.
     *
     * removeBond only RECORDS the swap-with-last; it is replayed inside the
     * next solve. The host applies its own replaceWithLast immediately, so
     * while this is true the two impulse arrays are indexed differently and
     * anything reading device impulses by host bond index reads the wrong
     * bond.
     */
    virtual bool hasPendingTopologyChange() const = 0;

    /**
     * Replay just the queued impulse swap-with-lasts, so the device impulse
     * array is indexed the way the host's already is.
     *
     * Narrow on purpose: it does not repartition islands, re-upload topology
     * or drop the captured graph, and it leaves the topology dirty so the next
     * real solve still does all of that. It exists because a reader that only
     * needs impulses BY INDEX should not have to wait for a solve that may
     * never come -- at rest the solve early-outs every tick, so the swaps
     * would otherwise sit unapplied indefinitely and the reader would be
     * locked out of exactly the regime it is cheapest to measure in.
     */
    virtual bool flushImpulsePermutation() = 0;


    virtual void resetWarmStart() = 0;

    virtual std::uint32_t nodeCount() const = 0;
    virtual std::uint32_t bondCount() const = 0;
    virtual const ExtStressGpuTelemetry& telemetry() const = 0;

protected:
    virtual ~ExtStressGpuSolver() = default;
};

} // namespace Blast
} // namespace Nv
