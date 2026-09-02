// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "NvBlastExtStressGpu.h"
#include "NvBlastExtStressFormula.h"

/// NV_NORMALIZATION_EPSILON, without dragging NvFoundation into the .cu.
#define NVBLAST_STRESS_NORMALIZATION_EPSILON float(1e-20f)

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace Nv
{
namespace Blast
{
namespace
{

constexpr std::uint32_t kBlockSize = 256;

/// BLAST_GPU_SKIP_DEBUG=1 traces the settled-island decision every 60 solves.
/// Kept in because "islands skipped = 0" has two very different causes -- the
/// mechanism is off, or the scene genuinely never settles -- and they are
/// indistinguishable from the aggregate telemetry.
const bool s_debug = std::getenv("BLAST_GPU_SKIP_DEBUG") != nullptr;

/// BLAST_GPU_WHOLE_RESET_ON_TOPOLOGY=1 restores the pre-incremental behaviour:
/// full warm-start memset + whole-baseline drop on every topology change and
/// every break. Value-checked kill switch for the incremental path below.
/// G1: gather instead of scatter in the right-multiply (default ON, =0
/// restores the atomic scatter).
///
/// The scatter wrote each bond's contribution into its two nodes with
/// atomicAdd -- 3 floats x 2 vectors x 2 nodes = 12 global float atomics per
/// bond, every iteration. That is the dominant cost of the solve (the code
/// said so itself) and it is also why the solve is not reproducible: float
/// addition is not associative and the atomic order is not deterministic, so
/// the same scene gives slightly different impulses every run, which is why
/// every experiment on this tree needs n>=2 arms and a noise floor.
///
/// The gather visits the same bonds and computes the same terms; only the
/// summation moves. Each node loops over its own incident bonds in a fixed
/// CSR order and writes its slot once, exclusively. No atomics, and the
/// per-node sum order is fixed by the topology, so it is reproducible.
///
/// MEASURED, AND IT DOES NOT PAY -- DEFAULT OFF. City-scale A/B, one binary,
/// alternating arms, n=2, in the `stress` bracket (medians, ms):
///
///                       at rest        under load (>800 awake)
///   scatter (atomics)   3.24  3.27     6.87  7.77
///   gather              3.95  3.56     7.12  7.90
///
/// Within-arm spread at rest is 0.03 ms, so +0.5 ms at rest is real, and the
/// load case is a wash inside its own 0.9 ms spread. The premise -- that the
/// twelve atomics per bond dominated -- was not supported: at 230k bonds the
/// atomic traffic spreads across thousands of distinct node addresses and
/// does not serialize the way a single hot address would.
///
/// Why it LOSES at rest: the scatter iterates the compacted ACTIVE BOND list,
/// which is nearly empty once islands settle, so it costs almost nothing.
/// The gather iterates active NODES, and a static node is never settled by
/// design (nodeSettled keeps kNoIsland nodes always live, because they are
/// boundaries shared by every island), so the gather walks the full adjacency
/// of every support node every iteration just to find that all of its bonds
/// are skipped.
///
/// Kept, off by default, for two reasons: it is proven correct (the
/// equivalence harness passes all eight checks, including identical
/// broken-bond counts on every tick), and it is the only atomics-free
/// formulation available if the determinism work resumes -- the reproducible
/// solve needs this half. Turning it on requires first making the active-node
/// list exclude nodes with no active bonds, which is what would restore the
/// compaction the scatter gets for free.
/// Per-kernel timing for the solve, because the external profiler is not
/// available here: ncu needs GPU performance counters, and the driver refuses
/// them without a host-level modprobe setting this container cannot change
/// (ERR_NVGPUCTRPERM). Rather than reason about the kernel mix from source, we
/// measure it.
///
/// BLAST_GPU_KERNEL_PROFILE=1 bypasses the captured graph and launches the
/// same kernels eagerly with a CUDA event around each, accumulating per-name
/// totals. Bypassing the graph is itself informative: the difference between
/// the graph total and the sum of the kernels IS the launch overhead the graph
/// exists to remove.
struct KernelProfile
{
    static constexpr std::size_t kMaxSlots = 512;
    struct Slot
    {
        const char* name{nullptr};
        cudaEvent_t start{nullptr};
        cudaEvent_t stop{nullptr};
    };
    std::vector<Slot> slots;
    std::size_t used{0};
    std::map<std::string, std::pair<double, std::uint32_t>> totals;
    bool active{false};

    void ensure()
    {
        if (slots.empty())
        {
            slots.resize(kMaxSlots);
            for (Slot& slot : slots)
            {
                cudaEventCreate(&slot.start);
                cudaEventCreate(&slot.stop);
            }
        }
    }

    void begin(const char* name, cudaStream_t stream)
    {
        if (!active || used >= kMaxSlots)
        {
            return;
        }
        ensure();
        slots[used].name = name;
        cudaEventRecord(slots[used].start, stream);
    }

    void end(cudaStream_t stream)
    {
        if (!active || used >= kMaxSlots)
        {
            return;
        }
        cudaEventRecord(slots[used].stop, stream);
        ++used;
    }

    /// Drain after a synchronize: event elapsed time is only valid once the
    /// stream has caught up.
    void harvest()
    {
        for (std::size_t i = 0; i < used; ++i)
        {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, slots[i].start, slots[i].stop) == cudaSuccess)
            {
                auto& entry = totals[slots[i].name];
                entry.first += static_cast<double>(ms);
                entry.second += 1;
            }
        }
        used = 0;
    }

    void dump(const char* label, std::uint32_t solves) const
    {
        double grand = 0.0;
        for (const auto& entry : totals)
        {
            grand += entry.second.first;
        }
        std::fprintf(stderr, "\n=== kernel profile (%s), %u solves ===\n", label, solves);
        std::fprintf(
            stderr, "%-34s %10s %9s %10s %7s\n",
            "kernel", "ms/solve", "launches", "us/launch", "share");
        std::vector<std::pair<std::string, std::pair<double, std::uint32_t>>> rows(
            totals.begin(), totals.end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            return a.second.first > b.second.first;
        });
        for (const auto& row : rows)
        {
            const double msPerSolve = row.second.first / std::max(1u, solves);
            const double launchesPerSolve =
                static_cast<double>(row.second.second) / std::max(1u, solves);
            std::fprintf(
                stderr, "%-34s %10.4f %9.1f %10.2f %6.1f%%\n",
                row.first.c_str(), msPerSolve, launchesPerSolve,
                row.second.first * 1000.0 / std::max(1u, row.second.second),
                100.0 * row.second.first / std::max(grand, 1e-9));
        }
        std::fprintf(
            stderr, "%-34s %10.4f\n", "TOTAL (sum of kernels)",
            grand / std::max(1u, solves));
    }
};

static bool kernelProfileEnabled()
{
    static const bool enabled = [] {
        const char* raw = std::getenv("BLAST_GPU_KERNEL_PROFILE");
        return raw != nullptr && std::string(raw) != "0";
    }();
    return enabled;
}

static bool gatherRightMultiplyEnabled()
{
    static const bool enabled = [] {
        // Default ON. This was default-off for one commit after the gather
        // measured SLOWER than the scatter it replaces; that measurement was
        // real but its cause was the static-node degree walk inside the kernel
        // (see gatherRightMultiply), not the gather itself. With that removed
        // the gather wins at load and ties at rest, so it is the path.
        const char* raw = std::getenv("BLAST_GPU_GATHER");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return enabled;
}

/// BLAST_GPU_GRAPH_STATS=1: how often the active lists are actually refreshed,
/// how often the captured graph is re-instantiated, and what the mid-enqueue
/// sync costs. Pure diagnostics. Guessing these three wrong is how a sound
/// idea gets implemented badly and then measured as a regression.
static bool islandTraceEnabled()
{
    static const bool enabled = std::getenv("BLAST_ISLAND_TRACE") != nullptr;
    return enabled;
}

static /// Device-side early exit via a CUDA graph conditional while-node.
///
/// On by default: it is a strict improvement whenever the solve converges
/// before its budget, and identical work when it does not. `BLAST_GPU_COND_LOOP=0`
/// restores the unrolled loop for A/B measurement. Requires CUDA 12.3+ for
/// cudaGraphCondTypeWhile; the toolkit here is 12.8. If any part of the setup
/// is unsupported at runtime the code falls back to the unrolled form, so this
/// can never change the answer -- only the cost.
bool conditionalLoopEnabled()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("BLAST_GPU_COND_LOOP");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return enabled;
}

/// Iterations per condition check in the conditional CG loop. See
/// launchConditionalLoopCaptured for why this is not 1.
std::uint32_t conditionalLoopChunk()
{
    static const std::uint32_t chunk = []() {
        const char* raw = std::getenv("BLAST_GPU_COND_CHUNK");
        const long parsed = raw ? std::atol(raw) : 0;
        return parsed > 0 ? static_cast<std::uint32_t>(std::min(parsed, 64L)) : 2u;
    }();
    return chunk;
}

/// Rebuild the island partition only when a removal actually disconnected
/// something. `BLAST_GPU_DEFER_REPARTITION=0` restores the unconditional
/// rebuild for A/B measurement.
/// Sparse topology upload. `BLAST_GPU_DELTA_UPLOAD=0` forces the whole-array
/// upload, for A/B and as a safety valve.
/// Node-space CGLS: solve for mu with lambda = lambda0 + W mu instead of
/// carrying bond-length vectors through the loop. Default ON.
///
/// Measured on the city scenario: 1.31x at 298k bonds and 1.98x at 1.19M bonds
/// -- the win grows with scale because it is fundamentally about the working
/// set fitting the L2, and the bond-length vectors are what pushed it out.
/// `BLAST_GPU_NODE_SPACE=0` restores the bond-space loop.
/// Block-Jacobi (6x6 per node) preconditioning of the node-space iteration.
/// Default OFF: derivation says it costs 2 matvecs/iteration against a 1.5-3x
/// iteration reduction, i.e. a wash, and this flag exists to MEASURE that
/// rather than assume it. Also the smoother a multigrid V-cycle would need.
bool jacobiEnabled()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("BLAST_GPU_JACOBI");
        return raw != nullptr && std::string(raw) != "0";
    }();
    return enabled;
}

/// Skip within-solve converged islands in the matvec. A/B switch: it saves the
/// matvec work but costs one scattered load per node, so which way it pays
/// depends on how many islands converge early.
bool skipConvergedEnabled()
{
    static const bool e = []() { const char* r = std::getenv("BLAST_GPU_SKIP_CONVERGED"); return r == nullptr || std::string(r) != "0"; }();
    return e;
}

bool nodeSpaceEnabled()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("BLAST_GPU_NODE_SPACE");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return enabled;
}

/// Relabel a separated piece locally instead of repartitioning the whole graph.
///
/// DEFAULT OFF: it is a large cost win (fracture tick 3.40 -> 2.23 ms) that
/// measurably changes the physics -- residual/tolerance went 1.05x -> 5.06x,
/// islands over tolerance 1.6% -> 10.5%, and 12% more bonds broke over the same
/// 1200-tick demolition. Something about the incrementally-maintained partition
/// is not equivalent to the rebuilt one and it is not yet understood, so the
/// verified path stays the default. `BLAST_GPU_LOCAL_SPLIT=1` enables it for
/// investigation.
/// Assert the incremental partition against a full rebuild every tick. Debug
/// only: it does the very work the incremental path exists to avoid.
bool verifyPartitionEnabled()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("BLAST_GPU_VERIFY_PARTITION");
        return raw != nullptr && std::string(raw) != "0";
    }();
    return enabled;
}

bool localSplitRelabelEnabled()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("BLAST_GPU_LOCAL_SPLIT");
        return raw != nullptr && std::string(raw) != "0";
    }();
    return enabled;
}

bool deltaUploadEnabled()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("BLAST_GPU_DELTA_UPLOAD");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return enabled;
}

bool deferRepartitionEnabled()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("BLAST_GPU_DEFER_REPARTITION");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return enabled;
}

bool graphStatsEnabled()
{
    static const bool enabled = [] {
        const char* raw = std::getenv("BLAST_GPU_GRAPH_STATS");
        return raw != nullptr && raw[0] == '1';
    }();
    return enabled;
}

/// Bake the captured graph at FULL launch width and stop tracking the active
/// counts on the host. BLAST_GPU_FULL_LAUNCH=0 restores the sized-capture path.
///
/// The active lists still do their job: the kernels walk the compacted lists
/// and bound themselves from DEVICE memory, so a settled city still touches
/// only what moved. All that changes is the LAUNCH DIMENSION, where extra
/// threads return immediately.
///
/// What this buys is measured, not assumed. At baseline the captured graph was
/// being re-instantiated on 45.7% of solves, because graphMatches has an upper
/// bound as well as a lower one -- the cap must also stay within 4x the ideal
/// size, so a scene whose active set keeps shrinking recaptures forever. A
/// fixed full-width cap satisfies both bounds permanently, and once the caps
/// are fixed the counts are not needed on the host at all, which removes the
/// mid-enqueue cudaStreamSynchronize (0.383 ms/solve) as a side effect.
///
/// The first attempt at this forced full width but LEFT the upper bound in
/// place, so every tick failed graphMatches and recaptured. It measured as a
/// regression and was very nearly recorded as a dead end.
///
/// MEASURED, corrected version, default OFF: it does remove the sync
/// (0.423 -> 0.000 ms/solve) but costs more than it saves. gpu_host_work
/// 1.581 -> 3.320 ms and stress_solve p50 14.00 -> 20.52 ms, because every
/// kernel now launches at full width on every iteration. Recapture also did
/// NOT fall (48.0% -> 62.7%), which is the useful part of the result: cap
/// sizing was never what drove recapture. With both cap bounds bypassed
/// entirely the rate stayed high, so the driver is the rest of graphMatches --
/// applyTopologyChange nulls m_graphExec on every fracture tick, and
/// WHOLE_RESET_ON_TOPOLOGY flips warmStart false->true->false around it.
///
/// The route to removing the sync WITHOUT paying full width is therefore to
/// make the launch dimension a performance hint rather than a correctness
/// requirement: grid-stride the solve kernels (they already bound from device
/// memory) so a stale, undersized cap is merely slower, never wrong. Then the
/// counts can come from the previous tick and the sync goes away for free.
static bool fullLaunchCapacity()
{
    static const bool enabled = [] {
        // Default OFF: measured a net regression (see the note above).
        const char* raw = std::getenv("BLAST_GPU_FULL_LAUNCH");
        return raw != nullptr && raw[0] == '1';
    }();
    return enabled;
}

/// Patch the instantiated graph in place instead of re-instantiating it.
/// BLAST_GPU_GRAPH_UPDATE=0 restores destroy + cudaGraphInstantiate.
///
/// Measured: the graph was being re-instantiated on 80% of solves at 0.644 ms
/// each -- 0.516 ms/solve of pure host overhead, MORE than the mid-enqueue
/// sync beside it. applyTopologyChange destroyed it whenever grid sizes or the
/// per-island memset width changed, and graphMatches forces it again whenever
/// the baked launch caps drift outside a 4x band.
///
/// None of that is structural. Every device pointer is allocated once in the
/// constructor and never moves, so a fracture tick changes kernel gridDim, a
/// few scalar arguments and one memset width -- exactly what
/// cudaGraphExecUpdate patches. Only three things change the NODE SET:
/// warmStart (it adds/removes the gather+subtract pair), applyDamage and
/// maxIterations; the latter two are constant in practice.
/// Keep the captured graph's NODE SET independent of warmStart, so a
/// warm-start flip is a parameter change the exec can absorb instead of a
/// rebuild. BLAST_GPU_STABLE_GRAPH=0 restores the branch.
static bool stableGraphEnabled()
{
    static const bool enabled = [] {
        const char* raw = std::getenv("BLAST_GPU_STABLE_GRAPH");
        return raw == nullptr || raw[0] != '0';
    }();
    return enabled;
}

static bool graphUpdateEnabled()
{
    static const bool enabled = [] {
        const char* raw = std::getenv("BLAST_GPU_GRAPH_UPDATE");
        return raw == nullptr || raw[0] != '0';
    }();
    return enabled;
}

static bool wholeResetOnTopology()
{
    static const bool enabled = [] {
        const char* raw = std::getenv("BLAST_GPU_WHOLE_RESET_ON_TOPOLOGY");
        return raw != nullptr && raw[0] == '1';
    }();
    return enabled;
}

struct alignas(16) Vec4
{
    float x;
    float y;
    float z;
    float w;
};

struct alignas(16) AngLin
{
    Vec4 angular;
    Vec4 linear;
};

struct Inertia
{
    float angular;
    float linear;
};

struct SolveStatus
{
    std::uint32_t active;
    std::uint32_t iterations;
    std::uint32_t converged;
};

/// Bonds/nodes that belong to no solvable island (static-static bonds, and
/// static nodes, which are fixed boundaries carrying no coupling).
static constexpr std::uint32_t kNoIsland = 0xFFFFFFFFu;

/// Padded per-island reduction accumulators, as a power-of-two shift. The
/// stride is fixed at compile time so the indexing is a shift rather than a
/// multiply, and so the scratch allocation never has to be resized; the number
/// of slots ACTUALLY summed is chosen per call and is <= 1 << this.
static constexpr std::uint32_t kReductionSlotShiftMax = 5;   // 32 slots
static constexpr std::uint32_t kReductionSlotsMax = 1u << kReductionSlotShiftMax;

/// How often node-space CGLS recomputes q = L pi explicitly rather than
/// advancing it by recurrence. See nodeSpaceMatvec's refresh mode.
///
/// Measured at 8 it cost ~6% of the solve and recovered only ~2% of the
/// residual gap against the CPU reference, so the recurrence is NOT the main
/// source of that gap -- worth knowing before spending more on it. Kept at 16
/// as cheap insurance for very long solves, where drift does compound.
static constexpr std::uint32_t kQRefresh = 16u;

/// A node->bond CSR entry whose bond is gone. The CSR is patched in place on a
/// removal rather than rebuilt, so a node's run keeps its length and the
/// vacated slots are tombstoned. 0xFFFFFFFF cannot collide with a live ref
/// (bond index is masked to 31 bits) and is one compare to test.
static constexpr std::uint32_t kDeadBondRef = 0xFFFFFFFFu;

/**
 * Settled-island skipping: is this BOND's work already done?
 *
 * `skip` is null whenever ExtStressGpuSolveParams::skipSettledIslands is off,
 * which restores the pre-skip behaviour exactly rather than approximately.
 *
 * A kNoIsland bond joins two static nodes. It carries no coupling, the
 * conjugate-gradient update never writes it, and it starts at zero from the
 * construction-time memset -- so its impulse is zero forever and every pass
 * over it is waste. Treating it as permanently settled is what lets the
 * readback list be exactly "the bonds whose impulses can have changed".
 */
__device__ inline bool bondSettled(const std::uint32_t* skip, std::uint32_t island)
{
    return skip != nullptr && (island == kNoIsland || skip[island] != 0u);
}

/**
 * ...and is this NODE's?
 *
 * The opposite rule for kNoIsland: a static node is a boundary SHARED by every
 * island that touches it, so its right-hand side must be rebuilt whichever
 * islands are being solved. (It is multiplied by a zero inverse inertia
 * downstream, so it contributes nothing -- but leaving a stale value there
 * would make a non-finite input outlive the frame that produced it.)
 */
__device__ inline bool nodeSettled(const std::uint32_t* skip, std::uint32_t island)
{
    return skip != nullptr && island != kNoIsland && skip[island] != 0u;
}

void checkCuda(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void checkDriver(CUresult result, const char* operation)
{
    if (result != CUDA_SUCCESS)
    {
        const char* message = nullptr;
        cuGetErrorString(result, &message);
        throw std::runtime_error(
            std::string(operation) + ": " + (message ? message : "CUDA driver error"));
    }
}

class ContextGuard
{
public:
    explicit ContextGuard(CUcontext context)
        : m_active(context != nullptr)
    {
        if (m_active)
        {
            checkDriver(cuCtxPushCurrent(context), "activate PhysX CUDA context");
        }
    }

    ~ContextGuard()
    {
        if (m_active)
        {
            CUcontext popped = nullptr;
            cuCtxPopCurrent(&popped);
        }
    }

private:
    bool m_active;
};

__host__ __device__ Vec4 makeVec(float x, float y, float z)
{
    return {x, y, z, 0.0f};
}

__host__ __device__ Vec4 add(const Vec4& a, const Vec4& b)
{
    return makeVec(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ Vec4 sub(const Vec4& a, const Vec4& b)
{
    return makeVec(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ Vec4 mul(const Vec4& value, float scale)
{
    return makeVec(value.x * scale, value.y * scale, value.z * scale);
}

__host__ __device__ Vec4 cross(const Vec4& a, const Vec4& b)
{
    return makeVec(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ void atomicAddVec(Vec4& target, const Vec4& value)
{
    atomicAdd(&target.x, value.x);
    atomicAdd(&target.y, value.y);
    atomicAdd(&target.z, value.z);
}

__global__ void initializeSolve(
    const ExtStressGpuImpulse* input,
    AngLin* rhs,
    AngLin* residual,
    AngLin* impulses,
    const Inertia* inertia,
    const std::uint32_t* nodeIsland,
    const std::uint32_t* bondIsland,
    const std::uint32_t* islandSkip,
    const std::uint32_t* activeNodes,
    const std::uint32_t* activeBonds,
    const std::uint32_t* activeCounts,
    float reciprocalLengthScale,
    float reciprocalLinearImpulseScale,
    float reciprocalAngularImpulseScale,
    bool warmStart)
{
    // Threads walk the compacted active lists, not the raw arrays: a settled
    // city launches what moved, not what exists. The settled guards below are
    // kept even though the lists pre-filter -- they are what make this kernel
    // bit-identical to the full launch, list or no list.
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < activeCounts[1])
    {
        const std::uint32_t index = activeNodes[slot];
        if (!nodeSettled(islandSkip, nodeIsland[index]))
        {
        const Inertia value = inertia[index];
        const ExtStressGpuImpulse velocity = input[index];
        AngLin b{};
        // Reciprocate ONCE. This was six float divisions per node, and the
        // compiler cannot hoist them because the denominators are data --
        // even though under equalizeMasses they are only ever 0 or 1.
        const float invAngular = value.angular > 0.0f ? 1.0f / value.angular : 1.0f;
        const float invLinear =
            value.linear > 0.0f ? reciprocalLengthScale / value.linear : reciprocalLengthScale;
        b.angular = makeVec(
            -velocity.angular.x * invAngular,
            -velocity.angular.y * invAngular,
            -velocity.angular.z * invAngular);
        b.linear = makeVec(
            -velocity.linear.x * invLinear,
            -velocity.linear.y * invLinear,
            -velocity.linear.z * invLinear);
        rhs[index] = b;
        residual[index] = b;
        }
    }
    // A settled island's impulses are the output the caller is still holding.
    // They must not even be rescaled: impulse*(1/s) followed by impulse*s is
    // not the identity in float, so re-entering the solve at all would walk a
    // frozen island's stress by an ulp per tick for as long as it sits there.
    if (slot < activeCounts[0])
    {
        const std::uint32_t index = activeBonds[slot];
        if (!bondSettled(islandSkip, bondIsland[index]))
        {
        // NOTHING to do on a warm start. Impulses are stored in SOLVER-SCALED
        // units on the device and never round-tripped: this used to multiply
        // every bond by 1/scale here so unscaleImpulses could multiply it back
        // at the end of the solve -- a full read-modify-write over every bond,
        // twice per solve, purely to undo itself. It was 70% of the fixed
        // overhead, which is itself a third of the solve.
        //
        // Keeping them scaled also removes a real numerical wart: x*(1/s)
        // followed by x*s is not the identity in float, so the old round-trip
        // walked a warm-started island's stress by an ulp per tick forever.
        (void)reciprocalAngularImpulseScale;
        (void)reciprocalLinearImpulseScale;
        if (!warmStart)
        {
            impulses[index] = AngLin{};
        }
        }
    }
}

__global__ void couplingRightMultiply(
    AngLin* nodes,
    const AngLin* bonds,
    const std::uint32_t* node0,
    const std::uint32_t* node1,
    const Vec4* offset0,
    const Vec4* offset1,
    const float* health,
    const float* colScale,
    const std::uint32_t* bondIsland,
    const std::uint32_t* islandSkip,
    const std::uint32_t* activeBonds,
    const std::uint32_t* activeCounts)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[0])
    {
        return;
    }
    const std::uint32_t bond = activeBonds[slot];
    if (health[bond] <= 0.0f)
    {
        return;
    }
    // A settled island contributes nothing to any node another island reads:
    // dynamic nodes belong to exactly one island, and the static nodes it does
    // share are zeroed by their inverse inertia. Dropping the atomics here is
    // most of the saving, because they are the expensive part of the solve.
    if (bondSettled(islandSkip, bondIsland[bond]))
    {
        return;
    }

    // Column scale: the operator is C*S, one multiply per bond.
    AngLin impulse = bonds[bond];
    {
        const float s_j = colScale[bond];
        impulse.angular = mul(impulse.angular, s_j);
        impulse.linear = mul(impulse.linear, s_j);
    }
    const Vec4 node0Angular = sub(impulse.angular, cross(offset0[bond], impulse.linear));
    const Vec4 node1Angular = sub(cross(offset1[bond], impulse.linear), impulse.angular);
    atomicAddVec(nodes[node0[bond]].angular, node0Angular);
    atomicAddVec(nodes[node0[bond]].linear, impulse.linear);
    atomicAddVec(nodes[node1[bond]].angular, node1Angular);
    atomicAddVec(nodes[node1[bond]].linear, mul(impulse.linear, -1.0f));
}

/// The transpose of couplingRightMultiply: one thread per ACTIVE NODE, summing
/// its own incident bonds instead of every bond racing to add into its nodes.
///
/// The CSR is node -> refs, each ref packing (bondIndex, endpoint) so the
/// thread knows which of the two sign conventions to apply. The two branches
/// below are exactly couplingRightMultiply's two halves, unchanged; what is
/// gone is the atomicAdd and the memset that had to precede it, because this
/// kernel writes every active node's slot unconditionally.
///
/// The scale by inverse inertia is folded in here, which also retires the
/// separate scaleNodes launch on this path.
__global__ void gatherRightMultiply(
    AngLin* nodes,
    const AngLin* bonds,
    const std::uint32_t* nodeBondBegin,
    const std::uint32_t* nodeBondRef,
    const Vec4* offset0,
    const Vec4* offset1,
    const float* health,
    const float* colScale,
    const std::uint32_t* bondIsland,
    const std::uint32_t* islandSkip,
    const Inertia* inertia,
    const std::uint32_t* activeNodes,
    const std::uint32_t* activeCounts)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[1])
    {
        return;
    }
    const std::uint32_t node = activeNodes[slot];

    Vec4 angular{0.0f, 0.0f, 0.0f, 0.0f};
    Vec4 linear{0.0f, 0.0f, 0.0f, 0.0f};

    // A static node's sum is multiplied by a zero inverse inertia below, so the
    // whole accumulation is discarded -- and static nodes are exactly the
    // high-degree ones. The city's terrain is a single node shared by every
    // building, so ONE thread was walking thousands of scattered bonds while
    // its warpmates sat idle, to produce a value that is defined to be zero.
    // Measured: dropping this loop took the kernel from 200 us to 4 us.
    const Inertia inv = inertia[node];
    if (inv.angular == 0.0f && inv.linear == 0.0f)
    {
        nodes[node].angular = angular;
        nodes[node].linear = linear;
        return;
    }

    const std::uint32_t begin = nodeBondBegin[node];
    const std::uint32_t end = nodeBondBegin[node + 1];
    for (std::uint32_t i = begin; i < end; ++i)
    {
        const std::uint32_t ref = nodeBondRef[i];
        if (ref == kDeadBondRef)
        {
            continue;   // tombstone from an in-place removal
        }
        const std::uint32_t bond = ref & 0x7FFFFFFFu;
        const bool isSecond = (ref & 0x80000000u) != 0u;
        if (health[bond] <= 0.0f)
        {
            continue;
        }
        if (bondSettled(islandSkip, bondIsland[bond]))
        {
            continue;
        }
        AngLin impulse = bonds[bond];
        {
            const float s_j = colScale[bond];
            impulse.angular = mul(impulse.angular, s_j);
            impulse.linear = mul(impulse.linear, s_j);
        }
        if (!isSecond)
        {
            angular = add(angular, sub(impulse.angular, cross(offset0[bond], impulse.linear)));
            linear = add(linear, impulse.linear);
        }
        else
        {
            angular = add(angular, sub(cross(offset1[bond], impulse.linear), impulse.angular));
            linear = add(linear, mul(impulse.linear, -1.0f));
        }
    }

    // Exclusive write, and the inverse-inertia scale folded in: a static node
    // has zero inertia, so its slot lands at zero exactly as the memset +
    // scaleNodes pair produced before.
    nodes[node].angular = mul(angular, inertia[node].angular);
    nodes[node].linear = mul(linear, inertia[node].linear);
}

__global__ void scaleNodes(
    AngLin* nodes,
    const Inertia* inertia,
    const std::uint32_t* activeNodes,
    const std::uint32_t* activeCounts)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < activeCounts[1])
    {
        const std::uint32_t node = activeNodes[slot];
        nodes[node].angular = mul(nodes[node].angular, inertia[node].angular);
        nodes[node].linear = mul(nodes[node].linear, inertia[node].linear);
    }
}

__global__ void subtractResidual(
    AngLin* residual,
    const AngLin* value,
    const std::uint32_t* activeNodes,
    const std::uint32_t* activeCounts)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < activeCounts[1])
    {
        const std::uint32_t index = activeNodes[slot];
        residual[index].angular =
            sub(residual[index].angular, value[index].angular);
        residual[index].linear =
            sub(residual[index].linear, value[index].linear);
    }
}

__global__ void couplingLeftMultiply(
    AngLin* bonds,
    const AngLin* nodes,
    const Inertia* inertia,
    const std::uint32_t* node0,
    const std::uint32_t* node1,
    const Vec4* offset0,
    const Vec4* offset1,
    const float* health,
    const float* colScale,
    const std::uint32_t* bondIsland,
    const std::uint32_t* islandSkip,
    const std::uint32_t* activeBonds,
    const std::uint32_t* activeCounts)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[0])
    {
        return;
    }
    const std::uint32_t bond = activeBonds[slot];
    if (bondSettled(islandSkip, bondIsland[bond]))
    {
        return;     // gradient for a settled island is never read
    }
    if (health[bond] <= 0.0f)
    {
        bonds[bond] = {};
        return;
    }

    const std::uint32_t first = node0[bond];
    const std::uint32_t second = node1[bond];
    AngLin x0 = nodes[first];
    AngLin x1 = nodes[second];
    x0.angular = mul(x0.angular, inertia[first].angular);
    x0.linear = mul(x0.linear, inertia[first].linear);
    x1.angular = mul(x1.angular, inertia[second].angular);
    x1.linear = mul(x1.linear, inertia[second].linear);

    AngLin result;
    result.angular = sub(x0.angular, x1.angular);
    result.linear = add(
        sub(x0.linear, x1.linear),
        sub(cross(offset0[bond], x0.angular), cross(offset1[bond], x1.angular)));
    // Transpose of the column scale: y = S * C^T * (...).
    {
        const float s_j = colScale[bond];
        result.angular = mul(result.angular, s_j);
        result.linear = mul(result.linear, s_j);
    }
    bonds[bond] = result;
}

__global__ void squaredMagnitude(
    const AngLin* values,
    float* output,
    std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        const AngLin value = values[index];
        output[index] =
            value.angular.x * value.angular.x
            + value.angular.y * value.angular.y
            + value.angular.z * value.angular.z
            + value.linear.x * value.linear.x
            + value.linear.y * value.linear.y
            + value.linear.z * value.linear.z;
    }
}

__global__ void setTolerance(float* deltaSquared, const float* rhsSquared, float tolerance)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *deltaSquared = tolerance * tolerance * *rhsSquared;
    }
}

/// Seed each island's squared residual target and mark it active.
__global__ void setTolerancePerIsland(
    float* deltaSquared,
    std::uint32_t* islandActive,
    std::uint32_t* islandConverged,
    const std::uint32_t* islandSkip,
    const float* rhsSquared,
    float tolerance,
    std::uint32_t islandCount)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= islandCount)
    {
        return;
    }
    if (islandSkip != nullptr && islandSkip[id] != 0u)
    {
        // Inactive for every kernel below, and its convergence flag is left
        // alone: that flag IS the baseline saying this island may be skipped
        // again next frame, and clearing it here would make a settled island
        // alternate solve/skip forever.
        islandActive[id] = 0;
        return;
    }
    deltaSquared[id] = rhsSquared[id] * tolerance * tolerance;
    islandActive[id] = 1;
    islandConverged[id] = 0;
}

__global__ void initializeStatus(
    SolveStatus* status, std::uint32_t* iteration, std::uint32_t maxIterations)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        status->active = 1;
        status->iterations = maxIterations;
        status->converged = 0;
        // The loop counter lives on the device so the CG body can be ONE graph
        // node executed repeatedly, instead of maxIterations copies of it.
        *iteration = 0u;
    }
}



/// ---------------------------------------------------------------------------
/// NODE-SPACE CGLS
///
/// The bond-space form solves  C^T D C lambda = -C^T D v  by CGNR on the factor
/// B = D C, carrying three BOND-length vectors (gradient, direction, impulses)
/// through every iteration. Substituting
///
///     lambda = lambda0 + W mu,      W := B^T = C^T D,   L := W^T W = D C C^T D
///
/// eliminates the bond-space vectors z, p and s ALGEBRAICALLY -- this is an
/// identity, not an approximation, and every scalar (z_sq, beta, alpha) is
/// unchanged. `r` in cgnr.h is already a node vector, so rho == r literally.
///
///     w  = L rho      and, from the same pass, z_sq = ||W rho||^2 = sum ||t_j||^2
///     beta = z_sq / z_sq_prev
///     pi = rho + beta pi ;  q = w + beta q          (q == L pi, by linearity)
///     alpha = z_sq / ||q||^2
///     mu += alpha pi ;      rho -= alpha q
///     ... and once, at the end, lambda = lambda0 + W mu
///
/// Why bother: the hot loop drops from ~112 B/bond to ~48 B/bond, because
/// `gradient` and `direction` cease to exist. At 671k bonds the old working set
/// no longer fits the 4090's 72 MB L2 and per-bond cost rises 72%; this brings
/// it back inside.
///
/// THREE TRAPS, all of which produce plausible-looking wrong answers:
///
/// 1. THIS IS NOT PLAIN CG ON L. Plain CG uses numerator rho^T rho; the correct
///    one is rho^T L rho = ||W rho||^2. null(L) is the rigid-body mode of any
///    island not anchored to a static node -- the stress suite measures 29,041
///    of those in one scene. With rho^T rho both alpha and beta inflate and the
///    residual test stalls forever on every free-floating fragment.
/// 2. z_sq MUST COUNT EACH BOND EXACTLY ONCE. Canonical owner: the node-0 side,
///    or the dynamic endpoint when node 0 is static. Double counting scales
///    alpha, beta and delta consistently, so it still converges -- just slower.
///    An endpoint test will not catch it.
/// 3. q = w + beta q IS A RECURRENCE where s = Bp was a fresh application. That
///    is the pipelined-CG trade; refresh q = L pi explicitly every kQRefresh
///    iterations.
/// ---------------------------------------------------------------------------

/// w = L rho, fused: the bond-space intermediate t_j is recomputed at each
/// endpoint rather than stored. The workload is ~2 flops/byte on a machine that
/// does 80, so recomputing is free and it removes a whole bond-length vector
/// from the loop.
__global__ void nodeSpaceMatvec(
    AngLin* w,
    const AngLin* rho,
    const Inertia* inertia,
    const std::uint32_t* nodeBondBegin,
    const std::uint32_t* nodeBondRef,
    const std::uint32_t* node0,
    const std::uint32_t* node1,
    const Vec4* offset0,
    const Vec4* offset1,
    const float* health,
    const float* colScale,
    const std::uint32_t* bondIsland,
    const std::uint32_t* islandSkip,
    const std::uint32_t* nodeIsland,
    const std::uint32_t* islandActive,
    bool skipConverged,
    float* zSqSlots,
    std::uint32_t slotCount,
    const std::uint32_t* activeNodes,
    const std::uint32_t* activeCounts,
    // Refresh mode: when non-zero, this launch only does work on iterations
    // divisible by `refreshEvery`, and skips the z_sq accumulation. That is the
    // periodic explicit recomputation of q = L pi which keeps the recurrence
    // q = w + beta q from drifting (the pipelined-CG trade).
    const std::uint32_t* iterationPtr,
    std::uint32_t refreshEvery)
{
    if (refreshEvery != 0u && (*iterationPtr % refreshEvery) != 0u)
    {
        return;
    }
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[1])
    {
        return;
    }
    const std::uint32_t node = activeNodes[slot];
    // An island that reached tolerance EARLIER IN THIS SOLVE is done: its pi,
    // q, mu and rho are frozen by the islandActive guard in the update kernels,
    // so the w and z_sq computed for it here would be written and never read.
    // The cross-tick islandSkip mask does not cover this -- that one only
    // retires islands that were already settled when the solve began. Islands
    // converge at very different rates once a scene is fragmented, so at
    // realistic island counts this is a large fraction of the matvec.
    // Read the island once and reuse it for the z_sq atomic below: the guard
    // then costs a single extra scattered load, not two.
    const std::uint32_t myIsland = nodeIsland[node];
    if (skipConverged && myIsland != kNoIsland && islandActive != nullptr
        && !islandActive[myIsland])
    {
        return;
    }
    const Inertia inv = inertia[node];
    if (inv.angular == 0.0f && inv.linear == 0.0f)
    {
        // Static: annihilated on both sides of L, so its row is identically
        // zero. Also the high-degree terrain node, which would otherwise walk
        // thousands of bonds to produce zero.
        w[node].angular = Vec4{0.0f, 0.0f, 0.0f, 0.0f};
        w[node].linear = Vec4{0.0f, 0.0f, 0.0f, 0.0f};
        return;
    }

    const AngLin selfRho = rho[node];
    const Vec4 selfAng = mul(selfRho.angular, inv.angular);
    const Vec4 selfLin = mul(selfRho.linear, inv.linear);

    Vec4 accAng{0.0f, 0.0f, 0.0f, 0.0f};
    Vec4 accLin{0.0f, 0.0f, 0.0f, 0.0f};
    float zSq = 0.0f;

    const std::uint32_t begin = nodeBondBegin[node];
    const std::uint32_t end = nodeBondBegin[node + 1];
    for (std::uint32_t i = begin; i < end; ++i)
    {
        const std::uint32_t ref = nodeBondRef[i];
        if (ref == kDeadBondRef)
        {
            continue;   // tombstone from an in-place removal
        }
        const std::uint32_t bond = ref & 0x7FFFFFFFu;
        const bool isSecond = (ref & 0x80000000u) != 0u;
        if (health[bond] <= 0.0f || bondSettled(islandSkip, bondIsland[bond]))
        {
            continue;
        }
        const std::uint32_t other = isSecond ? node0[bond] : node1[bond];
        const Inertia otherInv = inertia[other];
        const AngLin otherRho = rho[other];
        const Vec4 otherAng = mul(otherRho.angular, otherInv.angular);
        const Vec4 otherLin = mul(otherRho.linear, otherInv.linear);

        // t_j = (C^T D rho)_j, with node 0 first regardless of which side we
        // are on -- the sign convention is a property of the bond, not of the
        // walker.
        const Vec4 a0 = isSecond ? otherAng : selfAng;
        const Vec4 l0 = isSecond ? otherLin : selfLin;
        const Vec4 a1 = isSecond ? selfAng : otherAng;
        const Vec4 l1 = isSecond ? selfLin : otherLin;
        const Vec4 o0 = offset0[bond];
        const Vec4 o1 = offset1[bond];

        // Column scaling: the operator is L_S = D C S^2 C^T D, where S is the
        // per-bond compliance weight (Young's modulus) the CPU processor
        // solves with. The bond-space intermediate is y_j = s_j t_j, the
        // numerator is ||y||^2, and the node accumulation applies S once more
        // on the way back -- so the whole bond term carries s_j^2.
        const float s_j = colScale[bond];
        const float s2 = s_j * s_j;
        const Vec4 tAng = mul(sub(a0, a1), s2);
        const Vec4 tLin =
            mul(add(sub(l0, l1), sub(cross(o0, a0), cross(o1, a1))), s2);

        // Own the bond from the node-0 side, or from the dynamic side when
        // node 0 is static (that side never runs). Exactly once, either way.
        const bool node0Dynamic =
            !(inertia[node0[bond]].angular == 0.0f && inertia[node0[bond]].linear == 0.0f);
        const bool owns = isSecond ? !node0Dynamic : true;
        if (owns)
        {
            // ||s_j t_j||^2 == |s_j^2 t_j|^2 / s_j^2
            zSq += (tAng.x * tAng.x + tAng.y * tAng.y + tAng.z * tAng.z
                 + tLin.x * tLin.x + tLin.y * tLin.y + tLin.z * tLin.z) / s2;
        }

        // (D C S^2 t)_node, the same accumulation gatherRightMultiply performs.
        if (!isSecond)
        {
            accAng = add(accAng, sub(tAng, cross(o0, tLin)));
            accLin = add(accLin, tLin);
        }
        else
        {
            accAng = add(accAng, sub(cross(o1, tLin), tAng));
            accLin = sub(accLin, tLin);
        }
    }

    w[node].angular = mul(accAng, inv.angular);
    w[node].linear = mul(accLin, inv.linear);

    if (zSqSlots != nullptr && zSq > 0.0f && myIsland != kNoIsland)
    {
        atomicAdd(&zSqSlots[myIsland * slotCount + (slot & (slotCount - 1u))], zSq);
    }
}

/// Assemble and invert the 6x6 diagonal block of L per node: block-Jacobi.
///
/// L = D C C^T D, so the diagonal block at node i is D_i (sum over incident
/// bonds of A A^T) D_i, where A = [[I, -[r]x],[0, I]] is that bond's coupling
/// block. A A^T works out to
///
///     A A^T = [[ I - [r]x [r]x ,  -[r]x ],
///              [   +[r]x        ,    I   ]]
///
/// and since [r]x [r]x = r r^T - |r|^2 I, the top-left is I - r r^T + |r|^2 I.
/// Note BOTH the sign of that product and the ASYMMETRY of the off-diagonal
/// blocks (-[r]x above, +[r]x below, because [r]x^T = -[r]x). Getting either
/// wrong makes the block non-SPD and the preconditioned iteration diverges
/// outright -- measured, first attempt: residual 9e+04 against 2.6e-02.
///
/// Correctness safety: if the assembled block will not factor, this stores the
/// IDENTITY for that node. A preconditioner that degenerates to no
/// preconditioner is still a valid preconditioner -- it costs iterations, never
/// the answer -- so no input can make this produce a wrong solve.
__global__ void nodeSpaceBuildJacobi(
    float* inverse,                  // 36 floats per node, row major
    const Inertia* inertia,
    const std::uint32_t* nodeBondBegin,
    const std::uint32_t* nodeBondRef,
    const std::uint32_t* node0,
    const std::uint32_t* node1,
    const Vec4* offset0,
    const Vec4* offset1,
    const float* health,
    const float* colScale,
    std::uint32_t nodeCount,
    std::uint32_t bondCount)
{
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodeCount)
    {
        return;
    }
    float m[36];
    for (int i = 0; i < 36; ++i) m[i] = 0.0f;

    const Inertia inv = inertia[node];
    const bool dynamic = !(inv.angular == 0.0f && inv.linear == 0.0f);
    if (dynamic)
    {
        for (std::uint32_t i = nodeBondBegin[node]; i < nodeBondBegin[node + 1]; ++i)
        {
            const std::uint32_t ref = nodeBondRef[i];
            if (ref == kDeadBondRef) continue;
            const std::uint32_t bond = ref & 0x7FFFFFFFu;
            if (bond >= bondCount || health[bond] <= 0.0f) continue;
            const Vec4 r = (ref & 0x80000000u) ? offset1[bond] : offset0[bond];
            const float rx = r.x, ry = r.y, rz = r.z;
            const float r2 = rx * rx + ry * ry + rz * rz;
            // Column scale enters the diagonal block as s_j^2 (L_S = D C S^2 C^T D).
            const float w = colScale[bond] * colScale[bond];
            // top-left: I - r r^T + |r|^2 I
            m[0*6+0] += w * (1.0f - rx*rx + r2);  m[0*6+1] += w * (-rx*ry);  m[0*6+2] += w * (-rx*rz);
            m[1*6+0] += w * (-ry*rx);  m[1*6+1] += w * (1.0f - ry*ry + r2);  m[1*6+2] += w * (-ry*rz);
            m[2*6+0] += w * (-rz*rx);  m[2*6+1] += w * (-rz*ry);  m[2*6+2] += w * (1.0f - rz*rz + r2);
            // off-diagonal: -[r]x above the diagonal, +[r]x below.
            const float sx[9] = {0.0f, -rz, ry,  rz, 0.0f, -rx,  -ry, rx, 0.0f};
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                {
                    m[(a)*6 + (3+b)] += -w * sx[a*3+b];
                    m[(3+a)*6 + (b)] += +w * sx[a*3+b];
                }
            // bottom-right: I
            m[3*6+3] += w;  m[4*6+4] += w;  m[5*6+5] += w;
        }
        // Apply the 0/1 inertia mask on both sides, as L does.
        for (int a = 0; a < 6; ++a)
            for (int b = 0; b < 6; ++b)
            {
                const float da = (a < 3) ? inv.angular : inv.linear;
                const float db = (b < 3) ? inv.angular : inv.linear;
                m[a*6+b] *= da * db;
            }
    }

    // Gauss-Jordan with partial pivoting into `out`, starting from identity.
    float out[36];
    for (int i = 0; i < 36; ++i) out[i] = 0.0f;
    for (int i = 0; i < 6; ++i) out[i*6+i] = 1.0f;

    bool ok = dynamic;
    if (ok)
    {
        for (int col = 0; col < 6 && ok; ++col)
        {
            int piv = col;
            float best = fabsf(m[col*6+col]);
            for (int r2i = col + 1; r2i < 6; ++r2i)
            {
                const float v = fabsf(m[r2i*6+col]);
                if (v > best) { best = v; piv = r2i; }
            }
            if (!(best > 1e-12f)) { ok = false; break; }
            if (piv != col)
                for (int c = 0; c < 6; ++c)
                {
                    float t = m[col*6+c]; m[col*6+c] = m[piv*6+c]; m[piv*6+c] = t;
                    t = out[col*6+c]; out[col*6+c] = out[piv*6+c]; out[piv*6+c] = t;
                }
            const float inv0 = 1.0f / m[col*6+col];
            for (int c = 0; c < 6; ++c) { m[col*6+c] *= inv0; out[col*6+c] *= inv0; }
            for (int r2i = 0; r2i < 6; ++r2i)
            {
                if (r2i == col) continue;
                const float f = m[r2i*6+col];
                if (f == 0.0f) continue;
                for (int c = 0; c < 6; ++c)
                { m[r2i*6+c] -= f * m[col*6+c]; out[r2i*6+c] -= f * out[col*6+c]; }
            }
        }
        for (int i = 0; i < 36 && ok; ++i) if (!isfinite(out[i])) ok = false;
    }
    if (!ok)
    {
        for (int i = 0; i < 36; ++i) out[i] = 0.0f;
        for (int i = 0; i < 6; ++i) out[i*6+i] = 1.0f;   // identity fallback
    }
    for (int i = 0; i < 36; ++i) inverse[node * 36 + i] = out[i];
}

/// g = N w, and gamma = w^T g accumulated per island.
__global__ void nodeSpaceApplyJacobi(
    AngLin* g,
    const AngLin* w,
    const float* inverse,
    const std::uint32_t* nodeIsland,
    const std::uint32_t* islandActive,
    float* gammaSlots,
    std::uint32_t slotCount,
    const std::uint32_t* activeNodes,
    const std::uint32_t* activeCounts)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[1]) return;
    const std::uint32_t node = activeNodes[slot];
    const std::uint32_t island = nodeIsland[node];
    if (island == kNoIsland || !islandActive[island]) return;

    const float wv[6] = {w[node].angular.x, w[node].angular.y, w[node].angular.z,
                         w[node].linear.x, w[node].linear.y, w[node].linear.z};
    const float* M = inverse + node * 36;
    // N MUST BE APPLIED TWICE.
    //
    // CGLS wants P ~ S^+ in bond space. With P = W N B and B(W rho) = L rho,
    // P s = W N L rho, so matching S^+ needs N L = L^+, i.e. N ~ L^-2 -- NOT
    // L^-1. Applying an L^-1 approximation once yields P ~ I, which is the
    // UNPRECONDITIONED operator back again, except perturbed by the
    // approximation error -- a noisy identity, strictly worse than identity.
    // Measured with a single apply: residual 1.52e+00 against 2.59e-02
    // unpreconditioned, stable across three unrelated bug fixes, which is what
    // finally identified this as structural rather than a memory bug.
    float half[6];
    for (int a = 0; a < 6; ++a)
    {
        float acc = 0.0f;
        for (int b = 0; b < 6; ++b) acc += M[a*6+b] * wv[b];
        half[a] = acc;
    }
    float gv[6];
    for (int a = 0; a < 6; ++a)
    {
        float acc = 0.0f;
        for (int b = 0; b < 6; ++b) acc += M[a*6+b] * half[b];
        gv[a] = acc;
    }
    g[node].angular = Vec4{gv[0], gv[1], gv[2], 0.0f};
    g[node].linear = Vec4{gv[3], gv[4], gv[5], 0.0f};
    float gamma = 0.0f;
    for (int a = 0; a < 6; ++a) gamma += wv[a] * gv[a];
    atomicAdd(&gammaSlots[island * slotCount + (slot & (slotCount - 1u))], gamma);
}

/// pi = rho + beta pi ;  q = w + beta q, and ||q||^2 accumulated in the same
/// pass.
///
/// This kernel already has the new q in registers, so reducing it here removes
/// a whole separate pass that re-read q from memory (32 B/node) plus its kernel
/// launch, every iteration.
__global__ void nodeSpaceUpdateDirection(
    AngLin* pi,
    AngLin* q,
    const AngLin* rho,
    const AngLin* w,
    const float* zSq,
    const float* zSqPrev,
    const std::uint32_t* nodeIsland,
    const std::uint32_t* islandActive,
    float* qSqSlots,
    std::uint32_t slotCount,
    const std::uint32_t* activeNodes,
    const std::uint32_t* activeCounts)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[1])
    {
        return;
    }
    const std::uint32_t node = activeNodes[slot];
    const std::uint32_t island = nodeIsland[node];
    if (island == kNoIsland || !islandActive[island])
    {
        return;
    }
    const float denominator = zSqPrev[island];
    const float beta = denominator > 0.0f ? zSq[island] / denominator : 0.0f;
    // `rho` is the preconditioned direction source: it is rho itself when
    // unpreconditioned, and g = N w when preconditioned.
    pi[node].angular = add(rho[node].angular, mul(pi[node].angular, beta));
    pi[node].linear = add(rho[node].linear, mul(pi[node].linear, beta));
    const Vec4 qa = add(w[node].angular, mul(q[node].angular, beta));
    const Vec4 ql = add(w[node].linear, mul(q[node].linear, beta));
    q[node].angular = qa;
    q[node].linear = ql;
    const float qSq = qa.x * qa.x + qa.y * qa.y + qa.z * qa.z
                    + ql.x * ql.x + ql.y * ql.y + ql.z * ql.z;
    atomicAdd(&qSqSlots[island * slotCount + (slot & (slotCount - 1u))], qSq);
}

/// mu += alpha pi ;  rho -= alpha q
__global__ void nodeSpaceUpdateSolution(
    const std::uint32_t* iterationPtr,
    std::uint32_t maxIterations,
    AngLin* mu,
    AngLin* rho,
    const AngLin* pi,
    const AngLin* q,
    const float* zSq,
    const float* qSq,
    const std::uint32_t* nodeIsland,
    const std::uint32_t* islandActive,
    const std::uint32_t* activeNodes,
    const std::uint32_t* activeCounts)
{
    if (*iterationPtr > maxIterations)
    {
        return;   // chunked-loop overshoot guard; see the bond-space twin
    }
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[1])
    {
        return;
    }
    const std::uint32_t node = activeNodes[slot];
    const std::uint32_t island = nodeIsland[node];
    if (island == kNoIsland || !islandActive[island])
    {
        return;
    }
    const float denominator = qSq[island];
    if (!(denominator > 0.0f) || !isfinite(denominator))
    {
        return;
    }
    const float alpha = zSq[island] / denominator;
    mu[node].angular = add(mu[node].angular, mul(pi[node].angular, alpha));
    mu[node].linear = add(mu[node].linear, mul(pi[node].linear, alpha));
    rho[node].angular = sub(rho[node].angular, mul(q[node].angular, alpha));
    rho[node].linear = sub(rho[node].linear, mul(q[node].linear, alpha));
}

/// lambda += W mu, i.e. one C^T D pass, run once at the end of the solve.
__global__ void nodeSpaceApplySolution(
    AngLin* impulses,
    const AngLin* mu,
    const Inertia* inertia,
    const std::uint32_t* node0,
    const std::uint32_t* node1,
    const Vec4* offset0,
    const Vec4* offset1,
    const float* health,
    const float* colScale,
    const std::uint32_t* bondIsland,
    const std::uint32_t* islandSkip,
    const std::uint32_t* activeBonds,
    const std::uint32_t* activeCounts)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[0])
    {
        return;
    }
    const std::uint32_t bond = activeBonds[slot];
    if (bondSettled(islandSkip, bondIsland[bond]))
    {
        return;
    }
    if (health[bond] <= 0.0f)
    {
        impulses[bond] = AngLin{};
        return;
    }
    const std::uint32_t first = node0[bond];
    const std::uint32_t second = node1[bond];
    AngLin x0 = mu[first];
    AngLin x1 = mu[second];
    x0.angular = mul(x0.angular, inertia[first].angular);
    x0.linear = mul(x0.linear, inertia[first].linear);
    x1.angular = mul(x1.angular, inertia[second].angular);
    x1.linear = mul(x1.linear, inertia[second].linear);

    // J' = J'0 + S C^T D mu: one column-scale multiply per bond.
    const float s_j = colScale[bond];
    const Vec4 tAng = mul(sub(x0.angular, x1.angular), s_j);
    const Vec4 tLin = mul(add(
        sub(x0.linear, x1.linear),
        sub(cross(offset0[bond], x0.angular), cross(offset1[bond], x1.angular))), s_j);
    impulses[bond].angular = add(impulses[bond].angular, tAng);
    impulses[bond].linear = add(impulses[bond].linear, tLin);
}

/// Zero the node-space accumulators that must start each solve at zero.
__global__ void nodeSpaceReset(
    AngLin* mu,
    AngLin* pi,
    AngLin* q,
    AngLin* g,
    std::uint32_t nodeCount)
{
    // Over ALL nodes, not the active list. The matvec reads its neighbour's
    // value at every half-edge including static ones, where it is multiplied by
    // a zero inertia -- and 0 * NaN is NaN, so an uninitialised slot that the
    // active list never covers still poisons the result. Measured: it turned a
    // 2.5e-02 residual into 1.5e+00.
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodeCount)
    {
        return;
    }
    mu[node] = AngLin{};
    pi[node] = AngLin{};
    q[node] = AngLin{};
    g[node] = AngLin{};
}

/// Apply patched node->bond CSR entries. Four slots per removal instead of
/// re-uploading the whole 2*bondCount reference array.
__global__ void scatterNodeBondRefs(
    const std::uint32_t* slots,
    const std::uint32_t* values,
    std::uint32_t count,
    std::uint32_t* nodeBondRef)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        nodeBondRef[slots[i]] = values[i];
    }
}

/// One bond's full topology record, for the sparse upload path.
struct alignas(16) BondDelta
{
    Vec4 offset0;
    Vec4 offset1;
    Vec4 normal;
    std::uint32_t node0;
    std::uint32_t node1;
    std::uint32_t material;
    std::uint32_t island;
    float area;
    float nodeDistance;
    float health;
    float colScale;
};

/// Apply a sparse topology update.
///
/// removeBond is swap-with-last, so a removal rewrites exactly ONE live slot
/// (the removed index, which receives the former last bond); everything past
/// the new bond count is simply never read again. The whole-array re-upload
/// this replaces moved ~16 MB per fracture tick to change a few hundred bonds.
__global__ void scatterBondTopology(
    const std::uint32_t* slots,
    const BondDelta* values,
    std::uint32_t count,
    std::uint32_t* node0,
    std::uint32_t* node1,
    Vec4* offset0,
    Vec4* offset1,
    Vec4* normals,
    float* areas,
    float* nodeDistances,
    float* health,
    float* colScales,
    std::uint32_t* materials,
    std::uint32_t* island)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    const std::uint32_t b = slots[i];
    const BondDelta v = values[i];
    colScales[b] = v.colScale;
    node0[b] = v.node0;
    node1[b] = v.node1;
    offset0[b] = v.offset0;
    offset1[b] = v.offset1;
    normals[b] = v.normal;
    areas[b] = v.area;
    nodeDistances[b] = v.nodeDistance;
    health[b] = v.health;
    materials[b] = v.material;
    island[b] = v.island;
}

/// Per-island sum of squared magnitudes, accumulated into PADDED slots.
///
/// Islands are disconnected components, so their conjugate-gradient scalars
/// must be independent: a shared alpha/beta compromises every island toward
/// the average, and a shared convergence test lets a large well-conditioned
/// island hide a small badly-solved one. Bonds and nodes are not stored
/// island-contiguous, so a segmented library reduce is not available and this
/// has to be an atomic scatter.
///
/// The naive form -- one atomicAdd per element onto ONE accumulator per island
/// -- was measured at 76% of the entire solve (6.88 ms of 9.04 ms, 106 us per
/// launch, twice per CG iteration) on the 298k-bond city. The cause is
/// contention, not bandwidth: 298k atomics onto 108 addresses is 2,760-way
/// serialization per address, while the two matvecs either side of it run at
/// 2-3 TB/s because the working set is L2-resident.
///
/// So give each island `slots` accumulators and hash the element onto one by
/// its thread index. Consecutive lanes in a warp land on consecutive slots, so
/// a warp's 32 atomics go to 32 distinct addresses and never serialize against
/// each other. `finalizeIslandReduction` then sums the slots.
///
/// `slots` is chosen per call from the average island occupancy, because the
/// padding is a pure loss in the regime it is not needed for: a fully
/// fractured city is ~96k islands of ~4 elements, where contention is already
/// nil and a wide second pass would cost more than the atomics it saves.
/// See ExtStressGpuSolverImpl::reductionSlots.
__global__ void accumulateSquaredByIsland(
    const AngLin* values,
    const std::uint32_t* island,
    const std::uint32_t* islandSkip,
    float* perIslandSlots,
    const std::uint32_t* activeList,
    const std::uint32_t* activeCounts,
    std::uint32_t whichCount,
    std::uint32_t slotCount)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[whichCount])
    {
        return;
    }
    const std::uint32_t index = activeList[slot];
    const std::uint32_t id = island[index];
    if (id == kNoIsland)
    {
        return;
    }
    if (islandSkip != nullptr && islandSkip[id] != 0u)
    {
        return;     // settled: its scalars are never consulted this solve
    }
    const AngLin& value = values[index];
    const float squared =
        value.angular.x * value.angular.x + value.angular.y * value.angular.y +
        value.angular.z * value.angular.z + value.linear.x * value.linear.x +
        value.linear.y * value.linear.y + value.linear.z * value.linear.z;
    atomicAdd(&perIslandSlots[id * slotCount + (slot & (slotCount - 1u))], squared);
}

/// Collapse the padded slots to one value per island.
///
/// One thread per island, reading `slots` contiguous floats. At 108 islands
/// and 32 slots that is 13.8 KB -- far below the cost of the contention it
/// removes. Summation order is fixed (ascending slot), so this pass is
/// deterministic; the atomics that produced the slots are not, which is why
/// bit-exact reproducibility has to wait for island-contiguous ordering.
__global__ void finalizeIslandReduction(
    const float* perIslandSlots,
    float* result,
    std::uint32_t islandCount,
    std::uint32_t slots)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= islandCount)
    {
        return;
    }
    const float* base = perIslandSlots + static_cast<std::size_t>(id) * slots;
    float sum = 0.0f;
    for (std::uint32_t i = 0; i < slots; ++i)
    {
        sum += base[i];
    }
    result[id] = sum;
}

/// One thread per island: retire islands that have reached tolerance. An
/// island that stops here stops costing iterations, which is what makes
/// solving to a residual tolerance affordable instead of running a fixed
/// budget for the whole graph.
/// Convergence test AND the active tally, fused.
///
/// They were two launches over the same island grid, back to back, reading
/// the same array. At ~3 us of launch latency per kernel -- which is what
/// every small kernel in this solve costs, measured -- a launch that does one
/// comparison per island is nearly all overhead. Fusing removes one launch
/// per iteration and the separate counter reset with it.
///
/// The tally writes ONE partial per block instead of accumulating into a
/// shared counter: with ~8 blocks at city scale the final sum is trivial for
/// the latch kernel, it needs no reset beforehand, and it is exact and
/// order-independent by construction rather than by argument.
/// Slot-sum + convergence test + block tally, in one launch.
///
/// These were three separate island-grid kernels. At realistic island counts
/// that is almost pure launch overhead: 1,243 islands is five blocks of work,
/// and the three kernels measured 3.5-4.6 us EACH per launch against ~10-25 us
/// for the kernels that do real work. Fusing the pairs that were already
/// adjacent and share a grid shape removes two launches per CG iteration.
__global__ void finalizeAndCheckConvergence(
    const float* perIslandSlots,
    float* result,
    std::uint32_t slots,
    std::uint32_t* islandActive,
    std::uint32_t* islandConverged,
    const float* deltaSquared,
    std::uint32_t* blockActiveCounts,
    std::uint32_t islandCount)
{
    __shared__ std::uint32_t partial[kBlockSize];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t id = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;
    if (id < islandCount)
    {
        const float* base = perIslandSlots + static_cast<std::size_t>(id) * slots;
        for (std::uint32_t i = 0; i < slots; ++i)
        {
            sum += base[i];
        }
        result[id] = sum;
    }

    std::uint32_t active = 0;
    if (id < islandCount && islandActive[id])
    {
        if (sum <= deltaSquared[id])
        {
            islandActive[id] = 0;
            islandConverged[id] = 1;
        }
        else
        {
            active = 1;
        }
    }
    partial[tid] = active;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        blockActiveCounts[blockIdx.x] = partial[0];
    }
}

/// Slot-sum + degenerate retirement + loop control, in one launch.
__global__ void finalizeAndRetire(
    const float* perIslandSlots,
    float* result,
    std::uint32_t slots,
    std::uint32_t* islandActive,
    float* previousNumerator,
    const float* numerator,
    SolveStatus* status,
    const std::uint32_t* blockActiveCounts,
    std::uint32_t blockCount,
    std::uint32_t* iterationPtr,
    std::uint32_t islandCount,
    cudaGraphConditionalHandle loopHandle,
    std::uint32_t maxIterations)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < islandCount)
    {
        const float* base = perIslandSlots + static_cast<std::size_t>(id) * slots;
        float sum = 0.0f;
        for (std::uint32_t i = 0; i < slots; ++i)
        {
            sum += base[i];
        }
        result[id] = sum;
        if (islandActive[id])
        {
            if (!(sum > 0.0f) || !isfinite(sum))
            {
                islandActive[id] = 0;
            }
            previousNumerator[id] = numerator[id];
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const std::uint32_t iteration = *iterationPtr;
        std::uint32_t active = 0;
        for (std::uint32_t i = 0; i < blockCount; ++i)
        {
            active += blockActiveCounts[i];
        }
        status->active = active;
        if (active == 0 && !status->converged)
        {
            status->converged = 1;
            status->iterations = iteration;
        }
        const std::uint32_t next = iteration + 1u;
        *iterationPtr = next;
        if (loopHandle != 0)
        {
            cudaGraphSetConditional(
                loopHandle, (active != 0u && next < maxIterations) ? 1u : 0u);
        }
    }
}

__global__ void checkConvergencePerIsland(
    std::uint32_t* islandActive,
    std::uint32_t* islandConverged,
    const float* residualSquared,
    const float* deltaSquared,
    std::uint32_t* blockActiveCounts,
    std::uint32_t islandCount)
{
    __shared__ std::uint32_t partial[kBlockSize];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t id = blockIdx.x * blockDim.x + tid;

    std::uint32_t active = 0;
    if (id < islandCount && islandActive[id])
    {
        if (residualSquared[id] <= deltaSquared[id])
        {
            islandActive[id] = 0;
            // Reaching tolerance is what earns the right to be skipped later,
            // and it is recorded separately from `active` because retiring a
            // degenerate island also clears `active` -- freezing an island
            // that gave up rather than converged would preserve stale,
            // inflated stress and keep breaking bonds off it.
            islandConverged[id] = 1;
        }
        else
        {
            active = 1;
        }
    }

    partial[tid] = active;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        blockActiveCounts[blockIdx.x] = partial[0];
    }
}

/// Roll the per-island flags up into the single status the host reads.
/// Zero the active counter before the parallel tally below accumulates into it.
///
/// Split from the tally so the tally can be a grid-wide reduction: a kernel
/// that both clears and accumulates a shared counter would race itself across
/// blocks.
__global__ void resetActiveCount(SolveStatus* status)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        status->active = 0;
    }
}

/// Count active islands in parallel and latch convergence.
///
/// MEASURED, and this was the single biggest cost in the whole solve. The
/// previous version ran <<<1,1>>> -- one thread walking every island, every
/// iteration. Per-kernel event timing at city scale (24k nodes, 34k bonds,
/// 2000 islands, 32 iterations):
///
///   summarizeIslands            0.7932 ms/solve   24.79 us/launch   34.8%
///   couplingRightMultiply       0.3846 ms/solve   11.66 us/launch   16.9%
///   accumulateSquaredByIsland   0.2506 ms/solve    3.86 us/launch   11.0%
///   ...every other kernel        2.8-4.2 us/launch
///
/// 2000 serial iterations on one CUDA core is ~24 us, which is exactly what
/// it measured -- real serial work, not launch overhead, and it stalled the
/// pipeline between the two halves of every CG iteration. One thread was
/// doing what 16,384 could.
///
/// Block-level reduction in shared memory, then one integer atomicAdd per
/// block. Integer atomics are exact and order-independent, so the result is
/// identical to the serial sum, not merely close -- the same count, every
/// run.
__global__ void summarizeIslands(
    SolveStatus* status,
    const std::uint32_t* islandActive,
    std::uint32_t islandCount,
    std::uint32_t iteration)
{
    __shared__ std::uint32_t partial[kBlockSize];
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t index = blockIdx.x * blockDim.x + tid;
    partial[tid] = (index < islandCount && islandActive[index] != 0u) ? 1u : 0u;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0 && partial[0] != 0u)
    {
        atomicAdd(&status->active, partial[0]);
    }
}

/// Latch convergence once the tally above is complete.
///
/// Separate launch because the decision needs the FINAL count, and a block
/// cannot know whether it was the last to contribute without another
/// synchronisation. One thread is correct here: the work is O(1).
__global__ void latchConvergence(SolveStatus* status, std::uint32_t iteration)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        if (status->active == 0 && !status->converged)
        {
            status->converged = 1;
            status->iterations = iteration;
        }
    }
}

__global__ void checkConvergence(
    SolveStatus* status,
    const float* residualSquared,
    const float* deltaSquared,
    std::uint32_t iteration)
{
    if (threadIdx.x == 0 && blockIdx.x == 0
        && status->active
        && *residualSquared <= *deltaSquared)
    {
        status->active = 0;
        status->iterations = iteration;
        status->converged = 1;
    }
}

__global__ void updateDirectionPerIsland(
    AngLin* direction,
    const AngLin* gradient,
    const float* gradientSquared,
    const float* previousGradientSquared,
    const std::uint32_t* bondIsland,
    const std::uint32_t* islandActive,
    const std::uint32_t* activeBonds,
    const std::uint32_t* activeCounts,
    const std::uint32_t* iterationPtr)
{
    const std::uint32_t iteration = *iterationPtr;
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[0])
    {
        return;
    }
    const std::uint32_t index = activeBonds[slot];
    const std::uint32_t id = bondIsland[index];
    if (id == kNoIsland || !islandActive[id])
    {
        return;
    }
    if (iteration == 0)
    {
        direction[index] = gradient[index];
        return;
    }
    const float denominator = previousGradientSquared[id];
    const float beta = denominator > 0.0f ? gradientSquared[id] / denominator : 0.0f;
    direction[index].angular =
        add(gradient[index].angular, mul(direction[index].angular, beta));
    direction[index].linear =
        add(gradient[index].linear, mul(direction[index].linear, beta));
}

__global__ void saveGradientSquaredPerIsland(
    float* previousGradientSquared,
    const float* gradientSquared,
    const std::uint32_t* islandActive,
    std::uint32_t islandCount)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < islandCount && islandActive[id])
    {
        previousGradientSquared[id] = gradientSquared[id];
    }
}

/// Each island advances by its own step size. A degenerate island (zero or
/// non-finite denominator) retires itself without disturbing the others.
__global__ void updateSolutionAndResidualPerIsland(
    const std::uint32_t* iterationPtr,
    std::uint32_t maxIterations,
    AngLin* solution,
    AngLin* residual,
    const AngLin* direction,
    const AngLin* projectedDirection,
    const float* gradientSquared,
    const float* projectedDirectionSquared,
    std::uint32_t* islandActive,
    const std::uint32_t* bondIsland,
    const std::uint32_t* nodeIsland,
    const std::uint32_t* activeBonds,
    const std::uint32_t* activeNodes,
    const std::uint32_t* activeCounts)
{
    // Chunked loop control (see launchConditionalLoopCaptured): the while-node
    // condition is evaluated every kChunk iterations, so the body can overshoot
    // the budget by up to kChunk-1 iterations. This is the one kernel that
    // mutates the solution, so guarding it here makes the overshoot a pure
    // no-op and keeps the chunked, unchunked and unrolled paths bit-identical.
    // Every other per-iteration buffer is scratch that only feeds this kernel.
    if (*iterationPtr > maxIterations)
    {
        return;
    }

    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < activeCounts[0])
    {
        const std::uint32_t index = activeBonds[slot];
        const std::uint32_t id = bondIsland[index];
        if (id != kNoIsland && islandActive[id])
        {
            const float denominator = projectedDirectionSquared[id];
            if (denominator > 0.0f && isfinite(denominator))
            {
                const float step = gradientSquared[id] / denominator;
                solution[index].angular =
                    add(solution[index].angular, mul(direction[index].angular, step));
                solution[index].linear =
                    add(solution[index].linear, mul(direction[index].linear, step));
            }
        }
    }
    if (slot < activeCounts[1])
    {
        const std::uint32_t index = activeNodes[slot];
        const std::uint32_t id = nodeIsland[index];
        if (id != kNoIsland && islandActive[id])
        {
            const float denominator = projectedDirectionSquared[id];
            if (denominator > 0.0f && isfinite(denominator))
            {
                const float step = gradientSquared[id] / denominator;
                residual[index].angular =
                    sub(residual[index].angular, mul(projectedDirection[index].angular, step));
                residual[index].linear =
                    sub(residual[index].linear, mul(projectedDirection[index].linear, step));
            }
        }
    }
}

/// Retire islands whose step is degenerate. Separate pass so the update above
/// stays branch-simple and every island sees a consistent active flag.
/// Retire degenerate islands, roll the gradient norm forward, and latch
/// convergence -- three island-grid launches fused into one.
///
/// The ordering that makes this legal: saveGradientSquared must run after
/// updateDirection has READ previousGradientSquared, and by this point in the
/// iteration it has. Latching needs the completed active tally from
/// checkConvergencePerIsland, and that kernel finished earlier on the same
/// stream. Every read here is of a value already final.
__global__ void retireDegenerateIslands(
    std::uint32_t* islandActive,
    const float* projectedDirectionSquared,
    float* previousGradientSquared,
    const float* gradientSquared,
    SolveStatus* status,
    const std::uint32_t* blockActiveCounts,
    std::uint32_t blockCount,
    std::uint32_t* iterationPtr,
    std::uint32_t islandCount,
    // Loop control, folded in here rather than run as its own kernel: this is
    // already the one place that knows how many islands are still active, and
    // at ~30k islands an extra launch per iteration cost ~8% of the intact
    // city's solve for a value that was sitting in a register.
    cudaGraphConditionalHandle loopHandle,
    std::uint32_t maxIterations)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < islandCount)
    {
        if (islandActive[id])
        {
            const float denominator = projectedDirectionSquared[id];
            if (!(denominator > 0.0f) || !isfinite(denominator))
            {
                islandActive[id] = 0;
            }
            // Fold of saveGradientSquaredPerIsland: an island that is not
            // active never reads this again, so rolling it forward only for
            // active islands preserves every value the split kernels
            // produced. (The original guarded on islandActive too.)
            previousGradientSquared[id] = gradientSquared[id];
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // Only this thread touches the counter, and only after every other
        // kernel in the iteration has read it, so the read-modify-write needs
        // no synchronisation.
        const std::uint32_t iteration = *iterationPtr;
        std::uint32_t active = 0;
        for (std::uint32_t i = 0; i < blockCount; ++i)
        {
            active += blockActiveCounts[i];
        }
        status->active = active;
        if (active == 0 && !status->converged)
        {
            status->converged = 1;
            status->iterations = iteration;
        }
        const std::uint32_t next = iteration + 1u;
        *iterationPtr = next;
        if (loopHandle != 0)
        {
            // Non-zero keeps the enclosing cudaGraphCondTypeWhile node looping.
            // `active` is this iteration's count, not last iteration's, which
            // is why folding the decision in here is also more accurate than
            // the separate kernel it replaces.
            cudaGraphSetConditional(
                loopHandle, (active != 0u && next < maxIterations) ? 1u : 0u);
        }
    }
}

/// Pack the impulses of the islands that were actually solved into a dense
/// block, so the device-to-host copy and the host's conversion loop cost what
/// changed rather than what exists.
__global__ void gatherImpulses(
    const AngLin* impulses,
    const std::uint32_t* indices,
    AngLin* output,
    std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        output[index] = impulses[indices[index]];
    }
}

/// Write only the node velocities that changed. The device keeps the rest from
/// the previous upload, which is why they are known to be unchanged at all.
__global__ void scatterVelocities(
    ExtStressGpuImpulse* input,
    const std::uint32_t* indices,
    const ExtStressGpuImpulse* values,
    std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
        input[indices[index]] = values[index];
    }
}

/// Flag the entries the compacted index lists should keep. One pass over the
/// raw arrays per REBUILD (skip-set or topology change), instead of one pass
/// per kernel per iteration per tick. The predicates are bondSettled /
/// nodeSettled themselves, so the lists cannot disagree with the guards.
///
/// The flags feed cub::DeviceSelect::Flagged rather than an atomicAdd append:
/// select preserves index order, so a compacted list walks memory in the same
/// ascending order the underlying bond/node arrays are laid out in. With the
/// atomic append the order was whatever the scheduler raced out, and every CG
/// kernel's global loads through the list were uncoalesced -- invisible while
/// most islands skip, ~2x kernel time when nothing skips (fresh whole-map
/// cascade, the regime the solver exists for).
__global__ void flagActiveBonds(
    const std::uint32_t* bondIsland,
    const std::uint32_t* islandSkip,
    std::uint32_t bondCount,
    std::uint32_t* flags)
{
    const std::uint32_t bond = blockIdx.x * blockDim.x + threadIdx.x;
    if (bond < bondCount)
    {
        flags[bond] = bondSettled(islandSkip, bondIsland[bond]) ? 0u : 1u;
    }
}

__global__ void flagActiveNodes(
    const std::uint32_t* nodeIsland,
    const std::uint32_t* islandSkip,
    std::uint32_t nodeCount,
    std::uint32_t* flags)
{
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < nodeCount)
    {
        flags[node] = nodeSettled(islandSkip, nodeIsland[node]) ? 0u : 1u;
    }
}

__global__ void applyStressDamage(
    float bendGainMax,
    const AngLin* impulses,
    const float* colScale,
    const Vec4* normals,
    const float* areas,
    const float* nodeDistances,
    const std::uint32_t* bondMaterials,
    const ExtStressGpuMaterial* materials,
    std::uint32_t materialCount,
    float* health,
    std::uint32_t* brokenBonds,
    std::uint32_t* brokenCount,
    std::uint32_t bondCount,
    float linearImpulseScale,
    float angularImpulseScale)
{
    const std::uint32_t bond = blockIdx.x * blockDim.x + threadIdx.x;
    if (bond >= bondCount || health[bond] <= 0.0f)
    {
        return;
    }

    // Device impulses are solver-scaled; convert to physical units here. Two
    // multiplies on a value this kernel already has in registers.
    AngLin impulse = impulses[bond];
    const float s_j = colScale[bond];
    impulse.angular = mul(impulse.angular, angularImpulseScale * s_j);
    impulse.linear = mul(impulse.linear, linearImpulseScale * s_j);
    const Vec4 normal = normals[bond];
    const float area = areas[bond];
    const float distance = nodeDistances[bond];
    // Same body as the host walk -- see NvBlastExtStressFormula.h.
    float stressNormal, stressShear, stressBend;
    extStressCalcBondStress(
        ExtStressVec3{impulse.linear.x, impulse.linear.y, impulse.linear.z},
        ExtStressVec3{impulse.angular.x, impulse.angular.y, impulse.angular.z},
        ExtStressVec3{normal.x, normal.y, normal.z},
        area, distance, bendGainMax, stressNormal, stressShear, stressBend);

    // Each bond fails against its OWN material. The table is tiny and
    // L2-resident; a plain global read is sufficient.
    const std::uint32_t materialIndex = bondMaterials[bond];
    const ExtStressGpuMaterial material =
        materials[materialIndex < materialCount ? materialIndex : 0];

    const float compression = fmaxf(0.0f, -stressNormal);
    const float tension = fmaxf(0.0f, stressNormal);
    float multiplier = 0.0f;
    if (compression > material.compressionElasticLimit)
    {
        multiplier +=
            (compression - material.compressionElasticLimit)
            / fmaxf(material.compressionFatalLimit - material.compressionElasticLimit, 1.0f);
    }
    if (tension > material.tensionElasticLimit)
    {
        multiplier +=
            (tension - material.tensionElasticLimit)
            / fmaxf(material.tensionFatalLimit - material.tensionElasticLimit, 1.0f);
    }
    if (stressShear > material.shearElasticLimit)
    {
        multiplier +=
            (stressShear - material.shearElasticLimit)
            / fmaxf(material.shearFatalLimit - material.shearElasticLimit, 1.0f);
    }
    if (multiplier <= 0.0f)
    {
        return;
    }

    const float oldHealth = health[bond];
    const float newHealth = fmaxf(0.0f, oldHealth - oldHealth * multiplier);
    health[bond] = newHealth;
    if (oldHealth > 0.0f && newHealth <= 0.0f)
    {
        const std::uint32_t output = atomicAdd(brokenCount, 1u);
        if (output < bondCount)
        {
            brokenBonds[output] = bond;
        }
    }
}


/// The bond-stress walk, one thread per solver bond group.
///
/// One thread per group with the member loop run SERIALLY in slot order, not a
/// warp-level segmented reduction. Two reasons, and the second is the deciding
/// one:
///   - groups hold 1-4 blast bonds in practice, so a warp per group would idle
///     most of its lanes;
///   - a tree reduction sums the area-weighted accumulators in a different
///     order than the host walk, which would make the dual-run audit a
///     tolerance comparison. Accumulating serially in slot order reproduces the
///     host's float arithmetic operation for operation, so the audit can demand
///     equality instead. Given that a 0.0975% divergence once passed every
///     broad gate in this repo, an exact audit is worth more than the lanes.
///
/// Everything here mirrors SupportGraphProcessor::processBondGroup, including
/// the parts that are easy to miss: the unbreakable member takes over the
/// group's values and STOPS the walk (so later members are never even examined
/// for removal), normalizeSafe multiplies by a reciprocal and leaves a
/// degenerate normal untouched rather than zeroing it, and the centroid and
/// node displacement are only divided through when the total area can take
/// damage.
__global__ void bondStressWalk(
    float bendGainMax,
    const std::uint32_t* groupBegin,
    const std::uint32_t* groupSize,
    const std::uint32_t* memberBlastBond,
    const std::uint32_t* bondNode0,
    const std::uint32_t* bondNode1,
    const std::uint32_t* bondMaterial,
    const float* bondNormal,
    const float* bondCentroid,
    const float* bondNodeDisp,
    const float* health,
    const AngLin* impulses,
    const float* colScale,
    float bsLinearScale,
    float bsAngularScale,
    const float* materialElasticLimits,
    std::uint32_t materialCount,
    float unbreakableLimit,
    std::uint32_t groupCount,
    float* groupStressNormal,
    float* groupStressShear,
    float* groupNormalOut,
    float* groupCentroidOut,
    std::uint8_t* nodeOverstressed,
    std::uint32_t* groupRemoveCount,
    std::uint32_t* removeFlag,
    std::uint32_t* overstressedCount)
{
    const std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groupCount)
    {
        return;
    }

    const std::uint32_t begin = groupBegin[g];
    const std::uint32_t size = groupSize[g];

    // Cleared up front rather than as we go: the unbreakable break leaves the
    // tail of the group unvisited, and a stale flag there would resurrect a
    // removal the host never emitted.
    for (std::uint32_t k = 0; k < size; ++k)
    {
        removeFlag[begin + k] = 0u;
    }

    float totalArea = 0.0f;
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    std::uint32_t removeCount = 0;

    for (std::uint32_t k = 0; k < size; ++k)
    {
        const std::uint32_t slot = begin + k;
        const std::uint32_t bb = memberBlastBond[slot];
        const float remainingArea = health[bb];
        if (remainingArea > 0.0f)
        {
            const std::size_t base = 3u * static_cast<std::size_t>(bb);
            const float mnx = bondNormal[base + 0];
            const float mny = bondNormal[base + 1];
            const float mnz = bondNormal[base + 2];
            const float mcx = bondCentroid[base + 0];
            const float mcy = bondCentroid[base + 1];
            const float mcz = bondCentroid[base + 2];
            const float mdx = bondNodeDisp[base + 0];
            const float mdy = bondNodeDisp[base + 1];
            const float mdz = bondNodeDisp[base + 2];

            // canTakeDamage: health in (0, kUnbreakableLimit).
            if (!(remainingArea < unbreakableLimit))
            {
                totalArea = unbreakableLimit;
                nx = mnx; ny = mny; nz = mnz;
                cx = mcx; cy = mcy; cz = mcz;
                dx = mdx; dy = mdy; dz = mdz;
                break;
            }

            // fmaf, because the host accumulates with `v += u * area` and is
            // compiled with -mfma, so gcc contracts it. Matching that
            // explicitly is what keeps the two walks bit-identical.
            nx = fmaf(mnx, remainingArea, nx);
            ny = fmaf(mny, remainingArea, ny);
            nz = fmaf(mnz, remainingArea, nz);
            cx = fmaf(mcx, remainingArea, cx);
            cy = fmaf(mcy, remainingArea, cy);
            cz = fmaf(mcz, remainingArea, cz);
            dx = fmaf(mdx, remainingArea, dx);
            dy = fmaf(mdy, remainingArea, dy);
            dz = fmaf(mdz, remainingArea, dz);
            totalArea += remainingArea;
        }
        else
        {
            removeFlag[slot] = 1u;
            ++removeCount;
        }
    }

    groupRemoveCount[g] = removeCount;

    if (totalArea == 0.0f)
    {
        return;
    }

    // NvVec3::normalizeSafe -- reciprocal multiply, and a degenerate normal is
    // left alone rather than zeroed.
    {
        const float mag =
            sqrtf(extStressDot(ExtStressVec3{nx, ny, nz}, ExtStressVec3{nx, ny, nz}));
        if (!(mag < NVBLAST_STRESS_NORMALIZATION_EPSILON))
        {
            const float inv = 1.0f / mag;
            nx *= inv;
            ny *= inv;
            nz *= inv;
        }
    }

    if (totalArea > 0.0f && totalArea < unbreakableLimit)
    {
        // NvVec3::operator/= takes the reciprocal ONCE and multiplies. A true
        // per-component division is a different value in the last bit, and
        // that difference showed up as whole batches of bonds flipping across
        // an elastic limit together while the city was still settling.
        const float inv = 1.0f / totalArea;
        cx *= inv; cy *= inv; cz *= inv;
        dx *= inv; dy *= inv; dz *= inv;
    }

    float stressNormal = 0.0f;
    float stressShear = 0.0f;
    float stressBend = 0.0f;
    if (totalArea > 0.0f && totalArea < unbreakableLimit)
    {
        // Solver-scaled -> physical, same as applyStressDamage.
        AngLin impulse = impulses[g];
        const float s_g = colScale[g];
        impulse.angular = mul(impulse.angular, bsAngularScale * s_g);
        impulse.linear = mul(impulse.linear, bsLinearScale * s_g);
        const float nodeDist =
            sqrtf(extStressDot(ExtStressVec3{dx, dy, dz}, ExtStressVec3{dx, dy, dz}));
        extStressCalcBondStress(
            ExtStressVec3{impulse.linear.x, impulse.linear.y, impulse.linear.z},
            ExtStressVec3{impulse.angular.x, impulse.angular.y, impulse.angular.z},
            ExtStressVec3{nx, ny, nz},
            totalArea, nodeDist, bendGainMax, stressNormal, stressShear, stressBend);
    }

    groupStressNormal[g] = stressNormal;
    groupStressShear[g] = stressShear;
    groupNormalOut[3u * g + 0u] = nx;
    groupNormalOut[3u * g + 1u] = ny;
    groupNormalOut[3u * g + 2u] = nz;
    groupCentroidOut[3u * g + 0u] = cx;
    groupCentroidOut[3u * g + 1u] = cy;
    groupCentroidOut[3u * g + 2u] = cz;

    // Every member shares the group's stress but fails against its OWN
    // material, so overstress is counted per member.
    std::uint32_t overstressed = 0;
    for (std::uint32_t k = 0; k < size; ++k)
    {
        const std::uint32_t bb = memberBlastBond[begin + k];
        const std::uint32_t node0 = bondNode0[bb];
        if (node0 == 0xFFFFFFFFu || !(health[bb] > 0.0f))
        {
            continue;
        }
        const std::uint32_t rawIndex = bondMaterial[bb];
        const std::uint32_t materialIndex = rawIndex < materialCount ? rawIndex : 0u;
        const float compressionElastic = materialElasticLimits[3u * materialIndex + 0u];
        const float tensionElastic = materialElasticLimits[3u * materialIndex + 1u];
        const float shearElastic = materialElasticLimits[3u * materialIndex + 2u];
        if (-stressNormal > compressionElastic
            || stressNormal > tensionElastic
            || stressShear > shearElastic)
        {
            ++overstressed;
            nodeOverstressed[node0] = 1u;
            nodeOverstressed[bondNode1[bb]] = 1u;
        }
    }
    if (overstressed != 0)
    {
        atomicAdd(overstressedCount, overstressed);
    }
}

/// Stable segmented compaction of the removal list.
///
/// Groups ascending (the offsets come from an exclusive scan over group index)
/// and slots ascending inside each group -- which IS the serial walk's emission
/// order. It is emphatically not sorted by blast bond index: measured on a
/// grid-1 shot run, 2 of 200 non-empty removal lists came out non-ascending,
/// so a sort would diverge from the host on ~1% of the ticks that break bonds,
/// and removal order feeds back into topology.
__global__ void bondStressNullKernel() {}

__global__ void bondStressCompactRemovals(
    const std::uint32_t* groupBegin,
    const std::uint32_t* groupSize,
    const std::uint32_t* memberBlastBond,
    const std::uint32_t* removeFlag,
    const std::uint32_t* removeOffset,
    std::uint32_t groupCount,
    std::uint32_t* removeList)
{
    const std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groupCount)
    {
        return;
    }
    const std::uint32_t begin = groupBegin[g];
    const std::uint32_t size = groupSize[g];
    std::uint32_t out = removeOffset[g];
    for (std::uint32_t k = 0; k < size; ++k)
    {
        if (removeFlag[begin + k] != 0u)
        {
            removeList[out++] = memberBlastBond[begin + k];
        }
    }
}

bool bondStressProbe()
{
    static const bool on = [] {
        const char* raw = std::getenv("BLAST_BOND_STRESS_PROBE");
        return raw != nullptr && std::string(raw) != "0";
    }();
    return on;
}

/// std::vector allocator backed by cudaMallocHost.
///
/// The topology upload was a DOUBLE copy: computeIslands and buildNodeBondCsr
/// filled pageable std::vectors, then those were memcpy'd into a pinned staging
/// arena, then DMA'd. Measured, the DMA enqueue cost 0.09 ms and the host
/// memcpy 1.52 ms -- the copy, not the transfer, was the whole cost. Backing
/// the vectors with pinned memory deletes the middle step: the host code writes
/// where the DMA reads.
///
/// Worth recording why the more obvious idea was NOT built. Uploading only the
/// changed bytes sounds strictly better, but 5.048 MB of the 5.31 MB is dirty
/// every fracture tick -- computeIslands renumbers island ids wholesale -- so
/// there is no narrow dirty range to exploit. That was measured first.
template <typename T>
struct PinnedAllocator
{
    using value_type = T;
    PinnedAllocator() noexcept = default;
    template <typename U>
    PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n)
    {
        void* p = nullptr;
        if (n != 0 && cudaMallocHost(&p, n * sizeof(T)) != cudaSuccess)
        {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept
    {
        if (p != nullptr)
        {
            cudaFreeHost(p);
        }
    }
    template <typename U>
    bool operator==(const PinnedAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const PinnedAllocator<U>&) const noexcept { return false; }
};

template <typename T>
using PinnedVector = std::vector<T, PinnedAllocator<T>>;

class ExtStressGpuSolverImpl final : public ExtStressGpuSolver
{
public:
    ExtStressGpuSolverImpl(
        const ExtStressGpuNode* nodes,
        std::uint32_t nodeCount,
        const ExtStressGpuBond* bonds,
        std::uint32_t bondCount,
        const ExtStressGpuMaterial* materials,
        std::uint32_t materialCount,
        CUcontext cudaContext)
        : m_nodeCount(nodeCount)
        , m_bondCount(bondCount)
        , m_cudaContext(cudaContext)
    {
        // A solver always has a material table; default = one entry with the
        // struct's defaults so limit-free callers keep historical behavior.
        if (materials && materialCount > 0)
        {
            m_hostMaterials.assign(materials, materials + materialCount);
        }
        else
        {
            m_hostMaterials.assign(1, ExtStressGpuMaterial());
        }
        m_materialCount = static_cast<std::uint32_t>(m_hostMaterials.size());
        ContextGuard context(m_cudaContext);
        prepare(nodes, bonds);
computeIslands();
        buildNodeBondCsr();
        groupBondsByIsland();
        allocate();
        uploadTopology();
uploadIslands();
        uploadNodeBondCsr();
        if (s_debug)
        {
            std::fprintf(
                stderr,
                "[blast-gpu-skip] CREATE nodes %u bonds %u islands %u\n",
                m_nodeCount,
                m_bondCount,
                m_islandCount);
        }
    }

    ~ExtStressGpuSolverImpl() override
    {
        ContextGuard context(m_cudaContext);
        cudaStreamSynchronize(m_stream);
        if (m_graphExec)
        {
            cudaGraphExecDestroy(m_graphExec);
        }
        if (m_graph)
        {
            cudaGraphDestroy(m_graph);
        }
        cudaEventDestroy(m_uploadStart);
        cudaEventDestroy(m_uploadStop);
        cudaEventDestroy(m_solveStart);
        cudaEventDestroy(m_solveStop);
        cudaEventDestroy(m_statusReady);
        cudaEventDestroy(m_topoUploadDone);
        cudaEventDestroy(m_downloadStart);
        cudaEventDestroy(m_downloadStop);
        cudaStreamDestroy(m_stream);
        if (m_bodyStream) { cudaStreamDestroy(m_bodyStream); m_bodyStream = nullptr; }
        freeBondStress();
        cudaFreeHost(m_topoStaging);
        cudaFreeHost(m_hostStatus);
        cudaFreeHost(m_hostBrokenCount);
        cudaFreeHost(m_hostIslandConvergedPinned);
        cudaFreeHost(m_hostImpulses);
        cudaFreeHost(m_hostInput);
        cudaFreeHost(m_hostScatterIndices);
        cudaFreeHost(m_hostScatterValues);
        cudaFreeHost(m_hostGatherIndices);
        cudaFreeHost(m_hostActiveCounts);
        cudaFree(m_selectScratch);
        cudaFree(m_activeFlags);
        cudaFree(m_activeCounts);
        cudaFree(m_activeNodes);
        cudaFree(m_activeBonds);
        cudaFree(m_gatherOutput);
        cudaFree(m_gatherIndices);
        cudaFree(m_scatterValues);
        cudaFree(m_scatterIndices);
        cudaFree(m_islandConverged);
        cudaFree(m_islandSkip);
        cudaFree(m_reduceScratch);
        cudaFree(m_status);
        cudaFree(m_previousGradientSquared);
        cudaFree(m_deltaSquared);
        cudaFree(m_islandActive);
        cudaFree(m_blockActiveCounts);
        cudaFree(m_nodeBondBegin);
        cudaFree(m_nodeBondRef);
        cudaFree(m_bondIsland);
        cudaFree(m_nodeIsland);
        cudaFree(m_projectedDirectionSquared);
        cudaFree(m_gradientSquared);
        cudaFree(m_reduceSlots);
        cudaFree(m_iteration);
        cudaFree(m_devDeltaSlots);
        cudaFree(m_devDeltaValues);
        cudaFree(m_nsPi);
        cudaFree(m_nsQ);
        cudaFree(m_nsW);
        cudaFree(m_nsMu);
        cudaFree(m_nsG);
        cudaFree(m_nsW2);
        cudaFree(m_nsJacobi);
        cudaFree(m_nsGamma);
        cudaFree(m_nsGammaPrev);
        cudaFree(m_devRefSlots);
        cudaFree(m_devRefValues);
        cudaFree(m_devNodeIslandSlots);
        cudaFree(m_devNodeIslandValues);
        cudaFree(m_reductionInput);
        cudaFree(m_projectedDirection);
        cudaFree(m_residual);
        cudaFree(m_direction);
        cudaFree(m_gradient);
        cudaFree(m_rhs);
        cudaFree(m_impulses);
        cudaFree(m_input);
        cudaFree(m_brokenCount);
        cudaFree(m_brokenBonds);
        cudaFree(m_health);
        cudaFree(m_colScales);
        cudaFree(m_nodeDistances);
        cudaFree(m_materials);
        cudaFree(m_bondMaterials);
        cudaFree(m_areas);
        cudaFree(m_normals);
        cudaFree(m_inertia);
        cudaFree(m_offset1);
        cudaFree(m_offset0);
        cudaFree(m_node1);
        cudaFree(m_node0);
    }

    void release() override
    {
        if (m_kernelProfile.active && m_profiledSolves > 0)
        {
            m_kernelProfile.dump("eager launches, no CUDA graph", m_profiledSolves);
        }
        delete this;
    }

    /// Periodic BLAST_GPU_GRAPH_STATS dump.
    ///
    /// Called from BOTH solve entry points. It used to live inline in
    /// solveAndReadbackImpulses only, so any caller using the plain solve()
    /// path -- which is what a headless harness naturally uses -- got no host
    /// breakdown at all, and the absence looked like "the stats are off"
    /// rather than "you are on the other entry point".
    void maybeDumpStats()
    {
        if (graphStatsEnabled() && m_islandCount > 0)
        {
            accumulateResidualStats();
        }
        if (graphStatsEnabled() && (++m_statSolves % 600u) == 0u)
        {
            fprintf(stderr,
                    "[gpu-graph] solves=%u listRefresh=%u listSkip=%u (%.1f%% refreshed) "
                    "graphRecapture=%u (%.2f%% of solves) midSync=%.4f ms/solve "
                    "| recapture %.4f ms/solve (%.3f each) "
                    "| graphUpdate=%u %.4f ms/solve (%.3f each)\n",
                    m_statSolves, m_statListRefreshes, m_statListSkips,
                    100.0 * double(m_statListRefreshes)
                        / double(m_statListRefreshes + m_statListSkips + 1u),
                    m_statGraphRecaptures,
                    100.0 * double(m_statGraphRecaptures) / double(m_statSolves),
                    m_statMidSyncMs / double(m_statSolves),
                    (m_statCaptureMs + m_statInstantiateMs) / double(m_statSolves),
                    (m_statCaptureMs + m_statInstantiateMs)
                        / double(m_statGraphRecaptures + 1u),
                    m_statGraphUpdates,
                    m_statUpdateMs / double(m_statSolves),
                    m_statUpdateMs / double(m_statGraphUpdates + 1u));
            // Where the HOST time actually goes, split from where it merely
            // waits. Only `plan` and `finish` are reclaimable by writing
            // faster host code; `wait` is the device computing, and shrinking
            // it means less device work or more overlap, not tighter host code.
            fprintf(stderr,
                    "[gpu-host] plan=%.4f (graphUpdate=%.4f, midSync=%.4f, "
                    "other=%.4f | planSkip=%.4f refresh=%.4f topo=%.4f "
                    "[%u calls, %.3f each: islands=%.3f csr=%.3f group=%.3f "
                    "upload=%.3f memset=%.3f | %.2f MB in %u copies, "
                    "stageMemcpy=%.3f, %u growths, %.2f GB/s]) "
                    "wait=%.4f finish=%.4f ms/solve "
                    "| iters=%.1f/solve unconverged=%u (%.1f%% of solves) "
                    "| residual/tolerance: mean=%.2fx max=%.2fx, %.1f%% of islands over tol\n",
                    m_statPlanMs / double(m_statSolves),
                    m_statUpdateMs / double(m_statSolves),
                    m_statMidSyncMs / double(m_statSolves),
                    (m_statPlanMs - m_statUpdateMs) / double(m_statSolves),
                    m_statPlanSkipMs / double(m_statSolves),
                    m_statRefreshMs / double(m_statSolves),
                    m_statTopoMs / double(m_statSolves),
                    m_statTopoCalls,
                    m_statTopoMs / double(m_statTopoCalls + 1u),
                    m_statTopoIslandsMs / double(m_statTopoCalls + 1u),
                    m_statTopoCsrMs / double(m_statTopoCalls + 1u),
                    m_statTopoGroupMs / double(m_statTopoCalls + 1u),
                    m_statTopoUploadMs / double(m_statTopoCalls + 1u),
                    m_statTopoMemsetMs / double(m_statTopoCalls + 1u),
                    double(m_statTopoBytes) / double(m_statTopoCalls + 1u) / 1048576.0,
                    m_statTopoCopies / (m_statTopoCalls + 1u),
                    m_statStageCopyMs / double(m_statTopoCalls + 1u),
                    m_statTopoGrowths,
                    (double(m_statTopoBytes) / 1.0e9)
                        / ((m_statTopoUploadMs > 0.0 ? m_statTopoUploadMs : 1.0) / 1000.0),
                    m_statWaitMs / double(m_statSolves),
                    m_statFinishMs / double(m_statSolves),
                    double(m_statIterations) / double(m_statSolves),
                    m_statUnconverged,
                    100.0 * double(m_statUnconverged) / double(m_statSolves),
                    m_statResCount ? m_statResSum / double(m_statResCount) : 0.0,
                    m_statResMax,
                    m_statResCount
                        ? 100.0 * double(m_statResOverTol) / double(m_statResCount)
                        : 0.0);
        }
    }

    bool solve(
        const ExtStressGpuImpulse* nodeVelocities,
        const ExtStressGpuSolveParams& params) override
    {
        ContextGuard context(m_cudaContext);
        using HostClock = std::chrono::steady_clock;
        const auto hostMs = [](HostClock::time_point from) {
            return std::chrono::duration<float, std::milli>(HostClock::now() - from)
                .count();
        };
        const auto planStart = HostClock::now();
        if (!enqueueSolve(nodeVelocities, params))
        {
            return false;
        }
        // enqueueSolve contains the mid-solve stall in refreshActiveLists, so
        // its own sync time is subtracted out below rather than counted as
        // planning work.
        m_telemetry.hostPlanMilliseconds =
            hostMs(planStart) - m_activeListSyncMilliseconds;
        m_telemetry.hostSyncMilliseconds = m_activeListSyncMilliseconds;
        m_activeListSyncMilliseconds = 0.0f;
        if (m_solveWasNoOp)
        {
            return true;
        }
        const auto waitStart = HostClock::now();
        checkCuda(cudaEventSynchronize(m_statusReady), "wait for stress solve");
        const float solveWaitMs = hostMs(waitStart);
        m_telemetry.hostSyncMilliseconds += solveWaitMs;
        m_statWaitMs += solveWaitMs;
        const auto finishStart = HostClock::now();
        finishSolve();
        m_telemetry.hostFinishMilliseconds = hostMs(finishStart);
        m_statPlanMs += m_telemetry.hostPlanMilliseconds;
        m_statFinishMs += m_telemetry.hostFinishMilliseconds;
        maybeDumpStats();
        return true;
    }

    bool readbackImpulses(
        ExtStressGpuImpulse* bondImpulses,
        std::uint32_t capacity) override
    {
        ContextGuard context(m_cudaContext);
        if (!bondImpulses || capacity < m_bondCount)
        {
            return false;
        }
        // Public whole-array contract: the caller has not been told which
        // bonds moved, so every one is copied.
        m_changedBonds.resize(m_bondCount);
        for (std::uint32_t i = 0; i < m_bondCount; ++i)
        {
            m_changedBonds[i] = i;
        }
        enqueueImpulseReadback(false);
        checkCuda(cudaEventSynchronize(m_downloadStop), "wait for impulse readback");
        finishImpulseReadback(bondImpulses, false);
        return true;
    }

    bool solveAndReadbackImpulses(
        const ExtStressGpuImpulse* nodeVelocities,
        const ExtStressGpuSolveParams& params,
        ExtStressGpuImpulse* bondImpulses,
        std::uint32_t capacity) override
    {
        m_skipStableUnconverged = params.skipStableUnconverged;
        ContextGuard context(m_cudaContext);
        // Instrumented here as well as in solve(), because THIS is the entry
        // point production uses -- the adapter calls
        // solveAndReadbackImpulses, and a host split measured only in solve()
        // reported a clean 0.00 ms live while the real path went unmeasured.
        using HostClock = std::chrono::steady_clock;
        const auto hostMs = [](HostClock::time_point from) {
            return std::chrono::duration<float, std::milli>(HostClock::now() - from)
                .count();
        };
        const auto planStart = HostClock::now();
        if (!bondImpulses || capacity < m_bondCount
            || !enqueueSolve(nodeVelocities, params))
        {
            return false;
        }
        m_telemetry.hostPlanMilliseconds =
            hostMs(planStart) - m_activeListSyncMilliseconds;
        m_telemetry.hostSyncMilliseconds = m_activeListSyncMilliseconds;
        m_activeListSyncMilliseconds = 0.0f;
        if (m_solveWasNoOp)
        {
            // Every island settled: the impulses on the device are the ones
            // the caller is already holding, so there is nothing to launch,
            // synchronize on, or copy back.
            return true;
        }
        // The status and impulse copies share one stream. Waiting for the
        // latter completes the upload, solve, status, and impulse readback.
        const bool compacted = m_changedBonds.size() < m_bondCount;
        const auto enqueueStart = HostClock::now();
        enqueueImpulseReadback(compacted);
        m_telemetry.hostPlanMilliseconds += hostMs(enqueueStart);
        const auto waitStart = HostClock::now();
        checkCuda(cudaEventSynchronize(m_downloadStop), "wait for stress solve and readback");
        m_telemetry.hostSyncMilliseconds += hostMs(waitStart);
        const auto finishStart = HostClock::now();
        finishSolve();
        finishImpulseReadback(bondImpulses, compacted);
        m_telemetry.hostFinishMilliseconds = hostMs(finishStart);
        m_statPlanMs += m_telemetry.hostPlanMilliseconds;
        m_statFinishMs += m_telemetry.hostFinishMilliseconds;
        m_statWaitMs += m_telemetry.hostSyncMilliseconds - m_statLastMidSync;
        // Convergence, not just cost. The whole question about the incremental
        // topology path is whether keeping a stale warm start leaves the solve
        // UNCONVERGED at the iteration cap -- an unconverged stress field is
        // what decides which bonds break, so it is a correctness signal, not a
        // performance one.
        m_statIterations += m_telemetry.iterations;
        if (!m_telemetry.converged)
        {
            ++m_statUnconverged;
        }
        // HOW FAR from converged, not just whether. The convergence test is
        // residual^2 <= tolerance^2 per island, so sqrt(gradientSq/deltaSq) is
        // the factor by which an island misses its own tolerance: 1.0 is
        // exactly converged, 10.0 means the residual is ten times the target.
        //
        // This is the number that answers "how much accuracy is the iteration
        // cap costing us". Bonds-broken cannot answer it -- breakage is a
        // threshold crossing on top of a chaotic cascade, so it amplifies a
        // small stress error into a large outcome difference and tells you
        // nothing about the size of the error itself.
        maybeDumpStats();
        return true;
    }

    bool readbackBrokenBonds(
        std::uint32_t* bondIndices,
        std::uint32_t capacity,
        std::uint32_t& count) override
    {
        ContextGuard context(m_cudaContext);
        count = 0;
        checkCuda(
            cudaMemcpy(&count, m_brokenCount, sizeof(count), cudaMemcpyDeviceToHost),
            "read broken bond count");
        if (count > m_bondCount || count > capacity || (count > 0 && !bondIndices))
        {
            return false;
        }
        if (count > 0)
        {
            checkCuda(
                cudaMemcpy(
                    bondIndices,
                    m_brokenBonds,
                    sizeof(std::uint32_t) * count,
                    cudaMemcpyDeviceToHost),
                "read broken bond indices");
        }
        m_telemetry.deviceToHostBytes +=
            sizeof(count) + sizeof(std::uint32_t) * static_cast<std::uint64_t>(count);
        return true;
    }

    bool readbackBondHealth(float* health, std::uint32_t capacity) override
    {
        ContextGuard context(m_cudaContext);
        if (!health || capacity < m_bondCount)
        {
            return false;
        }
        checkCuda(
            cudaMemcpy(
                health,
                m_health,
                sizeof(float) * m_bondCount,
                cudaMemcpyDeviceToHost),
            "read bond health");
        m_telemetry.deviceToHostBytes +=
            sizeof(float) * static_cast<std::uint64_t>(m_bondCount);
        return true;
    }

    bool removeBond(std::uint32_t bondIndex) override
    {
        if (bondIndex >= m_bondCount)
        {
            return false;
        }
        // Swap-with-last, exactly as ConjugateGradientImpulseSolver does to its
        // own bond and impulse arrays. Both sides must apply the same
        // permutation or the impulse the host reads back belongs to a
        // different bond -- a silent, physics-shaped corruption rather than an
        // error.
        const std::uint32_t last = m_bondCount - 1;
        // The removed bond's island must re-solve: its frozen impulses no
        // longer describe the post-break topology. Nodes are the stable key --
        // island ids get remapped by the repartition this removal triggers.
        m_forceDirtyNodes.push_back(m_hostNode0[bondIndex]);
        m_forceDirtyNodes.push_back(m_hostNode1[bondIndex]);
        // Captured here, before swapWithLast overwrites slot `bondIndex`.
        m_splitChecks.push_back({m_hostNode0[bondIndex], m_hostNode1[bondIndex]});
        // Device impulses must follow the same permutation as the host arrays;
        // replayed in order in applyTopologyChange.
        m_pendingImpulseSwaps.emplace_back(bondIndex, last);
        swapWithLast(m_hostNode0, bondIndex, last);
        swapWithLast(m_hostNode1, bondIndex, last);
        swapWithLast(m_hostOffset0, bondIndex, last);
        swapWithLast(m_hostOffset1, bondIndex, last);
        swapWithLast(m_hostNormals, bondIndex, last);
        swapWithLast(m_hostAreas, bondIndex, last);
        swapWithLast(m_hostColScale, bondIndex, last);
        swapWithLast(m_hostNodeDistances, bondIndex, last);
        swapWithLast(m_hostHealth, bondIndex, last);
        swapWithLast(m_hostBondMaterials, bondIndex, last);
        // The island label travels with the bond, so the existing partition
        // stays self-consistent without being recomputed. That is what lets
        // the repartition be deferred (see shouldRepartition).
        if (bondIndex < m_hostBondIsland.size() && last < m_hostBondIsland.size())
        {
            m_hostBondIsland[bondIndex] = m_hostBondIsland[last];
        }
        // m_bondsByIsland indexes bonds, and two of them just moved.
        m_bondsByIslandValid = false;
        ++m_removalsSinceRepartition;
        // Exactly one live slot changed: `bondIndex` now holds what `last`
        // held. Slots at or past the new bond count are never read again.
        if (bondIndex != last)
        {
            m_changedBondSlots.push_back(bondIndex);
        }
        patchCsrForRemoval(bondIndex, last);
        m_bondCount = last;
        m_topologyDirty = true;
        return true;
    }

    const std::uint32_t* lastChangedBonds(std::uint32_t& count) const override
    {
        count = static_cast<std::uint32_t>(m_changedBonds.size());
        return m_changedBonds.data();
    }


    // ── Device bond-stress walk ────────────────────────────────────────────
    //
    // The reason this exists: the impulses this walk consumes are already
    // resident here, and are copied back to the host every tick for no other
    // purpose than feeding it. On the host it is the largest single cost in
    // the tick and it is linear in TOTAL live bonds, not in activity, so no
    // amount of skipping or CPU fan-out reaches it.

    bool setBondStressTopology(const ExtStressGpuBondStressTopology& topology) override
    {
        // Every other public entry point pushes PhysX's CUDA context before
        // touching the device; these did not, so their allocations, streams
        // and launches landed in whatever context the calling thread happened
        // to have -- the primary one -- while the impulses they read were
        // allocated under the guard in PhysX's. Cross-context by construction,
        // and the reason a tiny kernel could not be submitted without the GPU
        // switching contexts around it.
        ContextGuard context(m_cudaContext);
        if (topology.groupCount == 0 || topology.blastBondCount == 0)
        {
            m_bsReady = false;
            return false;
        }
        try
        {
            ensureBondStressCapacity(topology);

            // Static payload, indexed by blast bond: uploaded once per resync
            // because blast bond indices are never permuted.
            uploadBondStress(m_bsBondNode0, topology.bondNode0, topology.blastBondCount, "bs node0");
            uploadBondStress(m_bsBondNode1, topology.bondNode1, topology.blastBondCount, "bs node1");
            uploadBondStress(m_bsBondMaterial, topology.bondMaterial, topology.blastBondCount, "bs material");
            uploadBondStress(m_bsBondNormal, topology.bondNormal, 3u * topology.blastBondCount, "bs normal");
            uploadBondStress(m_bsBondCentroid, topology.bondCentroid, 3u * topology.blastBondCount, "bs centroid");
            uploadBondStress(m_bsBondNodeDisp, topology.bondNodeDisp, 3u * topology.blastBondCount, "bs nodeDisp");

            // syncBonds zeroes every bond's stress when the graph structure
            // changes, because internal bonds stop being updated. Match it.
            checkCuda(cudaMemset(m_bsGroupStressNormal, 0, sizeof(float) * topology.groupCount), "bs clear sn");
            checkCuda(cudaMemset(m_bsGroupStressShear, 0, sizeof(float) * topology.groupCount), "bs clear ss");
            checkCuda(cudaMemset(m_bsGroupNormal, 0, sizeof(float) * 3u * topology.groupCount), "bs clear n");
            checkCuda(cudaMemset(m_bsGroupCentroid, 0, sizeof(float) * 3u * topology.groupCount), "bs clear c");
            m_bsCsrResident = false;
            m_bsHealthResident = false;
            m_bsReady = true;
        }
        catch (const std::exception&)
        {
            m_bsReady = false;
            return false;
        }
        return true;
    }

    bool readbackGroupStresses(const float*& stressNormal, const float*& stressShear) override
    {
        // Every other public entry point pushes PhysX's CUDA context before
        // touching the device; these did not, so their allocations, streams
        // and launches landed in whatever context the calling thread happened
        // to have -- the primary one -- while the impulses they read were
        // allocated under the guard in PhysX's. Cross-context by construction,
        // and the reason a tiny kernel could not be submitted without the GPU
        // switching contexts around it.
        ContextGuard context(m_cudaContext);
        if (!m_bsReady || m_bsVectorGroups == 0 || m_bsStream == nullptr)
        {
            return false;
        }
        if (!m_bsStressFetched)
        {
            try
            {
                // On m_bsStream, not the legacy default stream: a blocking
                // cudaMemcpy there implicitly synchronises with EVERY stream
                // on the device, so the lazy fetch would wait on PhysX's whole
                // pipeline. That turned a cheap deferred read into a stall and
                // showed up as a 2.2x spread between otherwise identical arms.
                checkCuda(
                    cudaMemcpyAsync(m_bsHostGroupStressNormal, m_bsGroupStressNormal,
                                    sizeof(float) * m_bsVectorGroups,
                                    cudaMemcpyDeviceToHost, m_bsStream),
                    "bs lazy read sn");
                checkCuda(
                    cudaMemcpyAsync(m_bsHostGroupStressShear, m_bsGroupStressShear,
                                    sizeof(float) * m_bsVectorGroups,
                                    cudaMemcpyDeviceToHost, m_bsStream),
                    "bs lazy read ss");
                checkCuda(cudaStreamSynchronize(m_bsStream), "bs lazy stress sync");
            }
            catch (const std::exception&)
            {
                return false;
            }
            m_bsStressFetched = true;
        }
        stressNormal = m_bsHostGroupStressNormal;
        stressShear = m_bsHostGroupStressShear;
        return true;
    }

    bool readbackGroupVectors(const float*& groupNormal, const float*& groupCentroid) override
    {
        // Every other public entry point pushes PhysX's CUDA context before
        // touching the device; these did not, so their allocations, streams
        // and launches landed in whatever context the calling thread happened
        // to have -- the primary one -- while the impulses they read were
        // allocated under the guard in PhysX's. Cross-context by construction,
        // and the reason a tiny kernel could not be submitted without the GPU
        // switching contexts around it.
        ContextGuard context(m_cudaContext);
        if (!m_bsReady || m_bsVectorGroups == 0 || m_bsStream == nullptr)
        {
            return false;
        }
        if (!m_bsVectorsFetched)
        {
            try
            {
                checkCuda(
                    cudaMemcpyAsync(
                        m_bsHostGroupNormal, m_bsGroupNormal,
                        sizeof(float) * 3u * m_bsVectorGroups,
                        cudaMemcpyDeviceToHost, m_bsStream),
                    "bs lazy read normal");
                checkCuda(
                    cudaMemcpyAsync(
                        m_bsHostGroupCentroid, m_bsGroupCentroid,
                        sizeof(float) * 3u * m_bsVectorGroups,
                        cudaMemcpyDeviceToHost, m_bsStream),
                    "bs lazy read centroid");
                checkCuda(cudaStreamSynchronize(m_bsStream), "bs lazy vector sync");
            }
            catch (const std::exception&)
            {
                return false;
            }
            m_bsVectorsFetched = true;
        }
        groupNormal = m_bsHostGroupNormal;
        groupCentroid = m_bsHostGroupCentroid;
        return true;
    }

    bool updateBondStress(
        const ExtStressGpuBondStressTopology& csr,
        const float* blastBondHealth,
        float unbreakableLimit,
        ExtStressGpuBondStressResult& result) override
    {
        // Every other public entry point pushes PhysX's CUDA context before
        // touching the device; these did not, so their allocations, streams
        // and launches landed in whatever context the calling thread happened
        // to have -- the primary one -- while the impulses they read were
        // allocated under the guard in PhysX's. Cross-context by construction,
        // and the reason a tiny kernel could not be submitted without the GPU
        // switching contexts around it.
        ContextGuard context(m_cudaContext);
        if (!m_bsReady
            || csr.groupCount == 0
            || csr.groupCount > m_bsGroupCapacity
            || csr.memberSlotCount > m_bsSlotCapacity
            || csr.blastBondCount > m_bsBondCapacity
            || csr.graphNodeCount > m_bsNodeCapacity
            || csr.materialCount == 0
            || csr.materialElasticLimits == nullptr
            || csr.groupCount > m_bondCount)
        {
            return false;
        }
        try
        {
            const auto bsHostStart = std::chrono::steady_clock::now();
            if (m_bsStream == nullptr)
            {
                // Its own stream, at the highest priority the device offers.
                //
                // The walk does not have to be ordered behind anything on the
                // solver stream: it reads impulses the solve has already
                // synchronised, and deviceImpulsesUsable() is what guarantees
                // that. Sharing a stream only forced it to queue.
                //
                // Standalone this walk costs 0.056 ms at the city's group
                // count and is unaffected by other PROCESSES hammering the
                // GPU (+7%). In-game the identical work was costing 1.183 ms,
                // so what it was waiting for was in this process: PhysX's own
                // rigid-body simulation, in the same context.
                int lo = 0, hi = 0;
                cudaDeviceGetStreamPriorityRange(&lo, &hi);
                cudaStreamCreateWithPriority(&m_bsStream, cudaStreamNonBlocking, hi);
            }
            if (m_bsEvUploadStart == nullptr)
            {
                cudaEventCreate(&m_bsEvUploadStart);
                cudaEventCreate(&m_bsEvUploadStop);
                cudaEventCreate(&m_bsEvKernelStop);
                cudaEventCreate(&m_bsEvReadStop);
            }
            // Probe: what does synchronising cost when there is NOTHING to
            // wait for? Three points, at the exact place in the tick the real
            // walk runs. If an EMPTY sync already costs what the real one
            // does, the fixed cost has nothing to do with our work.
            if (bondStressProbe())
            {
                const auto p0 = std::chrono::steady_clock::now();
                cudaStreamSynchronize(m_bsStream);          // nothing queued
                const auto p1 = std::chrono::steady_clock::now();
                bondStressNullKernel<<<1, 1, 0, m_bsStream>>>();
                cudaStreamSynchronize(m_bsStream);          // FIRST empty launch
                const auto p2 = std::chrono::steady_clock::now();
                bondStressNullKernel<<<1, 1, 0, m_bsStream>>>();
                cudaStreamSynchronize(m_bsStream);          // SECOND, back to back
                const auto p2b = std::chrono::steady_clock::now();
                m_telemetry.bondStressProbeKernel2Ms =
                    std::chrono::duration<float, std::milli>(p2b - p2).count();
                const auto p2c = std::chrono::steady_clock::now();
                cudaMemcpyAsync(m_bsHostCounts, m_bsOverstressedCount,
                                sizeof(std::uint32_t), cudaMemcpyDeviceToHost, m_bsStream);
                cudaStreamSynchronize(m_bsStream);          // one 4-byte D2H
                const auto p3 = std::chrono::steady_clock::now();
                m_telemetry.bondStressProbeEmptyMs =
                    std::chrono::duration<float, std::milli>(p1 - p0).count();
                m_telemetry.bondStressProbeKernelMs =
                    std::chrono::duration<float, std::milli>(p2 - p1).count();
                m_telemetry.bondStressProbeCopyMs =
                    std::chrono::duration<float, std::milli>(p3 - p2c).count();
            }

            const std::uint32_t groups = csr.groupCount;
            m_telemetry.bondStressBytesUp = 0;
            m_telemetry.bondStressBytesDown = 0;
            cudaEventRecord(m_bsEvUploadStart, m_bsStream);

            // Uploads go through PINNED staging, and only when the bytes
            // actually changed.
            //
            // Measured before this: 1.13 MB per call taking 1.250 ms, which is
            // 0.9 GB/s -- pageable-memory speed, not PCIe speed. cudaMemcpyAsync
            // from a std::vector cannot DMA, so it stages through a driver
            // bounce buffer and serialises against the stream. The kernel it
            // was feeding runs in 0.041 ms.
            //
            // The CSR only changes when a bond breaks, and health only changes
            // when something takes damage; at rest neither moves, so both
            // uploads collapse to a compare.
            std::uint64_t bytesUp = 0;
            const auto prepStart = std::chrono::steady_clock::now();
            if (csr.csrDirty || !m_bsCsrResident)
            {
                memcpy(m_bsPinGroupBegin, csr.groupBegin, sizeof(std::uint32_t) * groups);
                memcpy(m_bsPinGroupSize, csr.groupSize, sizeof(std::uint32_t) * groups);
                memcpy(m_bsPinMembers, csr.memberBlastBond,
                       sizeof(std::uint32_t) * csr.memberSlotCount);
                uploadBondStressAsync(m_bsGroupBegin, m_bsPinGroupBegin, groups, "bs groupBegin");
                uploadBondStressAsync(m_bsGroupSize, m_bsPinGroupSize, groups, "bs groupSize");
                uploadBondStressAsync(
                    m_bsMemberBlastBond, m_bsPinMembers, csr.memberSlotCount, "bs members");
                bytesUp += sizeof(std::uint32_t) * (2ull * groups + csr.memberSlotCount);
                m_bsCsrResident = true;
            }
            const std::size_t healthBytes = sizeof(float) * csr.blastBondCount;
            if (!m_bsHealthResident
                || memcmp(m_bsPinHealth, blastBondHealth, healthBytes) != 0)
            {
                memcpy(m_bsPinHealth, blastBondHealth, healthBytes);
                uploadBondStressAsync(
                    m_bsHealth, m_bsPinHealth, csr.blastBondCount, "bs health");
                bytesUp += healthBytes;
                m_bsHealthResident = true;
            }
            // Small, and it can change without a topology rebuild, so it is
            // refreshed every call rather than cached.
            if (csr.materialCount > m_bsMaterialCapacity)
            {
                cudaFree(m_bsMaterialLimits);
                m_bsMaterialLimits = nullptr;
                allocateDevice(m_bsMaterialLimits, 3u * csr.materialCount, "bs alloc materials");
                cudaFreeHost(m_bsPinMaterials);
                allocateHost(m_bsPinMaterials, 3u * csr.materialCount, "bs pin materials");
                m_bsMaterialCapacity = csr.materialCount;
            }
            memcpy(m_bsPinMaterials, csr.materialElasticLimits,
                   sizeof(float) * 3u * csr.materialCount);
            uploadBondStressAsync(
                m_bsMaterialLimits, m_bsPinMaterials, 3u * csr.materialCount, "bs materials");

            checkCuda(
                cudaMemsetAsync(
                    m_bsNodeOverstressed, 0, sizeof(std::uint8_t) * csr.graphNodeCount, m_bsStream),
                "bs clear node mask");
            checkCuda(
                cudaMemsetAsync(m_bsOverstressedCount, 0, sizeof(std::uint32_t), m_bsStream),
                "bs clear count");
            // Sentinel entry for the exclusive scan, so offset[groups] is the
            // total without a second reduction.
            checkCuda(
                cudaMemsetAsync(
                    m_bsGroupRemoveCount + groups, 0, sizeof(std::uint32_t), m_bsStream),
                "bs clear scan sentinel");

            bytesUp += sizeof(float) * 3ull * csr.materialCount;
            m_telemetry.bondStressBytesUp = bytesUp;
            m_telemetry.bondStressPrepMs =
                std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - prepStart).count();
            cudaEventRecord(m_bsEvUploadStop, m_bsStream);

            // Carry each reassigned slot's stored stress with it, so a slot
            // handed to a different group does not answer with the old one's
            // values on any tick that group is not reprocessed.
            for (std::uint32_t i = 0; i < csr.groupSwapCount; ++i)
            {
                const std::uint32_t dst = csr.groupSwapDst[i];
                const std::uint32_t src = csr.groupSwapSrc[i];
                if (dst == src || dst >= m_bsGroupCapacity || src >= m_bsGroupCapacity)
                {
                    continue;
                }
                cudaMemcpyAsync(m_bsGroupStressNormal + dst, m_bsGroupStressNormal + src,
                                sizeof(float), cudaMemcpyDeviceToDevice, m_bsStream);
                cudaMemcpyAsync(m_bsGroupStressShear + dst, m_bsGroupStressShear + src,
                                sizeof(float), cudaMemcpyDeviceToDevice, m_bsStream);
                cudaMemcpyAsync(m_bsGroupNormal + 3 * dst, m_bsGroupNormal + 3 * src,
                                sizeof(float) * 3, cudaMemcpyDeviceToDevice, m_bsStream);
                cudaMemcpyAsync(m_bsGroupCentroid + 3 * dst, m_bsGroupCentroid + 3 * src,
                                sizeof(float) * 3, cudaMemcpyDeviceToDevice, m_bsStream);
            }

            const std::uint32_t block = 128;
            const std::uint32_t grid = (groups + block - 1) / block;
            bondStressWalk<<<grid, block, 0, m_bsStream>>>(
                m_bendGainMax,
                m_bsGroupBegin, m_bsGroupSize, m_bsMemberBlastBond,
                m_bsBondNode0, m_bsBondNode1, m_bsBondMaterial,
                m_bsBondNormal, m_bsBondCentroid, m_bsBondNodeDisp,
                m_bsHealth, m_impulses, m_colScales,
                m_lengthScale * m_massScale,
                m_lengthScale * m_lengthScale * m_massScale,
                m_bsMaterialLimits, csr.materialCount,
                unbreakableLimit, groups,
                m_bsGroupStressNormal, m_bsGroupStressShear,
                m_bsGroupNormal, m_bsGroupCentroid,
                m_bsNodeOverstressed, m_bsGroupRemoveCount,
                m_bsRemoveFlag, m_bsOverstressedCount);
            checkCuda(cudaGetLastError(), "bs walk launch");

            std::size_t scanBytes = 0;
            checkCuda(
                cub::DeviceScan::ExclusiveSum(
                    nullptr, scanBytes, m_bsGroupRemoveCount, m_bsGroupRemoveOffset,
                    static_cast<int>(groups + 1), m_bsStream),
                "bs scan sizing");
            if (scanBytes > m_bsScanScratchBytes)
            {
                cudaFree(m_bsScanScratch);
                m_bsScanScratch = nullptr;
                allocateDevice(
                    reinterpret_cast<char*&>(m_bsScanScratch), scanBytes, "bs scan scratch");
                m_bsScanScratchBytes = scanBytes;
            }
            checkCuda(
                cub::DeviceScan::ExclusiveSum(
                    m_bsScanScratch, scanBytes, m_bsGroupRemoveCount, m_bsGroupRemoveOffset,
                    static_cast<int>(groups + 1), m_bsStream),
                "bs scan");

            bondStressCompactRemovals<<<grid, block, 0, m_bsStream>>>(
                m_bsGroupBegin, m_bsGroupSize, m_bsMemberBlastBond,
                m_bsRemoveFlag, m_bsGroupRemoveOffset, groups, m_bsRemoveList);
            checkCuda(cudaGetLastError(), "bs compact launch");
            cudaEventRecord(m_bsEvKernelStop, m_bsStream);

            // Fixed-size readbacks first, then one sync, then the removal list
            // whose length we only know after it.
            checkCuda(
                cudaMemcpyAsync(
                    m_bsHostCounts, m_bsOverstressedCount, sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, m_bsStream),
                "bs read count");
            checkCuda(
                cudaMemcpyAsync(
                    m_bsHostCounts + 1, m_bsGroupRemoveOffset + groups, sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost, m_bsStream),
                "bs read remove total");
            checkCuda(
                cudaMemcpyAsync(
                    m_bsHostNodeOverstressed, m_bsNodeOverstressed,
                    sizeof(std::uint8_t) * csr.graphNodeCount,
                    cudaMemcpyDeviceToHost, m_bsStream),
                "bs read node mask");
            m_telemetry.bondStressBytesDown =
                2ull * sizeof(std::uint32_t)
                + sizeof(std::uint8_t) * csr.graphNodeCount;
            m_bsVectorsFetched = false;
            m_bsStressFetched = false;
            m_bsVectorGroups = groups;
            cudaEventRecord(m_bsEvReadStop, m_bsStream);
            const auto bsSyncStart = std::chrono::steady_clock::now();
            m_telemetry.bondStressEnqueueMs =
                std::chrono::duration<float, std::milli>(bsSyncStart - prepStart).count()
                - m_telemetry.bondStressPrepMs;
            // Spin, then fall back to blocking.
            //
            // The device work here is ~0.13 ms. cudaStreamSynchronize under
            // the default scheduling policy is free to hand the core back to
            // the OS, and getting it back costs more than the work did.
            // Spinning on a query for a bounded number of iterations covers
            // the common case without giving up the thread; anything longer
            // than that is a real stall and worth blocking on.
            {
                bool done = false;
                for (int spin = 0; spin < 20000; ++spin)
                {
                    const cudaError_t q = cudaStreamQuery(m_bsStream);
                    if (q == cudaSuccess) { done = true; break; }
                    if (q != cudaErrorNotReady) { checkCuda(q, "bs spin"); }
                }
                if (!done)
                {
                    checkCuda(cudaStreamSynchronize(m_bsStream), "bs sync");
                }
            }
            const auto bsSyncEnd = std::chrono::steady_clock::now();
            m_telemetry.bondStressSyncMs =
                std::chrono::duration<float, std::milli>(bsSyncEnd - bsSyncStart).count();
            {
                float up = 0.0f, kern = 0.0f, read = 0.0f;
                cudaEventElapsedTime(&up, m_bsEvUploadStart, m_bsEvUploadStop);
                cudaEventElapsedTime(&kern, m_bsEvUploadStop, m_bsEvKernelStop);
                cudaEventElapsedTime(&read, m_bsEvKernelStop, m_bsEvReadStop);
                m_telemetry.bondStressUploadMs = up;
                m_telemetry.bondStressKernelMs = kern;
                m_telemetry.bondStressReadbackMs = read;
            }

            const std::uint32_t removeCount =
                m_bsHostCounts[1] > csr.memberSlotCount ? csr.memberSlotCount : m_bsHostCounts[1];
            if (removeCount != 0)
            {
                checkCuda(
                    cudaMemcpyAsync(
                        m_bsHostRemoveList, m_bsRemoveList,
                        sizeof(std::uint32_t) * removeCount,
                        cudaMemcpyDeviceToHost, m_bsStream),
                    "bs read removals");
                checkCuda(cudaStreamSynchronize(m_bsStream), "bs sync removals");
            }

            result.bondIndicesToRemove = m_bsHostRemoveList;
            result.removeCount = removeCount;
            result.overstressedBondCount = m_bsHostCounts[0];
            result.nodeOverstressed = m_bsHostNodeOverstressed;
            // Also deferred. Nothing reads a bond's stress unless its node
            // came back flagged, so on a tick with no overstress -- which is
            // every tick of a settled city -- these are never wanted.
            result.groupStressNormal = nullptr;
            result.groupStressShear = nullptr;
            // Left null on purpose: fetched only if someone asks.
            result.groupNormal = nullptr;
            result.groupCentroid = nullptr;
            m_telemetry.bondStressHostMs =
                std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - bsHostStart).count();
        }
        catch (const std::exception&)
        {
            return false;
        }
        return true;
    }

    /// Stage a host buffer through pinned memory and enqueue an ASYNC H2D copy
    /// on the solver stream.
    ///
    /// The topology uploads were blocking `cudaMemcpy` straight out of
    /// `std::vector`, i.e. out of PAGEABLE memory, which cannot DMA: the driver
    /// bounces it through its own staging buffer a chunk at a time and the
    /// transfer runs at roughly 0.9 GB/s instead of the ~12 GB/s the link is
    /// good for. At ~2.6 MB per fracture tick that is ~2.2 ms of the host
    /// simply waiting, and it was 64% of all reclaimable host time in the
    /// solve. Exactly the same trap, and the same fix, as the bond-stress CSR.
    ///
    /// Lifetime: the staged bytes must outlive the async copy. The arena is
    /// only ever rewound at the START of applyTopologyChange, and every solve
    /// waits on m_statusReady before the next one begins, so a copy enqueued in
    /// one topology change has necessarily completed before the arena is reused
    /// in the next. Growth is the one case that could overlap a live copy, so
    /// it synchronises first.
    /// How the topology arrays reach the device. Three real modes, because the
    /// obvious two were not enough to find the right answer.
    ///
    ///   pageable : the original -- blocking cudaMemcpy out of pageable
    ///              std::vector, ~0.9 GB/s, 2.6 ms/solve of pure host stall.
    ///   async    : pinned-backed arrays, copies enqueued on the solver stream.
    ///              Host plan time 3.87 -> 1.99 ms/solve, and MEASURABLY WORSE
    ///              overall: the 5.3 MB transfer then overlaps the CG kernels
    ///              and competes with them for memory bandwidth, which cost
    ///              8.25% of device solve time (p=0.021, 16 pairs). The host
    ///              time it saved just became host WAITING, because this phase
    ///              is device-bound.
    ///   sync     : pinned-backed arrays, blocking copy. Keeps the win that
    ///              mattered -- no staging memcpy, and pinned memory DMAs at
    ///              full rate instead of being bounced -- while finishing the
    ///              transfer before the solve is enqueued, so it never steals
    ///              bandwidth from the kernels.
    ///
    /// Default `async` (was `sync`). The measurement above that rejected async
    /// -- 8.25% of device solve time lost to the transfer competing with the CG
    /// kernels for bandwidth -- was taken against a CG loop that has since got
    /// roughly 4x cheaper (padded per-island reductions, device-side early
    /// exit). The competing kernels are now short enough that overlapping wins:
    /// re-measured on a 600-tick demolition at 298k bonds, the fracture tick
    /// goes 5.92 -> 5.47 ms wall, async faster in 3/3 paired runs. Host plan
    /// falls 3.59 -> 2.76 ms and roughly half of that does reappear as host
    /// waiting, exactly as the original note predicted -- but only half, so the
    /// trade is now net positive rather than net negative.
    ///
    /// This is worth re-checking whenever the device solve changes materially
    /// again; it has already flipped once.
    enum class TopoUploadMode { Pageable, Async, Sync };
    static TopoUploadMode topoUploadMode()
    {
        static const TopoUploadMode mode = [] {
            const char* raw = std::getenv("BLAST_TOPO_UPLOAD");
            if (raw != nullptr)
            {
                const std::string value(raw);
                if (value == "pageable") { return TopoUploadMode::Pageable; }
                if (value == "async")    { return TopoUploadMode::Async; }
                if (value == "sync")     { return TopoUploadMode::Sync; }
            }
            return TopoUploadMode::Async;
        }();
        return mode;
    }

    void stageUpload(void* dst, const void* src, std::size_t bytes, const char* name)
    {
        if (bytes == 0)
        {
            return;
        }
        if (topoUploadMode() == TopoUploadMode::Pageable)
        {
            // Faithfully reproduce the ORIGINAL cost structure for A/B: a
            // BLOCKING copy out of PAGEABLE memory. Copying via a pageable
            // bounce is necessary because the sources are pinned now, and a
            // blocking copy from pinned memory is a different (much faster)
            // thing than the one this change replaced. An arm that does not
            // reproduce the old behaviour measures nothing.
            if (m_pageableBounce.size() < bytes)
            {
                m_pageableBounce.resize(bytes);
            }
            std::memcpy(m_pageableBounce.data(), src, bytes);
            checkCuda(
                cudaMemcpy(dst, m_pageableBounce.data(), bytes, cudaMemcpyHostToDevice),
                name);
            return;
        }
        if (m_topoStagingUsed + bytes > m_topoStagingBytes)
        {
            checkCuda(cudaStreamSynchronize(m_stream), "sync before staging growth");
            const std::size_t want = (m_topoStagingUsed + bytes) * 2u;
            if (m_topoStaging)
            {
                cudaFreeHost(m_topoStaging);
                m_topoStaging = nullptr;
            }
            checkCuda(cudaMallocHost(&m_topoStaging, want), "alloc topology staging");
            ++m_statTopoGrowths;
            m_topoStagingBytes = want;
            m_topoStagingUsed = 0;
        }
        m_statTopoBytes += bytes;
        ++m_statTopoCopies;
        // If the source is ALREADY pinned, DMA straight out of it: no staging
        // copy at all. cudaHostGetDevicePointer succeeds only for memory the
        // driver has registered, which is exactly the PinnedVector case, so
        // this is a reliable test rather than a guess.
        void* devicePtr = nullptr;
        if (cudaHostGetDevicePointer(&devicePtr, const_cast<void*>(src), 0) == cudaSuccess)
        {
            // Async on the SOLVER stream in both modes. A blocking cudaMemcpy
            // would run on the legacy default stream, which implicitly
            // synchronises with every stream on the device -- so each of the
            // fifteen copies would wait for whatever the GPU already had
            // queued. Measured, that made "sync" mode 1.99 ms of mostly
            // waiting, against 0.08 ms for the same bytes enqueued async.
            //
            // Sync mode differs only in WHEN it waits: once, after all fifteen
            // are enqueued (see applyTopologyChange), so the transfer is
            // complete before the solve kernels are enqueued and cannot
            // compete with them for memory bandwidth.
            checkCuda(
                cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, m_stream),
                name);
            return;
        }
        cudaGetLastError();   // clear the probe's error state
        char* slot = static_cast<char*>(m_topoStaging) + m_topoStagingUsed;
        const auto cpStart = StatClock::now();
        std::memcpy(slot, src, bytes);
        m_statStageCopyMs += statMs(cpStart);
        m_topoStagingUsed += bytes;
        checkCuda(
            cudaMemcpyAsync(dst, slot, bytes, cudaMemcpyHostToDevice, m_stream),
            name);
    }

    template <typename T>
    void uploadBondStress(T* dst, const T* src, std::uint32_t count, const char* name)
    {
        checkCuda(cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice), name);
    }

    template <typename T>
    void uploadBondStressAsync(T* dst, const T* src, std::uint32_t count, const char* name)
    {
        checkCuda(
            cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice, m_bsStream),
            name);
    }

    /// Grow-only: a resync can add groups back, and reallocating on every
    /// shrink would churn for nothing.
    void ensureBondStressCapacity(const ExtStressGpuBondStressTopology& topology)
    {
        if (topology.groupCount > m_bsGroupCapacity)
        {
            freeBondStressGroupBuffers();
            const std::uint32_t n = topology.groupCount;
            allocateDevice(m_bsGroupBegin, n, "bs alloc groupBegin");
            allocateDevice(m_bsGroupSize, n, "bs alloc groupSize");
            allocateDevice(m_bsGroupStressNormal, n, "bs alloc sn");
            allocateDevice(m_bsGroupStressShear, n, "bs alloc ss");
            allocateDevice(m_bsGroupNormal, 3u * n, "bs alloc normal");
            allocateDevice(m_bsGroupCentroid, 3u * n, "bs alloc centroid");
            allocateDevice(m_bsGroupRemoveCount, n + 1u, "bs alloc removeCount");
            allocateDevice(m_bsGroupRemoveOffset, n + 1u, "bs alloc removeOffset");
            allocateHost(m_bsPinGroupBegin, n, "bs pin groupBegin");
            allocateHost(m_bsPinGroupSize, n, "bs pin groupSize");
            allocateHost(m_bsHostGroupStressNormal, n, "bs host sn");
            allocateHost(m_bsHostGroupStressShear, n, "bs host ss");
            allocateHost(m_bsHostGroupNormal, 3u * n, "bs host normal");
            allocateHost(m_bsHostGroupCentroid, 3u * n, "bs host centroid");
            m_bsGroupCapacity = n;
        }
        if (topology.memberSlotCount > m_bsSlotCapacity)
        {
            cudaFree(m_bsMemberBlastBond); m_bsMemberBlastBond = nullptr;
            cudaFree(m_bsRemoveFlag); m_bsRemoveFlag = nullptr;
            cudaFree(m_bsRemoveList); m_bsRemoveList = nullptr;
            cudaFreeHost(m_bsHostRemoveList); m_bsHostRemoveList = nullptr;
            const std::uint32_t n = topology.memberSlotCount;
            allocateDevice(m_bsMemberBlastBond, n, "bs alloc members");
            allocateDevice(m_bsRemoveFlag, n, "bs alloc removeFlag");
            allocateDevice(m_bsRemoveList, n, "bs alloc removeList");
            allocateHost(m_bsHostRemoveList, n, "bs host removeList");
            cudaFreeHost(m_bsPinMembers);
            allocateHost(m_bsPinMembers, n, "bs pin members");
            m_bsCsrResident = false;
            m_bsSlotCapacity = n;
        }
        if (topology.blastBondCount > m_bsBondCapacity)
        {
            cudaFree(m_bsBondNode0); m_bsBondNode0 = nullptr;
            cudaFree(m_bsBondNode1); m_bsBondNode1 = nullptr;
            cudaFree(m_bsBondMaterial); m_bsBondMaterial = nullptr;
            cudaFree(m_bsBondNormal); m_bsBondNormal = nullptr;
            cudaFree(m_bsBondCentroid); m_bsBondCentroid = nullptr;
            cudaFree(m_bsBondNodeDisp); m_bsBondNodeDisp = nullptr;
            cudaFree(m_bsHealth); m_bsHealth = nullptr;
            const std::uint32_t n = topology.blastBondCount;
            allocateDevice(m_bsBondNode0, n, "bs alloc node0");
            allocateDevice(m_bsBondNode1, n, "bs alloc node1");
            allocateDevice(m_bsBondMaterial, n, "bs alloc material");
            allocateDevice(m_bsBondNormal, 3u * n, "bs alloc bnormal");
            allocateDevice(m_bsBondCentroid, 3u * n, "bs alloc bcentroid");
            allocateDevice(m_bsBondNodeDisp, 3u * n, "bs alloc bdisp");
            allocateDevice(m_bsHealth, n, "bs alloc health");
            cudaFreeHost(m_bsPinHealth);
            allocateHost(m_bsPinHealth, n, "bs pin health");
            m_bsHealthResident = false;
            m_bsBondCapacity = n;
        }
        if (topology.graphNodeCount > m_bsNodeCapacity)
        {
            cudaFree(m_bsNodeOverstressed); m_bsNodeOverstressed = nullptr;
            cudaFreeHost(m_bsHostNodeOverstressed); m_bsHostNodeOverstressed = nullptr;
            allocateDevice(m_bsNodeOverstressed, topology.graphNodeCount, "bs alloc node mask");
            allocateHost(
                m_bsHostNodeOverstressed, topology.graphNodeCount, "bs host node mask");
            m_bsNodeCapacity = topology.graphNodeCount;
        }
        if (m_bsOverstressedCount == nullptr)
        {
            allocateDevice(m_bsOverstressedCount, 1, "bs alloc count");
            allocateHost(m_bsHostCounts, 2, "bs host counts");
        }
    }

    void freeBondStressGroupBuffers()
    {
        cudaFree(m_bsGroupBegin); m_bsGroupBegin = nullptr;
        cudaFree(m_bsGroupSize); m_bsGroupSize = nullptr;
        cudaFree(m_bsGroupStressNormal); m_bsGroupStressNormal = nullptr;
        cudaFree(m_bsGroupStressShear); m_bsGroupStressShear = nullptr;
        cudaFree(m_bsGroupNormal); m_bsGroupNormal = nullptr;
        cudaFree(m_bsGroupCentroid); m_bsGroupCentroid = nullptr;
        cudaFree(m_bsGroupRemoveCount); m_bsGroupRemoveCount = nullptr;
        cudaFree(m_bsGroupRemoveOffset); m_bsGroupRemoveOffset = nullptr;
        cudaFreeHost(m_bsPinGroupBegin); m_bsPinGroupBegin = nullptr;
        cudaFreeHost(m_bsPinGroupSize); m_bsPinGroupSize = nullptr;
        cudaFreeHost(m_bsHostGroupStressNormal); m_bsHostGroupStressNormal = nullptr;
        cudaFreeHost(m_bsHostGroupStressShear); m_bsHostGroupStressShear = nullptr;
        cudaFreeHost(m_bsHostGroupNormal); m_bsHostGroupNormal = nullptr;
        cudaFreeHost(m_bsHostGroupCentroid); m_bsHostGroupCentroid = nullptr;
    }

    void freeBondStress()
    {
        freeBondStressGroupBuffers();
        cudaFree(m_bsMemberBlastBond);
        cudaFree(m_bsRemoveFlag);
        cudaFree(m_bsRemoveList);
        cudaFree(m_bsBondNode0);
        cudaFree(m_bsBondNode1);
        cudaFree(m_bsBondMaterial);
        cudaFree(m_bsBondNormal);
        cudaFree(m_bsBondCentroid);
        cudaFree(m_bsBondNodeDisp);
        cudaFree(m_bsHealth);
        cudaFree(m_bsNodeOverstressed);
        cudaFree(m_bsOverstressedCount);
        cudaFree(m_bsScanScratch);
        cudaFree(m_bsMaterialLimits);
        cudaFreeHost(m_bsPinMembers);
        cudaFreeHost(m_bsPinHealth);
        cudaFreeHost(m_bsPinMaterials);
        cudaFreeHost(m_bsHostRemoveList);
        cudaFreeHost(m_bsHostNodeOverstressed);
        cudaFreeHost(m_bsHostCounts);
        if (m_bsStream) { cudaStreamDestroy(m_bsStream); m_bsStream = nullptr; }
    }

    bool hasPendingTopologyChange() const override
    {
        return m_topologyDirty || !m_pendingImpulseSwaps.empty();
    }

    bool flushImpulsePermutation() override
    {
        // Every other public entry point pushes PhysX's CUDA context before
        // touching the device; these did not, so their allocations, streams
        // and launches landed in whatever context the calling thread happened
        // to have -- the primary one -- while the impulses they read were
        // allocated under the guard in PhysX's. Cross-context by construction,
        // and the reason a tiny kernel could not be submitted without the GPU
        // switching contexts around it.
        ContextGuard context(m_cudaContext);
        if (m_pendingImpulseSwaps.empty())
        {
            return true;
        }
        try
        {
            for (const auto& swap : m_pendingImpulseSwaps)
            {
                if (swap.first != swap.second)
                {
                    checkCuda(
                        cudaMemcpy(
                            m_impulses + swap.first,
                            m_impulses + swap.second,
                            sizeof(AngLin),
                            cudaMemcpyDeviceToDevice),
                        "flush impulse swap");
                }
            }
            m_pendingImpulseSwaps.clear();
        }
        catch (const std::exception&)
        {
            return false;
        }
        return true;
    }

    void resetWarmStart() override
    {
        ContextGuard context(m_cudaContext);
        checkCuda(cudaMemset(m_impulses, 0, sizeof(AngLin) * m_bondCount), "reset warm start");
        m_hasWarmStart = false;
        // The impulses these flags certified are gone, so nothing may be
        // skipped against them.
        invalidateSettledBaseline();
    }

    std::uint32_t nodeCount() const override
    {
        return m_nodeCount;
    }

    std::uint32_t bondCount() const override
    {
        return m_bondCount;
    }

    const ExtStressGpuTelemetry& telemetry() const override
    {
        return m_telemetry;
    }

private:
    /// Read back per-island residual^2 and tolerance^2 and record how far past
    /// tolerance the solve stopped. Diagnostic only; gated on graph stats.
    void accumulateResidualStats()
    {
        const std::size_t n = m_islandCount;
        m_statResGrad.resize(n);
        m_statResDelta.resize(n);
        m_statResActive.resize(n);
        if (cudaMemcpy(m_statResGrad.data(), m_gradientSquared,
                       sizeof(float) * n, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(m_statResDelta.data(), m_deltaSquared,
                       sizeof(float) * n, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(m_statResActive.data(), m_islandActive,
                       sizeof(std::uint32_t) * n, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaGetLastError();
            return;
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            // The population is "islands this solve SEEDED", i.e. not skipped.
            //
            // Two wrong filters were tried first and both are instructive.
            // Filtering on nothing counts skipped islands, whose deltaSquared
            // and gradientSquared still hold values from whenever they last
            // ran -- that measures history, and it flatters whichever arm
            // skips more (the warm-started one). Filtering on islandActive==1
            // is circular in the other direction: checkConvergencePerIsland
            // CLEARS islandActive when an island converges, so the survivors
            // are by definition the ones that failed, and the answer is always
            // "100% over tolerance".
            //
            // islandSkip is the honest marker: it says what this solve was
            // asked to do, before it knew how it would go.
            if (m_hostIslandSkip != nullptr && m_hostIslandSkip[i] != 0u)
            {
                continue;
            }
            const float d = m_statResDelta[i];
            const float g = m_statResGrad[i];
            if (!(d > 0.0f) || !(g >= 0.0f))
            {
                continue;   // island carries no target: not part of this solve
            }
            const double ratio = std::sqrt(double(g) / double(d));
            m_statResSum += ratio;
            ++m_statResCount;
            if (ratio > m_statResMax) { m_statResMax = ratio; }
            if (ratio > 1.0) { ++m_statResOverTol; }
        }
    }

    bool enqueueSolve(
        const ExtStressGpuImpulse* nodeVelocities,
        const ExtStressGpuSolveParams& params)
    {
        // Latch the bending policy the host resolved, so every kernel this
        // solve launches uses the same one the CPU walk does.
        m_bendGainMax = params.bendGainMax;

        if (!nodeVelocities || params.maxIterations == 0)
        {
            return false;
        }
        // Sampled HERE, not in launchSolve: that runs inside CUDA graph
        // capture, where a synchronous memcpy is illegal and returns 900
        // (operation not permitted while capturing) -- which corrupts the
        // capture rather than just failing the read.
        if (islandTraceEnabled())
        {
            checkCuda(cudaStreamSynchronize(m_stream), "island trace pre-sync");
            m_dbgSumBefore = debugImpulseMagnitude();
            m_dbgWarm = params.warmStart && m_hasWarmStart;
        }
        if (m_topologyDirty)
        {
            const auto topoStart = StatClock::now();
            applyTopologyChange();
            m_statTopoMs += statMs(topoStart);
            ++m_statTopoCalls;
        }
        if (m_bondCount == 0)
        {
            return false;   // nothing bonded left; the caller falls back to its CPU path
        }

        m_telemetry = {};
        m_telemetry.islandCount = m_islandCount;
        m_solveWasNoOp = false;

        // The settled baseline certifies "these impulses are what THIS solve
        // would produce". Change the solve -- tolerance, iteration budget,
        // damage, warm start -- and that stops being true: an island left at
        // its own tolerance by a loose solve would be refined by a tighter
        // one, so freezing it would silently pin the scene to the loosest
        // settings it ever ran with. graphMatches is the same test that
        // decides whether the captured CUDA graph is still valid, which is not
        // a coincidence: both ask whether this is the same solve as last time.
        if (!graphMatches(params, params.warmStart && m_hasWarmStart))
        {
            invalidateSettledBaseline();
        }

        // Skipping is only coherent while warm-starting. Without a warm start
        // initializeSolve ZEROES every bond it touches, so a settled island
        // that kept its impulses and a solved one that started from zero would
        // be two different physics in one graph. The kernels apply the same
        // condition (launchSolve), so the two sides cannot drift apart.
        const bool skipping =
            params.skipSettledIslands && params.warmStart && m_hasWarmStart;

        // m_hostInput is both the staging buffer and the settled baseline: it
        // is a byte-for-byte record of what the device already holds, so
        // "differs from the baseline" and "the device's copy is stale" are the
        // same question and get asked once.
        if (skipping)
        {
            const auto skipStart = StatClock::now();
            planSettledSkip(nodeVelocities);
            m_statPlanSkipMs += statMs(skipStart);
        }
        else
        {
            std::memcpy(
                m_hostInput,
                nodeVelocities,
                sizeof(ExtStressGpuImpulse) * m_nodeCount);
            std::fill(m_hostIslandSkip, m_hostIslandSkip + m_islandCount, 0u);
            m_changedBonds.resize(m_bondCount);
            for (std::uint32_t i = 0; i < m_bondCount; ++i)
            {
                m_changedBonds[i] = i;
            }
            m_changedNodes.clear();
        }

        if (skipping
            && m_changedNodes.empty()
            && m_telemetry.islandsSkipped == m_islandCount)
        {
            // Nothing anywhere in the graph moved. Even the empty graph replay
            // costs a launch per kernel per iteration, so the whole frame is
            // elided rather than run over an all-skip mask.
            m_solveWasNoOp = true;
            m_changedBonds.clear();
            m_telemetry.converged = true;
            return true;
        }

        checkCuda(cudaEventRecord(m_uploadStart, m_stream), "record upload start");
        if (skipping)
        {
            checkCuda(
                cudaMemcpyAsync(
                    m_islandSkip,
                    m_hostIslandSkip,
                    sizeof(std::uint32_t) * m_islandCount,
                    cudaMemcpyHostToDevice,
                    m_stream),
                "upload island skip mask");
            m_telemetry.hostToDeviceBytes +=
                sizeof(std::uint32_t) * static_cast<std::uint64_t>(m_islandCount);
            uploadChangedVelocities();
        }
        else
        {
            checkCuda(
                cudaMemcpyAsync(
                    m_input,
                    m_hostInput,
                    sizeof(ExtStressGpuImpulse) * m_nodeCount,
                    cudaMemcpyHostToDevice,
                    m_stream),
                "upload stress inputs");
            m_telemetry.hostToDeviceBytes +=
                sizeof(ExtStressGpuImpulse) * static_cast<std::uint64_t>(m_nodeCount);
        }
        checkCuda(cudaEventRecord(m_uploadStop, m_stream), "record upload stop");

        // After the mask upload (compaction reads it), before the graph
        // launch (the kernels read the lists).
        const auto refreshStart = StatClock::now();
        refreshActiveLists(skipping);
        m_statRefreshMs += statMs(refreshStart);

        checkCuda(cudaEventRecord(m_solveStart, m_stream), "record solve start");
        executeSolve(params);
        checkCuda(cudaEventRecord(m_solveStop, m_stream), "record solve stop");
        checkCuda(
            cudaMemcpyAsync(
                m_hostStatus,
                m_status,
                sizeof(SolveStatus),
                cudaMemcpyDeviceToHost,
                m_stream),
            "read stress status");
        checkCuda(
            cudaMemcpyAsync(
                m_hostIslandConvergedPinned,
                m_islandConverged,
                sizeof(std::uint32_t) * m_islandCount,
                cudaMemcpyDeviceToHost,
                m_stream),
            "read per-island convergence");
        // Bond health is an input to the solve -- but only at the boundary,
        // since the kernels read nothing from it but `health <= 0`. Partial
        // damage therefore cannot change an island's answer, and a BREAK
        // always can. Four bytes per solve buys the difference.
        checkCuda(
            cudaMemcpyAsync(
                m_hostBrokenCount,
                m_brokenCount,
                sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost,
                m_stream),
            "read broken bond count");
        m_telemetry.deviceToHostBytes +=
            sizeof(SolveStatus)
            + sizeof(std::uint32_t) * static_cast<std::uint64_t>(m_islandCount);
        checkCuda(cudaEventRecord(m_statusReady, m_stream), "record status ready");
        return true;
    }

    /**
     * Rebuild the compacted active lists when the skip set they encode has
     * changed. One kernel pass per array plus an 8-byte synchronous readback
     * of the counts -- the sync is what lets executeSolve know, on the host,
     * whether the baked launch capacity still covers the lists. In a settled
     * scene the skip set is stable and this runs never; during demolition it
     * runs at most once per tick.
     */
    void refreshActiveLists(bool skipping)
    {
        const std::uint32_t* mask = skipping ? m_islandSkip : nullptr;
        const bool maskChanged = skipping
            && (m_prevIslandSkip.size() != m_islandCount
                || std::memcmp(
                       m_prevIslandSkip.data(),
                       m_hostIslandSkip,
                       sizeof(std::uint32_t) * m_islandCount)
                       != 0);
        if (!m_activeListsDirty && m_prevListsSkipping == skipping && !maskChanged)
        {
            ++m_statListSkips;
            return;
        }
        ++m_statListRefreshes;
        // Flag, then order-preserving select. The flag buffer is shared by the
        // two selects; the stream orders bond-select before node-flag, so the
        // reuse cannot race. DeviceSelect writes each count straight into its
        // m_activeCounts slot, which is why the memset the atomic append
        // needed is gone.
        cub::CountingInputIterator<std::uint32_t> identity(0u);
        flagActiveBonds<<<
            (m_bondCount + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(m_bondIsland, mask, m_bondCount, m_activeFlags);
        checkCuda(
            cub::DeviceSelect::Flagged(
                m_selectScratch,
                m_selectScratchBytes,
                identity,
                m_activeFlags,
                m_activeBonds,
                m_activeCounts,
                static_cast<int>(m_bondCount),
                m_stream),
            "select active bonds");
        flagActiveNodes<<<
            (m_nodeCount + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(m_nodeIsland, mask, m_nodeCount, m_activeFlags);
        checkCuda(
            cub::DeviceSelect::Flagged(
                m_selectScratch,
                m_selectScratchBytes,
                identity,
                m_activeFlags,
                m_activeNodes,
                m_activeCounts + 1,
                static_cast<int>(m_nodeCount),
                m_stream),
            "select active nodes");
        checkCuda(
            cudaMemcpyAsync(
                m_hostActiveCounts,
                m_activeCounts,
                sizeof(std::uint32_t) * 2,
                cudaMemcpyDeviceToHost,
                m_stream),
            "read active counts");
        if (!fullLaunchCapacity())
        {
            // Priced separately: this is the host BLOCKED, not the host
            // working, and only the latter is reclaimable by faster host code.
            const auto syncStart = std::chrono::steady_clock::now();
            checkCuda(cudaStreamSynchronize(m_stream), "sync active counts");
            const double midSync =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - syncStart)
                    .count();
            m_activeListSyncMilliseconds += static_cast<float>(midSync);
            m_statMidSyncMs += midSync;
            m_statLastMidSync = midSync;
        }
        if (!fullLaunchCapacity())
        {
            m_activeBondCount = m_hostActiveCounts[0];
            m_activeNodeCount = m_hostActiveCounts[1];
        }
        if (skipping)
        {
            m_prevIslandSkip.assign(m_hostIslandSkip, m_hostIslandSkip + m_islandCount);
        }
        else
        {
            m_prevIslandSkip.clear();
        }
        m_prevListsSkipping = skipping;
        m_activeListsDirty = false;
    }

    /**
     * Decide which islands are settled, and refresh the baseline in the same
     * pass.
     *
     * The predicate is the CPU solver's, unchanged: an island is settled when
     * every one of its DYNAMIC nodes carries a velocity identical to the one
     * it was last solved with, AND that solve reached tolerance. Static nodes
     * are excluded because they are boundaries whose velocity is multiplied by
     * a zero inverse inertia -- they cannot change an island's answer, and
     * including them would make a moving kinematic support re-solve the whole
     * city for nothing.
     */
    void planSettledSkip(const ExtStressGpuImpulse* nodeVelocities)
    {
        const bool haveBaseline = m_settledBaselineValid;
        m_islandDirty.assign(m_islandCount, haveBaseline ? 0u : 1u);
        m_changedNodes.clear();
        for (std::uint32_t i = 0; i < m_nodeCount; ++i)
        {
            if (haveBaseline && velocityBitsEqual(nodeVelocities[i], m_hostInput[i]))
            {
                continue;
            }
            m_hostInput[i] = nodeVelocities[i];
            m_changedNodes.push_back(i);
            const std::uint32_t id = m_hostNodeIsland[i];
            if (id != kNoIsland)
            {
                m_islandDirty[id] = 1;
            }
        }

        // Nodes whose bonds broke or were removed since the last solve: their
        // islands must re-solve regardless of the velocity compare, because
        // frozen impulses no longer describe the changed topology.
        for (const std::uint32_t node : m_forceDirtyNodes)
        {
            if (node < m_nodeCount)
            {
                const std::uint32_t id = m_hostNodeIsland[node];
                if (id != kNoIsland)
                {
                    m_islandDirty[id] = 1;
                }
            }
        }
        m_forceDirtyNodes.clear();

        m_changedBonds.clear();
        std::uint32_t skipped = 0;
        for (std::uint32_t k = 0; k < m_islandCount; ++k)
        {
            // An island may be skipped only once it has CONVERGED. Skipping a
            // stable-but-unconverged island freezes it at whatever partial
            // answer the iteration budget happened to reach, permanently.
            //
            // That was the depth-truncation defect. Conjugate gradient
            // propagates load roughly one graph hop per iteration, so a
            // structure deeper than the budget cannot finish in one solve. It
            // would then go quiet -- inputs unchanged -- and be skipped for
            // ever, pinned at exactly `iterations` panels' worth of load.
            // Measured on a column: footing stress plateaued at exactly
            // 25 panels at 25 iterations and 50 at 50, bit-identical from depth
            // 48 to 128. A 10-floor building is far deeper than that, so the
            // city read as unloaded and stood up under the GPU solver while the
            // CPU solver -- whose skip has always required convergence --
            // collapsed it. One cause, both symptoms.
            //
            // `skipStableUnconverged` was meant to stop unconverged islands
            // re-solving for ever at rest. It cannot do that safely: they never
            // converge BECAUSE they are skipped. The way out is the documented
            // one -- pursue convergence (unconvergedExtraUpdates), converge in
            // the first second, and skip at ~zero cost thereafter.
            const bool skip =
                haveBaseline && !m_islandDirty[k] && m_hostIslandConverged[k] != 0u;
            m_hostIslandSkip[k] = skip ? 1u : 0u;
            if (skip)
            {
                ++skipped;
                continue;
            }
            if (m_bondsByIslandValid)
            {
                for (std::uint32_t t = m_islandBondBegin[k]; t < m_islandBondBegin[k + 1]; ++t)
                {
                    m_changedBonds.push_back(m_bondsByIsland[t]);
                }
            }
        }
        if (!m_bondsByIslandValid)
        {
            // Bonds moved under the island->bond index since it was built, so
            // enumerate from the labels instead, which removeBond keeps
            // current. One pass over bonds rather than a gather per island: no
            // worse than the gather in exactly the case this runs -- a tick on
            // which something broke, where little or nothing is skippable
            // anyway -- and it lets the repartition be deferred.
            m_changedBonds.clear();
            for (std::uint32_t b = 0; b < m_bondCount; ++b)
            {
                const std::uint32_t island = m_hostBondIsland[b];
                if (island != kNoIsland && m_hostIslandSkip[island] == 0u)
                {
                    m_changedBonds.push_back(b);
                }
            }
        }
        m_telemetry.islandsSkipped = skipped;
        if (s_debug)
        {
            std::uint32_t clean = 0;
            std::uint32_t converged = 0;
            for (std::uint32_t k = 0; k < m_islandCount; ++k)
            {
                clean += m_islandDirty[k] ? 0u : 1u;
                converged += m_hostIslandConverged[k] ? 1u : 0u;
            }
            if ((m_debugSolves++ % 60u) == 0u)
            {
                std::fprintf(
                    stderr,
                    "[blast-gpu-skip] nodes %u changed %u | islands %u clean %u converged %u "
                    "skipped %u | bonds %u changed %zu | baseline %d\n",
                    m_nodeCount,
                    static_cast<std::uint32_t>(m_changedNodes.size()),
                    m_islandCount,
                    clean,
                    converged,
                    skipped,
                    m_bondCount,
                    m_changedBonds.size(),
                    haveBaseline ? 1 : 0);
            }
        }
        if (!haveBaseline)
        {
            // Nothing was skipped, and bonds in no island have never been
            // written -- but the caller's mirror has never been written
            // either, so hand it the whole array once.
            m_changedBonds.resize(m_bondCount);
            for (std::uint32_t i = 0; i < m_bondCount; ++i)
            {
                m_changedBonds[i] = i;
            }
        }
    }

    /**
     * Bit-exact velocity comparison, matching stress.cpp's angLin6Equal.
     *
     * Float `==` rather than std::memcmp, because it is the CPU path's own
     * test and both of its disagreements with a byte compare fall the safe
     * way. -0.0f == +0.0f declares a node unchanged, and it is: the right-hand
     * side differs only in the sign of a zero, which no later operation can
     * turn into a different magnitude. NaN != NaN declares it changed, so a
     * poisoned input can never be frozen in place by the skip.
     */
    static bool velocityBitsEqual(
        const ExtStressGpuImpulse& a,
        const ExtStressGpuImpulse& b)
    {
        return a.angular.x == b.angular.x && a.angular.y == b.angular.y
            && a.angular.z == b.angular.z && a.linear.x == b.linear.x
            && a.linear.y == b.linear.y && a.linear.z == b.linear.z;
    }

    /// Push only the node velocities that moved. Falls back to the straight
    /// copy of the whole array once enough of them have: the scatter costs an
    /// index and a kernel launch per node, so below about half it is cheaper
    /// and above it is not. Both write exactly the same device state.
    void uploadChangedVelocities()
    {
        const std::uint32_t count = static_cast<std::uint32_t>(m_changedNodes.size());
        if (count == 0)
        {
            return;     // the device already holds every value this solve needs
        }
        if (static_cast<std::uint64_t>(count) * 2u >= m_nodeCount)
        {
            checkCuda(
                cudaMemcpyAsync(
                    m_input,
                    m_hostInput,
                    sizeof(ExtStressGpuImpulse) * m_nodeCount,
                    cudaMemcpyHostToDevice,
                    m_stream),
                "upload stress inputs");
            m_telemetry.hostToDeviceBytes +=
                sizeof(ExtStressGpuImpulse) * static_cast<std::uint64_t>(m_nodeCount);
            return;
        }
        for (std::uint32_t j = 0; j < count; ++j)
        {
            const std::uint32_t node = m_changedNodes[j];
            m_hostScatterIndices[j] = node;
            m_hostScatterValues[j] = m_hostInput[node];
        }
        checkCuda(
            cudaMemcpyAsync(
                m_scatterIndices,
                m_hostScatterIndices,
                sizeof(std::uint32_t) * count,
                cudaMemcpyHostToDevice,
                m_stream),
            "upload changed node indices");
        checkCuda(
            cudaMemcpyAsync(
                m_scatterValues,
                m_hostScatterValues,
                sizeof(ExtStressGpuImpulse) * count,
                cudaMemcpyHostToDevice,
                m_stream),
            "upload changed node velocities");
        scatterVelocities<<<
            (count + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(m_input, m_scatterIndices, m_scatterValues, count);
        m_telemetry.hostToDeviceBytes +=
            (sizeof(std::uint32_t) + sizeof(ExtStressGpuImpulse))
            * static_cast<std::uint64_t>(count);
    }

    void finishSolve()
    {
        if (m_kernelProfile.active)
        {
            // Elapsed time is only readable once the stream has caught up,
            // and finishSolve is where the caller has already synchronized.
            m_kernelProfile.harvest();
        }
        checkCuda(cudaGetLastError(), "execute stress kernels");
        checkCuda(
            cudaEventElapsedTime(
                &m_telemetry.uploadMilliseconds,
                m_uploadStart,
                m_uploadStop),
            "measure input upload");
        checkCuda(
            cudaEventElapsedTime(
                &m_telemetry.solveMilliseconds,
                m_solveStart,
                m_solveStop),
            "measure stress solve");
        m_telemetry.iterations = m_hostStatus->iterations;
        m_telemetry.converged = m_hostStatus->converged != 0;
        if (islandTraceEnabled())
        {
            const double after = debugImpulseMagnitude();
            m_dbgActive.resize(m_islandCount);
            m_dbgConverged.resize(m_islandCount);
            cudaMemcpy(m_dbgActive.data(), m_islandActive,
                       sizeof(std::uint32_t) * m_islandCount, cudaMemcpyDeviceToHost);
            cudaMemcpy(m_dbgConverged.data(), m_islandConverged,
                       sizeof(std::uint32_t) * m_islandCount, cudaMemcpyDeviceToHost);
            cudaGetLastError();
            std::uint32_t active = 0, conv = 0;
            for (std::uint32_t i = 0; i < m_islandCount; ++i)
            {
                active += (m_dbgActive[i] != 0u) ? 1u : 0u;
                conv += (m_dbgConverged[i] != 0u) ? 1u : 0u;
            }
            static std::uint32_t dbgTick = 0;
            if (dbgTick < 60)
            {
                std::fprintf(stderr,
                             "[island] tick=%-3u warm=%d iters=%-3u conv=%d "
                             "|x|before=%.6f |x|after=%.6f delta=%.6f "
                             "islands=%u active_end=%u converged_end=%u\n",
                             dbgTick, int(m_dbgWarm), unsigned(m_hostStatus->iterations),
                             int(m_hostStatus->converged != 0),
                             m_dbgSumBefore, after, after - m_dbgSumBefore,
                             unsigned(m_islandCount), unsigned(active), unsigned(conv));
            }
            ++dbgTick;
        }
        m_hasWarmStart = true;
        // Carry each solved island's convergence forward as next frame's
        // permission to skip. A skipped island reports nothing, and must not:
        // the flag it still holds is the one that certified the impulses it
        // still holds.
        for (std::uint32_t k = 0; k < m_islandCount; ++k)
        {
            if (m_hostIslandSkip[k] == 0u)
            {
                m_hostIslandConverged[k] = m_hostIslandConvergedPinned[k];
            }
        }
        m_settledBaselineValid = true;
        if (*m_hostBrokenCount != 0u && !wholeResetOnTopology())
        {
            // Per-island invalidation: only the islands whose own bonds broke
            // lose their baseline. The removal path force-dirties the same
            // nodes; this covers damage-driven breaks the adapter has not yet
            // converted to removeBond calls, closing the window the
            // equivalence harness caught (a settled island taking damage from
            // frozen stress and never re-solving after its own bond failed --
            // 322 broken against 287).
            const std::uint32_t count =
                std::min(*m_hostBrokenCount, m_bondCount);
            std::vector<std::uint32_t> broken(count);
            if (count > 0)
            {
                checkCuda(
                    cudaMemcpy(
                        broken.data(),
                        m_brokenBonds,
                        sizeof(std::uint32_t) * count,
                        cudaMemcpyDeviceToHost),
                    "read broken bonds for invalidation");
            }
            for (const std::uint32_t bond : broken)
            {
                if (bond < m_bondCount)
                {
                    m_forceDirtyNodes.push_back(m_hostNode0[bond]);
                    m_forceDirtyNodes.push_back(m_hostNode1[bond]);
                }
            }
        }
        else if (*m_hostBrokenCount != 0u)
        {
            // A bond crossed zero health somewhere, so at least one island's
            // effective topology changed and its frozen impulses no longer
            // describe it. Drop the whole baseline rather than work out which:
            // it is the CPU solver's own rule (stress.cpp removeBond clears
            // m_skipValid), and in a settled scene -- the only place skipping
            // pays -- nothing breaks, so it costs nothing there.
            //
            // Found by the equivalence test, not by inspection: with
            // applyDamage on, skipping produced 322 broken bonds against 287.
            // A settled island keeps taking damage from its frozen stress, and
            // without this it was never re-solved after one of its own bonds
            // failed.
            invalidateSettledBaseline();
        }
    }

    // Allocator-generic: the topology arrays are PinnedVector, not
    // std::vector, so a signature naming the default allocator no longer
    // matches them.
    template <typename T, typename Alloc>
    static void swapWithLast(std::vector<T, Alloc>& values, std::uint32_t index, std::uint32_t last)
    {
        values[index] = values[last];
        values.resize(last);
    }

    /**
     * Fold the removals since the last solve into device state.
     *
     * Deliberately the same observable state a freshly constructed solver
     * would have: re-uploaded topology, a repartition from the surviving
     * bonds, and a cold warm start. The old code got there by destroying the
     * solver; this gets there without releasing and re-cudaMalloc-ing twenty
     * buffers, and -- the reason it matters -- a tick with NO removals now
     * changes nothing at all, so the warm-start impulses and the settled
     * baseline survive into it.
     */
    void applyTopologyChange()
    {
        m_topologyDirty = false;
        m_topoStagingUsed = 0;
        // The host topology arrays are pinned and DMA'd from directly, and the
        // rebuilds below RESIZE them -- which frees that pinned memory. If a
        // copy enqueued last tick were still in flight, that is a use-after-
        // free the driver would read straight through.
        //
        // In practice every solve waits on m_statusReady first, so the copies
        // are long done. "In practice" is not good enough for a dangling DMA:
        // the no-op solve path returns early without waiting, so the guarantee
        // has a hole in it. Waiting on the upload event closes it, and costs
        // nothing when the work has already completed.
        if (m_topoUploadsPending)
        {
            checkCuda(cudaEventSynchronize(m_topoUploadDone), "wait topology uploads");
            m_topoUploadsPending = false;
        }
        // The island partition is about to be remapped; the compacted lists
        // index into the OLD partition and must be rebuilt before next solve.
        m_activeListsDirty = true;
        m_jacobiBuilt = false;   // topology moved: the diagonal blocks are stale
        auto tstep = StatClock::now();
        // Repartitioning is O(nodes + bonds) of pointer-chasing union-find and
        // measured 2.2 ms per fracture tick at 298k bonds -- the single largest
        // cost in the whole tick, device work included. It is also usually
        // unnecessary, because REMOVING A BOND CAN ONLY SPLIT AN ISLAND, NEVER
        // MERGE ONE, and solving two disconnected components as one island is
        // exact: the operator is block-diagonal across them, so the CG iterates
        // are identical. Only the convergence test and the skip granularity are
        // shared, which costs at most a few extra iterations for whichever
        // component converges first.
        //
        // So carry the stale, coarser partition and rebuild on a budget. The
        // labels stay self-consistent because removeBond moves a bond's island
        // label with it.
        // CSR first: the split test walks it, and it has to describe the graph
        // AFTER this tick's removals. Usually it already does, because
        // removeBond patched it in place -- rebuild only once the tombstones
        // have accumulated enough to be worth compacting away.
        const bool csrPatched = m_csrValid && csrPatchAcceptable();
        if (!csrPatched)
        {
            buildNodeBondCsr();
        }
        m_statTopoCsrMs += statMs(tstep); tstep = StatClock::now();
        const bool repartition = shouldRepartition();
        if (repartition)
        {
            computeIslands();
            m_islandCountAtRebuild = m_islandCount;
            m_removalsSinceRepartition = 0;
            // computeIslands rewrote every label, so any deltas queued by a
            // local relabel are stale -- the full upload below carries them.
            m_changedNodeIslandSlots.clear();
            m_changedBondSlots.clear();
        }
        m_splitChecks.clear();
        m_statTopoIslandsMs += statMs(tstep); tstep = StatClock::now();
        if (repartition)
        {
            groupBondsByIsland();
            m_bondsByIslandValid = true;
        }
        m_statTopoGroupMs += statMs(tstep); tstep = StatClock::now();
        // The bond arrays change in a handful of slots per tick; the island
        // arrays only change at all when the partition was rebuilt. Fall back
        // to the full upload whenever the sparse path declines.
        const bool sparse =
            !repartition && uploadNodeIslandDelta() && uploadTopologyDelta();
        if (!sparse)
        {
            uploadTopology();
            uploadIslands();
        }
        else if (!m_inertiaUploaded)
        {
            // Per-node inertia is fixed by prepare(); it was being re-sent on
            // every fracture tick for nothing.
            stageUpload(m_inertia, m_hostInertia.data(),
                        sizeof(Inertia) * m_nodeCount, "upload inertia");
            m_inertiaUploaded = true;
        }
        m_changedBondSlots.clear();
        m_changedNodeIslandSlots.clear();
        if (csrPatched && uploadCsrDelta())
        {
            // nodeBondBegin is untouched by a patch: runs keep their length.
        }
        else
        {
            uploadNodeBondCsr();
        }
        m_changedRefSlots.clear();
        checkCuda(cudaEventRecord(m_topoUploadDone, m_stream), "record topology uploads");
        m_topoUploadsPending = true;
        if (topoUploadMode() == TopoUploadMode::Sync)
        {
            // Drain the transfers here, ONCE, so they finish before the solve
            // is enqueued. Letting them ride alongside the CG kernels cost
            // 8.25% of device solve time (p=0.021, 16 pairs) -- the host time
            // it saved merely became host waiting, because this phase is
            // device-bound.
            checkCuda(cudaStreamSynchronize(m_stream), "drain topology uploads");
            m_topoUploadsPending = false;
        }
        m_statTopoUploadMs += statMs(tstep);
        // Grid sizes and the per-island memset lengths are baked into the
        // captured graph, and both just changed -- but they are node
        // PARAMETERS, not structure, so the exec can be patched rather than
        // rebuilt. Destroying it here is what made every fracture tick pay a
        // full re-instantiation. executeSolve now decides, and still falls back
        // to destroy + instantiate if the patch will not take.
        m_graphParamsDirty = true;
        if (!graphUpdateEnabled())
        {
            if (m_graphExec)
            {
                checkCuda(cudaGraphExecDestroy(m_graphExec), "destroy solver graph exec");
                m_graphExec = nullptr;
            }
            if (m_graph)
            {
                checkCuda(cudaGraphDestroy(m_graph), "destroy solver graph");
                m_graph = nullptr;
            }
        }
        if (wholeResetOnTopology())
        {
            const auto memsetStart = StatClock::now();
            // Async, on the solver stream, NOT the synchronous form. Plain
            // cudaMemset runs on the legacy default stream, which implicitly
            // synchronises with every other stream on the device -- so it
            // would have blocked on the topology uploads enqueued just above
            // and handed back exactly the stall they were made async to avoid.
            checkCuda(
                cudaMemsetAsync(
                    m_impulses, 0, sizeof(AngLin) * m_bondCount, m_stream),
                "reset warm start after topology change");
            m_statTopoMemsetMs += statMs(memsetStart);
            m_hasWarmStart = false;
            m_settledBaselineValid = false;
            m_pendingImpulseSwaps.clear();
        }
        else
        {
            // Incremental: a removal only permutes the bond arrays. Every
            // OTHER island is a disconnected component whose inputs and
            // topology are untouched -- its warm-start impulses and settled
            // baseline stay exactly valid, so a fracture tick no longer
            // cold-starts the whole city (the memset above did, every time a
            // bond broke anywhere). The device impulse array replays the same
            // swap-with-last permutation the host arrays already applied; the
            // affected islands re-solve via the force-dirty nodes recorded in
            // removeBond.
            for (const auto& swap : m_pendingImpulseSwaps)
            {
                if (swap.first != swap.second)
                {
                    checkCuda(
                        cudaMemcpyAsync(
                            m_impulses + swap.first,
                            m_impulses + swap.second,
                            sizeof(AngLin),
                            cudaMemcpyDeviceToDevice,
                            m_stream),
                        "replay impulse swap");
                }
            }
            m_pendingImpulseSwaps.clear();
        }
        // Converged flags are keyed by island id, so they cannot survive a
        // renumbering -- but they CAN survive a tick that did not renumber.
        // That distinction matters a lot: wiping unconditionally meant one bond
        // breaking anywhere disabled the settled skip for the entire scene on
        // the next tick, so a long demolition never got to skip anything.
        // The islands actually touched by this tick's removals are dirtied
        // separately, through m_forceDirtyNodes.
        if (repartition)
        {
            m_hostIslandConverged.assign(m_islandCapacity, 0u);
            std::fill(m_hostIslandSkip, m_hostIslandSkip + m_islandCapacity, 0u);
            // Async on the solver stream: the plain form runs on the legacy
            // default stream and implicitly synchronises every stream on the
            // device, including the topology uploads enqueued just above.
            checkCuda(
                cudaMemsetAsync(
                    m_islandConverged, 0,
                    sizeof(std::uint32_t) * m_islandCapacity, m_stream),
                "clear island converged flags");
        }
    }

    /// Is the patched CSR still worth using, or have the tombstones piled up?
    ///
    /// Dead entries cost the matvec a load and a compare each; compacting them
    /// away costs a full rebuild. 12.5% is where the rebuild starts paying.
    bool csrPatchAcceptable() const
    {
        const std::size_t total = m_hostNodeBondRef.size();
        return total != 0 && m_deadRefCount * 8u < total;
    }

    /// Upload just the CSR slots a patch touched.
    bool uploadCsrDelta()
    {
        const std::uint32_t count =
            static_cast<std::uint32_t>(m_changedRefSlots.size());
        if (count == 0)
        {
            return true;
        }
        if (count * 8u >= m_hostNodeBondRef.size() || !deltaUploadEnabled())
        {
            return false;
        }
        if (count > m_refDeltaCapacity)
        {
            const std::uint32_t want =
                std::max(count, m_refDeltaCapacity ? m_refDeltaCapacity * 2u : 2048u);
            cudaFree(m_devRefSlots);
            cudaFree(m_devRefValues);
            if (cudaMalloc(&m_devRefSlots, sizeof(std::uint32_t) * want) != cudaSuccess
                || cudaMalloc(&m_devRefValues, sizeof(std::uint32_t) * want) != cudaSuccess)
            {
                cudaGetLastError();
                m_refDeltaCapacity = 0;
                return false;
            }
            m_refDeltaCapacity = want;
        }
        m_refSlots.resize(count);
        m_refValues.resize(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const std::uint32_t pos = m_changedRefSlots[i];
            m_refSlots[i] = pos;
            m_refValues[i] = m_hostNodeBondRef[pos];
        }
        stageUpload(m_devRefSlots, m_refSlots.data(),
                    sizeof(std::uint32_t) * count, "upload csr delta slots");
        stageUpload(m_devRefValues, m_refValues.data(),
                    sizeof(std::uint32_t) * count, "upload csr delta values");
        scatterNodeBondRefs<<<(count + kBlockSize - 1) / kBlockSize, kBlockSize, 0, m_stream>>>(
            m_devRefSlots, m_devRefValues, count, m_nodeBondRef);
        return true;
    }

    /// Upload only the node island labels a local split rewrote.
    bool uploadNodeIslandDelta()
    {
        const std::uint32_t count =
            static_cast<std::uint32_t>(m_changedNodeIslandSlots.size());
        if (count == 0)
        {
            return true;
        }
        if (count * 8u >= m_nodeCount || !deltaUploadEnabled())
        {
            return false;
        }
        if (count > m_nodeIslandDeltaCapacity)
        {
            const std::uint32_t want = std::max(
                count, m_nodeIslandDeltaCapacity ? m_nodeIslandDeltaCapacity * 2u : 2048u);
            cudaFree(m_devNodeIslandSlots);
            cudaFree(m_devNodeIslandValues);
            if (cudaMalloc(&m_devNodeIslandSlots, sizeof(std::uint32_t) * want) != cudaSuccess
                || cudaMalloc(&m_devNodeIslandValues, sizeof(std::uint32_t) * want) != cudaSuccess)
            {
                cudaGetLastError();
                m_nodeIslandDeltaCapacity = 0;
                return false;
            }
            m_nodeIslandDeltaCapacity = want;
        }
        m_nodeIslandSlots.resize(count);
        m_nodeIslandValues.resize(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const std::uint32_t node = m_changedNodeIslandSlots[i];
            m_nodeIslandSlots[i] = node;
            m_nodeIslandValues[i] = m_hostNodeIsland[node];
        }
        stageUpload(m_devNodeIslandSlots, m_nodeIslandSlots.data(),
                    sizeof(std::uint32_t) * count, "upload node island delta slots");
        stageUpload(m_devNodeIslandValues, m_nodeIslandValues.data(),
                    sizeof(std::uint32_t) * count, "upload node island delta values");
        scatterNodeBondRefs<<<(count + kBlockSize - 1) / kBlockSize, kBlockSize, 0, m_stream>>>(
            m_devNodeIslandSlots, m_devNodeIslandValues, count, m_nodeIsland);
        return true;
    }

    /// Rebuild the island partition, or carry the stale one another tick?
    ///
    /// Carrying it is always CORRECT (see applyTopologyChange); it only costs
    /// skip granularity, because two components that have actually separated
    /// keep sharing one convergence flag and one skip decision. Rebuild once
    /// that has had a chance to matter -- measured against the alternative of
    /// rebuilding every tick, which spends 2.2 ms to sharpen a partition that
    /// is usually unchanged.
    bool shouldRepartition()
    {
        // `<`, not `!=`: removeBond shrinks m_bondCount without shrinking the
        // label array, so a stale array is legitimately LONGER than the graph.
        if (m_islandCount == 0 || m_hostBondIsland.size() < m_bondCount)
        {
            return true;   // never partitioned, or the labels do not cover the graph
        }
        // Local relabelling never reuses an id, so the count drifts above the
        // true island number and every per-island array, memset and serial tail
        // scales with the drift. Compact it once it has grown materially.
        if (m_islandCount > m_islandCountAtRebuild + m_islandCountAtRebuild / 4u + 64u)
        {
            return true;
        }
        if (!deferRepartitionEnabled())
        {
            return true;   // A/B control: rebuild every fracture tick, as before
        }
        return applyRemovalSplits();
    }

    /// Did any of this tick's removals actually disconnect something?
    ///
    /// Removing one edge can only ever separate the two sides of THAT edge, so
    /// "did the partition change" is exactly "are these two endpoints still
    /// connected to each other". If they are, the component is unchanged and
    /// the existing labels remain the true partition -- not an approximation of
    /// it. That is what makes deferring the rebuild free of quality cost.
    ///
    /// Deferring on a fixed budget instead was measurably wrong: it let two
    /// components that HAD separated keep sharing one convergence test, so a
    /// large component's tolerance hid a small under-solved one and the
    /// residual went from 0.63x tolerance to 22x, with 42% of islands over.
    ///
    /// The walk is bounded; exhausting the budget reports "split" and forces a
    /// rebuild, which is the conservative direction.
    /// Handle this tick's removals: detect splits, and relabel them LOCALLY
    /// when the separated piece is small. Returns true if a full repartition is
    /// still required.
    ///
    /// Removing one edge can only separate the two sides of THAT edge, so a
    /// bounded walk from one endpoint answers both questions at once: if it
    /// reaches the other endpoint the component is unchanged; if it exhausts
    /// naturally the visited set IS the separated piece, and giving that piece a
    /// fresh island id is the entire partition update. A chunk falling off a
    /// building costs work proportional to the chunk, not to the city.
    ///
    /// Falling back to a full rebuild is always safe -- it is what this
    /// replaces -- so every uncertain case returns true.
    bool applyRemovalSplits()
    {
        constexpr std::uint32_t kVisitBudget = 512u;
        const auto isStatic = [&](std::uint32_t node) {
            return !(m_hostInertia[node].linear > 0.0f);
        };
        m_splitVisited.resize(m_nodeCount, 0u);
        m_affectedIslands.clear();
        for (const auto& pair : m_splitChecks)
        {
            const std::uint32_t from = pair.first;
            const std::uint32_t to = pair.second;
            if (from >= m_nodeCount || to >= m_nodeCount)
            {
                return true;
            }
            // A bond with a static endpoint was already a cut in the partition.
            if (isStatic(from) || isStatic(to))
            {
                continue;
            }
            if (++m_splitStamp == 0u)
            {
                std::fill(m_splitVisited.begin(), m_splitVisited.end(), 0u);
                m_splitStamp = 1u;
            }
            m_splitQueue.clear();
            m_splitQueue.push_back(from);
            m_splitVisited[from] = m_splitStamp;
            bool reconnected = false;
            for (std::size_t qi = 0; qi < m_splitQueue.size() && !reconnected; ++qi)
            {
                if (m_splitQueue.size() > kVisitBudget)
                {
                    return true;   // piece too big to relabel cheaply
                }
                const std::uint32_t node = m_splitQueue[qi];
                for (std::uint32_t i = m_hostNodeBondBegin[node];
                     i < m_hostNodeBondBegin[node + 1]; ++i)
                {
                    const std::uint32_t ref = m_hostNodeBondRef[i];
                    if (ref == kDeadBondRef)
                    {
                        continue;
                    }
                    const std::uint32_t bond = ref & 0x7FFFFFFFu;
                    if (bond >= m_bondCount)
                    {
                        continue;
                    }
                    const std::uint32_t other =
                        (ref & 0x80000000u) ? m_hostNode0[bond] : m_hostNode1[bond];
                    if (other >= m_nodeCount || isStatic(other))
                    {
                        continue;   // static nodes cut the graph, as in the union rule
                    }
                    if (other == to)
                    {
                        reconnected = true;
                        break;
                    }
                    if (m_splitVisited[other] != m_splitStamp)
                    {
                        m_splitVisited[other] = m_splitStamp;
                        m_splitQueue.push_back(other);
                    }
                }
            }
            if (reconnected)
            {
                continue;   // no split: the labels are still the true partition
            }
            if (!localSplitRelabelEnabled() || m_islandCount >= m_islandCapacity)
            {
                return true;   // fall back to a full repartition
            }

            // A split. Do NOT relabel just this side.
            //
            // Relabelling only the BFS-source component is wrong when an island
            // splits three or more ways: the source side gets a fresh id, but
            // the REMAINDER can be two disjoint pieces still sharing the old
            // id. They then share one convergence test, and a large piece's
            // tolerance hides a small under-solved one -- measured as
            // residual/tolerance 1.05x -> 5.06x and 12% more bonds broken.
            //
            // Instead, recompute the partition for the affected island only.
            // That is O(that island), not O(the city).
            m_affectedIslands.push_back(m_hostNodeIsland[from]);
            ++m_statLocalSplits;
        }
        return m_affectedIslands.empty() ? false : !rebuildAffectedIslands();
    }

    /// Diagnostic: is the incrementally-maintained partition IDENTICAL, as a
    /// partition, to what a full computeIslands would produce right now?
    ///
    /// This is the probe that settles whether an observed behaviour difference
    /// is a real partition error or just chaotic amplification of a changed
    /// reduction order. Ids are allowed to differ; only the equivalence classes
    /// must match, so it checks for a bijection both ways.
    ///
    /// Destroys and restores the partition around a full rebuild, so it is
    /// strictly a debug path (BLAST_GPU_VERIFY_PARTITION=1).
    void verifyPartitionAgainstRebuild()
    {
        m_verifyNodeIsland.assign(m_hostNodeIsland.begin(), m_hostNodeIsland.end());
        m_verifyBondIsland.assign(m_hostBondIsland.begin(), m_hostBondIsland.end());
        const std::uint32_t incrementalCount = m_islandCount;

        computeIslands();   // overwrites the labels with the reference partition

        std::map<std::uint32_t, std::uint32_t> incToRef, refToInc;
        std::uint64_t mismatches = 0;
        for (std::uint32_t n = 0; n < m_nodeCount; ++n)
        {
            const std::uint32_t inc = m_verifyNodeIsland[n];
            const std::uint32_t ref = m_hostNodeIsland[n];
            if ((inc == kNoIsland) != (ref == kNoIsland)) { ++mismatches; continue; }
            if (inc == kNoIsland) { continue; }
            auto a = incToRef.emplace(inc, ref);
            if (!a.second && a.first->second != ref) { ++mismatches; }
            auto b = refToInc.emplace(ref, inc);
            if (!b.second && b.first->second != inc) { ++mismatches; }
        }
        std::uint64_t bondMismatches = 0;
        for (std::uint32_t b = 0; b < m_bondCount; ++b)
        {
            const std::uint32_t inc = m_verifyBondIsland[b];
            const std::uint32_t ref = m_hostBondIsland[b];
            if ((inc == kNoIsland) != (ref == kNoIsland)) { ++bondMismatches; continue; }
            if (inc == kNoIsland) { continue; }
            auto it = incToRef.find(inc);
            if (it == incToRef.end() || it->second != ref) { ++bondMismatches; }
        }
        if (mismatches || bondMismatches || ++m_verifyCalls % 200u == 0u)
        {
            std::fprintf(stderr,
                "[partition-verify] call=%llu incrementalIslands=%u referenceIslands=%u "
                "nodeMismatches=%llu bondMismatches=%llu\n",
                static_cast<unsigned long long>(m_verifyCalls),
                incrementalCount, m_islandCount,
                static_cast<unsigned long long>(mismatches),
                static_cast<unsigned long long>(bondMismatches));
        }

        // Put the incremental partition back so the run continues to measure it.
        m_hostNodeIsland.assign(m_verifyNodeIsland.begin(), m_verifyNodeIsland.end());
        m_hostBondIsland.assign(m_verifyBondIsland.begin(), m_verifyBondIsland.end());
        m_islandCount = incrementalCount;
    }

    /// Recompute connected components for just the islands a removal split.
    ///
    /// Correct by construction: every node of an affected island is re-derived
    /// from scratch, so no two disconnected pieces can be left sharing an id.
    /// Returns false if it declines (too much of the graph affected), leaving
    /// the caller to do a full repartition.
    bool rebuildAffectedIslands()
    {
        const auto isStatic = [&](std::uint32_t node) {
            return !(m_hostInertia[node].linear > 0.0f);
        };
        std::sort(m_affectedIslands.begin(), m_affectedIslands.end());
        m_affectedIslands.erase(
            std::unique(m_affectedIslands.begin(), m_affectedIslands.end()),
            m_affectedIslands.end());

        // Gather the affected islands' nodes. One pass over nodes; the union
        // work below is proportional to those islands, not to the graph.
        m_affectedNodes.clear();
        for (std::uint32_t n = 0; n < m_nodeCount; ++n)
        {
            if (isStatic(n))
            {
                continue;
            }
            const std::uint32_t island = m_hostNodeIsland[n];
            if (island != kNoIsland
                && std::binary_search(m_affectedIslands.begin(), m_affectedIslands.end(), island))
            {
                m_affectedNodes.push_back(n);
            }
        }
        // If the affected islands are most of the graph, a full rebuild is both
        // simpler and no more expensive.
        if (m_affectedNodes.size() * 2u >= m_nodeCount)
        {
            return false;
        }

        for (std::uint32_t n : m_affectedNodes)
        {
            m_hostParent[n] = n;
        }
        for (std::uint32_t n : m_affectedNodes)
        {
            for (std::uint32_t i = m_hostNodeBondBegin[n]; i < m_hostNodeBondBegin[n + 1]; ++i)
            {
                const std::uint32_t ref = m_hostNodeBondRef[i];
                if (ref == kDeadBondRef)
                {
                    continue;
                }
                const std::uint32_t bond = ref & 0x7FFFFFFFu;
                if (bond >= m_bondCount)
                {
                    continue;
                }
                const std::uint32_t other =
                    (ref & 0x80000000u) ? m_hostNode0[bond] : m_hostNode1[bond];
                if (other >= m_nodeCount || isStatic(other))
                {
                    continue;   // static nodes cut, exactly as computeIslands does
                }
                unite(n, other);
            }
        }

        // Reuse the affected ids first so the island count does not drift.
        std::size_t nextReuse = 0;
        m_rootRemap.clear();
        for (std::uint32_t n : m_affectedNodes)
        {
            const std::uint32_t root = findRoot(n);
            auto it = m_rootRemap.find(root);
            std::uint32_t id;
            if (it == m_rootRemap.end())
            {
                if (nextReuse < m_affectedIslands.size())
                {
                    id = m_affectedIslands[nextReuse++];
                }
                else if (m_islandCount < m_islandCapacity)
                {
                    id = m_islandCount++;
                }
                else
                {
                    return false;   // out of id space; a full rebuild compacts it
                }
                m_rootRemap.emplace(root, id);
                if (id < m_islandCapacity)
                {
                    // Every piece here changed shape; none may be skipped on trust.
                    m_hostIslandConverged[id] = 0u;
                    m_hostIslandSkip[id] = 0u;
                }
            }
            else
            {
                id = it->second;
            }
            m_hostNodeIsland[n] = id;
            m_changedNodeIslandSlots.push_back(n);
            for (std::uint32_t i = m_hostNodeBondBegin[n]; i < m_hostNodeBondBegin[n + 1]; ++i)
            {
                const std::uint32_t ref = m_hostNodeBondRef[i];
                if (ref == kDeadBondRef)
                {
                    continue;
                }
                const std::uint32_t bond = ref & 0x7FFFFFFFu;
                if (bond >= m_bondCount)
                {
                    continue;
                }
                m_hostBondIsland[bond] = id;
                m_changedBondSlots.push_back(bond);
            }
        }
        m_bondsByIslandValid = false;
        return true;
    }

    /// Forget that anything is settled. Used whenever the impulses the
    /// convergence flags refer to are no longer the ones on the device.
    void invalidateSettledBaseline()
    {
        m_settledBaselineValid = false;
        std::fill(m_hostIslandConverged.begin(), m_hostIslandConverged.end(), 0u);
        if (m_hostIslandSkip)
        {
            std::fill(m_hostIslandSkip, m_hostIslandSkip + m_islandCapacity, 0u);
        }
    }

    void enqueueImpulseReadback(bool compacted)
    {
        const std::uint32_t count = static_cast<std::uint32_t>(m_changedBonds.size());
        checkCuda(cudaEventRecord(m_downloadStart, m_stream), "record readback start");
        if (!compacted)
        {
            checkCuda(
                cudaMemcpyAsync(
                    m_hostImpulses,
                    m_impulses,
                    sizeof(AngLin) * m_bondCount,
                    cudaMemcpyDeviceToHost,
                    m_stream),
                "read stress impulses");
        }
        else if (count > 0)
        {
            // Gather on the device, copy dense. Reading the whole array and
            // discarding most of it would hand back on the PCIe bus and in the
            // host conversion loop exactly what the skip just saved.
            std::memcpy(
                m_hostGatherIndices,
                m_changedBonds.data(),
                sizeof(std::uint32_t) * count);
            checkCuda(
                cudaMemcpyAsync(
                    m_gatherIndices,
                    m_hostGatherIndices,
                    sizeof(std::uint32_t) * count,
                    cudaMemcpyHostToDevice,
                    m_stream),
                "upload changed bond indices");
            gatherImpulses<<<
                (count + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(m_impulses, m_gatherIndices, m_gatherOutput, count);
            checkCuda(
                cudaMemcpyAsync(
                    m_hostImpulses,
                    m_gatherOutput,
                    sizeof(AngLin) * count,
                    cudaMemcpyDeviceToHost,
                    m_stream),
                "read changed stress impulses");
            m_telemetry.hostToDeviceBytes +=
                sizeof(std::uint32_t) * static_cast<std::uint64_t>(count);
        }
        checkCuda(cudaEventRecord(m_downloadStop, m_stream), "record readback stop");
    }

    void finishImpulseReadback(ExtStressGpuImpulse* bondImpulses, bool compacted)
    {
        checkCuda(
            cudaEventElapsedTime(
                &m_telemetry.downloadMilliseconds,
                m_downloadStart,
                m_downloadStop),
            "measure impulse readback");
        const std::uint32_t count = static_cast<std::uint32_t>(m_changedBonds.size());
        const float linearScale = m_lengthScale * m_massScale;
        const float angularScale = m_lengthScale * linearScale;
        for (std::uint32_t j = 0; j < count; ++j)
        {
            const std::uint32_t bond = m_changedBonds[j];
            const AngLin& value = m_hostImpulses[compacted ? j : bond];
            // Device impulses are also column-scaled (J' = J / colScale).
            const float s_j = m_hostColScale[bond];
            // Device impulses are solver-scaled; this loop already touches
            // every changed bond for the 32 B -> 24 B repack, so converting to
            // physical units here is six multiplies on data already in cache.
            // Doing it on the device instead would need a full-array pass on
            // the non-compacted path -- which is the path taken after every
            // fracture tick, where it would cost as much as it saves.
            bondImpulses[bond].angular =
                {value.angular.x * angularScale * s_j,
                 value.angular.y * angularScale * s_j,
                 value.angular.z * angularScale * s_j};
            bondImpulses[bond].linear =
                {value.linear.x * linearScale * s_j,
                 value.linear.y * linearScale * s_j,
                 value.linear.z * linearScale * s_j};
        }
        m_telemetry.deviceToHostBytes +=
            sizeof(AngLin) * static_cast<std::uint64_t>(count);
    }

    void prepare(const ExtStressGpuNode* nodes, const ExtStressGpuBond* bonds)
    {
        m_hostNode0.resize(m_bondCount);
        m_hostNode1.resize(m_bondCount);
        m_hostOffset0.resize(m_bondCount);
        m_hostOffset1.resize(m_bondCount);
        m_hostInertia.resize(m_nodeCount);
        m_hostNormals.resize(m_bondCount);
        m_hostAreas.resize(m_bondCount);
        m_hostColScale.resize(m_bondCount);
        m_hostNodeDistances.resize(m_bondCount);
        m_hostHealth.resize(m_bondCount);
        m_hostBondMaterials.resize(m_bondCount);

        double logMass = 0.0;
        std::uint32_t massCount = 0;
        for (std::uint32_t i = 0; i < m_nodeCount; ++i)
        {
            if (nodes[i].mass > 0.0f)
            {
                logMass += std::log(nodes[i].mass);
                ++massCount;
            }
            m_hostInertia[i] = {
                nodes[i].inertia > 0.0f ? 1.0f : 0.0f,
                nodes[i].mass > 0.0f ? 1.0f : 0.0f};
        }
        m_massScale =
            massCount ? static_cast<float>(std::exp(logMass / massCount)) : 1.0f;

        double lengthSum = 0.0;
        std::uint32_t offsetCount = 0;
        for (std::uint32_t i = 0; i < m_bondCount; ++i)
        {
            if (bonds[i].node0 >= m_nodeCount || bonds[i].node1 >= m_nodeCount)
            {
                throw std::runtime_error("GPU stress bond node index is out of range");
            }
            m_hostNode0[i] = bonds[i].node0;
            m_hostNode1[i] = bonds[i].node1;
            const ExtStressGpuNode& first = nodes[bonds[i].node0];
            const ExtStressGpuNode& second = nodes[bonds[i].node1];
            const Vec4 displacement = makeVec(
                second.position[0] - first.position[0],
                second.position[1] - first.position[1],
                second.position[2] - first.position[2]);
            const float distance = std::sqrt(
                displacement.x * displacement.x
                + displacement.y * displacement.y
                + displacement.z * displacement.z);
            Vec4 normal = makeVec(
                bonds[i].normal[0],
                bonds[i].normal[1],
                bonds[i].normal[2]);
            float normalLength = std::sqrt(
                normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
            if (!(normalLength > 0.0f))
            {
                normal = distance > 0.0f
                    ? mul(displacement, 1.0f / distance)
                    : makeVec(1.0f, 0.0f, 0.0f);
            }
            else
            {
                normal = mul(normal, 1.0f / normalLength);
                const float alignment =
                    normal.x * displacement.x
                    + normal.y * displacement.y
                    + normal.z * displacement.z;
                if (alignment < 0.0f)
                {
                    normal = mul(normal, -1.0f);
                }
            }
            m_hostNormals[i] = normal;
            m_hostAreas[i] = bonds[i].area > 0.0f ? bonds[i].area : 1.0f;
            m_hostColScale[i] = bonds[i].colScale > 0.0f ? bonds[i].colScale : 1.0f;
            m_hostNodeDistances[i] = distance > 1.0e-6f ? distance : 1.0f;
            m_hostHealth[i] = std::max(0.0f, bonds[i].health);
            m_hostBondMaterials[i] =
                bonds[i].material < m_materialCount ? bonds[i].material : 0;
            Vec4 offset0{};
            Vec4 offset1{};
            if (first.mass <= 0.0f)
            {
                offset1 = makeVec(
                    bonds[i].centroid[0] - second.position[0],
                    bonds[i].centroid[1] - second.position[1],
                    bonds[i].centroid[2] - second.position[2]);
                offset0 = mul(offset1, -1.0f);
            }
            else if (second.mass <= 0.0f)
            {
                offset0 = makeVec(
                    bonds[i].centroid[0] - first.position[0],
                    bonds[i].centroid[1] - first.position[1],
                    bonds[i].centroid[2] - first.position[2]);
                offset1 = mul(offset0, -1.0f);
            }
            else
            {
                offset0 = makeVec(
                    0.5f * (second.position[0] - first.position[0]),
                    0.5f * (second.position[1] - first.position[1]),
                    0.5f * (second.position[2] - first.position[2]));
                offset1 = mul(offset0, -1.0f);
            }
            if (first.mass > 0.0f)
            {
                lengthSum += std::sqrt(
                    offset0.x * offset0.x + offset0.y * offset0.y + offset0.z * offset0.z);
                ++offsetCount;
            }
            if (second.mass > 0.0f)
            {
                lengthSum += std::sqrt(
                    offset1.x * offset1.x + offset1.y * offset1.y + offset1.z * offset1.z);
                ++offsetCount;
            }
            m_hostOffset0[i] = offset0;
            m_hostOffset1[i] = offset1;
        }
        m_lengthScale =
            offsetCount ? static_cast<float>(lengthSum / offsetCount) : 1.0f;
        if (!(m_lengthScale > 0.0f))
        {
            m_lengthScale = 1.0f;
        }
        const float reciprocalLength = 1.0f / m_lengthScale;
        for (std::uint32_t i = 0; i < m_bondCount; ++i)
        {
            m_hostOffset0[i] = mul(m_hostOffset0[i], reciprocalLength);
            m_hostOffset1[i] = mul(m_hostOffset1[i], reciprocalLength);
        }
    }

    template <typename T>
    void allocateDevice(T*& pointer, std::size_t count, const char* name)
    {
        checkCuda(cudaMalloc(&pointer, sizeof(T) * count), name);
    }

    template <typename T>
    void allocateHost(T*& pointer, std::size_t count, const char* name)
    {
        checkCuda(cudaMallocHost(&pointer, sizeof(T) * count), name);
    }

    void allocate()
    {
        allocateDevice(m_node0, m_bondCount, "allocate node0");
        allocateDevice(m_node1, m_bondCount, "allocate node1");
        allocateDevice(m_offset0, m_bondCount, "allocate offset0");
        allocateDevice(m_offset1, m_bondCount, "allocate offset1");
        allocateDevice(m_inertia, m_nodeCount, "allocate inertia");
        allocateDevice(m_normals, m_bondCount, "allocate normals");
        allocateDevice(m_areas, m_bondCount, "allocate areas");
        allocateDevice(m_colScales, m_bondCount, "allocate compliance weights");
        allocateDevice(m_bondMaterials, m_bondCount, "allocate bond materials");
        allocateDevice(m_materials, m_materialCount, "allocate material table");
        allocateDevice(m_nodeDistances, m_bondCount, "allocate node distances");
        allocateDevice(m_health, m_bondCount, "allocate health");
        allocateDevice(m_brokenBonds, m_bondCount, "allocate broken bonds");
        allocateDevice(m_brokenCount, 1, "allocate broken count");
        allocateDevice(m_input, m_nodeCount, "allocate input");
        allocateDevice(m_impulses, m_bondCount, "allocate impulses");
        allocateDevice(m_rhs, m_nodeCount, "allocate rhs");
        allocateDevice(m_gradient, m_bondCount, "allocate gradient");
        allocateDevice(m_direction, m_bondCount, "allocate direction");
        allocateDevice(m_residual, m_nodeCount, "allocate residual");
        allocateDevice(m_projectedDirection, m_nodeCount, "allocate projected direction");
        allocateDevice(
            m_reductionInput,
            std::max(m_nodeCount, m_bondCount),
            "allocate reduction input");
        // Per-island conjugate-gradient scalars. One set per disconnected
        // component, so islands converge and step independently.
        // Sized for the worst case, not for today's partition: every bond that
        // breaks can split an island, and an island contains at least one
        // dynamic node, so the count can rise to m_nodeCount over the solver's
        // life. Allocating for that once is a few hundred kilobytes and means
        // a repartition never has to reallocate -- which is what would drag
        // the full-rebuild cost back in through the side door.
        m_islandCapacity = std::max(m_nodeCount, 1u);
        allocateDevice(m_gradientSquared, m_islandCapacity, "allocate gradient norm");
        allocateDevice(m_iteration, 1, "allocate device iteration counter");
        // Node-space CGLS working set. Node-length, so ~1/3 of the bond-length
        // vectors they replace.
        allocateDevice(m_nsPi, m_nodeCount, "allocate node-space direction");
        allocateDevice(m_nsQ, m_nodeCount, "allocate node-space projected direction");
        allocateDevice(m_nsW, m_nodeCount, "allocate node-space matvec output");
        allocateDevice(m_nsMu, m_nodeCount, "allocate node-space correction");
        allocateDevice(m_nsG, m_nodeCount, "allocate node-space preconditioned residual");
        allocateDevice(m_nsW2, m_nodeCount, "allocate node-space second matvec output");
        allocateDevice(m_nsJacobi, static_cast<std::size_t>(m_nodeCount) * 36,
                       "allocate node-space block-Jacobi inverses");
        allocateDevice(m_nsGamma, m_islandCapacity, "allocate node-space gamma");
        allocateDevice(m_nsGammaPrev, m_islandCapacity, "allocate node-space gamma prev");
        // Padded accumulators for the per-island reduction. One buffer serves
        // every reduction because each is finalized into its own result array
        // before the next one starts.
        allocateDevice(
            m_reduceSlots,
            static_cast<std::size_t>(m_islandCapacity) * kReductionSlotsMax,
            "allocate padded island reduction slots");
        allocateDevice(
            m_projectedDirectionSquared, m_islandCapacity, "allocate projected norm");
        allocateDevice(m_deltaSquared, m_islandCapacity, "allocate tolerance");
        allocateDevice(
            m_previousGradientSquared, m_islandCapacity, "allocate previous gradient norm");
        allocateDevice(m_islandActive, m_islandCapacity, "allocate island active flags");
        allocateDevice(m_islandConverged, m_islandCapacity, "allocate island converged flags");
        allocateDevice(m_islandSkip, m_islandCapacity, "allocate island skip mask");
        allocateDevice(
            m_blockActiveCounts,
            (m_islandCapacity + kBlockSize - 1) / kBlockSize + 1,
            "allocate island block tallies");
        allocateDevice(m_nodeBondBegin, m_nodeCount + 1, "allocate node-bond csr offsets");
        allocateDevice(m_nodeBondRef, m_bondCount * 2 + 1, "allocate node-bond csr refs");
        allocateDevice(m_bondIsland, m_bondCount, "allocate bond island ids");
        allocateDevice(m_nodeIsland, m_nodeCount, "allocate node island ids");
        allocateDevice(m_scatterIndices, m_nodeCount, "allocate scatter indices");
        allocateDevice(m_scatterValues, m_nodeCount, "allocate scatter values");
        allocateDevice(m_gatherIndices, m_bondCount, "allocate gather indices");
        allocateDevice(m_gatherOutput, m_bondCount, "allocate gather output");
        allocateDevice(m_status, 1, "allocate solve status");
        allocateDevice(m_activeBonds, m_bondCount, "allocate active bond list");
        allocateDevice(m_activeNodes, m_nodeCount, "allocate active node list");
        allocateDevice(m_activeCounts, 2, "allocate active counts");
        allocateDevice(
            m_activeFlags, std::max(m_bondCount, m_nodeCount), "allocate active flags");
        {
            // Scratch for the order-preserving selects; sized for the larger
            // of the two so one buffer serves both.
            cub::CountingInputIterator<std::uint32_t> identity(0u);
            std::size_t bondSelectBytes = 0;
            std::size_t nodeSelectBytes = 0;
            cub::DeviceSelect::Flagged(
                nullptr,
                bondSelectBytes,
                identity,
                m_activeFlags,
                m_activeBonds,
                m_activeCounts,
                static_cast<int>(m_bondCount));
            cub::DeviceSelect::Flagged(
                nullptr,
                nodeSelectBytes,
                identity,
                m_activeFlags,
                m_activeNodes,
                m_activeCounts + 1,
                static_cast<int>(m_nodeCount));
            m_selectScratchBytes = std::max(bondSelectBytes, nodeSelectBytes);
            checkCuda(
                cudaMalloc(&m_selectScratch, m_selectScratchBytes),
                "allocate select scratch");
        }
        allocateHost(m_hostActiveCounts, 2, "allocate pinned active counts");
        m_hostActiveCounts[0] = 0u;
        m_hostActiveCounts[1] = 0u;
        allocateHost(m_hostInput, m_nodeCount, "allocate pinned stress input");
        allocateHost(m_hostImpulses, m_bondCount, "allocate pinned stress impulses");
        allocateHost(m_hostStatus, 1, "allocate pinned stress status");
        allocateHost(m_hostBrokenCount, 1, "allocate pinned broken count");
        *m_hostBrokenCount = 0u;
        allocateHost(m_hostIslandSkip, m_islandCapacity, "allocate pinned island skip mask");
        allocateHost(
            m_hostIslandConvergedPinned, m_islandCapacity, "allocate pinned island convergence");
        allocateHost(m_hostScatterIndices, m_nodeCount, "allocate pinned scatter indices");
        allocateHost(m_hostScatterValues, m_nodeCount, "allocate pinned scatter values");
        allocateHost(m_hostGatherIndices, m_bondCount, "allocate pinned gather indices");
        m_hostIslandConverged.assign(m_islandCapacity, 0u);
        m_islandDirty.assign(m_islandCapacity, 1u);
        m_changedNodes.reserve(m_nodeCount);
        m_changedBonds.reserve(m_bondCount);
        std::fill(m_hostIslandSkip, m_hostIslandSkip + m_islandCount, 0u);

        const std::uint32_t reductionCount = std::max(m_nodeCount, m_bondCount);
        cub::DeviceReduce::Sum(
            nullptr,
            m_reduceScratchBytes,
            m_reductionInput,
            m_gradientSquared,
            reductionCount);
        checkCuda(cudaMalloc(&m_reduceScratch, m_reduceScratchBytes), "allocate reduction scratch");
        checkCuda(
            cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking),
            "create solver stream");
        // Capture-only: the conditional loop body is captured onto this stream
        // into the while-node's body graph. Nothing is ever launched on it
        // outside capture.
        checkCuda(
            cudaStreamCreateWithFlags(&m_bodyStream, cudaStreamNonBlocking),
            "create conditional body capture stream");
        checkCuda(cudaEventCreate(&m_uploadStart), "create upload start event");
        checkCuda(cudaEventCreate(&m_uploadStop), "create upload stop event");
        checkCuda(cudaEventCreate(&m_solveStart), "create solve start event");
        checkCuda(cudaEventCreate(&m_solveStop), "create solve stop event");
        checkCuda(cudaEventCreate(&m_statusReady), "create status-ready event");
        checkCuda(cudaEventCreate(&m_topoUploadDone), "create topology-upload event");
        checkCuda(cudaEventCreate(&m_downloadStart), "create download start event");
        checkCuda(cudaEventCreate(&m_downloadStop), "create download stop event");
        checkCuda(cudaMemset(m_impulses, 0, sizeof(AngLin) * m_bondCount), "clear impulses");
        checkCuda(cudaMemset(m_brokenCount, 0, sizeof(std::uint32_t)), "clear broken count");
        checkCuda(
            cudaMemset(m_islandConverged, 0, sizeof(std::uint32_t) * m_islandCapacity),
            "clear island converged flags");
        checkCuda(
            cudaMemset(m_islandSkip, 0, sizeof(std::uint32_t) * m_islandCapacity),
            "clear island skip mask");
        checkCuda(
            cudaMemset(m_input, 0, sizeof(ExtStressGpuImpulse) * m_nodeCount),
            "clear stress inputs");
        // Nothing has been solved yet, so nothing may be skipped against it.
        std::memset(m_hostInput, 0, sizeof(ExtStressGpuImpulse) * m_nodeCount);
    }

    /**
     * Group bonds by island, once.
     *
     * The partition cannot change under a live solver -- a broken bond is
     * expressed as zero health, and any real topology change goes through a
     * new solver -- so this CSR is built at construction and read every frame
     * to answer "which bonds belong to the islands I am solving?" in time
     * proportional to that answer rather than to the whole graph.
     */
    void groupBondsByIsland()
    {
        m_islandBondBegin.assign(m_islandCount + 1, 0u);
        for (std::uint32_t b = 0; b < m_bondCount; ++b)
        {
            if (m_hostBondIsland[b] != kNoIsland)
            {
                ++m_islandBondBegin[m_hostBondIsland[b] + 1];
            }
        }
        for (std::uint32_t k = 0; k < m_islandCount; ++k)
        {
            m_islandBondBegin[k + 1] += m_islandBondBegin[k];
        }
        m_bondsByIsland.resize(m_islandBondBegin[m_islandCount]);
        m_groupCursor.assign(
            m_islandBondBegin.begin(), m_islandBondBegin.end());
        std::vector<std::uint32_t>& cursor = m_groupCursor;
        for (std::uint32_t b = 0; b < m_bondCount; ++b)
        {
            const std::uint32_t island = m_hostBondIsland[b];
            if (island != kNoIsland)
            {
                m_bondsByIsland[cursor[island]++] = b;
            }
        }
    }

    /// Partition the graph into islands, matching the CPU solver's rule
    /// exactly (stress.cpp solveIslandAware): union only across bonds whose
    /// endpoints are both dynamic, because a static (zero-mass) node is a fixed
    /// boundary that transmits no coupling and therefore cuts the graph. The
    /// partitions must agree for CPU and GPU results to agree.
    void computeIslands()
    {
        m_hostParent.resize(m_nodeCount);
        for (std::uint32_t i = 0; i < m_nodeCount; ++i)
        {
            m_hostParent[i] = i;
        }
        // A node is static exactly when prepare() gave it a zero inverse
        // inertia, so the partition can be rebuilt from the host mirrors alone
        // -- which is what lets a bond removal repartition in place instead of
        // going back to the caller's descriptors.
        const auto isStatic = [&](std::uint32_t node) {
            return !(m_hostInertia[node].linear > 0.0f);
        };
        for (std::uint32_t b = 0; b < m_bondCount; ++b)
        {
            const std::uint32_t n0 = m_hostNode0[b];
            const std::uint32_t n1 = m_hostNode1[b];
            if (n0 >= m_nodeCount || n1 >= m_nodeCount)
            {
                continue;
            }
            if (isStatic(n0) || isStatic(n1))
            {
                continue; // cut at static nodes
            }
            unite(n0, n1);
        }

        // Generation stamping instead of re-filling the id map: this used to
        // clear three arrays totalling ~1.6 MB on every fracture tick, all of
        // which are then overwritten by the loops below. Only m_hostRootIsland
        // is read before it is written, and a stamp answers "written this
        // rebuild?" without touching memory proportional to the graph.
        if (++m_rootIslandGeneration == 0u)
        {
            // Wrapped: the stamps would alias the new generation.
            std::fill(m_hostRootStamp.begin(), m_hostRootStamp.end(), 0u);
            m_rootIslandGeneration = 1u;
        }
        m_hostRootIsland.resize(m_nodeCount);
        m_hostRootStamp.resize(m_nodeCount, 0u);
        m_hostBondIsland.resize(m_bondCount);
        m_islandCount = 0;
        for (std::uint32_t b = 0; b < m_bondCount; ++b)
        {
            const std::uint32_t n0 = m_hostNode0[b];
            const std::uint32_t n1 = m_hostNode1[b];
            if (n0 >= m_nodeCount || n1 >= m_nodeCount)
            {
                m_hostBondIsland[b] = kNoIsland;
                continue;
            }
            const bool s0 = isStatic(n0);
            const bool s1 = isStatic(n1);
            if (s0 && s1)
            {
                m_hostBondIsland[b] = kNoIsland; // degenerate: no coupling
                continue;
            }
            const std::uint32_t rep = findRoot(s0 ? n1 : n0);
            if (m_hostRootStamp[rep] != m_rootIslandGeneration)
            {
                m_hostRootStamp[rep] = m_rootIslandGeneration;
                m_hostRootIsland[rep] = m_islandCount++;
            }
            m_hostBondIsland[b] = m_hostRootIsland[rep];
        }

        // Static nodes stay unassigned: they are boundaries, excluded from the
        // per-island residual just as they are from the CPU sub-systems.
        m_hostNodeIsland.resize(m_nodeCount);
        for (std::uint32_t i = 0; i < m_nodeCount; ++i)
        {
            if (isStatic(i))
            {
                m_hostNodeIsland[i] = kNoIsland;
                continue;
            }
            const std::uint32_t rep = findRoot(i);
            // A dynamic node with no surviving bond never had an id created
            // for it, and must stay unassigned exactly as before.
            m_hostNodeIsland[i] = m_hostRootStamp[rep] == m_rootIslandGeneration
                ? m_hostRootIsland[rep] : kNoIsland;
        }
        if (m_islandCount == 0)
        {
            m_islandCount = 1; // keep buffers valid on a graph with no coupling
        }
    }

    std::uint32_t findRoot(std::uint32_t node)
    {
        while (m_hostParent[node] != node)
        {
            m_hostParent[node] = m_hostParent[m_hostParent[node]];
            node = m_hostParent[node];
        }
        return node;
    }

    void unite(std::uint32_t a, std::uint32_t b)
    {
        const std::uint32_t ra = findRoot(a);
        const std::uint32_t rb = findRoot(b);
        if (ra != rb)
        {
            m_hostParent[rb] = ra;
        }
    }

    /// Build the node -> incident-bond CSR the gather right-multiply walks.
    ///
    /// Pure host work over mirrors that already exist, done wherever topology
    /// is (re)built, so the per-iteration kernel never has to discover which
    /// bonds touch a node. Each ref packs the bond index with the endpoint
    /// bit (bit 31) that selects the sign convention -- a bond appears in
    /// both of its nodes' lists, once per side.
    ///
    /// A kNoIsland bond (static-static) is included: the kernel skips it via
    /// bondSettled exactly as the scatter did, and excluding it here would
    /// make the CSR disagree with the scatter path under BLAST_GPU_GATHER=0.
    void buildNodeBondCsr()
    {
        m_hostNodeBondBegin.assign(m_nodeCount + 1, 0u);
        for (std::uint32_t bond = 0; bond < m_bondCount; ++bond)
        {
            ++m_hostNodeBondBegin[m_hostNode0[bond] + 1];
            ++m_hostNodeBondBegin[m_hostNode1[bond] + 1];
        }
        for (std::uint32_t node = 0; node < m_nodeCount; ++node)
        {
            m_hostNodeBondBegin[node + 1] += m_hostNodeBondBegin[node];
        }
        // resize, not assign: every slot is written by the scatter below, so
        // the value-fill was ~2.4 MB of pointless stores per fracture tick.
        m_hostNodeBondRef.resize(m_hostNodeBondBegin[m_nodeCount]);
        // Reused across ticks; this was a fresh heap allocation every rebuild.
        m_csrCursor.assign(
            m_hostNodeBondBegin.begin(), m_hostNodeBondBegin.end());
        std::vector<std::uint32_t>& cursor = m_csrCursor;
        // Ascending bond order within each node's list: the sum order is then
        // a pure function of topology, which is what makes the gather
        // reproducible run to run.
        // Remember where each bond's two refs landed, so a removal can patch
        // them in O(1) instead of rebuilding the whole CSR.
        m_hostBondRefPos.resize(2u * static_cast<std::size_t>(m_bondCount));
        for (std::uint32_t bond = 0; bond < m_bondCount; ++bond)
        {
            const std::uint32_t p0 = cursor[m_hostNode0[bond]]++;
            const std::uint32_t p1 = cursor[m_hostNode1[bond]]++;
            m_hostNodeBondRef[p0] = bond;
            m_hostNodeBondRef[p1] = bond | 0x80000000u;
            m_hostBondRefPos[2u * bond] = p0;
            m_hostBondRefPos[2u * bond + 1u] = p1;
        }
        m_deadRefCount = 0;
        m_csrValid = true;
        m_changedRefSlots.clear();
    }

    /// Patch the CSR for one swap-with-last removal, in place.
    ///
    /// Node degrees change but their CSR RUNS do not move, so tombstoning the
    /// removed bond's two entries and retargeting the moved bond's two entries
    /// is the whole update -- four slots, versus rebuilding 2*bondCount entries
    /// and re-uploading them (measured 0.82 ms per fracture tick at 298k bonds).
    void patchCsrForRemoval(std::uint32_t removed, std::uint32_t last)
    {
        if (!m_csrValid || 2u * static_cast<std::size_t>(last) + 1u >= m_hostBondRefPos.size())
        {
            m_csrValid = false;
            return;
        }
        const std::uint32_t deadA = m_hostBondRefPos[2u * removed];
        const std::uint32_t deadB = m_hostBondRefPos[2u * removed + 1u];
        m_hostNodeBondRef[deadA] = kDeadBondRef;
        m_hostNodeBondRef[deadB] = kDeadBondRef;
        m_changedRefSlots.push_back(deadA);
        m_changedRefSlots.push_back(deadB);
        m_deadRefCount += 2;

        if (last != removed)
        {
            const std::uint32_t moveA = m_hostBondRefPos[2u * last];
            const std::uint32_t moveB = m_hostBondRefPos[2u * last + 1u];
            // Keep the endpoint bit; only the bond index changes.
            m_hostNodeBondRef[moveA] = removed;
            m_hostNodeBondRef[moveB] = removed | 0x80000000u;
            m_changedRefSlots.push_back(moveA);
            m_changedRefSlots.push_back(moveB);
            m_hostBondRefPos[2u * removed] = moveA;
            m_hostBondRefPos[2u * removed + 1u] = moveB;
        }
    }

    void uploadNodeBondCsr()
    {
        if (m_hostNodeBondBegin.empty())
        {
            return;
        }
        stageUpload(m_nodeBondBegin, m_hostNodeBondBegin.data(), sizeof(std::uint32_t) * m_hostNodeBondBegin.size(), "upload node-bond csr offsets");
        if (!m_hostNodeBondRef.empty())
        {
            stageUpload(m_nodeBondRef, m_hostNodeBondRef.data(), sizeof(std::uint32_t) * m_hostNodeBondRef.size(), "upload node-bond csr refs");
        }
    }

    /// Sparse alternative to uploadTopology + uploadIslands.
    ///
    /// Returns false when the change set is too large to be worth it, or does
    /// not cover everything that moved -- the caller then does the full upload.
    /// Correctness never depends on this returning true.
    bool uploadTopologyDelta()
    {
        const std::uint32_t count =
            static_cast<std::uint32_t>(m_changedBondSlots.size());
        if (count == 0)
        {
            return true;   // nothing moved; the device already matches
        }
        // Below ~1/8 of the graph the sparse path wins; above it the gather,
        // the extra kernel and the indirection cost more than a clean stream.
        if (count * 8u >= m_bondCount || !deltaUploadEnabled())
        {
            return false;
        }
        if (count > m_deltaCapacity)
        {
            const std::uint32_t want = std::max(count, m_deltaCapacity ? m_deltaCapacity * 2u : 1024u);
            cudaFree(m_devDeltaSlots);
            cudaFree(m_devDeltaValues);
            if (cudaMalloc(&m_devDeltaSlots, sizeof(std::uint32_t) * want) != cudaSuccess
                || cudaMalloc(&m_devDeltaValues, sizeof(BondDelta) * want) != cudaSuccess)
            {
                cudaGetLastError();
                m_deltaCapacity = 0;
                return false;
            }
            m_deltaCapacity = want;
        }
        m_deltaSlots.resize(count);
        m_deltaValues.resize(count);
        for (std::uint32_t i = 0; i < count; ++i)
        {
            const std::uint32_t b = m_changedBondSlots[i];
            if (b >= m_bondCount)
            {
                // The slot was itself removed later in the same tick; it is
                // past the live range now, so nothing needs to reach the device.
                m_deltaSlots[i] = 0u;
                m_deltaValues[i] = BondDelta{};
                m_deltaValues[i].island = kNoIsland;
                m_deltaSlots[i] = m_bondCount ? m_bondCount - 1u : 0u;
                continue;
            }
            m_deltaSlots[i] = b;
            BondDelta& v = m_deltaValues[i];
            v.offset0 = m_hostOffset0[b];
            v.offset1 = m_hostOffset1[b];
            v.normal = m_hostNormals[b];
            v.node0 = m_hostNode0[b];
            v.node1 = m_hostNode1[b];
            v.material = m_hostBondMaterials[b];
            v.island = b < m_hostBondIsland.size() ? m_hostBondIsland[b] : kNoIsland;
            v.area = m_hostAreas[b];
            v.nodeDistance = m_hostNodeDistances[b];
            v.health = m_hostHealth[b];
            v.colScale = m_hostColScale[b];
        }
        stageUpload(m_devDeltaSlots, m_deltaSlots.data(),
                    sizeof(std::uint32_t) * count, "upload delta slots");
        stageUpload(m_devDeltaValues, m_deltaValues.data(),
                    sizeof(BondDelta) * count, "upload delta values");
        scatterBondTopology<<<(count + kBlockSize - 1) / kBlockSize, kBlockSize, 0, m_stream>>>(
            m_devDeltaSlots, m_devDeltaValues, count,
            m_node0, m_node1, m_offset0, m_offset1, m_normals,
            m_areas, m_nodeDistances, m_health, m_colScales, m_bondMaterials, m_bondIsland);
        return true;
    }

    void uploadIslands()
    {
        stageUpload(m_bondIsland, m_hostBondIsland.data(), sizeof(std::uint32_t) * m_bondCount, "upload bond island ids");
        stageUpload(m_nodeIsland, m_hostNodeIsland.data(), sizeof(std::uint32_t) * m_nodeCount, "upload node island ids");
    }

    void uploadTopology()
    {
        stageUpload(m_node0, m_hostNode0.data(), sizeof(std::uint32_t) * m_bondCount, "upload node0");
        stageUpload(m_node1, m_hostNode1.data(), sizeof(std::uint32_t) * m_bondCount, "upload node1");
        stageUpload(m_offset0, m_hostOffset0.data(), sizeof(Vec4) * m_bondCount, "upload offset0");
        stageUpload(m_offset1, m_hostOffset1.data(), sizeof(Vec4) * m_bondCount, "upload offset1");
        stageUpload(m_inertia, m_hostInertia.data(), sizeof(Inertia) * m_nodeCount, "upload inertia");
        stageUpload(m_normals, m_hostNormals.data(), sizeof(Vec4) * m_bondCount, "upload normals");
        stageUpload(m_areas, m_hostAreas.data(), sizeof(float) * m_bondCount, "upload areas");
        stageUpload(m_colScales, m_hostColScale.data(), sizeof(float) * m_bondCount, "upload compliance weights");
        stageUpload(m_nodeDistances, m_hostNodeDistances.data(), sizeof(float) * m_bondCount, "upload node distances");
        stageUpload(m_health, m_hostHealth.data(), sizeof(float) * m_bondCount, "upload health");
        stageUpload(m_bondMaterials, m_hostBondMaterials.data(), sizeof(std::uint32_t) * m_bondCount, "upload bond materials");
        stageUpload(m_materials, m_hostMaterials.data(), sizeof(ExtStressGpuMaterial) * m_materialCount, "upload material table");
    }

    /// Per-island reduction. Replaces the whole-graph cub::DeviceReduce, which
    /// imposed a grid-wide barrier every iteration and produced a single
    /// residual that could not distinguish a converged island from a starved
    /// one.
    void reduceByIsland(
        cudaStream_t stream,
        const AngLin* values,
        const std::uint32_t* island,
        const std::uint32_t* islandSkip,
        const std::uint32_t* activeList,
        std::uint32_t whichCount,
        std::uint32_t launchCap,
        float* result)
    {
        const std::uint32_t slots = reductionSlots(launchCap);
        checkCuda(
            cudaMemsetAsync(
                m_reduceSlots,
                0,
                sizeof(float) * static_cast<std::size_t>(m_islandCount) * slots,
                stream),
            "clear per-island reduction");
        m_kernelProfile.begin("accumulateSquaredByIsland", stream);
        accumulateSquaredByIsland<<<
            (launchCap + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            stream>>>(
            values,
            island,
            islandSkip,
            m_reduceSlots,
            activeList,
            m_activeCounts,
            whichCount,
            slots);
        m_kernelProfile.end(stream);
        m_kernelProfile.begin("finalizeIslandReduction", stream);
        finalizeIslandReduction<<<
            (m_islandCount + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            stream>>>(m_reduceSlots, result, m_islandCount, slots);
        m_kernelProfile.end(stream);
    }

    /// How many accumulators to spread each island's atomics across.
    ///
    /// Padding buys contention relief proportional to the slot count and costs
    /// a second pass proportional to islandCount * slots. It is worth it only
    /// when islands are large: at ~2,760 elements per island (the intact city)
    /// 32 slots turn 2,760-way serialization into 86-way; at ~4 elements per
    /// island (fully fractured) there is no contention to relieve and the wide
    /// second pass would be pure loss. Round down to a power of two so the
    /// kernel can mask instead of divide.
    std::uint32_t reductionSlots(std::uint32_t elements) const
    {
        if (m_islandCount == 0)
        {
            return 1u;
        }
        const std::uint32_t perIsland = elements / m_islandCount;
        std::uint32_t slots = 1u;
        // One slot per 8 elements, so a slot is never contended by fewer
        // atomics than the second pass costs to read it back.
        while (slots < kReductionSlotsMax && slots * 8u < perIsland)
        {
            slots <<= 1u;
        }
        return slots;
    }

    void rightMultiply(
        cudaStream_t stream, const AngLin* bonds, AngLin* nodes,
        const std::uint32_t* islandSkip)
    {
        if (gatherRightMultiplyEnabled())
        {
            // No memset: every active node writes its own slot. No scaleNodes
            // either: the inverse-inertia scale is folded into the write. Two
            // graph nodes become one, and twelve global float atomics per
            // bond become zero.
            m_kernelProfile.begin("gatherRightMultiply", stream);
            gatherRightMultiply<<<
                (m_graphNodeCap + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                stream>>>(
                nodes,
                bonds,
                m_nodeBondBegin,
                m_nodeBondRef,
                m_offset0,
                m_offset1,
                m_health,
                m_colScales,
                m_bondIsland,
                islandSkip,
                m_inertia,
                m_activeNodes,
                m_activeCounts);
            m_kernelProfile.end(stream);
            return;
        }
        checkCuda(
            cudaMemsetAsync(
                nodes,
                0,
                sizeof(AngLin) * m_nodeCount,
                stream),
            "clear node product");
        m_kernelProfile.begin("couplingRightMultiply", stream);
        couplingRightMultiply<<<
            (m_graphBondCap + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            stream>>>(
            nodes,
            bonds,
            m_node0,
            m_node1,
            m_offset0,
            m_offset1,
            m_health,
            m_colScales,
            m_bondIsland,
            islandSkip,
            m_activeBonds,
            m_activeCounts);
        m_kernelProfile.end(stream);
        m_kernelProfile.begin("scaleNodes", stream);
        scaleNodes<<<
            (m_graphNodeCap + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            stream>>>(
            nodes,
            m_inertia,
            m_activeNodes,
            m_activeCounts);
        m_kernelProfile.end(stream);
    }

    /// Sum of |impulse| over all bonds, read back from the device.
    /// Debug only: a full D2H copy, gated on BLAST_ISLAND_TRACE.
    double debugImpulseMagnitude()
    {
        m_dbgImpulses.resize(m_bondCount);
        if (cudaMemcpy(m_dbgImpulses.data(), m_impulses,
                       sizeof(AngLin) * m_bondCount,
                       cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            cudaGetLastError();
            return -1.0;
        }
        double total = 0.0;
        for (const AngLin& v : m_dbgImpulses)
        {
            total += (std::fabs(double(v.linear.x)) + std::fabs(double(v.linear.y))
                   + std::fabs(double(v.linear.z))) * double(m_lengthScale * m_massScale);
        }
        return total;
    }

    /// One CG iteration. Parameterised on the stream so it can be captured
    /// either into the main graph (unrolled fallback) or into a conditional
    /// while-node's body graph, which is what gives device-side early exit.
    /// One node-space CGLS iteration. Same scalars, same stopping test, one
    /// fewer per-island reduction, and no bond-length vector in the loop.
    void launchNodeSpaceIteration(
        const ExtStressGpuSolveParams& params,
        const std::uint32_t* islandSkip,
        cudaStream_t stream,
        cudaGraphConditionalHandle loopHandle)
    {
        const std::uint32_t nodeBlocks =
            (m_graphNodeCap + kBlockSize - 1) / kBlockSize;
        const std::uint32_t islandBlocks =
            (m_islandCount + kBlockSize - 1) / kBlockSize;
        const std::uint32_t slots = reductionSlots(m_graphNodeCap);

        // w = L rho, and z_sq = ||W rho||^2 out of the same pass.
        checkCuda(
            cudaMemsetAsync(
                m_reduceSlots, 0,
                sizeof(float) * static_cast<std::size_t>(m_islandCount) * slots, stream),
            "clear node-space z_sq");
        m_kernelProfile.begin("nodeSpaceMatvec", stream);
        nodeSpaceMatvec<<<nodeBlocks, kBlockSize, 0, stream>>>(
            m_nsW, m_residual, m_inertia, m_nodeBondBegin, m_nodeBondRef,
            m_node0, m_node1, m_offset0, m_offset1, m_health, m_colScales,
            m_bondIsland, islandSkip, m_nodeIsland, m_islandActive, skipConvergedEnabled(),
            m_reduceSlots, slots, m_activeNodes, m_activeCounts,
            m_iteration, 0u);
        m_kernelProfile.end(stream);
        m_kernelProfile.begin("finalizeAndCheckConvergence", stream);
        finalizeAndCheckConvergence<<<islandBlocks, kBlockSize, 0, stream>>>(
            m_reduceSlots, m_gradientSquared, slots,
            m_islandActive, m_islandConverged, m_deltaSquared,
            m_blockActiveCounts, m_islandCount);
        m_kernelProfile.end(stream);

        // Preconditioned form: the direction is built from g = N w rather than
        // from rho, the numerator becomes gamma = w^T g, and q therefore needs
        // its own matvec (L g) because it can no longer reuse w. That second
        // matvec is the price of preconditioning here -- see the derivation in
        // the plan; it is why block-Jacobi has to more than halve the iteration
        // count merely to break even.
        const AngLin* directionSource = m_residual;
        const float* numerator = m_gradientSquared;
        float* numeratorPrev = m_previousGradientSquared;
        if (jacobiEnabled())
        {
            checkCuda(
                cudaMemsetAsync(
                    m_reduceSlots, 0,
                    sizeof(float) * static_cast<std::size_t>(m_islandCount) * slots, stream),
                "clear node-space gamma");
            m_kernelProfile.begin("nodeSpaceApplyJacobi", stream);
            nodeSpaceApplyJacobi<<<nodeBlocks, kBlockSize, 0, stream>>>(
                m_nsG, m_nsW, m_nsJacobi, m_nodeIsland, m_islandActive,
                m_reduceSlots, slots, m_activeNodes, m_activeCounts);
            m_kernelProfile.end(stream);
            m_kernelProfile.begin("finalizeIslandReduction", stream);
            finalizeIslandReduction<<<islandBlocks, kBlockSize, 0, stream>>>(
                m_reduceSlots, m_nsGamma, m_islandCount, slots);
            m_kernelProfile.end(stream);
            directionSource = m_nsG;
            numerator = m_nsGamma;
            numeratorPrev = m_nsGammaPrev;

            // q's own matvec: L g, written into m_nsW2.
            m_kernelProfile.begin("nodeSpaceMatvecG", stream);
            nodeSpaceMatvec<<<nodeBlocks, kBlockSize, 0, stream>>>(
                m_nsW2, m_nsG, m_inertia, m_nodeBondBegin, m_nodeBondRef,
                m_node0, m_node1, m_offset0, m_offset1, m_health, m_colScales,
                m_bondIsland, islandSkip, m_nodeIsland, m_islandActive, skipConvergedEnabled(),
                nullptr, slots, m_activeNodes, m_activeCounts,
                m_iteration, 0u);
            m_kernelProfile.end(stream);
        }
        checkCuda(
            cudaMemsetAsync(
                m_reduceSlots, 0,
                sizeof(float) * static_cast<std::size_t>(m_islandCount) * slots, stream),
            "clear node-space q_sq");
        m_kernelProfile.begin("nodeSpaceUpdateDirection", stream);
        nodeSpaceUpdateDirection<<<nodeBlocks, kBlockSize, 0, stream>>>(
            m_nsPi, m_nsQ, directionSource, jacobiEnabled() ? m_nsW2 : m_nsW,
            numerator, numeratorPrev,
            m_nodeIsland, m_islandActive, m_reduceSlots, slots,
            m_activeNodes, m_activeCounts);
        m_kernelProfile.end(stream);

        // NOTE: the periodic explicit refresh of q = L pi was removed here.
        // Once ||q||^2 is accumulated by the direction update that WRITES q, a
        // later refresh would replace q while leaving its norm stale, so alpha
        // would be computed from a vector that no longer exists. It was also
        // not earning its place: at every-8 it cost ~6% of the solve and closed
        // only ~2% of the residual gap against the CPU, which is what showed
        // the recurrence is not the main source of that gap.
        // ||q||^2 was accumulated by the direction update itself; just collapse
        // the padded slots.
        m_kernelProfile.begin("finalizeAndRetire", stream);
        finalizeAndRetire<<<islandBlocks, kBlockSize, 0, stream>>>(
            m_reduceSlots, m_projectedDirectionSquared, slots,
            m_islandActive, numeratorPrev, numerator,
            m_status, m_blockActiveCounts, islandBlocks,
            m_iteration, m_islandCount, loopHandle, params.maxIterations);
        m_kernelProfile.end(stream);

        m_kernelProfile.begin("nodeSpaceUpdateSolution", stream);
        nodeSpaceUpdateSolution<<<nodeBlocks, kBlockSize, 0, stream>>>(
            m_iteration, params.maxIterations, m_nsMu, m_residual, m_nsPi, m_nsQ,
            numerator, m_projectedDirectionSquared,
            m_nodeIsland, m_islandActive, m_activeNodes, m_activeCounts);
        m_kernelProfile.end(stream);
    }

    void launchIterationBody(
        const ExtStressGpuSolveParams& params,
        const std::uint32_t* islandSkip,
        std::uint32_t maxCap,
        cudaStream_t stream,
        cudaGraphConditionalHandle loopHandle)
    {
        if (nodeSpaceEnabled())
        {
            launchNodeSpaceIteration(params, islandSkip, stream, loopHandle);
            return;
        }

            m_kernelProfile.begin("couplingLeftMultiply", stream);
            couplingLeftMultiply<<<
                (m_graphBondCap + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                stream>>>(
                m_gradient,
                m_residual,
                m_inertia,
                m_node0,
                m_node1,
                m_offset0,
                m_offset1,
                m_health,
                m_colScales,
                m_bondIsland,
                islandSkip,
                m_activeBonds,
                m_activeCounts);
            m_kernelProfile.end(stream);
            const std::uint32_t islandBlocks =
                (m_islandCount + kBlockSize - 1) / kBlockSize;
            reduceByIsland(stream, 
                m_gradient, m_bondIsland, islandSkip, m_activeBonds, 0u,
                m_graphBondCap, m_gradientSquared);
            m_kernelProfile.begin("checkConvergencePerIsland", stream);
            checkConvergencePerIsland<<<islandBlocks, kBlockSize, 0, stream>>>(
                m_islandActive,
                m_islandConverged,
                m_gradientSquared,
                m_deltaSquared,
                m_blockActiveCounts,
                m_islandCount);
            m_kernelProfile.end(stream);
            m_kernelProfile.begin("updateDirectionPerIsland", stream);
            updateDirectionPerIsland<<<
                (m_graphBondCap + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                stream>>>(
                m_direction,
                m_gradient,
                m_gradientSquared,
                m_previousGradientSquared,
                m_bondIsland,
                m_islandActive,
                m_activeBonds,
                m_activeCounts,
                m_iteration);
            m_kernelProfile.end(stream);

            rightMultiply(stream, m_direction, m_projectedDirection, islandSkip);
            reduceByIsland(stream, 
                m_projectedDirection,
                m_nodeIsland,
                islandSkip,
                m_activeNodes,
                1u,
                m_graphNodeCap,
                m_projectedDirectionSquared);
            m_kernelProfile.begin("retireDegenerateIslands", stream);
            retireDegenerateIslands<<<islandBlocks, kBlockSize, 0, stream>>>(
                m_islandActive,
                m_projectedDirectionSquared,
                m_previousGradientSquared,
                m_gradientSquared,
                m_status,
                m_blockActiveCounts,
                islandBlocks,
                m_iteration,
                m_islandCount,
                loopHandle,
                params.maxIterations);
            m_kernelProfile.end(stream);
            m_kernelProfile.begin("updateSolutionAndResidualPerIsland", stream);
            updateSolutionAndResidualPerIsland<<<
                (maxCap + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                stream>>>(
                m_iteration,
                params.maxIterations,
                m_impulses,
                m_residual,
                m_direction,
                m_projectedDirection,
                m_gradientSquared,
                m_projectedDirectionSquared,
                m_islandActive,
                m_bondIsland,
                m_nodeIsland,
                m_activeBonds,
                m_activeNodes,
                m_activeCounts);
            m_kernelProfile.end(stream);
            }

    /// Run the CG loop, with device-side early exit when it is available.
    ///
    /// Two shapes, and the difference is the whole point:
    ///
    /// - CONDITIONAL (preferred): one `cudaGraphCondTypeWhile` node whose body
    ///   is a single copy of the iteration. The device decides each round
    ///   whether to go again, so a scene whose islands all converged at
    ///   iteration 3 costs three iterations, not maxIterations. Requires that
    ///   we are inside stream capture, which is the normal path.
    ///
    /// - UNROLLED (fallback): maxIterations copies of the body, as before.
    ///   Used when not capturing (the eager kernel-profile path), and when the
    ///   conditional path is switched off. Bit-identical results, just slower
    ///   whenever the solve converges early.
    ///
    /// Both call the same launchIterationBody, so there is one implementation
    /// of the iteration and no chance of the two shapes drifting apart.
    void launchConditionalLoop(
        const ExtStressGpuSolveParams& params,
        const std::uint32_t* islandSkip,
        std::uint32_t maxCap)
    {
        if (conditionalLoopEnabled())
        {
            cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
            unsigned long long captureId = 0;
            cudaGraph_t capturing = nullptr;
            const cudaGraphNode_t* deps = nullptr;
            std::size_t depCount = 0;
            if (cudaStreamGetCaptureInfo(
                    m_stream, &captureStatus, &captureId, &capturing, &deps, &depCount)
                    == cudaSuccess
                && captureStatus == cudaStreamCaptureStatusActive
                && capturing != nullptr)
            {
                if (launchConditionalLoopCaptured(
                        params, islandSkip, maxCap, capturing, deps, depCount))
                {
                    return;
                }
                // Fall through to the unrolled form. Any failure here is a
                // performance regression, never a wrong answer, because the
                // two shapes compute the same thing.
                cudaGetLastError();
            }
        }

        for (std::uint32_t iteration = 0; iteration < params.maxIterations; ++iteration)
        {
            // The unrolled form still reads the counter from device memory, so
            // the two paths share one kernel signature. Bump it per copy.
            launchIterationBody(params, islandSkip, maxCap, m_stream, 0);
        }
    }

    /// Build the while-node. Returns false if anything is unsupported, leaving
    /// the caller to use the unrolled form.
    bool launchConditionalLoopCaptured(
        const ExtStressGpuSolveParams& params,
        const std::uint32_t* islandSkip,
        std::uint32_t maxCap,
        cudaGraph_t capturing,
        const cudaGraphNode_t* deps,
        std::size_t depCount)
    {
        cudaGraphConditionalHandle handle = 0;
        // Default 1: the body must run at least once, exactly like a do/while.
        // A solve that is already converged still has to produce its outputs.
        if (cudaGraphConditionalHandleCreate(
                &handle, capturing, 1, cudaGraphCondAssignDefault) != cudaSuccess)
        {
            return false;
        }

        cudaGraphNodeParams nodeParams{};
        nodeParams.type = cudaGraphNodeTypeConditional;
        nodeParams.conditional.handle = handle;
        nodeParams.conditional.type = cudaGraphCondTypeWhile;
        nodeParams.conditional.size = 1;

        cudaGraphNode_t whileNode = nullptr;
        if (cudaGraphAddNode(&whileNode, capturing, deps, depCount, &nodeParams)
                != cudaSuccess
            || nodeParams.conditional.phGraph_out == nullptr)
        {
            return false;
        }

        // Everything captured after this point must depend on the while node,
        // or the epilogue would be free to run alongside the loop.
        if (cudaStreamUpdateCaptureDependencies(
                m_stream, &whileNode, 1, cudaStreamSetCaptureDependencies) != cudaSuccess)
        {
            return false;
        }

        cudaGraph_t body = nodeParams.conditional.phGraph_out[0];
        if (cudaStreamBeginCaptureToGraph(
                m_bodyStream, body, nullptr, nullptr, 0,
                cudaStreamCaptureModeThreadLocal) != cudaSuccess)
        {
            return false;
        }

        // Check the condition every kChunk iterations instead of every one.
        // The while-node's per-execution dispatch cost showed up as a stable
        // +3.7% on scenes that run their full budget (0/7 pairs faster,
        // counterbalanced n=8); amortising it over a chunk removes most of
        // that while still exiting ~8x earlier than the budget on scenes that
        // converge immediately. Exactness comes from the guard in
        // updateSolutionAndResidualPerIsland, so kChunk is purely a cost knob.
        const std::uint32_t chunk = conditionalLoopChunk();
        for (std::uint32_t k = 0; k < chunk; ++k)
        {
            launchIterationBody(params, islandSkip, maxCap, m_bodyStream, handle);
        }

        cudaGraph_t bodyOut = nullptr;
        if (cudaStreamEndCapture(m_bodyStream, &bodyOut) != cudaSuccess)
        {
            return false;
        }
        ++m_statConditionalLoops;
        return true;
    }

    void launchSolve(const ExtStressGpuSolveParams& params)
    {
        const bool warmStart = params.warmStart && m_hasWarmStart;

        const float reciprocalLengthScale = 1.0f / m_lengthScale;
        const float reciprocalMassScale = 1.0f / m_massScale;
        const float reciprocalLinearImpulseScale =
            reciprocalLengthScale * reciprocalMassScale;
        const float reciprocalAngularImpulseScale =
            reciprocalLengthScale * reciprocalLinearImpulseScale;
        const std::uint32_t maxCap = std::max(m_graphNodeCap, m_graphBondCap);
        // Null when the feature is off, so every skip test below compiles down
        // to one comparison against a register and the pre-skip behaviour is
        // restored exactly rather than approximately. It is a capture-time
        // argument, so flipping the flag recaptures the graph (graphMatches);
        // the mask it points at changes every frame and does not.
        const std::uint32_t* islandSkip =
            (params.skipSettledIslands && warmStart) ? m_islandSkip : nullptr;

        m_kernelProfile.begin("initializeSolve", m_stream);
        initializeSolve<<<
            (maxCap + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            m_input,
            m_rhs,
            m_residual,
            m_impulses,
            m_inertia,
            m_nodeIsland,
            m_bondIsland,
            islandSkip,
            m_activeNodes,
            m_activeBonds,
            m_activeCounts,
            reciprocalLengthScale,
            reciprocalLinearImpulseScale,
            reciprocalAngularImpulseScale,
            warmStart);
        m_kernelProfile.end(m_stream);
        // Deliberately NOT guarded on warmStart. These two are a no-op on a
        // cold start, exactly: initializeSolve has just written impulses = {},
        // gatherRightMultiply WRITES each active node's slot rather than
        // accumulating into it (see the comment in rightMultiply), so
        // m_projectedDirection comes out zero, and subtractResidual then
        // subtracts zero from the residual.
        //
        // Running them unconditionally is what makes warmStart a pure kernel
        // ARGUMENT rather than a change to the graph's node set. That is worth
        // two no-op kernels on cold-start ticks: with the branch in place, a
        // warm-start flip forced a full graph re-instantiation, and
        // BLAST_GPU_WHOLE_RESET_ON_TOPOLOGY flips it on every fracture tick --
        // measured at 33.6% of solves even after cudaGraphExecUpdate had
        // removed every other recapture driver.
        //
        // That flag cannot simply be turned off to avoid this: it is load
        // bearing for the scene. Without it the suite fails T2 "one shot
        // pulverizes" (3726 bonds vs a <=2028 band), T2 bodies (1087 vs <=707)
        // and T4 awake-declined (0.42 vs <=0.10). So the graph has to tolerate
        // the flip instead.
        if (warmStart || stableGraphEnabled())
        {
            rightMultiply(m_stream, m_impulses, m_projectedDirection, islandSkip);
            m_kernelProfile.begin("subtractResidual", m_stream);
            subtractResidual<<<
                (m_graphNodeCap + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(
                m_residual, m_projectedDirection, m_activeNodes, m_activeCounts);
            m_kernelProfile.end(m_stream);
        }

        // Tolerance is relative to each island's own load, not the whole
        // graph's: a small island next to a heavily loaded one would otherwise
        // inherit a threshold it can never meaningfully meet.
        reduceByIsland(
            m_stream,
            m_rhs, m_nodeIsland, islandSkip, m_activeNodes, 1u, m_graphNodeCap,
            m_gradientSquared);
        m_kernelProfile.begin("setTolerancePerIsland", m_stream);
        setTolerancePerIsland<<<
            (m_islandCount + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            m_deltaSquared,
            m_islandActive,
            m_islandConverged,
            islandSkip,
            m_gradientSquared,
            params.tolerance,
            m_islandCount);
        m_kernelProfile.end(m_stream);
        m_kernelProfile.begin("initializeStatus", m_stream);
        initializeStatus<<<1, 1, 0, m_stream>>>(
            m_status, m_iteration, params.maxIterations);
        m_kernelProfile.end(m_stream);

        if (nodeSpaceEnabled() && jacobiEnabled() && !m_jacobiBuilt)
        {
            m_kernelProfile.begin("nodeSpaceBuildJacobi", m_stream);
            nodeSpaceBuildJacobi<<<
                (m_nodeCount + kBlockSize - 1) / kBlockSize, kBlockSize, 0, m_stream>>>(
                m_nsJacobi, m_inertia, m_nodeBondBegin, m_nodeBondRef,
                m_node0, m_node1, m_offset0, m_offset1, m_health, m_colScales,
                m_nodeCount, m_bondCount);
            m_kernelProfile.end(m_stream);
            m_jacobiBuilt = true;
        }
        if (nodeSpaceEnabled() && jacobiEnabled())
        {
            // gamma_prev must be zero at iteration 0 so beta is zero there.
            // Left uninitialised it is whatever the allocator handed back, and
            // if those bits happen to be NaN then beta is NaN, pi = g + NaN*0
            // is NaN, and mu accumulates the poison for the rest of the solve.
            // The unpreconditioned path never showed this because it reuses
            // m_previousGradientSquared, which earlier solves have already
            // filled with real values.
            checkCuda(
                cudaMemsetAsync(m_nsGammaPrev, 0,
                                sizeof(float) * m_islandCapacity, m_stream),
                "clear node-space gamma prev");
            checkCuda(
                cudaMemsetAsync(m_nsGamma, 0,
                                sizeof(float) * m_islandCapacity, m_stream),
                "clear node-space gamma");
        }
        if (nodeSpaceEnabled())
        {
            // mu accumulates the whole correction, so it must start at zero
            // every solve; pi and q are recurrences and must not inherit the
            // previous solve's Krylov state.
            m_kernelProfile.begin("nodeSpaceReset", m_stream);
            nodeSpaceReset<<<
                (m_nodeCount + kBlockSize - 1) / kBlockSize,
                kBlockSize, 0, m_stream>>>(
                m_nsMu, m_nsPi, m_nsQ, m_nsG, m_nodeCount);
            m_kernelProfile.end(m_stream);
        }

        launchConditionalLoop(params, islandSkip, maxCap);

        if (nodeSpaceEnabled())
        {
            // lambda = lambda0 + W mu. One C^T D pass over bonds, once per
            // solve, instead of a bond-length update every iteration.
            m_kernelProfile.begin("nodeSpaceApplySolution", m_stream);
            nodeSpaceApplySolution<<<
                (m_graphBondCap + kBlockSize - 1) / kBlockSize,
                kBlockSize, 0, m_stream>>>(
                m_impulses, m_nsMu, m_inertia, m_node0, m_node1,
                m_offset0, m_offset1, m_health, m_colScales, m_bondIsland, islandSkip,
                m_activeBonds, m_activeCounts);
            m_kernelProfile.end(m_stream);
        }


        // No unscale pass. Impulses stay in solver-scaled units on the device;
        // the three places that read them for the OUTSIDE world apply the scale
        // themselves (applyStressDamage, bondStressWalk, and the host repack in
        // finishImpulseReadback). See initializeSolve.
        //
        // Column scaling (the CPU processor's compliance weights, colScale) is
        // part of the stored variable too: the device holds J' = J / colScale,
        // the operator is C*S on the way out and S*C^T on the way back, and the
        // same three consumers multiply by colScale when converting to physical.
        checkCuda(
            cudaMemsetAsync(
                m_brokenCount,
                0,
                sizeof(std::uint32_t),
                m_stream),
            "clear broken bond count");
        // Deliberately NOT gated on the skip mask. A settled island's stress
        // is unchanged, not absent: re-solving it would produce the same
        // impulses and charge the same damage, so charging it here is what
        // makes skipping observationally identical. This is also what the CPU
        // path does -- its updateBondStress runs over every bond whether or
        // not the island was skipped.
        if (params.applyDamage)
        {
            m_kernelProfile.begin("applyStressDamage", m_stream);
            applyStressDamage<<<
                (m_bondCount + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(
                m_bendGainMax,
                m_impulses,
                m_colScales,
                m_normals,
                m_areas,
                m_nodeDistances,
                m_bondMaterials,
                m_materials,
                m_materialCount,
                m_health,
                m_brokenBonds,
                m_brokenCount,
                m_bondCount,
                m_lengthScale * m_massScale,
                m_lengthScale * m_lengthScale * m_massScale);
            m_kernelProfile.end(m_stream);
        }
    }

    /// Smallest power-of-two launch size covering `active`, with one bucket
    /// of headroom so a few more islands waking does not recapture the graph.
    /// Never below 1024 (recaptures at that scale cost more than the waste)
    /// and never above the raw array size.
    static std::uint32_t launchCapacity(std::uint32_t active, std::uint32_t full)
    {
        if (active >= full)
        {
            return full;
        }
        std::uint32_t cap = 1024u;
        while (cap < active)
        {
            cap <<= 1u;
        }
        cap <<= 1u;
        return std::min(cap, full);
    }

    /// The subset of graphMatches that changes the graph's NODE SET rather
    /// than its node parameters. When this holds, a changed solve can be
    /// patched into the existing exec instead of rebuilt. Deliberately excludes
    /// skipSettledIslands: it only selects whether the kernels receive
    /// m_islandSkip or nullptr, and a pointer arg is a parameter like any other.
    bool graphTopologyMatches(const ExtStressGpuSolveParams& params, bool warmStart) const
    {
        return m_graphExec != nullptr
            && m_graph != nullptr
            && (stableGraphEnabled() || m_graphWarmStart == warmStart)
            && m_graphParams.maxIterations == params.maxIterations
            && m_graphParams.applyDamage == params.applyDamage;
    }

    bool graphMatches(const ExtStressGpuSolveParams& params, bool warmStart) const
    {
        return m_graphExec
            && m_graphWarmStart == warmStart
            && m_graphParams.maxIterations == params.maxIterations
            && m_graphParams.tolerance == params.tolerance
            && m_graphParams.applyDamage == params.applyDamage
            // Capture-time: it selects whether the kernels are handed the skip
            // mask at all. The mask's CONTENTS change every frame and are read
            // from device memory, so a settling scene never recaptures.
            && m_graphParams.skipSettledIslands == params.skipSettledIslands
            // The active lists' CONTENTS are device-side and free to change;
            // the launch dimensions that must cover them are baked. Recapture
            // when the lists outgrow the baked caps (mandatory -- threads past
            // the cap simply do not exist) or shrank far below them (waste).
            && (fullLaunchCapacity()
                || (m_graphBondCap >= m_activeBondCount
                    && m_graphNodeCap >= m_activeNodeCount
                    && m_graphBondCap <= 4u * std::max(
                           launchCapacity(m_activeBondCount, m_bondCount), 1024u)
                    && m_graphNodeCap <= 4u * std::max(
                           launchCapacity(m_activeNodeCount, m_nodeCount), 1024u)));
    }

    void executeSolve(const ExtStressGpuSolveParams& params)
    {
        const bool warmStart = params.warmStart && m_hasWarmStart;
        if (kernelProfileEnabled())
        {
            // Eager launches, one event pair per kernel. The graph is bypassed
            // on purpose: comparing this total against the graph-replay total
            // prices the launch overhead the graph removes, which is the one
            // number source-reading cannot supply.
            m_kernelProfile.active = true;
            // The launch caps are normally computed in the capture branch
            // below; the eager path skips it, so they must be set here or
            // every grid is sized from a stale (or zero) cap.
            m_graphBondCap = fullLaunchCapacity()
                ? m_bondCount : launchCapacity(m_activeBondCount, m_bondCount);
            m_graphNodeCap = fullLaunchCapacity()
                ? m_nodeCount : launchCapacity(m_activeNodeCount, m_nodeCount);
            launchSolve(params);
            ++m_profiledSolves;
            return;
        }
        if (m_graphParamsDirty || !graphMatches(params, warmStart))
        {
            m_graphParamsDirty = false;
            m_graphBondCap = fullLaunchCapacity()
                ? m_bondCount : launchCapacity(m_activeBondCount, m_bondCount);
            m_graphNodeCap = fullLaunchCapacity()
                ? m_nodeCount : launchCapacity(m_activeNodeCount, m_nodeCount);
            if (graphUpdateEnabled() && graphTopologyMatches(params, warmStart))
            {
                const auto updateStart = std::chrono::steady_clock::now();
                cudaGraph_t fresh = nullptr;
                bool captured =
                    cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeThreadLocal)
                    == cudaSuccess;
                if (captured)
                {
                    launchSolve(params);
                    captured = cudaStreamEndCapture(m_stream, &fresh) == cudaSuccess
                        && fresh != nullptr;
                }
                if (captured)
                {
                    cudaGraphExecUpdateResultInfo info{};
                    const cudaError_t rc = cudaGraphExecUpdate(m_graphExec, fresh, &info);
                    if (rc == cudaSuccess && info.result == cudaGraphExecUpdateSuccess)
                    {
                        cudaGraphDestroy(m_graph);
                        m_graph = fresh;
                        m_graphParams = params;
                        m_graphWarmStart = warmStart;
                        ++m_statGraphUpdates;
                        m_statUpdateMs += std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now()
                                              - updateStart).count();
                        checkCuda(cudaGraphLaunch(m_graphExec, m_stream),
                                  "launch patched solver graph");
                        return;
                    }
                    cudaGraphDestroy(fresh);
                }
                // Do not let a failed attempt poison the fallback's error state.
                cudaGetLastError();
            }
            if (m_graphExec)
            {
                checkCuda(cudaGraphExecDestroy(m_graphExec), "destroy old solver graph exec");
                m_graphExec = nullptr;
            }
            if (m_graph)
            {
                checkCuda(cudaGraphDestroy(m_graph), "destroy old solver graph");
                m_graph = nullptr;
            }
            const auto recaptureStart = std::chrono::steady_clock::now();
            checkCuda(
                cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeThreadLocal),
                "begin solver graph capture");
            launchSolve(params);
            checkCuda(
                cudaStreamEndCapture(m_stream, &m_graph),
                "end solver graph capture");
            const auto captureDone = std::chrono::steady_clock::now();
            checkCuda(
                (++m_statGraphRecaptures,
                 cudaGraphInstantiate(&m_graphExec, m_graph, 0)),
                "instantiate solver graph");
            m_statCaptureMs += std::chrono::duration<double, std::milli>(
                                   captureDone - recaptureStart).count();
            m_statInstantiateMs += std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - captureDone).count();
            m_graphParams = params;
            m_graphWarmStart = warmStart;
        }
        checkCuda(cudaGraphLaunch(m_graphExec, m_stream), "launch solver graph");
    }

    std::uint32_t m_nodeCount;
    std::uint32_t m_bondCount;
    CUcontext m_cudaContext{nullptr};
    float m_massScale{1.0f};
    float m_lengthScale{1.0f};
    bool m_hasWarmStart{false};
    std::vector<AngLin> m_dbgImpulses;
    std::vector<std::uint32_t> m_dbgActive;
    std::vector<std::uint32_t> m_dbgConverged;
    double m_dbgSumBefore{0.0};
    bool m_dbgWarm{false};
    ExtStressGpuTelemetry m_telemetry{};

    PinnedVector<std::uint32_t> m_hostNode0;
    PinnedVector<std::uint32_t> m_hostNode1;
    PinnedVector<Vec4> m_hostOffset0;
    PinnedVector<Vec4> m_hostOffset1;
    PinnedVector<Inertia> m_hostInertia;
    PinnedVector<Vec4> m_hostNormals;
    PinnedVector<float> m_hostAreas;
    /// Per-bond compliance weight, the CPU processor's column scale.
    PinnedVector<float> m_hostColScale;
    PinnedVector<float> m_hostNodeDistances;
    PinnedVector<float> m_hostHealth;
    PinnedVector<std::uint32_t> m_hostBondMaterials;
    PinnedVector<ExtStressGpuMaterial> m_hostMaterials;

    std::uint32_t* m_node0{nullptr};
    std::uint32_t* m_node1{nullptr};
    Vec4* m_offset0{nullptr};
    Vec4* m_offset1{nullptr};
    Inertia* m_inertia{nullptr};
    Vec4* m_normals{nullptr};
    /// Bending policy, taken from solve params so the device matches the host.
    float m_bendGainMax{0.0f};
    float* m_areas{nullptr};
    // Per-bond compliance weight, uploaded verbatim from the CPU processor.
    float* m_colScales{nullptr};
    float* m_nodeDistances{nullptr};
    float* m_health{nullptr};
    std::uint32_t* m_bondMaterials{nullptr};
    ExtStressGpuMaterial* m_materials{nullptr};
    std::uint32_t m_materialCount{0};
    std::uint32_t* m_brokenBonds{nullptr};
    std::uint32_t* m_brokenCount{nullptr};
    ExtStressGpuImpulse* m_input{nullptr};
    AngLin* m_impulses{nullptr};
    AngLin* m_rhs{nullptr};
    AngLin* m_gradient{nullptr};
    AngLin* m_direction{nullptr};
    AngLin* m_residual{nullptr};
    AngLin* m_projectedDirection{nullptr};
    float* m_reductionInput{nullptr};
    float* m_gradientSquared{nullptr};
    /// Padded scratch for the per-island reduction; see accumulateSquaredByIsland.
    float* m_reduceSlots{nullptr};
    /// Device-side CG loop counter; advanced by retireDegenerateIslands.
    std::uint32_t* m_iteration{nullptr};
    /// Scratch reused across topology rebuilds; see computeIslands /
    /// buildNodeBondCsr / groupBondsByIsland.
    std::vector<std::uint32_t> m_hostRootStamp;
    std::uint32_t m_rootIslandGeneration{0};
    std::vector<std::uint32_t> m_csrCursor;
    std::vector<std::uint32_t> m_groupCursor;
    /// Bond removals since the island partition was last rebuilt.
    std::uint32_t m_removalsSinceRepartition{0};
    /// False when swap-with-last moved bonds out from under m_bondsByIsland.
    bool m_bondsByIslandValid{true};
    /// Endpoint pairs of the bonds removed since the last topology apply.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> m_splitChecks;
    std::vector<std::uint32_t> m_splitVisited;
    std::vector<std::uint32_t> m_splitQueue;
    std::uint32_t m_splitStamp{0};
    /// Bond slots rewritten since the last topology upload.
    std::vector<std::uint32_t> m_changedBondSlots;
    PinnedVector<std::uint32_t> m_deltaSlots;
    PinnedVector<BondDelta> m_deltaValues;
    std::uint32_t* m_devDeltaSlots{nullptr};
    BondDelta* m_devDeltaValues{nullptr};
    std::uint32_t m_deltaCapacity{0};
    /// Per-node inertia never changes after prepare(); upload it once.
    bool m_inertiaUploaded{false};
    AngLin* m_nsPi{nullptr};
    AngLin* m_nsQ{nullptr};
    AngLin* m_nsW{nullptr};
    AngLin* m_nsMu{nullptr};
    AngLin* m_nsG{nullptr};
    AngLin* m_nsW2{nullptr};
    float* m_nsJacobi{nullptr};
    float* m_nsGamma{nullptr};
    float* m_nsGammaPrev{nullptr};
    bool m_jacobiBuilt{false};
    /// Position of each bond's two CSR refs; lets a removal patch in O(1).
    std::vector<std::uint32_t> m_hostBondRefPos;
    std::vector<std::uint32_t> m_changedRefSlots;
    PinnedVector<std::uint32_t> m_refSlots;
    PinnedVector<std::uint32_t> m_refValues;
    std::uint32_t* m_devRefSlots{nullptr};
    std::uint32_t* m_devRefValues{nullptr};
    std::uint32_t m_refDeltaCapacity{0};
    std::size_t m_deadRefCount{0};
    bool m_csrValid{false};
    std::vector<std::uint32_t> m_changedNodeIslandSlots;
    PinnedVector<std::uint32_t> m_nodeIslandSlots;
    PinnedVector<std::uint32_t> m_nodeIslandValues;
    std::uint32_t* m_devNodeIslandSlots{nullptr};
    std::uint32_t* m_devNodeIslandValues{nullptr};
    std::uint32_t m_nodeIslandDeltaCapacity{0};
    std::uint64_t m_statLocalSplits{0};
    std::uint32_t m_islandCountAtRebuild{0};
    std::vector<std::uint32_t> m_affectedIslands;
    std::vector<std::uint32_t> m_affectedNodes;
    std::map<std::uint32_t, std::uint32_t> m_rootRemap;
    std::vector<std::uint32_t> m_verifyNodeIsland;
    std::vector<std::uint32_t> m_verifyBondIsland;
    std::uint64_t m_verifyCalls{0};
    std::uint64_t m_statConditionalLoops{0};
    /// Per-island conjugate-gradient state. Islands are disconnected
    /// components: they must converge and step independently, matching the CPU
    /// solver's per-island sub-solves.
    std::uint32_t* m_islandActive{nullptr};
    /// Set when an island reaches tolerance; kept separate from `active`
    /// because a degenerate island also goes inactive, and only a CONVERGED
    /// one has earned the right to be skipped next frame.
    std::uint32_t* m_islandConverged{nullptr};
    /// This frame's decision, one entry per island, uploaded before the graph
    /// runs. 1 = settled, do not touch.
    std::uint32_t* m_islandSkip{nullptr};
    KernelProfile m_kernelProfile;
    std::uint32_t m_profiledSolves{0};
    std::uint32_t* m_blockActiveCounts{nullptr};
    std::uint32_t* m_nodeBondBegin{nullptr};
    std::uint32_t* m_nodeBondRef{nullptr};
    std::uint32_t* m_bondIsland{nullptr};
    std::uint32_t* m_nodeIsland{nullptr};
    std::uint32_t m_islandCount{1};
    std::uint32_t m_islandCapacity{1};
    /// Set by removeBond; consumed by applyTopologyChange on the next solve.
    bool m_topologyDirty{false};
    /// See ExtStressGpuSolveParams::skipStableUnconverged.
    bool m_skipStableUnconverged{false};
    /// Device-impulse permutations recorded by removeBond, replayed at the
    /// next applyTopologyChange so device and host bond order stay identical.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> m_pendingImpulseSwaps;
    /// Nodes whose bonds broke or were removed; consumed by planSettledSkip.
    std::vector<std::uint32_t> m_forceDirtyNodes;
    /// Bonds grouped by island (CSR), built once: the partition is fixed for
    /// the solver's lifetime.
    std::vector<std::uint32_t> m_islandBondBegin;
    std::vector<std::uint32_t> m_bondsByIsland;
    /// Host-side settled state. m_hostIslandConverged is the baseline carried
    /// between frames; m_settledBaselineValid says whether it refers to
    /// anything.
    std::vector<std::uint32_t> m_hostIslandConverged;
    std::vector<std::uint8_t> m_islandDirty;
    std::vector<std::uint32_t> m_changedNodes;
    std::vector<std::uint32_t> m_changedBonds;
    bool m_settledBaselineValid{false};
    /// Set when a solve() call found nothing to do at all, so the caller can
    /// skip its own post-solve work too.
    bool m_solveWasNoOp{false};
    /// Time spent BLOCKED inside refreshActiveLists this solve, kept apart
    /// from planning work so the host split means something.
    float m_activeListSyncMilliseconds{0.0f};
    std::uint32_t m_debugSolves{0};
    std::uint32_t* m_scatterIndices{nullptr};
    ExtStressGpuImpulse* m_scatterValues{nullptr};
    std::uint32_t* m_gatherIndices{nullptr};
    AngLin* m_gatherOutput{nullptr};
    std::vector<std::uint32_t> m_hostParent;
    std::vector<std::uint32_t> m_hostRootIsland;
    /// G1: node -> incident-bond CSR (offsets, and refs packing bond index +
    /// endpoint bit). Host-built at every topology change, uploaded once.
    PinnedVector<std::uint32_t> m_hostNodeBondBegin;
    PinnedVector<std::uint32_t> m_hostNodeBondRef;
    PinnedVector<std::uint32_t> m_hostBondIsland;
    PinnedVector<std::uint32_t> m_hostNodeIsland;
    float* m_projectedDirectionSquared{nullptr};
    float* m_deltaSquared{nullptr};
    float* m_previousGradientSquared{nullptr};
    SolveStatus* m_status{nullptr};
    /// Pinned staging that doubles as the settled baseline: a byte-exact
    /// record of the velocities the device currently holds.
    ExtStressGpuImpulse* m_hostInput{nullptr};
    AngLin* m_hostImpulses{nullptr};
    SolveStatus* m_hostStatus{nullptr};
    std::uint32_t* m_hostBrokenCount{nullptr};
    std::uint32_t* m_hostIslandSkip{nullptr};
    std::uint32_t* m_hostIslandConvergedPinned{nullptr};
    std::uint32_t* m_hostScatterIndices{nullptr};
    ExtStressGpuImpulse* m_hostScatterValues{nullptr};
    std::uint32_t* m_hostGatherIndices{nullptr};
    void* m_reduceScratch{nullptr};
    std::size_t m_reduceScratchBytes{0};
    cudaEvent_t m_uploadStart{};
    cudaEvent_t m_uploadStop{};
    cudaEvent_t m_solveStart{};
    cudaEvent_t m_solveStop{};
    cudaEvent_t m_statusReady{};
    cudaEvent_t m_downloadStart{};
    cudaEvent_t m_downloadStop{};
    cudaStream_t m_stream{};
    /// Capture-only stream for the conditional loop body graph.
    cudaStream_t m_bodyStream{};
    cudaGraph_t m_graph{};
    cudaGraphExec_t m_graphExec{};
    ExtStressGpuSolveParams m_graphParams{};
    bool m_graphWarmStart{false};
    // Active-set state: the solve kernels walk these compacted lists so a
    // tick costs what is moving, not what exists. Rebuilt (one device pass)
    // only when the skip set or the topology changes; counts live at
    // m_activeCounts[0]=bonds / [1]=nodes and are read by the kernels from
    // device memory, so their CONTENTS never force a graph recapture -- only
    // outgrowing the baked launch capacity does (executeSolve).
    std::uint32_t* m_activeBonds{nullptr};
    std::uint32_t* m_activeNodes{nullptr};
    std::uint32_t* m_activeCounts{nullptr};
    /// Shared 0/1 flag buffer for the two DeviceSelect compactions, sized
    /// max(bonds, nodes); stream order serializes the reuse.
    std::uint32_t* m_activeFlags{nullptr};
    void* m_selectScratch{nullptr};
    std::size_t m_selectScratchBytes{0};
    std::uint32_t* m_hostActiveCounts{nullptr};
    std::uint32_t m_activeBondCount{0};
    std::uint32_t m_activeNodeCount{0};
    std::uint32_t m_graphBondCap{0};
    std::uint32_t m_graphNodeCap{0};
    std::vector<std::uint32_t> m_prevIslandSkip;
    bool m_prevListsSkipping{false};
    bool m_activeListsDirty{true};
    std::uint32_t m_statSolves{0};
    std::uint32_t m_statListRefreshes{0};
    std::uint32_t m_statListSkips{0};
    std::uint32_t m_statGraphRecaptures{0};
    double m_statMidSyncMs{0.0};
    double m_statPlanMs{0.0};
    std::uint64_t m_statIterations{0};
    std::uint32_t m_statUnconverged{0};
    std::vector<float> m_statResGrad;
    std::vector<float> m_statResDelta;
    std::vector<std::uint32_t> m_statResActive;
    double m_statResSum{0.0};
    double m_statResMax{0.0};
    std::uint64_t m_statResCount{0};
    std::uint64_t m_statResOverTol{0};
    double m_statFinishMs{0.0};
    double m_statWaitMs{0.0};
    double m_statLastMidSync{0.0};

    /// Class-scope host timer. solve() has its own local HostClock/hostMs
    /// lambda; enqueueSolve needs the same thing and cannot see them.
    using StatClock = std::chrono::steady_clock;
    static double statMs(StatClock::time_point from)
    {
        return std::chrono::duration<double, std::milli>(StatClock::now() - from)
            .count();
    }
    double m_statPlanSkipMs{0.0};
    double m_statTopoMs{0.0};
    double m_statTopoIslandsMs{0.0};
    double m_statTopoCsrMs{0.0};
    double m_statTopoGroupMs{0.0};
    double m_statTopoUploadMs{0.0};
    double m_statTopoMemsetMs{0.0};
    void* m_topoStaging{nullptr};
    std::size_t m_topoStagingBytes{0};
    std::size_t m_topoStagingUsed{0};
    cudaEvent_t m_topoUploadDone{nullptr};
    bool m_topoUploadsPending{false};
    std::vector<char> m_pageableBounce;
    std::uint64_t m_statTopoBytes{0};
    std::uint32_t m_statTopoCopies{0};
    std::uint32_t m_statTopoGrowths{0};
    double m_statStageCopyMs{0.0};
    std::uint32_t m_statTopoCalls{0};
    double m_statRefreshMs{0.0};
    double m_statCaptureMs{0.0};
    double m_statInstantiateMs{0.0};
    std::uint32_t m_statGraphUpdates{0};
    double m_statUpdateMs{0.0};
    bool m_graphParamsDirty{false};

    // Device bond-stress walk. Capacities grow only.
    std::uint32_t* m_bsGroupBegin{nullptr};
    std::uint32_t* m_bsGroupSize{nullptr};
    std::uint32_t* m_bsMemberBlastBond{nullptr};
    std::uint32_t* m_bsBondNode0{nullptr};
    std::uint32_t* m_bsBondNode1{nullptr};
    std::uint32_t* m_bsBondMaterial{nullptr};
    float* m_bsBondNormal{nullptr};
    float* m_bsBondCentroid{nullptr};
    float* m_bsBondNodeDisp{nullptr};
    float* m_bsHealth{nullptr};
    float* m_bsGroupStressNormal{nullptr};
    float* m_bsGroupStressShear{nullptr};
    float* m_bsGroupNormal{nullptr};
    float* m_bsGroupCentroid{nullptr};
    std::uint8_t* m_bsNodeOverstressed{nullptr};
    std::uint32_t* m_bsGroupRemoveCount{nullptr};
    std::uint32_t* m_bsGroupRemoveOffset{nullptr};
    std::uint32_t* m_bsRemoveFlag{nullptr};
    std::uint32_t* m_bsRemoveList{nullptr};
    std::uint32_t* m_bsOverstressedCount{nullptr};
    void* m_bsScanScratch{nullptr};
    std::size_t m_bsScanScratchBytes{0};
    std::uint32_t* m_bsHostRemoveList{nullptr};
    std::uint8_t* m_bsHostNodeOverstressed{nullptr};
    float* m_bsHostGroupStressNormal{nullptr};
    float* m_bsHostGroupStressShear{nullptr};
    float* m_bsHostGroupNormal{nullptr};
    float* m_bsHostGroupCentroid{nullptr};
    std::uint32_t* m_bsHostCounts{nullptr};
    std::uint32_t m_bsGroupCapacity{0};
    std::uint32_t m_bsSlotCapacity{0};
    std::uint32_t m_bsNodeCapacity{0};
    std::uint32_t m_bsBondCapacity{0};
    float* m_bsMaterialLimits{nullptr};
    std::uint32_t m_bsMaterialCapacity{0};
    bool m_bsReady{false};
    /// Pinned staging: pageable source memory cannot DMA.
    std::uint32_t* m_bsPinGroupBegin{nullptr};
    std::uint32_t* m_bsPinGroupSize{nullptr};
    std::uint32_t* m_bsPinMembers{nullptr};
    float* m_bsPinHealth{nullptr};
    float* m_bsPinMaterials{nullptr};
    bool m_bsCsrResident{false};
    bool m_bsHealthResident{false};
    bool m_bsVectorsFetched{false};
    bool m_bsStressFetched{false};
    std::uint32_t m_bsVectorGroups{0};
    cudaStream_t m_bsStream{nullptr};
    cudaEvent_t m_bsEvUploadStart{nullptr};
    cudaEvent_t m_bsEvUploadStop{nullptr};
    cudaEvent_t m_bsEvKernelStop{nullptr};
    cudaEvent_t m_bsEvReadStop{nullptr};
};

} // namespace

ExtStressGpuSolver* ExtStressGpuSolver::create(
    const ExtStressGpuNode* nodes,
    std::uint32_t nodeCount,
    const ExtStressGpuBond* bonds,
    std::uint32_t bondCount,
    const ExtStressGpuMaterial* materials,
    std::uint32_t materialCount,
    void* cudaContext)
{
    if (!nodes || !bonds || nodeCount == 0 || bondCount == 0)
    {
        return nullptr;
    }
    try
    {
        return new ExtStressGpuSolverImpl(
            nodes,
            nodeCount,
            bonds,
            bondCount,
            materials,
            materialCount,
            reinterpret_cast<CUcontext>(cudaContext));
    }
    catch (...)
    {
        return nullptr;
    }
}

} // namespace Blast
} // namespace Nv
