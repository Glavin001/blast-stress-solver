// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "NvBlastExtStressGpu.h"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <map>
#include <string>
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
        const float angularDenominator = value.angular > 0.0f ? value.angular : 1.0f;
        const float linearDenominator = value.linear > 0.0f ? value.linear : 1.0f;
        b.angular = makeVec(
            -velocity.angular.x / angularDenominator,
            -velocity.angular.y / angularDenominator,
            -velocity.angular.z / angularDenominator);
        b.linear = makeVec(
            -reciprocalLengthScale * velocity.linear.x / linearDenominator,
            -reciprocalLengthScale * velocity.linear.y / linearDenominator,
            -reciprocalLengthScale * velocity.linear.z / linearDenominator);
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
        if (warmStart)
        {
            impulses[index].angular =
                mul(impulses[index].angular, reciprocalAngularImpulseScale);
            impulses[index].linear =
                mul(impulses[index].linear, reciprocalLinearImpulseScale);
        }
        else
        {
            impulses[index] = {};
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

    const AngLin impulse = bonds[bond];
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
        const AngLin impulse = bonds[bond];
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

__global__ void initializeStatus(SolveStatus* status, std::uint32_t maxIterations)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        status->active = 1;
        status->iterations = maxIterations;
        status->converged = 0;
    }
}

/// Per-island sum of squared magnitudes.
///
/// Islands are disconnected components, so their conjugate-gradient scalars
/// must be independent: a shared alpha/beta compromises every island toward
/// the average, and a shared convergence test lets a large well-conditioned
/// island hide a small badly-solved one. atomicAdd rather than a segmented
/// library reduce because bonds and nodes are not stored island-contiguous.
__global__ void accumulateSquaredByIsland(
    const AngLin* values,
    const std::uint32_t* island,
    const std::uint32_t* islandSkip,
    float* perIsland,
    const std::uint32_t* activeList,
    const std::uint32_t* activeCounts,
    std::uint32_t whichCount)
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
    atomicAdd(&perIsland[id], squared);
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
    std::uint32_t iteration)
{
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
    std::uint32_t iteration,
    std::uint32_t islandCount)
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
    }
}

__global__ void unscaleImpulses(
    AngLin* impulses,
    const std::uint32_t* bondIsland,
    const std::uint32_t* islandSkip,
    const std::uint32_t* activeBonds,
    const std::uint32_t* activeCounts,
    float linearScale,
    float angularScale)
{
    const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= activeCounts[0])
    {
        return;
    }
    const std::uint32_t index = activeBonds[slot];
    if (!bondSettled(islandSkip, bondIsland[index]))
    {
        impulses[index].angular = mul(impulses[index].angular, angularScale);
        impulses[index].linear = mul(impulses[index].linear, linearScale);
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
    const AngLin* impulses,
    const Vec4* normals,
    const float* areas,
    const float* nodeDistances,
    const std::uint32_t* bondMaterials,
    const ExtStressGpuMaterial* materials,
    std::uint32_t materialCount,
    float* health,
    std::uint32_t* brokenBonds,
    std::uint32_t* brokenCount,
    std::uint32_t bondCount)
{
    const std::uint32_t bond = blockIdx.x * blockDim.x + threadIdx.x;
    if (bond >= bondCount || health[bond] <= 0.0f)
    {
        return;
    }

    const AngLin impulse = impulses[bond];
    const Vec4 normal = normals[bond];
    const float area = areas[bond];
    const float distance = nodeDistances[bond];
    const float linearNormal =
        impulse.linear.x * normal.x
        + impulse.linear.y * normal.y
        + impulse.linear.z * normal.z;
    const float linearMagnitudeSquared =
        impulse.linear.x * impulse.linear.x
        + impulse.linear.y * impulse.linear.y
        + impulse.linear.z * impulse.linear.z;
    float stressNormal = linearNormal / area;
    float stressShear =
        sqrtf(fmaxf(0.0f, linearMagnitudeSquared - linearNormal * linearNormal)) / area;

    const float angularNormal = fabsf(
        impulse.angular.x * normal.x
        + impulse.angular.y * normal.y
        + impulse.angular.z * normal.z);
    const float angularMagnitudeSquared =
        impulse.angular.x * impulse.angular.x
        + impulse.angular.y * impulse.angular.y
        + impulse.angular.z * impulse.angular.z;
    const float twist = angularNormal / area;
    const float bend =
        sqrtf(fmaxf(0.0f, angularMagnitudeSquared - angularNormal * angularNormal)) / area;
    stressShear += twist * 2.0f / distance;
    stressNormal += copysignf(bend * 2.0f / distance, stressNormal);

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
        cudaEventDestroy(m_downloadStart);
        cudaEventDestroy(m_downloadStop);
        cudaStreamDestroy(m_stream);
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

    bool solve(
        const ExtStressGpuImpulse* nodeVelocities,
        const ExtStressGpuSolveParams& params) override
    {
        ContextGuard context(m_cudaContext);
        if (!enqueueSolve(nodeVelocities, params))
        {
            return false;
        }
        if (m_solveWasNoOp)
        {
            return true;
        }
        checkCuda(cudaEventSynchronize(m_statusReady), "wait for stress solve");
        finishSolve();
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
        if (!bondImpulses || capacity < m_bondCount
            || !enqueueSolve(nodeVelocities, params))
        {
            return false;
        }
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
        enqueueImpulseReadback(compacted);
        checkCuda(cudaEventSynchronize(m_downloadStop), "wait for stress solve and readback");
        finishSolve();
        finishImpulseReadback(bondImpulses, compacted);
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
        // Device impulses must follow the same permutation as the host arrays;
        // replayed in order in applyTopologyChange.
        m_pendingImpulseSwaps.emplace_back(bondIndex, last);
        swapWithLast(m_hostNode0, bondIndex, last);
        swapWithLast(m_hostNode1, bondIndex, last);
        swapWithLast(m_hostOffset0, bondIndex, last);
        swapWithLast(m_hostOffset1, bondIndex, last);
        swapWithLast(m_hostNormals, bondIndex, last);
        swapWithLast(m_hostAreas, bondIndex, last);
        swapWithLast(m_hostNodeDistances, bondIndex, last);
        swapWithLast(m_hostHealth, bondIndex, last);
        swapWithLast(m_hostBondMaterials, bondIndex, last);
        m_bondCount = last;
        m_topologyDirty = true;
        return true;
    }

    const std::uint32_t* lastChangedBonds(std::uint32_t& count) const override
    {
        count = static_cast<std::uint32_t>(m_changedBonds.size());
        return m_changedBonds.data();
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
    bool enqueueSolve(
        const ExtStressGpuImpulse* nodeVelocities,
        const ExtStressGpuSolveParams& params)
    {
        if (!nodeVelocities || params.maxIterations == 0)
        {
            return false;
        }
        if (m_topologyDirty)
        {
            applyTopologyChange();
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
            planSettledSkip(nodeVelocities);
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
        refreshActiveLists(skipping);

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
            return;
        }
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
        checkCuda(cudaStreamSynchronize(m_stream), "sync active counts");
        m_activeBondCount = m_hostActiveCounts[0];
        m_activeNodeCount = m_hostActiveCounts[1];
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
            const bool skip =
                haveBaseline && !m_islandDirty[k]
                && (m_skipStableUnconverged || m_hostIslandConverged[k] != 0u);
            m_hostIslandSkip[k] = skip ? 1u : 0u;
            if (skip)
            {
                ++skipped;
                continue;
            }
            for (std::uint32_t t = m_islandBondBegin[k]; t < m_islandBondBegin[k + 1]; ++t)
            {
                m_changedBonds.push_back(m_bondsByIsland[t]);
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

    template <typename T>
    static void swapWithLast(std::vector<T>& values, std::uint32_t index, std::uint32_t last)
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
        // The island partition is about to be remapped; the compacted lists
        // index into the OLD partition and must be rebuilt before next solve.
        m_activeListsDirty = true;
computeIslands();
        buildNodeBondCsr();
        groupBondsByIsland();
        uploadTopology();
uploadIslands();
        uploadNodeBondCsr();
        // Grid sizes and the per-island memset lengths are baked into the
        // captured graph, and both just changed.
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
        if (wholeResetOnTopology())
        {
            checkCuda(
                cudaMemset(m_impulses, 0, sizeof(AngLin) * m_bondCount),
                "reset warm start after topology change");
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
                        cudaMemcpy(
                            m_impulses + swap.first,
                            m_impulses + swap.second,
                            sizeof(AngLin),
                            cudaMemcpyDeviceToDevice),
                        "replay impulse swap");
                }
            }
            m_pendingImpulseSwaps.clear();
        }
        // Island ids were remapped by the repartition either way; converged
        // flags are keyed by id and must not survive it. Under
        // skipStableUnconverged this does not block skipping; under
        // converged-required semantics it conservatively forces a re-solve.
        m_hostIslandConverged.assign(m_islandCapacity, 0u);
        std::fill(m_hostIslandSkip, m_hostIslandSkip + m_islandCapacity, 0u);
        checkCuda(
            cudaMemset(m_islandConverged, 0, sizeof(std::uint32_t) * m_islandCapacity),
            "clear island converged flags");
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
        for (std::uint32_t j = 0; j < count; ++j)
        {
            const std::uint32_t bond = m_changedBonds[j];
            const AngLin& value = m_hostImpulses[compacted ? j : bond];
            bondImpulses[bond].angular =
                {value.angular.x, value.angular.y, value.angular.z};
            bondImpulses[bond].linear =
                {value.linear.x, value.linear.y, value.linear.z};
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
        checkCuda(cudaEventCreate(&m_uploadStart), "create upload start event");
        checkCuda(cudaEventCreate(&m_uploadStop), "create upload stop event");
        checkCuda(cudaEventCreate(&m_solveStart), "create solve start event");
        checkCuda(cudaEventCreate(&m_solveStop), "create solve stop event");
        checkCuda(cudaEventCreate(&m_statusReady), "create status-ready event");
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
        std::vector<std::uint32_t> cursor(
            m_islandBondBegin.begin(), m_islandBondBegin.end());
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

        m_hostRootIsland.assign(m_nodeCount, kNoIsland);
        m_hostBondIsland.assign(m_bondCount, kNoIsland);
        m_islandCount = 0;
        for (std::uint32_t b = 0; b < m_bondCount; ++b)
        {
            const std::uint32_t n0 = m_hostNode0[b];
            const std::uint32_t n1 = m_hostNode1[b];
            if (n0 >= m_nodeCount || n1 >= m_nodeCount)
            {
                continue;
            }
            const bool s0 = isStatic(n0);
            const bool s1 = isStatic(n1);
            if (s0 && s1)
            {
                continue; // degenerate static-static bond: no coupling
            }
            const std::uint32_t rep = findRoot(s0 ? n1 : n0);
            if (m_hostRootIsland[rep] == kNoIsland)
            {
                m_hostRootIsland[rep] = m_islandCount++;
            }
            m_hostBondIsland[b] = m_hostRootIsland[rep];
        }

        // Static nodes stay unassigned: they are boundaries, excluded from the
        // per-island residual just as they are from the CPU sub-systems.
        m_hostNodeIsland.assign(m_nodeCount, kNoIsland);
        for (std::uint32_t i = 0; i < m_nodeCount; ++i)
        {
            if (!isStatic(i))
            {
                m_hostNodeIsland[i] = m_hostRootIsland[findRoot(i)];
            }
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
        m_hostNodeBondRef.assign(m_hostNodeBondBegin[m_nodeCount], 0u);
        std::vector<std::uint32_t> cursor(
            m_hostNodeBondBegin.begin(), m_hostNodeBondBegin.end());
        // Ascending bond order within each node's list: the sum order is then
        // a pure function of topology, which is what makes the gather
        // reproducible run to run.
        for (std::uint32_t bond = 0; bond < m_bondCount; ++bond)
        {
            m_hostNodeBondRef[cursor[m_hostNode0[bond]]++] = bond;
            m_hostNodeBondRef[cursor[m_hostNode1[bond]]++] = bond | 0x80000000u;
        }
    }

    void uploadNodeBondCsr()
    {
        if (m_hostNodeBondBegin.empty())
        {
            return;
        }
        checkCuda(
            cudaMemcpy(
                m_nodeBondBegin,
                m_hostNodeBondBegin.data(),
                sizeof(std::uint32_t) * m_hostNodeBondBegin.size(),
                cudaMemcpyHostToDevice),
            "upload node-bond csr offsets");
        if (!m_hostNodeBondRef.empty())
        {
            checkCuda(
                cudaMemcpy(
                    m_nodeBondRef,
                    m_hostNodeBondRef.data(),
                    sizeof(std::uint32_t) * m_hostNodeBondRef.size(),
                    cudaMemcpyHostToDevice),
                "upload node-bond csr refs");
        }
    }

    void uploadIslands()
    {
        checkCuda(
            cudaMemcpy(
                m_bondIsland,
                m_hostBondIsland.data(),
                sizeof(std::uint32_t) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload bond island ids");
        checkCuda(
            cudaMemcpy(
                m_nodeIsland,
                m_hostNodeIsland.data(),
                sizeof(std::uint32_t) * m_nodeCount,
                cudaMemcpyHostToDevice),
            "upload node island ids");
    }

    void uploadTopology()
    {
        checkCuda(
            cudaMemcpy(
                m_node0,
                m_hostNode0.data(),
                sizeof(std::uint32_t) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload node0");
        checkCuda(
            cudaMemcpy(
                m_node1,
                m_hostNode1.data(),
                sizeof(std::uint32_t) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload node1");
        checkCuda(
            cudaMemcpy(
                m_offset0,
                m_hostOffset0.data(),
                sizeof(Vec4) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload offset0");
        checkCuda(
            cudaMemcpy(
                m_offset1,
                m_hostOffset1.data(),
                sizeof(Vec4) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload offset1");
        checkCuda(
            cudaMemcpy(
                m_inertia,
                m_hostInertia.data(),
                sizeof(Inertia) * m_nodeCount,
                cudaMemcpyHostToDevice),
            "upload inertia");
        checkCuda(
            cudaMemcpy(
                m_normals,
                m_hostNormals.data(),
                sizeof(Vec4) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload normals");
        checkCuda(
            cudaMemcpy(
                m_areas,
                m_hostAreas.data(),
                sizeof(float) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload areas");
        checkCuda(
            cudaMemcpy(
                m_nodeDistances,
                m_hostNodeDistances.data(),
                sizeof(float) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload node distances");
        checkCuda(
            cudaMemcpy(
                m_health,
                m_hostHealth.data(),
                sizeof(float) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload health");
        checkCuda(
            cudaMemcpy(
                m_bondMaterials,
                m_hostBondMaterials.data(),
                sizeof(std::uint32_t) * m_bondCount,
                cudaMemcpyHostToDevice),
            "upload bond materials");
        checkCuda(
            cudaMemcpy(
                m_materials,
                m_hostMaterials.data(),
                sizeof(ExtStressGpuMaterial) * m_materialCount,
                cudaMemcpyHostToDevice),
            "upload material table");
    }

    /// Per-island reduction. Replaces the whole-graph cub::DeviceReduce, which
    /// imposed a grid-wide barrier every iteration and produced a single
    /// residual that could not distinguish a converged island from a starved
    /// one.
    void reduceByIsland(
        const AngLin* values,
        const std::uint32_t* island,
        const std::uint32_t* islandSkip,
        const std::uint32_t* activeList,
        std::uint32_t whichCount,
        std::uint32_t launchCap,
        float* result)
    {
        checkCuda(
            cudaMemsetAsync(result, 0, sizeof(float) * m_islandCount, m_stream),
            "clear per-island reduction");
        m_kernelProfile.begin("accumulateSquaredByIsland", m_stream);
        accumulateSquaredByIsland<<<
            (launchCap + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            values,
            island,
            islandSkip,
            result,
            activeList,
            m_activeCounts,
            whichCount);
        m_kernelProfile.end(m_stream);
    }

    void rightMultiply(const AngLin* bonds, AngLin* nodes, const std::uint32_t* islandSkip)
    {
        if (gatherRightMultiplyEnabled())
        {
            // No memset: every active node writes its own slot. No scaleNodes
            // either: the inverse-inertia scale is folded into the write. Two
            // graph nodes become one, and twelve global float atomics per
            // bond become zero.
            m_kernelProfile.begin("gatherRightMultiply", m_stream);
            gatherRightMultiply<<<
                (m_graphNodeCap + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(
                nodes,
                bonds,
                m_nodeBondBegin,
                m_nodeBondRef,
                m_offset0,
                m_offset1,
                m_health,
                m_bondIsland,
                islandSkip,
                m_inertia,
                m_activeNodes,
                m_activeCounts);
            m_kernelProfile.end(m_stream);
            return;
        }
        checkCuda(
            cudaMemsetAsync(
                nodes,
                0,
                sizeof(AngLin) * m_nodeCount,
                m_stream),
            "clear node product");
        m_kernelProfile.begin("couplingRightMultiply", m_stream);
        couplingRightMultiply<<<
            (m_graphBondCap + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            nodes,
            bonds,
            m_node0,
            m_node1,
            m_offset0,
            m_offset1,
            m_health,
            m_bondIsland,
            islandSkip,
            m_activeBonds,
            m_activeCounts);
        m_kernelProfile.end(m_stream);
        m_kernelProfile.begin("scaleNodes", m_stream);
        scaleNodes<<<
            (m_graphNodeCap + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            nodes,
            m_inertia,
            m_activeNodes,
            m_activeCounts);
        m_kernelProfile.end(m_stream);
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
        if (warmStart)
        {
            rightMultiply(m_impulses, m_projectedDirection, islandSkip);
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
        initializeStatus<<<1, 1, 0, m_stream>>>(m_status, params.maxIterations);
        m_kernelProfile.end(m_stream);

        for (std::uint32_t iteration = 0; iteration < params.maxIterations; ++iteration)
        {
            m_kernelProfile.begin("couplingLeftMultiply", m_stream);
            couplingLeftMultiply<<<
                (m_graphBondCap + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(
                m_gradient,
                m_residual,
                m_inertia,
                m_node0,
                m_node1,
                m_offset0,
                m_offset1,
                m_health,
                m_bondIsland,
                islandSkip,
                m_activeBonds,
                m_activeCounts);
            m_kernelProfile.end(m_stream);
            const std::uint32_t islandBlocks =
                (m_islandCount + kBlockSize - 1) / kBlockSize;
            reduceByIsland(
                m_gradient, m_bondIsland, islandSkip, m_activeBonds, 0u,
                m_graphBondCap, m_gradientSquared);
            m_kernelProfile.begin("checkConvergencePerIsland", m_stream);
            checkConvergencePerIsland<<<islandBlocks, kBlockSize, 0, m_stream>>>(
                m_islandActive,
                m_islandConverged,
                m_gradientSquared,
                m_deltaSquared,
                m_blockActiveCounts,
                m_islandCount);
            m_kernelProfile.end(m_stream);
            m_kernelProfile.begin("updateDirectionPerIsland", m_stream);
            updateDirectionPerIsland<<<
                (m_graphBondCap + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(
                m_direction,
                m_gradient,
                m_gradientSquared,
                m_previousGradientSquared,
                m_bondIsland,
                m_islandActive,
                m_activeBonds,
                m_activeCounts,
                iteration);
            m_kernelProfile.end(m_stream);

            rightMultiply(m_direction, m_projectedDirection, islandSkip);
            reduceByIsland(
                m_projectedDirection,
                m_nodeIsland,
                islandSkip,
                m_activeNodes,
                1u,
                m_graphNodeCap,
                m_projectedDirectionSquared);
            m_kernelProfile.begin("retireDegenerateIslands", m_stream);
            retireDegenerateIslands<<<islandBlocks, kBlockSize, 0, m_stream>>>(
                m_islandActive,
                m_projectedDirectionSquared,
                m_previousGradientSquared,
                m_gradientSquared,
                m_status,
                m_blockActiveCounts,
                islandBlocks,
                iteration,
                m_islandCount);
            m_kernelProfile.end(m_stream);
            m_kernelProfile.begin("updateSolutionAndResidualPerIsland", m_stream);
            updateSolutionAndResidualPerIsland<<<
                (maxCap + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(
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
            m_kernelProfile.end(m_stream);
        }

        const float linearScale = m_lengthScale * m_massScale;
        const float angularScale = m_lengthScale * linearScale;
        m_kernelProfile.begin("unscaleImpulses", m_stream);
        unscaleImpulses<<<
            (m_graphBondCap + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            m_impulses,
            m_bondIsland,
            islandSkip,
            m_activeBonds,
            m_activeCounts,
            linearScale,
            angularScale);
        m_kernelProfile.end(m_stream);
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
                m_impulses,
                m_normals,
                m_areas,
                m_nodeDistances,
                m_bondMaterials,
                m_materials,
                m_materialCount,
                m_health,
                m_brokenBonds,
                m_brokenCount,
                m_bondCount);
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
            && m_graphBondCap >= m_activeBondCount
            && m_graphNodeCap >= m_activeNodeCount
            && m_graphBondCap <= 4u * std::max(
                   launchCapacity(m_activeBondCount, m_bondCount), 1024u)
            && m_graphNodeCap <= 4u * std::max(
                   launchCapacity(m_activeNodeCount, m_nodeCount), 1024u);
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
            m_graphBondCap = launchCapacity(m_activeBondCount, m_bondCount);
            m_graphNodeCap = launchCapacity(m_activeNodeCount, m_nodeCount);
            launchSolve(params);
            ++m_profiledSolves;
            return;
        }
        if (!graphMatches(params, warmStart))
        {
            m_graphBondCap = launchCapacity(m_activeBondCount, m_bondCount);
            m_graphNodeCap = launchCapacity(m_activeNodeCount, m_nodeCount);
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
            checkCuda(
                cudaStreamBeginCapture(m_stream, cudaStreamCaptureModeThreadLocal),
                "begin solver graph capture");
            launchSolve(params);
            checkCuda(
                cudaStreamEndCapture(m_stream, &m_graph),
                "end solver graph capture");
            checkCuda(
                cudaGraphInstantiate(&m_graphExec, m_graph, 0),
                "instantiate solver graph");
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
    ExtStressGpuTelemetry m_telemetry{};

    std::vector<std::uint32_t> m_hostNode0;
    std::vector<std::uint32_t> m_hostNode1;
    std::vector<Vec4> m_hostOffset0;
    std::vector<Vec4> m_hostOffset1;
    std::vector<Inertia> m_hostInertia;
    std::vector<Vec4> m_hostNormals;
    std::vector<float> m_hostAreas;
    std::vector<float> m_hostNodeDistances;
    std::vector<float> m_hostHealth;
    std::vector<std::uint32_t> m_hostBondMaterials;
    std::vector<ExtStressGpuMaterial> m_hostMaterials;

    std::uint32_t* m_node0{nullptr};
    std::uint32_t* m_node1{nullptr};
    Vec4* m_offset0{nullptr};
    Vec4* m_offset1{nullptr};
    Inertia* m_inertia{nullptr};
    Vec4* m_normals{nullptr};
    float* m_areas{nullptr};
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
    std::uint32_t m_debugSolves{0};
    std::uint32_t* m_scatterIndices{nullptr};
    ExtStressGpuImpulse* m_scatterValues{nullptr};
    std::uint32_t* m_gatherIndices{nullptr};
    AngLin* m_gatherOutput{nullptr};
    std::vector<std::uint32_t> m_hostParent;
    std::vector<std::uint32_t> m_hostRootIsland;
    /// G1: node -> incident-bond CSR (offsets, and refs packing bond index +
    /// endpoint bit). Host-built at every topology change, uploaded once.
    std::vector<std::uint32_t> m_hostNodeBondBegin;
    std::vector<std::uint32_t> m_hostNodeBondRef;
    std::vector<std::uint32_t> m_hostBondIsland;
    std::vector<std::uint32_t> m_hostNodeIsland;
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
