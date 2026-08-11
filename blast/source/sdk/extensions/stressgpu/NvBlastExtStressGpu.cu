// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "NvBlastExtStressGpu.h"

#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace Nv
{
namespace Blast
{
namespace
{

constexpr std::uint32_t kBlockSize = 256;

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
    std::uint32_t nodeCount,
    std::uint32_t bondCount,
    float reciprocalLengthScale,
    float reciprocalLinearImpulseScale,
    float reciprocalAngularImpulseScale,
    bool warmStart)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nodeCount)
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
    if (index < bondCount)
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

__global__ void couplingRightMultiply(
    AngLin* nodes,
    const AngLin* bonds,
    const std::uint32_t* node0,
    const std::uint32_t* node1,
    const Vec4* offset0,
    const Vec4* offset1,
    const float* health,
    std::uint32_t bondCount)
{
    const std::uint32_t bond = blockIdx.x * blockDim.x + threadIdx.x;
    if (bond >= bondCount || health[bond] <= 0.0f)
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

__global__ void scaleNodes(
    AngLin* nodes,
    const Inertia* inertia,
    std::uint32_t nodeCount)
{
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < nodeCount)
    {
        nodes[node].angular = mul(nodes[node].angular, inertia[node].angular);
        nodes[node].linear = mul(nodes[node].linear, inertia[node].linear);
    }
}

__global__ void subtractResidual(
    AngLin* residual,
    const AngLin* value,
    std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
    {
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
    std::uint32_t bondCount)
{
    const std::uint32_t bond = blockIdx.x * blockDim.x + threadIdx.x;
    if (bond >= bondCount)
    {
        return;
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
    const float* rhsSquared,
    float tolerance,
    std::uint32_t islandCount)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= islandCount)
    {
        return;
    }
    deltaSquared[id] = rhsSquared[id] * tolerance * tolerance;
    islandActive[id] = 1;
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
    float* perIsland,
    std::uint32_t count)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
    {
        return;
    }
    const std::uint32_t id = island[index];
    if (id == kNoIsland)
    {
        return;
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
__global__ void checkConvergencePerIsland(
    std::uint32_t* islandActive,
    const float* residualSquared,
    const float* deltaSquared,
    std::uint32_t islandCount)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= islandCount || !islandActive[id])
    {
        return;
    }
    if (residualSquared[id] <= deltaSquared[id])
    {
        islandActive[id] = 0;
    }
}

/// Roll the per-island flags up into the single status the host reads.
__global__ void summarizeIslands(
    SolveStatus* status,
    const std::uint32_t* islandActive,
    std::uint32_t islandCount,
    std::uint32_t iteration)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    std::uint32_t active = 0;
    for (std::uint32_t i = 0; i < islandCount; ++i)
    {
        active += islandActive[i] ? 1u : 0u;
    }
    status->active = active;
    if (active == 0 && !status->converged)
    {
        status->converged = 1;
        status->iterations = iteration;
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
    std::uint32_t count,
    std::uint32_t iteration)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
    {
        return;
    }
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
    std::uint32_t bondCount,
    std::uint32_t nodeCount)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < bondCount)
    {
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
    if (index < nodeCount)
    {
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
__global__ void retireDegenerateIslands(
    std::uint32_t* islandActive,
    const float* projectedDirectionSquared,
    std::uint32_t islandCount)
{
    const std::uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= islandCount || !islandActive[id])
    {
        return;
    }
    const float denominator = projectedDirectionSquared[id];
    if (!(denominator > 0.0f) || !isfinite(denominator))
    {
        islandActive[id] = 0;
    }
}

__global__ void unscaleImpulses(
    AngLin* impulses,
    std::uint32_t bondCount,
    float linearScale,
    float angularScale)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < bondCount)
    {
        impulses[index].angular = mul(impulses[index].angular, angularScale);
        impulses[index].linear = mul(impulses[index].linear, linearScale);
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
        computeIslands(nodes, bonds);
        allocate();
        uploadTopology();
        uploadIslands();
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
        cudaFreeHost(m_hostImpulses);
        cudaFreeHost(m_hostInput);
        cudaFree(m_reduceScratch);
        cudaFree(m_status);
        cudaFree(m_previousGradientSquared);
        cudaFree(m_deltaSquared);
        cudaFree(m_islandActive);
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
        enqueueImpulseReadback();
        checkCuda(cudaEventSynchronize(m_downloadStop), "wait for impulse readback");
        finishImpulseReadback(bondImpulses);
        return true;
    }

    bool solveAndReadbackImpulses(
        const ExtStressGpuImpulse* nodeVelocities,
        const ExtStressGpuSolveParams& params,
        ExtStressGpuImpulse* bondImpulses,
        std::uint32_t capacity) override
    {
        ContextGuard context(m_cudaContext);
        if (!bondImpulses || capacity < m_bondCount
            || !enqueueSolve(nodeVelocities, params))
        {
            return false;
        }
        // The status and impulse copies share one stream. Waiting for the
        // latter completes the upload, solve, status, and impulse readback.
        enqueueImpulseReadback();
        checkCuda(cudaEventSynchronize(m_downloadStop), "wait for stress solve and readback");
        finishSolve();
        finishImpulseReadback(bondImpulses);
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

    void resetWarmStart() override
    {
        ContextGuard context(m_cudaContext);
        checkCuda(cudaMemset(m_impulses, 0, sizeof(AngLin) * m_bondCount), "reset warm start");
        m_hasWarmStart = false;
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

        m_telemetry = {};
        std::memcpy(
            m_hostInput,
            nodeVelocities,
            sizeof(ExtStressGpuImpulse) * m_nodeCount);
        checkCuda(cudaEventRecord(m_uploadStart, m_stream), "record upload start");
        checkCuda(
            cudaMemcpyAsync(
                m_input,
                m_hostInput,
                sizeof(ExtStressGpuImpulse) * m_nodeCount,
                cudaMemcpyHostToDevice,
                m_stream),
            "upload stress inputs");
        checkCuda(cudaEventRecord(m_uploadStop, m_stream), "record upload stop");
        m_telemetry.hostToDeviceBytes =
            sizeof(ExtStressGpuImpulse) * static_cast<std::uint64_t>(m_nodeCount);

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
        checkCuda(cudaEventRecord(m_statusReady, m_stream), "record status ready");
        return true;
    }

    void finishSolve()
    {
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
    }

    void enqueueImpulseReadback()
    {
        checkCuda(cudaEventRecord(m_downloadStart, m_stream), "record readback start");
        checkCuda(
            cudaMemcpyAsync(
                m_hostImpulses,
                m_impulses,
                sizeof(AngLin) * m_bondCount,
                cudaMemcpyDeviceToHost,
                m_stream),
            "read stress impulses");
        checkCuda(cudaEventRecord(m_downloadStop, m_stream), "record readback stop");
    }

    void finishImpulseReadback(ExtStressGpuImpulse* bondImpulses)
    {
        checkCuda(
            cudaEventElapsedTime(
                &m_telemetry.downloadMilliseconds,
                m_downloadStart,
                m_downloadStop),
            "measure impulse readback");
        for (std::uint32_t i = 0; i < m_bondCount; ++i)
        {
            bondImpulses[i].angular =
                {
                    m_hostImpulses[i].angular.x,
                    m_hostImpulses[i].angular.y,
                    m_hostImpulses[i].angular.z};
            bondImpulses[i].linear =
                {
                    m_hostImpulses[i].linear.x,
                    m_hostImpulses[i].linear.y,
                    m_hostImpulses[i].linear.z};
        }
        m_telemetry.deviceToHostBytes =
            sizeof(AngLin) * static_cast<std::uint64_t>(m_bondCount);
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
        allocateDevice(m_gradientSquared, m_islandCount, "allocate gradient norm");
        allocateDevice(
            m_projectedDirectionSquared, m_islandCount, "allocate projected norm");
        allocateDevice(m_deltaSquared, m_islandCount, "allocate tolerance");
        allocateDevice(
            m_previousGradientSquared, m_islandCount, "allocate previous gradient norm");
        allocateDevice(m_islandActive, m_islandCount, "allocate island active flags");
        allocateDevice(m_bondIsland, m_bondCount, "allocate bond island ids");
        allocateDevice(m_nodeIsland, m_nodeCount, "allocate node island ids");
        allocateDevice(m_status, 1, "allocate solve status");
        allocateHost(m_hostInput, m_nodeCount, "allocate pinned stress input");
        allocateHost(m_hostImpulses, m_bondCount, "allocate pinned stress impulses");
        allocateHost(m_hostStatus, 1, "allocate pinned stress status");

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
    }

    /// Partition the graph into islands, matching the CPU solver's rule
    /// exactly (stress.cpp solveIslandAware): union only across bonds whose
    /// endpoints are both dynamic, because a static (zero-mass) node is a fixed
    /// boundary that transmits no coupling and therefore cuts the graph. The
    /// partitions must agree for CPU and GPU results to agree.
    void computeIslands(const ExtStressGpuNode* nodes, const ExtStressGpuBond* bonds)
    {
        m_hostParent.resize(m_nodeCount);
        for (std::uint32_t i = 0; i < m_nodeCount; ++i)
        {
            m_hostParent[i] = i;
        }
        const auto isStatic = [&](std::uint32_t node) {
            return !(nodes[node].mass > 0.0f);
        };
        for (std::uint32_t b = 0; b < m_bondCount; ++b)
        {
            const std::uint32_t n0 = bonds[b].node0;
            const std::uint32_t n1 = bonds[b].node1;
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
            const std::uint32_t n0 = bonds[b].node0;
            const std::uint32_t n1 = bonds[b].node1;
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
        std::uint32_t count,
        float* result)
    {
        checkCuda(
            cudaMemsetAsync(result, 0, sizeof(float) * m_islandCount, m_stream),
            "clear per-island reduction");
        accumulateSquaredByIsland<<<
            (count + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            values,
            island,
            result,
            count);
    }

    void rightMultiply(const AngLin* bonds, AngLin* nodes)
    {
        checkCuda(
            cudaMemsetAsync(
                nodes,
                0,
                sizeof(AngLin) * m_nodeCount,
                m_stream),
            "clear node product");
        couplingRightMultiply<<<
            (m_bondCount + kBlockSize - 1) / kBlockSize,
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
            m_bondCount);
        scaleNodes<<<
            (m_nodeCount + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            nodes,
            m_inertia,
            m_nodeCount);
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
        const std::uint32_t maxCount = std::max(m_nodeCount, m_bondCount);

        initializeSolve<<<
            (maxCount + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            m_input,
            m_rhs,
            m_residual,
            m_impulses,
            m_inertia,
            m_nodeCount,
            m_bondCount,
            reciprocalLengthScale,
            reciprocalLinearImpulseScale,
            reciprocalAngularImpulseScale,
            warmStart);
        if (warmStart)
        {
            rightMultiply(m_impulses, m_projectedDirection);
            subtractResidual<<<
                (m_nodeCount + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(m_residual, m_projectedDirection, m_nodeCount);
        }

        // Tolerance is relative to each island's own load, not the whole
        // graph's: a small island next to a heavily loaded one would otherwise
        // inherit a threshold it can never meaningfully meet.
        reduceByIsland(m_rhs, m_nodeIsland, m_nodeCount, m_gradientSquared);
        setTolerancePerIsland<<<
            (m_islandCount + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            m_deltaSquared,
            m_islandActive,
            m_gradientSquared,
            params.tolerance,
            m_islandCount);
        initializeStatus<<<1, 1, 0, m_stream>>>(m_status, params.maxIterations);

        for (std::uint32_t iteration = 0; iteration < params.maxIterations; ++iteration)
        {
            couplingLeftMultiply<<<
                (m_bondCount + kBlockSize - 1) / kBlockSize,
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
                m_bondCount);
            const std::uint32_t islandBlocks =
                (m_islandCount + kBlockSize - 1) / kBlockSize;
            reduceByIsland(m_gradient, m_bondIsland, m_bondCount, m_gradientSquared);
            checkConvergencePerIsland<<<islandBlocks, kBlockSize, 0, m_stream>>>(
                m_islandActive,
                m_gradientSquared,
                m_deltaSquared,
                m_islandCount);
            summarizeIslands<<<1, 1, 0, m_stream>>>(
                m_status,
                m_islandActive,
                m_islandCount,
                iteration);
            updateDirectionPerIsland<<<
                (m_bondCount + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(
                m_direction,
                m_gradient,
                m_gradientSquared,
                m_previousGradientSquared,
                m_bondIsland,
                m_islandActive,
                m_bondCount,
                iteration);
            saveGradientSquaredPerIsland<<<islandBlocks, kBlockSize, 0, m_stream>>>(
                m_previousGradientSquared,
                m_gradientSquared,
                m_islandActive,
                m_islandCount);

            rightMultiply(m_direction, m_projectedDirection);
            reduceByIsland(
                m_projectedDirection,
                m_nodeIsland,
                m_nodeCount,
                m_projectedDirectionSquared);
            retireDegenerateIslands<<<islandBlocks, kBlockSize, 0, m_stream>>>(
                m_islandActive,
                m_projectedDirectionSquared,
                m_islandCount);
            updateSolutionAndResidualPerIsland<<<
                (maxCount + kBlockSize - 1) / kBlockSize,
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
                m_bondCount,
                m_nodeCount);
        }

        const float linearScale = m_lengthScale * m_massScale;
        const float angularScale = m_lengthScale * linearScale;
        unscaleImpulses<<<
            (m_bondCount + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            m_impulses,
            m_bondCount,
            linearScale,
            angularScale);
        checkCuda(
            cudaMemsetAsync(
                m_brokenCount,
                0,
                sizeof(std::uint32_t),
                m_stream),
            "clear broken bond count");
        if (params.applyDamage)
        {
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
        }
    }

    bool graphMatches(const ExtStressGpuSolveParams& params, bool warmStart) const
    {
        return m_graphExec
            && m_graphWarmStart == warmStart
            && m_graphParams.maxIterations == params.maxIterations
            && m_graphParams.tolerance == params.tolerance
            && m_graphParams.applyDamage == params.applyDamage;
    }

    void executeSolve(const ExtStressGpuSolveParams& params)
    {
        const bool warmStart = params.warmStart && m_hasWarmStart;
        if (!graphMatches(params, warmStart))
        {
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
    std::uint32_t* m_bondIsland{nullptr};
    std::uint32_t* m_nodeIsland{nullptr};
    std::uint32_t m_islandCount{1};
    std::vector<std::uint32_t> m_hostParent;
    std::vector<std::uint32_t> m_hostRootIsland;
    std::vector<std::uint32_t> m_hostBondIsland;
    std::vector<std::uint32_t> m_hostNodeIsland;
    float* m_projectedDirectionSquared{nullptr};
    float* m_deltaSquared{nullptr};
    float* m_previousGradientSquared{nullptr};
    SolveStatus* m_status{nullptr};
    ExtStressGpuImpulse* m_hostInput{nullptr};
    AngLin* m_hostImpulses{nullptr};
    SolveStatus* m_hostStatus{nullptr};
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
