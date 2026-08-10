// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#include "NvBlastExtStressGpu.h"

#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

__global__ void initializeStatus(SolveStatus* status, std::uint32_t maxIterations)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        status->active = 1;
        status->iterations = maxIterations;
        status->converged = 0;
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

__global__ void updateDirection(
    AngLin* direction,
    const AngLin* gradient,
    const float* gradientSquared,
    const float* previousGradientSquared,
    const SolveStatus* status,
    std::uint32_t count,
    std::uint32_t iteration)
{
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (!status->active || index >= count)
    {
        return;
    }
    if (iteration == 0)
    {
        direction[index] = gradient[index];
        return;
    }
    const float denominator = *previousGradientSquared;
    const float beta = denominator > 0.0f ? *gradientSquared / denominator : 0.0f;
    direction[index].angular =
        add(gradient[index].angular, mul(direction[index].angular, beta));
    direction[index].linear =
        add(gradient[index].linear, mul(direction[index].linear, beta));
}

__global__ void saveGradientSquared(
    float* previousGradientSquared,
    const float* gradientSquared,
    const SolveStatus* status)
{
    if (threadIdx.x == 0 && blockIdx.x == 0 && status->active)
    {
        *previousGradientSquared = *gradientSquared;
    }
}

__global__ void updateSolutionAndResidual(
    AngLin* solution,
    AngLin* residual,
    const AngLin* direction,
    const AngLin* projectedDirection,
    const float* gradientSquared,
    const float* projectedDirectionSquared,
    SolveStatus* status,
    std::uint32_t bondCount,
    std::uint32_t nodeCount)
{
    if (!status->active)
    {
        return;
    }
    const float denominator = *projectedDirectionSquared;
    if (!(denominator > 0.0f) || !isfinite(denominator))
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            status->active = 0;
        }
        return;
    }
    const float step = *gradientSquared / denominator;
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < bondCount)
    {
        solution[index].angular =
            add(solution[index].angular, mul(direction[index].angular, step));
        solution[index].linear =
            add(solution[index].linear, mul(direction[index].linear, step));
    }
    if (index < nodeCount)
    {
        residual[index].angular =
            sub(residual[index].angular, mul(projectedDirection[index].angular, step));
        residual[index].linear =
            sub(residual[index].linear, mul(projectedDirection[index].linear, step));
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
    float* health,
    std::uint32_t* brokenBonds,
    std::uint32_t* brokenCount,
    std::uint32_t bondCount,
    float compressionElastic,
    float compressionFatal,
    float tensionElastic,
    float tensionFatal,
    float shearElastic,
    float shearFatal)
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

    const float compression = fmaxf(0.0f, -stressNormal);
    const float tension = fmaxf(0.0f, stressNormal);
    float multiplier = 0.0f;
    if (compression > compressionElastic)
    {
        multiplier +=
            (compression - compressionElastic)
            / fmaxf(compressionFatal - compressionElastic, 1.0f);
    }
    if (tension > tensionElastic)
    {
        multiplier +=
            (tension - tensionElastic)
            / fmaxf(tensionFatal - tensionElastic, 1.0f);
    }
    if (stressShear > shearElastic)
    {
        multiplier +=
            (stressShear - shearElastic)
            / fmaxf(shearFatal - shearElastic, 1.0f);
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
        CUcontext cudaContext)
        : m_nodeCount(nodeCount)
        , m_bondCount(bondCount)
        , m_cudaContext(cudaContext)
    {
        ContextGuard context(m_cudaContext);
        prepare(nodes, bonds);
        allocate();
        uploadTopology();
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
        cudaEventDestroy(m_eventStart);
        cudaEventDestroy(m_eventStop);
        cudaStreamDestroy(m_stream);
        cudaFree(m_reduceScratch);
        cudaFree(m_status);
        cudaFree(m_previousGradientSquared);
        cudaFree(m_deltaSquared);
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
        if (!nodeVelocities || params.maxIterations == 0)
        {
            return false;
        }

        m_telemetry = {};
        checkCuda(cudaEventRecord(m_eventStart, m_stream), "record upload start");
        checkCuda(
            cudaMemcpy(
                m_input,
                nodeVelocities,
                sizeof(ExtStressGpuImpulse) * m_nodeCount,
                cudaMemcpyHostToDevice),
            "upload stress inputs");
        checkCuda(cudaEventRecord(m_eventStop, m_stream), "record upload stop");
        checkCuda(cudaEventSynchronize(m_eventStop), "wait for input upload");
        checkCuda(
            cudaEventElapsedTime(
                &m_telemetry.uploadMilliseconds,
                m_eventStart,
                m_eventStop),
            "measure input upload");
        m_telemetry.hostToDeviceBytes =
            sizeof(ExtStressGpuImpulse) * static_cast<std::uint64_t>(m_nodeCount);

        checkCuda(cudaEventRecord(m_eventStart, m_stream), "record solve start");
        executeSolve(params);
        checkCuda(cudaEventRecord(m_eventStop, m_stream), "record solve stop");
        checkCuda(cudaEventSynchronize(m_eventStop), "wait for stress solve");
        checkCuda(cudaGetLastError(), "execute stress kernels");
        checkCuda(
            cudaEventElapsedTime(
                &m_telemetry.solveMilliseconds,
                m_eventStart,
                m_eventStop),
            "measure stress solve");

        SolveStatus status{};
        checkCuda(
            cudaMemcpy(&status, m_status, sizeof(status), cudaMemcpyDeviceToHost),
            "read stress status");
        m_telemetry.iterations = status.iterations;
        m_telemetry.converged = status.converged != 0;
        m_hasWarmStart = true;
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
        std::vector<AngLin> host(m_bondCount);
        checkCuda(cudaEventRecord(m_eventStart, m_stream), "record readback start");
        checkCuda(
            cudaMemcpy(
                host.data(),
                m_impulses,
                sizeof(AngLin) * m_bondCount,
                cudaMemcpyDeviceToHost),
            "read stress impulses");
        checkCuda(cudaEventRecord(m_eventStop, m_stream), "record readback stop");
        checkCuda(cudaEventSynchronize(m_eventStop), "wait for impulse readback");
        checkCuda(
            cudaEventElapsedTime(
                &m_telemetry.downloadMilliseconds,
                m_eventStart,
                m_eventStop),
            "measure impulse readback");
        for (std::uint32_t i = 0; i < m_bondCount; ++i)
        {
            bondImpulses[i].angular =
                {host[i].angular.x, host[i].angular.y, host[i].angular.z};
            bondImpulses[i].linear =
                {host[i].linear.x, host[i].linear.y, host[i].linear.z};
        }
        m_telemetry.deviceToHostBytes =
            sizeof(AngLin) * static_cast<std::uint64_t>(m_bondCount);
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

    void allocate()
    {
        allocateDevice(m_node0, m_bondCount, "allocate node0");
        allocateDevice(m_node1, m_bondCount, "allocate node1");
        allocateDevice(m_offset0, m_bondCount, "allocate offset0");
        allocateDevice(m_offset1, m_bondCount, "allocate offset1");
        allocateDevice(m_inertia, m_nodeCount, "allocate inertia");
        allocateDevice(m_normals, m_bondCount, "allocate normals");
        allocateDevice(m_areas, m_bondCount, "allocate areas");
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
        allocateDevice(m_gradientSquared, 1, "allocate gradient norm");
        allocateDevice(m_projectedDirectionSquared, 1, "allocate projected norm");
        allocateDevice(m_deltaSquared, 1, "allocate tolerance");
        allocateDevice(m_previousGradientSquared, 1, "allocate previous gradient norm");
        allocateDevice(m_status, 1, "allocate solve status");

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
        checkCuda(cudaEventCreate(&m_eventStart), "create start event");
        checkCuda(cudaEventCreate(&m_eventStop), "create stop event");
        checkCuda(cudaMemset(m_impulses, 0, sizeof(AngLin) * m_bondCount), "clear impulses");
        checkCuda(cudaMemset(m_brokenCount, 0, sizeof(std::uint32_t)), "clear broken count");
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
    }

    void reduce(const AngLin* values, std::uint32_t count, float* result)
    {
        squaredMagnitude<<<
            (count + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            m_stream>>>(
            values,
            m_reductionInput,
            count);
        cub::DeviceReduce::Sum(
            m_reduceScratch,
            m_reduceScratchBytes,
            m_reductionInput,
            result,
            count,
            m_stream);
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

        reduce(m_rhs, m_nodeCount, m_gradientSquared);
        setTolerance<<<1, 1, 0, m_stream>>>(
            m_deltaSquared,
            m_gradientSquared,
            params.tolerance);
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
            reduce(m_gradient, m_bondCount, m_gradientSquared);
            checkConvergence<<<1, 1, 0, m_stream>>>(
                m_status,
                m_gradientSquared,
                m_deltaSquared,
                iteration);
            updateDirection<<<
                (m_bondCount + kBlockSize - 1) / kBlockSize,
                kBlockSize,
                0,
                m_stream>>>(
                m_direction,
                m_gradient,
                m_gradientSquared,
                m_previousGradientSquared,
                m_status,
                m_bondCount,
                iteration);
            saveGradientSquared<<<1, 1, 0, m_stream>>>(
                m_previousGradientSquared,
                m_gradientSquared,
                m_status);

            rightMultiply(m_direction, m_projectedDirection);
            reduce(
                m_projectedDirection,
                m_nodeCount,
                m_projectedDirectionSquared);
            updateSolutionAndResidual<<<
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
                m_status,
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
                m_health,
                m_brokenBonds,
                m_brokenCount,
                m_bondCount,
                params.compressionElasticLimit,
                params.compressionFatalLimit,
                params.tensionElasticLimit,
                params.tensionFatalLimit,
                params.shearElasticLimit,
                params.shearFatalLimit);
        }
    }

    bool graphMatches(const ExtStressGpuSolveParams& params, bool warmStart) const
    {
        return m_graphExec
            && m_graphWarmStart == warmStart
            && m_graphParams.maxIterations == params.maxIterations
            && m_graphParams.tolerance == params.tolerance
            && m_graphParams.applyDamage == params.applyDamage
            && m_graphParams.compressionElasticLimit == params.compressionElasticLimit
            && m_graphParams.compressionFatalLimit == params.compressionFatalLimit
            && m_graphParams.tensionElasticLimit == params.tensionElasticLimit
            && m_graphParams.tensionFatalLimit == params.tensionFatalLimit
            && m_graphParams.shearElasticLimit == params.shearElasticLimit
            && m_graphParams.shearFatalLimit == params.shearFatalLimit;
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

    std::uint32_t* m_node0{nullptr};
    std::uint32_t* m_node1{nullptr};
    Vec4* m_offset0{nullptr};
    Vec4* m_offset1{nullptr};
    Inertia* m_inertia{nullptr};
    Vec4* m_normals{nullptr};
    float* m_areas{nullptr};
    float* m_nodeDistances{nullptr};
    float* m_health{nullptr};
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
    float* m_projectedDirectionSquared{nullptr};
    float* m_deltaSquared{nullptr};
    float* m_previousGradientSquared{nullptr};
    SolveStatus* m_status{nullptr};
    void* m_reduceScratch{nullptr};
    std::size_t m_reduceScratchBytes{0};
    cudaEvent_t m_eventStart{};
    cudaEvent_t m_eventStop{};
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
            reinterpret_cast<CUcontext>(cudaContext));
    }
    catch (...)
    {
        return nullptr;
    }
}

} // namespace Blast
} // namespace Nv
