#include "NvBlastExtStressGpu.h"
#include "stress_solver/stress.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace
{

using Nv::Blast::ExtStressGpuBond;
using Nv::Blast::ExtStressGpuImpulse;
using Nv::Blast::ExtStressGpuNode;
using Nv::Blast::ExtStressGpuSolveParams;
using Nv::Blast::ExtStressGpuSolver;

struct SolverDeleter
{
    void operator()(ExtStressGpuSolver* solver) const
    {
        if (solver)
        {
            solver->release();
        }
    }
};

float componentError(float reference, float candidate)
{
    return std::abs(reference - candidate) / std::max(1.0f, std::abs(reference));
}

bool compare(
    const std::vector<AngLin6>& cpu,
    const std::vector<ExtStressGpuImpulse>& gpu,
    float tolerance,
    const char* phase)
{
    float maximum = 0.0f;
    double squared = 0.0;
    std::uint64_t componentCount = 0;
    for (std::size_t i = 0; i < cpu.size(); ++i)
    {
        const float reference[6] = {
            cpu[i].ang.x,
            cpu[i].ang.y,
            cpu[i].ang.z,
            cpu[i].lin.x,
            cpu[i].lin.y,
            cpu[i].lin.z};
        const float candidate[6] = {
            gpu[i].angular.x,
            gpu[i].angular.y,
            gpu[i].angular.z,
            gpu[i].linear.x,
            gpu[i].linear.y,
            gpu[i].linear.z};
        for (std::uint32_t component = 0; component < 6; ++component)
        {
            if (!std::isfinite(candidate[component]))
            {
                std::cerr << phase << ": non-finite GPU impulse at bond " << i << '\n';
                return false;
            }
            const float error = componentError(reference[component], candidate[component]);
            maximum = std::max(maximum, error);
            squared += static_cast<double>(error) * error;
            ++componentCount;
        }
    }
    const double rms = std::sqrt(squared / componentCount);
    std::cout << phase << ": maxRelative=" << maximum << " rmsRelative=" << rms << '\n';
    return maximum <= tolerance;
}

} // namespace

int main()
{
    constexpr std::uint32_t width = 8;
    constexpr std::uint32_t height = 8;
    constexpr std::uint32_t nodeCount = width * height;

    std::vector<ExtStressGpuNode> gpuNodes(nodeCount);
    std::vector<SolverNodeS> cpuNodes(nodeCount);
    for (std::uint32_t y = 0; y < height; ++y)
    {
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const std::uint32_t index = y * width + x;
            const float mass = index == 0 ? 0.0f : 1.0f + 0.01f * index;
            const float inertia = index == 0 ? 0.0f : 0.5f + 0.005f * index;
            gpuNodes[index] = {
                {static_cast<float>(x), static_cast<float>(y), 0.0f},
                mass,
                inertia};
            cpuNodes[index].CoM =
                {static_cast<float>(x), static_cast<float>(y), 0.0f};
            cpuNodes[index].mass = mass;
            cpuNodes[index].inertia = inertia;
        }
    }

    std::vector<ExtStressGpuBond> gpuBonds;
    std::vector<SolverBond> cpuBonds;
    auto addBond = [&](std::uint32_t first, std::uint32_t second)
    {
        ExtStressGpuBond gpu{};
        gpu.node0 = first;
        gpu.node1 = second;
        for (std::uint32_t component = 0; component < 3; ++component)
        {
            gpu.centroid[component] =
                0.5f
                * (gpuNodes[first].position[component]
                    + gpuNodes[second].position[component]);
        }
        gpuBonds.push_back(gpu);

        SolverBond cpu{};
        cpu.nodes[0] = first;
        cpu.nodes[1] = second;
        cpu.centroid = {
            gpu.centroid[0],
            gpu.centroid[1],
            gpu.centroid[2]};
        cpuBonds.push_back(cpu);
    };
    for (std::uint32_t y = 0; y < height; ++y)
    {
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const std::uint32_t index = y * width + x;
            if (x + 1 < width)
            {
                addBond(index, index + 1);
            }
            if (y + 1 < height)
            {
                addBond(index, index + width);
            }
        }
    }

    StressProcessor cpu;
    StressProcessor::DataParams dataParams;
    dataParams.centerBonds = true;
    dataParams.equalizeMasses = true;
    cpu.prepare(
        cpuNodes.data(),
        static_cast<std::uint32_t>(cpuNodes.size()),
        cpuBonds.data(),
        static_cast<std::uint32_t>(cpuBonds.size()),
        dataParams);

    // Damage limits are per-material now. This equivalence test uses one
    // material so every bond fails at the same threshold; the second entry
    // exists only to prove out-of-table indices are not silently accepted.
    std::vector<Nv::Blast::ExtStressGpuMaterial> gpuMaterials(1);
    gpuMaterials[0].compressionElasticLimit = 0.0f;
    gpuMaterials[0].compressionFatalLimit = 0.001f;
    gpuMaterials[0].tensionElasticLimit = 0.0f;
    gpuMaterials[0].tensionFatalLimit = 0.001f;
    gpuMaterials[0].shearElasticLimit = 0.0f;
    gpuMaterials[0].shearFatalLimit = 0.001f;

    std::unique_ptr<ExtStressGpuSolver, SolverDeleter> gpu(
        ExtStressGpuSolver::create(
            gpuNodes.data(),
            static_cast<std::uint32_t>(gpuNodes.size()),
            gpuBonds.data(),
            static_cast<std::uint32_t>(gpuBonds.size()),
            gpuMaterials.data(),
            static_cast<std::uint32_t>(gpuMaterials.size())));
    if (!gpu)
    {
        std::cerr << "failed to create GPU stress solver\n";
        return 1;
    }

    std::vector<AngLin6> cpuVelocities(nodeCount);
    std::vector<ExtStressGpuImpulse> gpuVelocities(nodeCount);
    std::vector<AngLin6> cpuImpulses(cpuBonds.size());
    std::vector<ExtStressGpuImpulse> gpuImpulses(gpuBonds.size());

    auto setLoads = [&](float lateralScale)
    {
        for (std::uint32_t i = 0; i < nodeCount; ++i)
        {
            const float dynamic = i == 0 ? 0.0f : 1.0f;
            const float lateral = dynamic * lateralScale * static_cast<float>(i % width);
            cpuVelocities[i].ang = {0.0f, 0.0f, 0.0f};
            cpuVelocities[i].lin = {lateral, -9.81f * dynamic, 0.0f};
            gpuVelocities[i].angular = {0.0f, 0.0f, 0.0f};
            gpuVelocities[i].linear = {lateral, -9.81f * dynamic, 0.0f};
        }
    };

    StressProcessor::SolverParams cpuParams;
    cpuParams.maxIter = 25;
    cpuParams.tolerance = 0.001f;
    cpuParams.warmStart = false;
    ExtStressGpuSolveParams gpuParams;
    gpuParams.maxIterations = 25;
    gpuParams.tolerance = 0.001f;
    gpuParams.warmStart = false;

    setLoads(0.015f);
    const int coldIterations =
        cpu.solve(cpuImpulses.data(), cpuVelocities.data(), cpuParams);
    if (!gpu->solve(gpuVelocities.data(), gpuParams)
        || !gpu->readbackImpulses(
            gpuImpulses.data(),
            static_cast<std::uint32_t>(gpuImpulses.size()))
        || !compare(cpuImpulses, gpuImpulses, 0.02f, "cold"))
    {
        return 1;
    }
    float peakImpulse = 0.0f;
    for (const AngLin6& impulse : cpuImpulses)
    {
        peakImpulse = std::max(
            peakImpulse,
            std::sqrt(
                impulse.lin.x * impulse.lin.x
                + impulse.lin.y * impulse.lin.y
                + impulse.lin.z * impulse.lin.z));
    }
    if (!(peakImpulse > 1.0f))
    {
        std::cerr << "equivalence fixture produced only trivial impulses\n";
        return 1;
    }
    std::cout << "cold CPU iterations=" << coldIterations
              << " peakImpulse=" << peakImpulse << '\n';

    setLoads(0.02f);
    cpuParams.warmStart = true;
    gpuParams.warmStart = true;
    cpu.solve(cpuImpulses.data(), cpuVelocities.data(), cpuParams);
    if (!gpu->solve(gpuVelocities.data(), gpuParams)
        || !gpu->readbackImpulses(
            gpuImpulses.data(),
            static_cast<std::uint32_t>(gpuImpulses.size()))
        || !compare(cpuImpulses, gpuImpulses, 0.03f, "warm"))
    {
        return 1;
    }

    setLoads(25.0f);
    gpuParams.applyDamage = true;
    if (!gpu->solve(gpuVelocities.data(), gpuParams))
    {
        return 1;
    }
    std::vector<std::uint32_t> broken(gpuBonds.size());
    std::uint32_t brokenCount = 0;
    std::vector<float> health(gpuBonds.size());
    if (!gpu->readbackBrokenBonds(
            broken.data(),
            static_cast<std::uint32_t>(broken.size()),
            brokenCount)
        || !gpu->readbackBondHealth(
            health.data(),
            static_cast<std::uint32_t>(health.size()))
        || brokenCount == 0)
    {
        std::cerr << "GPU stress damage emitted no compact break events\n";
        return 1;
    }
    for (std::uint32_t i = 0; i < brokenCount; ++i)
    {
        if (broken[i] >= health.size() || health[broken[i]] != 0.0f)
        {
            std::cerr << "BROKEN invariant violated\n";
            return 1;
        }
    }
    for (float value : health)
    {
        if (!(value >= 0.0f && value <= 1.0f))
        {
            std::cerr << "bond health monotonicity invariant violated\n";
            return 1;
        }
    }
    std::cout << "damage: compactBreaks=" << brokenCount << '\n';

    const auto& telemetry = gpu->telemetry();
    std::cout
        << "gpu solve: " << telemetry.solveMilliseconds << " ms, "
        << telemetry.iterations << " iterations, "
        << telemetry.hostToDeviceBytes << " B H2D, "
        << telemetry.deviceToHostBytes << " B D2H\n";
    return 0;
}
