#include "NvBlastExtStressGpu.h"
#include "stress_solver/stress.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

namespace
{

using namespace Nv::Blast;

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

} // namespace

int main(int argc, char** argv)
{
    const std::uint32_t width =
        argc > 1 ? static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10)) : 113;
    const std::uint32_t repetitions =
        argc > 2 ? static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 30;
    if (width < 2 || repetitions == 0)
    {
        return 1;
    }
    const std::uint32_t nodeCount = width * width;

    std::vector<ExtStressGpuNode> gpuNodes(nodeCount);
    std::vector<SolverNodeS> cpuNodes(nodeCount);
    for (std::uint32_t y = 0; y < width; ++y)
    {
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const std::uint32_t index = y * width + x;
            const float mass = index == 0 ? 0.0f : 1.0f;
            gpuNodes[index] = {
                {static_cast<float>(x), static_cast<float>(y), 0.0f},
                mass,
                mass};
            cpuNodes[index].CoM =
                {static_cast<float>(x), static_cast<float>(y), 0.0f};
            cpuNodes[index].mass = mass;
            cpuNodes[index].inertia = mass;
        }
    }

    std::vector<ExtStressGpuBond> gpuBonds;
    std::vector<SolverBond> cpuBonds;
    gpuBonds.reserve(2 * width * (width - 1));
    cpuBonds.reserve(gpuBonds.capacity());
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
    for (std::uint32_t y = 0; y < width; ++y)
    {
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const std::uint32_t index = y * width + x;
            if (x + 1 < width)
            {
                addBond(index, index + 1);
            }
            if (y + 1 < width)
            {
                addBond(index, index + width);
            }
        }
    }

    StressProcessor cpu;
    StressProcessor::DataParams data;
    data.centerBonds = true;
    data.equalizeMasses = true;
    cpu.prepare(
        cpuNodes.data(),
        nodeCount,
        cpuBonds.data(),
        static_cast<std::uint32_t>(cpuBonds.size()),
        data);

    std::unique_ptr<ExtStressGpuSolver, SolverDeleter> gpu(
        ExtStressGpuSolver::create(
            gpuNodes.data(),
            nodeCount,
            gpuBonds.data(),
            static_cast<std::uint32_t>(gpuBonds.size())));
    if (!gpu)
    {
        std::cerr << "GPU solver creation failed\n";
        return 1;
    }

    std::vector<AngLin6> cpuVelocity(nodeCount);
    std::vector<AngLin6> cpuImpulse(cpuBonds.size());
    std::vector<ExtStressGpuImpulse> gpuVelocity(nodeCount);
    auto setInput = [&](std::uint32_t frame)
    {
        const float phase = 0.0001f * static_cast<float>(frame);
        for (std::uint32_t i = 0; i < nodeCount; ++i)
        {
            const float dynamic = i == 0 ? 0.0f : 1.0f;
            const float lateral =
                dynamic * (0.01f * static_cast<float>(i % width) + phase);
            cpuVelocity[i].ang = {0.0f, 0.0f, 0.0f};
            cpuVelocity[i].lin = {lateral, -9.81f * dynamic, 0.0f};
            gpuVelocity[i].angular = {0.0f, 0.0f, 0.0f};
            gpuVelocity[i].linear = {lateral, -9.81f * dynamic, 0.0f};
        }
    };

    StressProcessor::SolverParams cpuParams;
    cpuParams.maxIter = 25;
    cpuParams.tolerance = 0.001f;
    ExtStressGpuSolveParams gpuParams;
    gpuParams.maxIterations = 25;
    gpuParams.tolerance = 0.001f;

    setInput(0);
    cpuParams.warmStart = false;
    gpuParams.warmStart = false;
    cpu.solve(cpuImpulse.data(), cpuVelocity.data(), cpuParams);
    gpu->solve(gpuVelocity.data(), gpuParams);

    cpuParams.warmStart = true;
    gpuParams.warmStart = true;
    const auto cpuStart = std::chrono::steady_clock::now();
    for (std::uint32_t frame = 1; frame <= repetitions; ++frame)
    {
        setInput(frame);
        cpu.solve(cpuImpulse.data(), cpuVelocity.data(), cpuParams);
    }
    const double cpuMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cpuStart).count()
        / repetitions;

    double gpuMilliseconds = 0.0;
    const auto gpuHostStart = std::chrono::steady_clock::now();
    for (std::uint32_t frame = 1; frame <= repetitions; ++frame)
    {
        setInput(frame);
        if (!gpu->solve(gpuVelocity.data(), gpuParams))
        {
            return 1;
        }
        gpuMilliseconds += gpu->telemetry().solveMilliseconds;
    }
    const double gpuHostMilliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - gpuHostStart).count()
        / repetitions;
    gpuMilliseconds /= repetitions;
    std::vector<std::uint32_t> broken(gpuBonds.size());
    std::uint32_t brokenCount = 0;
    if (!gpu->readbackBrokenBonds(
            broken.data(),
            static_cast<std::uint32_t>(broken.size()),
            brokenCount))
    {
        return 1;
    }

    std::cout
        << "{\n"
        << "  \"nodes\": " << nodeCount << ",\n"
        << "  \"bonds\": " << gpuBonds.size() << ",\n"
        << "  \"iterations\": 25,\n"
        << "  \"repetitions\": " << repetitions << ",\n"
        << "  \"cpuMeanMilliseconds\": " << cpuMilliseconds << ",\n"
        << "  \"gpuKernelMeanMilliseconds\": " << gpuMilliseconds << ",\n"
        << "  \"gpuHostMeanMilliseconds\": " << gpuHostMilliseconds << ",\n"
        << "  \"speedupVsCpu\": " << cpuMilliseconds / gpuHostMilliseconds << ",\n"
        << "  \"h2dBytesPerSolve\": " << gpu->telemetry().hostToDeviceBytes << ",\n"
        << "  \"compactBreakCount\": " << brokenCount << ",\n"
        << "  \"compactReadbackBytes\": " << gpu->telemetry().deviceToHostBytes << "\n"
        << "}\n";
    return 0;
}
