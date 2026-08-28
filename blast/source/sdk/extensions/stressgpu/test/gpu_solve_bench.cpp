// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// A city-scale profiling target for the GPU stress solve.
//
// The existing equivalence harness runs 48 towers x 10 floors: 480 bonds, 48
// islands. Production runs ~24,000 nodes, ~74,000 bonds and ~2,000 islands,
// and the costs that matter there (launch overhead per iteration, per-island
// serial work, reduction contention) either do not appear at 480 bonds or
// appear with the wrong sign. Profiling the small scene would answer a
// question nobody asked.
//
// This builds a graph with production's shape and drives it with production's
// solve parameters, so `ncu` and the wall-clock numbers below describe the
// kernel mix the game actually pays for.
//
// Build and run:
//   source/sdk/extensions/stressgpu/test/bench_and_profile.sh
//
// Env:
//   BENCH_ISLANDS   number of disconnected structures   (default 2000)
//   BENCH_NODES     dynamic nodes per structure         (default 12)
//   BENCH_ITERS     solver maxIterations                (default 32)
//   BENCH_SOLVES    timed solves                        (default 200)
//   BENCH_SKIP      1 = skipSettledIslands              (default 0)
//   BENCH_STABLE    1 = feed identical velocities, so the skip can engage

#include "NvBlastExtStressGpu.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace Nv::Blast;

namespace
{

std::uint32_t envU32(const char* name, std::uint32_t fallback)
{
    if (const char* raw = std::getenv(name))
    {
        const long parsed = std::atol(raw);
        if (parsed > 0)
        {
            return static_cast<std::uint32_t>(parsed);
        }
    }
    return fallback;
}

bool envFlag(const char* name)
{
    const char* raw = std::getenv(name);
    return raw != nullptr && std::string(raw) != "0";
}

/// Production's shape: many independent structures, each a column of chunks
/// bonded to a shared static ground node. One static node for the whole scene
/// mirrors the city (every building sits on the same terrain) and, because
/// static-to-dynamic bonds are island cut points, each column is still its own
/// island — the partition the skip machinery operates on.
struct Scene
{
    std::vector<ExtStressGpuNode> nodes;
    std::vector<ExtStressGpuBond> bonds;
    std::vector<ExtStressGpuMaterial> materials;
    std::uint32_t islands = 0;
    std::uint32_t perIsland = 0;

    Scene(std::uint32_t islandCount, std::uint32_t nodesPerIsland)
        : islands(islandCount), perIsland(nodesPerIsland)
    {
        materials.resize(1);
        materials[0] = {3.0e2f, 6.0e2f, 6.0e1f, 1.2e2f, 6.0e1f, 1.2e2f};

        ExtStressGpuNode ground{};
        ground.mass = 0.0f;
        ground.inertia = 0.0f;
        nodes.push_back(ground);

        for (std::uint32_t island = 0; island < islandCount; ++island)
        {
            std::uint32_t previous = 0;
            const float x = 4.0f * static_cast<float>(island % 64);
            const float z = 4.0f * static_cast<float>(island / 64);
            for (std::uint32_t f = 0; f < nodesPerIsland; ++f)
            {
                ExtStressGpuNode node{};
                node.position[0] = x;
                node.position[1] = 1.0f + 2.0f * static_cast<float>(f);
                node.position[2] = z;
                node.mass = 900.0f;
                node.inertia = 600.0f;
                const std::uint32_t index = static_cast<std::uint32_t>(nodes.size());
                nodes.push_back(node);

                ExtStressGpuBond bond{};
                bond.node0 = previous;
                bond.node1 = index;
                bond.centroid[0] = node.position[0];
                bond.centroid[1] = node.position[1] - 1.0f;
                bond.centroid[2] = node.position[2];
                bond.normal[1] = 1.0f;
                bond.area = 4.0f;
                bond.health = bond.area;
                bond.material = 0;
                bonds.push_back(bond);

                // A lateral bond every other floor thickens the graph toward
                // the city's ~3 bonds per node instead of a pure chain's 1.
                if (f > 0 && (f % 2) == 0)
                {
                    ExtStressGpuBond brace{};
                    brace.node0 = index - 2;
                    brace.node1 = index;
                    brace.centroid[0] = node.position[0];
                    brace.centroid[1] = node.position[1] - 2.0f;
                    brace.centroid[2] = node.position[2];
                    brace.normal[1] = 1.0f;
                    brace.area = 2.0f;
                    brace.health = brace.area;
                    brace.material = 0;
                    bonds.push_back(brace);
                }
                previous = index;
            }
        }
    }
};

} // namespace

int main()
{
    const std::uint32_t islands = envU32("BENCH_ISLANDS", 2000);
    const std::uint32_t perIsland = envU32("BENCH_NODES", 12);
    const std::uint32_t iterations = envU32("BENCH_ITERS", 32);
    const std::uint32_t solves = envU32("BENCH_SOLVES", 200);
    const bool skip = envFlag("BENCH_SKIP");
    const bool stable = envFlag("BENCH_STABLE");

    Scene scene(islands, perIsland);
    std::printf(
        "scene: %zu nodes, %zu bonds, %u islands (%.2f bonds/node), "
        "iters=%u solves=%u skip=%d stable=%d\n",
        scene.nodes.size(), scene.bonds.size(), islands,
        static_cast<double>(scene.bonds.size()) / static_cast<double>(scene.nodes.size()),
        iterations, solves, skip ? 1 : 0, stable ? 1 : 0);

    ExtStressGpuSolver* solver = ExtStressGpuSolver::create(
        scene.nodes.data(),
        static_cast<std::uint32_t>(scene.nodes.size()),
        scene.bonds.data(),
        static_cast<std::uint32_t>(scene.bonds.size()),
        scene.materials.data(),
        static_cast<std::uint32_t>(scene.materials.size()));
    if (solver == nullptr)
    {
        std::fprintf(stderr, "FAIL: could not create the GPU solver\n");
        return 1;
    }

    std::vector<ExtStressGpuImpulse> velocities(scene.nodes.size());
    ExtStressGpuSolveParams params{};
    params.maxIterations = iterations;
    params.tolerance = 0.001f;
    params.warmStart = true;
    params.skipSettledIslands = skip;
    params.skipStableUnconverged = skip;
    params.applyDamage = true;

    auto drive = [&](std::uint32_t step) {
        // Every dynamic node gets a velocity; with BENCH_STABLE the same one
        // every step, so the settled skip can latch and the run measures the
        // at-rest path instead of the loaded one.
        const float phase = stable ? 1.0f : 1.0f + 0.01f * static_cast<float>(step % 17);
        for (std::size_t i = 1; i < velocities.size(); ++i)
        {
            velocities[i].linear.y = -0.35f * phase;
            velocities[i].linear.x = 0.02f * phase * static_cast<float>(i % 7);
            velocities[i].angular.z = 0.01f * phase;
        }
    };

    // Warm up: first solves allocate, capture the CUDA graph and settle the
    // launch-cap buckets. Timing those would measure setup, not steady state.
    for (std::uint32_t i = 0; i < 20; ++i)
    {
        drive(i);
        if (!solver->solve(velocities.data(), params))
        {
            std::fprintf(stderr, "FAIL: warmup solve %u\n", i);
            solver->release();
            return 1;
        }
    }

    double solveMsTotal = 0.0;
    std::uint32_t iterationTotal = 0;
    std::uint32_t skippedTotal = 0;
    const auto wallStart = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < solves; ++i)
    {
        drive(i + 100);
        if (!solver->solve(velocities.data(), params))
        {
            std::fprintf(stderr, "FAIL: timed solve %u\n", i);
            solver->release();
            return 1;
        }
        const ExtStressGpuTelemetry& telemetry = solver->telemetry();
        solveMsTotal += telemetry.solveMilliseconds;
        iterationTotal += telemetry.iterations;
        skippedTotal += telemetry.islandsSkipped;
    }
    const double wallMs =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wallStart)
            .count();

    std::printf(
        "solve: device %.3f ms/solve, wall %.3f ms/solve, mean %.1f iterations, "
        "%.0f islands skipped/solve\n",
        solveMsTotal / solves,
        wallMs / solves,
        static_cast<double>(iterationTotal) / solves,
        static_cast<double>(skippedTotal) / solves);
    // Per-iteration is the number the kernel mix is judged on: the loop body
    // is what a launch-overhead or occupancy problem shows up in.
    std::printf(
        "per-iteration: %.4f ms device (%zu bonds, %u islands)\n",
        (solveMsTotal / solves) / std::max(1.0, static_cast<double>(iterationTotal) / solves),
        scene.bonds.size(),
        islands);

    solver->release();
    return 0;
}
