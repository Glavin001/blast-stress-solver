// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Does skipping settled islands change the answer?
//
// The claim under test is that ExtStressGpuSolveParams::skipSettledIslands is
// free: an island whose inputs are bit-identical to its last solve, and whose
// last solve converged, keeps impulses that a re-solve would reproduce. A
// feature that claims to be free has to prove it, so this drives THREE solvers
// over one scripted velocity stream:
//
//   offA, offB  -- both with skipping disabled
//   on          -- skipping enabled
//
// offA vs offB is the noise floor. The GPU solve accumulates node loads with
// float atomicAdd, whose ordering is not reproducible, so two identical runs
// already differ in the last few bits and nothing here can be asserted
// bit-for-bit. The test is therefore: `on` must not differ from `offA` by more
// than `offB` does. Comparing against a fixed tolerance instead would either
// pass a real regression or fail on the hardware's own jitter.
//
// The fingerprint that must match exactly is the physics one: which bonds
// broke, and how many. Impulses are compared as a distribution.
//
// Build and run:
//   source/sdk/extensions/stressgpu/test/build_and_run.sh

#include "NvBlastExtStressGpu.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace Nv::Blast;

namespace
{

int g_failures = 0;

void check(bool condition, const std::string& what)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
    else
    {
        std::fprintf(stderr, "ok:   %s\n", what.c_str());
    }
}

/// Towers of stacked cubes, each tower bonded to the ground node at index 0.
/// Static-to-dynamic bonds are cut points, so every tower is its own island --
/// the same partition rule the CPU solver uses, and the thing the skip
/// operates on.
struct Scene
{
    static constexpr std::uint32_t kTowers = 48;
    static constexpr std::uint32_t kFloors = 10;

    std::vector<ExtStressGpuNode> nodes;
    std::vector<ExtStressGpuBond> bonds;
    std::vector<ExtStressGpuMaterial> materials;
    /// Which tower (island) each dynamic node belongs to, for the driver.
    std::vector<std::uint32_t> nodeTower;

    Scene()
    {
        // Weak enough that the scripted load actually breaks bonds. A run in
        // which nothing breaks proves nothing about the damage path, which is
        // where a skipped island's frozen stress would show up first.
        materials.resize(1);
        materials[0] = {3.0e2f, 6.0e2f, 6.0e1f, 1.2e2f, 6.0e1f, 1.2e2f};

        ExtStressGpuNode ground{};
        ground.position[0] = 0.0f;
        ground.position[1] = 0.0f;
        ground.position[2] = 0.0f;
        ground.mass = 0.0f;         // static: a boundary, and an island cut point
        ground.inertia = 0.0f;
        nodes.push_back(ground);
        nodeTower.push_back(0xFFFFFFFFu);

        for (std::uint32_t t = 0; t < kTowers; ++t)
        {
            std::uint32_t previous = 0;     // ground
            for (std::uint32_t f = 0; f < kFloors; ++f)
            {
                ExtStressGpuNode node{};
                node.position[0] = 4.0f * static_cast<float>(t);
                node.position[1] = 1.0f + 2.0f * static_cast<float>(f);
                node.position[2] = 0.0f;
                node.mass = 900.0f;
                node.inertia = 600.0f;
                const std::uint32_t index = static_cast<std::uint32_t>(nodes.size());
                nodes.push_back(node);
                nodeTower.push_back(t);

                ExtStressGpuBond bond{};
                bond.node0 = previous;
                bond.node1 = index;
                bond.centroid[0] = node.position[0];
                bond.centroid[1] = node.position[1] - 1.0f;
                bond.centroid[2] = node.position[2];
                bond.normal[0] = 0.0f;
                bond.normal[1] = 1.0f;
                bond.normal[2] = 0.0f;
                bond.area = 4.0f;
                bond.health = bond.area;
                bond.material = 0;
                bonds.push_back(bond);
                previous = index;
            }
        }
    }
};

/// One solver plus the host mirror a real caller keeps: the whole point of
/// lastChangedBonds is that the mirror is updated in place, so a bug that left
/// a stale impulse behind would show up here and nowhere else.
struct Runner
{
    ExtStressGpuSolver* solver = nullptr;
    std::vector<ExtStressGpuImpulse> impulses;
    std::vector<float> health;
    std::uint64_t brokenTotal = 0;
    std::uint32_t skippedTotal = 0;
    std::uint32_t solves = 0;
    std::uint32_t damaged = 0;
    std::uint32_t converged = 0;
    std::uint64_t iterations = 0;
    std::vector<std::uint32_t> brokenPerTick;

    Runner(const Scene& scene, bool skip)
        : skipSettled(skip)
    {
        solver = ExtStressGpuSolver::create(
            scene.nodes.data(),
            static_cast<std::uint32_t>(scene.nodes.size()),
            scene.bonds.data(),
            static_cast<std::uint32_t>(scene.bonds.size()),
            scene.materials.data(),
            static_cast<std::uint32_t>(scene.materials.size()));
        impulses.assign(scene.bonds.size(), ExtStressGpuImpulse{});
        health.assign(scene.bonds.size(), 0.0f);
    }

    ~Runner()
    {
        if (solver)
        {
            solver->release();
        }
    }

    void step(const std::vector<ExtStressGpuImpulse>& velocities, std::uint32_t tick)
    {
        ExtStressGpuSolveParams params;
        // Alternating budgets. The loose half converges in a couple of
        // iterations, so almost every settled island is skippable; the tight
        // half cannot reach tolerance inside the budget, so islands come back
        // CLEAN BUT UNCONVERGED and must not be skipped -- freezing an
        // under-converged island keeps stale, inflated stress alive and
        // carries on breaking bonds off it. Without this half the convergence
        // half of the predicate is never exercised. It also flips the
        // capture-time parameters, so the CUDA graph is rebuilt mid-run.
        const bool loose = (tick % 40) < 20;
        params.maxIterations = loose ? 32 : 4;
        params.tolerance = loose ? 1.0e-3f : 1.0e-9f;
        params.warmStart = true;
        params.applyDamage = true;
        params.skipSettledIslands = skipSettled;
        const bool okay = solver->solveAndReadbackImpulses(
            velocities.data(),
            params,
            impulses.data(),
            static_cast<std::uint32_t>(impulses.size()));
        if (!okay)
        {
            std::fprintf(stderr, "FAIL: solveAndReadbackImpulses returned false\n");
            ++g_failures;
            return;
        }
        ++solves;
        skippedTotal += solver->telemetry().islandsSkipped;

        std::uint32_t brokenCount = 0;
        std::vector<std::uint32_t> broken(impulses.size());
        solver->readbackBrokenBonds(
            broken.data(), static_cast<std::uint32_t>(broken.size()), brokenCount);
        brokenTotal += brokenCount;
        brokenPerTick.push_back(brokenCount);
        solver->readbackBondHealth(health.data(), static_cast<std::uint32_t>(health.size()));
        damaged = 0;
        for (std::size_t i = 0; i < health.size(); ++i)
        {
            damaged += (health[i] < 4.0f) ? 1u : 0u;
        }
        converged += solver->telemetry().converged ? 1u : 0u;
        iterations += solver->telemetry().iterations;
    }

    bool skipSettled;
};

/// Largest difference between two impulse mirrors, as a fraction of the
/// largest impulse in the frame.
///
/// Scaled globally rather than per bond on purpose: a bond carrying
/// essentially no load reads 1e-9 in one run and 1.3e-9 in the next, which is
/// a 23% "relative" difference and pure noise. Measuring against the frame's
/// own scale asks the question that matters -- is any bond's stress
/// meaningfully different -- instead of amplifying the bonds that carry
/// nothing.
double maxScaledDifference(
    const std::vector<ExtStressGpuImpulse>& a,
    const std::vector<ExtStressGpuImpulse>& b)
{
    double scale = 0.0;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const float lhs[6] = {
            a[i].angular.x, a[i].angular.y, a[i].angular.z,
            a[i].linear.x, a[i].linear.y, a[i].linear.z};
        const float rhs[6] = {
            b[i].angular.x, b[i].angular.y, b[i].angular.z,
            b[i].linear.x, b[i].linear.y, b[i].linear.z};
        for (int k = 0; k < 6; ++k)
        {
            scale = std::max(scale, static_cast<double>(std::fabs(lhs[k])));
            scale = std::max(scale, static_cast<double>(std::fabs(rhs[k])));
            worst = std::max(worst, std::fabs(static_cast<double>(lhs[k] - rhs[k])));
        }
    }
    return scale > 0.0 ? worst / scale : 0.0;
}

double maxHealthDifference(const std::vector<float>& a, const std::vector<float>& b)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        worst = std::max(worst, std::fabs(static_cast<double>(a[i] - b[i])));
    }
    return worst;
}

}   // namespace

int main()
{
    Scene scene;
    const std::uint32_t nodeCount = static_cast<std::uint32_t>(scene.nodes.size());

    Runner offA(scene, false);
    Runner offB(scene, false);
    Runner on(scene, true);
    if (!offA.solver || !offB.solver || !on.solver)
    {
        std::fprintf(stderr, "FAIL: no CUDA stress solver (no GPU?)\n");
        return 1;
    }

    // The velocity stream. Towers below `settledTowers` reach a fixed load and
    // then repeat it bit-for-bit -- which is what a settled island looks like
    // to the solver, and the only thing the skip is allowed to react to. The
    // rest keep moving, so the solve can never be skipped wholesale.
    const std::uint32_t settledTowers = Scene::kTowers * 3 / 4;
    const std::uint32_t ticks = 160;
    std::vector<ExtStressGpuImpulse> velocities(nodeCount);

    double worstOnVsA = 0.0;
    double worstBVsA = 0.0;
    double worstHealthOn = 0.0;
    double worstHealthB = 0.0;
    std::uint32_t removedAt = 90;

    for (std::uint32_t tick = 0; tick < ticks; ++tick)
    {
        for (std::uint32_t i = 0; i < nodeCount; ++i)
        {
            const std::uint32_t tower = scene.nodeTower[i];
            if (tower == 0xFFFFFFFFu)
            {
                velocities[i] = ExtStressGpuImpulse{};
                continue;
            }
            const bool settles = tower < settledTowers && tick >= 20;
            // A settled tower's input is a pure function of the tower, so it
            // is reproduced to the bit every tick from tick 20 on.
            const float phase = settles
                ? static_cast<float>(tower)
                : static_cast<float>(tower) + 0.017f * static_cast<float>(tick);
            velocities[i].linear = {
                0.02f * std::sin(phase),
                -0.35f - 0.01f * std::cos(phase),
                0.02f * std::cos(phase)};
            velocities[i].angular = {0.001f * std::sin(phase), 0.0f, 0.0f};
        }

        // Mid-run topology change: a bond breaks out from under all three
        // solvers. Skipping must survive it, and must survive it identically.
        if (tick == removedAt)
        {
            const std::uint32_t victim = 17;
            offA.solver->removeBond(victim);
            offB.solver->removeBond(victim);
            on.solver->removeBond(victim);
        }

        offA.step(velocities, tick);
        offB.step(velocities, tick);
        on.step(velocities, tick);

        worstOnVsA = std::max(worstOnVsA, maxScaledDifference(offA.impulses, on.impulses));
        worstBVsA = std::max(worstBVsA, maxScaledDifference(offA.impulses, offB.impulses));
        worstHealthOn = std::max(worstHealthOn, maxHealthDifference(offA.health, on.health));
        worstHealthB = std::max(worstHealthB, maxHealthDifference(offA.health, offB.health));
    }

    std::fprintf(
        stderr,
        "\nislands skipped: on=%u over %u solves, offA=%u, offB=%u\n",
        on.skippedTotal, on.solves, offA.skippedTotal, offB.skippedTotal);
    std::fprintf(
        stderr,
        "broken bonds:    on=%llu offA=%llu offB=%llu  (damaged now on=%u offA=%u)\n",
        static_cast<unsigned long long>(on.brokenTotal),
        static_cast<unsigned long long>(offA.brokenTotal),
        static_cast<unsigned long long>(offB.brokenTotal),
        on.damaged,
        offA.damaged);
    std::fprintf(
        stderr,
        "max rel impulse: on vs offA = %.3e   offB vs offA = %.3e (noise floor)\n",
        worstOnVsA, worstBVsA);
    std::fprintf(
        stderr,
        "max health diff: on vs offA = %.3e   offB vs offA = %.3e (noise floor)\n\n",
        worstHealthOn, worstHealthB);

    std::fprintf(
        stderr,
        "convergence:     on=%u/%u solves converged, mean %.1f iterations; offA=%u/%u\n\n",
        on.converged, on.solves,
        on.solves ? static_cast<double>(on.iterations) / on.solves : 0.0,
        offA.converged, offA.solves);
    check(on.skippedTotal > 0, "skipping actually fires with skipSettledIslands on");
    check(offA.skippedTotal == 0 && offB.skippedTotal == 0,
          "skipping never fires with skipSettledIslands off");
    check(offA.brokenTotal > 0, "the scripted load actually breaks bonds");
    check(on.damaged > 0, "the scripted load actually damages bonds");
    check(on.brokenTotal == offA.brokenTotal && offA.brokenTotal == offB.brokenTotal,
          "identical broken-bond count with and without skipping");
    check(on.brokenPerTick == offA.brokenPerTick,
          "identical broken-bond count on every single tick, not just in total");
    // The noise floor may be exactly zero on a run where the atomics happened
    // to line up, so allow the skip a floor of one float epsilon on top of it.
    const double floorRel = std::max(worstBVsA, 1.0e-6);
    (void)removedAt;
    const double floorHealth = std::max(worstHealthB, 1.0e-3);
    check(worstOnVsA <= floorRel,
          "impulses with skipping differ no more than two skip-off runs do");
    check(worstHealthOn <= floorHealth,
          "bond healths with skipping differ no more than two skip-off runs do");

    std::fprintf(stderr, g_failures ? "\nFAILED (%d)\n" : "\nPASSED\n", g_failures);
    return g_failures ? 1 : 0;
}
