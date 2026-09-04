// Standalone timing harness for the device bond-stress walk.
//
// Drives the real ExtStressGpuSolver API on a synthetic graph, outside the
// game, so the cost of the walk can be separated from whatever else is on the
// GPU and swept against problem size. In-game the host was seen blocking
// 0.878 ms on ~0.18 ms of device work, and there is no way to tell from inside
// a 4200-tick trace whether that is inherent launch/sync latency, contention
// from another process, or something the implementation is doing to itself.
//
// Reports, per group count: the phase split, and the derived scaling.

#include "NvBlastExtStressGpu.h"
#include "NvBlastExtStressFormula.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>

using namespace Nv::Blast;

namespace
{

struct AngLinHost { float ang[3]; float lin[3]; };

struct Scene
{
    std::vector<ExtStressGpuNode> nodes;
    std::vector<ExtStressGpuBond> bonds;
    std::vector<ExtStressGpuMaterial> materials;
    // Flattened bond-stress topology.
    std::vector<std::uint32_t> groupBegin, groupSize, memberBlastBond;
    std::vector<std::uint32_t> bondNode0, bondNode1, bondMaterial;
    std::vector<float> bondNormal, bondCentroid, bondNodeDisp;
    std::vector<float> health, materialElastic;
};

/// A chain of nodes, each solver bond aggregating `membersPerGroup` blast
/// bonds -- the same shape the city produces (1-4 members per group).
Scene makeScene(std::uint32_t groups, std::uint32_t membersPerGroup)
{
    Scene s;
    const std::uint32_t nodeCount = groups + 1;
    const std::uint32_t blastBonds = groups * membersPerGroup;

    s.nodes.resize(nodeCount);
    for (std::uint32_t i = 0; i < nodeCount; ++i)
    {
        s.nodes[i].position[0] = static_cast<float>(i);
        s.nodes[i].position[1] = 0.0f;
        s.nodes[i].position[2] = 0.0f;
        s.nodes[i].mass = 1.0f;
        s.nodes[i].inertia = 1.0f;
    }
    s.bonds.resize(groups);
    for (std::uint32_t g = 0; g < groups; ++g)
    {
        s.bonds[g].node0 = g;
        s.bonds[g].node1 = g + 1;
        s.bonds[g].centroid[0] = static_cast<float>(g) + 0.5f;
        s.bonds[g].centroid[1] = 0.0f;
        s.bonds[g].centroid[2] = 0.0f;
        s.bonds[g].normal[0] = 1.0f;
        s.bonds[g].normal[1] = 0.0f;
        s.bonds[g].normal[2] = 0.0f;
        s.bonds[g].area = 1.0f;
        s.bonds[g].health = 1.0f;
        s.bonds[g].material = 0;
    }
    s.materials.resize(1);
    s.materialElastic = {1.0e6f, 1.0e6f, 1.0e6f};

    s.groupBegin.resize(groups);
    s.groupSize.resize(groups);
    s.memberBlastBond.resize(blastBonds);
    s.bondNode0.resize(blastBonds);
    s.bondNode1.resize(blastBonds);
    s.bondMaterial.assign(blastBonds, 0);
    s.bondNormal.resize(3ull * blastBonds);
    s.bondCentroid.resize(3ull * blastBonds);
    s.bondNodeDisp.resize(3ull * blastBonds);
    s.health.assign(blastBonds, 0.25f);

    std::uint32_t slot = 0;
    for (std::uint32_t g = 0; g < groups; ++g)
    {
        s.groupBegin[g] = slot;
        s.groupSize[g] = membersPerGroup;
        for (std::uint32_t k = 0; k < membersPerGroup; ++k)
        {
            const std::uint32_t bb = slot;
            s.memberBlastBond[slot] = bb;
            s.bondNode0[bb] = g;
            s.bondNode1[bb] = g + 1;
            s.bondNormal[3ull * bb + 0] = 1.0f;
            s.bondCentroid[3ull * bb + 0] = static_cast<float>(g) + 0.5f;
            s.bondNodeDisp[3ull * bb + 0] = 1.0f;
            ++slot;
        }
    }
    return s;
}

ExtStressGpuBondStressTopology topologyOf(const Scene& s, std::uint32_t groups,
                                          std::uint32_t blastBonds, bool csrDirty)
{
    ExtStressGpuBondStressTopology t{};
    t.groupCount = groups;
    t.memberSlotCount = blastBonds;
    t.graphNodeCount = groups + 1;
    t.blastBondCount = blastBonds;
    t.groupBegin = s.groupBegin.data();
    t.groupSize = s.groupSize.data();
    t.memberBlastBond = s.memberBlastBond.data();
    t.bondNode0 = s.bondNode0.data();
    t.bondNode1 = s.bondNode1.data();
    t.bondMaterial = s.bondMaterial.data();
    t.bondNormal = s.bondNormal.data();
    t.bondCentroid = s.bondCentroid.data();
    t.bondNodeDisp = s.bondNodeDisp.data();
    t.materialElasticLimits = s.materialElastic.data();
    t.materialCount = 1;
    t.csrDirty = csrDirty;
    return t;
}

/// The serial walk, transliterated from SupportGraphProcessor::processBondGroup
/// against the same flattened arrays. This is the thing the GPU path replaces,
/// so the benchmark can state a speedup rather than a device timing.
void cpuWalk(const Scene& s, std::uint32_t groups,
             std::vector<float>& sn, std::vector<float>& ss,
             std::vector<std::uint8_t>& nodeFlag, std::uint32_t& overstressed,
             std::vector<std::uint32_t>& removals, const std::vector<AngLinHost>& impulses)
{
    const float unbreakable = 0.5f * 3.4028235e38f;
    overstressed = 0;
    removals.clear();
    std::fill(nodeFlag.begin(), nodeFlag.end(), 0);
    for (std::uint32_t g = 0; g < groups; ++g)
    {
        const std::uint32_t begin = s.groupBegin[g];
        const std::uint32_t size = s.groupSize[g];
        float totalArea = 0.0f;
        float nx = 0, ny = 0, nz = 0, cx = 0, cy = 0, cz = 0, dx = 0, dy = 0, dz = 0;
        for (std::uint32_t k = 0; k < size; ++k)
        {
            const std::uint32_t bb = s.memberBlastBond[begin + k];
            const float area = s.health[bb];
            if (area > 0.0f)
            {
                const std::size_t b = 3ull * bb;
                if (!(area < unbreakable))
                {
                    totalArea = unbreakable;
                    nx = s.bondNormal[b]; ny = s.bondNormal[b + 1]; nz = s.bondNormal[b + 2];
                    dx = s.bondNodeDisp[b]; dy = s.bondNodeDisp[b + 1]; dz = s.bondNodeDisp[b + 2];
                    break;
                }
                nx = fmaf(s.bondNormal[b], area, nx);
                ny = fmaf(s.bondNormal[b + 1], area, ny);
                nz = fmaf(s.bondNormal[b + 2], area, nz);
                cx = fmaf(s.bondCentroid[b], area, cx);
                cy = fmaf(s.bondCentroid[b + 1], area, cy);
                cz = fmaf(s.bondCentroid[b + 2], area, cz);
                dx = fmaf(s.bondNodeDisp[b], area, dx);
                dy = fmaf(s.bondNodeDisp[b + 1], area, dy);
                dz = fmaf(s.bondNodeDisp[b + 2], area, dz);
                totalArea += area;
            }
            else
            {
                removals.push_back(bb);
            }
        }
        if (totalArea == 0.0f) { continue; }
        const float mag = sqrtf(fmaf(nz, nz, fmaf(ny, ny, nx * nx)));
        if (!(mag < 1e-20f)) { const float inv = 1.0f / mag; nx *= inv; ny *= inv; nz *= inv; }
        if (totalArea > 0.0f && totalArea < unbreakable)
        {
            const float inv = 1.0f / totalArea;
            cx *= inv; cy *= inv; cz *= inv; dx *= inv; dy *= inv; dz *= inv;
        }
        float stressNormal = 0.0f, stressShear = 0.0f;
        if (totalArea > 0.0f && totalArea < unbreakable)
        {
            const AngLinHost& im = impulses[g];
            const float nodeDist = sqrtf(fmaf(dz, dz, fmaf(dy, dy, dx * dx)));
            extStressCalcBondStress(
                ExtStressVec3{im.lin[0], im.lin[1], im.lin[2]},
                ExtStressVec3{im.ang[0], im.ang[1], im.ang[2]},
                ExtStressVec3{nx, ny, nz}, totalArea, nodeDist, stressNormal, stressShear);
        }
        sn[g] = stressNormal;
        ss[g] = stressShear;
        for (std::uint32_t k = 0; k < size; ++k)
        {
            const std::uint32_t bb = s.memberBlastBond[begin + k];
            if (!(s.health[bb] > 0.0f)) { continue; }
            if (-stressNormal > s.materialElastic[0]
                || stressNormal > s.materialElastic[1]
                || stressShear > s.materialElastic[2])
            {
                ++overstressed;
                nodeFlag[s.bondNode0[bb]] = 1;
                nodeFlag[s.bondNode1[bb]] = 1;
            }
        }
    }
}

double medianOf(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 200;
    // Microseconds of idle between reps. The game submits this walk once per
    // 16 ms tick; a tight loop keeps the context hot and hides whatever a cold
    // submission costs. This makes the harness able to reproduce the game's
    // duty cycle.
    const int idleUs = argc > 2 ? atoi(argv[2]) : 0;
    std::uint64_t mismatches = 0;
    const std::uint32_t membersPerGroup = 2;

    printf("== device bond-stress walk, standalone (%d reps, %u members/group, %d us idle between reps)\n",
           reps, membersPerGroup, idleUs);
    printf("%9s %9s | %9s %9s %9s %9s | %9s %8s %9s %9s\n",
           "groups", "bonds", "gpu ms", "prep", "enqueue", "sync", "kernel",
           "cpu ms", "speedup", "gpu ns/grp");

    for (std::uint32_t groups : {1000u, 4000u, 16000u, 32000u, 64000u, 128000u, 256000u})
    {
        const std::uint32_t blastBonds = groups * membersPerGroup;
        Scene s = makeScene(groups, membersPerGroup);

        ExtStressGpuSolver* solver = ExtStressGpuSolver::create(
            s.nodes.data(), static_cast<std::uint32_t>(s.nodes.size()),
            s.bonds.data(), static_cast<std::uint32_t>(s.bonds.size()),
            s.materials.data(), 1, nullptr);
        if (solver == nullptr)
        {
            printf("  solver create failed at %u groups\n", groups);
            continue;
        }

        auto topo = topologyOf(s, groups, blastBonds, true);
        if (!solver->setBondStressTopology(topo))
        {
            printf("  setBondStressTopology failed at %u groups\n", groups);
            solver->release();
            continue;
        }

        ExtStressGpuBondStressResult result{};
        // Warm up: first call allocates and uploads.
        solver->updateBondStress(topo, s.health.data(), 0.5f * 3.4028235e38f, result);
        // The CPU reference must read the impulses the device actually holds.
        std::vector<ExtStressGpuImpulse> devImp(groups);
        solver->readbackImpulses(devImp.data(), groups);

        std::vector<double> host, prep, enq, sync, kern;
        for (int r = 0; r < reps; ++r)
        {
            if (idleUs > 0)
            {
                const auto until = std::chrono::steady_clock::now()
                    + std::chrono::microseconds(idleUs);
                while (std::chrono::steady_clock::now() < until) { }
            }
            auto t = topologyOf(s, groups, blastBonds, false);
            const auto t0 = std::chrono::steady_clock::now();
            solver->updateBondStress(t, s.health.data(), 0.5f * 3.4028235e38f, result);
            const auto t1 = std::chrono::steady_clock::now();
            host.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            const ExtStressGpuTelemetry& tel = solver->telemetry();
            prep.push_back(tel.bondStressPrepMs);
            enq.push_back(tel.bondStressEnqueueMs);
            sync.push_back(tel.bondStressSyncMs);
            kern.push_back(tel.bondStressKernelMs);
        }
        // Same walk on the host, for the speedup.
        std::vector<float> csn(groups, 0.0f), css(groups, 0.0f);
        std::vector<std::uint8_t> cflag(groups + 1, 0);
        std::vector<std::uint32_t> crem;
        std::vector<AngLinHost> imp(groups);
        for (std::uint32_t g = 0; g < groups; ++g)
        {
            imp[g].ang[0] = devImp[g].angular.x;
            imp[g].ang[1] = devImp[g].angular.y;
            imp[g].ang[2] = devImp[g].angular.z;
            imp[g].lin[0] = devImp[g].linear.x;
            imp[g].lin[1] = devImp[g].linear.y;
            imp[g].lin[2] = devImp[g].linear.z;
        }
        std::uint32_t cover = 0;
        std::vector<double> cpu;
        for (int r = 0; r < reps; ++r)
        {
            const auto c0 = std::chrono::steady_clock::now();
            cpuWalk(s, groups, csn, css, cflag, cover, crem, imp);
            const auto c1 = std::chrono::steady_clock::now();
            cpu.push_back(std::chrono::duration<double, std::milli>(c1 - c0).count());
        }
        // Correctness cross-check at every size, not just timing: the walk
        // kernel is otherwise only exercised in-game, where a failure is
        // expensive to localise.
        {
            const float* gsn = nullptr;
            const float* gss = nullptr;
            const float* gsb = nullptr;
            if (solver->readbackGroupStresses(gsn, gss, gsb) && gsn != nullptr)
            {
                std::uint32_t bad = 0;
                for (std::uint32_t g = 0; g < groups; ++g)
                {
                    if (memcmp(&gsn[g], &csn[g], sizeof(float)) != 0
                        || memcmp(&gss[g], &css[g], sizeof(float)) != 0)
                    {
                        ++bad;
                    }
                }
                if (bad != 0)
                {
                    printf("  MISMATCH at %u groups: %u of %u differ from the CPU walk\n",
                           groups, bad, groups);
                    mismatches += bad;
                }
            }
            if (result.overstressedBondCount != cover)
            {
                printf("  MISMATCH at %u groups: overstressed %u (gpu) vs %u (cpu)\n",
                       groups, result.overstressedBondCount, cover);
                ++mismatches;
            }
        }

        const double h = medianOf(host);
        const double c = medianOf(cpu);
        printf("%9u %9u | %9.4f %9.4f %9.4f %9.4f | %9.4f %8.4f %8.2fx %9.2f\n",
               groups, blastBonds, h, medianOf(prep), medianOf(enq), medianOf(sync),
               medianOf(kern), c, c / h, h * 1e6 / groups);
        solver->release();
    }
    printf("\n%s (%llu cross-check mismatches against the CPU walk)\n",
           mismatches == 0 ? "PASSED" : "FAILED", (unsigned long long)mismatches);
    return mismatches == 0 ? 0 : 1;
}
