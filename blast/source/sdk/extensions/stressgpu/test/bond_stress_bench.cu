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

double medianOf(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 200;
    const std::uint32_t membersPerGroup = 2;

    printf("== device bond-stress walk, standalone (%d reps, %u members/group)\n",
           reps, membersPerGroup);
    printf("%9s %9s | %9s %9s %9s %9s | %9s %10s\n",
           "groups", "bonds", "host ms", "prep", "enqueue", "sync", "kernel", "ns/group");

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

        std::vector<double> host, prep, enq, sync, kern;
        for (int r = 0; r < reps; ++r)
        {
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
        const double h = medianOf(host);
        printf("%9u %9u | %9.4f %9.4f %9.4f %9.4f | %9.4f %10.2f\n",
               groups, blastBonds, h, medianOf(prep), medianOf(enq), medianOf(sync),
               medianOf(kern), h * 1e6 / groups);
        solver->release();
    }
    return 0;
}
