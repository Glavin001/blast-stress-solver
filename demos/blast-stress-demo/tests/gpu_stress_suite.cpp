// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
//
// Topology stress suite for the GPU stress solve.
//
// WHY THIS EXISTS
//
// The solver's cost and its correctness both depend on the shape of the island
// partition, and that shape changes completely over a structure's life:
//
//   intact          a handful of islands of thousands of nodes each
//   under fire      a few large islands plus a growing cloud of small ones
//   shattered       hundreds of thousands of islands of two to four chunks
//   dust            every bond broken; every chunk free, nothing to solve
//
// These stress opposite things. The intact end is bandwidth and load balance
// across a few huge islands. The shattered end is per-island overhead, launch
// count, and reduction contention across a vast number of tiny ones -- and it
// is also where the solver's rigid-body null space actually shows up, because
// a fragment that has come off the ground is unanchored and its operator is
// singular. A change validated only on the intact scene is validated on half
// the problem, and it is the easy half.
//
// So this measures the whole trajectory, on the REAL pack rather than a
// synthetic model of it, and it checks correctness at every point rather than
// only speed.
//
// The suite deliberately covers combinations that "cannot happen" in a tidy
// run -- an unanchored fragment under gravity, an island of exactly one bond,
// a scene where every bond is gone -- because over a long session they all do.
//
//   ./gpu_stress_suite [--pack <json>] [--grid N] [--iters N] [--solves N]
//                      [--only <name>] [--histogram] [--progressive] [--quick]

#include "NvBlastExtStressGpu.h"
#include "scene_pack.h"
#include "stress_solver/stress.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <map>
#include <string>
#include <vector>

using namespace Nv::Blast;
using blast_demo::ScenePack;

namespace
{

// ---------------------------------------------------------------- utilities

/// Deterministic. A fixed sequence matters more than a good one: two runs must
/// fracture identically or their timings describe different scenes.
struct Lcg
{
    std::uint64_t state;
    explicit Lcg(std::uint64_t seed) : state(seed ? seed : 1u) {}
    std::uint32_t next()
    {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(state >> 33);
    }
    std::uint32_t below(std::uint32_t n) { return n ? next() % n : 0u; }
};

struct UnionFind
{
    std::vector<std::uint32_t> parent;
    explicit UnionFind(std::size_t n) : parent(n)
    {
        for (std::size_t i = 0; i < n; ++i) parent[i] = static_cast<std::uint32_t>(i);
    }
    std::uint32_t find(std::uint32_t a)
    {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    }
    void unite(std::uint32_t a, std::uint32_t b)
    {
        a = find(a); b = find(b);
        if (a != b) parent[b] = a;
    }
};

double processCpuMs()
{
    timespec ts{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return 0.0;
    return static_cast<double>(ts.tv_sec) * 1000.0 + static_cast<double>(ts.tv_nsec) / 1.0e6;
}

// -------------------------------------------------------------------- scene

struct Scene
{
    std::vector<ExtStressGpuNode> nodes;
    std::vector<ExtStressGpuBond> bonds;
    std::vector<ExtStressGpuMaterial> materials;
};

/// Island statistics as the SOLVER sees them.
///
/// Two rules have to match `computeIslands` exactly or the numbers describe a
/// different graph than the one being solved:
///   - union only across bonds whose BOTH endpoints are dynamic, because a
///     zero-mass node is a boundary shared by everything resting on it;
///   - a dynamic node with no surviving bond carries no unknown and gets no
///     island at all. It is a free chunk, not an island of one.
struct IslandStats
{
    std::vector<std::uint32_t> nodesPerIsland;   // descending
    std::vector<std::uint32_t> bondsPerIsland;   // descending
    std::uint64_t solvedBonds{0};
    std::uint32_t freeChunks{0};
    std::uint32_t staticNodes{0};
    std::uint32_t anchoredBonds{0};
    std::uint32_t unanchoredIslands{0};          // no static neighbour: singular operator
    std::uint32_t largestIslandNodes{0};
};

IslandStats computeIslandStats(const Scene& scene)
{
    IslandStats st;
    const std::size_t n = scene.nodes.size();
    UnionFind uf(n);
    std::vector<std::uint32_t> degree(n, 0u);

    for (const auto& b : scene.bonds)
    {
        ++degree[b.node0]; ++degree[b.node1];
        if (scene.nodes[b.node0].mass > 0.0f && scene.nodes[b.node1].mass > 0.0f)
            uf.unite(b.node0, b.node1);
    }

    std::map<std::uint32_t, std::uint32_t> nodesPer, bondsPer;
    std::map<std::uint32_t, bool> anchored;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (scene.nodes[i].mass <= 0.0f) { ++st.staticNodes; continue; }
        if (degree[i] == 0u) { ++st.freeChunks; continue; }
        const std::uint32_t root = uf.find(static_cast<std::uint32_t>(i));
        ++nodesPer[root];
        anchored.emplace(root, false);
    }
    for (const auto& b : scene.bonds)
    {
        const bool d0 = scene.nodes[b.node0].mass > 0.0f;
        const bool d1 = scene.nodes[b.node1].mass > 0.0f;
        if (d0 && d1) { ++bondsPer[uf.find(b.node0)]; }
        else if (d0)  { ++bondsPer[uf.find(b.node0)]; ++st.anchoredBonds; anchored[uf.find(b.node0)] = true; }
        else if (d1)  { ++bondsPer[uf.find(b.node1)]; ++st.anchoredBonds; anchored[uf.find(b.node1)] = true; }
    }
    for (const auto& kv : anchored) if (!kv.second) ++st.unanchoredIslands;
    for (const auto& kv : nodesPer) st.nodesPerIsland.push_back(kv.second);
    for (const auto& kv : bondsPer) { st.bondsPerIsland.push_back(kv.second); st.solvedBonds += kv.second; }
    std::sort(st.nodesPerIsland.begin(), st.nodesPerIsland.end(), std::greater<std::uint32_t>());
    std::sort(st.bondsPerIsland.begin(), st.bondsPerIsland.end(), std::greater<std::uint32_t>());
    st.largestIslandNodes = st.nodesPerIsland.empty() ? 0u : st.nodesPerIsland[0];
    return st;
}

void printIslandStats(const IslandStats& st, const char* label)
{
    std::printf("\n--- island histogram [%s] ---\n", label);
    std::printf("islands=%zu  solved bonds=%llu  free chunks=%u  static=%u  "
                "anchored bonds=%u  UNANCHORED islands=%u\n",
                st.nodesPerIsland.size(),
                static_cast<unsigned long long>(st.solvedBonds),
                st.freeChunks, st.staticNodes, st.anchoredBonds, st.unanchoredIslands);

    const std::uint32_t edges[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                                   1024, 4096, 16384, 65536, 0xFFFFFFFFu};
    std::uint64_t nodeTotal = 0;
    for (auto v : st.nodesPerIsland) nodeTotal += v;
    std::printf("  nodes/island     islands       nodes   %% nodes\n");
    for (std::size_t e = 0; e < sizeof(edges) / sizeof(edges[0]); ++e)
    {
        const std::uint32_t hi = edges[e];
        const std::uint32_t lo = e == 0 ? 0 : edges[e - 1];
        std::uint32_t count = 0; std::uint64_t sum = 0;
        for (auto v : st.nodesPerIsland) if (v > lo && v <= hi) { ++count; sum += v; }
        if (!count) continue;
        std::printf("  %6u..%-8u %8u %11llu   %6.2f%%\n", lo + 1, hi, count,
                    static_cast<unsigned long long>(sum),
                    100.0 * static_cast<double>(sum) / static_cast<double>(nodeTotal ? nodeTotal : 1));
    }
    if (!st.bondsPerIsland.empty())
    {
        std::uint64_t running = 0;
        const std::size_t marks[] = {1, 4, 16, 64, 256};
        for (std::size_t m = 0, i = 0; m < 5 && i < st.bondsPerIsland.size(); ++i)
        {
            running += st.bondsPerIsland[i];
            if (i + 1 == marks[m])
            {
                std::printf("    top %4zu islands hold %6.2f%% of solved bonds\n", marks[m],
                            100.0 * static_cast<double>(running) /
                            static_cast<double>(st.solvedBonds ? st.solvedBonds : 1));
                ++m;
            }
        }
        std::printf("    largest island: %u bonds, %u nodes\n",
                    st.bondsPerIsland[0], st.largestIslandNodes);
    }
}

// ------------------------------------------------------------ scene builders

Scene buildScene(const ScenePack& pack, std::uint32_t grid, float limitScale)
{
    Scene scene;
    float minX = 1e30f, maxX = -1e30f, minZ = 1e30f, maxZ = -1e30f;
    for (const auto& n : pack.nodes)
    {
        minX = std::min(minX, n.centroid.x); maxX = std::max(maxX, n.centroid.x);
        minZ = std::min(minZ, n.centroid.z); maxZ = std::max(maxZ, n.centroid.z);
    }
    const float spanX = (maxX - minX) * 1.05f + 8.0f;
    const float spanZ = (maxZ - minZ) * 1.05f + 8.0f;

    for (const auto& m : pack.materials)
    {
        ExtStressGpuMaterial gm{};
        gm.compressionElasticLimit = m.limits.compressionElastic * limitScale;
        gm.compressionFatalLimit   = m.limits.compressionFatal   * limitScale;
        gm.tensionElasticLimit     = m.limits.tensionElastic     * limitScale;
        gm.tensionFatalLimit       = m.limits.tensionFatal       * limitScale;
        gm.shearElasticLimit       = m.limits.shearElastic       * limitScale;
        gm.shearFatalLimit         = m.limits.shearFatal         * limitScale;
        scene.materials.push_back(gm);
    }
    if (scene.materials.empty()) scene.materials.resize(1);

    const std::uint32_t copies = std::max(1u, grid * grid);
    scene.nodes.reserve(pack.nodes.size() * copies);
    scene.bonds.reserve(pack.bonds.size() * copies);
    for (std::uint32_t c = 0; c < copies; ++c)
    {
        const float dx = spanX * static_cast<float>(c % std::max(1u, grid));
        const float dz = spanZ * static_cast<float>(c / std::max(1u, grid));
        const std::uint32_t base = static_cast<std::uint32_t>(scene.nodes.size());
        for (const auto& n : pack.nodes)
        {
            ExtStressGpuNode gn{};
            gn.position[0] = n.centroid.x + dx;
            gn.position[1] = n.centroid.y;
            gn.position[2] = n.centroid.z + dz;
            gn.mass = n.mass;
            // Sphere approximation, identical to NvBlastExtStressSolver.cpp.
            // A different inertia here is a different operator.
            gn.inertia = (n.mass > 0.0f && n.volume > 0.0f)
                ? n.mass * 0.4f * std::pow(n.volume * 3.0f / (4.0f * 3.14159265358979f), 2.0f / 3.0f)
                : 0.0f;
            scene.nodes.push_back(gn);
        }
        for (const auto& b : pack.bonds)
        {
            ExtStressGpuBond gb{};
            gb.node0 = base + b.node0; gb.node1 = base + b.node1;
            gb.centroid[0] = b.centroid.x + dx;
            gb.centroid[1] = b.centroid.y;
            gb.centroid[2] = b.centroid.z + dz;
            gb.normal[0] = b.normal.x; gb.normal[1] = b.normal.y; gb.normal[2] = b.normal.z;
            gb.area = b.area; gb.health = b.area;
            gb.material = b.material < scene.materials.size() ? b.material : 0u;
            scene.bonds.push_back(gb);
        }
    }
    return scene;
}

/// Keep only the largest connected structure: "one really large building".
/// Worst case for load balance -- a single island that no amount of
/// parallelism across islands can help with.
void keepLargestComponent(Scene& scene)
{
    const IslandStats st = computeIslandStats(scene);
    if (st.nodesPerIsland.empty()) return;

    UnionFind uf(scene.nodes.size());
    for (const auto& b : scene.bonds)
        if (scene.nodes[b.node0].mass > 0.0f && scene.nodes[b.node1].mass > 0.0f)
            uf.unite(b.node0, b.node1);

    std::map<std::uint32_t, std::uint32_t> count;
    for (std::size_t i = 0; i < scene.nodes.size(); ++i)
        if (scene.nodes[i].mass > 0.0f) ++count[uf.find(static_cast<std::uint32_t>(i))];
    std::uint32_t best = 0, bestRoot = 0;
    for (const auto& kv : count) if (kv.second > best) { best = kv.second; bestRoot = kv.first; }

    std::vector<ExtStressGpuBond> kept;
    for (const auto& b : scene.bonds)
    {
        const bool d0 = scene.nodes[b.node0].mass > 0.0f;
        const bool d1 = scene.nodes[b.node1].mass > 0.0f;
        const std::uint32_t root = d0 ? uf.find(b.node0) : (d1 ? uf.find(b.node1) : 0xFFFFFFFFu);
        if (root == bestRoot) kept.push_back(b);
    }
    scene.bonds.swap(kept);
}

/// Remove a fraction of bonds at random: the "under fire" middle of the
/// trajectory, where large and tiny islands coexist.
void breakFraction(Scene& scene, float fraction, Lcg& rng)
{
    const std::size_t target =
        static_cast<std::size_t>(static_cast<double>(scene.bonds.size()) * (1.0 - fraction));
    while (scene.bonds.size() > target && !scene.bonds.empty())
    {
        const std::uint32_t victim = rng.below(static_cast<std::uint32_t>(scene.bonds.size()));
        scene.bonds[victim] = scene.bonds.back();
        scene.bonds.pop_back();
    }
}

/// Blast-localised fracture: remove bonds inside spherical impact volumes.
///
/// This is the shape real destruction has, and it is NOT what uniform random
/// removal produces. Measured against a live /city session: at 27,990 of
/// 298,172 bonds broken (9.4%) the game reports **1,353 solver islands and
/// 7,912 free chunk bodies**. Uniform removal of 25% of bonds -- nearly three
/// times the damage -- yields only 231 islands, because scattered single-bond
/// cuts almost never disconnect anything. Concentrated damage carves pieces
/// off; scattered damage just weakens.
///
/// So the island-count regime the solver actually runs in is reachable only by
/// clustering the removals. Bonds are removed when their centroid falls inside
/// any blast sphere, which is how the game's own shot damage is applied.
void blastFracture(Scene& scene, std::uint32_t blasts, float coreRadius,
                   float shellRadius, Lcg& rng)
{
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (const auto& n : scene.nodes)
        for (int c = 0; c < 3; ++c)
        {
            lo[c] = std::min(lo[c], n.position[c]);
            hi[c] = std::max(hi[c], n.position[c]);
        }

    std::vector<std::array<float, 3>> centres;
    centres.reserve(blasts);
    for (std::uint32_t i = 0; i < blasts; ++i)
    {
        const float fx = static_cast<float>(rng.below(10001)) / 10000.0f;
        const float fy = static_cast<float>(rng.below(10001)) / 10000.0f;
        const float fz = static_cast<float>(rng.below(10001)) / 10000.0f;
        centres.push_back({lo[0] + fx * (hi[0] - lo[0]),
                           lo[1] + fy * fy * (hi[1] - lo[1]),
                           lo[2] + fz * (hi[2] - lo[2])});
    }

    const float c2 = coreRadius * coreRadius, s2 = shellRadius * shellRadius;
    auto d2 = [](const float* p, const std::array<float, 3>& c) {
        const float dx = p[0] - c[0], dy = p[1] - c[1], dz = p[2] - c[2];
        return dx * dx + dy * dy + dz * dz;
    };

    std::vector<ExtStressGpuBond> kept;
    kept.reserve(scene.bonds.size());
    for (const auto& b : scene.bonds)
    {
        bool remove = false;
        for (const auto& c : centres)
        {
            // Core: everything inside is pulverised into free chunks.
            if (d2(b.centroid, c) <= c2) { remove = true; break; }
            // Shell: sever bonds that STRADDLE the boundary. This detaches the
            // interior as a connected cluster instead of destroying it, which
            // is what actually creates islands -- a solid sphere alone produces
            // free chunks and almost no new islands.
            const float da = d2(scene.nodes[b.node0].position, c);
            const float db = d2(scene.nodes[b.node1].position, c);
            if ((da <= s2) != (db <= s2)) { remove = true; break; }
        }
        if (!remove) kept.push_back(b);
    }
    scene.bonds.swap(kept);
}

/// Shatter into fragments of at most `maxNodes` chunks, sparsely bonded.
///
/// Grown by BFS so fragments are contiguous, which is what real fracture
/// produces; a purely random cut leaves a long tail of accidental large
/// components and would not actually test the small-island regime. Bonds
/// crossing fragments are dropped, EXCEPT bonds to static nodes: some
/// fragments stay anchored to the ground and some do not, and the unanchored
/// ones are the singular-operator case that has to keep working.
void shatterToFragments(Scene& scene, std::uint32_t maxNodes, Lcg& rng)
{
    const std::size_t n = scene.nodes.size();
    std::vector<std::vector<std::uint32_t>> adj(n);
    for (std::size_t i = 0; i < scene.bonds.size(); ++i)
    {
        const auto& b = scene.bonds[i];
        if (scene.nodes[b.node0].mass > 0.0f && scene.nodes[b.node1].mass > 0.0f)
        {
            adj[b.node0].push_back(b.node1);
            adj[b.node1].push_back(b.node0);
        }
    }

    std::vector<std::uint32_t> frag(n, 0xFFFFFFFFu);
    std::uint32_t nextFrag = 0;
    std::vector<std::uint32_t> queue;
    for (std::size_t seed = 0; seed < n; ++seed)
    {
        if (scene.nodes[seed].mass <= 0.0f || frag[seed] != 0xFFFFFFFFu) continue;
        const std::uint32_t id = nextFrag++;
        const std::uint32_t cap = 1u + rng.below(maxNodes);   // 1..maxNodes
        queue.clear();
        queue.push_back(static_cast<std::uint32_t>(seed));
        frag[seed] = id;
        std::uint32_t taken = 1;
        for (std::size_t qi = 0; qi < queue.size() && taken < cap; ++qi)
        {
            for (std::uint32_t nb : adj[queue[qi]])
            {
                if (taken >= cap) break;
                if (frag[nb] != 0xFFFFFFFFu) continue;
                frag[nb] = id; queue.push_back(nb); ++taken;
            }
        }
    }

    std::vector<ExtStressGpuBond> kept;
    kept.reserve(scene.bonds.size() / 4);
    for (const auto& b : scene.bonds)
    {
        const bool d0 = scene.nodes[b.node0].mass > 0.0f;
        const bool d1 = scene.nodes[b.node1].mass > 0.0f;
        if (d0 && d1) { if (frag[b.node0] == frag[b.node1]) kept.push_back(b); }
        else
        {
            // A bond to ground: keep only some, so the scene ends up with both
            // anchored and free-floating fragments.
            if (rng.below(4u) == 0u) kept.push_back(b);
        }
    }
    scene.bonds.swap(kept);
}

// ------------------------------------------------------- CPU reference solver

/// Result of running the stock NVIDIA Blast CPU solver on the same scene.
struct CpuResult
{
    double cpuMs{0};          // process CPU time: the honest cost on a shared box
    double wallMs{0};
    int iterations{0};        // negative => did not converge (StressProcessor contract)
    bool converged{false};
    AngLin6ErrorSq errorSq{0.0f, 0.0f};
    std::vector<AngLin6> impulses;
};

/// Drive StressProcessor over the identical scene.
///
/// The two DataParams below are not defaults -- they are what
/// NvBlastExtStressSolver.cpp:170-171 hard-codes in production, and the GPU
/// backend mirrors them (its Inertia is literally {inertia>0, mass>0}). Setting
/// them differently here would compare two different operators and call the
/// difference a GPU bug.
///
/// islandAware matches the GPU too: the GPU partitions into islands always, so
/// a whole-graph CPU solve would be a different algorithm, not a reference.
CpuResult runCpu(const Scene& scene, std::uint32_t iters, std::uint32_t solves,
                 bool warmStart, float gravity)
{
    CpuResult r;

    std::vector<SolverNodeS> nodes(scene.nodes.size());
    for (std::size_t i = 0; i < scene.nodes.size(); ++i)
    {
        nodes[i].CoM = {scene.nodes[i].position[0], scene.nodes[i].position[1],
                        scene.nodes[i].position[2]};
        nodes[i].mass = scene.nodes[i].mass;
        nodes[i].inertia = scene.nodes[i].inertia;
    }
    std::vector<SolverBond> bonds(scene.bonds.size());
    for (std::size_t i = 0; i < scene.bonds.size(); ++i)
    {
        bonds[i].centroid = {scene.bonds[i].centroid[0], scene.bonds[i].centroid[1],
                             scene.bonds[i].centroid[2]};
        bonds[i].nodes[0] = scene.bonds[i].node0;
        bonds[i].nodes[1] = scene.bonds[i].node1;
        bonds[i].area = scene.bonds[i].area;
        bonds[i].material = scene.bonds[i].material;
    }

    StressProcessor cpu;
    StressProcessor::DataParams dp;
    dp.centerBonds = true;
    dp.equalizeMasses = true;
    cpu.prepare(nodes.data(), static_cast<std::uint32_t>(nodes.size()),
                bonds.data(), static_cast<std::uint32_t>(bonds.size()), dp);

    std::vector<AngLin6> velocities(scene.nodes.size());
    for (std::size_t i = 0; i < velocities.size(); ++i)
    {
        velocities[i].ang = {0.0f, 0.0f, 0.0f};
        velocities[i].lin = {0.0f, scene.nodes[i].mass > 0.0f ? gravity : 0.0f, 0.0f};
    }

    StressProcessor::SolverParams sp;
    sp.maxIter = iters;
    sp.tolerance = 0.001f;
    sp.warmStart = warmStart;
    sp.islandAware = true;
    sp.skipSettled = false;

    r.impulses.assign(scene.bonds.size(), AngLin6{});

    // One untimed solve so a warm-start comparison starts from the same place
    // the GPU's does, and so first-touch allocation is not charged to the mean.
    cpu.solve(r.impulses.data(), velocities.data(), sp, &r.errorSq);

    const double cpu0 = processCpuMs();
    const auto wall0 = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < solves; ++i)
    {
        if (!warmStart) std::fill(r.impulses.begin(), r.impulses.end(), AngLin6{});
        r.iterations = cpu.solve(r.impulses.data(), velocities.data(), sp, &r.errorSq);
    }
    r.cpuMs = (processCpuMs() - cpu0) / solves;
    r.wallMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall0).count() / solves;
    r.converged = r.iterations >= 0;
    return r;
}

/// Componentwise agreement, using the same formula as the shipped equivalence
/// test (gpu_stress_test.cpp): relative to max(1, |reference|), so a component
/// that is legitimately near zero cannot manufacture a huge relative error.
struct Agreement
{
    float maxRelative{0.0f};
    double rmsRelative{0.0};
    double cpuPeak{0.0}, gpuPeak{0.0};
    std::size_t worstBond{0};
};

Agreement compareImpulses(const std::vector<AngLin6>& cpu,
                          const std::vector<ExtStressGpuImpulse>& gpu)
{
    Agreement a;
    double squared = 0.0; std::uint64_t n = 0;
    const std::size_t count = std::min(cpu.size(), gpu.size());
    for (std::size_t i = 0; i < count; ++i)
    {
        const float ref[6] = {cpu[i].ang.x, cpu[i].ang.y, cpu[i].ang.z,
                              cpu[i].lin.x, cpu[i].lin.y, cpu[i].lin.z};
        const float cand[6] = {gpu[i].angular.x, gpu[i].angular.y, gpu[i].angular.z,
                               gpu[i].linear.x, gpu[i].linear.y, gpu[i].linear.z};
        for (int c = 0; c < 6; ++c)
        {
            a.cpuPeak = std::max(a.cpuPeak, static_cast<double>(std::fabs(ref[c])));
            a.gpuPeak = std::max(a.gpuPeak, static_cast<double>(std::fabs(cand[c])));
            const float e = std::fabs(ref[c] - cand[c]) / std::max(1.0f, std::fabs(ref[c]));
            if (e > a.maxRelative) { a.maxRelative = e; a.worstBond = i; }
            squared += static_cast<double>(e) * e; ++n;
        }
    }
    a.rmsRelative = n ? std::sqrt(squared / static_cast<double>(n)) : 0.0;
    return a;
}

/// How well a solution satisfies EQUILIBRIUM -- the metric that actually says
/// whether two solvers agree on the physics.
///
/// Componentwise impulse agreement is the wrong gate for this problem, and the
/// measurements say so: with a cold start and a fully converged solve
/// (single-building, 429 iterations) CPU and GPU still differ by 70x on
/// individual components. That is not a bug in either. The operator is rank
/// deficient -- null(B) is most of the bond unknowns -- and badly conditioned,
/// so a residual tolerance of 1e-3 bounds the residual, NOT the solution: two
/// solutions can sit far apart along near-null directions and both be correct
/// answers to the same question. Componentwise comparison measures the
/// conditioning of the problem, not the quality of the solver.
///
/// What is well posed is the residual itself. For each dynamic node, the net
/// wrench its bonds apply must balance the applied load:
///     sum_j +/-(lambda_j.ang - r_ij x lambda_j.lin,  lambda_j.lin)  =  -k * v_i
/// The solvers work in internally scaled units, so `k` is recovered by least
/// squares rather than reconstructed from their private length/mass scales --
/// which makes this measurement independent of both solvers' conventions and
/// therefore a fair referee between them.
struct Equilibrium
{
    double relLinear{0.0};    // ||net.lin + k v|| / ||k v||
    double relAngular{0.0};   // ||net.ang|| / typical wrench magnitude
    double k{0.0};
};

/// Restricted to ANCHORED islands, and that restriction is not a convenience.
/// A free-floating fragment is in freefall: no set of internal bond impulses
/// can balance gravity on it, the correct answer is lambda = 0, and its
/// "residual" is the entire applied load. Including those nodes reports ~46 for
/// a scene that is behaving perfectly, which would read as a catastrophic
/// failure and is instead a statement that most of the scene is falling.
Equilibrium equilibriumResidual(const Scene& scene,
                                const std::vector<ExtStressGpuImpulse>& lambda,
                                float gravity)
{
    const std::size_t n = scene.nodes.size();

    UnionFind uf(n);
    for (const auto& b : scene.bonds)
        if (scene.nodes[b.node0].mass > 0.0f && scene.nodes[b.node1].mass > 0.0f)
            uf.unite(b.node0, b.node1);
    std::vector<char> anchored(n, 0);
    for (const auto& b : scene.bonds)
    {
        const bool d0 = scene.nodes[b.node0].mass > 0.0f;
        const bool d1 = scene.nodes[b.node1].mass > 0.0f;
        if (d0 && !d1)      anchored[uf.find(b.node0)] = 1;
        else if (d1 && !d0) anchored[uf.find(b.node1)] = 1;
    }
    std::vector<double> netAng(3 * n, 0.0), netLin(3 * n, 0.0);
    double wrenchScale = 0.0; std::uint64_t wrenchCount = 0;

    for (std::size_t j = 0; j < scene.bonds.size(); ++j)
    {
        const auto& b = scene.bonds[j];
        const auto& im = lambda[j];
        const double la[3] = {im.angular.x, im.angular.y, im.angular.z};
        const double ll[3] = {im.linear.x, im.linear.y, im.linear.z};
        const bool d0 = scene.nodes[b.node0].mass > 0.0f;
        const bool d1 = scene.nodes[b.node1].mass > 0.0f;

        // centerBonds = true, exactly as StressProcessor::prepare builds it.
        double o0[3], o1[3];
        const auto& p0 = scene.nodes[b.node0].position;
        const auto& p1 = scene.nodes[b.node1].position;
        if (!d0)       { for (int c = 0; c < 3; ++c) { o1[c] = b.centroid[c] - p1[c]; o0[c] = -o1[c]; } }
        else if (!d1)  { for (int c = 0; c < 3; ++c) { o0[c] = b.centroid[c] - p0[c]; o1[c] = -o0[c]; } }
        else           { for (int c = 0; c < 3; ++c) { o0[c] = 0.5 * (p1[c] - p0[c]); o1[c] = -o0[c]; } }

        auto cross = [](const double* a, const double* b_, double* out) {
            out[0] = a[1] * b_[2] - a[2] * b_[1];
            out[1] = a[2] * b_[0] - a[0] * b_[2];
            out[2] = a[0] * b_[1] - a[1] * b_[0];
        };
        double c0[3], c1[3];
        cross(o0, ll, c0); cross(o1, ll, c1);
        for (int c = 0; c < 3; ++c)
        {
            netAng[3 * b.node0 + c] += la[c] - c0[c];
            netLin[3 * b.node0 + c] += ll[c];
            netAng[3 * b.node1 + c] -= la[c] - c1[c];
            netLin[3 * b.node1 + c] -= ll[c];
            wrenchScale += (la[c] - c0[c]) * (la[c] - c0[c]);
        }
        ++wrenchCount;
    }

    // Least-squares k for  net.lin = -k * v.lin,  v.lin = (0, gravity, 0).
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (scene.nodes[i].mass <= 0.0f || !anchored[uf.find(static_cast<std::uint32_t>(i))])
            continue;
        num += netLin[3 * i + 1] * gravity;
        den += static_cast<double>(gravity) * gravity;
    }
    Equilibrium e;
    e.k = den > 0.0 ? num / den : 0.0;

    double linErr = 0.0, linRef = 0.0, angErr = 0.0;
    std::uint64_t counted = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        // Static nodes: the ground reaction absorbs anything. Unanchored
        // islands: in freefall, see above.
        if (scene.nodes[i].mass <= 0.0f || !anchored[uf.find(static_cast<std::uint32_t>(i))])
            continue;
        ++counted;
        const double want[3] = {0.0, e.k * gravity, 0.0};
        for (int c = 0; c < 3; ++c)
        {
            const double d = netLin[3 * i + c] - want[c];
            linErr += d * d; linRef += want[c] * want[c];
            angErr += netAng[3 * i + c] * netAng[3 * i + c];
        }
    }
    e.relLinear = linRef > 0.0 ? std::sqrt(linErr / linRef) : 0.0;
    const double wr = wrenchCount ? std::sqrt(wrenchScale / static_cast<double>(wrenchCount)) : 1.0;
    e.relAngular = (wr > 0.0 && counted)
        ? std::sqrt(angErr / static_cast<double>(counted)) / wr : 0.0;
    return e;
}

/// Progressive destruction: the path the game actually runs.
///
/// Every other scenario here builds a topology and holds it still. That misses
/// the cost the game pays most often, because `applyTopologyChange` fires on
/// every tick on which ANY bond broke: it rebuilds the island partition and the
/// node-bond CSR, re-uploads the whole topology, and wipes every island's
/// converged flag -- which disables the settled-skip scene-wide until the next
/// quiet tick. None of that appears in a static scenario.
///
/// So: solve with damage on, read back what broke, remove it, repeat, and split
/// the per-tick cost by whether anything broke that tick.
struct DestroyStats
{
    std::uint32_t ticks{0}, fractureTicks{0}, bondsRemoved{0};
    double devFracture{0}, devQuiet{0};
    double planFracture{0}, planQuiet{0};
    double syncFracture{0}, syncQuiet{0};
    double finishFracture{0}, finishQuiet{0};
    double wallFracture{0}, wallQuiet{0};
    double worstTickWall{0};
    std::size_t finalBonds{0}, finalNonFinite{0};
    double finalResidual{-1.0};
    bool ok{true};
    std::string note;
};

/// Stamp the CPU processor's compliance weights (column scale) onto the GPU
/// bonds, exactly as ConjugateGradientImpulseSolver::initialize does in
/// production (NvBlastExtStressSolver.cpp, `gpu.colScale = getColumnScale(i)`).
///
/// Without this the GPU solves the UNWEIGHTED system while the CPU reference
/// solves the Young's-modulus-weighted one. Both satisfy equilibrium, so the
/// residual gate cannot tell them apart -- it passed with single-building at
/// peak |J| cpu 1.68e6 / gpu 1.24e6 -- but the load DISTRIBUTION between
/// parallel paths, and therefore which bonds break, differs. The comparison
/// must feed both backends the same operator.
void stampColumnScale(Scene& scene)
{
    if (scene.bonds.empty()) return;
    std::vector<SolverNodeS> nodes(scene.nodes.size());
    for (std::size_t i = 0; i < scene.nodes.size(); ++i)
    {
        nodes[i].CoM = {scene.nodes[i].position[0], scene.nodes[i].position[1],
                        scene.nodes[i].position[2]};
        nodes[i].mass = scene.nodes[i].mass;
        nodes[i].inertia = scene.nodes[i].inertia;
    }
    std::vector<SolverBond> bonds(scene.bonds.size());
    for (std::size_t i = 0; i < scene.bonds.size(); ++i)
    {
        bonds[i].centroid = {scene.bonds[i].centroid[0], scene.bonds[i].centroid[1],
                             scene.bonds[i].centroid[2]};
        bonds[i].nodes[0] = scene.bonds[i].node0;
        bonds[i].nodes[1] = scene.bonds[i].node1;
        bonds[i].area = scene.bonds[i].area;
        bonds[i].material = scene.bonds[i].material;
    }
    StressProcessor cpu;
    StressProcessor::DataParams dp;
    dp.centerBonds = true;
    dp.equalizeMasses = true;
    cpu.prepare(nodes.data(), static_cast<std::uint32_t>(nodes.size()),
                bonds.data(), static_cast<std::uint32_t>(bonds.size()), dp);
    for (std::size_t i = 0; i < scene.bonds.size(); ++i)
    {
        scene.bonds[i].colScale = cpu.getColumnScale(static_cast<std::uint32_t>(i));
    }
}

DestroyStats runDestroy(const Scene& sceneIn, std::uint32_t iters, std::uint32_t ticks,
                        float gravity, bool skip)
{
    DestroyStats st;
    Scene scene = sceneIn;
    stampColumnScale(scene);

    ExtStressGpuSolver* solver = ExtStressGpuSolver::create(
        scene.nodes.data(), static_cast<std::uint32_t>(scene.nodes.size()),
        scene.bonds.data(), static_cast<std::uint32_t>(scene.bonds.size()),
        scene.materials.data(), static_cast<std::uint32_t>(scene.materials.size()));
    if (!solver) { st.ok = false; st.note = "solver create FAILED"; return st; }

    std::vector<ExtStressGpuImpulse> velocities(scene.nodes.size());
    for (std::size_t i = 0; i < velocities.size(); ++i)
    {
        velocities[i] = {};
        if (scene.nodes[i].mass > 0.0f) velocities[i].linear.y = gravity;
    }

    ExtStressGpuSolveParams params{};
    params.maxIterations = iters;
    params.tolerance = 0.001f;
    params.warmStart = true;
    params.skipSettledIslands = skip;
    params.skipStableUnconverged = false;
    params.applyDamage = true;

    // Mirror of the solver's bond array. removeBond is swap-with-last, so
    // applying the identical permutation here keeps this describing exactly the
    // topology the solver holds -- which is what lets the residual below be a
    // real check rather than a check against a graph we imagined.
    std::vector<std::uint32_t> broken(scene.bonds.size());
    for (std::uint32_t t = 0; t < ticks; ++t)
    {
        const auto tick0 = std::chrono::steady_clock::now();
        if (!solver->solve(velocities.data(), params))
        { st.ok = false; st.note = "solve FAILED"; break; }

        std::uint32_t count = 0;
        if (!solver->readbackBrokenBonds(broken.data(),
                                         static_cast<std::uint32_t>(broken.size()), count))
        { st.ok = false; st.note = "readbackBrokenBonds FAILED"; break; }

        // Descending, because removeBond is swap-with-last: taking the highest
        // index first means the element swapped in is always beyond the indices
        // still to be removed. Ascending silently removes the wrong bonds.
        if (count > 1) std::sort(broken.begin(), broken.begin() + count,
                                 std::greater<std::uint32_t>());
        for (std::uint32_t i = 0; i < count; ++i)
        {
            if (broken[i] >= scene.bonds.size())
            { st.ok = false; st.note = "broken index out of range"; break; }
            if (!solver->removeBond(broken[i])) { st.ok = false; st.note = "removeBond FAILED"; break; }
            scene.bonds[broken[i]] = scene.bonds.back();
            scene.bonds.pop_back();
        }
        const double wall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tick0).count();

        const ExtStressGpuTelemetry& tm = solver->telemetry();
        ++st.ticks;
        st.worstTickWall = std::max(st.worstTickWall, wall);
        if (count)
        {
            ++st.fractureTicks; st.bondsRemoved += count;
            st.devFracture += tm.solveMilliseconds; st.planFracture += tm.hostPlanMilliseconds;
            st.syncFracture += tm.hostSyncMilliseconds; st.finishFracture += tm.hostFinishMilliseconds;
            st.wallFracture += wall;
        }
        else
        {
            st.devQuiet += tm.solveMilliseconds; st.planQuiet += tm.hostPlanMilliseconds;
            st.syncQuiet += tm.hostSyncMilliseconds; st.finishQuiet += tm.hostFinishMilliseconds;
            st.wallQuiet += wall;
        }
    }
    // Validate the END STATE, not just the cost. This is the path where the
    // island partition is carried across ticks and converged flags survive, so
    // an error here would show up as a physically wrong answer after a long
    // demolition and nowhere else.
    if (st.ok && !scene.bonds.empty())
    {
        std::vector<ExtStressGpuImpulse> impulses(scene.bonds.size());
        if (solver->readbackImpulses(impulses.data(),
                                     static_cast<std::uint32_t>(impulses.size())))
        {
            std::size_t nonFinite = 0;
            for (const auto& im : impulses)
            {
                const float v[6] = {im.angular.x, im.angular.y, im.angular.z,
                                    im.linear.x, im.linear.y, im.linear.z};
                for (float f : v) if (!std::isfinite(f)) { ++nonFinite; break; }
            }
            const Equilibrium eq = equilibriumResidual(scene, impulses, gravity);
            st.finalBonds = scene.bonds.size();
            st.finalResidual = eq.relLinear;
            st.finalNonFinite = nonFinite;
            if (nonFinite) { st.ok = false; st.note = "NON-FINITE impulses after destruction"; }
        }
    }

    solver->release();
    return st;
}

/// Verify the exact-Galerkin property the AMG design rests on.
///
/// The claim: with the RIGID-BODY LIFT as prolongator -- aggregate `a` at
/// position x_a carries (omega_a, v_a), and member i receives
/// omega_i = omega_a, v_i = v_a + (x_i - x_a) x omega_a -- we have
///
///     C^T R = C_c^T     EXACTLY,
///
/// where C_c is a Coupling matrix on the aggregated graph with
/// offset = (bond centroid - aggregate position). If that holds, then
/// R^T L R = L_c is exactly Galerkin, the coarse operator is just another bond
/// graph, and EVERY EXISTING KERNEL RUNS UNCHANGED AT EVERY LEVEL -- no
/// block-CSR, no new SpMV. That is what makes multigrid cheap to build here
/// rather than a rewrite, so it is worth proving numerically before anyone
/// spends a session on the hierarchy.
///
/// Also checks the second half of the claim: a bond INTERNAL to an aggregate
/// must map to exactly zero, which is what makes the coarse graph smaller.
struct GalerkinReport
{
    std::size_t coarseNodes{0}, crossingBonds{0}, internalBonds{0};
    double maxCrossErr{0.0}, maxInternalMag{0.0}, refMag{0.0};
    bool ok{false};
};

GalerkinReport verifyGalerkin(const Scene& scene, Lcg& rng, Scene* coarseOut, bool mergeParallel)
{
    GalerkinReport rep;
    const std::size_t n = scene.nodes.size();

    // Greedy pairwise aggregation over dynamic nodes. Static nodes are never
    // aggregated: mixing them would break P R = R P_c, since a static node is
    // annihilated on the fine level but its aggregate would not be.
    std::vector<std::vector<std::uint32_t>> adj(n);
    for (const auto& b : scene.bonds)
    {
        if (scene.nodes[b.node0].mass > 0.0f && scene.nodes[b.node1].mass > 0.0f)
        { adj[b.node0].push_back(b.node1); adj[b.node1].push_back(b.node0); }
    }
    std::vector<std::uint32_t> agg(n, 0xFFFFFFFFu);
    std::uint32_t nextAgg = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (scene.nodes[i].mass <= 0.0f || agg[i] != 0xFFFFFFFFu) continue;
        const std::uint32_t id = nextAgg++;
        agg[i] = id;
        for (std::uint32_t nb : adj[i])
        {
            if (agg[nb] == 0xFFFFFFFFu && scene.nodes[nb].mass > 0.0f) { agg[nb] = id; break; }
        }
    }
    rep.coarseNodes = nextAgg;

    // Aggregate positions = centroid of members.
    std::vector<double> ax(nextAgg, 0.0), ay(nextAgg, 0.0), az(nextAgg, 0.0);
    std::vector<std::uint32_t> cnt(nextAgg, 0);
    for (std::size_t i = 0; i < n; ++i)
    {
        if (agg[i] == 0xFFFFFFFFu) continue;
        ax[agg[i]] += scene.nodes[i].position[0];
        ay[agg[i]] += scene.nodes[i].position[1];
        az[agg[i]] += scene.nodes[i].position[2];
        ++cnt[agg[i]];
    }
    for (std::uint32_t a = 0; a < nextAgg; ++a)
    { if (cnt[a]) { ax[a] /= cnt[a]; ay[a] /= cnt[a]; az[a] /= cnt[a]; } }

    // A random coarse 6-DOF field.
    auto rnd = [&]() { return (double)(rng.next() % 20000) / 10000.0 - 1.0; };
    std::vector<double> cOmega(3 * nextAgg), cVel(3 * nextAgg);
    for (std::uint32_t a = 0; a < nextAgg; ++a)
    { for (int c = 0; c < 3; ++c) { cOmega[3*a+c] = rnd(); cVel[3*a+c] = rnd(); } }

    // Lift to the fine level: the rigid-body prolongator.
    std::vector<double> fOmega(3 * n, 0.0), fVel(3 * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        const std::uint32_t a = agg[i];
        if (a == 0xFFFFFFFFu) continue;   // static: annihilated, stays zero
        const double d[3] = {scene.nodes[i].position[0] - ax[a],
                             scene.nodes[i].position[1] - ay[a],
                             scene.nodes[i].position[2] - az[a]};
        const double* w = &cOmega[3*a];
        for (int c = 0; c < 3; ++c) fOmega[3*i+c] = w[c];
        // v_i = v_a + (x_i - x_a) x omega_a
        fVel[3*i+0] = cVel[3*a+0] + (d[1]*w[2] - d[2]*w[1]);
        fVel[3*i+1] = cVel[3*a+1] + (d[2]*w[0] - d[0]*w[2]);
        fVel[3*i+2] = cVel[3*a+2] + (d[0]*w[1] - d[1]*w[0]);
    }

    auto cross = [](const double* a, const double* b, double* o) {
        o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0]; };

    // C^T applied to the lifted field, per fine bond, vs C_c^T applied to the
    // coarse field with offset = (bond centroid - aggregate position).
    for (const auto& b : scene.bonds)
    {
        const std::uint32_t a0 = agg[b.node0], a1 = agg[b.node1];
        if (a0 == 0xFFFFFFFFu || a1 == 0xFFFFFFFFu) continue;   // anchored: separate case

        const double o0[3] = {b.centroid[0] - scene.nodes[b.node0].position[0],
                              b.centroid[1] - scene.nodes[b.node0].position[1],
                              b.centroid[2] - scene.nodes[b.node0].position[2]};
        const double o1[3] = {b.centroid[0] - scene.nodes[b.node1].position[0],
                              b.centroid[1] - scene.nodes[b.node1].position[1],
                              b.centroid[2] - scene.nodes[b.node1].position[2]};
        double c0[3], c1[3];
        cross(o0, &fOmega[3*b.node0], c0);
        cross(o1, &fOmega[3*b.node1], c1);
        double fineAng[3], fineLin[3];
        for (int c = 0; c < 3; ++c)
        {
            fineAng[c] = fOmega[3*b.node0+c] - fOmega[3*b.node1+c];
            fineLin[c] = fVel[3*b.node0+c] - fVel[3*b.node1+c] + c0[c] - c1[c];
        }

        if (a0 == a1)
        {
            // Internal to an aggregate: must vanish, which is what shrinks the
            // coarse graph.
            ++rep.internalBonds;
            for (int c = 0; c < 3; ++c)
            {
                rep.maxInternalMag = std::max(rep.maxInternalMag, std::fabs(fineAng[c]));
                rep.maxInternalMag = std::max(rep.maxInternalMag, std::fabs(fineLin[c]));
            }
            continue;
        }

        ++rep.crossingBonds;
        const double q0[3] = {b.centroid[0] - ax[a0], b.centroid[1] - ay[a0], b.centroid[2] - az[a0]};
        const double q1[3] = {b.centroid[0] - ax[a1], b.centroid[1] - ay[a1], b.centroid[2] - az[a1]};
        double d0[3], d1[3];
        cross(q0, &cOmega[3*a0], d0);
        cross(q1, &cOmega[3*a1], d1);
        for (int c = 0; c < 3; ++c)
        {
            const double coarseAng = cOmega[3*a0+c] - cOmega[3*a1+c];
            const double coarseLin = cVel[3*a0+c] - cVel[3*a1+c] + d0[c] - d1[c];
            rep.maxCrossErr = std::max(rep.maxCrossErr, std::fabs(fineAng[c] - coarseAng));
            rep.maxCrossErr = std::max(rep.maxCrossErr, std::fabs(fineLin[c] - coarseLin));
            rep.refMag = std::max(rep.refMag, std::fabs(coarseLin));
        }
    }
    // RELATIVE, not absolute: node positions and bond centroids are float32,
    // so the identity can only hold to float epsilon (~1.2e-7) times the
    // magnitude of the quantities involved, however exact the algebra is.
    const double scale = std::max(rep.refMag, 1.0);
    rep.ok = rep.maxCrossErr / scale < 1e-6 && rep.maxInternalMag / scale < 1e-6;

    // Emit the coarse graph so the next level can be built from it. This is
    // what a V-cycle actually needs: the identity has to survive RECURSIVE
    // coarsening, not just one level, and the coarse graph must be a valid
    // Coupling graph in its own right or the recursion cannot even be typed.
    if (coarseOut != nullptr)
    {
        coarseOut->materials = scene.materials;
        coarseOut->nodes.assign(nextAgg, ExtStressGpuNode{});
        for (std::uint32_t a = 0; a < nextAgg; ++a)
        {
            coarseOut->nodes[a].position[0] = static_cast<float>(ax[a]);
            coarseOut->nodes[a].position[1] = static_cast<float>(ay[a]);
            coarseOut->nodes[a].position[2] = static_cast<float>(az[a]);
            coarseOut->nodes[a].mass = 1.0f;      // aggregates of dynamic nodes
            coarseOut->nodes[a].inertia = 1.0f;
        }
        // One static node carrying every anchored bond, mirroring how the fine
        // level treats the ground: static nodes are boundaries, never aggregated.
        const std::uint32_t coarseGround = static_cast<std::uint32_t>(coarseOut->nodes.size());
        coarseOut->nodes.push_back(ExtStressGpuNode{});   // mass 0 => static
        for (const auto& b : scene.bonds)
        {
            const std::uint32_t a0 = agg[b.node0], a1 = agg[b.node1];
            ExtStressGpuBond cb{};
            cb.centroid[0] = b.centroid[0];
            cb.centroid[1] = b.centroid[1];
            cb.centroid[2] = b.centroid[2];
            cb.normal[0] = b.normal[0]; cb.normal[1] = b.normal[1]; cb.normal[2] = b.normal[2];
            cb.area = b.area; cb.health = b.health; cb.material = b.material;
            if (a0 != 0xFFFFFFFFu && a1 != 0xFFFFFFFFu)
            {
                if (a0 == a1) continue;           // internal: vanishes, verified above
                cb.node0 = a0; cb.node1 = a1;
            }
            else if (a0 != 0xFFFFFFFFu) { cb.node0 = a0; cb.node1 = coarseGround; }
            else if (a1 != 0xFFFFFFFFu) { cb.node0 = coarseGround; cb.node1 = a1; }
            else continue;
            coarseOut->bonds.push_back(cb);
        }

        if (mergeParallel)
        {
            // Collapse parallel coarse bonds (same aggregate pair) into one,
            // area-weighted. This BREAKS the exact-Galerkin identity, because
            // the merged bond has a single centroid where the originals had
            // several -- but it is the only way to make BONDS coarsen, and a
            // non-exact coarse operator is still a legitimate preconditioner:
            // it costs iterations, never correctness. The question this
            // measures is whether the coarsening it buys outweighs that.
            std::map<std::uint64_t, std::size_t> seen;
            std::vector<ExtStressGpuBond> merged;
            std::vector<float> weight;
            for (const auto& cb : coarseOut->bonds)
            {
                const std::uint32_t lo = std::min(cb.node0, cb.node1);
                const std::uint32_t hi = std::max(cb.node0, cb.node1);
                const std::uint64_t key = (static_cast<std::uint64_t>(lo) << 32) | hi;
                auto it = seen.find(key);
                if (it == seen.end())
                {
                    seen.emplace(key, merged.size());
                    merged.push_back(cb);
                    weight.push_back(cb.area);
                    for (int c = 0; c < 3; ++c) merged.back().centroid[c] = cb.centroid[c] * cb.area;
                }
                else
                {
                    ExtStressGpuBond& m = merged[it->second];
                    for (int c = 0; c < 3; ++c) m.centroid[c] += cb.centroid[c] * cb.area;
                    m.area += cb.area;
                    m.health += cb.health;
                    weight[it->second] += cb.area;
                }
            }
            for (std::size_t i = 0; i < merged.size(); ++i)
            {
                const float w = weight[i] > 0.0f ? weight[i] : 1.0f;
                for (int c = 0; c < 3; ++c) merged[i].centroid[c] /= w;
            }
            coarseOut->bonds.swap(merged);
        }
    }
    return rep;
}

// -------------------------------------------------------------------- runner

struct RunResult
{
    std::string name;
    std::size_t nodes{0}, bonds{0};
    std::uint32_t islands{0};
    double deviceMs{0}, wallMs{0}, perIterMs{0};
    /// The iteration at which every island had converged, as reported by the
    /// solver. NOT the number of iterations executed: there is no device-side
    /// early exit, so the kernel sequence always runs maxIterations times. In
    /// the shattered regime convergedAt is 1 and 32 run anyway -- which is the
    /// single largest waste in that regime and is invisible if you divide
    /// device time by convergedAt.
    double convergedAt{0};
    std::uint32_t executedIters{0};
    // CPU reference, filled only with --compare.
    bool compared{false};
    double cpuMs{0}, cpuWallMs{0};
    int cpuIterations{0};
    float maxRelative{0.0f};
    double rmsRelative{0.0};
    double cpuPeak{0.0}, gpuPeak{0.0};
    double eqCpuLin{0.0}, eqGpuLin{0.0}, eqCpuAng{0.0}, eqGpuAng{0.0};
    bool ok{false};
    std::string note;
};

/// One scenario: fresh solver, warmup, timed solves, correctness checks.
///
/// The checks matter as much as the timing. A solver that is fast because it
/// silently skipped an island, or that emits a NaN on an unanchored fragment,
/// is not fast -- and both of those are failure modes this suite exists to
/// provoke.
RunResult runScenario(const std::string& name, const Scene& scene,
                      std::uint32_t iters, std::uint32_t solves,
                      bool skip, bool damage, float gravity, bool histogram,
                      bool compare, std::uint32_t cpuSolves)
{
    RunResult r; r.name = name;
    r.nodes = scene.nodes.size(); r.bonds = scene.bonds.size();

    const IslandStats st = computeIslandStats(scene);
    if (histogram) printIslandStats(st, name.c_str());

    if (scene.bonds.empty())
    {
        // The far extreme: every bond gone. There is nothing to solve, and the
        // solver must say so rather than misbehave.
        r.ok = true; r.note = "no bonds (all free chunks) - nothing to solve";
        return r;
    }

    // Same compliance weights the CPU reference below will use.
    Scene stamped = scene;
    stampColumnScale(stamped);
    ExtStressGpuSolver* solver = ExtStressGpuSolver::create(
        stamped.nodes.data(), static_cast<std::uint32_t>(stamped.nodes.size()),
        stamped.bonds.data(), static_cast<std::uint32_t>(stamped.bonds.size()),
        stamped.materials.data(), static_cast<std::uint32_t>(stamped.materials.size()));
    if (!solver) { r.note = "solver create FAILED"; return r; }

    std::vector<ExtStressGpuImpulse> velocities(scene.nodes.size());
    for (std::size_t i = 0; i < velocities.size(); ++i)
    {
        velocities[i] = {};
        // Production feeds gravity as an ACCELERATION: it lands in the node
        // velocity regardless of mass, and the angular half is never written.
        if (scene.nodes[i].mass > 0.0f) velocities[i].linear.y = gravity;
    }

    ExtStressGpuSolveParams params{};
    params.maxIterations = iters;
    params.tolerance = 0.001f;
    // Cold start when comparing. The operator is rank deficient -- null(B) is
    // most of the bond unknowns -- so CG only pins the component of lambda in
    // range(B^T) and carries whatever the starting guess had in the null space.
    // From zero both solvers converge to the SAME minimum-norm solution and
    // must agree; from a warm start they may legitimately sit at different
    // points of the same solution manifold, and calling that a mismatch would
    // be wrong.
    params.warmStart = !compare;
    params.skipSettledIslands = skip;
    params.skipStableUnconverged = false;   // the depth-truncation defect
    params.applyDamage = damage;

    for (std::uint32_t i = 0; i < 5; ++i)
    {
        if (!solver->solve(velocities.data(), params))
        { r.note = "warmup solve FAILED"; solver->release(); return r; }
    }

    double solveMs = 0; std::uint64_t iterTotal = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    for (std::uint32_t i = 0; i < solves; ++i)
    {
        if (!solver->solve(velocities.data(), params))
        { r.note = "timed solve FAILED"; solver->release(); return r; }
        const ExtStressGpuTelemetry& t = solver->telemetry();
        solveMs += t.solveMilliseconds; iterTotal += t.iterations;
        r.islands = t.islandCount;
    }
    r.wallMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - wall0).count() / solves;
    r.deviceMs = solveMs / solves;
    r.convergedAt = static_cast<double>(iterTotal) / solves;
    r.executedIters = iters;
    r.perIterMs = r.deviceMs / static_cast<double>(iters);

    // Correctness: every impulse finite, and the solve did real work.
    std::vector<ExtStressGpuImpulse> impulses(scene.bonds.size());
    if (!solver->readbackImpulses(impulses.data(), static_cast<std::uint32_t>(impulses.size())))
    { r.note = "impulse readback FAILED"; solver->release(); return r; }

    std::size_t nonFinite = 0; double peak = 0.0;
    for (const auto& im : impulses)
    {
        const float v[6] = {im.angular.x, im.angular.y, im.angular.z,
                            im.linear.x, im.linear.y, im.linear.z};
        for (float f : v)
        {
            if (!std::isfinite(f)) { ++nonFinite; break; }
            peak = std::max(peak, static_cast<double>(std::fabs(f)));
        }
    }
    {
        // Always report how far the solve is from equilibrium, not only under
        // --compare. This is what makes "does a restarted CG converge?"
        // answerable from the warm-start path, which --compare cannot do
        // because it deliberately starts cold.
        const Equilibrium eq = equilibriumResidual(scene, impulses, gravity);
        r.eqGpuLin = eq.relLinear;
        r.eqGpuAng = eq.relAngular;
    }
    if (nonFinite) { char b[96]; std::snprintf(b, sizeof(b), "%zu NON-FINITE impulses", nonFinite); r.note = b; }
    else if (peak <= 0.0) r.note = "all impulses zero - solver did nothing";
    else { r.ok = true; char b[64]; std::snprintf(b, sizeof(b), "peak |J| %.3g", peak); r.note = b; }

    if (compare && r.ok)
    {
        const CpuResult cpu = runCpu(scene, iters, cpuSolves, /*warmStart*/ false, gravity);
        const Agreement ag = compareImpulses(cpu.impulses, impulses);

        std::vector<ExtStressGpuImpulse> cpuAsGpu(cpu.impulses.size());
        for (std::size_t i = 0; i < cpu.impulses.size(); ++i)
        {
            cpuAsGpu[i].angular = {cpu.impulses[i].ang.x, cpu.impulses[i].ang.y, cpu.impulses[i].ang.z};
            cpuAsGpu[i].linear  = {cpu.impulses[i].lin.x, cpu.impulses[i].lin.y, cpu.impulses[i].lin.z};
        }
        const Equilibrium eqCpu = equilibriumResidual(scene, cpuAsGpu, gravity);
        const Equilibrium eqGpu = equilibriumResidual(scene, impulses, gravity);
        r.eqCpuLin = eqCpu.relLinear; r.eqGpuLin = eqGpu.relLinear;
        r.eqCpuAng = eqCpu.relAngular; r.eqGpuAng = eqGpu.relAngular;
        r.compared = true;
        r.cpuMs = cpu.cpuMs; r.cpuWallMs = cpu.wallMs;
        r.cpuIterations = cpu.iterations;
        r.maxRelative = ag.maxRelative; r.rmsRelative = ag.rmsRelative;
        r.cpuPeak = ag.cpuPeak; r.gpuPeak = ag.gpuPeak;
        char b[160];
        std::snprintf(b, sizeof(b), "peak |J| cpu %.3g / gpu %.3g", ag.cpuPeak, ag.gpuPeak);
        r.note = b;
    }

    solver->release();
    return r;
}

} // namespace

int main(int argc, char** argv)
{
    std::string packPath =
        "../../../blast/blast-stress-solver/assets/mini-city/fractured-downtown.json";
    std::uint32_t grid = 2, iters = 32, solves = 20;
    float limitScale = 0.45f;
    bool histogram = false, quick = false, compare = false, destroy = false;
    std::uint32_t ticks = 120;
    bool galerkin = false;
    std::uint32_t cpuSolves = 3;
    std::string only;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : ""; };
        if      (a == "--pack")        packPath = next();
        else if (a == "--grid")        grid = static_cast<std::uint32_t>(std::atoi(next()));
        else if (a == "--iters")       iters = static_cast<std::uint32_t>(std::atoi(next()));
        else if (a == "--solves")      solves = static_cast<std::uint32_t>(std::atoi(next()));
        else if (a == "--limit-scale") limitScale = static_cast<float>(std::atof(next()));
        else if (a == "--only")        only = next();
        else if (a == "--histogram")   histogram = true;
        else if (a == "--quick")       quick = true;
        else if (a == "--compare")     compare = true;
        else if (a == "--destroy")     destroy = true;
        else if (a == "--galerkin")    galerkin = true;
        else if (a == "--ticks")       ticks = static_cast<std::uint32_t>(std::atoi(next()));
        else if (a == "--cpu-solves")  cpuSolves = static_cast<std::uint32_t>(std::atoi(next()));
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    ScenePack pack;
    try { pack = blast_demo::loadScenePack(packPath); }
    catch (const std::exception& e)
    { std::fprintf(stderr, "FAIL: pack '%s': %s\n", packPath.c_str(), e.what()); return 1; }

    std::printf("pack '%s': %zu nodes, %zu bonds | grid %u | iters %u | solves %u\n",
                pack.title.c_str(), pack.nodes.size(), pack.bonds.size(), grid, iters, solves);

    struct Case { const char* name; const char* what; };
    const std::vector<Case> cases = {
        {"single-building",  "largest connected structure only"},
        {"city",             "all buildings, intact"},
        {"broken-25",        "25% of bonds gone"},
        {"broken-50",        "50% of bonds gone"},
        {"broken-75",        "75% of bonds gone"},
        {"broken-90",        "90% of bonds gone"},
        {"broken-99",        "99% of bonds gone"},
        {"live-light",       "blast-localised, ~4% bonds gone (early bombardment)"},
        {"live-city",         "blast-localised; matches a real /city session: ~1.35k islands, ~28k bonds gone"},
        {"live-heavy",        "blast-localised, ~2x the live session's damage"},
        {"shatter-4",        "fragments of <=4 chunks, sparsely bonded"},
        {"shatter-3",        "fragments of <=3 chunks"},
        {"shatter-2",        "fragments of <=2 chunks (pairs)"},
        {"dust",             "every bond broken; every chunk free"},
    };

    if (galerkin)
    {
        Scene scene = buildScene(pack, grid, limitScale);
        Lcg rng(0xA11CEu);
        std::printf("\n=== exact-Galerkin check, RECURSIVE (what a V-cycle needs) ===\n");
        std::printf("%-6s %10s %10s %12s %14s %16s %s\n", "level", "nodes", "bonds",
                    "coarsening", "internal drop", "rel error", "verdict");
        bool allLevelsOk = true;
        Scene cur = scene;
        std::vector<std::size_t> exactBonds;
        for (int level = 0; level < 8; ++level)
        {
            Scene next;
            const GalerkinReport r = verifyGalerkin(cur, rng, &next, /*merge*/ false);
            exactBonds.push_back(cur.bonds.size());
            if (r.coarseNodes == 0 || r.crossingBonds == 0) break;
            const double sc = std::max(r.refMag, 1.0);
            std::printf("%-6d %10zu %10zu %11.2fx %13zu %16.2e %s\n",
                        level, cur.nodes.size(), cur.bonds.size(),
                        r.coarseNodes ? double(cur.nodes.size()) / double(r.coarseNodes) : 0.0,
                        r.internalBonds, r.maxCrossErr / sc,
                        r.ok ? "exact" : "*** FAILED ***");
            allLevelsOk = allLevelsOk && r.ok;
            if (next.nodes.size() <= 8 || next.bonds.empty()) { cur = next; break; }
            cur = next;
        }
        std::printf("coarsest level: %zu nodes, %zu bonds\n", cur.nodes.size(), cur.bonds.size());
        std::printf("%s\n", allLevelsOk
            ? "GALERKIN EXACT AT EVERY LEVEL - the hierarchy is a stack of Coupling graphs"
            : "*** GALERKIN IDENTITY FAILED AT SOME LEVEL ***");

        // Now the same recursion with parallel coarse bonds MERGED. Exactness
        // is lost by construction; what matters is how much bond coarsening it
        // buys and how large the resulting approximation is.
        std::printf("\n=== MERGED-bond variant (approximate coarse operator) ===\n");
        std::printf("%-6s %10s %10s %12s %16s\n", "level", "nodes", "bonds",
                    "bond coarse", "rel error");
        Scene m = buildScene(pack, grid, limitScale);
        Lcg rng2(0xA11CEu);
        std::vector<std::size_t> mergedBonds;
        for (int level = 0; level < 8; ++level)
        {
            Scene next;
            const GalerkinReport r = verifyGalerkin(m, rng2, &next, /*merge*/ true);
            if (r.coarseNodes == 0 || r.crossingBonds == 0) break;
            mergedBonds.push_back(m.bonds.size());
            const double sc = std::max(r.refMag, 1.0);
            std::printf("%-6d %10zu %10zu %11.2fx %16.2e\n", level, m.nodes.size(),
                        m.bonds.size(),
                        next.bonds.empty() ? 0.0 : double(m.bonds.size()) / double(next.bonds.size()),
                        r.maxCrossErr / sc);
            if (next.nodes.size() <= 8 || next.bonds.empty()) { m = next; break; }
            m = next;
        }
        std::size_t se = 0, sm = 0;
        for (auto v : exactBonds) se += v;
        for (auto v : mergedBonds) sm += v;
        std::printf("\nV-cycle cost (bond-visits, relative to one fine matvec):\n");
        std::printf("  exact  : %8zu = %.2fx\n", se, double(se) / double(exactBonds[0]));
        std::printf("  merged : %8zu = %.2fx\n", sm, double(sm) / double(mergedBonds[0]));
        return allLevelsOk ? 0 : 1;
    }

    if (false)
    {
        Scene scene = buildScene(pack, grid, limitScale);
        Lcg rng(0xA11CEu);
        const GalerkinReport g = verifyGalerkin(scene, rng, nullptr, false);
        std::printf("\n=== exact-Galerkin check (rigid-body lift as prolongator) ===\n");
        std::printf("fine nodes=%zu -> coarse nodes=%zu (%.2fx coarsening)\n",
                    scene.nodes.size(), g.coarseNodes,
                    g.coarseNodes ? double(scene.nodes.size()) / double(g.coarseNodes) : 0.0);
        std::printf("crossing bonds=%zu  internal bonds=%zu (must vanish)\n",
                    g.crossingBonds, g.internalBonds);
        const double sc = std::max(g.refMag, 1.0);
        std::printf("max |C^T R x - C_c^T x| = %.3e  (%.2e relative to %.3e)\n",
                    g.maxCrossErr, g.maxCrossErr / sc, g.refMag);
        std::printf("max |C^T R x| on internal bonds = %.3e  (%.2e relative)\n",
                    g.maxInternalMag, g.maxInternalMag / sc);
        std::printf("float32 epsilon is 1.2e-07; positions and centroids are float.\n");
        std::printf("%s\n", g.ok
            ? "GALERKIN EXACT - the coarse operator is another Coupling graph"
            : "*** GALERKIN IDENTITY FAILED ***");
        return g.ok ? 0 : 1;
    }

    if (destroy)
    {
        // A low limit scale so gravity alone progressively demolishes the city:
        // it produces a long, realistic fracture cascade without needing a
        // projectile model, and every tick of it exercises applyTopologyChange.
        Scene scene = buildScene(pack, grid, limitScale);
        std::printf("\n=== progressive destruction: %u ticks, limit scale %.3f ===\n",
                    ticks, limitScale);
        const DestroyStats d = runDestroy(scene, iters, ticks, pack.gravity, /*skip*/ true);
        const double fT = d.fractureTicks ? double(d.fractureTicks) : 1.0;
        const double qT = (d.ticks - d.fractureTicks) ? double(d.ticks - d.fractureTicks) : 1.0;
        std::printf("ticks=%u  fracture ticks=%u (%.1f%%)  bonds removed=%u\n",
                    d.ticks, d.fractureTicks,
                    100.0 * d.fractureTicks / std::max(1u, d.ticks), d.bondsRemoved);
        std::printf("%-16s %10s %10s %10s %10s %10s\n",
                    "tick kind", "wall ms", "device ms", "host plan", "host sync", "host fin");
        std::printf("%-16s %10.3f %10.3f %10.3f %10.3f %10.3f\n", "FRACTURE",
                    d.wallFracture / fT, d.devFracture / fT, d.planFracture / fT,
                    d.syncFracture / fT, d.finishFracture / fT);
        std::printf("%-16s %10.3f %10.3f %10.3f %10.3f %10.3f\n", "quiet",
                    d.wallQuiet / qT, d.devQuiet / qT, d.planQuiet / qT,
                    d.syncQuiet / qT, d.finishQuiet / qT);
        std::printf("worst single tick: %.3f ms wall\n", d.worstTickWall);
        std::printf("end state: %zu bonds left, equilibrium residual %.3e, "
                    "non-finite %zu\n",
                    d.finalBonds, d.finalResidual, d.finalNonFinite);
        std::printf("%s\n", d.ok ? "DESTROY OK" : ("*** " + d.note + " ***").c_str());
        return d.ok ? 0 : 1;
    }

    std::vector<RunResult> results;
    for (const auto& c : cases)
    {
        if (!only.empty() && only != c.name) continue;
        if (quick && (std::string(c.name) == "broken-25" || std::string(c.name) == "broken-75"))
            continue;

        Lcg rng(0x5EEDu);
        Scene scene = buildScene(pack, grid, limitScale);
        const std::string n = c.name;
        // Blast counts scale with the replicated area so damage DENSITY is
        // constant across grids. A fixed count would make a grid-4 scene
        // proportionally less damaged than a grid-2 one and quietly change
        // which regime the larger test is measuring.
        const std::uint32_t areaScale = std::max(1u, grid * grid);
        if      (n == "single-building") keepLargestComponent(scene);
        else if (n == "broken-25")       breakFraction(scene, 0.25f, rng);
        else if (n == "broken-50")       breakFraction(scene, 0.50f, rng);
        else if (n == "broken-75")       breakFraction(scene, 0.75f, rng);
        else if (n == "broken-90")       breakFraction(scene, 0.90f, rng);
        else if (n == "broken-99")       breakFraction(scene, 0.99f, rng);
        else if (n == "live-light")      blastFracture(scene, 900u * areaScale / 4u, 2.4f, 3.4f, rng);
        else if (n == "live-city")
        {
            // Calibrated against a real /city session; env-overridable so the
            // calibration can be redone when the scene or shot model changes.
            auto ef = [](const char* k, float d) { const char* r = std::getenv(k); return r ? (float)std::atof(r) : d; };
            auto eu = [](const char* k, std::uint32_t d) { const char* r = std::getenv(k); return r ? (std::uint32_t)std::atol(r) : d; };
            blastFracture(scene, eu("LIVE_BLASTS", 2200u) * areaScale / 4u, ef("LIVE_CORE", 2.8f), ef("LIVE_SHELL", 3.8f), rng);
        }
        else if (n == "live-heavy")      blastFracture(scene, 4500u * areaScale / 4u, 3.2f, 4.2f, rng);
        else if (n == "shatter-4")       shatterToFragments(scene, 4u, rng);
        else if (n == "shatter-3")       shatterToFragments(scene, 3u, rng);
        else if (n == "shatter-2")       shatterToFragments(scene, 2u, rng);
        else if (n == "dust")            scene.bonds.clear();

        std::printf("\n=== %s (%s) ===\n", c.name, c.what);
        results.push_back(runScenario(c.name, scene, iters, solves,
                                      /*skip*/ false, /*damage*/ false,
                                      pack.gravity, histogram, compare, cpuSolves));
    }

    std::printf("\n================================ SUMMARY ================================\n");
    std::printf("%-17s %9s %9s %8s %10s %11s %7s %6s  %s\n",
                "scenario", "nodes", "bonds", "islands", "device ms", "ms/iter",
                "conv@", "ok", "note");
    bool allOk = true;
    for (const auto& r : results)
    {
        std::printf("%-17s %9zu %9zu %8u %10.3f %11.5f %7.1f %6s  %s\n",
                    r.name.c_str(), r.nodes, r.bonds, r.islands,
                    r.deviceMs, r.perIterMs, r.convergedAt,
                    r.ok ? "yes" : "NO", r.note.c_str());
        if (r.eqGpuLin > 0.0) {
            std::printf("%-17s %9s %9s %8s %10s %11s %7s %6s  equilibrium residual "
                        "lin=%.4e ang=%.4e\n", "", "", "", "", "", "", "", "",
                        r.eqGpuLin, r.eqGpuAng);
        }
        allOk = allOk && r.ok;
    }
    std::printf("=========================================================================\n");

    if (compare)
    {
        std::printf("\n=================== CPU (Blast reference) vs GPU ====================\n");
        std::printf("%-17s %10s %10s %8s %9s %11s %11s\n",
                    "scenario", "cpu ms", "gpu ms", "speedup", "cpu iters",
                    "max rel err", "rms rel err");
        for (const auto& r : results)
        {
            if (!r.compared) continue;
            std::printf("%-17s %10.3f %10.3f %7.1fx %9d %11.2e %11.2e\n",
                        r.name.c_str(), r.cpuMs, r.deviceMs,
                        r.deviceMs > 0 ? r.cpuMs / r.deviceMs : 0.0,
                        r.cpuIterations, static_cast<double>(r.maxRelative), r.rmsRelative);
        }
        std::printf("====================================================================\n");

        std::printf("\n============ EQUILIBRIUM RESIDUAL (the real quality gate) ============\n");
        std::printf("%-17s %12s %12s %12s %12s\n", "scenario",
                    "cpu lin", "gpu lin", "cpu ang", "gpu ang");
        bool qualityOk = true;
        for (const auto& r : results)
        {
            if (!r.compared) continue;
            std::printf("%-17s %12.3e %12.3e %12.3e %12.3e\n", r.name.c_str(),
                        r.eqCpuLin, r.eqGpuLin, r.eqCpuAng, r.eqGpuAng);
            // The GPU must not be materially WORSE at satisfying equilibrium
            // than the CPU reference on the same budget. Being better is fine.
            if (r.eqGpuLin > std::max(1e-3, r.eqCpuLin * 2.0)) qualityOk = false;
            if (r.eqGpuAng > std::max(1e-3, r.eqCpuAng * 2.0)) qualityOk = false;
        }
        std::printf("=====================================================================\n");
        std::printf("quality gate (GPU equilibrium residual <= 2x CPU): %s\n",
                    qualityOk ? "PASS" : "*** FAIL ***");
        allOk = allOk && qualityOk;
    }

    std::printf("%s\n", allOk ? "ALL SCENARIOS OK" : "*** FAILURES PRESENT ***");
    return allOk ? 0 : 1;
}
