// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2016-2024 NVIDIA Corporation. All rights reserved.


#include <chrono>
#include "NvBlastExtStressSolver.h"
#include "NvBlast.h"
#include "NvBlastGlobals.h"
#include "NvBlastArray.h"
#include "NvBlastHashMap.h"
#include "NvBlastHashSet.h"
#include "NvBlastAssert.h"
#include "NvBlastIndexFns.h"

#include "NsFPU.h"
#include "NvBlastNvSharedHelpers.h"
#include "NvCMath.h"

#include "stress.h"
#include "buffer.h"
#include "simd/simd_device_query.h"
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
#include "NvBlastExtStressGpu.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#define USE_SCALAR_IMPL 0
#define WARM_START 1
#define GRAPH_INTERGRIRY_CHECK 0

#if GRAPH_INTERGRIRY_CHECK
#include <set>
#endif


namespace Nv
{
namespace Blast
{

using namespace nvidia;

#if defined(NVBLAST_ENABLE_CUDA_STRESS)
/// See the A/B branch in SupportGraphProcessor::removeBondIfExists.
static bool gpuIncrementalRemovalEnabled()
{
    static const bool enabled =
        std::getenv("BLAST_GPU_NO_INCREMENTAL_REMOVAL") == nullptr;
    return enabled;
}
#else
static bool gpuIncrementalRemovalEnabled() { return true; }
#endif

static_assert(sizeof(NvVec3) == sizeof(NvcVec3), "sizeof(NvVec3) must equal sizeof(NvcVec3).");
static_assert(offsetof(NvVec3, x) == offsetof(NvcVec3, x) &&
              offsetof(NvVec3, y) == offsetof(NvcVec3, y) &&
              offsetof(NvVec3, z) == offsetof(NvcVec3, z),
              "Elements of NvVec3 and NvcVec3 must have the same struct offset.");


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                           Conjugate Gradient Solver
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ConjugateGradientImpulseSolver
{
public:
    ConjugateGradientImpulseSolver(uint32_t nodeCount, uint32_t maxBondCount)
    {
        m_bonds.reserve(maxBondCount);
        m_impulses.reserve(maxBondCount);
        reset(nodeCount);
    }

    ~ConjugateGradientImpulseSolver()
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        if (m_gpuSolver)
        {
            m_gpuSolver->release();
        }
#endif
    }

    void getBondImpulses(uint32_t bond, NvVec3& impulseLinear, NvVec3& impulseAngular) const
    {
        NVBLAST_ASSERT(bond < m_impulses.size());
        const AngLin6& f = m_impulses[bond];
        *(NvcVec3*)&impulseAngular = f.ang;
        *(NvcVec3*)&impulseLinear = f.lin;
    }

    void getBondNodes(uint32_t bond, uint32_t& node0, uint32_t& node1) const
    {
        NVBLAST_ASSERT(bond < m_bonds.size());
        const SolverBond& b = m_bonds[bond];
        node0 = b.nodes[0];
        node1 = b.nodes[1];
    }

    uint32_t getBondCount() const
    {
        return m_bonds.size();
    }

    uint32_t getNodeCount() const
    {
        return m_nodes.size();
    }

    void setNodeMassInfo(uint32_t node, const NvVec3& CoM, float mass, float inertia)
    {
        NVBLAST_ASSERT(node < m_nodes.size());
        SolverNodeS& n = m_nodes[node];
        n.CoM = { CoM.x, CoM.y, CoM.z };
        n.mass = std::max(mass, 0.0f);  // No negative masses, but 0 is meaningful (== infinite)
        n.inertia = std::max(inertia, 0.0f);    // Ditto for inertia
        m_forceColdStart = true;
    }

    void initialize()
    {
        StressProcessor::DataParams params;
        params.centerBonds = true;
        params.equalizeMasses = true;
        m_stressProcessor.prepare(m_nodes.begin(), m_nodes.size(), m_bonds.begin(), m_bonds.size(), params);
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        if (m_gpuSolver)
        {
            m_gpuSolver->release();
            m_gpuSolver = nullptr;
        }
        if (m_gpuRequested
            && !m_nodes.empty()
            && m_bonds.size() >= m_gpuMinimumBondCount)
        {
            m_gpuNodes.resize(m_nodes.size());
            for (uint32_t i = 0; i < m_nodes.size(); ++i)
            {
                const SolverNodeS& node = m_nodes[i];
                ExtStressGpuNode& gpu = m_gpuNodes[i];
                gpu.position[0] = node.CoM.x;
                gpu.position[1] = node.CoM.y;
                gpu.position[2] = node.CoM.z;
                gpu.mass = node.mass;
                gpu.inertia = node.inertia;
            }
            m_gpuBonds.resize(m_bonds.size());
            for (uint32_t i = 0; i < m_bonds.size(); ++i)
            {
                const SolverBond& bond = m_bonds[i];
                ExtStressGpuBond& gpu = m_gpuBonds[i];
                gpu.node0 = bond.nodes[0];
                gpu.node1 = bond.nodes[1];
                gpu.centroid[0] = bond.centroid.x;
                gpu.centroid[1] = bond.centroid.y;
                gpu.centroid[2] = bond.centroid.z;
                // Normals are recomputed from node geometry in ExtStressGpuSolver.
                gpu.normal[0] = gpu.normal[1] = gpu.normal[2] = 0.0f;
                // Area is geometry: the summed authored contact patch of the
                // group's members. Health IS the damage pool and must seed from
                // area, not 1.0 — a uniform seed is exactly the bug that once
                // made authored strength meaningless on the CPU path. Caveat:
                // bonds partially damaged before a topology rebuild re-seed at
                // full area here; only the unwired on-device damage path reads
                // this, and it must re-sync healths when it is wired.
                gpu.area = bond.area > 0.0f ? bond.area : 1.0f;
                gpu.health = gpu.area;
                gpu.material = bond.material;
            }
            // Convert the resolved material table for the device-side damage
            // kernel; layouts match field-for-field.
            m_gpuMaterials.resize(m_materialCount ? m_materialCount : 1);
            for (uint32_t material = 0; material < m_gpuMaterials.size(); ++material)
            {
                const ExtStressMaterial source = (m_materials && material < m_materialCount)
                    ? m_materials[material]
                    : ExtStressMaterial();
                ExtStressGpuMaterial& gpu = m_gpuMaterials[material];
                gpu.compressionElasticLimit = source.compressionElasticLimit;
                gpu.compressionFatalLimit = source.compressionFatalLimit;
                gpu.tensionElasticLimit = source.tensionElasticLimit;
                gpu.tensionFatalLimit = source.tensionFatalLimit;
                gpu.shearElasticLimit = source.shearElasticLimit;
                gpu.shearFatalLimit = source.shearFatalLimit;
            }
            m_gpuSolver = ExtStressGpuSolver::create(
                m_gpuNodes.data(),
                static_cast<uint32_t>(m_gpuNodes.size()),
                m_gpuBonds.data(),
                static_cast<uint32_t>(m_gpuBonds.size()),
                m_gpuMaterials.data(),
                static_cast<uint32_t>(m_gpuMaterials.size()),
                m_gpuCudaContext);
            m_gpuVelocities.resize(m_nodes.size());
            m_gpuImpulses.resize(m_bonds.size());
            m_gpuIslandsSkipped = 0;
            m_gpuIslandsTotal = 0;
        }
#endif
    }

    bool setGpuAccelerated(bool enabled)
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        m_gpuRequested = enabled;
        m_forceColdStart = true;
        return true;
#else
        return !enabled;
#endif
    }

    void setGpuCudaContext(void* cudaContext)
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        m_gpuCudaContext = cudaContext;
        m_forceColdStart = true;
#else
        NV_UNUSED(cudaContext);
#endif
    }

    void setGpuMinimumBondCount(uint32_t bondCount)
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        m_gpuMinimumBondCount = bondCount;
        m_forceColdStart = true;
#else
        NV_UNUSED(bondCount);
#endif
    }

    bool getGpuAccelerated() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        return m_gpuRequested && m_gpuSolver != nullptr;
#else
        return false;
#endif
    }

    float getGpuSolveMilliseconds() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        return m_gpuFrameSolveMilliseconds;
#else
        return 0.0f;
#endif
    }

    float getGpuImpulseCopyMilliseconds() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        return m_gpuImpulseCopyMilliseconds;
#else
        return 0.0f;
#endif
    }

    uint32_t getGpuImpulseCopyCount() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        return m_gpuImpulseCopyCount;
#else
        return 0;
#endif
    }

    float getGpuHostWorkMilliseconds() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        return m_gpuFrameHostWorkMilliseconds;
#else
        return 0.0f;
#endif
    }

    float getGpuHostBlockedMilliseconds() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        return m_gpuFrameHostBlockedMilliseconds;
#else
        return 0.0f;
#endif
    }

    uint64_t getGpuHostToDeviceBytes() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        return m_gpuFrameHostToDeviceBytes;
#else
        return 0;
#endif
    }

    uint64_t getGpuDeviceToHostBytes() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        return m_gpuFrameDeviceToHostBytes;
#else
        return 0;
#endif
    }

    void setNodeVelocities(uint32_t node, const NvVec3& velocityLinear, const NvVec3& velocityAngular)
    {
        NVBLAST_ASSERT(node < m_velocities.size());
        AngLin6& v = m_velocities[node];
        m_inputsChanged = m_inputsChanged
            || v.ang.x != velocityAngular.x
            || v.ang.y != velocityAngular.y
            || v.ang.z != velocityAngular.z
            || v.lin.x != velocityLinear.x
            || v.lin.y != velocityLinear.y
            || v.lin.z != velocityLinear.z;
        v.ang = { velocityAngular.x, velocityAngular.y, velocityAngular.z };
        v.lin = { velocityLinear.x, velocityLinear.y, velocityLinear.z };
    }

    uint32_t addBond(uint32_t node0, uint32_t node1, const NvVec3& bondCentroid, float area = 1.0f, uint32_t material = 0)
    {
        SolverBond b;
        b.nodes[0] = node0;
        b.nodes[1] = node1;
        b.centroid = { bondCentroid.x, bondCentroid.y, bondCentroid.z };
        b.area = area > 0.0f ? area : 1.0f;
        b.material = material;
        m_bonds.pushBack(b);
        m_impulses.push_back({{0,0,0},{0,0,0}});
        m_forceColdStart = true;
        return m_bonds.size() - 1;
    }

    void addBondArea(uint32_t bondIndex, float area)
    {
        NVBLAST_ASSERT(bondIndex < m_bonds.size());
        if (area > 0.0f)
        {
            m_bonds[bondIndex].area += area;
            m_forceColdStart = true;
        }
    }

    void setMaterialTable(const ExtStressMaterial* materials, uint32_t materialCount)
    {
        m_materials = materials;
        m_materialCount = materialCount;
        // The device-side table is uploaded at GPU-solver creation; force a
        // rebuild so a changed table reaches the device. Tables change at
        // authoring/sweep time, not per frame, so the cold start is cheap.
        m_forceColdStart = true;
    }

    uint32_t getBondMaterial(uint32_t bondIndex) const
    {
        NVBLAST_ASSERT(bondIndex < m_bonds.size());
        return m_bonds[bondIndex].material;
    }

    void setBondMaterial(uint32_t bondIndex, uint32_t material)
    {
        NVBLAST_ASSERT(bondIndex < m_bonds.size());
        m_bonds[bondIndex].material = material;
    }

    void replaceWithLast(uint32_t bondIndex)
    {
        m_bonds.replaceWithLast(bondIndex);
        if ((size_t)bondIndex + 2 < m_impulses.size())
        {
            m_impulses[bondIndex] = m_impulses.back();
            m_impulses.resize(m_impulses.size() - 1);
        }
        m_stressProcessor.removeBond(bondIndex);
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        // The CUDA solver applies the same swap-with-last rather than being
        // rebuilt. See the note in removeBondIfExists.
        if (m_gpuSolver)
        {
            m_gpuSolver->removeBond(bondIndex);
        }
#endif
    }

    void reset(uint32_t nodeCount)
    {
        m_nodes.resize(nodeCount);
        memset(m_nodes.begin(), 0, sizeof(SolverNodeS)*nodeCount);
        m_velocities.resize(nodeCount);
        memset(m_velocities.data(), 0, sizeof(AngLin6)*nodeCount);
        clearBonds();
        m_error_sq = {FLT_MAX, FLT_MAX};
        m_converged = false;
        m_forceColdStart = true;
        m_inputsChanged = true;
    }

    void clearBonds()
    {
        m_bonds.clear();
        m_impulses.resize(0);
        m_forceColdStart = true;
    }

    void setSkipStableUnconverged(bool enabled) { m_skipStableUnconverged = enabled; }
    void solve(uint32_t iterationCount, bool warmStart = true, bool islandAware = false, bool skipSettled = false)
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        m_gpuFrameSolveMilliseconds = 0.0f;
        m_gpuFrameHostWorkMilliseconds = 0.0f;
        m_gpuImpulseCopyMilliseconds = 0.0f;
        m_gpuImpulseCopyCount = 0;
        m_gpuFrameHostBlockedMilliseconds = 0.0f;
        m_gpuFrameHostToDeviceBytes = 0;
        m_gpuFrameDeviceToHostBytes = 0;
#endif
        if (skipSettled
            && warmStart
            && m_converged
            && !m_forceColdStart
            && !m_inputsChanged)
        {
            m_error_sq = {0.0f, 0.0f};
            return;
        }
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        if (m_gpuSolver)
        {
            for (uint32_t i = 0; i < m_velocities.size(); ++i)
            {
                m_gpuVelocities[i].angular =
                    {m_velocities[i].ang.x, m_velocities[i].ang.y, m_velocities[i].ang.z};
                m_gpuVelocities[i].linear =
                    {m_velocities[i].lin.x, m_velocities[i].lin.y, m_velocities[i].lin.z};
            }
            ExtStressGpuSolveParams gpuParams;
            // The GPU solver is island-aware: each disconnected component gets
            // its own conjugate-gradient scalars and its own convergence test,
            // matching the CPU sub-solves. It therefore takes the same
            // iteration budget as the CPU path, and the same budget means the
            // same answer -- which is what makes a scene portable between the
            // two backends instead of needing per-backend tuning.
            //
            // This used to inflate the budget to a 32/44 floor, because the
            // solve was global: one shared residual could sit under tolerance
            // while a small component was still badly resolved, and extra
            // iterations were the only lever. Per-island convergence removes
            // the need, and removing it restores CPU/GPU agreement.
            gpuParams.maxIterations = iterationCount;
            gpuParams.tolerance = 0.001f;
            gpuParams.warmStart = warmStart && !m_forceColdStart;
            // Settled-island skipping on the GPU, and it means the same thing
            // it means on the CPU (stress.cpp solveIslandAware): an island
            // whose dynamic nodes' velocities are bit-identical to its last
            // solve, and whose last solve converged, keeps the impulses it
            // already has. Its warm-start state stays resident on the device
            // and is not even rescaled, so nothing about it drifts while it
            // sits still.
            //
            // Without this the GPU uploaded and solved the whole graph every
            // tick regardless -- correct, and the right trade when a solve was
            // 2.4 ms over a few hundred islands, but not at city scale where
            // nearly every island is a settled debris cluster.
            gpuParams.skipSettledIslands = skipSettled && warmStart;
            gpuParams.skipStableUnconverged = m_skipStableUnconverged;
            if (m_gpuSolver->solveAndReadbackImpulses(
                    m_gpuVelocities.data(),
                    gpuParams,
                    m_gpuImpulses.data(),
                    static_cast<uint32_t>(m_gpuImpulses.size())))
            {
                // Only the bonds the solver actually re-solved. Copying all of
                // them would hand the saving straight back: at city scale this
                // loop is tens of thousands of iterations of pure host work,
                // and it is inside the "blast solve" phase.
                // Timed: solve_ms minus the kernel, host work and blocking
                // left 32% unnamed, and it is not initialize or calcError
                // (0.05 and 0.00 ms). This loop is the remaining candidate --
                // a per-bond host copy inside the solve phase.
                const auto copyStart = std::chrono::steady_clock::now();
                uint32_t changedCount = 0;
                const uint32_t* changed = m_gpuSolver->lastChangedBonds(changedCount);
                const uint32_t impulseCount = static_cast<uint32_t>(m_impulses.size());
                for (uint32_t k = 0; k < changedCount; ++k)
                {
                    const uint32_t i = changed[k];
                    if (i >= impulseCount)
                    {
                        continue;
                    }
                    m_impulses[i].ang = {
                        m_gpuImpulses[i].angular.x,
                        m_gpuImpulses[i].angular.y,
                        m_gpuImpulses[i].angular.z};
                    m_impulses[i].lin = {
                        m_gpuImpulses[i].linear.x,
                        m_gpuImpulses[i].linear.y,
                        m_gpuImpulses[i].linear.z};
                }
                m_gpuImpulseCopyMilliseconds =
                    std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - copyStart).count();
                m_gpuImpulseCopyCount = changedCount;
                m_gpuIslandsSkipped = m_gpuSolver->telemetry().islandsSkipped;
                m_gpuIslandsTotal = m_gpuSolver->telemetry().islandCount;
                m_converged = m_gpuSolver->telemetry().converged;
                m_gpuFrameSolveMilliseconds =
                    m_gpuSolver->telemetry().solveMilliseconds;
                // Host wall inside solve(), split into working and blocked.
                // The bench says the wrapper is almost entirely blocked; the
                // live gap is 3.8 ms against a 1.2 ms kernel, which the bench
                // cannot reproduce, so the split has to be measured where the
                // gap actually is.
                m_gpuFrameHostWorkMilliseconds =
                    m_gpuSolver->telemetry().hostPlanMilliseconds
                    + m_gpuSolver->telemetry().hostFinishMilliseconds;
                m_gpuFrameHostBlockedMilliseconds =
                    m_gpuSolver->telemetry().hostSyncMilliseconds;
                m_gpuFrameHostToDeviceBytes =
                    m_gpuSolver->telemetry().hostToDeviceBytes;
                m_gpuFrameDeviceToHostBytes =
                    m_gpuSolver->telemetry().deviceToHostBytes;
                m_error_sq = m_converged
                    ? AngLin6ErrorSq{0.0f, 0.0f}
                    : AngLin6ErrorSq{FLT_MAX, FLT_MAX};
                m_forceColdStart = false;
                m_inputsChanged = false;
                return;
            }
        }
#endif
        StressProcessor::SolverParams params;
        params.maxIter = iterationCount;
        params.tolerance = 0.001f;
        params.warmStart = warmStart && !m_forceColdStart;
        params.islandAware = islandAware;
        params.skipSettled = skipSettled;
        m_converged = (m_stressProcessor.solve(m_impulses.data(), m_velocities.data(), params, &m_error_sq) >= 0);
        m_forceColdStart = false;
        m_inputsChanged = false;
    }

    // Number of settled islands skipped in the last island-aware solve (Stage 3 instrumentation).
    // The GPU path keeps its own partition -- the same one, by the same rule -- so it reports
    // its own count; reading the CPU processor's here is what made solver_islands_skipped
    // permanently 0 on CUDA and hid that the GPU was re-solving a settled city every tick.
    uint32_t getIslandsSkipped() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        if (m_gpuSolver)
        {
            return m_gpuIslandsSkipped;
        }
#endif
        return m_stressProcessor.getLastIslandsSkipped();
    }

    uint32_t getIslandsTotal() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        if (m_gpuSolver)
        {
            return m_gpuIslandsTotal;
        }
#endif
        return m_stressProcessor.getLastIslandsTotal();
    }

    bool calcError(float& linear, float& angular) const
    {
        linear = sqrtf(m_error_sq.lin);
        angular = sqrtf(m_error_sq.ang);
        return m_converged;
    }

private:
    Array<SolverNodeS>::type    m_nodes;
    Array<SolverBond>::type     m_bonds;
    StressProcessor             m_stressProcessor;
    POD_Buffer<AngLin6>         m_velocities;
    POD_Buffer<AngLin6>         m_impulses;
    AngLin6ErrorSq              m_error_sq;
    bool                        m_converged;
    bool                        m_forceColdStart;
    bool m_skipStableUnconverged = false;
    bool                        m_inputsChanged;
    // Borrowed from the owning solver: resolved material table for the GPU
    // damage-kernel seed. Not consumed by the CPU solve.
    const ExtStressMaterial*    m_materials{nullptr};
    uint32_t                    m_materialCount{0};
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
    bool                                m_gpuRequested{false};
    void*                               m_gpuCudaContext{nullptr};
    uint32_t                            m_gpuMinimumBondCount{4096};
    ExtStressGpuSolver*                 m_gpuSolver{nullptr};
    std::vector<ExtStressGpuNode>       m_gpuNodes;
    std::vector<ExtStressGpuBond>       m_gpuBonds;
    std::vector<ExtStressGpuMaterial>   m_gpuMaterials;
    std::vector<ExtStressGpuImpulse>    m_gpuVelocities;
    std::vector<ExtStressGpuImpulse>    m_gpuImpulses;
    uint32_t                            m_gpuIslandsSkipped{0};
    uint32_t                            m_gpuIslandsTotal{0};
    float                               m_gpuFrameSolveMilliseconds{0.0f};
    float                               m_gpuFrameHostWorkMilliseconds{0.0f};
    /// Per-bond host copy of solved impulses, and how many bonds it touched.
    float                               m_gpuImpulseCopyMilliseconds{0.0f};
    uint32_t                            m_gpuImpulseCopyCount{0};
    float                               m_gpuFrameHostBlockedMilliseconds{0.0f};
    uint64_t                            m_gpuFrameHostToDeviceBytes{0};
    uint64_t                            m_gpuFrameDeviceToHostBytes{0};
#endif
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                   Graph Processor
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if GRAPH_INTERGRIRY_CHECK
#define CHECK_GRAPH_INTEGRITY checkGraphIntegrity()
#else
#define CHECK_GRAPH_INTEGRITY ((void)0)
#endif

class SupportGraphProcessor
{

public:
    struct BondData
    {
        uint32_t node0;
        uint32_t node1;
        uint32_t blastBondIndex;
        // linear stresses
        float    stressNormal;  // negative values represent compression pressure, positive represent tension
        float    stressShear;

        // The normal used to compute stress values
        // Can be different than the bond normal if graph reduction is used
        // and multiple bonds are grouped together
        nvidia::NvVec3 normal;

        // Centroid used to compute node offsets, instead of assuming the bond is halfway between node positions.
        // This also allows the bonds to the world node to be drawn
        nvidia::NvVec3 centroid;
    };

    struct NodeData
    {
        float mass;
        float volume;
        NvVec3 localPos;
        NvVec3 localVel;
        uint32_t solverNode;
        uint32_t neighborsCount;

        // --- chunk crushing (see ExtStressCrushProperties) ---
        // Love-Weber virial accumulator, sigma*V in Voigt order
        // [xx, yy, zz, xy, xz, yz]. Symmetric by construction: each
        // contribution is added as its symmetric part.
        float virial[6];
        float strainRate;       // 1/s, supplied by the host
        float crushDamage;      // [0,1], 1 = pulverized
        float pressure;         // Pa, p = -trace(sigma)/3, positive in compression
        float deviator;         // Pa, von Mises equivalent q
        // How close this chunk is to its crush yield surface: max of
        // q / (cohesion + slope*p) and p / capPressure. 1 means at yield.
        // The crush analogue of bond utilisation.
        float crushUtilisation;
        bool  crushed;              // latched once crushDamage reaches 1
        bool  crushReported;        // drained through getCrushedNodes()
        bool  crushCommandIssued;   // a chunk fracture command was already emitted
    };

    struct SolverNodeData
    {
        uint32_t supportNodesCount;
        NvVec3 localPos;
        union
        {
            float mass;
            int32_t indexShift;
        };
        float volume;
    };

    struct SolverBondData
    {
        InlineArray<uint32_t, 8>::type blastBondIndices;
    };

    SupportGraphProcessor(uint32_t nodeCount, uint32_t maxBondCount) :
        m_solver(nodeCount, maxBondCount), m_nodesDirty(true), m_bondsDirty(true), m_islandCount(0)
    {
        m_nodesData.resize(nodeCount);
        m_bondsData.reserve(maxBondCount);

        m_solverNodesData.resize(nodeCount);
        m_solverBondsData.reserve(maxBondCount);

        m_solverBondsMap.reserve(maxBondCount);

        m_blastBondIndexMap.resize(maxBondCount);
        memset(m_blastBondIndexMap.begin(), 0xFF, m_blastBondIndexMap.size() * sizeof(uint32_t));

        resetVelocities();
        resetCrushState();
    }

    const NodeData& getNodeData(uint32_t node) const
    {
        return m_nodesData[node];
    }

    const BondData& getBondData(uint32_t bond) const
    {
        return m_bondsData[bond];
    }

    const SolverNodeData& getSolverNodeData(uint32_t node) const
    {
        return m_solverNodesData[node];
    }

    const SolverBondData& getSolverBondData(uint32_t bond) const
    {
        return m_solverBondsData[bond];
    }

    void getSolverInternalBondImpulses(uint32_t bond, NvVec3& impulseLinear, NvVec3& impulseAngular) const
    {
        m_solver.getBondImpulses(bond, impulseLinear, impulseAngular);
    }

    void getSolverInternalBondNodes(uint32_t bond, uint32_t& node0, uint32_t& node1) const
    {
        m_solver.getBondNodes(bond, node0, node1);
    }

    uint32_t getBondCount() const
    {
        return m_bondsData.size();
    }

    uint32_t getNodeCount() const
    {
        return m_nodesData.size();;
    }

    uint32_t getSolverBondCount() const
    {
        return m_solverBondsData.size();
    }

    uint32_t getSolverNodeCount() const
    {
        return m_solverNodesData.size();;
    }

    uint32_t getOverstressedBondCount() const
    {
        return m_overstressedBondCount;
    }

    /// E1: may this node's adjacency contain an overstressed bond? TRUE when
    /// the mask is missing (pre-first-update), so callers degrade to the full
    /// walk rather than silently skipping real work.
    bool isNodeOverstressed(uint32_t node) const
    {
        return node >= m_nodeOverstressed.size() || m_nodeOverstressed[node] != 0;
    }

    // Number of connected components (islands) in the solver graph, computed in
    // sync() on topology change. Static (mass<=0) nodes are cut points.
    uint32_t getIslandCount() const
    {
        // Report the partition the SOLVE actually ran over, when there is one.
        // m_islandCount is recomputed only on a graph resync, and now that the
        // CUDA solver survives fracture those are rare -- it went stale at 27
        // next to 1063 islands skipped, which is worse than no number at all.
        // Both paths' solver-side counts mean the same thing: connected
        // components that contain at least one bond.
        const uint32_t solverIslands = m_solver.getIslandsTotal();
        return solverIslands ? solverIslands : m_islandCount;
    }

    // Settled islands skipped in the last island-aware solve (Stage 3 instrumentation).
    uint32_t getIslandsSkipped() const
    {
        return m_solver.getIslandsSkipped();
    }

    uint32_t getIslandsTotal() const
    {
        return m_solver.getIslandsTotal();
    }

    bool setGpuAccelerated(bool enabled)
    {
        const bool available = m_solver.setGpuAccelerated(enabled);
        if (available)
        {
            m_nodesDirty = true;
        }
        return available;
    }

    void setGpuCudaContext(void* cudaContext)
    {
        m_solver.setGpuCudaContext(cudaContext);
        m_nodesDirty = true;
    }

    void setGpuMinimumBondCount(uint32_t bondCount)
    {
        m_solver.setGpuMinimumBondCount(bondCount);
        m_nodesDirty = true;
    }

    bool getGpuAccelerated() const { return m_solver.getGpuAccelerated(); }
    float getGpuSolveMilliseconds() const { return m_solver.getGpuSolveMilliseconds(); }
    float getGpuHostWorkMilliseconds() const { return m_solver.getGpuHostWorkMilliseconds(); }
    float getGpuImpulseCopyMilliseconds() const { return m_solver.getGpuImpulseCopyMilliseconds(); }
    uint32_t getGpuImpulseCopyCount() const { return m_solver.getGpuImpulseCopyCount(); }
    float getGpuHostBlockedMilliseconds() const { return m_solver.getGpuHostBlockedMilliseconds(); }
    uint64_t getGpuHostToDeviceBytes() const { return m_solver.getGpuHostToDeviceBytes(); }
    uint64_t getGpuDeviceToHostBytes() const { return m_solver.getGpuDeviceToHostBytes(); }

    void calcSolverBondStresses(
        uint32_t bondIdx, float bondArea, float nodeDist, const nvidia::NvVec3& bondNormal,
        float& stressNormal, float& stressShear) const
    {
        if (!canTakeDamage(bondArea))
        {
            stressNormal = stressShear = 0.0f;
            return;
        }

        // impulseLinear in the direction of the bond normal is stressNormal, perpendicular is stressShear
        // ignore impulseAngular for now, not sure how to account for that
        // convert to pressure to factor out area
        NvVec3 impulseLinear, impulseAngular;
        getSolverInternalBondImpulses(bondIdx, impulseLinear, impulseAngular);
        const float normalComponentLinear = impulseLinear.dot(bondNormal);
        stressNormal = normalComponentLinear / bondArea;
        const float impulseLinearMagSqr = impulseLinear.magnitudeSquared();
        stressShear = sqrtf(impulseLinearMagSqr - normalComponentLinear * normalComponentLinear) / bondArea;

        // impulseAngular in the direction of the bond normal is twist, perpendicular is bend
        // take abs() of the dot product because only the magnitude of the twist matters, not direction
        const float normalComponentAngular = abs(impulseAngular.dot(bondNormal));
        const float twist = normalComponentAngular / bondArea;
        const float impulseAngularMagSqr = impulseAngular.magnitudeSquared();
        const float bend = sqrtf(impulseAngularMagSqr - normalComponentAngular * normalComponentAngular) / bondArea;

        // interpret angular pressure as a composition of linear pressures
        // dividing by nodeDist for scaling
        const float twistContribution = twist * 2.0f / nodeDist;
        stressShear += twistContribution;
        const float bendContribution = bend * 2.0f / nodeDist;
        stressNormal += std::copysign(bendContribution, stressNormal);
    }

    float mapStressToRange(float stress, float elasticLimit, float fatalLimit) const
    {
        if (stress < elasticLimit)
        {
            return 0.5f * stress / elasticLimit;
        }
        else
        {
            return fatalLimit > elasticLimit ? 0.5f + 0.5f * (stress - elasticLimit) / (fatalLimit - elasticLimit) : 1.0f;
        }
    }

    float getSolverBondStressPct(uint32_t bondIdx, const float* bondHealths, ExtStressSolver::DebugRenderMode mode) const
    {
        // All member bonds share the group's stress values, but each maps to a
        // percentage against its OWN material limits — so every intact member
        // must be visited (no early break): a weak member can be near-fatal
        // while a strong one in the same group is barely loaded.
        float compressionStress, tensionStress, shearStress;
        float stress = -1.0f;
        const auto& blastBondIndices = m_solverBondsData[bondIdx].blastBondIndices;
        for (const auto blastBondIndex : blastBondIndices)
        {
            // only consider the stress values on bonds that are intact
            if (bondHealths[blastBondIndex] > 0.0f && getBondStress(blastBondIndex, compressionStress, tensionStress, shearStress))
            {
                const ExtStressMaterial& material = materialForBlastBond(blastBondIndex);
                if (mode == ExtStressSolver::STRESS_PCT_COMPRESSION || mode == ExtStressSolver::STRESS_PCT_MAX)
                {
                    compressionStress = mapStressToRange(compressionStress, material.compressionElasticLimit, material.compressionFatalLimit);
                    stress = std::max(compressionStress, stress);
                }

                if (mode == ExtStressSolver::STRESS_PCT_TENSION || mode == ExtStressSolver::STRESS_PCT_MAX)
                {
                    tensionStress = mapStressToRange(tensionStress, material.tensionElasticLimit, material.tensionFatalLimit);
                    stress = std::max(tensionStress, stress);
                }

                if (mode == ExtStressSolver::STRESS_PCT_SHEAR || mode == ExtStressSolver::STRESS_PCT_MAX)
                {
                    shearStress = mapStressToRange(shearStress, material.shearElasticLimit, material.shearFatalLimit);
                    stress = std::max(shearStress, stress);
                }
            }
        }

        // return a value < 0.0f if all bonds are broken
        return stress;
    }

    void setNodeInfo(uint32_t node, float mass, float volume, NvVec3 localPos)
    {
        m_nodesData[node].mass = mass;
        m_nodesData[node].volume = volume;
        m_nodesData[node].localPos = localPos;
        m_nodesDirty = true;
    }

    void setNodeNeighborsCount(uint32_t node, uint32_t neighborsCount)
    {
        // neighbors count is expected to be the number of nodes on 1 island/actor.
        m_nodesData[node].neighborsCount = neighborsCount;

        // check for too huge aggregates (happens after island's split)
        if (!m_nodesDirty)
        {
            m_nodesDirty |= (m_solverNodesData[m_nodesData[node].solverNode].supportNodesCount > neighborsCount / 2);
        }
    }

    void addNodeForce(uint32_t node, const NvVec3& force, ExtForceMode::Enum mode)
    {
        const float mass = m_nodesData[node].mass;
        if (mass > 0)
        {
            // NOTE - passing in acceleration as velocity.  The impulse solver's output will be interpreted as force.
            m_nodesData[node].localVel += (mode == ExtForceMode::FORCE) ? force/mass : force;
        }
    }

    /**
    addNodeForce that also records the force's application point, so an
    EXTERNAL contact enters the node's Love-Weber virial sum.

    The plain addNodeForce above discards the point (it only ever needed the
    resultant), which is fine for the bond solve but loses exactly the
    information a per-chunk stress tensor is built from. Only surface
    tractions belong in the sum -- gravity and centrifugal loads are body
    forces and keep using addNodeForce.
    */
    void addNodeForceAt(uint32_t node, const NvVec3& localPosition, const NvVec3& force, ExtForceMode::Enum mode)
    {
        if (m_crushEnabled && materialForNode(node).crush.enabled())
        {
            // The virial is a sum over forces; convert acceleration first.
            const NvVec3 actualForce =
                (mode == ExtForceMode::FORCE) ? force : force * m_nodesData[node].mass;
            accumulateVirial(node, localPosition - m_nodesData[node].localPos, actualForce);
        }
        addNodeForce(node, force, mode);
    }

    void addBond(uint32_t node0, uint32_t node1, uint32_t blastBondIndex)
    {
        if (isInvalidIndex(m_blastBondIndexMap[blastBondIndex]))
        {
            const BondData data = {
                node0,
                node1,
                blastBondIndex,
                0.0f
            };
            m_bondsData.pushBack(data);
            m_blastBondIndexMap[blastBondIndex] = m_bondsData.size() - 1;
        }
    }

    void removeBondIfExists(uint32_t blastBondIndex)
    {
        const uint32_t bondIndex = m_blastBondIndexMap[blastBondIndex];

        if (!isInvalidIndex(bondIndex))
        {
            const BondData& bond = m_bondsData[bondIndex];
            const uint32_t solverNode0 = m_nodesData[bond.node0].solverNode;
            const uint32_t solverNode1 = m_nodesData[bond.node1].solverNode;
            bool isBondInternal = (solverNode0 == solverNode1);

            if (isBondInternal)
            {
                // internal bond sadly requires graph resync (it never happens on reduction level '0')
                m_nodesDirty = true;
            }
            else if (!m_nodesDirty && m_solver.getGpuAccelerated() && !gpuIncrementalRemovalEnabled())
            {
                // A/B switch (BLAST_GPU_NO_INCREMENTAL_REMOVAL=1): restore the
                // full graph rebuild a broken bond used to force whenever the
                // CUDA solver was active, so the cost of the rebuild can be
                // measured against the cost of not doing it on the same binary
                // and the same scene.
                m_nodesDirty = true;
            }
            else if (!m_nodesDirty)
            {
                // otherwise it's external bond, we can remove it manually and keep graph synced
                // we don't need to spend time there if (m_nodesDirty == true), graph will be resynced anyways
                //
                // This branch used to be unreachable whenever the CUDA solver
                // was active: a broken bond forced a full graph resync, which
                // destroyed and rebuilt the GPU solver. At city scale a bond
                // breaks on most ticks, so the device's topology, island
                // partition, warm-start impulses and settled baseline were
                // discarded and rebuilt every tick -- which is why the GPU path
                // could never carry anything forward and skipped nothing.
                // ExtStressGpuSolver::removeBond now applies the same
                // swap-with-last the host does, so the solver survives its own
                // scene falling apart.

                BondKey solverBondKey(solverNode0, solverNode1);
                auto entry = m_solverBondsMap.find(solverBondKey);
                if (entry)
                {
                    const uint32_t solverBondIndex = entry->second;
                    auto& blastBondIndices = m_solverBondsData[solverBondIndex].blastBondIndices;
                    blastBondIndices.findAndReplaceWithLast(blastBondIndex);
                    if (blastBondIndices.empty())
                    {
                        // all bonds associated with this solver bond were removed, so let's remove solver bond

                        m_solverBondsData.replaceWithLast(solverBondIndex);
                        m_solver.replaceWithLast(solverBondIndex);
                        if (m_solver.getBondCount() > 0)
                        {
                            // update 'previously last' solver bond mapping
                            uint32_t node0, node1;
                            m_solver.getBondNodes(solverBondIndex, node0, node1);
                            m_solverBondsMap[BondKey(node0, node1)] = solverBondIndex;
                        }

                        m_solverBondsMap.erase(solverBondKey);
                    }
                }

                CHECK_GRAPH_INTEGRITY;
            }

            // remove bond from graph processor's list
            m_blastBondIndexMap[blastBondIndex] = invalidIndex<uint32_t>();
            m_bondsData.replaceWithLast(bondIndex);
            m_blastBondIndexMap[m_bondsData[bondIndex].blastBondIndex] = m_bondsData.size() > bondIndex ? bondIndex : invalidIndex<uint32_t>();
        }
    }

    void setGraphReductionLevel(uint32_t level)
    {
        m_graphReductionLevel = level;
        m_nodesDirty = true;
    }

    uint32_t getGraphReductionLevel() const
    {
        return m_graphReductionLevel;
    }

    void setSkipStableUnconverged(bool enabled)
    {
        m_solver.setSkipStableUnconverged(enabled);
    }

    void solve(const ExtStressSolverSettings& settings, const float* bondHealth, const NvBlastBond* bonds, bool warmStart = true, bool islandAware = false, bool skipSettled = false, float deltaTime = 0.0f)
    {
        sync(bonds, islandAware);

        for (const NodeData& node : m_nodesData)
        {
            m_solver.setNodeVelocities(node.solverNode, node.localVel, NvVec3(NvZero));
        }

        m_solver.solve(settings.maxSolverIterationsPerFrame, warmStart, islandAware, skipSettled);

        resetVelocities();

        updateBondStress(bondHealth, bonds);

        // Per-chunk stress runs after the bonds because it consumes the same
        // solved impulses. Skipped entirely when no material enables crush.
        updateNodeStress(bondHealth, deltaTime);
    }

    bool calcError(float& linear, float& angular) const
    {
        return m_solver.calcError(linear, angular);
    }

    bool getBondStress(uint32_t blastBondIndex, float& compression, float& tension, float& shear) const
    {
        const uint32_t bondIndex = m_blastBondIndexMap[blastBondIndex];
        if (isInvalidIndex(bondIndex))
        {
            return false;
        }

        // compression and tension are mutually exclusive since they operate in opposite directions
        // they both measure stress parallel to the bond normal direction
        // compression is the force resisting two nodes being pushed together (it pushes them apart)
        // tension is the force resisting two nodes being pulled apart (it pulls them together)
        if (m_bondsData[bondIndex].stressNormal <= 0.0f)
        {
            compression = -m_bondsData[bondIndex].stressNormal;
            tension = 0.0f;
        }
        else
        {
            compression = 0.0f;
            tension = m_bondsData[bondIndex].stressNormal;
        }

        // shear is independent and can co-exist with compression and tension
        shear = m_bondsData[bondIndex].stressShear;         // the force perpendicular to the bond normal direction

        return true;
    }

    // Convert from Blast bond index to internal stress solver bond index
    // Will be InvalidIndex if the internal bond was removed from the stress solver
    uint32_t getInternalBondIndex(uint32_t blastBondIndex)
    {
        return m_blastBondIndexMap[blastBondIndex];
    }

private:

    void resetVelocities()
    {
        for (auto& node : m_nodesData)
        {
            node.localVel = NvVec3(NvZero);
        }
    }

    void resetCrushState()
    {
        for (auto& node : m_nodesData)
        {
            memset(node.virial, 0, sizeof(node.virial));
            node.strainRate = 0.0f;
            node.crushDamage = 0.0f;
            node.pressure = 0.0f;
            node.deviator = 0.0f;
            node.crushUtilisation = 0.0f;
            node.crushed = false;
            node.crushReported = false;
            node.crushCommandIssued = false;
        }
        m_crushedNodes.clear();
    }

    void clearVirials()
    {
        for (auto& node : m_nodesData)
        {
            memset(node.virial, 0, sizeof(node.virial));
        }
    }

    /**
    Add the symmetric part of (branch (outer) force) to a node's virial.
    Only the symmetric part is meaningful: the antisymmetric part of the sum is
    the net couple on the chunk, which a Cauchy stress tensor does not carry.
    */
    void accumulateVirial(uint32_t node, const NvVec3& branch, const NvVec3& force)
    {
        float* v = m_nodesData[node].virial;
        v[0] += branch.x * force.x;
        v[1] += branch.y * force.y;
        v[2] += branch.z * force.z;
        v[3] += 0.5f * (branch.x * force.y + branch.y * force.x);
        v[4] += 0.5f * (branch.x * force.z + branch.z * force.x);
        v[5] += 0.5f * (branch.y * force.z + branch.z * force.y);
    }

    /**
    Build each crush-enabled node's Cauchy stress tensor from the solved bond
    forces plus any external contacts recorded this frame, reduce it to the
    (p, q) invariants, and integrate crush damage.

    The construction is the Love-Weber (virial) sum, the standard definition of
    stress in a discrete assembly:

        sigma = (1/V) * sum over contacts of sym( branch (outer) force )

    Bond forces are attributed to member bonds by area share, because graph
    reduction can merge several asset bonds into one solver constraint that
    carries a single impulse.

    Body forces (gravity, centrifugal) are deliberately absent: they are not
    surface tractions and do not belong in the sum. A chunk buried in a pile is
    still crushed by the weight above it, because that weight reaches it
    through its bonds and contacts.
    */
    void updateNodeStress(const float* bondHealth, float deltaTime)
    {
        if (!m_crushEnabled)
        {
            return;
        }

        accumulateBondVirial(bondHealth);

        for (uint32_t node = 0; node < m_nodesData.size(); ++node)
        {
            NodeData& data = m_nodesData[node];
            const ExtStressCrushProperties& crush = materialForNode(node).crush;

            if (!crush.enabled() || data.crushed || data.volume <= 0.0f || data.mass <= 0.0f)
            {
                data.pressure = 0.0f;
                data.deviator = 0.0f;
                data.crushUtilisation = 0.0f;
                continue;
            }

            const float recipVolume = 1.0f / data.volume;
            const float sxx = data.virial[0] * recipVolume;
            const float syy = data.virial[1] * recipVolume;
            const float szz = data.virial[2] * recipVolume;
            const float sxy = data.virial[3] * recipVolume;
            const float sxz = data.virial[4] * recipVolume;
            const float syz = data.virial[5] * recipVolume;

            // p is positive in compression, so it carries the opposite sign to
            // the trace of the stress tensor.
            const float pressure = -(sxx + syy + szz) / 3.0f;

            // Deviator s = sigma + p*I, then q = sqrt(1.5 * s:s).
            const float dxx = sxx + pressure;
            const float dyy = syy + pressure;
            const float dzz = szz + pressure;
            const float deviatorSq =
                dxx * dxx + dyy * dyy + dzz * dzz + 2.0f * (sxy * sxy + sxz * sxz + syz * syz);
            const float deviator = sqrtf(1.5f * (deviatorSq > 0.0f ? deviatorSq : 0.0f));

            // A jammed debris pile can feed the solver forces large enough to
            // overflow the virial into inf/nan (measured on a 20k-chunk city:
            // one degenerate island reported q = inf, which crushed everything
            // it touched and poisoned every peak statistic downstream).
            // A non-finite stress state is not "very crushed", it is "not a
            // number": treat the tick as unreadable and do nothing.
            if (!std::isfinite(pressure) || !std::isfinite(deviator))
            {
                data.pressure = 0.0f;
                data.deviator = 0.0f;
                data.crushUtilisation = 0.0f;
                continue;
            }

            data.pressure = pressure;
            data.deviator = deviator;

            // TENSION CUTOFF. Comminution is a compressive phenomenon: a chunk
            // in net tension fails by cracking, which is exactly what the bond
            // model already represents, and it does not turn to powder. Without
            // this the Drucker-Prager cone would make crushing EASIER in
            // tension (its limit falls with pressure), so free-floating debris
            // -- which has no confining pressure at all -- would crumble
            // instead of tumbling. That is the failure mode this whole model
            // exists to avoid.
            if (pressure <= 0.0f)
            {
                data.crushUtilisation = 0.0f;
                continue;
            }

            // Utilisation is a property of the STRESS STATE alone, so compute
            // it whether or not anything is closing on the chunk. That is what
            // makes it usable the way bond utilisation is: sample it after a
            // gravity settle to see how much of a chunk's crush capacity its
            // own structure already consumes, before anything moves.
            {
                const float coneLimitNow = crush.cohesion + crush.frictionSlope * pressure;
                const float coneUse = coneLimitNow > 0.0f ? deviator / coneLimitNow : 0.0f;
                const float capUse =
                    crush.capPressure > 0.0f ? pressure / crush.capPressure : 0.0f;
                data.crushUtilisation = coneUse > capUse ? coneUse : capUse;
            }

            if (deltaTime <= 0.0f)
            {
                continue;
            }
            const float strainRate = data.strainRate;

            // Optional CEB-style dynamic increase factor: concrete is stronger
            // the faster it is loaded, which is why a fast projectile spalls
            // where a slow press crushes.
            float strengthScale = 1.0f;
            if (crush.strainRateExponent > 0.0f && crush.referenceStrainRate > 0.0f)
            {
                strengthScale = powf(strainRate / crush.referenceStrainRate, crush.strainRateExponent);
                if (strengthScale < 1.0f)
                {
                    strengthScale = 1.0f;   // rate hardening only, never softening
                }
            }

            // Drucker-Prager cone with a pressure cap. Using a cap surface
            // rather than a bare pressure threshold is what distinguishes
            // CONFINED crushing from unconfined shear: unconfined debris sits
            // at low p, stays inside the cone, and tumbles intact.
            const float coneLimit = strengthScale * (crush.cohesion + crush.frictionSlope * pressure);
            const float capLimit = strengthScale * crush.capPressure;
            const float shearExcess = deviator - coneLimit;
            const float capExcess = pressure - capLimit;
            const float excess = shearExcess > capExcess ? shearExcess : capExcess;

            if (excess <= 0.0f)
            {
                continue;
            }

            // Perzyna overstress flow: the plastic strain rate is how far
            // outside the yield surface the stress sits, divided by the
            // material's viscosity. Damage is that plastic work per unit
            // volume, normalized by the specific comminution energy.
            //
            //     epsdot_p = excess / crushViscosity
            //     dD       = excess * epsdot_p * dt / crushEnergy
            //
            // Quadratic in overstress, which is what keeps a barely-yielding
            // chunk intact for a long time while a hard hit comminutes almost
            // at once. Crucially it needs no strain measurement, so a chunk
            // loaded purely through its BONDS -- buried in a collapse, never
            // touched directly -- crushes exactly as a struck one does.
            const float crushEnergy = crush.crushEnergy > 0.0f ? crush.crushEnergy : 1.0f;
            const float crushViscosity = crush.crushViscosity > 0.0f ? crush.crushViscosity : 1.0f;
            data.crushDamage += excess * excess * deltaTime / (crushViscosity * crushEnergy);

            if (data.crushDamage >= 1.0f)
            {
                data.crushDamage = 1.0f;
                data.crushed = true;
                m_crushedNodes.pushBack(node);
            }
        }
    }

    /**
    Distribute each solver bond's solved force onto its member bonds by area
    share and add the result to both endpoint nodes' virials.

    Sign convention: the solver's linear impulse acts on node0 as given and on
    node1 negated. Combined with the branch vectors this makes a bond in
    compression produce a positive pressure at both of its nodes, matching
    getBondStress's convention that a negative stressNormal is compression.
    */
    void accumulateBondVirial(const float* bondHealth)
    {
        clearVirials();

        for (uint32_t group = 0; group < m_solverBondsData.size(); ++group)
        {
            const auto& blastBondIndices = m_solverBondsData[group].blastBondIndices;

            float totalArea = 0.0f;
            for (auto blastBondIndex : blastBondIndices)
            {
                const float remainingArea = bondHealth[blastBondIndex];
                if (remainingArea > 0.0f && canTakeDamage(remainingArea))
                {
                    totalArea += remainingArea;
                }
            }
            if (totalArea <= 0.0f)
            {
                continue;
            }

            NvVec3 impulseLinear, impulseAngular;
            getSolverInternalBondImpulses(group, impulseLinear, impulseAngular);
            // impulseAngular is intentionally unused: a self-equilibrated couple
            // contributes only an antisymmetric term, which vanishes when the
            // Cauchy stress is symmetrized. Bending-driven edge spall is
            // therefore not represented in the mean stress.
            (void)impulseAngular;

            const float recipTotalArea = 1.0f / totalArea;
            for (auto blastBondIndex : blastBondIndices)
            {
                const float remainingArea = bondHealth[blastBondIndex];
                if (remainingArea <= 0.0f || !canTakeDamage(remainingArea))
                {
                    continue;
                }
                const uint32_t bondIndex = m_blastBondIndexMap[blastBondIndex];
                if (isInvalidIndex(bondIndex))
                {
                    continue;
                }
                const BondData& bond = m_bondsData[bondIndex];
                const NvVec3 force = impulseLinear * (remainingArea * recipTotalArea);

                if (materialForNode(bond.node0).crush.enabled())
                {
                    accumulateVirial(bond.node0, bond.centroid - m_nodesData[bond.node0].localPos, force);
                }
                if (materialForNode(bond.node1).crush.enabled())
                {
                    accumulateVirial(bond.node1, bond.centroid - m_nodesData[bond.node1].localPos, -force);
                }
            }
        }
    }

    void updateBondStress(const float* bondHealth, const NvBlastBond* bonds)
    {
        m_overstressedBondCount = 0;
        // E1: reset alongside the count so mask and count always describe the
        // same update. resize is a no-op after the first call.
        m_nodeOverstressed.resize(m_nodesData.size());
        if (!m_nodeOverstressed.empty())
        {
            memset(m_nodeOverstressed.begin(), 0, m_nodeOverstressed.size());
        }

        // Reuse a persistent scratch buffer instead of allocating one every solve.
        // NsArray::clear() keeps capacity, so reserve() only allocates on the first call
        // (or when the bond count grows); this loop runs every frame the solver is active.
        Array<uint32_t>::type& bondIndicesToRemove = m_bondIndicesToRemove;
        bondIndicesToRemove.clear();
        bondIndicesToRemove.reserve(getBondCount());
        for (uint32_t i = 0; i < m_solverBondsData.size(); ++i)
        {
            // calculate the total area of all bonds involved so pressure can be calculated
            float totalArea = 0.0f;
            // calculate an average normal and centroid for all bonds as well, weighted by their area
            nvidia::NvVec3 bondNormal(NvZero);
            nvidia::NvVec3 bondCentroid(NvZero);
            nvidia::NvVec3 averageNodeDisp(NvZero);
            const auto& blastBondIndices = m_solverBondsData[i].blastBondIndices;
            for (auto blastBondIndex : blastBondIndices)
            {
                if (bondHealth[blastBondIndex] > 0.0f)
                {
                    const uint32_t bondIndex = m_blastBondIndexMap[blastBondIndex];
                    const BondData& bond = m_bondsData[bondIndex];
                    const nvidia::NvVec3 nodeDisp = m_nodesData[bond.node1].localPos - m_nodesData[bond.node0].localPos;

                    // the current health of a bond is the effective area remaining
                    const float remainingArea = bondHealth[blastBondIndex];
                    const NvBlastBond& blastBond = bonds[blastBondIndex];

                    // Align normal(s) with node displacement, so that compressive/tensile distinction is correct
                    const nvidia::NvVec3 assetBondNormal(blastBond.normal[0], blastBond.normal[1], blastBond.normal[2]);
                    const nvidia::NvVec3 blastBondNormal = std::copysign(1.0f, assetBondNormal.dot(nodeDisp))*assetBondNormal;

                    const nvidia::NvVec3 blastBondCentroid(blastBond.centroid[0], blastBond.centroid[1], blastBond.centroid[2]);

                    if (!canTakeDamage(remainingArea))  // Check unbreakable limit
                    {
                        totalArea = kUnbreakableLimit;  // Don't add this in, in case of overflow
                        bondNormal = blastBondNormal;
                        bondCentroid = blastBondCentroid;
                        averageNodeDisp = nodeDisp;
                        break;
                    }

                    bondNormal += blastBondNormal*remainingArea;
                    bondCentroid += blastBondCentroid*remainingArea;
                    averageNodeDisp += nodeDisp*remainingArea;

                    totalArea += remainingArea;
                }
                else
                {
                    // if the bond is broken, try to remove it after processing is complete
                    bondIndicesToRemove.pushBack(blastBondIndex);
                }
            }

            if (totalArea == 0.0f)
            {
                continue;
            }

            // normalized the aggregate normal now that all contributing bonds have been combined
            bondNormal.normalizeSafe();

            // divide by total area for the weighted position, if the area is valid
            if (canTakeDamage(totalArea))
            {
                bondCentroid /= totalArea;
                averageNodeDisp /= totalArea;
            }

            // The stress in a merged solver bond is shared by all member bonds
            // (one constraint, one sigma over the summed area) — but each
            // member fails against its OWN material limits, so overstress is
            // counted per member rather than per group.
            float stressNormal, stressShear;
            calcSolverBondStresses(i, totalArea, averageNodeDisp.magnitude(), bondNormal, stressNormal, stressShear);
            NVBLAST_ASSERT(!std::isnan(stressNormal) && !std::isnan(stressShear));

            // store the stress values for all the bonds involved
            for (auto blastBondIndex : blastBondIndices)
            {
                const uint32_t bondIndex = m_blastBondIndexMap[blastBondIndex];
                if (!isInvalidIndex(bondIndex) && bondHealth[blastBondIndex] > 0.0f)
                {
                    const ExtStressMaterial& material = materialForBlastBond(blastBondIndex);
                    BondData& bond = m_bondsData[bondIndex];
                    if (-stressNormal > material.compressionElasticLimit
                        || stressNormal > material.tensionElasticLimit
                        || stressShear > material.shearElasticLimit)
                    {
                        ++m_overstressedBondCount;
                        // E1: exactly the condition generateStressDamage keys
                        // its command on, evaluated on the same stored floats
                        // — so the mask is neither wider nor narrower than
                        // the set of bonds that will produce commands.
                        m_nodeOverstressed[bond.node0] = 1;
                        m_nodeOverstressed[bond.node1] = 1;
                    }

                    NVBLAST_ASSERT(getNodeData(bond.node0).solverNode != getNodeData(bond.node1).solverNode);
                    NVBLAST_ASSERT(bond.blastBondIndex == blastBondIndex);

                    bond.stressNormal = stressNormal;
                    bond.stressShear = stressShear;

                    // store the normal used to calc stresses so it can be used later to determine forces
                    bond.normal = bondNormal;

                    // store the bond centroid
                    bond.centroid = bondCentroid;
                }
            }
        }

        // now that processing is done, remove any dead bonds
        for (uint32_t bondIndex : bondIndicesToRemove)
        {
            removeBondIfExists(bondIndex);
        }
    }

    void sync(const NvBlastBond* bonds, bool islandAware)
    {
        const bool resynced = m_nodesDirty || m_bondsDirty;
        if (m_nodesDirty)
        {
            syncNodes(bonds);
            m_solver.initialize();
        }
        if (m_bondsDirty)
        {
            syncBonds(bonds);
        }
        // Island partitioning only feeds the islandCount stat — the solve never reads it — so it is
        // only computed when island-aware solving is enabled. This keeps flag-off truly zero-cost
        // (no per-resync union-find when island solving is off).
        if (resynced && islandAware)
        {
            computeIslands();
        }

        CHECK_GRAPH_INTEGRITY;
    }

    // ── Island partitioning ──
    // Connected components of the solver graph via union-find over the solver
    // bonds. Static (mass<=0) nodes carry no coupling (sqrt_I_inv == 0), so they
    // are CUT POINTS: two structures sharing only a static/world node are
    // physically independent islands. Recomputed only on topology change.
    // m_nodeIsland[solverNode] holds the compacted island id (InvalidIndex for
    // static nodes); m_islandCount is the number of dynamic islands. This is the
    // foundation for solving each island independently and skipping settled ones.
    uint32_t islandFind(uint32_t x)
    {
        uint32_t r = x;
        while (m_islandParent[r] != r) r = m_islandParent[r];
        while (m_islandParent[x] != r) { const uint32_t nx = m_islandParent[x]; m_islandParent[x] = r; x = nx; }
        return r;
    }

    void computeIslands()
    {
        const uint32_t kInvalid = static_cast<uint32_t>(-1);
        const uint32_t nodeCount = getSolverNodeCount();
        m_islandParent.resize(nodeCount);
        for (uint32_t i = 0; i < nodeCount; ++i) m_islandParent[i] = i;

        const uint32_t bondCount = getSolverBondCount();
        for (uint32_t b = 0; b < bondCount; ++b)
        {
            uint32_t n0, n1;
            getSolverInternalBondNodes(b, n0, n1);
            if (n0 >= nodeCount || n1 >= nodeCount) continue;
            // static node = cut point, do not union through it
            if (m_solverNodesData[n0].mass <= 0.0f || m_solverNodesData[n1].mass <= 0.0f) continue;
            const uint32_t ra = islandFind(n0), rb = islandFind(n1);
            if (ra != rb) m_islandParent[ra] = rb;
        }

        // Compact island ids over dynamic nodes; static nodes get InvalidIndex.
        m_nodeIsland.resize(nodeCount);
        m_islandRootId.resize(nodeCount);
        for (uint32_t i = 0; i < nodeCount; ++i) m_islandRootId[i] = kInvalid;
        m_islandCount = 0;
        for (uint32_t i = 0; i < nodeCount; ++i)
        {
            if (m_solverNodesData[i].mass <= 0.0f) { m_nodeIsland[i] = kInvalid; continue; }
            const uint32_t r = islandFind(i);
            if (m_islandRootId[r] == kInvalid) m_islandRootId[r] = m_islandCount++;
            m_nodeIsland[i] = m_islandRootId[r];
        }
    }

    void syncNodes(const NvBlastBond* bonds)
    {
        // init with 1<->1 blast nodes to solver nodes mapping
        m_solverNodesData.resize(m_nodesData.size());
        for (uint32_t i = 0; i < m_nodesData.size(); ++i)
        {
            m_nodesData[i].solverNode = i;
            m_solverNodesData[i].supportNodesCount = 1;
            m_solverNodesData[i].indexShift = 0;
        }

        // for static nodes aggregate size per graph reduction level is lower, it
        // falls behind on few levels. (can be made as parameter)
        const uint32_t STATIC_NODES_COUNT_PENALTY = 2 << 2;

        // reducing graph by aggregating nodes level by level
        // NOTE (@anovoselov):  Recently, I found a flow in the algorithm below. In very rare situations aggregate (solver node)
        // can contain more then one connected component. I didn't notice it to produce any visual artifacts and it's
        // unlikely to influence stress solvement a lot. Possible solution is to merge *whole* solver nodes, that
        // will raise complexity a bit (at least will add another loop on nodes for every reduction level.
        for (uint32_t k = 0; k < m_graphReductionLevel; k++)
        {
            const uint32_t maxAggregateSize = 1 << (k + 1);

            for (const BondData& bond : m_bondsData)
            {
                NodeData& node0 = m_nodesData[bond.node0];
                NodeData& node1 = m_nodesData[bond.node1];

                if (node0.solverNode == node1.solverNode)
                    continue;

                SolverNodeData& solverNode0 = m_solverNodesData[node0.solverNode];
                SolverNodeData& solverNode1 = m_solverNodesData[node1.solverNode];

                const int countPenalty = 1;   // This was being set to STATIC_NODES_COUNT_PENALTY for static nodes, may want to revisit
                const uint32_t aggregateSize = std::min<uint32_t>(maxAggregateSize, node0.neighborsCount / 2);

                if (solverNode0.supportNodesCount * countPenalty >= aggregateSize)
                    continue;
                if (solverNode1.supportNodesCount * countPenalty >= aggregateSize)
                    continue;

                if (solverNode0.supportNodesCount >= solverNode1.supportNodesCount)
                {
                    solverNode1.supportNodesCount--;
                    solverNode0.supportNodesCount++;
                    node1.solverNode = node0.solverNode;
                }
                else if (solverNode1.supportNodesCount >= solverNode0.supportNodesCount)
                {
                    solverNode1.supportNodesCount++;
                    solverNode0.supportNodesCount--;
                    node0.solverNode = node1.solverNode;
                }
            }
        }

        // Solver Nodes now sparse, a lot of empty ones. Rearrange them by moving all non-empty to the front
        // 2 passes used for that
        {
            uint32_t currentNode = 0;
            for (; currentNode < m_solverNodesData.size(); ++currentNode)
            {
                if (m_solverNodesData[currentNode].supportNodesCount > 0)
                    continue;

                // 'currentNode' is free

                // search next occupied node
                uint32_t k = currentNode + 1;
                for (; k < m_solverNodesData.size(); ++k)
                {
                    if (m_solverNodesData[k].supportNodesCount > 0)
                    {
                        // replace currentNode and keep indexShift
                        m_solverNodesData[currentNode].supportNodesCount = m_solverNodesData[k].supportNodesCount;
                        m_solverNodesData[k].indexShift = k - currentNode;
                        m_solverNodesData[k].supportNodesCount = 0;
                        break;
                    }
                }

                if (k == m_solverNodesData.size())
                {
                    break;
                }
            }
            for (auto& node : m_nodesData)
            {
                node.solverNode -= m_solverNodesData[node.solverNode].indexShift;
            }

            // now, we know total solver nodes count and which nodes are aggregated into them
            m_solverNodesData.resize(currentNode);
        }


        // calculate all needed data
        for (SolverNodeData& solverNode : m_solverNodesData)
        {
            solverNode.supportNodesCount = 0;
            solverNode.localPos = NvVec3(NvZero);
            solverNode.mass = 0.0f;
            solverNode.volume = 0.0f;
        }

        for (NodeData& node : m_nodesData)
        {
            SolverNodeData& solverNode = m_solverNodesData[node.solverNode];
            solverNode.supportNodesCount++;
            solverNode.localPos += node.localPos;
            solverNode.mass += node.mass;
            solverNode.volume += node.volume;
        }

        for (SolverNodeData& solverNode : m_solverNodesData)
        {
            solverNode.localPos /= (float)solverNode.supportNodesCount;
        }

        m_solver.reset(m_solverNodesData.size());
        for (uint32_t nodeIndex = 0; nodeIndex < m_solverNodesData.size(); ++nodeIndex)
        {
            const SolverNodeData& solverNode = m_solverNodesData[nodeIndex];

            const float R = NvPow(solverNode.volume * 3.0f * NvInvPi / 4.0f, 1.0f / 3.0f); // sphere volume approximation
            const float inertia = solverNode.mass * (R * R * 0.4f); // sphere inertia tensor approximation: I = 2/5 * M * R^2 ; invI = 1 / I;
            m_solver.setNodeMassInfo(nodeIndex, solverNode.localPos, solverNode.mass, inertia);
        }

        m_nodesDirty = false;

        syncBonds(bonds);
    }

    void syncBonds(const NvBlastBond* bonds)
    {
        // traverse all blast bonds and aggregate
        m_solver.clearBonds();
        m_solverBondsMap.clear();
        m_solverBondsData.clear();
        for (BondData& bond : m_bondsData)
        {
            const NodeData& node0 = m_nodesData[bond.node0];
            const NodeData& node1 = m_nodesData[bond.node1];

            // reset stress, bond structure changed and internal bonds stress won't be updated during updateBondStress()
            bond.stressNormal = 0.0f;
            bond.stressShear = 0.0f;

            // initialize normal and centroid using blast values
            bond.normal = *(NvVec3*)bonds[bond.blastBondIndex].normal;
            bond.centroid = *(NvVec3*)bonds[bond.blastBondIndex].centroid;

            // fix normal direction to point from node0 to node1
            bond.normal *= std::copysign(1.0f, bond.normal.dot(node1.localPos - node0.localPos));

            if (node0.solverNode == node1.solverNode)
                continue; // skip (internal)

            BondKey key(node0.solverNode, node1.solverNode);
            auto entry = m_solverBondsMap.find(key);
            SolverBondData* data;
            const float bondArea = bonds[bond.blastBondIndex].area;
            if (!entry)
            {
                m_solverBondsData.pushBack(SolverBondData());
                data = &m_solverBondsData.back();
                m_solverBondsMap[key] = m_solverBondsData.size() - 1;

                const uint32_t bondMaterial =
                    m_bondMaterials ? m_bondMaterials[bond.blastBondIndex] : 0;
                m_solver.addBond(
                    node0.solverNode, node1.solverNode, bond.centroid, bondArea, bondMaterial);
            }
            else
            {
                data = &m_solverBondsData[entry->second];
                m_solver.addBondArea(entry->second, bondArea);
            }
            data->blastBondIndices.pushBack(bond.blastBondIndex);
        }
        // Graph reduction can merge bonds of different materials into one
        // solver bond; settle each group's representative to its weakest
        // member (feeds only the GPU-side damage seed).
        refreshGroupMaterials();

        m_bondsDirty = false;
    }

#if GRAPH_INTERGRIRY_CHECK
    void checkGraphIntegrity()
    {
        NVBLAST_ASSERT(m_solver.getBondCount() == m_solverBondsData.size());
        NVBLAST_ASSERT(m_solver.getNodeCount() == m_solverNodesData.size());

        std::set<uint64_t> solverBonds;
        for (uint32_t i = 0; i < m_solverBondsData.size(); ++i)
        {
            const auto& bondData = m_solver.getBondData(i);
            BondKey key(bondData.node0, bondData.node1);
            NVBLAST_ASSERT(solverBonds.find(key) == solverBonds.end());
            solverBonds.emplace(key);
            auto entry = m_solverBondsMap.find(key);
            NVBLAST_ASSERT(entry != nullptr);
            const auto& solverBond = m_solverBondsData[entry->second];
            for (auto& blastBondIndex : solverBond.blastBondIndices)
            {
                if (!isInvalidIndex(m_blastBondIndexMap[blastBondIndex]))
                {
                    auto& b = m_bondsData[m_blastBondIndexMap[blastBondIndex]];
                    BondKey key2(m_nodesData[b.node0].solverNode, m_nodesData[b.node1].solverNode);
                    NVBLAST_ASSERT(key2 == key);
                }
            }
        }

        for (auto& solverBond : m_solverBondsData)
        {
            for (auto& blastBondIndex : solverBond.blastBondIndices)
            {
                if (!isInvalidIndex(m_blastBondIndexMap[blastBondIndex]))
                {
                    auto& b = m_bondsData[m_blastBondIndexMap[blastBondIndex]];
                    NVBLAST_ASSERT(m_nodesData[b.node0].solverNode != m_nodesData[b.node1].solverNode);
                }
            }
        }
        uint32_t mappedBondCount = 0;
        for (uint32_t i = 0; i < m_blastBondIndexMap.size(); i++)
        {
            const auto& bondIndex = m_blastBondIndexMap[i];
            if (!isInvalidIndex(bondIndex))
            {
                mappedBondCount++;
                NVBLAST_ASSERT(m_bondsData[bondIndex].blastBondIndex == i);
            }
        }
        NVBLAST_ASSERT(m_bondsData.size() == mappedBondCount);
    }
#endif

    struct BondKey
    {
        uint32_t node0;
        uint32_t node1;

        BondKey(uint32_t n0, uint32_t n1) : node0(n0), node1(n1) {}

        operator uint64_t() const
        {
            // Szudzik's function
            return node0 >= node1 ? (uint64_t)node0 * node0 + node0 + node1 : (uint64_t)node1 * node1 + node0;
        }
    };

    ConjugateGradientImpulseSolver      m_solver;
    Array<SolverNodeData>::type         m_solverNodesData;
    Array<SolverBondData>::type         m_solverBondsData;

    uint32_t                            m_graphReductionLevel;

    bool                                m_nodesDirty;
    bool                                m_bondsDirty;

    uint32_t                            m_overstressedBondCount;

    /// E1: nodes incident to at least one overstressed bond, refreshed beside
    /// m_overstressedBondCount in updateBondStress. Lets fracture-command
    /// generation visit only the neighborhoods that can produce a command
    /// instead of walking every bond of every actor (~230k visits to find
    /// ~200 overstressed). Indexed by asset-graph node id — the same space
    /// fillFractureCommands walks (addBond() is fed m_graph node indices).
    Array<uint8_t>::type                m_nodeOverstressed;

    // Island partitioning (connected components of the solver graph)
    Array<uint32_t>::type               m_islandParent;
    Array<uint32_t>::type               m_islandRootId;
    Array<uint32_t>::type               m_nodeIsland;
    uint32_t                            m_islandCount;

    HashMap<BondKey, uint32_t>::type    m_solverBondsMap;
    Array<uint32_t>::type               m_blastBondIndexMap;

    Array<BondData>::type               m_bondsData;
    Array<NodeData>::type               m_nodesData;

    // Persistent scratch for updateBondStress() so it doesn't heap-allocate per solve.
    Array<uint32_t>::type               m_bondIndicesToRemove;

    // Material table + per-asset-bond material indices, owned by
    // ExtStressSolverImpl (stable storage; re-pointed whenever they change).
    // Null until set: materialForBlastBond falls back to a default material so
    // low-level use without a table keeps the historical 1.0/2.0 Pa behavior.
    const ExtStressMaterial*            m_materials{nullptr};
    uint32_t                            m_materialCount{0};
    const uint32_t*                     m_bondMaterials{nullptr};
    // Per-graph-node material indices, selecting each chunk's crush properties.
    const uint32_t*                     m_nodeMaterials{nullptr};
    // True only when some material enables crush AND the graph is unreduced.
    bool                                m_crushEnabled{false};
    Array<uint32_t>::type               m_crushedNodes;

public:
    void setMaterialTables(
        const ExtStressMaterial* materials,
        uint32_t materialCount,
        const uint32_t* bondMaterials,
        const uint32_t* nodeMaterials,
        bool crushEnabled)
    {
        m_materials = materials;
        m_materialCount = materialCount;
        m_bondMaterials = bondMaterials;
        m_nodeMaterials = nodeMaterials;
        m_crushEnabled = crushEnabled;
        m_solver.setMaterialTable(materials, materialCount);
    }

    const ExtStressMaterial& materialForNode(uint32_t node) const
    {
        static const ExtStressMaterial defaultMaterial;
        if (!m_materials || m_materialCount == 0)
        {
            return defaultMaterial;
        }
        const uint32_t material = m_nodeMaterials ? m_nodeMaterials[node] : 0;
        return m_materials[material < m_materialCount ? material : 0];
    }

    bool isCrushEnabled() const { return m_crushEnabled; }

    void setNodeStrainRate(uint32_t node, float strainRate)
    {
        m_nodesData[node].strainRate = strainRate > 0.0f ? strainRate : 0.0f;
    }

    float getNodeCrushDamage(uint32_t node) const { return m_nodesData[node].crushDamage; }
    float getNodePressure(uint32_t node) const { return m_nodesData[node].pressure; }
    float getNodeDeviator(uint32_t node) const { return m_nodesData[node].deviator; }
    float getNodeCrushUtilisation(uint32_t node) const { return m_nodesData[node].crushUtilisation; }
    bool  isNodeCrushed(uint32_t node) const { return m_nodesData[node].crushed; }
    bool  needsCrushCommand(uint32_t node) const
    {
        return m_nodesData[node].crushed && !m_nodesData[node].crushCommandIssued;
    }
    void  markCrushCommandIssued(uint32_t node) { m_nodesData[node].crushCommandIssued = true; }
    uint32_t getPendingCrushCount() const
    {
        uint32_t pending = 0;
        for (const auto& node : m_nodesData)
        {
            if (node.crushed && !node.crushCommandIssued) ++pending;
        }
        return pending;
    }

    /**
    Drain the crushed-node queue. Each node is reported exactly once; the
    latched `crushed` flag keeps it out of the stress loop forever after.
    */
    uint32_t drainCrushedNodes(uint32_t* nodeIndices, uint32_t capacity)
    {
        const uint32_t count = m_crushedNodes.size() < capacity ? m_crushedNodes.size() : capacity;
        for (uint32_t i = 0; i < count; ++i)
        {
            nodeIndices[i] = m_crushedNodes[i];
            m_nodesData[m_crushedNodes[i]].crushReported = true;
        }
        // Keep anything that did not fit; it drains on the next call.
        if (count == m_crushedNodes.size())
        {
            m_crushedNodes.clear();
        }
        else
        {
            const uint32_t remaining = m_crushedNodes.size() - count;
            for (uint32_t i = 0; i < remaining; ++i)
            {
                m_crushedNodes[i] = m_crushedNodes[i + count];
            }
            m_crushedNodes.resize(remaining);
        }
        return count;
    }

    const ExtStressMaterial& materialForBlastBond(uint32_t blastBondIndex) const
    {
        static const ExtStressMaterial defaultMaterial = []() {
            ExtStressMaterial resolved;
            resolved.tensionElasticLimit = resolved.compressionElasticLimit;
            resolved.tensionFatalLimit = resolved.compressionFatalLimit;
            resolved.shearElasticLimit = resolved.compressionElasticLimit;
            resolved.shearFatalLimit = resolved.compressionFatalLimit;
            return resolved;
        }();
        if (!m_materials || m_materialCount == 0)
        {
            return defaultMaterial;
        }
        const uint32_t material =
            m_bondMaterials ? m_bondMaterials[blastBondIndex] : 0;
        return m_materials[material < m_materialCount ? material : 0];
    }

    // Re-derive each solver bond group's material as its WEAKEST member's
    // (lowest resolved tension elastic limit). Consumed only by the GPU-side
    // damage kernel seed; the CPU damage path reads each member's own material.
    void refreshGroupMaterials()
    {
        for (uint32_t group = 0; group < m_solverBondsData.size(); ++group)
        {
            uint32_t weakest = 0;
            float weakestLimit = FLT_MAX;
            for (const uint32_t blastBondIndex : m_solverBondsData[group].blastBondIndices)
            {
                const uint32_t material =
                    m_bondMaterials ? m_bondMaterials[blastBondIndex] : 0;
                const ExtStressMaterial& resolved = materialForBlastBond(blastBondIndex);
                if (resolved.tensionElasticLimit < weakestLimit)
                {
                    weakestLimit = resolved.tensionElasticLimit;
                    weakest = material;
                }
            }
            m_solver.setBondMaterial(group, weakest);
        }
    }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                           ExtStressSolver
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
*/
class ExtStressSolverImpl final : public ExtStressSolver
{
    NV_NOCOPY(ExtStressSolverImpl)

public:
    ExtStressSolverImpl(const NvBlastFamily& family, const ExtStressSolverSettings& settings);
    virtual void                            release() override;


    //////// ExtStressSolverImpl interface ////////

    virtual void                            setAllNodesInfoFromLL(float density = 1.0f) override;

    virtual void                            setNodeInfo(uint32_t graphNode, float mass, float volume, NvcVec3 localPos) override;

    virtual void                            setSettings(const ExtStressSolverSettings& settings) override
    {
        m_settings = settings;
    }

    virtual const ExtStressSolverSettings&  getSettings() const override
    {
        return m_settings;
    }

    virtual bool                            addForce(const NvBlastActor& actor, NvcVec3 localPosition, NvcVec3 localForce, ExtForceMode::Enum mode) override;

    virtual void                            addForce(uint32_t graphNode, NvcVec3 localForce, ExtForceMode::Enum mode) override;
    virtual void                            addForceAt(uint32_t graphNode, NvcVec3 localPosition, NvcVec3 localForce, ExtForceMode::Enum mode) override;

    virtual bool                            addGravity(const NvBlastActor& actor, NvcVec3 localGravity) override;
    virtual bool                            addCentrifugalAcceleration(const NvBlastActor& actor, NvcVec3 localCenterMass, NvcVec3 localAngularVelocity) override;

    virtual void                            update() override;

    virtual uint32_t                        getOverstressedBondCount() const override
    {
        return m_graphProcessor->getOverstressedBondCount();
    }

    virtual uint32_t                        getIslandCount() const override
    {
        return m_graphProcessor->getIslandCount();
    }

    virtual void                            setIslandAware(bool enabled) override
    {
        m_islandAware = enabled;
    }

    virtual bool                            getIslandAware() const override
    {
        return m_islandAware;
    }

    virtual void                            setSkipSettled(bool enabled) override
    {
        m_skipSettled = enabled;
    }

    void setSkipStableUnconverged(bool enabled) override
    {
        m_graphProcessor->setSkipStableUnconverged(enabled);
    }

    virtual bool                            getSkipSettled() const override
    {
        return m_skipSettled;
    }

    virtual uint32_t                        getIslandsSkipped() const override
    {
        return m_graphProcessor->getIslandsSkipped();
    }

    virtual uint32_t                        getIslandsTotal() const override
    {
        return m_graphProcessor->getIslandsTotal();
    }

    virtual bool                            setGpuAccelerated(bool enabled) override
    {
        return m_graphProcessor->setGpuAccelerated(enabled);
    }

    virtual void                            setGpuCudaContext(void* cudaContext) override
    {
        m_graphProcessor->setGpuCudaContext(cudaContext);
    }

    virtual void                            setGpuMinimumBondCount(uint32_t bondCount) override
    {
        m_graphProcessor->setGpuMinimumBondCount(bondCount);
    }

    virtual bool                            getGpuAccelerated() const override
    {
        return m_graphProcessor->getGpuAccelerated();
    }

    virtual float                           getGpuSolveMilliseconds() const override
    {
        return m_graphProcessor->getGpuSolveMilliseconds();
    }

    virtual float                           getInitializeMilliseconds() const override
    {
        return m_initializeMilliseconds;
    }

    virtual float                           getGraphSolveMilliseconds() const override
    {
        return m_graphSolveMilliseconds;
    }

    virtual float                           getCalcErrorMilliseconds() const override
    {
        return m_calcErrorMilliseconds;
    }

    virtual float                           getGpuImpulseCopyMilliseconds() const override
    {
        return m_graphProcessor->getGpuImpulseCopyMilliseconds();
    }

    virtual uint32_t                        getGpuImpulseCopyCount() const override
    {
        return m_graphProcessor->getGpuImpulseCopyCount();
    }

    virtual float                           getGpuHostWorkMilliseconds() const override
    {
        return m_graphProcessor->getGpuHostWorkMilliseconds();
    }

    virtual float                           getGpuHostBlockedMilliseconds() const override
    {
        return m_graphProcessor->getGpuHostBlockedMilliseconds();
    }

    virtual uint64_t                        getGpuHostToDeviceBytes() const override
    {
        return m_graphProcessor->getGpuHostToDeviceBytes();
    }

    virtual uint64_t                        getGpuDeviceToHostBytes() const override
    {
        return m_graphProcessor->getGpuDeviceToHostBytes();
    }

    virtual void                            generateFractureCommands(const NvBlastActor& actor, NvBlastFractureBuffers& commands) override;
    virtual uint32_t                        generateFractureCommandsPerActor(const NvBlastActor** actorBuffer, NvBlastFractureBuffers* commandsBuffer, uint32_t bufferSize) override;


    virtual void                            reset() override
    {
        m_reset = true;
    }

    virtual float                           getStressErrorLinear() const override
    {
        return m_errorLinear;
    }

    virtual float                           getStressErrorAngular() const override
    {
        return m_errorAngular;
    }

    virtual bool                            converged() const override
    {
        return m_converged;
    }

    virtual uint32_t                        getFrameCount() const override
    {
        return m_framesCount;
    }

    virtual uint32_t                        getBondCount() const override
    {
        return m_graphProcessor->getSolverBondCount();
    }

    virtual bool                            getExcessForces(uint32_t actorIndex, const NvcVec3& com, NvcVec3& force, NvcVec3& torque) override;

    virtual uint32_t                        getBondStresses(float* compression, float* tension, float* shear, uint32_t capacity) const override
    {
        const uint32_t count = std::min(capacity, m_assetBondCount);
        for (uint32_t bondIndex = 0; bondIndex < count; ++bondIndex)
        {
            float bondCompression = 0.0f;
            float bondTension = 0.0f;
            float bondShear = 0.0f;
            // Broken bonds keep whatever stress they carried on the tick they failed;
            // report them as unloaded so utilisation reflects the live load path.
            if (m_bondHealths[bondIndex] > 0.0f)
            {
                m_graphProcessor->getBondStress(bondIndex, bondCompression, bondTension, bondShear);
            }
            if (compression) compression[bondIndex] = bondCompression;
            if (tension)     tension[bondIndex]     = bondTension;
            if (shear)       shear[bondIndex]       = bondShear;
        }
        return count;
    }

    virtual void                            setMaterials(const ExtStressMaterial* materials, uint32_t count) override
    {
        if (!materials || count == 0)
        {
            NVBLAST_LOG_ERROR("ExtStressSolver::setMaterials: at least one material is required");
            return;
        }
        m_materials.resize(count);
        for (uint32_t material = 0; material < count; ++material)
        {
            m_materials[material] = resolveMaterial(materials[material]);
        }
        // Storage may have reallocated; re-point the processor and refresh the
        // solver-bond group materials (weakness ordering may have changed).
        pushMaterialTables();
    }

    virtual void                            setBondMaterials(const uint32_t* materialIndices, uint32_t bondCount) override
    {
        const uint32_t materialCount = static_cast<uint32_t>(m_materials.size());
        const uint32_t count = std::min(bondCount, m_assetBondCount);
        for (uint32_t bond = 0; bond < m_assetBondCount; ++bond)
        {
            const uint32_t material =
                (materialIndices && bond < count) ? materialIndices[bond] : 0;
            m_bondMaterials[bond] = material < materialCount ? material : 0;
        }
        pushMaterialTables();
    }

    virtual void                            setNodeMaterials(const uint32_t* materialIndices, uint32_t nodeCount) override
    {
        const uint32_t materialCount = static_cast<uint32_t>(m_materials.size());
        const uint32_t count = std::min(nodeCount, m_graphNodeCount);
        for (uint32_t node = 0; node < m_graphNodeCount; ++node)
        {
            const uint32_t material =
                (materialIndices && node < count) ? materialIndices[node] : 0;
            m_nodeMaterials[node] = material < materialCount ? material : 0;
        }
        pushMaterialTables();
    }

    virtual void                            setNodeStrainRates(const float* strainRates, uint32_t nodeCount, float deltaTime) override
    {
        m_deltaTime = deltaTime > 0.0f ? deltaTime : 0.0f;
        if (!m_graphProcessor)
        {
            return;
        }
        const uint32_t count = std::min(nodeCount, m_graphNodeCount);
        for (uint32_t node = 0; node < m_graphNodeCount; ++node)
        {
            m_graphProcessor->setNodeStrainRate(
                node, (strainRates && node < count) ? strainRates[node] : 0.0f);
        }
    }

    virtual uint32_t                        getNodeCrushDamage(float* damage, uint32_t capacity) const override
    {
        if (!damage || !m_graphProcessor)
        {
            return 0;
        }
        const uint32_t count = std::min(capacity, m_graphNodeCount);
        for (uint32_t node = 0; node < count; ++node)
        {
            damage[node] = m_graphProcessor->getNodeCrushDamage(node);
        }
        return count;
    }

    virtual uint32_t                        getNodeStressInvariants(float* pressure, float* deviator, uint32_t capacity) const override
    {
        if (!m_graphProcessor)
        {
            return 0;
        }
        const uint32_t count = std::min(capacity, m_graphNodeCount);
        for (uint32_t node = 0; node < count; ++node)
        {
            if (pressure) pressure[node] = m_graphProcessor->getNodePressure(node);
            if (deviator) deviator[node] = m_graphProcessor->getNodeDeviator(node);
        }
        return count;
    }

    virtual uint32_t                        getNodeCrushUtilisation(float* utilisation, uint32_t capacity) const override
    {
        if (!utilisation || !m_graphProcessor)
        {
            return 0;
        }
        const uint32_t count = std::min(capacity, m_graphNodeCount);
        for (uint32_t node = 0; node < count; ++node)
        {
            utilisation[node] = m_graphProcessor->getNodeCrushUtilisation(node);
        }
        return count;
    }

    virtual uint32_t                        getCrushedNodes(uint32_t* nodeIndices, uint32_t capacity) override
    {
        if (!nodeIndices || !m_graphProcessor)
        {
            return 0;
        }
        return m_graphProcessor->drainCrushedNodes(nodeIndices, capacity);
    }

    virtual bool                            isCrushEnabled() const override
    {
        return m_graphProcessor && m_graphProcessor->isCrushEnabled();
    }

    virtual uint32_t                        getBondUtilisations(float* utilisation, uint32_t capacity) const override
    {
        if (!utilisation)
        {
            return 0;
        }
        const uint32_t count = std::min(capacity, m_assetBondCount);
        for (uint32_t bondIndex = 0; bondIndex < count; ++bondIndex)
        {
            float bondUtilisation = 0.0f;
            float bondCompression, bondTension, bondShear;
            if (m_bondHealths[bondIndex] > 0.0f
                && m_graphProcessor->getBondStress(
                    bondIndex, bondCompression, bondTension, bondShear))
            {
                // Divide by the BOND's own material limits: with mixed
                // materials a global divisor would misreport every joint
                // whose material differs from it.
                const ExtStressMaterial& material = materialForBond(bondIndex);
                if (material.compressionElasticLimit > 0.0f)
                {
                    bondUtilisation = std::max(
                        bondUtilisation, bondCompression / material.compressionElasticLimit);
                }
                if (material.tensionElasticLimit > 0.0f)
                {
                    bondUtilisation = std::max(
                        bondUtilisation, bondTension / material.tensionElasticLimit);
                }
                if (material.shearElasticLimit > 0.0f)
                {
                    bondUtilisation = std::max(
                        bondUtilisation, bondShear / material.shearElasticLimit);
                }
            }
            utilisation[bondIndex] = bondUtilisation;
        }
        return count;
    }

    virtual bool                            notifyActorCreated(const NvBlastActor& actor) override;

    virtual void                            notifyActorDestroyed(const NvBlastActor& actor) override;

    virtual const DebugBuffer               fillDebugRender(const uint32_t* nodes, uint32_t nodeCount, DebugRenderMode mode, float scale) override;


    //////// ExtStressSolverImpl public methods ////////

    bool                                    valid() { return m_valid; }

private:
    ~ExtStressSolverImpl();


    //////// private methods ////////

    void                                    solve();

    void                                    fillFractureCommands(const NvBlastActor& actor, NvBlastFractureBuffers& commands);

    void                                    initialize();

    void                                    iterate();

    void                                    removeBrokenBonds();

    template<class T>
    T*                                      getScratchArray(uint32_t size);

    bool                                    generateStressDamage(const NvBlastActor& actor, uint32_t bondIndex, uint32_t node0, uint32_t node1);

    static ExtStressMaterial                resolveMaterial(const ExtStressMaterial& material)
    {
        NVBLAST_ASSERT(material.compressionElasticLimit >= 0.0f && material.compressionFatalLimit >= 0.0f);
        ExtStressMaterial resolved = material;
        // Negative tension/shear limits inherit the compression values.
        if (resolved.tensionElasticLimit < 0.0f)
        {
            resolved.tensionElasticLimit = resolved.compressionElasticLimit;
        }
        if (resolved.tensionFatalLimit < 0.0f)
        {
            resolved.tensionFatalLimit = resolved.compressionFatalLimit;
        }
        if (resolved.shearElasticLimit < 0.0f)
        {
            resolved.shearElasticLimit = resolved.compressionElasticLimit;
        }
        if (resolved.shearFatalLimit < 0.0f)
        {
            resolved.shearFatalLimit = resolved.compressionFatalLimit;
        }
        return resolved;
    }

    /**
    Re-point the graph processor at the (possibly reallocated) material storage
    and recompute whether crush is active at all.

    Crush requires an UNREDUCED graph. Graph reduction merges support nodes into
    a single solver node, so a per-node stress tensor would describe an
    aggregate rather than a chunk. Rather than report a plausible wrong number,
    refuse: with reduction > 0 crush stays off and nothing is ever crushed.
    */
    void                                    pushMaterialTables()
    {
        if (!m_graphProcessor)
        {
            return;
        }

        bool anyCrush = false;
        for (const ExtStressMaterial& material : m_materials)
        {
            if (material.crush.enabled())
            {
                anyCrush = true;
                break;
            }
        }

        const bool reduced = m_settings.graphReductionLevel > 0;
        if (anyCrush && reduced && !m_crushReductionWarned)
        {
            m_crushReductionWarned = true;
            NVBLAST_LOG_ERROR(
                "ExtStressSolver: chunk crushing requires graphReductionLevel 0 "
                "(reduction merges chunks into aggregate solver nodes, so a per-chunk "
                "stress tensor would be meaningless). Crushing is disabled.");
        }

        m_graphProcessor->setMaterialTables(
            m_materials.data(),
            static_cast<uint32_t>(m_materials.size()),
            m_bondMaterials.data(),
            m_nodeMaterials.empty() ? nullptr : m_nodeMaterials.data(),
            anyCrush && !reduced);
        m_graphProcessor->refreshGroupMaterials();
    }

    const ExtStressMaterial&                materialForBond(uint32_t blastBondIndex) const
    {
        const uint32_t material = blastBondIndex < m_bondMaterials.size()
            ? m_bondMaterials[blastBondIndex]
            : 0;
        return m_materials[material < m_materials.size() ? material : 0];
    }


    //////// data ////////

    const NvBlastFamily&                                                m_family;
    HashSet<const NvBlastActor*>::type                                  m_activeActors;
    ExtStressSolverSettings                                             m_settings;
    // Resolved material table (>= 1 entry; index 0 = scene default) and the
    // per-ASSET-bond material indices. Storage is stable between set calls so
    // the graph processor can hold raw pointers into it.
    std::vector<ExtStressMaterial>                                      m_materials;
    std::vector<uint32_t>                                               m_bondMaterials;
    // Per-GRAPH-NODE material indices, selecting each chunk's crush properties.
    std::vector<uint32_t>                                               m_nodeMaterials;
    uint32_t                                                            m_graphNodeCount{0};
    // Timestep of the pending update(), used to integrate crush plastic work.
    float                                                               m_deltaTime{0.0f};
    NvBlastSupportGraph                                                 m_graph;
    bool                                                                m_isDirty;
    bool                                                                m_reset;
    const float*                                                        m_bondHealths;
    const float*                                                        m_cachedBondHealths;
    const NvBlastBond*                                                  m_bonds;
    uint32_t                                                            m_assetBondCount;
    SupportGraphProcessor*                                              m_graphProcessor;
    float                                                               m_errorAngular;
    float                                                               m_errorLinear;
    bool                                                                m_converged;
    bool                                                                m_islandAware;
    bool                                                                m_skipSettled;
    uint32_t                                                            m_framesCount;
    /// Host walls around the GPU solve, previously only visible as solve_ms's
    /// 30.4% remainder.
    float                                                               m_initializeMilliseconds{0.0f};
    float                                                               m_graphSolveMilliseconds{0.0f};
    float                                                               m_calcErrorMilliseconds{0.0f};
    bool                                                                m_crushReductionWarned{false};
    Array<NvBlastBondFractureData>::type                                m_bondFractureBuffer;
    Array<NvBlastChunkFractureData>::type                               m_chunkFractureBuffer;
    Array<uint8_t>::type                                                m_scratch;
    Array<DebugLine>::type                                              m_debugLineBuffer;
    bool                                                                m_valid;
};


template<class T>
NV_INLINE T* ExtStressSolverImpl::getScratchArray(uint32_t size)
{
    const uint32_t scratchSize = sizeof(T) * size;
    if (m_scratch.size() < scratchSize)
    {
        m_scratch.resize(scratchSize);
    }
    return reinterpret_cast<T*>(m_scratch.begin());
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                  Creation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ExtStressSolverImpl::ExtStressSolverImpl(const NvBlastFamily& family, const ExtStressSolverSettings& settings)
    : m_family(family), m_settings(settings), m_isDirty(false), m_reset(false),
    m_bondHealths(nullptr), m_cachedBondHealths(nullptr), m_bonds(nullptr), m_assetBondCount(0),
    m_graphProcessor(nullptr),
    m_errorAngular(std::numeric_limits<float>::max()), m_errorLinear(std::numeric_limits<float>::max()),
    m_converged(false), m_islandAware(false), m_skipSettled(false), m_framesCount(0), m_valid(false)
{
    // A solver always has a material table: the 1-entry default preserves the
    // historical settings-defaults behavior for callers that never author one.
    m_materials.push_back(resolveMaterial(ExtStressMaterial()));

    const NvBlastAsset* asset = NvBlastFamilyGetAsset(&m_family, logLL);
    if (!asset)
    {
        NVBLAST_LOG_ERROR("ExtStressSolverImpl::ExtStressSolverImpl: family has NULL asset");
        return;
    }

    m_graph = NvBlastAssetGetSupportGraph(asset, logLL);
    const uint32_t bondCount = NvBlastAssetGetBondCount(asset, logLL);
    m_assetBondCount = bondCount;
    m_bondMaterials.assign(bondCount, 0);
    m_graphNodeCount = m_graph.nodeCount;
    m_nodeMaterials.assign(m_graphNodeCount, 0);

    m_bondFractureBuffer.reserve(bondCount);
    // Reserved for the same reason as the bond buffer, and the bug is worth
    // naming: fillFractureCommands stores POINTERS into this buffer while it
    // keeps growing across actors, so a reallocation mid-generate dangles
    // every earlier actor's chunkFractures. One command per node per generate
    // is the hard ceiling, so nodeCount capacity makes growth impossible.
    m_chunkFractureBuffer.reserve(m_graph.nodeCount);

    {
        NvBlastActor* actor;
        NvBlastFamilyGetActors(&actor, 1, &family, logLL);
        m_bondHealths = NvBlastActorGetBondHealths(actor, logLL);
        m_cachedBondHealths = NvBlastActorGetCachedBondHeaths(actor, logLL);
        m_bonds = NvBlastAssetGetBonds(asset, logLL);
    }

    m_graphProcessor = NVBLAST_NEW(SupportGraphProcessor)(m_graph.nodeCount, bondCount);
    pushMaterialTables();

    // traverse graph and fill bond info
    for (uint32_t node0 = 0; node0 < m_graph.nodeCount; ++node0)
    {
        for (uint32_t adjacencyIndex = m_graph.adjacencyPartition[node0]; adjacencyIndex < m_graph.adjacencyPartition[node0 + 1]; adjacencyIndex++)
        {
            uint32_t bondIndex = m_graph.adjacentBondIndices[adjacencyIndex];
            if (m_bondHealths[bondIndex] <= 0.0f)
                continue;
            uint32_t node1 = m_graph.adjacentNodeIndices[adjacencyIndex];

            if (node0 < node1)
            {
                m_graphProcessor->addBond(node0, node1, bondIndex);
            }
        }
    }

    // If we got here we should have a valid solver
    m_valid = true;
}

ExtStressSolverImpl::~ExtStressSolverImpl()
{
    NVBLAST_DELETE(m_graphProcessor, SupportGraphProcessor);
}

ExtStressSolver* ExtStressSolver::create(const NvBlastFamily& family, const ExtStressSolverSettings& settings)
{
    ExtStressSolverImpl* solver = NVBLAST_NEW(ExtStressSolverImpl) (family, settings);

    if (!solver->valid())
    {
        solver->release();
        solver = nullptr;
        NVBLAST_LOG_ERROR("ExtStressSolver::create: could not create solver");
    }

    return solver;
}

void ExtStressSolverImpl::release()
{
    NVBLAST_DELETE(this, ExtStressSolverImpl);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                          Actors & Graph Data
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ExtStressSolverImpl::setAllNodesInfoFromLL(float density)
{
    const NvBlastAsset* asset = NvBlastFamilyGetAsset(&m_family, logLL);
    NVBLAST_ASSERT(asset);

    const uint32_t chunkCount = NvBlastAssetGetChunkCount(asset, logLL);
    const NvBlastChunk* chunks = NvBlastAssetGetChunks(asset, logLL);

    // traverse graph and fill node info
    for (uint32_t node0 = 0; node0 < m_graph.nodeCount; ++node0)
    {
        const uint32_t chunkIndex0 = m_graph.chunkIndices[node0];
        if (chunkIndex0 >= chunkCount)
        {
            // chunkIndex is invalid means it is static node (represents world)
            m_graphProcessor->setNodeInfo(node0, 0.0f, 0.0f, NvVec3(NvZero));
        }
        else
        {
            // fill node info
            const NvBlastChunk& chunk = chunks[chunkIndex0];
            const float volume = chunk.volume;
            const float mass = volume * density;
            const NvVec3 localPos = *reinterpret_cast<const NvVec3*>(chunk.centroid);
            m_graphProcessor->setNodeInfo(node0, mass, volume, localPos);
        }
    }
}

void ExtStressSolverImpl::setNodeInfo(uint32_t graphNode, float mass, float volume, NvcVec3 localPos)
{
    m_graphProcessor->setNodeInfo(graphNode, mass, volume, toNvShared(localPos));
}

bool ExtStressSolverImpl::getExcessForces(uint32_t actorIndex, const NvcVec3& com, NvcVec3& force, NvcVec3& torque)
{
    // otherwise allocate enough space and query the Blast SDK
    const NvBlastActor* actor = NvBlastFamilyGetActorByIndex(&m_family, actorIndex, logLL);
    if (actor == nullptr)
    {
        return false;
    }

    const uint32_t nodeCount = NvBlastActorGetGraphNodeCount(actor, logLL);
    uint32_t* nodeIndices = getScratchArray<uint32_t>(nodeCount);
    const uint32_t retCount = NvBlastActorGetGraphNodeIndices(nodeIndices, nodeCount, actor, logLL);
    NVBLAST_ASSERT(retCount == nodeCount);

    // get the mapping between support chunks and actor indices
    // this is the fastest way to tell if two node/chunks are part of the same actor
    const uint32_t* actorIndices = NvBlastFamilyGetChunkActorIndices(&m_family, logLL);
    if (actorIndices == nullptr)
    {
        return false;
    }

    // walk the visible nodes for the actor looking for bonds that broke this frame
    nvidia::NvVec3 totalForce(0.0f);
    nvidia::NvVec3 totalTorque(0.0f);
    for (uint32_t n = 0; n < nodeCount; n++)
    {
        // find bonds that broke this frame (health <= 0 but internal stress bond index is still valid)
        const uint32_t nodeIdx = nodeIndices[n];
        for (uint32_t i = m_graph.adjacencyPartition[nodeIdx]; i < m_graph.adjacencyPartition[nodeIdx + 1]; i++)
        {
            // check if the bond is broken first of all
            const uint32_t blastBondIndex = m_graph.adjacentBondIndices[i];
            if (m_bondHealths[blastBondIndex] > 0.0f)
            {
                continue;
            }

            // broken bonds that have invalid internal indices broke before this frame
            const uint32_t internalBondIndex = m_graphProcessor->getInternalBondIndex(blastBondIndex);
            if (isInvalidIndex(internalBondIndex))
            {
                continue;
            }

            // make sure the other node in the bond isn't part of the same actor
            // forces should only be applied due to bonds breaking between actors, not within
            const uint32_t chunkIdx = m_graph.chunkIndices[nodeIdx];
            const uint32_t otherNodeIdx = m_graph.adjacentNodeIndices[i];
            const uint32_t otherChunkIdx = m_graph.chunkIndices[otherNodeIdx];
            if (!isInvalidIndex(chunkIdx) && !isInvalidIndex(otherChunkIdx) && actorIndices[chunkIdx] == actorIndices[otherChunkIdx])
            {
                continue;
            }

            // this bond should contribute forces to the output
            const auto bondData = m_graphProcessor->getBondData(internalBondIndex);
            NVBLAST_ASSERT(blastBondIndex == bondData.blastBondIndex);
            uint32_t node0, node1;
            m_graphProcessor->getSolverInternalBondNodes(internalBondIndex, node0, node1);
            NVBLAST_ASSERT(bondData.node0 == node0 && bondData.node1 == node1);

            // accumulators for forces just from this bond
            nvidia::NvVec3 nvLinearPressure(0.0f);
            nvidia::NvVec3 nvAngularPressure(0.0f);

            // deal with linear forces
            const ExtStressMaterial& bondMaterial = materialForBond(blastBondIndex);
            // Excess is DEMANDED load beyond the fatal limit -- and a joint can
            // never transmit more force than its own breaking strength: demand
            // beyond that was never carried through the joint, the joint failed
            // instead. Unbounded, this injected the full demanded overshoot as
            // impulse; utilisation spikes of 23-116x the elastic limit were
            // measured live, so the same break sometimes read as a shrug and
            // sometimes as an explosion, and the explosive tail seeded
            // ground-tunnelling escape velocities. Each component is therefore
            // bounded by that bond's OWN fatal limit: the material table that
            // decides breaking also bounds the release. No tunable constant is
            // involved, and typical (sub-fatal-overshoot) breaks are unchanged.
            const float excessCompression = std::max(
                bondData.stressNormal + bondMaterial.compressionFatalLimit,
                -bondMaterial.compressionFatalLimit);
            const float excessTension = std::min(
                bondData.stressNormal - bondMaterial.tensionFatalLimit,
                bondMaterial.tensionFatalLimit);
            if (excessCompression < 0.0f)
            {
                nvLinearPressure += excessCompression * bondData.normal;
            }
            else if (excessTension > 0.0f)
            {
                // tension is in the negative direction of the linear impulse
                nvLinearPressure += excessTension * bondData.normal;
            }

            const float excessShear = std::min(
                bondData.stressShear - bondMaterial.shearFatalLimit,
                bondMaterial.shearFatalLimit);
            if (excessShear > 0.0f)
            {
                NvVec3 impulseLinear, impulseAngular;
                m_graphProcessor->getSolverInternalBondImpulses(internalBondIndex, impulseLinear, impulseAngular);
                const nvidia::NvVec3 shearDir = impulseLinear - impulseLinear.dot(bondData.normal)*bondData.normal;
                nvLinearPressure += excessShear * shearDir.getNormalized();
            }

            if (nvLinearPressure.magnitudeSquared() > FLT_EPSILON)
            {
                const float* bondCenter = m_bonds[blastBondIndex].centroid;
                const nvidia::NvVec3 forceOffset =
                    nvidia::NvVec3(bondCenter[0], bondCenter[1], bondCenter[2])
                    - toNvShared(com);
                const nvidia::NvVec3 torqueFromForce = forceOffset.cross(nvLinearPressure);
                nvAngularPressure += torqueFromForce;
            }

            // add the contributions from this bond to the total forces for the actor
            // multiply by the area to convert back to force from pressure
            const float bondRemainingArea = m_cachedBondHealths[blastBondIndex];
            NVBLAST_ASSERT(bondRemainingArea <= m_bonds[blastBondIndex].area);

            const float sign = otherNodeIdx > nodeIdx ? 1.0f : -1.0f;

            totalForce += nvLinearPressure * (sign*bondRemainingArea);
            totalTorque += nvAngularPressure * (sign*bondRemainingArea);
        }
    }

    // convert to the output format and return true if non-zero forces were accumulated
    force = fromNvShared(totalForce);
    torque = fromNvShared(totalTorque);
    return (totalForce.magnitudeSquared() + totalTorque.magnitudeSquared()) > 0.0f;
}

bool ExtStressSolverImpl::notifyActorCreated(const NvBlastActor& actor)
{
    const uint32_t graphNodeCount = NvBlastActorGetGraphNodeCount(&actor, logLL);
    if (graphNodeCount > 1)
    {
        // update neighbors
        {
            uint32_t* graphNodeIndices = getScratchArray<uint32_t>(graphNodeCount);
            const uint32_t nodeCount = NvBlastActorGetGraphNodeIndices(graphNodeIndices, graphNodeCount, &actor, logLL);
            for (uint32_t i = 0; i < nodeCount; ++i)
            {
                m_graphProcessor->setNodeNeighborsCount(graphNodeIndices[i], nodeCount);
            }
        }

        m_activeActors.insert(&actor);
        m_isDirty = true;
        return true;
    }
    return false;
}

void ExtStressSolverImpl::notifyActorDestroyed(const NvBlastActor& actor)
{
    if (m_activeActors.erase(&actor))
    {
        m_isDirty = true;
    }
}

void ExtStressSolverImpl::removeBrokenBonds()
{
    // traverse graph and remove dead bonds
    for (uint32_t node0 = 0; node0 < m_graph.nodeCount; ++node0)
    {
        for (uint32_t adjacencyIndex = m_graph.adjacencyPartition[node0]; adjacencyIndex < m_graph.adjacencyPartition[node0 + 1]; adjacencyIndex++)
        {
            uint32_t node1 = m_graph.adjacentNodeIndices[adjacencyIndex];
            if (node0 < node1)
            {
                uint32_t bondIndex = m_graph.adjacentBondIndices[adjacencyIndex];
                if (m_bondHealths[bondIndex] <= 0.0f)
                {
                    m_graphProcessor->removeBondIfExists(bondIndex);
                }
            }
        }
    }

    m_isDirty = false;
}

void ExtStressSolverImpl::initialize()
{
    if (m_reset)
    {
        m_framesCount = 0;
    }

    if (m_isDirty)
    {
        removeBrokenBonds();
    }

    if (m_settings.graphReductionLevel != m_graphProcessor->getGraphReductionLevel())
    {
        m_graphProcessor->setGraphReductionLevel(m_settings.graphReductionLevel);
    }
}

bool ExtStressSolverImpl::addForce(const NvBlastActor& actor, NvcVec3 localPosition, NvcVec3 localForce, ExtForceMode::Enum mode)
{
    float bestDist = FLT_MAX;
    uint32_t bestNode = invalidIndex<uint32_t>();

    const uint32_t graphNodeCount = NvBlastActorGetGraphNodeCount(&actor, logLL);
    if (graphNodeCount > 1)
    {
        uint32_t* graphNodeIndices = getScratchArray<uint32_t>(graphNodeCount);
        const uint32_t nodeCount = NvBlastActorGetGraphNodeIndices(graphNodeIndices, graphNodeCount, &actor, logLL);

        for (uint32_t i = 0; i < nodeCount; ++i)
        {
            const uint32_t node = graphNodeIndices[i];
            const float sqrDist = (toNvShared(localPosition) - m_graphProcessor->getNodeData(node).localPos).magnitudeSquared();
            if (sqrDist < bestDist)
            {
                bestDist = sqrDist;
                bestNode = node;
            }
        }

        if (!isInvalidIndex(bestNode))
        {
            // Position-aware: an external contact is a surface traction and
            // belongs in the node's virial sum, unlike gravity below.
            m_graphProcessor->addNodeForceAt(
                bestNode, toNvShared(localPosition), toNvShared(localForce), mode);
            return true;
        }
    }
    return false;
}

void ExtStressSolverImpl::addForce(uint32_t graphNode, NvcVec3 localForce, ExtForceMode::Enum mode)
{
    m_graphProcessor->addNodeForce(graphNode, toNvShared(localForce), mode);
}

void ExtStressSolverImpl::addForceAt(uint32_t graphNode, NvcVec3 localPosition, NvcVec3 localForce, ExtForceMode::Enum mode)
{
    m_graphProcessor->addNodeForceAt(
        graphNode, toNvShared(localPosition), toNvShared(localForce), mode);
}

bool ExtStressSolverImpl::addGravity(const NvBlastActor& actor, NvcVec3 localGravity)
{
    const uint32_t graphNodeCount = NvBlastActorGetGraphNodeCount(&actor, logLL);
    if (graphNodeCount > 1)
    {
        uint32_t* graphNodeIndices = getScratchArray<uint32_t>(graphNodeCount);
        const uint32_t nodeCount = NvBlastActorGetGraphNodeIndices(graphNodeIndices, graphNodeCount, &actor, logLL);

        for (uint32_t i = 0; i < nodeCount; ++i)
        {
            const uint32_t node = graphNodeIndices[i];
            m_graphProcessor->addNodeForce(node, toNvShared(localGravity), ExtForceMode::ACCELERATION);
        }
        return true;
    }
    return false;
}

bool ExtStressSolverImpl::addCentrifugalAcceleration(const NvBlastActor& actor, NvcVec3 localCenterMass, NvcVec3 localAngularVelocity)
{
    const uint32_t graphNodeCount = NvBlastActorGetGraphNodeCount(&actor, logLL);
    if (graphNodeCount > 1)
    {
        uint32_t* graphNodeIndices = getScratchArray<uint32_t>(graphNodeCount);
        const uint32_t nodeCount = NvBlastActorGetGraphNodeIndices(graphNodeIndices, graphNodeCount, &actor, logLL);

        // Apply centrifugal force
        for (uint32_t i = 0; i < nodeCount; ++i)
        {
            const uint32_t node = graphNodeIndices[i];
            const auto& localPos = m_graphProcessor->getNodeData(node).localPos;
            // a = w x (w x r)
            const NvVec3 centrifugalAcceleration =
                toNvShared(localAngularVelocity)
                    .cross(toNvShared(localAngularVelocity).cross(localPos - toNvShared(localCenterMass)));
            m_graphProcessor->addNodeForce(node, centrifugalAcceleration, ExtForceMode::ACCELERATION);
        }
        return true;
    }
    return false;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                  Update
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ExtStressSolverImpl::update()
{
    // solve_ms minus (kernel + host work + blocked) left 30.4% unnamed. It is
    // not in the GPU solver at all -- these two walks sit around it and had no
    // timer, so the only way to see them was a subtraction nobody performed.
    const auto initStart = std::chrono::steady_clock::now();
    initialize();
    m_initializeMilliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - initStart).count();

    solve();

    m_framesCount++;
}

void ExtStressSolverImpl::solve()
{
    NV_SIMD_GUARD;

    const auto graphStart = std::chrono::steady_clock::now();
    m_graphProcessor->solve(m_settings, m_bondHealths, m_bonds, WARM_START && !m_reset, m_islandAware, m_skipSettled, m_deltaTime);
    m_reset = false;
    m_graphSolveMilliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - graphStart).count();

    const auto errStart = std::chrono::steady_clock::now();
    m_converged = m_graphProcessor->calcError(m_errorLinear, m_errorAngular);
    m_calcErrorMilliseconds =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - errStart).count();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                  Damage
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// check if this bond is over stressed in any way and generate a fracture command if it is
bool ExtStressSolverImpl::generateStressDamage(const NvBlastActor& actor, uint32_t bondIndex, uint32_t node0, uint32_t node1)
{
    const float bondHealth = m_bondHealths[bondIndex];
    float stressCompression, stressTension, stressShear;
    if (bondHealth > 0.0f && m_graphProcessor->getBondStress(bondIndex, stressCompression, stressTension, stressShear))
    {
        // compression and tension are mutually exclusive, only one can be positive at a time since they act in opposite directions
        const ExtStressMaterial& material = materialForBond(bondIndex);
        float stressMultiplier = 0.0f;
        if (stressCompression > material.compressionElasticLimit)
        {
            const float excessStress = stressCompression - material.compressionElasticLimit;
            const float compressionDenom = material.compressionFatalLimit - material.compressionElasticLimit;
            const float compressionMultiplier = excessStress / (compressionDenom > 0.0f ? compressionDenom : 1.0f);
            stressMultiplier += compressionMultiplier;
        }
        else if (stressTension > material.tensionElasticLimit)
        {
            const float excessStress = stressTension - material.tensionElasticLimit;
            const float tensionDenom = material.tensionFatalLimit - material.tensionElasticLimit;
            const float tensionMultiplier = excessStress / (tensionDenom > 0.0f ? tensionDenom : 1.0f);
            stressMultiplier += tensionMultiplier;
        }

        // shear can co-exist with either compression or tension so must be accounted for independently of them
        if (stressShear > material.shearElasticLimit)
        {
            const float excessStress = stressShear - material.shearElasticLimit;
            const float shearDenom = material.shearFatalLimit - material.shearElasticLimit;
            const float shearMultiplier = excessStress / (shearDenom > 0.0f ? shearDenom : 1.0f);
            stressMultiplier += shearMultiplier;
        }

        if (stressMultiplier > 0.0f)
        {
            // bond health/area is reduced by excess pressure to approximate micro bonds in the material breaking
            const float bondDamage = bondHealth * stressMultiplier;
            const NvBlastBondFractureData data = {
                0,
                node0,
                node1,
                bondDamage
            };
            m_bondFractureBuffer.pushBack(data);

            // cache off the current health value for this bond
            // so it can be used to calculate forces to apply if it breaks later
            NvBlastActorCacheBondHeath(&actor, bondIndex, logLL);
            return true;
        }
    }

    return false;
}

void ExtStressSolverImpl::fillFractureCommands(const NvBlastActor& actor, NvBlastFractureBuffers& commands)
{
    const uint32_t graphNodeCount = NvBlastActorGetGraphNodeCount(&actor, logLL);
    uint32_t commandCount = 0;
    uint32_t chunkCommandCount = 0;

    const bool anyBondWork = graphNodeCount > 1 && m_graphProcessor->getOverstressedBondCount() > 0;
    const bool anyCrushWork = m_graphProcessor->isCrushEnabled() && m_graphProcessor->getPendingCrushCount() > 0;

    if (anyBondWork || anyCrushWork)
    {
        uint32_t* graphNodeIndices = getScratchArray<uint32_t>(graphNodeCount);
        const uint32_t nodeCount = NvBlastActorGetGraphNodeIndices(graphNodeIndices, graphNodeCount, &actor, logLL);

        // E1: only nodes flagged during the stress readback can have an
        // overstressed incident bond, and generateStressDamage is pure for
        // every other bond — it mutates nothing unless a limit is exceeded —
        // so skipping unflagged nodes preserves every command, every value,
        // and the exact emission order. Walk order itself is unchanged.
        // BLAST_FRACTURE_NODE_SKIP=0 restores the full walk.
        static const bool nodeSkip = [] {
            const char* raw = std::getenv("BLAST_FRACTURE_NODE_SKIP");
            return raw == nullptr || std::string(raw) != "0";
        }();
        for (uint32_t i = 0; i < nodeCount; ++i)
        {
            const uint32_t node0 = graphNodeIndices[i];

            if (anyBondWork
                && (!nodeSkip || m_graphProcessor->isNodeOverstressed(node0)))
            {
                for (uint32_t adjacencyIndex = m_graph.adjacencyPartition[node0]; adjacencyIndex < m_graph.adjacencyPartition[node0 + 1]; adjacencyIndex++)
                {
                    const uint32_t node1 = m_graph.adjacentNodeIndices[adjacencyIndex];
                    if (node0 < node1)
                    {
                        const uint32_t bondIndex = m_graph.adjacentBondIndices[adjacencyIndex];
                        if (generateStressDamage(actor, bondIndex, node0, node1))
                        {
                            commandCount++;
                        }
                    }
                }
            }

            // A fully crushed chunk is severed through Blast's own chunk
            // fracture path, which zeroes every incident bond and drops the
            // node out of the island graph (NvBlastFamily.cpp, fractureWithEvents).
            //
            // Note the command is issued ONLY at full crush, never for partial
            // damage: Blast detaches a support chunk on ANY positive chunk
            // damage regardless of remaining health, so partial crush damage
            // cannot be represented in Blast's chunk health without severing
            // the chunk early. That is why crush damage accumulates here in the
            // solver instead.
            if (anyCrushWork && m_graphProcessor->needsCrushCommand(node0))
            {
                const NvBlastChunkFractureData data = {
                    0,
                    m_graph.chunkIndices[node0],
                    // Health is a DAMAGE amount here. Blast's health for a
                    // lower-support chunk is authored as 1.0 by this stack, so
                    // anything above it exhausts the chunk outright; the
                    // unbreakable sentinel keeps genuinely indestructible
                    // chunks out of this path in the first place.
                    2.0f
                };
                m_chunkFractureBuffer.pushBack(data);
                m_graphProcessor->markCrushCommandIssued(node0);
                chunkCommandCount++;
            }
        }
    }

    commands.chunkFractureCount = chunkCommandCount;
    commands.chunkFractures = chunkCommandCount > 0 ? m_chunkFractureBuffer.end() - chunkCommandCount : nullptr;
    commands.bondFractureCount = commandCount;
    commands.bondFractures = commandCount > 0 ? m_bondFractureBuffer.end() - commandCount : nullptr;
}

void ExtStressSolverImpl::generateFractureCommands(const NvBlastActor& actor, NvBlastFractureBuffers& commands)
{
    m_bondFractureBuffer.clear();
    m_chunkFractureBuffer.clear();
    fillFractureCommands(actor, commands);
}

uint32_t ExtStressSolverImpl::generateFractureCommandsPerActor(const NvBlastActor** actorBuffer, NvBlastFractureBuffers* commandsBuffer, uint32_t bufferSize)
{
    // A crushed chunk is fracture work even when no bond is overstressed: a
    // chunk can be pulverized while every joint around it is still intact.
    const bool crushPending =
        m_graphProcessor->isCrushEnabled() && m_graphProcessor->getPendingCrushCount() > 0;
    if (m_graphProcessor->getOverstressedBondCount() == 0 && !crushPending)
        return 0;

    m_bondFractureBuffer.clear();
    m_chunkFractureBuffer.clear();
    uint32_t index = 0;
    for (auto it = m_activeActors.getIterator(); !it.done() && index < bufferSize; ++it)
    {
        const NvBlastActor* actor = *it;
        NvBlastFractureBuffers& nextCommand = commandsBuffer[index];
        fillFractureCommands(*actor, nextCommand);
        if (nextCommand.bondFractureCount > 0 || nextCommand.chunkFractureCount > 0)
        {
            actorBuffer[index] = actor;
            index++;
        }
    }
    return index;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                  Debug Render
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline uint32_t NvVec4ToU32Color(const NvVec4& color)
{
    return ((uint32_t)(color.w * 255) << 24) | // A
           ((uint32_t)(color.x * 255) << 16) | // R
           ((uint32_t)(color.y * 255) << 8)  | // G
           ((uint32_t)(color.z * 255));        // B
}

static float Lerp(float v0, float v1, float val)
{
    return v0 * (1 - val) + v1 * val;
}

inline float clamp01(float v)
{
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

inline NvVec4 colorConvertHSVAtoRGBA(float h, float s, float v, float a)
{
    const float t = 6.0f * (h - std::floor(h));
    const int n = (int)t;
    const float m = t - (float)n;
    const float c = 1.0f - s;
    const float b[6] = { 1.0f, 1.0f - s * m, c, c, 1.0f - s * (1.0f - m), 1.0f };
    return NvVec4(v * b[n % 6], v * b[(n + 4) % 6], v * b[(n + 2) % 6], a); // n % 6 protects against roundoff errors
}

inline uint32_t bondHealthColor(float stressPct)
{
    stressPct = clamp01(stressPct);

    constexpr float BOND_HEALTHY_HUE = 1.0f/3.0f;   // Green
    constexpr float BOND_ELASTIC_HUE = 0.0f;        // Red
    constexpr float BOND_STRESSED_HUE = 2.0f/3.0f;  // Blue
    constexpr float BOND_FATAL_HUE = 5.0f/6.0f;     // Magenta

    const float hue = stressPct < 0.5f ?
        Lerp(BOND_HEALTHY_HUE, BOND_ELASTIC_HUE, 2.0f * stressPct) : Lerp(BOND_STRESSED_HUE, BOND_FATAL_HUE, 2.0f * stressPct - 1.0f);

    return NvVec4ToU32Color(colorConvertHSVAtoRGBA(hue, 1.0f, 1.0f, 1.0f));
}

const ExtStressSolver::DebugBuffer ExtStressSolverImpl::fillDebugRender(const uint32_t* nodes, uint32_t nodeCount, DebugRenderMode mode, float scale)
{
    NV_UNUSED(scale);

    const uint32_t BOND_UNBREAKABLE_COLOR = NvVec4ToU32Color(NvVec4(0.0f, 0.682f, 1.0f, 1.0f));

    ExtStressSolver::DebugBuffer debugBuffer = { nullptr, 0 };

    if (m_isDirty)
        return debugBuffer;

    m_debugLineBuffer.clear();

    Array<uint8_t>::type& nodesSet = m_scratch;

    nodesSet.resize(m_graphProcessor->getSolverNodeCount());
    memset(nodesSet.begin(), 0, nodesSet.size() * sizeof(uint8_t));
    for (uint32_t i = 0; i < nodeCount; ++i)
    {
        NVBLAST_ASSERT(m_graphProcessor->getNodeData(nodes[i]).solverNode < nodesSet.size());
        nodesSet[m_graphProcessor->getNodeData(nodes[i]).solverNode] = 1;
    }

    const uint32_t bondCount = m_graphProcessor->getSolverBondCount();
    for (uint32_t i = 0; i < bondCount; ++i)
    {
        const auto& bondData = m_graphProcessor->getBondData(i);
        uint32_t node0, node1;
        m_graphProcessor->getSolverInternalBondNodes(i, node0, node1);
        if (nodesSet[node0] != 0)
        {
            //NVBLAST_ASSERT(nodesSet[node1] != 0);
            const auto& solverNode0 = m_graphProcessor->getSolverNodeData(node0);
            const auto& solverNode1 = m_graphProcessor->getSolverNodeData(node1);
            const NvcVec3 p0 = fromNvShared(solverNode0.mass > 0.0f ? solverNode0.localPos : bondData.centroid);
            const NvcVec3 p1 = fromNvShared(solverNode1.mass > 0.0f ? solverNode1.localPos : bondData.centroid);

            // don't render lines for broken bonds
            const float stressPct = m_graphProcessor->getSolverBondStressPct(i, m_bondHealths, mode);
            if (stressPct >= 0.0f)
            {
                const uint32_t color = canTakeDamage(m_bondHealths[bondData.blastBondIndex]) ? bondHealthColor(stressPct) : BOND_UNBREAKABLE_COLOR;
                m_debugLineBuffer.pushBack(DebugLine(p0, p1, color));
            }
        }
    }

    debugBuffer.lines = m_debugLineBuffer.begin();
    debugBuffer.lineCount = m_debugLineBuffer.size();

    return debugBuffer;
}


} // namespace Blast
} // namespace Nv
