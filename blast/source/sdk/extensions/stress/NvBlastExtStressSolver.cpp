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
#include "NvBlastExtStressFormula.h"
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
#include <utility>
#include <vector>
#include <cmath>
#include <cstdio>
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
static ExtStressParallelFor s_parallelFor = nullptr;
static void*                s_parallelForCtx = nullptr;

void NvBlastExtStressSetParallelFor(ExtStressParallelFor fn, void* ctx)
{
    s_parallelFor = fn;
    s_parallelForCtx = ctx;
}

void getExtStressParallelFor(ExtStressParallelFor& fn, void*& ctx)
{
    fn = s_parallelFor;
    ctx = s_parallelForCtx;
}


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

// Bending policy, at file scope because two classes need it: the CG solver
// hands it to the GPU (which cannot read an environment variable), and the
// graph processor uses it on the host walk. Both must agree, which is the
// whole reason the formula itself lives in a shared header.
/// Ceiling on the section-modulus gain, tunable with BLAST_BEND_MAX_GAIN.
///
/// 6/sqrt(area) is the right shape for the gain and the wrong magnitude at
/// a small joint, because part of what it amplifies there is the
/// discretisation's own moments rather than the structure's. A chunk-to-
/// chunk seam in a wall carries a moment that the wall, as a wall, does
/// not.
///
/// Measured over ten seconds of gravity and nothing else on a 47,631-chunk
/// masonry structure -- concentric stone ring walls, which should stand in
/// almost pure compression, and whose stone yields in tension at 0.8 MPa:
///
///     gain 10   2,423 bonds broken and climbing
///     gain  5      28
///     gain  3       0
///     gain  2       1
///
/// 3 is where it stops cracking itself apart, and costs nothing measurable
/// elsewhere: the cantilever ladder behaves identically from 1.5 to 10 on
/// the CUDA solver, so nothing in the set is relying on the higher gain.
static float maxBendAmplification()
{
    static const float gain = []() {
        const char* raw = std::getenv("BLAST_BEND_MAX_GAIN");
        if (raw != nullptr)
        {
            const float parsed = static_cast<float>(std::atof(raw));
            if (parsed > 0.0f)
            {
                return parsed;
            }
        }
        return 3.0f;
    }();
    return gain;
}

/// Whether bending is scaled by a section modulus (see above).
///
/// A section modulus is what turns a moment into a fibre stress; the node
/// spacing this replaces was never a section dimension, so the old scaling
/// understated bending on a slab joint by roughly an order of magnitude.
///
/// This was briefly defaulted OFF on the strength of a measurement that
/// showed the shipping city breaking 2,016 of its own bonds in ten seconds
/// of gravity with it on. That measurement was worthless: it was taken on
/// the CPU CG solve, whose 8-iteration residual is reported as real stress
/// and which self-destructs cities on its own. On the CUDA solver the
/// server actually runs, the same scene breaks ZERO bonds with this on.
///
/// The lesson is older than this change and is recorded elsewhere in the
/// tree: a stress number from the CPU path is a property of the solver, not
/// of the structure. BLAST_BEND_SECTION_MODULUS=0 restores the old scaling
/// for A/B.
static bool sectionModulusBending()
{
    static const bool enabled = []() {
        const char* value = std::getenv("BLAST_BEND_SECTION_MODULUS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// Whether bending resolves into separate tension and compression fibres
/// rather than being added to the axial stress with its sign.
static bool fibreBending()
{
    static const bool enabled = []() {
        const char* value = std::getenv("BLAST_BEND_FIBER");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}


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
        // Resolve each bond's Young's modulus from its material before the
        // processor computes compliance weights: stiffness is EA/L, and
        // without E a steel tie and a stone bed joint of equal size would
        // share load equally when steel is ~20x stiffer.
        for (uint32_t i = 0; i < m_bonds.size(); ++i)
        {
            const uint32_t material = m_bonds[i].material;
            m_bonds[i].modulus = (m_materials && material < m_materialCount)
                ? m_materials[material].elasticModulusPa
                : 0.0f;
        }

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
        // Record why the GPU backend is or is not live. Every branch below
        // used to be silent, so "I asked for the GPU and got the CPU" was
        // indistinguishable from "I asked for the GPU and got it" without a
        // debugger.
        if (!m_gpuRequested)
        {
            m_gpuInactiveReason = "not requested (setGpuAccelerated was never called with true)";
        }
        else if (m_nodes.empty())
        {
            m_gpuInactiveReason = "the family has no nodes";
        }
        else if (m_bonds.size() < m_gpuMinimumBondCount)
        {
            snprintf(m_gpuReasonBuf, sizeof(m_gpuReasonBuf),
                     "bond count %u is below gpuStressMinimumBondCount %u "
                     "(raise the bond count or lower the threshold; it defaults to 4096, "
                     "which is deliberately above test-scene sizes)",
                     unsigned(m_bonds.size()), unsigned(m_gpuMinimumBondCount));
            m_gpuInactiveReason = m_gpuReasonBuf;
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
                // The exact weights the CPU processor computed, so the two
                // backends solve the same weighted system rather than two
                // systems that agree only when every joint is the same size.
                gpu.colScale = m_stressProcessor.getColumnScale(i);
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
            m_gpuInactiveReason = (m_gpuSolver != nullptr)
                ? nullptr
                : "ExtStressGpuSolver::create returned null "
                  "(no CUDA device, no context set via setGpuCudaContext, or allocation failed)";
            m_gpuVelocities.resize(m_nodes.size());
            m_gpuImpulses.resize(m_bonds.size());
            m_gpuIslandsSkipped = 0;
            m_gpuIslandsTotal = 0;
        }
#endif
    }

    /// Request the CUDA backend. The return value means "the request was
    /// accepted", NOT "the GPU is now in use": the backend is (re)built on the
    /// next prepare, and can still decline -- below the minimum bond count, no
    /// CUDA context, an empty family. Ask getGpuAccelerated() for what is
    /// actually running and getGpuInactiveReason() for why, rather than
    /// assuming this returning true means anything happened.
    bool setGpuAccelerated(bool enabled)
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        m_gpuRequested = enabled;
        m_forceColdStart = true;
        if (!enabled)
        {
            m_gpuInactiveReason = "not requested (setGpuAccelerated was called with false)";
        }
        return true;
#else
        m_gpuInactiveReason =
            "this build has no CUDA stress solver (NVBLAST_ENABLE_CUDA_STRESS undefined)";
        return !enabled;
#endif
    }

    /// Null when the CUDA backend is live; otherwise a human-readable reason.
    const char* getGpuInactiveReason() const
    {
        return m_gpuInactiveReason;
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

    /// Which solver bonds re-solved this tick. Empty means "assume all".
    const std::vector<uint8_t>& bondDirty() const { return m_bondDirty; }

    /// Advances only when a solve actually produced new impulses. The settled
    /// early-out does not touch it, so an unchanged serial is a positive
    /// statement that every impulse is bit-identical to the last observation.
    uint64_t solveSerial() const { return m_solveSerial; }

#if defined(NVBLAST_ENABLE_CUDA_STRESS)
    /// The resident device solver, for the bond-stress walk that consumes the
    /// impulses it already holds. Null when the GPU path is not active.
    ExtStressGpuSolver* gpuSolver() const { return m_gpuSolver; }

    /// Whether the LAST solve actually ran on the device. Having a solver is
    /// not the same thing: solveAndReadbackImpulses can fail over to the host
    /// StressProcessor, and the CPU early-out skips the solve entirely. In
    /// both cases the device's resident impulses are not what the host walk
    /// would read, so a device bond-stress walk would be computing from stale
    /// inputs.
    bool lastSolveOnDevice() const { return m_lastSolveOnDevice; }

#if defined(NVBLAST_ENABLE_CUDA_STRESS)
    /// Audit: does the host's impulse mirror still equal what the device
    /// holds? The host only copies back lastChangedBonds, so a bond the
    /// solver did not re-solve keeps its previous host value -- which is only
    /// correct if the device kept the same one.
    void auditImpulseMirror(uint64_t& checks, uint64_t& mismatches, double& maxRel)
    {
        if (!m_gpuSolver) { return; }
        const uint32_t n = static_cast<uint32_t>(m_impulses.size());
        m_auditImpulses.resize(n);
        if (!m_gpuSolver->readbackImpulses(m_auditImpulses.data(), n)) { return; }
        for (uint32_t i = 0; i < n; ++i)
        {
            const AngLin6& h = m_impulses[i];
            const ExtStressGpuImpulse& d = m_auditImpulses[i];
            const float hv[6] = {h.ang.x, h.ang.y, h.ang.z, h.lin.x, h.lin.y, h.lin.z};
            const float dv[6] = {d.angular.x, d.angular.y, d.angular.z,
                                 d.linear.x, d.linear.y, d.linear.z};
            ++checks;
            if (memcmp(hv, dv, sizeof(hv)) != 0)
            {
                ++mismatches;
                for (int k = 0; k < 6; ++k)
                {
                    const double a = hv[k], b = dv[k];
                    const double rel = std::abs(a) > 1e-9
                        ? std::abs(b - a) / std::abs(a) : std::abs(b - a);
                    if (rel > maxRel) maxRel = rel;
                }
            }
        }
    }
    std::vector<ExtStressGpuImpulse> m_auditImpulses;
#endif

    /// Device impulses usable BY INDEX: the last real solve ran there, and no
    /// bond removal has been queued since.
    ///
    /// Both halves are load-bearing. removeBond only RECORDS its
    /// swap-with-last and replays it inside the next solve, while the host
    /// applies its own replaceWithLast immediately -- so between a tick that
    /// broke bonds and the next real solve the two impulse arrays are indexed
    /// differently. Reading device impulses by host bond index there does not
    /// diverge slightly, it reads a DIFFERENT BOND: measured 15.8% of group
    /// stresses differing bit for bit, max relative difference 120.
    ///
    /// Demanding a solve THIS tick instead is also sound, and throws away most
    /// of the benefit: at grid 2 the settled structures early-out of the solve
    /// nearly every tick, and they are exactly the ones still paying the walk
    /// in full, because it is linear in total live bonds rather than in
    /// activity. Measured under that stricter rule: 8, 2 and 1 device runs out
    /// of ~4200 ticks on three of the four structures.
    /// Tried and rejected: a narrow flushImpulsePermutation() on the solver
    /// that replays only the queued swaps, so the walk could also run at rest
    /// (where no solve ever comes to apply them). It DID unblock engagement --
    /// refusals from this predicate went 4196 -> 0 on the settled structures
    /// -- and it broke the audit, 0 mismatches becoming 8. Replaying the
    /// permutation is not sufficient on its own: the device's bond count and
    /// island partition are part of the same topology transaction, and the
    /// walk's guards read them. Unblocking at-rest needs the whole transaction
    /// applied, not the cheap part of it.
    /// Non-const: applies the queued impulse permutation if one is
    /// outstanding, because at rest no solve ever comes to do it.
    bool deviceImpulsesUsable()
    {
        if (!m_lastSolveOnDevice || m_gpuSolver == nullptr)
        {
            return false;
        }
        return m_gpuSolver->flushImpulsePermutation();
    }
#endif

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
        m_solutionSteadyFrames = 0;
        m_lastImpulses.resize(0);
    }

    void clearBonds()
    {
        m_bonds.clear();
        m_impulses.resize(0);
        m_forceColdStart = true;
    }

    void setSkipStableUnconverged(bool enabled) { m_skipStableUnconverged = enabled; }
    /// Count a frame as steady when it converged and the impulses it produced
    /// are the ones the previous frame produced.
    ///
    /// Compared relative to the largest impulse in play, so the test means the
    /// same thing for a garden wall and for a tower: a small structure's
    /// absolute impulses are tiny and an absolute epsilon would call every
    /// frame steady.
    void updateSolutionSteadiness()
    {
        const uint32_t count = m_impulses.size();
        if (!m_converged || count == 0)
        {
            // Nothing to compare against next frame: an unconverged answer is
            // not a candidate for "steady", so copying it (an O(bonds) pass,
            // ~0.5 ms per structure on the 268k-bond downtown, every tick the
            // 32-iteration budget fails to converge -- which is every tick)
            // bought nothing. Dropping the history makes the next converged
            // frame start the count from zero, exactly as before.
            m_solutionSteadyFrames = 0;
            m_lastImpulses.resize(0);
            return;
        }
        bool comparable = (m_lastImpulses.size() == count);
        float worstDelta = 0.0f;
        float scale = 0.0f;
        if (comparable)
        {
            const auto len = [](float x, float y, float z) {
                return std::sqrt(x * x + y * y + z * z);
            };
            for (uint32_t i = 0; i < count; ++i)
            {
                const AngLin6& now = m_impulses[i];
                const AngLin6& before = m_lastImpulses[i];
                worstDelta = std::max(worstDelta, len(now.ang.x - before.ang.x,
                                                     now.ang.y - before.ang.y,
                                                     now.ang.z - before.ang.z));
                worstDelta = std::max(worstDelta, len(now.lin.x - before.lin.x,
                                                     now.lin.y - before.lin.y,
                                                     now.lin.z - before.lin.z));
                scale = std::max(scale, len(now.ang.x, now.ang.y, now.ang.z));
                scale = std::max(scale, len(now.lin.x, now.lin.y, now.lin.z));
            }
        }
        const bool steady = comparable
            && worstDelta <= STEADY_IMPULSE_TOLERANCE * std::max(scale, 1.0e-6f);
        m_solutionSteadyFrames = steady ? m_solutionSteadyFrames + 1 : 0;
        m_lastImpulses.resize(count);
        for (uint32_t i = 0; i < count; ++i) m_lastImpulses[i] = m_impulses[i];
    }

    /// Frames the answer must hold still before the whole solve may be skipped.
    ///
    /// The garage above needed ~18 to reach equilibrium; this is not a timer
    /// for that, it is a guard against calling a moving answer settled. Four
    /// consecutive frames within tolerance costs a few frames of solving on a
    /// static structure and nothing after that.
    static constexpr uint32_t STEADY_FRAMES_BEFORE_SKIP = 4;

    /// Relative movement below which two consecutive solves count as the same
    /// answer.
    static constexpr float STEADY_IMPULSE_TOLERANCE = 1.0e-3f;

    void solve(uint32_t iterationCount, bool warmStart = true, bool islandAware = false, bool skipSettled = false)
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        m_gpuFrameSolveMilliseconds = 0.0f;
        m_gpuFrameHostWorkMilliseconds = 0.0f;
        m_bondDirty.clear();
        m_gpuImpulseCopyMilliseconds = 0.0f;
        m_gpuImpulseCopyCount = 0;
        m_gpuFrameHostBlockedMilliseconds = 0.0f;
        m_gpuFrameHostToDeviceBytes = 0;
        m_gpuFrameDeviceToHostBytes = 0;
#endif
        // Skipping the whole solve is only sound once the ANSWER has stopped
        // moving, which is not the same as this frame reporting converged.
        //
        // The solve is warm-started and capped at a handful of iterations per
        // frame, so it walks toward equilibrium over many frames. `converged`
        // means "the residual is under tolerance for the system as posed this
        // frame"; a supported structure has near-zero node velocities from the
        // very first tick, so that can be true long before the impulses have
        // finished growing. Skipping on it freezes a partial answer forever.
        //
        // Measured on a five-storey parking garage, peak bond utilisation by
        // tick, with the two behaviours side by side:
        //
        //   skipping on `converged`   0.076 0.077 0.077 0.077 ... 0.077 forever
        //   always solving            0.076 0.083 0.152 0.272 ... 0.463 rising
        //
        // The frozen value is six times too low and never recovers. Downstream
        // that means a building cannot notice it is overloaded: cut 60% of that
        // garage's columns and it reported zero stress, zero damage and did not
        // move, which is not strength -- it is a structure that stopped being
        // solved on its second tick.
        //
        // So the early-out now needs a run of consecutive converged frames
        // during which the impulses barely changed. A genuinely static
        // structure reaches that in a few frames and is skipped from then on,
        // which is the whole point of the optimisation; one still settling
        // keeps solving until it has actually settled.
        if (skipSettled
            && warmStart
            && m_converged
            && m_solutionSteadyFrames >= STEADY_FRAMES_BEFORE_SKIP
            && !m_forceColdStart
            && !m_inputsChanged)
        {
            m_error_sq = {0.0f, 0.0f};
            return;
        }
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        // Sticky across the settled early-out on purpose: that path solves
        // nothing, so it cannot invalidate what the device holds. What DOES
        // invalidate it is a queued topology change, tested separately -- see
        // deviceImpulsesUsable().
        m_lastSolveOnDevice = false;
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
            // The same bending policy the host walk uses. Read once here
            // because the device cannot read an environment variable, and
            // because host and device disagreeing about bending is exactly
            // what the shared formula header exists to prevent.
            gpuParams.bendGainMax =
                sectionModulusBending() ? maxBendAmplification() : 0.0f;
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
            // Same condition as the whole-solve early-out, and for the same
            // reason: the device's own settled-island skip returns the
            // previous impulses untouched, so letting it run before the answer
            // has steadied both freezes the answer AND makes the steadiness
            // test above see no change and agree. The two have to be gated
            // together or the cheaper one quietly re-creates the bug.
            //
            //
            // The device skip is PER ISLAND and converged-only (planSettledSkip,
            // since the depth-truncation fix): an island is retired only once
            // its own residual is under tolerance and its inputs are
            // bit-identical, so it cannot freeze a moving answer. Gating it on
            // the WHOLE solve having converged for STEADY_FRAMES_BEFORE_SKIP
            // frames threw that away: at the 32-iteration budget the whole
            // solve never converges, so the gate never opened and not one
            // island was ever skipped (0% on a standing city, against 45%
            // before the gate existed). The whole-solve early-out above keeps
            // the gate; the per-island skip does not need it.
            gpuParams.skipSettledIslands = skipSettled && warmStart;
            if (std::getenv("BLAST_WARMSTART_TRACE"))
            {
                static uint32_t traceTick = 0;
                if (traceTick < 40)
                {
                    std::fprintf(stderr,
                                 "[warmstart] tick=%u warmStart=%d forceCold=%d "
                                 "converged=%d inputsChanged=%d iters=%u\n",
                                 traceTick, int(gpuParams.warmStart),
                                 int(m_forceColdStart), int(m_converged),
                                 int(m_inputsChanged), unsigned(iterationCount));
                }
                ++traceTick;
            }
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
                // Dirty mask over solver bonds, from the same compacted list
                // the impulse copy uses. A bond absent from it kept the
                // impulses it already had, so anything derived from those
                // impulses is unchanged too.
                const bool compactedList =
                    changed != nullptr && changedCount < impulseCount;
                m_bondDirty.assign(impulseCount, compactedList ? 0u : 1u);
                if (compactedList)
                {
                    for (uint32_t k = 0; k < changedCount; ++k)
                    {
                        if (changed[k] < m_bondDirty.size())
                        {
                            m_bondDirty[changed[k]] = 1u;
                        }
                    }
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
                updateSolutionSteadiness();
                m_forceColdStart = false;
                m_inputsChanged = false;
                m_lastSolveOnDevice = true;
                ++m_solveSerial;
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
        updateSolutionSteadiness();
        m_forceColdStart = false;
        m_inputsChanged = false;
        m_lastSolveOnDevice = false;
        ++m_solveSerial;
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
    bool                        m_lastSolveOnDevice{false};
    uint64_t                    m_solveSerial{0};
    POD_Buffer<AngLin6>         m_impulses;
    /// Per-solver-bond dirty flags; see bondDirty().
    std::vector<uint8_t>        m_bondDirty;
    AngLin6ErrorSq              m_error_sq;
    bool                        m_converged;
    bool                        m_forceColdStart;
    bool m_skipStableUnconverged = false;
    bool                        m_inputsChanged;
    /// Consecutive frames whose solve converged AND barely moved the impulses.
    /// See the early-out in solve() for why converged alone is not enough.
    uint32_t                    m_solutionSteadyFrames = 0;
    /// Impulses as of the previous solve, for measuring that movement.
    ///
    /// POD_Buffer, matching m_impulses: AngLin6 carries 16-byte-aligned
    /// members and a generic Array does not honour that, which is a segfault
    /// the first time the SIMD path touches it.
    POD_Buffer<AngLin6>         m_lastImpulses;
    // Borrowed from the owning solver: resolved material table for the GPU
    // damage-kernel seed. Not consumed by the CPU solve.
    const ExtStressMaterial*    m_materials{nullptr};
    uint32_t                    m_materialCount{0};
    // Declared unconditionally: getGpuInactiveReason() and the non-CUDA branch
    // of setGpuAccelerated() both report through it, and both exist in builds
    // without NVBLAST_ENABLE_CUDA_STRESS -- where "why is the GPU not running"
    // is exactly the question a caller needs answered.
    const char*                         m_gpuInactiveReason{
        "not requested (setGpuAccelerated was never called with true)"};
    char                                m_gpuReasonBuf[256]{};
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
        // Bending, as an equivalent pressure, always >= 0.
        //
        // Kept apart from stressNormal rather than folded into it because
        // bending is not axial load: it puts one face of the joint in tension
        // and the opposite face in compression at the same time. Only the
        // extreme fibres decide whether the joint fails, and which limit they
        // are checked against depends on their sign -- so the two have to
        // survive as separate numbers this far.
        float    stressBend;

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
        // Real rotational inertia (kg m^2) from the chunk's actual shape, or 0
        // to fall back to the sphere approximation. A building is made of
        // flat, wide things; a slab's inertia about its flat axis and about
        // its edge differ by an order of magnitude, and the sphere splits the
        // difference wrongly for exactly those pieces.
        float geometricInertia;

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
        float geometricInertia;
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
    const char* getGpuInactiveReason() const { return m_solver.getGpuInactiveReason(); }
    float getGpuSolveMilliseconds() const { return m_solver.getGpuSolveMilliseconds(); }
    float getGpuHostWorkMilliseconds() const { return m_solver.getGpuHostWorkMilliseconds(); }
    float getGpuImpulseCopyMilliseconds() const { return m_solver.getGpuImpulseCopyMilliseconds(); }
    uint32_t getGpuImpulseCopyCount() const { return m_solver.getGpuImpulseCopyCount(); }
    float getGpuHostBlockedMilliseconds() const { return m_solver.getGpuHostBlockedMilliseconds(); }
    uint64_t getGpuHostToDeviceBytes() const { return m_solver.getGpuHostToDeviceBytes(); }
    uint64_t getGpuDeviceToHostBytes() const { return m_solver.getGpuDeviceToHostBytes(); }

    void calcSolverBondStresses(
        uint32_t bondIdx, float bondArea, float nodeDist, const nvidia::NvVec3& bondNormal,
        float& stressNormal, float& stressShear, float& stressBend) const
    {
        if (!canTakeDamage(bondArea))
        {
            stressNormal = stressShear = stressBend = 0.0f;
            return;
        }

        // One body for this equation, shared with the device kernel -- see
        // NvBlastExtStressFormula.h for why, and for the two deliberate
        // differences from the form that used to be inlined here.
        NvVec3 impulseLinear, impulseAngular;
        getSolverInternalBondImpulses(bondIdx, impulseLinear, impulseAngular);
        // The shared formula, so the host and the device solve the same
        // equation -- upstream measured 32% of stress values differing between
        // them before this existed. Bending policy is passed in because the
        // body compiles for the device and cannot read an environment variable.
        extStressCalcBondStress(
            ExtStressVec3{impulseLinear.x, impulseLinear.y, impulseLinear.z},
            ExtStressVec3{impulseAngular.x, impulseAngular.y, impulseAngular.z},
            ExtStressVec3{bondNormal.x, bondNormal.y, bondNormal.z},
            bondArea, nodeDist,
            sectionModulusBending() ? maxBendAmplification() : 0.0f,
            stressNormal, stressShear, stressBend);
    }

    static void fibreStresses(float stressNormal, float stressBend,
                              float& compression, float& tension)
    {
        if (!fibreBending())
        {
            // Legacy: bend amplifies whatever the axial sign already is.
            const float combined = stressNormal + std::copysign(stressBend, stressNormal);
            compression = combined <= 0.0f ? -combined : 0.0f;
            tension = combined > 0.0f ? combined : 0.0f;
            return;
        }
        tension = std::max(0.0f, stressNormal + stressBend);
        compression = std::max(0.0f, stressBend - stressNormal);
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

    void setNodeGeometricInertia(uint32_t node, float inertia)
    {
        m_nodesData[node].geometricInertia = inertia > 0.0f ? inertia : 0.0f;
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
            // The ONLY place localVel is ever written. That is what lets the
            // walk-in below skip itself on a quiet tick.
            m_velocitiesTouched = true;
            m_localVelDirty = true;
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
                    // Same edit against the flattened mirror, so the device
                    // walk keeps seeing the membership the host walk sees.
                    bondStressCsrRemoveMember(solverBondIndex, blastBondIndex);
                    if (blastBondIndices.empty())
                    {
                        // all bonds associated with this solver bond were removed, so let's remove solver bond

                        m_solverBondsData.replaceWithLast(solverBondIndex);
                        bondStressCsrRemoveGroup(solverBondIndex);
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

    /// Default ON. Off-switch for the quiet-tick walk-in skip, so a bad soak
    /// is an env change rather than a rebuild.
    static bool walkInSkipEnabled()
    {
        static const bool enabled = [] {
            const char* raw = std::getenv("BLAST_WALKIN_SKIP");
            return raw == nullptr || std::string(raw)[0] != '0';
        }();
        return enabled;
    }

    void solve(const ExtStressSolverSettings& settings, const float* bondHealth, const NvBlastBond* bonds, bool warmStart = true, bool islandAware = false, bool skipSettled = false, float deltaTime = 0.0f)
    {
        // The 11.2 ms that looked like an unattributed remainder is in here,
        // not in the GPU call. Three host walks bracket the solve: a node
        // walk in, and two per-bond/per-node walks out.
        using SolveClock = std::chrono::steady_clock;
        const auto walkMs = [](SolveClock::time_point from) {
            return std::chrono::duration<float, std::milli>(
                       SolveClock::now() - from).count();
        };

        const auto syncStart = SolveClock::now();
        sync(bonds, islandAware);
        // The walk-in used to transpose EVERY node into the solver every tick,
        // unconditionally. On an idle city that is the single largest cost in
        // the whole tick -- 0.93 ms of a 3.65 ms idle tick, writing the same
        // zeros over ~87,000 nodes -- because nothing has moved and nothing
        // will read the result.
        //
        // It can be skipped outright, and the condition is exact rather than
        // heuristic. addNodeForce is the ONLY writer of localVel, and
        // resetVelocities() zeroes localVel after every solve. So if no force
        // was applied this tick AND none was applied last tick, then every
        // localVel is zero and the solver's m_velocities were already set to
        // zero by last tick's walk. Writing them again cannot change anything.
        //
        // Retaining last tick's velocities is the POINT, not a hazard, and
        // getting that backwards cost 15x. An earlier version also required
        // the previous tick to be quiet, reasoning that leaving stale non-zero
        // velocities standing would be a bug. But the caller only stops
        // applying forces when it has established that nothing moved, and in
        // that case last tick's velocities are exactly the right inputs. With
        // the two-tick rule the first quiet tick still walked in and wrote the
        // zeros that the skipped gravity had left behind -- which unloaded
        // every structure, woke the islands the solver had settled, and turned
        // a 3.7 ms idle tick into 54 ms.
        //
        // So: quiet means leave the solver's inputs alone. addNodeForce is the
        // only writer of localVel, so "quiet" is exact -- if nothing called it,
        // there is nothing to transpose.
        //
        // A node-count change means the mapping to solver nodes moved under
        // us, so that tick always walks in full.
        const bool nodeCountStable = (m_nodesData.size() == m_walkInNodeCount);
        const bool quietNow = !m_velocitiesTouched;
        if (!(walkInSkipEnabled() && quietNow && nodeCountStable))
        {
            for (const NodeData& node : m_nodesData)
            {
                m_solver.setNodeVelocities(node.solverNode, node.localVel, NvVec3(NvZero));
            }
        }
        else
        {
            ++m_walkInSkipped;
        }
        m_walkInNodeCount = m_nodesData.size();
        m_velocitiesTouched = false;
        m_hostWalkInMilliseconds = walkMs(syncStart);

        m_solver.solve(settings.maxSolverIterationsPerFrame, warmStart, islandAware, skipSettled);

        const auto outStart = SolveClock::now();
        resetVelocities();
        m_hostResetMilliseconds = walkMs(outStart);

        const auto bondStart = SolveClock::now();
        // Deferred: the CALLER drives bondStressBegin / N x bondStressStrip /
        // bondStressFinish, so the strips of every structure can be fanned out
        // in one flat top-level dispatch instead of one dispatch per slot.
        if (m_deferBondStress)
        {
            bondStressBegin(bondHealth, bonds);
            m_deferredBonds = bonds;
            m_deferredBondHealth = bondHealth;
            m_hostBondStressMilliseconds = walkMs(bondStart);
            return;
        }
        updateBondStress(bondHealth, bonds);
        m_hostBondStressMilliseconds = walkMs(bondStart);

        // Why the skip rate is what it is. Two causes need opposite fixes:
        // groups that genuinely re-solved (nothing to do about it here) and
        // groups that are unchanged but latched overstressed (their health
        // moves every tick). stderr rather than five layers of FFI, because
        // this answers one question once.
        if (std::getenv("BLAST_BOND_STRESS_STATS") != nullptr && (++m_bsTicks % 120) == 0)
        {
            const double t = m_bsTotal > 0 ? double(m_bsTotal) : 1.0;
            fprintf(stderr,
                    "[bond-stress] groups/tick %.0f | dirty %.1f%% | "
                    "overstress-blocked %.1f%% | SKIPPED %.1f%%\n",
                    t / 120.0,
                    100.0 * double(m_bsDirty) / t,
                    100.0 * double(m_bsOverstressBlocked) / t,
                    100.0 * double(m_bondStressGroupsSkipped - m_bsSkipMark) / t);
            m_bsTotal = m_bsDirty = m_bsOverstressBlocked = 0;
            m_bsSkipMark = m_bondStressGroupsSkipped;
        }

        // Per-chunk stress runs after the bonds because it consumes the same
        // solved impulses. Skipped entirely when no material enables crush.
        const auto nodeStart = SolveClock::now();
        updateNodeStress(bondHealth, deltaTime);
        m_hostNodeStressMilliseconds = walkMs(nodeStart);
    }

    /// A/B for the unchanged-bond-stress skip (default OFF until audited).
    /// Permanently default OFF, decided on measurement (2026-08-28): on top
    /// of the parallel walk this saves 6.4% of a 1.1 ms cost -- 0.07 ms.
    /// During demolition 41-98% of groups genuinely re-solve, so the settled
    /// fraction is smallest exactly when the walk costs most; this cannot
    /// grow into a win. Exact and audited (357/357/357), kept for re-test.
    static bool skipUnchangedBondStress()
    {
        static const bool enabled = [] {
            const char* raw = std::getenv("BLAST_SKIP_UNCHANGED_BOND_STRESS");
            return raw != nullptr && std::string(raw) != "0";
        }();
        return enabled;
    }

    /// Device bond-stress walk (Step 4). Default OFF.
    static bool bondStressGpu()
    {
        static const bool enabled = [] {
            const char* raw = std::getenv("BLAST_BOND_STRESS_GPU");
            return raw != nullptr && std::string(raw) != "0";
        }();
        return enabled;
    }

    /// Dual-run audit of the device walk against the serial one. Default OFF.
    static bool bondStressGpuVerify()
    {
        static const bool enabled = [] {
            const char* raw = std::getenv("BLAST_BOND_STRESS_GPU_VERIFY");
            return raw != nullptr && std::string(raw) != "0";
        }();
        return enabled;
    }

    /// Audit of the flattened mirror itself against m_solverBondsData.
    static bool bondStressCsrVerify()
    {
        static const bool enabled = [] {
            const char* raw = std::getenv("BLAST_BOND_STRESS_CSR_VERIFY");
            return raw != nullptr && std::string(raw) != "0";
        }();
        return enabled;
    }

    /// The mirror costs memory and a maintenance branch on every bond
    /// removal, so it is only built when something actually reads it. With
    /// every flag off this whole mechanism is inert.
    static bool bondStressMirrorEnabled()
    {
        static const bool enabled =
            bondStressGpu() || bondStressGpuVerify() || bondStressCsrVerify();
        return enabled;
    }

    uint64_t getBondStressGroupsSkipped() const { return m_bondStressGroupsSkipped; }
    uint64_t getBondStressParallelChecks() const { return m_bsParChecks; }
    uint64_t getBondStressParallelMismatches() const { return m_bsParMismatches; }

    bool calcError(float& linear, float& angular) const
    {
        return m_solver.calcError(linear, angular);
    }

    /// Host walks bracketing the solve. These are where the 11.2 ms that
    /// looked like an unattributed remainder actually lives.
    float getHostWalkInMilliseconds() const { return m_hostWalkInMilliseconds; }
    float getHostResetMilliseconds() const { return m_hostResetMilliseconds; }
    float getHostBondStressMilliseconds() const { return m_hostBondStressMilliseconds; }
    float getHostNodeStressMilliseconds() const { return m_hostNodeStressMilliseconds; }
    /// Per-group "was overstressed last tick", the second half of the skip
    /// condition. Starts all-1 so the first tick after a resize never skips.
    std::vector<uint8_t> m_groupOverstressed;
    uint64_t m_bondStressGroupsSkipped{0};
    uint64_t m_bsTotal{0}, m_bsDirty{0}, m_bsOverstressBlocked{0};
    uint64_t m_bsTicks{0}, m_bsSkipMark{0};
    /// Per-strip accumulators for the parallel bond-stress walk. Only these
    /// two outputs need isolating: stressNormal/stressShear are disjoint per
    /// group, and m_nodeOverstressed is set-to-1 so concurrent writes agree.
    struct BondStressStrip
    {
        Array<uint32_t>::type remove;
        uint32_t count = 0;
    };
    /// Fixed strip width, so a flat (slot x strip) dispatch has a stable
    /// shape and the merge order is deterministic regardless of dispatcher.
    static const uint32_t kBondStressStrips = 16;
    /// Inputs stashed by bondStressBegin for the strip phase.
    const float* m_bsBondHealth{nullptr};
    const NvBlastBond* m_bsBonds{nullptr};
    const std::vector<uint8_t>* m_bsDirtyPtr{nullptr};
    bool m_bsHaveDirty{false};
    uint32_t m_bsGroupCount{0};
    bool m_deferBondStress{false};
    const NvBlastBond* m_deferredBonds{nullptr};
    const float* m_deferredBondHealth{nullptr};
    /// Reused across ticks so the strip vectors keep their capacity.
    std::vector<BondStressStrip> m_bondStressStrips;
    /// Verify-mode scratch and audit counters. Mismatches must be zero.
    BondStressStrip m_verifySerialStrip;

    // ── Flattened mirror of m_solverBondsData, for the device bond-stress walk ──
    //
    // Every per-member INPUT to the walk is static between graph resyncs. Node
    // positions only move through setNodeInfo, which sets m_nodesDirty and so
    // forces a resync; that makes nodeDisp -- and the sign-aligned bond normal
    // derived from it -- fixed for the lifetime of this table. Only bond HEALTH
    // and the solved impulses change per tick, and health arrives as a flat
    // array already indexed by blast bond, so it needs no gather.
    //
    // Between resyncs groups only ever SHRINK, and by exactly the two edits the
    // host applies to m_solverBondsData:
    //   member removed -> findAndReplaceWithLast inside its group
    //   group emptied  -> replaceWithLast over the group table
    // Neither moves a member RANGE, because (begin, size) is stored per group
    // and it is those entries that get swapped.
    //
    // The payload is indexed by BLAST BOND rather than by member slot, and that
    // is the point of the layout. Blast bond indices are asset-level and are
    // never permuted, so the node pair, material, normal, centroid and node
    // displacement are written once per resync and never touched again. What a
    // bond removal edits is then only the CSR -- three uint32 arrays -- which
    // is small enough to refresh on the device wholesale, instead of needing
    // its own incremental scatter that could silently fall out of step.
    //
    // Slot order within a group is the order the device must emit removals in,
    // and it is NOT sorted -- see the note on bondStressFinish.
    std::vector<uint32_t> m_bsCsrGroupBegin;
    std::vector<uint32_t> m_bsCsrGroupSize;
    std::vector<uint32_t> m_bsCsrMemberBlastBond;
    // Static per blast bond, valid until the next resync.
    std::vector<uint32_t> m_bsBondNode0;
    std::vector<uint32_t> m_bsBondNode1;
    std::vector<uint32_t> m_bsBondMaterial;
    std::vector<float>    m_bsBondNormal;    // 3 floats per blast bond
    std::vector<float>    m_bsBondCentroid;  // 3 floats per blast bond
    std::vector<float>    m_bsBondNodeDisp;  // 3 floats per blast bond
    /// blast bond -> owning group, or InvalidIndex. Lets the device map its
    /// per-group outputs back without the host walking anything.
    std::vector<uint32_t> m_bsBlastBondGroup;
    std::vector<float> m_bsMaterialLimits;
    std::vector<uint32_t> m_bsGroupSwapDst;
    std::vector<uint32_t> m_bsGroupSwapSrc;
    /// The inputs the cached answer belongs to, and the answer.
    std::vector<float> m_bsHealthShadow;
    std::vector<float> m_bsMaterialShadow;
    std::vector<uint8_t> m_bsCacheMask;
    std::vector<uint32_t> m_bsCacheRemovals;
    uint64_t m_bsCacheSolveSerial{~0ull};
    uint32_t m_bsCacheOverstressed{0};
    bool m_bsCacheValid{false};
    uint64_t m_bsSkippedLaunches{0};
    /// Set by anything that invalidates the mirror; cleared by a rebuild.
    bool m_bsCsrDirty{true};
    /// The CSR changed since the device last took a copy.
    bool m_bsCsrUploadDirty{true};
    /// Audit counters. Explicit initialisers: a telemetry field without one
    /// has already reported 4.5e18 checks once.
    uint64_t m_bsCsrChecks{0};
    uint64_t m_bsCsrMismatches{0};
    uint64_t m_bsCsrRebuilds{0};
    uint64_t m_bsCsrTicks{0};
    /// Which rebuild generation the device currently holds.
    uint64_t m_bsTopologyUploaded{~0ull};
    /// Set for the tick when the device walk produced this tick's outputs.
    bool m_bsGpuActive{false};
    mutable const float* m_bsGpuStressNormal{nullptr};
    mutable const float* m_bsGpuStressShear{nullptr};
    mutable const float* m_bsGpuNormal{nullptr};
    mutable const float* m_bsGpuCentroid{nullptr};
    std::vector<uint32_t> m_bsGpuRemovals;
    std::vector<uint32_t> m_bsVerifyRemovals;
    std::vector<uint32_t> m_bsSerialRemovals;
    std::vector<uint8_t> m_bsVerifyMask;
    uint64_t m_bsGpuChecks{0};
    uint64_t m_bsGpuMismatches{0};
    uint64_t m_bsGpuMismCount{0};
    uint64_t m_bsGpuMismRemoveSize{0};
    uint64_t m_bsGpuMismRemoveOrder{0};
    uint64_t m_bsGpuMismMask{0};
    uint64_t m_bsGpuFirstBadTick{0};
    uint64_t m_bsGpuLastBadTick{0};
    uint64_t m_bsGpuAttempts{0};
    uint64_t m_bsGpuRuns{0};
    double m_bsPhUpload{0.0};
    double m_bsPhKernel{0.0};
    double m_bsPhRead{0.0};
    double m_bsPhSync{0.0};
    double m_bsPhHost{0.0};
    double m_bsPhPrep{0.0};
    double m_bsPhEnq{0.0};
    double m_bsPrEmpty{0.0};
    double m_bsPrKernel{0.0};
    double m_bsPrKernel2{0.0};
    double m_bsPrCopy{0.0};
    uint64_t m_bsPhUp{0};
    uint64_t m_bsPhDown{0};
    uint64_t m_bsPhCalls{0};
    uint64_t m_bsRefuseNoSolver{0};
    uint64_t m_bsRefuseStale{0};
    uint64_t m_bsRefusePendingTopo{0};
    uint64_t m_bsRefuseCsr{0};
    uint64_t m_bsGpuStressChecks{0};
    uint64_t m_bsGpuStressMismatches{0};
    uint64_t m_bsGpuStressBig{0};
    double m_bsGpuStressMaxRel{0.0};
    uint64_t m_bsImpChecks{0};
    uint64_t m_bsImpMismatches{0};
    double m_bsImpMaxRel{0.0};
    uint64_t m_bsPayChecks{0};
    uint64_t m_bsPayNode{0};
    uint64_t m_bsPayNormal{0};
    uint64_t m_bsPayDisp{0};
    uint64_t m_bsPayCentroid{0};
    uint32_t m_bsDumped{0};
    uint64_t m_bsParChecks{0};
    uint64_t m_bsParMismatches{0};
    float m_hostWalkInMilliseconds{0.0f};
    // Walk-in skip state. See the comment at the walk-in in solve().
    bool m_velocitiesTouched{true};
    bool m_localVelDirty{true};
    size_t m_walkInNodeCount{0};
    uint64_t m_walkInSkipped{0};
    float m_hostResetMilliseconds{0.0f};
    float m_hostBondStressMilliseconds{0.0f};
    float m_hostNodeStressMilliseconds{0.0f};

    bool getBondStress(uint32_t blastBondIndex, float& compression, float& tension, float& shear) const
    {
        float stressNormal, stressShear;
        nvidia::NvVec3 normal, centroid;
        if (!bondStressOutputs(blastBondIndex, stressNormal, stressShear, normal, centroid))
        {
            return false;
        }

        // Axial load is one or the other -- compression resists two nodes being
        // pushed together, tension resists them being pulled apart -- but a
        // bending moment produces both at once, on opposite faces of the joint.
        // These are therefore the two extreme fibres, and either can be the one
        // that fails.
        fibreStresses(m_bondsData[blastBondIndex].stressNormal,
                      m_bondsData[blastBondIndex].stressBend, compression, tension);

        // shear is independent and can co-exist with compression and tension
        shear = stressShear;         // the force perpendicular to the bond normal direction

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
        // Zeroing localVel is only necessary if something wrote it, and
        // addNodeForce is the only thing that ever does. On a quiet tick they
        // are all already zero, and this pass is a full sweep of ~87,000 nodes
        // to write zeros over zeros -- which became the single largest idle
        // cost (1.07 ms) once gravity and the walk-in stopped dominating.
        if (!m_localVelDirty)
        {
            return;
        }
        for (auto& node : m_nodesData)
        {
            node.localVel = NvVec3(NvZero);
        }
        m_localVelDirty = false;
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
                // In GPU mode the walk never wrote BondData, so the centroid
                // comes from the per-group device output instead.
                float ignoredNormalStress, ignoredShear;
                NvVec3 ignoredNormal, centroid = bond.centroid;
                bondStressOutputs(
                    blastBondIndex, ignoredNormalStress, ignoredShear, ignoredNormal, centroid);

                if (materialForNode(bond.node0).crush.enabled())
                {
                    accumulateVirial(bond.node0, centroid - m_nodesData[bond.node0].localPos, force);
                }
                if (materialForNode(bond.node1).crush.enabled())
                {
                    accumulateVirial(bond.node1, centroid - m_nodesData[bond.node1].localPos, -force);
                }
            }
        }
    }

    /**
    Per-bond stress from the solved impulses. THE largest named cost in the
    tick: 9.90 ms at grid 2, 64.5% of the graph solve, ~7x the GPU kernel it
    post-processes.

    Measured properties, so the next change starts from evidence:

      - Linear in TOTAL live bonds, not in activity. 2.41 ms at 74.5k bonds,
        9.90 ms at 268k -- 4.10x for 3.60x. Halving awake bodies does not
        move it.
      - The unchanged-stress skip (BLAST_SKIP_UNCHANGED_BOND_STRESS) removes
        only 6.5%, because during demolition 41-98% of groups genuinely
        re-solve. Skipping is the wrong lever; the ceiling is the settled
        fraction and it is small when it matters.
      - The parallel fan-out above this is over STRUCTURES, and there are
        four. Measured concurrency 2.21x on a 32-core box, so ~26 cores are
        idle while this runs.

    The two viable fixes, and what each needs:

    1. Parallelise this loop. Biggest available win and the cheaper of the
       two, but this library has no task dispatcher -- the pool lives in the
       caller (vibe-land-4 physx-bridge, stress_executor_). Needs a
       dispatch hook plumbed caller -> adapter -> here.

       Bit-exactness needs care in exactly one place. Per-bond writes
       (stressNormal/stressShear) are independent; m_nodeOverstressed is
       set-to-1 only so races are benign; m_overstressedBondCount is a sum.
       But bondIndicesToRemove is ORDER-SENSITIVE -- removal order feeds
       back into topology -- so it must be per-strip local vectors
       concatenated in strip order, exactly as resolve_support_loads does
       for its supporter ingest.

    2. Move it to CUDA. The impulses are already on the device and are
       copied back specifically so this walk can run on the host. Needs six
       host structures made device-resident and coherent: bondHealth, the
       asset NvBlastBond array, m_bondsData, m_nodesData positions,
       m_blastBondIndexMap, and the material table, plus the ragged
       per-group blastBondIndices as a CSR rebuilt on topology change.

    Either way the equivalence harness is the gate: it compares broken-bond
    counts per tick and caught a 0.0975% divergence in the flat-bondless
    change that every other gate passed.
    */
public:
    /// Phase 1 of the bond-stress walk: reset outputs, stash the inputs the
    /// strip phase needs. Split so the CALLER can fan strips out flatly over
    /// (slot x strip) in ONE top-level dispatch. The per-slot dispatch that
    /// preceded this measured ~2x worse at rest -- mutex serialisation plus
    /// 2:1 thread oversubscription from a second pool.
    void bondStressBegin(const float* bondHealth, const NvBlastBond* bonds)
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

        // Skip groups whose stress provably cannot have changed.
        //
        // This walk is 9.90 ms -- 64.5% of the graph solve and 24.7% of the
        // whole Blast tick, roughly 7x the GPU kernel it post-processes --
        // and it runs over every solver bond every tick regardless of what
        // moved. The solver already knows what moved: bondDirty() is the same
        // compacted list the impulse copy uses.
        //
        // The skip is exact, not approximate, and needs BOTH conditions:
        //
        //   not dirty        -- the impulses this group's stress is computed
        //                       from are byte-identical to last tick's, so
        //                       recomputing yields the same stressNormal and
        //                       stressShear it already holds.
        //   not overstressed -- an overstressed bond takes damage every tick,
        //                       so its HEALTH changes even when its impulse
        //                       does not, and health is the other input.
        //                       Skipping one would freeze a bond mid-failure.
        //
        // A group meeting both contributes nothing to m_overstressedBondCount
        // or m_nodeOverstressed (it was not overstressed) and nothing to
        // bondIndicesToRemove (unchanged health cannot cross a threshold it
        // did not cross last tick), so the reset-and-rebuild above stays
        // correct without re-applying anything for it.
        // Resize BEFORE the check that reads the size, not after. As written
        // the other way round, the very first tick at any new group count saw
        // m_groupOverstressed.size() == 0, failed the guard, and disabled the
        // skip for that tick -- and after a fracture that is every tick that
        // matters. Measured: dirty+blocked+skipped summed to 15-65%, never
        // 100%, and the missing share was ticks where the guard was false and
        // every group took the full walk.
        //
        // New groups start latched overstressed (1u), so a group that has
        // never been evaluated is never skipped.
        m_groupOverstressed.resize(m_solverBondsData.size(), 1u);
        m_bsBondHealth = bondHealth;
        m_bsBonds = bonds;
        m_bsDirtyPtr = &m_solver.bondDirty();
        m_bsHaveDirty = skipUnchangedBondStress()
            && m_bsDirtyPtr->size() == m_solverBondsData.size();
        m_bsGroupCount = static_cast<uint32_t>(m_solverBondsData.size());

        // A fallback tick must not leave the previous tick's device pointers
        // looking live.
        m_bsGpuActive = false;

        // Keep the flattened mirror in step before anything reads it.
        if (bondStressMirrorEnabled())
        {
            if (m_bsCsrDirty)
            {
                rebuildBondStressCsr(bonds);
                ++m_bsCsrRebuilds;
            }
            ++m_bsCsrTicks;
        }
        if (bondStressGpu() && (m_bsCsrTicks % 600) == 0 && m_bsCsrTicks > 0)
        {
            if (m_bsPhCalls > 0)
            {
                const double c = static_cast<double>(m_bsPhCalls);
                fprintf(stderr,
                        "[bs-phase] calls=%llu per-call ms: upload %.3f kernel %.3f "
                        "readback %.3f | prep %.3f enqueue %.3f sync %.3f | host total %.3f "
                        "| skipped %llu of %llu | up %.2f MB down %.2f MB | PROBE empty %.3f nullkernel1 %.3f nullkernel2 %.3f copy4B %.3f\n",
                        (unsigned long long)m_bsPhCalls,
                        m_bsPhUpload / c, m_bsPhKernel / c, m_bsPhRead / c,
                        m_bsPhPrep / c, m_bsPhEnq / c, m_bsPhSync / c, m_bsPhHost / c,
                        (unsigned long long)m_bsSkippedLaunches,
                        (unsigned long long)(m_bsSkippedLaunches + m_bsPhCalls),
                        double(m_bsPhUp) / c / 1048576.0,
                        double(m_bsPhDown) / c / 1048576.0,
                        m_bsPrEmpty / c, m_bsPrKernel / c, m_bsPrKernel2 / c, m_bsPrCopy / c);
            }
            fprintf(stderr,
                    "[bond-stress-gpu] ticks=%llu attempts=%llu ranOnDevice=%llu "
                    "checks=%llu MISMATCHES=%llu "
                    "(count=%llu removeSize=%llu removeOrder=%llu mask=%llu) "
                    "firstBadTick=%llu lastBadTick=%llu refuse(noSolver=%llu stale=%llu pendingTopo=%llu csr=%llu) STRESS_MISMATCH=%llu/%llu big=%llu maxRel=%.3e IMPULSE=%llu/%llu impMaxRel=%.3e PAYLOAD(n=%llu node=%llu normal=%llu disp=%llu centroid=%llu)\n",
                    (unsigned long long)m_bsCsrTicks,
                    (unsigned long long)m_bsGpuAttempts,
                    (unsigned long long)m_bsGpuRuns,
                    (unsigned long long)m_bsGpuChecks,
                    (unsigned long long)m_bsGpuMismatches,
                    (unsigned long long)m_bsGpuMismCount,
                    (unsigned long long)m_bsGpuMismRemoveSize,
                    (unsigned long long)m_bsGpuMismRemoveOrder,
                    (unsigned long long)m_bsGpuMismMask,
                    (unsigned long long)m_bsGpuFirstBadTick,
                    (unsigned long long)m_bsGpuLastBadTick,
                    (unsigned long long)m_bsRefuseNoSolver,
                    (unsigned long long)m_bsRefuseStale,
                    (unsigned long long)m_bsRefusePendingTopo,
                    (unsigned long long)m_bsRefuseCsr,
                    (unsigned long long)m_bsGpuStressMismatches,
                    (unsigned long long)m_bsGpuStressChecks,
                    (unsigned long long)m_bsGpuStressBig,
                    m_bsGpuStressMaxRel,
                    (unsigned long long)m_bsImpMismatches,
                    (unsigned long long)m_bsImpChecks,
                    m_bsImpMaxRel,
                    (unsigned long long)m_bsPayChecks,
                    (unsigned long long)m_bsPayNode,
                    (unsigned long long)m_bsPayNormal,
                    (unsigned long long)m_bsPayDisp,
                    (unsigned long long)m_bsPayCentroid);
        }
        if (bondStressCsrVerify())
        {
            verifyBondStressCsr();
            if ((m_bsCsrTicks % 600) == 0)
            {
                fprintf(stderr,
                        "[bond-stress-csr] ticks=%llu rebuilds=%llu (%.1f%%) "
                        "checks=%llu MISMATCHES=%llu\n",
                        (unsigned long long)m_bsCsrTicks,
                        (unsigned long long)m_bsCsrRebuilds,
                        100.0 * double(m_bsCsrRebuilds) / double(m_bsCsrTicks),
                        (unsigned long long)m_bsCsrChecks,
                        (unsigned long long)m_bsCsrMismatches);
            }
        }
        m_bondStressStrips.resize(kBondStressStrips);
        for (uint32_t sIdx = 0; sIdx < kBondStressStrips; ++sIdx)
        {
            m_bondStressStrips[sIdx].remove.clear();
            m_bondStressStrips[sIdx].count = 0;
        }
    }

    /// Phase 2: one contiguous ascending range of groups into its own strip.
    /// Pure with respect to other strips -- per-group stress writes are
    /// disjoint, m_nodeOverstressed is set-to-1 so concurrent writes agree,
    /// and counts/removals accumulate per strip.
    void bondStressStrip(uint32_t stripIdx)
    {
        if (stripIdx >= kBondStressStrips || m_bsGroupCount == 0)
        {
            return;
        }
        const uint32_t stripLen =
            (m_bsGroupCount + kBondStressStrips - 1) / kBondStressStrips;
        const uint32_t begin = stripIdx * stripLen;
        if (begin >= m_bsGroupCount)
        {
            return;
        }
        const uint32_t end = std::min(m_bsGroupCount, begin + stripLen);
        BondStressStrip& ctx = m_bondStressStrips[stripIdx];
        for (uint32_t i = begin; i < end; ++i)
        {
            processBondGroup(i, ctx);
        }
    }

    /// Phase 3: merge in STRIP ORDER, then apply removals. Strip order is
    /// serial order because the ranges are contiguous and ascending -- the
    /// property the 222M-check audit verified element by element, and the one
    /// that matters because removal order feeds back into topology.
    ///
    /// The emitted order is (group index, then member SLOT within the group).
    /// It is NOT ascending by blast bond index, and a GPU port must not assume
    /// it is:
    ///   - removeBondIfExists does blastBondIndices.findAndReplaceWithLast,
    ///     which moves the group's LAST member into the removed member's slot,
    ///     so within-group slot order is a permutation, not a sort;
    ///   - groups are built in m_bondsData order (syncBonds), and m_bondsData
    ///     is itself replaceWithLast'd, so group order is not sorted either.
    /// Measured on a grid-1 shot run: 2 of 200 non-empty removal lists came
    /// out non-ascending (5 descending adjacent pairs of 774). So compacting
    /// removals on the device and SORTING BY INDEX would reproduce a
    /// different order than this walk on ~1% of the ticks that break bonds.
    /// The order-preserving device equivalent is a stable segmented
    /// compaction: prefix-sum the per-group removal counts, then have each
    /// group write its own removals at its offset in ascending slot order.
    void bondStressFinish()
    {
        Array<uint32_t>::type& bondIndicesToRemove = m_bondIndicesToRemove;
        for (const BondStressStrip& strip : m_bondStressStrips)
        {
            m_overstressedBondCount += strip.count;
            for (uint32_t removed : strip.remove)
            {
                bondIndicesToRemove.pushBack(removed);
            }
        }

        // now that processing is done, remove any dead bonds
        for (uint32_t bondIndex : bondIndicesToRemove)
        {
            removeBondIfExists(bondIndex);
        }
    }

    /// One group. Was a lambda inside updateBondStress; promoted to a member
    /// so the three phases share it without re-capturing state.
    void processBondGroup(uint32_t i, BondStressStrip& ctx)
    {
        const float* bondHealth = m_bsBondHealth;
        const NvBlastBond* bonds = m_bsBonds;
        const std::vector<uint8_t>& dirty = *m_bsDirtyPtr;
        const bool haveDirty = m_bsHaveDirty;
        (void)bondHealth;
        (void)bonds;
        (void)dirty;
        (void)haveDirty;
            ++m_bsTotal;
            if (haveDirty)
            {
                if (dirty[i] != 0u)
                {
                    ++m_bsDirty;          // re-solved: must recompute
                }
                else if (m_groupOverstressed[i] != 0u)
                {
                    ++m_bsOverstressBlocked;  // unchanged, but taking damage
                }
            }
            if (haveDirty && dirty[i] == 0u && m_groupOverstressed[i] == 0u)
            {
                ++m_bondStressGroupsSkipped;
                return;
            }
            m_groupOverstressed[i] = 0u;
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
                    ctx.remove.pushBack(blastBondIndex);
                }
            }

            if (totalArea == 0.0f)
            {
                return;
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
            float stressNormal, stressShear, stressBend;
            calcSolverBondStresses(i, totalArea, averageNodeDisp.magnitude(), bondNormal, stressNormal,
                                   stressShear, stressBend);
            NVBLAST_ASSERT(!std::isnan(stressNormal) && !std::isnan(stressShear) && !std::isnan(stressBend));
            float fibreCompression, fibreTension;
            fibreStresses(stressNormal, stressBend, fibreCompression, fibreTension);

            // store the stress values for all the bonds involved
            for (auto blastBondIndex : blastBondIndices)
            {
                const uint32_t bondIndex = m_blastBondIndexMap[blastBondIndex];
                if (!isInvalidIndex(bondIndex) && bondHealth[blastBondIndex] > 0.0f)
                {
                    const ExtStressMaterial& material = materialForBlastBond(blastBondIndex);
                    BondData& bond = m_bondsData[bondIndex];
                    if (fibreCompression > material.compressionElasticLimit
                        || fibreTension > material.tensionElasticLimit
                        || stressShear > material.shearElasticLimit)
                    {
                        ++ctx.count;
                        // Latch the group as overstressed so next tick cannot
                        // skip it: its health will change even if its impulse
                        // does not.
                        m_groupOverstressed[i] = 1u;
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
                    bond.stressBend = stressBend;

                    // store the normal used to calc stresses so it can be used later to determine forces
                    bond.normal = bondNormal;

                    // store the bond centroid
                    bond.centroid = bondCentroid;
                }
            }
    }

    /// Rebuild the flattened mirror from m_solverBondsData. Cold path: the
    /// audit measured one rebuild per run, because the incremental edits below
    /// absorb essentially all topology churn.
    void rebuildBondStressCsr(const NvBlastBond* bonds)
    {
        const uint32_t groupCount = static_cast<uint32_t>(m_solverBondsData.size());
        const uint32_t blastBondCount = static_cast<uint32_t>(m_blastBondIndexMap.size());
        uint32_t total = 0;
        for (uint32_t g = 0; g < groupCount; ++g)
        {
            total += m_solverBondsData[g].blastBondIndices.size();
        }

        m_bsCsrGroupBegin.resize(groupCount);
        m_bsCsrGroupSize.resize(groupCount);
        m_bsCsrMemberBlastBond.resize(total);
        // 0xFFFFFFFF, not 0: the kernel uses it as the "no graph bond" sentinel
        // that stands in for the host walk's isInvalidIndex guard, and node 0
        // is a real node that would otherwise get flagged spuriously.
        m_bsBondNode0.assign(blastBondCount, invalidIndex<uint32_t>());
        m_bsBondNode1.assign(blastBondCount, invalidIndex<uint32_t>());
        m_bsBondMaterial.assign(blastBondCount, 0);
        m_bsBondNormal.assign(3 * static_cast<size_t>(blastBondCount), 0.0f);
        m_bsBondCentroid.assign(3 * static_cast<size_t>(blastBondCount), 0.0f);
        m_bsBondNodeDisp.assign(3 * static_cast<size_t>(blastBondCount), 0.0f);
        m_bsBlastBondGroup.assign(blastBondCount, invalidIndex<uint32_t>());

        uint32_t slot = 0;
        for (uint32_t g = 0; g < groupCount; ++g)
        {
            const auto& members = m_solverBondsData[g].blastBondIndices;
            m_bsCsrGroupBegin[g] = slot;
            m_bsCsrGroupSize[g] = members.size();
            for (const uint32_t blastBondIndex : members)
            {
                m_bsCsrMemberBlastBond[slot++] = blastBondIndex;
                fillBondStressBond(g, blastBondIndex, bonds);
            }
        }
        m_bsCsrDirty = false;
        m_bsCsrUploadDirty = true;
        m_bsCacheValid = false;
        m_bsGroupSwapDst.clear();
        m_bsGroupSwapSrc.clear();
    }

    /// The static payload for one blast bond. Everything here derives from
    /// asset geometry and node positions, both of which are pinned until the
    /// next resync.
    void fillBondStressBond(uint32_t group, uint32_t blastBondIndex, const NvBlastBond* bonds)
    {
        m_bsBlastBondGroup[blastBondIndex] = group;

        const uint32_t bondIndex = m_blastBondIndexMap[blastBondIndex];
        if (isInvalidIndex(bondIndex))
        {
            // No graph bond: the host walk's isInvalidIndex guard makes this
            // member contribute nothing, and a zero normal with a zero node
            // pair reproduces that on the device.
            return;
        }

        const BondData& bond = m_bondsData[bondIndex];
        m_bsBondNode0[blastBondIndex] = bond.node0;
        m_bsBondNode1[blastBondIndex] = bond.node1;
        m_bsBondMaterial[blastBondIndex] =
            m_bondMaterials ? m_bondMaterials[blastBondIndex] : 0;

        const NvBlastBond& blastBond = bonds[blastBondIndex];
        const nvidia::NvVec3 nodeDisp =
            m_nodesData[bond.node1].localPos - m_nodesData[bond.node0].localPos;
        const nvidia::NvVec3 assetNormal(
            blastBond.normal[0], blastBond.normal[1], blastBond.normal[2]);
        // The same alignment the walk does, hoisted here because it depends
        // only on asset geometry and node positions.
        const nvidia::NvVec3 aligned =
            std::copysign(1.0f, assetNormal.dot(nodeDisp)) * assetNormal;

        const size_t base = 3 * static_cast<size_t>(blastBondIndex);
        m_bsBondNormal[base + 0] = aligned.x;
        m_bsBondNormal[base + 1] = aligned.y;
        m_bsBondNormal[base + 2] = aligned.z;
        m_bsBondCentroid[base + 0] = blastBond.centroid[0];
        m_bsBondCentroid[base + 1] = blastBond.centroid[1];
        m_bsBondCentroid[base + 2] = blastBond.centroid[2];
        m_bsBondNodeDisp[base + 0] = nodeDisp.x;
        m_bsBondNodeDisp[base + 1] = nodeDisp.y;
        m_bsBondNodeDisp[base + 2] = nodeDisp.z;
    }

    /// Mirror of blastBondIndices.findAndReplaceWithLast: the group's LAST
    /// member moves into the removed member's slot. Reproducing this exactly
    /// is what keeps the device's removal order equal to the serial walk's.
    void bondStressCsrRemoveMember(uint32_t group, uint32_t blastBondIndex)
    {
        m_bsCacheValid = false;
        if (!bondStressMirrorEnabled() || m_bsCsrDirty
            || group >= m_bsCsrGroupSize.size())
        {
            return;
        }
        const uint32_t begin = m_bsCsrGroupBegin[group];
        const uint32_t size = m_bsCsrGroupSize[group];
        for (uint32_t k = 0; k < size; ++k)
        {
            if (m_bsCsrMemberBlastBond[begin + k] != blastBondIndex)
            {
                continue;
            }
            m_bsCsrMemberBlastBond[begin + k] = m_bsCsrMemberBlastBond[begin + size - 1];
            m_bsCsrGroupSize[group] = size - 1;
            m_bsCsrUploadDirty = true;
            if (blastBondIndex < m_bsBlastBondGroup.size())
            {
                m_bsBlastBondGroup[blastBondIndex] = invalidIndex<uint32_t>();
            }
            return;
        }
        // Not found: the mirror and m_solverBondsData disagree, so stop
        // trusting it rather than silently diverging.
        m_bsCsrDirty = true;
    }

    /// Mirror of m_solverBondsData.replaceWithLast. Only the (begin, size)
    /// entries move; the member ranges stay where they are.
    void bondStressCsrRemoveGroup(uint32_t group)
    {
        m_bsCacheValid = false;
        if (!bondStressMirrorEnabled() || m_bsCsrDirty
            || group >= m_bsCsrGroupSize.size())
        {
            return;
        }
        const uint32_t last = static_cast<uint32_t>(m_bsCsrGroupSize.size()) - 1;
        if (group != last)
        {
            m_bsCsrGroupBegin[group] = m_bsCsrGroupBegin[last];
            m_bsCsrGroupSize[group] = m_bsCsrGroupSize[last];
            // The group that just moved owns different members now.
            const uint32_t begin = m_bsCsrGroupBegin[group];
            for (uint32_t k = 0; k < m_bsCsrGroupSize[group]; ++k)
            {
                const uint32_t blastBondIndex = m_bsCsrMemberBlastBond[begin + k];
                if (blastBondIndex < m_bsBlastBondGroup.size())
                {
                    m_bsBlastBondGroup[blastBondIndex] = group;
                }
            }
        }
        m_bsCsrGroupBegin.pop_back();
        m_bsCsrGroupSize.pop_back();
        m_bsCsrUploadDirty = true;
        if (group != last)
        {
            m_bsGroupSwapDst.push_back(group);
            m_bsGroupSwapSrc.push_back(last);
        }
    }

    /// BLAST_BOND_STRESS_CSR_VERIFY=1: compare the incrementally-maintained
    /// mirror to m_solverBondsData, group by group and slot by slot IN ORDER.
    /// The incremental edits are the whole risk here, and "the gates are
    /// green" is not evidence about them -- a 0.0975% divergence once passed
    /// every broad gate in this repo.
    void verifyBondStressCsr()
    {
        const uint32_t groupCount = static_cast<uint32_t>(m_solverBondsData.size());
        ++m_bsCsrChecks;
        if (m_bsCsrGroupSize.size() != groupCount)
        {
            ++m_bsCsrMismatches;
            return;
        }
        for (uint32_t g = 0; g < groupCount; ++g)
        {
            const auto& members = m_solverBondsData[g].blastBondIndices;
            ++m_bsCsrChecks;
            if (m_bsCsrGroupSize[g] != members.size())
            {
                ++m_bsCsrMismatches;
                continue;
            }
            const uint32_t begin = m_bsCsrGroupBegin[g];
            for (uint32_t k = 0; k < m_bsCsrGroupSize[g]; ++k)
            {
                ++m_bsCsrChecks;
                if (m_bsCsrMemberBlastBond[begin + k] != members[k])
                {
                    ++m_bsCsrMismatches;
                }
            }
        }
    }

    uint64_t getBondStressCsrChecks() const { return m_bsCsrChecks; }
    uint64_t getBondStressCsrMismatches() const { return m_bsCsrMismatches; }

    /// Run the walk on the device, consuming the impulses already resident
    /// there. Returns false if anything is not ready, in which case the caller
    /// falls back to the serial walk.
    bool runBondStressOnDevice(const float* bondHealth)
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        ExtStressGpuSolver* gpu = m_solver.gpuSolver();
        if (gpu == nullptr) { ++m_bsRefuseNoSolver; return false; }
        if (!m_solver.lastSolveOnDevice()) { ++m_bsRefuseStale; return false; }
        if (!m_solver.deviceImpulsesUsable()) { ++m_bsRefusePendingTopo; return false; }
        if (m_bsCsrDirty || m_bsCsrGroupSize.empty()) { ++m_bsRefuseCsr; return false; }
        // The device walk has no equivalent of the unchanged-group skip, which
        // is permanently default OFF anyway. Refuse rather than quietly
        // computing something else.
        if (m_bsHaveDirty)
        {
            return false;
        }

        refreshBondStressMaterials();

        // Nothing to launch if nothing changed.
        //
        // The walk is a pure function of health, the CSR, the material limits
        // and the impulses. When none of them moved, last tick's answer is
        // still the answer, and the kernel is not worth its submission: on a
        // GPU shared with other work, getting ANY kernel onto an SM costs
        // ~0.5 ms -- a null kernel measures the same as this one -- while the
        // walk's own device time is ~0.07 ms. Skipping the launch removes the
        // whole fixed cost rather than amortising it, and at rest that is
        // every tick.
        //
        // This is the same idea as BLAST_SKIP_UNCHANGED_BOND_STRESS, which was
        // measured at only 6.4% on the CPU walk and left off. The arithmetic
        // is completely different here: on the CPU the skip saves per-bond
        // work proportional to what it skips, on the GPU it saves a fixed cost
        // that dwarfs the work.
        const uint64_t solveSerial = m_solver.solveSerial();
        const bool healthChanged =
            m_bsHealthShadow.size() != m_bsBondNode0.size()
            || memcmp(m_bsHealthShadow.data(), bondHealth,
                      sizeof(float) * m_bsHealthShadow.size()) != 0;
        const bool materialsChanged =
            m_bsMaterialShadow != m_bsMaterialLimits;
        const bool unchanged =
            m_bsCacheValid && !healthChanged && !materialsChanged
            && !m_bsCsrUploadDirty && solveSerial == m_bsCacheSolveSerial;
        if (unchanged)
        {
            m_overstressedBondCount = m_bsCacheOverstressed;
            if (!m_nodeOverstressed.empty())
            {
                memcpy(m_nodeOverstressed.begin(), m_bsCacheMask.data(),
                       m_nodeOverstressed.size());
            }
            m_bsGpuStressNormal = nullptr;
            m_bsGpuStressShear = nullptr;
            m_bsGpuNormal = nullptr;
            m_bsGpuCentroid = nullptr;
            m_bsGpuActive = true;
            m_bsGpuRemovals = m_bsCacheRemovals;
            ++m_bsSkippedLaunches;
            return true;
        }

        ExtStressGpuBondStressTopology csr{};
        fillBondStressTopology(csr);
        if (csr.groupCount == 0)
        {
            return false;
        }

        if (m_bsTopologyUploaded != m_bsCsrRebuilds)
        {
            if (!gpu->setBondStressTopology(csr))
            {
                return false;
            }
            m_bsTopologyUploaded = m_bsCsrRebuilds;
        }

        ExtStressGpuBondStressResult result{};
        if (!gpu->updateBondStress(csr, bondHealth, kUnbreakableLimit, result))
        {
            return false;
        }
        // Only once the device has actually taken them.
        m_bsCsrUploadDirty = false;
        m_bsGroupSwapDst.clear();
        m_bsGroupSwapSrc.clear();

        m_overstressedBondCount = result.overstressedBondCount;
        if (!m_nodeOverstressed.empty() && result.nodeOverstressed != nullptr)
        {
            memcpy(
                m_nodeOverstressed.begin(),
                result.nodeOverstressed,
                m_nodeOverstressed.size());
        }
        // Pinned staging owned by the solver, valid until the next call, and
        // there is exactly one call per tick.
        // Fetched on first use, like the vectors.
        m_bsGpuStressNormal = nullptr;
        m_bsGpuStressShear = nullptr;
        // Fetched on first use, not every tick -- see readbackGroupVectors.
        m_bsGpuNormal = nullptr;
        m_bsGpuCentroid = nullptr;
        m_bsGpuActive = true;

#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        {
            const ExtStressGpuTelemetry& t = gpu->telemetry();
            m_bsPhUpload += t.bondStressUploadMs;
            m_bsPhKernel += t.bondStressKernelMs;
            m_bsPhRead += t.bondStressReadbackMs;
            m_bsPhSync += t.bondStressSyncMs;
            m_bsPhHost += t.bondStressHostMs;
            m_bsPhPrep += t.bondStressPrepMs;
            m_bsPhEnq += t.bondStressEnqueueMs;
            m_bsPrEmpty += t.bondStressProbeEmptyMs;
            m_bsPrKernel += t.bondStressProbeKernelMs;
            m_bsPrKernel2 += t.bondStressProbeKernel2Ms;
            m_bsPrCopy += t.bondStressProbeCopyMs;
            m_bsPhUp += t.bondStressBytesUp;
            m_bsPhDown += t.bondStressBytesDown;
            ++m_bsPhCalls;
        }
#endif
        m_bsGpuRemovals.assign(
            result.bondIndicesToRemove,
            result.bondIndicesToRemove + result.removeCount);

        // Snapshot the inputs this answer belongs to, and the answer itself.
        m_bsHealthShadow.assign(bondHealth, bondHealth + m_bsBondNode0.size());
        m_bsMaterialShadow = m_bsMaterialLimits;
        m_bsCacheSolveSerial = solveSerial;
        m_bsCacheOverstressed = m_overstressedBondCount;
        m_bsCacheMask.assign(
            m_nodeOverstressed.begin(), m_nodeOverstressed.begin() + m_nodeOverstressed.size());
        m_bsCacheRemovals = m_bsGpuRemovals;
        m_bsCacheValid = true;
        return true;
#else
        NV_UNUSED(bondHealth);
        return false;
#endif
    }

#if defined(NVBLAST_ENABLE_CUDA_STRESS)
    void fillBondStressTopology(ExtStressGpuBondStressTopology& csr) const
    {
        csr.groupCount = static_cast<uint32_t>(m_bsCsrGroupSize.size());
        csr.memberSlotCount = static_cast<uint32_t>(m_bsCsrMemberBlastBond.size());
        csr.graphNodeCount = static_cast<uint32_t>(m_nodesData.size());
        csr.blastBondCount = static_cast<uint32_t>(m_bsBondNode0.size());
        csr.groupBegin = m_bsCsrGroupBegin.data();
        csr.groupSize = m_bsCsrGroupSize.data();
        csr.memberBlastBond = m_bsCsrMemberBlastBond.data();
        csr.bondNode0 = m_bsBondNode0.data();
        csr.bondNode1 = m_bsBondNode1.data();
        csr.bondMaterial = m_bsBondMaterial.data();
        csr.bondNormal = m_bsBondNormal.data();
        csr.bondCentroid = m_bsBondCentroid.data();
        csr.bondNodeDisp = m_bsBondNodeDisp.data();
        csr.materialElasticLimits = m_bsMaterialLimits.data();
        csr.materialCount = static_cast<uint32_t>(m_bsMaterialLimits.size() / 3);
        csr.csrDirty = m_bsCsrUploadDirty;
        csr.groupSwapDst = m_bsGroupSwapDst.empty() ? nullptr : m_bsGroupSwapDst.data();
        csr.groupSwapSrc = m_bsGroupSwapSrc.empty() ? nullptr : m_bsGroupSwapSrc.data();
        csr.groupSwapCount = static_cast<uint32_t>(m_bsGroupSwapDst.size());
    }
#endif

    /// The elastic limits the walk actually thresholds against, resolved
    /// exactly as materialForBlastBond resolves them. Rebuilt every tick: the
    /// table is a handful of entries, and it can be pushed again without a
    /// graph resync, so caching it against the rebuild generation would go
    /// stale silently.
    void refreshBondStressMaterials()
    {
        if (!m_materials || m_materialCount == 0)
        {
            // materialForBlastBond's default, resolved the same way.
            ExtStressMaterial d;
            d.tensionElasticLimit = d.compressionElasticLimit;
            d.shearElasticLimit = d.compressionElasticLimit;
            m_bsMaterialLimits.resize(3);
            m_bsMaterialLimits[0] = d.compressionElasticLimit;
            m_bsMaterialLimits[1] = d.tensionElasticLimit;
            m_bsMaterialLimits[2] = d.shearElasticLimit;
            return;
        }
        m_bsMaterialLimits.resize(3 * static_cast<size_t>(m_materialCount));
        for (uint32_t i = 0; i < m_materialCount; ++i)
        {
            m_bsMaterialLimits[3 * i + 0] = m_materials[i].compressionElasticLimit;
            m_bsMaterialLimits[3 * i + 1] = m_materials[i].tensionElasticLimit;
            m_bsMaterialLimits[3 * i + 2] = m_materials[i].shearElasticLimit;
        }
    }

    /// The walk's four outputs for one blast bond, from whichever path
    /// produced them. In GPU mode m_bondsData is never written -- avoiding
    /// that 268k-entry scatter is most of the point -- so every consumer goes
    /// through here instead of reading BondData directly.
    bool bondStressOutputs(
        uint32_t blastBondIndex, float& stressNormal, float& stressShear,
        nvidia::NvVec3& normal, nvidia::NvVec3& centroid) const
    {
        const uint32_t bondIndex = m_blastBondIndexMap[blastBondIndex];
        if (isInvalidIndex(bondIndex))
        {
            return false;
        }
        // Only "is the device path live", NOT "have the stresses been fetched".
        // Those became different questions when the readback went lazy, and
        // conflating them sent every caller down the host path to read a
        // BondData the device path never writes -- so every bond reported zero
        // stress, nothing was ever damaged, and the city stopped fracturing
        // entirely while the overstressed COUNT climbed to 4000. The dual-run
        // audit could not see it: it makes the serial path authoritative, so
        // it never exercises this branch.
        if (!m_bsGpuActive)
        {
            const BondData& bond = m_bondsData[bondIndex];
            stressNormal = bond.stressNormal;
            stressShear = bond.stressShear;
            normal = bond.normal;
            centroid = bond.centroid;
            return true;
        }
        const uint32_t group = blastBondIndex < m_bsBlastBondGroup.size()
            ? m_bsBlastBondGroup[blastBondIndex]
            : invalidIndex<uint32_t>();
        if (isInvalidIndex(group) || group >= m_bsCsrGroupSize.size())
        {
            // Internal bond (both endpoints in one solver node): the serial
            // walk never updates it either, and syncBonds left it at zero.
            stressNormal = 0.0f;
            stressShear = 0.0f;
            normal = m_bondsData[bondIndex].normal;
            centroid = m_bondsData[bondIndex].centroid;
            return true;
        }
        if (m_bsGpuStressNormal == nullptr && !fetchBondStressStresses())
        {
            const BondData& bond = m_bondsData[bondIndex];
            stressNormal = bond.stressNormal;
            stressShear = bond.stressShear;
            normal = bond.normal;
            centroid = bond.centroid;
            return true;
        }
        stressNormal = m_bsGpuStressNormal[group];
        stressShear = m_bsGpuStressShear[group];
        // The vectors cost 6 of the 8 floats per group and are wanted by
        // almost nothing on a given tick, so they are pulled across on first
        // use. Callers that only want stress never pay for them.
        if (m_bsGpuNormal == nullptr)
        {
            if (!fetchBondStressVectors())
            {
                normal = m_bondsData[bondIndex].normal;
                centroid = m_bondsData[bondIndex].centroid;
                return true;
            }
        }
        normal = nvidia::NvVec3(
            m_bsGpuNormal[3 * group + 0],
            m_bsGpuNormal[3 * group + 1],
            m_bsGpuNormal[3 * group + 2]);
        centroid = nvidia::NvVec3(
            m_bsGpuCentroid[3 * group + 0],
            m_bsGpuCentroid[3 * group + 1],
            m_bsGpuCentroid[3 * group + 2]);
        return true;
    }

    void setDeferBondStress(bool defer) { m_deferBondStress = defer; }
    uint32_t bondStressStripCount() const { return kBondStressStrips; }

    /// Completion for the deferred path: merge the strips the caller drove,
    /// then run the per-node stress the solve() tail skipped. calcError is
    /// unaffected -- it reads solver state, not bond stress.
    void bondStressComplete(float deltaTime)
    {
        bondStressFinish();
        updateNodeStress(m_deferredBondHealth, deltaTime);
    }

    /// Serial entry point: the same three phases the flat fan-out drives.
    /// One shared body on purpose -- two copies would drift, and a drift here
    /// is a physics divergence.
    void updateBondStress(const float* bondHealth, const NvBlastBond* bonds)
    {
        bondStressBegin(bondHealth, bonds);

        bool ranOnDevice = false;
        if (bondStressGpu())
        {
            ++m_bsGpuAttempts;
            ranOnDevice = runBondStressOnDevice(bondHealth);
            m_bsGpuRuns += ranOnDevice ? 1u : 0u;
        }
        if (ranOnDevice)
        {
            if (!bondStressGpuVerify())
            {
                for (const uint32_t blastBondIndex : m_bsGpuRemovals)
                {
                    removeBondIfExists(blastBondIndex);
                }
                return;
            }

            // Dual run. Snapshot what the device produced, then re-run the
            // serial walk on the SAME inputs -- health is not mutated by
            // either path, so the second run sees exactly what the first did
            // -- and compare before anything is applied.
            const uint32_t gpuCount = m_overstressedBondCount;
            m_bsVerifyMask.assign(
                m_nodeOverstressed.begin(), m_nodeOverstressed.begin() + m_nodeOverstressed.size());
            m_bsVerifyRemovals = m_bsGpuRemovals;

            bondStressBegin(bondHealth, bonds);
            for (uint32_t sIdx = 0; sIdx < kBondStressStrips; ++sIdx)
            {
                bondStressStrip(sIdx);
            }

            // The serial removal list, in strip order, which is serial order.
            m_bsSerialRemovals.clear();
            uint32_t serialCount = 0;
            for (const BondStressStrip& strip : m_bondStressStrips)
            {
                serialCount += strip.count;
                for (const uint32_t removed : strip.remove)
                {
                    m_bsSerialRemovals.push_back(removed);
                }
            }

            ++m_bsGpuChecks;
            if (gpuCount != serialCount)
            {
                ++m_bsGpuMismatches;
                ++m_bsGpuMismCount;
                if (m_bsGpuFirstBadTick == 0) m_bsGpuFirstBadTick = m_bsCsrTicks;
                m_bsGpuLastBadTick = m_bsCsrTicks;
            }
            ++m_bsGpuChecks;
            if (m_bsVerifyRemovals.size() != m_bsSerialRemovals.size())
            {
                ++m_bsGpuMismatches;
                ++m_bsGpuMismRemoveSize;
                if (m_bsGpuFirstBadTick == 0) m_bsGpuFirstBadTick = m_bsCsrTicks;
                m_bsGpuLastBadTick = m_bsCsrTicks;
            }
            else
            {
                // Element by element and IN ORDER: removal order feeds back
                // into topology, so "same set" is not sufficient.
                for (size_t k = 0; k < m_bsSerialRemovals.size(); ++k)
                {
                    ++m_bsGpuChecks;
                    if (m_bsVerifyRemovals[k] != m_bsSerialRemovals[k])
                    {
                        ++m_bsGpuMismatches;
                        ++m_bsGpuMismRemoveOrder;
                        if (m_bsGpuFirstBadTick == 0) m_bsGpuFirstBadTick = m_bsCsrTicks;
                m_bsGpuLastBadTick = m_bsCsrTicks;
                    }
                }
            }
            // Output 4: the stress values themselves, which every consumer
            // reads through bondStressOutputs. Checking the removal list, the
            // count and the node mask only establishes that the two paths
            // AGREE ABOUT WHAT CROSSED A LIMIT -- it says nothing about the
            // numbers they hand out afterwards, and those feed excess forces,
            // the virial, and every diagnostic. Compare exactly what a
            // consumer would see, for every live bond, on both paths.
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
            m_solver.auditImpulseMirror(
                m_bsImpChecks, m_bsImpMismatches, m_bsImpMaxRel);
#endif
            // The cached static payload, recomputed the way the walk does it.
            // This table is the one thing the device reads that the host
            // rebuilds from scratch every tick, so it is the obvious place for
            // a stale value to hide.
            for (uint32_t g = 0; g < m_bsCsrGroupSize.size(); ++g)
            {
                const uint32_t begin = m_bsCsrGroupBegin[g];
                for (uint32_t k = 0; k < m_bsCsrGroupSize[g]; ++k)
                {
                    const uint32_t bb = m_bsCsrMemberBlastBond[begin + k];
                    const uint32_t bondIndex = m_blastBondIndexMap[bb];
                    if (isInvalidIndex(bondIndex)) { continue; }
                    const BondData& bd = m_bondsData[bondIndex];
                    const NvBlastBond& ab = bonds[bb];
                    const nvidia::NvVec3 disp =
                        m_nodesData[bd.node1].localPos - m_nodesData[bd.node0].localPos;
                    const nvidia::NvVec3 an(ab.normal[0], ab.normal[1], ab.normal[2]);
                    const nvidia::NvVec3 aligned = std::copysign(1.0f, an.dot(disp)) * an;
                    const size_t base = 3 * static_cast<size_t>(bb);
                    ++m_bsPayChecks;
                    if (m_bsBondNode0[bb] != bd.node0 || m_bsBondNode1[bb] != bd.node1)
                    { ++m_bsPayNode; }
                    if (memcmp(&m_bsBondNormal[base], &aligned.x, 3 * sizeof(float)) != 0)
                    { ++m_bsPayNormal; }
                    if (memcmp(&m_bsBondNodeDisp[base], &disp.x, 3 * sizeof(float)) != 0)
                    { ++m_bsPayDisp; }
                    if (memcmp(&m_bsBondCentroid[base], ab.centroid, 3 * sizeof(float)) != 0)
                    { ++m_bsPayCentroid; }
                }
            }
            // The vectors are fetched lazily now, so the audit has to ask for
            // them rather than assume the tick already paid for them.
            const bool haveStresses = fetchBondStressStresses();
            const bool haveVectors = fetchBondStressVectors();
            if (!haveStresses)
            {
                m_bsGpuStressNormal = nullptr;
            }
            for (uint32_t g = 0; g < m_bsCsrGroupSize.size(); ++g)
            {
                const uint32_t begin = m_bsCsrGroupBegin[g];
                for (uint32_t k = 0; k < m_bsCsrGroupSize[g]; ++k)
                {
                    const uint32_t bb = m_bsCsrMemberBlastBond[begin + k];
                    if (!(bondHealth[bb] > 0.0f))
                    {
                        continue;
                    }
                    const uint32_t bondIndex = m_blastBondIndexMap[bb];
                    if (isInvalidIndex(bondIndex))
                    {
                        continue;
                    }
                    const BondData& host = m_bondsData[bondIndex];
                    if (m_bsGpuStressNormal == nullptr) { continue; }
                    const float devSn = m_bsGpuStressNormal[g];
                    const float devSs = m_bsGpuStressShear[g];
                    ++m_bsGpuStressChecks;
                    if (memcmp(&host.stressNormal, &devSn, sizeof(float)) != 0
                        || memcmp(&host.stressShear, &devSs, sizeof(float)) != 0)
                    {
                        ++m_bsGpuStressMismatches;
                        // Size, not just presence. A last-bit difference that
                        // never straddles a limit is a different finding from
                        // a materially different number, and the bit-compare
                        // alone cannot tell them apart.
                        const double hn = host.stressNormal, dn = devSn;
                        const double hs = host.stressShear, ds = devSs;
                        const double rn = std::abs(hn) > 1e-9
                            ? std::abs(dn - hn) / std::abs(hn) : std::abs(dn - hn);
                        const double rs = std::abs(hs) > 1e-9
                            ? std::abs(ds - hs) / std::abs(hs) : std::abs(ds - hs);
                        const double rel = rn > rs ? rn : rs;
                        if (rel > m_bsGpuStressMaxRel) m_bsGpuStressMaxRel = rel;
                        if (rel > 1e-5) ++m_bsGpuStressBig;
                        if (rel > 1e-3 && m_bsDumped < 3)
                        {
                            ++m_bsDumped;
                            float ta = 0.0f; uint32_t live = 0;
                            for (uint32_t j = 0; j < m_bsCsrGroupSize[g]; ++j)
                            {
                                const uint32_t mb = m_bsCsrMemberBlastBond[begin + j];
                                if (bondHealth[mb] > 0.0f) { ta += bondHealth[mb]; ++live; }
                            }
                            fprintf(stderr,
                                "[bs-dump] g=%u size=%u live=%u totalArea=%.6g | "
                                "host sn=%.9g ss=%.9g | dev sn=%.9g ss=%.9g\n",
                                g, m_bsCsrGroupSize[g], live, ta,
                                host.stressNormal, host.stressShear, devSn, devSs);
                        }
                    }
                    if (haveVectors && m_bsGpuNormal != nullptr && m_bsGpuCentroid != nullptr)
                    {
                        ++m_bsGpuStressChecks;
                        const float devN[3] = {m_bsGpuNormal[3 * g + 0],
                                               m_bsGpuNormal[3 * g + 1],
                                               m_bsGpuNormal[3 * g + 2]};
                        const float devC[3] = {m_bsGpuCentroid[3 * g + 0],
                                               m_bsGpuCentroid[3 * g + 1],
                                               m_bsGpuCentroid[3 * g + 2]};
                        if (memcmp(&host.normal.x, devN, sizeof(devN)) != 0
                            || memcmp(&host.centroid.x, devC, sizeof(devC)) != 0)
                        {
                            ++m_bsGpuStressMismatches;
                        }
                    }
                }
            }

            for (uint32_t n = 0; n < m_nodeOverstressed.size(); ++n)
            {
                ++m_bsGpuChecks;
                if (m_bsVerifyMask[n] != m_nodeOverstressed[n])
                {
                    ++m_bsGpuMismatches;
                    ++m_bsGpuMismMask;
                    if (m_bsGpuFirstBadTick == 0) m_bsGpuFirstBadTick = m_bsCsrTicks;
                m_bsGpuLastBadTick = m_bsCsrTicks;
                }
            }

            // The serial run is authoritative while auditing, and it wrote
            // m_bondsData, so consumers must read that rather than the device
            // outputs.
            m_bsGpuActive = false;
            if (false)
            {
                fprintf(stderr,
                        "[bond-stress-gpu] ticks=%llu checks=%llu MISMATCHES=%llu "
                        "(count=%llu removeSize=%llu removeOrder=%llu mask=%llu) "
                        "firstBadTick=%llu lastBadTick=%llu refuse(noSolver=%llu stale=%llu pendingTopo=%llu csr=%llu) STRESS_MISMATCH=%llu/%llu big=%llu maxRel=%.3e IMPULSE=%llu/%llu impMaxRel=%.3e PAYLOAD(n=%llu node=%llu normal=%llu disp=%llu centroid=%llu)\n",
                        (unsigned long long)m_bsCsrTicks,
                        (unsigned long long)m_bsGpuChecks,
                        (unsigned long long)m_bsGpuMismatches,
                        (unsigned long long)m_bsGpuMismCount,
                        (unsigned long long)m_bsGpuMismRemoveSize,
                        (unsigned long long)m_bsGpuMismRemoveOrder,
                        (unsigned long long)m_bsGpuMismMask,
                        (unsigned long long)m_bsGpuFirstBadTick,
                        (unsigned long long)m_bsGpuLastBadTick,
                    (unsigned long long)m_bsRefuseNoSolver,
                    (unsigned long long)m_bsRefuseStale,
                    (unsigned long long)m_bsRefusePendingTopo,
                    (unsigned long long)m_bsRefuseCsr,
                    (unsigned long long)m_bsGpuStressMismatches,
                    (unsigned long long)m_bsGpuStressChecks,
                    (unsigned long long)m_bsGpuStressBig,
                    m_bsGpuStressMaxRel,
                    (unsigned long long)m_bsImpMismatches,
                    (unsigned long long)m_bsImpChecks,
                    m_bsImpMaxRel,
                    (unsigned long long)m_bsPayChecks,
                    (unsigned long long)m_bsPayNode,
                    (unsigned long long)m_bsPayNormal,
                    (unsigned long long)m_bsPayDisp,
                    (unsigned long long)m_bsPayCentroid);
            }
            bondStressFinish();
            return;
        }

        for (uint32_t sIdx = 0; sIdx < kBondStressStrips; ++sIdx)
        {
            bondStressStrip(sIdx);
        }
        bondStressFinish();
    }

    /// Overlay the walk's live outputs onto a BondData copy, for the
    /// consumers that were written against the AoS.
    void overlayBondStressOutputs(uint32_t blastBondIndex, BondData& bond) const
    {
        float stressNormal, stressShear;
        nvidia::NvVec3 normal, centroid;
        if (bondStressOutputs(blastBondIndex, stressNormal, stressShear, normal, centroid))
        {
            bond.stressNormal = stressNormal;
            bond.stressShear = stressShear;
            bond.normal = normal;
            bond.centroid = centroid;
        }
    }

    bool fetchBondStressStresses() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        ExtStressGpuSolver* gpu = m_solver.gpuSolver();
        if (gpu == nullptr)
        {
            return false;
        }
        return gpu->readbackGroupStresses(m_bsGpuStressNormal, m_bsGpuStressShear);
#else
        return false;
#endif
    }

    bool fetchBondStressVectors() const
    {
#if defined(NVBLAST_ENABLE_CUDA_STRESS)
        ExtStressGpuSolver* gpu = m_solver.gpuSolver();
        if (gpu == nullptr)
        {
            return false;
        }
        return gpu->readbackGroupVectors(m_bsGpuNormal, m_bsGpuCentroid);
#else
        return false;
#endif
    }

    /// The device walk's two outcomes, reported so the skip rate is visible
    /// from a live report instead of only on stderr.
    ///
    /// skipped = answered from cache because every input was unchanged, no
    ///           kernel launched at all
    /// runs    = actually launched
    /// The two are disjoint and sum to the ticks the device path was taken;
    /// m_bsGpuRuns is deliberately NOT used here because it counts both.
    uint64_t getBondStressGpuSkipped() const { return m_bsSkippedLaunches; }
    uint64_t getBondStressGpuRuns() const { return m_bsPhCalls; }

    uint64_t getBondStressGpuChecks() const { return m_bsGpuChecks; }
    uint64_t getBondStressGpuMismatches() const { return m_bsGpuMismatches; }

private:


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
        if (resynced)
        {
            // A resync rebuilds m_solverBondsData wholesale, and can move node
            // positions, so every static per-member value the mirror caches is
            // stale. Rebuilt lazily at the next bondStressBegin.
            m_bsCsrDirty = true;
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
            solverNode.geometricInertia = 0.0f;
        }

        for (NodeData& node : m_nodesData)
        {
            SolverNodeData& solverNode = m_solverNodesData[node.solverNode];
            solverNode.supportNodesCount++;
            solverNode.localPos += node.localPos;
            solverNode.mass += node.mass;
            solverNode.volume += node.volume;
            // Summed without the parallel-axis term. Exact at graph reduction
            // 0, where every solver node is a single chunk -- which is what
            // production runs; an aggregate underestimates slightly, erring
            // toward the old sphere behaviour.
            solverNode.geometricInertia += node.geometricInertia;
        }

        for (SolverNodeData& solverNode : m_solverNodesData)
        {
            solverNode.localPos /= (float)solverNode.supportNodesCount;
        }

        m_solver.reset(m_solverNodesData.size());
        for (uint32_t nodeIndex = 0; nodeIndex < m_solverNodesData.size(); ++nodeIndex)
        {
            const SolverNodeData& solverNode = m_solverNodesData[nodeIndex];

            // Real shape-derived inertia when the host supplied one; the sphere
            // approximation (I = 2/5 M R^2 from volume) only as fallback.
            float inertia = solverNode.geometricInertia;
            if (inertia <= 0.0f)
            {
                const float R = NvPow(solverNode.volume * 3.0f * NvInvPi / 4.0f, 1.0f / 3.0f);
                inertia = solverNode.mass * (R * R * 0.4f);
            }
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
            bond.stressBend = 0.0f;

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
    virtual void                            setNodeGeometricInertia(uint32_t graphNode, float inertia) override;

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

    virtual const char*                     getGpuInactiveReason() const override
    {
        return m_graphProcessor->getGpuInactiveReason();
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

    virtual float                           getHostWalkInMilliseconds() const override
    { return m_graphProcessor->getHostWalkInMilliseconds(); }
    virtual float                           getHostResetMilliseconds() const override
    { return m_graphProcessor->getHostResetMilliseconds(); }
    virtual float                           getHostBondStressMilliseconds() const override
    { return m_graphProcessor->getHostBondStressMilliseconds(); }
    virtual float                           getHostNodeStressMilliseconds() const override
    { return m_graphProcessor->getHostNodeStressMilliseconds(); }

    virtual float                           getGpuImpulseCopyMilliseconds() const override
    {
        return m_graphProcessor->getGpuImpulseCopyMilliseconds();
    }

    virtual uint64_t                        getBondStressGroupsSkipped() const override
    {
        return m_graphProcessor->getBondStressGroupsSkipped();
    }

    virtual void                            setDeferBondStress(bool defer) override
    {
        m_graphProcessor->setDeferBondStress(defer);
    }

    virtual void                            bondStressStrip(uint32_t stripIdx) override
    {
        m_graphProcessor->bondStressStrip(stripIdx);
    }

    virtual void                            bondStressComplete() override
    {
        m_graphProcessor->bondStressComplete(m_deltaTime);
    }

    virtual uint32_t                        getBondStressStripCount() const override
    {
        return m_graphProcessor->bondStressStripCount();
    }

    virtual uint64_t                        getBondStressGpuSkipped() const override
    {
        return m_graphProcessor->getBondStressGpuSkipped();
    }

    virtual uint64_t                        getBondStressGpuRuns() const override
    {
        return m_graphProcessor->getBondStressGpuRuns();
    }

    virtual uint64_t                        getBondStressParallelChecks() const override
    {
        return m_graphProcessor->getBondStressParallelChecks();
    }

    virtual uint64_t                        getBondStressParallelMismatches() const override
    {
        return m_graphProcessor->getBondStressParallelMismatches();
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

    virtual void                            setDeltaTime(float deltaTime) override
    {
        m_deltaTime = deltaTime > 0.0f ? deltaTime : 0.0f;
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
    /// Bond-fracture commands for ONE actor, walking only the given graph
    /// nodes (ascending) instead of the actor's whole node list. Used by the
    /// candidate path of generateFractureCommandsPerActor.
    void                                    fillFractureCommandsForNodes(const NvBlastActor& actor, NvBlastFractureBuffers& commands, const uint32_t* nodes, uint32_t nodeCount);
    /// Scratch for the candidate path: (actor index, graph node) of every
    /// overstressed node, sorted so one actor's nodes are contiguous.
    std::vector<std::pair<uint32_t, uint32_t>> m_fractureCandidates;
    uint64_t                                m_fractureCandidateChecks{0};
    uint64_t                                m_fractureCandidateMismatches{0};
    uint64_t                                m_fractureCandidateTicks{0};

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

void ExtStressSolverImpl::setNodeGeometricInertia(uint32_t graphNode, float inertia)
{
    m_graphProcessor->setNodeGeometricInertia(graphNode, inertia);
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
            auto bondData = m_graphProcessor->getBondData(internalBondIndex);
            // GPU mode leaves BondData unwritten; overlay the live values.
            m_graphProcessor->overlayBondStressOutputs(blastBondIndex, bondData);
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
/**
How fast a bond past its elastic limit loses section, per second of being held
there, per unit of overstress.

This governs only the SUB-FATAL band: a joint held between its elastic and
fatal limits loses health * multiplier * this per second, where the multiplier
is how far into that band it sits. A joint at or past fatal fails immediately
regardless of this value, because that is a different mechanism (see
generateStressDamage).

So a joint 20% into the band is gone in 1/(0.2 * rate) seconds. At 0.5 that is
ten seconds; at the top of the band it is two. That spread is the point --
something barely overloaded should visibly strain for a while, something badly
overloaded should go almost at once, and the difference between them is what
makes a delayed collapse read as a structure losing a fight rather than a timer
expiring.

Measured on the cantilever ladder at 2.0: a 10 m overhang lets go at 4 s and a
12 m one at 3 s, both having visibly held first, while a parking deck with one
whole side of columns cut sags and comes down over roughly twenty seconds. BLAST_DAMAGE_RATE overrides.
*/
static float damageRatePerSecond()
{
    static const float rate = []() {
        const char* raw = std::getenv("BLAST_DAMAGE_RATE");
        if (raw != nullptr)
        {
            const float parsed = static_cast<float>(std::atof(raw));
            if (parsed > 0.0f)
            {
                return parsed;
            }
        }
        return 2.0f;
    }();
    return rate;
}

bool ExtStressSolverImpl::generateStressDamage(const NvBlastActor& actor, uint32_t bondIndex, uint32_t node0, uint32_t node1)
{
    const float bondHealth = m_bondHealths[bondIndex];
    float stressCompression, stressTension, stressShear;
    if (bondHealth > 0.0f && m_graphProcessor->getBondStress(bondIndex, stressCompression, stressTension, stressShear))
    {
        // Compression and tension are opposite directions of AXIAL load, but a
        // bending moment produces both at once on opposite faces, so both are
        // checked. They are combined with max() rather than added: it is one
        // cross-section failing, and summing would count the same failure twice.
        const ExtStressMaterial& material = materialForBond(bondIndex);
        float stressMultiplier = 0.0f;
        float axialMultiplier = 0.0f;
        if (stressCompression > material.compressionElasticLimit)
        {
            const float excessStress = stressCompression - material.compressionElasticLimit;
            const float compressionDenom = material.compressionFatalLimit - material.compressionElasticLimit;
            axialMultiplier = excessStress / (compressionDenom > 0.0f ? compressionDenom : 1.0f);
        }
        if (stressTension > material.tensionElasticLimit)
        {
            const float excessStress = stressTension - material.tensionElasticLimit;
            const float tensionDenom = material.tensionFatalLimit - material.tensionElasticLimit;
            const float tensionMultiplier = excessStress / (tensionDenom > 0.0f ? tensionDenom : 1.0f);
            axialMultiplier = std::max(axialMultiplier, tensionMultiplier);
        }
        stressMultiplier += axialMultiplier;

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
            // Bond health/area is reduced by excess pressure, approximating
            // micro bonds in the material breaking.
            //
            // Two DIFFERENT failure mechanisms live in this one number, and
            // they are separated here:
            //
            //   at or past the fatal limit (multiplier >= 1) the joint is
            //   overloaded outright and fails now, in this tick, however
            //   briefly the load was applied. This is what a blast or an
            //   impact does: microseconds of enormous stress, and the thing
            //   breaks.
            //
            //   between elastic and fatal it does not fail, it DAMAGES, at a
            //   rate -- losing section for as long as it is held there. This
            //   is what an overloaded cantilever does: holds, creaks, and lets
            //   go some seconds later.
            //
            // Scaling both by the timestep, as a single rate, cannot serve
            // both: slow enough for a floor to strain visibly before it goes
            // is slow enough that a rocket deposits under a percent of a
            // bond's health in its one tick and nothing breaks at all. Slow
            // enough for a rocket is a floor that vanishes the instant it is
            // overloaded. They are separate mechanisms and the model now says
            // so.
            //
            // m_deltaTime of 0 means nobody told us the timestep, which is the
            // offline case: fall back to per-tick, unchanged.
            float bondDamage;
            if (stressMultiplier >= 1.0f)
            {
                bondDamage = bondHealth;
            }
            else if (m_deltaTime > 0.0f)
            {
                bondDamage = bondHealth * std::min(
                    1.0f, stressMultiplier * m_deltaTime * damageRatePerSecond());
            }
            else
            {
                bondDamage = bondHealth * stressMultiplier;
            }
            // Damage ARRESTS at the residual the reinforcement represents.
            //
            // Health is remaining area and stress is force/health, so a joint
            // that keeps losing area keeps raising its own stress -- the
            // runaway that makes an overloaded joint fail eventually no matter
            // how slowly. Stopping at a floor turns that into what a
            // reinforced crack does: crack, weaken to the section the steel
            // holds, and stay there.
            // ...but only on the GRADUAL path. Past the fatal limit the joint
            // goes outright, arrest or no arrest: reinforcement holds a crack
            // open at a stable width, it does not survive the steel yielding.
            // Clamping the fatal case too let bonds sit at 172x their elastic
            // limit indefinitely, which is how a garage stripped of 90% of its
            // columns stood there with nothing broken.
            if (stressMultiplier < 1.0f && material.residualAreaFraction > 0.0f)
            {
                const float floorHealth =
                    m_bonds[bondIndex].area * material.residualAreaFraction;
                if (bondHealth - bondDamage < floorHealth)
                {
                    bondDamage = std::max(0.0f, bondHealth - floorHealth);
                }
            }

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

void ExtStressSolverImpl::fillFractureCommandsForNodes(const NvBlastActor& actor, NvBlastFractureBuffers& commands, const uint32_t* nodes, uint32_t nodeCount)
{
    uint32_t commandCount = 0;
    for (uint32_t i = 0; i < nodeCount; ++i)
    {
        const uint32_t node0 = nodes[i];
        for (uint32_t adjacencyIndex = m_graph.adjacencyPartition[node0]; adjacencyIndex < m_graph.adjacencyPartition[node0 + 1]; adjacencyIndex++)
        {
            const uint32_t node1 = m_graph.adjacentNodeIndices[adjacencyIndex];
            // Same ownership rule as the full walk: a bond is visited from its
            // lower-indexed endpoint, and both endpoints of an overstressed
            // bond are flagged, so every overstressed bond is visited exactly
            // once here too.
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
    commands.chunkFractureCount = 0;
    commands.chunkFractures = nullptr;
    commands.bondFractureCount = commandCount;
    commands.bondFractures = commandCount > 0 ? m_bondFractureBuffer.end() - commandCount : nullptr;
}

/// Generate commands only for the actors that own an overstressed node,
/// walking only those nodes.
///
/// The full walk visits EVERY active actor and fetches every actor's node
/// list, then checks the overstressed flag per node: O(actors + nodes) per
/// tick. Under the sub-fatal damage band a few hundred bonds sit overstressed
/// on essentially every tick of a standing city, so that walk ran every tick
/// -- 1.3 ms at rest on the 87k-node downtown to find ~300 bonds on four
/// actors. The flagged nodes name their actor directly through the family's
/// chunk -> actor table, so the work is O(overstressed nodes log n).
///
/// Values are identical: damage is a per-bond function of the bond's own
/// stress and health, and the adapter sorts commands by actor index before
/// applying them. Only the ORDER of bond commands within one actor differs
/// (ascending node here, the actor's node-list order in the full walk), which
/// nothing downstream depends on. BLAST_FRACTURE_CANDIDATES=0 restores the
/// full walk; BLAST_FRACTURE_CANDIDATES_VERIFY=1 runs both every tick and
/// compares the command sets.
static bool fractureCandidatesEnabled()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("BLAST_FRACTURE_CANDIDATES");
        return raw == nullptr || raw[0] != '0';
    }();
    return enabled;
}

static bool fractureCandidatesVerify()
{
    static const bool enabled = []() {
        const char* raw = std::getenv("BLAST_FRACTURE_CANDIDATES_VERIFY");
        return raw != nullptr && raw[0] == '1';
    }();
    return enabled;
}

uint32_t ExtStressSolverImpl::generateFractureCommandsPerActor(const NvBlastActor** actorBuffer, NvBlastFractureBuffers* commandsBuffer, uint32_t bufferSize)
{
    // A crushed chunk is fracture work even when no bond is overstressed: a
    // chunk can be pulverized while every joint around it is still intact.
    const bool crushPending =
        m_graphProcessor->isCrushEnabled() && m_graphProcessor->getPendingCrushCount() > 0;
    if (m_graphProcessor->getOverstressedBondCount() == 0 && !crushPending)
        return 0;

    // Crush commands are per node and not tracked by the overstressed flags,
    // so a tick with pending crush work takes the full walk.
    const uint32_t* chunkActorIndices =
        (!crushPending && fractureCandidatesEnabled())
            ? NvBlastFamilyGetChunkActorIndices(&m_family, logLL)
            : nullptr;

    // Verify mode: the full walk first, into a private copy, because the
    // candidate walk appends to the same command buffer.
    std::vector<NvBlastBondFractureData> verifyFull;
    if (chunkActorIndices != nullptr && fractureCandidatesVerify())
    {
        m_bondFractureBuffer.clear();
        m_chunkFractureBuffer.clear();
        for (auto it = m_activeActors.getIterator(); !it.done(); ++it)
        {
            NvBlastFractureBuffers scratch{};
            fillFractureCommands(**it, scratch);
        }
        verifyFull.assign(m_bondFractureBuffer.begin(), m_bondFractureBuffer.end());
    }

    m_bondFractureBuffer.clear();
    m_chunkFractureBuffer.clear();
    uint32_t index = 0;

    if (chunkActorIndices != nullptr)
    {
        m_fractureCandidates.clear();
        const uint32_t nodeCount = m_graph.nodeCount;
        for (uint32_t node = 0; node < nodeCount; ++node)
        {
            if (!m_graphProcessor->isNodeOverstressed(node))
            {
                continue;
            }
            const uint32_t actorIndex = chunkActorIndices[m_graph.chunkIndices[node]];
            if (isInvalidIndex(actorIndex))
            {
                continue;
            }
            m_fractureCandidates.emplace_back(actorIndex, node);
        }
        std::sort(m_fractureCandidates.begin(), m_fractureCandidates.end());

        std::vector<uint32_t> nodes;
        for (size_t i = 0; i < m_fractureCandidates.size() && index < bufferSize;)
        {
            const uint32_t actorIndex = m_fractureCandidates[i].first;
            nodes.clear();
            for (; i < m_fractureCandidates.size() && m_fractureCandidates[i].first == actorIndex; ++i)
            {
                nodes.push_back(m_fractureCandidates[i].second);
            }
            const NvBlastActor* actor = NvBlastFamilyGetActorByIndex(&m_family, actorIndex, logLL);
            if (actor == nullptr || !m_activeActors.contains(actor))
            {
                continue;
            }
            NvBlastFractureBuffers& nextCommand = commandsBuffer[index];
            fillFractureCommandsForNodes(*actor, nextCommand, nodes.data(), static_cast<uint32_t>(nodes.size()));
            if (nextCommand.bondFractureCount > 0)
            {
                actorBuffer[index] = actor;
                index++;
            }
        }

        if (fractureCandidatesVerify())
        {
            // Same multiset of (bond, damage) regardless of order.
            std::vector<NvBlastBondFractureData> got(m_bondFractureBuffer.begin(), m_bondFractureBuffer.end());
            auto key = [](const NvBlastBondFractureData& a, const NvBlastBondFractureData& b) {
                if (a.nodeIndex0 != b.nodeIndex0) return a.nodeIndex0 < b.nodeIndex0;
                if (a.nodeIndex1 != b.nodeIndex1) return a.nodeIndex1 < b.nodeIndex1;
                return a.health < b.health;
            };
            std::sort(got.begin(), got.end(), key);
            std::sort(verifyFull.begin(), verifyFull.end(), key);
            bool same = got.size() == verifyFull.size();
            for (size_t k = 0; same && k < got.size(); ++k)
            {
                same = got[k].nodeIndex0 == verifyFull[k].nodeIndex0
                    && got[k].nodeIndex1 == verifyFull[k].nodeIndex1
                    && got[k].health == verifyFull[k].health;
            }
            m_fractureCandidateChecks += verifyFull.size();
            ++m_fractureCandidateTicks;
            if (!same)
            {
                ++m_fractureCandidateMismatches;
                std::fprintf(stderr,
                             "[fracture-candidates] MISMATCH tick=%llu full=%zu candidates=%zu\n",
                             static_cast<unsigned long long>(m_fractureCandidateTicks),
                             verifyFull.size(), got.size());
            }
            if (m_fractureCandidateTicks % 600 == 0)
            {
                std::fprintf(stderr,
                             "[fracture-candidates] ticks=%llu checks=%llu mismatches=%llu\n",
                             static_cast<unsigned long long>(m_fractureCandidateTicks),
                             static_cast<unsigned long long>(m_fractureCandidateChecks),
                             static_cast<unsigned long long>(m_fractureCandidateMismatches));
            }
        }
        return index;
    }

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
