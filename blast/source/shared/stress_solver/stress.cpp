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
// Copyright (c) 2022-2024 NVIDIA Corporation. All rights reserved.

#include "stress.h"
#include "math/cgnr.h"
#if !defined(STRESS_SOLVER_NO_DEVICE_QUERY)
#include "simd/simd_device_query.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>

#define MASS_AND_LENGTH_SCALING 1


typedef CGNR<AngLin6, AngLin6Ops<Float_Scalar>, BondMatrixS, BondMatrixOpsS<Float_Scalar>, Float_Scalar, AngLin6ErrorSq>    CGNR_SISD;
typedef CGNR<AngLin6, AngLin6Ops<SIMD_Scalar>, BondMatrixS, BondMatrixOpsS<SIMD_Scalar>, SIMD_Scalar, AngLin6ErrorSq>       CGNR_SIMD;


// Bit-exact velocity comparison. A settled island is only skipped when its inputs are *identical*
// to its last solve, so any change (a new contact, a body waking) differs here and re-solves it
// that same frame. Used by StressProcessor::solveIslandAware.
static inline bool angLin6Equal(const AngLin6& a, const AngLin6& b)
{
    return a.ang.x == b.ang.x && a.ang.y == b.ang.y && a.ang.z == b.ang.z
        && a.lin.x == b.lin.x && a.lin.y == b.lin.y && a.lin.z == b.lin.z;
}


/**
 * StressProcessor static members
 */

#if defined(STRESS_SOLVER_FORCE_SCALAR)
const bool
StressProcessor::s_use_simd = false;
#elif defined(STRESS_SOLVER_NO_DEVICE_QUERY)
const bool
StressProcessor::s_use_simd = false;
#else
// Check for SSE, FMA3, and AVX support
const bool
StressProcessor::s_use_simd =
    device_supports_instruction_set(InstructionSet::SSE) &&     // Basic SSE
    device_supports_instruction_set(InstructionSet::FMA3) &&    // Fused Multiply-Add instructions
    device_supports_instruction_set(InstructionSet::OSXSAVE) && // OS uses XSAVE and XRSTORE instructions allowing saving YMM registers on context switch
    device_supports_instruction_set(InstructionSet::AVX) &&     // Advanced Vector Extensions (256 bit operations)
    os_supports_avx_restore();                                  // OS has enabled the required extended state for AVX
#endif


/**
 * StressProcessor methods
 */

void
StressProcessor::prepare(const SolverNodeS* nodes, uint32_t N_nodes, const SolverBond* bonds, uint32_t N_bonds, const DataParams& params)
{
    m_recip_sqrt_I.resize(N_nodes);
    m_couplings.resize(N_bonds);
    m_rhs.resize(N_nodes);
    m_B_scratch.resize(N_nodes);
    m_solver_cache.resize(s_use_simd ? CGNR_SIMD().required_cache_size(N_nodes, N_bonds) : CGNR_SISD().required_cache_size(N_nodes, N_bonds));
    m_can_resume = false;
    m_skipValid = false;        // topology changed: drop the settled-state baseline so a fresh one is rebuilt
    m_islandTopoValid = false;  // topology changed: rebuild the island grouping cache on the next island solve

    // Calculate bond offsets and length scale
    uint32_t offsets_to_scale = 0;
    m_length_scale = 0.0f;
    for (uint32_t i = 0; i < N_bonds; ++i)
    {
        const SolverBond& bond = bonds[i];
        const uint32_t b0 = bond.nodes[0];
        const uint32_t b1 = bond.nodes[1];
        Coupling& c = m_couplings[i];

        NvcVec3 offset0, offset1;
        if (!params.centerBonds)
        {
            offset0 = nodes[b0].mass > 0 ? bond.centroid - nodes[b0].CoM : nodes[b1].CoM - bond.centroid;
            offset1 = nodes[b1].mass > 0 ? bond.centroid - nodes[b1].CoM : nodes[b0].CoM - bond.centroid;
        }
        else
        {
            if (nodes[b0].mass <= 0)
            {
                offset1 = bond.centroid - nodes[b1].CoM;
                offset0 = -offset1;
            }
            else
            if (nodes[b1].mass <= 0)
            {
                offset0 = bond.centroid - nodes[b0].CoM;
                offset1 = -offset0;
            }
            else
            {
                offset0 = 0.5f*(nodes[b1].CoM - nodes[b0].CoM);
                offset1 = -offset0;
            }
        }

        if (nodes[b0].mass > 0.0f)
        {
            ++offsets_to_scale;
            m_length_scale += std::sqrt(offset0|offset0);
        }
        if (nodes[b1].mass > 0.0f)
        {
            ++offsets_to_scale;
            m_length_scale += std::sqrt(offset1|offset1);
        }

        c.offset0 = offset0;
        c.node0 = bond.nodes[0];
        c.offset1 = offset1;
        c.node1 = bond.nodes[1];
    }
#if MASS_AND_LENGTH_SCALING
    m_length_scale = offsets_to_scale ? m_length_scale / offsets_to_scale : 1.0f;
#else
    m_length_scale = 1.0f;
#endif

    // Scale offsets by length scale
    const float recip_length_scale = 1.0f/m_length_scale;
    for (uint32_t j = 0; j < N_bonds; ++j)
    {
        Coupling& coupling = m_couplings[j];
        coupling.offset0 *= recip_length_scale;
        coupling.offset1 *= recip_length_scale;
    }

    // Set mass scale to geometric mean of the masses
    m_mass_scale = 0.0f;
    uint32_t nonzero_mass_count = 0;
    for (uint32_t i = 0; i < N_nodes; ++i)
    {
        if (nodes[i].mass > 0.0f)
        {
            m_mass_scale += std::log(nodes[i].mass);
            ++nonzero_mass_count;
        }
    }

#if MASS_AND_LENGTH_SCALING
    m_mass_scale = nonzero_mass_count ? std::exp(m_mass_scale / nonzero_mass_count) : 1.0f;
#else
    m_mass_scale = 1.0f;
#endif

    // Generate I^-1/2
    std::vector<InertiaS> invI(N_nodes);
    const float inertia_scale = m_mass_scale*m_length_scale*m_length_scale;
    if (!params.equalizeMasses)
    {
        for (uint32_t i = 0; i < N_nodes; ++i)
        {
            invI[i] =
            {
                nodes[i].inertia > 0.0f ? inertia_scale/nodes[i].inertia : 0.0f,
                nodes[i].mass > 0.0f ? m_mass_scale/nodes[i].mass : 0.0f
            };
            m_recip_sqrt_I[i] = { std::sqrt(invI[i].I), std::sqrt(invI[i].m) };
        }
    }
    else
    {
        for (uint32_t i = 0; i < N_nodes; ++i)
        {
            invI[i] =
            {
                nodes[i].inertia > 0.0f ? 1.0f : 0.0f,
                nodes[i].mass > 0.0f ? 1.0f : 0.0f
            };
            m_recip_sqrt_I[i] = { std::sqrt(invI[i].I), std::sqrt(invI[i].m) };
        }
    }

    // Create sparse matrix representation for B = (I^-1/2)*C
    m_B.set(m_couplings.data(), m_recip_sqrt_I.data(), m_B_scratch.data(), N_nodes, N_bonds);
}


int
StressProcessor::solve(AngLin6* impulses, const AngLin6* velocities, const SolverParams& params, AngLin6ErrorSq* error_sq /* = nullptr */, bool resume /* = false */)
{
    // Island-aware path: solve each disconnected component ("island") independently. When there
    // is at most one island it reports handled=false and we fall through to the whole-graph code
    // below, which is then bit-identical to the legacy solve.
    if (params.islandAware)
    {
        bool handled = false;
        const int islandResult = solveIslandAware(impulses, velocities, params, error_sq, handled);
        if (handled) return islandResult;
    }

    const InertiaS* sqrt_I_inv = m_recip_sqrt_I.data();
    const uint32_t N_nodes = getNodeCount();
    const uint32_t N_bonds = getBondCount();
    void* cache = m_solver_cache.data();

    const float recip_length_scale = 1.0f/m_length_scale;

    // Apply length and mass scaling to impulses if warm-starting
    if (params.warmStart)
    {
        const float recip_mass_scale = 1.0f/m_mass_scale;
        const float recip_linear_impulse_scale = recip_length_scale*recip_mass_scale;
        const float recip_angular_impulse_scale = recip_length_scale*recip_linear_impulse_scale;
        for (uint32_t j = 0; j < N_bonds; ++j)
        {
            impulses[j].ang *= recip_angular_impulse_scale;
            impulses[j].lin *= recip_linear_impulse_scale;
        }
    }

    // Calculate r.h.s. vector b = -(I^1/2)*velocities
    AngLin6* b = m_rhs.data();
    for (uint32_t i = 0; i < N_nodes; ++i)
    {
        const InertiaS& I_i = sqrt_I_inv[i];
        const AngLin6& v_i = velocities[i];
        AngLin6& b_i = b[i];
        if (I_i.I <= 0.0f && (v_i.ang|v_i.ang) > 0.0f)
        {
            std::fprintf(stderr, "[Blast][StressSolver] node %u has zero angular inertia but non-zero angular velocity\n", i);
            return -32;
        }
        b_i.ang = v_i.ang/(-(I_i.I > 0 ? I_i.I : 1.0f));
        b_i.lin = (-recip_length_scale/(I_i.m > 0 ? I_i.m : 1.0f))*v_i.lin;
    }

    // Solve B*J = b for J, where B = (I^-1/2)*C and b = -(I^1/2)*v.
    // Since CGNR does this by solving (B^T)*B*J = (B^T)*b, this actually solves
    // (C^T)*(I^-1)*C*J = -(C^T)*v for J, which is the equation we really wanted to solve.
    const uint32_t maxIter = params.maxIter ? params.maxIter : 6*std::max(N_nodes, N_bonds);

    // Set solver warmth
    const unsigned warmth = params.warmStart ? (m_can_resume && resume ? 2 : 1) : 0;

    // Choose solver based on parameters
    const int result = s_use_simd ?
        CGNR_SIMD().solve(impulses, m_B, b, N_nodes, N_bonds, cache, error_sq, params.tolerance, maxIter, warmth) :
        CGNR_SISD().solve(impulses, m_B, b, N_nodes, N_bonds, cache, error_sq, params.tolerance, maxIter, warmth);

    // Undo length and mass scaling
    const float linear_impulse_scale = m_length_scale*m_mass_scale;
    const float angular_impulse_scale = m_length_scale*linear_impulse_scale;
    for (uint32_t j = 0; j < N_bonds; ++j)
    {
        impulses[j].ang *= angular_impulse_scale;
        impulses[j].lin *= linear_impulse_scale;
    }

    m_can_resume = true;

    return result;
}


int
StressProcessor::solveIslandAware(AngLin6* impulses, const AngLin6* velocities, const SolverParams& params, AngLin6ErrorSq* error_sq, bool& handled)
{
    handled = false;

    const uint32_t N_nodes = getNodeCount();
    const uint32_t N_bonds = getBondCount();
    if (N_bonds == 0) return 0;     // nothing bonded; let the caller use the whole-graph path

#ifdef STRESS_SOLVER_NO_ISLAND_CACHE
    m_islandTopoValid = false;      // A/B switch: rebuild the island grouping every frame (original behavior)
#endif

    const uint32_t kInvalid = (uint32_t)-1;
    const Coupling* C = m_couplings.data();
    const InertiaS* sqrt_I_inv = m_recip_sqrt_I.data();

    // ── 1-3. Build the island grouping (union-find + per-bond island id + CSR grouping). This is a
    //         pure function of topology, so it is built only when m_islandTopoValid is false (set by
    //         prepare()/removeBond()) and reused bit-identically on every other frame. m_islandCount,
    //         m_islandBondBegin and m_bondsByIsland persist across frames as the cache. ──
    if (!m_islandTopoValid)
    {
        // 1. Union-find over bonds. Static (zero-mass) nodes carry no coupling and act as cut points,
        //    so two structures sharing only a static/world node are separate islands.
        m_uf.resize(N_nodes);
        for (uint32_t i = 0; i < N_nodes; ++i) m_uf[i] = i;
        for (uint32_t b = 0; b < N_bonds; ++b)
        {
            const uint32_t n0 = C[b].node0;
            const uint32_t n1 = C[b].node1;
            if (!(sqrt_I_inv[n0].m > 0.0f) || !(sqrt_I_inv[n1].m > 0.0f)) continue;   // cut at static nodes
            uint32_t r0 = n0; while (m_uf[r0] != r0) { m_uf[r0] = m_uf[m_uf[r0]]; r0 = m_uf[r0]; }
            uint32_t r1 = n1; while (m_uf[r1] != r1) { m_uf[r1] = m_uf[m_uf[r1]]; r1 = m_uf[r1]; }
            if (r0 != r1) m_uf[r0] = r1;
        }

        // 2. Assign each bond a compacted island id (the component of its dynamic endpoint).
        m_bondIsland.assign(N_bonds, kInvalid);
        m_rootIsland.assign(N_nodes, kInvalid);
        uint32_t islandCount = 0;
        for (uint32_t b = 0; b < N_bonds; ++b)
        {
            const uint32_t n0 = C[b].node0;
            const uint32_t n1 = C[b].node1;
            const bool s0 = !(sqrt_I_inv[n0].m > 0.0f);
            const bool s1 = !(sqrt_I_inv[n1].m > 0.0f);
            if (s0 && s1) continue;                          // degenerate static-static bond: no coupling
            uint32_t rep = s0 ? n1 : n0;                     // a dynamic endpoint
            while (m_uf[rep] != rep) { m_uf[rep] = m_uf[m_uf[rep]]; rep = m_uf[rep]; }
            if (m_rootIsland[rep] == kInvalid) m_rootIsland[rep] = islandCount++;
            m_bondIsland[b] = m_rootIsland[rep];
        }

        // 3. Group bonds contiguously by island (CSR offsets via counting sort). Skipped when there is
        //    at most one island, since the whole-graph fallback below does not read the grouping.
        if (islandCount > 1)
        {
            m_islandBondBegin.assign(islandCount + 1, 0);
            for (uint32_t b = 0; b < N_bonds; ++b)
                if (m_bondIsland[b] != kInvalid) ++m_islandBondBegin[m_bondIsland[b] + 1];
            for (uint32_t k = 0; k < islandCount; ++k)
                m_islandBondBegin[k + 1] += m_islandBondBegin[k];
            m_bondsByIsland.resize(m_islandBondBegin[islandCount]);
            m_cursor.assign(m_islandBondBegin.begin(), m_islandBondBegin.end());
            for (uint32_t b = 0; b < N_bonds; ++b)
            {
                const uint32_t isl = m_bondIsland[b];
                if (isl != kInvalid) m_bondsByIsland[m_cursor[isl]++] = b;
            }
        }

        m_islandCount = islandCount;
        m_islandTopoValid = true;
    }

    const uint32_t islandCount = m_islandCount;

    if (islandCount <= 1)
    {
        m_skipValid = false;            // whole-graph path runs (no per-island baseline maintained this frame)
        m_lastIslandsTotal = 0;
        m_lastIslandsSkipped = 0;
        return 0;                       // single island: caller uses the bit-identical whole-graph path
    }

    handled = true;

    // ── 4. Scratch + scaling constants (identical to solve()). ──
    m_g2l.resize(N_nodes);
    m_g2lStamp.assign(N_nodes, 0);      // 0 == unstamped; island k uses stamp (k+1)
    m_l2g.resize(N_nodes);
    m_localC.resize(N_bonds);
    m_localI.resize(N_nodes);
    m_localImpulses.resize(N_bonds);
    m_rhs.resize(N_nodes);
    m_B_scratch.resize(N_nodes);
    m_lastVel.resize(N_nodes);
    m_nodeConverged.resize(N_nodes, 0);

    const float recip_length_scale          = 1.0f / m_length_scale;
    const float recip_mass_scale            = 1.0f / m_mass_scale;
    const float recip_linear_impulse_scale  = recip_length_scale * recip_mass_scale;
    const float recip_angular_impulse_scale = recip_length_scale * recip_linear_impulse_scale;
    const float linear_impulse_scale        = m_length_scale * m_mass_scale;
    const float angular_impulse_scale       = m_length_scale * linear_impulse_scale;

    void*     cache  = m_solver_cache.data();
    AngLin6*  b_rhs  = m_rhs.data();
    Coupling* localC = m_localC.data();
    InertiaS* localI = m_localI.data();
    AngLin6*  x      = m_localImpulses.data();

    int  totalIters   = 0;
    bool allConverged = true;
    AngLin6ErrorSq totalErr = { 0.0f, 0.0f };
    uint32_t skipped = 0;
    const bool trySkip = params.skipSettled && m_skipValid;

    // Per-bond "settled-skipped" flags for this solve (consumed by the caller to skip recomputing
    // unchanged bond stresses). Cleared here so it reflects only this solve; bonds in re-solved
    // islands and any bond with no island stay 0.
    m_bondSkipped.assign(N_bonds, 0);

    // ── 5. Per-island: gather a contiguous, island-local sub-system → run the same CGNR/scaling as
    //       solve() → scatter the bond impulses back. An island whose dynamic nodes' velocities are
    //       bit-identical to its last solve and that already converged is skipped (its solve would be
    //       a no-op): its impulses/stresses are kept unchanged. Paused, never evicted. ──
    for (uint32_t k = 0; k < islandCount; ++k)
    {
        const uint32_t bBegin = m_islandBondBegin[k];
        const uint32_t bEnd   = m_islandBondBegin[k + 1];
        const uint32_t localN = bEnd - bBegin;
        if (localN == 0) continue;
        const uint32_t stamp = k + 1;

        // Gather island-local nodes (renumbered 0..localM), couplings, and warm-start impulses, and
        // (when eligible) test settled-state: every DYNAMIC node's velocity unchanged and last solve
        // converged. Static (zero-mass) nodes are anchors that carry no coupling, so they are ignored.
        bool skippable = trySkip;
        uint32_t localM = 0;
        for (uint32_t t = bBegin; t < bEnd; ++t)
        {
            const uint32_t b  = m_bondsByIsland[t];
            const uint32_t g0 = C[b].node0;
            const uint32_t g1 = C[b].node1;
            if (m_g2lStamp[g0] != stamp)
            {
                m_g2lStamp[g0] = stamp; m_g2l[g0] = localM; m_l2g[localM] = g0; localI[localM] = sqrt_I_inv[g0]; ++localM;
                if (skippable && sqrt_I_inv[g0].m > 0.0f && !(m_nodeConverged[g0] && angLin6Equal(velocities[g0], m_lastVel[g0]))) skippable = false;
            }
            if (m_g2lStamp[g1] != stamp)
            {
                m_g2lStamp[g1] = stamp; m_g2l[g1] = localM; m_l2g[localM] = g1; localI[localM] = sqrt_I_inv[g1]; ++localM;
                if (skippable && sqrt_I_inv[g1].m > 0.0f && !(m_nodeConverged[g1] && angLin6Equal(velocities[g1], m_lastVel[g1]))) skippable = false;
            }
            const uint32_t lb = t - bBegin;
            Coupling lc = C[b];
            lc.node0 = m_g2l[g0];
            lc.node1 = m_g2l[g1];
            localC[lb] = lc;
            x[lb] = impulses[b];
        }

        if (skippable)
        {
            // Inputs unchanged and already converged → re-solving is a no-op. Keep this island's bond
            // impulses (and hence its stresses) exactly as they are; the baseline already matches.
            // Flag its bonds so the caller can skip recomputing their (unchanged) stresses too.
            for (uint32_t t = bBegin; t < bEnd; ++t) m_bondSkipped[m_bondsByIsland[t]] = 1;
            ++skipped;
            continue;
        }

        // Warm-start impulse scaling (matches solve()).
        if (params.warmStart)
        {
            for (uint32_t j = 0; j < localN; ++j)
            {
                x[j].ang *= recip_angular_impulse_scale;
                x[j].lin *= recip_linear_impulse_scale;
            }
        }

        // Right-hand side b = -(I^1/2)*velocities for this island's nodes.
        bool bad = false;
        for (uint32_t i = 0; i < localM; ++i)
        {
            const InertiaS& I_i = localI[i];
            const AngLin6&  v_i = velocities[m_l2g[i]];
            AngLin6& bi = b_rhs[i];
            if (I_i.I <= 0.0f && (v_i.ang | v_i.ang) > 0.0f) { bad = true; break; }
            bi.ang = v_i.ang / (-(I_i.I > 0 ? I_i.I : 1.0f));
            bi.lin = (-recip_length_scale / (I_i.m > 0 ? I_i.m : 1.0f)) * v_i.lin;
        }
        if (bad)
        {
            allConverged = false;
            for (uint32_t i = 0; i < localM; ++i) m_nodeConverged[m_l2g[i]] = 0;   // never skip a bad island
            continue;
        }

        // Island-local bond matrix view over the gathered couplings/inertia.
        BondMatrixS lB;
        lB.set(localC, localI, m_B_scratch.data(), localM, localN);

        const uint32_t maxIter = params.maxIter ? params.maxIter : 6 * std::max(localM, localN);
        const unsigned warmth  = params.warmStart ? 1u : 0u;   // hot-resume (cache) not used per island

        AngLin6ErrorSq err = { 0.0f, 0.0f };
        const int r = s_use_simd ?
            CGNR_SIMD().solve(x, lB, b_rhs, localM, localN, cache, &err, params.tolerance, maxIter, warmth) :
            CGNR_SISD().solve(x, lB, b_rhs, localM, localN, cache, &err, params.tolerance, maxIter, warmth);

        // Undo scaling and scatter the bond impulses back to their global positions.
        for (uint32_t j = 0; j < localN; ++j)
        {
            x[j].ang *= angular_impulse_scale;
            x[j].lin *= linear_impulse_scale;
            impulses[m_bondsByIsland[bBegin + j]] = x[j];
        }

        // Record the settled-state baseline for next frame: the inputs just solved and whether it converged.
        const uint8_t cv = (r >= 0) ? 1u : 0u;
        for (uint32_t i = 0; i < localM; ++i) { const uint32_t g = m_l2g[i]; m_lastVel[g] = velocities[g]; m_nodeConverged[g] = cv; }

        if (r < 0) { allConverged = false; totalIters += -r; } else { totalIters += r; }
        totalErr.ang += err.ang;
        totalErr.lin += err.lin;
    }

    m_can_resume = false;           // island path does not maintain the hot-resume cache
    m_skipValid = true;             // baseline is current; islands may be skipped next frame
    m_lastIslandsTotal = islandCount;
    m_lastIslandsSkipped = skipped;
    if (error_sq) *error_sq = totalErr;
    return allConverged ? totalIters : -totalIters;
}


bool
StressProcessor::removeBond(uint32_t bondIndex)
{
    if (bondIndex >= getBondCount()) return false;

    m_couplings[bondIndex] = m_couplings.back();
    m_couplings.pop_back();
    --m_B.N;
    m_can_resume = false;
    m_skipValid = false;        // topology changed: drop the settled-state baseline
    m_islandTopoValid = false;  // topology changed: rebuild the island grouping cache on the next island solve

    return true;
}
