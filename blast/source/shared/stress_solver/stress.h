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

#pragma once

#include "bond.h"
#include "buffer.h"


class StressProcessor
{
public:
    /** Constructor clears member data. */
    StressProcessor() : m_mass_scale(0.0f), m_length_scale(0.0f), m_can_resume(false),
        m_islandCount(0), m_islandTopoValid(false),
        m_skipValid(false), m_lastIslandsSkipped(0), m_lastIslandsTotal(0) {}

    /** Parameters controlling the data preparation. */
    struct DataParams
    {
        bool        equalizeMasses  = false;        // Use the geometric mean of the nodes' masses instead of the individual masses.
        bool        centerBonds     = false;        // Place the bond position halfway between adjoining nodes' CoMs.
    };

    /** Parameters controlling the solver behavior. */
    struct SolverParams
    {
        uint32_t    maxIter         = 0;        // The maximum number of iterations.  If 0, use CGNR for default value.
        float       tolerance       = 1.e-6f;   // The relative tolerance threshold for convergence.  Iteration will stop when this is reached.
        bool        warmStart       = false;    // Whether or not to use the solve function's 'impulses' parameter as a starting input vector.
        bool        islandAware     = false;    // Solve each disconnected component ("island") independently. With <=1 island this falls back to the
                                                // whole-graph path, so it is bit-identical to the legacy solve; with multiple islands the result matches
                                                // within solver tolerance (same scaling, same matrix entries; only the CG iteration is partitioned).
        bool        skipSettled   = false;    // Requires islandAware. Skip any island whose velocity inputs are bit-identical to its last solve and that
                                                // already converged: the solve would be a no-op (0 iterations), so its bond impulses/stresses are kept. Any
                                                // input change (new contact, wake) differs the velocity and re-solves that island the same frame. Never evicts.
    };

    /**
     * Build the internal representation of the stress network from nodes and bonds.
     * This only needs to be called initially, and any time the nodes or bonds change.
     * 
     * \param[in]   nodes   Array of SolverNodeS (scalar inertia).
     * \param[in]   N_nodes Number of elements in the nodes array.
     * \param[in]   bonds   Array of SolverBond.  The node indices in each bond entry correspond to the ordering of the nodes array.
     * \param[in]   N_bonds Number of elements in the bonds array.
     * \param[in]   params  Parameters affecting the processing of the input data (see DataParams).
     */
    void        prepare(const SolverNodeS* nodes, uint32_t N_nodes, const SolverBond* bonds, uint32_t N_bonds, const DataParams& params);

    /**
     * Solve for the bond impulses given the velocities of each node.  The function prepare(...) must be called
     * before this can be used, but then solve(...) may be called multiple times.
     * 
     * The vector elements (impulses and velocities) hold linear and angular parts.
     * 
     * \param[out]  impulses    Output array of impulses exerted by each bond.  For a warm or hot start, this is also used as an input.
     *                          Must be of length N_bonds passed into the prepare(...) function.
     * \param[in]   velocities  Input array of external velocities on each node.  Must be of length N_nodes passed into the prepare(...) function.
     * \param[in]   params      Parameters affecting the solver characteristics (see SolverParams).
     * \param[out]  error_sq    (Optional) If not NULL, *error_sq will be filled with the angular and linear square errors (solver residuals).  Default = NULL.
     * \param[in]   resume      (Optional) Set to true if impulses and velocities have not changed since last call, to resume solving.  Default = false.
     * 
     * \return the number of iterations taken to converge, if it converges.  Otherwise, returns minus the number of iterations before exiting.
     */
    int         solve(AngLin6* impulses, const AngLin6* velocities, const SolverParams& params, AngLin6ErrorSq* error_sq = nullptr, bool resume = false);

    /**
     * Removes the indexed bond from the solver.
     * 
     * \param[in]   bondIndex   The index of the bond to remove.  Must be less than getBondCount().
     * 
     * \return true iff successful.
     */
    bool        removeBond(uint32_t bondIndex);

    /**
     * \return the number of nodes in the stress network.  (Set by prepare(...).)
     */
    uint32_t    getNodeCount() const { return (uint32_t)m_recip_sqrt_I.size(); }

    /**
     * \return the number of bonds in the stress network.  (Set by prepare(...), possibly reduced by removeBond(...).)
     */
    uint32_t    getBondCount() const { return (uint32_t)m_couplings.size(); }

    /** \return number of islands skipped as settled in the last island-aware solve. */
    uint32_t    getLastIslandsSkipped() const { return m_lastIslandsSkipped; }

    /** \return number of islands processed in the last island-aware solve (0 if the whole-graph path ran). */
    uint32_t    getLastIslandsTotal() const { return m_lastIslandsTotal; }

    /**
     * \return whether or not the solver uses SIMD.  If the device and OS support SSE, AVX, and FMA instruction sets, SIMD is used. 
     */
    static bool usingSIMD() { return s_use_simd; }

protected:
    /**
     * Solve each disconnected component ("island") of the stress network independently, by gathering
     * each island's bonds/nodes into a contiguous sub-system, running the same CGNR/scaling as solve(),
     * and scattering the bond impulses back. The numeric kernels are unchanged; only the iteration is
     * partitioned. Static (zero-mass) nodes carry no coupling and act as cut points between islands.
     *
     * \param[out]  handled  Set to false when there is at most one island (the caller should then use
     *                       the whole-graph path, which is bit-identical to the legacy solve).
     * \return iteration count summed over islands (>=0 if every island converged, otherwise negative).
     */
    int         solveIslandAware(AngLin6* impulses, const AngLin6* velocities, const SolverParams& params, AngLin6ErrorSq* error_sq, bool& handled);

    float                   m_mass_scale;
    float                   m_length_scale;
    POD_Buffer<InertiaS>    m_recip_sqrt_I;
    POD_Buffer<Coupling>    m_couplings;
    BondMatrixS             m_B;
    POD_Buffer<AngLin6>     m_rhs;
    POD_Buffer<AngLin6>     m_B_scratch;
    POD_Buffer<AngLin6>     m_solver_cache;
    bool                    m_can_resume;

    // Island-aware solve scratch (only sized/used when SolverParams::islandAware and there is >1 island)
    std::vector<uint32_t>   m_uf;               // union-find parent over nodes
    std::vector<uint32_t>   m_bondIsland;       // bond -> island id
    std::vector<uint32_t>   m_islandBondBegin;  // CSR-style start offset per island (size islandCount+1)
    std::vector<uint32_t>   m_bondsByIsland;    // bond indices grouped contiguously by island
    std::vector<uint32_t>   m_rootIsland;       // scratch: node root -> compact island id (used only on rebuild)
    std::vector<uint32_t>   m_cursor;           // scratch: per-island write cursor (used only on rebuild)

    // Island grouping cache. The union-find / bond->island assignment / CSR grouping above are a pure
    // function of topology (bond node indices and which nodes are static), so they are identical every
    // frame until the topology changes. They are rebuilt only when m_islandTopoValid is false — set by
    // prepare()/removeBond() — and reused (bit-identically) otherwise, so the per-island solves are
    // unchanged. m_islandCount caches the island count (including the <=1 whole-graph-fallback case).
    uint32_t                m_islandCount;
    bool                    m_islandTopoValid;
    std::vector<uint32_t>   m_g2l;              // global node -> island-local node index
    std::vector<uint32_t>   m_g2lStamp;         // version stamp so m_g2l can be reused without clearing
    std::vector<uint32_t>   m_l2g;              // island-local node index -> global node
    POD_Buffer<Coupling>    m_localC;           // island-local couplings (node indices renumbered local)
    POD_Buffer<InertiaS>    m_localI;           // island-local recip_sqrt_I
    POD_Buffer<AngLin6>     m_localImpulses;    // island-local impulses (gather in, scatter out)

    // Settled skip state (Stage 3): per-node last-solved velocity + per-node convergence, so an
    // island whose inputs are bit-identical to its last solve and already converged can be skipped
    // (the solve would be a no-op). Invalidated on any topology change (prepare/removeBond) and on
    // the whole-graph fallback, so a fresh baseline is always re-established before any skip.
    POD_Buffer<AngLin6>     m_lastVel;
    std::vector<uint8_t>    m_nodeConverged;
    bool                    m_skipValid;
    uint32_t                m_lastIslandsSkipped;
    uint32_t                m_lastIslandsTotal;

    static const bool       s_use_simd;
};
