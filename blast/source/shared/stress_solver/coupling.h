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

#include "solver_types.h"
#include "anglin6.h"

#include "NvCMath.h"


/**
 * Bond coupling data used as a representation of a block column of a "coupling matrix" C,
 * which has exactly two non-zero blocks.  The non-zero blocks are of the form
 * 
 *               /   1  ~r_ij \
 *   C_ij = s_ij |            |.
 *               \   0    1   /
 * 
 * This represents the coupling of node i by bond j.  The scalar s_ij is +/-1, and for each
 * bond (column j of C) s_ij must take on both signs.  The matrix factor is again composed
 * of blocks, each element a 3x3 matrix.  The 0 and 1's are just multiples of the unit matrix,
 * and ~r_ij is the 3x3 antisymmetric matrix representing "crossing with the vector r_ij on the
 * left" (i.e. (~u)*v = (u) x (v)).  The vector r_ij represents the displacement from node i's
 * CoM to bond j's centroid.
 */

SIMD_ALIGN_32
(
struct Coupling
{
    NvcVec3 offset0;
    uint32_t node0;
    NvcVec3 offset1;
    uint32_t node1;
}
);


template <typename Elem, typename Scalar = Float_Scalar>
struct CouplingMatrixOps
{
    /**
     * Sparse matrix-vector multiply y = C*x, where C is a "coupling matrix" represented by columns
     * of type Coupling (see the comments for Coupling).
     *
     * \param[out]  y   Resulting column Elem vector of length M.
     * \param[in]   C   Input M x N coupling matrix.
     * \param[in]   x   Input column Elem vector of length N.
     * \param[in]   M   The number of rows in y and C.
     * \param[in]   N   The number of rows in x and columns in C.
     */
    inline void
    rmul(Elem* y, const Coupling* C, const Elem* x, uint32_t M, uint32_t N)
    {
        memset(y, 0, sizeof(AngLin6)*M);
        for (uint32_t j = 0 ; j < N; ++j)
        {
            const Coupling& c = C[j];
            const AngLin6& x_j = x[j];
            AngLin6& y0 = y[c.node0];
            AngLin6& y1 = y[c.node1];
            y0.ang += x_j.ang - (c.offset0^x_j.lin);
            y0.lin += x_j.lin;
            y1.ang -= x_j.ang - (c.offset1^x_j.lin);
            y1.lin -= x_j.lin;
        }
    }

    /**
     * Sparse matrix-vector multiply y = x*C, where C is a "coupling matrix" represented by columns
     * of type Coupling (see the comments for Coupling).
     *
     * \param[out]  y   Resulting row Elem vector of length N.
     * \param[in]   x   Input row Elem vector, must be long enough to be indexed by all values in B's representation.
     * \param[in]   C   Input M x N couping matrix.
     * \param[in]   M   The number of columns in x and rows in C.
     * \param[in]   N   The number of columns in y and C.
     */
    inline void
    lmul(Elem* y, const Elem* x, const Coupling* C, uint32_t M, uint32_t N)
    {
        NV_UNUSED(M);
        for (uint32_t j = 0; j < N; ++j)
        {
            const Coupling& c = C[j];
            const AngLin6& x0 = x[c.node0];
            const AngLin6& x1 = x[c.node1];
            AngLin6& y_j = y[j];
            y_j.ang = x0.ang - x1.ang;
            y_j.lin = x0.lin - x1.lin + (c.offset0^x0.ang) - (c.offset1^x1.ang);
        }
    }
};

#if !defined(STRESS_SOLVER_NO_SIMD)
#if defined(STRESS_SOLVER_WASM_SIMD_DIRECT)
/**
 * Direct wasm-simd128 coupling matrix ops — the prime SIMD target per the
 * profile (rmul + lmul = ~59% of the CGNR solve, cross-product alone is ~50%).
 *
 * Same math as the AVX specialization below, but each `__m256` op is unrolled
 * into the two natural 128-bit halves (one for the AngLin6's ang vector, one
 * for its lin vector).  Cross products are 6 v128 ops (4 shuffles + 2 muls +
 * 1 sub + 1 final shuffle); the AVX path packs two crosses into one `__m256`
 * via `pair_cross3`, but that lowers to the same primitive ops on wasm — with
 * extra `_mm256_set_m128` / `_mm256_extractf128_ps` round-trips that this
 * version skips.
 */
#include <wasm_simd128.h>

template <typename Elem>
struct CouplingMatrixOps<Elem, SIMD_Scalar>
{
    inline void
    rmul(Elem* y, const Coupling* C, const Elem* x, uint32_t M, uint32_t N)
    {
        memset(y, 0, sizeof(AngLin6)*M);
        for (uint32_t j = 0; j < N; ++j)
        {
            const Coupling& c = C[j];
            const AngLin6& x_j = x[j];
            AngLin6& y0 = y[c.node0];
            AngLin6& y1 = y[c.node1];

            v128_t xa = wasm_v128_load(&x_j.ang.x);
            v128_t xl = wasm_v128_load(&x_j.lin.x);
            v128_t y0a = wasm_v128_load(&y0.ang.x);
            v128_t y0l = wasm_v128_load(&y0.lin.x);
            v128_t y1a = wasm_v128_load(&y1.ang.x);
            v128_t y1l = wasm_v128_load(&y1.lin.x);
            v128_t off0 = wasm_v128_load(&c.offset0.x);
            v128_t off1 = wasm_v128_load(&c.offset1.x);

            // y0.ang += x.ang - off0 × x.lin
            // y0.lin += x.lin
            // y1.ang -= x.ang - off1 × x.lin
            // y1.lin -= x.lin
            v128_t cross0 = cross3(off0, xl);
            v128_t cross1 = cross3(off1, xl);

            wasm_v128_store(&y0.ang.x, wasm_f32x4_add(y0a, wasm_f32x4_sub(xa, cross0)));
            wasm_v128_store(&y0.lin.x, wasm_f32x4_add(y0l, xl));
            wasm_v128_store(&y1.ang.x, wasm_f32x4_sub(y1a, wasm_f32x4_sub(xa, cross1)));
            wasm_v128_store(&y1.lin.x, wasm_f32x4_sub(y1l, xl));
        }
    }

    inline void
    lmul(Elem* y, const Elem* x, const Coupling* C, uint32_t M, uint32_t N)
    {
        NV_UNUSED(M);
        for (uint32_t j = 0; j < N; ++j)
        {
            const Coupling& c = C[j];
            const AngLin6& x0 = x[c.node0];
            const AngLin6& x1 = x[c.node1];
            AngLin6& y_j = y[j];

            v128_t x0a = wasm_v128_load(&x0.ang.x), x0l = wasm_v128_load(&x0.lin.x);
            v128_t x1a = wasm_v128_load(&x1.ang.x), x1l = wasm_v128_load(&x1.lin.x);
            v128_t off0 = wasm_v128_load(&c.offset0.x);
            v128_t off1 = wasm_v128_load(&c.offset1.x);

            // y_j.ang = x0.ang - x1.ang
            // y_j.lin = x0.lin - x1.lin + off0 × x0.ang - off1 × x1.ang
            v128_t cross0 = cross3(off0, x0a);
            v128_t cross1 = cross3(off1, x1a);

            wasm_v128_store(&y_j.ang.x, wasm_f32x4_sub(x0a, x1a));
            wasm_v128_store(&y_j.lin.x,
                wasm_f32x4_add(wasm_f32x4_sub(x0l, x1l),
                               wasm_f32x4_sub(cross0, cross1)));
        }
    }

private:
    // Cross product on the first 3 lanes of a v128 (lane 3 is don't-care).
    //   (a × b).x = a.y*b.z - a.z*b.y
    //   (a × b).y = a.z*b.x - a.x*b.z
    //   (a × b).z = a.x*b.y - a.y*b.x
    // Computed as a.yzx * b.zxy − a.zxy * b.yzx, no final shuffle needed
    // because the result lanes already come out (.x, .y, .z, _).
    static inline v128_t cross3(v128_t a, v128_t b)
    {
        v128_t a_yzx = wasm_i32x4_shuffle(a, a, 1, 2, 0, 3);
        v128_t b_zxy = wasm_i32x4_shuffle(b, b, 2, 0, 1, 3);
        v128_t a_zxy = wasm_i32x4_shuffle(a, a, 2, 0, 1, 3);
        v128_t b_yzx = wasm_i32x4_shuffle(b, b, 1, 2, 0, 3);
        return wasm_f32x4_sub(wasm_f32x4_mul(a_yzx, b_zxy),
                              wasm_f32x4_mul(a_zxy, b_yzx));
    }
};
#else
template <typename Elem>
struct CouplingMatrixOps<Elem, SIMD_Scalar>
{
    /**
     * Sparse matrix-vector multiply y = C*x, where C is a "coupling matrix" represented by columns
     * of type Coupling (see the comments for Coupling).
     *
     * \param[out]  y   Resulting column Elem vector of length M.
     * \param[in]   C   Input M x N coupling matrix.
     * \param[in]   x   Input column Elem vector of length N.
     * \param[in]   M   The number of rows in y and C.
     * \param[in]   N   The number of rows in x and columns in C.
     */
    inline void
    rmul(Elem* y, const Coupling* C, const Elem* x, uint32_t M, uint32_t N)
    {
        memset(y, 0, sizeof(AngLin6)*M);
        for (uint32_t j = 0 ; j < N; ++j)
        {
            const Coupling& c = C[j];
            const AngLin6& x_j = x[j];
            AngLin6& y0 = y[c.node0];
            AngLin6& y1 = y[c.node1];

            __m256 _x = _mm256_load_ps(&x_j.ang.x);
            __m256 _y0 = _mm256_load_ps(&y0.ang.x);
            __m256 _y1 = _mm256_load_ps(&y1.ang.x);
            __m256 _c = _mm256_load_ps(&c.offset0.x);

            _y0 = _mm256_add_ps(_y0, _x);
            _y1 = _mm256_sub_ps(_y1, _x);

            __m128 _xl = _mm256_extractf128_ps(_x, 1);
            __m256 _a = pair_cross3(_mm256_set_m128(_xl, _xl), _c);
            _y0 = _mm256_add_ps(_y0, _mm256_set_m128(_mm_setzero_ps(), _mm256_castps256_ps128(_a)));
            _y1 = _mm256_sub_ps(_y1, _mm256_set_m128(_mm_setzero_ps(), _mm256_extractf128_ps(_a, 1)));

            _mm256_store_ps(&y0.ang.x, _y0);
            _mm256_store_ps(&y1.ang.x, _y1);
        }
    }

    /**
     * Sparse matrix-vector multiply y = x*C, where C is a "coupling matrix" represented by columns
     * of type Coupling (see the comments for Coupling).
     *
     * \param[out]  y   Resulting row Elem vector of length N.
     * \param[in]   x   Input row Elem vector, must be long enough to be indexed by all values in B's representation.
     * \param[in]   C   Input M x N couping matrix.
     * \param[in]   M   The number of columns in x and rows in C.
     * \param[in]   N   The number of columns in y and C.
     */
    inline void
    lmul(Elem* y, const Elem* x, const Coupling* C, uint32_t M, uint32_t N)
    {
        NV_UNUSED(M);
        for (uint32_t j = 0; j < N; ++j)
        {
            const Coupling& c = C[j];
            const AngLin6& x0 = x[c.node0];
            const AngLin6& x1 = x[c.node1];
            AngLin6& y_j = y[j];

            __m256 _x0 = _mm256_load_ps(&x0.ang.x);
            __m256 _x1 = _mm256_load_ps(&x1.ang.x);
            __m256 _c = _mm256_load_ps(&c.offset0.x);

            __m256 _y = _mm256_sub_ps(_x0, _x1);

            __m256 _a = pair_cross3(_c, _mm256_set_m128(_mm256_castps256_ps128(_x1), _mm256_castps256_ps128(_x0)));
            _y = _mm256_add_ps(_y, _mm256_set_m128(_mm_sub_ps(_mm256_castps256_ps128(_a), _mm256_extractf128_ps(_a, 1)), _mm_setzero_ps()));

            _mm256_store_ps(&y_j.ang.x, _y);
        }
    }

private:
    inline __m256
    pair_cross3(const __m256& v0, const __m256& v1)
    {
        __m256 prep0 = _mm256_shuffle_ps(v0, v0, 0xc9);
        __m256 prep1 = _mm256_shuffle_ps(v1, v1, 0xc9);
        __m256 res_shuffled = _mm256_fmsub_ps(v0, prep1, _mm256_mul_ps(prep0, v1));
        return _mm256_shuffle_ps(res_shuffled, res_shuffled, 0xc9);
    }
};
#endif // STRESS_SOLVER_WASM_SIMD_DIRECT
#endif // !STRESS_SOLVER_NO_SIMD
