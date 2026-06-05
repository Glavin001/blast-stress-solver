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

#include "NvCMath.h"

#if !defined(STRESS_SOLVER_NO_SIMD)
#include "simd/simd.h"
#else
#ifndef SIMD_ALIGN_16
#define SIMD_ALIGN_16(code) NV_ALIGN_PREFIX(16) code NV_ALIGN_SUFFIX(16)
#endif
#ifndef SIMD_ALIGN_32
#define SIMD_ALIGN_32(code) NV_ALIGN_PREFIX(32) code NV_ALIGN_SUFFIX(32)
#endif
#endif


/**
 * Holds an angular and linear component, for angular and linear velocities, accelerations, impulses, torques and forces, etc.
 */
SIMD_ALIGN_32(
struct AngLin6
{
    SIMD_ALIGN_16(NvcVec3 ang);
    SIMD_ALIGN_16(NvcVec3 lin);
}
);


/**
 * Holds the angular and linear components of the calculated error.
 */
struct AngLin6ErrorSq
{
    float ang, lin;
};


/**
 * SISD AngLin6 operations.
 */
template<typename Scalar = float>
struct AngLin6Ops
{
    /** r = x + y */
    inline  void add(AngLin6& r, const AngLin6& x, const AngLin6& y)            { r.ang = x.ang + y.ang; r.lin = x.lin + y.lin; }

    /** r = x - y */
    inline  void sub(AngLin6& r, const AngLin6& x, const AngLin6& y)            { r.ang = x.ang - y.ang; r.lin = x.lin - y.lin; }

    /** r = c*x + y */
    inline  void madd(AngLin6& r, float c, const AngLin6& x, const AngLin6& y)  { r.ang = c*x.ang + y.ang; r.lin = c*x.lin + y.lin; }

    /** r = -c*x + y */
    inline  void nmadd(AngLin6& r, float c, const AngLin6& x, const AngLin6& y) { r.ang = y.ang - c*x.ang; r.lin = y.lin - c*x.lin; }

    /** Vector add */
    inline  void vadd(AngLin6* r, const AngLin6* x, const AngLin6* y, uint32_t N)               { while (N--) add(*r++, *x++, *y++); }

    /** Vector sub */
    inline  void vsub(AngLin6* r, const AngLin6* x, const AngLin6* y, uint32_t N)               { while (N--) sub(*r++, *x++, *y++); }

    /** Vector madd */
    inline  void vmadd(AngLin6* r, float c, const AngLin6* x, const AngLin6* y, uint32_t N)     { while (N--) madd(*r++, c, *x++, *y++); }

    /** Vector nmadd */
    inline  void vnmadd(AngLin6* r, float c, const AngLin6* x, const AngLin6* y, uint32_t N)    { while (N--) nmadd(*r++, c, *x++, *y++); }

    /**
     * Vector-of-vectors dot product.
     * 
     * \param[in]   v   Vector of AngLin6, of length N.
     * \param[in]   w   Vector of AngLin6, of length N.
     * \param[in]   N   Number of elements in v and w.
     * 
     * return (v|w).
     */
    inline float
    dot(const AngLin6* v, const AngLin6* w, uint32_t N)
    {
        float result = 0.0f;
        for (uint32_t i = 0; i < N; ++i)
        {
            const AngLin6& v_i = v[i];
            const AngLin6& w_i = w[i];
            result += (v_i.ang|w_i.ang) + (v_i.lin|w_i.lin);
        }
        return result;
    }

    /**
     * Vector-of-vectors length squared.
     * 
     * Equivalent to dot(v, v N), but could be faster in some cases
     * 
     * \param[in]   v   Vector of AngLin6, of length N.
     * \param[in]   N   Number of elements in v.
     * 
     * return |v|^2.
     */
    inline float
    length_sq(const AngLin6* v, uint32_t N)
    {
        float result = 0.0f;
        for (uint32_t i = 0; i < N; ++i)
        {
            const AngLin6& v_i = v[i];
            result += (v_i.ang|v_i.ang) + (v_i.lin|v_i.lin);
        }
        return result;
    }

    /**
     * Vector-of-vectors length squared, split into angular and linear contributions.
     * 
     * \param[out]  error_sq    Sum of the squared angular and linear parts of v.
     * \param[in]   v           Vector of AngLin6, of length N.
     * \param[in]   N           Number of elements in v.
     * 
     * \return the sum of the squared angular and linear errors, error.ang + error.lin.
     */
    inline float
    calculate_error(AngLin6ErrorSq& error_sq, const AngLin6* v, uint32_t N)
    {
        error_sq.ang = error_sq.lin = 0.0f;
        for (uint32_t i = 0; i < N; ++i)
        {
            const AngLin6& v_i = v[i];
            error_sq.ang += v_i.ang|v_i.ang;
            error_sq.lin += v_i.lin|v_i.lin;
        }
        return error_sq.ang + error_sq.lin;
    }
};


#if !defined(STRESS_SOLVER_NO_SIMD)
#if defined(STRESS_SOLVER_WASM_SIMD_DIRECT)
/**
 * Direct wasm-simd128 AngLin6 operations.
 *
 * Mirrors the AVX specialization below, but written against `<wasm_simd128.h>`
 * straight rather than through `__m256` (which emscripten emulates as two
 * `__m128` halves + struct bookkeeping when targeting wasm).  Each AngLin6 is
 * 32 bytes — exactly two v128 — so the natural representation is
 * `(v128 ang_part, v128 lin_part)` with the trailing pad in lane 3.
 *
 * The Scalar template parameter stays as `__m128` for source compatibility
 * with the CGNR template; on emcc native AVX targets `__m128` and `v128_t`
 * are layout-compatible (both are 16-byte 4×f32 vectors), and the explicit
 * bitcasts here are no-ops at codegen time.
 *
 * Why this exists alongside the `STRESS_SOLVER_WASM_SIMD` path: the
 * AVX-intrinsic path through `__m256` was measured at +24% over scalar on
 * the mini-city replay, but it pays a per-op wrapper tax that hurts at
 * smaller scales.  Direct v128 should match at scale and not regress at
 * small scale — same work, less bookkeeping.
 */
#include <wasm_simd128.h>

template<>
struct AngLin6Ops<__m128>
{
    /** r = x + y */
    inline void
    add(AngLin6& r, const AngLin6& x, const AngLin6& y)
    {
        v128_t xa = wasm_v128_load(&x.ang.x), xl = wasm_v128_load(&x.lin.x);
        v128_t ya = wasm_v128_load(&y.ang.x), yl = wasm_v128_load(&y.lin.x);
        wasm_v128_store(&r.ang.x, wasm_f32x4_add(xa, ya));
        wasm_v128_store(&r.lin.x, wasm_f32x4_add(xl, yl));
    }

    /** r = x - y */
    inline void
    sub(AngLin6& r, const AngLin6& x, const AngLin6& y)
    {
        v128_t xa = wasm_v128_load(&x.ang.x), xl = wasm_v128_load(&x.lin.x);
        v128_t ya = wasm_v128_load(&y.ang.x), yl = wasm_v128_load(&y.lin.x);
        wasm_v128_store(&r.ang.x, wasm_f32x4_sub(xa, ya));
        wasm_v128_store(&r.lin.x, wasm_f32x4_sub(xl, yl));
    }

    /** r = c*x + y (c is splatted across the 4 lanes of a v128) */
    inline void
    madd(AngLin6& r, __m128 c, const AngLin6& x, const AngLin6& y)
    {
        v128_t cv = (v128_t)c;
        v128_t xa = wasm_v128_load(&x.ang.x), xl = wasm_v128_load(&x.lin.x);
        v128_t ya = wasm_v128_load(&y.ang.x), yl = wasm_v128_load(&y.lin.x);
        wasm_v128_store(&r.ang.x, wasm_f32x4_add(wasm_f32x4_mul(cv, xa), ya));
        wasm_v128_store(&r.lin.x, wasm_f32x4_add(wasm_f32x4_mul(cv, xl), yl));
    }

    /** r = -c*x + y = y - c*x */
    inline void
    nmadd(AngLin6& r, __m128 c, const AngLin6& x, const AngLin6& y)
    {
        v128_t cv = (v128_t)c;
        v128_t xa = wasm_v128_load(&x.ang.x), xl = wasm_v128_load(&x.lin.x);
        v128_t ya = wasm_v128_load(&y.ang.x), yl = wasm_v128_load(&y.lin.x);
        wasm_v128_store(&r.ang.x, wasm_f32x4_sub(ya, wasm_f32x4_mul(cv, xa)));
        wasm_v128_store(&r.lin.x, wasm_f32x4_sub(yl, wasm_f32x4_mul(cv, xl)));
    }

    inline  void vadd(AngLin6* r, const AngLin6* x, const AngLin6* y, uint32_t N)               { while (N--) add(*r++, *x++, *y++); }
    inline  void vsub(AngLin6* r, const AngLin6* x, const AngLin6* y, uint32_t N)               { while (N--) sub(*r++, *x++, *y++); }
    inline  void vmadd(AngLin6* r, __m128 c, const AngLin6* x, const AngLin6* y, uint32_t N)    { while (N--) madd(*r++, c, *x++, *y++); }
    inline  void vnmadd(AngLin6* r, __m128 c, const AngLin6* x, const AngLin6* y, uint32_t N)   { while (N--) nmadd(*r++, c, *x++, *y++); }

    /** dot(v, w, N) = sum_i (v[i].ang . w[i].ang + v[i].lin . w[i].lin),
     *  returned in lane 0 of the result (the CGNR code uses it as a Scalar). */
    inline __m128
    dot(const AngLin6* v, const AngLin6* w, uint32_t N)
    {
        v128_t acc = wasm_f32x4_splat(0.0f);
        for (uint32_t i = 0; i < N; ++i)
        {
            v128_t va = wasm_v128_load(&v[i].ang.x), vl = wasm_v128_load(&v[i].lin.x);
            v128_t wa = wasm_v128_load(&w[i].ang.x), wl = wasm_v128_load(&w[i].lin.x);
            // Mask off lane 3 (struct pad) so it doesn't contribute. Doing it via
            // `replace_lane` instead of `wasm_v128_and` keeps the inner loop a clean
            // mul/replace/add sequence the JIT folds tightly.
            v128_t pa = wasm_f32x4_replace_lane(wasm_f32x4_mul(va, wa), 3, 0.0f);
            v128_t pl = wasm_f32x4_replace_lane(wasm_f32x4_mul(vl, wl), 3, 0.0f);
            acc = wasm_f32x4_add(acc, wasm_f32x4_add(pa, pl));
        }
        return (__m128)horizontalSum(acc);
    }

    inline __m128
    length_sq(const AngLin6* v, uint32_t N)
    {
        v128_t acc = wasm_f32x4_splat(0.0f);
        for (uint32_t i = 0; i < N; ++i)
        {
            v128_t va = wasm_v128_load(&v[i].ang.x), vl = wasm_v128_load(&v[i].lin.x);
            v128_t pa = wasm_f32x4_replace_lane(wasm_f32x4_mul(va, va), 3, 0.0f);
            v128_t pl = wasm_f32x4_replace_lane(wasm_f32x4_mul(vl, vl), 3, 0.0f);
            acc = wasm_f32x4_add(acc, wasm_f32x4_add(pa, pl));
        }
        return (__m128)horizontalSum(acc);
    }

    /** Same |v|^2 split into ang_sq and lin_sq; keep separate accumulators. */
    inline __m128
    calculate_error(AngLin6ErrorSq& error_sq, const AngLin6* v, uint32_t N)
    {
        v128_t acc_a = wasm_f32x4_splat(0.0f), acc_l = wasm_f32x4_splat(0.0f);
        for (uint32_t i = 0; i < N; ++i)
        {
            v128_t va = wasm_v128_load(&v[i].ang.x), vl = wasm_v128_load(&v[i].lin.x);
            v128_t pa = wasm_f32x4_replace_lane(wasm_f32x4_mul(va, va), 3, 0.0f);
            v128_t pl = wasm_f32x4_replace_lane(wasm_f32x4_mul(vl, vl), 3, 0.0f);
            acc_a = wasm_f32x4_add(acc_a, pa);
            acc_l = wasm_f32x4_add(acc_l, pl);
        }
        const float ang_sq = horizontalSumScalar(acc_a);
        const float lin_sq = horizontalSumScalar(acc_l);
        error_sq.ang = ang_sq;
        error_sq.lin = lin_sq;
        return (__m128)wasm_f32x4_splat(ang_sq + lin_sq);
    }

private:
    /** Sum the 4 lanes of a v128, broadcast the result across all 4 lanes. */
    static inline v128_t horizontalSum(v128_t v)
    {
        v128_t hi = wasm_i32x4_shuffle(v, v, 2, 3, 0, 1);
        v128_t s1 = wasm_f32x4_add(v, hi);
        v128_t hi2 = wasm_i32x4_shuffle(s1, s1, 1, 0, 3, 2);
        return wasm_f32x4_add(s1, hi2);
    }
    /** Sum the 4 lanes of a v128 and return as scalar. */
    static inline float horizontalSumScalar(v128_t v)
    {
        v128_t s = horizontalSum(v);
        return wasm_f32x4_extract_lane(s, 0);
    }
};
#else
/**
 * SIMD AngLin6 operations (x86 AVX or emscripten's AVX intrinsic emulation).
 */
template<>
struct AngLin6Ops<__m128>
{
    /** r = x + y */
    inline void
    add(AngLin6& r, const AngLin6& x, const AngLin6& y)
    {
        __m256 _x = _mm256_load_ps(&x.ang.x);
        __m256 _y = _mm256_load_ps(&y.ang.x);
        __m256 _r = _mm256_add_ps(_x, _y);
        _mm256_store_ps(&r.ang.x, _r);
    }

    /** r = x - y */
    inline void
    sub(AngLin6& r, const AngLin6& x, const AngLin6& y)
    {
        __m256 _x = _mm256_load_ps(&x.ang.x);
        __m256 _y = _mm256_load_ps(&y.ang.x);
        __m256 _r = _mm256_sub_ps(_x, _y);
        _mm256_store_ps(&r.ang.x, _r);
    }

    /** r = c*x + y */
    inline void
    madd(AngLin6& r, __m128 c, const AngLin6& x, const AngLin6& y)
    {
        __m256 _c = _mm256_set_m128(c, c);
        __m256 _x = _mm256_load_ps(&x.ang.x);
        __m256 _y = _mm256_load_ps(&y.ang.x);
        __m256 _r = _mm256_fmadd_ps(_c, _x, _y);
        _mm256_store_ps(&r.ang.x, _r);
    }

    /** r = -c*x + y */
    inline void
    nmadd(AngLin6& r, __m128 c, const AngLin6& x, const AngLin6& y)
    {
        __m256 _c = _mm256_set_m128(c, c);
        __m256 _x = _mm256_load_ps(&x.ang.x);
        __m256 _y = _mm256_load_ps(&y.ang.x);
        __m256 _r = _mm256_fnmadd_ps(_c, _x, _y);
        _mm256_store_ps(&r.ang.x, _r);
    }

    /** Vector add */
    inline  void vadd(AngLin6* r, const AngLin6* x, const AngLin6* y, uint32_t N)               { while (N--) add(*r++, *x++, *y++); }

    /** Vector sub */
    inline  void vsub(AngLin6* r, const AngLin6* x, const AngLin6* y, uint32_t N)               { while (N--) sub(*r++, *x++, *y++); }

    /** Vector madd */
    inline  void vmadd(AngLin6* r, __m128 c, const AngLin6* x, const AngLin6* y, uint32_t N)    { while (N--) madd(*r++, c, *x++, *y++); }

    /** Vector nmadd */
    inline  void vnmadd(AngLin6* r, __m128 c, const AngLin6* x, const AngLin6* y, uint32_t N)   { while (N--) nmadd(*r++, c, *x++, *y++); }

    /**
     * Vector-of-vectors dot product.
     * 
     * \param[in]   v   Vector of AngLin6, of length N.
     * \param[in]   w   Vector of AngLin6, of length N.
     * \param[in]   N   Number of elements in v and w.
     * 
     * return (v|w).
     */
    inline __m128
    dot(const AngLin6* v, const AngLin6* w, uint32_t N)
    {
        __m256 _res = _mm256_setzero_ps();
        for (uint32_t i = 0; i < N; ++i)
        {
            __m256 _v = _mm256_load_ps((const float*)(v+i));
            __m256 _w = _mm256_load_ps((const float*)(w+i));
            _res = _mm256_add_ps(_res, _mm256_dp_ps(_v, _w, 0x7f));
        }
        return _mm_add_ps(_mm256_castps256_ps128(_res), _mm256_extractf128_ps(_res, 1));
    }

    /**
     * Vector-of-vectors length squared.
     * 
     * Equivalent to dot(v, v N), but could be faster in some cases
     * 
     * \param[in]   v   Vector of AngLin6, of length N.
     * \param[in]   N   Number of elements in v.
     * 
     * return |v|^2.
     */
    inline __m128
    length_sq(const AngLin6* v, uint32_t N)
    {
        __m256 _res = _mm256_setzero_ps();
        for (uint32_t i = 0; i < N; ++i)
        {
            __m256 _v = _mm256_load_ps((const float*)(v+i));
            _res = _mm256_add_ps(_res, _mm256_dp_ps(_v, _v, 0x7f));
        }
        return _mm_add_ps(_mm256_castps256_ps128(_res), _mm256_extractf128_ps(_res, 1));
    }

    /**
     * Vector-of-vectors length squared, split into angular and linear contributions.
     * 
     * \param[out]  error_sq    Sum of the squared angular and linear parts of v.
     * \param[in]   v           Vector of AngLin6, of length N.
     * \param[in]   N           Number of elements in v.
     * 
     * \return the sum of the squared angular and linear errors, error.ang + error.lin.
     */
    inline __m128
    calculate_error(AngLin6ErrorSq& error_sq, const AngLin6* v, uint32_t N)
    {
        __m256 _res = _mm256_setzero_ps();
        for (uint32_t i = 0; i < N; ++i)
        {
            __m256 _v = _mm256_load_ps((const float*)(v+i));
            _res = _mm256_add_ps(_res, _mm256_dp_ps(_v, _v, 0x7f));
        }
        __m128 _ang_sq = _mm256_castps256_ps128(_res);
        __m128 _lin_sq = _mm256_extractf128_ps(_res, 1);

        _mm_store_ss(&error_sq.ang, _ang_sq);
        _mm_store_ss(&error_sq.lin, _lin_sq);

        return _mm_add_ps(_ang_sq, _lin_sq);
    }
};
#endif // STRESS_SOLVER_WASM_SIMD_DIRECT
#endif // !STRESS_SOLVER_NO_SIMD
