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

#if defined(STRESS_SOLVER_WASM_SIMD)
// WASM build with the AVX kernels rerouted through simde
// (https://github.com/simd-everywhere/simde) onto wasm-simd128.  The vendored
// headers in this directory key on `__m128`/`__m256` and the AVX/FMA intrinsics
// — simde provides one-to-one emulations of those that lower to v128 ops at
// compile time, so the existing AVX code in anglin6.h / coupling.h / inertia.h
// builds unchanged.
//
// SIMDE_ENABLE_NATIVE_ALIASES makes `_mm256_*` / `__m256` etc. expand to the
// `simde_*` / `simde__m256` equivalents without source changes elsewhere.
#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/sse.h>
#include <simde/x86/sse2.h>
#include <simde/x86/sse3.h>
#include <simde/x86/ssse3.h>
#include <simde/x86/sse4.1.h>
#include <simde/x86/avx.h>
#include <simde/x86/fma.h>

// On WASM there is no hardware FMA in baseline simd128 (only relaxed-simd has
// `f32x4.relaxed_madd`, and its rounding is implementation-defined so we keep
// off it for determinism).  simde's 128-bit FMA fallback runs a 4-iteration
// scalar loop through a `simde__m128_private` union; clang's vectorizer cannot
// re-lift it back to f32x4 ops, so the autovec scalar AngLin6 path actually
// out-codegens a "naïve" simde build.  Override the four FMA intrinsics the
// AngLin6 kernels use with explicit (mul + add/sub) v128 sequences so the
// emitted WASM stays vectorized.  Numeric impact: 2 roundings instead of 1 per
// op — the same as `-O3 -msimd128` already does for the scalar path, so this
// matches existing semantics rather than introducing new drift.
#include <wasm_simd128.h>

#undef _mm_fmadd_ps
#undef _mm_fnmadd_ps
#undef _mm_fmsub_ps
#define _mm_fmadd_ps(a, b, c)  wasm_f32x4_add(wasm_f32x4_mul((a), (b)), (c))
#define _mm_fnmadd_ps(a, b, c) wasm_f32x4_sub((c), wasm_f32x4_mul((a), (b)))
#define _mm_fmsub_ps(a, b, c)  wasm_f32x4_sub(wasm_f32x4_mul((a), (b)), (c))

// For the 256-bit variants simde represents __m256 as a union of two 128-bit
// halves; reach into that representation via simde's public accessors so the
// override keeps the same struct layout simde expects.
static inline simde__m256 _stress_mm256_fmadd_ps(simde__m256 a, simde__m256 b, simde__m256 c) {
    simde__m256_private a_ = simde__m256_to_private(a), b_ = simde__m256_to_private(b),
                        c_ = simde__m256_to_private(c), r_;
    r_.m128[0] = _mm_fmadd_ps(a_.m128[0], b_.m128[0], c_.m128[0]);
    r_.m128[1] = _mm_fmadd_ps(a_.m128[1], b_.m128[1], c_.m128[1]);
    return simde__m256_from_private(r_);
}
static inline simde__m256 _stress_mm256_fnmadd_ps(simde__m256 a, simde__m256 b, simde__m256 c) {
    simde__m256_private a_ = simde__m256_to_private(a), b_ = simde__m256_to_private(b),
                        c_ = simde__m256_to_private(c), r_;
    r_.m128[0] = _mm_fnmadd_ps(a_.m128[0], b_.m128[0], c_.m128[0]);
    r_.m128[1] = _mm_fnmadd_ps(a_.m128[1], b_.m128[1], c_.m128[1]);
    return simde__m256_from_private(r_);
}
static inline simde__m256 _stress_mm256_fmsub_ps(simde__m256 a, simde__m256 b, simde__m256 c) {
    simde__m256_private a_ = simde__m256_to_private(a), b_ = simde__m256_to_private(b),
                        c_ = simde__m256_to_private(c), r_;
    r_.m128[0] = _mm_fmsub_ps(a_.m128[0], b_.m128[0], c_.m128[0]);
    r_.m128[1] = _mm_fmsub_ps(a_.m128[1], b_.m128[1], c_.m128[1]);
    return simde__m256_from_private(r_);
}
#undef _mm256_fmadd_ps
#undef _mm256_fnmadd_ps
#undef _mm256_fmsub_ps
#define _mm256_fmadd_ps(a, b, c)  _stress_mm256_fmadd_ps((a), (b), (c))
#define _mm256_fnmadd_ps(a, b, c) _stress_mm256_fnmadd_ps((a), (b), (c))
#define _mm256_fmsub_ps(a, b, c)  _stress_mm256_fmsub_ps((a), (b), (c))

// Same story as FMA for the SSE4.1 dot-product: simde's WASM fallback runs a
// scalar reduction loop over `simde__m128_private`, which clang doesn't lift
// back to v128.  All three call sites in anglin6.h use the same imm8 = 0x7f
// (multiply lanes 0..2 of the AngLin6's ang OR lin half, broadcast the sum to
// all four lanes), so override just that case with an explicit horizontal-add
// sequence — `mul`, zero lane 3, shuffle + add twice, splat.  Five v128 ops
// vs. simde's per-lane scalar reduction.  simde uses `simde_mm_dp_ps` for its
// own internal callers (not the macro alias), so the override only affects the
// AngLin6 kernels.
static inline simde__m128 _stress_mm_dp_ps_7f(simde__m128 a, simde__m128 b) {
    v128_t aw = simde__m128_to_wasm_v128(a);
    v128_t bw = simde__m128_to_wasm_v128(b);
    v128_t mul = wasm_f32x4_mul(aw, bw);
    mul = wasm_f32x4_replace_lane(mul, 3, 0.0f);  // imm8 high nibble 0x7 drops lane 3
    v128_t hi  = wasm_i32x4_shuffle(mul, mul, 2, 3, 2, 3);
    v128_t s1  = wasm_f32x4_add(mul, hi);
    v128_t s1h = wasm_i32x4_shuffle(s1, s1, 1, 1, 1, 1);
    v128_t sum = wasm_f32x4_add(s1, s1h);
    v128_t bcast = wasm_i32x4_shuffle(sum, sum, 0, 0, 0, 0);  // imm8 low nibble 0xf splats
    return simde__m128_from_wasm_v128(bcast);
}
static inline simde__m256 _stress_mm256_dp_ps_7f(simde__m256 a, simde__m256 b) {
    simde__m256_private a_ = simde__m256_to_private(a), b_ = simde__m256_to_private(b), r_;
    r_.m128[0] = _stress_mm_dp_ps_7f(a_.m128[0], b_.m128[0]);
    r_.m128[1] = _stress_mm_dp_ps_7f(a_.m128[1], b_.m128[1]);
    return simde__m256_from_private(r_);
}
#undef _mm_dp_ps
#undef _mm256_dp_ps
// Only 0x7f is used in this codebase.  Trap any other value at compile time so
// a future change to anglin6.h that uses a different imm8 surfaces here rather
// than silently regressing to the scalar fallback.
#define _mm_dp_ps(a, b, imm8) \
    ((imm8) == 0x7f ? _stress_mm_dp_ps_7f((a), (b)) \
                    : (__builtin_unreachable(), simde_mm_dp_ps((a), (b), (imm8))))
#define _mm256_dp_ps(a, b, imm8) \
    ((imm8) == 0x7f ? _stress_mm256_dp_ps_7f((a), (b)) \
                    : (__builtin_unreachable(), simde_mm256_dp_ps((a), (b), (imm8))))

#else
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>
#endif

#if defined(__GNUC__) && !defined(STRESS_SOLVER_WASM_SIMD)  // missing with gcc; simde already provides it
#define _mm256_set_m128(vh, vl) _mm256_insertf128_ps(_mm256_castps128_ps256(vl), (vh), 1)
#endif


#define SIMD_ALIGN_16(code) NV_ALIGN_PREFIX(16) code NV_ALIGN_SUFFIX(16)
#define SIMD_ALIGN_32(code) NV_ALIGN_PREFIX(32) code NV_ALIGN_SUFFIX(32)

inline __m128   add(const __m128& a, const __m128& b)   { return _mm_add_ps(a, b); }
inline __m128   add(float a, const __m128& b)           { return _mm_add_ps(_mm_load1_ps(&a), b); }
inline __m128   add(const __m128& a, float b)           { return _mm_add_ps(a, _mm_load1_ps(&b)); }
inline float    add(float a, float b)                   { return a + b; }

inline __m128   sub(const __m128& a, const __m128& b)   { return _mm_sub_ps(a, b); }
inline __m128   sub(float a, const __m128& b)           { return _mm_sub_ps(_mm_load1_ps(&a), b); }
inline __m128   sub(const __m128& a, float b)           { return _mm_sub_ps(a, _mm_load1_ps(&b)); }
inline float    sub(float a, float b)                   { return a - b; }

inline __m128   mul(const __m128& a, const __m128& b)   { return _mm_mul_ps(a, b); }
inline __m128   mul(float a, const __m128& b)           { return _mm_mul_ps(_mm_load1_ps(&a), b); }
inline __m128   mul(const __m128& a, float b)           { return _mm_mul_ps(a, _mm_load1_ps(&b)); }
inline float    mul(float a, float b)                   { return a * b; }

inline __m128   div(const __m128& a, const __m128& b)   { return _mm_div_ps(a, b); }
inline __m128   div(float a, const __m128& b)           { return _mm_div_ps(_mm_load1_ps(&a), b); }
inline __m128   div(const __m128& a, float b)           { return _mm_div_ps(a, _mm_load1_ps(&b)); }
inline float    div(float a, float b)                   { return a / b; }

inline bool     lt(const __m128& a, const __m128& b)    { return !!_mm_comilt_ss(a, b); }
inline bool     gt(const __m128& a, const __m128& b)    { return !!_mm_comigt_ss(a, b); }
inline bool     le(const __m128& a, const __m128& b)    { return !!_mm_comile_ss(a, b); }
inline bool     ge(const __m128& a, const __m128& b)    { return !!_mm_comige_ss(a, b); }
inline bool     eq(const __m128& a, const __m128& b)    { return !!_mm_comieq_ss(a, b); }
inline bool     ne(const __m128& a, const __m128& b)    { return !!_mm_comineq_ss(a, b); }

inline bool     lt(const float a, const float b)        { return a < b; }
inline bool     gt(const float a, const float b)        { return a > b; }
inline bool     le(const float a, const float b)        { return a <= b; }
inline bool     ge(const float a, const float b)        { return a >= b; }
inline bool     eq(const float a, const float b)        { return a == b; }
inline bool     ne(const float a, const float b)        { return a != b; }

inline float to_float(const __m128& x) { float f; _mm_store_ss(&f, x); return f; }
inline float to_float(float x) { return x; }

inline void from_float(__m128& x, float y) { x = _mm_load1_ps(&y); }
inline void from_float(float& x, float y) { x = y; }

inline void set_zero(__m128& x) { x = _mm_setzero_ps(); }
inline void set_zero(float& x) { x = 0.0f; }

inline void store_float(float* mem, const __m128& f) { _mm_store_ps(mem, f); }
inline void store_float(float* mem, float f) { *mem = f; }

inline void load_float(__m128& f, const float* mem) { f = _mm_load_ps(mem); }
inline void load_float(float& f, const float* mem) { f = *mem; }

inline __m128 prep_cross3(const __m128& v) { return _mm_shuffle_ps(v, v, 0xc9); } // w z y x -> w x z y

inline __m128
cross3(const __m128& v0, const __m128& v1)
{
    __m128 prep0 = prep_cross3(v0);
    __m128 prep1 = prep_cross3(v1);
    __m128 res_shuffled = _mm_sub_ps(_mm_mul_ps(v0, prep1), _mm_mul_ps(prep0, v1));
    return _mm_shuffle_ps(res_shuffled, res_shuffled, 0xc9);
}

inline __m128
cross3_prep0(const __m128& v0, const __m128& prep0, const __m128& v1)
{
    __m128 prep1 = prep_cross3(v1);
    __m128 res_shuffled = _mm_sub_ps(_mm_mul_ps(v0, prep1), _mm_mul_ps(prep0, v1));
    return _mm_shuffle_ps(res_shuffled, res_shuffled, 0xc9);
}

inline __m128
cross3_prep1(const __m128& v0, const __m128& v1, const __m128& prep1)
{
    __m128 prep0 = prep_cross3(v0);
    __m128 res_shuffled = _mm_sub_ps(_mm_mul_ps(v0, prep1), _mm_mul_ps(prep0, v1));
    return _mm_shuffle_ps(res_shuffled, res_shuffled, 0xc9);
}
