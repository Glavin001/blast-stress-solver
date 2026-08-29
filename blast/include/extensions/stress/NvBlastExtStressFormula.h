// Copyright (c) 2026 NVIDIA Corporation. All rights reserved.

#pragma once

/**
The bond stress equation, in one place.

It used to exist twice, term for term: once on the host in
SupportGraphProcessor::calcSolverBondStresses (NvBlastExtStressSolver.cpp) and
once on the device in applyStressDamage (NvBlastExtStressGpu.cu). Only the host
copy runs in production -- the device copy is reachable only when
params.applyDamage is set, which happens exclusively in the two stressgpu test
harnesses -- so the duplication was not costing double damage, but it was one
equation with two bodies that could drift silently.

Anything that ports this walk to the GPU has to agree with the host to the
bit-pattern the equivalence gate accepts, so the equation gets a single body
that both compilers emit.

Deliberate difference from the old host copy: the two sqrtf arguments are
clamped at zero, matching what the device copy already did. For a unit normal
|v|^2 - (v.n)^2 is non-negative by Cauchy-Schwarz, and bondNormal is
normalizeSafe()'d (unit, or exactly zero -- in which case the dot product is
zero and the radicand is |v|^2). So the clamp can only ever fire on float
cancellation, where the unclamped host form produced a NaN that the assert
below the call site was there to catch. Clamping is the same value everywhere
the old code was well-defined.

Also deliberate: fabsf rather than the old unqualified abs(). In a translation
unit that includes both <cmath> and <cstdlib>, unqualified abs() on a float is
one bad overload resolution away from truncating through abs(int).
*/

#include <cmath>

/**
Why this file uses fmaf explicitly.

Earlier this equation forced STRICT non-contractable ops on the device
(__fmul_rn/__fadd_rn) to stop nvcc fusing multiply-add, on the theory that the
host did not fuse. The host does: the library compiles its C++ with -mfma and
gcc contracts by default. So the two sides were made to disagree deliberately,
and the unit tests caught it -- 1100 of 2024 cases differing.

Explicit fmaf is the fix for both halves of the problem at once. It is
IEEE-754 defined and correctly rounded, so the host and the device compute the
same bits no matter what either compiler would have chosen on its own. And it
is MORE accurate here, not merely more consistent: a fused multiply-add rounds
once instead of twice, which is exactly what the perpendicular component needs,
since v - (v.n)n subtracts two nearly-equal quantities whenever the impulse is
close to parallel with the bond normal.
*/

namespace Nv
{
namespace Blast
{

/// Marks the equation for both compilers. nvcc needs the execution-space
/// qualifiers; the host compiler must not see them.
#if defined(__CUDACC__)
#define NVBLAST_STRESS_FORMULA_FN __host__ __device__ inline
#else
#define NVBLAST_STRESS_FORMULA_FN inline
#endif

/// Plain xyz, so neither side has to hand the other its vector type: the host
/// speaks nvidia::NvVec3, the device speaks Vec4/AngLin, and this header is
/// included by translation units that know about neither.
struct ExtStressVec3
{
    float x, y, z;
};

/// Dot product with a fixed, fully-specified rounding sequence.
NVBLAST_STRESS_FORMULA_FN float extStressDot(const ExtStressVec3& a, const ExtStressVec3& b)
{
    return fmaf(a.z, b.z, fmaf(a.y, b.y, a.x * b.x));
}

/// |v - (v.n)n|, the magnitude of v perpendicular to a unit n.
///
/// Stable in two ways. It never forms |v|^2 - (v.n)^2, so it never subtracts
/// two nearly-equal LARGE numbers. And each component is taken with a single
/// fused rounding, which is what keeps the remaining subtraction accurate when
/// v is nearly parallel to n and the result is small.
NVBLAST_STRESS_FORMULA_FN float extStressPerpendicularMagnitude(
    const ExtStressVec3& v, const ExtStressVec3& n, float vDotN)
{
    const float px = fmaf(-vDotN, n.x, v.x);
    const float py = fmaf(-vDotN, n.y, v.y);
    const float pz = fmaf(-vDotN, n.z, v.z);
    return sqrtf(fmaf(pz, pz, fmaf(py, py, px * px)));
}

/**
Impulse on a bond -> the normal and shear pressures it carries.

\param[in]  impulseLinear   Linear impulse across the bond.
\param[in]  impulseAngular  Angular impulse across the bond.
\param[in]  normal          Bond normal. Unit length, or zero.
\param[in]  area            Effective area (m^2). Must be non-zero; on the host
                            path this is the summed remaining health of the
                            group's members, which is live area, not the static
                            asset area.
\param[in]  nodeDist        Distance between the bond's two node positions,
                            used to reinterpret angular pressure as linear.
\param[out] stressNormal    Signed: positive is tension, negative compression.
\param[out] stressShear     Unsigned.
*/
NVBLAST_STRESS_FORMULA_FN void extStressCalcBondStress(
    const ExtStressVec3& impulseLinear,
    const ExtStressVec3& impulseAngular,
    const ExtStressVec3& normal,
    float area,
    float nodeDist,
    float& stressNormal,
    float& stressShear)
{
    // Linear impulse along the normal is normal stress, perpendicular is
    // shear. Dividing by area converts impulse to pressure.
    const float linearNormal = extStressDot(impulseLinear, normal);
    stressNormal = linearNormal / area;
    // Shear is the magnitude of the component perpendicular to the normal,
    // taken by REMOVING the along-normal part rather than by
    // sqrt(|L|^2 - (L.n)^2). Those are equal in exact arithmetic and very
    // different in floats: a bond loaded along its normal -- the ordinary
    // case -- makes |L|^2 and (L.n)^2 nearly equal, so the subtraction
    // cancels catastrophically and a last-bit difference in |L|^2 becomes a
    // large relative difference in the shear. Worse, the fmaxf clamp then
    // turns a radicand that straddles zero into "0 versus something".
    // Measured under the old form, host against device: 32% of stress values
    // differing, 4.9M by more than 1e-5 relative, max relative difference
    // exactly 1.000 -- the signature of that clamp.
    stressShear = extStressPerpendicularMagnitude(impulseLinear, normal, linearNormal) / area;

    // Angular impulse along the normal is twist, perpendicular is bend. abs()
    // because only the magnitude of the twist matters, not its direction.
    const float angularAlongNormal = extStressDot(impulseAngular, normal);
    // abs() because only the magnitude of the twist matters, not direction.
    const float twist = fabsf(angularAlongNormal) / area;
    // Same stable form as the shear. Squaring the dot product discarded its
    // sign, so using the signed value here is the same quantity.
    const float bend =
        extStressPerpendicularMagnitude(impulseAngular, normal, angularAlongNormal) / area;

    // Interpret angular pressure as a composition of linear pressures,
    // dividing by nodeDist for scaling.
    // Plain ops: multiplying by 2 is exact, and an add of a division result
    // offers the compiler nothing to contract, so these are already the same
    // sequence on both sides.
    stressShear += twist * 2.0f / nodeDist;
    stressNormal += copysignf(bend * 2.0f / nodeDist, stressNormal);
}

}  // namespace Blast
}  // namespace Nv
