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
Strict, non-contractable float ops on the device.

The host evaluates this equation with plain multiplies and adds. nvcc, with
its default --fmad=true, contracts every mul+add into an FMA, which is more
accurate but NOT the same value in the last bit. That difference is invisible
almost all the time and then very visible in one specific regime: a freshly
loaded city has enormous numbers of bonds sitting at essentially identical
stress, so a 1-ulp disagreement straddles an elastic limit for a whole batch
of them at once. Measured, before these were introduced: mismatches confined
to ticks 8-120 -- the settling window -- and exactly zero afterwards.

Forcing the individual ops keeps the device answer bit-identical to the host's
instead, which is what lets the dual-run audit demand equality rather than a
tolerance. It is deliberately local to this equation: building the whole .cu
with -fmad=false would also change the CG solver kernels that ship today.
*/
#if defined(__CUDA_ARCH__)
#define NVBLAST_SFMUL(a, b) __fmul_rn((a), (b))
#define NVBLAST_SFADD(a, b) __fadd_rn((a), (b))
#define NVBLAST_SFSUB(a, b) __fsub_rn((a), (b))
#else
#define NVBLAST_SFMUL(a, b) ((a) * (b))
#define NVBLAST_SFADD(a, b) ((a) + (b))
#define NVBLAST_SFSUB(a, b) ((a) - (b))
#endif

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

/// |v - (v.n)n|, the magnitude of v perpendicular to a unit n.
///
/// The numerically stable way to get the perpendicular magnitude: it never
/// forms |v|^2 - (v.n)^2, so it never subtracts two nearly-equal large
/// numbers when v is close to parallel with n.
NVBLAST_STRESS_FORMULA_FN float extStressPerpendicularMagnitude(
    const ExtStressVec3& v, const ExtStressVec3& n, float vDotN)
{
    const float px = NVBLAST_SFSUB(v.x, NVBLAST_SFMUL(vDotN, n.x));
    const float py = NVBLAST_SFSUB(v.y, NVBLAST_SFMUL(vDotN, n.y));
    const float pz = NVBLAST_SFSUB(v.z, NVBLAST_SFMUL(vDotN, n.z));
    return sqrtf(NVBLAST_SFADD(
        NVBLAST_SFADD(NVBLAST_SFMUL(px, px), NVBLAST_SFMUL(py, py)),
        NVBLAST_SFMUL(pz, pz)));
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
    const float linearNormal = NVBLAST_SFADD(
        NVBLAST_SFADD(
            NVBLAST_SFMUL(impulseLinear.x, normal.x),
            NVBLAST_SFMUL(impulseLinear.y, normal.y)),
        NVBLAST_SFMUL(impulseLinear.z, normal.z));
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
    const float angularAlongNormal = NVBLAST_SFADD(
        NVBLAST_SFADD(
            NVBLAST_SFMUL(impulseAngular.x, normal.x),
            NVBLAST_SFMUL(impulseAngular.y, normal.y)),
        NVBLAST_SFMUL(impulseAngular.z, normal.z));
    // abs() because only the magnitude of the twist matters, not direction.
    const float twist = fabsf(angularAlongNormal) / area;
    // Same stable form as the shear. Squaring the dot product discarded its
    // sign, so using the signed value here is the same quantity.
    const float bend =
        extStressPerpendicularMagnitude(impulseAngular, normal, angularAlongNormal) / area;

    // Interpret angular pressure as a composition of linear pressures,
    // dividing by nodeDist for scaling.
    stressShear = NVBLAST_SFADD(stressShear, NVBLAST_SFMUL(twist, 2.0f) / nodeDist);
    stressNormal = NVBLAST_SFADD(
        stressNormal, copysignf(NVBLAST_SFMUL(bend, 2.0f) / nodeDist, stressNormal));
}

}  // namespace Blast
}  // namespace Nv
