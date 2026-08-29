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
    const float linearNormal =
        impulseLinear.x * normal.x + impulseLinear.y * normal.y + impulseLinear.z * normal.z;
    const float linearMagnitudeSquared =
        impulseLinear.x * impulseLinear.x
        + impulseLinear.y * impulseLinear.y
        + impulseLinear.z * impulseLinear.z;
    stressNormal = linearNormal / area;
    stressShear =
        sqrtf(fmaxf(0.0f, linearMagnitudeSquared - linearNormal * linearNormal)) / area;

    // Angular impulse along the normal is twist, perpendicular is bend. abs()
    // because only the magnitude of the twist matters, not its direction.
    const float angularNormal = fabsf(
        impulseAngular.x * normal.x + impulseAngular.y * normal.y + impulseAngular.z * normal.z);
    const float angularMagnitudeSquared =
        impulseAngular.x * impulseAngular.x
        + impulseAngular.y * impulseAngular.y
        + impulseAngular.z * impulseAngular.z;
    const float twist = angularNormal / area;
    const float bend =
        sqrtf(fmaxf(0.0f, angularMagnitudeSquared - angularNormal * angularNormal)) / area;

    // Interpret angular pressure as a composition of linear pressures,
    // dividing by nodeDist for scaling.
    stressShear += twist * 2.0f / nodeDist;
    stressNormal += copysignf(bend * 2.0f / nodeDist, stressNormal);
}

}  // namespace Blast
}  // namespace Nv
