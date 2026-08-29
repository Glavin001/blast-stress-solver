// Unit tests for the shared bond-stress equation.
//
// Three implementations of the same math are run on identical inputs:
//
//   host    -- extStressCalcBondStress compiled for the CPU
//   device  -- the same inline compiled by nvcc and run in a kernel
//   exact   -- a double-precision reference written straight from the
//              definitions, used as ground truth for THESE float inputs
//
// Comparing host against device only shows the two agree. It cannot show
// either is right, and for a while both were wrong in the same place. The
// double reference is what makes a case a test rather than a diff.
//
// Also carries the legacy sqrt(|v|^2 - (v.n)^2) form, so the cancellation it
// suffered from is a measured property in a test rather than a claim in a
// commit message.

#include "NvBlastExtStressFormula.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace Nv::Blast;

namespace
{

struct Case
{
    const char* name;
    float linear[3];
    float angular[3];
    float normal[3];
    float area;
    float nodeDist;
};

struct Result
{
    float stressNormal;
    float stressShear;
};

// ── exact reference ────────────────────────────────────────────────────────
// Written from the definitions in double, not derived from the float code.
void exactStress(const Case& c, double& stressNormal, double& stressShear)
{
    const double L[3] = {c.linear[0], c.linear[1], c.linear[2]};
    const double A[3] = {c.angular[0], c.angular[1], c.angular[2]};
    const double n[3] = {c.normal[0], c.normal[1], c.normal[2]};
    const double area = c.area;
    const double dist = c.nodeDist;

    const double ldn = L[0] * n[0] + L[1] * n[1] + L[2] * n[2];
    const double lp[3] = {L[0] - ldn * n[0], L[1] - ldn * n[1], L[2] - ldn * n[2]};
    const double lperp = std::sqrt(lp[0] * lp[0] + lp[1] * lp[1] + lp[2] * lp[2]);

    const double adn = A[0] * n[0] + A[1] * n[1] + A[2] * n[2];
    const double ap[3] = {A[0] - adn * n[0], A[1] - adn * n[1], A[2] - adn * n[2]};
    const double aperp = std::sqrt(ap[0] * ap[0] + ap[1] * ap[1] + ap[2] * ap[2]);

    stressNormal = ldn / area;
    stressShear = lperp / area;

    const double twist = std::fabs(adn) / area;
    const double bend = aperp / area;
    stressShear += twist * 2.0 / dist;
    stressNormal += std::copysign(bend * 2.0 / dist, stressNormal);
}

// The form this equation used to have, for the cancellation comparison.
void legacyStress(const Case& c, double& stressNormal, double& stressShear)
{
    const float* L = c.linear;
    const float* A = c.angular;
    const float* n = c.normal;
    const float area = c.area;

    const float ldn = L[0] * n[0] + L[1] * n[1] + L[2] * n[2];
    const float lmag2 = L[0] * L[0] + L[1] * L[1] + L[2] * L[2];
    float sn = ldn / area;
    float ss = std::sqrt(std::fmax(0.0f, lmag2 - ldn * ldn)) / area;

    const float adn = std::fabs(A[0] * n[0] + A[1] * n[1] + A[2] * n[2]);
    const float amag2 = A[0] * A[0] + A[1] * A[1] + A[2] * A[2];
    const float twist = adn / area;
    const float bend = std::sqrt(std::fmax(0.0f, amag2 - adn * adn)) / area;
    ss += twist * 2.0f / c.nodeDist;
    sn += std::copysign(bend * 2.0f / c.nodeDist, sn);

    stressNormal = sn;
    stressShear = ss;
}

// ── device runner ──────────────────────────────────────────────────────────
__global__ void runOnDevice(const Case* cases, Result* out, int count)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
    {
        return;
    }
    const Case c = cases[i];
    float sn = 0.0f, ss = 0.0f;
    extStressCalcBondStress(
        ExtStressVec3{c.linear[0], c.linear[1], c.linear[2]},
        ExtStressVec3{c.angular[0], c.angular[1], c.angular[2]},
        ExtStressVec3{c.normal[0], c.normal[1], c.normal[2]},
        c.area, c.nodeDist, sn, ss);
    out[i].stressNormal = sn;
    out[i].stressShear = ss;
}

Result runOnHost(const Case& c)
{
    Result r{};
    extStressCalcBondStress(
        ExtStressVec3{c.linear[0], c.linear[1], c.linear[2]},
        ExtStressVec3{c.angular[0], c.angular[1], c.angular[2]},
        ExtStressVec3{c.normal[0], c.normal[1], c.normal[2]},
        c.area, c.nodeDist, r.stressNormal, r.stressShear);
    return r;
}

double relative(double got, double want)
{
    const double d = std::fabs(got - want);
    return std::fabs(want) > 1e-12 ? d / std::fabs(want) : d;
}

int failures = 0;
int checks = 0;

void expectNear(const char* what, const char* caseName, double got, double want, double tol)
{
    ++checks;
    const double rel = relative(got, want);
    if (!(rel <= tol) || std::isnan(got))
    {
        ++failures;
        printf("  FAIL  %-28s %-34s got %-16.9g want %-16.9g rel %.3e > %.1e\n",
               caseName, what, got, want, rel, tol);
    }
}

/// Error judged against a scale, not against the value itself.
void expectScaled(
    const char* what, const char* caseName, double got, double want, double scale, double tol)
{
    ++checks;
    const double err = std::fabs(got - want);
    const double bound = tol * (scale > 0.0 ? scale : 1.0);
    if (!(err <= bound) || std::isnan(got))
    {
        ++failures;
        printf("  FAIL  %-28s %-34s got %-16.9g want %-16.9g err %.3e > %.1e (scale %.3g)\n",
               caseName, what, got, want, err, bound, scale);
    }
}

void expectExact(const char* what, const char* caseName, float got, float want)
{
    ++checks;
    if (memcmp(&got, &want, sizeof(float)) != 0)
    {
        ++failures;
        printf("  FAIL  %-28s %-34s got %.9g want %.9g (not bit-identical)\n",
               caseName, what, got, want);
    }
}

}  // namespace

int main()
{
    const float invSqrt2 = 0.70710678118654752f;
    std::vector<Case> cases = {
        // name                          linear             angular          normal        area  dist
        {"zero-impulse",               {0, 0, 0},        {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"pure-tension",               {0, 0, 10},       {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"pure-compression",           {0, 0, -10},      {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"pure-shear",                 {10, 0, 0},       {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"shear-other-axis",           {0, 10, 0},       {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"45-degree",                  {10, 0, 10},      {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"area-scaling",               {0, 0, 10},       {0, 0, 0},      {0, 0, 1},      4.0f, 1.0f},
        {"tiny-area",                  {0, 0, 10},       {0, 0, 0},      {0, 0, 1},      1e-6f, 1.0f},
        {"huge-impulse",               {0, 0, 1e12f},    {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"tiny-impulse",               {0, 0, 1e-12f},   {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"oblique-normal",             {3, 4, 0},        {0, 0, 0},      {invSqrt2, invSqrt2, 0}, 1.0f, 1.0f},
        {"pure-twist",                 {0, 0, 0},        {0, 0, 5},      {0, 0, 1},      1.0f, 1.0f},
        {"pure-bend",                  {0, 0, 0},        {5, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"bend-sign-follows-normal",   {0, 0, -10},      {5, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"twist-and-bend",             {0, 0, 0},        {3, 0, 4},      {0, 0, 1},      1.0f, 1.0f},
        {"combined",                   {1, 2, 3},        {4, 5, 6},      {0, 0, 1},      2.0f, 3.0f},
        {"degenerate-zero-normal",     {1, 2, 3},        {4, 5, 6},      {0, 0, 0},      1.0f, 1.0f},
        {"long-nodedist",              {0, 0, 10},       {5, 0, 0},      {0, 0, 1},      1.0f, 1e6f},
        // The regime that broke host/device agreement: impulse almost exactly
        // along the normal, so |L|^2 and (L.n)^2 are nearly equal.
        {"near-parallel-1e-3",         {1e-3f, 0, 1e6f}, {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"near-parallel-1e-4",         {1e-4f, 0, 1e6f}, {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"near-parallel-1e-6",         {1e-6f, 0, 1e6f}, {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"near-parallel-huge",         {1.0f, 0, 1e9f},  {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"exactly-parallel-huge",      {0, 0, 1e9f},     {0, 0, 0},      {0, 0, 1},      1.0f, 1.0f},
        {"near-parallel-angular",      {0, 0, 0},        {1e-3f, 0, 1e6f}, {0, 0, 1},    1.0f, 1.0f},
    };

    // Fuzz: deterministic LCG, no Date/random dependency.
    unsigned int seed = 12345u;
    auto nextFloat = [&seed](float lo, float hi) {
        seed = seed * 1664525u + 1013904223u;
        const float t = static_cast<float>((seed >> 8) & 0xFFFFFF) / 16777215.0f;
        return lo + t * (hi - lo);
    };
    std::vector<std::string> fuzzNames;
    for (int i = 0; i < 2000; ++i)
    {
        Case c{};
        const float scale = std::pow(10.0f, nextFloat(-6.0f, 9.0f));
        // Half the fuzz cases are deliberately near-parallel.
        float nx = nextFloat(-1.0f, 1.0f), ny = nextFloat(-1.0f, 1.0f), nz = nextFloat(-1.0f, 1.0f);
        const float nm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (nm < 1e-6f) { nx = 0.0f; ny = 0.0f; nz = 1.0f; }
        else { nx /= nm; ny /= nm; nz /= nm; }
        c.normal[0] = nx; c.normal[1] = ny; c.normal[2] = nz;
        if (i % 2 == 0)
        {
            const float eps = std::pow(10.0f, nextFloat(-8.0f, -1.0f)) * scale;
            c.linear[0] = nx * scale + eps * nz;
            c.linear[1] = ny * scale;
            c.linear[2] = nz * scale - eps * nx;
        }
        else
        {
            c.linear[0] = nextFloat(-1.0f, 1.0f) * scale;
            c.linear[1] = nextFloat(-1.0f, 1.0f) * scale;
            c.linear[2] = nextFloat(-1.0f, 1.0f) * scale;
        }
        c.angular[0] = nextFloat(-1.0f, 1.0f) * scale;
        c.angular[1] = nextFloat(-1.0f, 1.0f) * scale;
        c.angular[2] = nextFloat(-1.0f, 1.0f) * scale;
        c.area = std::pow(10.0f, nextFloat(-4.0f, 2.0f));
        c.nodeDist = std::pow(10.0f, nextFloat(-2.0f, 3.0f));
        fuzzNames.push_back("fuzz-" + std::to_string(i));
        cases.push_back(c);
    }
    for (size_t i = 0; i < fuzzNames.size(); ++i)
    {
        cases[cases.size() - fuzzNames.size() + i].name = fuzzNames[i].c_str();
    }

    const int n = static_cast<int>(cases.size());

    // Run the device copy.
    Case* dCases = nullptr;
    Result* dOut = nullptr;
    if (cudaMalloc(&dCases, sizeof(Case) * n) != cudaSuccess
        || cudaMalloc(&dOut, sizeof(Result) * n) != cudaSuccess)
    {
        printf("cudaMalloc failed\n");
        return 2;
    }
    cudaMemcpy(dCases, cases.data(), sizeof(Case) * n, cudaMemcpyHostToDevice);
    runOnDevice<<<(n + 127) / 128, 128>>>(dCases, dOut, n);
    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        printf("kernel failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 2;
    }
    std::vector<Result> device(n);
    cudaMemcpy(device.data(), dOut, sizeof(Result) * n, cudaMemcpyDeviceToHost);

    printf("== bond stress formula: %d cases (%zu named, %zu fuzz)\n",
           n, cases.size() - fuzzNames.size(), fuzzNames.size());

    // 1. Accuracy against exact math.
    //
    // Tolerance is relative to the DOMINANT stress scale, max(|normal|,
    // |shear|), not to each component separately. That is the physically
    // meaningful bound: a shear of 4.16 sitting beside a normal stress of 1e6
    // is noise, and every consumer of these numbers compares them against
    // material limits that live at the dominant scale. Judging the shear
    // against itself instead demands precision float cannot deliver when the
    // impulse is nearly parallel to the normal -- |v - (v.n)n| is then a small
    // difference of large numbers, and its own relative error is bounded below
    // by eps*|v|/|perp| no matter how the subtraction is arranged.
    printf("\n-- accuracy vs double-precision reference (tol 1e-5 of dominant stress)\n");
    double worstHost = 0.0, worstLegacy = 0.0;
    const char* worstHostCase = "";
    const char* worstLegacyCase = "";
    for (int i = 0; i < n; ++i)
    {
        double exN, exS;
        exactStress(cases[i], exN, exS);
        const Result h = runOnHost(cases[i]);
        const double scale = std::fmax(std::fabs(exN), std::fabs(exS));
        expectScaled("stressNormal vs exact", cases[i].name, h.stressNormal, exN, scale, 1e-5);
        expectScaled("stressShear vs exact", cases[i].name, h.stressShear, exS, scale, 1e-5);

        const double rh = std::fmax(relative(h.stressNormal, exN), relative(h.stressShear, exS));
        if (rh > worstHost) { worstHost = rh; worstHostCase = cases[i].name; }

        double lgN, lgS;
        legacyStress(cases[i], lgN, lgS);
        const double rl = std::fmax(relative(lgN, exN), relative(lgS, exS));
        if (rl > worstLegacy) { worstLegacy = rl; worstLegacyCase = cases[i].name; }
    }
    printf("   worst relative error  current %.3e (%s)\n", worstHost, worstHostCase);
    printf("   worst relative error  legacy  %.3e (%s)   <- the form that was replaced\n",
           worstLegacy, worstLegacyCase);

    // 2. Host and device must agree bit for bit.
    printf("\n-- host vs device, bit-identical\n");
    int bitDiffs = 0;
    double worstHD = 0.0;
    const char* worstHDCase = "";
    for (int i = 0; i < n; ++i)
    {
        const Result h = runOnHost(cases[i]);
        const Result d = device[i];
        if (memcmp(&h, &d, sizeof(Result)) != 0)
        {
            ++bitDiffs;
            const double r = std::fmax(
                relative(d.stressNormal, h.stressNormal), relative(d.stressShear, h.stressShear));
            if (r > worstHD) { worstHD = r; worstHDCase = cases[i].name; }
        }
    }
    printf("   %d of %d cases differ; worst relative %.3e (%s)\n",
           bitDiffs, n, worstHD, worstHDCase);
    // A hard requirement, not an observation. Both sides route every
    // multiply-add through fmaf, which is IEEE-defined and correctly rounded,
    // so there is no compiler freedom left for them to disagree about. If this
    // ever fails again, someone has reintroduced a contractable a*b+c.
    ++checks;
    if (bitDiffs != 0)
    {
        ++failures;
        printf("  FAIL  host and device are not bit-identical (%d cases)\n", bitDiffs);
    }

    // 3. Analytic identities that must hold regardless of implementation.
    printf("\n-- analytic identities\n");
    {
        // Pure normal load carries no shear.
        Case c{"identity/pure-normal", {0, 0, 7}, {0, 0, 0}, {0, 0, 1}, 2.0f, 1.0f};
        const Result r = runOnHost(c);
        expectNear("shear is zero", c.name, r.stressShear, 0.0, 1e-12);
        expectNear("normal is F/A", c.name, r.stressNormal, 3.5, 1e-6);
    }
    {
        // Pure shear load carries no normal stress.
        Case c{"identity/pure-shear", {7, 0, 0}, {0, 0, 0}, {0, 0, 1}, 2.0f, 1.0f};
        const Result r = runOnHost(c);
        expectNear("normal is zero", c.name, r.stressNormal, 0.0, 1e-12);
        expectNear("shear is F/A", c.name, r.stressShear, 3.5, 1e-6);
    }
    {
        // Doubling the impulse doubles both stresses.
        Case a{"identity/linearity", {1, 2, 3}, {4, 5, 6}, {0, 0, 1}, 2.0f, 3.0f};
        Case b = a;
        for (int k = 0; k < 3; ++k) { b.linear[k] *= 2.0f; b.angular[k] *= 2.0f; }
        const Result ra = runOnHost(a), rb = runOnHost(b);
        expectNear("normal doubles", a.name, rb.stressNormal, 2.0 * ra.stressNormal, 1e-6);
        expectNear("shear doubles", a.name, rb.stressShear, 2.0 * ra.stressShear, 1e-6);
    }
    {
        // Halving the area doubles both stresses.
        Case a{"identity/area-inverse", {1, 2, 3}, {4, 5, 6}, {0, 0, 1}, 2.0f, 3.0f};
        Case b = a; b.area = 1.0f;
        const Result ra = runOnHost(a), rb = runOnHost(b);
        expectNear("normal doubles", a.name, rb.stressNormal, 2.0 * ra.stressNormal, 1e-6);
        expectNear("shear doubles", a.name, rb.stressShear, 2.0 * ra.stressShear, 1e-6);
    }
    {
        // Rotating load and normal together changes nothing.
        Case a{"identity/rotation", {1, 2, 3}, {4, 5, 6}, {0, 0, 1}, 2.0f, 3.0f};
        // Rotate 90 degrees about x: (x,y,z) -> (x,-z,y)
        Case b = a;
        b.linear[0] = a.linear[0]; b.linear[1] = -a.linear[2]; b.linear[2] = a.linear[1];
        b.angular[0] = a.angular[0]; b.angular[1] = -a.angular[2]; b.angular[2] = a.angular[1];
        b.normal[0] = a.normal[0]; b.normal[1] = -a.normal[2]; b.normal[2] = a.normal[1];
        const Result ra = runOnHost(a), rb = runOnHost(b);
        expectNear("normal invariant", a.name, rb.stressNormal, ra.stressNormal, 1e-5);
        expectNear("shear invariant", a.name, rb.stressShear, ra.stressShear, 1e-5);
    }
    {
        // Flipping the normal flips the sign of the normal stress and leaves
        // shear alone.
        Case a{"identity/normal-flip", {1, 2, 3}, {0, 0, 0}, {0, 0, 1}, 2.0f, 3.0f};
        Case b = a; b.normal[2] = -1.0f;
        const Result ra = runOnHost(a), rb = runOnHost(b);
        expectNear("normal negates", a.name, rb.stressNormal, -ra.stressNormal, 1e-6);
        expectNear("shear unchanged", a.name, rb.stressShear, ra.stressShear, 1e-6);
    }
    {
        // Shear is never negative, and nothing is ever NaN.
        int bad = 0;
        for (int i = 0; i < n; ++i)
        {
            const Result r = runOnHost(cases[i]);
            if (r.stressShear < 0.0f || std::isnan(r.stressShear) || std::isnan(r.stressNormal))
            {
                ++bad;
            }
        }
        ++checks;
        if (bad != 0)
        {
            ++failures;
            printf("  FAIL  %-28s %d cases with negative or NaN stress\n", "identity/well-formed", bad);
        }
    }

    printf("\n%d checks, %d failures\n", checks, failures);
    printf(failures == 0 ? "PASSED\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
