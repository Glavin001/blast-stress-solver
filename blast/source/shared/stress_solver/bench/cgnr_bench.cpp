// CGNR stress-solver micro/whole-solve benchmark + bottleneck attribution.
//
// PURPOSE
//   The web build ships the *scalar* CGNR path (js_stress_example/scripts/build.js
//   compiles stress.cpp with -DSTRESS_SOLVER_FORCE_SCALAR -DSTRESS_SOLVER_NO_SIMD,
//   relying only on -O3 -msimd128 auto-vectorization). In the mini-city recording
//   "CGNR solve (WASM)" is ~62% of the stress solver (4.32 ms/frame). This harness
//   compiles the *exact same* scalar kernels natively so we can answer, with
//   evidence and no WASM toolchain:
//
//     1. Within one CGNR iteration, where does the time go? (lmul vs rmul vs the
//        vector/reduction ops; and inside the mat-vecs, coupling vs inertia.)
//     2. How much of that is memory traffic (streaming passes) vs FP work?
//     3. Does a candidate optimization keep the output *bit-identical*?
//
//   It deliberately does NOT use SIMD — that lever is owned elsewhere. The point
//   here is the algorithmic / memory-traffic / O(n) structure of the solve, which
//   is independent of (and stacks with) any SIMD work.
//
// BUILD/RUN
//   See run.sh in this directory (clang++/g++, scalar config matching the WASM build).
//
// This file has no NvBlast dependencies beyond the stress-solver headers + std.

#include "../stress.h"           // StressProcessor (real prepare()/solve(), incl. island-aware)
#include "../math/cgnr.h"
#include "../bond.h"
#include "../coupling.h"
#include "../inertia.h"
#include "../anglin6.h"
#include "../solver_types.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

// ── The scalar CGNR the WASM build uses (Scalar = float, SISD ops). ──
typedef CGNR<AngLin6, AngLin6Ops<float>, BondMatrixS, BondMatrixOpsS<float>, float, AngLin6ErrorSq> CGNR_Scalar;

// ──────────────────────────────────────────────────────────────────────────────
// Timing utilities
// ──────────────────────────────────────────────────────────────────────────────
using Clock = std::chrono::steady_clock;
static inline double now_ns() {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}

// Volatile sink so the optimizer can't delete the work we are timing.
static volatile float g_sink = 0.0f;
static void consume(const AngLin6* v, uint32_t n) {
    float s = 0.0f;
    for (uint32_t i = 0; i < n; ++i) s += v[i].ang.x + v[i].lin.z;
    g_sink += s;
}

// Auto-calibrated timer: returns nanoseconds per single invocation of f().
template <class F>
static double bench_ns(F&& f, double target_ms = 60.0, int min_reps = 3) {
    // warmup
    f(); f();
    // calibrate
    double t0 = now_ns();
    f();
    double one = now_ns() - t0;
    int reps = (int)std::max((double)min_reps, (target_ms * 1e6) / std::max(one, 1.0));
    reps = std::min(reps, 2000000);
    // measure best-of-3 windows to dodge scheduler noise
    double best = 1e300;
    for (int w = 0; w < 3; ++w) {
        double a = now_ns();
        for (int r = 0; r < reps; ++r) f();
        double dt = now_ns() - a;
        best = std::min(best, dt / reps);
    }
    return best;
}

// ──────────────────────────────────────────────────────────────────────────────
// Problem generation: 3D lattice "buildings" (single connected island each).
//   ~2.6-3.0 bonds/node interior, matching the mini-city (19356 bonds / 7264 nodes
//   = 2.66). Bonds connect 6-neighbors; nodes numbered in natural (x,y,z) order so
//   the bond/node access pattern mirrors how a building is assembled floor by floor.
// ──────────────────────────────────────────────────────────────────────────────
struct Problem {
    std::vector<SolverNodeS> nodes;
    std::vector<SolverBond>  bonds;
    std::vector<AngLin6>     velocities;
    std::string              name;
    uint32_t M() const { return (uint32_t)nodes.size(); }
    uint32_t N() const { return (uint32_t)bonds.size(); }
};

static void addLattice(Problem& p, int W, int D, int H, float spacing,
                       float ox, float oy, float oz, std::mt19937& rng) {
    std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);
    std::uniform_real_distribution<float> vel(-0.5f, 0.5f);
    const uint32_t base = (uint32_t)p.nodes.size();
    auto idx = [&](int x, int y, int z) { return base + (uint32_t)((z * D + y) * W + x); };

    for (int z = 0; z < H; ++z)
        for (int y = 0; y < D; ++y)
            for (int x = 0; x < W; ++x) {
                SolverNodeS n;
                n.CoM = { ox + x * spacing + jitter(rng),
                          oy + z * spacing + jitter(rng),
                          oz + y * spacing + jitter(rng) };
                // bottom layer anchored to the world (static, mass 0 == infinite).
                n.mass    = (z == 0) ? 0.0f : (800.0f + 200.0f * (float)((x + y + z) % 5));
                n.inertia = (z == 0) ? 0.0f : (n.mass * 0.18f);
                p.nodes.push_back(n);
                AngLin6 v{};
                // a small downward + jitter velocity field, like settling debris/gravity load.
                // Static (anchored, mass/inertia 0) nodes carry no velocity load.
                if (z != 0) {
                    v.lin = { vel(rng), -1.0f + vel(rng), vel(rng) };
                    v.ang = { 0.1f * vel(rng), 0.1f * vel(rng), 0.1f * vel(rng) };
                }
                p.velocities.push_back(v);
            }

    auto bond = [&](uint32_t a, uint32_t b, float bx, float by, float bz) {
        SolverBond s; s.nodes[0] = a; s.nodes[1] = b; s.centroid = { bx, by, bz };
        p.bonds.push_back(s);
    };
    for (int z = 0; z < H; ++z)
        for (int y = 0; y < D; ++y)
            for (int x = 0; x < W; ++x) {
                const NvcVec3 c = p.nodes[idx(x, y, z)].CoM;
                if (x + 1 < W) { NvcVec3 c2 = p.nodes[idx(x+1,y,z)].CoM; bond(idx(x,y,z), idx(x+1,y,z), 0.5f*(c.x+c2.x),0.5f*(c.y+c2.y),0.5f*(c.z+c2.z)); }
                if (y + 1 < D) { NvcVec3 c2 = p.nodes[idx(x,y+1,z)].CoM; bond(idx(x,y,z), idx(x,y+1,z), 0.5f*(c.x+c2.x),0.5f*(c.y+c2.y),0.5f*(c.z+c2.z)); }
                if (z + 1 < H) { NvcVec3 c2 = p.nodes[idx(x,y,z+1)].CoM; bond(idx(x,y,z), idx(x,y,z+1), 0.5f*(c.x+c2.x),0.5f*(c.y+c2.y),0.5f*(c.z+c2.z)); }
            }
}

static Problem makeBuilding(int W, int D, int H, const char* name) {
    Problem p; p.name = name;
    std::mt19937 rng(1234567u);
    addLattice(p, W, D, H, 1.0f, 0, 0, 0, rng);
    return p;
}

// A city: many disjoint buildings (=> many islands), sized to ~match the recording.
static Problem makeCity(int buildings, int W, int D, int Hmin, int Hmax, const char* name) {
    Problem p; p.name = name;
    std::mt19937 rng(7654321u);
    std::uniform_int_distribution<int> hd(Hmin, Hmax);
    int cols = (int)std::ceil(std::sqrt((double)buildings));
    for (int b = 0; b < buildings; ++b) {
        int H = hd(rng);
        float ox = (b % cols) * (W + 6) * 1.0f;
        float oz = (b / cols) * (D + 6) * 1.0f;
        addLattice(p, W, D, H, 1.0f, ox, 0, oz, rng);
    }
    return p;
}

// ──────────────────────────────────────────────────────────────────────────────
// Build the solver-internal representation (couplings + recip-sqrt-inertia) the way
// stress.cpp::prepare does, so the isolated kernel microbenchmarks see a realistic
// matrix. (Scaling constants don't affect kernel *cost*, only convergence, so we use
// representative unit-ish scaling here; end-to-end convergence is measured via the
// real StressProcessor path below.)
// ──────────────────────────────────────────────────────────────────────────────
struct Matrix {
    std::vector<Coupling> C;
    std::vector<InertiaS> sqrtIinv;
    std::vector<AngLin6>  scratch;
    BondMatrixS B;
    void build(const Problem& p) {
        const uint32_t M = p.M(), N = p.N();
        C.resize(N); sqrtIinv.resize(M); scratch.resize(M);
        for (uint32_t i = 0; i < M; ++i) {
            const SolverNodeS& n = p.nodes[i];
            sqrtIinv[i].I = n.inertia > 0 ? std::sqrt(1.0f / n.inertia) : 0.0f;
            sqrtIinv[i].m = n.mass    > 0 ? std::sqrt(1.0f / n.mass)    : 0.0f;
        }
        for (uint32_t j = 0; j < N; ++j) {
            const SolverBond& b = p.bonds[j];
            const uint32_t b0 = b.nodes[0], b1 = b.nodes[1];
            Coupling& c = C[j];
            c.node0 = b0; c.node1 = b1;
            c.offset0 = b.centroid - p.nodes[b0].CoM;
            c.offset1 = b.centroid - p.nodes[b1].CoM;
        }
        B.set(C.data(), sqrtIinv.data(), scratch.data(), M, N);
    }
};

// ── Local "fused-reduction" kernels, kept in the harness (NOT in production) purely to
//    A/B the idea of folding |z|²/|s|² into the producing mat-vec. They are bit-identical
//    to (mat-vec + separate reduction); the benchmark shows the fusion does NOT pay off in
//    the auto-vectorized scalar build (the reduction vectorizes better as its own loop and
//    the working set is cache-resident), which is why production keeps the two passes. ──
static float local_rmul_lensq(AngLin6* y, const Matrix& mat, const AngLin6* x, uint32_t M, uint32_t N) {
    CouplingMatrixOps<AngLin6, float>().rmul(y, mat.C.data(), x, M, N);   // y = C*x
    float nrm = 0.0f;
    for (uint32_t i = 0; i < M; ++i) {
        const InertiaS& I = mat.sqrtIinv[i];
        y[i].ang = I.I * y[i].ang; y[i].lin = I.m * y[i].lin;             // fused I^-½ + |y|²
        nrm += (y[i].ang | y[i].ang) + (y[i].lin | y[i].lin);
    }
    return nrm;
}
static float local_lmul_errsq(AngLin6* y, const AngLin6* x, Matrix& mat, uint32_t M, uint32_t N, AngLin6ErrorSq& err) {
    AngLin6* s = mat.scratch.data();
    InertiaMatrixOps<float>().mul(s, mat.sqrtIinv.data(), x, M);          // s = I^-½ x
    err.ang = err.lin = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {                                    // fused C^T + split error
        const Coupling& c = mat.C[j];
        const AngLin6& x0 = s[c.node0]; const AngLin6& x1 = s[c.node1];
        AngLin6& yj = y[j];
        yj.ang = x0.ang - x1.ang;
        yj.lin = x0.lin - x1.lin + (c.offset0 ^ x0.ang) - (c.offset1 ^ x1.ang);
        err.ang += yj.ang | yj.ang; err.lin += yj.lin | yj.lin;
    }
    return err.ang + err.lin;
}

// ──────────────────────────────────────────────────────────────────────────────
// 1) ISOLATED KERNEL MICROBENCHMARKS — clean ns/element throughput, no timer in loop.
// ──────────────────────────────────────────────────────────────────────────────
static void microbench(const Problem& p) {
    const uint32_t M = p.M(), N = p.N();
    Matrix mat; mat.build(p);

    std::vector<AngLin6> xN(N), yN(N), zN(N), pN(N);
    std::vector<AngLin6> rM(M), sM(M), tM(M);
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto fill = [&](std::vector<AngLin6>& v){ for (auto& e : v){ e.ang={d(rng),d(rng),d(rng)}; e.lin={d(rng),d(rng),d(rng)}; } };
    fill(xN); fill(yN); fill(zN); fill(pN); fill(rM); fill(sM); fill(tM);

    CouplingMatrixOps<AngLin6, float> cops;
    InertiaMatrixOps<float>           iops;
    BondMatrixOpsS<float>             bops;
    AngLin6Ops<float>                 vops;
    AngLin6ErrorSq err{};

    struct Row { const char* name; double ns; double elems; const char* unit; };
    std::vector<Row> rows;

    // Composed mat-vecs (what CGNR actually calls):
    rows.push_back({"BondMatrixOps::lmul  (z = Bᵀr  = Cᵀ·I^-½·r)", bench_ns([&]{ bops.lmul(zN.data(), rM.data(), mat.B, M, N); consume(zN.data(),1);} ), (double)N, "bond"});
    rows.push_back({"BondMatrixOps::rmul  (s = B·p  = I^-½·C·p)",  bench_ns([&]{ bops.rmul(sM.data(), mat.B, pN.data(), M, N); consume(sM.data(),1);} ), (double)N, "bond"});

    // Fusion A/B at the kernel level: separate (mat-vec + reduction) vs fused single pass.
    // (Local fused kernels — see note above. Demonstrates fusion is NOT a win here.)
    AngLin6ErrorSq fe{};
    rows.push_back({"  lmul + calculate_error  (separate, 2 passes)", bench_ns([&]{ bops.lmul(zN.data(), rM.data(), mat.B, M, N); g_sink += vops.calculate_error(fe, zN.data(), N); } ), (double)N, "bond"});
    rows.push_back({"  lmul_errsq              (fused, 1 pass)",      bench_ns([&]{ g_sink += local_lmul_errsq(zN.data(), rM.data(), mat, M, N, fe); } ), (double)N, "bond"});
    rows.push_back({"  rmul + length_sq        (separate, 2 passes)", bench_ns([&]{ bops.rmul(sM.data(), mat.B, pN.data(), M, N); g_sink += vops.length_sq(sM.data(), M); } ), (double)N, "bond"});
    rows.push_back({"  rmul_lensq              (fused, 1 pass)",      bench_ns([&]{ g_sink += local_rmul_lensq(sM.data(), mat, pN.data(), M, N); } ), (double)N, "bond"});

    // Their internal halves (coupling vs inertia diagonal):
    rows.push_back({"  CouplingMatrixOps::rmul (C·x, scatter+2×cross+memset)", bench_ns([&]{ cops.rmul(sM.data(), mat.C.data(), pN.data(), M, N); consume(sM.data(),1);} ), (double)N, "bond"});
    rows.push_back({"  CouplingMatrixOps::lmul (Cᵀ·x, gather+2×cross)",        bench_ns([&]{ cops.lmul(zN.data(), rM.data(), mat.C.data(), M, N); consume(zN.data(),1);} ), (double)N, "bond"});
    rows.push_back({"  InertiaMatrixOps::mul   (I^-½ diag, M nodes)",          bench_ns([&]{ iops.mul(tM.data(), mat.sqrtIinv.data(), rM.data(), M); consume(tM.data(),1);} ), (double)M, "node"});

    // Vector / reduction ops (per CGNR iteration):
    rows.push_back({"AngLin6Ops::calculate_error (reduce |z|², N)", bench_ns([&]{ g_sink += vops.calculate_error(err, zN.data(), N);} ), (double)N, "bond"});
    rows.push_back({"AngLin6Ops::length_sq       (reduce |s|², M)", bench_ns([&]{ g_sink += vops.length_sq(sM.data(), M);} ), (double)M, "node"});
    rows.push_back({"AngLin6Ops::vmadd  (x += μ·p, N)",  bench_ns([&]{ vops.vmadd(xN.data(), 0.5f, pN.data(), xN.data(), N); consume(xN.data(),1);} ), (double)N, "bond"});
    rows.push_back({"AngLin6Ops::vnmadd (r -= μ·s, M)",  bench_ns([&]{ vops.vnmadd(rM.data(), 0.5f, sM.data(), rM.data(), M); consume(rM.data(),1);} ), (double)M, "node"});

    printf("\n── Isolated kernel microbenchmarks · %s (M=%u nodes, N=%u bonds, %.2f bonds/node) ──\n",
           p.name.c_str(), M, N, (double)N / M);
    printf("  %-52s %12s %12s\n", "kernel", "µs/call", "ns/elem");
    for (auto& r : rows)
        printf("  %-52s %12.3f %12.3f\n", r.name, r.ns / 1000.0, r.ns / r.elems);
}

// ──────────────────────────────────────────────────────────────────────────────
// 2) INSTRUMENTED SOLVE — wrap the real ops to attribute time per kernel *as the
//    real CGNR solve calls them* (real iteration count, real convergence). Timer
//    overhead is a few % but the *shares* are faithful.
// ──────────────────────────────────────────────────────────────────────────────
struct Acc { double ns = 0; uint64_t calls = 0; };
static Acc gLmul, gRmul, gErr, gLenSq, gVmadd, gVnmadd, gOther;
static void resetAcc() { gLmul = gRmul = gErr = gLenSq = gVmadd = gVnmadd = gOther = Acc{}; }

struct InstrElemOps {
    AngLin6Ops<float> base;
    void  add (AngLin6& r, const AngLin6& x, const AngLin6& y) { base.add(r,x,y); }
    void  sub (AngLin6& r, const AngLin6& x, const AngLin6& y) { base.sub(r,x,y); }
    void  madd(AngLin6& r, float c, const AngLin6& x, const AngLin6& y) { base.madd(r,c,x,y); }
    void  nmadd(AngLin6& r, float c, const AngLin6& x, const AngLin6& y){ base.nmadd(r,c,x,y); }
    void  vadd (AngLin6* r, const AngLin6* x, const AngLin6* y, uint32_t N){ double t=now_ns(); base.vadd(r,x,y,N); gOther.ns+=now_ns()-t; gOther.calls++; }
    void  vsub (AngLin6* r, const AngLin6* x, const AngLin6* y, uint32_t N){ double t=now_ns(); base.vsub(r,x,y,N); gOther.ns+=now_ns()-t; gOther.calls++; }
    void  vmadd(AngLin6* r, float c, const AngLin6* x, const AngLin6* y, uint32_t N){ double t=now_ns(); base.vmadd(r,c,x,y,N); gVmadd.ns+=now_ns()-t; gVmadd.calls++; }
    void  vnmadd(AngLin6* r, float c, const AngLin6* x, const AngLin6* y, uint32_t N){ double t=now_ns(); base.vnmadd(r,c,x,y,N); gVnmadd.ns+=now_ns()-t; gVnmadd.calls++; }
    float dot(const AngLin6* v, const AngLin6* w, uint32_t N){ double t=now_ns(); float r=base.dot(v,w,N); gOther.ns+=now_ns()-t; gOther.calls++; return r; }
    float length_sq(const AngLin6* v, uint32_t N){ double t=now_ns(); float r=base.length_sq(v,N); gLenSq.ns+=now_ns()-t; gLenSq.calls++; return r; }
    float calculate_error(AngLin6ErrorSq& e, const AngLin6* v, uint32_t N){ double t=now_ns(); float r=base.calculate_error(e,v,N); gErr.ns+=now_ns()-t; gErr.calls++; return r; }
};
struct InstrMatOps {
    BondMatrixOpsS<float> base;
    void rmul(AngLin6* y, const BondMatrixS& B, const AngLin6* x, uint32_t M, uint32_t N) const { double t=now_ns(); base.rmul(y,B,x,M,N); gRmul.ns+=now_ns()-t; gRmul.calls++; }
    void lmul(AngLin6* y, const AngLin6* x, const BondMatrixS& B, uint32_t M, uint32_t N) const { double t=now_ns(); base.lmul(y,x,B,M,N); gLmul.ns+=now_ns()-t; gLmul.calls++; }
};
typedef CGNR<AngLin6, InstrElemOps, BondMatrixS, InstrMatOps, float, AngLin6ErrorSq> CGNR_Instr;

static void instrumentedSolve(const Problem& p, uint32_t maxIter, float tol) {
    const uint32_t M = p.M(), N = p.N();
    Matrix mat; mat.build(p);
    std::vector<AngLin6> b(M), x(N);
    for (uint32_t i = 0; i < M; ++i) { b[i].ang = p.velocities[i].ang; b[i].lin = p.velocities[i].lin; }
    std::vector<uint8_t> cache(CGNR_Scalar().required_cache_size(M, N));

    resetAcc();
    AngLin6ErrorSq err{};
    double t0 = now_ns();
    int iters = CGNR_Instr().solve(x.data(), mat.B, b.data(), M, N, cache.data(), &err, tol, maxIter, 0);
    double total = now_ns() - t0;

    double kern = gLmul.ns + gRmul.ns + gErr.ns + gLenSq.ns + gVmadd.ns + gVnmadd.ns + gOther.ns;
    int it = iters < 0 ? -iters : iters;
    printf("\n── Instrumented whole-graph solve · %s (M=%u, N=%u) · %d iters%s · %.3f ms ──\n",
           p.name.c_str(), M, N, it, iters < 0 ? " (capped)" : " (converged)", total / 1e6);
    printf("  %-26s %10s %8s %10s %14s\n", "kernel", "Σ ms", "calls", "%solve", "µs/call");
    struct R { const char* n; Acc a; };
    R rs[] = {{"lmul  (Bᵀr)", gLmul}, {"rmul  (Bp)", gRmul}, {"calculate_error |z|²", gErr},
              {"length_sq |s|²", gLenSq}, {"vmadd (p,x updates)", gVmadd}, {"vnmadd (r update)", gVnmadd},
              {"other (vsub/dot)", gOther}};
    std::sort(rs, rs + 7, [](const R&a, const R&b){ return a.a.ns > b.a.ns; });
    for (auto& r : rs)
        printf("  %-26s %10.3f %8llu %9.1f%% %14.3f\n", r.n, r.a.ns/1e6,
               (unsigned long long)r.a.calls, 100.0*r.a.ns/kern, r.a.calls? r.a.ns/1000.0/r.a.calls : 0.0);
    printf("  %-26s %10.3f (timer-overhead excluded from shares)\n", "Σ kernels", kern/1e6);
}

// ──────────────────────────────────────────────────────────────────────────────
// 3) END-TO-END via the REAL StressProcessor (prepare + island-aware solve), the
//    faithful production path. Reports ms/solve at production settings.
// ──────────────────────────────────────────────────────────────────────────────
static void endToEnd(const Problem& p, bool islandAware, uint32_t maxIter, float tol) {
    StressProcessor sp;
    StressProcessor::DataParams dp; dp.centerBonds = true; dp.equalizeMasses = true;
    sp.prepare(p.nodes.data(), p.M(), p.bonds.data(), p.N(), dp);

    std::vector<AngLin6> impulses(p.N());
    std::memset(impulses.data(), 0, sizeof(AngLin6) * p.N());

    StressProcessor::SolverParams params;
    params.maxIter = maxIter; params.tolerance = tol; params.warmStart = true;
    params.islandAware = islandAware; params.skipSettled = false;

    AngLin6ErrorSq err{};
    // Frame 0: cold-ish warm start. Then warm-resume frames (production reuses impulses).
    auto once = [&](bool resume){
        double t = now_ns();
        int it = sp.solve(impulses.data(), p.velocities.data(), params, &err, resume);
        return std::pair<double,int>(now_ns() - t, it);
    };
    auto first = once(false);
    // steady-state warm frames
    double best = 1e300, sum = 0; int frames = 30, itlast = 0;
    for (int f = 0; f < frames; ++f) { auto r = once(true); best = std::min(best, r.first); sum += r.first; itlast = r.second; }
    printf("\n── End-to-end StressProcessor::solve · %s · islandAware=%d · maxIter=%u ──\n",
           p.name.c_str(), (int)islandAware, maxIter);
    printf("  cold/first frame : %.3f ms (%d iters)\n", first.first/1e6, first.second<0?-first.second:first.second);
    printf("  warm steady-state: %.3f ms mean · %.3f ms best (last=%d iters%s) · islands=%u\n",
           sum/frames/1e6, best/1e6, itlast<0?-itlast:itlast, itlast<0?" capped":"",
           sp.getLastIslandsTotal());
}

// ──────────────────────────────────────────────────────────────────────────────
// 4) SOLUTION FINGERPRINT — a stable hash of the solved impulses + iteration count.
//    This is the bit-exactness lock: any "same output, faster" optimization MUST
//    reproduce these exact values. Run before & after a change to prove no drift.
// ──────────────────────────────────────────────────────────────────────────────
static uint64_t fnv1a(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data; uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
static void fingerprint(const Problem& p, bool islandAware, uint32_t maxIter, float tol) {
    StressProcessor sp;
    StressProcessor::DataParams dp; dp.centerBonds = true; dp.equalizeMasses = true;
    sp.prepare(p.nodes.data(), p.M(), p.bonds.data(), p.N(), dp);
    std::vector<AngLin6> impulses(p.N());
    std::memset(impulses.data(), 0, sizeof(AngLin6) * p.N());
    StressProcessor::SolverParams params;
    params.maxIter = maxIter; params.tolerance = tol; params.warmStart = true;
    params.islandAware = islandAware; params.skipSettled = false;
    AngLin6ErrorSq err{};
    int it = 0;
    for (int f = 0; f < 5; ++f) it = sp.solve(impulses.data(), p.velocities.data(), params, &err, f > 0);
    // canonicalize -0.0f / NaN-free expectation, hash the 6 meaningful floats per impulse
    std::vector<float> flat; flat.reserve(p.N() * 6);
    for (uint32_t j = 0; j < p.N(); ++j) {
        const AngLin6& a = impulses[j];
        flat.insert(flat.end(), { a.ang.x, a.ang.y, a.ang.z, a.lin.x, a.lin.y, a.lin.z });
    }
    printf("  %-30s islandAware=%d  iters=%-4d  hash=%016llx  err=(%.6e,%.6e)\n",
           p.name.c_str(), (int)islandAware, it, (unsigned long long)fnv1a(flat.data(), flat.size()*sizeof(float)),
           err.ang, err.lin);
}

// Exercises island-cache invalidation: solve a few frames (build + reuse the cache), remove
// some bonds (must invalidate), then solve more (rebuild for the new topology + reuse). The
// hash must be identical with the cache on (default) and off (-DSTRESS_SOLVER_NO_ISLAND_CACHE).
static void fingerprintRemoveBond(const Problem& p, const char* label, uint32_t maxIter, float tol) {
    StressProcessor sp;
    StressProcessor::DataParams dp; dp.centerBonds = true; dp.equalizeMasses = true;
    sp.prepare(p.nodes.data(), p.M(), p.bonds.data(), p.N(), dp);
    std::vector<AngLin6> impulses(p.N());
    std::memset(impulses.data(), 0, sizeof(AngLin6) * p.N());
    StressProcessor::SolverParams params;
    params.maxIter = maxIter; params.tolerance = tol; params.warmStart = true;
    params.islandAware = true; params.skipSettled = false;
    AngLin6ErrorSq err{};
    for (int f = 0; f < 3; ++f) sp.solve(impulses.data(), p.velocities.data(), params, &err, f > 0);
    for (uint32_t k = 0; k < 8 && sp.getBondCount() > 16; ++k) {           // remove bonds (invalidates cache)
        const uint32_t idx = (sp.getBondCount() * 37u + 11u * k) % sp.getBondCount();
        sp.removeBond(idx);
        impulses[idx] = impulses[sp.getBondCount()];                       // mirror replaceWithLast on the caller's array
    }
    for (int f = 0; f < 3; ++f) sp.solve(impulses.data(), p.velocities.data(), params, &err, f > 0);
    const uint32_t nb = sp.getBondCount();
    std::vector<float> flat; flat.reserve(nb * 6);
    for (uint32_t j = 0; j < nb; ++j) { const AngLin6& a = impulses[j]; flat.insert(flat.end(), { a.ang.x, a.ang.y, a.ang.z, a.lin.x, a.lin.y, a.lin.z }); }
    printf("  %-30s islandAware=1  bonds=%-5u  hash=%016llx  err=(%.6e,%.6e)\n",
           label, nb, (unsigned long long)fnv1a(flat.data(), flat.size()*sizeof(float)), err.ang, err.lin);
}

int main() {
    printf("CGNR scalar-path benchmark (WASM-config: NO_SIMD, FORCE_SCALAR, -O3)\n");
    printf("AngLin6=%zuB Coupling=%zuB InertiaS=%zuB\n", sizeof(AngLin6), sizeof(Coupling), sizeof(InertiaS));

    const uint32_t MAXIT = 25;   // production: maxSolverIterationsPerFrame
    const float    TOL   = 0.001f;

    // A single mid-rise building ≈ one island the per-island solver actually runs.
    Problem building = makeBuilding(6, 6, 12, "building-6x6x12");
    // City-scale ≈ the recording (target ~7264 nodes / 19356 bonds). 28 buildings ×
    // 6×6×(avg ~8) ≈ 8k nodes; whole-graph path stresses scale.
    Problem city = makeCity(28, 6, 6, 4, 12, "city-28x(6x6x4..12)");

    microbench(building);
    instrumentedSolve(building, MAXIT, TOL);
    endToEnd(building, /*islandAware=*/false, MAXIT, TOL);

    printf("\n========================  CITY SCALE  ========================\n");
    microbench(city);
    instrumentedSolve(city, MAXIT, TOL);
    endToEnd(city, /*islandAware=*/false, MAXIT, TOL);
    endToEnd(city, /*islandAware=*/true,  MAXIT, TOL);

    printf("\n── Solution fingerprints (bit-exactness lock; must be invariant under any "
           "\"same output, faster\" change) ──\n");
    fingerprint(building, false, MAXIT, TOL);
    fingerprint(city,     false, MAXIT, TOL);
    fingerprint(city,     true,  MAXIT, TOL);
    fingerprintRemoveBond(city, "city+removeBond", MAXIT, TOL);

    printf("\n(sink=%g)\n", (double)g_sink);
    return 0;
}
