// Experiment: does bond/node REORDERING for cache locality speed up the CGNR stress solve?
//
// CONTEXT
//   The coupling mat-vec (CouplingMatrixOps::rmul/lmul) scatters/gathers into node slots via
//   bond endpoint indices — an irregular access pattern. The SpMV literature says reordering a
//   sparse matrix for locality (RCM/Cuthill-McKee + bandwidth reduction) can speed SpMV up to
//   ~2.6×. BUT that regime is matrices too big for cache (DRAM-bound). This harness tests
//   whether it helps THIS workload, where:
//     - the web build solves island-AWARE → each island is a few-hundred-node, cache-resident
//       sub-system, and the island gather already renumbers nodes first-touch-local;
//     - even the whole-graph mini-city (~7.3k nodes) is only ~256 KB of node vectors → L3.
//
//   For each problem it builds three orderings of the SAME matrix and times rmul/lmul:
//     natural   — as generated (nodes in (x,y,z) order: already decent locality)
//     shuffled  — random node renumber + random bond order (worst-case locality)
//     CM-local  — Cuthill-McKee node renumber (BFS) + bonds sorted by min endpoint (best case)
//   and measures the solution DRIFT (reordering changes the rmul scatter-accumulation order, so
//   it is NOT bit-exact — the solve moves within solver tolerance).
//
// BUILD/RUN: see reorder_run.sh (scalar config matching the wasm build; clang++/g++, -O3).

#include "../stress.h"
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

typedef CGNR<AngLin6, AngLin6Ops<float>, BondMatrixS, BondMatrixOpsS<float>, float, AngLin6ErrorSq> CGNR_Scalar;

using Clock = std::chrono::steady_clock;
static inline double now_ns() {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
static volatile float g_sink = 0.0f;
static void consume(const AngLin6* v, uint32_t n) { float s=0; for (uint32_t i=0;i<n;++i) s+=v[i].ang.x+v[i].lin.z; g_sink+=s; }

template <class F> static double bench_ns(F&& f, double target_ms = 60.0) {
    f(); f();
    double t0 = now_ns(); f(); double one = now_ns()-t0;
    int reps = (int)std::max(3.0, (target_ms*1e6)/std::max(one,1.0)); reps = std::min(reps, 2000000);
    double best = 1e300;
    for (int w=0; w<3; ++w) { double a=now_ns(); for (int r=0;r<reps;++r) f(); best=std::min(best,(now_ns()-a)/reps); }
    return best;
}

// ── Problem: 3D lattice "buildings" (~2.6 bonds/node, matching the mini-city). ──
struct Problem {
    std::vector<SolverNodeS> nodes; std::vector<SolverBond> bonds; std::vector<AngLin6> velocities; std::string name;
    uint32_t M() const { return (uint32_t)nodes.size(); } uint32_t N() const { return (uint32_t)bonds.size(); }
};
static void addLattice(Problem& p, int W, int D, int H, float sp, float ox, float oy, float oz, std::mt19937& rng) {
    std::uniform_real_distribution<float> jit(-0.02f,0.02f), vel(-0.5f,0.5f);
    const uint32_t base = (uint32_t)p.nodes.size();
    auto idx = [&](int x,int y,int z){ return base + (uint32_t)((z*D+y)*W+x); };
    for (int z=0;z<H;++z) for (int y=0;y<D;++y) for (int x=0;x<W;++x) {
        SolverNodeS n; n.CoM={ox+x*sp+jit(rng), oy+z*sp+jit(rng), oz+y*sp+jit(rng)};
        n.mass=(z==0)?0.0f:(800.0f+200.0f*(float)((x+y+z)%5)); n.inertia=(z==0)?0.0f:(n.mass*0.18f);
        p.nodes.push_back(n);
        AngLin6 v{}; if (z!=0){ v.lin={vel(rng),-1.0f+vel(rng),vel(rng)}; v.ang={0.1f*vel(rng),0.1f*vel(rng),0.1f*vel(rng)}; }
        p.velocities.push_back(v);
    }
    auto bond=[&](uint32_t a,uint32_t b,float bx,float by,float bz){ SolverBond s; s.nodes[0]=a; s.nodes[1]=b; s.centroid={bx,by,bz}; p.bonds.push_back(s); };
    for (int z=0;z<H;++z) for (int y=0;y<D;++y) for (int x=0;x<W;++x) {
        const NvcVec3 c=p.nodes[idx(x,y,z)].CoM;
        if (x+1<W){NvcVec3 c2=p.nodes[idx(x+1,y,z)].CoM; bond(idx(x,y,z),idx(x+1,y,z),0.5f*(c.x+c2.x),0.5f*(c.y+c2.y),0.5f*(c.z+c2.z));}
        if (y+1<D){NvcVec3 c2=p.nodes[idx(x,y+1,z)].CoM; bond(idx(x,y,z),idx(x,y+1,z),0.5f*(c.x+c2.x),0.5f*(c.y+c2.y),0.5f*(c.z+c2.z));}
        if (z+1<H){NvcVec3 c2=p.nodes[idx(x,y,z+1)].CoM; bond(idx(x,y,z),idx(x,y,z+1),0.5f*(c.x+c2.x),0.5f*(c.y+c2.y),0.5f*(c.z+c2.z));}
    }
}
static Problem makeBuilding(int W,int D,int H,const char* name){ Problem p; p.name=name; std::mt19937 rng(1234567u); addLattice(p,W,D,H,1,0,0,0,rng); return p; }
static Problem makeCity(int b,int W,int D,int Hmin,int Hmax,const char* name){
    Problem p; p.name=name; std::mt19937 rng(7654321u); std::uniform_int_distribution<int> hd(Hmin,Hmax);
    int cols=(int)std::ceil(std::sqrt((double)b));
    for (int i=0;i<b;++i){ int H=hd(rng); addLattice(p,W,D,H,1,(i%cols)*(W+6)*1.0f,0,(i/cols)*(D+6)*1.0f,rng); }
    return p;
}

struct Matrix { std::vector<Coupling> C; std::vector<InertiaS> sqrtIinv; std::vector<AngLin6> scratch; BondMatrixS B; };

// Build the matrix under a node renumber (oldToNew) and a bond order (bondPerm).
static void buildReordered(Matrix& m, const Problem& p, const std::vector<uint32_t>& oldToNew, const std::vector<uint32_t>& bondPerm) {
    const uint32_t M=p.M(), N=p.N(); m.C.resize(N); m.sqrtIinv.resize(M); m.scratch.resize(M);
    for (uint32_t i=0;i<M;++i){ const SolverNodeS& n=p.nodes[i]; InertiaS I; I.I=n.inertia>0?std::sqrt(1.0f/n.inertia):0.0f; I.m=n.mass>0?std::sqrt(1.0f/n.mass):0.0f; m.sqrtIinv[oldToNew[i]]=I; }
    for (uint32_t jj=0;jj<N;++jj){ const SolverBond& b=p.bonds[bondPerm[jj]]; Coupling c; c.node0=oldToNew[b.nodes[0]]; c.node1=oldToNew[b.nodes[1]];
        c.offset0=b.centroid-p.nodes[b.nodes[0]].CoM; c.offset1=b.centroid-p.nodes[b.nodes[1]].CoM; m.C[jj]=c; }
    m.B.set(m.C.data(), m.sqrtIinv.data(), m.scratch.data(), M, N);
}
static std::vector<uint32_t> identityPerm(uint32_t n){ std::vector<uint32_t> v(n); for(uint32_t i=0;i<n;++i) v[i]=i; return v; }
static std::vector<uint32_t> cuthillMcKee(const Problem& p){           // BFS node renumber (bandwidth reduction)
    const uint32_t M=p.M(), N=p.N(); const uint32_t kInv=(uint32_t)-1;
    std::vector<std::vector<uint32_t>> adj(M);
    for (uint32_t j=0;j<N;++j){ uint32_t a=p.bonds[j].nodes[0], b=p.bonds[j].nodes[1]; adj[a].push_back(b); adj[b].push_back(a); }
    std::vector<uint32_t> o2n(M,kInv), q; uint32_t next=0;
    for (uint32_t s=0;s<M;++s){ if (o2n[s]!=kInv) continue; q.clear(); q.push_back(s); o2n[s]=next++;
        for (size_t h=0;h<q.size();++h) for (uint32_t v:adj[q[h]]) if (o2n[v]==kInv){ o2n[v]=next++; q.push_back(v); } }
    return o2n;
}

static void solveImp(const Problem& q, std::vector<AngLin6>& out) {
    StressProcessor sp; StressProcessor::DataParams dp; dp.centerBonds=true; dp.equalizeMasses=true;
    sp.prepare(q.nodes.data(), q.M(), q.bonds.data(), q.N(), dp);
    out.assign(q.N(), AngLin6{});
    StressProcessor::SolverParams pr; pr.maxIter=25; pr.tolerance=0.001f; pr.warmStart=true; pr.islandAware=true;
    AngLin6ErrorSq e{}; for (int f=0;f<5;++f) sp.solve(out.data(), q.velocities.data(), pr, &e, f>0);
}

static void experiment(const Problem& p) {
    const uint32_t M=p.M(), N=p.N(); std::mt19937 rng(2024);
    const double kb = M*(double)sizeof(AngLin6)/1024.0;

    std::vector<uint32_t> idN=identityPerm(M), idB=identityPerm(N);
    std::vector<uint32_t> shufN=idN, shufB=idB; std::shuffle(shufN.begin(),shufN.end(),rng); std::shuffle(shufB.begin(),shufB.end(),rng);
    std::vector<uint32_t> cmN=cuthillMcKee(p), cmB=idB;
    std::sort(cmB.begin(), cmB.end(), [&](uint32_t a,uint32_t b){
        return std::min(cmN[p.bonds[a].nodes[0]],cmN[p.bonds[a].nodes[1]]) < std::min(cmN[p.bonds[b].nodes[0]],cmN[p.bonds[b].nodes[1]]); });

    Matrix nat, shuf, cm; buildReordered(nat,p,idN,idB); buildReordered(shuf,p,shufN,shufB); buildReordered(cm,p,cmN,cmB);

    std::vector<AngLin6> zN(N), pN(N), sM(M), rM(M);
    std::mt19937 r2(7); std::uniform_real_distribution<float> d(-1,1);
    auto fill=[&](std::vector<AngLin6>&v){ for(auto&e:v){ e.ang={d(r2),d(r2),d(r2)}; e.lin={d(r2),d(r2),d(r2)}; } };
    fill(pN); fill(rM);
    BondMatrixOpsS<float> bops;
    auto tr=[&](Matrix& m){ return bench_ns([&]{ bops.rmul(sM.data(), m.B, pN.data(), M, N); consume(sM.data(),1);} )/N; };
    auto tl=[&](Matrix& m){ return bench_ns([&]{ bops.lmul(zN.data(), rM.data(), m.B, M, N); consume(zN.data(),1);} )/N; };

    printf("\n── %s (M=%u, N=%u; node vectors %.0f KB %s) ──\n", p.name.c_str(), M, N, kb,
           kb<48?"→ L1-resident":kb<8192?"→ L2/L3-resident":"→ exceeds L3 (DRAM)");
    printf("  %-9s %13s %13s   %s\n","ordering","rmul ns/bond","lmul ns/bond","vs natural");
    double rn=tr(nat), ln=tl(nat), rs=tr(shuf), ls=tl(shuf), rc=tr(cm), lc=tl(cm);
    printf("  %-9s %13.3f %13.3f   (baseline)\n","natural",rn,ln);
    printf("  %-9s %13.3f %13.3f   rmul %+.0f%% lmul %+.0f%%\n","shuffled",rs,ls,100*(rs-rn)/rn,100*(ls-ln)/ln);
    printf("  %-9s %13.3f %13.3f   rmul %+.0f%% lmul %+.0f%%\n","CM-local",rc,lc,100*(rc-rn)/rn,100*(lc-ln)/ln);
    printf("  → CM-local vs natural (the realistic gain): rmul %+.1f%%, lmul %+.1f%%\n",100*(rc-rn)/rn,100*(lc-ln)/ln);
    printf("  → shuffled→CM-local (max gain, only vs a BAD input): rmul %.0f%%, lmul %.0f%%\n",100*(rs-rc)/rs,100*(ls-lc)/ls);

    // Solution drift: solve natural vs CM-reordered (island-aware), map CM impulses back, relative L2.
    if (N > 60000) { printf("  → solution drift: (skipped for the oversized stress case)\n"); return; }
    Problem q; q.name=p.name; q.nodes.resize(M); q.velocities.resize(M); q.bonds.resize(N);
    for (uint32_t i=0;i<M;++i){ q.nodes[cmN[i]]=p.nodes[i]; q.velocities[cmN[i]]=p.velocities[i]; }
    for (uint32_t jj=0;jj<N;++jj){ SolverBond b=p.bonds[cmB[jj]]; b.nodes[0]=cmN[b.nodes[0]]; b.nodes[1]=cmN[b.nodes[1]]; q.bonds[jj]=b; }
    std::vector<AngLin6> impNat, impCM; solveImp(p, impNat); solveImp(q, impCM);
    double num=0, den=0;
    for (uint32_t jj=0;jj<N;++jj){ const AngLin6&a=impNat[cmB[jj]]; const AngLin6&b=impCM[jj];
        float dx[6]={a.ang.x-b.ang.x,a.ang.y-b.ang.y,a.ang.z-b.ang.z,a.lin.x-b.lin.x,a.lin.y-b.lin.y,a.lin.z-b.lin.z};
        float ax[6]={a.ang.x,a.ang.y,a.ang.z,a.lin.x,a.lin.y,a.lin.z};
        for (int k=0;k<6;++k){ num+=dx[k]*(double)dx[k]; den+=ax[k]*(double)ax[k]; } }
    printf("  → solution drift (CM-local vs natural, island-aware): relative L2 = %.2e  (NOT bit-exact; solver tol 1e-3)\n",
           den>0?std::sqrt(num/den):0.0);
}

int main() {
    printf("Bond/node reordering-for-locality experiment (scalar CGNR, WASM config, -O3)\n");
    printf("AngLin6=%zuB Coupling=%zuB\n", sizeof(AngLin6), sizeof(Coupling));
    experiment(makeBuilding(6,6,12, "building-6x6x12  (≈ one island the island-aware path solves)"));
    experiment(makeCity(28,6,6,4,12, "city-28 buildings (≈ recording's whole-graph scale)"));
    // A deliberately large single block whose node vectors spill past L3, to show where reorder
    // WOULD matter (and confirm this workload is not in that regime).
    experiment(makeBuilding(40,40,40, "big-block-40^3   (DRAM-scale stress test, not a real scene)"));
    printf("\n(sink=%g)\n", (double)g_sink);
    return 0;
}
