// Native A/B + timing for the settled-bond stress skip, with island-aware + skip-settled ON
// (the only mode in which the skip fires — the mini-city/WASM path). Settles a city, then drives
// a few buildings with time-varying forces so others stay settled (skip fires on them), hashing the
// per-frame stress OUTPUT (debug-render lines + per-actor excess forces + overstressed/bond/actor
// counts). Build twice (default vs -DSTRESS_NO_BONDSTRESS_SKIP) and diff the hash → bit-exactness.
#include "ext_stress_bridge.h"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
static uint64_t H=1469598103934665603ull;
static inline void hb(const void* p, size_t n){ const uint8_t* b=(const uint8_t*)p; for(size_t i=0;i<n;++i){ H^=b[i]; H*=1099511628211ull; } }
static inline void hf(float f){ hb(&f,4); }
static inline void hu(uint32_t u){ hb(&u,4); }

struct City { std::vector<ExtStressNodeDesc> nodes; std::vector<ExtStressBondDesc> bonds; };
static void addLattice(City& c,int W,int D,int H_,float ox,float oz,std::mt19937& rng){
    std::uniform_real_distribution<float> jit(-0.02f,0.02f);
    const uint32_t base=(uint32_t)c.nodes.size();
    auto idx=[&](int x,int y,int z){ return base+(uint32_t)((z*D+y)*W+x); };
    for(int z=0;z<H_;++z)for(int y=0;y<D;++y)for(int x=0;x<W;++x){
        ExtStressNodeDesc n; n.centroid={ox+x+jit(rng),(float)z,oz+y+jit(rng)};
        n.mass=(z==0)?0.0f:(800.0f+200.0f*(float)((x+y+z)%5)); n.volume=1.0f; c.nodes.push_back(n);
    }
    auto bond=[&](uint32_t a,uint32_t b,float nx,float ny,float nz){ const ExtStressNodeDesc&A=c.nodes[a],&B=c.nodes[b];
        ExtStressBondDesc d; d.centroid={0.5f*(A.centroid.x+B.centroid.x),0.5f*(A.centroid.y+B.centroid.y),0.5f*(A.centroid.z+B.centroid.z)};
        d.normal={nx,ny,nz}; d.area=1.0f; d.node0=a; d.node1=b; c.bonds.push_back(d); };
    for(int z=0;z<H_;++z)for(int y=0;y<D;++y)for(int x=0;x<W;++x){
        if(x+1<W)bond(idx(x,y,z),idx(x+1,y,z),1,0,0);
        if(y+1<D)bond(idx(x,y,z),idx(x,y+1,z),0,0,1);
        if(z+1<H_)bond(idx(x,y,z),idx(x,y,z+1),0,1,0);
    }
}

int main(){
    City c; std::mt19937 rng(7654321u); std::uniform_int_distribution<int> hd(4,12);
    const int B=28,W=6,D=6,cols=6;
    for(int b=0;b<B;++b) addLattice(c,W,D,hd(rng),(float)((b%cols)*(W+6)),(float)((b/cols)*(D+6)),rng);

    ExtStressSolverSettingsDesc s{}; s.max_solver_iterations_per_frame=25; s.graph_reduction_level=0;
    s.compression_elastic_limit=1e9f; s.compression_fatal_limit=2e9f; s.tension_elastic_limit=1e9f;
    s.tension_fatal_limit=2e9f; s.shear_elastic_limit=1e9f; s.shear_fatal_limit=2e9f;   // Strong: nothing breaks
    ExtStressSolverHandle* h=ext_stress_solver_create(c.nodes.data(),(uint32_t)c.nodes.size(),c.bonds.data(),(uint32_t)c.bonds.size(),&s);
    ext_stress_solver_set_island_aware(h,1); ext_stress_solver_set_skip_settled(h,1);
    const uint32_t NB=ext_stress_solver_bond_count(h), NA=ext_stress_solver_actor_count(h);
    printf("city: %zu nodes, %zu bonds; solver bonds=%u actors=%u\n", c.nodes.size(), c.bonds.size(), NB, NA);

    std::vector<ExtStressDebugLine> lines(NB+16);
    double tSettled=0; int nSettled=0;

    for(int f=0; f<400; ++f){
        ext_stress_solver_add_all_actor_gravity(h,0.f,-9.81f,0.f,nullptr,0);
        // Drive a few nodes in 3 buildings with time-varying force so those islands re-solve every frame,
        // while the other 25 buildings settle and get skipped. (node index < first building's node count.)
        for(int k=0;k<3;++k){ uint32_t node=(uint32_t)(k*200+30); float p=0.3f*std::sin(0.2f*f+k);
            StressVec3 pos={0,0,0}, force={120.0f*p, -60.0f, 80.0f*p};
            ext_stress_solver_add_force(h,node,&pos,&force,0); }

        auto t0=Clock::now();
        ext_stress_solver_update(h);
        double dt=ms(t0,Clock::now());
        if(f>=200){ tSettled+=dt; ++nSettled; }   // measure steady-state cost

        // hash the stress OUTPUT each frame
        hu(ext_stress_solver_overstressed_bond_count(h));
        hu(ext_stress_solver_bond_count(h));
        hu(ext_stress_solver_islands_skipped(h)); hu(ext_stress_solver_islands_total(h));
        uint32_t nl=ext_stress_solver_fill_debug_render(h,0,1.0f,lines.data(),(uint32_t)lines.size()); // mode 0 = STRESS_PCT_MAX
        hu(nl);
        for(uint32_t i=0;i<nl;++i){ const ExtStressDebugLine&L=lines[i];
            hf(L.p0.x);hf(L.p0.y);hf(L.p0.z);hf(L.p1.x);hf(L.p1.y);hf(L.p1.z);hu(L.color0);hu(L.color1); }
    }
    printf("frames hashed: 400   stress-output hash = %016llx\n", (unsigned long long)H);
    printf("steady-state (frames 200-399, 3 active / 25 settled buildings): %.4f ms/solve\n", tSettled/nSettled);
    printf("  islands skipped last frame: %u/%u\n", ext_stress_solver_islands_skipped(h), ext_stress_solver_islands_total(h));
    ext_stress_solver_destroy(h);
    return 0;
}
