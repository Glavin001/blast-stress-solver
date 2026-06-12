// Scaling sweep: the settled-bond stress skip saves time proportional to the settled bonds it skips.
// Sweeps city size (fixed small "active" fraction = mostly-settled city with localized action, the
// realistic case) and prints the per-frame solve cost. Build default vs -DSTRESS_NO_BONDSTRESS_SKIP
// and the run script joins the two columns.
#include "ext_stress_bridge.h"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double,std::milli>(b-a).count(); }
struct City{ std::vector<ExtStressNodeDesc> nodes; std::vector<ExtStressBondDesc> bonds; };
static void addLattice(City& c,int W,int D,int Hh,float ox,float oz,std::mt19937& rng){
    std::uniform_real_distribution<float> jit(-0.02f,0.02f); const uint32_t base=(uint32_t)c.nodes.size();
    auto idx=[&](int x,int y,int z){ return base+(uint32_t)((z*D+y)*W+x); };
    for(int z=0;z<Hh;++z)for(int y=0;y<D;++y)for(int x=0;x<W;++x){ ExtStressNodeDesc n;
        n.centroid={ox+x+jit(rng),(float)z,oz+y+jit(rng)}; n.mass=(z==0)?0.0f:1000.0f; n.volume=1.0f; c.nodes.push_back(n);}
    auto bond=[&](uint32_t a,uint32_t b,float nx,float ny,float nz){ const auto&A=c.nodes[a],&B=c.nodes[b];
        ExtStressBondDesc d; d.centroid={0.5f*(A.centroid.x+B.centroid.x),0.5f*(A.centroid.y+B.centroid.y),0.5f*(A.centroid.z+B.centroid.z)};
        d.normal={nx,ny,nz}; d.area=1.0f; d.node0=a; d.node1=b; c.bonds.push_back(d);};
    for(int z=0;z<Hh;++z)for(int y=0;y<D;++y)for(int x=0;x<W;++x){
        if(x+1<W)bond(idx(x,y,z),idx(x+1,y,z),1,0,0); if(y+1<D)bond(idx(x,y,z),idx(x,y+1,z),0,0,1); if(z+1<Hh)bond(idx(x,y,z),idx(x,y,z+1),0,1,0);}
}
static void run(const char* label,int nB,int W,int D,int Hmin,int Hmax,int active){
    City c; std::mt19937 rng(7654321u); std::uniform_int_distribution<int> hd(Hmin,Hmax);
    int cols=(int)std::ceil(std::sqrt((double)nB));
    std::vector<uint32_t> driveNode;   // a real, dynamic node in each building (top-centre)
    for(int b=0;b<nB;++b){ uint32_t base=(uint32_t)c.nodes.size(); int H=hd(rng);
        addLattice(c,W,D,H,(float)((b%cols)*(W+6)),(float)((b/cols)*(D+6)),rng);
        driveNode.push_back(base + (uint32_t)(((H-1)*D + D/2)*W + W/2)); }   // top layer, centre column
    ExtStressSolverSettingsDesc s{}; s.max_solver_iterations_per_frame=25; s.graph_reduction_level=0;
    s.compression_elastic_limit=1e9f; s.compression_fatal_limit=2e9f; s.tension_elastic_limit=1e9f;
    s.tension_fatal_limit=2e9f; s.shear_elastic_limit=1e9f; s.shear_fatal_limit=2e9f;
    ExtStressSolverHandle* h=ext_stress_solver_create(c.nodes.data(),(uint32_t)c.nodes.size(),c.bonds.data(),(uint32_t)c.bonds.size(),&s);
    ext_stress_solver_set_island_aware(h,1); ext_stress_solver_set_skip_settled(h,1);
    double tSum=0; int n=0;
    for(int f=0; f<260; ++f){
        ext_stress_solver_add_all_actor_gravity(h,0.f,-9.81f,0.f,nullptr,0);
        for(int k=0;k<active && k<(int)driveNode.size();++k){ float p=std::sin(0.2f*f+k);
            const ExtStressNodeDesc& nd=c.nodes[driveNode[k]];
            StressVec3 pos={nd.centroid.x,nd.centroid.y,nd.centroid.z};  // actor add_force uses nearest-node-to-pos
            StressVec3 force={6000.0f*p,0.f,4000.0f*p};                  // varies each frame → defeats skip-settled for this island
            ext_stress_solver_add_force(h,driveNode[k],&pos,&force,0);}
        auto t0=Clock::now(); ext_stress_solver_update(h); double dt=ms(t0,Clock::now());
        if(f>=160){ tSum+=dt; ++n; }
    }
    printf("%-10s %8zu %8zu %7d %7.3f  %u/%u\n", label, c.nodes.size(), c.bonds.size(), active,
           tSum/n, ext_stress_solver_islands_skipped(h), ext_stress_solver_islands_total(h));
    ext_stress_solver_destroy(h);
}
int main(){
#if defined(STRESS_NO_BONDSTRESS_SKIP)
    printf("# build: skip OFF\n");
#else
    printf("# build: skip ON\n");
#endif
    printf("# ── size sweep (mostly settled, 3 buildings active = localized action) ──\n");
    printf("%-10s %8s %8s %7s %7s  %s\n","scene","nodes","bonds","active","ms","skip/total");
    run("tiny",    16, 5,5, 6,10, 3);
    run("small",   36, 6,6, 8,12, 3);
    run("medium",  64, 7,7, 8,14, 3);
    run("large",  100, 8,8,10,16, 3);
    run("xlarge", 169, 8,8,12,18, 3);
    printf("# ── activity sweep (fixed 'large' city, 100 buildings, ~219k bonds) ──\n");
    printf("%-10s %8s %8s %7s %7s  %s\n","scene","nodes","bonds","active","ms","skip/total");
    run("settled",  100, 8,8,10,16, 0);    // fully at rest
    run("act5",     100, 8,8,10,16, 5);
    run("act25",    100, 8,8,10,16, 25);
    run("act50",    100, 8,8,10,16, 50);
    run("act100",   100, 8,8,10,16, 100);   // whole city in motion → nothing skipped
    return 0;
}
