// Tier-2: bit-exactness THROUGH the fracture cycle (the partial-damage case). Weak material so
// gravity overstresses bonds → generate+apply fracture commands → bond health drops (partial damage)
// and islands split → settle → skip fires. Hashes the stress output + fracture decisions + topology
// every frame. Build default vs -DSTRESS_NO_BONDSTRESS_SKIP and diff → must be identical.
#include "ext_stress_bridge.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <random>

static uint64_t H=1469598103934665603ull;
static inline void hb(const void* p,size_t n){ const uint8_t* b=(const uint8_t*)p; for(size_t i=0;i<n;++i){ H^=b[i]; H*=1099511628211ull; } }
static inline void hf(float f){ hb(&f,4);} static inline void hu(uint32_t u){ hb(&u,4);}

struct City{ std::vector<ExtStressNodeDesc> nodes; std::vector<ExtStressBondDesc> bonds; };
static void addLattice(City& c,int W,int D,int Hh,float ox,float oz,std::mt19937& rng){
    std::uniform_real_distribution<float> jit(-0.02f,0.02f); const uint32_t base=(uint32_t)c.nodes.size();
    auto idx=[&](int x,int y,int z){ return base+(uint32_t)((z*D+y)*W+x); };
    for(int z=0;z<Hh;++z)for(int y=0;y<D;++y)for(int x=0;x<W;++x){ ExtStressNodeDesc n;
        n.centroid={ox+x+jit(rng),(float)z,oz+y+jit(rng)}; n.mass=(z==0)?0.0f:1000.0f; n.volume=1.0f; c.nodes.push_back(n); }
    auto bond=[&](uint32_t a,uint32_t b,float nx,float ny,float nz){ const auto&A=c.nodes[a],&B=c.nodes[b];
        ExtStressBondDesc d; d.centroid={0.5f*(A.centroid.x+B.centroid.x),0.5f*(A.centroid.y+B.centroid.y),0.5f*(A.centroid.z+B.centroid.z)};
        d.normal={nx,ny,nz}; d.area=1.0f; d.node0=a; d.node1=b; c.bonds.push_back(d); };
    for(int z=0;z<Hh;++z)for(int y=0;y<D;++y)for(int x=0;x<W;++x){
        if(x+1<W)bond(idx(x,y,z),idx(x+1,y,z),1,0,0); if(y+1<D)bond(idx(x,y,z),idx(x,y+1,z),0,0,1); if(z+1<Hh)bond(idx(x,y,z),idx(x,y,z+1),0,1,0); }
}

int main(){
    City c; std::mt19937 rng(13u); std::uniform_int_distribution<int> hd(5,11);
    for(int b=0;b<16;++b) addLattice(c,5,5,hd(rng),(float)((b%4)*11),(float)((b/4)*11),rng);
    ExtStressSolverSettingsDesc s{}; s.max_solver_iterations_per_frame=25; s.graph_reduction_level=0;
    // Weak limits → gravity alone overstresses & progressively damages bonds.
    s.compression_elastic_limit=2000.f; s.compression_fatal_limit=8000.f; s.tension_elastic_limit=1500.f;
    s.tension_fatal_limit=6000.f; s.shear_elastic_limit=1500.f; s.shear_fatal_limit=6000.f;
    ExtStressSolverHandle* h=ext_stress_solver_create(c.nodes.data(),(uint32_t)c.nodes.size(),c.bonds.data(),(uint32_t)c.bonds.size(),&s);
    ext_stress_solver_set_island_aware(h,1); ext_stress_solver_set_skip_settled(h,1);

    const uint32_t maxBonds=(uint32_t)c.bonds.size()+16, maxNodes=(uint32_t)c.nodes.size()+16;
    std::vector<ExtStressFractureCommands> cmd(64); std::vector<ExtStressBondFracture> bondBuf(maxBonds);
    std::vector<ExtStressSplitEvent> evt(64); std::vector<ExtStressActor> child(maxNodes); std::vector<uint32_t> nodesBuf(maxNodes);
    int totalFractures=0, totalSplits=0;

    for(int f=0; f<300; ++f){
        ext_stress_solver_add_all_actor_gravity(h,0.f,-9.81f,0.f,nullptr,0);
        ext_stress_solver_update(h);
        // hash stress output
        hu(ext_stress_solver_overstressed_bond_count(h)); hu(ext_stress_solver_bond_count(h));
        hu(ext_stress_solver_actor_count(h)); hu(ext_stress_solver_islands_skipped(h)); hu(ext_stress_solver_islands_total(h));
        hf(ext_stress_solver_get_linear_error(h)); hf(ext_stress_solver_get_angular_error(h));
        // generate + hash + apply fracture commands (this is what changes bond health → the partial-damage case)
        uint32_t outCmd=0,outBond=0;
        ext_stress_solver_generate_fracture_commands_per_actor(h,cmd.data(),(uint32_t)cmd.size(),bondBuf.data(),maxBonds,&outCmd,&outBond);
        hu(outCmd); hu(outBond);
        for(uint32_t i=0;i<outBond;++i){ hu(bondBuf[i].nodeIndex0); hu(bondBuf[i].nodeIndex1); hf(bondBuf[i].health); }
        totalFractures+=outBond;
        if(outCmd>0){
            uint32_t oe=0,oc=0,on=0;
            ext_stress_solver_apply_fracture_commands(h,cmd.data(),outCmd,evt.data(),(uint32_t)evt.size(),
                child.data(),(uint32_t)child.size(),&oe,&oc,nodesBuf.data(),maxNodes,&on);
            totalSplits+=oe; hu(oe); hu(oc);
        }
    }
    printf("frames=300  total bond-fractures applied=%d  splits=%d\n", totalFractures, totalSplits);
    printf("stress+fracture+topology hash = %016llx\n", (unsigned long long)H);
    printf("final: actors=%u islands=%u skipped=%u\n", ext_stress_solver_actor_count(h),
           ext_stress_solver_islands_total(h), ext_stress_solver_islands_skipped(h));
    ext_stress_solver_destroy(h);
    return 0;
}
