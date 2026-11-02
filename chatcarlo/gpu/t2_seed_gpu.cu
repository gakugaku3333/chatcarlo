// G5 A2b fixed P9c specialization; shared t2 physics and Philox.
#ifndef NAN
#define NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
#include "t2_kernel.h"
#include "t2_seed_entry.h"
extern "C" __global__ void t2_seed_init(double *data,const long long *meta,const double *constants,int nm){if(blockIdx.x||threadIdx.x)return;for(int j=0;j<nm;++j){P8gMaterial m=p8g_material(data,meta+136*j,constants);p8g_initialize(m);}}
extern "C" __global__ void t2_seed_kernel(const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,const double *input,double *d,long long *i,double *ledger,long long *sites,long long *source_counts,double *regions){
   
    T2SeedState q;for(int h=blockIdx.x*blockDim.x+threadIdx.x;h<n;h+=gridDim.x*blockDim.x)
        t2_seed_one(&q,data,meta,constants,g,seed,start+h,maxsteps,input?input+23*h:0,d+37*h,i+79*h,ledger+6*h,sites+6*h,source_counts+4*h,regions?regions+T2S_REGION_ROW*h:0);
}
extern "C" __global__ void t2_seed_boundary_probe(const T2SeedGeometry *g,const double *a,int n,double *out){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)t2_seed_boundary_one(g,a+23*j,out+6*j);}
extern "C" __global__ void t2_seed_score_probe(const T2SeedGeometry *g,const double *a,const double *b,double loss,int site,int repeats){for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<repeats;j+=gridDim.x*blockDim.x)for(int k=0;k<2;++k){if(site==3)t2_seed_segment(g->grids+k,a,b,loss);else t2_seed_point(g->grids+k,b,loss);}}
// T2Seed: fixed-state entry, no source or clock reinitialization (P9b G5 contract).
extern "C" __global__ void t2_seed_fixed_kernel(const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,const double *input,uint64_t seed,double *out){
    if(blockIdx.x||threadIdx.x)return;T2SeedState q={};q.cached_ev=-1.;q.M=constants[0];q.source_energy=q.local.source=input[6];
    q.touch[0]=g->nc;q.touch[1]=-1;q.touch[2]=g->nf;q.touch[3]=-1;q.resume_top=-2;
    for(int j=0;j<23;++j)q.state.a[j]=input[j];rng_init(&q.rng,seed,0);
    if(input[19]==0){q.stack[0]=q.state;q.top=0;q.phase=T2S_P;}else{q.top=-1;q.phase=T2S_C;}
    while(q.phase!=T2S_D)t2_seed_advance(&q,data,meta,constants,g,0,100000,0);
    for(int j=0;j<6;++j){out[j]=q.sites[j];out[6+j]=q.ledger_energy[j];}out[12]=q.dep;out[13]=q.steps;out[14]=q.failed?1.:0.;
}

extern "C" __global__ void t2_seed_identity(unsigned long long *out){if(!blockIdx.x&&!threadIdx.x){out[0]=G5_SET_ID;out[1]=G5_SET_TAG;}}

#include "t2_seed_probe.h"
extern "C" __global__ void t2_seed_probe(const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,const double *input,const double *fixed,int mode,uint64_t seed,uint64_t start,int n,double *out){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<n)t2_seed_probe_one(data,meta,constants,g,input+23*h,fixed+3*h,mode,seed,start+h,out+74*h);}
extern "C" __global__ void t2_seed_source_probe(const T2SeedGeometry *g,const double *s,const double *d,double *out){if(blockIdx.x||threadIdx.x)return;out[0]=t2_seed_ellipsoid(s,d);double t=t2_seed_entry(g,s,d);out[1]=t;for(int k=0;k<3;++k)out[k+2]=s[k]+t*d[k];}
