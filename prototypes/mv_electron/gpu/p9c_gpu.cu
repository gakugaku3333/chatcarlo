#ifndef NAN
#define NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
#include "p9c_entry.h"
extern "C" __global__ void p9c_init(double *data,const long long *meta,const double *constants,int nm){if(blockIdx.x||threadIdx.x)return;for(int j=0;j<nm;++j){P8gMaterial m=p8g_material(data,meta+136*j,constants);p8g_initialize(m);}}
extern "C" __global__ void p9c_kernel(const double *data,const long long *meta,const double *constants,const P9cGeometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,const double *input,double *d,long long *i,double *ledger,long long *sites,long long *source_counts,double *regions){
    // P9c: a door is a separate launch; history IDs stay global across batches.
    P9cState q;for(int h=blockIdx.x*blockDim.x+threadIdx.x;h<n;h+=gridDim.x*blockDim.x)
        p9c_one(&q,data,meta,constants,g,seed,start+h,maxsteps,input?input+23*h:0,d+37*h,i+79*h,ledger+6*h,sites+6*h,source_counts+2*h,regions?regions+3420*h:0);
}
extern "C" __global__ void p9c_boundary_probe(const P9cGeometry *g,const double *a,int n,double *out){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)p9c_boundary_one(g,a+23*j,out+6*j);}
extern "C" __global__ void p9c_score_probe(const P9cGeometry *g,const double *a,const double *b,double loss,int site,int repeats){for(int j=blockIdx.x*blockDim.x+threadIdx.x;j<repeats;j+=gridDim.x*blockDim.x)for(int k=0;k<2;++k){if(site==3)p9c_segment(g->grids+k,a,b,loss);else p9c_point(g->grids+k,b,loss);}}
// P9c: fixed-state entry, no source or clock reinitialization (P9b G5 contract).
extern "C" __global__ void p9c_fixed_kernel(const double *data,const long long *meta,const double *constants,const P9cGeometry *g,const double *input,uint64_t seed,double *out){
    if(blockIdx.x||threadIdx.x)return;P9cState q={};q.cached_ev=-1.;q.M=constants[0];q.source_energy=q.local.source=input[6];
    q.touch[0]=g->nc;q.touch[1]=-1;q.touch[2]=g->nf;q.touch[3]=-1;q.resume_top=-2;
    for(int j=0;j<23;++j)q.state.a[j]=input[j];rng_init(&q.rng,seed,0);
    if(input[19]==0){q.stack[0]=q.state;q.top=0;q.phase=P8G2_P;}else{q.top=-1;q.phase=P8G2_C;}
    while(q.phase!=P8G2_D)p9c_advance(&q,data,meta,constants,g,0,100000,0);
    for(int j=0;j<6;++j){out[j]=q.sites[j];out[6+j]=q.ledger_energy[j];}out[12]=q.dep;out[13]=q.steps;out[14]=q.failed?1.:0.;
}
