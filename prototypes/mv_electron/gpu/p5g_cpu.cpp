#include "p5g_kernel.h"
// CPU-only entry points; the common history also targets CUDA C++.
extern "C" void p5a_deterministic(int which,const double *x,int n,const P5Tables *t,double *out) {
    for(int j=0;j<n;++j) {
        if(which==0)out[j]=p5_collision(x[j],t);
        else if(which==1)p5_hard(x[j],t,out+2*j,out+2*j+1);
        else if(which==2)p5_coefficients(x[j],t,out+4*j);
        else if(which==3)p5_table_rates(x[j],t,out+2*j,out+2*j+1);
    }
}
extern "C" void p5a_random(int which,uint64_t seed,const double *x,int n,const P5Tables *t,double *out,double *next,uint64_t *calls) {
    RngState s;rng_init(&s,seed,0);
    for(int j=0;j<n;++j) {
        if(which==0)out[j]=p5_sample_W(x[j],t,&s);
        else {
            double total,G1;p5_table_rates(x[2*j],t,&total,&G1);
            out[j]=p5_gs_mu_cached(x[2*j],x[2*j+1],t,total,G1,&s);
        }
    }
    *calls=s.calls;*next=rng_uniform(&s);
}
extern "C" void p5a_score(int which,const double *x,int n,int nb,double dz,double *out) {
    for(int j=0;j<n;++j) {
        if(which==0)p5_score_segment(out+j*nb,nb,x[3*j],x[3*j+1],x[3*j+2],dz);
        else p5_local(out+j*nb,nb,x[2*j],x[2*j+1],dz);
    }
}
extern "C" void p5a_rng(uint64_t seed,uint64_t h,int n,double *out) {
    RngState s;rng_init(&s,seed,h);for(int j=0;j<n;++j)out[j]=rng_uniform(&s);
}
extern "C" void p5a_run(const P5Tables *t,uint64_t seed,uint64_t start,int n,int history_rng,int maxsteps,
    double *batches,double *fine,double *totals,double *maxima,long long *violations,long long *counts,
    long long *nspec,long long *steps,double *brems,long long *maxdepth,
    double *diagnostic,int diag_capacity,long long *diag_meta) {
    static_assert(sizeof(long long)==8,"64-bit signed counter required");
    P5Output o={batches,fine,totals,maxima,brems,violations,counts,nspec,steps,0};
    P5Diagnostic d={diagnostic,diag_capacity,0,0};RngState rng;rng_init(&rng,seed,0);
    for(int h=0;h<n;++h) {
#ifdef P5_RNG_MT
        if(history_rng)rng_init(&rng,uint32_t(seed+start+uint64_t(h)),0);
#else
        (void)history_rng;rng_init(&rng,seed,start+uint64_t(h));
#endif
        int b=h*100/n;if(b>99)b=99;
        p5g_history(t,&rng,b,maxsteps,&o,diag_capacity?&d:0);
    }
    *maxdepth=o.maxdepth;diag_meta[0]=d.used;diag_meta[1]=d.overflow;
}

extern "C" void p5g_trace(const P5Tables *t,uint64_t seed,uint64_t h,int maxsteps,double *trace,int capacity,long long *meta) {
    double coarse[150]={0},fine[600]={0},totals[6]={0},maxima[3]={0},brems[10]={0};
    long long violations[3]={0},counts[16]={0},nspec[640]={0},steps=0;
    P5Output o={coarse,fine,totals,maxima,brems,violations,counts,nspec,&steps,0};
    RngState s;rng_init(&s,seed,h);P5gTrace tr={trace,capacity,0,0};
    p5g_history(t,&s,0,maxsteps,&o,0,&tr);meta[0]=tr.used;meta[1]=tr.overflow;
}
