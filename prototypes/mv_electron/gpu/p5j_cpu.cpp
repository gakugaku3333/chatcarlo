#include "p5j_kernel.h"
static P5jTables configured;
extern "C" void p5j_configure(const P5Tables *base,double *logs,unsigned long long *counts,int initialize) {
 static_cast<P5Tables&>(configured)=*base;configured.logs=logs;configured.log_counts=counts;
 if(initialize)p5j_initialize(&configured);
}
// CPU-only entry points; the common history also targets CUDA C++.
extern "C" void p5a_deterministic(int which,const double *x,int n,const P5Tables *base,double *out) {
    const P5jTables *t=&configured;
    for(int j=0;j<n;++j) {
        if(which==0)out[j]=p5_collision(x[j],t);
        else if(which==1)p5_hard(x[j],t,out+2*j,out+2*j+1);
        else if(which==2)p5j_coefficients(x[j],t,out+4*j);
        else if(which==3)p5j_table_rates(x[j],t,out+2*j,out+2*j+1);
    }
}
extern "C" void p5a_random(int which,uint64_t seed,const double *x,int n,const P5Tables *base,double *out,double *next,uint64_t *calls) {
    const P5jTables *t=&configured;
    RngState s;rng_init(&s,seed,0);
    for(int j=0;j<n;++j) {
        if(which==0)out[j]=p5_sample_W(x[j],t,&s);
        else {
            double total,G1;p5j_table_rates(x[2*j],t,&total,&G1);
            out[j]=p5j_gs_mu_cached(x[2*j],x[2*j+1],t,total,G1,&s);
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
extern "C" void p5a_run(const P5Tables *base,uint64_t seed,uint64_t start,int n,int history_rng,int maxsteps,
    double *batches,double *fine,double *totals,double *maxima,long long *violations,long long *counts,
    long long *nspec,long long *steps,double *brems,long long *maxdepth,
    double *diagnostic,int diag_capacity,long long *diag_meta) {
    const P5jTables *t=&configured;
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
        p5j_history(t,&rng,b,maxsteps,&o,diag_capacity?&d:0);
    }
    *maxdepth=o.maxdepth;diag_meta[0]=d.used;diag_meta[1]=d.overflow;
}

extern "C" void p5j_trace(const P5Tables *base,uint64_t seed,uint64_t h,int maxsteps,double *trace,int capacity,long long *meta) {
    const P5jTables *t=&configured;
    double coarse[150]={0},fine[600]={0},totals[6]={0},maxima[3]={0},brems[10]={0};
    long long violations[3]={0},counts[16]={0},nspec[640]={0},steps=0;
    P5Output o={coarse,fine,totals,maxima,brems,violations,counts,nspec,&steps,0};
    RngState s;rng_init(&s,seed,h);P5jTrace tr={trace,capacity,0,0};
    p5j_history(t,&s,0,maxsteps,&o,0,&tr);meta[0]=tr.used;meta[1]=tr.overflow;
}

extern "C" void p5j_gate0(int which,const double *x,int n,double *out) {
    for(int j=0;j<n;++j)p5j_compare(&configured,which,x[2*j],x[2*j+1],j,out+40*j);
}
extern "C" void p5j_diagnostic(double *out) {p5j_auxiliary(&configured,out);}

extern "C" void p5j_initialization_diagnostic(double *out) {p5j_initialization_records(&configured,out);}

extern "C" void p5j_contract(double *out) {
    double sources[4]={NAN,.1,0.,-0.};
    for(int j=0;j<4;++j) {
        for(int k=0;k<40;++k)out[j*40+k]=0.;
        out[j*40]=p5_collision(.2,&configured);
        out[j*40+20]=p5j_collision(.2,&configured,sources[j],log(.2));
        // Signed zero must not reuse; payload NaNs must not reuse.
        double arg=j==2?-0.:j==3?0.:.2;
        out[j*40+1]=log(arg);
        out[j*40+21]=p5j_log(arg,sources[j],123.,&configured,5);
    }
}
