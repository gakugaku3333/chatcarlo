#ifndef P5B_WRAP_H
#define P5B_WRAP_H
// NVRTC does not provide the host <cmath> INFINITY macro.
#if defined(P5_GPU) && !defined(INFINITY)
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
#include "p5_kernel.h"

#ifdef P5_GPU
typedef unsigned long long P5bCount;
#else
typedef long long P5bCount;
#endif

// The argument order is the P5Tables declaration order, never a struct by value.
#define P5B_TABLE_ARGS \
 const double *constants,const double *loget,const double *ed,const double *ts, \
 const double *ks,const double *us,const double *ys,const double *rates, \
 const double *pes,const double *water,const double *thresholds, \
 int ne,int nt,int nk,int nu,int nrates,int ratecols,int np,int watercols, \
 const double *er,const double *Z,const double *NJ,const double *dl, \
 const double *brems_e,const double *brems_v,const double *edges,const double *cum, \
 double bpar,double delcm,double delpos,int nz,int nbrems,int nspec
#define P5B_TABLE_INIT \
 P5Tables table={constants,loget,ed,ts,ks,us,ys,rates,pes,water,thresholds, \
 ne,nt,nk,nu,nrates,ratecols,np,watercols,er,Z,NJ,dl,brems_e,brems_v,edges,cum, \
 bpar,delcm,delpos,nz,nbrems,nspec}; const P5Tables *t=&table
#define P5B_HISTORY_ARGS \
 double *batches,double *fine_batches,double *totals,double *maxima,double *brems, \
 P5bCount *violations,P5bCount *counts,P5bCount *nspec_index,P5bCount *steps, \
 int *maxdepth,P5bCount *calls,double *diagnostic,P5bCount *diag_used,P5bCount *nan_count
#define P5B_HISTORY_PASS \
 batches,fine_batches,totals,maxima,brems,violations,counts,nspec_index,steps, \
 maxdepth,calls,diagnostic,diag_used,nan_count

struct P5bLocal {
    double batches[150],fine_batches[600],totals[6],maxima[3],brems[10];
    long long violations[3],counts[16],steps;
    int nspec_index,maxdepth,diag_used;
    uint64_t calls;
    double diagnostic[28];
};
P5_FN int p5b_row(int h,int n) {return int(h*100/n);}
P5_FN void p5b_history(const P5Tables *t,uint64_t seed,uint64_t global_h,int maxsteps,P5bLocal *L) {
    *L=P5bLocal{};
    long long nspec_work[640]={0};
    P5Output o={L->batches,L->fine_batches,L->totals,L->maxima,L->brems,
        L->violations,L->counts,nspec_work,&L->steps,0};
    P5Diagnostic d={L->diagnostic,4,0,0};
    RngState s;rng_init(&s,seed,global_h);
    history(t,&s,0,maxsteps,&o,&d);
    L->calls=s.calls;L->maxdepth=o.maxdepth;L->diag_used=d.used<4?d.used:4;
    L->nspec_index=-1;
    for(int j=0;j<t->nspec*10;++j)if(nspec_work[j])L->nspec_index=j;
}
P5_FN bool p5b_nan(double v) {return v!=v;}
P5_FN int p5b_nans(const P5bLocal *L) {
    int n=0;
    for(int j=0;j<150;++j)n+=p5b_nan(L->batches[j]);
    for(int j=0;j<600;++j)n+=p5b_nan(L->fine_batches[j]);
    for(int j=0;j<6;++j)n+=p5b_nan(L->totals[j]);
    for(int j=0;j<3;++j)n+=p5b_nan(L->maxima[j]);
    for(int j=0;j<10;++j)n+=p5b_nan(L->brems[j]);
    for(int j=0;j<L->diag_used*7;++j)n+=p5b_nan(L->diagnostic[j]);
    return n;
}
P5_FN void p5b_store(const P5bLocal *L,int i,P5B_HISTORY_ARGS) {
    // The approved n limit protects h*100; array strides can exceed int32.
    for(int j=0;j<150;++j)batches[uint64_t(i)*150+j]=L->batches[j];
    for(int j=0;j<600;++j)fine_batches[uint64_t(i)*600+j]=L->fine_batches[j];
    for(int j=0;j<6;++j)totals[i*6+j]=L->totals[j];
    for(int j=0;j<3;++j){maxima[i*3+j]=L->maxima[j];violations[i*3+j]=L->violations[j];}
    for(int j=0;j<10;++j)brems[i*10+j]=L->brems[j];
    for(int j=0;j<16;++j)counts[i*16+j]=L->counts[j];
    for(int j=0;j<28;++j)diagnostic[i*28+j]=L->diagnostic[j];
    nspec_index[i]=L->nspec_index;steps[i]=L->steps;maxdepth[i]=L->maxdepth;
    calls[i]=L->calls;diag_used[i]=L->diag_used;nan_count[i]=p5b_nans(L);
}
// All nineteen pointer fields, followed by the three scalar fields.
P5_FN double p5b_field(const P5Tables *t,int i) {
    const double *p[19]={t->constants,t->loget,t->ed,t->ts,t->ks,t->us,t->ys,
        t->rates,t->pes,t->water,t->thresholds,t->er,t->Z,t->NJ,t->dl,
        t->brems_e,t->brems_v,t->edges,t->cum};
    int n[19]={6,t->ne,t->ne,t->nt,t->nk,t->nu,t->nt*t->nk*t->nu,
        t->nrates*t->ratecols,t->np,t->np*t->watercols,2,t->ne,t->nz,t->nz,36,
        t->nbrems,t->nbrems*5,t->nspec+1,t->nspec};
    for(int j=0;j<19;++j){if(i<n[j])return p[j][i];i-=n[j];}
    return i==0?t->bpar:(i==1?t->delcm:t->delpos);
}
P5_FN void p5b_dimensions(const P5Tables *t,int *out) {
    int v[11]={t->ne,t->nt,t->nk,t->nu,t->nrates,t->ratecols,t->np,t->watercols,
        t->nz,t->nbrems,t->nspec};
    for(int j=0;j<11;++j)out[j]=v[j];
}
P5_FN void p5b_function(const P5Tables *t,int which,uint64_t seed,int j,
    const double *x,double *out,uint64_t *calls) {
    RngState s;rng_init(&s,seed,uint64_t(j));
    if(which==0)out[j]=p5_collision(x[j],t);
    else if(which==1)p5_hard(x[j],t,out+2*j,out+2*j+1);
    else if(which==2)p5_coefficients(x[j],t,out+4*j);
    else if(which==3)p5_table_rates(x[j],t,out+2*j,out+2*j+1);
    else if(which==4)out[j]=p5_sample_W(x[j],t,&s);
    else {double total,G1;p5_table_rates(x[2*j],t,&total,&G1);
        out[j]=p5_gs_mu_cached(x[2*j],x[2*j+1],t,total,G1,&s);}
    calls[j]=s.calls;
}
P5_FN void p5b_rng_words(const uint64_t *triples,int j,uint32_t *words,double *values) {
    philox(triples[3*j],triples[3*j+1],triples[3*j+2],words+4*j);
    values[j]=uniform(words+4*j);
}
#endif
