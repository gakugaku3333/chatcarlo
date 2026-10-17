#ifndef P5K_DRIVER_H
#define P5K_DRIVER_H
#include "p5j_kernel.h"
#include "p5b_wrap.h"
// Only Q allocates these in global memory. F keeps one in its lane's local storage.
struct P5kSlot {
    P5jState q; RngState rng; P5bLocal L; long long nspec_work[640];
    int i,active,diag_used,trace_used; long long diag_overflow,trace_overflow;
};
P5_FN P5Output p5k_output(P5kSlot *s) {
    return P5Output{s->L.batches,s->L.fine_batches,s->L.totals,s->L.maxima,s->L.brems,
        s->L.violations,s->L.counts,s->nspec_work,&s->L.steps,s->L.maxdepth};
}
P5_FN P5jTrace p5k_trace(P5kSlot *s,double *trace,int capacity) {
    return P5jTrace{trace?trace+uint64_t(s->i)*capacity*10:0,capacity,s->trace_used,s->trace_overflow};
}
P5_FN void p5k_initialize(P5kSlot *s,int i,int n,const P5jTables *t,uint64_t seed,uint64_t start,
                         double *trace,int capacity) {
    s->active=0;s->q.phase=P5J_D;s->i=i;s->L=P5bLocal{};
    s->diag_used=0;s->diag_overflow=0;s->trace_used=0;s->trace_overflow=0;
    for(int j=0;j<640;++j)s->nspec_work[j]=0;
    if(i>=n)return;
    s->active=1;rng_init(&s->rng,seed,start+uint64_t(i));
    P5Output o=p5k_output(s);P5jTrace tr=p5k_trace(s,trace,capacity);
    p5j_init(&s->q,t,&s->rng,0,&o,trace?&tr:0);
    s->L.maxdepth=o.maxdepth;s->trace_used=tr.used;s->trace_overflow=tr.overflow;
}
// One snapshot job. It never adds its new phase to this round's lists.
P5_FN void p5k_work(P5kSlot *s,const P5jTables *t,int maxsteps,int K,double *trace,int capacity) {
    P5Output o=p5k_output(s);P5Diagnostic d={s->L.diagnostic,4,s->diag_used,s->diag_overflow};
    P5jTrace tr=p5k_trace(s,trace,capacity);int phase=s->q.phase;
    for(int k=0;k<(phase==P5J_C?K:1);++k) {
        p5j_transition(&s->q,t,&s->rng,maxsteps,&o,&d,trace?&tr:0);
        if(s->q.phase!=P5J_C)break;
    }
    s->L.maxdepth=o.maxdepth;s->diag_used=d.used;s->diag_overflow=d.overflow;
    s->trace_used=tr.used;s->trace_overflow=tr.overflow;
}
P5_FN void p5k_finalize(P5kSlot *s,const P5jTables *t,double *trace,int capacity,long long *meta) {
    s->L.calls=s->rng.calls;s->L.diag_used=s->diag_used<4?s->diag_used:4;s->L.nspec_index=-1;
    for(int j=0;j<t->nspec*10;++j)if(s->nspec_work[j])s->L.nspec_index=j;
    P5Output o=p5k_output(s);P5jTrace tr=p5k_trace(s,trace,capacity);
    p5j_record(trace?&tr:0,5,&s->q,&s->rng,&o);
    s->trace_used=tr.used;s->trace_overflow=tr.overflow;
    if(meta){meta[2*s->i]=tr.used;meta[2*s->i+1]=tr.overflow;}
}
P5_FN int p5k_phase(const P5kSlot *s) {return s->active?s->q.phase:P5J_D;}
P5_FN bool p5k_done(const P5kSlot *s) {return s->active && s->q.phase==P5J_D;}
// CPU classification is the serial equivalent of the GPU ballot/prefix construction.
P5_FN void p5k_classify(const P5kSlot *slots,int R,int *lists,int *counts) {
    for(int p=0;p<4;++p)counts[p]=0;
    for(int j=0;j<R;++j){int p=p5k_phase(slots+j);if(p<4)lists[p*R+counts[p]++]=j;}
}
P5_FN bool p5k_finished(const int *counts) {return !(counts[0]+counts[1]+counts[2]+counts[3]);}
#endif
