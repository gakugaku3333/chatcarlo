// P5j's unchanged GPU translation unit and p6_kernel.h precede this file.
#define P6_RUN_ARGS double *rb,double *rf,double *rt,double *rm,double *re, \
 unsigned long long *rv,unsigned long long *rc,unsigned long long *rn, \
 unsigned long long *rs,int *rd,unsigned long long *rnan
#define P6_RUN_PASS rb,rf,rt,rm,re,rv,rc,rn,rs,rd,rnan
P5_FN void p6_emit(P6Slot *s,const P5jTables *t,int n,int history,
 P5B_HISTORY_ARGS,P6_RUN_ARGS,double *hr,long long *hl,double *hs,
 double *rr,unsigned long long *rl,double *trace,int capacity,long long *meta) {
    p6_finalize(s,t,trace,capacity,meta);P5bLocal &L=s->L;int i=s->i;
    if(history) {
        p5b_store(&L,i,P5B_HISTORY_PASS);
        for(int j=0;j<P6_BINS;++j){hr[uint64_t(i)*P6_BINS+j]=s->regions[j];if(!isfinite(s->regions[j]))++nan_count[i];}
        for(int j=0;j<15;++j)hl[i*15+j]=s->ledger[j];
        for(int j=0;j<7;++j)hs[i*7+j]=s->sources[j];return;
    }
    int row=p5b_row(i,n);
    for(int j=0;j<150;++j)atomicAdd(rb+row*150+j,L.batches[j]);
    for(int j=0;j<600;++j)atomicAdd(rf+row*600+j,L.fine_batches[j]);
    for(int j=0;j<6;++j)atomicAdd(rt+j,L.totals[j]);
    for(int j=0;j<10;++j)atomicAdd(re+j,L.brems[j]);
    for(int j=0;j<3;++j) {
        if(isfinite(L.maxima[j]))atomicMax(reinterpret_cast<unsigned long long*>(rm+j),static_cast<unsigned long long>(__double_as_longlong(L.maxima[j])));
        atomicAdd(rv+j,static_cast<unsigned long long>(L.violations[j]));
    }
    for(int j=0;j<16;++j)atomicAdd(rc+j,static_cast<unsigned long long>(L.counts[j]));
    atomicAdd(rn+L.nspec_index,1ULL);atomicAdd(rs+row,static_cast<unsigned long long>(L.steps));
    atomicMax(rd,L.maxdepth);atomicAdd(rnan,static_cast<unsigned long long>(p5b_nans(&L)));
    for(int j=0;j<P6_BINS;++j){atomicAdd(rr+row*P6_BINS+j,s->regions[j]);if(!isfinite(s->regions[j]))atomicAdd(rnan,1ULL);}
    for(int j=0;j<15;++j)atomicAdd(rl+j,static_cast<unsigned long long>(s->ledger[j]));
}
extern "C" __global__ void p6_kernel(P5J_TABLE_ARGS,const P6Geometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,
 int field,int enabled,int *next,int history,P5B_HISTORY_ARGS,P6_RUN_ARGS,
 double *hr,long long *hl,double *hs,double *rr,unsigned long long *rl,double *trace,int capacity,long long *meta,unsigned long long *additions) {
    P5J_TABLE_INIT;P6Slot local={};P6Slot *s=&local;
    p6_initialize(s,atomicAdd(next,1),n,t,seed,start,field,trace,capacity);
    for(;;) {
        int snapshot=s->active?s->q.base.phase:P5J_D;
        unsigned done=__ballot_sync(0xffffffffu,snapshot==P5J_D);
        unsigned masks[4];for(int p=0;p<4;++p)masks[p]=__ballot_sync(0xffffffffu,snapshot==p);
        if(done==0xffffffffu)break;
        unsigned lane=1u<<(threadIdx.x&31);
        for(int p=0;p<4;++p)if(masks[p]&lane) {
            p6_work(s,t,g,enabled,maxsteps,trace,capacity,additions?additions+p5b_row(s->i,n)*P6_BINS:0);
            if(s->active && s->q.base.phase==P5J_D) {
                p6_emit(s,t,n,history,P5B_HISTORY_PASS,P6_RUN_PASS,hr,hl,hs,rr,rl,trace,capacity,meta);
                p6_initialize(s,atomicAdd(next,1),n,t,seed,start,field,trace,capacity);
            }
        }
    }
}
extern "C" __global__ void p6_score_kernel(const P6Geometry *g,const double *records,int n,double *scores,double *sums) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const double *v=records+9*i;P6Sink sink={scores+uint64_t(i)*P6_BINS,sums+15*i,g,1};
    p6_deposit(&sink,0,3,int(v[1]),v+2,v+5,v[8]);
}
