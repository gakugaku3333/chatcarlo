#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
//@@COMMON@@
#define P5H_TABLE_ARGS P5B_TABLE_ARGS,const int *ranges,const double *lookup_meta,const int *pair_start,unsigned long long *lookup_counts
#define P5H_TABLE_INIT \
 P5Tables base={constants,loget,ed,ts,ks,us,ys,rates,pes,water,thresholds, \
 ne,nt,nk,nu,nrates,ratecols,np,watercols,er,Z,NJ,dl,brems_e,brems_v,edges,cum, \
 bpar,delcm,delpos,nz,nbrems,nspec}; \
 P5hTables table;static_cast<P5Tables&>(table)=base;table.ranges=ranges;table.meta=lookup_meta; \
 table.pair_start=pair_start;table.lookup_counts=lookup_counts;const P5hTables *t=&table

P5_FN void p5h_function(const P5hTables *t,int which,uint64_t seed,int j,
    const double *x,double *out,uint64_t *calls) {
    RngState s;rng_init(&s,seed,uint64_t(j));
    if(which==0)out[j]=p5h_collision(x[j],t);
    else if(which==1)p5_hard(x[j],t,out+2*j,out+2*j+1);
    else if(which==2)p5h_coefficients(x[j],t,out+4*j);
    else if(which==3)p5h_table_rates(x[j],t,out+2*j,out+2*j+1);
    else if(which==4)out[j]=p5_sample_W(x[j],t,&s);
    else {double total,G1;p5h_table_rates(x[2*j],t,&total,&G1);
        out[j]=p5h_gs_mu_cached(x[2*j],x[2*j+1],t,total,G1,&s);}
    calls[j]=s.calls;
}
P5_FN void p5h_warp(const P5hTables *t,uint64_t seed,uint64_t h,bool valid,int maxsteps,P5bLocal *L,P5hTrace *tr=0) {
    P5hState q;RngState s;long long nspec_work[640];P5Output o;P5Diagnostic d;
    q.phase=P5H_D;
    if(valid) {
        *L=P5bLocal{};for(int j=0;j<640;++j)nspec_work[j]=0;
        o=P5Output{L->batches,L->fine_batches,L->totals,L->maxima,L->brems,
            L->violations,L->counts,nspec_work,&L->steps,0};
        d=P5Diagnostic{L->diagnostic,4,0,0};rng_init(&s,seed,h);
        p5h_init(&q,t,&s,0,&o,tr);
    }
    for(;;) {
        int snapshot=q.phase;
        unsigned done=__ballot_sync(0xffffffffu,snapshot==P5H_D);
        unsigned pm=__ballot_sync(0xffffffffu,snapshot==P5H_P);
        unsigned cm=__ballot_sync(0xffffffffu,snapshot==P5H_C);
        unsigned em=__ballot_sync(0xffffffffu,snapshot==P5H_E);
        unsigned hm=__ballot_sync(0xffffffffu,snapshot==P5H_H);
        if(done==0xffffffffu)break;
        unsigned lane=1u<<(threadIdx.x&31);
        if(pm&lane)p5h_transition(&q,t,&s,maxsteps,&o,&d,tr);
        if(cm&lane)p5h_transition(&q,t,&s,maxsteps,&o,&d,tr);
        if(em&lane)p5h_transition(&q,t,&s,maxsteps,&o,&d,tr);
        if(hm&lane)p5h_transition(&q,t,&s,maxsteps,&o,&d,tr);
    }
    if(valid) {
        L->calls=s.calls;L->maxdepth=o.maxdepth;L->diag_used=d.used<4?d.used:4;
        L->nspec_index=-1;
        for(int j=0;j<t->nspec*10;++j)if(nspec_work[j])L->nspec_index=j;
        p5h_record(tr,5,&q,&s,&o); // exactly one output-ready marker
    }
}
#ifndef P5B_MERGE_FAULT
#define P5B_MERGE_FAULT 0
#endif
extern "C" __global__ void p5h_per_history_kernel(P5H_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,P5B_HISTORY_ARGS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    P5H_TABLE_INIT;P5bLocal L;p5h_warp(t,seed,start+uint64_t(i),i<n,maxsteps,&L);
    if(i>=n)return;
    p5b_store(&L,i,P5B_HISTORY_PASS);
}
extern "C" __global__ void p5h_run_kernel(P5H_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,
    double *batches,double *fine_batches,double *totals,double *maxima,double *brems,
    unsigned long long *violations,unsigned long long *counts,unsigned long long *spectrum,
    unsigned long long *step_batches,int *maxdepth,unsigned long long *nan_count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    P5H_TABLE_INIT;P5bLocal L;p5h_warp(t,seed,start+uint64_t(i),i<n,maxsteps,&L);
    if(i>=n)return;
    int row=p5b_row(i,n),drop=-1;
#if P5B_MERGE_FAULT == 1
    for(int j=0;j<150;++j)if(L.batches[j]!=0.)drop=j;
#endif
    for(int j=0;j<150;++j)if(j!=drop)atomicAdd(batches+row*150+j,L.batches[j]);
    for(int j=0;j<600;++j)atomicAdd(fine_batches+row*600+j,L.fine_batches[j]);
    for(int j=0;j<6;++j)atomicAdd(totals+j,L.totals[j]);
    for(int j=0;j<10;++j)atomicAdd(brems+j,L.brems[j]);
    for(int j=0;j<3;++j){
        if(!p5b_nan(L.maxima[j]))atomicMax(reinterpret_cast<unsigned long long*>(maxima+j),
            static_cast<unsigned long long>(__double_as_longlong(L.maxima[j])));
        atomicAdd(violations+j,static_cast<unsigned long long>(L.violations[j]));
    }
    for(int j=0;j<16;++j)atomicAdd(counts+j,static_cast<unsigned long long>(L.counts[j]));
    atomicAdd(spectrum+L.nspec_index,1ULL);
    atomicAdd(step_batches+row,static_cast<unsigned long long>(L.steps));
    atomicMax(maxdepth,L.maxdepth);atomicAdd(nan_count,static_cast<unsigned long long>(p5b_nans(&L)));
}
extern "C" __global__ void p5b_fields_kernel(P5H_TABLE_ARGS,int n,double *out,int *dimensions) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;P5H_TABLE_INIT;
    out[i]=p5b_field(t,i);if(i==0)p5b_dimensions(t,dimensions);
}
extern "C" __global__ void p5b_functions_kernel(P5H_TABLE_ARGS,int which,uint64_t seed,
    const double *x,int n,double *out,uint64_t *calls) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;P5H_TABLE_INIT;
    p5h_function(t,which,seed,j,x,out,calls);
}
extern "C" __global__ void p5b_philox_kernel(const uint64_t *triples,int n,uint32_t *words,double *values) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)p5b_rng_words(triples,j,words,values);
}
extern "C" __global__ void p5b_rows_kernel(uint64_t start,int n,int *rows,uint64_t *histories) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n){rows[j]=p5b_row(j,n);histories[j]=start+uint64_t(j);}
}
extern "C" __global__ void p5h_audit_kernel(P5H_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,P5B_HISTORY_ARGS,double *trace,int capacity,long long *meta) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    P5H_TABLE_INIT;P5bLocal L;
    P5hTrace tr={i<n?trace+uint64_t(i)*capacity*10:0,capacity,0,0};
    p5h_warp(t,seed,start+uint64_t(i),i<n,maxsteps,&L,i<n?&tr:0);
    if(i>=n)return;
    p5b_store(&L,i,P5B_HISTORY_PASS);meta[2*i]=tr.used;meta[2*i+1]=tr.overflow;
}

extern "C" __global__ void p5h_gate0_kernel(P5H_TABLE_ARGS,
 const double *a,int size,int logarithmic,int start,int map,int site,
 const double *x,int count,int narrow,long long *out) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=count+3*size)return;
 P5H_TABLE_INIT;
 int broken[2]={start,start};double meta[30];
 if(narrow) {
   for(int k=0;k<30;++k)meta[k]=lookup_meta[k];
   meta[map*5+3]=1;meta[map*5+4]=0;
   meta[map*5+2]=1/(meta[map*5+1]-meta[map*5]);table.meta=meta;table.ranges=broken;
 }
 double value;
 if(j<count)value=x[j];
 else {int k=(j-count)%3;value=p5h_value(a,(j-count)/3,1,logarithmic);
       if(k==1)value=nextafter(value,-INFINITY);if(k==2)value=nextafter(value,INFINITY);}
 out[2*j]=p5h_find(value,a,size,1,logarithmic,start,t,map,site);
 out[2*j+1]=p5h_original(value,a,size,1,logarithmic,start);
}
