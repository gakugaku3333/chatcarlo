
#ifndef NAN
#define NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
//@@COMMON@@

#define P5J_TABLE_ARGS P5B_TABLE_ARGS,double *logs,unsigned long long *log_counts
#define P5J_TABLE_INIT \
 P5Tables base={constants,loget,ed,ts,ks,us,ys,rates,pes,water,thresholds, \
 ne,nt,nk,nu,nrates,ratecols,np,watercols,er,Z,NJ,dl,brems_e,brems_v,edges,cum, \
 bpar,delcm,delpos,nz,nbrems,nspec}; \
 P5jTables table;static_cast<P5Tables&>(table)=base;table.logs=logs;table.log_counts=log_counts;const P5jTables *t=&table
P5_FN void p5j_function(const P5jTables *t,int which,uint64_t seed,int j,
    const double *x,double *out,uint64_t *calls) {
    RngState s;rng_init(&s,seed,uint64_t(j));
    if(which==0)out[j]=p5_collision(x[j],t);
    else if(which==1)p5_hard(x[j],t,out+2*j,out+2*j+1);
    else if(which==2)p5j_coefficients(x[j],t,out+4*j);
    else if(which==3)p5j_table_rates(x[j],t,out+2*j,out+2*j+1);
    else if(which==4)out[j]=p5_sample_W(x[j],t,&s);
    else {double total,G1;p5j_table_rates(x[2*j],t,&total,&G1);
        out[j]=p5j_gs_mu_cached(x[2*j],x[2*j+1],t,total,G1,&s);}
    calls[j]=s.calls;
}
P5_FN void p5j_warp(const P5jTables *t,uint64_t seed,uint64_t h,bool valid,int maxsteps,P5bLocal *L,P5jTrace *tr=0) {
    P5jState q;RngState s;long long nspec_work[640];P5Output o;P5Diagnostic d;
    q.phase=P5J_D;
    if(valid) {
        *L=P5bLocal{};for(int j=0;j<640;++j)nspec_work[j]=0;
        o=P5Output{L->batches,L->fine_batches,L->totals,L->maxima,L->brems,
            L->violations,L->counts,nspec_work,&L->steps,0};
        d=P5Diagnostic{L->diagnostic,4,0,0};rng_init(&s,seed,h);
        p5j_init(&q,t,&s,0,&o,tr);
    }
    for(;;) {
        int snapshot=q.phase;
        unsigned done=__ballot_sync(0xffffffffu,snapshot==P5J_D);
        unsigned pm=__ballot_sync(0xffffffffu,snapshot==P5J_P);
        unsigned cm=__ballot_sync(0xffffffffu,snapshot==P5J_C);
        unsigned em=__ballot_sync(0xffffffffu,snapshot==P5J_E);
        unsigned hm=__ballot_sync(0xffffffffu,snapshot==P5J_H);
        if(done==0xffffffffu)break;
        unsigned lane=1u<<(threadIdx.x&31);
        if(pm&lane)p5j_transition(&q,t,&s,maxsteps,&o,&d,tr);
        if(cm&lane)p5j_transition(&q,t,&s,maxsteps,&o,&d,tr);
        if(em&lane)p5j_transition(&q,t,&s,maxsteps,&o,&d,tr);
        if(hm&lane)p5j_transition(&q,t,&s,maxsteps,&o,&d,tr);
    }
    if(valid) {
        L->calls=s.calls;L->maxdepth=o.maxdepth;L->diag_used=d.used<4?d.used:4;
        L->nspec_index=-1;
        for(int j=0;j<t->nspec*10;++j)if(nspec_work[j])L->nspec_index=j;
        p5j_record(tr,5,&q,&s,&o); // exactly one output-ready marker
    }
}
#ifndef P5B_MERGE_FAULT
#define P5B_MERGE_FAULT 0
#endif
extern "C" __global__ void p5j_per_history_kernel(P5J_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,P5B_HISTORY_ARGS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    P5J_TABLE_INIT;P5bLocal L;p5j_warp(t,seed,start+uint64_t(i),i<n,maxsteps,&L);
    if(i>=n)return;
    p5b_store(&L,i,P5B_HISTORY_PASS);
}
extern "C" __global__ void p5j_run_kernel(P5J_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,
    double *batches,double *fine_batches,double *totals,double *maxima,double *brems,
    unsigned long long *violations,unsigned long long *counts,unsigned long long *spectrum,
    unsigned long long *step_batches,int *maxdepth,unsigned long long *nan_count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    P5J_TABLE_INIT;P5bLocal L;p5j_warp(t,seed,start+uint64_t(i),i<n,maxsteps,&L);
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
extern "C" __global__ void p5b_fields_kernel(P5J_TABLE_ARGS,int n,double *out,int *dimensions) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;P5J_TABLE_INIT;
    out[i]=p5b_field(t,i);if(i==0)p5b_dimensions(t,dimensions);
}
extern "C" __global__ void p5b_functions_kernel(P5J_TABLE_ARGS,int which,uint64_t seed,
    const double *x,int n,double *out,uint64_t *calls) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;P5J_TABLE_INIT;
    p5j_function(t,which,seed,j,x,out,calls);
}
extern "C" __global__ void p5b_philox_kernel(const uint64_t *triples,int n,uint32_t *words,double *values) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)p5b_rng_words(triples,j,words,values);
}
extern "C" __global__ void p5b_rows_kernel(uint64_t start,int n,int *rows,uint64_t *histories) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n){rows[j]=p5b_row(j,n);histories[j]=start+uint64_t(j);}
}
extern "C" __global__ void p5j_audit_kernel(P5J_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,P5B_HISTORY_ARGS,double *trace,int capacity,long long *meta) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    P5J_TABLE_INIT;P5bLocal L;
    P5jTrace tr={i<n?trace+uint64_t(i)*capacity*10:0,capacity,0,0};
    p5j_warp(t,seed,start+uint64_t(i),i<n,maxsteps,&L,i<n?&tr:0);
    if(i>=n)return;
    p5b_store(&L,i,P5B_HISTORY_PASS);meta[2*i]=tr.used;meta[2*i+1]=tr.overflow;
}

extern "C" __global__ void p5j_initialize_kernel(P5J_TABLE_ARGS) {
 P5J_TABLE_INIT;if(blockIdx.x==0 && threadIdx.x==0)p5j_initialize(&table);
}
extern "C" __global__ void p5j_gate0_kernel(P5J_TABLE_ARGS,int which,const double *x,int n,double *out) {
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;P5J_TABLE_INIT;
 p5j_compare(t,which,x[2*j],x[2*j+1],j,out+40*j);
}
extern "C" __global__ void p5j_diagnostic_kernel(P5J_TABLE_ARGS,double *out) {
 P5J_TABLE_INIT;if(blockIdx.x || threadIdx.x)return;p5j_auxiliary(t,out);
}

extern "C" __global__ void p5j_initialization_diagnostic_kernel(P5J_TABLE_ARGS,double *out) {
 P5J_TABLE_INIT;if(blockIdx.x || threadIdx.x)return;p5j_initialization_records(t,out);
}

extern "C" __global__ void p5j_contract_kernel(P5J_TABLE_ARGS,double *out) {
 P5J_TABLE_INIT;if(blockIdx.x || threadIdx.x)return;
    double sources[4]={NAN,.1,0.,-0.};
    for(int j=0;j<4;++j) {
        for(int k=0;k<40;++k)out[j*40+k]=0.;
        out[j*40]=p5_collision(.2,t);
        out[j*40+20]=p5j_collision(.2,t,sources[j],log(.2));
        // Signed zero must not reuse; payload NaNs must not reuse.
        double arg=j==2?-0.:j==3?0.:.2;
        out[j*40+1]=log(arg);
        out[j*40+21]=p5j_log(arg,sources[j],123.,t,5);
    }
}
