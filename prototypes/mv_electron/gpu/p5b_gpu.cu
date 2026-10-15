#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
//@@COMMON@@
#ifndef P5B_MERGE_FAULT
#define P5B_MERGE_FAULT 0
#endif
extern "C" __global__ void p5b_per_history_kernel(P5B_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,P5B_HISTORY_ARGS) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    P5B_TABLE_INIT;P5bLocal L;p5b_history(t,seed,start+uint64_t(i),maxsteps,&L);
    p5b_store(&L,i,P5B_HISTORY_PASS);
}
extern "C" __global__ void p5b_run_kernel(P5B_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,
    double *batches,double *fine_batches,double *totals,double *maxima,double *brems,
    unsigned long long *violations,unsigned long long *counts,unsigned long long *spectrum,
    unsigned long long *step_batches,int *maxdepth,unsigned long long *nan_count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    P5B_TABLE_INIT;P5bLocal L;p5b_history(t,seed,start+uint64_t(i),maxsteps,&L);
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
extern "C" __global__ void p5b_fields_kernel(P5B_TABLE_ARGS,int n,double *out,int *dimensions) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;P5B_TABLE_INIT;
    out[i]=p5b_field(t,i);if(i==0)p5b_dimensions(t,dimensions);
}
extern "C" __global__ void p5b_functions_kernel(P5B_TABLE_ARGS,int which,uint64_t seed,
    const double *x,int n,double *out,uint64_t *calls) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;P5B_TABLE_INIT;
    p5b_function(t,which,seed,j,x,out,calls);
}
extern "C" __global__ void p5b_philox_kernel(const uint64_t *triples,int n,uint32_t *words,double *values) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n)p5b_rng_words(triples,j,words,values);
}
extern "C" __global__ void p5b_rows_kernel(uint64_t start,int n,int *rows,uint64_t *histories) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;if(j<n){rows[j]=p5b_row(j,n);histories[j]=start+uint64_t(j);}
}
