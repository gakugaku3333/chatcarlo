// P5f static one-thread-per-history entries. Assembled after derived common code.
extern "C" __global__ void p5f_per_history_kernel(P5B_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,int p5f_base,
    P5B_HISTORY_ARGS,unsigned long long *reads) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    P5B_TABLE_INIT;P5fLocal L;p5b_history(t,seed,start+uint64_t(i),maxsteps,p5f_base,&L);
    p5b_store(&L,i,P5B_HISTORY_PASS);reads[i]=L.reads;
}
extern "C" __global__ void p5f_run_kernel(P5B_TABLE_ARGS,
    uint64_t seed,uint64_t start,int n,int maxsteps,int p5f_base,
    double *batches,double *fine_batches,double *totals,double *maxima,double *brems,
    unsigned long long *violations,unsigned long long *counts,unsigned long long *spectrum,
    unsigned long long *step_batches,int *maxdepth,unsigned long long *nan_count,
    unsigned long long *calls,unsigned long long *reads) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    P5B_TABLE_INIT;P5fLocal L;p5b_history(t,seed,start+uint64_t(i),maxsteps,p5f_base,&L);
    int row=p5b_row(i,n);
#if P5F_WRITE
    for(int j=0;j<2;++j)atomicAdd(batches+row*2+j,L.batches[j]);
#else
    for(int j=0;j<150;++j)atomicAdd(batches+row*150+j,L.batches[j]);
    for(int j=0;j<600;++j)atomicAdd(fine_batches+row*600+j,L.fine_batches[j]);
#endif
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
    atomicAdd(calls,L.calls);atomicAdd(reads,L.reads);
}
