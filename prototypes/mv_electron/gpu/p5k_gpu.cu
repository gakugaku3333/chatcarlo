// P5j's translation unit precedes this file, unchanged.
P5_FN void p5k_gpu_classify(P5kSlot *slots,int R,int *lists,int *counts,int *prefix) {
    int tid=threadIdx.x,lane=tid&31,warp=tid/32;
    // Prefixes are ordered by slot, including R=512's second pass.
    for(int p=0;p<4;++p) {
        for(int base=0;base<R;base+=256) {
            int j=base+tid;unsigned mask=__ballot_sync(0xffffffffu,j<R && p5k_phase(slots+j)==p);
            int w=base/32+warp;
            if(lane==0)prefix[p*16+w]=__popc(mask);
        }
    }
    __syncthreads();
    if(tid==0)for(int p=0;p<4;++p) {
        int sum=0;for(int w=0;w<R/32;++w){int c=prefix[p*16+w];prefix[p*16+w]=sum;sum+=c;}counts[p]=sum;
    }
    __syncthreads();
    for(int p=0;p<4;++p)for(int base=0;base<R;base+=256) {
        int j=base+tid;unsigned mask=__ballot_sync(0xffffffffu,j<R && p5k_phase(slots+j)==p);
        if(j<R && p5k_phase(slots+j)==p)
            lists[p*R+prefix[p*16+base/32+warp]+__popc(mask&((1u<<lane)-1))]=j;
    }
    __syncthreads();
}
#define P5K_RUN_ARGS double *rb,double *rf,double *rt,double *rm,double *re, \
 unsigned long long *rv,unsigned long long *rc,unsigned long long *rn, \
 unsigned long long *rs,int *rd,unsigned long long *rnan
#define P5K_RUN_PASS rb,rf,rt,rm,re,rv,rc,rn,rs,rd,rnan
P5_FN void p5k_emit(P5kSlot *s,const P5jTables *t,int n,int history,
 P5B_HISTORY_ARGS,P5K_RUN_ARGS,double *trace,int capacity,long long *meta) {
    p5k_finalize(s,t,trace,capacity,meta);P5bLocal &L=s->L;int i=s->i;
    if(history){p5b_store(&L,i,P5B_HISTORY_PASS);return;}
    int row=p5b_row(i,n),drop=-1;
#if P5B_MERGE_FAULT == 1
    for(int j=0;j<150;++j)if(L.batches[j]!=0.)drop=j;
#endif
    for(int j=0;j<150;++j)if(j!=drop)atomicAdd(rb+row*150+j,L.batches[j]);
    for(int j=0;j<600;++j)atomicAdd(rf+row*600+j,L.fine_batches[j]);
    for(int j=0;j<6;++j)atomicAdd(rt+j,L.totals[j]);
    for(int j=0;j<10;++j)atomicAdd(re+j,L.brems[j]);
    for(int j=0;j<3;++j){
        if(!p5b_nan(L.maxima[j]))atomicMax(reinterpret_cast<unsigned long long*>(rm+j),
            static_cast<unsigned long long>(__double_as_longlong(L.maxima[j])));
        atomicAdd(rv+j,static_cast<unsigned long long>(L.violations[j]));
    }
    for(int j=0;j<16;++j)atomicAdd(rc+j,static_cast<unsigned long long>(L.counts[j]));
    atomicAdd(rn+L.nspec_index,1ULL);atomicAdd(rs+row,static_cast<unsigned long long>(L.steps));
    atomicMax(rd,L.maxdepth);atomicAdd(rnan,static_cast<unsigned long long>(p5b_nans(&L)));
}
#define P5K_DRIVER_ARGS P5J_TABLE_ARGS,uint64_t seed,uint64_t start,int n,int maxsteps, \
 int variant,P5kSlot *arena,int *next,int history,P5B_HISTORY_ARGS,P5K_RUN_ARGS, \
 double *trace,int capacity,long long *meta
#define P5K_EMIT p5k_emit(s,t,n,history,P5B_HISTORY_PASS,P5K_RUN_PASS,trace,capacity,meta)
// variant is uniform. F never accesses arena and keeps state inside its lane.
extern "C" __global__ void p5k_kernel(P5K_DRIVER_ARGS) {
    P5J_TABLE_INIT;
    __shared__ int lists[4*512],counts_phase[4],prefix[4*16];
    if(variant==0) {
        P5kSlot local;P5kSlot *s=&local;
        p5k_initialize(s,atomicAdd(next,1),n,t,seed,start,trace,capacity);
        for(;;) {
            int snapshot=p5k_phase(s);
            unsigned done=__ballot_sync(0xffffffffu,snapshot==P5J_D);
            unsigned masks[4];for(int p=0;p<4;++p)masks[p]=__ballot_sync(0xffffffffu,snapshot==p);
            if(done==0xffffffffu)break;
            unsigned lane=1u<<(threadIdx.x&31);
            for(int p=0;p<4;++p)if(masks[p]&lane) {
                p5k_work(s,t,maxsteps,1,trace,capacity);
                if(p5k_done(s)){P5K_EMIT;p5k_initialize(s,atomicAdd(next,1),n,t,seed,start,trace,capacity);}
            }
        }
        return;
    }
    int R=variant==4?512:256,K=variant==2?4:variant==3?8:1;
    P5kSlot *slots=arena+blockIdx.x*R;
    for(int j=threadIdx.x;j<R;j+=blockDim.x)p5k_initialize(slots+j,atomicAdd(next,1),n,t,seed,start,trace,capacity);
    __syncthreads();
    for(;;) {
        p5k_gpu_classify(slots,R,lists,counts_phase,prefix);
        if(p5k_finished(counts_phase))break;
        for(int p=0;p<4;++p)for(int k=threadIdx.x;k<counts_phase[p];k+=blockDim.x) {
            P5kSlot *s=slots+lists[p*R+k];p5k_work(s,t,maxsteps,K,trace,capacity);
            if(p5k_done(s)){P5K_EMIT;p5k_initialize(s,atomicAdd(next,1),n,t,seed,start,trace,capacity);}
        }
        __syncthreads();
    }
}
// Synthetic transitions exercise the actual GPU classification and work assignment.
extern "C" __global__ void p5k_schedule_kernel(P5kSlot *arena,int R,int n,uint64_t start,int *next,
 int *visits,int *rows,uint64_t *histories,int length,int phase,int *ordered) {
    if(R==0) {
        int i=atomicAdd(next,1);
        for(;;) {
            int snapshot=i<n?i%4:P5J_D;
            unsigned done=__ballot_sync(0xffffffffu,snapshot==P5J_D);
            unsigned mask[4];for(int p=0;p<4;++p)mask[p]=__ballot_sync(0xffffffffu,snapshot==p);
            if(done==0xffffffffu)break;
            for(int p=0;p<4;++p)if(mask[p]&(1u<<(threadIdx.x&31))) {
                atomicAdd(visits+i,1);rows[i]=p5b_row(i,n);histories[i]=start+uint64_t(i);i=atomicAdd(next,1);
            }
        }
        return;
    }
    __shared__ int lists[4*512],counts[4],prefix[4*16];P5kSlot *slots=arena+blockIdx.x*R;
    for(int j=threadIdx.x;j<R;j+=blockDim.x) {
        P5kSlot *s=slots+j;s->i=length>=0?j:atomicAdd(next,1);
        s->active=s->i<(length>=0?length:n);s->q.phase=length>=0?phase:s->i%4;
    }
    __syncthreads();
    for(;;) {
        p5k_gpu_classify(slots,R,lists,counts,prefix);
        if(length>=0) {
            for(int k=threadIdx.x;k<counts[phase];k+=blockDim.x) {
                int j=lists[phase*R+k];ordered[k]=slots[j].q.phase==phase?j:-2;
                slots[j].q.phase=P5J_D;
            }
            if(threadIdx.x==0)ordered[R]=counts[phase];return;
        }
        if(p5k_finished(counts))break;
        for(int p=0;p<4;++p)for(int k=threadIdx.x;k<counts[p];k+=blockDim.x) {
            P5kSlot *s=slots+lists[p*R+k];int i=s->i;
            atomicAdd(visits+i,1);rows[i]=p5b_row(i,n);histories[i]=start+uint64_t(i);
            s->i=atomicAdd(next,1);s->active=s->i<n;s->q.phase=s->i%4;
        }
        __syncthreads();
    }
}
extern "C" __global__ void p5k_size_kernel(int *bytes) {if(!threadIdx.x && !blockIdx.x)*bytes=sizeof(P5kSlot);}
// Isolated replay for reference timing, never added up as a wall-time decomposition.
// Prepare leaves completed local outputs resident; stages reuse exactly that sample.
extern "C" __global__ void p5k_stage_kernel(P5K_DRIVER_ARGS,int stage) {
    P5J_TABLE_INIT;int R=variant==4?512:256;P5kSlot *slots=arena+blockIdx.x*R;
    __shared__ int lists[4*512],phases[4],prefix[4*16];
    if(stage==1) {p5k_gpu_classify(slots,R,lists,phases,prefix);return;}
    for(int j=threadIdx.x;j<R;j+=blockDim.x) {
        P5kSlot *s=slots+j;
        if(stage==0)p5k_initialize(s,blockIdx.x*R+j,n,t,seed,start,0,0);
        else if(stage==2) {
            // Preserve sample ownership; only replay the output operation.
            P5K_EMIT;
        } else if(stage==3) {
            while(s->active && s->q.phase!=P5J_D)p5k_work(s,t,maxsteps,1,0,0);
        }
    }
}
