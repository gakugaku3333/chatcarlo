#ifndef NAN
#define NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
//@@COMMON@@
//@@PROBE@@
// The host packs the geometry ABI with device addresses, never host pointers.
extern "C" __global__ void p9b_init(double *data,const long long *meta,const double *constants,int nm){if(blockIdx.x||threadIdx.x)return;for(int mid=0;mid<nm;++mid){P8gMaterial m=p8g_material(data,meta+136*mid,constants);p8g_initialize(m);}}
extern "C" __global__ void p9b_probe_kernel(const double *data,const long long *meta,const double *constants,const P9bGeometry *g,const double *input,const double *fixed,int mode,uint64_t seed,uint64_t start,int n,double *out){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<n)p9b_probe_one(data,meta,constants,g,input+23*h,fixed+3*h,mode,seed,start+h,out+74*h);}
extern "C" __global__ void p9b_kernel(const double *data,const long long *meta,const double *constants,const P9bGeometry *g,uint64_t seed,uint64_t start,int n,int *next,int history,int maxsteps,const double *injected,double *hd,long long *hi,double *groups,unsigned long long *group_ints,long long *visits,double *field_groups,double *field_histories,unsigned long long *adds,unsigned long long *sites,double *ledger,unsigned long long *lateral){
    P9bState local;P9bState *q=&local;p9b_slot_init(q,g);q->field_additions=adds;
    int h=atomicAdd(next,1);
    if(h<n){q->field_additions=adds+(h*100/n)*3420;P5State a;if(injected)for(int j=0;j<23;++j)a.a[j]=injected[23*h+j];p9b_begin(q,data,meta,constants,g,seed,start+h,0,injected?&a:0);}else q->phase=P8G2_D;
    for(;;){
        int snapshot=q->phase;unsigned done=__ballot_sync(0xffffffffu,snapshot==P8G2_D);
        unsigned masks[4];for(int p=0;p<4;++p)masks[p]=__ballot_sync(0xffffffffu,snapshot==p);
        if(done==0xffffffffu)break;unsigned lane=1u<<(threadIdx.x&31);
        for(int p=0;p<4;++p)if(masks[p]&lane){
            p9b_advance(q,data,meta,constants,g,0,maxsteps,0);
            if(q->phase==P8G2_D){
                atomicAdd(reinterpret_cast<unsigned long long*>(visits+h),1ULL);
                P9bRecord &r=q->record;int dw=p8g_dwidth(g),iw=p8g_iwidth(g),b=h*100/n,f=g->nb+g->nc+g->nf;
                if(history){for(int j=0;j<dw;++j)hd[h*dw+j]=r.d[j];for(int j=0;j<iw;++j)hi[h*iw+j]=r.i[j];}
                for(int j=0;j<f+21;++j){if((j>=f+6&&j<f+9)||(j>=f+19&&j<f+21))atomicMax(reinterpret_cast<unsigned long long*>(groups+b*dw+j),(unsigned long long)__double_as_longlong(r.d[j]));else atomicAdd(groups+b*dw+j,r.d[j]);}
                for(int j=0;j<iw;++j){if(j==77)atomicMax(group_ints+b*iw+j,(unsigned long long)r.i[j]);else if(j<75||j>=79)atomicAdd(group_ints+b*iw+j,(unsigned long long)r.i[j]);}
                if(g->scoring_enabled){for(int j=0;j<3420;++j){atomicAdd(field_groups+b*3420+j,q->field_scores[j]);if(history)field_histories[h*3420+j]=q->field_scores[j];}
                    for(int j=0;j<6;++j){atomicAdd(sites+j,(unsigned long long)q->sites[j]);atomicAdd(ledger+j,q->ledger_energy[j]);if(history){sites[6+6*h+j]=q->sites[j];ledger[6+6*h+j]=q->ledger_energy[j];}}}
                atomicAdd(lateral,(unsigned long long)q->lateral);
                h=atomicAdd(next,1);if(h<n){q->field_additions=adds+(h*100/n)*3420;P5State a;if(injected)for(int j=0;j<23;++j)a.a[j]=injected[23*h+j];p9b_begin(q,data,meta,constants,g,seed,start+h,0,injected?&a:0);}else q->phase=P8G2_D;
            }
        }
    }
}
//@@SCORE@@
extern "C" __global__ void p9b_score_probe_kernel(const P9bGeometry *g,int region,int site,const double *p0,const double *p1,double energy,double *field,double *depth,unsigned long long *adds,long long *sites,double *ledger){if(blockIdx.x||threadIdx.x)return;p9b_score_one(g,region,site,p0,p1,energy,field,depth,adds,sites,ledger);}
//@@INJECTED@@
extern "C" __global__ void p9b_injected_kernel(const double *data,const long long *meta,const double *constants,const P9bGeometry *g,const double *input,uint64_t seed,double *out,unsigned long long *adds){if(blockIdx.x||threadIdx.x)return;p9b_injected_one(data,meta,constants,g,input,seed,out,adds);}
