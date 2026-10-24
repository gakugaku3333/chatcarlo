#ifndef NAN
#define NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
//@@COMMON@@
P5_FN void p8g2_gpu_begin(P8g2State *q,int h,int n,const double *data,const long long *meta,const double *constants,const P8gGeometry *g,uint64_t seed,uint64_t start,unsigned long long *hits,const double *injected){
    if(h>=n){q->phase=P8G2_D;return;}P5State a;
    if(injected)for(int j=0;j<23;++j)a.a[j]=injected[h*23+j];
    p8g2_begin(q,data,meta,constants,g,seed,start+uint64_t(h),hits,injected?&a:0);
}
extern "C" __global__ void p8g2_kernel(const double *data,const long long *meta,const double *constants,
    const double *geometry,const long long *mapping,int nl,int nb,int nc,int nf,int nspec,
    uint64_t seed,uint64_t start,int n,int *next,int history,int maxsteps,int reference,const double *injected,
    double *hd,long long *hi,double *groups,unsigned long long *group_ints,double *absolute,long long *visits,unsigned long long *hits){
    P8gGeometry g={geometry,geometry+nl+1,geometry+2*nl+1,geometry+2*nl+nb+2,
        geometry+2*nl+nb+19,geometry+2*nl+nb+20+nspec,mapping,mapping+nl,nl,nb,nc,nf,nspec,1,1,1.};
    P8g2State local;P8g2State *q=&local;p8g2_slot_init(q,&g);
    int h=atomicAdd(next,1);p8g2_gpu_begin(q,h,n,data,meta,constants,&g,seed,start,hits,injected);
    for(;;){
        int snapshot=q->phase;
        unsigned done=__ballot_sync(0xffffffffu,snapshot==P8G2_D);
        unsigned masks[4];for(int p=0;p<4;++p)masks[p]=__ballot_sync(0xffffffffu,snapshot==p);
        if(done==0xffffffffu)break;
        unsigned lane=1u<<(threadIdx.x&31);
        for(int p=0;p<4;++p)if(masks[p]&lane){
            if(reference){P5State a;if(injected)for(int j=0;j<23;++j)a.a[j]=injected[h*23+j];
                rng_init(&q->rng,seed,start+uint64_t(h));p8g_clear(&g,&q->local);
                p8g_record_history(data,meta,constants,&g,&q->rng,maxsteps,&q->record,&q->local,injected?&a:0);q->phase=P8G2_D;
            }else p8g2_advance(q,data,meta,constants,&g,0,maxsteps,hits);
            if(q->phase==P8G2_D){
                int copies=1;
#if P8G2_FAULT == 3
                copies=2;p8g2_hit(hits,32);
#elif P8G2_FAULT == 6
                copies=0;p8g2_hit(hits,33);
#endif
                for(int copy=0;copy<copies;++copy){
                    atomicAdd(reinterpret_cast<unsigned long long*>(visits+h),1ULL);
                    P8gRecord &r=q->record;int dw=p8g_dwidth(&g),iw=p8g_iwidth(&g),b=h*100/n,f=nb+nc+nf;
                    if(history){for(int j=0;j<dw;++j)hd[h*dw+j]=r.d[j];for(int j=0;j<iw;++j)hi[h*iw+j]=r.i[j];}
#if P8G2_FAULT == 5
                    b=(b+1)%100;p8g2_hit(hits,35);
#endif
                    for(int j=0;j<f+21;++j){
#if P8G2_FAULT == 4
                        if(j==f){p8g2_hit(hits,34);continue;}
#endif
                        if((j>=f+6 && j<f+9)||(j>=f+19 && j<f+21))atomicMax(reinterpret_cast<unsigned long long*>(groups+b*dw+j),(unsigned long long)__double_as_longlong(r.d[j]));
                        else atomicAdd(groups+b*dw+j,r.d[j]);
                        atomicAdd(absolute+b*dw+j,fabs(r.d[j]));
                    }
                    for(int j=0;j<iw;++j){if(j==77)atomicMax(group_ints+b*iw+j,(unsigned long long)r.i[j]);else if(j<75||j>=79)atomicAdd(group_ints+b*iw+j,(unsigned long long)r.i[j]);}
                }
                h=atomicAdd(next,1);p8g2_gpu_begin(q,h,n,data,meta,constants,&g,seed,start,hits,injected);
            }
        }
    }
}
extern "C" __global__ void p8g2_init(double *data,const long long *meta,const double *constants,int nm){if(blockIdx.x || threadIdx.x)return;for(int mid=0;mid<nm;++mid){P8gMaterial m=p8g_material(data,meta+136*mid,constants);p8g_initialize(m);}}
