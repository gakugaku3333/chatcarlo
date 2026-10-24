#ifndef NAN
#define NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif
#ifndef INFINITY
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
//@@COMMON@@
// Complete histories, then dynamic fetch; no state-machine scheduling.
extern "C" __global__ void p8g_kernel(const double *data,const long long *meta,const double *constants,
    const double *geometry,const long long *mapping,int nl,int nb,int nc,int nf,int nspec,
    uint64_t seed,uint64_t start,int n,int *next,int history,
    double *hd,long long *hi,double *groups,unsigned long long *group_ints,double *absolute) {
    P8gGeometry g={geometry,geometry+nl+1,geometry+2*nl+1,geometry+2*nl+nb+2,
        geometry+2*nl+nb+19,geometry+2*nl+nb+20+nspec,mapping,mapping+nl,nl,nb,nc,nf,nspec,1,1,1.};
    P8gLocal l={};P8gRecord r;RngState rng;
    int h=blockIdx.x*blockDim.x+threadIdx.x;
    while(h<n){rng_init(&rng,seed,start+uint64_t(h));p8g_record_history(data,meta,constants,&g,&rng,10000000,&r,&l);
        int dw=p8g_dwidth(&g),iw=p8g_iwidth(&g),b=h*100/n;if(b>99)b=99;
        if(history){for(int j=0;j<dw;++j)hd[h*dw+j]=r.d[j];for(int j=0;j<iw;++j)hi[h*iw+j]=r.i[j];}
int f=nb+nc+nf;
        for(int j=0;j<f+21;++j){
            if((j>=f+6 && j<f+9) || (j>=f+19 && j<f+21))
                atomicMax(reinterpret_cast<unsigned long long*>(groups+b*dw+j),(unsigned long long)__double_as_longlong(r.d[j]));
            else atomicAdd(groups+b*dw+j,r.d[j]);
            atomicAdd(absolute+b*dw+j,fabs(r.d[j]));
        }
        for(int j=0;j<iw;++j){if(j==77)atomicMax(group_ints+b*iw+j,(unsigned long long)r.i[j]);else if(j<75 || j>=79)atomicAdd(group_ints+b*iw+j,(unsigned long long)r.i[j]);}
        p8g_clear(&g,&l);h=atomicAdd(next,1);
    }
}
extern "C" __global__ void p8g_fixed_kernel(const double *data,const long long *meta,const double *constants,
    const double *geometry,const long long *mapping,int nl,int nb,int nc,int nf,
    uint64_t seed,const double *state,double *hd,long long *hi,double *events,long long *transitions) {
    if(blockIdx.x || threadIdx.x)return;
    P8gGeometry g={geometry,geometry+nl+1,geometry+2*nl+1,geometry+2*nl+nb+2,0,0,mapping,mapping+nl,nl,nb,nc,nf,0,1,1,state[6]};
    P5State a;for(int j=0;j<23;++j)a.a[j]=state[j];P8gRecord r;P8gLocal l={};RngState rng;rng_init(&rng,seed,0);
    p8g_record_history(data,meta,constants,&g,&rng,10000000,&r,&l,&a);
    for(int j=0;j<p8g_dwidth(&g);++j)hd[j]=r.d[j];for(int j=0;j<p8g_iwidth(&g);++j)hi[j]=r.i[j];
    for(int j=0;j<160;++j)events[j]=l.events[j];for(int j=0;j<80;++j)transitions[j]=l.transitions[j];
}
//@@PROBE@@

extern "C" __global__ void p8g_init(double *data,const long long *meta,const double *constants,int nm){if(blockIdx.x || threadIdx.x)return;for(int mid=0;mid<nm;++mid){P8gMaterial m=p8g_material(data,meta+136*mid,constants);p8g_initialize(m);}}
