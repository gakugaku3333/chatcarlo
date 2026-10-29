//@@COMMON@@
extern "C" __global__ void g1_histories_kernel(const double *data,const long long *offsets,int me,int ng,int nm,
    const double *centers,const double *halves,const long long *materials,int nb,int bg,const double *world,const double *det,
    unsigned long long seed,unsigned long long start,int n,const double *o,const double *dir,const double *energy,int fluor,
    int record,double *d,long long *ints,double *agg,double *trace,int trace_capacity) {
    int h=blockIdx.x*blockDim.x+threadIdx.x;if(h>=n)return;
    G1Tables t={data,offsets,me,ng,nm};G1Geometry g={centers,halves,materials,nb,bg,world,det};
    RngState rng;rng_init(&rng,seed,start+uint64_t(h));double local[34];long long li[9];
    g1_history(&t,&g,&rng,o+3*h,dir+3*h,energy[h],fluor,local,li,trace,trace_capacity);
    g1_add_record(&t,&g,local,li,agg);
    if(record){for(int j=0;j<2+nm;++j)d[h*(2+nm)+j]=local[j];for(int j=0;j<9;++j)ints[h*9+j]=li[j];}
}
extern "C" __global__ void g1_geometry_probe_flat(int which,const double *centers,const double *halves,const long long *materials,int nb,int bg,const double *world,const double *det,const double *x,int n,double *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;G1Geometry g={centers,halves,materials,nb,bg,world,det};
    const double *a=x+10*i;double *y=out+4*i;
    if(which==0)g1_box(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],y);
    if(which==1)y[0]=g1_material(a[0],a[1],a[2],&g);
    if(which==2)g1_boundary(a[0],a[1],a[2],a[3],a[4],a[5],&g,y);
    if(which==3)g1_detector(a,a+3,a[6],det,y);
}


extern "C" __global__ void g2_histories_kernel(const double *data,const long long *offsets,int me,int ng,int nm,
    const double *centers,const double *halves,const long long *materials,int nb,int bg,const double *world,const double *det,
    unsigned long long seed,unsigned long long start,int n,const double *o,const double *dir,const double *energy,int fluor,
    int record,double *d,long long *ints,double *agg,double *trace,int trace_capacity,const double *table,const double *cfg,double *kerma,double *h10,double *count,double *sum,double *sum2,double *box_count) {
    G2Tally tally={table,cfg,nm,kerma,h10,count,sum,sum2,box_count};
    int h=blockIdx.x*blockDim.x+threadIdx.x;if(h>=n)return;
    G1Tables t={data,offsets,me,ng,nm};G1Geometry g={centers,halves,materials,nb,bg,world,det};
    RngState rng;rng_init(&rng,seed,start+uint64_t(h));double local[34];long long li[9];
    g1_history(&t,&g,&rng,o+3*h,dir+3*h,energy[h],fluor,local,li,trace,trace_capacity,&tally);
    g1_add_record(&t,&g,local,li,agg);
    if(record){for(int j=0;j<2+nm;++j)d[h*(2+nm)+j]=local[j];for(int j=0;j<9;++j)ints[h*9+j]=li[j];}
}

extern "C" __global__ void g5_physics_set_kernel(uint64_t *out) {if(blockIdx.x || threadIdx.x)return;out[0]=G5_SET_ID;out[1]=G5_SET_TAG;}
