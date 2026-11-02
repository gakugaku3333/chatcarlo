// G5 A2b fixed P9c specialization; shared t2 physics and Philox.
#ifndef T2_SEED_ENTRY_H
#define T2_SEED_ENTRY_H
#include "t2_seed_kernel.h"
// T2Seed: compact history records, shared by CPU and CUDA. Grid memory is external.
P5_FN void t2_seed_one(T2SeedState *q,const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,uint64_t seed,uint64_t hid,int maxsteps,const double *input,double *d,long long *i,double *ledger,long long *sites,long long *source_counts,double *regions){
    t2_seed_slot_init(q,g);q->field_additions=0;q->region_out=regions;P5State a;if(input)for(int j=0;j<23;++j)a.a[j]=input[j];
    t2_seed_begin(q,data,meta,constants,g,seed,hid,0,input?&a:0);
    while(q->phase!=T2S_D)t2_seed_advance(q,data,meta,constants,g,0,maxsteps,0);
    for(int j=0;j<37;++j)d[j]=q->record.d[j];for(int j=0;j<79;++j)i[j]=q->record.i[j];
    for(int j=0;j<6;++j){ledger[j]=q->ledger_energy[j];sites[j]=q->sites[j];}
    source_counts[0]=q->rejected;source_counts[1]=q->missed;source_counts[2]=(long long)hid;source_counts[3]=1;
#if T2_SEED_FAULT == 3
source_counts[3]=0;
#endif
#if T2_SEED_FAULT == 4
source_counts[2]=(long long)hid-1;
#endif
}
P5_FN void t2_seed_boundary_one(const T2SeedGeometry *g,const double *a,double *out){
    T2SeedHit h=t2_seed_boundary(a,g);out[0]=h.distance;out[1]=h.next;
    double b[23];for(int k=0;k<23;++k)b[k]=a[k];for(int k=0;k<3;++k)b[k]+=h.distance*a[k+3];t2_seed_snap(b,g,h);
    for(int k=0;k<3;++k)out[k+2]=b[k];out[5]=t2_seed_destination(b,g,h,int(a[16]));
}
#endif
