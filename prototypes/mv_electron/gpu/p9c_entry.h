#ifndef P9C_ENTRY_H
#define P9C_ENTRY_H
#include "p9c_kernel.h"
// P9c: compact history records, shared by CPU and CUDA. Grid memory is external.
P5_FN void p9c_one(P9cState *q,const double *data,const long long *meta,const double *constants,const P9cGeometry *g,uint64_t seed,uint64_t hid,int maxsteps,const double *input,double *d,long long *i,double *ledger,long long *sites,long long *source_counts,double *regions){
    p9c_slot_init(q,g);q->field_additions=0;P5State a;if(input)for(int j=0;j<23;++j)a.a[j]=input[j];
    p9c_begin(q,data,meta,constants,g,seed,hid,0,input?&a:0);
    while(q->phase!=P8G2_D)p9c_advance(q,data,meta,constants,g,0,maxsteps,0);
    for(int j=0;j<37;++j)d[j]=q->record.d[j];for(int j=0;j<79;++j)i[j]=q->record.i[j];
    for(int j=0;j<6;++j){ledger[j]=q->ledger_energy[j];sites[j]=q->sites[j];}
    source_counts[0]=q->rejected;source_counts[1]=q->missed;
    if(regions)for(int j=0;j<3420;++j)regions[j]=q->field_scores[j];
}
P5_FN void p9c_boundary_one(const P9cGeometry *g,const double *a,double *out){
    P9cHit h=p9c_boundary(a,g);out[0]=h.distance;out[1]=h.next;
    double b[23];for(int k=0;k<23;++k)b[k]=a[k];for(int k=0;k<3;++k)b[k]+=h.distance*a[k+3];p9c_snap(b,g,h);
    for(int k=0;k<3;++k)out[k+2]=b[k];out[5]=p9c_destination(b,g,h,int(a[16]));
}
#endif
