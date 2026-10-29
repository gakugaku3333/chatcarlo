#include "g1_physics.h"
// Verification-only entry: one sequential MT stream, including the complete state.
extern "C" void g1_direction_probe(int which,const double *x,int n,uint64_t seed,double *out,uint64_t *state,double *next) {
    RngState rng;rng_init(&rng,seed,0);
    for(int i=0;i<n;++i) {
        uint64_t before=rng.calls;
        if(which==0)g1_scatter(x[4*i],x[4*i+1],x[4*i+2],x[4*i+3],&rng,out+3*i);
        else g1_isotropic(&rng,out+3*i);
        state[626+i]=rng.calls-before;
    }
#ifdef P5_RNG_MT
    for(int j=0;j<624;++j)state[j]=rng.mt[j];state[624]=rng.index;
#endif
    state[625]=rng.calls;
    for(int j=0;j<16;++j)next[j]=rng_uniform(&rng);
}
#include "g1_geometry.h"
extern "C" void g1_physics_probe(int which,const double *data,const long long *offsets,int me,int ng,int nm,const double *x,int n,uint64_t seed,double *out,uint64_t *state,double *next) {
    G1Tables t={data,offsets,me,ng,nm};RngState rng;rng_init(&rng,seed,0);
    for(int j=0;j<n;++j){
        uint64_t before=rng.calls;const double *a=x+8*j;double *y=out+(8+5*me)*j;
        int m=int(a[1]),ne=int(g1_row(&t,NE)[m]);
        double p[32]={},c[32]={},r[32]={},frac[32]={},parts[4]={};int idx[32]={};
        if(which<=3)g1_mu(a[0],m,&t,p,c,r,idx,frac,parts);
        if(which==0){for(int k=0;k<4;++k)y[k]=parts[k];
            for(int k=0;k<ne;++k){y[8+k]=p[k];y[8+me+k]=c[k];y[8+2*me+k]=r[k];y[8+3*me+k]=idx[k];y[8+4*me+k]=frac[k];}}
        if(which==1)g1_compton(m,a[0],&t,c,parts[2],&rng,y);
        if(which==2)g1_rayleigh(m,a[0],&t,r,parts[3],&rng,y);
        if(which==3){int e=int(a[2]);double line=g1_fluorescence(m,e,a[0],idx[e],frac[e],&t,&rng);y[0]=line>0.;y[1]=line;}
        if(which==4)y[0]=g1_select(ne,g1_row(&t,FRACS,m),g1_row(&t,FRACS,m),a[0],&rng);
        if(which==5)y[0]=g1_lerp(g1_row(&t,LOGE,m,int(a[2]),ng),g1_row(&t,PHOTO,m,int(a[2]),ng),int(g1_row(&t,NG,m,int(a[2]))[0]),a[0]);
        if(which==6){int id;double f;g1_index(g1_row(&t,LOGE,m,int(a[2]),ng),g1_row(&t,STEP,m,int(a[2]))[0],int(g1_row(&t,NG,m,int(a[2]))[0]),a[0],&id,&f);y[0]=id;y[1]=f;}
        state[626+j]=rng.calls-before;
    }
#ifdef P5_RNG_MT
    for(int j=0;j<624;++j)state[j]=rng.mt[j];state[624]=rng.index;
#endif
    state[625]=rng.calls;for(int j=0;j<16;++j)next[j]=rng_uniform(&rng);
}
extern "C" void g1_geometry_probe(int which,const G1Geometry *g,const double *x,int n,double *out) {
    for(int i=0;i<n;++i){const double *a=x+10*i;double *y=out+4*i;
        if(which==0)g1_box(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],y);
        if(which==1)y[0]=g1_material(a[0],a[1],a[2],g);
        if(which==2)g1_boundary(a[0],a[1],a[2],a[3],a[4],a[5],g,y);
        if(which==3)g1_detector(a,a+3,a[6],g->det,y);
    }
}
#include "g1_history.h"
extern "C" void g1_histories(const double *data,const long long *offsets,int me,int ng,int nm,const G1Geometry *g,
    uint64_t seed,uint64_t start,int n,const double *o,const double *dir,const double *energy,int fluor,
    int record,double *d,long long *ints,double *agg,double *trace,int trace_capacity) {
    G1Tables t={data,offsets,me,ng,nm};RngState rng;rng_init(&rng,seed,0);
    for(int h=0;h<n;++h){
#ifdef P5_RNG_PHILOX
        rng_init(&rng,seed,start+uint64_t(h));
#endif
        double local[34];long long li[9];uint64_t before=rng.calls;
        g1_history(&t,g,&rng,o+3*h,dir+3*h,energy[h],fluor,local,li,trace,trace_capacity);
        li[7]-=before;g1_add_record(&t,g,local,li,agg);
        if(record){for(int j=0;j<2+nm;++j)d[h*(2+nm)+j]=local[j];for(int j=0;j<9;++j)ints[h*9+j]=li[j];}
    }
}
extern "C" void g1_reduce_probe(int nm,const double *det,const double *d,const long long *i,int n,double *out) {
    G1Tables t={0,0,0,0,nm};G1Geometry g={0,0,0,0,0,0,det};
    for(int h=0;h<n;++h)g1_add_record(&t,&g,d+h*(nm+2),i+h*9,out);
}

extern "C" void g2_histories(const double *data,const long long *offsets,int me,int ng,int nm,const G1Geometry *g,
    uint64_t seed,uint64_t start,int n,const double *o,const double *dir,const double *energy,int fluor,
    int record,double *d,long long *ints,double *agg,double *trace,int trace_capacity,const double *table,const double *cfg,double *kerma,double *h10,double *count,double *sum,double *sum2,double *box_count) {
    G2Tally tally={table,cfg,nm,kerma,h10,count,sum,sum2,box_count};
    G1Tables t={data,offsets,me,ng,nm};RngState rng;rng_init(&rng,seed,0);
    for(int h=0;h<n;++h){
#ifdef P5_RNG_PHILOX
        rng_init(&rng,seed,start+uint64_t(h));
#endif
        double local[34];long long li[9];uint64_t before=rng.calls;
        g1_history(&t,g,&rng,o+3*h,dir+3*h,energy[h],fluor,local,li,trace,trace_capacity,&tally);
        li[7]-=before;g1_add_record(&t,g,local,li,agg);
        if(record){for(int j=0;j<2+nm;++j)d[h*(2+nm)+j]=local[j];for(int j=0;j<9;++j)ints[h*9+j]=li[j];}
    }
}

extern "C" void g2_probe(int which,int nm,const double *table,const double *cfg,const double *x,int n,double *out,double *h10,double *count) {
    G2Tally t={table,cfg,nm,out,h10,count,0,0,0};
    for(int j=0;j<n;++j){const double *a=x+10*j;
        if(which==0){out[2*j]=g2_muen(&t,int(a[1]),a[0]);out[2*j+1]=g2_h10(&t,a[0]);}
        if(which==1)g2_grid_segment(&t,a,a+3,a[6],a[7],a[8]);
        if(which==2)out[j]=g2_overlap(a,a+3,cfg,cfg+3);
    }
}
extern "C" void g2_moment_probe(int nb,const double *x,int n,double *sum,double *sum2,double *count) {
    double cfg[13]={};cfg[12]=nb;G2Tally t={0,cfg,2,0,0,0,sum,sum2,count};
    for(int h=0;h<n;++h)g2_end_history(&t,x+h*nb);
}
extern "C" double g2_first_distance(const double *data,const long long *offsets,int me,int ng,int nm,uint64_t seed,int m,double e) {
    G1Tables t={data,offsets,me,ng,nm};RngState rng;rng_init(&rng,seed,0);
    double tau=-log(rng_uniform(&rng)),p[32],c[32],r[32],frac[32],part[4];int idx[32];
    g1_mu(e,m,&t,p,c,r,idx,frac,part);return tau/part[0];
}
extern "C" void g2_box_history_probe(int nm,const double *table,const double *cfg,const double *x,const long long *ids,int n,int nh,double *sum,double *sum2,double *count) {
    G2Tally t={table,cfg,nm,0,0,0,sum,sum2,count};
    for(int h=0;h<nh;++h){double local[64]={};
        for(int j=0;j<n;++j)if(ids[j]==h){const double *a=x+10*j;
            g2_segment(&t,int(a[8]),a,a+3,a[6],a[7],local);}
        g2_end_history(&t,local);}
}

extern "C" void g5_physics_set(uint64_t *out) {out[0]=G5_SET_ID;out[1]=G5_SET_TAG;}
