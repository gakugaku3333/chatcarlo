// G5 A2b fixed P9c specialization; shared t2 physics and Philox.
#include <cmath>
#include <cstdint>
using std::isfinite;
#include "t2_kernel.h"
#include "t2_seed_entry.h"
#include <memory>
extern "C" int t2_seed_geometry_size(){return sizeof(T2SeedGeometry);}
extern "C" void t2_seed_init(double *data,const long long *meta,const double *constants,int nm){for(int j=0;j<nm;++j){P8gMaterial m=p8g_material(data,meta+136*j,constants);p8g_initialize(m);}}
extern "C" void t2_seed_histories(const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,const double *input,double *d,long long *i,double *ledger,long long *sites,long long *source_counts,double *regions){
    std::unique_ptr<T2SeedState> q(new T2SeedState);
    for(int h=0;h<n;++h)t2_seed_one(q.get(),data,meta,constants,g,seed,start+h,maxsteps,input?input+23*h:0,d+37*h,i+79*h,ledger+6*h,sites+6*h,source_counts+4*h,regions?regions+T2S_REGION_ROW*h:0);
}
extern "C" void t2_seed_boundary_probe(const T2SeedGeometry *g,const double *a,int n,double *out){for(int j=0;j<n;++j)t2_seed_boundary_one(g,a+23*j,out+6*j);}
extern "C" void t2_seed_score_probe(const T2SeedGeometry *g,const double *a,const double *b,double loss,int site){for(int k=0;k<2;++k){if(site==3)t2_seed_segment(g->grids+k,a,b,loss);else t2_seed_point(g->grids+k,b,loss);}}
extern "C" void t2_seed_source_probe(const T2SeedGeometry *g,const double *s,const double *d,double *out){out[0]=t2_seed_ellipsoid(s,d);double t=t2_seed_entry(g,s,d);out[1]=t;for(int k=0;k<3;++k)out[k+2]=s[k]+t*d[k];}

extern "C" int t2_seed_set_id(){return G5_SET_ID;}
extern "C" uint64_t t2_seed_set_tag(){return G5_SET_TAG;}

#include "t2_seed_probe.h"
extern "C" void t2_seed_probe(const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,const double *input,const double *fixed,int mode,uint64_t seed,uint64_t start,int n,double *out){for(int h=0;h<n;++h)t2_seed_probe_one(data,meta,constants,g,input+23*h,fixed+3*h,mode,seed,start+h,out+74*h);}
