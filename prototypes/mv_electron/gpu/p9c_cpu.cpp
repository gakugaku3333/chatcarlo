#include "p5d_gxx_compat.h"
#include "p9c_entry.h"
#include <memory>
extern "C" int p9c_geometry_size(){return sizeof(P9cGeometry);}
extern "C" void p9c_init(double *data,const long long *meta,const double *constants,int nm){for(int j=0;j<nm;++j){P8gMaterial m=p8g_material(data,meta+136*j,constants);p8g_initialize(m);}}
extern "C" void p9c_histories(const double *data,const long long *meta,const double *constants,const P9cGeometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,const double *input,double *d,long long *i,double *ledger,long long *sites,long long *source_counts,double *regions){
    std::unique_ptr<P9cState> q(new P9cState);
    for(int h=0;h<n;++h)p9c_one(q.get(),data,meta,constants,g,seed,start+h,maxsteps,input?input+23*h:0,d+37*h,i+79*h,ledger+6*h,sites+6*h,source_counts+2*h,regions?regions+3420*h:0);
}
extern "C" void p9c_boundary_probe(const P9cGeometry *g,const double *a,int n,double *out){for(int j=0;j<n;++j)p9c_boundary_one(g,a+23*j,out+6*j);}
extern "C" void p9c_score_probe(const P9cGeometry *g,const double *a,const double *b,double loss,int site){for(int k=0;k<2;++k){if(site==3)p9c_segment(g->grids+k,a,b,loss);else p9c_point(g->grids+k,b,loss);}}
extern "C" void p9c_source_probe(const P9cGeometry *g,const double *s,const double *d,double *out){out[0]=p9c_ellipsoid(s,d);double t=p9c_entry(g,s,d);out[1]=t;for(int k=0;k<3;++k)out[k+2]=s[k]+t*d[k];}
