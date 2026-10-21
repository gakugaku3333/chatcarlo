#ifndef G1_GEOMETRY_H
#define G1_GEOMETRY_H
#include "g1_common.h"
// Ports: kernel.py:539-620, detector.py:47-66.
struct G1Geometry {const double *centers,*halves;const long long *materials;int nb,bg;const double *world;const double *det;};
P5_FN void g1_box(double ox,double oy,double oz,double dx,double dy,double dz,double hx,double hy,double hz,double *out) {
    double o[3]={ox,oy,oz},d[3]={dx,dy,dz},h[3]={hx,hy,hz};
    double enter=-INFINITY,exit=INFINITY;bool miss=false;
    for(int j=0;j<3;++j){
        if(fabs(d[j])<1e-12){if(o[j]< -h[j] || o[j]>h[j])miss=true;}
        else {double a=(-h[j]-o[j])/d[j],b=(h[j]-o[j])/d[j];
            double mn=a<b?a:b,mx=a>b?a:b;enter=enter>mn?enter:mn;exit=exit<mx?exit:mx;}
    }
    out[0]=enter;out[1]=exit;out[2]=!miss && enter<=exit;
}
P5_FN int g1_material(double x,double y,double z,const G1Geometry *g) {
    int mat=g->bg;
    for(int b=0;b<g->nb;++b){const double *c=g->centers+3*b,*h=g->halves+3*b;
        if(fabs(x-c[0])<=h[0] && fabs(y-c[1])<=h[1] && fabs(z-c[2])<=h[2]) {
            mat=int(g->materials[b]);if(P5_FAULT==4)break;
        }
    }
    return mat;
}
P5_FN void g1_boundary(double x,double y,double z,double dx,double dy,double dz,const G1Geometry *g,double *out) {
    double obj=INFINITY,v[3];
    for(int b=0;b<g->nb;++b){const double *c=g->centers+3*b,*h=g->halves+3*b;
        g1_box(x-c[0],y-c[1],z-c[2],dx,dy,dz,h[0],h[1],h[2],v);
        if(v[2]){if(v[0]>1e-6 && v[0]<obj)obj=v[0];if(v[1]>1e-6 && v[1]<obj)obj=v[1];}
    }
    const double *w=g->world;g1_box(x-w[0],y-w[1],z-w[2],dx,dy,dz,w[3],w[4],w[5],v);
    double exit=v[2]?v[1]:INFINITY;out[0]=obj<exit?obj:exit;out[1]=exit<=obj;
}
P5_FN void g1_detector(const double *o,const double *d,double ds,const double *p,double *out) {
    // Packed plane: center, normal, u, v, size[2], shape[2].
    out[0]=INFINITY;out[1]=-1.;out[2]=-1.;out[3]=0.;
    if(!p)return;
    double denom=(d[0]*p[3]+d[1]*p[4])+d[2]*p[5];
    if(denom==0.)return;
    double t=(((p[0]-o[0])*p[3]+(p[1]-o[1])*p[4])+(p[2]-o[2])*p[5])/denom;
    double rel[3];for(int j=0;j<3;++j)rel[j]=(o[j]+d[j]*t)-p[j];
    double uu=(rel[0]*p[6]+rel[1]*p[7])+rel[2]*p[8];
    double vv=(rel[0]*p[9]+rel[1]*p[10])+rel[2]*p[11];
    double du=p[12]/p[14],dv=p[13]/p[15];
    long long iu=(long long)floor((uu+p[12]/2.)/du),iv=(long long)floor((vv+p[13]/2.)/dv);
    if(denom<0. && t>=0. && t<=ds && iu>=0 && iu<p[14] && iv>=0 && iv<p[15]){
        out[0]=t;out[1]=double(iu);out[2]=double(iv);out[3]=1.;
    }
}
#endif
