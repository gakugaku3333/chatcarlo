#ifndef P9B_GEOMETRY_H
#define P9B_GEOMETRY_H
#include "p8g_kernel.h"
#include "p7_field.h"
// Seeds are y-aligned cylinders: cx, cy, cz, radius, half-length.
struct P9bGeometry : P8gGeometry {
    const double *seeds;const long long *parents;
    int axis,ns,field,scoring_enabled;double half[3],field_half,ssd;
    P7Geometry scoring;
};
struct P9bHit {double distance;int next,planes,side,cap,seed;double values[3],normal[3];};
P5_FN bool p9b_tie(double a,double b){return a==b || (isfinite(a)&&isfinite(b)&&fabs(a-b)<=8.*2.220446049250313e-16*p5_max(1.,p5_max(fabs(a),fabs(b))));}
P5_FN int p9b_parent(const P9bGeometry *g,int r){return r<g->nl?r:int(g->parents[r-g->nl]);}
P5_FN P9bHit p9b_boundary(const double *a,const P9bGeometry *g){
    int r=int(a[16]),parent=p9b_parent(g,r);P9bHit h={INFINITY,r,0,0,0,-1,{0.,0.,0.},{0.,0.,0.}};
    // Preserve the exact P8g-2 division and max for the layer axis.
    if(r<g->nl)for(int k=0;k<3;++k){double u=a[k+3];if(u==0.)continue;
        double face=k==g->axis?g->faces[r+(u>0)]:((u>0)?g->half[k]:-g->half[k]);
        double t=p5_max(0.,(face-a[k])/u);int nr=k==g->axis?r+(u>0?1:-1):-1;
        if(t<h.distance && !p9b_tie(t,h.distance)){h.distance=t;h.next=nr;h.planes=0;}
        if(p9b_tie(t,h.distance)){h.planes|=1<<k;h.values[k]=face;h.normal[k]=u>0?1.:-1.;if(nr<0||nr>=g->nl)h.next=-1;}
    }
    for(int j=0;j<g->ns;++j){if(int(g->parents[j])!=parent)continue;if(r>=g->nl && r!=g->nl+j)continue;
        const double *c=g->seeds+5*j;double x=a[0]-c[0],y=a[1]-c[1],z=a[2]-c[2],ux=a[3],uy=a[4],uz=a[5];
        double A=ux*ux+uz*uz,B=x*ux+z*uz,C=x*x+z*z-c[3]*c[3];
        double sl=-INFINITY,sh=INFINITY,cl=-INFINITY,ch=INFINITY;int capin=0,capout=0;
        if(A==0.){if(C>=0.)continue;}else{double cross=x*uz-z*ux;double D=A*(c[3]*c[3])-cross*cross;if(D<=0.)continue;
            double root=sqrt(D),q=-B-(B>=0.?root:-root);double t0=q/A,t1=C/q;
            sl=p5_min(t0,t1);sh=p5_max(t0,t1);}
        if(uy==0.){if(fabs(y)>=c[4])continue;}else{double t0=(-c[4]-y)/uy,t1=(c[4]-y)/uy;cl=p5_min(t0,t1);ch=p5_max(t0,t1);capin=uy>0?-1:1;capout=-capin;}
        double enter=p5_max(sl,cl),exit=p5_min(sh,ch);if(enter>=exit)continue;
        bool inside=r>=g->nl;double t=inside?exit:enter;
        if(!inside){if(exit<=0.)continue;
            // A topological region may be one ulp inside the geometric surface.
            // Admit a zero entry only while heading inward through its entry face.
            if(t<0.){bool radial=C>=0. || (sl>=cl && B<0.);bool axial=fabs(y)>=c[4] || (cl>=sl && y*uy<0.);if(!(sl>=cl?radial:axial))continue;t=0.;}}
        t=p5_max(0.,t);if(t>=h.distance && !p9b_tie(t,h.distance))continue;
        h.distance=t;h.next=inside?parent:g->nl+j;h.planes=0;h.seed=j;
        double chosen=inside?exit:enter;h.side=p9b_tie(chosen,inside?sh:sl);h.cap=p9b_tie(chosen,inside?ch:cl)?(inside?capout:capin):0;
        h.normal[0]=(x+t*ux)/c[3];h.normal[1]=0.;h.normal[2]=(z+t*uz)/c[3];
    }return h;
}
P5_FN void p9b_snap(double *a,const P9bGeometry *g,const P9bHit &h){
    for(int k=0;k<3;++k)if(h.planes&(1<<k))a[k]=h.values[k];
    if(h.cap)a[1]=g->seeds[5*h.seed+1]+h.cap*g->seeds[5*h.seed+4];
}
P5_FN int p9b_destination(const double *a,const P9bGeometry *g,const P9bHit &h,int old){
    if(h.seed>=0){double radial=a[3]*h.normal[0]+a[5]*h.normal[2],axial=h.cap*a[4];
        if(old<g->nl)return ((!h.side||radial<0.)&&(!h.cap||axial<0.))?h.next:old;
        return ((h.side&&radial>0.)||(h.cap&&axial>0.))?h.next:old;}
    bool crossed=false;for(int k=0;k<3;++k)if(h.planes&(1<<k)){
        bool outward=a[k+3]*h.normal[k]>0.;
        if(outward && (k!=g->axis || h.values[k]==g->faces[0] || h.values[k]==g->faces[g->nl]))return -1;
        if(outward && k==g->axis)crossed=true;
    }return crossed?h.next:old;
}
P5_FN void p9b_transition(double *a,const P9bGeometry *g,const P9bHit &h,const P8gMaterial &m,double total,double G1,RngState *rng,double *info,double override_mu=2.,const double *fixed=0){
    int old=int(a[16]);p9b_snap(a,g,h);
    if(h.next>=0 && h.next<g->nl+g->ns && g->ids[old]!=g->ids[h.next]){
        // The frozen routine retains scattering and RNG order; only its z-only decision is replaced.
        p8g_material_boundary(a,old,h.next,m,total,G1,p5_rule(p5_max(.01,a[7])),rng,info,override_mu);
        if(fixed)for(int k=0;k<3;++k)a[3+k]=fixed[k];
        a[16]=p9b_destination(a,g,h,old);
    }else a[16]=h.next;
}
// Photon transitions are shared by production's optical-depth loop and G3.
P5_FN void p9b_photon_transition(double *a,const P9bGeometry *g,const P9bHit &hit,const double *data,const long long *meta,const double *constants,double E,P8gMaterial *m,double *c,double *mu,int *rebuilds){
    p9b_snap(a,g,hit);a[16]=hit.next;
    *m=p8g_material(data,meta+p8g_mid(g->ids[hit.next])*136,constants);
    p8g_coefficients(E,*m,c);*mu=0.;
    for(int z=0;z<4;++z){c[z]*=g->rho[hit.next];*mu+=c[z];}++*rebuilds;
}
#endif
