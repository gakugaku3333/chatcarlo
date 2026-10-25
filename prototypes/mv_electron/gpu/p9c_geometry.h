#ifndef P9C_GEOMETRY_H // P9c: copied namespace / extension.
#define P9C_GEOMETRY_H // P9c: copied namespace / extension.
#include "p8g_kernel.h" // P9c: copied namespace / extension.
#include "p7_field.h" // P9c: copied namespace / extension.
#include "p9c_grid.h" // P9c: copied namespace / extension.
// Seeds are y-aligned cylinders: cx, cy, cz, radius, half-length. // P9c: copied namespace / extension.
struct P9cGeometry : P8gGeometry { // P9c: copied namespace / extension.
    const double *seeds;const long long *parents; // P9c: copied namespace / extension.
    int axis,ns,field,scoring_enabled;double half[3],field_half,ssd; // P9c: copied namespace / extension.
    P7Geometry scoring; // P9c: copied namespace / extension.
    int outer,beam;double radii[3],angle;P9cGrid grids[2]; // P9c: copied namespace / extension.
}; // P9c: copied namespace / extension.
struct P9cHit {double distance;int next,planes,side,cap,seed;double values[3],normal[3];}; // P9c: copied namespace / extension.
P5_FN bool p9c_tie(double a,double b){return a==b || (isfinite(a)&&isfinite(b)&&fabs(a-b)<=8.*2.220446049250313e-16*p5_max(1.,p5_max(fabs(a),fabs(b))));} // P9c: copied namespace / extension.
P5_FN int p9c_parent(const P9cGeometry *g,int r){return r<g->nl?r:int(g->parents[r-g->nl]);} // P9c: copied namespace / extension.
P5_FN P9cHit p9c_boundary(const double *a,const P9cGeometry *g){ // P9c: copied namespace / extension.
    int r=int(a[16]),parent=p9c_parent(g,r);P9cHit h={INFINITY,r,0,0,0,-1,{0.,0.,0.},{0.,0.,0.}}; // P9c: copied namespace / extension.
    if(g->outer && r<g->nl){ // P9c: copied namespace / extension.
        double A=a[3]*a[3]/(g->radii[0]*g->radii[0])+a[5]*a[5]/(g->radii[2]*g->radii[2]); // P9c: copied namespace / extension.
        double B=a[0]*a[3]/(g->radii[0]*g->radii[0])+a[2]*a[5]/(g->radii[2]*g->radii[2]); // P9c: copied namespace / extension.
        double C=a[0]*a[0]/(g->radii[0]*g->radii[0])+a[2]*a[2]/(g->radii[2]*g->radii[2])-1.; // P9c: copied namespace / extension.
        double t=INFINITY;if(A>0.){double D=p5_max(0.,B*B-A*C),root=sqrt(D);t=B>=0.?-C/(B+root):(-B+root)/A;t=p5_max(0.,t);} // P9c: copied namespace / extension.
        h.distance=t;h.next=-1;h.side=2; // P9c: copied namespace / extension.
        h.normal[0]=(a[0]+t*a[3])/(g->radii[0]*g->radii[0]);h.normal[2]=(a[2]+t*a[5])/(g->radii[2]*g->radii[2]); // P9c: copied namespace / extension.
        if(a[4]!=0.){double face=a[4]>0.?g->radii[1]:-g->radii[1],cap=p5_max(0.,(face-a[1])/a[4]); // P9c: copied namespace / extension.
            if(cap<t && !p9c_tie(cap,t)){h.distance=cap;h.side=0;h.normal[0]=h.normal[2]=0.;} // P9c: copied namespace / extension.
            if(p9c_tie(cap,h.distance)){h.planes=2;h.values[1]=face;h.normal[1]=a[4]>0.?1.:-1.;}} // P9c: copied namespace / extension.
    } // P9c: copied namespace / extension.
    if(!g->outer){ // P9c: copied namespace / extension.
    // Preserve the exact P8g-2 division and max for the layer axis. // P9c: copied namespace / extension.
    if(r<g->nl)for(int k=0;k<3;++k){double u=a[k+3];if(u==0.)continue; // P9c: copied namespace / extension.
        double face=k==g->axis?g->faces[r+(u>0)]:((u>0)?g->half[k]:-g->half[k]); // P9c: copied namespace / extension.
        double t=p5_max(0.,(face-a[k])/u);int nr=k==g->axis?r+(u>0?1:-1):-1; // P9c: copied namespace / extension.
        if(t<h.distance && !p9c_tie(t,h.distance)){h.distance=t;h.next=nr;h.planes=0;} // P9c: copied namespace / extension.
        if(p9c_tie(t,h.distance)){h.planes|=1<<k;h.values[k]=face;h.normal[k]=u>0?1.:-1.;if(nr<0||nr>=g->nl)h.next=-1;} // P9c: copied namespace / extension.
    } // P9c: copied namespace / extension.
    } // P9c: copied namespace / extension.
    for(int j=0;j<g->ns;++j){if(int(g->parents[j])!=parent)continue;if(r>=g->nl && r!=g->nl+j)continue; // P9c: copied namespace / extension.
        const double *c=g->seeds+5*j;double x=a[0]-c[0],y=a[1]-c[1],z=a[2]-c[2],ux=a[3],uy=a[4],uz=a[5]; // P9c: copied namespace / extension.
        double A=ux*ux+uz*uz,B=x*ux+z*uz,C=x*x+z*z-c[3]*c[3]; // P9c: copied namespace / extension.
        double sl=-INFINITY,sh=INFINITY,cl=-INFINITY,ch=INFINITY;int capin=0,capout=0; // P9c: copied namespace / extension.
        if(A==0.){if(C>=0.)continue;}else{double cross=x*uz-z*ux;double D=A*(c[3]*c[3])-cross*cross;if(D<=0.)continue; // P9c: copied namespace / extension.
            double root=sqrt(D),q=-B-(B>=0.?root:-root);double t0=q/A,t1=C/q; // P9c: copied namespace / extension.
            sl=p5_min(t0,t1);sh=p5_max(t0,t1);} // P9c: copied namespace / extension.
        if(uy==0.){if(fabs(y)>=c[4])continue;}else{double t0=(-c[4]-y)/uy,t1=(c[4]-y)/uy;cl=p5_min(t0,t1);ch=p5_max(t0,t1);capin=uy>0?-1:1;capout=-capin;} // P9c: copied namespace / extension.
        double enter=p5_max(sl,cl),exit=p5_min(sh,ch);if(enter>=exit)continue; // P9c: copied namespace / extension.
        bool inside=r>=g->nl;double t=inside?exit:enter; // P9c: copied namespace / extension.
        if(!inside){if(exit-p5_max(enter,0.)<=1e-12)continue; // P9c: ignore negligible forward chords before zero-entry admission.
            if(exit<=0.)continue; // P9c: copied namespace / extension.
            // A topological region may be one ulp inside the geometric surface. // P9c: copied namespace / extension.
            // Admit a zero entry only while heading inward through its entry face. // P9c: copied namespace / extension.
            if(t<0.){bool radial=C>=0. || (sl>=cl && B<0.);bool axial=fabs(y)>=c[4] || (cl>=sl && y*uy<0.);if(!(sl>=cl?radial:axial))continue;t=0.;}} // P9c: copied namespace / extension.
        t=p5_max(0.,t);if(t>=h.distance && !p9c_tie(t,h.distance))continue; // P9c: copied namespace / extension.
        h.distance=t;h.next=inside?parent:g->nl+j;h.planes=0;h.seed=j; // P9c: copied namespace / extension.
        double chosen=inside?exit:enter;h.side=p9c_tie(chosen,inside?sh:sl);h.cap=p9c_tie(chosen,inside?ch:cl)?(inside?capout:capin):0; // P9c: copied namespace / extension.
        h.normal[0]=(x+t*ux)/c[3];h.normal[1]=0.;h.normal[2]=(z+t*uz)/c[3]; // P9c: copied namespace / extension.
    }return h; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN void p9c_snap(double *a,const P9cGeometry *g,const P9cHit &h){ // P9c: copied namespace / extension.
    for(int k=0;k<3;++k)if(h.planes&(1<<k))a[k]=h.values[k]; // P9c: copied namespace / extension.
    if(h.cap)a[1]=g->seeds[5*h.seed+1]+h.cap*g->seeds[5*h.seed+4]; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN int p9c_destination(const double *a,const P9cGeometry *g,const P9cHit &h,int old){ // P9c: copied namespace / extension.
    if(h.seed>=0){double radial=a[3]*h.normal[0]+a[5]*h.normal[2],axial=h.cap*a[4]; // P9c: copied namespace / extension.
        if(old<g->nl)return ((!h.side||radial<0.)&&(!h.cap||axial<0.))?h.next:old; // P9c: copied namespace / extension.
        return ((h.side&&radial>0.)||(h.cap&&axial>0.))?h.next:old;} // P9c: copied namespace / extension.
    if(g->outer)return (a[3]*h.normal[0]+a[4]*h.normal[1]+a[5]*h.normal[2]>0.)?-1:old; // P9c: copied namespace / extension.
    bool crossed=false;for(int k=0;k<3;++k)if(h.planes&(1<<k)){ // P9c: copied namespace / extension.
        bool outward=a[k+3]*h.normal[k]>0.; // P9c: copied namespace / extension.
        if(outward && (k!=g->axis || h.values[k]==g->faces[0] || h.values[k]==g->faces[g->nl]))return -1; // P9c: copied namespace / extension.
        if(outward && k==g->axis)crossed=true; // P9c: copied namespace / extension.
    }return crossed?h.next:old; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
P5_FN void p9c_transition(double *a,const P9cGeometry *g,const P9cHit &h,const P8gMaterial &m,double total,double G1,RngState *rng,double *info,double override_mu=2.,const double *fixed=0){ // P9c: copied namespace / extension.
    int old=int(a[16]);p9c_snap(a,g,h); // P9c: copied namespace / extension.
    if(h.next>=0 && h.next<g->nl+g->ns && g->ids[old]!=g->ids[h.next]){ // P9c: copied namespace / extension.
        // The frozen routine retains scattering and RNG order; only its z-only decision is replaced. // P9c: copied namespace / extension.
        p8g_material_boundary(a,old,h.next,m,total,G1,p5_rule(p5_max(.01,a[7])),rng,info,override_mu); // P9c: copied namespace / extension.
        if(fixed)for(int k=0;k<3;++k)a[3+k]=fixed[k]; // P9c: copied namespace / extension.
        a[16]=p9c_destination(a,g,h,old); // P9c: copied namespace / extension.
    }else a[16]=h.next; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
// Photon transitions are shared by production's optical-depth loop and G3. // P9c: copied namespace / extension.
P5_FN void p9c_photon_transition(double *a,const P9cGeometry *g,const P9cHit &hit,const double *data,const long long *meta,const double *constants,double E,P8gMaterial *m,double *c,double *mu,int *rebuilds){ // P9c: copied namespace / extension.
    p9c_snap(a,g,hit);a[16]=hit.next; // P9c: copied namespace / extension.
    *m=p8g_material(data,meta+p8g_mid(g->ids[hit.next])*136,constants); // P9c: copied namespace / extension.
    p8g_coefficients(E,*m,c);*mu=0.; // P9c: copied namespace / extension.
    for(int z=0;z<4;++z){c[z]*=g->rho[hit.next];*mu+=c[z];}++*rebuilds; // P9c: copied namespace / extension.
} // P9c: copied namespace / extension.
#endif // P9c: copied namespace / extension.
