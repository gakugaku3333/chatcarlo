#ifndef G1_PHYSICS_H
#define G1_PHYSICS_H
#include "g1_common.h"
// Scalar ports of chatcarlo/kernel.py:285-536. Preserve evaluation order.
P5_FN double g1_lerp(const double *x_grid,const double *y_grid,int n,double x) {
    if(x<=x_grid[0])return y_grid[0];
    if(x>=x_grid[n-1])return y_grid[n-1];
    int lo=0,hi=n-1;
    while(hi-lo>1) {int mid=(lo+hi)/2;if(x_grid[mid]<=x)lo=mid;else hi=mid;}
    double x0=x_grid[lo],x1=x_grid[hi],y0=y_grid[lo],y1=y_grid[hi];
    return y0+(x-x0)/(x1-x0)*(y1-y0);
}
P5_FN void g1_index(const double *log_e_row,double step,int n,double log_eq,int *index,double *fraction) {
    int hi=n-1,idx;
    if(step>0.) {
        idx=int((log_eq-log_e_row[0])/step)+1;
        if(idx<1)idx=1;if(idx>hi)idx=hi;
        if(log_eq<=log_e_row[idx-1]){--idx;if(idx<1)idx=1;}
        if(log_e_row[idx]<log_eq){++idx;if(idx>hi)idx=hi;}
    } else {
        int lo_i=0,hi_i=n;
        while(lo_i<hi_i){int mid=(lo_i+hi_i)/2;if(log_e_row[mid]<log_eq)lo_i=mid+1;else hi_i=mid;}
        idx=lo_i;if(idx<1)idx=1;if(idx>hi)idx=hi;
    }
    double x0=log_e_row[idx-1],x1=log_e_row[idx];
    *index=idx;*fraction=(log_eq-x0)/(x1-x0);
}
P5_FN int g1_select(int n,const double *fracs,const double *cs,double total,RngState *rng) {
    double target=rng_uniform(rng)*total,cum=0.;
    for(int i=0;i<n;++i){cum+=fracs[i]*cs[i];if(target<=cum)return i;}
    return n-1;
}
P5_FN void g1_scatter(double dx,double dy,double dz,double cos_theta,RngState *rng,double *out) {
    double s=1.-cos_theta*cos_theta;
    double sin_theta=sqrt(s>0.?s:0.);
    double phi=rng_uniform(rng)*2.0*3.141592653589793;
    double upx,upy,upz;
    if(fabs(dz)<.999){upx=0.;upy=0.;upz=1.;}else{upx=1.;upy=0.;upz=0.;}
    double ux=upy*dz-upz*dy,uy=upz*dx-upx*dz,uz=upx*dy-upy*dx;
    double un=sqrt(ux*ux+uy*uy+uz*uz);
    ux/=un;uy/=un;uz/=un;
    double vx=dy*uz-dz*uy,vy=dz*ux-dx*uz,vz=dx*uy-dy*ux;
    double cphi=cos(phi),sphi=sin(phi);
    double nx=sin_theta*cphi*ux+sin_theta*sphi*vx+cos_theta*dx;
    double ny=sin_theta*cphi*uy+sin_theta*sphi*vy+cos_theta*dy;
    double nz=sin_theta*cphi*uz+sin_theta*sphi*vz+cos_theta*dz;
    double nn=sqrt(nx*nx+ny*ny+nz*nz);
    out[0]=nx/nn;out[1]=ny/nn;out[2]=nz/nn;
}
P5_FN void g1_isotropic(RngState *rng,double *out) {
    double cos_theta=rng_uniform(rng)*2.0-1.0;
    double s=1.0-cos_theta*cos_theta;
    double sin_theta=sqrt(s>0.?s:0.);
    double phi=rng_uniform(rng)*2.0*3.141592653589793;
    out[0]=sin_theta*cos(phi);out[1]=sin_theta*sin(phi);out[2]=cos_theta;
}

// Field order follows SceneMaterialTables (bake_scene_materials).
enum {NE,ZS,FRACS,LOGE,STEP,NG,PHOTO,COMPT,RAYL,IQ,IS,RX,RA,KE,KO,KLE,KLP,NL,KF,DENS};
struct G1Tables {const double *data;const long long *offsets;int me,ng,nm;};
P5_FN const double *g1_row(const G1Tables *t,int field,int m=0,int e=0,int width=1) {
    return t->data+t->offsets[field]+(m*t->me+e)*width;
}
P5_FN void g1_mu(double energy,int m,const G1Tables *t,double *p,double *c,double *r,int *idx,double *frac,double *out) {
    int n=int(g1_row(t,NE)[m]);double log_eq=log(energy);
    double tp=0.,tc=0.,tr=0.;
    for(int i=0;i<n;++i){
        g1_index(g1_row(t,LOGE,m,i,t->ng),g1_row(t,STEP,m,i)[0],int(g1_row(t,NG,m,i)[0]),log_eq,idx+i,frac+i);
        int j=idx[i];double f=frac[i];
        const double *pp=g1_row(t,PHOTO,m,i,t->ng),*cc=g1_row(t,COMPT,m,i,t->ng),*rr=g1_row(t,RAYL,m,i,t->ng);
        p[i]=exp(pp[j-1]+f*(pp[j]-pp[j-1]));
        c[i]=exp(cc[j-1]+f*(cc[j]-cc[j-1]));
        r[i]=exp(rr[j-1]+f*(rr[j]-rr[j-1]));
        double fr=g1_row(t,FRACS,m,i)[0];tp+=fr*p[i];tc+=fr*c[i];tr+=fr*r[i];
    }
    out[0]=(tp+tc+tr)*g1_row(t,DENS)[m];out[1]=tp;out[2]=tc;out[3]=tr;
}
P5_FN double g1_fluorescence(int m,int ei,double energy,int idx,double frac,const G1Tables *t,RngState *rng) {
    double edge=g1_row(t,KE,m,ei)[0];if(energy<=edge)return 0.;
    double omega=g1_row(t,KO,m,ei)[0];if(omega<=0.)return 0.;
    const double *kf=g1_row(t,KF,m,ei,t->ng);
    double k_frac=kf[idx-1]+frac*(kf[idx]-kf[idx-1]);
    if(rng_uniform(rng)>=k_frac)return 0.;
    if(rng_uniform(rng)>=omega)return 0.;
    int n=int(g1_row(t,NL,m,ei)[0]);double r=rng_uniform(rng),cum=0.;int chosen=n-1;
    for(int j=0;j<n;++j){cum+=g1_row(t,KLP,m,ei,8)[j];if(r<=cum){chosen=j;break;}}
    double line=g1_row(t,KLE,m,ei,8)[chosen];
#if P5_FAULT != 3
    if(line<5.)return 0.;
#endif
    return line;
}
P5_FN void g1_compton(int m,double energy,const G1Tables *t,const double *c,double total,RngState *rng,double *out) {
    int ei=g1_select(int(g1_row(t,NE)[m]),g1_row(t,FRACS,m),c,total,rng);
    double z=g1_row(t,ZS,m,ei)[0],alpha=energy/511.;
    double eps_min=1./(1.+2.*alpha),envelope=1./eps_min+eps_min;
    while(true){
        double xi1=rng_uniform(rng),xi2=rng_uniform(rng);
        double ep=eps_min+xi1*(1.-eps_min);
        double co=1.-(1./ep-1.)/alpha;
        double sin2=1.-co*co,g=1./ep+ep-sin2;
        if(xi2*envelope>g)continue;
        double cc=co;if(cc>1.)cc=1.;if(cc< -1.)cc=-1.;
        double theta=acos(cc),q=energy*sin(theta/2.)/12.3984193;
        double s_over_z=g1_lerp(g1_row(t,IQ,m,ei,2000),g1_row(t,IS,m,ei,2000),2000,q)/z;
        double xi3=rng_uniform(rng);
        if(P5_FAULT==2 || xi3<=s_over_z){out[0]=ep;out[1]=co;out[2]=ei;return;}
    }
}
P5_FN void g1_rayleigh(int m,double energy,const G1Tables *t,const double *r,double total,RngState *rng,double *out) {
    int ei=g1_select(int(g1_row(t,NE)[m]),g1_row(t,FRACS,m),r,total,rng);
    double ratio=energy/12.3984193,x_max=ratio*ratio;
    while(true){
        double cut=g1_lerp(g1_row(t,RX,m,ei,2000),g1_row(t,RA,m,ei,2000),2000,x_max);
        double xi1=rng_uniform(rng);
        double x=g1_lerp(g1_row(t,RA,m,ei,2000),g1_row(t,RX,m,ei,2000),2000,xi1*cut);
        if(x>x_max)x=x_max;
        double co=1.-2.*x/x_max;if(co>1.)co=1.;if(co< -1.)co=-1.;
        double xi2=rng_uniform(rng);
        if(P5_FAULT==1 || xi2<=(1.+co*co)/2.){out[0]=co;out[1]=ei;return;}
    }
}
#endif
