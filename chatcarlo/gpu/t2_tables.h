// Derived from p8f_materials.h; G5 A2 standalone copy.
#ifndef P8F_MATERIALS_H
#define P8F_MATERIALS_H
#include "t2_physics.h"
#ifndef P8F_FAULT
#define P8F_FAULT 0
#endif
// Numeric bundle fields retain P8e Material order. Offsets index doubles;
// descriptors are (offset, size, first dimension, second dimension).
struct P8Material {
    P5Tables p;
    const double *ne,*excitation,*c,*z,*nj,*stop_T,*stop_delta,*stop_collision,*stop_rad;
    const double *gs_T,*gs_K,*gs_u,*gs_y,*rates,*photo_E,*photo,*pair_E,*pair_n,*pair_e,*thresholds;
    const double *dl,*bpar,*delcm,*delpos,*bindings,*pm,*shell_nj,*yields,*lines,*brems_E,*brems_V,*log_stop_T;
    const double *line_counts;
    int nstop,ncollision,nphoto,npair,nz,nline;
};
P5_FN P8Material p8f_material(const double *data,const long long *q,const double *constants) {
    P8Material m={};
    const double **fields[]={&m.ne,&m.excitation,&m.c,&m.z,&m.nj,&m.stop_T,&m.stop_delta,&m.stop_collision,&m.stop_rad,
        &m.gs_T,&m.gs_K,&m.gs_u,&m.gs_y,&m.rates,&m.photo_E,&m.photo,&m.pair_E,&m.pair_n,&m.pair_e,&m.thresholds,
        &m.dl,&m.bpar,&m.delcm,&m.delpos,&m.bindings,&m.pm,&m.shell_nj,&m.yields,&m.lines,&m.line_counts,
        &m.brems_E,&m.brems_V,&m.log_stop_T};
    for(int i=0;i<33;++i)*fields[i]=data+q[4*i];
    m.nstop=q[4*5+1];m.ncollision=q[4*7+1];m.nphoto=q[4*14+1];m.npair=q[4*16+1];m.nz=q[4*3+1];
    m.nline=q[4*28+3]; // lines shape (element, shell, line, 3); supplied as line stride below
    P5Tables &p=m.p;p.photon=data+q[133]+37;p.constants=constants;p.loget=m.log_stop_T;p.ed=m.stop_delta;p.ne=m.nstop;
    p.ts=m.gs_T;p.ks=m.gs_K;p.us=m.gs_u;p.ys=m.gs_y;p.rates=m.rates;
    p.nt=q[4*9+1];p.nk=q[4*10+1];p.nu=q[4*11+1];p.nrates=q[4*13+2];p.ratecols=q[4*13+3];
    p.pes=m.pair_E;p.np=m.npair;p.thresholds=m.thresholds;p.dl=m.dl;p.bpar=*m.bpar;p.delcm=*m.delcm;p.delpos=*m.delpos;
    p.Z=m.z;p.NJ=m.nj;p.nz=m.nz;p.brems_e=m.brems_E;p.brems_v=m.brems_V;p.nbrems=q[4*30+1];
    return m;
}
P5_FN int p8f_upper(double x,const double *a,int n) {
    int lo=0,hi=n;while(lo<hi){int j=(lo+hi)/2;if(x<a[j])hi=j;else lo=j+1;}return lo;
}
// Numba's np.interp chooses the final duplicate. Search the rounded log grid.
P5_FN double p8f_log_interp(double E,const double *x,const double *y,int n) {
    double l=log(E);if(l<log(x[0]))return exp(log(y[0]));if(l>log(x[n-1]))return exp(log(y[n-1]));
    int lo=0,hi=n;while(lo<hi){int j=(lo+hi)/2;if(l<log(x[j]))hi=j;else lo=j+1;}
    int j=lo-1;if(l==log(x[j]) || j==n-1)return exp(log(y[j]));
    double slope=(log(y[j+1])-log(y[j]))/(log(x[j+1])-log(x[j]));
    return exp(slope*(l-log(x[j]))+log(y[j]));
}
P5_FN double p8f_collision(double T,const P8Material &m) {
    if(m.ncollision)return p8f_log_interp(T,m.stop_T,m.stop_collision,m.nstop);
    double M=m.p.constants[0],tau=T/M,g=1+tau,b2=1-1/(g*g),d=tau/2;
    double F=-1-b2+log((tau-d)*d)+tau/(tau-d)+(d*d/2+(2*tau+1)*log1p(-d/tau))/(g*g);
    double im=*m.excitation/M;
    return *m.c/b2*(log(2*(tau+2)/(im*im))+F-interp(log(T),m.log_stop_T,m.stop_delta,m.nstop));
}
P5_FN void p8f_hard(double T,const P8Material &m,double *rate,double *stop) {
    *rate=0.;*stop=0.;if(T<=2*.01)return;
    double g=1+T/m.p.constants[0],gg=g*g,b2=1-1/gg,a=(g-1)*(g-1)/gg,b=(2*g-1)/gg,c=*m.c/(b2*T),x=.01/T;
    if(.5-x<1e-7) {double h=(T/2-.01)/T,mid=.5-h/2,om=1-mid;
        double v=a+1/(mid*mid)+1/(om*om)-b/(mid*(1-mid));*rate=c*h*v;*stop=c*T*h*mid*v;return;}
    *rate=c*(p5_primitive0(.5,a,b)-p5_primitive0(x,a,b));*stop=c*T*(p5_primitive1(.5,a,b)-p5_primitive1(x,a,b));
}
P5_FN double p8f_annihilation_rate(double T,const P8Material &m) {
    double g=1+T/m.p.constants[0],p=sqrt(g*g-1),RE=m.p.constants[3];
    double cross=3.141592653589793*(RE*RE)/(g+1)*((g*g+4*g+1)/(g*g-1)*log(g+p)-(g+3)/p);return *m.ne*cross;
}

P5_FN void p8f_quantities(double T,const P8Material &m,double *q) {
    int idx;double raw[12];quantities(T,m.brems_E,m.brems_V,m.p.nbrems,m.log_stop_T,m.stop_rad,m.nstop,.01,0,q,q+1,q+2,q+3,&idx,raw);
    if(T>.01 && m.ncollision)q[2]=interp(log(T),m.log_stop_T,m.stop_rad,m.nstop)-q[1];
}
P5_FN double p8f_sample_k(double T,const P8Material &m,RngState *s,int *excess) {
    double q[4];p8f_quantities(T,m,q);double lo=log(.01),width=log(T/.01);*excess=0;
    for(;;){double u=rng_uniform(s),k=p5_min(T,p5_max(.01,exp(lo+width*u)));double f=p5_shape(T,k,&m.p);
        if(f>q[3]){*excess=1;return k;}double accept=rng_uniform(s);if(accept*q[3]<=f)return k;}
}




#define G5_PHOTON_P8F
#include "photon_physics.h"
// t2_materials.h is a frozen reference; active helpers live here/shared photons.
#define P8G_MATERIALS_H
#include "t2_functions.h"
P5_FN int p8g_upper(double x,const double *a,int n) {
    int lo=0,hi=n;while(lo<hi){int j=(lo+hi)/2;if(x<a[j])hi=j;else lo=j+1;}return lo;
}
// Numba's np.interp chooses the final duplicate. Search the rounded log grid.
P5_FN double p8g_log_interp(double E,const double *x,const double *y,int n,const P8gMaterial &m8,int sx,int sy) {
    double l=log(E);if(l<p8g_read(m8,sx,0))return exp(p8g_read(m8,sy,0));if(l>p8g_read(m8,sx,n-1))return exp(p8g_read(m8,sy,n-1));
    int lo=0,hi=n;while(lo<hi){int j=(lo+hi)/2;if(l<p8g_read(m8,sx,j))hi=j;else lo=j+1;}
    int j=lo-1;if(l==p8g_read(m8,sx,j) || j==n-1)return exp(p8g_read(m8,sy,j));
    double slope=(p8g_read(m8,sy,j+1)-p8g_read(m8,sy,j))/(p8g_read(m8,sx,j+1)-p8g_read(m8,sx,j));
    return exp(slope*(l-p8g_read(m8,sx,j))+p8g_read(m8,sy,j));
}
P5_FN double p8g_collision(double T,const P8gMaterial &m) {
    if(m.ncollision)return p8g_log_interp(T,m.stop_T,m.stop_collision,m.nstop,m,8,9);
    double M=m.p.constants[0],tau=T/M,g=1+tau,b2=1-1/(g*g),d=tau/2;
    double F=-1-b2+log((tau-d)*d)+tau/(tau-d)+(d*d/2+(2*tau+1)*log1p(-d/tau))/(g*g);
    double im=*m.excitation/M;
    return *m.c/b2*(log(2*(tau+2)/(im*im))+F-interp(log(T),m.log_stop_T,m.stop_delta,m.nstop));
}
P5_FN void p8g_hard(double T,const P8gMaterial &m,double *rate,double *stop) {
    *rate=0.;*stop=0.;if(T<=2*.01)return;
    double g=1+T/m.p.constants[0],gg=g*g,b2=1-1/gg,a=(g-1)*(g-1)/gg,b=(2*g-1)/gg,c=*m.c/(b2*T),x=.01/T;
    if(.5-x<1e-7) {double h=(T/2-.01)/T,mid=.5-h/2,om=1-mid;
        double v=a+1/(mid*mid)+1/(om*om)-b/(mid*(1-mid));*rate=c*h*v;*stop=c*T*h*mid*v;return;}
    *rate=c*(p5_primitive0(.5,a,b)-p5_primitive0(x,a,b));*stop=c*T*(p5_primitive1(.5,a,b)-p5_primitive1(x,a,b));
}
P5_FN double p8g_annihilation_rate(double T,const P8gMaterial &m) {
    double g=1+T/m.p.constants[0],p=sqrt(g*g-1),RE=m.p.constants[3];
    double cross=3.141592653589793*(RE*RE)/(g+1)*((g*g+4*g+1)/(g*g-1)*log(g+p)-(g+3)/p);return *m.ne*cross;
}

P5_FN void p8g_quantities(double T,const P8gMaterial &m,double *q) {
    int idx;double raw[12];p8g_quantities_raw(T,m.brems_E,m.brems_V,m.p.nbrems,m.log_stop_T,m.stop_rad,m.nstop,.01,0,q,q+1,q+2,q+3,&idx,raw,m);
    if(T>.01 && m.ncollision)q[2]=interp(log(T),m.log_stop_T,m.stop_rad,m.nstop)-q[1];
}
P5_FN double p8g_sample_k(double T,const P8gMaterial &m,RngState *s,int *excess) {
    double q[4];p8g_quantities(T,m,q);double lo=log(.01),width=log(T/.01);*excess=0;
    for(;;){double u=rng_uniform(s),k=p5_min(T,p5_max(.01,exp(lo+width*u)));double f=p8g_shape(T,k,m);

#if P8G_CHECK
        if(!isfinite(f)){*excess=1;return k;}
#endif
        if(f>q[3]){*excess=1;return k;}double accept=rng_uniform(s);if(accept*q[3]<=f)return k;}
}





#define G5_PHOTON_P8G
#include "photon_physics.h"
#endif
