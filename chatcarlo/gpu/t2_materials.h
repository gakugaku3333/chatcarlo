// Derived from p8g_materials.h; G5 A2 standalone copy.
#ifndef P8G_MATERIALS_H
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
P5_FN void p8g_coefficients(double E,const P8gMaterial &m,double *c) {
    double x=E/m.p.constants[0],L=log1p(2*x),RE=m.p.constants[3],one=1+2*x;
    c[0]=2*3.141592653589793*(RE*RE)*(*m.ne)*((1+x)/(x*x)*(2*(1+x)/(1+2*x)-L/x)+L/(2*x)-(1+3*x)/(one*one));
    double pe=p8g_log_interp(E,m.photo_E,m.photo,m.nphoto,m,10,11);int i=p8g_upper(E,m.photo_E,m.nphoto)-1;
    if(i>=0 && i<m.nphoto-1) {
        bool near=(i>0 && m.photo_E[i]==m.photo_E[i-1]) || (i+2<m.nphoto && m.photo_E[i+1]==m.photo_E[i+2]);
        if(near) {
            int left=i,right=i+1;
#if P8G_FAULT == 4
            if(i>0 && m.photo_E[i]==m.photo_E[i-1])--left;
            if(i+2<m.nphoto && m.photo_E[i+1]==m.photo_E[i+2])++right;
#endif
            double u=log(E/m.photo_E[i])/p8g_read(m,12,i);
            pe=exp(p8g_read(m,14,left)+u*p8g_read(m,13,left));
        }
    }
    c[1]=pe;c[2]=p8g_pair_interp(E,m.pair_E,m.pair_n,m.npair,1,m.thresholds[0],m,6);
    c[3]=p8g_pair_interp(E,m.pair_E,m.pair_e,m.npair,1,m.thresholds[1],m,7);
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
P5_FN void p8g_shell_distribution(double E,const P8gMaterial &m,double *xs) {
    double l=log(E*1000),total=0.;
    for(int i=0;i<m.nz;++i) {
        for(int s=0;s<5;++s) {const double *q=m.pm+(i*5+s)*4;double v=0.;
            if(E>m.bindings[i*5+s] && (s==0 || q[0]!=0))v=exp(q[0]+q[1]*l+q[2]*l*l+q[3]*l*l*l);
            if(E<=m.bindings[i*5+4])v=0.;xs[i*5+s]=v*m.shell_nj[i];}
        if(E<=m.bindings[i*5+3]){double row=0.;for(int s=0;s<5;++s)row+=xs[i*5+s];for(int s=0;s<5;++s)xs[i*5+s]=0.;xs[i*5+4]=row;}
    }
    for(int j=0;j<m.nz*5;++j)total+=xs[j];if(total>0)for(int j=0;j<m.nz*5;++j)xs[j]/=total;
}
P5_FN void p8g_choose_shell(const double *xs,int nz,double r,int *element,int *shell){
    *element=-1;*shell=-1;for(int i=0;i<nz;++i)for(int j=0;j<5;++j)if(xs[i*5+j]>0){*element=i;*shell=j;if(r<xs[i*5+j])return;r-=xs[i*5+j];}
}
P5_FN void p8g_select_shell(double E,const P8gMaterial &m,RngState *s,int *element,int *shell) {
    double xs[150];p8g_shell_distribution(E,m,xs);double total=0.;for(int j=0;j<m.nz*5;++j)total+=xs[j];
    *element=-1;*shell=-1;if(total==0)return;double r=rng_uniform(s);
    p8g_choose_shell(xs,m.nz,r,element,shell);
}
P5_FN int p8g_relax(int element,int shell,const P8gMaterial &m,RngState *s,bool enabled,double *out,double *local,long long *trace,int *events) {
    int queue[8],top=0,used=0;queue[0]=shell;double emitted=0.;*events=0;
    for(int j=0;j<24;++j)trace[j]=-1;for(int j=0;j<32;++j)out[j]=0.;
    while(top>=0){int sh=queue[top--];if(sh>=4 || !enabled)continue;
        double u=rng_uniform(s);if(u>=m.yields[element*4+sh]){trace[3*(*events)]=sh;++*events;continue;}
        double r=rng_uniform(s);int j=0;const double *lines=m.lines+((element*4+sh)*m.nline)*4;
        while(j<int(m.line_counts[element*4+sh])-1 && r>=lines[j*4+1]){r-=lines[j*4+1];++j;}
        double E=lines[j*4];int child=int(lines[j*4+2]);trace[3*(*events)]=sh;trace[3*(*events)+1]=j;trace[3*(*events)+2]=child;++*events;
        if(E>=.01){double v[3];p5_annihilation(s,v);out[4*used]=E;for(int k=0;k<3;++k)out[4*used+1+k]=v[k];++used;emitted+=E;}
        if(child<4)queue[++top]=child;
    }
    *local=m.bindings[element*5+shell]-emitted;return used;
}
#endif
