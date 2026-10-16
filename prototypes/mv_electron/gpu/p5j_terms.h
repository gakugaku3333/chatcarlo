#ifndef P5J_TERMS_H
#define P5J_TERMS_H
#include "p5j_logs.h"
#ifndef P5_GPU
#include <string.h>
#endif // NVRTC provides the CUDA runtime builtins, including memcpy.
P5_FN void p5j_count(const P5jTables *t,int site) {
#if P5J_CHECK
#ifdef P5_GPU
    atomicAdd(t->log_counts+18+2*site,1ULL);
#else
    ++t->log_counts[18+2*site];
#endif
#endif
}
P5_FN double p5j_constant(const P5jTables *t,int k,int site) {
    p5j_count(t,site);return t->logs[p5j_offset(t,8)+p5j_length(t,8)+k];
}
P5_FN double p5j_log(double arg,double Lsrc,double Lval,const P5jTables *t,int site) {
    unsigned long long a,b;memcpy(&a,&arg,8);memcpy(&b,&Lsrc,8);
    bool reuse=!isnan(Lsrc) && a==b;p5j_count(t,site+int(!reuse));
    return reuse?Lval:log(arg);
}
P5_FN double p5j_c_log(double tau,double d,const P5jTables *t) {
    double arg=-d/tau;double K2=t->logs[p5j_offset(t,8)+p5j_length(t,8)+2];
    p5j_count(t,arg==-0.5?3:4);
    double L=(arg==-0.5)?K2:log1p(arg);return L;
}
P5_FN double p5j_collision(double T,const P5jTables *t,double Lsrc,double Lval) {
    double M=t->constants[0],C=t->constants[1],I=t->constants[2];
    double tau=T/M,g=1+tau,b2=1-1/(g*g),d=tau/2;
    double F=-1-b2+log((tau-d)*d)+tau/(tau-d)
        +(d*d/2+(2*tau+1)*p5j_c_log(tau,d,t))/(g*g);
    double density=interp(p5j_log(T,Lsrc,Lval,t,9),t->loget,t->ed,t->ne);
    double im=I/M;
    return C/b2*(log(2*(tau+2)/(im*im))+F-density);
}
P5_FN double p5j_primitive0(double x,double a,double b,const P5jTables *t) {
    return a*x-1/x+1/(1-x)-b*p5j_constant(t,0,0);
}
P5_FN double p5j_primitive1(double x,double a,double b,const P5jTables *t) {
    return a*x*x/2+p5j_constant(t,1,1)+1/(1-x)+(1+b)*p5j_constant(t,2,2);
}
P5_FN void p5j_hard(double T,const P5jTables *t,double *rate,double *stop) {
    *rate=0.;*stop=0.;if(T<=2*.01)return;
    double a,b,c;p5_moller_params(T,t,&a,&b,&c);double x=.01/T;
    if(.5-x<1e-7) {
        double h=(T/2-.01)/T,mid=.5-h/2,om=1-mid;
        double v=a+1/(mid*mid)+1/(om*om)-b/(mid*(1-mid));
        *rate=c*h*v;*stop=c*T*h*mid*v;return;
    }
    *rate=c*(p5j_primitive0(.5,a,b,t)-p5_primitive0(x,a,b));
    *stop=c*T*(p5j_primitive1(.5,a,b,t)-p5_primitive1(x,a,b));
}

#endif
