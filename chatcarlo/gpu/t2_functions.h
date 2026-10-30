// Derived from p8g_functions.h; G5 A2 standalone copy.
#ifndef P8G_FUNCTIONS_H
#define P8G_FUNCTIONS_H
#include "t2_logs.h"
P5_FN double p8g_pair_interp(double E,const double *es,const double *vs,int n,int stride,double th,const P8gMaterial &m8,int site) {
    if(E<=th)return 0.;
    int start=0;while(start<n && es[start]<=th)++start;
    double x=log(E),y;
    if(x<p8g_read(m8,5,start))y=p8g_read(m8,site,start);
    else if(x>p8g_read(m8,5,n-1))y=p8g_read(m8,site,n-1);
    else {
        int lo=start,hi=n;
        while(lo<hi) {int m=lo+(hi-lo)/2;if(p8g_read(m8,5,m)<x)lo=m+1;else hi=m;}
        int j=lo;
        if(j<n && x==p8g_read(m8,5,j))y=p8g_read(m8,site,j);
        else {
            --j;double y0=p8g_read(m8,site,j);
            double y1=p8g_read(m8,site,j+1);
            double slope=(y1-y0)/(p8g_read(m8,5,j+1)-p8g_read(m8,5,j));
            y=slope*(x-p8g_read(m8,5,j))+y0;
        }
    }
    double q=1-th/E;return exp(y)*((q*q)*q);
}
P5_FN void p8g_quantities_raw(double t,const double *e,const double *v,int n,
    const double *le,const double *er,int m,double cut,int disabled,
    double *rate,double *discrete,double *soft,double *env,int *idx,double *raw,const P8gMaterial &m8) {
    *idx=-1;for(int j=0;j<12;++j)raw[j]=0.;
    double rad=interp(log(t>cut?t:cut),le,er,m);
    *rate=0.;*discrete=0.;*soft=rad;*env=0.;
    if(disabled || t<=cut)return;
    int i=lower_bound(t,e,n)-1;if(i<0)i=0;if(i>n-2)i=n-2;*idx=i;
    raw[0]=e[i];raw[1]=e[i+1];
    for(int j=0;j<5;++j){raw[2+j]=v[5*i+j];raw[7+j]=v[5*(i+1)+j];}
    double u=(log(t)-p8g_read(m8,0,i))/(p8g_read(m8,0,i+1)-p8g_read(m8,0,i));
    if(i==0) {
        *rate=u*v[5];*discrete=u*v[6];*soft=rad-*discrete;
        *env=u*v[8]+(1-u)*v[3];return;
    }
    *rate=exp((1-u)*p8g_read(m8,1,3*i+0)+u*p8g_read(m8,1,3*(i+1)+0));
    *soft=exp((1-u)*p8g_read(m8,1,3*i+2)+u*p8g_read(m8,1,3*(i+1)+2));
    *discrete=exp((1-u)*p8g_read(m8,1,3*i+1)+u*p8g_read(m8,1,3*(i+1)+1));
    *env=(1-u)*v[5*i+3]+u*v[5*(i+1)+3];
}
P5_FN void p8g_table_rates(double T,const P8gMaterial &m8,double *total,double *G1) {
 const P5Tables *t=&m8.p;
    int shift=t->ratecols==3?1:0,n=shift?t->nrates:t->nt,lo=0,hi=n;
    while(lo<hi) {int m=lo+(hi-lo)/2;double e=shift?t->rates[m*t->ratecols]:t->ts[m];
        if(e<T)lo=m+1;else hi=m;}
    int i=lo-1;if(i<0)i=0;if(i>n-2)i=n-2;
    double e0=shift?t->rates[i*t->ratecols]:t->ts[i];
    double e1=shift?t->rates[(i+1)*t->ratecols]:t->ts[i+1];
    double a=log(T/e0)/p8g_read(m8,2,i);
    *total=exp((1-a)*p8g_read(m8,3,2*i+0)+a*p8g_read(m8,3,2*(i+1)+0));
    *G1=exp((1-a)*p8g_read(m8,3,2*i+1)+a*p8g_read(m8,3,2*(i+1)+1));
}
P5_FN double p8g_gs_mu_cached(double T,double K,const P8gMaterial &m8,double total,double G1,RngState *s) {
 const P5Tables *t=&m8.p;
    double p0=exp(-K/G1*total),zero=rng_uniform(s);
    if(zero<p0)return 1.;
    double u=rng_uniform(s);int i=lower_bound(T,t->ts,t->nt)-1;
    if(i<0)i=0;if(i>t->nt-2)i=t->nt-2;
    double a=log(T/t->ts[i])/p8g_read(m8,4,i);int j,h;double b,c;
    p5_bracket(t->ks,t->nk,K,&j,&b);p5_bracket(t->us,t->nu,u,&h,&c);
    int q00=(i*t->nk+j)*t->nu+h,q01=q00+t->nu;
    int q10=q00+t->nk*t->nu,q11=q10+t->nu;
    double y00=(1-c)*t->ys[q00]+c*t->ys[q00+1];
    double y01=(1-c)*t->ys[q01]+c*t->ys[q01+1];
    double y10=(1-c)*t->ys[q10]+c*t->ys[q10+1];
    double y11=(1-c)*t->ys[q11]+c*t->ys[q11+1];
    return 1-((1-a)*((1-b)*y00+b*y01)+a*((1-b)*y10+b*y11));
}
P5_FN double p8g_shape(double T,double k,const P8gMaterial &m8) {
 const P5Tables *t=&m8.p;
    if(k<0. || k>T)return 0.;
    double M=t->constants[0],RE=t->constants[3],alpha=t->constants[5];
    double E=T+M,rem=E-k,y=rem/E,result=0.;
    for(int j=0;j<t->nz;++j) {
        double z=t->Z[j],az=alpha*z,az2=az*az;
        double fc=az2*(1/(1+az2)+.20206+az2*(-.0369+az2*(.0083-.002*az2)));
        double xi=z==1?6.144/(5.31-fc):p8g_read(m8,15,j)/(p8g_read(m8,16,j)-fc);
        double xl=4./3.*p8g_read(m8,17,j),d=136.*pow(z,-1./3.)*M/E*k/rem;
        if(d>=exp((21.12-xl)/4.184)-.952)continue;
        double s1,s2;
        if(d<=1) {s1=20.867+d*(-3.242+.625*d)-xl;s2=20.209+d*(-1.930-.086*d)-xl;}
        else {s1=21.12-4.184*log(d+.952)-xl;s2=s1;}
        result+=t->NJ[j]*(RE*RE)*alpha*z*(z+xi)*((1+y*y)*s1-2./3.*y*s2);
    }
    return result;
}
#endif
