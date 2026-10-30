// Derived from p5_rng.h; G5 A2 standalone copy.
#ifndef P5_RNG_H
#define P5_RNG_H
#include "g5_base.h"

P5_FN int lower_bound(double x,const double *a,int n) {
    int lo=0,hi=n;
    while(lo<hi) {int m=lo+(hi-lo)/2;if(a[m]<x)lo=m+1;else hi=m;}
    return lo;
}
P5_FN double interp(double x,const double *a,const double *v,int n) {
    // Numba np_interp_impl_inner: slope first, multiply, then add.
    if(x<a[0])return v[0];if(x>a[n-1])return v[n-1];
    int j=lower_bound(x,a,n);
    if(j<n && x==a[j])return v[j];
    --j;
    double slope=(v[j+1]-v[j])/(a[j+1]-a[j]);
    return slope*(x-a[j])+v[j];
}
P5_FN void quantities(double t,const double *e,const double *v,int n,
    const double *le,const double *er,int m,double cut,int disabled,
    double *rate,double *discrete,double *soft,double *env,int *idx,double *raw) {
    *idx=-1;for(int j=0;j<12;++j)raw[j]=0.;
    double rad=interp(log(t>cut?t:cut),le,er,m);
    *rate=0.;*discrete=0.;*soft=rad;*env=0.;
    if(disabled || t<=cut)return;
    int i=lower_bound(t,e,n)-1;if(i<0)i=0;if(i>n-2)i=n-2;*idx=i;
    raw[0]=e[i];raw[1]=e[i+1];
    for(int j=0;j<5;++j){raw[2+j]=v[5*i+j];raw[7+j]=v[5*(i+1)+j];}
    double u=(log(t)-log(e[i]))/(log(e[i+1])-log(e[i]));
    if(i==0) {
        *rate=u*v[5];*discrete=u*v[6];*soft=rad-*discrete;
        *env=u*v[8]+(1-u)*v[3];return;
    }
    *rate=exp((1-u)*log(v[5*i])+u*log(v[5*(i+1)]));
    *soft=exp((1-u)*log(v[5*i+2])+u*log(v[5*(i+1)+2]));
    *discrete=exp((1-u)*log(v[5*i+1])+u*log(v[5*(i+1)+1]));
    *env=(1-u)*v[5*i+3]+u*v[5*(i+1)+3];
}
P5_FN void maths(const double *x,double *out) {
    double u=x[0],u2=x[1],a=x[2],b=x[3],c=x[4];
    out[0]=log(u);out[1]=exp(-20*u);out[2]=sqrt(u);
    out[3]=sin(6.283185307179586*u);out[4]=cos(6.283185307179586*u);
    out[5]=pow(u+0.1,1.3);out[6]=atan2(u-0.5,u2);out[7]=acos(2*u-1);
    out[8]=a*b+c*u-b/(1.0+a);
}

// MT is verification-only and is excluded from GPU compilation.
#if defined(P5_RNG_MT)
#ifdef P5_GPU
#error MT19937 is CPU-only
#endif
struct RngState {uint32_t mt[624];int index;uint64_t calls;};
P5_FN void rng_init(RngState *s,uint64_t seed,uint64_t history) {
    (void)history;s->mt[0]=uint32_t(seed);s->index=624;s->calls=0;
    for(int i=1;i<624;++i)
        s->mt[i]=1812433253U*(s->mt[i-1]^(s->mt[i-1]>>30))+uint32_t(i);
}
P5_FN uint32_t rng_word(RngState *s) {
    if(s->index==624) {
        for(int j=0;j<624;++j) {
            uint32_t y=(s->mt[j]&0x80000000U)|(s->mt[(j+1)%624]&0x7fffffffU);
            s->mt[j]=s->mt[(j+397)%624]^(y>>1)^((y&1U)?0x9908b0dfU:0U);
        }
        s->index=0;
    }
    uint32_t y=s->mt[s->index++];y^=y>>11;y^=(y<<7)&0x9d2c5680U;
    y^=(y<<15)&0xefc60000U;y^=y>>18;return y;
}
P5_FN double rng_uniform(RngState *s) {
    uint32_t w[2];w[0]=rng_word(s);w[1]=rng_word(s);++s->calls;
    return uniform(w);
}
#elif defined(P5_RNG_PHILOX)
struct RngState {uint64_t seed,history,calls;};
P5_FN void rng_init(RngState *s,uint64_t seed,uint64_t history) {
    s->seed=seed;s->history=history;s->calls=0;
}
P5_FN double rng_uniform(RngState *s) {
    return draw(s->seed,s->history,&s->calls);
}
#else
#error Select P5_RNG_MT or P5_RNG_PHILOX
#endif
#endif
