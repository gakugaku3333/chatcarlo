// Philox4x32-10 definition: DEShawResearch/random123,
// commit 9545ff6413f258be2f04c1d319d99aaef7521150,
// include/Random123/philox.h; independently expressed round arithmetic.
#ifdef P5_GPU
#define P5_FN __device__ inline
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#else
#include <cmath>
#include <cstdint>
#define P5_FN static inline
#endif
static_assert(sizeof(uint32_t)==4, "32-bit word required");
static_assert(sizeof(uint64_t)==8, "64-bit word required");
#ifndef P5_CAP
#define P5_CAP 16
#endif
#ifndef P5_FAULT
#define P5_FAULT 0
#endif
P5_FN void philox(uint64_t seed, uint64_t history, uint64_t count, uint32_t *out) {
    uint32_t x0=uint32_t(history), x1=uint32_t(history>>32);
    uint32_t x2=uint32_t(count), x3=uint32_t(count>>32);
    uint32_t k0=uint32_t(seed), k1=uint32_t(seed>>32);
    for(int r=0;r<10;++r) {
        uint64_t p=uint64_t(0xD2511F53U)*x0, q=uint64_t(0xCD9E8D57U)*x2;
        uint32_t y0=uint32_t(q>>32)^x1^k0, y2=uint32_t(p>>32)^x3^k1;
        x0=y0; x1=uint32_t(q); x2=y2; x3=uint32_t(p);
        k0+=0x9E3779B9U; k1+=0xBB67AE85U;
    }
    out[0]=x0;out[1]=x1;out[2]=x2;out[3]=x3;
}
P5_FN double uniform(const uint32_t *w) {
    return ((w[0]>>5)*67108864.0+(w[1]>>6))/9007199254740992.0;
}
P5_FN double draw(uint64_t seed,uint64_t h,uint64_t *c) {
    uint32_t w[4];philox(seed,h,*c,w);++*c;return uniform(w);
}
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
struct Particle {double s[23];int generation;};
P5_FN void toy(uint64_t seed,uint64_t h,uint64_t *meta,double *bins) {
    // Depth-first bounded binary tree (at most 511 nodes). State entries are
    // exact small integers; all 23 contribute to both the bin and deposit.
    Particle stack[P5_CAP];int top=1,maxdepth=1,steps=0,status=0;
    uint64_t count=0;
    for(int j=0;j<23;++j)stack[0].s[j]=double(j+1);
    stack[0].generation=0;for(int b=0;b<50;++b)bins[b]=0.;
#if P5_FAULT == 3
    double unused=draw(seed,h,&count);(void)unused;
#endif
    while(top>0) {
        if(steps==1000){status=2;break;}
        Particle p=stack[--top];++steps;
        double sum=0.;
        for(int j=0;j<23;++j) {
            if(!(p.s[j]>=0. && p.s[j]<=1.e12)){status=3;break;}
            sum+=(j+1)*p.s[j];
        }
        if(status)break;
        int bin=int(sum)%50;double deposit=(int(sum)%1024+1)/1024.0;
#if P5_FAULT == 1
        bin=(bin+1)%50;
#endif
#if P5_FAULT == 2
        if(steps%10!=0)bins[bin]+=deposit;
#else
        bins[bin]+=deposit;
#endif
        double u=draw(seed,h,&count);
        int children=p.generation<8?(u<0.05?0:(u<0.15?1:2)):0;
        for(int k=0;k<children;++k) {
            if(top==P5_CAP){status=1;break;}
            Particle child;child.generation=p.generation+1;
            for(int j=0;j<23;++j)child.s[j]=p.s[j]+(k+1)*(j+1)+p.s[(j+1)%23];
            stack[top++]=child;if(top>maxdepth)maxdepth=top;
        }
        if(status)break;
    }
    meta[0]=count;meta[1]=steps;meta[2]=maxdepth;meta[3]=status;
}
