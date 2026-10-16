#ifndef P5I_LOGS_H
#define P5I_LOGS_H
#include "p5_kernel.h"
#ifndef P5I_CHECK
#define P5I_CHECK 0
#endif
struct P5iTables : P5Tables {double *logs;unsigned long long *log_counts;};
// Array IDs: brems_e, brems_v[0:3], rates_ratio, rates_values,
// ts_ratio, pes, water[2], pair(threshold0), pair(threshold1).
P5_FN int p5i_length(const P5Tables *t,int site) {
    int n=t->ratecols==3?t->nrates:t->nt;
    return site==0?t->nbrems:site==1?3*t->nbrems:site==2?n-1:
        site==3?2*n:site==4?t->nt-1:t->np;
}
P5_FN int p5i_offset(const P5Tables *t,int site) {
    int off=0;for(int s=0;s<site;++s)off+=p5i_length(t,s);return off;
}
P5_FN bool p5i_valid(const P5Tables *t,int site,int j) {
    return !(site==1 && j<3) && !(site>=7 && t->pes[j]<=t->thresholds[site-7]);
}
P5_FN double p5i_expression(const P5Tables *t,int site,int j,double *record=0) {
    if(site==0) {if(record)record[4]=t->brems_e[j];return log(t->brems_e[j]);}
    if(site==1) {if(record)record[4]=t->brems_v[5*(j/3)+j%3];return log(t->brems_v[5*(j/3)+j%3]);}
    if(site==2) {
        int i=j,shift=t->ratecols==3?1:0;
        double e0=shift?t->rates[i*t->ratecols]:t->ts[i];
        double e1=shift?t->rates[(i+1)*t->ratecols]:t->ts[i+1];
        if(record)record[1]=record[4]=e1/e0;return log(e1/e0);
    }
    if(site==3) {int i=j/2,shift=t->ratecols==3?1:0;
        if(record)record[4]=t->rates[i*t->ratecols+shift+j%2];return log(t->rates[i*t->ratecols+shift+j%2]);}
    if(site==4) {int i=j;if(record)record[1]=record[4]=t->ts[i+1]/t->ts[i];return log(t->ts[i+1]/t->ts[i]);}
    if(site==5) {if(record)record[4]=t->pes[j];return log(t->pes[j]);}
    if(site==6) {if(record)record[4]=t->water[j*t->watercols+2];return log(t->water[j*t->watercols+2]);}
    const double *es=t->pes,*vs=t->water+3+(site-7);int stride=t->watercols;
    double th=t->thresholds[site-7];double q=1-th/es[j];
    if(record) {record[2]=q;record[3]=(q*q)*q;record[1]=record[4]=vs[j*stride]/((q*q)*q);}
    return log(vs[j*stride]/((q*q)*q));
}
P5_FN void p5i_initialize(P5iTables *t) {
    for(int site=0;site<9;++site)for(int j=0;j<p5i_length(t,site);++j)
        t->logs[p5i_offset(t,site)+j]=p5i_valid(t,site,j)?p5i_expression(t,site,j):NAN;
}
P5_FN double p5i_read(const P5iTables *t,int site,int j) {
#if P5I_CHECK
#ifdef P5_GPU
    atomicAdd(t->log_counts+2*site,1ULL);
    if(!p5i_valid(t,site,j))atomicAdd(t->log_counts+2*site+1,1ULL);
#else
    ++t->log_counts[2*site];if(!p5i_valid(t,site,j))++t->log_counts[2*site+1];
#endif
#endif
    return t->logs[p5i_offset(t,site)+j];
}

// Auxiliary original-expression records: log value, quotient, q, q^3, log argument.
// Invalid slots remain NaN and are not evaluated. These records are not gate0 criteria.
P5_FN void p5i_auxiliary(const P5iTables *t,double *out) {
    for(int site=0;site<9;++site)for(int j=0;j<p5i_length(t,site);++j) {
        int o=p5i_offset(t,site)+j;double *v=out+5*o;
        for(int z=0;z<5;++z)v[z]=NAN;
        if(!p5i_valid(t,site,j))continue;
        double arg;
        if(site==0)arg=t->brems_e[j];
        else if(site==1)arg=t->brems_v[5*(j/3)+j%3];
        else if(site==2) {
            int i=j,shift=t->ratecols==3?1:0;
            double e0=shift?t->rates[i*t->ratecols]:t->ts[i];
            double e1=shift?t->rates[(i+1)*t->ratecols]:t->ts[i+1];
            arg=e1/e0;v[1]=arg;
        } else if(site==3) {int i=j/2,shift=t->ratecols==3?1:0;arg=t->rates[i*t->ratecols+shift+j%2];}
        else if(site==4) {int i=j;arg=t->ts[i+1]/t->ts[i];v[1]=arg;}
        else if(site==5)arg=t->pes[j];
        else if(site==6)arg=t->water[j*t->watercols+2];
        else {
            const double *es=t->pes,*vs=t->water+3+(site-7);int stride=t->watercols;
            double th=t->thresholds[site-7];double q=1-th/es[j];
            v[2]=q;v[3]=(q*q)*q;arg=vs[j*stride]/((q*q)*q);v[1]=arg;
        }
        v[0]=log(arg);v[4]=arg;
    }
}
// Re-evaluate the initializer, independently of the rewritten original diagnostic.
P5_FN void p5i_initialization_records(const P5iTables *t,double *out) {
    for(int site=0;site<9;++site)for(int j=0;j<p5i_length(t,site);++j) {
        double *v=out+5*(p5i_offset(t,site)+j);
        for(int z=0;z<5;++z)v[z]=NAN;
        if(p5i_valid(t,site,j))v[0]=p5i_expression(t,site,j,v);
    }
}
#endif
