#ifndef G2_TALLY_H
#define G2_TALLY_H
#include "g1_common.h"
struct G2Tally {
    const double *table,*cfg; int nm;
    double *kerma,*h10,*count,*sum,*sum2,*box_count;
};
P5_FN void g2_add(double *p,double v) {
    if(!p)return;
#ifdef P5_GPU
    atomicAdd(p,v);
#else
    *p+=v;
#endif
}
P5_FN int g2_interval(const double *x,int n,double e) {
    int lo=0,hi=n-1;
    while(hi-lo>1){int mid=(lo+hi)/2;if(e<x[mid])hi=mid;else lo=mid;}
    return lo;
}
P5_FN double g2_muen(const G2Tally *t,int m,double e) {
    int s=int(t->cfg[0]),n=int(t->cfg[2+m]);const double *x=t->table+m*s;
    double le=log(e);int i=g2_interval(x,n,le);double u=le-x[i];
    const double *c=t->table+t->nm*s+m*4*s;
    return exp(((c[i]*u+c[s+i])*u+c[2*s+i])*u+c[3*s+i]);
}
P5_FN double g2_h10(const G2Tally *t,double e) {
    int s=int(t->cfg[0]),n=int(t->cfg[1]);const double *x=t->table+5*t->nm*s+t->nm,*y=x+n;
    double le=log(e);if(le<x[0])le=x[0];if(le>x[n-1])le=x[n-1];
    int i=g2_interval(x,n,le);double f=(le-x[i])/(x[i+1]-x[i]);
    return exp(y[i]+f*(y[i+1]-y[i]));
}
// The historical overlap uses endpoint displacement, including its 1e-12 cutoff.
P5_FN double g2_overlap(const double *a,const double *b,const double *lo,const double *hi) {
    double enter=0.,exit=1.,norm=0.;
    for(int k=0;k<3;++k){double d=b[k]-a[k];norm+=d*d;
        if(fabs(d)<1e-12){if(a[k]<lo[k] || a[k]>hi[k])return 0.;}
        else {double u=(lo[k]-a[k])/d,v=(hi[k]-a[k])/d;
            enter=fmax(enter,fmin(u,v));exit=fmin(exit,fmax(u,v));}}
    return fmax(exit-enter,0.)*sqrt(norm);
}
P5_FN void g2_grid_segment(const G2Tally *t,const double *o,const double *d,double len,double kw,double hw) {
    const double *cfg=t->cfg+2+t->nm;if(!int(cfg[0]))return;
    const double *lo=cfg+1;double h=cfg[7],enter=0.,exit=len;int shape[3];
    for(int k=0;k<3;++k){shape[k]=int(cfg[4+k]);double hi=lo[k]+shape[k]*h;
        if(fabs(d[k])<1e-12){if(o[k]<lo[k] || o[k]>hi)return;}
        else {double u=(lo[k]-o[k])/d[k],v=(hi-o[k])/d[k];
            enter=fmax(enter,fmin(u,v));exit=fmin(exit,fmax(u,v));}}
    if(exit<=enter)return;
    // Enumerate the same internal planes as tally.py, then merge the three
    // sorted streams. Midpoints decide cells (upper external face is excluded).
    int plane[3],last[3],step[3];double next[3];
    for(int k=0;k<3;++k){
        double a=(o[k]+enter*d[k]-lo[k])/h,b=(o[k]+exit*d[k]-lo[k])/h;
        int low=int(ceil(fmin(a,b)+1e-7)),high=int(floor(fmax(a,b)-1e-7));
        step[k]=d[k]>0?1:-1;plane[k]=d[k]>0?low:high;last[k]=d[k]>0?high:low;
        next[k]=(fabs(d[k])<1e-12 || high<low)?exit:(lo[k]+plane[k]*h-o[k])/d[k];
    }
    double left=enter;
    while(left<exit){double right=exit;for(int k=0;k<3;++k)right=fmin(right,next[k]);
        if(right>left){double mid=(left+right)/2.;int ix[3];bool valid=true;
            for(int k=0;k<3;++k){ix[k]=int(floor((o[k]+d[k]*mid-lo[k])/h));if(ix[k]<0 || ix[k]>=shape[k])valid=false;}
            if(valid){int j=(ix[0]*shape[1]+ix[1])*shape[2]+ix[2];double dl=right-left;
                g2_add(t->kerma+j,kw*dl);g2_add(t->h10+j,hw*dl);if(t->count)g2_add(t->count+j,1.);}}
        if(right>=exit)break;
        for(int k=0;k<3;++k)if(next[k]<=right){plane[k]+=step[k];
            bool done=step[k]>0?plane[k]>last[k]:plane[k]<last[k];
            next[k]=done?exit:(lo[k]+plane[k]*h-o[k])/d[k];}
        left=right;
    }
}
P5_FN void g2_box_segment(const G2Tally *t,int m,const double *o,const double *d,double len,double kw,double *local) {
    const double *cfg=t->cfg+2+t->nm;int nb=int(cfg[8]);if(m!=int(cfg[9]))return;
    double end[3];for(int k=0;k<3;++k)end[k]=o[k]+d[k]*len;
    for(int j=0;j<nb;++j)local[j]+=g2_overlap(o,end,cfg+10+6*j,cfg+13+6*j)*kw;
}
P5_FN void g2_segment(const G2Tally *t,int m,const double *o,const double *d,double len,double e,double *local) {
    double kw=e*g2_muen(t,m,e)*t->table[5*t->nm*int(t->cfg[0])+m];
    g2_grid_segment(t,o,d,len,kw,g2_h10(t,e));g2_box_segment(t,m,o,d,len,kw,local);
}
P5_FN void g2_end_history(const G2Tally *t,const double *local) {
    int nb=int(t->cfg[2+t->nm+8]);for(int j=0;j<nb;++j){
        g2_add(t->sum+j,local[j]);g2_add(t->sum2+j,local[j]*local[j]);
        if(t->box_count && local[j]!=0.)g2_add(t->box_count+j,1.);}
}
#endif
