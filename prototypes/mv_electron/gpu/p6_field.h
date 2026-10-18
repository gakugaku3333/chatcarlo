#ifndef P6_FIELD_H
#define P6_FIELD_H
#include "p5j_kernel.h"
#ifndef P6_FAULT
#define P6_FAULT 0
#endif
// Fixed P3b-3 geometry, packed without padding: 1,836 doubles per history.
enum {P6_REGIONS=15,P6_BINS=1836};
struct P6Geometry {
    double lo[45],hi[45],origins[15],widths[15];
    int axes[15],sizes[15],offsets[15];
};
static_assert(sizeof(P6Geometry)==1144,"P6 CPU/CUDA geometry ABI");
struct P6Sink {double *scores,*sums;const P6Geometry *g;int enabled;unsigned long long *additions;};
struct P6Records {
    double *deposits,*sources;long long *sites;
    int deposit_capacity,source_capacity;long long used,overflow,source_used,source_overflow;
    uint64_t h;
    long long *site_counts; // CPU diagnostics, independent of record capacity.
};
P5_FN void p6_clip(const double *p0,const double *p1,const double *lo,const double *hi,double *t0,double *t1) {
    *t0=0.;*t1=1.;
    for(int j=0;j<3;++j) {
        double d=p1[j]-p0[j];
        if(d==0.) {if(p0[j]<lo[j] || p0[j]>=hi[j]){*t0=1.;*t1=0.;return;}}
        else {double a=(lo[j]-p0[j])/d,b=(hi[j]-p0[j])/d;
            *t0=p5_max(*t0,p5_min(a,b));*t1=p5_min(*t1,p5_max(a,b));}
    }
}
P5_FN int p6_bin(int i,int nb,int axis) {
#if P6_FAULT == 1 || (defined(P5_GPU) && P6_FAULT == 5)
    if(axis!=2 && i>=0 && i<nb)return (i+1)%nb;
#endif
    return i;
}
// Optional diagnostics count the actual += operations, including zero amounts.
// Null in ordinary transport; no scoring arithmetic or RNG calls are changed.
P5_FN void p6_count_addition(unsigned long long *additions,int i) {
    if(!additions)return;
#ifdef P5_GPU
    atomicAdd(additions+i,1ULL);
#else
    ++additions[i];
#endif
}
P5_FN double p6_score_box_point(double *score,int nb,const double *p,double loss,int axis,double origin,double width,const double *lo,const double *hi,unsigned long long *additions=0) {
    for(int j=0;j<3;++j)if(p[j]<lo[j] || p[j]>=hi[j])return 0.;
    int i=int(floor((p[axis]-origin)/width));
    if(i>=0 && i<nb){int bin=p6_bin(i,nb,axis);score[bin]+=loss;p6_count_addition(additions,bin);return loss;}return 0.;
}
P5_FN double p6_score_box_bins(double *score,int nb,const double *p0,const double *p1,double loss,int axis,double origin,double width,const double *lo,const double *hi,unsigned long long *additions=0) {
    if(p0[0]==p1[0] && p0[1]==p1[1] && p0[2]==p1[2])return p6_score_box_point(score,nb,p0,loss,axis,origin,width,lo,hi,additions);
    double t0,t1;p6_clip(p0,p1,lo,hi,&t0,&t1);if(t0>=t1)return 0.;
    double d=p1[axis]-p0[axis];
    if(d==0.) {int i=int(floor((p0[axis]-origin)/width));
        if(i>=0 && i<nb){double amount=loss*(t1-t0);int bin=p6_bin(i,nb,axis);score[bin]+=amount;p6_count_addition(additions,bin);return amount;}return 0.;}
    double q0=p0[axis]+t0*d,q1=p0[axis]+t1*d;
    int first=int(floor((p5_min(q0,q1)-origin)/width));if(first<0)first=0;
    int last=int(floor((p5_max(q0,q1)-origin)/width));if(last>nb-1)last=nb-1;
    double total=0.;
    for(int i=first;i<=last;++i) {
        double a=(origin+i*width-p0[axis])/d,b=(origin+(i+1)*width-p0[axis])/d;
        double amount=loss*p5_max(0.,p5_min(t1,p5_max(a,b))-p5_max(t0,p5_min(a,b)));
        int bin=p6_bin(i,nb,axis);score[bin]+=amount;p6_count_addition(additions,bin);total+=amount;
    }return total;
}
P5_FN void p6_record_deposit(P6Records *r,int site,int kind,const double *p0,const double *p1,double loss) {
    if(!r)return;
    if(r->site_counts)++r->site_counts[site-1];
    if(!r->deposit_capacity)return;
    if(r->used<r->deposit_capacity) {
        double *v=r->deposits+9*r->used;v[0]=double(r->h);v[1]=kind;
        for(int j=0;j<3;++j){v[2+j]=p0[j];v[5+j]=p1[j];}v[8]=loss;r->sites[r->used]=site;
    }else ++r->overflow;
    ++r->used;
}
P5_FN void p6_deposit(P6Sink *sink,P6Records *r,int site,int kind,const double *p0,const double *p1,double loss) {
    p6_record_deposit(r,site,kind,p0,p1,loss);
    if(!sink->enabled)return;
#if P6_FAULT == 2
    if(site==3)return;
#endif
#if defined(P5_GPU) && P6_FAULT == 6
    return;
#endif
    double u[3],v[3];
#if defined(P5_GPU) && P6_FAULT == 7
    for(int k=0;k<3;++k){u[k]=p0[k];v[k]=p1[k];}u[0]=-p0[1];u[1]=p0[0];v[0]=-p1[1];v[1]=p1[0];p0=u;p1=v;
#endif
    const P6Geometry *g=sink->g;
    for(int j=0;j<P6_REGIONS;++j) {
        const double *lo=g->lo+3*j,*hi=g->hi+3*j;
        unsigned long long *additions=sink->additions?sink->additions+g->offsets[j]:0;
        if(kind) {
            if(p0[0]<lo[0] || p0[0]>=hi[0] || p0[1]<lo[1] || p0[1]>=hi[1] || p0[2]<lo[2] || p0[2]>=hi[2])continue;
            sink->sums[j]+=p6_score_box_point(sink->scores+g->offsets[j],g->sizes[j],p0,loss,g->axes[j],g->origins[j],g->widths[j],lo,hi,additions);
        }else {
            bool outside=false;
            for(int k=0;k<3;++k)if((p0[k]<lo[k] && p1[k]<lo[k]) || (p0[k]>=hi[k] && p1[k]>=hi[k]))outside=true;
            if(outside)continue;
            sink->sums[j]+=p6_score_box_bins(sink->scores+g->offsets[j],g->sizes[j],p0,p1,loss,g->axes[j],g->origins[j],g->widths[j],lo,hi,additions);
        }
    }
}
P5_FN void p6_deposit_point(P6Sink *sink,P6Records *r,int site,const double *p,double loss) {p6_deposit(sink,r,site,1,p,p,loss);}
P5_FN void p6_deposit_segment(P6Sink *sink,P6Records *r,const double *p0,const double *p1,double loss) {p6_deposit(sink,r,3,0,p0,p1,loss);}
P5_FN void p6_ledger(const P6Sink *sink,double dep,double incoming,long long *ledger) {
    if(!sink->enabled)return;
    for(int j=0;j<15;++j) {
        double error=j==2?fabs(sink->sums[j]-dep):p5_max(0.,sink->sums[j]-dep);
        if(error>1e-9*incoming)++ledger[j];
    }
}
#endif
