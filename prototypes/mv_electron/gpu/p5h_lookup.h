#ifndef P5H_LOOKUP_H
#define P5H_LOOKUP_H
#include "p5_kernel.h"
#ifndef P5H_CHECK
#define P5H_CHECK 0
#endif
// Maps: us, brems_e, rates, loget, log(pes), ts. Global read-only memory.
// Per-site counters: calls, full-range fallbacks, mismatches; never early-return a lane.
struct P5hTables : P5Tables {
    const int *ranges;const double *meta;const int *pair_start;
    unsigned long long *lookup_counts;
};
P5_FN void p5h_count(unsigned long long *p) {
#ifdef P5_GPU
    atomicAdd(p,1ULL);
#else
    ++*p;
#endif
}
P5_FN double p5h_value(const double *a,int i,int stride,int logarithmic) {
    return logarithmic?log(a[i*stride]):a[i*stride];
}
P5_FN int p5h_original(double x,const double *a,int n,int stride,int logarithmic,int start) {
    int lo=start,hi=n;
    while(lo<hi) {int m=lo+(hi-lo)/2;if(p5h_value(a,m,stride,logarithmic)<x)lo=m+1;else hi=m;}
    return lo;
}
P5_FN int p5h_find(double x,const double *a,int n,int stride,int logarithmic,int start,
                  const P5hTables *t,int map,int site) {
    const double *m=t->meta+map*5;
    int L=start,H=n;bool fallback=true;
    // Check domain before conversion, including overflow in the coordinate.
    if(isfinite(x) && x>=m[0] && x<=m[1]) {
        double q=(x-m[0])*m[2];
        if(isfinite(q) && q>=0. && q<=m[3]) {
            int b=int(q);if(b==int(m[3]))--b;
            int offset=int(m[4])+2*b;
            L=t->ranges[offset];H=t->ranges[offset+1];
            if(L<start)L=start;
            fallback=!(start<=L && L<=H && H<=n
                && (L==start || p5h_value(a,L-1,stride,logarithmic)<x)
                && (H==n || !(p5h_value(a,H,stride,logarithmic)<x)));
        }
    }
    if(fallback){L=start;H=n;}
    int lo=L,hi=H;
    while(lo<hi) {int m=lo+(hi-lo)/2;if(p5h_value(a,m,stride,logarithmic)<x)lo=m+1;else hi=m;}
#if P5H_CHECK
    p5h_count(t->lookup_counts+site*3);
    if(fallback)p5h_count(t->lookup_counts+site*3+1);
    if(lo!=p5h_original(x,a,n,stride,logarithmic,start))p5h_count(t->lookup_counts+site*3+2);
#endif
    return lo;
}
#endif
