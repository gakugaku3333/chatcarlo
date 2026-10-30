// Derived from p8g_kernel.h; G5 A2 standalone copy.
#ifndef P8G_KERNEL_H
#define P8G_KERNEL_H
#include "t2_materials.h"
#ifndef P8G_STACK_CAP
#define P8G_STACK_CAP 4096
#endif
struct P8gGeometry {const double *faces,*rho,*bins,*strength,*edges,*cum;const long long *ids,*owner;int nl,nb,nc,nf,nspec,binding,fluorescence;double energy;};
struct P8gOutput {double *batches,*fine_batches,*totals,*maxima,*brems,*p8_batches,*p8_max;long long *violations,*counts,*step_batches,*boundary_counts,*boundary_hist,*boundary_omitted_hist,*fluorescence_counts,*branches;int maxdepth;};
#define P8G_BIN_CAP 300
#define P8G_COARSE_CAP 150
#define P8G_FINE_CAP 600
struct P8gLocal {double bins[P8G_BIN_CAP],coarse[P8G_COARSE_CAP],fine[P8G_FINE_CAP];long long additions[P8G_BIN_CAP];double source,dep;uint64_t calls;int nspec_index,event_used,skipped;double events[160];long long transitions[80];};
P5_FN double p8g_sum(const double *x,int n){double v=0.;for(int j=0;j<n;++j)v+=x[j];return v;}
P5_FN void p8g_boundary(double z,double w,int region,const double *faces,double *distance,int *nr){*nr=region;if(w>0){*distance=p5_max(0.,(faces[region+1]-z)/w);*nr=region+1;}else if(w<0){*distance=p5_max(0.,(faces[region]-z)/w);*nr=region-1;}else *distance=INFINITY;}
P5_FN void p8g_score(double *score,int n,double z0,double z1,int region,double loss,const double *edges,const long long *owner,long long *adds=0){
    if(loss==0)return;double lo=p5_min(z0,z1),hi=p5_max(z0,z1);
    if(hi-lo<=1e-15){int j=p5_clip_bin(p8g_upper(z0,edges,n+1)-1,n);if(owner[j]!=region && j>0 && z0==edges[j] && owner[j-1]==region)--j;score[j]+=loss;if(adds)++adds[j];return;}
    int first=p5_max(0,p8g_upper(lo,edges,n+1)-1),last=p5_min(n-1,lower_bound(hi,edges,n+1));
    for(int j=first;j<=last;++j)if(owner[j]==region){double overlap=p5_max(0.,p5_min(hi,edges[j+1])-p5_max(lo,edges[j]));score[j]+=loss*overlap/(hi-lo);if(adds)++adds[j];}
}
P5_FN void p8g_sink(const P8gGeometry *g,P8gLocal *l,double z0,double z1,int region,double loss){
#if P8F_FAULT == 5
    if(loss>0 && !l->skipped){l->skipped=1;return;}
#endif
    p8g_score(l->bins,g->nb,z0,z1,region,loss,g->bins,g->owner,l->additions);
}
P5_FN void p8g_material_boundary(double *a,int old,int nr,const P8gMaterial &m,double total,double G1,double knew,RngState *s,double *info,double override_mu=2.){
    double strength=p5_max(0.,(a[14]-a[13])-a[12]),mu=1.;bool terminal=a[18]==1,omitted=false;
    if(!terminal){if(strength>=m.gs_K[0]){mu=override_mu>1?p8g_gs_mu_cached(p5_max(.01,a[7]),strength,m,total,G1,s):override_mu;double angle=rng_uniform(s),v[3];p5_rotate(a[3],a[4],a[5],mu,2*3.141592653589793*angle,v);for(int j=0;j<3;++j)a[j+3]=v[j];}else if(strength>0)omitted=true;}
    bool forward=(nr>old && a[5]>0)||(nr<old && a[5]<0);a[16]=forward?nr:old;
    if(!terminal){double u=rng_uniform(s);a[12]=u*knew;a[13]=(1-u)*knew;a[14]=knew;}
    info[0]=strength;info[1]=mu;info[2]=omitted?1.:0.;info[3]=terminal?1.:0.;info[4]=-1.;
}
P5_FN int p8g_event(double step,double bb,double eb,double mb,double hb,double rb,double ab){
    if(bb<=step+1e-13)return 0;if(eb<=step+1e-13)return 1;if(mb<=step+1e-13)return 2;if(hb<=step+1e-13)return 3;if(rb<=step+1e-13)return 4;if(ab<=step+1e-13)return 5;return 6;
}
P5_FN int p8g_mid(int id){
#if P8F_FAULT == 1
    return (id+1)%4;
#else
    return id;
#endif
}

struct P8gRecord {double d[1200];long long i[1200];};
P5_FN int p8g_dwidth(const P8gGeometry *g){return g->nb+g->nc+g->nf+37;}
P5_FN int p8g_iwidth(const P8gGeometry *g){return 79+g->nb;}
#endif
