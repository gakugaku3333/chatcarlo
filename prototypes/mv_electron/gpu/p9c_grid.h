#ifndef P9C_GRID_H
#define P9C_GRID_H
// P9c: double atomic tallies. All output pointers are caller-owned global memory.
struct P9cGrid {double lo[3],step[3];int n[3];double *values;unsigned long long *adds;};
// P9c: resolve decimal grid faces consistently with the half-open extractor.
P5_FN double p9c_cell_coordinate(double u){double nearest=round(u);return fabs(u-nearest)<=4.*2.220446049250313e-16*p5_max(1.,fabs(u))?nearest:u;}
P5_FN void p9c_add(const P9cGrid *g,int i,int j,int k,double value){
    if(!g->values || value==0.)return;long long cell=((long long)i*g->n[1]+j)*g->n[2]+k;
#ifdef P5_GPU
    atomicAdd(g->values+cell,value);if(g->adds)atomicAdd(g->adds+cell,1ULL);
#else
    g->values[cell]+=value;if(g->adds)++g->adds[cell];
#endif
}
P5_FN double p9c_point(const P9cGrid *g,const double *p,double loss){
    int ix[3];for(int k=0;k<3;++k){double v=p9c_cell_coordinate((p[k]-g->lo[k])/g->step[k]);if(v<0.||v>=g->n[k])return 0.;ix[k]=int(floor(v));}
    p9c_add(g,ix[0],ix[1],ix[2],loss);return 1.;
}
P5_FN double p9c_segment(const P9cGrid *g,const double *a,const double *b,double loss){
    double d[3],lo=0.,hi=1.;bool zero=true;
    for(int k=0;k<3;++k){d[k]=b[k]-a[k];zero=zero&&d[k]==0.;double upper=g->lo[k]+g->step[k]*g->n[k];
        if(d[k]==0.){if(a[k]<g->lo[k]||a[k]>=upper)return 0.;}
        else{double s=(g->lo[k]-a[k])/d[k],t=(upper-a[k])/d[k];lo=p5_max(lo,p5_min(s,t));hi=p5_min(hi,p5_max(s,t));}}
    if(zero)return p9c_point(g,a,loss);if(hi<=lo)return 0.;
    int ix[3];double next[3];
    for(int k=0;k<3;++k){double u=p9c_cell_coordinate((a[k]+lo*d[k]-g->lo[k])/g->step[k]);int cell=int(floor(u));if(d[k]<0.&&u==floor(u))--cell;
        ix[k]=cell<0?0:(cell>=g->n[k]?g->n[k]-1:cell);
        next[k]=d[k]==0.?INFINITY:(g->lo[k]+(ix[k]+(d[k]>0.))*g->step[k]-a[k])/d[k];}
    double t=lo;int bound=g->n[0]+g->n[1]+g->n[2]+6;
    for(int j=0;j<bound && t<hi;++j){double end=p5_min(hi,p5_min(next[0],p5_min(next[1],next[2])));
        if(end>t)p9c_add(g,ix[0],ix[1],ix[2],loss*(end-t));
        if(end>=hi)break;
        for(int k=0;k<3;++k)if(next[k]<=end){ix[k]+=d[k]>0.?1:-1;next[k]=d[k]==0.?INFINITY:(g->lo[k]+(ix[k]+(d[k]>0.))*g->step[k]-a[k])/d[k];}
        t=end;
    }return hi-lo;
}
#endif
