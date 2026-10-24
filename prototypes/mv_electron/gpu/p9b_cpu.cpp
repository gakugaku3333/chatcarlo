#include "p5d_gxx_compat.h"
#include "p9b_kernel.h"
#include <memory>
extern "C" void p9b_init(double *data,const long long *meta,const double *constants,int nm){for(int mid=0;mid<nm;++mid){P8gMaterial m=p8g_material(data,meta+136*mid,constants);p8g_initialize(m);}}
extern "C" int p9b_geometry_size(){return sizeof(P9bGeometry);}
extern "C" void p9b_histories(const double *data,const long long *meta,const double *constants,const P9bGeometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,const double *injected,double *d,long long *ints,double *field,long long *sites,double *ledger,unsigned long long *adds,long long *lateral){
    std::unique_ptr<P9bState> q(new P9bState);p9b_slot_init(q.get(),g);q->field_additions=adds;
    for(int h=0;h<n;++h){P5State a;if(injected)for(int j=0;j<23;++j)a.a[j]=injected[23*h+j];
        p9b_begin(q.get(),data,meta,constants,g,seed,start+h,0,injected?&a:0);
        while(q->phase!=P8G2_D)p9b_advance(q.get(),data,meta,constants,g,0,maxsteps,0);
        int dw=p8g_dwidth(g),iw=p8g_iwidth(g);
        for(int j=0;j<dw;++j)d[h*dw+j]=q->record.d[j];for(int j=0;j<iw;++j)ints[h*iw+j]=q->record.i[j];
        if(field)for(int j=0;j<3420;++j)field[h*3420+j]=q->field_scores[j];
        for(int j=0;j<6;++j){sites[h*6+j]=q->sites[j];ledger[h*6+j]=q->ledger_energy[j];}lateral[h]=q->lateral;
    }
}
// G3 bypasses begin entirely. The 23-component state and clocks are preserved.
P5_FN void p9b_probe_one(const double *data,const long long *meta,const double *constants,const P9bGeometry *g,const double *input,const double *fixed,int mode,uint64_t seed,uint64_t hid,double *out){
    double a[23];for(int j=0;j<23;++j){a[j]=input[j];out[j]=a[j];}RngState rng;rng_init(&rng,seed,hid);
    P9bHit hit=p9b_boundary(a,g);out[46]=hit.distance;out[47]=hit.next;
    if(isfinite(hit.distance)){for(int k=0;k<3;++k)a[k]+=hit.distance*a[3+k];p9b_snap(a,g,hit);for(int j=0;j<23;++j)out[51+j]=a[j];
        int old=int(a[16]);P8gMaterial m=p8g_material(data,meta+136*g->ids[old],constants);double info[5]={0.,1.,0.,0.,-1.},total,G1;p8g_table_rates(p5_max(.01,a[7]),m,&total,&G1);
        int photon_rebuilds=0;
        if(a[19]==0){if(hit.next>=0){double c[4],mu;p9b_photon_transition(a,g,hit,data,meta,constants,a[6],&m,c,&mu,&photon_rebuilds);}else a[16]=hit.next;}else p9b_transition(a,g,hit,m,total,G1,&rng,info,2.,mode==0?fixed:0);
        out[48]=(int(a[16])>=0 && int(a[16])<g->nl+g->ns)?g->ids[int(a[16])]:-1;
        out[49]=0.;
        if(int(a[16])>=0 && ((a[19]==0 && int(a[16])!=old) || (a[19]!=0 && hit.next>=0 && g->ids[old]!=g->ids[hit.next]))){
            P8gMaterial next=p8g_material(data,meta+136*g->ids[int(a[16])],constants);
            if(a[19]==0){out[49]=photon_rebuilds;}
            else{P9bState q;for(int j=0;j<23;++j)q.state.a[j]=a[j];q.recomputes=0;q.arate=0.;p9b_refresh(&q,next,p5_max(.01,a[7]));out[49]=q.recomputes;}
        }out[50]=rng.calls;
    }else{out[48]=g->ids[int(a[16])];out[49]=out[50]=0.;}
    for(int j=0;j<23;++j)out[23+j]=a[j];
}
extern "C" void p9b_probe(const double *data,const long long *meta,const double *constants,const P9bGeometry *g,const double *input,const double *fixed,int mode,uint64_t seed,uint64_t start,int n,double *out){for(int h=0;h<n;++h)p9b_probe_one(data,meta,constants,g,input+23*h,fixed+3*h,mode,seed,start+h,out+74*h);}
P5_FN void p9b_score_one(const P9bGeometry *g,int region,int site,const double *p0,const double *p1,double energy,double *field,double *depth,unsigned long long *adds,long long *sites,double *ledger){
    P9bState q;q.field_additions=adds;for(int j=0;j<3420;++j)q.field_scores[j]=0.;for(int j=0;j<15;++j)q.field_sums[j]=0.;for(int j=0;j<6;++j){q.sites[j]=0;q.ledger_energy[j]=0.;}
    for(int j=0;j<g->nb;++j){q.local.bins[j]=0.;q.local.additions[j]=0;}
    p9b_field(&q,g,site,p0,p1,energy);p9b_sink(g,&q.local,p0[g->axis],p1[g->axis],region,energy);
    for(int j=0;j<3420;++j)field[j]=q.field_scores[j];for(int j=0;j<g->nb;++j)depth[j]=q.local.bins[j];
    for(int j=0;j<6;++j){sites[j]=q.sites[j];ledger[j]=q.ledger_energy[j];}
}
extern "C" void p9b_score_probe(const P9bGeometry *g,int region,int site,const double *p0,const double *p1,double energy,double *field,double *depth,unsigned long long *adds,long long *sites,double *ledger){p9b_score_one(g,region,site,p0,p1,energy,field,depth,adds,sites,ledger);}
// G3/G5 fixed-state entry: no begin, source draws, or clock reinitialization.
P5_FN void p9b_injected_one(const double *data,const long long *meta,const double *constants,const P9bGeometry *g,const double *input,uint64_t seed,double *out,unsigned long long *adds){
    P9bState q={};q.field_additions=adds;q.cached_ev=-1.;q.M=constants[0];q.source_energy=q.local.source=input[6];
    q.touch[0]=g->nc;q.touch[1]=-1;q.touch[2]=g->nf;q.touch[3]=-1;q.resume_top=-2;
    for(int j=0;j<23;++j)q.state.a[j]=input[j];rng_init(&q.rng,seed,0);
    if(input[19]==0){q.stack[0]=q.state;q.top=0;q.phase=P8G2_P;}else{q.top=-1;q.phase=P8G2_C;}
    while(q.phase!=P8G2_D)p9b_advance(&q,data,meta,constants,g,0,100000,0);
    for(int j=0;j<6;++j){out[j]=q.sites[j];out[6+j]=q.ledger_energy[j];}out[12]=q.dep;out[13]=q.steps;out[14]=q.failed?1.:0.;
}
extern "C" void p9b_injected(const double *data,const long long *meta,const double *constants,const P9bGeometry *g,const double *input,uint64_t seed,double *out,unsigned long long *adds){p9b_injected_one(data,meta,constants,g,input,seed,out,adds);}
