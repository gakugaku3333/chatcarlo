// Derived from p8g2_cpu.cpp; G5 A2 standalone copy.
#include <cmath>
using std::isfinite;
#include "t2_kernel.h"
#include <vector>
#include <memory>
extern "C" void p8g2_init(double *data,const long long *meta,const double *constants,int nm){for(int mid=0;mid<nm;++mid){P8gMaterial m=p8g_material(data,meta+136*mid,constants);p8g_initialize(m);}}
// Frozen reference wrapper deliberately accepts capacities below 16 and maxsteps.
extern "C" void p8g2_histories(const double *data,const long long *meta,const double *constants,const P8gGeometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,int slots,int reference,const double *injected,const P6Geometry *fg,double hx,double hy,double ssd,double *fs,double *fl,unsigned long long *fa,double *sources,double *d,long long *ints,long long *visits,unsigned long long *hits,double *groups,unsigned long long *group_ints){
    std::unique_ptr<P8g2State[]> arena(new P8g2State[slots]);for(int j=0;j<slots;++j)p8g2_slot_init(&arena[j],g);std::vector<int> ids(slots,-1),snapshot(slots);int next=0;
    for(int j=0;j<slots && next<n;++j){ids[j]=next++;P5State a;if(injected)for(int k=0;k<23;++k)a.a[k]=injected[ids[j]*23+k];p8g2_begin(&arena[j],data,meta,constants,g,seed,start+ids[j],hits,injected?&a:0,fg,hx,hy,ssd);}
    for(;;){bool active=false;for(int j=0;j<slots;++j){snapshot[j]=ids[j]>=0?arena[j].phase:P8G2_D;if(ids[j]>=0)active=true;}if(!active)break;
        for(int phase=0;phase<4;++phase)for(int j=0;j<slots;++j)if(snapshot[j]==phase){
            P8g2State *q=&arena[j];int h=ids[j];
            p8g2_advance(q,data,meta,constants,g,0,maxsteps,hits);
            if(q->phase==P8G2_D){
                for(int k=0;k<3420;++k){fs[k]+=q->field_scores[k];fa[k]+=q->field_additions[k];}
for(int k=0;k<15;++k)fl[k]+=q->field_sums[k];
for(int k=0;k<6;++k)sources[h*6+k]=q->source_state[k];
int copies=1;
#if P8G2_FAULT == 3
                copies=2;p8g2_hit(hits,32);
#elif P8G2_FAULT == 6
                copies=0;p8g2_hit(hits,33);
#endif
                for(int copy=0;copy<copies;++copy){++visits[h];int dw=p8g_dwidth(g),iw=p8g_iwidth(g),f=g->nb+g->nc+g->nf,b=h*100/n;
                    for(int k=0;k<dw;++k)d[h*dw+k]=q->record.d[k];for(int k=0;k<iw;++k)ints[h*iw+k]=q->record.i[k];
#if P8G2_FAULT == 5
                    b=(b+1)%100;p8g2_hit(hits,35);
#endif
                    for(int k=0;k<f+21;++k){
#if P8G2_FAULT == 4
                        if(k==g->nb+g->nc+g->nf){p8g2_hit(hits,34);continue;}
#endif
                        if((k>=f+6 && k<f+9)||(k>=f+19 && k<f+21))groups[b*dw+k]=p5_max(groups[b*dw+k],q->record.d[k]);else groups[b*dw+k]+=q->record.d[k];}
                    for(int k=0;k<iw;++k)if(k==77)group_ints[b*iw+k]=p5_max(group_ints[b*iw+k],(unsigned long long)q->record.i[k]);else if(k<75||k>=79)group_ints[b*iw+k]+=q->record.i[k];
                }
                if(next<n){ids[j]=next++;P5State a;if(injected)for(int k=0;k<23;++k)a.a[k]=injected[ids[j]*23+k];p8g2_begin(q,data,meta,constants,g,seed,start+ids[j],hits,injected?&a:0,fg,hx,hy,ssd);}else ids[j]=-1;
            }
        }
    }
}

extern "C" int t2_set_id(){return G5_SET_ID;}
extern "C" uint64_t t2_set_tag(){return G5_SET_TAG;}
// Fixed scorer probes use exactly the sink called by physical transitions.
extern "C" void t2_score(const P6Geometry *g,const double *records,int n,double *scores,double *sums){
 for(int h=0;h<n;++h){const double *v=records+h*9;P6Sink sink={scores+h*3420,sums+h*15,g,1,0};
 p6_deposit(&sink,0,1,int(v[1]),v+2,v+5,v[8]);}
}
extern "C" void t2_flight(const double *constants,uint64_t seed,double T,double *out){
 P5Tables t={};t.constants=constants;RngState s;rng_init(&s,seed,0);p5_flight_photons(T,0.,0.,1.,&t,&s,out);
}
