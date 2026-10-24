#include "p5d_gxx_compat.h"
#include "p8g2_kernel.h"
#include <vector>
#include <memory>
extern "C" void p8g2_init(double *data,const long long *meta,const double *constants,int nm){for(int mid=0;mid<nm;++mid){P8gMaterial m=p8g_material(data,meta+136*mid,constants);p8g_initialize(m);}}
// Frozen reference wrapper deliberately accepts capacities below 16 and maxsteps.
extern "C" void p8g2_histories(const double *data,const long long *meta,const double *constants,const P8gGeometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,int slots,int reference,const double *injected,double *d,long long *ints,long long *visits,unsigned long long *hits,double *groups,unsigned long long *group_ints){
    std::unique_ptr<P8g2State[]> arena(new P8g2State[slots]);for(int j=0;j<slots;++j)p8g2_slot_init(&arena[j],g);std::vector<int> ids(slots,-1),snapshot(slots);int next=0;
    for(int j=0;j<slots && next<n;++j){ids[j]=next++;P5State a;if(injected)for(int k=0;k<23;++k)a.a[k]=injected[ids[j]*23+k];p8g2_begin(&arena[j],data,meta,constants,g,seed,start+ids[j],hits,injected?&a:0);}
    for(;;){bool active=false;for(int j=0;j<slots;++j){snapshot[j]=ids[j]>=0?arena[j].phase:P8G2_D;if(ids[j]>=0)active=true;}if(!active)break;
        for(int phase=0;phase<4;++phase)for(int j=0;j<slots;++j)if(snapshot[j]==phase){
            P8g2State *q=&arena[j];int h=ids[j];
            if(reference){P5State a;if(injected)for(int k=0;k<23;++k)a.a[k]=injected[h*23+k];
                // Reset RNG to undo begin's source/injection draws before calling frozen history.
                rng_init(&q->rng,seed,start+h);p8g_clear(g,&q->local);
                p8g_record_history(data,meta,constants,g,&q->rng,maxsteps,&q->record,&q->local,injected?&a:0);q->phase=P8G2_D;
            }else p8g2_advance(q,data,meta,constants,g,0,maxsteps,hits);
            if(q->phase==P8G2_D){
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
                if(next<n){ids[j]=next++;P5State a;if(injected)for(int k=0;k<23;++k)a.a[k]=injected[ids[j]*23+k];p8g2_begin(q,data,meta,constants,g,seed,start+ids[j],hits,injected?&a:0);}else ids[j]=-1;
            }
        }
    }
}
// Original accumulation order: direct shared output from each transition.
extern "C" void p8g_run(const double *data,const long long *meta,const double *constants,const P8gGeometry *g,uint64_t seed,uint64_t start,int n,int maxsteps,P8gOutput *o,double *histories,long long *hcalls,double *sources,double *deps,double *diagnostic,long long *dm){
    P8g2State q;p8g2_slot_init(&q,g);
    for(int h=0;h<n;++h){p8g2_begin(&q,data,meta,constants,g,seed,start+h,0);int b=h*100/n;
        while(q.phase!=P8G2_D)p8g2_advance(&q,data,meta,constants,g,b,maxsteps,0,o);
        if(histories)for(int j=0;j<g->nb;++j)histories[h*g->nb+j]=q.local.bins[j];
        if(hcalls)hcalls[h]=q.local.calls;if(sources)sources[h]=q.local.source;if(deps)deps[h]=q.local.dep;
        if(dm)dm[h]=q.diag_used<2?q.diag_used:2;
        if(diagnostic){int f=g->nb+g->nc+g->nf;for(int j=0;j<14;++j)diagnostic[14*h+j]=q.record.d[f+23+j];}
    }
}
