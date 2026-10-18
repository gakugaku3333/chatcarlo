#include "p5j_cpu.cpp"
#include "p7_kernel.h"
extern "C" void p7_mt_run(uint64_t seed,int n,int maxsteps,const P6Geometry *g,int field,double field_half,double ssd,int enabled,
 double *batches,double *fine,double *totals,double *maxima,long long *violations,long long *counts,
 long long *nspec,long long *steps,double *brems,long long *maxdepth,double *regions,long long *ledger,
 double *records,long long *sites,int record_capacity,double *sources,int source_capacity,long long *meta,
 long long *site_counts,const double *source_state) {
    P5Output o={batches,fine,totals,maxima,brems,violations,counts,nspec,steps,0};
    P6Records rec={records,sources,sites,record_capacity,source_capacity,0,0,0,0,0,site_counts};
    RngState rng;rng_init(&rng,seed,0);
    for(int h=0;h<n;++h) {
        int b=h*100/n;P7State q={};rec.h=uint64_t(h);
        if(source_state) {
            // CPU-only fixed electron source, matching P3b's source_state path.
            // No spectrum/field draws; P3b renews only the brems clock here.
            q.base.batch=b;q.base.touch[0]=150;q.base.touch[1]=-1;
            q.base.touch[2]=600;q.base.touch[3]=-1;
            for(int j=0;j<23;++j)q.base.stack[0].a[j]=source_state[j];
            q.base.stack[0].a[22]=-log(rng_uniform(&rng));
            q.base.source_energy=source_state[6];q.base.phase=P5J_P;
            if(source_capacity) {
                if(rec.source_used<source_capacity)
                    for(int j=0;j<7;++j)rec.sources[7*rec.source_used+j]=source_state[j];
                else ++rec.source_overflow;
                ++rec.source_used;
            }
        }else p7_init(&q,&configured,&rng,b,&o,field,field_half,ssd,&rec);
        P6Sink sink={regions+b*P7_BINS,q.region_sums,g,enabled};
        while(q.base.phase!=P5J_D)p7_transition(&q,&configured,&rng,maxsteps,&o,0,&sink,ledger,&rec);
    }
    *maxdepth=o.maxdepth;meta[0]=rec.used;meta[1]=rec.overflow;meta[2]=rec.source_used;meta[3]=rec.source_overflow;
}
extern "C" void p7_histories(uint64_t seed,uint64_t start,int n,int maxsteps,const P6Geometry *g,int field,double field_half,double ssd,int enabled,int fresh,
 P5B_HISTORY_ARGS,double *regions,long long *ledger,double *sources,double *trace,int capacity,long long *meta) {
    if(!capacity)trace=0;
    P7Slot slot={};
    for(int i=0;i<n;++i) {
        if(fresh)slot=P7Slot{};
        p7_initialize(&slot,i,n,&configured,seed,start,field,field_half,ssd,trace,capacity);
        while(slot.q.base.phase!=P5J_D)p7_work(&slot,&configured,g,enabled,maxsteps,trace,capacity);
        p7_finalize(&slot,&configured,trace,capacity,meta);p5b_store(&slot.L,i,P5B_HISTORY_PASS);
        for(int j=0;j<P7_BINS;++j)regions[uint64_t(i)*P7_BINS+j]=slot.regions[j];
        for(int j=0;j<15;++j)ledger[i*15+j]=slot.ledger[j];
        for(int j=0;j<7;++j)sources[i*7+j]=slot.sources[j];
    }
}
extern "C" void p7_run(uint64_t seed,uint64_t start,int n,int maxsteps,const P6Geometry *g,int field,double field_half,double ssd,int enabled,
 double *rb,double *rf,double *rt,double *rm,double *re,long long *rv,long long *rc,long long *rn,
 long long *rs,int *rd,long long *rnan,double *regions,long long *ledger,unsigned long long *additions) {
    P7Slot s={};
    for(int i=0;i<n;++i) {
        p7_initialize(&s,i,n,&configured,seed,start,field,field_half,ssd);
        while(s.q.base.phase!=P5J_D)p7_work(&s,&configured,g,enabled,maxsteps,0,0,additions?additions+p5b_row(i,n)*P7_BINS:0);
        p7_finalize(&s,&configured);P5bLocal &L=s.L;int row=p5b_row(i,n);
        for(int j=0;j<150;++j)rb[row*150+j]+=L.batches[j];
        for(int j=0;j<600;++j)rf[row*600+j]+=L.fine_batches[j];
        for(int j=0;j<6;++j)rt[j]+=L.totals[j];
        for(int j=0;j<10;++j)re[j]+=L.brems[j];
        for(int j=0;j<3;++j){rm[j]=p5_max(rm[j],L.maxima[j]);rv[j]+=L.violations[j];}
        for(int j=0;j<16;++j)rc[j]+=L.counts[j];
        ++rn[L.nspec_index];rs[row]+=L.steps;if(L.maxdepth>*rd)*rd=L.maxdepth;*rnan+=p5b_nans(&L);
        for(int j=0;j<P7_BINS;++j){regions[row*P7_BINS+j]+=s.regions[j];if(!isfinite(s.regions[j]))++*rnan;}
        for(int j=0;j<15;++j)ledger[j]+=s.ledger[j];
    }
}
// All scorer primitives, with per-row geometry for deterministic CPU/GPU replay.
extern "C" void p7_score(const double *x,int n,int nb,double *out,double *points,double *values) {
    for(int i=0;i<n;++i) {
        const double *v=x+16*i;double *r=values+4*i;
        p6_clip(v,v+3,v+10,v+13,r,r+1);
        r[2]=p6_score_box_bins(out+uint64_t(i)*nb,nb,v,v+3,v[6],int(v[7]),v[8],v[9],v+10,v+13);
        r[3]=p6_score_box_point(points+uint64_t(i)*nb,nb,v,v[6],int(v[7]),v[8],v[9],v+10,v+13);
    }
}
// Same deposit stream, grouped directly and locally then grouped; preserve per-history sums.
extern "C" void p7_sinks(const P6Geometry *g,const double *records,const long long *histories,int n,int nh,
 double *groups,double *local,double *sums_group,double *sums_local,unsigned long long *additions) {
    for(int i=0;i<n;++i) {
        int h=int(histories[i]),b=h*100/nh;const double *r=records+9*i;
        P6Sink a={groups+b*P7_BINS,sums_group+h*15,g,1,additions+b*P7_BINS},c={local+h*P7_BINS,sums_local+h*15,g,1};
        p6_deposit(&a,0,3,int(r[1]),r+2,r+5,r[8]);p6_deposit(&c,0,3,int(r[1]),r+2,r+5,r[8]);
    }
}
