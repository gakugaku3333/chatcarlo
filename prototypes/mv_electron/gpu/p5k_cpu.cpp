#include "p5j_cpu.cpp"
#include "p5k_driver.h"
#include <vector>
static std::vector<P5kSlot> arena;
extern "C" int p5k_slot_bytes() {return sizeof(P5kSlot);}
// reference=-1 executes P5j's history; variants 0..4 are F,Q1,Q4,Q8,Q1-512.
extern "C" void p5k_run(int variant,int blocks,uint64_t seed,uint64_t start,int n,int maxsteps,
    P5B_HISTORY_ARGS,double *trace,int capacity,long long *meta) {
    const P5jTables *t=&configured;
    int R=variant==4?512:256,K=variant==2?4:variant==3?8:1,next=0;
    if(variant<0) {
        for(int i=0;i<n;++i) {
            P5kSlot s;s.i=i;s.L=P5bLocal{};
            for(int j=0;j<640;++j)s.nspec_work[j]=0;
            rng_init(&s.rng,seed,start+uint64_t(i));P5Output o=p5k_output(&s);
            P5Diagnostic d={s.L.diagnostic,4,0,0};P5jTrace tr={trace?trace+uint64_t(i)*capacity*10:0,capacity,0,0};
            tr.used=0;tr.overflow=0;
            p5j_history(t,&s.rng,0,maxsteps,&o,&d,trace?&tr:0);
            // Marker 5 has the final transition's post-state, with only from changed.
            if(trace) {
                if(tr.used<capacity) {
                    double *v=tr.data+10*tr.used,*last=v-10;
                    for(int j=0;j<10;++j)v[j]=last[j];v[0]=5;
                } else ++tr.overflow;
                ++tr.used;
            }
            s.L.calls=s.rng.calls;s.L.maxdepth=o.maxdepth;s.L.diag_used=d.used<4?d.used:4;
            s.L.nspec_index=-1;for(int j=0;j<t->nspec*10;++j)if(s.nspec_work[j])s.L.nspec_index=j;
            p5b_store(&s.L,i,P5B_HISTORY_PASS);
            if(meta){meta[2*i]=tr.used;meta[2*i+1]=tr.overflow;}
        }
        return;
    }
    arena.resize(blocks*R);
    for(int j=0;j<blocks*R;++j)p5k_initialize(&arena[j],next++,n,t,seed,start,trace,capacity);
    std::vector<int> lists(4*R);int count[4];
    for(;;) {
        bool live=false;
        for(int b=0;b<blocks;++b) {
            P5kSlot *slots=arena.data()+b*R;
            // F snapshots each warp separately; Q snapshots the entire block once.
            int groups=variant==0?R/32:1,width=variant==0?32:R;
            for(int g=0;g<groups;++g) {
                P5kSlot *base=slots+g*width;p5k_classify(base,width,lists.data(),count);
                if(p5k_finished(count))continue;live=true;
                for(int p=0;p<4;++p)for(int tid=0;tid<256;++tid)
                    for(int k=tid;k<count[p];k+=256) {
                        P5kSlot *s=base+lists[p*width+k];
                        p5k_work(s,t,maxsteps,K,trace,capacity);
                        if(p5k_done(s)) {
                            p5k_finalize(s,t,trace,capacity,meta);p5b_store(&s->L,s->i,P5B_HISTORY_PASS);
                            p5k_initialize(s,next++,n,t,seed,start,trace,capacity);
                        }
                    }
            }
        }
        if(!live)break;
    }
}
// The same snapshot scheduling with synthetic transitions; faults exercise execution ownership.
extern "C" void p5k_schedule(int variant,int B,int n,uint64_t start,int fault,int *visits,int *rows,uint64_t *histories) {
    int R=variant==4?512:256;std::vector<int> ids(B*R,-1),phase(B*R,4);int next=0;
    for(int j=0;j<B*R;++j){ids[j]=next++;if(ids[j]<n)phase[j]=ids[j]%4;}
    bool live=true;
    while(live) {
        live=false;
        for(int b=0;b<B;++b) {
            std::vector<int> lists[4];
            for(int j=0;j<R;++j)if(phase[b*R+j]<4)lists[phase[b*R+j]].push_back(b*R+j);
            for(int p=0;p<4;++p)for(int tid=0;tid<256;++tid)for(int k=tid;k<int(lists[p].size());k+=256) {
                live=true;int j=lists[p][k],i=ids[j];
                if(fault!=3){visits[i]+=fault==1?2:1;rows[i]=p5b_row(i,n);histories[i]=start+i;}
                ids[j]=next;next+=fault==2?2:1;phase[j]=ids[j]<n?ids[j]%4:4;
            }
        }
    }
}
extern "C" int p5k_lists(int R,int phase,int length,int *out) {
    std::vector<P5kSlot> slots(R);for(int j=0;j<R;++j){slots[j].active=j<length;slots[j].q.phase=phase;}
    std::vector<int> lists(4*R);int counts[4];p5k_classify(slots.data(),R,lists.data(),counts);
    for(int tid=0;tid<256;++tid)for(int k=tid;k<counts[phase];k+=256) {
        int j=lists[phase*R+k];out[k]=slots[j].q.phase==phase?j:-2;
        slots[j].q.phase=P5J_D;
    }
    return counts[phase];
}
