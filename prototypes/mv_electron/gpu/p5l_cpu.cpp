#include "p5k_cpu.cpp"
#include "p5l_contract.h"
// Persistent arena belongs only to this new driver.
static std::vector<P5kSlot> p5l_arena;
extern "C" int p5l_slot_bytes(){return sizeof(P5kSlot);}
extern "C" uint64_t p5l_retained_bytes(){return p5l_arena.capacity()*sizeof(P5kSlot);}
extern "C" void p5l_run(int variant,int blocks,uint64_t seed,uint64_t start,int n,int maxsteps,
 P5B_HISTORY_ARGS,double *trace,int capacity,long long *meta) {
 if(variant<0){p5k_run(-1,blocks,seed,start,n,maxsteps,P5B_HISTORY_PASS,trace,capacity,meta);return;}
 const P5jTables *t=&configured;int R=32*P5L_M(variant),K=P5L_K(variant),next=0;
 p5l_arena.resize(blocks*8*R);
 for(int j=0;j<blocks*8*R;++j)p5k_initialize(&p5l_arena[j],next++,n,t,seed,start,trace,capacity);
 std::vector<int> lists(4*R);int count[4];
 for(;;){bool live=false;
  for(int w=0;w<blocks*8;++w){P5kSlot *base=p5l_arena.data()+w*R;
   p5k_classify(base,R,lists.data(),count);if(p5k_finished(count))continue;live=true;
   for(int p=0;p<4;++p)for(int k=0;k<count[p];++k){P5kSlot *s=base+lists[p*R+k];
    p5l_work(s,t,maxsteps,K,trace,capacity);
    if(p5k_done(s)){p5k_finalize(s,t,trace,capacity,meta);p5b_store(&s->L,s->i,P5B_HISTORY_PASS);
     p5k_initialize(s,next++,n,t,seed,start,trace,capacity);}
   }
  }if(!live)break;
 }
}
// The same snapshot scheduling with synthetic transitions; faults exercise execution ownership.
extern "C" void p5l_schedule(int variant,int B,int n,uint64_t start,int fault,int *visits,int *rows,uint64_t *histories) {
 int R=32*P5L_M(variant),W=B*8,next=0;
 std::vector<int> ids(W*R,-1),phase(W*R,4);
 for(int j=0;j<W*R;++j){ids[j]=next++;if(ids[j]<n)phase[j]=ids[j]%4;}
 for(;;){bool live=false;
  for(int w=0;w<W;++w){std::vector<int> lists[4];
   for(int j=w*R;j<(w+1)*R;++j)if(phase[j]<4)lists[phase[j]].push_back(j);
   for(int p=0;p<4;++p)for(int base=0;base<int(lists[p].size());base+=32)
    for(int lane=0;lane<32&&base+lane<int(lists[p].size());++lane){
     live=true;int j=lists[p][base+lane],i=ids[j];
     if(fault!=3){visits[i]+=fault==1?2:1;rows[i]=p5b_row(i,n);histories[i]=start+i;}
     ids[j]=next;next+=fault==2?2:1;phase[j]=ids[j]<n?ids[j]%4:4;
    }
  }if(!live)break;
 }
}
extern "C" int p5l_lists(int R,int phase,int length,int *out) {
    std::vector<P5kSlot> slots(R);for(int j=0;j<R;++j){slots[j].active=j<length;slots[j].q.phase=phase;}
    std::vector<int> lists(4*R);int counts[4];p5k_classify(slots.data(),R,lists.data(),counts);
    for(int tid=0;tid<256;++tid)for(int k=tid;k<counts[phase];k+=256) {
        int j=lists[phase*R+k];out[k]=slots[j].q.phase==phase?j:-2;
        slots[j].q.phase=P5J_D;
    }
    return counts[phase];
}
extern "C" int p5l_contract(int variant){
 int errors=p5l_k_contract(P5L_K(variant)),R=32*P5L_M(variant);
 std::vector<P5kSlot> slots(2*R);std::vector<int> lists(8*R),visits(2*R,0);int counts[2][4];
 // Warp zero drains while warp one remains live, with mixed phases in every 32-slot interval.
 for(int round=0;round<2;++round){
  for(int w=0;w<2;++w){
   for(int j=0;j<R;++j){P5kSlot *s=&slots[w*R+j];s->active=!(round&&w==0)&&p5l_mixed_phase(j)!=P5J_D;s->q.phase=p5l_mixed_phase(j);}
   p5k_classify(slots.data()+w*R,R,lists.data()+w*4*R,counts[w]);
  }
  for(int w=0;w<2;++w)for(int p=0;p<4;++p){int previous=-1,expected=0;
   for(int j=0;j<R;++j)expected+=!(round&&w==0)&&p5l_mixed_phase(j)==p;
   errors+=counts[w][p]!=expected;
   for(int k=0;k<counts[w][p];++k){int j=lists[w*4*R+p*R+k];
    errors+=j<=previous||p5l_mixed_phase(j)!=p;previous=j;visits[w*R+j]++;
    // P->C, C->E, E->P, H->D->refill(P), all deferred until next round.
    slots[w*R+j].q.phase=p==0?1:p==1?2:0;
   }
  }
  for(int w=0;w<2;++w)for(int j=0;j<R;++j){errors+=visits[w*R+j]!=int(!(round&&w==0)&&p5l_mixed_phase(j)!=P5J_D);visits[w*R+j]=0;}
 }
 return errors;
}
