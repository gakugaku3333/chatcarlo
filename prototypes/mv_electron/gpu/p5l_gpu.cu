// Warp-owned ordered index lists. Every lane participates, including tail lanes.
P5_FN void p5l_classify(P5kSlot *s,int m,int *lists,int *counts){
 int lane=threadIdx.x&31,R=32*m;
 for(int p=0;p<4;++p){int total=0;
  for(int a=0;a<m;++a){int j=32*a+lane;bool match=p5k_phase(s+j)==p;
   unsigned mask=__ballot_sync(0xffffffffu,match);
   if(match)lists[p*R+total+__popc(mask&((1u<<lane)-1))]=j;
   total+=__popc(mask);
  }if(lane==0)counts[p]=total;
 }__syncwarp();
}
extern "C" __global__ void p5l_kernel(P5K_DRIVER_ARGS){
 P5J_TABLE_INIT;int lane=threadIdx.x&31,warp=threadIdx.x/32,m=P5L_M(variant),K=P5L_K(variant);
 if(!P5L_W(variant)){
  P5kSlot local;P5kSlot *s=P5L_LOCAL(variant)?&local:arena+(blockIdx.x*8+warp)*32+lane;
  p5k_initialize(s,atomicAdd(next,1),n,t,seed,start,trace,capacity);
  for(;;){int snapshot=p5k_phase(s);unsigned masks[4];
   unsigned done=__ballot_sync(0xffffffffu,snapshot==P5J_D);
   for(int p=0;p<4;++p)masks[p]=__ballot_sync(0xffffffffu,snapshot==p);
   if(done==0xffffffffu)break;
   for(int p=0;p<4;++p)if(masks[p]&(1u<<lane)){
    p5l_work(s,t,maxsteps,K,trace,capacity);
    if(p5k_done(s)){P5K_EMIT;p5k_initialize(s,atomicAdd(next,1),n,t,seed,start,trace,capacity);}
   }
  }return;
 }
 __shared__ int all_lists[8*4*128],all_counts[8*4];
 int R=32*m,*lists=all_lists+warp*4*128,*phase_counts=all_counts+warp*4;
 P5kSlot *slots=arena+(blockIdx.x*8+warp)*R;
 for(int a=0;a<m;++a)p5k_initialize(slots+32*a+lane,atomicAdd(next,1),n,t,seed,start,trace,capacity);
 __syncwarp();
 for(;;){p5l_classify(slots,m,lists,phase_counts);if(p5k_finished(phase_counts))break;
  for(int p=0;p<4;++p)for(int base=0;base<phase_counts[p];base+=32){int k=base+lane;
   if(k<phase_counts[p]){P5kSlot *s=slots+lists[p*R+k];p5l_work(s,t,maxsteps,K,trace,capacity);
    if(p5k_done(s)){P5K_EMIT;p5k_initialize(s,atomicAdd(next,1),n,t,seed,start,trace,capacity);}
   }
  }__syncwarp();
 }
}
// Management uses the same classifier and snapshot ownership as transport.
extern "C" __global__ void p5l_schedule_kernel(P5kSlot *arena,int variant,int n,uint64_t start,int *next,
 int *visits,int *rows,uint64_t *histories,int length,int phase,int *ordered){
 int lane=threadIdx.x&31,warp=threadIdx.x/32,m=P5L_M(variant),R=32*m,w=blockIdx.x*8+warp;
 if(!P5L_W(variant)&&length<0){int i=atomicAdd(next,1);
  for(;;){int snapshot=i<n?i%4:P5J_D;unsigned masks[4];
   unsigned done=__ballot_sync(0xffffffffu,snapshot==P5J_D);
   for(int p=0;p<4;++p)masks[p]=__ballot_sync(0xffffffffu,snapshot==p);
   if(done==0xffffffffu)break;
   for(int p=0;p<4;++p)if(masks[p]&(1u<<lane)){
    atomicAdd(visits+i,1);rows[i]=p5b_row(i,n);histories[i]=start+uint64_t(i);i=atomicAdd(next,1);
   }
  }return;
 }
 __shared__ int all_lists[8*4*128],all_counts[8*4];int *lists=all_lists+warp*4*128,*counts=all_counts+warp*4;
 P5kSlot *slots=arena+w*R;
 for(int a=0;a<m;++a){int j=32*a+lane;P5kSlot *s=slots+j;
  s->i=length>=0?j:atomicAdd(next,1);s->active=s->i<(length>=0?length:n);s->q.phase=length>=0?phase:s->i%4;
 }__syncwarp();
 for(;;){p5l_classify(slots,m,lists,counts);
  if(length>=0){for(int k=lane;k<counts[phase];k+=32)ordered[w*(R+1)+k]=lists[phase*R+k];
   if(!lane)ordered[w*(R+1)+R]=counts[phase];return;}
  if(p5k_finished(counts))break;
  for(int p=0;p<4;++p)for(int base=0;base<counts[p];base+=32){int k=base+lane;
   if(k<counts[p]){P5kSlot *s=slots+lists[p*R+k];int i=s->i;
    atomicAdd(visits+i,1);rows[i]=p5b_row(i,n);histories[i]=start+uint64_t(i);
    s->i=atomicAdd(next,1);s->active=s->i<n;s->q.phase=s->i%4;
   }
  }__syncwarp();
 }
}
extern "C" __global__ void p5l_size_kernel(int *bytes){if(!threadIdx.x&&!blockIdx.x)*bytes=sizeof(P5kSlot);}
#include "p5l_contract.h"
extern "C" __global__ void p5l_contract_kernel(P5kSlot *arena,int *out){
 int lane=threadIdx.x&31,warp=threadIdx.x/32,w=blockIdx.x*8+warp;
 __shared__ int ll[8*4*128],cc[8*4],vv[8*128];
 int *lists=ll+warp*4*128,*counts=cc+warp*4,*visits=vv+warp*128;
 P5kSlot *slots=arena+w*128;int errors=0;
 for(int m=1;m<=4;m*=2)for(int round=0;round<2;++round){int R=32*m;
  for(int a=0;a<m;++a){int j=32*a+lane;slots[j].active=!(round&&warp%2==0)&&p5l_mixed_phase(j)!=P5J_D;slots[j].q.phase=p5l_mixed_phase(j);visits[j]=0;}
  __syncwarp();p5l_classify(slots,m,lists,counts);
  if(!lane)for(int p=0;p<4;++p){int expected=0,previous=-1;
   for(int j=0;j<R;++j)expected+=!(round&&warp%2==0)&&p5l_mixed_phase(j)==p;
   errors+=counts[p]!=expected;
   for(int k=0;k<counts[p];++k){int j=lists[p*R+k];errors+=j<=previous||p5l_mixed_phase(j)!=p;previous=j;}
  }
  for(int p=0;p<4;++p)for(int base=0;base<counts[p];base+=32){int k=base+lane;
   if(k<counts[p]){int j=lists[p*R+k];visits[j]++;slots[j].q.phase=p==0?1:p==1?2:0;}
  }__syncwarp();
  for(int a=0;a<m;++a){int j=32*a+lane;errors+=visits[j]!=int(!(round&&warp%2==0)&&p5l_mixed_phase(j)!=P5J_D);}
  __syncwarp();
 }
 errors+=p5l_k_contract(1)+p5l_k_contract(4);
 unsigned bad=__ballot_sync(0xffffffffu,errors!=0);
 if(!lane)out[w*4]=bad!=0;
}
