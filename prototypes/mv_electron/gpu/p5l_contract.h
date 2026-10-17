#ifndef P5L_CONTRACT_H
#define P5L_CONTRACT_H
// Mixed states in every 32 interval, with >32 C slots for both W pool sizes.
P5_FN int p5l_mixed_phase(int j){int r=j%8;return r==0?P5J_P:r==1?P5J_E:r==2?P5J_H:r==3?P5J_D:P5J_C;}
// Synthetic transition outputs: five C continuations then exit to E.
P5_FN int p5l_synthetic_job(int *phase,int *position,const int *next_phase,int K){
 int initial=*phase,used=0;
 for(int k=0;k<(initial==P5J_C?K:1);++k){*phase=next_phase[(*position)++];++used;if(*phase!=P5J_C)break;}
 return used;
}
P5_FN int p5l_k_contract(int K){
 int continuation[6]={P5J_C,P5J_C,P5J_C,P5J_C,P5J_C,P5J_E};
 int phase=P5J_C,pos=0,errors=0;
 if(K==4){errors+=p5l_synthetic_job(&phase,&pos,continuation,K)!=4;
  errors+=p5l_synthetic_job(&phase,&pos,continuation,K)!=2;}
 else errors+=p5l_synthetic_job(&phase,&pos,continuation,K)!=1;
 int exit[1]={P5J_E};phase=P5J_C;pos=0;
 errors+=p5l_synthetic_job(&phase,&pos,exit,K)!=1;
 int enter[1]={P5J_C};phase=P5J_P;pos=0;
 errors+=p5l_synthetic_job(&phase,&pos,enter,K)!=1;
 return errors;
}
// Audit the actual frozen work function on every traced job, without changing its arithmetic.
P5_FN void p5l_work(P5kSlot *s,const P5jTables *t,int maxsteps,int K,double *trace,int capacity){
 int before=s->trace_used,initial=s->q.phase;
 p5k_work(s,t,maxsteps,K,trace,capacity);
 if(trace && !s->trace_overflow){
  int used=s->trace_used-before,limit=initial==P5J_C?K:1;
  bool good=used>=1 && used<=limit;
  if(initial==P5J_C && used<limit && s->q.phase==P5J_C)good=false;
  const double *rows=trace+uint64_t(s->i)*capacity*10;
  for(int k=0;k<used;++k){int from=int(rows[(before+k)*10]);
   good=good && from==(k?P5J_C:initial);
   if(k+1<used)good=good && int(rows[(before+k)*10+1])==P5J_C;
  }
  if(!good)++s->trace_overflow;
 }
}
#endif
