#include "t2_seed_cpu.cpp"
// CPU copy of t2_seed_fixed_kernel (body identical to t2_seed_gpu.cu) for the A2b CPU dry run (a2b_cpu_dryrun.py) only.
extern "C" void t2_seed_fixed(const double *data,const long long *meta,const double *constants,const T2SeedGeometry *g,const double *input,uint64_t seed,double *out){
    std::unique_ptr<T2SeedState> qq(new T2SeedState());T2SeedState &q=*qq;q.cached_ev=-1.;q.M=constants[0];q.source_energy=q.local.source=input[6];
    q.touch[0]=g->nc;q.touch[1]=-1;q.touch[2]=g->nf;q.touch[3]=-1;q.resume_top=-2;
    for(int j=0;j<23;++j)q.state.a[j]=input[j];rng_init(&q.rng,seed,0);
    if(input[19]==0){q.stack[0]=q.state;q.top=0;q.phase=T2S_P;}else{q.top=-1;q.phase=T2S_C;}
    while(q.phase!=T2S_D)t2_seed_advance(&q,data,meta,constants,g,0,100000,0);
    for(int j=0;j<6;++j){out[j]=q.sites[j];out[6+j]=q.ledger_energy[j];}out[12]=q.dep;out[13]=q.steps;out[14]=q.failed?1.:0.;
}
extern "C" int t2_seed_state_size(){return sizeof(T2SeedState);}
