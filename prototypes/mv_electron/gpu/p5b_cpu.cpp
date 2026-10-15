#include "p5b_wrap.h"
// No CPU production wrapper: P5a's shared-output addition order is the reference.
extern "C" void p5b_per_history(const P5Tables *t,uint64_t seed,uint64_t start,
    int n,int maxsteps,P5B_HISTORY_ARGS) {
    for(int i=0;i<n;++i){P5bLocal L;p5b_history(t,seed,start+uint64_t(i),maxsteps,&L);
        p5b_store(&L,i,P5B_HISTORY_PASS);}
}
extern "C" void p5b_fields(const P5Tables *t,int n,double *out,int *dimensions) {
    for(int i=0;i<n;++i)out[i]=p5b_field(t,i);p5b_dimensions(t,dimensions);
}
extern "C" void p5b_functions(const P5Tables *t,int which,uint64_t seed,
    const double *x,int n,double *out,uint64_t *calls) {
    for(int j=0;j<n;++j)p5b_function(t,which,seed,j,x,out,calls);
}
extern "C" void p5b_philox(const uint64_t *triples,int n,uint32_t *words,double *values) {
    for(int j=0;j<n;++j)p5b_rng_words(triples,j,words,values);
}
extern "C" void p5b_rows(uint64_t start,int n,int *rows,uint64_t *histories) {
    for(int j=0;j<n;++j){rows[j]=p5b_row(j,n);histories[j]=start+uint64_t(j);}
}
