#include "p5a0_common.h"
extern "C" void p5_rng(const uint64_t *triples,int n,uint32_t *words,double *u) {
    for(int i=0;i<n;++i){philox(triples[3*i],triples[3*i+1],triples[3*i+2],words+4*i);u[i]=uniform(words+4*i);}
}
extern "C" void p5_quant(const double *t,int nt,const double *e,const double *v,int n,
    const double *le,const double *er,int m,double cut,int disabled,double *out,int *idx,double *raw) {
    for(int j=0;j<nt;++j)quantities(t[j],e,v,n,le,er,m,cut,disabled,out+4*j,out+4*j+1,out+4*j+2,out+4*j+3,idx+j,raw+12*j);
}
extern "C" void p5_math(const double *x,int n,double *out) {
    for(int i=0;i<n;++i)maths(x+5*i,out+9*i);
}
extern "C" void p5_toy(uint64_t seed,uint64_t start,int n,int shared,uint64_t *meta,double *out) {
    for(int i=0;i<n;++i) {
        double bins[50];uint64_t m[4];toy(seed,start+uint64_t(i),m,bins);
        for(int j=0;j<4;++j)meta[4*i+j]=m[j];
        for(int j=0;j<50;++j){if(shared)out[j]+=bins[j];else out[50*i+j]=bins[j];}
    }
}
// Verification-only MT19937, reference recurrence and 53-bit NumPy conversion.
extern "C" void p5_mt(uint32_t seed,int n,double *out) {
    uint32_t state[624];state[0]=seed;
    for(int i=1;i<624;++i)state[i]=1812433253U*(state[i-1]^(state[i-1]>>30))+uint32_t(i);
    int index=624;
    for(int i=0;i<n;++i) {
        uint32_t w[2];
        for(int k=0;k<2;++k) {
            if(index==624) {
                for(int j=0;j<624;++j) {
                    uint32_t y=(state[j]&0x80000000U)|(state[(j+1)%624]&0x7fffffffU);
                    state[j]=state[(j+397)%624]^(y>>1)^((y&1U)?0x9908b0dfU:0U);
                }index=0;
            }
            uint32_t y=state[index++];y^=y>>11;y^=(y<<7)&0x9d2c5680U;
            y^=(y<<15)&0xefc60000U;y^=y>>18;w[k]=y;
        }out[i]=uniform(w);
    }
}
