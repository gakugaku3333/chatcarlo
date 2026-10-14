#ifndef P5_RNG_H
#define P5_RNG_H
#include "p5a0_common.h"

// MT is verification-only and is excluded from GPU compilation.
#if defined(P5_RNG_MT)
#ifdef P5_GPU
#error MT19937 is CPU-only
#endif
struct RngState {uint32_t mt[624];int index;uint64_t calls;};
P5_FN void rng_init(RngState *s,uint64_t seed,uint64_t history) {
    (void)history;s->mt[0]=uint32_t(seed);s->index=624;s->calls=0;
    for(int i=1;i<624;++i)
        s->mt[i]=1812433253U*(s->mt[i-1]^(s->mt[i-1]>>30))+uint32_t(i);
}
P5_FN uint32_t rng_word(RngState *s) {
    if(s->index==624) {
        for(int j=0;j<624;++j) {
            uint32_t y=(s->mt[j]&0x80000000U)|(s->mt[(j+1)%624]&0x7fffffffU);
            s->mt[j]=s->mt[(j+397)%624]^(y>>1)^((y&1U)?0x9908b0dfU:0U);
        }
        s->index=0;
    }
    uint32_t y=s->mt[s->index++];y^=y>>11;y^=(y<<7)&0x9d2c5680U;
    y^=(y<<15)&0xefc60000U;y^=y>>18;return y;
}
P5_FN double rng_uniform(RngState *s) {
    uint32_t w[2];w[0]=rng_word(s);w[1]=rng_word(s);++s->calls;
    return uniform(w);
}
#elif defined(P5_RNG_PHILOX)
struct RngState {uint64_t seed,history,calls;};
P5_FN void rng_init(RngState *s,uint64_t seed,uint64_t history) {
    s->seed=seed;s->history=history;s->calls=0;
}
P5_FN double rng_uniform(RngState *s) {
    return draw(s->seed,s->history,&s->calls);
}
#else
#error Select P5_RNG_MT or P5_RNG_PHILOX
#endif
#endif
