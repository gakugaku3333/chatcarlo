#ifndef G5_BASE_H
#define G5_BASE_H
// Philox4x32-10 definition: DEShawResearch/random123,
// commit 9545ff6413f258be2f04c1d319d99aaef7521150,
// include/Random123/philox.h; independently expressed round arithmetic.
#ifdef P5_GPU
// NVRTC does not provide the host <cmath> INFINITY macro.
#if defined(P5_GPU) && !defined(INFINITY)
#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))
#endif
#define P5_FN __device__ inline
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
#else
#include <cmath>
#include <cstdint>
#define P5_FN static inline
#endif
static_assert(sizeof(uint32_t)==4, "32-bit word required");
static_assert(sizeof(uint64_t)==8, "64-bit word required");
#ifndef P5_CAP
#define P5_CAP 16
#endif
#ifndef P5_FAULT
#define P5_FAULT 0
#endif
P5_FN void philox(uint64_t seed, uint64_t history, uint64_t count, uint32_t *out) {
    uint32_t x0=uint32_t(history), x1=uint32_t(history>>32);
    uint32_t x2=uint32_t(count), x3=uint32_t(count>>32);
    uint32_t k0=uint32_t(seed), k1=uint32_t(seed>>32);
    for(int r=0;r<10;++r) {
        uint64_t p=uint64_t(0xD2511F53U)*x0, q=uint64_t(0xCD9E8D57U)*x2;
        uint32_t y0=uint32_t(q>>32)^x1^k0, y2=uint32_t(p>>32)^x3^k1;
        x0=y0; x1=uint32_t(q); x2=y2; x3=uint32_t(p);
        k0+=0x9E3779B9U; k1+=0xBB67AE85U;
    }
    out[0]=x0;out[1]=x1;out[2]=x2;out[3]=x3;
}
P5_FN double uniform(const uint32_t *w) {
    return ((w[0]>>5)*67108864.0+(w[1]>>6))/9007199254740992.0;
}
P5_FN double draw(uint64_t seed,uint64_t h,uint64_t *c) {
    uint32_t w[4];philox(seed,h,*c,w);++*c;return uniform(w);
}
#endif
