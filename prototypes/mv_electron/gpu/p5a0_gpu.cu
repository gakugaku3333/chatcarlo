#define P5_GPU 1
//@@COMMON@@
extern "C" __global__ void p5_rng(const uint64_t *triples,int n,uint32_t *words,double *u) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){philox(triples[3*i],triples[3*i+1],triples[3*i+2],words+4*i);u[i]=uniform(words+4*i);}
}
extern "C" __global__ void p5_quant(const double *t,int nt,const double *e,const double *v,int n,
    const double *le,const double *er,int m,double cut,int disabled,double *out,int *idx,double *raw) {
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<nt)quantities(t[j],e,v,n,le,er,m,cut,disabled,out+4*j,out+4*j+1,out+4*j+2,out+4*j+3,idx+j,raw+12*j);
}
extern "C" __global__ void p5_math(const double *x,int n,double *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)maths(x+5*i,out+9*i);
}
extern "C" __global__ void p5_toy(uint64_t seed,uint64_t start,int n,int shared,uint64_t *meta,double *out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {
        double bins[50];uint64_t m[4];toy(seed,start+uint64_t(i),m,bins);
        for(int j=0;j<4;++j)meta[4*i+j]=m[j];
        for(int j=0;j<50;++j){if(shared)atomicAdd(out+j,bins[j]);else out[50*i+j]=bins[j];}
    }
}
