"""Separate ncu process. Only the selected V0 launch is profiled after warmup."""
import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--bundle')
    a=parser.parse_args()
    import cupy as cp
    if a.probe:
        kernel=cp.RawKernel('extern "C" __global__ void probe(double *x){x[threadIdx.x]+=1.;}','probe')
        x=cp.zeros(32)
        kernel((1,),(32,),(x,))
        cp.cuda.Device().synchronize()
        print('probe complete')
        return
    from . import p5b_core as core, p5d_core as d
    from .p5b_colab import GPU
    data,_,_=d.load_bundle(a.bundle)
    kernel=cp.RawKernel(core.source(),'p5b_run_kernel',options=core.OPTIONS,backend='nvrtc')
    gpu=GPU(cp,data,{'p5b_run_kernel':kernel})
    gpu.production(1000)
    cp.cuda.Device().synchronize()
    gpu.production(100000)
    cp.cuda.Device().synchronize()


if __name__=='__main__':
    main()
