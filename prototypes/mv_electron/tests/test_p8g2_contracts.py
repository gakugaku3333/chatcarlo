"""Compile exported CUDA as host syntax only; does not execute GPU gates."""
import shutil
import subprocess
import pytest
from prototypes.mv_electron.gpu import p8g2_core as core

STUB=r'''
#include <cmath>
#include <cstring>
#define __device__
#define __global__
struct Dim {int x;};Dim blockIdx,blockDim,threadIdx;
static double __longlong_as_double(unsigned long long x){double y;memcpy(&y,&x,8);return y;}
static long long __double_as_longlong(double x){long long y;memcpy(&y,&x,8);return y;}
template<class T,class U> T atomicAdd(T* p,U v){T x=*p;*p+=v;return x;}
template<class T,class U> T atomicMax(T* p,U v){T x=*p;if(*p<v)*p=v;return x;}
unsigned __ballot_sync(unsigned,int){return 0;}
using std::isfinite;
#define P5_GPU
#define P5_RNG_PHILOX
#define P8G_STACK_CAP 18
'''

@pytest.mark.parametrize('fault',range(7))
def test_p8g2_exported_cuda_syntax(tmp_path,fault):
    path=tmp_path/'syntax.cpp';path.write_text(STUB+f'\n#define P8G2_FAULT {fault}\n#define P8G2_CHECK 1\n'+core.cuda_source())
    result=subprocess.run([shutil.which('clang++'),'-std=c++11','-ffp-contract=off','-fsyntax-only',str(path)],text=True,capture_output=True)
    assert result.returncode==0,result.stderr
