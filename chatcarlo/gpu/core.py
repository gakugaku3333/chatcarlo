"""NumPy/ctypes driver; no diagnostic physics packages are required on a VM."""
import ctypes
import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from . import physics_sets

HERE=Path(__file__).resolve().parent
FIELDS='n_elem zs fracs log_e step n_grid photo compt rayl incoh_q incoh_s rayl_x rayl_a k_edge k_omega k_line_e k_line_p n_lines k_frac density_g_cm3'.split()
CPU_FLAGS=('-O2','-ffp-contract=off','-std=c++11','-shared','-fPIC')
CUDA_OPTIONS=('-DP5_GPU','-DP5_RNG_PHILOX','--fmad=false')
SOURCES=('g5_base.h','physics_set.h','g2_tally.h','g1_common.h','g1_physics.h','g1_geometry.h','g1_history.h','g1_cpu.cpp','g1_gpu.cu')

class Geometry(ctypes.Structure):
    _fields_=[('centers',ctypes.c_void_p),('halves',ctypes.c_void_p),('materials',ctypes.c_void_p),('nb',ctypes.c_int),('bg',ctypes.c_int),('world',ctypes.c_void_p),('det',ctypes.c_void_p)]

def pack(tables):
    chunks=[np.asarray(getattr(tables,f),np.float64).ravel() for f in FIELDS]
    offsets=np.r_[0,np.cumsum([len(c) for c in chunks])[:-1]].astype(np.int64)
    return dict(data=np.concatenate(chunks),offsets=offsets,me=np.array(tables.zs.shape[1]),ng=np.array(tables.log_e.shape[2]),nm=np.array(len(tables.material_names)))

def cuda_source():
    def expand(name):
        lines=[]
        for line in (HERE/name).read_text().splitlines():
            if line.startswith('#include "'):lines.append(expand(line.split('"')[1]))
            else:lines.append(line)
        return '\n'.join(lines)
    probes=(HERE/'g1_cpu.cpp').read_text()
    probes=probes[probes.index('extern "C" void g1_direction_probe'):probes.index('#include "g1_history.h"')]
    probes=probes.replace('#include "g1_geometry.h"','')
    for name in ('g1_direction_probe','g1_physics_probe','g1_geometry_probe'):
        probes=probes.replace('extern "C" void '+name+'(', 'extern "C" __global__ void '+name+'_kernel(')
    # Probes deliberately run serially; each input consumes the same single stream.
    probes=probes.replace('{\n    RngState rng;', '{\n    if(blockIdx.x || threadIdx.x)return;RngState rng;')
    probes=probes.replace('G1Tables t={data,offsets,me,ng,nm};RngState', 'if(blockIdx.x || threadIdx.x)return;G1Tables t={data,offsets,me,ng,nm};RngState')
    probes=probes.replace('    for(int i=0;i<n;++i){const double *a=x+10*i;', '    if(blockIdx.x || threadIdx.x)return;for(int i=0;i<n;++i){const double *a=x+10*i;')
    return (HERE/'g1_gpu.cu').read_text().replace('//@@COMMON@@',expand('g1_history.h'))+'\n'+probes

class Core:
    def __init__(self,arrays,device='cpu',*,build_dir=None,rng='philox',fault=0,physics_set=physics_sets.DIAGNOSTIC,physics_declaration=None):
        if device not in ('cpu','cuda'):raise ValueError('device must be cpu or cuda')
        if rng not in ('mt','philox') or (device=='cuda' and rng!='philox'):raise ValueError('MT is CPU verification only')
        if physics_set != physics_sets.DIAGNOSTIC:
            raise ValueError('physics_set not implemented by diagnostic core: '+str(physics_set))
        self.physics_set=physics_sets.declaration(physics_set)
        if physics_declaration is not None:physics_sets.check(physics_declaration,self.physics_set)
        self.set_options=(f'-DG5_SET_ID={self.physics_set["id"]}',f'-DG5_SET_TAG={self.physics_set["tag"]}ULL')
        self.cuda_options=CUDA_OPTIONS+self.set_options+((f'-DP5_FAULT={fault}',) if fault else ())
        self.a={k:np.ascontiguousarray(v) for k,v in arrays.items()};self.device=device;self.rng=rng
        self.me=int(self.a['me'].item());self.ng=int(self.a['ng'].item());self.nm=int(self.a['nm'].item())
        if not 1<=self.me<=32 or not 1<=self.nm<=32:raise ValueError('G1 table capacity exceeded')
        self._hash_cache = {}
        self.npix=0 if not self.a['detector'].size else int(np.prod(self.a['detector'][14:16]))
        if device=='cpu':
            self._temp=None
            if build_dir is None:self._temp=tempfile.TemporaryDirectory(prefix='chatcarlo-g1-');build_dir=self._temp.name
            build_dir=Path(build_dir);build_dir.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256(''.join((HERE/s).read_text() for s in SOURCES).encode()+f'{rng}:{fault}:{self.set_options}'.encode()).hexdigest()[:20]
            target=build_dir/f'g1-{digest}.so'
            if not target.exists():
                compiler=shutil.which('clang++') or shutil.which('g++')
                if compiler is None:raise RuntimeError('clang++ or g++ is required for CPU twin')
                subprocess.run([compiler,*CPU_FLAGS,*self.set_options,'-DP5_RNG_'+rng.upper(),f'-DP5_FAULT={fault}',str(HERE/'g1_cpu.cpp'),'-o',str(target)],check=True,capture_output=True)
            self.lib=ctypes.CDLL(str(target));self.fn=self.lib.g1_histories
            self.fn.restype=None
            self.fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(Geometry),ctypes.c_uint64,ctypes.c_uint64,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
        else:
            import cupy as cp
            self.cp=cp;self.gpu={k:cp.asarray(v) for k,v in self.a.items()}
            self.module=cp.RawModule(code=cuda_source(),backend='nvrtc',options=self.cuda_options)
            self.fn=self.module.get_function('g1_histories_kernel')
            self.fn.compile()
            cp.cuda.Stream.null.synchronize()
        self.check_physics_set()
        self.max_log_energy=self._energy_limit()
        # Immutable static inputs are hashed once, at construction. Cache keys
        # retain the ABI position and actual address, so a pointer swap misses.
        layout = CPU_INPUTS if device == 'cpu' else CUDA_INPUTS
        for position, name in layout.items():
            if name not in {'data', 'offsets', *GEOMETRY_INPUTS}: continue
            if device == 'cpu':
                if name in ('data', 'offsets'):
                    self._hash_cache[(position, name, self.a[name].ctypes.data)] = array_sha(self.a[name])
            else:
                null_detector = name == 'detector' and not self.npix
                pointer = 0 if null_detector else int(self.gpu[name].data.ptr)
                value = np.array([]) if null_detector else self.cp.asnumpy(self.gpu[name])
                self._hash_cache[(position, name, pointer)] = array_sha(value)
        if device == 'cpu':
            for name in GEOMETRY_INPUTS:
                pointer = None if name == 'detector' and not self.npix else self.a[name].ctypes.data
                self._hash_cache[('geometry', name, pointer)] = array_sha(self.a[name])

    def check_physics_set(self):
        out=np.zeros(2,np.uint64)
        if self.device=='cpu':
            fn=self.lib.g5_physics_set
            fn.argtypes=[ctypes.c_void_p];fn.restype=None
            fn(out.ctypes.data)
        else:
            gpu=self.cp.zeros(2,self.cp.uint64)
            self.module.get_function('g5_physics_set_kernel')((1,),(1,),(gpu,))
            out=self.cp.asnumpy(gpu)
        if tuple(map(int,out))!=(self.physics_set['id'],self.physics_set['tag']):
            raise ValueError('compiled physics_set ID/tag mismatch')

    def _energy_limit(self):
        a=self.a
        logs=a['data'][a['offsets'][3]:a['offsets'][4]].reshape(self.nm,self.me,self.ng)
        counts=a['data'][a['offsets'][5]:a['offsets'][6]].reshape(self.nm,self.me)
        elements=a['data'][a['offsets'][0]:a['offsets'][1]]
        return max(float(logs[m,j,int(counts[m,j])-1]) for m in range(self.nm) for j in range(int(elements[m])))

    def _observe(self, args, arrays):
        layout = CPU_INPUTS if self.device == 'cpu' else CUDA_INPUTS
        observed = {}
        for position, name in layout.items():
            value = args[position]
            if name in ARRAY_INPUTS:
                # Hash the actual array associated with this pointer position.
                device_array = self.device == 'cuda' and isinstance(args[position], self.cp.ndarray)
                cache_key = (position, name, int(args[position].data.ptr) if device_array else int(args[position]))
                static = name not in ('origins', 'directions', 'energies')
                if static and cache_key in self._hash_cache:
                    value = self._hash_cache[cache_key]
                else:
                    if self.device == 'cuda':
                        value = self.cp.asnumpy(args[position]) if device_array else np.array([])
                    else:
                        value = next(v for v in arrays.values() if v.ctypes.data == args[position])
                    value = array_sha(value)
                    if static: self._hash_cache[cache_key] = value
            else:
                value = int(value.value) if hasattr(value, 'value') else int(value)
            observed[f'{position}:{name}'] = value
        if self.device == 'cpu':
            geometry = ctypes.cast(args[5], ctypes.POINTER(Geometry)).contents
            for name in GEOMETRY_INPUTS:
                pointer = getattr(geometry, 'det' if name == 'detector' else name)
                array = np.array([]) if pointer is None else next(v for v in self.a.values() if v.ctypes.data == pointer)
                key = ('geometry', name, pointer)
                if key not in self._hash_cache: self._hash_cache[key] = array_sha(array)
                observed['5:geometry.'+name] = self._hash_cache[key]
            observed['5:geometry.nb'] = geometry.nb
            observed['5:geometry.bg'] = geometry.bg
        return observed

    def run(self,origins,directions,energies,*,seed,start=0,fluorescence=True,record=False,trace_capacity=0,observe=False):
        o,d,e=[np.ascontiguousarray(v,np.float64) for v in (origins,directions,energies)]
        n=len(e)
        if o.shape!=(n,3) or d.shape!=(n,3) or e.shape!=(n,) or n<1:raise ValueError('source array shape')
        if not all(np.isfinite(v).all() for v in (o,d,e)) or np.any(e<=0):raise ValueError('invalid source values')
        if np.any(np.log(e)>self.max_log_energy+1e-12):raise ValueError('source energy exceeds table upper bound')
        if trace_capacity and n!=1:raise ValueError('diagnostic replay requires exactly one history')
        a=self.a;agg=np.zeros(self.nm+6+11*self.npix);rd=np.zeros((n if record else 1,self.nm+2));ri=np.zeros((n if record else 1,9),np.int64);trace=np.full((trace_capacity,8),np.nan)
        fault = os.environ.get('CHATCARLO_G4_FAULT')
        seed = int(seed) + (fault == 'seed')
        start = int(start) + (fault == 'start')
        observed = None
        if self.device=='cpu':
            g=Geometry(a['centers'].ctypes.data,a['halves'].ctypes.data,a['materials'].ctypes.data,len(a['materials']),int(a['background'].item()),a['world'].ctypes.data,a['detector'].ctypes.data if self.npix else None)
            args=(a['data'].ctypes.data,a['offsets'].ctypes.data,self.me,self.ng,self.nm,ctypes.byref(g),seed,start,n,o.ctypes.data,d.ctypes.data,e.ctypes.data,int(fluorescence),int(record),rd.ctypes.data,ri.ctypes.data,agg.ctypes.data,trace.ctypes.data if trace_capacity else None,trace_capacity)
            args=tuple(kind(value) if kind in (ctypes.c_int, ctypes.c_uint64) else value for kind,value in zip(self.fn.argtypes,args))
            if observe:
                observed = self._observe(args, dict(data=a['data'], offsets=a['offsets'], origins=o, directions=d, energies=e))
            self.fn(*args)
        else:
            cp=self.cp;g=self.gpu
            go,gd,ge=map(cp.asarray,(o,d,e));ga=cp.zeros_like(cp.asarray(agg));grd=cp.zeros_like(cp.asarray(rd));gri=cp.zeros_like(cp.asarray(ri));gt=cp.asarray(trace)
            args=(g['data'],g['offsets'],np.int32(self.me),np.int32(self.ng),np.int32(self.nm),g['centers'],g['halves'],g['materials'],np.int32(len(a['materials'])),np.int32(a['background'].item()),g['world'],g['detector'] if self.npix else np.uint64(0),np.uint64(seed),np.uint64(start),np.int32(n),go,gd,ge,np.int32(fluorescence),np.int32(record),grd,gri,ga,gt if trace_capacity else np.uint64(0),np.int32(trace_capacity))
            if observe:
                observed = self._observe(args, dict(g, origins=go, directions=gd, energies=ge))
            self.fn(((n+127)//128,),(128,),args)
            cp.cuda.Stream.null.synchronize();agg,rd,ri,trace=map(cp.asnumpy,(ga,grd,gri,gt))
        if agg[-1]!=0:raise ValueError('invalid history output in device aggregation')
        result=dict(deposits=agg[:self.nm],endings=agg[self.nm:self.nm+3],scatter_sum=agg[self.nm+3],n_fluorescence=agg[self.nm+4])
        if self.npix:
            result['images']=agg[self.nm+5:self.nm+5+7*self.npix].reshape(7,int(a['detector'][14]),int(a['detector'][15]))
            result['category_count']=agg[self.nm+5+7*self.npix:-1].reshape(4,int(a['detector'][14]),int(a['detector'][15]))
        if record:result.update(d=rd,i=ri)
        if trace_capacity:result['trace']=trace[np.isfinite(trace[:,0])]
        health(result,n,self.nm,a['detector'])
        return (result, observed) if observe else result

def health(result,n,nm,plane):
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite batch output')
    if np.any(result['deposits']<0) or np.any(result['endings']<0) or result['endings'].sum()!=n:raise ValueError('invalid deposits/endings or history conservation')
    if 'images' in result:
        if np.any(result['images']<0) or np.any(result['category_count']<0):raise ValueError('negative detector output')
        if result['category_count'].sum()!=result['endings'][2] or result['images'][4].sum()!=result['endings'][2]:raise ValueError('detector counts do not match endings')
    if 'i' in result:
        i=result['i'];d=result['d'];hit=i[:,3]==2
        if np.any(i[:,8]) or np.any((i[:,3]<0)|(i[:,3]>2)) or np.any(d[:,1:]<0):raise ValueError('invalid history output')
        if hit.any() and (not len(plane) or np.any(i[hit,4:6]<0) or np.any(i[hit,4]>=plane[14]) or np.any(i[hit,5]>=plane[15]) or np.any((i[hit,6]<0)|(i[hit,6]>3))):raise ValueError('detector pixel out of bounds')


GEOMETRY_INPUTS = ('centers', 'halves', 'materials', 'world', 'detector')
ARRAY_INPUTS = {'data', 'offsets', *GEOMETRY_INPUTS, 'origins', 'directions', 'energies'}
CPU_INPUTS = {0:'data', 1:'offsets', 2:'me', 3:'ng', 4:'nm', 6:'seed', 7:'start', 8:'n', 9:'origins', 10:'directions', 11:'energies', 12:'fluorescence', 13:'record', 18:'trace_capacity'}
CUDA_INPUTS = {0:'data', 1:'offsets', 2:'me', 3:'ng', 4:'nm', 5:'centers', 6:'halves', 7:'materials', 8:'nb', 9:'bg', 10:'world', 11:'detector', 12:'seed', 13:'start', 14:'n', 15:'origins', 16:'directions', 17:'energies', 18:'fluorescence', 19:'record', 24:'trace_capacity'}


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def expected_observation(a, o, d, e, seed, start, fluorescence, device):
    values = {k:array_sha(a[k]) for k in ('data','offsets',*GEOMETRY_INPUTS)}
    values.update({k:int(a[k].item()) for k in ('me','ng','nm')})
    values.update(seed=int(seed), start=int(start), n=len(e), origins=array_sha(o), directions=array_sha(d), energies=array_sha(e), fluorescence=int(fluorescence), record=0, trace_capacity=0, nb=len(a['materials']), bg=int(a['background'].item()))
    layout = CPU_INPUTS if device == 'cpu' else CUDA_INPUTS
    observed = {f'{i}:{name}':values[name] for i,name in layout.items()}
    if device == 'cpu':
        observed.update({'5:geometry.'+name:values[name] for name in GEOMETRY_INPUTS})
        observed.update({'5:geometry.nb':values['nb'], '5:geometry.bg':values['bg']})
    return observed
