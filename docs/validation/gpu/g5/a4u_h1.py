"""A4u deterministic H1 report: sealed XCOM reads and native coefficients, no MC."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from chatcarlo.gpu.generate_photon_tables import element,ensure

CHANNELS=('photo','compt','rayleigh')
ROOT=Path(__file__).resolve().parents[4]


def native_coefficients(materials,constants,energies):
    """Call the actual production p8g_coefficients after initializing its logs."""
    from chatcarlo.gpu import therapy
    data,meta=therapy.pack(materials);constants=np.ascontiguousarray(constants,float)
    energies=np.ascontiguousarray(energies,float);out=np.empty((len(materials),len(energies),3))
    source='''#include "t2_cpu.cpp"
extern "C" void a4u_h1_coeff(double *data,const long long *meta,const double *constants,int nm,const double *e,int n,double *out){
    p8g2_init(data,meta,constants,nm);
    for(int m=0;m<nm;++m){P8gMaterial mat=p8g_material(data,meta+136*m,constants);
        for(int j=0;j<n;++j){double c[5];p8g_coefficients(e[j]*.001,mat,c);double *row=out+3*(m*n+j);row[0]=c[1];row[1]=c[0];row[2]=c[4];}}
}
'''
    with tempfile.TemporaryDirectory(prefix='a4u-h1-') as tmp:
        tmp=Path(tmp);cpp=tmp/'probe.cpp';cpp.write_text(source);target=tmp/'probe.so'
        compiler=shutil.which('clang++') or shutil.which('g++')
        if compiler is None:raise RuntimeError('C++ compiler required for H1 native coefficients')
        subprocess.run([compiler,'-O2','-ffp-contract=off','-std=c++11','-shared','-fPIC','-DP5_RNG_PHILOX','-DG5_TEST_LEGACY_ELECTRON=0','-I',str(therapy.HERE),str(cpp),'-o',str(target)],check=True,capture_output=True)
        lib=ct.CDLL(str(target));fn=lib.a4u_h1_coeff;fn.restype=None
        fn.argtypes=[therapy.D,therapy.I,therapy.D,ct.c_int,therapy.D,ct.c_int,therapy.D]
        fn(data,meta,constants,len(materials),energies,len(energies),out)
    return out


def elemental_value(p,i,E,channel):
    d=p[2+17*i:2+17*(i+1)];n=int(d[2]);grid=p[int(d[4]):int(d[4])+n];off=int(d[5+channel])
    return float(np.exp(np.interp(np.log(E),np.log(grid),np.log(p[off:off+n]))))


def report(materials,names,constants):
    photons=[ensure(m) for m in materials];energies={5.,20.,60.,150.,1000.,10000.}
    for p in photons:
        tail=int(p[1]);nz=int(p[0])
        for hi in p[tail+1:tail+1+3*nz]:energies.update((float(hi)*(1-1e-8),float(hi)*(1+1e-8)))
    energies=np.array(sorted(energies));native=native_coefficients(materials,constants,energies)
    records={};zs=sorted({int(z) for m in materials for z in m.z})
    previous_path=Path(__file__).with_name('a4u_s2_h1.json')
    previous=json.loads(previous_path.read_text()) if previous_path.exists() else None
    for mid,(name,m,p) in enumerate(zip(names,materials,photons)):
        tail=int(p[1]);nz=int(p[0]);nf=int(p[tail]);fallback=p[tail+1+3*nz:].reshape(4,nf)
        def xs(E,channel):
            return float(native[mid,np.searchsorted(energies,E),channel])
        def xcom(E,channel):
            return float(np.exp(np.interp(np.log(E),np.log(fallback[0]),np.log(fallback[channel+1]))))
        joins=[]
        for channel in range(3):
            # Report every distinct elemental transition of a mixed material.
            for join in sorted(set(p[tail+1+channel:tail+1+3*nz:3])):
                below=float(join)*(1-1e-8);above=float(join)*(1+1e-8)
                lo=xs(below,channel);hi=xs(above,channel);ref=xcom(above,channel)
                joins.append(dict(channel=CHANNELS[channel],keV=float(join),below=lo,above=hi,relative_step=hi/lo-1,
                                  XCOM_above=ref,relative_difference_XCOM_above=hi/ref-1))
        differences=[]
        for energy in (5.,20.,60.,150.,1000.,10000.):
            for channel,label in enumerate(CHANNELS):
                ref=xcom(energy,channel);new=xs(energy,channel)
                differences.append(dict(channel=label,keV=energy,XCOM=ref,unified=new,relative_difference=new/ref-1))
        # Independent reconstruction from packed elemental tables, compared to
        # the actual native path. In particular this rejects a residual KN total.
        for j,E in enumerate(energies):
            expected=np.array([sum(p[3+17*i]*elemental_value(p,i,E,ch) for i in range(nz)) for ch in range(3)])
            np.testing.assert_allclose(native[mid,j],expected,rtol=2e-13,atol=0)
        records[name]=dict(photon_sha256=hashlib.sha256(p.tobytes()).hexdigest(),photon_size=p.size,joins=joins,xs_vs_XCOM=differences,
            coefficient_low_keV=5.,shell_freeze_keV={str(int(z)):min(v['high'] for k,v in element(int(z))[-1]['ranges'].items() if k.startswith('partial_')) for z in m.z},
            previous_joins=None if previous is None else previous['materials'][name]['joins'])
    return dict(status='recorded_not_a_dose_gate',generator_sha256=hashlib.sha256((ROOT/'chatcarlo/gpu/generate_photon_tables.py').read_bytes()).hexdigest(),
        photon_source_sha256=hashlib.sha256((ROOT/'chatcarlo/gpu/photon_physics.h').read_bytes()).hexdigest(),
        native_probe='p8g_coefficients; production specialization; no photon histories',
        fallback_columns=['energy_keV','photo','compt','rayleigh'],
        previous_record_note='Previous H1 reconstructed mixture fallbacks and analytic KN; current H1 probes actual native coefficients.',
        previous_record_sha256=None if previous is None else hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        materials=records,elements=[element(z)[-1] for z in zs],note='Cross-section comparison only; no MV dose, EGS5 or T4 gate is claimed.')

if __name__=='__main__':
    import argparse
    from . import a2_common
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    _,materials,arrays=a2_common.inputs()
    args.out.write_text(json.dumps(report(materials,('water','lung','bone','lead'),arrays['constants']),indent=2,allow_nan=False)+'\n')
