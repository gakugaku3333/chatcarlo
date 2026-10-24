"""P8g-2 identity and the frozen record checks."""
from .p8g_core import *
from . import p8g_core as frozen

def fingerprint():
    names=set(frozen.fingerprint())|{'p8g2_kernel.h','p8g2_cpu.cpp','p8g2_gpu.cu','p8g2_core.py','p8g2.py'}
    return {n:sha(HERE/n) for n in sorted(names)}

def cuda_source():
    def expand(name,seen):
        lines=[]
        for line in (HERE/name).read_text().splitlines():
            if line.startswith('#include "'):
                child=line.split('"')[1]
                if child not in seen:seen.add(child);lines.append(expand(child,seen))
            else:lines.append(line)
        return '\n'.join(lines)
    return (HERE/'p8g2_gpu.cu').read_text().replace('//@@COMMON@@',expand('p8g2_kernel.h',set()))

def check_visits(visits):
    if not np.all(visits==1):
        h=int(np.flatnonzero(visits!=1)[0]);raise AssertionError(f'output count history {h}: {visits[h]} != 1')

PATHS=('photon_layers','multiple_relaxation','relaxation_capacity','hard_suspend','brems_suspend',
       'annihilation_rest','annihilation_flight','positron_escape','E_capacity','boundary_invalidate',
       'low_photon','boundary_stay','hard_repop','cache_recompute','hard_capacity','brems_capacity','brems_repop','boundary_recompute')
TRANSITIONS={'P_P':40,'P_C':41,'P_H':43,'C_C':46,'C_E':47,'C_H':48,'E_P':50,'E_H':53,'H_D':59}
LAUNCHES=((40,128),(40,256),(80,128))
CAPACITY=18


def fixed_cases():
    """Registered transport states; energies/clocks, never material-table data.

    Each case is used unchanged by CPU and T4; a missing target is a failed gate.
    starts are global Philox history numbers with SEED2.
    """
    cases=[]
    def photon(name,E,z,region,start,targets,cap=18):
        a=np.zeros(23);a[2]=z;a[5]=1.;a[6]=E;a[16]=region
        cases.append(dict(name=name,state=a.tolist(),start=start,cap=cap,maxsteps=10000000,targets=targets))
    def charged(name,T,z,region,kind,start,targets,cap=18,horizontal=True,hard=False):
        a=np.zeros(23);a[2]=z;a[3 if horizontal else 5]=1.;a[6]=a[7]=T;a[8]=T;a[9]=a[10]=.025*T;a[11]=.05*T;a[12]=a[15]=1e100;a[14]=.025;a[16]=region;a[19]=kind
        if hard:a[15]=0.
        cases.append(dict(name=name,state=a.tolist(),start=start,cap=cap,maxsteps=10000000,targets=targets))
    photon('low-photon',.007,21.2,5,0,[10])
    photon('relaxation-multiple',.1,21.25,5,1,[1])
    photon('photon-layers',10.,0.,0,0,[0])
    photon('relaxation-capacity',.1,21.25,5,1,[1,2],cap=3)
    charged('rest',.01,21.25,5,1,0,[5])
    charged('flight',10.,21.25,5,1,6,[6])
    charged('escape',1.,21.499999999,5,1,0,[7],horizontal=False)
    charged('E-capacity',.01,21.25,5,1,0,[8],cap=2)
    charged('hard',10.,21.25,5,-1,0,[3,12],hard=True)
    charged('brems',10.,21.25,5,-1,0,[4,16])
    charged('hard-capacity',10.,21.25,5,-1,0,[14],cap=1,hard=True)
    charged('brems-capacity',10.,21.25,5,-1,0,[15],cap=1)
    charged('boundary',1.,4.999999999,0,-1,0,[9,17],horizontal=False)
    charged('boundary-stay',1.,4.999999999,0,-1,0,[11],horizontal=False)
    a=cases[-1]['state'];a[5]=.001;a[3]=float(np.sqrt(1-a[5]**2));a[12]=.01;a[13]=.025;a[14]=.05
    # Maxsteps failure in both P and C, with frozen unresolved order.
    for name,idx in (('P-maxsteps',0),('C-maxsteps',4)):
        v=dict(cases[idx]);v.update(name=name,maxsteps=0,targets=[]);cases.append(v)
    return cases
