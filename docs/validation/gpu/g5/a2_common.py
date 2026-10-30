"""Local A2 paths, sealed inputs and small JSON evidence."""
import sys
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chatcarlo.gpu.therapy import space
CACHE=Path.home()/'.cache/chatcarlo/g5/a2'
SOURCE=Path.home()/'.cache/chatcarlo/p8g2/export'
SUMMARY=HERE/'a2_run'


def write(path,value):
    path=Path(path);space(path.parent);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')


def inputs(source=SOURCE):
    from prototypes.mv_electron.gpu.p8g2_colab import load
    return load(source)


def field_geometry():
    from prototypes.mv_electron.gpu.p7_core import geometry
    from chatcarlo.gpu.therapy import FieldGeometry
    g,_=geometry('p7')
    return FieldGeometry.from_buffer_copy(g)


def water_arrays(arrays,condition='A'):
    from prototypes.mv_electron.gpu import p7_core,p8g
    c=p7_core.CONDITIONS[condition]
    edges,_,cum=p7_core.spectrum(c['spectrum'])
    _,a,grid=p8g.geometry([(1,30.,.1)],[1.,.26,1.85,11.35],edges,cum)
    a.update(constants=arrays['constants'],sizes=np.array([1,len(a['owner']),150,600,len(cum)],np.int64),mass_g_cm2=grid['mass_g_cm2'])
    return a,c


def no_spectrum(a):
    a={**a,'sizes':a['sizes'].copy(),'edges':np.empty(0),'cum':np.empty(0)};a['sizes'][-1]=0
    return a
