"""ELSEPA atomic tables; identical dcslib interpolation and water composition."""
from functools import lru_cache
import numpy as np
from scipy.interpolate import CubicSpline
from .dcs import Dcslib
from .physics import Z, NJ
from .p1e_common import WORK

class Elsepa(Dcslib):
    name='elsepa'
    def __init__(self, directory=None, charge=-1, *, expected_ihef=None, expected_hashes=None):
        directory=WORK/'tables' if directory is None else directory
        self.atomic={}; self.raw={}
        for z in Z:
            path=directory/f'z{int(z)}_q{charge}.npz'
            if not path.exists(): raise FileNotFoundError('ELSEPA tables absent; run generate_elsepa.py')
            if expected_hashes is not None:
                from .p1e_common import sha
                if expected_hashes.get(path.name)!=sha(path): raise ValueError('ELSEPA table hash mismatch')
            with np.load(path,allow_pickle=False) as f: a={k:f[k].copy() for k in f.files}
            if expected_ihef is not None and ('IHEF' not in a or int(a['IHEF'])!=expected_ihef): raise ValueError('ELSEPA IHEF mismatch')
            if int(a['Z'])!=z or int(a['charge'])!=charge: raise ValueError('ELSEPA identity mismatch')
            if np.any(~np.isfinite(a['dcs'])) or np.any(a['dcs']<=0): raise ValueError('invalid ELSEPA DCS')
            if np.any((a['dcs']==1e-45)&(a['error']==1)): raise ValueError('ELSEPA invalid-value replacement')
            self.raw[int(z)]=a
            self.atomic[int(z)]=(a['energy_MeV'],a['dcs'],a['table_xs'])
            if not hasattr(self,'rmu'): self.rmu=a['mu']
            elif not np.array_equal(self.rmu,a['mu']): raise ValueError('ELSEPA angular grids differ')
        self.ts=self.atomic[1][0]
        if not np.array_equal(self.ts,self.atomic[8][0]): raise ValueError('ELSEPA energy grids differ')
        self.energy_min=max(.01,float(self.ts[0])); self.energy_max=min(20.,float(self.ts[-1]))
        self.native_energies=self.ts[(self.ts>=.01)&(self.ts<=20)]
        self.scale=NJ[1]; stoich=NJ/self.scale
        combined=sum(stoich[j]*(z+1)/z*self.atomic[int(z)][1] for j,z in enumerate(Z))
        self.combined=CubicSpline(np.log(self.ts),np.log(combined),axis=0,bc_type='natural')
        self.atomic_splines={z:CubicSpline(np.log(e),np.log(a),axis=0,bc_type='natural') for z,(e,a,h) in self.atomic.items()}
    @lru_cache(maxsize=256)
    def spline(self,T,z=None):
        self.validate_energy(T)
        return super().spline(T,z)

    def validate_energy(self,T):
        if not self.energy_min<=T<=self.energy_max:
            raise ValueError(f'ELSEPA supported energy range: {self.energy_min:g}–{self.energy_max:g} MeV')
