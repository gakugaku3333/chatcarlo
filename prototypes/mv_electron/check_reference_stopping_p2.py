"""Read-only self-check of generated PEGS stopping coefficients, not dose fitting."""
import json
import re
import numpy as np
from .p2_common import CACHE,OUT,ENERGIES,write
from .physics import stopping

def read_stopping(work):
    lines=(work/'pgs5job.pegs5dat').read_text().splitlines()
    rlc=float(lines[4].split()[0]);flags=list(map(int,lines[5].split()));neke=flags[3]
    eke0,eke1=map(float,lines[17].split())
    values=[]
    for line in lines[18:]:
        try:values.extend(map(float,line.split()))
        except ValueError:break
        if len(values)>=neke*30:break
    if len(values)<neke*30:raise RuntimeError('incomplete PEGS electron table')
    data=np.array(values[:neke*30]).reshape(neke,30)
    energies=np.geomspace(.011,10.,200)
    rows=[]
    for T in energies:
        logT=np.log(T);j=max(0,min(neke-1,int(eke1*logT+eke0)-1))
        electron=(data[j,4]+data[j,5]*logT)/rlc;positron=(data[j,6]+data[j,7]*logT)/rlc
        rows.append((T,electron,positron,float(stopping(T))))
    return np.array(rows)

def check():
    rows=[];missing=[]
    for j,E in enumerate(ENERGIES):
        for mode in (0,4):
            work=CACHE/'egs5'/f'selfcheck_{j}_{mode}'
            if not (OUT/'egs5'/work.name/'summary.json').exists():missing.append(work.name);continue
            table=read_stopping(work)
            finite=bool(np.all(np.isfinite(table)) and np.all(table[:,1:3]>0))
            selected=[]
            for t in (.02,.1,2.,10.):
                jx=int(np.argmin(abs(np.log(table[:,0]/t))));v=table[jx]
                selected.append(dict(T=float(v[0]),electron_MeV_cm2_g=float(v[1]),positron_MeV_cm2_g=float(v[2]),
                          prototype_electron_restricted_plus_full_radiation=float(v[3]),electron_relative_to_prototype=float(v[1]/v[3]-1)))
            rows.append(dict(E=E,IUNRST=mode,finite_positive=finite,selected=selected,
                        electron_min=float(table[:,1].min()),positron_min=float(table[:,2].min()),
                        note='Comparison to prototype is diagnostic only. IUNRST=0 includes restricted radiation; IUNRST=4 includes unrestricted radiation.'))
    result=dict(status='未評価' if missing else '合格' if all(r['finite_positive'] for r in rows) else '不合格',missing=missing,rows=rows,
                source_mapping='egs5_hatch.f:435–446; per-energy 30 fields, EDEDX0/1 fields 5/6, PDEDX0/1 fields 7/8; divide by PEGS radiation length in g/cm2',
                independent_source_audit='pegs5.f:1342–1362 and 5027–5028 / 5051–5052 confirm restricted collision + full radiative loss for IUNRST=4; annihilation removed from discrete competition')
    write('t5_stopping_selfcheck.json',result);return result

if __name__=='__main__':check()
