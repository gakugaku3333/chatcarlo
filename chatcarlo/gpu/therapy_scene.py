"""Strict P7 water-field entry schema, without loading transport data."""
from pathlib import Path
import yaml
import numpy as np
from . import physics_sets
from .therapy import NAME

RANGE='検証していない区間。許される範囲: 診断は最大150 keV以下、therapy_mv は mohan10 または mohan4'


def spectrum_arrays(source):
    """Canonical explicit source contract; arrays use MeV like Mohan bins."""
    if isinstance(source.get('spectrum'),str) and source['spectrum'] in ('mohan10','mohan4'):
        return None
    if 'energy_keV' in source:
        e=source['energy_keV']
        if type(e) not in (int,float) or not np.isfinite(e) or not 20<=e<=10000:
            raise ValueError('source.energy_keV: 20〜10000 keV のみ')
        # A4-D8's existing narrow-bin convention; clip at permitted endpoints.
        # USER DECISION 2026-11-04 (recorded, kept as is): at exactly 20 or
        # exactly 10000 the clip halves the bin and moves its centre by
        # +-0.05 keV, so `energy_keV: 20` samples 20.00-20.10 keV (centre
        # 20.05), not 20.00+-0.10. Every value from 20.1 up to 9999.9 is exact.
        # The A4u low-energy validation therefore judged 20 keV through
        # a4u_h5_gate (edges 19.9-20.1), not through this entry.
        edges=np.array([max(20.,e-.1),min(10000.,e+.1)])/1000
        return edges,np.array([1.])
    bins=source.get('spectrum')
    if not isinstance(bins,list) or not 1<=len(bins)<=64:
        raise ValueError('source.spectrum: mohan10 | mohan4 または連続 bin（64 以下）が必要')
    rows=[]
    for b in bins:
        if not isinstance(b,dict) or set(b)!={'low_keV','high_keV','weight'}:
            raise ValueError('source.spectrum: bin は low_keV, high_keV, weight')
        vals=[b[k] for k in ('low_keV','high_keV','weight')]
        if any(type(v) not in (int,float) for v in vals) or not np.isfinite(vals).all():raise ValueError('source.spectrum: 有限な数値が必要')
        lo,hi,w=vals
        if not 20<=lo<hi<=10000 or w<=0:raise ValueError('source.spectrum: 20〜10000 keV、正の重みが必要')
        if rows and rows[-1][1]!=lo:raise ValueError('source.spectrum: bin が不連続')
        rows.append(vals)
    a=np.asarray(rows);weights=a[:,2]/a[:,2].sum();cum=np.cumsum(weights);cum[-1]=1.
    return np.r_[a[0,0],a[:,1]]/1000,cum


def source_spec(source):
    explicit=spectrum_arrays(source)
    return source['spectrum'] if explicit is None else dict(edges=explicit[0].tolist(),cum=explicit[1].tolist())

def validate(raw):
    def keys(value, required, path='', optional=()):
        if not isinstance(value,dict):raise ValueError(path+': mapping が必要')
        missing=set(required)-set(value);unknown=set(value)-set(required)-set(optional)
        if missing or unknown:raise ValueError(path+': '+('必須キー '+str(sorted(missing)) if missing else '未知のキー '+str(sorted(unknown))))
    keys(raw,('domain','source','phantom','scoring','run'),optional=('physics',))
    keys(raw['source'],('field_cm','ssd_cm'),'source',('spectrum','energy_keV'))
    if ('spectrum' in raw['source'])==('energy_keV' in raw['source']):raise ValueError('source: spectrum または energy_keV を一つだけ指定')
    keys(raw['phantom'],('material','thickness_cm'),'phantom')
    keys(raw['run'],('histories','seed','engine'),'run')
    spectrum_arrays(raw['source'])
    if 'physics' in raw:
        keys(raw['physics'],(), 'physics', ('set',))
        selected=raw['physics'].get('set',NAME)
        physics_sets.declaration(selected,specialization=physics_sets.ELECTRON)
        if selected!=NAME:raise ValueError('physics.set: MV 線源には chatcarlo-photon-electron-v1 のみ許可')
    for path,want in (('domain','therapy_mv'),('scoring','pdd_ocr'),('source.field_cm',[10,10]),('source.ssd_cm',100),('phantom.material','water'),('phantom.thickness_cm',30)):
        value=raw
        for key in path.split('.'):value=value[key]
        if value!=want or (isinstance(want,(int,list)) and any(type(v) not in (int,float) for v in (value if isinstance(value,list) else [value]))):raise ValueError(path+': '+str(want)+' のみ')
    run=raw['run']
    if run['engine'] not in ('cpu_local','colab_t4'):raise ValueError('run.engine: cpu_local | colab_t4 のみ')
    cap=200000 if run['engine']=='cpu_local' else 240000000
    if type(run['histories']) is not int or not 0<run['histories']<=cap or run['histories']%100:raise ValueError('run.histories: 100 の倍数、上限 '+str(cap))
    if type(run['seed']) is not int or not 0<=run['seed']<2**64:raise ValueError('run.seed: uint64 が必要')
    return raw


def load(path):
    return validate(yaml.safe_load(Path(path).read_text()))
