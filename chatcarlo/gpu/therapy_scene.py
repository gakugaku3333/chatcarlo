"""Strict P7 water-field entry schema, without loading transport data."""
from pathlib import Path
import yaml
from . import physics_sets
from .therapy import NAME

RANGE='検証していない区間。許される範囲: 診断は最大150 keV以下、therapy_mv は mohan10 または mohan4'


def validate(raw):
    def keys(value, required, path='', optional=()):
        if not isinstance(value,dict):raise ValueError(path+': mapping が必要')
        missing=set(required)-set(value);unknown=set(value)-set(required)-set(optional)
        if missing or unknown:raise ValueError(path+': '+('必須キー '+str(sorted(missing)) if missing else '未知のキー '+str(sorted(unknown))))
    keys(raw,('domain','source','phantom','scoring','run'),optional=('physics',))
    keys(raw['source'],('spectrum','field_cm','ssd_cm'),'source')
    keys(raw['phantom'],('material','thickness_cm'),'phantom')
    keys(raw['run'],('histories','seed','engine'),'run')
    if raw['source']['spectrum'] not in ('mohan10','mohan4'):raise ValueError('source.spectrum: '+RANGE)
    if 'physics' in raw:
        keys(raw['physics'],(), 'physics', ('set',))
        selected=raw['physics'].get('set',NAME)
        physics_sets.declaration(selected)
        if selected!=NAME:raise ValueError('physics.set: MV 線源には therapy-electron-v1 のみ許可')
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
