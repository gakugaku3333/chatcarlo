"""Array-slice catalogue. NumPy/stdlib only, including on a worker VM.

Offsets are element offsets (not bytes). Shape always describes the original
array, including scalar and higher-dimensional P8 arrays. Schema definitions
are independent of the supplied catalogue so swapping equal-shaped fields is
not accepted by verification.
"""
import hashlib
import math
from pathlib import Path
import numpy as np

FORMAT = 'chatcarlo-catalog-v1'
DIAGNOSTIC = 'diagnostic-photon-kerma-v1'
THERAPY = 'therapy-electron-v1'
FIELDS = 'n_elem zs fracs log_e step n_grid photo compt rayl incoh_q incoh_s rayl_x rayl_a k_edge k_omega k_line_e k_line_p n_lines k_frac density_g_cm3'.split()
# G2 prepare() adds scoring descriptions after table creation. These are not
# transport tables and are deliberately excluded from the catalogue contract.
G2_METADATA = {'boxes', 'box_names', 'water', 'seed', 'n_histories', 'batch_size',
               'depth', 'grid_origin', 'grid_shape', 'grid_resolution'}
UNITS = dict(zip(FIELDS, ['count','atomic-number','mass-fraction','ln(keV)',
    'ln(keV)','count','ln(cm2/g)','ln(cm2/g)','ln(cm2/g)','angstrom^-1',
    'electrons','angstrom^-2','integral(F2 dx)','keV','probability','keV',
    'probability','count','fraction','g/cm3']))
ENERGY_FIELDS = {'log_e','photo','compt','rayl','k_frac'}


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def generator(path, root, options=None):
    return dict(path=path, sha256=hashlib.sha256((Path(root)/path).read_bytes()).hexdigest(),
                options=options or {})


def entry(name, blob, offset, array, *, unit, used_by, generator, provenance,
          grid=None, grid_kind=None, band=None, **attributes):
    a = np.asarray(array)
    return dict(name=name, used_by=list(used_by), blob=blob, offset=int(offset),
                size=int(a.size), dtype=a.dtype.str, shape=list(a.shape), unit=unit,
                energy_unit='keV' if band is not None else None, energy_band_keV=band,
                grid_kind=grid_kind, grid_nodes_sha256=None if grid is None else digest(grid),
                provenance=provenance, generator=generator, data_sha256=digest(a), **attributes)


def diagnostic_arrays(a):
    nm, me, ng = (int(a[k]) for k in ('nm','me','ng'))
    shapes = {f:(nm,me) for f in FIELDS}
    for f in ('n_elem','density_g_cm3'): shapes[f]=(nm,)
    for f in ENERGY_FIELDS: shapes[f]=(nm,me,ng)
    for f in ('incoh_q','incoh_s','rayl_x','rayl_a'): shapes[f]=(nm,me,2000)
    for f in ('k_line_e','k_line_p'): shapes[f]=(nm,me,8)
    off = np.asarray(a['offsets'])
    if off.shape != (len(FIELDS),) or off.dtype != np.dtype('int64'):
        raise ValueError('offsets: invalid descriptor')
    arrays = {}
    expected = 0
    for i,f in enumerate(FIELDS):
        size = math.prod(shapes[f])
        if int(off[i]) != expected or expected+size > len(a['data']):
            raise ValueError(f+': invalid packed offset/size')
        arrays[f] = a['data'][expected:expected+size].reshape(shapes[f])
        expected += size
    if expected != len(a['data']): raise ValueError('data: unexpected trailing elements')
    return arrays


def energy_band(arrays):
    logs = arrays['log_e']; counts = arrays['n_grid']; ne = arrays['n_elem']
    nodes = [logs[m,j,:int(counts[m,j])] for m in range(len(ne)) for j in range(int(ne[m]))]
    if not nodes or any(len(g)<2 or not np.isfinite(g).all() for g in nodes):
        raise ValueError('log_e: invalid effective energy grid')
    return [float(np.exp(min(g[0] for g in nodes))), float(np.exp(max(g[-1] for g in nodes)))]


def diagnostic(a, generators, material_names):
    arrays = diagnostic_arrays(a); band = energy_band(arrays); fields=[]
    for i,f in enumerate(FIELDS):
        grid_name = 'log_e' if f in ENERGY_FIELDS else {'incoh_q':'incoh_q',
            'incoh_s':'incoh_q','rayl_x':'rayl_x','rayl_a':'rayl_x'}.get(f)
        fields.append(entry(f,'tables',a['offsets'][i],arrays[f], unit=UNITS[f],
            used_by=[DIAGNOSTIC], generator=generators['diagnostic'],
            provenance='xraylib / material composition; chatcarlo/kernel.py:bake_scene_materials',
            grid=None if grid_name is None else arrays[grid_name], grid_kind=grid_name,
            band=band if f in ENERGY_FIELDS else None, material_names=list(material_names),
            valid_length='n_grid' if f in ENERGY_FIELDS else None,
            padding='per-element to ng' if f in ENERGY_FIELDS else None))
    if 'g2_table' in a:
        stride, nh = map(int,a['g2_sizes'][:2]); nm=len(material_names)
        defs=[('g2_x',(nm,stride),'ln(keV)',None),
              ('g2_c',(nm,4,stride),'ln(muen/rho) PCHIP coefficients','g2_x'),
              ('g2_density',(nm,),'g/cm3',None),
              ('g2_log_e',(nh,),'ln(keV)',None),
              ('g2_log_h',(nh,),'ln(pSv cm2)','g2_log_e')]
        cursor=0; g2={}
        for name,shape,unit,grid_name in defs:
            size=math.prod(shape);value=a['g2_table'][cursor:cursor+size].reshape(shape);g2[name]=value
            grid_name = grid_name or (name if name in ('g2_x','g2_log_e') else None)
            grid = g2.get(grid_name)
            if name in ('g2_x','g2_c'):
                x=g2['g2_x'];valid=[x[m,:int(a['g2_sizes'][m+2])] for m in range(nm)]
                b=[float(np.exp(min(v[0] for v in valid))),float(np.exp(max(v[-1] for v in valid)))]
            elif name in ('g2_log_e','g2_log_h'): b=[float(np.exp(grid[0])),float(np.exp(grid[-1]))]
            else:b=None
            fields.append(entry(name,'g2_tables',cursor,value,unit=unit,used_by=[DIAGNOSTIC],
                generator=generators['g2'],provenance='NIST XAAMDI; ICRP74/ICRU57; material density',
                grid=grid,grid_kind=grid_name,band=b,material_names=list(material_names),
                valid_length='g2_sizes[2:]' if name in ('g2_x','g2_c') else None,
                padding='zero to stride' if name in ('g2_x','g2_c') else None))
            cursor+=size
        if cursor!=len(a['g2_table']):raise ValueError('g2_table: unexpected size')
        fields.append(entry('g2_sizes','g2_sizes',0,a['g2_sizes'],unit='count',used_by=[DIAGNOSTIC],
            generator=generators['g2'],provenance='g2_tables.pack dimensions'))
    return dict(format=FORMAT,fields=fields)


def verify(catalog, blobs, expected=None):
    if catalog.get('format') != FORMAT:raise ValueError('unknown catalog format')
    fields=catalog.get('fields',[]);names=[f['name'] for f in fields]
    if len(names)!=len(set(names)):raise ValueError('duplicate catalog field: '+next(n for n in names if names.count(n)>1))
    if expected is not None:
        want={f['name']:f for f in expected['fields']}
        missing=set(want)-set(names);extra=set(names)-set(want)
        if missing or extra:raise ValueError('catalog missing/unexpected field: '+','.join(sorted(missing|extra)))
    for f in fields:
        name=f['name']
        try:
            blob=np.asarray(blobs[f['blob']]).ravel();off=f['offset'];size=f['size'];shape=f['shape']
            if type(off) is not int or type(size) is not int or off<0 or size<0 or off+size>blob.size:
                raise ValueError('offset/size out of bounds')
            if any(type(d) is not int or d<0 for d in shape) or math.prod(shape)!=size:
                raise ValueError('shape/size mismatch')
            if blob.dtype.str!=f['dtype']:raise ValueError('dtype mismatch')
            if digest(blob[off:off+size])!=f['data_sha256']:raise ValueError('slice SHA256 mismatch')
            if expected is not None and f != want[name]:raise ValueError('field schema/grid/data mismatch')
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(name+': '+str(exc)) from exc
    return True


def reconstruct(catalog, blobs):
    verify(catalog,blobs)
    return {f['name']:np.asarray(blobs[f['blob']]).ravel()[f['offset']:f['offset']+f['size']].reshape(f['shape']).copy()
            for f in catalog['fields']}


# Fixed from A0 table 2; no prototype import in the runtime.
P8_DEFINITIONS = {'ne': {'unit': '電子/g',
        'provenance': '組成からの電子密度',
        'definition': 'scalar',
        'generator_path': 'prototypes/mv_electron/materials_p8a.py'},
 'excitation': {'unit': 'MeV',
                'provenance': 'ESTAR mean excitation',
                'definition': 'scalar',
                'generator_path': 'prototypes/mv_electron/fetch_materials_p8a.py'},
 'c': {'unit': 'MeV cm²/g',
       'provenance': '2π r_e² m_e ne',
       'definition': 'scalar',
       'generator_path': 'prototypes/mv_electron/photon_transport_p8e.py'},
 'z': {'unit': '原子番号',
       'provenance': '材料組成',
       'definition': 'nz',
       'generator_path': 'prototypes/mv_electron/materials_p8a.py'},
 'nj': {'unit': '原子/g',
        'provenance': '材料組成',
        'definition': 'nz',
        'generator_path': 'prototypes/mv_electron/materials_p8a.py'},
 'stop_T': {'unit': 'MeV',
            'provenance': 'NIST ESTAR',
            'definition': '1201 log 点 .01–20（P8a記録、4材料共通）',
            'generator_path': 'prototypes/mv_electron/fetch_materials_p8a.py'},
 'stop_delta': {'unit': '無次元',
                'provenance': 'ESTAR/密度効果。肺は補正 A',
                'definition': 'stop_T と同じ',
                'generator_path': 'prototypes/mv_electron/fetch_materials_p8a.py'},
 'stop_collision': {'unit': 'MeV cm²/g',
                    'provenance': 'NIST ESTAR collision stopping',
                    'definition': 'stop_T と同じ。解析モデルの検証束のみ空',
                    'generator_path': 'prototypes/mv_electron/photon_transport_p8e.py'},
 'stop_rad': {'unit': 'MeV cm²/g',
              'provenance': 'NIST ESTAR radiative stopping',
              'definition': 'stop_T と同じ',
              'generator_path': 'prototypes/mv_electron/photon_transport_p8e.py'},
 'gs_T': {'unit': 'MeV',
          'provenance': 'ELSEPA GS 積分',
          'definition': '49 log 点 .01–10',
          'generator_path': 'prototypes/mv_electron/gs_backend.py'},
 'gs_K': {'unit': '無次元',
          'provenance': 'step の比 K',
          'definition': '9 非等間隔点（.0005–.2）',
          'generator_path': 'prototypes/mv_electron/gs_backend.py'},
 'gs_u': {'unit': '確率',
          'provenance': '逆 CDF の確率軸',
          'definition': '20483 非等間隔点、log ではない',
          'generator_path': 'prototypes/mv_electron/single_table.py'},
 'gs_y': {'unit': '1−cosθ',
          'provenance': 'ELSEPA の GS 逆 CDF',
          'definition': '49 × 9 × 20483',
          'generator_path': 'prototypes/mv_electron/gs_backend.py'},
 'rates': {'unit': 'MeV; cm²/g',
           'provenance': 'ELSEPA rates（密度で除算）',
           'definition': '独立に収束した可変エネルギー格子。列 E,total,G1',
           'generator_path': 'prototypes/mv_electron/single_table.py'},
 'photo_E': {'unit': 'MeV',
             'provenance': 'NIST XCOM refined edges',
             'definition': '材料固有、吸収端の重複を保持。log 評価',
             'generator_path': 'prototypes/mv_electron/fetch_materials_p8a.py'},
 'photo': {'unit': 'cm²/g',
           'provenance': 'NIST XCOM photoelectric',
           'definition': 'photo_E と同じ',
           'generator_path': 'prototypes/mv_electron/fetch_materials_p8a.py'},
 'pair_E': {'unit': 'MeV',
            'provenance': 'NIST XCOM',
            'definition': 'XCOM 固有点、log 評価',
            'generator_path': 'prototypes/mv_electron/fetch_materials_p8a.py'},
 'pair_n': {'unit': 'cm²/g',
            'provenance': 'NIST XCOM nuclear pair',
            'definition': 'pair_E と同じ、閾値以下は 0',
            'generator_path': 'prototypes/mv_electron/fetch_materials_p8a.py'},
 'pair_e': {'unit': 'cm²/g',
            'provenance': 'NIST XCOM electron pair',
            'definition': 'pair_E と同じ、閾値以下は 0',
            'generator_path': 'prototypes/mv_electron/fetch_materials_p8a.py'},
 'thresholds': {'unit': 'MeV',
                'provenance': '対生成の閾値',
                'definition': '2',
                'generator_path': 'prototypes/mv_electron/fetch_materials_p8a.py'},
 'dl': {'unit': '無次元',
        'provenance': 'PEGS DECK/HATCH（研究用）',
        'definition': '6 × 6',
        'generator_path': 'prototypes/mv_electron/pegs_p8a.py'},
 'bpar': {'unit': '無次元',
          'provenance': 'PEGS（入力から先頭 1 値）',
          'definition': 'scalar',
          'generator_path': 'prototypes/mv_electron/photon_transport_p8e.py'},
 'delcm': {'unit': 'MeV',
           'provenance': 'PEGS screening',
           'definition': 'scalar',
           'generator_path': 'prototypes/mv_electron/pegs_p8a.py'},
 'delpos': {'unit': 'MeV⁻¹',
            'provenance': 'PEGS（入力から先頭 1 値）',
            'definition': 'scalar',
            'generator_path': 'prototypes/mv_electron/photon_transport_p8e.py'},
 'bindings': {'unit': 'MeV',
              'provenance': 'EGS5 block_data_atom（研究用）',
              'definition': 'nz × 5 shell',
              'generator_path': 'prototypes/mv_electron/pegs_p8a.py'},
 'pm': {'unit': 'log(keV) 多項式の係数',
        'provenance': 'EGS5 shell fit（研究用）',
        'definition': 'nz × 5 × 4',
        'generator_path': 'prototypes/mv_electron/pegs_p8a.py'},
 'shell_nj': {'unit': '原子/g',
              'provenance': 'shell 用材料組成',
              'definition': 'nz',
              'generator_path': 'prototypes/mv_electron/pegs_p8a.py'},
 'yields': {'unit': '確率',
            'provenance': 'xraylib fluorescence yields',
            'definition': 'nz × 4 shell',
            'generator_path': 'prototypes/mv_electron/generate_fluorescence_p8e.py'},
 'lines': {'unit': 'MeV; 確率; child shell; line id',
           'provenance': 'xraylib fluorescence lines',
           'definition': 'nz × 4 × nline × 4',
           'generator_path': 'prototypes/mv_electron/generate_fluorescence_p8e.py'},
 'line_counts': {'unit': '個数',
                 'provenance': '有効線数',
                 'definition': 'nz × 4',
                 'generator_path': 'prototypes/mv_electron/generate_fluorescence_p8e.py'},
 'brems_E': {'unit': 'MeV',
             'provenance': 'ESTAR 規格化の解析 brems',
             'definition': 'ESTAR ノード + log 刻み≤.003 + cutoff 近傍 401 点、可変長',
             'generator_path': 'prototypes/mv_electron/bremsstrahlung_p3r.py'},
 'brems_V': {'unit': 'cm²/g; MeV cm²/g; cm²/g; 無次元A',
             'provenance': 'ESTAR 規格化の解析 brems',
             'definition': 'len(brems_E) × 5（rate,hard,soft,envelope,A）',
             'generator_path': 'prototypes/mv_electron/bremsstrahlung_p3r.py'},
 'log_stop_T': {'unit': 'ln(MeV)',
                'provenance': 'stop_T から生成',
                'definition': 'stop_T の対数',
                'generator_path': 'prototypes/mv_electron/photon_transport_p8e.py'}}
P8_GRIDS = {**dict.fromkeys(('stop_T','stop_delta','stop_collision','stop_rad','log_stop_T'),'stop_T'),
            **dict.fromkeys(('gs_T','gs_y'),'gs_T'), **dict.fromkeys(('photo_E','photo'),'photo_E'),
            **dict.fromkeys(('pair_E','pair_n','pair_e'),'pair_E'),
            **dict.fromkeys(('brems_E','brems_V'),'brems_E'), 'rates':'rates[:,0]'}
# Explicit heterogeneous component units, not one unit for the whole array.
P8_DEFINITIONS['lines']['unit']=['MeV','probability','child-shell','line-id']
P8_DEFINITIONS['brems_V']['unit']=['cm2/g','MeV cm2/g','MeV cm2/g','cm2/g','dimensionless']


def p8_material(material, material_name, generators):
    """Input is a duck-typed material; preserve every original dimension."""
    fields=[];chunks=[];cursor=0
    for name,definition in P8_DEFINITIONS.items():
        a=np.asarray(getattr(material,name),dtype=np.float64)
        grid_name=P8_GRIDS.get(name)
        grid=(np.asarray(material.rates)[:,0] if grid_name=='rates[:,0]' else
              np.asarray(getattr(material,grid_name)) if grid_name else None)
        band=None if grid is None else [float(grid[0]*1000),float(grid[-1]*1000)]
        # stop_collision may be empty in analytical validation materials.
        meta=dict(material=material_name,grid_reference=grid_name,
                  valid_length=None if grid is None else int(len(grid)),padding=None,
                  log_transform='ln(MeV)' if name=='log_stop_T' else None,
                  energy_to_keV=1000 if grid is not None or name in ('excitation','thresholds','bindings','lines','delcm') else None,
                  density_factor='divide by material density' if name=='rates' else None,
                  definition=definition['definition'])
        if name=='gs_y':meta['axes']=['gs_T','gs_K','gs_u']
        if name=='lines':meta['valid_length']='line_counts';meta['padding']='unused lines'
        fields.append(entry(material_name+'.'+name,'p8',cursor,a,unit=definition['unit'],
            used_by=[THERAPY],generator=generators[definition['generator_path']],
            provenance=definition['provenance'],grid=grid,grid_kind=grid_name,band=band,**meta))
        chunks.append(a.ravel());cursor+=a.size
    return dict(format=FORMAT,fields=fields), np.concatenate(chunks)


def therapy(materials, generators):
    """Four independent blobs: offsets restart only inside a named material."""
    names=('water','lung','bone','lead')
    if len(materials)!=4:raise ValueError('four therapy materials required')
    fields=[];blobs={}
    for name,material in zip(names,materials):
        cat,blob=p8_material(material,name,generators)
        key='p8_'+name
        for field in cat['fields']:field['blob']=key
        fields.extend(cat['fields']);blobs[key]=blob
    return dict(format=FORMAT,fields=fields),blobs
