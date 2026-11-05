"""D2: sealed XCOM incoherent column, production native totals and legacy KN."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from chatcarlo.gpu import catalog,therapy,seed_core
from chatcarlo.gpu.generate_photon_tables import ensure,material
from docs.validation.gpu.g5 import a4u_h1

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def real_tables():
    sys.path.insert(0,str(ROOT/'docs/validation/gpu/g5'))
    import a2_common
    _,m,a=a2_common.inputs()
    return m,a


@pytest.fixture(scope='module')
def h1(real_tables):
    m,a=real_tables
    return a4u_h1.report(m,('water','lung','bone','lead'),a['constants'])


def raw_rows(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        parts=line.split()
        if len(parts)==8:
            try:rows.append([float(v) for v in parts])
            except ValueError:pass  # XCOM text headings, never numeric samples
    return rows


def test_elemental_incoherent_uses_sealed_column(real_tables):
    m,_=real_tables
    for mat in m:
        sources=[];p=ensure(mat,sources=sources);nz=int(p[0])
        for i in range(nz):
            z=int(p[2+17*i]);files=[s for s in sources if s['kind']=='elemental' and s['Z']==z]
            assert files
            rows=[]
            for s in files:
                path=Path(s['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==s['sha256']
                rows.extend(raw_rows(path))
            raw=np.array(rows)
            for energy in (1000.,2000.,10000.):
                expected=np.exp(np.interp(np.log(energy),np.log(raw[:,0]*1000),np.log(raw[:,2])))
                actual=a4u_h1.elemental_value(p,i,energy,1)
                assert actual==pytest.approx(expected,rel=2e-13)
        tail=int(p[1]);nf=int(p[tail]);assert nf>0
        assert len(p)==tail+1+3*nz+4*nf


def test_native_totals_and_lead_seam(h1):
    for name,row in h1['materials'].items():
        for d in row['xs_vs_XCOM']:
            if d['keV']>=1000:
                # Mixtures include the original input composition precision.
                assert abs(d['relative_difference'])<1e-4
    lead=h1['materials']['lead']
    new=next(j for j in lead['joins'] if j['channel']=='compt')
    previous=next(j for j in lead['previous_joins'] if j['channel']=='compt')
    assert abs(new['relative_step'])<abs(previous['relative_step'])
    assert new['above']==pytest.approx(new['XCOM_above'],rel=2e-13)
    assert set(h1['fallback_columns'])=={'energy_keV','photo','compt','rayleigh'}


def test_photon_hashes_include_all_four_columns(real_tables):
    from chatcarlo.gpu.therapy_observation import table_hashes
    m,_=real_tables;data,meta=therapy.pack(m);observed=table_hashes(data,meta)
    for i,mat in enumerate(m):
        expected=catalog.digest(ensure(mat))
        assert observed[str(i)+'.photon']==expected
        assert seed_core.photon_digest(data,meta[i])==expected


def test_missing_incoherent_column_rejected(real_tables):
    p=ensure(real_tables[0][3]);tail=int(p[1]);nz=int(p[0]);nf=int(p[tail])
    table=p[tail+1+3*nz:].reshape(4,nf)
    old_three=(table[0],table[1],table[3])
    with pytest.raises(ValueError):material([82],[1.],xcom=old_three,elemental_xcom={82:old_three})


@pytest.mark.parametrize('legacy',[0,1])
def test_analytic_kn_total_only_compiles_for_legacy(legacy,tmp_path):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if compiler is None:pytest.fail('C++ compiler required')
    source=tmp_path/'active.cpp';source.write_text('#include "t2_cpu.cpp"\n')
    active=subprocess.check_output([compiler,'-E','-P','-x','c++','-DP5_RNG_PHILOX',f'-DG5_TEST_LEGACY_ELECTRON={legacy}','-I',str(therapy.HERE),str(source)],text=True)
    assert ('p5_kn_total(' in active)==bool(legacy)
    generator=(therapy.HERE/'generate_photon_tables.py').read_text()
    assert 'CS_KN' not in generator
