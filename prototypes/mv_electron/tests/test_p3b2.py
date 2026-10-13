"""P3b-2 standalone-driver and refusal tests; never launch EGS5."""
import ast
import json
import struct
import subprocess
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron import run_p3b2 as m
from prototypes.mv_electron.tests import test_p3b1 as previous


def point_cases():
    rows = []
    for mark in previous.test_b4_points.pytestmark:
        if mark.name == 'parametrize' and mark.args[0] == 'point,expected':
            rows.extend(mark.args[1])
    assert rows and len(rows) == 5
    return rows


@pytest.fixture
def score_driver(tmp_path):
    return m.build_driver(tmp_path, 'p3b_score_driver')


@pytest.mark.parametrize('p0,p1,axis,expected,interval', previous.GEOMETRY)
def test_e2a_segments(tmp_path, score_driver, p0, p1, axis, expected, interval):
    g = m.read_regions('1\n0 0 0 2 2 2 %d 0 1 2\n'%(axis+1))
    actual = m.driver_scores(tmp_path, score_driver, g, [[*p0,*p1,4.]])[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('p,expected', point_cases())
def test_e2a_points(tmp_path, score_driver, p, expected):
    g = m.read_regions('1\n0 0 0 2 2 2 3 0 1 2\n')
    actual = m.driver_scores(tmp_path, score_driver, g, [[*p,*p,4.]])[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_e2a_entry(tmp_path, score_driver):
    assert m.analytic_check(tmp_path, score_driver)['status'] == '合格'


def test_e2d(tmp_path):
    assert m.ausgab_check(tmp_path, m.build_driver(tmp_path, 'p3b_ausgab_driver'))['status'] == '合格'


def test_regions_roundtrip_and_inf(tmp_path, score_driver):
    g = m.geometry(m.REGIONS)
    text = m.regions_text(g)
    assert 'inf' in text
    back = m.read_regions(text)
    for key in back:
        np.testing.assert_array_equal(back[key], g[key])
    # Infinity is also read by Fortran, not only Python.
    rows = m.driver_scores(tmp_path, score_driver, g, [[0.,0.,0.,0.,0.,30.,30.]])
    np.testing.assert_allclose(rows[2], .25, rtol=0, atol=1e-12)


@pytest.mark.parametrize('count,size', [(17,2),(1,321),(0,2),(1,0)])
def test_limits_python(count,size):
    text = str(count)+'\n'+('0 0 0 2 2 2 3 0 1 %d\n'%size)*count
    with pytest.raises(ValueError):
        m.read_regions(text)


@pytest.mark.parametrize('count,size', [(17,2),(1,321)])
def test_limits_fortran(tmp_path, score_driver,count,size):
    (tmp_path/'regions.dat').write_text(str(count)+'\n'+('0 0 0 2 2 2 3 0 1 %d\n'%size)*count)
    (tmp_path/'segments.dat').write_text('0 0 0 1 1 1 1\n')
    result = subprocess.run([str(score_driver)], cwd=tmp_path, capture_output=True)
    assert result.returncode != 0


def extra_text(N=1000,field=(5.,100.),regions=m.REGIONS):
    g = m.geometry(regions)
    lines = ['FIELD '+' '.join(map(str,m.field_values(field))), 'NREG '+str(len(m.box_rows(g)))]
    for j,row in enumerate(m.box_rows(g),1):
        values = ['%.17g'%x for x in row]
        lines.append('REGION '+str(j)+' '+' '.join(values))
        lines.append(f'RLEDGER {j} 0')
        for b in range(1,101):
            for k in range(1,int(row[-1])+1):
                lines.append(f'REG {b} {j} {k} {b*100000+j*1000+k}')
    for x in range(1,11):
        for y in range(1,11):
            lines.append(f'SRCHIST {x} {y} {N if (x,y)==(6,6) else 0}')
    for h in range(1,min(N,1000)+1):
        lines.append(f'SRC {h} 0 0 0 0 1')
    return '\n'.join(lines)+'\n'


@pytest.fixture(scope='module')
def text():
    return extra_text()


def test_parse_shapes_and_mapping(text):
    out = m.parse_extra(text,1000,[5.,100.],m.REGIONS)
    for key,shape in zip(m.REGION_KEYS,[(100,120),(100,300),(100,120),(100,2,3,72),(100,2,3,144)]):
        assert out[key].shape == shape
    assert out['axis_coarse'][1,2] == 201003
    assert out['profile'][1,1,2,3] == 209004
    assert out['profile_fine'][1,1,2,3] == 215004
    assert out['region_ledger'].shape == (15,)
    assert out['sources'].shape == (1000,5)
    assert out['source_hist'].sum() == 1000


# Every malformed string exercises the public strict parser.
@pytest.mark.parametrize('old,new',[
    ('FIELD 1 5.0 100.0','FIELD 1 4.0 100.0'),
    ('FIELD 1 5.0 100.0','FIELD 1 nan 100.0'),
    ('FIELD 1 5.0 100.0','FIELD 1 5.0 100.0 0'),
    ('FIELD 1 5.0 100.0',''),
    ('FIELD 1 5.0 100.0','FIELD 1 5.0 100.0\nFIELD 1 5.0 100.0'),
    ('NREG 15','NREG 16'), ('NREG 15','NREG 15.0'), ('NREG 15',''),
    ('NREG 15','NREG 15 0'), ('NREG 15','NREG 15\nNREG 15'),
    ('REGION 1 -1 -1 0 1 1 30 3 0 0.25 120',''),
    ('REGION 1 -1 -1 0 1 1 30 3 0 0.25 120','REGION 1 -1 -1 0 1 1 30 3 0 0.25 120\nREGION 1 -1 -1 0 1 1 30 3 0 0.25 120'),
    ('REGION 1 -1 -1 0 1 1 30 3 0 0.25 120','REGION 1 -1 -1 0 1 1 30 3 0 0.25 120 0'),
    ('RLEDGER 1 0','RLEDGER 1 0\nRLEDGER 1 0'),
    ('RLEDGER 1 0','RLEDGER 1 0 0'),
    ('REGION 1 -1 -1 0 1 1 30 3 0 0.25 120','REGION 1 -1 -1 0 2 1 30 3 0 0.25 120'),
    ('REGION 1 -1 -1 0 1 1 30 3 0 0.25 120','REGION 16 -1 -1 0 1 1 30 3 0 0.25 120'),
    ('REGION 1 -1 -1 0 1 1 30 3 0 0.25 120','REGION 1 -1 -1 0 1 1 30 3 inf 0.25 120'),
    ('REGION 1 -1 -1 0 1 1 30 3 0 0.25 120','REGION 1 nan -1 0 1 1 30 3 0 0.25 120'),
    ('REG 1 1 1 101001',''),
    ('REG 1 1 1 101001','REG 1 1 1 101001\nREG 1 1 1 101001'),
    ('REG 1 1 1 101001','REG 101 1 1 101001'),
    ('REG 1 1 1 101001','REG 0 1 1 101001'),
    ('REG 1 1 1 101001','REG 1 16 1 101001'),
    ('REG 1 1 1 101001','REG 1 1 121 101001'),
    ('REG 1 1 1 101001','REG 1 1 1 101001 2'),
    ('REG 1 1 1 101001','REG 1 1 1'),
    ('REG 1 1 1 101001','REG 1 1 1 nan'),
    ('REG 1 1 1 101001','REG 1 1 1 inf'),
    ('REG 1 1 1 101001','REG 1 1 1 -1'),
    ('REG 1 1 1 101001','REG 1.5 1 1 0'),
    ('RLEDGER 1 0','RLEDGER 1 -1'),
    ('RLEDGER 1 0','RLEDGER 1 0.5'),
    ('RLEDGER 1 0','RLEDGER 16 0'),
    ('RLEDGER 1 0',''),
    ('SRC 1 0 0 0 0 1',''),
    ('SRC 1 0 0 0 0 1','SRC 1 0 0 0 0 1\nSRC 1 0 0 0 0 1'),
    ('SRC 1 0 0 0 0 1','SRC 1001 0 0 0 0 1'),
    ('SRC 1 0 0 0 0 1','SRC 0 0 0 0 0 1'),
    ('SRC 1 0 0 0 0 1','SRC 1 nan 0 0 0 1'),
    ('SRC 1 0 0 0 0 1','SRC 1 0 0 0 0 inf'),
    ('SRC 1 0 0 0 0 1','SRC 1 0 0 0 1'),
    ('SRCHIST 1 1 0',''),
    ('SRCHIST 1 1 0','SRCHIST 1 1 0\nSRCHIST 1 1 0'),
    ('SRCHIST 1 1 0','SRCHIST 11 1 0'),
    ('SRCHIST 1 1 0','SRCHIST 1 1 -1'),
    ('SRCHIST 1 1 0','SRCHIST 1 1 1.1'),
    ('SRCHIST 1 1 0','SRCHIST 1 1 1'),
])
def test_malformed_output(text,old,new):
    assert old in text
    with pytest.raises((RuntimeError,ValueError)):
        m.parse_extra(text.replace(old,new,1),1000,[5.,100.],m.REGIONS)


def test_no_field_or_regions():
    v = m.parse_extra(extra_text(field=None,regions=None),1000,None,None)
    assert v['region_ledger'].size == 0
    for key in m.REGION_KEYS:
        assert v[key].shape == (100,0)


def fake_record(data):
    return struct.pack('=i',len(data))+data+struct.pack('=i',len(data))


def synthetic_checkpoint(config,done=80000):
    state = struct.pack('=i',done)+bytes(607600-4)
    extra = bytes(8*320*16*100+4*16+4*100+8*5*1000+4)
    return b''.join(fake_record(x) for x in (m.checkpoint_header(config),state,extra))


@pytest.fixture
def stopped(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'CACHE',tmp_path/'cache')
    monkeypatch.setattr(m,'OUT',tmp_path/'out')
    monkeypatch.setattr(m,'CHECKS',tmp_path/'checks.json')
    monkeypatch.setattr(m,'egs_hashes',lambda: {})
    work = m.CACHE/'egs5'/'resume'; dest = m.OUT/'egs5'/'resume'
    work.mkdir(parents=True); dest.mkdir(parents=True)
    config = dict(N=200000,seed=20291751,E=10.,IUNRST=0,noflight=0,thickness=30.,batches=100,
                  input=m.r.pegs_input(0),user_code='p3b_egs',field=[5.,100.],regions=m.REGIONS,
                  egs_input_hashes={},source_hashes=m.source_hashes(),**m.r.spec_config(None,None,m.r.SPECTRUM_SHA))
    for parent in (work,dest):
        (parent/'config.json').write_text(json.dumps(config))
    for name in m.SOURCES[:4]:
        (work/name).write_bytes((m.CODE/name).read_bytes())
    (work/'config.dat').write_text('10 200000 20291751 0\n')
    (work/'p3b_egs.inp').write_text(config['input'])
    (work/'field.dat').write_text('5 100\n')
    (work/'regions.dat').write_text(m.regions_text(m.geometry(m.REGIONS)))
    (work/'checkpoint.bin').write_bytes(synthetic_checkpoint(config))
    def forbidden(*args,**kwargs):
        pytest.fail('EGS5 launch/build must not be called')
    monkeypatch.setattr(m,'launch',forbidden)
    monkeypatch.setattr(m.subprocess,'run',forbidden)
    return work,dest,config


@pytest.mark.parametrize('target',['field.dat','regions.dat','p3b_egs_score.inc'])
def test_e5_refuses_changed_files_before_process(stopped,target):
    work,_,_ = stopped
    (work/target).write_text((work/target).read_text()+'changed\n')
    with pytest.raises(RuntimeError,match='mismatch'):
        m.run_egs('resume',200000,20291751,None,None,field=[5.,100.],regions=m.REGIONS,resume=True)


@pytest.mark.parametrize('target',m.SOURCES)
def test_e5_source_hash_before_process(stopped,monkeypatch,target):
    original=m.source_hashes
    monkeypatch.setattr(m,'source_hashes',lambda:dict(original(),**{str((m.CODE/target).relative_to(m.ROOT)):'changed'}))
    with pytest.raises(RuntimeError,match='config mismatch'):
        m.run_egs('resume',200000,20291751,None,None,field=[5.,100.],regions=m.REGIONS,resume=True)


@pytest.mark.parametrize('key,value',[('field',[4.,100.]),('regions',[.1]+m.REGIONS[1:])])
def test_e5_changed_configuration(stopped,key,value):
    kw=dict(field=[5.,100.],regions=m.REGIONS,resume=True);kw[key]=value
    with pytest.raises(RuntimeError,match='config mismatch'):
        m.run_egs('resume',200000,20291751,None,None,**kw)


def test_checkpoint_valid(stopped):
    work,_,config=stopped
    assert m.verify_checkpoint(work,config)==80000


@pytest.mark.parametrize('done',[0,-1,200001,80001])
def test_checkpoint_progress(stopped,done):
    work,_,config=stopped
    (work/'checkpoint.bin').write_bytes(synthetic_checkpoint(config,done))
    with pytest.raises(RuntimeError):
        m.run_egs('resume',200000,20291751,None,None,field=[5.,100.],regions=m.REGIONS,resume=True)


@pytest.mark.parametrize('mutation',['truncated','wrong_magic','changed_header','missing_record','wrong_marker'])
def test_checkpoint_bad(stopped,mutation):
    work,_,config=stopped
    data=synthetic_checkpoint(config)
    if mutation=='truncated':data=data[:-1]
    elif mutation=='wrong_magic':data=data[:4]+b'X'+data[5:]
    elif mutation=='changed_header':data=synthetic_checkpoint(dict(config,seed=1))
    elif mutation=='missing_record':data=fake_record(m.checkpoint_header(config))
    else:data=bytes(4)+data[4:]
    (work/'checkpoint.bin').write_bytes(data)
    with pytest.raises(RuntimeError):
        m.run_egs('resume',200000,20291751,None,None,field=[5.,100.],regions=m.REGIONS,resume=True)


@pytest.mark.parametrize('target',['field.dat','regions.dat'])
def test_unexpected_input(tmp_path,target):
    (tmp_path/target).write_text('unexpected')
    with pytest.raises(RuntimeError,match='unexpected'):
        m.verify_inputs(tmp_path,{})


def test_existing_job_is_not_new(stopped):
    with pytest.raises(RuntimeError,match='existing job'):
        m.run_egs('resume',200000,20291751,None,None,field=[5.,100.],regions=m.REGIONS)


def test_failed_checks_block(stopped):
    m.CHECKS.write_text(json.dumps({'E4':dict(status='不合格')}))
    with pytest.raises(RuntimeError,match='previous'):
        m.run_egs('resume',200000,20291751,None,None,field=[5.,100.],regions=m.REGIONS,resume=True)


def test_equality_excludes_time():
    a={key:np.array([1.]) for key in m.EQUALITY_KEYS}
    b=dict(a,seconds=99.)
    assert all(m.equality(a,b).values())
    for key in m.EQUALITY_KEYS:
        assert not all(m.equality(a,dict(b,**{key:np.array([2.])})).values())


def test_no_deletion_commands():
    banned={'unlink','remove','rmtree','rmdir','removedirs'}
    for name in ('run_p3b2.py','tests/test_p3b2.py'):
        tree=ast.parse((m.CODE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                assert getattr(node.func,'id',None) not in banned
                assert getattr(node.func,'attr',None) not in banned
    for name in m.SOURCES[:6]:
        text=(m.CODE/name).read_text().lower()
        assert 'call system(' not in text
        assert "status='delete'" not in text


@pytest.mark.parametrize('target',['field.dat','regions.dat'])
def test_absent_configuration_refuses_leftover_before_process(stopped,target):
    work,dest,config=stopped
    key='field' if target=='field.dat' else 'regions'
    config=dict(config,**{key:None})
    for parent in (work,dest):
        (parent/'config.json').write_text(json.dumps(config))
    with pytest.raises(RuntimeError,match='unexpected '+target):
        m.run_egs('resume',200000,20291751,None,None,
                  field=config['field'],regions=config['regions'],resume=True)


def test_fixed_form_and_preserved_depth_and_howfar():
    original=(m.CODE/'p3r_egs.f').read_text().replace('p3r_egs_common.inc','p3b_egs_common.inc')
    depth=original[original.index('      subroutine p2point'):original.index('      subroutine howfar')]
    howfar=original[original.index('      subroutine howfar'):]
    assert depth in (m.CODE/'p3b_egs.f').read_text()
    assert depth in (m.CODE/'p3b_ausgab_driver.f').read_text()
    assert (m.CODE/'p3b_egs_ausgab.inc').read_text().endswith(howfar)
    for name in m.SOURCES[:6]:
        assert all(len(line)<=72 or line.startswith('!')
                   for line in (m.CODE/name).read_text().splitlines())
