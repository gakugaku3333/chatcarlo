"""Part 1 tests: standalone Fortran only; never run EGS5 or write ~/.cache."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from prototypes.mv_electron import run_p8e as p

# Artificial test densities; physical definitions are only loaded by the runner.
RHO = [1., .25, 2., 10.]


def test_layout_hand_calculated():
    g = p.layout(p.LAYERS, RHO)
    assert len(g['mass_g_cm2']) == 220
    np.testing.assert_allclose(g['edges_cm'][[0, 50, 110, 150, 170, 210, 220]],
                               [0, 5, 11, 15, 17, 21, 21.5])
    np.testing.assert_array_equal(g['bin_layer'], [0]*50+[1]*60+[2]*40+[3]*20+[4]*40+[5]*10)
    np.testing.assert_allclose(g['mass_g_cm2'], [.1]*50+[.025]*60+[.1]*40+[.2]*20+[.1]*40+[.5]*10)
    np.testing.assert_allclose(g['blocks_cm'], [(i, i+1) for i in range(3, 21)]+[(21, 21.5)])
    np.testing.assert_array_equal(g['block_index'], [-1]*30+sum(([i]*10 for i in range(19)), []))
    np.testing.assert_allclose(g['block_weights'].sum(axis=1), 1)
    assert np.dot(g['mass_g_cm2'], np.ones(220)) == pytest.approx(23.5)


def test_correlated_batches_and_projection():
    g = p.layout(p.LAYERS, RHO)
    d = np.repeat(np.arange(1., 101.)[:, None], 220, axis=1)
    values, mean, sem = p.block_statistics(d, g)
    np.testing.assert_allclose(values, np.repeat(np.arange(1., 101.)[:, None], 19, axis=1))
    np.testing.assert_allclose(mean, 50.5)
    np.testing.assert_allclose(sem, np.std(np.arange(1., 101.), ddof=1)/10)
    result = p.projection(d, g, 1000)
    assert result['projected_CPU_seconds'] == 60000
    assert result['exceeds_12_hours']
    expected = sem[0]/np.sqrt(60)/50.5
    np.testing.assert_allclose(result['projected_SEM_over_Dref'], expected)


@pytest.fixture(scope='module')
def fortran(tmp_path_factory):
    compiler = shutil.which('gfortran')
    if not compiler:
        pytest.fail('gfortran required for Part 1 acceptance tests')
    work = tmp_path_factory.mktemp('p8e_fortran')
    source = p.CODE.read_text()
    # Check the complete user code against real EGS declarations; no link/run.
    subprocess.run([compiler, '-std=legacy', '-fcheck=all', '-c',
                    '-I'+str(p.EGS), str(p.CODE), '-o', str(work/'user.o')], check=True, capture_output=True)
    routines = source[source.index('      subroutine p8config'):]
    driver = '''      program test
      implicit none
      integer nl,nt,lmed(16),first(16),last(16)
      real*8 bound(0:16),bw(16),bleft(1000),bright(1000)
      common/p8geom/bound,bw,bleft,bright,nl,nt,lmed,first,last
      integer n,seed,fluor,media_set,reg,j,ios
      real*8 z0,z1,loss,score(1000)
      call p8config(n,seed,fluor,media_set)
      do
        read(*,*,iostat=ios) reg,z0,z1,loss
        if(ios.ne.0) exit
        score=0.d0
        call p8deposit(reg,z0,z1,loss,score)
        write(*,*) (score(j),j=1,nt)
      enddo
      end
'''
    (work/'test.f').write_text(driver+routines)
    subprocess.run([compiler, '-std=legacy', '-fcheck=all', str(work/'test.f'),
                    '-o', str(work/'test')], check=True, capture_output=True)
    integrated = """      program integration
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
      integer nl,nt,lmed(16),first(16),last(16)
      real*8 bound(0:16),bw(16),bleft(1000),bright(1000)
      common/p8geom/bound,bw,bleft,bright,nl,nt,lmed,first,last
      real*8 p8score(1000),p8dep,p8escape,z0,dir,step,loss
      common/p8score/p8score,p8dep,p8escape
"""
    integrated += """      integer n,seed,fluor,media_set,reg,j,ios
      call p8config(n,seed,fluor,media_set)
      do
        read(*,*,iostat=ios) reg,z0,dir,step,loss
        if(ios.ne.0) exit
        p8score=0.d0
        p8dep=0.d0
        p8escape=0.d0
        np=1
        ir(1)=reg
        z(1)=z0
        w(1)=dir
        iq(1)=-1
        wt(1)=1.d0
        ustep=step
        irnew=reg
        idisc=0
        edep=loss
        call howfar
        call ausgab(0)
        write(*,*) (p8score(j),j=1,nt),ustep,irnew,idisc
      enddo
      end
"""
    callbacks = source[source.index('      subroutine ausgab'):source.index('! Standalone routines:')]
    (work/'integration.f').write_text(integrated+callbacks+routines)
    subprocess.run([compiler, '-std=legacy', '-fcheck=all', '-I'+str(p.EGS),
                    str(work/'integration.f'), '-o', str(work/'integration')],
                   check=True, capture_output=True)
    return work


def drive(work, config, events=''):
    (work/'config.dat').write_text(config)
    return subprocess.run([str(work/'test')], cwd=work, input=events, text=True, capture_output=True)


@pytest.mark.parametrize('config,code', [
    ('0 1 1 1 4\n1 1 .1\n', 41), ('101 1 1 1 4\n1 1 .1\n', 41),
    ('100 1 2 1 4\n1 1 .1\n', 42), ('100 1 -1 1 4\n1 1 .1\n', 42),
    ('100 1 1 0 4\n', 43), ('100 1 1 17 4\n', 43),
    ('100 1 1 1 4\n0 1 .1\n', 44), ('100 1 1 1 4\n5 1 .1\n', 44),
    ('100 1 1 1 4\n1 NaN .1\n', 45), ('100 1 1 1 4\n1 1 Inf\n', 45),
    ('100 1 1 1 4\n1 0 .1\n', 46), ('100 1 1 1 4\n1 -1 .1\n', 46),
    ('100 1 1 1 4\n1 1 0\n', 46), ('100 1 1 1 4\n1 1 -.1\n', 46),
    ('100 1 1 1 4\n1 1 .3\n', 47), ('100 1 1 1 4\n1 .1 1\n', 47),
    ('100 1 1 1 4\n1 1001 1\n', 48),
    ('100 1 1 2 4\n1 600 1\n2 401 1\n', 48),
    ('100 1 1 1 4\n1 1d308 1d-308\n', 48),
])
def test_fortran_input_stops(fortran, config, code):
    result = drive(fortran, config)
    assert result.returncode == code, result.stderr
    assert f'STOP {code}' in result.stderr


def test_fortran_fine_bin_attribution(fortran):
    # Two adjacent layers of the SAME material, then thin lead bins.
    config = '100 1 1 3 4\n1 1 .25\n1 1 .25\n4 .5 .125\n'
    events = '''2 .125 .875 6
2 .875 .125 6
2 .3 .3 2
2 1 1 3
3 1 1 3
2 0 0 4
4 2.5 2.5 5
3 1.25 1.25 7
4 2.0625 2.4375 6
4 2.4375 2.0625 6
'''
    result = drive(fortran, config, events)
    assert result.returncode == 0, result.stderr
    actual = np.array([list(map(float, line.split())) for line in result.stdout.splitlines()])
    expected = np.zeros((10, 12))
    expected[0, :4] = expected[1, :4] = [1, 2, 2, 1]
    expected[2, 1] = 2
    expected[3, 3] = 3
    expected[4, 4] = 3
    expected[5, 0] = 4
    expected[6, 11] = 5
    # w=0 charged step gives z1=z0 and uses the same point branch.
    expected[7, 5] = 7
    expected[8, 8:] = expected[9, 8:] = [1, 2, 2, 1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


def test_fortran_fixed_layout_and_integer_tolerance(fortran):
    result = drive(fortran, p.config_text(100, 1, p.LAYERS), '2 0 .1 1\n')
    assert result.returncode == 0
    assert len(result.stdout.split()) == 220
    assert drive(fortran, '100 1 1 1 4\n1 1.0000000005 .1\n').returncode == 0
    assert drive(fortran, '100 1 1 1 4\n1 1.000000002 .1\n').returncode == 47
    for t, w in [(0.9999999995, 1.), (1000.0000005, 1.)]:
        config = p.config_text(100, 1, [(1, t, w)])
        assert drive(fortran, config).returncode == 0


def output():
    """Synthetic output with independent manually specified layer/bin counts."""
    lines = ['CONFIG 100 9 1', 'SETTING 0 1']
    layers = [(1, 5, .1, 50), (2, 6, .1, 60), (1, 4, .1, 40),
              (3, 2, .1, 20), (1, 4, .1, 40), (4, .5, .05, 10)]
    for i, row in enumerate(layers, 1):
        lines.append('LAYER '+str(i)+' '+' '.join(map(str, row)))
    for i, medium in enumerate([0, 1, 2, 1, 3, 1, 4, 0], 1):
        rho = RHO[medium-1] if medium else 0
        fluor = int(medium > 0)
        lines.append(f'REGION {i} {medium} {rho} .521 .01 {fluor} {fluor} 0 0 0 0 1 0 0')
    for i, name in enumerate(['WATER', 'LUNG', 'BONE', 'LEAD'], 1):
        lines.append(f'MEDIUM {i} P8E_{name} {RHO[i-1]} .521 .01 1 .5 0 0 0 0')
    # Constant dose .01, mass 23.5 g/cm2, N=100 => deposition 23.5 MeV.
    lines += ['LEDGER 100 23.5 76.5 1e-8 0 1e-14', 'CPU 1']
    lines.extend(f'DOSE {b} {j} .01' for b in range(1, 101) for j in range(1, 221))
    return '\n'.join(lines+['DONE'])+'\n'


def test_parser():
    dose, health = p.parse_egs(output(), 100, 9, p.LAYERS, RHO)
    np.testing.assert_array_equal(dose, np.full((100, 220), .01))
    assert health['energy_balance']['nleft'] == 0


@pytest.mark.parametrize('old,new', [
    ('SETTING 0 1', 'SETTING 1 1'), ('CONFIG 100 9 1', 'CONFIG 100 8 1'),
    ('P8E_LUNG', 'P8E_BONE'), ('MEDIUM 2 P8E_LUNG 0.25', 'MEDIUM 2 P8E_LUNG 2'),
    ('REGION 3 2', 'REGION 3 3'), ('0.521', '0.6'),
    (' .521 .01 1 1 0 0 0 0', ' .521 .01 0 1 0 0 0 0'),
    ('76.5 1e-8 0 1e-14', '76.5 1e-8 1 1e-14'),
    ('76.5 1e-8 0 1e-14', '76.5 2e-6 0 1e-14'),
    ('76.5 1e-8 0 1e-14', '76.5 1e-8 0 2e-12'),
    ('100 23.5 76.5', '100 23.5 76.4'), ('DOSE 1 1 .01', 'DOSE 1 1 NaN'),
    ('DOSE 1 1 .01', 'DOSE 1 1 -.01'), ('DOSE 1 1 .01', 'DOSE 1 2 .01'),
    ('DOSE 1 1 .01\n', ''), ('DOSE 1 1 .01', 'DOSE 101 1 .01'),
    ('DONE', ''), ('CPU 1', 'CPU 1\nCPU 1'), ('LAYER 1 1 5 0.1 50', 'bad'),
])
def test_parser_rejects(old, new):
    text = output()
    # The fixture uses .521 rather than 0.521.
    old = old.replace('0.521', '.521')
    assert old in text
    with pytest.raises(ValueError):
        p.parse_egs(text.replace(old, new, 1), 100, 9, p.LAYERS, RHO)


def test_s7_replay_threshold_and_zero():
    old = np.arange(5500.).reshape(100, 55)
    fine = np.repeat(old, 2, axis=1)
    assert p.replay_comparison(fine, old)['status'] == 'pass'
    fine[0, 0] = 1e-30
    with pytest.raises(ValueError, match='stop'):
        p.replay_comparison(fine, old)
    fine = np.repeat(old, 2, axis=1)*1.000000000002
    with pytest.raises(ValueError):
        p.replay_comparison(fine, old)


def test_execute_seal_resume_failure_and_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'external_sources', lambda: {'test': 'hash'})
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        for name in ('egs5job.exe', 'pgs5job.pegs5dat', 'pgs5job.pegs5lst'):
            (tmp_path/name).write_text('test')
        (tmp_path/'egs5job.out').write_text('good')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(p.subprocess, 'run', fake)
    def valid(text):
        assert text == 'good'
    _, identity = p.execute(tmp_path, 'input', 'config', 'spectrum', valid)
    assert len(calls) == 2
    p.execute(tmp_path, 'input', 'config', 'spectrum', valid)
    assert len(calls) == 2
    saved = json.loads((tmp_path/'identity.json').read_text())
    assert set(saved) == {'sources', 'input', 'config', 'spectrum_sha256', 'egs_inputs'}
    assert 'run_p8e.py' not in saved['sources']
    with pytest.raises(ValueError):
        p.execute(tmp_path, 'changed', 'config', 'spectrum', valid)
    (tmp_path/'egs5job.out').write_text('changed')
    with pytest.raises(ValueError, match='sealed'):
        p.execute(tmp_path, 'input', 'config', 'spectrum', valid)


def test_failed_validation_never_sealed(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'external_sources', lambda: {})
    def fake(command, **kwargs):
        (tmp_path/'egs5job.exe').write_text('test')
        (tmp_path/'egs5job.out').write_text('invalid')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(p.subprocess, 'run', fake)
    def fail(text):
        raise ValueError('bad health or replay')
    with pytest.raises(ValueError):
        p.execute(tmp_path, '', '', '', fail)
    assert not (tmp_path/'complete.json').exists()
    assert json.loads((tmp_path/'execution.json').read_text())['status'] == 'failed'
    with pytest.raises(RuntimeError, match='retained'):
        p.execute(tmp_path, '', '', '', fail)


def test_actual_ausgab_howfar_and_zero_direction(fortran):
    (fortran/'config.dat').write_text('100 1 1 3 4\n1 1 .25\n1 1 .25\n4 .5 .125\n')
    result = subprocess.run([str(fortran/'integration')], cwd=fortran,
        input='2 .125 1 .75 6\n2 .875 -1 .75 6\n3 1.25 0 100 7\n'
              '2 1 0 1 3\n3 1 0 1 3\n2 0 0 1 4\n4 2.5 0 1 5\n'
              '2 .875 1 1 2\n3 1.125 -1 1 2\n1 0 -1 1 0\n5 2.5 1 1 0\n',
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    a = np.array([list(map(float, line.split())) for line in result.stdout.splitlines()])
    expected = np.zeros((11, 12))
    expected[0, :4] = expected[1, :4] = [1, 2, 2, 1]
    expected[2, 5] = 7
    expected[3, 3] = 3
    expected[4, 4] = 3
    expected[5, 0] = 4
    expected[6, 11] = 5
    expected[7, 3] = 2
    expected[8, 4] = 2
    np.testing.assert_allclose(a[:, :12], expected, atol=1e-14)
    np.testing.assert_array_equal(a[7:9, 12:], [[.125, 3, 0], [.125, 2, 0]])
    np.testing.assert_array_equal(a[9:, -1], [1, 1])


@pytest.mark.parametrize('mode,slow', [('egs-pilot', False), ('egs', False), ('egs', True)])
def test_runner_saves_raw_batches_and_guards_production(tmp_path, monkeypatch, mode, slow):
    from prototypes.mv_electron import materials_p8a, run_p3a
    definitions = {m: {'density_g_cm3': RHO[i]} for i, m in enumerate(p.MATERIALS)}
    (tmp_path/'manifest.json').write_text(json.dumps({'materials': {
        m: {'definition': d} for m, d in definitions.items()}}))
    monkeypatch.setattr(materials_p8a, 'OUT', tmp_path)
    monkeypatch.setattr(p, 'load_reference', lambda ref, material: ref)
    monkeypatch.setattr(p, 'pegs_input', lambda ds: 'test input')
    monkeypatch.setattr(run_p3a, 'load_spectrum', lambda: (np.array([1., 2.]), np.array([1.]), np.array([1.])))
    actual_run = p.run
    dependencies = []
    def prerequisite(command, out, cache):
        dependencies.append(command)
        if command == 's7_replay':
            return {'s7_replay': {'status': 'pass'}}
        assert command == 'egs-pilot'
        return {'projection': {'exceeds_12_hours': slow}}
    monkeypatch.setattr(p, 'run', prerequisite)
    calls = []
    def execute(work, inp, config, spectrum, validate):
        calls.append(config)
        n, seed, fluor, nl, media_set = map(int, config.splitlines()[0].split())
        assert media_set == 4
        assert n == (p.PILOT_N if mode == 'egs-pilot' else 60_000_000)
        assert seed == (p.PILOT_SEED if mode == 'egs-pilot' else 820001)
        text = output().replace('CONFIG 100 9 1', f'CONFIG {n} {seed} 1')
        text = text.replace('LEDGER 100 23.5 76.5', f'LEDGER {n} {n*.235} {n*.765}')
        validate(text)
        work.mkdir(parents=True)
        (work/'complete.json').write_text('{}')
        return text, 'test-identity'
    monkeypatch.setattr(p, 'execute', execute)
    if slow:
        with pytest.raises(RuntimeError, match='12 hours'):
            actual_run(mode, tmp_path/'out', tmp_path/'cache')
        assert not calls
    else:
        report = actual_run(mode, tmp_path/'out', tmp_path/'cache')
        folder = 'pilot' if mode == 'egs-pilot' else 'production'
        with np.load(tmp_path/f'out/egs_{folder}.npz') as a:
            assert a['dose'].shape == (100, 220)
            assert a['batch_blocks'].shape == (100, 19)
            np.testing.assert_array_equal(a['dose'], np.full((100, 220), .01))
        assert report['identity_sha256'] == 'test-identity'
        assert report['layers'] == p.LAYERS
        if mode == 'egs-pilot':
            assert 'projection' in report
    assert dependencies == (['s7_replay'] if mode == 'egs-pilot' else ['s7_replay', 'egs-pilot'])



@pytest.mark.parametrize('header,medium,code', [
    ('100 1 1 1', 1, 50),
    ('100 1 1 1 bad', 1, 50),
    ('100 1 1 1 0', 1, 51),
    ('100 1 1 1 3', 1, 51),
    ('100 1 1 1 5', 1, 51),
    ('100 1 1 1 2', 0, 44),
    ('100 1 1 1 2', 3, 44),
    ('100 1 1 1 2', 4, 44),
])
def test_fortran_media_set_validation(fortran, header, medium, code):
    result = drive(fortran, header+f'\n{medium} 1 .1\n')
    assert result.returncode == code, result.stderr


def test_two_media_config_and_mass(fortran):
    config = p.config_text(100, 9, p.S7_LAYERS, media_set=2)
    assert config.splitlines()[0] == '100 9 1 2 2'
    assert config.splitlines()[2].startswith('2 ')
    result = drive(fortran, config, '3 10.025 10.475 9\n')
    assert result.returncode == 0, result.stderr
    dose = np.array(list(map(float, result.stdout.split())))
    assert dose.shape == (110,)
    assert dose[:100].sum() == 0
    assert dose[100:].sum() == pytest.approx(9)
    grid = p.egs_layout(p.S7_LAYERS, RHO, media_set=2)
    np.testing.assert_allclose(grid['mass_g_cm2'], [.1]*100+[.5]*10)
    np.testing.assert_allclose(grid['blocks_cm'][-1], [10, 10.5])
    assert p.config_text(100, 9, p.LAYERS).splitlines()[0] == '100 9 1 6 4'
    for media_set in (0, 1, 3, 5):
        with pytest.raises(ValueError, match='media set'):
            p.config_text(100, 9, p.S7_LAYERS, media_set=media_set)
    for medium in (0, 3, 4):
        with pytest.raises(ValueError, match='medium'):
            p.config_text(100, 9, [(medium, 1., .1)], media_set=2)


def two_media_output(n=100, seed=9):
    lines = [f'CONFIG {n} {seed} 1', 'SETTING 0 1',
             'LAYER 1 1 10 .1 100', 'LAYER 2 2 .5 .05 10']
    for i, m in enumerate([0, 1, 2, 0], 1):
        rho = {0: 0, 1: 1., 2: 10.}[m]
        flag = int(m > 0)
        lines.append(f'REGION {i} {m} {rho} .521 .01 {flag} {flag} 0 0 0 0 1 0 0')
    lines += ['MEDIUM 1 P8E_WATER 1 .521 .01 1 .5 0 0 0 0',
              'MEDIUM 2 P8E_LEAD 10 .521 .01 1 .5 0 0 0 0',
              f'LEDGER {n} {n*.15} {n*.85} 1e-8 0 1e-14', 'CPU 1']
    lines += [f'DOSE {b} {j} .01' for b in range(1, 101) for j in range(1, 111)]
    return '\n'.join(lines+['DONE'])+'\n'


def test_two_media_parser():
    text = two_media_output()
    d, health = p.parse_egs(text, 100, 9, p.S7_LAYERS, RHO, media_set=2)
    np.testing.assert_array_equal(d, np.full((100, 110), .01))
    assert len(health['media']) == 2
    for old, new in [('P8E_LEAD', 'P8E_LUNG'), ('REGION 3 2 10.', 'REGION 3 2 .25'),
                     ('LAYER 2 2', 'LAYER 2 4'), ('DONE', 'MEDIUM 3 P8E_BONE\nDONE')]:
        assert old in text
        with pytest.raises(ValueError):
            p.parse_egs(text.replace(old, new), 100, 9, p.S7_LAYERS, RHO, media_set=2)
    with pytest.raises(ValueError):
        p.parse_egs(text, 100, 9, p.S7_LAYERS, RHO)


def test_s7_runner_selects_two_media_and_preserves_failure(tmp_path, monkeypatch):
    import hashlib
    from prototypes.mv_electron import materials_p8a as mat, run_p3a
    definitions = [{'density_g_cm3': r} for r in RHO]
    (tmp_path/'manifest.json').write_text(json.dumps({'materials': {
        m: {'definition': d} for m, d in zip(p.MATERIALS, definitions)}}))
    monkeypatch.setattr(mat, 'OUT', tmp_path)
    monkeypatch.setattr(mat, 'CACHE', tmp_path/'reference')
    monkeypatch.setattr(p, 'load_reference', lambda ref, material: ref)
    monkeypatch.setattr(p, 'external_sources', lambda: {})
    monkeypatch.setattr(run_p3a, 'load_spectrum', lambda: (np.array([1., 2.]), None, np.array([1.])))
    def pegs(ds):
        return json.dumps(ds, sort_keys=True)
    monkeypatch.setattr(p, 'pegs_input', pegs)
    ref = mat.CACHE/'fluorescence/pilot_on'
    ref.mkdir(parents=True)
    spectrum = p.spectrum_text(np.array([1., 2.]), np.array([1.]))
    n, seed = p.PILOT_N, p.SEEDS['pilot_on']
    old_input = pegs([definitions[0], definitions[3]])
    sources = {'p8a_egs.f': p.CODE.with_name('p8a_egs.f').read_text()}
    identity = dict(config=f'{n} {seed} 1\n', input=old_input, sources=sources,
                    spectrum_sha256=hashlib.sha256(spectrum.encode()).hexdigest(), egs_inputs={})
    for name, content in {'identity.json': json.dumps(identity), 'config.dat': identity['config'],
                          'spectrum.dat': spectrum, 'p8a_egs.inp': old_input,
                          'egs5job.out': 'old output', **sources}.items():
        (ref/name).write_text(content)
    monkeypatch.setattr(p, 'verify_seal', lambda work: None)
    monkeypatch.setattr(p, 'parse_s7', lambda *args: (np.full((100, 55), .01), {}))
    cache = tmp_path/'cache'
    failed = cache/'s7_replay'
    failed.mkdir(parents=True)
    (failed/'egs5job.out').write_text('preserved four-media failure')
    def execute(work, inp, config, spectrum, validate):
        assert work == cache/'s7_replay_2media'
        assert config == p.config_text(n, seed, p.S7_LAYERS, media_set=2)
        assert [d['medium_name'] for d in json.loads(inp)] == ['P8E_WATER', 'P8E_LEAD']
        assert [d['density_g_cm3'] for d in json.loads(inp)] == [RHO[0], RHO[3]]
        text = two_media_output(n, seed)
        validate(text)
        work.mkdir()
        (work/'complete.json').write_text('{}')
        return text, 'identity'
    monkeypatch.setattr(p, 'execute', execute)
    report = p.run('s7_replay', tmp_path/'out', cache)
    assert report['media_set'] == 2
    assert report['s7_replay']['tolerance'] == 1e-12
    assert (failed/'egs5job.out').read_text() == 'preserved four-media failure'
    assert sorted(x.name for x in failed.iterdir()) == ['egs5job.out']
    with np.load(tmp_path/'out/egs_s7_replay_2media.npz') as result:
        assert result['dose'].shape == (100, 110)
        np.testing.assert_allclose(result['mass_g_cm2'], [.1]*100+[.5]*10)

# Part 2: read-only reference data; generated tables stay in memory or tmp_path.
from prototypes.mv_electron import photon_transport_p8e as k8
from prototypes.mv_electron import generate_fluorescence_p8e as f8
from numba import njit


@pytest.fixture(scope='module')
def legacy8():
    return k8.legacy_bundle()


@pytest.fixture(scope='module')
def material_data8():
    from prototypes.mv_electron import materials_p8a as mat
    manifest=json.loads((mat.OUT/'manifest.json').read_text())
    result=[]
    for name in mat.MATERIALS:
        row=manifest['materials'][name]
        data={field:mat.load_reference(row[field],name)
              for field in ('definition','constants','stopping','photon','gs','pair','shells','range')}
        data['fluor']=f8.fluorescence_table(data['shells'])
        result.append(data)
    return result


@pytest.fixture(scope='module')
def materials8(material_data8):
    result=[]
    for d in material_data8:
        c=d['constants'];z=np.asarray(c['Z'],float);nj=np.asarray(c['NJ'],float)
        brems=k8.brems_tables(z,nj,d['stopping'])
        result.append(k8.make_material(c,d['stopping'],d['photon'],d['gs'],d['pair'],
            d['shells'],d['fluor'],brems,density=c['density_g_cm3']))
    return result


def test_part2_registration_and_code_fingerprint():
    from prototypes.mv_electron.batch_kernels_p3u import KERNELS
    assert KERNELS['p8e_stack'].run is k8.batch_run
    sources=k8.part2_sources()
    assert 'photon_transport_p8e.py' in sources
    assert not any('plans' in key or 'RESULTS' in key for key in sources)


@pytest.mark.parametrize('i',range(4))
def test_all_material_rates_units_and_brems(i,materials8,material_data8):
    from prototypes.mv_electron.physics import hard,NE
    from prototypes.mv_electron.flight_annihilation_p2c import annihilation_rate
    from prototypes.mv_electron.backend_sampling import table_rates
    m=materials8[i];data=material_data8[i];rho=data['constants']['density_g_cm3']
    assert k8.bundle_checks(m)['status']=='pass'
    for T in (.03,.1,1.,10.):
        np.testing.assert_allclose(k8.material_hard(T,m),np.asarray(hard(T))*m.ne/NE,rtol=1e-12)
        assert k8.material_annihilation_rate(T,m.ne)==pytest.approx(annihilation_rate(T)*m.ne/NE,rel=1e-12)
        np.testing.assert_allclose(table_rates(T,m.gs_T,m.rates),
            np.asarray(table_rates(T,data['gs']['T'],data['gs']['rates']))/rho,rtol=1e-12)
        actual=k8.material_collision(T,m)
        expected=np.exp(np.interp(np.log(T),np.log(data['stopping']['T']),np.log(data['stopping']['collision'])))
        assert actual==pytest.approx(expected,rel=1e-13)
    for j in (1,len(m.brems_E)//2,len(m.brems_E)-1):
        T=m.brems_E[j];norm=m.brems_V[j,4]
        rad=np.interp(np.log(T),np.log(data['stopping']['T']),data['stopping']['radiative'])
        assert norm*k8.material_integral(T,0,T,m.z,m.nj)==pytest.approx(rad,rel=1e-12)
        rate,hard_stop,soft,envelope=k8.material_quantities(T,m)
        assert hard_stop+soft==pytest.approx(rad,rel=1e-12)
        assert all(k8.material_shape(T,x,m.z,m.nj)<=envelope for x in np.geomspace(.01,T,4097))


def test_legacy_scalar_functions_bitwise(legacy8):
    from prototypes.mv_electron.physics import collision,hard
    from prototypes.mv_electron.photon_data_p2 import coefficients
    from prototypes.mv_electron.bremsstrahlung_p3r import quantities
    m=legacy8
    water=np.zeros((len(m.pair_E),7));water[:,2]=m.photo;water[:,3]=m.pair_n;water[:,4]=m.pair_e
    for T in (.010001,.019,.02,.0200001,.1,1.,9.9):
        assert k8.material_collision(T,m)==collision(T)
        np.testing.assert_array_equal(k8.material_hard(T,m),hard(T))
        np.testing.assert_array_equal(k8.material_coefficients(T,m),coefficients(T,m.pair_E,water,m.thresholds))
        np.testing.assert_array_equal(k8.material_quantities(T,m),quantities(T,m.brems_E,m.brems_V))


def test_actual_p4_water_10000_history_regression(legacy8):
    from prototypes.mv_electron import photon_transport_p4 as p4
    from prototypes.mv_electron.run_coupled_p2 import inputs
    from prototypes.mv_electron.run_p3s import rule_kwargs
    # Read-only protected inputs; the pre-existing pair artifact is present.
    args=inputs('elsepa')
    old=p4.run_coupled(1.,10000,2026102082,.05,.025,30.,*args,
        annihilate_in_flight=True,score_kerma=False,score_pair=False,**rule_kwargs('own'))
    new=k8.run_coupled([legacy8],[(1,30.,.1)],10000,2026102082,
        binding_enabled=False,fluorescence_enabled=False)
    assert np.array_equal(new['coarse'],old[0])
    assert new['coarse'].tobytes()==old[0].tobytes()
    np.testing.assert_array_equal(new['counts'],old[8])
    assert k8.healthy(new)


def test_p8a_water_subdivision_is_transparent(materials8):
    one=k8.run_coupled(materials8,[(1,21.5,.1)],10000,2026102083)
    six=k8.run_coupled(materials8,[(1,t,.1) for t in (5.,6.,4.,2.,4.,.5)],10000,2026102083)
    assert one['dose'].tobytes()==six['dose'].tobytes()
    assert one['coarse'].tobytes()==six['coarse'].tobytes()
    np.testing.assert_array_equal(one['counts'],six['counts'])
    assert k8.healthy(one) and k8.healthy(six)


@njit
def _boundary8(a,m,mu):
    np.random.seed(74)
    return k8.material_boundary(a,0,1,m,10.,1.,.025,mu)


def test_boundary_clocks_scattering_reflection_terminal(legacy8):
    # State is at the end of an old half-step. Only direction/region and the
    # scattering clocks may change; loss and Poisson clocks include exact ties.
    for strength,mu,expected_region in ((.01,1.,1),(.01,-1.,0),(.0001,-1.,1),(0.,-1.,1)):
        a=np.zeros(23);a[5]=1;a[7]=1;a[14]=.05;a[13]=.02;a[12]=.03-strength
        a[8]=.123;a[15]=0.;a[21]=.27;a[22]=.37
        before=a.copy();info=_boundary8(a,legacy8,mu)
        np.testing.assert_array_equal(a[[8,15,21,22]],before[[8,15,21,22]])
        assert info[0]==pytest.approx(strength,abs=1e-16)
        assert a[16]==expected_region and info[4]==-1
        assert a[14]==.025 and a[12]+a[13]==pytest.approx(.025)
        if strength<legacy8.gs_K[0]:assert a[5]==1
        if expected_region==0:
            # A second boundary operation at the same point has no old strength
            # to scatter; it must cross without reversal rather than cycling.
            second=k8.material_boundary(a,0,1,legacy8,10.,1.,.025,-1.)
            assert second[0]<legacy8.gs_K[0] and second[1]==1.
    a=np.zeros(23);a[5]=1;a[7]=1;a[18]=1;a[12:15]=[.01,.02,.05]
    before=a.copy();info=_boundary8(a,legacy8,-1.)
    np.testing.assert_array_equal(a[12:15],before[12:15]);assert a[5]==1 and a[16]==1 and info[3]==1


def test_actual_stack_boundary_records_and_health(materials8,material_data8):
    densities=[d['constants']['density_g_cm3'] for d in material_data8]
    result=k8.run_coupled(materials8,p.LAYERS,200,82007,densities=densities,E=10.,record_boundaries=100)
    assert k8.healthy(result)
    assert result['boundary_counts'][0]>0 and len(result['boundary_records'])>0
    assert result['dose'].shape==(220,)
    assert sum(result['boundary_hist'])==result['boundary_counts'][0]
    assert sum(result['boundary_omitted_hist'])==result['boundary_counts'][1]
    assert result['p8_max'][0]<=1e-12 and result['maxima'][0]<=1e-12


@njit
def _joint_samples8(m,E,n):
    np.random.seed(123082)
    counts=np.zeros(m.bindings.shape,np.int64)
    for j in range(n):
        a,b=k8.select_shell(E,m)
        if a>=0:counts[a,b]+=1
    return counts


@pytest.mark.parametrize('i',(0,2,3))
def test_joint_element_shell_million(i,materials8,material_data8):
    from prototypes.mv_electron.pegs_p8a import shell_probabilities
    m=materials8[i];expected,local=shell_probabilities(.1,material_data8[i]['shells'])
    np.testing.assert_allclose(k8.shell_distribution(.1,m),expected,rtol=1e-12)
    actual=_joint_samples8(m,.1,1_000_000)
    np.testing.assert_array_less(abs(actual-expected*1_000_000),5*np.sqrt(1_000_000*expected*(1-expected))+1)


@njit
def _relax_samples8(m,shell,n):
    np.random.seed(123083+shell)
    first=np.zeros(m.lines.shape[2]+1,np.int64)
    directions=np.zeros(4);max_error=0.;minimum=1.
    children=np.zeros(5,np.int64)
    for i in range(n):
        photons,local,trace=k8.relax_vacancy(0,shell,m)
        selected=-1
        if len(trace):
            selected=trace[0,1]
            if selected>=0:children[trace[0,2]]+=1
        first[selected+1]+=1
        energy=local
        minimum=min(minimum,local)
        for j in range(len(photons)):
            energy+=photons[j,0]
            directions[0]+=1;directions[1]+=photons[j,3];directions[2]+=photons[j,3]**2
            directions[3]=max(directions[3],abs(np.sum(photons[j,1:]**2)-1.))
        max_error=max(max_error,abs(energy-m.bindings[0,shell]))
    return first,children,directions,max_error,minimum


@pytest.mark.parametrize('shell',range(4))
def test_fluorescence_million_each_vacancy(shell,materials8):
    m=materials8[3];first,children,directions,error,minimum=_relax_samples8(m,shell,1_000_000)
    n=1_000_000;omega=m.yields[0,shell]
    expected=np.r_[1-omega,omega*m.lines[0,shell,:,1]]
    np.testing.assert_array_less(abs(first-n*expected),5*np.sqrt(n*expected*(1-expected))+1)
    child_probs=np.zeros(5)
    for energy,rate,child,code in m.lines[0,shell,:m.line_counts[0,shell]]:child_probs[int(child)]+=omega*rate
    np.testing.assert_array_less(abs(children-n*child_probs),5*np.sqrt(n*child_probs*(1-child_probs))+1)
    count=directions[0]
    if count:
        assert abs(directions[1]/count)<=5/np.sqrt(3*count)
        assert abs(directions[2]/count-1/3)<=5*np.sqrt((1/5-1/9)/count)
    assert error<1e-15 and minimum>=0 and directions[3]<1e-14


def test_absorption_edges_cutoff_and_budget_rejection(materials8,material_data8):
    m=materials8[3];edge=m.bindings[0,0]
    assert k8.shell_distribution(np.nextafter(edge,0),m)[0,0]==0
    assert k8.shell_distribution(np.nextafter(edge,np.inf),m)[0,0]>0
    # All below-cutoff lines deposit locally. M vacancies also terminate locally.
    low=materials8[0]
    for s in range(5):
        out,local,trace=k8.relax_vacancy(0,s,low)
        assert not len(out) and local==low.bindings[0,s]
    bad=dict(material_data8[3]['shells'])
    bad['binding_MeV']=bad['binding_MeV']*.01
    with pytest.raises(ValueError,match='negative PEGS'):
        f8.fluorescence_table(bad)
    # The lead line table contains individual lines on BOTH sides of 10 keV.
    energies=m.lines[...,0];assert np.any((energies>0)&(energies<.01)) and np.any((energies>.01)&(energies<.02))


def test_judgement_correlations_limits_stopping_and_allocation():
    grid=p.layout(p.LAYERS,[1.]*4)
    reference=np.ones((100,220))
    prototype=reference.copy()
    prototype[:,30:40]+=0.01
    result=p.judge_batches(prototype,reference,grid)
    # Binary representation puts 1.01-1 just above .01; test the exact numerical
    # threshold directly using a Dref that makes the subtraction exact.
    reference[:]=100.;prototype[:]=100.;prototype[:,30:40]=101.
    assert p.judge_batches(prototype,reference,grid)['status'][0]=='pass'
    prototype[:,40:50]=102.
    result=p.judge_batches(prototype,reference,grid)
    assert result['status'][:3]==['pass','fail','not-judged']
    reference[:]=1.;prototype[:]=1.
    prototype[:,30:40]=1+np.linspace(-1,1,100)[:,None]
    result=p.judge_batches(prototype,reference,grid)
    assert result['status'][0]=='insufficient-statistics' and result['status'][1]=='pass'
    actual=np.std(np.linspace(-1,1,100),ddof=1)/10
    assert result['SEM'][0]==pytest.approx(actual)
    pilot=np.ones((100,220))+np.linspace(-.05,.05,100)[:,None]
    allocation=p.required_prototype_size(pilot,reference,grid)
    needed=1_000_000*(np.std(np.linspace(-.05,.05,100),ddof=1)/10)**2/.0033**2
    assert allocation['histories']==max(200,int(np.ceil(1.5*needed/200))*200)
    assert allocation['histories']%200==0
    reference[:,30:40]=1+np.linspace(-1,1,100)[:,None]
    with pytest.raises(ValueError,match='EGS SEM'):
        p.required_prototype_size(pilot,reference,grid)


def test_result_hash_and_bin_tampering(tmp_path):
    grid=p.layout(p.LAYERS,[1.]*4);artifact=tmp_path/'data.npz'
    np.savez(artifact,dose=np.ones((100,220)),**grid)
    meta=dict(N=100,seed=1,layers=[list(x) for x in p.LAYERS],densities_g_cm3=[1.]*4,
              units='MeV cm2/g/history',artifact_sha256=p.sha(artifact))
    assert p._read_result(artifact,meta,100,1,grid,[1.]*4).shape==(100,220)
    artifact.write_bytes(artifact.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='hash'):
        p._read_result(artifact,meta,100,1,grid,[1.]*4)


@pytest.mark.parametrize('fault',('unresolved','steps','stack','score','envelope','balance'))
def test_new_healthy_rejects_faults(fault):
    batch=dict(dose=np.ones(220),totals=np.zeros(6),maxima=np.zeros(3),violations=np.zeros(3),
        counts=np.zeros(16),brems=np.zeros(10),p8_max=np.zeros(2))
    assert k8.healthy(batch)
    if fault=='unresolved':batch['totals'][3]=1
    if fault=='steps':batch['counts'][5]=1
    if fault=='stack':batch['counts'][6]=1
    if fault=='score':batch['p8_max'][0]=1.01e-12
    if fault=='envelope':batch['brems'][2]=1
    if fault=='balance':batch['maxima'][0]=1.01e-12
    assert not k8.healthy(batch)


def test_lead_fluorescence_escape_reabsorption_and_birth_positions(materials8,material_data8):
    rho=[d['constants']['density_g_cm3'] for d in material_data8]
    result=k8.run_coupled(materials8,[(4,.5,.05)],1000,820088,densities=rho,E=.1,record_fluorescence=3000)
    assert k8.healthy(result)
    emitted,escaped,absorbed,subcutoff=result['fluorescence_counts'][:4]
    assert emitted>0 and escaped>0 and absorbed>0
    assert emitted==escaped+absorbed+subcutoff
    kemit,kesc,kabs,kcut=result['fluorescence_counts'][4:]
    assert kemit>0 and kesc>0 and kabs>0 and kemit==kesc+kabs+kcut
    photons=result['fluorescence_records']
    assert len(photons)>0 and np.all(photons[:,0]>=.01)
    assert np.all(photons[:,3]>=0) and np.all(photons[:,3]<=.5)
    np.testing.assert_array_equal(photons[:,7],0)
    # At 100 keV only the characteristic line list may produce these photons.
    lines=materials8[3].lines[...,0].ravel()
    assert np.all(np.isin(photons[:,0],lines))


def test_actual_one_particle_boundary_energy_hard_priority(materials8,material_data8):
    mats=list(materials8)
    # A deterministic forward GS distribution keeps the boundary direction fixed.
    mats[0]=mats[0]._replace(gs_y=np.zeros_like(mats[0].gs_y))
    a=np.zeros(23);a[2]=5.;a[5]=1.;a[6]=a[7]=1.;a[9]=.025;a[10]=.025;a[11]=.05
    a[12]=0.;a[13]=.025;a[14]=.05;a[15]=0.;a[19]=-1
    rho=[d['constants']['density_g_cm3'] for d in material_data8]
    result=k8.run_coupled(mats,p.LAYERS,1,820089,densities=rho,source_state=a,record_events=16,record_boundaries=16)
    events=result['event_records']
    np.testing.assert_array_equal(events[:3,0],[0,1,3])
    # A boundary win retains the simultaneous loss/collision clocks at zero.
    first=result['boundary_records'][0]
    assert first[6]==0 and first[10]==0
    assert events[1,1]==1 and events[1,3]==0 and events[2,3]==0
    assert k8.healthy(result) and not np.any(result['counts'][5:7])


def test_judgement_exact_thresholds_and_nextafter():
    assert p.classify_blocks([.01,-.01],[.0033,.0033])==['pass','pass']
    assert p.classify_blocks([np.nextafter(.01,np.inf),0],[.0033,0])==['fail','not-judged']
    assert p.classify_blocks([.5,0],[np.nextafter(.0033,np.inf),0])==['insufficient-statistics','pass']


def test_lead_csda_integral_read_only(materials8,material_data8,tmp_path,monkeypatch):
    monkeypatch.setattr(k8,'load_bundles',lambda folder:(materials8,{}))
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'bundles.json').write_text('{}')
    result=p.run_csda(bundle,tmp_path)
    assert result['status']=='pass'
    assert all(abs(v['relative'])<=.005 for v in result['values'])


def test_proto_runner_seals_reuses_and_checks_judgement_inputs(tmp_path,monkeypatch):
    from prototypes.mv_electron import batch_runner_p3u as runner
    egs_cache=tmp_path/'egs';out=tmp_path/'out';out.mkdir()
    bundle=tmp_path/'materials';bundle.mkdir()
    monkeypatch.setattr(k8,'load_bundles',lambda folder:([],dict(densities=[1.]*4)))
    monkeypatch.setattr(k8,'fingerprint',lambda params:dict(source='test fingerprint'))
    monkeypatch.setattr(p,'load_egs_result',lambda out,cache,grid,densities:
        (np.ones((100,220)),dict(artifact_sha256='fixed-egs')))
    calls=[]
    def run_batches(work,kernel,params,N,seed,workers,nbatch=100):
        calls.append((kernel,N,seed,workers,nbatch))
        work.mkdir(parents=True,exist_ok=True)
        config=dict(kernel=kernel,params=params,N=N,seed=seed,nbatch=nbatch)
        (work/'config.json').write_text(json.dumps(config))
        for j in range(100):(work/f'batch_{j:03}.npz').write_bytes(b'test')
        target=work/'result.npz'
        np.savez(target,dose=np.ones((100,220)),totals=np.zeros((100,6)),
            maxima=np.zeros((100,3)),p8_max=np.zeros((100,2)),boundary_counts=np.zeros((100,5),int),
            boundary_hist=np.zeros((100,len(k8.BOUNDARY_STRENGTH_EDGES)+1),int),boundary_omitted_hist=np.zeros((100,len(k8.BOUNDARY_STRENGTH_EDGES)+1),int),fluorescence_counts=np.zeros((100,8),int),wall_seconds=1.)
        return target
    monkeypatch.setattr(runner,'run_batches',run_batches)
    pilot=p.run_part2('proto-pilot',out,egs_cache)
    assert pilot['N']==1_000_000 and pilot['allocation']['histories']==200
    production=p.run_part2('proto',out,egs_cache)
    assert production['N']==200 and production['seed']==p.PROTO_SEED
    assert all(c[3:]==(2,100) for c in calls)
    with pytest.raises(ValueError,match='review'):
        p.run_part2('judge',out,egs_cache)
    judgement=p.run_part2('judge',out,egs_cache,reviewed=True)
    assert judgement['status']==['pass']*19
    # Raw results remain immutable even if the analysis metadata is untouched.
    (tmp_path/'proto/production/batch_000.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='sealed'):
        p.run_part2('judge',out,egs_cache,reviewed=True)


def test_allocation_cap_and_result_condition_checks(tmp_path):
    grid=p.layout(p.LAYERS,[1.]*4)
    reference=np.ones((100,220))
    pilot=1+np.linspace(-.9,.9,100)[:,None]*np.ones((1,220))
    assert p.required_prototype_size(pilot,reference,grid)['status']=='stop-consult-user'
    target=tmp_path/'result.npz'
    bad_grid=dict(grid);bad_grid['edges_cm']=grid['edges_cm']+.001
    np.savez(target,dose=reference,**bad_grid)
    meta=dict(N=100,seed=2,layers=[list(x) for x in p.LAYERS],densities_g_cm3=[1.]*4,
        units='MeV cm2/g/history',artifact_sha256=p.sha(target))
    with pytest.raises(ValueError,match='bin mapping'):
        p._read_result(target,meta,100,2,grid,[1.]*4)
    with pytest.raises(ValueError,match='N/seed'):
        p._read_result(target,meta,100,1,grid,[1.]*4)


def test_boundary_invalidates_rates_and_uses_new_material(materials8,material_data8):
    mats=list(materials8);mats[0]=mats[0]._replace(gs_y=np.zeros_like(mats[0].gs_y))
    a=np.zeros(23);a[2]=5.;a[5]=1.;a[6]=a[7]=1.;a[8]=.02;a[9]=.025;a[10]=.025;a[11]=.05
    a[12]=.01;a[13]=.025;a[14]=.05;a[15]=.4;a[19]=-1
    rho=[d['constants']['density_g_cm3'] for d in material_data8]
    result=k8.run_coupled(mats,p.LAYERS,1,820090,densities=rho,source_state=a,record_events=8,record_boundaries=8)
    m=mats[1]
    stopping=(k8.material_collision(1.,m)-k8.material_hard(1.,m)[1]+k8.material_quantities(1.,m)[2])*rho[1]
    rate=k8.material_hard(1.,m)[0]*rho[1]
    next_event=result['event_records'][1]
    assert next_event[1]==1
    assert next_event[5]==pytest.approx(.02/stopping,rel=1e-12)
    assert next_event[7]==pytest.approx(.4/rate,rel=1e-12)
    assert k8.healthy(result)


def test_bundle_roundtrip_fingerprint_and_modified_table(tmp_path,monkeypatch,legacy8):
    from prototypes.mv_electron import materials_p8a as mat
    fake_out=tmp_path/'p8a';fake_out.mkdir();(fake_out/'manifest.json').write_text('{}')
    manifest={'materials':{name:{'constants':name} for name in mat.MATERIALS}}
    monkeypatch.setattr(mat,'OUT',fake_out)
    monkeypatch.setattr(mat,'load_manifest',lambda:manifest)
    monkeypatch.setattr(mat,'load_reference',lambda ref,name:dict(density_g_cm3=1.))
    m=legacy8._replace(gs_y=legacy8.gs_y[:2,:2,:3].copy())
    for name in mat.MATERIALS:np.savez(tmp_path/(name+'_bundle.npz'),**m._asdict())
    identity=dict(manifest_sha256=p.sha(fake_out/'manifest.json'),
        source_hashes=k8.part2_sources(),files={file.name:p.sha(file) for file in tmp_path.glob('*.npz')},
        checks={name:{'status':'pass'} for name in mat.MATERIALS},units='mass',densities=[1.]*4)
    (tmp_path/'bundles.json').write_text(json.dumps(identity))
    loaded,receipt=k8.load_bundles(tmp_path)
    assert len(loaded)==4 and receipt==identity
    for field in k8.Material._fields:np.testing.assert_array_equal(getattr(loaded[0],field),getattr(m,field))
    fp=k8.fingerprint(dict(bundle=str(tmp_path)))
    assert fp['source_hashes']['photon_transport_p8e.py']==p.sha(Path(k8.__file__))
    (tmp_path/'water_bundle.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bundle hash'):
        k8.fingerprint(dict(bundle=str(tmp_path)))


def test_photon_coefficients_preserve_each_absorption_edge_side(materials8,material_data8):
    from prototypes.mv_electron.materials_p8a import photo_interp
    count=0
    for m,data in zip(materials8,material_data8):
        table=data['photon'];nodes=table['photo_E']
        edges=nodes[:-1][np.diff(nodes)==0]
        for edge in edges:
            for E in (np.nextafter(edge,0),edge,np.nextafter(edge,np.inf)):
                expected=photo_interp(E,table['photo_E'],table['photo'])
                assert k8.material_coefficients(E,m)[1]==pytest.approx(float(expected),rel=1e-12)
                count+=1
    assert count>0
