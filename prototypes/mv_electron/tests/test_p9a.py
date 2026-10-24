"""Offline P9a EGS-side tests only; all writes are confined to pytest temp dirs.

Compile callbacks/standalone scorers, never link or execute PEGS/EGS5.
"""
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from prototypes.mv_electron import p9a as p

RHO = [1., .26, 1.85, 11.35, 2.47]


@pytest.fixture(autouse=True)
def forbid_external(monkeypatch):
    """Fail closed if a test accidentally enters the real EGS execution path."""
    actual = subprocess.run
    def guarded(command, **kwargs):
        assert 'egs5' not in Path(str(command[0])).name.lower()
        assert 'pegs' not in Path(str(command[0])).name.lower()
        return actual(command, **kwargs)
    monkeypatch.setattr(subprocess, 'run', guarded)


def synthetic(layers=p.LAYERS, media_set=5, n=100, seed=9, fluor=1):
    """Uniform dose: independent conservation from total layer areal mass."""
    lines = [f'CONFIG {n} {seed} {fluor}', 'SETTING 0 1']
    for i, (m, t, w) in enumerate(layers, 1):
        lines.append(f'LAYER {i} {m} {t} {w} {round(t/w)}')
    for i, m in enumerate([0]+[m for m, _, _ in layers]+[0], 1):
        f = fluor if m else 0
        lines.append(f'REGION {i} {m} {RHO[m-1] if m else 0} .521 .01 {f} {f} 0 0 0 0 1 0 0')
    for i in range(media_set):
        lines.append(f'MEDIUM {i+1} {p.MEDIA_NAMES[i]} {RHO[i]} .521 .01 1 .5 0 0 0 0')
    dep = n*.01*sum(RHO[m-1]*t for m, t, _ in layers)
    lines += [f'LEDGER {n} {dep} {n-dep} 1e-8 0 1e-14', 'CPU 1']
    for tag, nb in [('DOSE', sum(round(t/w) for _, t, w in layers)),
                    ('FINE', round(sum(t for _, t, _ in layers)/.05))]:
        lines.extend(f'{tag} {b} {j} .01' for b in range(1, 101) for j in range(1, nb+1))
    return '\n'.join(lines+['DONE'])+'\n'


def deck(media_set=5):
    lines = []
    for i, name in enumerate(p.MEDIA_NAMES[:media_set]):
        if i == 4:
            d = p.seed_definition()
            lines += [f'MEDIUM={name:<24},STERNCID=P8A_SEED_AGTI',
                      'MIXT,RHO= 2.4700E+00,NE= 2, IUNRST=0',
                      f'ASYM=TI,Z=22.,A={d["atomic_weights"]["22"]:.3f},RHOZ=4.04500E-01',
                      f'ASYM=AG,Z=47.,A={d["atomic_weights"]["47"]:.3f},RHOZ=5.95500E-01']
        else:
            lines += [f'MEDIUM={name:<24},STERNCID=TEST', f'old material {i}']
    return '\n'.join(lines)+'\n'


@pytest.mark.parametrize('layers,nb,nf', [(p.LAYERS, 216, 424), (p.p8e.LAYERS, 220, 430), (p.SLAB_LAYERS, 20, 40)])
def test_geometry(layers, nb, nf):
    g = p.layout(layers, RHO)
    f = p.fine_layout(layers, RHO)
    assert len(g['mass_g_cm2']) == nb and len(f['mass_g_cm2']) == nf
    np.testing.assert_allclose(f['mass_g_cm2'].sum(), g['mass_g_cm2'].sum())
    if layers == p.LAYERS:
        expected = [(3, 4), (4, 5), (5, 5.1)]
        expected += [(5.1+i, 6.1+i) for i in range(5)]
        expected += [(10.1, 11.1)]+[(11.1+i, 12.1+i) for i in range(10)]
        np.testing.assert_allclose(g['blocks_cm'], expected)
        assert g['block_index'][-1] == -1
        assert g['mass_g_cm2'][50] == pytest.approx(.0494)
        np.testing.assert_allclose(g['block_weights'].sum(axis=1), 1.)
    assert p.config_text(100, 9, layers).splitlines()[0].endswith('5')


@pytest.mark.parametrize('layers', [((1, .11, .01),), ((1, .05, .01), (5, .025, .005)),
                                    ((6, 1, .1),), ((1, 1, .3),), ((1, float('nan'), .1),),
                                    ((1, 50.05, .1),), ((1, 0, .1),)])
def test_bad_geometry(layers):
    with pytest.raises(ValueError):
        p.config_text(100, 1, layers)


def test_seed_pegs_input_no_I_identity():
    import xraylib
    d = p.seed_definition()
    assert d['weights'] == {'47': .5955, '22': .4045}
    assert d['density_g_cm3'] == 2.47
    for z in (22, 47):
        assert d['atomic_weights'][str(z)] == xraylib.AtomicWeight(z)
    inp = p.pegs_input([d])
    assert 'P9A_SEED' in inp and 'AE=0.521,AP=0.01' in inp
    assert p.digest(inp) == p.digest(p.pegs_input([dict(d, I_MeV=.02)]))
    assert p.digest(inp) != p.digest(p.pegs_input([dict(d, density_g_cm3=2.48)]))
    assert p.digest(inp) != p.digest(p.pegs_input([dict(d, weights={'47': .6, '22': .4})]))


@pytest.mark.parametrize('layers,media_set', [(p.LAYERS, 5), (p.p8e.LAYERS, 4), (p.p8e.LAYERS, 5), (p.SLAB_LAYERS, 5)])
def test_parser(layers, media_set):
    dose, fine, health = p.parse_egs(synthetic(layers, media_set), 100, 9, layers, RHO, media_set=media_set)
    assert dose.shape == (100, sum(round(t/w) for _, t, w in layers))
    assert fine.shape == (100, round(sum(t for _, t, _ in layers)/.05))
    np.testing.assert_array_equal(dose, .01)
    assert health['energy_balance']['nleft'] == 0


@pytest.mark.parametrize('old,new', [
    ('SETTING 0 1', 'SETTING 1 1'), ('CONFIG 100 9 1', 'CONFIG 100 10 1'),
    ('P9A_SEED', 'P8A_SEED_AGTI'), ('MEDIUM 5 P9A_SEED 2.47', 'MEDIUM 5 P9A_SEED 2.48'),
    ('REGION 3 5', 'REGION 3 4'), ('.521 .01', '.522 .01'),
    ('1e-8 0 1e-14', '2e-6 0 1e-14'), ('1e-8 0 1e-14', '1e-8 1 1e-14'),
    ('1e-8 0 1e-14', '1e-8 0 2e-12'), ('DOSE 1 1 .01', 'DOSE 1 1 NaN'),
    ('DOSE 1 1 .01', 'DOSE 1 2 .01'), ('DOSE 1 1 .01\n', ''),
    ('FINE 1 1 .01', 'FINE 1 1 NaN'), ('FINE 1 1 .01', 'FINE 1 1 -.01'),
    ('FINE 1 1 .01', 'FINE 1 2 .01'), ('FINE 1 1 .01\n', ''),
    ('FINE 1 1 .01', 'FINE 101 1 .01'), ('DONE', ''), ('CPU 1', 'CPU NaN'),
])
def test_parser_faults(old, new):
    with pytest.raises(ValueError):
        p.parse_egs(synthetic().replace(old, new, 1), 100, 9, p.LAYERS, RHO)


def test_fine_conservation_independent():
    text = synthetic()
    text = text.replace(' .01\nFINE', ' .02\nFINE')
    with pytest.raises(ValueError, match='ledger'):
        p.parse_egs(text, 100, 9, p.LAYERS, RHO)


def test_pegs_two_stage():
    assert p.check_pegs(deck(), 5, deck(4))['status'] == 'pass'
    assert p.check_pegs(deck(4), 4, deck(4))['status'] == 'pass'
    _, _, old = p.parse_egs(synthetic(p.p8e.LAYERS, 4), 100, 9, p.p8e.LAYERS, RHO, media_set=4)
    _, _, five = p.parse_egs(synthetic(p.p8e.LAYERS, 5), 100, 9, p.p8e.LAYERS, RHO)
    p.compare_settings(five, old)
    five['regions'][1] += ' 1'
    with pytest.raises(ValueError):
        p.compare_settings(five, old)


@pytest.mark.parametrize('old,new', [('2.4700E+00', '2.4800E+00'), ('NE= 2', 'NE= 3'),
    ('5.95500E-01', '5.90000E-01'), ('Z=47.', 'Z=48.'), ('old material 0', 'altered material 0'),
    ('P9A_SEED', 'P8A_SEED_AGTI')])
def test_pegs_faults(old, new):
    with pytest.raises(ValueError):
        p.check_pegs(deck().replace(old, new), 5, deck(4))


def test_existing_p8e_replay_read_only():
    with np.load(p.p8e.OUT/'egs_pilot.npz') as saved:
        dose = saved['dose'].copy()
    assert p.replay_comparison(dose, dose.copy())['status'] == 'pass'
    dose[0, 0] = np.nextafter(dose[0, 0], np.inf)
    with np.load(p.p8e.OUT/'egs_pilot.npz') as saved:
        with pytest.raises(ValueError, match='stop'):
            p.replay_comparison(dose, saved['dose'])


def test_projection_correlations_zeros_limit():
    g = p.layout(p.LAYERS, RHO)
    fluct = np.linspace(-.1, .1, 100)
    d = np.ones((100, 216))+fluct[:, None]
    r = p.projection(d, g, 1000)
    sem = np.std(fluct, ddof=1)/10
    expected = max(100, math.ceil(1.5*p.PILOT_N*sem**2/(.0033**2/2)/100)*100)
    assert r['histories'] == expected
    np.testing.assert_allclose(r['projected_SEM_over_Dref'], sem*np.sqrt(p.PILOT_N/expected))
    cpu = 43200*p.PILOT_N/expected
    assert not p.projection(d, g, cpu)['exceeds_12_hours']
    assert p.projection(d, g, cpu*1.00001)['exceeds_12_hours']
    assert p.projection(np.ones_like(d), g, 0)['histories'] == 100
    with pytest.raises(ValueError, match='Dref'):
        p.projection(np.zeros_like(d), g, 1)
    for bad in (np.nan, np.inf, -1):
        with pytest.raises(ValueError):
            p.projection(d, g, bad)


@pytest.fixture
def fake_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'external_sources', lambda: {'frozen': 'hash'})
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        work = Path(kwargs['cwd'])
        for name in ('egs5job.exe', 'pgs5job.pegs5dat', 'pgs5job.pegs5lst'):
            (work/name).write_text('test')
        (work/'egs5job.out').write_text('good')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(p.subprocess, 'run', fake)
    return tmp_path, calls


def test_execute_identity_seal_resume(fake_execution):
    work, calls = fake_execution
    def validate(text):
        assert text == 'good'
    p.execute(work, 'pegs input', 'config', 'spectrum', validate)
    assert len(calls) == 2
    p.verify_seal(work)
    identity = json.loads((work/'identity.json').read_text())
    assert identity['pegs_input_sha256'] == p.digest('pegs input')
    assert 'definition' not in identity and 'I_MeV' not in str(identity)
    assert set(identity['sources']) == {'p9a_egs.f'}
    p.execute(work, 'pegs input', 'config', 'spectrum', validate)
    assert len(calls) == 2
    with pytest.raises(ValueError, match='resume'):
        p.execute(work, 'changed', 'config', 'spectrum', validate)
    (work/'p9a_egs.inp').write_text('changed')
    with pytest.raises(ValueError, match='sealed'):
        p.execute(work, 'pegs input', 'config', 'spectrum', validate)


@pytest.mark.parametrize('file', ['identity.json', 'p9a_egs.f', 'pgs5job.pegs5dat',
                                 'pgs5job.pegs5lst', 'egs5job.out', 'config.dat', 'spectrum.dat'])
def test_seal_tampering(fake_execution, file):
    work, _ = fake_execution
    p.execute(work, 'inp', 'config', 'spectrum', lambda text: None)
    (work/file).write_text((work/file).read_text()+'x')
    with pytest.raises(ValueError):
        p.verify_seal(work)


def test_failed_validation_retains_attempt(fake_execution):
    work, calls = fake_execution
    def fail(text):
        raise ValueError('unhealthy output')
    with pytest.raises(ValueError):
        p.execute(work, 'inp', 'config', 'spectrum', fail)
    assert not (work/'complete.json').exists()
    assert json.loads((work/'execution.json').read_text())['status'] == 'failed'
    with pytest.raises(RuntimeError, match='retained'):
        p.execute(work, 'inp', 'config', 'spectrum', fail)
    assert len(calls) == 2


@pytest.fixture(scope='module')
def fortran(tmp_path_factory):
    compiler = shutil.which('gfortran')
    assert compiler, 'gfortran required to verify standalone scorers'
    work = tmp_path_factory.mktemp('p9a_fortran')
    source = p.CODE.read_text()
    # Syntax/type-check full source against real declarations; never link EGS.
    subprocess.run([compiler, '-std=legacy', '-fcheck=all', '-c', '-I'+str(p.EGS),
                    str(p.CODE), '-o', str(work/'user.o')], check=True, capture_output=True)
    common = '''      integer nl,nt,lmed(16),first(16),last(16)
      real*8 bound(0:16),bw(16),bleft(1000),bright(1000)
      common/p8geom/bound,bw,bleft,bright,nl,nt,lmed,first,last
      real*8 fscore(1000)
      common/p9score/fscore
      integer nf,ffirst(16),flast(16)
      real*8 fleft(1000),fright(1000)
      common/p9geom/fleft,fright,nf,ffirst,flast
'''
    driver = '''      program test
      implicit none
      include 'include/egs5_h.f'
      include 'include/egs5_epcont.f'
      include 'include/egs5_stack.f'
'''+common+'''      real*8 p8score(1000),p8dep,p8escape
      common/p8score/p8score,p8dep,p8escape
      integer n,seed,fluor,media_set,reg,j,ios
      real*8 pos,dir,step,loss
      call p8config(n,seed,fluor,media_set)
      do
        read(*,*,iostat=ios) reg,pos,dir,step,loss
        if(ios.ne.0) exit
        p8score=0.d0
        fscore=0.d0
        p8dep=0.d0
        p8escape=0.d0
        np=1
        ir(1)=reg
        z(1)=pos
        w(1)=dir
        iq(1)=-1
        wt(1)=1.d0
        ustep=step
        irnew=reg
        idisc=0
        edep=loss
        call howfar
        call ausgab(0)
        write(*,*) (p8score(j),j=1,nt),(fscore(j),j=1,nf),
     *             p8dep,p8escape,ustep,irnew,idisc
      enddo
      end
'''
    (work/'test.f').write_text(driver+source[source.index('      subroutine ausgab'):])
    subprocess.run([compiler, '-std=legacy', '-fcheck=all', '-I'+str(p.EGS),
                    str(work/'test.f'), '-o', str(work/'test')], check=True, capture_output=True)
    return work


def drive(work, config, events=''):
    (work/'config.dat').write_text(config)
    return subprocess.run([str(work/'test')], cwd=work, input=events, text=True, capture_output=True)


def test_fortran_main_unchanged():
    source, old = p.CODE.read_text(), p.p8e.CODE.read_text()
    a, b = '      subroutine howfar', '! Standalone routines:'
    assert source[source.index(a):source.index(b)] == old[old.index(a):old.index(b)]
    assert source[source.index('      subroutine p8deposit'):source.index('      subroutine p9deposit')].strip() == old[old.index('      subroutine p8deposit'):].strip()
    # The primary accumulation statements are preserved exactly.
    for line in ('        batches(:,b)=batches(:,b)+p8score',
                 '     *      batches(j,b)/(ncase/100*width*rho)'):
        assert line in source and line in old


@pytest.mark.parametrize('layers,media_set', [(p.LAYERS, 5), (p.p8e.LAYERS, 4),
                                           (p.p8e.LAYERS, 5), (p.SLAB_LAYERS, 5)])
def test_fortran_attribution_conservation(fortran, layers, media_set):
    events = []
    pos = 0.
    for i, (_, t, _) in enumerate(layers):
        # Both directions, interface points in the current region, zero w,
        # and a proposed step beyond the layer (howfar clips the step).
        events += [f'{i+2} {pos} 1 {t} 2', f'{i+2} {pos+t} -1 {t} 2',
                   f'{i+2} {pos} 0 1 2', f'{i+2} {pos+t} 0 1 2',
                   f'{i+2} {pos+t/2} 1 {t} 2']
        pos += t
    result = drive(fortran, p.config_text(100, 9, layers, media_set=media_set), '\n'.join(events)+'\n')
    assert result.returncode == 0, result.stderr
    arr = np.array([list(map(float, row.split())) for row in result.stdout.splitlines()])
    nb = sum(round(t/w) for _, t, w in layers)
    nf = round(sum(t for _, t, _ in layers)/.05)
    np.testing.assert_allclose(arr[:, :nb].sum(axis=1), 2., atol=2e-12)
    np.testing.assert_allclose(arr[:, nb:nb+nf].sum(axis=1), 2., atol=2e-12)
    np.testing.assert_array_equal(arr[:, nb+nf:nb+nf+2], [[2., 0.]]*len(events))
    # Uniform traversal of the thin plate: distinct .02 and .05 cm arrays.
    if layers == p.LAYERS:
        np.testing.assert_allclose(arr[5, 50:55], .4, atol=2e-13)
        np.testing.assert_allclose(arr[5, nb+100:nb+102], 1., atol=2e-13)


@pytest.mark.parametrize('config,code', [
    ('100 1 1 1 2\n1 1 .1\n', 51), ('100 1 1 1 4\n5 1 .1\n', 44),
    ('101 1 1 1 5\n1 1 .1\n', 41), ('100 1 2 1 5\n1 1 .1\n', 42),
    ('100 1 1 1 5\n1 .11 .01\n', 52), ('100 1 1 1 5\n1 50.1 .1\n', 52),
    ('100 1 1 1 5\n1 NaN .1\n', 45), ('100 1 1 1 5\n1 1 .3\n', 47)])
def test_fortran_bad_config(fortran, config, code):
    result = drive(fortran, config)
    assert result.returncode == code, result.stderr


@pytest.mark.parametrize('mode,slow', [('egs-pilot', False), ('egs', False), ('egs', True)])
def test_runner_prerequisites_seeds_stop(tmp_path, monkeypatch, mode, slow):
    monkeypatch.setattr(p, 'load_inputs', lambda: ([], 'spectrum'))
    actual = p.run
    dependencies, calls = [], []
    def prerequisite(command, out, cache):
        dependencies.append(command)
        return dict(projection=dict(histories=123400, exceeds_12_hours=slow))
    monkeypatch.setattr(p, 'run', prerequisite)
    def case(folder, n, seed, layers, media_set, definitions, spectrum, out, cache):
        calls.append((folder, n, seed, layers, media_set))
        return dict(status='saved')
    monkeypatch.setattr(p, 'run_case', case)
    if slow:
        with pytest.raises(RuntimeError, match='12 hours'):
            actual(mode, tmp_path/'out', tmp_path/'cache')
        assert not calls
    else:
        actual(mode, tmp_path/'out', tmp_path/'cache')
        expected = ('pilot', 1000000, 920002) if mode == 'egs-pilot' else ('production', 123400, 920001)
        assert calls == [(*expected, p.LAYERS, 5)]
    assert dependencies == ['egs-replay' if mode == 'egs-pilot' else 'egs-pilot']


def test_run_case_artifact_validation_before_seal(tmp_path, monkeypatch):
    definitions = [dict(density_g_cm3=r) for r in RHO]
    monkeypatch.setattr(p, 'pegs_input', lambda ds: 'pegs input')
    def execute(work, inp, config, spectrum, validate):
        work.mkdir(parents=True)
        (work/'pgs5job.pegs5dat').write_text(deck())
        text = synthetic(n=1000000, seed=920002)
        validate(text)
        (work/'complete.json').write_text('{}')
        return text, 'identity hash'
    monkeypatch.setattr(p, 'execute', execute)
    report = p.run_case('pilot', 1000000, 920002, p.LAYERS, 5, definitions, 'spectrum', tmp_path/'out', tmp_path/'cache')
    with np.load(tmp_path/'out/egs_pilot.npz') as saved:
        assert saved['dose'].shape == (100, 216)
        assert saved['fine'].shape == (100, 424)
        assert saved['batch_blocks'].shape == (100, 19)
        assert saved['fine_edges_cm'].shape == (425,)
    assert report['pegs_input_sha256'] == p.digest('pegs input')
    assert report['seed'] == 920002 and report['projection']['histories'] >= 100


@pytest.mark.parametrize('bad_four,bad_five', [(False, False), (True, False), (False, True)])
def test_replay_driver_two_stages(tmp_path, monkeypatch, bad_four, bad_five):
    monkeypatch.setattr(p, 'load_inputs', lambda: ([], 'spectrum'))
    old_dose, _, old_health = p.parse_egs(synthetic(p.p8e.LAYERS, 4), 100, 9, p.p8e.LAYERS, RHO, media_set=4)
    monkeypatch.setattr(p, 'load_replay_reference', lambda ds, s: (old_dose, old_health, deck(4)))
    calls = []
    def case(folder, n, seed, layers, media_set, definitions, spectrum, out, cache, validate):
        calls.append((folder, n, seed, media_set))
        dose = old_dose.copy()
        _, _, health = p.parse_egs(synthetic(p.p8e.LAYERS, media_set), 100, 9, p.p8e.LAYERS, RHO, media_set=media_set)
        if media_set == 4 and bad_four:
            dose[0, 0] = np.nextafter(dose[0, 0], np.inf)
        if media_set == 5:
            dose *= 2  # Five media must NOT require a byte replay.
            if bad_five:
                health['regions'][1] += ' changed'
        validate(dose, health, deck(media_set))
        return {'status': 'sealed'}
    monkeypatch.setattr(p, 'run_case', case)
    if bad_four or bad_five:
        with pytest.raises(ValueError):
            p.run('egs-replay', tmp_path/'out', tmp_path/'cache')
    else:
        assert p.run('egs-replay', tmp_path/'out', tmp_path/'cache')['status'] == 'pass'
    assert calls[0] == ('replay_4media', 1000000, 820000, 4)
    if not bad_four:
        assert calls[1] == ('replay_5media', 100, 820000, 5)
    else:
        assert len(calls) == 1


def test_read_reference_seal_and_metadata(tmp_path, monkeypatch):
    work = tmp_path/'cache/pilot'
    work.mkdir(parents=True)
    out = tmp_path/'out'
    out.mkdir()
    monkeypatch.setattr(p, 'pegs_input', lambda ds: 'pegs input')
    monkeypatch.setattr(p, 'external_sources', lambda: {'frozen': 'hash'})
    n, seed = p.p8e.PILOT_N, p.p8e.PILOT_SEED
    config = p.p8e.config_text(n, seed, p.p8e.LAYERS)
    text = synthetic(p.p8e.LAYERS, 4, n, seed)
    text = '\n'.join(line for line in text.splitlines() if not line.startswith('FINE'))+'\n'
    identity = dict(input='pegs input', egs_inputs={'frozen': 'hash'}, config=config,
                    spectrum_sha256=p.digest('spectrum'), sources={p.p8e.CODE.name: p.p8e.CODE.read_text()})
    files = {'egs5job.out': text, 'config.dat': config, 'spectrum.dat': 'spectrum',
             'p8e_egs.inp': 'pegs input', 'pgs5job.pegs5dat': deck(4),
             'pgs5job.pegs5lst': 'listing', p.p8e.CODE.name: p.p8e.CODE.read_text(),
             'identity.json': json.dumps(identity)}
    for name, content in files.items():
        (work/name).write_text(content)
    p.write(work/'complete.json', {name: p.sha(work/name) for name in files})
    g = p.p8e.layout(p.p8e.LAYERS, RHO[:4])
    np.savez(out/'egs_pilot.npz', dose=np.full((100, 220), .01), **g)
    report = dict(N=n, seed=seed, layers=[list(x) for x in p.p8e.LAYERS],
                  densities_g_cm3=RHO[:4], units='MeV cm2/g/history',
                  artifact_sha256=p.sha(out/'egs_pilot.npz'), complete_sha256=p.sha(work/'complete.json'),
                  identity_sha256=p.sha(work/'identity.json'))
    p.write(out/'egs_pilot.json', report)
    ds = [dict(density_g_cm3=r) for r in RHO]
    dose, health, pegs = p.load_replay_reference(ds, 'spectrum', out, tmp_path/'cache')
    assert dose.shape == (100, 220) and pegs == deck(4)
    report['seed'] += 1
    p.write(out/'egs_pilot.json', report)
    with pytest.raises(ValueError, match='N/seed'):
        p.load_replay_reference(ds, 'spectrum', out, tmp_path/'cache')
    report['seed'] -= 1
    p.write(out/'egs_pilot.json', report)
    (work/'config.dat').write_text('changed')
    with pytest.raises(ValueError, match='sealed'):
        p.load_replay_reference(ds, 'spectrum', out, tmp_path/'cache')

# Second implementation: material pipeline, portable storage and judgement.
from prototypes.mv_electron import p9a_data as d9, p9a_judge as j9
from prototypes.mv_electron.gpu import p9a as g9, p9a_store as s9


def test_protection_and_stage_seals(tmp_path):
    a=tmp_path/'old';a.write_bytes(b'old')
    out=tmp_path/'out'
    assert d9.protect(out,[a])['status']=='pass'
    assert d9.protect(out,[a])['files']==1
    d9.finish('example',[a],out)
    d9.verified('example',out)
    a.write_bytes(b'changed')
    with pytest.raises(ValueError,match='protected'):d9.verify_protected(out)
    with pytest.raises(ValueError,match='hash'):d9.verified('example',out)


def test_formal_I_from_META_and_stopping(monkeypatch):
    from prototypes.mv_electron import pegs_p8a as pg
    monkeypatch.setattr(pg,'validate_diagnostic_state',lambda text,d:None)
    d=d9.finalized_definition('META .511 .521 .01 2 2.47 233.7\n')
    assert d['I_MeV']==pytest.approx(233.7e-6)
    assert d['I_MeV']!=p.seed_definition()['I_MeV']
    assert d['medium_name']=='P8A_SEED_AGTI'
    assert p.seed_definition()['medium_name']=='P9A_SEED'
    a=dict(T=np.array([.01,.1,1.]),collision=np.array([2.,3.,4.]),radiative=np.array([1.,2.,3.]),delta=np.zeros(3))
    b=dict(T=np.array([.03,.3]),collision=np.array([2.5,3.5]),radiative=np.array([1.5,2.5]),delta=np.zeros(2))
    s=d9.stopping_data(a,b)
    np.testing.assert_array_equal(s['total'],[3,5,7])
    np.testing.assert_array_equal(s['check_total'],[4,6])
    report=d9.estar_report(d,s,{z:dict(T=a['T'],collision=a['collision'],I_MeV=1e-4) for z in (22,47)})
    assert report['status']=='report-only'
    assert report['Bragg_I_MeV']==pytest.approx(1e-4)


def test_photon_calls_existing_generator_only(tmp_path,monkeypatch):
    from prototypes.mv_electron import fetch_materials_p8a as fetch
    calls=[]
    def fake(work,d):
        calls.append((work,d));return dict(E=np.array([.01,.1]))
    monkeypatch.setattr(fetch,'photon_table',fake)
    d9.photon(tmp_path/'cache',tmp_path/'out')
    d9.photon(tmp_path/'cache',tmp_path/'out')
    assert len(calls)==1 and calls[0][1]['weights']=={'47':.5955,'22':.4045}
    assert not (tmp_path/'cache/seed_agti/definition.json').exists()


@pytest.mark.parametrize('layers', [p.LAYERS,p.SLAB_LAYERS,p.p8e.LAYERS])
def test_gpu_single_input_shapes(layers):
    bundle=dict(names=['water','lung','bone','lead','seed_agti'],densities=RHO)
    record,arrays=g9.input_record([None]*5,bundle,[.1,10.],[1.],layers,200000)
    nl,nb,nc,nf,ns=record['sizes']
    assert len(layers)==nl and ns==1
    assert record['schedule']['shapes']['_groups_d']==[100,nb+nc+nf+37]
    assert record['schedule']['shapes']['_groups_i']==[100,79+nb]
    assert record['schedule']['shapes']['fine']==[100,nf]
    assert arrays['mass_g_cm2'].shape==(nb,)
    assert arrays['ids'].tolist()==[m-1 for m,_,_ in layers]
    np.testing.assert_array_equal(arrays['bins'],p.layout(layers,RHO)['edges_cm'])


def fake_result(sizes=(5,216,106,424,1)):
    nb,nc,nf=sizes[1:4]
    return dict(dose=np.ones(nb),p8_batches=np.ones((100,nb)),
        coarse=np.ones((100,nc)),fine=np.ones((100,nf)),additions=np.ones((100,nb),np.uint64),
        _groups_d=np.zeros((100,nb+nc+nf+37)),_groups_i=np.zeros((100,79+nb),np.uint64),
        _absolute=np.ones((100,nb+nc+nf+37)),seconds=np.array(1.),
        totals=np.array([1.,1.,0.,0.,0.,0.]),maxima=np.zeros(3),violations=np.zeros(3,np.int64),
        counts=np.zeros(16,np.int64),brems=np.zeros(10),p8_max=np.zeros(2),maxdepth=np.array(3))


def small_schedule(count=2,seed=920102):
    return s9.spec(count=count,n=100,mode='gpu',seed=seed,
                   shapes=g9.shapes_from_sizes([5,216,106,424,1]))


def fake_session(root,identity,schedule):
    return s9.new_session(root,identity,schedule,'T4',
        dict(status='pass',n=2000,seed=g9.core.SEED2,comparison='P9a CPU/GPU history comparison'),dict(status='new'))


def test_storage_999_1000_shapes_hash_and_schedule(tmp_path):
    schedule=small_schedule(1001);identity={'input':'synthetic'}
    sid=fake_session(tmp_path,identity,schedule)
    result=fake_result()
    for j in (999,1000):
        s9.save_call(tmp_path,j,result,np.ones(100),s9.invocation(j,schedule),identity,schedule,sid)
    assert s9.call_path(tmp_path,999).name=='call_0999.npz'
    assert s9.call_path(tmp_path,1000).name=='call_1000.npz'
    assert set(s9.available(tmp_path,identity,schedule))=={999,1000}
    changed=dict(schedule,seed=schedule['seed']+1)
    with pytest.raises(ValueError):s9.available(tmp_path,identity,changed)
    path=s9.call_path(tmp_path,999);path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash'):s9.load_call(tmp_path,999,identity,schedule)


def test_drive_resume_and_conflicting_schedule(tmp_path):
    schedule=small_schedule(2);identity={'input':'synthetic'}
    sid=fake_session(tmp_path,identity,schedule);calls=[]
    def runner(**kw):calls.append(kw);return fake_result(),np.ones(100)
    s9.drive(tmp_path,runner,identity,schedule,sid,limit=1)
    assert len(calls)==1
    assert s9.resume_check(tmp_path,runner,identity,schedule)['status']=='pass'
    s9.drive(tmp_path,runner,identity,schedule,sid)
    assert s9.verify(tmp_path,identity,schedule)['n']==200
    with pytest.raises(ValueError):s9.drive(tmp_path,runner,identity,dict(schedule,count=3),sid)
    with pytest.raises(ValueError):s9.resume_check(tmp_path,runner,identity,dict(schedule,count=3))


@pytest.mark.parametrize('layers',[p.LAYERS,p.SLAB_LAYERS,p.p8e.LAYERS])
def test_storage_all_layout_shapes(layers):
    record,_=g9.input_record([None]*5,dict(names=list('abcde'),densities=RHO),[.1,10.],[1.],layers)
    r=fake_result(record['sizes']);s9.check_result(r,record['schedule'])
    r['fine']=r['fine'][:,:-1]
    with pytest.raises(ValueError,match='shape'):s9.check_result(r,record['schedule'])


def test_correlated_raw_normalization_to_block_sem():
    grid=p.layout(p.LAYERS,RHO);nb=len(grid['mass_g_cm2'])
    fluctuations=np.linspace(-.1,.1,100)
    target=1+fluctuations[:,None]*np.ones((1,nb))
    # Two calls, each contributing 2000 histories to every batch.
    raw=[dict(p8_batches=target*2000*grid['mass_g_cm2']) for _ in range(2)]
    dose=j9.normalize_calls(raw,[200000,200000],grid['mass_g_cm2'])
    np.testing.assert_allclose(dose,target)
    values,mean,sem=p.block_statistics(dose,grid)
    assert values.shape==(100,19)
    np.testing.assert_allclose(mean,1)
    np.testing.assert_allclose(sem,fluctuations.std(ddof=1)/10)
    # Correlated bin noise must not be reduced by sqrt(bins per block).
    report=j9.judge_batches(dose,np.ones_like(dose),grid)
    np.testing.assert_allclose(report['SEM'],sem)


def test_normalization_rejects_wrong_counts_shapes_values():
    with pytest.raises(ValueError):j9.normalize_calls([],[],[1.])
    for n in (0,99,101):
        with pytest.raises(ValueError):j9.normalize_calls([dict(p8_batches=np.ones((100,1)))],[n],[1.])
    with pytest.raises(ValueError):j9.normalize_calls([dict(p8_batches=np.full((100,1),np.nan))],[100],[1.])


def noisy_dose(grid,relative_sem):
    z=np.linspace(-1,1,100);z=z/(z.std(ddof=1)/10)
    return 1+z[:,None]*relative_sem


def test_gpu_projection_formula_zero_sem_cap_and_dref():
    grid=p.layout(p.LAYERS,RHO)
    e=np.ones((100,len(grid['mass_g_cm2'])))
    g=noisy_dose(grid,np.full(e.shape[1],.003))
    result=j9.gpu_projection(e,g,grid,1.)
    required=200000*.003**2/.0033**2
    assert result['histories']==math.ceil(1.5*required/200000)*200000
    assert len(result['projected_SEM_over_Dref'])==19
    n=result['histories']
    assert j9.gpu_projection(e,g,grid,1.,cap=n)['status']=='ready'
    assert j9.gpu_projection(e,g,grid,1.,cap=n-1)['status']=='stop-consult-user'
    assert j9.gpu_projection(e,e,grid,1.)['histories']==200000
    for bad in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):j9.gpu_projection(e,g,grid,bad)
    with pytest.raises(ValueError,match='EGS SEM'):j9.gpu_projection(noisy_dose(grid,np.full(e.shape[1],.004)),g,grid,1.)


def test_projection_different_limiting_blocks():
    grid=p.layout(p.LAYERS,RHO);nb=len(grid['mass_g_cm2'])
    se=np.zeros(nb);sg=np.zeros(nb)
    se[grid['block_index']==0]=.003
    sg[grid['block_index']==1]=.01
    a=j9.gpu_projection(noisy_dose(grid,se),noisy_dose(grid,sg),grid,1.)
    assert np.argmax(a['EGS_SEM_over_Dref'])==0
    assert np.argmax(a['required_by_block'])==1
    assert a['histories']%200000==0


def test_19_block_sequential_judgement_and_thresholds():
    from prototypes.mv_electron.run_p8e import classify_blocks
    assert classify_blocks([.01,0,.02,0],[.0033,.004,0,0])==['pass','insufficient-statistics','fail','not-judged']
    grid=p.layout(p.LAYERS,RHO);reference=np.ones((100,len(grid['mass_g_cm2'])))
    actual=reference.copy()
    actual[:,grid['block_index']==1]+=.02
    report=j9.judge_batches(actual,reference,grid)
    assert len(report['status'])==19
    assert report['status'][:3]==['pass','fail','not-judged']


def test_kernel_frozen_fingerprint():
    assert g9.kernel_check()['status']=='pass'


def test_numeric_and_storage_faults(tmp_path):
    from prototypes.mv_electron.gpu.p9a_colab import faults
    record=dict(schedule=small_schedule())
    detected=faults(fake_result(),record)
    assert {'nonfinite','actual interval','schedule','artifact hash'}<=set(detected)


def test_egs_analysis_hash_rejected_before_statistics(tmp_path,monkeypatch):
    out=tmp_path/'out';out.mkdir()
    p.write(out/'egs_pilot.json',dict(N=p.PILOT_N))
    monkeypatch.setattr(p,'verify_seal',lambda work:None)
    (tmp_path/'pilot').mkdir()
    for name in ('identity.json','complete.json'):(tmp_path/'pilot'/name).write_text('{}')
    (out/'egs_pilot.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash/N/seed/layout'):j9.load_egs(out,tmp_path,RHO,'pilot')


@pytest.fixture(scope='module')
def p9_cpu_builds(tmp_path_factory):
    # Read existing bundles only. A fifth synthetic material exercises ID 4.
    from prototypes.mv_electron.gpu import p8g
    materials,identity,edges,cum=p8g.inputs()
    work=tmp_path_factory.mktemp('p9a-cpu')
    kw=dict(layers=p.p8e.LAYERS,densities=[*identity['densities'],2.47],edges=edges,cum=cum)
    return g9.CPU(work,materials,**kw),g9.CPU(work,[*materials,materials[3]],**kw),materials,work,kw


@pytest.mark.parametrize('n,start',[(20000,0),(2000,7000)])
def test_old_four_material_cpu_byte_regression(p9_cpu_builds,n,start):
    old,new,_,_,_=p9_cpu_builds
    assert old.identity==new.identity
    g9.core.exact_histories(new.histories(n,g9.core.SEED2,start),old.histories(n,g9.core.SEED2,start),'P9a old four')


def test_fifth_material_cpu_slab(p9_cpu_builds):
    _,_,materials,work,kw=p9_cpu_builds
    cpu=g9.CPU(work,[*materials,materials[3]],**dict(kw,layers=p.SLAB_LAYERS))
    h,v,_,ag=cpu.histories(100,g9.core.SEED2,details=True)
    expected=cpu.histories(100,g9.core.SEED2,reference=True)
    g9.core.exact_histories(h,expected,'fifth ID')
    g9.core.check_visits(v)
    g9.core.check_group_records(ag,h,100)


def test_pegs_pipeline_final_definition_and_grid(tmp_path,monkeypatch):
    from prototypes.mv_electron import pegs_p8a as pg
    nodes=np.geomspace(.01,20,1201);mid=np.sqrt(nodes[:-1]*nodes[1:])
    old=dict(T=nodes,check_T=mid)
    monkeypatch.setattr(d9.mat,'load_manifest',lambda:dict(materials=dict(water=dict(stopping={}))))
    monkeypatch.setattr(d9.mat,'load_reference',lambda r,m:old)
    seen=[]
    def diagnostic(d,energies,work):
        assert d['I_MeV']==1e-4 and d['medium_name']=='P8A_SEED_AGTI'
        seen.append(energies.copy());work.mkdir(parents=True)
        (work/'pegs').write_text('synthetic')
        return dict(T=energies,collision=np.full(len(energies),2.),radiative=energies,
                    delta=np.zeros(len(energies)),text='META .511 .521 .01 2 2.47 233.7\n',pegs=work/'pegs')
    monkeypatch.setattr(pg,'run_diagnostic',diagnostic)
    monkeypatch.setattr(pg,'validate_diagnostic_state',lambda text,d:None)
    monkeypatch.setattr(pg,'extract_shells',lambda text,d:dict(I=d['I_MeV']))
    monkeypatch.setattr(pg,'pair_parameters',lambda text,name,d:dict(I=d['I_MeV']))
    monkeypatch.setattr(d9,'acquire_estar',lambda *args:dict(status='report-only'))
    d9.pegs(tmp_path/'cache',tmp_path/'out')
    d9.pegs(tmp_path/'cache',tmp_path/'out')
    assert len(seen)==2
    np.testing.assert_array_equal(seen[0],nodes);np.testing.assert_array_equal(seen[1],mid)
    folder=tmp_path/'cache/seed_agti'
    assert d9.read(folder/'definition.json')['I_MeV']==pytest.approx(233.7e-6)
    assert d9.read(folder/'constants.json')['I_MeV']==pytest.approx(233.7e-6)
    assert d9.read(folder/'stopping.npz')['T'].shape==(1201,)
    assert d9.read(folder/'stopping.npz')['check_T'].shape==(1200,)


def mock_elsepa_pipeline(tmp_path,monkeypatch,timeout=False):
    from prototypes.mv_electron import generate_elsepa_p8a as el
    from prototypes.mv_electron import gs_backend as gs
    cache=tmp_path/'cache';folder=cache/'seed_agti';folder.mkdir(parents=True)
    d=p.seed_definition();d['I_MeV']=.0002337;d9.mat.write(folder/'definition.json',d)
    d9.finish('pegs',[folder/'definition.json'],tmp_path/'out')
    monkeypatch.setattr(el,'prepare',lambda:('source','exe'))
    monkeypatch.setattr(el,'energies',lambda:np.array([.01,10.]))
    calls=[]
    def case(src,exe,work,z,q,t,**kw):
        calls.append((work,z,q,t,kw))
        assert q==-1 and z in (22,47)
        if 'replacements' in work.parts and timeout:
            raise el.ElsepaCaseTimeout(dict(Z=z,charge=q,energy_MeV=t))
        return dict(value=t)
    monkeypatch.setattr(el,'case',case)
    def stack(rows,z,q,es):
        return dict(Z=np.array(z),charge=np.array(q),IHEF=np.array(0),energy_MeV=es,
                    mu=np.array([0.,1.]),dcs=np.ones((len(es),2)))
    monkeypatch.setattr(el,'stack_rows',stack)
    def direct(src,exe,work,z,q,es,nodes):
        d9.mat.write(work/'direct_status.json',dict(omitted=[dict(energy_MeV=.1)] if timeout else []))
        np.savez(work/'direct.npz',**nodes)
        return d9.read(work/'direct_status.json')
    monkeypatch.setattr(el,'collect_direct',direct)
    monkeypatch.setattr(gs,'build',lambda provider,path:np.savez(path,T=[.01,10.]))
    return calls,cache


def test_elsepa_two_pass_case_reuse_and_electron_only(tmp_path,monkeypatch):
    calls,cache=mock_elsepa_pipeline(tmp_path,monkeypatch)
    # The real case() is the existing sealed-point reader; paths must match the
    # precomputed pass1/pass2/<float.hex> tree exactly, with no alternative cache.
    d9.elsepa(cache,tmp_path/'out')
    assert len(calls)==8
    for work,z,q,t,kw in calls:
        assert work==cache/'elsepa'/f'z{z}_q-1'/work.parent.name/t.hex()
        assert work.parent.name in ('pass1','pass2')
    d9.elsepa(cache,tmp_path/'out')
    assert len(calls)==8


def test_elsepa_one_replacement_then_stop(tmp_path,monkeypatch):
    from prototypes.mv_electron import generate_elsepa_p8a as el
    calls,cache=mock_elsepa_pipeline(tmp_path,monkeypatch,timeout=True)
    with pytest.raises(el.ElsepaCaseTimeout):d9.elsepa(cache,tmp_path/'out')
    replacements=[x for x in calls if 'replacements' in x[0].parts]
    assert len(replacements)==1
    assert replacements[0][3]==pytest.approx(.1*el.REPLACEMENT_FACTOR)
    assert not (tmp_path/'out/elsepa.json').exists()


def test_bundle_passes_density_and_PEGS_I(tmp_path,monkeypatch):
    from prototypes.mv_electron import photon_transport_p8e as t
    from prototypes.mv_electron import generate_fluorescence_p8e as fl
    folder=tmp_path/'cache/seed_agti';folder.mkdir(parents=True)
    constants=dict(Z=[22,47],NJ=[1.,2.],NE=116.,I_MeV=.0002337)
    d9.mat.write(folder/'constants.json',constants)
    d9.mat.write(folder/'definition.json',dict(I_MeV=.0002337))
    for name in ('stopping','photon','gs','pair','shells'):np.savez(folder/(name+'.npz'),T=[.01,10.])
    for stage in ('photon','pegs','elsepa'):d9.finish(stage,[folder/'constants.json'],tmp_path/'out')
    old=tmp_path/'old';old.mkdir();(old/'bundles.json').write_text('{}')
    monkeypatch.setattr(d9,'OLD',old)
    identity=dict(densities=RHO[:4]);monkeypatch.setattr(t,'load_bundles',lambda path:([None]*4,identity))
    monkeypatch.setattr(fl,'fluorescence_table',lambda shells:dict(yields=np.zeros((2,4))))
    monkeypatch.setattr(t,'brems_tables',lambda *args:('E','V'))
    called=[]
    class Material:
        excitation=.0002337
        def _asdict(self):return dict(excitation=self.excitation)
    def make(c,*args,**kw):
        called.append(kw);assert c['I_MeV']==.0002337;return Material()
    monkeypatch.setattr(t,'make_material',make)
    monkeypatch.setattr(t,'bundle_checks',lambda m:dict(status='pass'))
    d9.bundle(tmp_path/'cache',tmp_path/'out')
    assert called==[dict(density=2.47)]
    result=d9.read(tmp_path/'cache/materials/seed_agti_bundle.npz')
    assert float(result['excitation'])==.0002337


@pytest.mark.parametrize('layout_name,phase,n,seed',[
    ('slab','component',2000,g9.core.SEED2),('old','component',2000,g9.core.SEED2),
    ('stack','pilot',200000,g9.PILOT_SEED)])
def test_export_load_portable_input_roundtrip(tmp_path,monkeypatch,p9_cpu_builds,layout_name,phase,n,seed):
    from prototypes.mv_electron.gpu import p9a_colab as colab
    _,_,materials,_,kw=p9_cpu_builds
    bundle=dict(names=['water','lung','bone','lead','seed_agti'],densities=RHO)
    out=tmp_path/'out'
    protected=tmp_path/'frozen';protected.write_text('immutable')
    d9.protect(out,[protected]);d9.finish('bundle',[protected],out)
    d9.mat.write(out/'check.json',dict(status='pass',bundles=bundle))
    monkeypatch.setattr(d9,'load_bundles',lambda cache,out:([*materials,materials[3]],bundle))
    class FakeCPU:
        def __init__(self,*args,**kwargs):pass
        def histories(self,n,seed):return dict(offline=np.array([n,seed]))
    monkeypatch.setattr(g9,'CPU',FakeCPU)
    work=tmp_path/'export'
    result=g9.export(work,tmp_path/'cache',out,layout_name,n,seed,phase)
    record,m,arrays=colab.load(work)
    assert result['input']==record
    assert len(m)==5 and record['schedule']['n']==n
    assert record['schedule']['count']==1
    assert arrays['sizes'].tolist()==record['sizes']
    # A portable import needs only the GPU sources; the CPU/data acquisition
    # modules (which read legacy data on import) are not Colab dependencies.
    import zipfile
    with zipfile.ZipFile(work/'source.zip') as z:
        for name in g9.core.fingerprint():assert 'prototypes/mv_electron/gpu/'+name in z.namelist()
    path=work/'seed_agti_bundle.npz';path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='export hash'):colab.load(work)


def test_sealed_gpu_load_normalization_and_tamper(tmp_path,p9_cpu_builds):
    _,_,_,_,kw=p9_cpu_builds
    bundle=dict(names=['water','lung','bone','lead','seed_agti'],densities=RHO)
    record,arrays=g9.input_record([None]*5,bundle,kw['edges'],kw['cum'],p.LAYERS,
                                  200000,g9.PILOT_SEED)
    record['phase']='pilot'
    schedule=record['schedule'];sid=fake_session(tmp_path,record,schedule)
    result=fake_result(record['sizes'])
    def runner(**kw):return result,np.ones(200000,np.int64)
    s9.drive(tmp_path,runner,record,schedule,sid)
    dose,fine,receipt=j9.load_gpu(tmp_path,bundle,g9.PILOT_SEED,200000)
    expected=1/2000/arrays['mass_g_cm2']
    np.testing.assert_allclose(dose,np.tile(expected,(100,1)))
    assert fine.shape==(100,424) and receipt['complete']['n']==200000
    with pytest.raises(ValueError,match='seed'):j9.load_gpu(tmp_path,bundle,g9.GPU_SEED,200000)
    path=s9.call_path(tmp_path,0);path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='hash'):j9.load_gpu(tmp_path,bundle,g9.PILOT_SEED,200000)


@pytest.mark.parametrize('atomic_status',['pass','fail'])
def test_check_data_uses_P8a_thresholds_and_resolved_status(tmp_path,monkeypatch,atomic_status):
    from prototypes.mv_electron import check_p8a as ck, fetch_materials_p8a as fetch
    folder=tmp_path/'cache/seed_agti';folder.mkdir(parents=True)
    definition=p.seed_definition();definition['I_MeV']=.0002337
    d9.mat.write(folder/'definition.json',definition)
    d9.mat.write(folder/'elastic.json',{f'z{z}_q-1.npz':dict(Z=z) for z in (22,47)})
    d9.mat.write(folder/'replacements.json',{})
    for name in ('stopping','photon','shells','gs'):
        np.savez(folder/(name+'.npz'),E=[.01,10.],photo=[1.,1.],edges=[])
    monkeypatch.setattr(d9,'verified',lambda *args:None)
    monkeypatch.setattr(d9.mat,'load_reference',lambda r,m:r)
    monkeypatch.setattr(d9.mat,'MixtureElsepa',lambda d,t:(d,t))
    monkeypatch.setattr(fetch,'check_edge_jumps',lambda *args:None)
    monkeypatch.setattr(ck,'interpolation_checks',lambda *args:dict(collision=dict(status='pass')))
    monkeypatch.setattr(ck,'shell_checks',lambda *args:dict(status='pass',binding=[dict(status='report-only')]))
    monkeypatch.setattr(ck,'atomic_checks',lambda *args:dict(status=atomic_status,rows=[dict(status='not-evaluated',kind='direct-timeout')]))
    seen=[]
    def mixture(d,t):
        seen.append((d,t));assert set(t)=={22,47};return dict(status='pass')
    monkeypatch.setattr(ck,'mixture_checks',mixture)
    monkeypatch.setattr(ck,'gs_checks',lambda *args:dict(status='pass'))
    if atomic_status=='pass':assert d9.check_data(tmp_path/'cache',tmp_path/'out')['status']=='pass'
    else:
        with pytest.raises(ValueError,match='data gate failed'):d9.check_data(tmp_path/'cache',tmp_path/'out')
    assert len(seen)==1 and seen[0][0]['material']=='seed_agti'


def test_drive_limit_on_final_call_seals_completion(tmp_path):
    schedule=small_schedule(1);identity={'input':'synthetic'}
    sid=fake_session(tmp_path,identity,schedule)
    s9.drive(tmp_path,lambda **kw:(fake_result(),np.ones(100)),identity,schedule,sid,limit=1)
    assert s9.verify(tmp_path,identity,schedule)['status']=='pass'
