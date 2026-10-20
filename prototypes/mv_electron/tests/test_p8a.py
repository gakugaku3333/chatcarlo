"""P8a offline contracts and fault injection. No network, HOME cache or EGS run.

Use synthetic, analytic tables and read-only historical water inputs. Physical
S1--S7 acceptance is deliberately not asserted by these unit tests.
"""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from prototypes.mv_electron import materials_p8a as m
from prototypes.mv_electron import fetch_materials_p8a as fetch
from prototypes.mv_electron import pegs_p8a as pegs
from prototypes.mv_electron import check_p8a as check
from prototypes.mv_electron import estimate_p8a as estimate
from prototypes.mv_electron import generate_elsepa_p8a as generate


@pytest.fixture(autouse=True)
def no_external_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('offline P8a tests must not execute external programs')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(fetch, 'request', forbidden)
    # No tests may touch the actual cache, including reads.
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.expanduser().resolve().is_relative_to(Path.home()/'.cache/chatcarlo'):
            raise AssertionError('offline P8a test touched HOME cache')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)


@pytest.fixture
def water():
    raw = (m.ROOT/'docs/validation/mv/p1/data/composition.html').read_text()
    return fetch.parse_composition(raw, 'water', 'Water, Liquid')


@pytest.fixture
def atomic():
    es = np.array([.01, .02, .1, 1., 2., 10.])
    mu = np.linspace(0., 1., 17)
    return {z: dict(Z=z, charge=-1, IHEF=0, energy_MeV=es,
                   mu=mu, dcs=es[:, None]**-.1*np.ones((len(es), len(mu)))*(z+1)*1e-24,
                   table_xs=np.ones((len(es), 3)), error=np.zeros((len(es), len(mu)))) for z in (1, 8, 82)}


@pytest.fixture
def synthetic_tables():
    t = np.geomspace(.01, 20., 21)
    mid = np.sqrt(t[:-1]*t[1:])
    stopping = dict(T=t, collision=t**-.3, radiative=.01*t**1.1,
                    total=t**-.3+.01*t**1.1, delta=np.zeros_like(t),
                    check_T=mid, check_collision=mid**-.3, check_radiative=.01*mid**1.1)
    th, edge = np.array([1., 2.]), .04
    e = np.unique(np.r_[np.geomspace(.001, 20., 150), edge*(1-1e-6), edge, th, th*1.00001])
    photo_e = e.copy()
    photo_e[np.searchsorted(e, edge*(1-1e-6))] = edge
    mid = np.sqrt(e[:-1]*e[1:])
    def photo(v):
        return np.where(v >= edge, 5., 1.)*v**-2
    def pair(v, threshold):
        return .05*v**.2*np.maximum(0., 1-threshold/v)**3
    photon = dict(E=e, photo_E=photo_e, photo=photo(e),
                  pair_nuclear=pair(e, th[0]), pair_electron=pair(e, th[1]),
                  thresholds=th, edges=np.array([edge]), check_E=mid,
                  check_photo=photo(mid), check_pair_nuclear=pair(mid, th[0]),
                  check_pair_electron=pair(mid, th[1]),
                  check_total=photo(mid)+pair(mid, th[0])+pair(mid, th[1])+1.)
    return stopping, photon


def test_nist_selection_uses_retrieved_listing(water):
    raw = (m.ROOT/'docs/validation/mv/p1/data/composition.html').read_text()
    selected = fetch.material_list(raw)
    assert set(selected) == set(m.MATERIALS)
    assert selected['water']['name'] == 'Water, Liquid'
    with pytest.raises(ValueError, match='uniquely'):
        fetch.material_list(raw.replace('Lung (ICRP)', 'Missing material'))
    with pytest.raises(ValueError, match='identity'):
        fetch.parse_composition(raw, 'water', 'Lung (ICRP)')
    assert m.constants(water)['NE'] > 0


def test_existing_water_collision_and_consistency(water):
    from prototypes.mv_electron import physics
    c = m.constants(water)
    assert c['NE'] != physics.NE  # separate compatibility and consistency paths
    with np.load(physics.DATA/'estar.npz') as f:
        t = f['T']
        assert np.allclose(m.collision(t, physics.NE, physics.I, f['delta']),
                           [physics.collision(float(v)) for v in t], rtol=2e-15)


@pytest.mark.parametrize('defect', ['sum', 'nan', 'missing', 'negative', 'duplicate'])
def test_invalid_composition_rejected(water, defect):
    d = copy.deepcopy(water)
    if defect == 'sum': d['weights']['1'] *= 2
    if defect == 'nan': d['weights']['1'] = np.nan
    if defect == 'missing': del d['atomic_weights']['8']
    if defect == 'negative': d['weights']['1'] = -1
    if defect == 'duplicate': d['weights']['01'] = d['weights']['1']
    with pytest.raises(ValueError):
        m.constants(d)


@pytest.mark.parametrize('energies', [[1., 1.], [2., 1.], [0., 1.], [1., np.nan]])
def test_energy_faults_rejected(energies):
    with pytest.raises(ValueError): m.energy_grid(energies)


def test_absorption_edges_remain_discontinuous():
    e = np.array([.01, .02, .02, .04])
    v = np.array([4., 1., 10., 2.5])
    assert m.photo_interp(.02, e, v) == pytest.approx(10.)
    assert m.photo_interp(np.nextafter(.02, 0), e, v) == pytest.approx(1.)
    assert m.photo_interp(np.sqrt(.01*.02), e, v) == pytest.approx(2.)
    assert m.photo_interp(np.sqrt(.02*.04), e, v) == pytest.approx(5.)
    with pytest.raises(ValueError): m.photo_interp(.001, e, v)
    with pytest.raises(ValueError): m.photo_interp(.02, [.01, .02, .02, .02], [1., 2., 3., 4.])


def test_pair_threshold_and_endpoint_continuation(synthetic_tables):
    _, p = synthetic_tables
    for k, threshold in zip(('pair_nuclear', 'pair_electron'), p['thresholds']):
        e = np.array([np.nextafter(threshold, 0), threshold, np.nextafter(threshold, np.inf)])
        values = m.pair_interp(e, p['E'], p[k], threshold)
        assert values[0] == values[1] == 0 and values[2] > 0
        bad = p[k].copy(); bad[0] = 1.
        with pytest.raises(ValueError, match='threshold'): m.pair_interp(e, p['E'], bad, threshold)


def test_process_interpolation_at_independent_midpoints(synthetic_tables, water):
    s, p = synthetic_tables
    results = check.interpolation_checks(s, p, m.constants(water)['NE'])
    assert all(row['status'] == 'pass' for row in results.values()), results
    assert results['compton']['max_relative'] <= 1e-12
    s['check_T'][0] = s['T'][0]
    with pytest.raises(ValueError, match='held out'): check.interpolation_checks(s, p, 1.)


def test_interpolation_failure_is_not_pass(synthetic_tables, water):
    s, p = synthetic_tables
    s['check_collision'] *= 1.01
    result = check.interpolation_checks(s, p, m.constants(water)['NE'])
    assert result['collision']['status'] == 'fail'
    assert check.aggregate([r['status'] for r in result.values()]) == 'fail'


def test_mixture_order_density_and_pure_lead(water, atomic):
    a = m.MixtureElsepa(water, atomic)
    reversed_definition = dict(water, weights=dict(reversed(list(water['weights'].items()))),
                               atomic_weights=dict(reversed(list(water['atomic_weights'].items()))))
    b = m.MixtureElsepa(reversed_definition, atomic)
    c = m.MixtureElsepa(dict(water, density_g_cm3=.26), atomic)
    from prototypes.mv_electron.dcs import moments
    from prototypes.mv_electron.gs_backend import distribution
    y = np.linspace(0., 2., 33)
    for t in (.01, .2, 10.):
        assert a.density(t, y).tobytes() == b.density(t, y).tobytes()
        assert np.allclose(c.density(t, y), .26*a.density(t, y), rtol=2e-15)
    ga, total, _ = moments(a, 1.)
    gc, scaled, _ = moments(c, 1.)
    assert scaled/total == pytest.approx(.26)
    assert gc[1]/ga[1] == pytest.approx(.26)
    ya, ca, p0a, *_ = distribution(a, 1., .1, L=16, n=4097)
    yc, cc, p0c, *_ = distribution(c, 1., .1, L=16, n=4097)
    assert np.allclose(ca, cc, atol=1e-12) and p0a == pytest.approx(p0c)
    # Synthetic pure element: exact atom mixture including the Z+1 factor.
    lead = dict(material='lead', density_g_cm3=2., I_MeV=1e-4,
                weights={'82': 1.}, atomic_weights={'82': 200.})
    provider = m.MixtureElsepa(lead, atomic)
    scale = 2*m.NA/200*(83/82)
    assert provider.density(1., y) == pytest.approx(np.full_like(y, 2*np.pi*83e-24*scale))
    with pytest.raises(ValueError, match='range'): provider.density(10.1, y)


def test_legacy_water_elastic_bytes_with_synthetic_atoms(water, atomic, tmp_path):
    from prototypes.mv_electron.elsepa import Elsepa
    for z in (1, 8): np.savez(tmp_path/f'z{z}_q-1.npz', **atomic[z])
    old = Elsepa(tmp_path, expected_ihef=0)
    new = m.MixtureElsepa(water, atomic, legacy_water=True)
    y = np.linspace(0., 2., 33)
    for e in np.geomspace(.01, 10., 21):
        assert old.density(e, y).tobytes() == new.density(e, y).tobytes()
    with pytest.raises(ValueError, match='legacy'): m.MixtureElsepa(dict(water, density_g_cm3=.5), atomic, legacy_water=True)


@pytest.mark.parametrize('defect', ['Z', 'charge', 'IHEF', 'grid', 'dcs', 'shape'])
def test_elastic_identity_and_data_faults(water, atomic, defect):
    a = copy.deepcopy(atomic)
    if defect in ('Z', 'charge', 'IHEF'): a[1][defect] = 3
    if defect == 'grid': a[1]['energy_MeV'][2] = a[1]['energy_MeV'][1]
    if defect == 'dcs': a[1]['dcs'][0, 0] = np.nan
    if defect == 'shape': a[1]['dcs'] = a[1]['dcs'][:, :-1]
    with pytest.raises(ValueError): m.MixtureElsepa(water, a)


def test_pair_parser_matches_existing_water_and_variable_headers(water):
    source = (m.ROOT/'docs/validation/mv/p1/egs5/base_2/pgs5job.pegs5dat').read_text()
    parsed = pegs.pair_parameters(source, 'H2O', water)
    values = np.array([float(v) for line in source.splitlines()[6:16] for v in line.split()])
    assert parsed['dl'].tobytes() == values[:36].reshape(6, 6).tobytes()
    assert parsed['delcm'] == values[36] and parsed['bpar'][0] == values[38] and parsed['delpos'][0] == values[39]
    prefixed = 'unrelated header\n'+source
    assert np.array_equal(pegs.pair_parameters(prefixed, 'H2O', water)['dl'], parsed['dl'])
    with pytest.raises(ValueError, match='position'): pegs.pair_parameters(prefixed, 'H2O', water, expected_start=0)
    with pytest.raises(ValueError, match='material'): pegs.pair_parameters(source, 'P8A_LEAD', water)
    with pytest.raises(ValueError, match='density'): pegs.pair_parameters(source, 'H2O', dict(water, density_g_cm3=.5))
    with pytest.raises(ValueError, match='shape'): pegs.pair_parameters(source.replace(source.splitlines()[7], '1 2'), 'H2O', water)


def test_pair_provenance_shape_material_and_offset_injection(water, tmp_path):
    source = tmp_path/'pgs.pegs5dat'
    source.write_text((m.ROOT/'docs/validation/mv/p1/egs5/base_2/pgs5job.pegs5dat').read_text().replace('MEDIUM=H2O ', 'MEDIUM=P8A_WATER '))
    p = pegs.pair_parameters(source.read_text(), 'P8A_WATER', water)
    p.update(material=np.array('water'), source_sha256=np.array(m.sha(source)))
    dest = tmp_path/'pair.npz'; np.savez(dest, **p)
    ref = m.reference(dest, str(source), 'DECK 36+7', 'dimensionless', 'water', source_sha256=m.sha(source), medium_line=0, dl_line=int(p['dl_line']))
    assert check.verify_pair(ref, 'water', water)['status'] == 'pass'
    for key, val in [('dl', np.ones((5, 6))), ('material', np.array('lead')), ('dl_line', np.array(999))]:
        changed = dict(p, **{key: val}); np.savez(dest, **changed)
        bad = dict(ref, sha256=m.sha(dest), shape={k: list(np.asarray(v).shape) for k, v in changed.items()})
        with pytest.raises(ValueError): check.verify_pair(bad, 'water', water)
    np.savez(dest, **p)
    ref = dict(ref, sha256=m.sha(dest))
    with pytest.raises(ValueError, match='position'): check.verify_pair(dict(ref, dl_line=999), 'water', water)
    source.write_text(source.read_text()+'tampered\n')
    with pytest.raises(ValueError, match='hash'): check.verify_pair(ref, 'water', water)


def test_manifest_hash_material_shape_missing_and_resume(tmp_path, water, synthetic_tables):
    definition = tmp_path/'definition.json'; m.write(definition, water)
    ref = m.reference(definition, 'NIST fixture', 'parser', 'MeV; g/cm3', 'water')
    assert m.load_reference(ref, 'water') == water
    with pytest.raises(ValueError, match='material'): m.load_reference(ref, 'lung')
    definition.write_text('{}')
    with pytest.raises(ValueError, match='hash'): m.load_reference(ref, 'water')
    arr = tmp_path/'table.npz'; np.savez(arr, x=np.array([1., np.nan]))
    ref = m.reference(arr, 'synthetic', 'fixture', '1', 'water')
    with pytest.raises(ValueError, match='nonfinite'): m.load_reference(ref, 'water')
    np.savez(arr, x=np.array([1., 2.]))
    ref = m.reference(arr, 'synthetic', 'fixture', '1', 'water')
    with pytest.raises(ValueError, match='shape'): m.load_reference(dict(ref, shape={}), 'water')
    manifest = tmp_path/'manifest.json'; m.write(manifest, {'materials': {}})
    with pytest.raises(ValueError, match='missing material'): m.load_manifest(manifest)
    m.write(manifest, {'materials': {k: {} for k in m.MATERIALS}})
    with pytest.raises(ValueError, match='missing material tables'): m.load_manifest(manifest)
    identity = tmp_path/'identity.json'
    m.resume(identity, {'IHEF': 0, 'Z': 82}); m.resume(identity, {'IHEF': 0, 'Z': 82})
    with pytest.raises(ValueError, match='conditions'): m.resume(identity, {'IHEF': 1, 'Z': 82})


@pytest.mark.parametrize('states', [[], ['skip'], ['pass', 'skip'], ['not-evaluated'], ['report-only'], ['pass', 'insufficient-statistics']])
def test_skips_and_unperformed_checks_are_never_passes(states):
    assert check.aggregate(states) == 'not-evaluated'


def test_valid_and_failed_aggregation():
    assert check.aggregate(['pass', 'pass']) == 'pass'
    assert check.aggregate(['pass', 'fail']) == 'fail'


def test_acquisition_receipt_hash_and_request_change(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, 'request', lambda *args: b'kept raw response')
    p = tmp_path/'response.html'
    assert fetch.acquire(p, 'https://example.invalid', {'matno': 'fixture'}) == 'kept raw response'
    assert fetch.acquire(p, 'https://example.invalid', {'matno': 'fixture'}) == 'kept raw response'
    with pytest.raises(ValueError, match='conditions'): fetch.acquire(p, 'https://example.invalid', {'matno': 'other'})
    p.write_text('corrupt')
    with pytest.raises(ValueError, match='hash'): fetch.acquire(p, 'https://example.invalid', {'matno': 'fixture'})


def test_estar_parsers_keep_range_separate():
    raw = '<pre>1.000E-02 2.000E+00 1.000E-01 2.100E+00 0.000E+00<br>2.000E-02 1.000E+00 2.000E-01 1.200E+00 1.000E-01</pre>'
    result = fetch.parse_stopping(raw, [.01, .02])
    assert np.array_equal(result['delta'], [0., .1])
    with pytest.raises(ValueError, match='CSDA'): fetch.parse_range(raw)
    ranges = '<pre>CSDA RANGE g/cm2\n.01 2 .1 2.1 .001 0 0\n.02 1 .2 1.2 .004 0 0</pre>'
    assert np.array_equal(fetch.parse_range(ranges)['csda_g_cm2'], [.001, .004])
    with pytest.raises(ValueError): fetch.parse_stopping(raw, [.01, .03])
    with pytest.raises(ValueError): fetch.parse_range(ranges.replace('.004', 'nan'))


def test_pegs_input_composition_cutoffs_and_direct_units(water):
    text = pegs.pegs_input([water])
    assert 'P8A_WATER' in text
    assert float(text.split('RHOZ=')[1].split(',')[0]) == water['weights']['1']
    assert 'IUNRST=0' in text and 'EPSTFL=0' in text and 'AE=0.521,AP=0.01' in text
    assert 'INCOH=0' in text and 'IRAYL=0' in text
    code = pegs.diagnostic_source()
    assert 'spione(en,en)/(rlc*rho)' in code
    assert 'spionp(en,en)/(rlc*rho)' in code
    assert 'brmstm(en,en)/(rlc*rho)' in code
    assert 'call shower' not in code.lower()
    assert 'amoltm(en)' in code and 'bhabtm(en)' in code
    assert all(len(line) <= 72 for line in code.splitlines() if not line.startswith('!'))


def test_claude_s1_s2_fixes_are_preserved(water, monkeypatch):
    row = '<pre>1.000E-02 2.000E+00 1.000E-01 2.100E+00 0.000E+00</pre>'
    assert fetch.parse_stopping(row, [.01])['T'].tolist() == [.01]
    import xraylib
    monkeypatch.setattr(xraylib, 'AtomicNumberToSymbol', lambda z: 'Na')
    assert 'NA NA ' in pegs.pegs_input([water])
    code = pegs.diagnostic_source()
    assert "open(22,file='data/aprime.data',status='old')" in code
    assert code.index('read(43,*) n,rho') < code.index('call spinit')
    assert code.index('call spinit') < code.index("write(42,*) 'META'")
    assert code.index('stop 33') < code.index('spione(en,en)')


@pytest.fixture
def diagnostic_state(water):
    # Arbitrary synthetic state; zero DELTA0 is valid, not a reset by itself.
    return (f'META .511 .521 .01 10 {water["density_g_cm3"]} 100\n'
            'DENSITY .1 3 .2 2 4 0\n'
            f'SPION 1 -10 {2*np.log(10):.17g}\n')


def test_diagnostic_state_accepts_general_formula_zero_delta0(water, diagnostic_state):
    pegs.validate_diagnostic_state(diagnostic_state, water)
    pegs.validate_diagnostic_state(diagnostic_state.replace('.2 2 4', '0 2 4'), water)


@pytest.mark.parametrize('before,after', [
    ('DENSITY .1 3 .2 2 4 0', 'DENSITY 0 0 0 0 0 0'),
    ('DENSITY .1', 'DENSITY 0'),
    ('3 .2 2 4', '0 .2 2 4'),
    ('.2 2 4', '.2 .2 4'),
    ('2 4 0', '2 0 0'),
    ('2 4 0', '2 4 -1'),
    ('1.0 100', '1.0 0'),
    ('SPION 1', 'SPION 0'),
    ('SPION 1 -10', 'SPION 1 nan'),
    ('DENSITY', 'MISSING'),
])
def test_diagnostic_reset_state_is_rejected(water, diagnostic_state, before, after):
    assert before in diagnostic_state
    with pytest.raises(ValueError, match='diagnostic state|nonfinite'):
        pegs.validate_diagnostic_state(diagnostic_state.replace(before, after), water)


def test_run_diagnostic_rejects_zero_coefficients_before_sealing(water, diagnostic_state, tmp_path, monkeypatch):
    # Mock compilation and execution only; the autouse guard still forbids any
    # real PEGS program. Exercise both new and previously sealed output paths.
    monkeypatch.setattr(pegs, 'EGS', tmp_path/'source')
    bad = diagnostic_state.replace('DENSITY .1 3 .2 2 4 0', 'DENSITY 0 0 0 0 0 0')
    bad += 'STOP .01 1 1 1 1 1 1 1 1 1 1\nSTOP 1 1 1 1 1 1 1 1 1 1 1\nDONE\n'
    work = tmp_path/'work'
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        if command[0] == './egs5job.exe':
            (work/'direct.txt').write_text(bad)
        else:
            (work/'egs5job.exe').write_text('synthetic executable')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(subprocess, 'run', fake)
    with pytest.raises(ValueError, match='Sternheimer'):
        pegs.run_diagnostic(water, [.01, 1.], work)
    assert len(calls) == 2 and not (work/'complete.json').exists()
    # A hash receipt cannot make a reset state acceptable on the cached path.
    m.write(work/'complete.json', {'direct.txt': m.sha(work/'direct.txt')})
    with pytest.raises(ValueError, match='Sternheimer'):
        pegs.run_diagnostic(water, [.01, 1.], work)
    assert len(calls) == 2


def test_shell_selection_polynomials_and_bindings():
    shells = dict(Z=np.array([82]), NJ=np.array([1.]),
                  binding_MeV=np.array([[.08, .015, .014, .013, .003]]),
                  pm=np.zeros((1, 5, 4)))
    shells['pm'][..., 0] = np.log([1., 2., 3., 4., 5.])
    p, local = pegs.shell_probabilities(.1, shells)
    assert p.sum()+local == pytest.approx(1.)
    assert p[0] == pytest.approx(np.array([1, 2, 3, 4, 5])/15)
    p, local = pegs.shell_probabilities(.01, shells)
    assert p[0, 4] == 1 and p.sum() == 1
    p, local = pegs.shell_probabilities(.001, shells)
    assert local == 1 and p.sum() == 0
    with pytest.raises(ValueError, match='shape'): pegs.shell_probabilities(.1, dict(shells, pm=np.zeros((1, 4, 4))))
    with pytest.raises(ValueError, match='binding'): pegs.shell_probabilities(.1, dict(shells, binding_MeV=-shells['binding_MeV']))


def test_shell_material_mixup_is_rejected(water, synthetic_tables):
    _, photon = synthetic_tables
    shells = dict(Z=np.array([82]), NJ=np.array([1.]), binding_MeV=np.zeros((1, 5)))
    with pytest.raises(ValueError, match='material composition'): check.shell_checks(shells, photon, water)


def test_fluorescence_averages_preserve_bin_covariance_and_fixed_dref():
    x = np.linspace(-1., 1., 100)
    off = np.ones((100, 55))
    off[:, 45:50] += x[:, None]*.0001
    on = off.copy(); on[:, 45:50] += .001
    result = estimate.fluorescence(off, on, 1, 2)
    assert result['Dref'] == 1
    assert result['difference'] == pytest.approx([.001, 0.])
    expected = np.sqrt(2)*x.std(ddof=1)*.0001/10
    assert result['sem'][0] == pytest.approx(expected)
    assert result['decisions'] == ['continue-without-fluorescence']*2
    with pytest.raises(ValueError, match='independent'): estimate.fluorescence(off, on, 1, 1)
    for difference, decision in ((.005, 'consult-user'), (.003, 'undetermined-consult-user')):
        changed = off.copy(); changed[:, 45:50] += difference
        assert estimate.fluorescence(off, changed, 1, 2)['decisions'][0] == decision
    noisy = on.copy(); noisy[:, 45:50] += x[:, None]*.1
    assert estimate.fluorescence(off, noisy, 1, 2)['decisions'][0] == 'insufficient-statistics'


def test_pilot_scaling_cap_and_distinct_streams():
    assert len(set(estimate.SEEDS.values())) == len(estimate.SEEDS)
    assert estimate.production_size([.002, .001])['histories_each'] == 4_000_000
    assert estimate.production_size([.1, .1])['status'] == 'insufficient-statistics'
    assert estimate.production_size([.1, .1])['histories_each'] == 100_000_000
    with pytest.raises(ValueError): estimate.production_size([np.nan, .1])


@pytest.fixture
def egs_output():
    n, seed, fluor = 1000, 1, 0
    lines = [f'CONFIG {n} {seed} {fluor}', 'LEDGER 100000 11000 89000 0 0 0', 'CPU 1']
    for r, med in enumerate((0, 1, 2, 0), 1):
        lines.append(f'REGION {r} {med} 0 0 0 0 0 0 .521 .01')
    for i, rho in enumerate((1, 2), 1):
        lines.append(f'MEDIUM {i} {rho} .521 .01 0 0 0 1 0')
    lines += [f'DOSE {b} {j} 1' for b in range(1, 101) for j in range(1, 56)]
    return '\n'.join(lines+['DONE'])


def test_egs_output_health_and_mass_normalization(egs_output):
    dose, report = estimate.parse_egs(egs_output, 1000, 1, 0, [1, 2])
    assert dose.shape == (100, 55) and report['status'] == 'pass'


@pytest.mark.parametrize('before,after', [
    ('DONE', ''), ('DOSE 100 55 1', 'DOSE 1 1 1'),
    ('DOSE 100 55 1', 'DOSE 100 55 nan'),
    ('LEDGER 100000 11000 89000', 'LEDGER 100000 11000 88000'),
    ('REGION 3 2 0', 'REGION 3 2 1'),
    ('MEDIUM 2 2', 'MEDIUM 2 3'),
    ('CONFIG 1000 1 0', 'CONFIG 1000 2 0'),
    ('DOSE 100 55 1', 'DOSE 100 55 2'),
])
def test_egs_fault_injection(egs_output, before, after):
    with pytest.raises(ValueError): estimate.parse_egs(egs_output.replace(before, after), 1000, 1, 0, [1, 2])


def test_water_control_statistical_gate():
    a = np.ones((100, 55))
    assert estimate.water_control(a, a)['status'] == 'pass'
    assert estimate.water_control(a, a+.001)['status'] == 'fail'


def test_fortran_geometry_and_scoring_contract():
    code = (m.ROOT/'prototypes/mv_electron/p8a_egs.f').read_text()
    assert 'nmed=2' in code and 'nreg=4' in code
    assert 'med(3)=2' in code and 'med(4)=0' in code
    assert 'right=10.5d0' in code and 'left=10.d0' in code
    assert 'irnew=ir(np)+1' in code and 'irnew=ir(np)-1' in code
    assert 'width=.2d0' in code and 'width=.1d0' in code
    assert 'width*rho' in code and 'iedgfl(i)=fluor' in code and 'iauger(i)=fluor' in code
    assert not any(len(line) > 72 for line in code.splitlines() if not line.startswith('!'))


def test_protected_snapshot_detects_change(tmp_path, monkeypatch):
    p = tmp_path/'frozen.py'; p.write_text('frozen')
    snapshot = tmp_path/'protected.json'; m.write(snapshot, {'frozen.py': m.sha(p)})
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    assert m.verify_protected(snapshot) == 1
    p.write_text('changed')
    with pytest.raises(ValueError, match='protected'): m.verify_protected(snapshot)


def test_two_pass_arrays_check_dtype_and_values():
    a = dict(x=np.array([1., 2.]))
    assert generate.arrays_identical(a, copy.deepcopy(a))
    assert not generate.arrays_identical(a, dict(x=np.array([1., 2.], dtype=np.float32)))
    assert not generate.arrays_identical(a, dict(x=np.array([1., 3.])))


def test_build_manifest_and_reject_wrong_table_material(tmp_path, water, synthetic_tables, monkeypatch):
    out, cache = tmp_path/'out', tmp_path/'cache'
    monkeypatch.setattr(m, 'verify_protected', lambda: 0)
    m.write(out/'lung_state.json', {'status': 'pass'})
    for material in m.MATERIALS:
        folder = cache/material
        d = dict(water, material=material, density_g_cm3=.26 if material == 'lung' else 1.)
        m.write(folder/'definition.json', d)
        stopping, photon = copy.deepcopy(synthetic_tables)
        for data in (stopping, photon):
            data.update(material=np.array(material), composition_sha256=np.array(m.composition_id(d)))
        np.savez(folder/'stopping.npz', **stopping)
        np.savez(folder/'photon.npz', **photon)
        np.savez(folder/'range.npz', T=np.array([.01, 1.]), csda_g_cm2=np.array([.001, .1]),
                 material=np.array(material), composition_sha256=np.array(m.composition_id(d)))
    built = m.build(out, cache)
    assert set(built['materials']) == set(m.MATERIALS)
    assert m.load_manifest(out/'manifest.json', complete=False) == built
    # Rehashing a wrong material's npz cannot conceal its embedded identity.
    row = built['materials']['bone']
    row['stopping'] = dict(built['materials']['lung']['stopping'], material='bone')
    m.write(out/'manifest.json', built)
    with pytest.raises(ValueError, match='material identity'): m.load_manifest(out/'manifest.json', complete=False)


def test_mixture_check_and_photon_api(water, atomic, synthetic_tables):
    result = check.mixture_checks(water, atomic)
    assert result['status'] == 'pass'
    _, table = synthetic_tables
    e = np.array([.01, .1, 1., 10.])
    c = m.photon_coefficients(e, water, table)
    assert c.shape == (4, 4) and np.all(c >= 0)
    assert np.array_equal(c[:, 0], m.kn_total(e, m.constants(water)['NE']))


def test_sensitivity_runner_build_resume_and_tamper(tmp_path, monkeypatch, water, egs_output):
    monkeypatch.setattr(estimate, 'external_sources', lambda: {'fixture': 'hash'})
    calls = []
    def fake(command, **kwargs):
        work = Path(kwargs['cwd'])
        calls.append(command)
        if command[0] == './egs5job.exe':
            (work/'egs5job.out').write_text(egs_output)
            (work/'pgs5job.pegs5dat').write_text('synthetic pegs data')
            (work/'pgs5job.pegs5lst').write_text('synthetic listing')
        else:
            (work/'egs5job.exe').write_text('synthetic binary')
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(subprocess, 'run', fake)
    definitions = dict(water=water, lead=dict(water, material='lead', density_g_cm3=2.))
    args = (tmp_path/'egs', 1000, 1, 0, definitions, np.array([0., 1.]), np.array([1.]))
    dose, _ = estimate.run_egs(*args)
    assert dose.shape == (100, 55) and len(calls) == 2
    estimate.run_egs(*args)
    assert len(calls) == 2
    with pytest.raises(ValueError, match='conditions'):
        estimate.run_egs(args[0], 1100, *args[2:])
    (args[0]/'egs5job.out').write_text('tampered')
    with pytest.raises(ValueError, match='sealed'): estimate.run_egs(*args)


def test_approximation_indicators_are_report_only(synthetic_tables):
    _, table = synthetic_tables
    shells = dict(Z=np.array([82]), NJ=np.array([1.]),
                  binding_MeV=np.array([[.08, .015, .014, .013, .003]]), pm=np.ones((1, 5, 4))*.01)
    result = estimate.spectrum_indicators(np.array([0., 1., 10.]), np.array([.5, .5]), table, shells, 1e24)
    assert result['status'] == 'report-only' and not result['dose_upper_bound']
    assert 0 < result['pair_interaction_fraction'] < 1
    assert 0 <= result['binding_energy_fraction'] <= result['photoelectric_energy_fraction']
    assert result['sub_table_probability'] > 0
    assert 'P8b' in estimate.BREMS_REVIEW and 'SLAC-R-730' in estimate.BREMS_REVIEW


@pytest.mark.parametrize('module', [m, fetch, generate, pegs, check, estimate])
def test_cli_help_has_no_side_effects(module, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, 'argv', [module.__name__, '--help'])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 0
    assert 'usage:' in capsys.readouterr().out


@pytest.fixture
def lung_a_inputs():
    # Synthetic tables: PEGS has a large absolute offset from ESTAR, while its
    # density ratio remains small below .2 MeV. This must pass appendix 3.
    es = np.array([.01, .1, .2, 1., 5.])
    estar = dict(T=es, collision=np.array([10., 3., 2., 1.5, 2.]),
                 radiative=np.array([.01, .02, .03, .04, .05]),
                 delta=np.array([0., 0., .01, 1., 3.]))
    standard = dict(T=es.copy(), collision=estar['collision']*1.02,
                    delta=np.array([0., 0., .02, 1.2, 3.2]))
    selected = dict(T=es.copy(), collision=standard['collision']*np.array([1., 1.0005, 1.0009, 1.02, 1.05]),
                    delta=np.array([0., 0., 0., .8, 2.8]))
    return estar, standard, selected


def test_lung_a_ratio_and_additive_delta(lung_a_inputs):
    estar, standard, selected = lung_a_inputs
    result = fetch.lung_stopping_a(estar, standard, selected)
    np.testing.assert_array_equal(result['collision'], estar['collision']*(selected['collision']/standard['collision']))
    np.testing.assert_array_equal(result['delta'], estar['delta']+(selected['delta']-standard['delta']))
    np.testing.assert_array_equal(result['radiative'], estar['radiative'])
    np.testing.assert_array_equal(result['standard_ratio'], np.ones(len(estar['T'])))
    assert result['delta'][2] < 0  # Keep the specified additive correction.
    report = fetch.check_lung_stopping_a(estar, result)
    assert report['status'] == 'pass'
    assert .2 in report['low_energy_T_MeV'] and 1. not in report['low_energy_T_MeV']
    # A 2% PEGS/ESTAR offset must not reintroduce the old .5% gate.
    assert np.max(abs(standard['collision']/estar['collision']-1)) > .005


@pytest.mark.parametrize('ratio', [np.nextafter(1., 2.), .99])
def test_lung_a_rejects_nonunit_standard_ratio(lung_a_inputs, ratio):
    estar, standard, selected = lung_a_inputs
    result = fetch.lung_stopping_a(estar, standard, selected)
    result['standard_ratio'][-1] = ratio  # Also check above .2 MeV, exactly.
    check = fetch.check_lung_stopping_a(estar, result)
    assert check['status'] == 'fail' and not check['standard_ratio_exactly_one']


@pytest.mark.parametrize('factor', [1.0011, .9989])
def test_lung_a_rejects_low_energy_error_over_point_one_percent(lung_a_inputs, factor):
    estar, standard, selected = lung_a_inputs
    result = fetch.lung_stopping_a(estar, standard, selected)
    result['collision'][2] = estar['collision'][2]*factor  # Inclusive .2 MeV.
    check = fetch.check_lung_stopping_a(estar, result)
    assert check['status'] == 'fail' and check['low_energy_max_relative'] > .001


def test_lung_a_rejects_mismatched_pegs_grid(lung_a_inputs):
    estar, standard, selected = lung_a_inputs
    selected['T'][1] *= 1.01
    with pytest.raises(ValueError, match='energy grids'):
        fetch.lung_stopping_a(estar, standard, selected)


@pytest.mark.parametrize('fault', [None, 'node_ratio', 'check_ratio', 'node_low_energy', 'check_low_energy'])
def test_lung_state_a_persistence_and_failure(tmp_path, water, monkeypatch, fault):
    cache, out = tmp_path/'cache', tmp_path/'out'
    folder = cache/'lung'
    definition = dict(water, material='lung', density_g_cm3=1.05, standard_density_g_cm3=1.05)
    m.write(folder/'standard_definition.json', definition)
    t, mid = np.array([.01, .2, 1., 5.]), np.array([.02, .15, .5, 2.])
    old = dict(T=t, check_T=mid, collision=t**-.3, check_collision=mid**-.3,
               radiative=.01*t, check_radiative=.01*mid, delta=np.log1p(t), check_delta=np.log1p(mid))
    old.update(total=old['collision']+old['radiative'], check_total=old['check_collision']+old['check_radiative'])
    np.savez(folder/'standard_stopping.npz', **old)
    m.write(folder/'fetch_outputs.json', {name: m.sha(folder/name) for name in ('standard_definition.json', 'standard_stopping.npz')})
    m.write(folder/'definition.json', definition)
    np.savez(folder/'stopping.npz', **old)
    original = {name: m.sha(folder/name) for name in ('definition.json', 'stopping.npz')}
    calls = []
    def diagnostic(d, energies, work):
        calls.append((d['density_g_cm3'], energies.copy()))
        prefix = '' if np.array_equal(energies, t) else 'check_'
        ratio = np.where(energies <= .2, 1.0004, 1.05)
        if fault == ('node_low_energy' if prefix == '' else 'check_low_energy'):
            ratio[0] = 1.002
        return dict(T=energies.copy(), collision=old[prefix+'collision']*1.02*(ratio if d['density_g_cm3'] == .26 else 1.),
                    delta=old[prefix+'delta']+(np.zeros(len(energies)) if d['density_g_cm3'] == 1.05 else -.01))
    monkeypatch.setattr(pegs, 'density_stopping', diagnostic)
    builder = fetch.lung_stopping_a
    def inject(estar, standard, selected):
        result = builder(estar, standard, selected)
        tag = 'node_ratio' if np.array_equal(estar['T'], t) else 'check_ratio'
        if fault == tag:
            result['standard_ratio'][-1] = np.nextafter(1., 2.)
        return result
    monkeypatch.setattr(fetch, 'lung_stopping_a', inject)
    if fault:
        with pytest.raises(ValueError, match='S2 A checks failed'):
            fetch.lung_state(cache, out)
        assert {name: m.sha(folder/name) for name in original} == original
        assert not (folder/'lung_outputs.json').exists()
    else:
        fetch.lung_state(cache, out)
        with np.load(folder/'stopping.npz') as result:
            for prefix in ('', 'check_'):
                es = old[prefix+'T']
                np.testing.assert_allclose(result[prefix+'collision'], old[prefix+'collision']*np.where(es <= .2, 1.0004, 1.05))
                np.testing.assert_allclose(result[prefix+'delta'], old[prefix+'delta']-.01)
                np.testing.assert_array_equal(result[prefix+'radiative'], old[prefix+'radiative'])
                np.testing.assert_array_equal(result[prefix+'total'], result[prefix+'collision']+result[prefix+'radiative'])
        assert json.loads((folder/'definition.json').read_text())['density_g_cm3'] == .26
        assert (folder/'lung_outputs.json').exists()
    report = json.loads((out/'lung_state.json').read_text())
    assert report['status'] == ('fail' if fault else 'pass')
    assert report['standard_collision_max_relative'] == pytest.approx(.02)
    assert report['check_standard_collision_max_relative'] == pytest.approx(.02)
    assert report['standard_pegs_estar_comparison'].startswith('report-only')
    assert len(calls) == 4  # Both densities, both grids; no external program.


@pytest.mark.parametrize('method', ['estar_pegs_density_ratio_A', None])
def test_signed_lung_delta_is_only_an_a_record(tmp_path, water, method):
    manifest = dict(materials={})
    for material in m.MATERIALS:
        d = dict(water, material=material)
        if material == 'lung' and method is not None:
            d['stopping_method'] = method
        definition = tmp_path/(material+'.json')
        m.write(definition, d)
        table = tmp_path/(material+'.npz')
        np.savez(table, T=np.array([.01, 1.]), collision=np.ones(2), radiative=np.ones(2),
                 total=np.full(2, 2.), delta=np.full(2, -.01 if material == 'lung' else 0.),
                 material=np.array(material), composition_sha256=np.array(m.composition_id(d)))
        manifest['materials'][material] = dict(
            definition=m.reference(definition, 'synthetic', 'fixture', '1', material),
            stopping=m.reference(table, 'synthetic', 'fixture', '1', material))
    path = tmp_path/'manifest.json'
    m.write(path, manifest)
    if method:
        m.load_manifest(path, complete=False)
    else:
        with pytest.raises(ValueError, match='invalid stopping'):
            m.load_manifest(path, complete=False)


@pytest.fixture
def elsepa_job(tmp_path):
    source = tmp_path/'source'; (source/'database').mkdir(parents=True)
    exe = tmp_path/'elscata'; exe.write_bytes(b'synthetic executable')
    return source, exe, tmp_path/'case'


def saved_elsepa_job(job, z=17, charge=1, energy=1.):
    source, exe, work = job
    identity = dict(Z=z, charge=charge, energy_MeV=energy, IHEF=0,
                    input=generate.input_text(z, charge, energy), executable_sha256=m.sha(exe))
    m.write(work/'identity.json', identity)
    (work/'input.in').write_text(identity['input'])
    return identity


def synthetic_case_row(z=17, charge=1, energy=1.):
    header = (f'Z = {z}\nProjectile: {"electron" if charge == -1 else "positron"}\n'
              f'Kinetic energy = {energy*1e6:.12E}\nNuclear model: Fermi distribution\n'
              f'./database/z_{z:03}.den\nNo absorption potential\nNo correlation-polarization potential\n')
    if charge == -1: header += 'Furness-McCarthy exchange potential\n'
    return dict(header=header, mu=np.linspace(0, 1, 5), theta_deg=np.linspace(0, 180, 5),
                dcs=np.ones(5), error=np.zeros(5), series_xs=np.ones(3), table_xs=np.ones(3),
                warnings=np.array([], dtype=str), notices=np.array([], dtype=str))


@pytest.mark.parametrize('kind', ['node', 'direct'])
def test_elsepa_timeout_retains_and_seals_raw_attempt(elsepa_job, monkeypatch, kind):
    source, exe, work = elsepa_job
    calls = []
    def timeout(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 600
        (Path(kwargs['cwd'])/'dcs_failed.dat').write_bytes(b'')
        raise subprocess.TimeoutExpired(command, 600, output=b'partial output', stderr=b'partial error')
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(generate.ElsepaCaseTimeout) as error:
        generate.case(source, exe, work, 17, 1, 1., kind=kind)
    receipt = error.value.record
    archive = work.with_name(work.name+'.timed-out')
    assert receipt['status'] == 'not-evaluated' and receipt['energy_MeV'] == 1.
    assert (archive/'stdout.log').read_bytes() == b'partial output'
    assert (archive/'dcs_failed.dat').read_bytes() == b''
    assert not (work/'complete.json').exists()
    with pytest.raises(generate.ElsepaCaseTimeout): generate.case(source, exe, work, 17, 1, 1., kind=kind)
    assert len(calls) == 1  # No automatic retry on resume.
    (archive/'stdout.log').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='archived output changed'):
        generate.case(source, exe, work, 17, 1, 1., kind=kind)


def test_approved_interrupted_direct_case_is_archived_without_execution(elsepa_job):
    source, exe, work = elsepa_job
    t = float.fromhex('0x1.01de978eb7400p+3')
    saved_elsepa_job(elsepa_job, energy=t)
    (work/'dcs_8p058e06.dat').write_bytes(b'')
    (work/'dpwa.dat').write_bytes(b'partial DPWA output')
    with pytest.raises(generate.ElsepaCaseTimeout) as error:
        generate.case(source, exe, work, 17, 1, t, kind='direct')
    assert error.value.record['origin'].startswith('appendix-4')
    archive = work.with_name(work.name+'.timed-out')
    assert (archive/'dpwa.dat').read_bytes() == b'partial DPWA output'
    # Recovery also works if interrupted after archive rename but before marker.
    (work/'timeout.json').unlink()
    with pytest.raises(generate.ElsepaCaseTimeout): generate.case(source, exe, work, 17, 1, t, kind='direct')
    assert (work/'timeout.json').exists()


def test_unknown_interrupted_case_is_not_silently_omitted(elsepa_job):
    source, exe, work = elsepa_job
    saved_elsepa_job(elsepa_job)
    (work/'dcs_zero.dat').write_bytes(b'')
    with pytest.raises(RuntimeError, match='unfinished'):
        generate.case(source, exe, work, 17, 1, 1., kind='direct')
    assert not (work/'timeout.json').exists()


def test_completed_elsepa_case_reuse_checks_hashes(elsepa_job, monkeypatch):
    source, exe, work = elsepa_job
    saved_elsepa_job(elsepa_job)
    for name, text in [('dcs_done.dat', 'synthetic DCS'), ('stdout.log', ''), ('stderr.log', '')]:
        (work/name).write_text(text)
    m.write(work/'complete.json', dict(dcs_file='dcs_done.dat', files={p.name: m.sha(p) for p in work.iterdir() if p.is_file()}))
    monkeypatch.setattr(generate, 'parse', lambda *a: synthetic_case_row())
    assert np.array_equal(generate.case(source, exe, work, 17, 1, 1.)['dcs'], np.ones(5))
    (work/'stdout.log').write_text('changed')
    with pytest.raises(ValueError, match='raw output changed'):
        generate.case(source, exe, work, 17, 1, 1.)


def test_saved_elsepa_table_is_reused_byte_for_byte_and_tampering_rejected(tmp_path):
    path = tmp_path/'z17_q1.npz'
    arrays = dict(energy_MeV=np.array([.01, 10.]), dcs=np.ones((2, 5)))
    np.savez(path, **arrays)  # Legacy table without a separate SHA receipt.
    original = path.read_bytes()
    generate.save_verified_table(path, arrays)
    generate.save_verified_table(path, arrays)
    assert path.read_bytes() == original
    np.savez(path, energy_MeV=arrays['energy_MeV'], dcs=np.full((2, 5), 2.))
    with pytest.raises(ValueError, match='hash mismatch'): generate.save_verified_table(path, arrays)
    unsealed = tmp_path/'unsealed.npz'; np.savez(unsealed, dcs=np.zeros(5))
    with pytest.raises(ValueError, match='differs from verified'): generate.save_verified_table(unsealed, arrays)


def test_s4_identity_migration_allows_only_authorized_wrapper_change(tmp_path):
    identity = dict(Z=17, charge=1, IHEF=0, energy_MeV=[.01, 10.], executable_sha256='exe',
                    implementations=dict(generate.PRE_TIMEOUT_IMPLEMENTATIONS, **{'physics.py': 'frozen'}))
    path = tmp_path/'generation.json'; m.write(path, identity)
    original = path.read_bytes()
    current = copy.deepcopy(identity)
    current['implementations'].update({'generate_elsepa_p8a.py': 'new wrapper', 'check_p8a.py': 'new checker'})
    generate.resume_s4(path, current); generate.resume_s4(path, current)
    assert path.read_bytes() == original
    assert path.with_suffix('.resume_v4.json').exists()
    for key, value in [('IHEF', 1), ('executable_sha256', 'other exe'), ('energy_MeV', [.02, 10.])]:
        changed = dict(current, **{key: value})
        with pytest.raises(ValueError, match='beyond appendix'): generate.resume_s4(path, changed)
    changed = copy.deepcopy(current); changed['implementations']['physics.py'] = 'changed'
    with pytest.raises(ValueError, match='beyond appendix'): generate.resume_s4(path, changed)


def test_node_timeout_stops_generation_without_partial_table(tmp_path, monkeypatch):
    from prototypes.mv_electron import p1f_common
    p1f = tmp_path/'p1f'; m.write(p1f/'t1.json', {'table_hashes': {}})
    monkeypatch.setattr(p1f_common, 'OUT', p1f)
    monkeypatch.setattr(generate, 'verify_protected', lambda: 0)
    monkeypatch.setattr(generate, 'MATERIALS', ('lead',))
    monkeypatch.setattr(generate, 'load_manifest', lambda *a, **kw: {'materials': {'lead': {'definition': {}}}})
    monkeypatch.setattr(generate, 'load_reference', lambda *a: {'weights': {'17': 1.}})
    exe = tmp_path/'exe'; exe.write_bytes(b'fake')
    monkeypatch.setattr(generate, 'prepare', lambda: (tmp_path, exe))
    monkeypatch.setattr(generate, 'energies', lambda: np.array([.01, 10.]))
    def timeout(*args, **kwargs):
        raise generate.ElsepaCaseTimeout(dict(Z=17, charge=-1, energy_MeV=.01))
    monkeypatch.setattr(generate, 'case', timeout)
    with pytest.raises(generate.ElsepaCaseTimeout): generate.generate(tmp_path/'out', tmp_path/'cache')
    assert not list((tmp_path/'cache').rglob('*.npz'))


@pytest.mark.parametrize('all_timeout', [False, True])
def test_direct_timeouts_are_explicitly_not_evaluated(tmp_path, monkeypatch, all_timeout):
    es = np.array([.01, .1, 1., 10.])
    row = synthetic_case_row(); row.pop('header')
    nodes = generate.stack_rows([row]*len(es), 17, 1, es)
    intended = generate.direct_energies(es, 17)
    def direct(source, exe, work, z, charge, t, **kwargs):
        assert kwargs['kind'] == 'direct'
        if all_timeout or t == intended[-1]:
            raise generate.ElsepaCaseTimeout(dict(reason='synthetic timeout', Z=z, charge=charge, energy_MeV=t))
        return copy.deepcopy(row)
    monkeypatch.setattr(generate, 'case', direct)
    work = tmp_path/'z17_q1'; work.mkdir()
    coverage = generate.collect_direct(None, None, work, 17, 1, es, nodes)
    with np.load(work/'direct.npz') as f: data = {k: f[k].copy() for k in f.files}
    omitted = check.direct_coverage(nodes, data, coverage)
    assert coverage['not_evaluated_count'] == (len(intended) if all_timeout else 1)
    assert coverage['status'] == 'not-evaluated'
    assert check.aggregate(['pass']+[r['status'] for r in omitted]) == 'not-evaluated'
    with pytest.raises(ValueError, match='missing without omission'): check.direct_coverage(nodes, data)
    wrong = copy.deepcopy(coverage); wrong['omitted'][0]['charge'] = -1
    with pytest.raises(ValueError, match='omission'): check.direct_coverage(nodes, data, wrong)
    if all_timeout:
        monkeypatch.setattr(check, 'load_reference', lambda ref, *a: ref)
        monkeypatch.setattr(check, 'inspect_atomic', lambda *a: {'status': 'pass'})
        result = check.atomic_checks({'z17_q1.npz': dict(nodes, direct=data, direct_status=coverage)})
        assert result['status'] == 'not-evaluated'
        assert result['direct_omissions'][0]['not_evaluated_count'] == len(intended)


@pytest.mark.parametrize('material', m.MATERIALS)
def test_xcom_requests_both_edge_sides_and_preserves_upper_convention(tmp_path, monkeypatch, material):
    from prototypes.mv_electron import photon_data_p2
    edges = np.array([.0025856, .0130352])  # Synthetic rounded-label fixture.
    (tmp_path/'XCOM.f').write_text('DATA EPAIR1/1.0D6/,EPAIR2/2.0D6/')
    monkeypatch.setattr(fetch, 'prepare_xcom', lambda work: tmp_path)
    monkeypatch.setattr(photon_data_p2, 'grid', lambda thresholds: np.array([.001, .01, .1, 1., 2., 20.]))
    monkeypatch.setattr(fetch, 'xcom_edges', lambda source, zs:
                        edges if len(zs) == 2 else edges[:1] if 17 in zs else edges[1:])
    requests = []
    def xcom(work, es, name, weights):
        requests.append((np.array(es), dict(weights)))
        values = np.ones((len(es), 7))
        # Evaluating the rounded label itself wrongly gives the lower side.
        values[:, 2] = 10.+10.*np.sum(np.asarray(es)[:, None] >= edges*(1+2e-7), axis=1)
        values[:, 5] = 123.  # XCOM total including coherent attenuation.
        return values
    monkeypatch.setattr(fetch, 'xcom', xcom)
    p = fetch.photon_table(tmp_path/'work', dict(material=material, weights={'17': .5, '82': .5}))
    assert np.array_equal(p['edge_upper_requests'], edges*(1+1e-6))
    assert np.array_equal(p['edge_lower_requests'], edges*(1-1e-6))
    assert np.all(p['check_total'] == 123.)
    assert len(requests) == 4  # mixture, held-out checks and both constituents
    assert np.all(np.diff(p['E']) > 0)
    for edge in edges:
        ids = np.flatnonzero(p['photo_E'] == edge)
        assert len(ids) == 2 and p['photo'][ids[1]] > p['photo'][ids[0]]
        assert m.photo_interp(edge, p['photo_E'], p['photo']) == pytest.approx(p['photo'][ids[1]])
        assert edge not in p['E']


@pytest.mark.parametrize('defect', ['mixture_down', 'element_flat', 'element_down', 'missing_side'])
def test_xcom_edge_jump_faults_stop_table_creation(tmp_path, monkeypatch, defect):
    from prototypes.mv_electron import photon_data_p2
    edge = .0130352
    (tmp_path/'XCOM.f').write_text('DATA EPAIR1/1.0D6/,EPAIR2/2.0D6/')
    monkeypatch.setattr(fetch, 'prepare_xcom', lambda work: tmp_path)
    monkeypatch.setattr(fetch, 'xcom_edges', lambda *a: np.array([edge]))
    monkeypatch.setattr(photon_data_p2, 'grid', lambda t: np.array([.001, .01, .1, 1., 2., 20.]))
    def xcom(work, es, name, weights):
        a = np.ones((len(es), 7))
        upper = es == edge*(1+1e-6)
        if name == 'mixture' and defect == 'mixture_down': a[upper, 2] = .9
        if name.startswith('nodes_') and defect == 'element_down': a[upper, 2] = .9
        return a  # Flat mixture is allowed, but its elemental edge must jump.
    monkeypatch.setattr(fetch, 'xcom', xcom)
    if defect == 'missing_side':
        with pytest.raises(ValueError, match='side request missing'):
            fetch.check_edge_jumps(np.array([edge]), np.ones(1), [edge])
    else:
        with pytest.raises(ValueError, match='did not jump'):
            fetch.photon_table(tmp_path/'work', dict(material='lead', weights={'82': 1.}))


def test_weak_mixture_edge_allows_equality_but_element_requires_jump():
    edge = .02
    e = edge*np.array([1-1e-6, 1+1e-6])
    fetch.check_edge_jumps(e, np.ones(2), [edge])
    with pytest.raises(ValueError, match='did not jump'):
        fetch.check_edge_jumps(e, np.ones(2), [edge], elemental=True)


@pytest.mark.parametrize('threshold', [1.022, 2.044])
def test_pair_near_threshold_uses_total_attenuation_and_strict_boundary(threshold):
    expected = np.array([.1])
    actual = expected*1.001107  # Fails the former relative-only rule.
    near = np.array([threshold*1.0001])
    r = check.pair_comparison(near, actual, expected, np.array([100.]), threshold)
    assert r['max_relative'] > .001 and r['status'] == 'pass'
    assert r['max_absolute_over_total'] < 1e-5
    assert check.pair_comparison(near, actual, expected, np.ones(1), threshold)['status'] == 'fail'
    # At exactly 1.01*threshold, the relative rule applies even for tiny xs.
    assert check.pair_comparison(np.array([1.01*threshold]), actual, expected,
                                 np.array([100.]), threshold)['status'] == 'fail'
    assert check.pair_comparison(np.array([1.02*threshold]), actual, expected,
                                 np.array([100.]), threshold)['status'] == 'fail'
    with pytest.raises(ValueError, match='coefficient'):
        check.pair_comparison(near, actual, expected, np.zeros(1), threshold)


@pytest.mark.parametrize('defect', ['none', 'large_tst', 'bad_total', 'bad_g1', 'bad_g2',
                                   'doubling', 'unknown', 'mixed', 'nonfinite_tst', 'negative_tst'])
def test_p8a_tst_warning_requires_all_appendix5_conditions(monkeypatch, defect):
    from prototypes.mv_electron import check_elsepa_p1f
    delta = np.zeros(3)
    if defect == 'bad_total': delta[0] = .0051
    if defect == 'bad_g1': delta[1] = .0011
    if defect == 'bad_g2': delta[2] = .0011
    def integrated(*args, **kwargs):
        return np.ones(3)*(1+1.1e-6 if defect == 'doubling' and kwargs.get('order') == 32 else 1.)
    monkeypatch.setattr(check_elsepa_p1f, 'integrated', integrated)
    tst = {'large_tst': '1.001E-4', 'nonfinite_tst': 'NaN', 'negative_tst': '-1E-8'}.get(defect, '1.0D-4')
    warning = f'WARNING: TST = {tst}. Check convergence.\n\n  Total elastic cross section = 1.0E-18 cm**2'
    if defect == 'unknown': warning = 'WARNING: Possible accumulation of round-off errors.'
    if defect == 'mixed': warning += '\nWARNING: another problem'
    a = dict(mu=np.linspace(0, 1, 5), dcs=np.ones(5), error=np.zeros(5), table_xs=np.ones(3),
             series_xs=1/(1+delta), warnings=np.array([warning]), notices=np.array([], dtype=str), convergence='[]')
    result = check.inspect_atomic(a, 17, -1, 8., 'node')
    assert result['status'] == ('pass' if defect == 'none' else 'fail')
    assert bool(result['accepted_tst_warnings']) == (defect == 'none')
    assert result['raw_warning_blocks'] == [warning]
    assert check_elsepa_p1f.inspect(a, 17, -1, 8., 'node')['status'] == 'fail'
    if defect == 'none':
        assert result['accepted_tst_max'] == 1e-4
        # Duplicate raw warning blocks count as one accepted point.
        result['accepted_tst_warnings'] *= 2
        second = dict(result, charge=1, accepted_tst_max=2e-8)
        groups = check.tst_summary([result, result, second, {'status': 'not-evaluated'}])
        assert groups == [dict(Z=17, charge=-1, accepted_point_count=2, max_TST=1e-4),
                          dict(Z=17, charge=1, accepted_point_count=1, max_TST=2e-8)]


@pytest.fixture
def photon_migration_job(tmp_path, water, synthetic_tables, monkeypatch):
    out, cache = tmp_path/'out', tmp_path/'cache'
    monkeypatch.setattr(m, 'verify_protected', lambda: 0)
    monkeypatch.setattr(fetch, 'verify_protected', lambda: 0)
    m.write(out/'lung_state.json', dict(status='pass', density_g_cm3=.26, report='synthetic S2'))
    for material in m.MATERIALS:
        folder = cache/material
        d = dict(water, material=material, density_g_cm3=.26 if material == 'lung' else 1.)
        s, p = copy.deepcopy(synthetic_tables)
        for data in (s, p):
            data.update(material=np.array(material), composition_sha256=np.array(m.composition_id(d)))
        m.write(folder/'definition.json', d)
        np.savez(folder/'stopping.npz', **s)
        np.savez(folder/'photon.npz', **p)
        np.savez(folder/'range.npz', T=np.array([.01, 1.]), csda_g_cm2=np.array([.001, .1]),
                 material=np.array(material), composition_sha256=np.array(m.composition_id(d)))
        receipt = {name: m.sha(folder/name) for name in ('definition.json', 'stopping.npz', 'photon.npz', 'range.npz')}
        if material == 'lung':
            m.write(folder/'standard_definition.json', dict(d, density_g_cm3=1.05))
            np.savez(folder/'standard_stopping.npz', **dict(s, collision=s['collision']*.99))
            for name in ('standard_definition.json', 'standard_stopping.npz'):
                receipt[name] = m.sha(folder/name)
            # S1's original definition/stopping hashes differ from selected S2.
            receipt['definition.json'] = receipt['standard_definition.json']
            receipt['stopping.npz'] = receipt['standard_stopping.npz']
            m.write(folder/'lung_outputs.json', {name: m.sha(folder/name) for name in ('definition.json', 'stopping.npz')})
        m.write(folder/'fetch_outputs.json', receipt)
    manifest = m.build(out, cache)
    # A downstream S5 entry must survive the photon-only manifest update.
    pair = cache/'lead'/'pair.npz'
    np.savez(pair, material=np.array('lead'), sentinel=np.array([1., 2.]))
    manifest['materials']['lead']['pair'] = m.reference(pair, 'fixture', 'S5 fixture', '1', 'lead')
    m.write(out/'manifest.json', manifest)
    frozen = {str(p): p.read_bytes() for p in cache.rglob('*') if p.is_file() and p.name not in ('photon.npz', 'fetch_outputs.json')}
    frozen[str(out/'lung_state.json')] = (out/'lung_state.json').read_bytes()
    calls = []
    def photons(work, definition):
        assert work == cache/'xcom_appendix5'/definition['material']
        if definition['material'] == 'lung': assert definition['density_g_cm3'] == .26
        calls.append(definition['material'])
        data = copy.deepcopy(synthetic_tables[1])
        data['photo'] *= 1.002
        return data
    monkeypatch.setattr(fetch, 'photon_table', photons)
    return out, cache, manifest, frozen, calls


def test_photon_only_updates_only_photons_and_preserves_s2_s4_s5(photon_migration_job):
    out, cache, original, frozen, calls = photon_migration_job
    old_manifest = (out/'manifest.json').read_bytes()
    result = fetch.photon_only(out, cache)
    assert result['status'] == 'complete' and calls == list(m.MATERIALS)
    assert 'appendix 5 (1)' in result['reason']
    assert result['original_manifest'] == original
    assert result['original_manifest_sha256'] != m.sha(out/'manifest.json')
    updated = m.load_manifest(out/'manifest.json', complete=False)
    for material in m.MATERIALS:
        change = result['photons'][material]
        assert change['before'] == original['materials'][material]['photon']['sha256']
        assert change['after'] == m.sha(cache/material/'photon.npz') != change['before']
        assert updated['materials'][material]['photon'] == change['reference']
        assert {k: v for k, v in updated['materials'][material].items() if k != 'photon'} == {
            k: v for k, v in original['materials'][material].items() if k != 'photon'}
        receipt = json.loads((cache/material/'fetch_outputs.json').read_text())
        old_receipt = result['original_receipts'][material]['values']
        assert receipt == dict(old_receipt, **{'photon.npz': change['after']})
    assert all(Path(name).read_bytes() == data for name, data in frozen.items())
    journal_hash = m.sha(out/'photon_migration_v5.json')
    fetch.photon_only(out, cache)
    assert calls == list(m.MATERIALS) and m.sha(out/'photon_migration_v5.json') == journal_hash
    assert old_manifest != (out/'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='S2 already started'):
        fetch.fetch(out, cache)  # Full S1 still cannot overwrite selected S2.


@pytest.mark.parametrize('fault', ['stopping.npz', 'range.npz', 'definition.json', 'photon.npz',
                                   'standard_stopping.npz', 'standard_definition.json'])
def test_photon_only_rejects_preexisting_source_tampering_before_work(photon_migration_job, fault):
    out, cache, original, frozen, calls = photon_migration_job
    path = cache/'lung'/fault
    path.write_bytes(path.read_bytes()+b'tampered')
    manifest_before = (out/'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='hash mismatch'):
        fetch.photon_only(out, cache)
    assert not calls and (out/'manifest.json').read_bytes() == manifest_before
    assert not (out/'photon_migration_v5.json').exists()


@pytest.mark.parametrize('fault', [None, 'lung_state', 'photon', 'staged', 'manifest', 'implementation'])
def test_photon_only_resume_retains_old_and_staged_hashes(photon_migration_job, monkeypatch, fault):
    out, cache, original, frozen, calls = photon_migration_job
    photons = fetch.photon_table
    def interrupted(work, definition):
        if definition['material'] == 'lung': raise RuntimeError('synthetic interruption')
        return photons(work, definition)
    monkeypatch.setattr(fetch, 'photon_table', interrupted)
    with pytest.raises(RuntimeError, match='interruption'): fetch.photon_only(out, cache)
    record = json.loads((out/'photon_migration_v5.json').read_text())
    assert record['status'] == 'pending' and set(record['photons']) == {'water'}
    assert json.loads((out/'manifest.json').read_text()) == original
    assert all(Path(name).read_bytes() == data for name, data in frozen.items())
    if fault:
        if fault == 'implementation':
            monkeypatch.setattr(fetch, 'implementation_hashes', lambda: {'changed': 'unknown'})
        else:
            target = {'lung_state': out/'lung_state.json', 'photon': cache/'water'/'photon.npz',
                      'staged': Path(record['photons']['water']['staged']), 'manifest': out/'manifest.json'}[fault]
            target.write_bytes(target.read_bytes()+b'changed')
        monkeypatch.setattr(fetch, 'photon_table', lambda *a: pytest.fail('must fail before generation'))
        with pytest.raises(ValueError, match='changed|hash mismatch'):
            fetch.photon_only(out, cache)
    else:
        monkeypatch.setattr(fetch, 'photon_table', photons)
        assert fetch.photon_only(out, cache)['status'] == 'complete'
        assert calls == list(m.MATERIALS)  # Reuse the sealed staged water table.


def test_photon_only_recovers_interrupted_install_without_regeneration(photon_migration_job, monkeypatch):
    out, cache, original, frozen, calls = photon_migration_job
    copyfile = fetch.shutil.copyfile
    count = []
    def interrupted(source, target):
        count.append(source)
        if len(count) == 2: raise RuntimeError('install interrupted')
        return copyfile(source, target)
    monkeypatch.setattr(fetch.shutil, 'copyfile', interrupted)
    with pytest.raises(RuntimeError, match='install interrupted'): fetch.photon_only(out, cache)
    record = json.loads((out/'photon_migration_v5.json').read_text())
    assert record['status'] == 'ready'
    assert m.sha(cache/'water'/'photon.npz') == record['photons']['water']['after']
    assert json.loads((out/'manifest.json').read_text()) == original
    monkeypatch.setattr(fetch.shutil, 'copyfile', copyfile)
    monkeypatch.setattr(fetch, 'photon_table', lambda *a: pytest.fail('unexpected regeneration'))
    assert fetch.photon_only(out, cache)['status'] == 'complete'
    assert all(Path(name).read_bytes() == data for name, data in frozen.items())


@pytest.mark.parametrize('stage', ['S4', 'S5'])
def test_appendix6_migration_seals_only_authorized_files(tmp_path, stage):
    old = dict(Z=17, charge=1, energy_MeV=[.01, 10.], definition_sha256='same definition',
               executable_sha256='same exe', implementations={
                   n: 'old '+n for n in ('fetch_materials_p8a.py', 'check_p8a.py', 'pegs_p8a.py',
                                       'materials_p8a.py', 'generate_elsepa_p8a.py', 'gs_backend.py')})
    path = tmp_path/'generation.json'; m.write(path, old)
    before = path.read_bytes()
    new = copy.deepcopy(old)
    for name in new['implementations']:
        if name != 'gs_backend.py': new['implementations'][name] = 'new '+name
    evidence = {'executable_sha256': 'binary hash'}
    m.resume_appendix6(path, new, stage, evidence)
    m.resume_appendix6(path, new, stage, evidence)
    assert path.read_bytes() == before
    receipt = json.loads(path.with_suffix('.resume_v6.json').read_text())
    assert receipt['original_sha256'] == m.sha(path) and len(receipt['changes']) == 5
    assert receipt['current_identity'] == new and receipt['evidence'] == evidence
    with pytest.raises(ValueError, match='conditions changed'):
        m.resume_appendix6(path, new, stage, {'executable_sha256': 'changed binary'})
    for field, value in [('Z', 18), ('charge', -1), ('energy_MeV', [.02, 10.]),
                         ('definition_sha256', 'changed'), ('executable_sha256', 'changed')]:
        with pytest.raises(ValueError, match='beyond appendix 6'):
            m.resume_appendix6(path, dict(new, **{field: value}), stage, evidence)
    for fault in ('helper', 'added', 'removed'):
        changed = copy.deepcopy(new)
        if fault == 'helper': changed['implementations']['gs_backend.py'] = 'other helper'
        if fault == 'added': changed['implementations']['other.py'] = 'other'
        if fault == 'removed': del changed['implementations']['gs_backend.py']
        with pytest.raises(ValueError, match='beyond appendix 6'):
            m.resume_appendix6(path, changed, stage, evidence)
    changed = copy.deepcopy(new); changed['implementations']['check_p8a.py'] = 'another edit'
    with pytest.raises(ValueError, match='conditions changed'):
        m.resume_appendix6(path, changed, stage, evidence)


def test_s4_migration_chains_appendix4_and_retains_both_receipts(tmp_path):
    old = dict(Z=17, charge=1, IHEF=0, energy_MeV=[.01, 10.], executable_sha256='same',
               implementations=dict(generate.PRE_TIMEOUT_IMPLEMENTATIONS,
                                    **{'materials_p8a.py': 'old materials', 'physics.py': 'frozen'}))
    path = tmp_path/'generation.json'; m.write(path, old)
    v4 = copy.deepcopy(old)
    v4['implementations'].update({'generate_elsepa_p8a.py': 'v4 wrapper', 'check_p8a.py': 'v4 checker'})
    generate.resume_s4(path, v4)
    before = path.read_bytes(), path.with_suffix('.resume_v4.json').read_bytes()
    v6 = copy.deepcopy(v4)
    v6['implementations'].update({'materials_p8a.py': 'v6 materials', 'generate_elsepa_p8a.py': 'v6 wrapper', 'check_p8a.py': 'v5 checker'})
    generate.resume_s4(path, v6); generate.resume_s4(path, v6)
    assert before == (path.read_bytes(), path.with_suffix('.resume_v4.json').read_bytes())
    record = json.loads(path.with_suffix('.resume_v6.json').read_text())
    assert record['previous_v4_sha256'] == m.sha(path.with_suffix('.resume_v4.json'))
    path.with_suffix('.resume_v4.json').write_bytes(before[1]+b' ')
    with pytest.raises(ValueError, match='conditions changed'): generate.resume_s4(path, v6)


def test_s5_migration_reuses_completed_diagnostic_without_execution(tmp_path, water, diagnostic_state, monkeypatch):
    monkeypatch.setattr(pegs, 'EGS', tmp_path/'source')
    es = np.array([.01, 1.])
    work = tmp_path/'pegs'
    implementations = m.implementation_hashes()
    previous = dict(implementations, **{'fetch_materials_p8a.py': 'old fetch', 'check_p8a.py': 'old check',
                                       'pegs_p8a.py': 'old pegs', 'materials_p8a.py': 'old materials',
                                       'generate_elsepa_p8a.py': 'v4 wrapper'})
    identity = dict(definition=water, energies=es.tolist(), source=pegs.diagnostic_source(),
                    implementations=previous, pegs_input=pegs.pegs_input([water], unrestricted=4), egs_sources={})
    m.write(work/'identity.json', identity)
    text = diagnostic_state+'STOP .01 1 1 1 1 1 1 1 1 1 1\nSTOP 1 1 1 1 1 1 1 1 1 1 1\nDONE\n'
    for name, content in {'direct.txt': text, 'p8adiag.f': identity['source'], 'p8adiag.inp': identity['pegs_input'],
                          'pgs5job.pegs5dat': 'synthetic pegs', 'pgs5job.pegs5lst': 'synthetic listing',
                          'egs5job.exe': 'synthetic executable'}.items():
        (work/name).write_text(content)
    m.write(work/'complete.json', {name: m.sha(work/name) for name in
                                 ('direct.txt', 'p8adiag.f', 'p8adiag.inp', 'pgs5job.pegs5dat', 'pgs5job.pegs5lst')})
    frozen = {p: p.read_bytes() for p in work.iterdir()}
    result = pegs.run_diagnostic(water, es, work)
    assert np.array_equal(result['collision'], np.ones(2))
    pegs.run_diagnostic(water, es, work)
    assert all(p.read_bytes() == b for p, b in frozen.items())
    with pytest.raises(ValueError, match='beyond appendix 6'):
        pegs.run_diagnostic(dict(water, density_g_cm3=2.), es, work)
    with pytest.raises(ValueError, match='beyond appendix 6'):
        pegs.run_diagnostic(water, [.02, 1.], work)
    (work/'egs5job.exe').write_text('changed executable')
    with pytest.raises(ValueError, match='conditions changed'):
        pegs.run_diagnostic(water, es, work)


@pytest.mark.parametrize('option,expected', [(None, 'S1'), ('--lung', 'S2'), ('--photon-only', 'photon')])
def test_fetch_cli_selects_photon_only_without_running_s1_s2(monkeypatch, option, expected):
    import sys
    calls = []
    monkeypatch.setattr(sys, 'argv', ['fetch_materials_p8a']+([] if option is None else [option]))
    monkeypatch.setattr(fetch, 'verify_protected', lambda: 0)
    monkeypatch.setattr(fetch, 'fetch', lambda: calls.append('S1'))
    monkeypatch.setattr(fetch, 'lung_state', lambda: calls.append('S2'))
    monkeypatch.setattr(fetch, 'photon_only', lambda: calls.append('photon'))
    fetch.main()
    assert calls == [expected]
    monkeypatch.setattr(sys, 'argv', ['fetch_materials_p8a', '--lung', '--photon-only'])
    with pytest.raises(SystemExit) as error: fetch.main()
    assert error.value.code == 2


def test_s4_reuses_completed_elemental_direct_and_gs_bytes(tmp_path, water, monkeypatch):
    from prototypes.mv_electron import p1f_common, gs_backend
    out, cache, p1f = tmp_path/'out', tmp_path/'cache', tmp_path/'p1f'
    m.write(p1f/'t1.json', {'table_hashes': {}})
    monkeypatch.setattr(p1f_common, 'OUT', p1f)
    monkeypatch.setattr(generate, 'verify_protected', lambda: 0)
    monkeypatch.setattr(generate, 'MATERIALS', ('lead',))
    es = np.array([.01, 10.])
    monkeypatch.setattr(generate, 'energies', lambda: es)
    exe = tmp_path/'elscata'; exe.write_bytes(b'synthetic binary')
    monkeypatch.setattr(generate, 'prepare', lambda: (tmp_path/'source', exe))
    monkeypatch.setattr(gs_backend, 'build', lambda *a: pytest.fail('must reuse completed GS'))
    d = dict(water, material='lead', weights={'17': 1.}, atomic_weights={'17': 1.})
    m.write(cache/'lead'/'definition.json', d)
    definition = m.reference(cache/'lead'/'definition.json', 'fixture', 'fixture', '1', 'lead')
    manifest = {'materials': {'lead': {'definition': definition}}}
    monkeypatch.setattr(generate, 'load_manifest', lambda *a, **kw: copy.deepcopy(manifest))
    implementations = m.implementation_hashes()
    old = dict(implementations, **{'fetch_materials_p8a.py': 'old fetch', 'check_p8a.py': 'old checker',
                                  'materials_p8a.py': 'old materials', 'pegs_p8a.py': 'old pegs',
                                  'generate_elsepa_p8a.py': 'v4 generator'})
    tables = {}
    for charge in (-1, 1):
        work = cache/'elsepa'/f'z17_q{charge}'
        m.write(work/'generation.json', dict(Z=17, charge=charge, IHEF=0, energy_MeV=es.tolist(),
                                            executable_sha256=m.sha(exe), implementations=old))
        direct = generate.direct_energies(es, 17)
        for group, energies in (('pass1', es), ('pass2', es), ('direct', direct)):
            for t in energies:
                case_work = work/group/float(t).hex()
                saved_elsepa_job((tmp_path/'source', exe, case_work), 17, charge, float(t))
                for name, text in [('dcs_fixture.dat', 'synthetic DCS'), ('stdout.log', ''), ('stderr.log', '')]:
                    (case_work/name).write_text(text)
                m.write(case_work/'complete.json', dict(dcs_file='dcs_fixture.dat', files={
                    p.name: m.sha(p) for p in case_work.iterdir() if p.is_file()}))
        row = synthetic_case_row(17, charge); row.pop('header')
        table = work/f'z17_q{charge}.npz'
        np.savez(table, **generate.stack_rows([row]*len(es), 17, charge, es))
        np.savez(work/'direct.npz', **generate.stack_rows([row]*len(direct), 17, charge, direct))
        tables[table.name] = m.sha(table)
    gs = cache/'lead'/'gs.npz'
    np.savez(gs, sentinel=np.array([1., 2.]))
    m.write(gs.with_suffix('.sha.json'), dict(sha256=m.sha(gs)))
    m.write(gs.with_suffix('.identity.json'), dict(definition_sha256=definition['sha256'],
                                                 elemental=tables, implementations=old))
    frozen = {p: p.read_bytes() for p in cache.rglob('*') if p.is_file()}
    def parse(path, stdout):
        charge = -1 if any(p.name == 'z17_q-1' for p in path.parents) else 1
        return synthetic_case_row(17, charge, float.fromhex(path.parent.name))
    monkeypatch.setattr(generate, 'parse', parse)
    generate.generate(out, cache)
    generate.generate(out, cache)
    assert all(p.read_bytes() == data for p, data in frozen.items())
    assert gs.with_suffix('.identity.resume_v6.json').exists()
    for charge in (-1, 1):
        assert (cache/'elsepa'/f'z17_q{charge}'/'generation.resume_v6.json').exists()


def test_appendix7_mixture_falloff_floor_and_strict_element_jump():
    edge = .0010428
    es = edge*np.array([1-1e-6, 1+1e-6])
    floor = (es[0]/es[1])**4
    smooth = (es[0]/es[1])**3
    fetch.check_edge_jumps(es, np.array([1., smooth]), [edge])
    fetch.check_edge_jumps(es, np.array([1., floor]), [edge])
    with pytest.raises(ValueError, match='did not jump'):
        fetch.check_edge_jumps(es, np.array([1., floor*(1-1e-8)]), [edge])
    with pytest.raises(ValueError, match='did not jump'):
        fetch.check_edge_jumps(es, np.array([1., smooth]), [edge], elemental=True)
    fetch.check_edge_jumps(es, np.array([1., 1.01]), [edge], elemental=True)


def test_appendix8_excludes_only_edge_gap_midpoints(synthetic_tables, water):
    s, p = synthetic_tables
    gap = p['photo_E'][:-1] == p['photo_E'][1:]
    assert gap.sum() == 1
    p['check_photo'][gap] *= 100  # The edge itself must not judge interpolation.
    result = check.interpolation_checks(s, p, m.constants(water)['NE'])
    assert result['photoelectric']['status'] == 'pass'
    assert result['photoelectric']['excluded_edge_midpoints'] == 1
    p['check_photo'][np.flatnonzero(~gap)[0]] *= 1.002
    result = check.interpolation_checks(s, p, m.constants(water)['NE'])
    assert result['photoelectric']['status'] == 'fail'  # Still relative 0.1%.
    assert result['pair_nuclear']['status'] == result['pair_electron']['status'] == 'pass'


def test_appendix9_refines_all_low_edges_only_and_keeps_edge_gaps():
    from prototypes.mv_electron.photon_data_p2 import grid
    thresholds = np.array([1.022, 2.044])
    edges = np.array([.0010428, .003066, .00307, .02, .04])
    base = np.unique(np.r_[grid(thresholds), edges*(1-1e-6), edges*(1+1e-6)])
    refined = fetch.refine_edge_grid(base, edges)
    assert np.all(np.isin(base, refined)) and np.all(np.diff(refined) > 0)
    np.testing.assert_array_equal(refined[refined > .02], base[base > .02])
    for threshold in thresholds:
        for es in (base, refined):
            assert np.array_equal(es[(es > .99*threshold) & (es < 1.02*threshold)],
                                  base[(base > .99*threshold) & (base < 1.02*threshold)])
    added = refined[~np.isin(refined, base)]
    assert len(added) > 0
    assert all(any(.9*edge <= value <= edge*(1-1e-6) for edge in edges if edge <= .02) for value in added)
    for edge in edges:
        lower, upper = edge*np.array([1-1e-6, 1+1e-6])
        assert not np.any((refined > lower) & (refined < upper))
        if edge <= .02:
            assert np.any((added > .9*edge) & (added < lower))
            # Every clipped original interval has log spacing at most 1/4.
            for a, b in zip(base[:-1], base[1:]):
                lo, hi = max(a, .9*edge), min(b, lower)
                if lo < hi and not any(a == e*(1-1e-6) and b == e*(1+1e-6) for e in edges):
                    local = refined[(refined >= lo) & (refined <= hi)]
                    assert np.max(np.diff(np.log(local))) <= np.log(hi/lo)/4+1e-14
    # Even overlapping windows must leave exactly two conceptual edge sides.
    photo_e = refined.copy()
    for edge in edges:
        photo_e[np.isin(refined, edge*np.array([1-1e-6, 1+1e-6]))] = edge
        assert np.sum(photo_e == edge) == 2
    assert np.sum(photo_e[:-1] == photo_e[1:]) == len(edges)


def test_appendix9_refinement_reduces_synthetic_curvature_error_below_point_one_percent():
    from prototypes.mv_electron.photon_data_p2 import grid
    edge = .003066
    original = np.unique(np.r_[grid(np.array([1., 2.])), edge*(1-1e-6), edge*(1+1e-6)])
    es = original[(original >= .9*edge) & (original <= edge*(1-1e-6))]
    fine = fetch.refine_edge_grid(original, np.array([edge]))
    fine = fine[(fine >= es[0]) & (fine <= es[-1])]
    # An arbitrary smooth positive curve, not an XCOM value or physical claim.
    def curve(e):
        x = np.log(e/edge)
        return np.exp(-3*x+80*x*x)
    def error(nodes):
        mid = np.sqrt(nodes[:-1]*nodes[1:])
        return check.relative(m.log_interp(mid, nodes, curve(nodes)), curve(mid))
    assert error(es) > .001 and error(fine) < .001


@pytest.fixture
def second_photon_migration_job(photon_migration_job, synthetic_tables, monkeypatch):
    out, cache, original, frozen, first_calls = photon_migration_job
    current = m.implementation_hashes()
    old = dict(current, **{'fetch_materials_p8a.py': 'old edge acquisition', 'check_p8a.py': 'old edge checks'})
    monkeypatch.setattr(fetch, 'implementation_hashes', lambda: old.copy())
    first = fetch.photon_only(out, cache)
    first_journal = out/'photon_migration_v5.json'
    retained = {p: p.read_bytes() for p in (cache/'xcom_appendix5').rglob('*') if p.is_file()}
    retained[first_journal] = first_journal.read_bytes()
    retained[out/'photon_migration_v5.manifest.json'] = (out/'photon_migration_v5.manifest.json').read_bytes()
    monkeypatch.setattr(fetch, 'implementation_hashes', lambda: current.copy())
    second_calls = []
    def photons(work, definition):
        assert work == cache/'xcom_appendix9_round2'/definition['material']
        second_calls.append(definition['material'])
        data = copy.deepcopy(synthetic_tables[1])
        data['photo'] *= 1.003
        return data
    monkeypatch.setattr(fetch, 'photon_table', photons)
    return out, cache, first, frozen, retained, second_calls


def test_photon_only_second_round_keeps_first_history_and_chains_hashes(second_photon_migration_job):
    out, cache, first, frozen, retained, calls = second_photon_migration_job
    second = fetch.photon_only(out, cache)
    assert second['status'] == 'complete' and second['round'] == 2
    assert 'appendix 9' in second['reason'] and calls == list(m.MATERIALS)
    assert second['previous_migration'] == dict(path=str(out/'photon_migration_v5.json'), sha256=m.sha(out/'photon_migration_v5.json'))
    assert second['original_manifest'] == json.loads((out/'photon_migration_v5.manifest.json').read_text())
    manifest = m.load_manifest(out/'manifest.json', complete=False)
    for material, change in second['photons'].items():
        assert change['before'] == first['photons'][material]['after']
        assert change['after'] != change['before']
        assert manifest['materials'][material]['photon']['sha256'] == change['after']
        assert json.loads((cache/material/'fetch_outputs.json').read_text())['photon.npz'] == change['after']
    assert all(Path(p).read_bytes() == data for p, data in frozen.items())
    assert all(p.read_bytes() == data for p, data in retained.items())
    journal = out/'photon_migration_v9_round2.json'
    before = journal.read_bytes()
    fetch.photon_only(out, cache)
    assert journal.read_bytes() == before and calls == list(m.MATERIALS)


@pytest.mark.parametrize('fault', ['preserved', 'photon', 'previous_staged', 'unfinished', 'implementation', 'new_key'])
def test_second_photon_round_rejects_changed_baseline_before_requests(second_photon_migration_job, monkeypatch, fault):
    out, cache, first, frozen, retained, calls = second_photon_migration_job
    if fault == 'preserved': (cache/'lung'/'stopping.npz').write_bytes(b'changed')
    if fault == 'photon': (cache/'water'/'photon.npz').write_bytes(b'changed')
    if fault == 'previous_staged': Path(first['photons']['water']['staged']).write_bytes(b'changed')
    if fault == 'unfinished':
        first['status'] = 'pending'; m.write(out/'photon_migration_v5.json', first)
    if fault in ('implementation', 'new_key'):
        current = m.implementation_hashes()
        current['gs_backend.py' if fault == 'implementation' else 'unknown.py'] = 'unauthorized edit'
        monkeypatch.setattr(fetch, 'implementation_hashes', lambda: current)
    with pytest.raises(ValueError, match='changed|hash mismatch'):
        fetch.photon_only(out, cache)
    assert not calls and not (out/'photon_migration_v9_round2.json').exists()


@pytest.mark.parametrize('fault', [None, 'previous_journal', 'second_staged', 'implementation'])
def test_second_photon_round_resumes_without_overwriting_first(second_photon_migration_job, monkeypatch, fault):
    out, cache, first, frozen, retained, calls = second_photon_migration_job
    photons = fetch.photon_table
    def interrupted(work, definition):
        if definition['material'] == 'lung': raise RuntimeError('second round interrupted')
        return photons(work, definition)
    monkeypatch.setattr(fetch, 'photon_table', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'): fetch.photon_only(out, cache)
    second = json.loads((out/'photon_migration_v9_round2.json').read_text())
    assert second['status'] == 'pending' and set(second['photons']) == {'water'}
    assert all(p.read_bytes() == data for p, data in retained.items())
    if fault:
        if fault == 'previous_journal':
            p = out/'photon_migration_v5.json'; p.write_bytes(p.read_bytes()+b' ')
        if fault == 'second_staged': Path(second['photons']['water']['staged']).write_bytes(b'changed')
        if fault == 'implementation':
            current = m.implementation_hashes(); current['fetch_materials_p8a.py'] = 'another edit'
            monkeypatch.setattr(fetch, 'implementation_hashes', lambda: current)
        monkeypatch.setattr(fetch, 'photon_table', lambda *a: pytest.fail('must reject before requests'))
        with pytest.raises(ValueError, match='changed|hash mismatch'): fetch.photon_only(out, cache)
    else:
        monkeypatch.setattr(fetch, 'photon_table', photons)
        assert fetch.photon_only(out, cache)['status'] == 'complete'
        assert calls == list(m.MATERIALS)
        assert all(p.read_bytes() == data for p, data in retained.items())
        assert all(Path(p).read_bytes() == data for p, data in frozen.items())


def test_second_photon_round_resumes_partial_install(second_photon_migration_job, monkeypatch):
    out, cache, first, frozen, retained, calls = second_photon_migration_job
    copyfile = fetch.shutil.copyfile
    attempts = []
    def interrupted(src, dst):
        attempts.append(src)
        if len(attempts) == 2: raise RuntimeError('install interrupted')
        return copyfile(src, dst)
    monkeypatch.setattr(fetch.shutil, 'copyfile', interrupted)
    with pytest.raises(RuntimeError, match='install interrupted'): fetch.photon_only(out, cache)
    second = json.loads((out/'photon_migration_v9_round2.json').read_text())
    assert second['status'] == 'ready'
    assert m.sha(cache/'water'/'photon.npz') == second['photons']['water']['after']
    monkeypatch.setattr(fetch.shutil, 'copyfile', copyfile)
    monkeypatch.setattr(fetch, 'photon_table', lambda *a: pytest.fail('no repeat requests'))
    assert fetch.photon_only(out, cache)['status'] == 'complete'
    assert calls == list(m.MATERIALS)
    assert all(p.read_bytes() == data for p, data in retained.items())
    assert all(Path(p).read_bytes() == data for p, data in frozen.items())


@pytest.mark.parametrize('z,index', [(17, -2), (20, -1)])
def test_appendix10_energy_shift_is_unique_and_in_range(z, index):
    es = generate.energies()
    checks = generate.direct_energies(es, z)
    original = float(checks[index])
    value = generate.replacement_energy(original, es, checks)
    assert value == original*(1+1e-3) and value not in es and value not in checks
    with pytest.raises(ValueError, match='overlaps'):
        generate.replacement_energy(original, np.r_[es, value], checks)
    with pytest.raises(ValueError, match='overlaps'):
        generate.replacement_energy(original, es, np.r_[checks, value])
    with pytest.raises(ValueError, match='overlaps'):
        generate.replacement_energy(original, es, np.r_[checks, value*(1+1e-10)])
    with pytest.raises(ValueError, match='range'):
        generate.replacement_energy(9.999, es, checks)


@pytest.fixture
def replacement_job(tmp_path, water, monkeypatch):
    from prototypes.mv_electron import check_elsepa, check_elsepa_p1f, gs_backend
    out, cache = tmp_path/'out', tmp_path/'cache'
    source, exe = tmp_path/'source', tmp_path/'elscata'
    (source/'database').mkdir(parents=True)
    exe.write_bytes(b'synthetic ELSEPA executable')
    monkeypatch.setattr(generate, 'prepare', lambda: (source, exe))
    monkeypatch.setattr(generate, 'verify_protected', lambda: 0)
    monkeypatch.setattr(gs_backend, 'build', lambda *a: pytest.fail('GS must not be regenerated'))
    # Synthetic angular integration preserves amplitude and gives stable G1/G2;
    # exercise the real warning policy and relative/CDF decision code.
    def integrated(atom, *a, **kw): return np.ones(3)*float(np.exp(atom.curve(.5)))
    monkeypatch.setattr(check_elsepa, 'integrated', integrated)
    monkeypatch.setattr(check_elsepa_p1f, 'integrated', integrated)
    state = {'mode': 'pass'}
    def cdf(atom, *args):
        amplitude = float(np.exp(atom.curve(.5)))
        return np.array([0., .5, 1.]), np.array([0., .5+(amplitude-1)*10 if state['mode'] == 'cdf' else .5, 1.])
    monkeypatch.setattr(gs_backend, 'single_cdf', cdf)
    es = generate.energies()
    tables = {}
    for charge in (-1, 1):
        z = 20
        work = cache/'elsepa'/f'z{z}_q{charge}'
        row = synthetic_case_row(z, charge); row.pop('header')
        nodes = generate.stack_rows([row]*len(es), z, charge, es)
        m.write(work/'placeholder.json', {})  # Create the isolated synthetic job.
        np.savez(work/f'z{z}_q{charge}.npz', **nodes)
        intended = generate.direct_energies(es, z)
        completed = intended.copy()
        omitted = []
        if charge == 1:
            original = float(intended[-1])
            completed = intended[:-1]
            raw = work/'direct'/original.hex()
            identity = saved_elsepa_job((source, exe, raw), z, charge, original)
            (raw/'dcs_failed.dat').write_bytes(b'')
            record = generate._retain_timeout(raw, identity, 'synthetic original timeout')
            omitted = [dict(status='not-evaluated', reason=record['reason'], Z=z, charge=charge,
                            energy_MeV=original, timeout_record=str(raw/'timeout.json'), timeout_seconds=600)]
        np.savez(work/'direct.npz', **generate.stack_rows([row]*len(completed), z, charge, completed))
        m.write(work/'direct_status.json', dict(Z=z, charge=charge, expected_energy_MeV=intended.tolist(),
                completed_count=len(completed), not_evaluated_count=len(omitted), omitted=omitted,
                status='not-evaluated' if omitted else 'pass'))
        old = dict(Z=z, charge=charge, IHEF=0, energy_MeV=es.tolist(), executable_sha256=m.sha(exe),
                   implementations=dict(m.implementation_hashes(), **{'generate_elsepa_p8a.py': 'original wrapper'}))
        m.write(work/'generation.json', old)
        v6 = copy.deepcopy(old)
        v6['implementations'].update({'generate_elsepa_p8a.py': 'v6 wrapper', 'fetch_materials_p8a.py': 'old fetch',
                                       'check_p8a.py': 'old checks'})
        m.resume_appendix6(work/'generation.json', v6, 'S4')
        tables[f'z{z}_q{charge}.npz'] = m.reference(work/f'z{z}_q{charge}.npz', 'fixture', 'synthetic node table', '1', 'element',
            Z=z, charge=charge, direct=m.reference(work/'direct.npz', 'fixture', 'synthetic direct checks', '1', 'element'),
            direct_status=m.reference(work/'direct_status.json', 'fixture', 'synthetic coverage', '1', 'element'))
    manifest = dict(materials={})
    for material in ('lung', 'bone'):  # Shared Ca must be calculated only once.
        folder = cache/material
        definition = dict(water, material=material, weights={'20': 1.}, atomic_weights={'20': 1.})
        m.write(folder/'definition.json', definition)
        m.write(folder/'elastic.json', tables)
        np.savez(folder/'gs.npz', sentinel=np.ones(1), material=np.array(material))
        np.savez(folder/'stopping.npz', sentinel=np.ones(1), material=np.array(material))
        row = {k: m.reference(folder/(k+('.json' if k in ('definition', 'elastic') else '.npz')),
                              'fixture', 'synthetic input', '1', material) for k in ('definition', 'elastic', 'gs', 'stopping')}
        m.write(folder/'gs.identity.json', dict(definition_sha256=row['definition']['sha256'],
                 elemental={n: r['sha256'] for n, r in tables.items()}, implementations=m.implementation_hashes()))
        manifest['materials'][material] = row
    m.write(out/'manifest.json', manifest)
    monkeypatch.setattr(generate, 'load_manifest', lambda path, **kw: json.loads(Path(path).read_text()))
    calls = []
    def run(command, **kwargs):
        work = Path(kwargs['cwd'])
        assert 'replacements' in work.parts and kwargs['timeout'] == 600
        calls.append(float.fromhex(work.name))
        (work/'dcs_fixture.dat').write_bytes(b'' if state['mode'] == 'timeout' else b'synthetic DCS')
        if state['mode'] == 'timeout': raise subprocess.TimeoutExpired(command, 600)
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(subprocess, 'run', run)
    def parse(path, stdout):
        energy = float.fromhex(path.parent.name)
        row = synthetic_case_row(20, 1, energy)
        amplitude = 1.003 if state['mode'] == 'interpolation' else 1.001 if state['mode'] == 'cdf' else 1.
        for k in ('dcs', 'table_xs', 'series_xs'): row[k] *= amplitude
        if state['mode'] == 'angular': row['error'][:] = .00101
        if state['mode'] in ('tst', 'bad_tst'):
            row['warnings'] = np.array(['WARNING: TST = '+('1E-5' if state['mode'] == 'tst' else '1.01E-4')+'. Check convergence.'])
        return row
    monkeypatch.setattr(generate, 'parse', parse)
    frozen = {p: p.read_bytes() for root in (out, cache) for p in root.rglob('*') if p.is_file()}
    return dict(out=out, cache=cache, tables=tables, exe=exe, state=state, calls=calls, frozen=frozen, original=original)


@pytest.mark.parametrize('mode', ['pass', 'tst'])
def test_appendix10_replacement_pass_resolves_shared_timeout_and_preserves_all_old_bytes(replacement_job, mode):
    job = replacement_job; job['state']['mode'] = mode
    report = generate.replace_timeouts(job['out'], job['cache'])
    assert report['status'] == 'pass' and job['calls'] == [job['original']*(1+1e-3)]
    pair = report['groups']['z20_q1.npz']['pairs'][0]
    assert pair['original']['status'] == 'not-evaluated' and pair['status'] == 'pass'
    assert pair['checks']['angular']['status'] == pair['checks']['interpolation']['status'] == 'pass'
    groups = check.load_replacement_report(job['out'])
    before = check.atomic_checks(job['tables'])
    assert before['status'] == 'not-evaluated'
    for _ in ('lung', 'bone'):
        result = check.atomic_checks(job['tables'], groups)
        assert result['status'] == 'pass'
        omission = result['direct_omissions'][1]
        assert omission['not_evaluated_count'] == 1 and omission['unresolved_count'] == 0
        assert omission['replacement_count'] == 1 and omission['omitted'][0]['status'] == 'not-evaluated'
        assert any(r.get('kind') == 'direct-timeout' and r['status'] == 'not-evaluated' for r in result['rows'])
        if mode == 'tst':
            assert any(r['accepted_point_count'] == 1 and r['max_TST'] == 1e-5 for r in result['tst_warnings'])
    generate.replace_timeouts(job['out'], job['cache'])
    assert len(job['calls']) == 1  # Real case() reuses its sealed raw replacement.
    assert all(p.read_bytes() == data for p, data in job['frozen'].items())


@pytest.mark.parametrize('mode', ['timeout', 'angular', 'interpolation', 'cdf', 'bad_tst'])
def test_appendix10_replacement_timeout_or_failure_stops_without_another_shift(replacement_job, mode):
    job = replacement_job; job['state']['mode'] = mode
    with pytest.raises(RuntimeError, match='stop without another shift'):
        generate.replace_timeouts(job['out'], job['cache'])
    report = json.loads((job['out']/'elsepa_replacements_v12.json').read_text())
    expected = 'not-evaluated' if mode == 'timeout' else 'fail'
    assert report['status'] == expected
    pair = report['groups']['z20_q1.npz']['pairs'][0]
    assert pair['status'] == expected and pair['original']['status'] == 'not-evaluated'
    result = check.atomic_checks(job['tables'], check.load_replacement_report(job['out']))
    assert result['status'] == (generate.OMISSION_STATUS if mode == 'timeout' else expected)
    assert result['direct_omissions'][1]['unresolved_count'] == 1
    if mode == 'timeout':
        assert generate.replace_timeouts(job['out'], job['cache'])['status'] == generate.OMISSION_STATUS
    else:
        with pytest.raises(RuntimeError, match='stop without another shift'):
            generate.replace_timeouts(job['out'], job['cache'])
    assert job['calls'] == [job['original']*(1+1e-3)]
    assert all(p.read_bytes() == data for p, data in job['frozen'].items())


@pytest.mark.parametrize('fault', ['shift', 'original', 'duplicate', 'charge', 'data_hash', 'declared_pass'])
def test_appendix10_checker_rejects_bad_pair_or_rechecks_false_pass(replacement_job, fault):
    job = replacement_job
    report = generate.replace_timeouts(job['out'], job['cache'])
    groups = report['groups']
    pair = groups['z20_q1.npz']['pairs'][0]
    if fault == 'shift': pair['energy_MeV'] *= 1+1e-3  # No second shift is permitted.
    if fault == 'original': pair['original']['energy_MeV'] *= .99
    if fault == 'duplicate': groups['z20_q1.npz']['pairs'].append(copy.deepcopy(pair))
    if fault == 'charge': groups['z20_q1.npz']['charge'] = -1
    if fault == 'data_hash': Path(pair['data']['path']).write_bytes(b'changed')
    if fault == 'declared_pass':
        p = Path(pair['data']['path'])
        with np.load(p) as f: data = {k: f[k].copy() for k in f.files}
        data['error'][:] = .00101
        np.savez(p, **data)
        pair['data'] = m.reference(p, 'fixture', 'false pass injection', '1', 'element')
        assert check.atomic_checks(job['tables'], groups)['status'] == 'fail'
    else:
        with pytest.raises(ValueError, match='replacement|hash mismatch'):
            check.atomic_checks(job['tables'], groups)


@pytest.mark.parametrize('fault', ['node', 'gs', 'definition', 'executable', 'original_timeout', 'migration', 'helper'])
def test_appendix10_resume_rejects_changed_conditions_and_evidence(replacement_job, monkeypatch, fault):
    job = replacement_job
    generate.replace_timeouts(job['out'], job['cache'])
    if fault == 'node': Path(job['tables']['z20_q1.npz']['path']).write_bytes(b'changed')
    if fault == 'gs': (job['cache']/'lung'/'gs.npz').write_bytes(b'changed')
    if fault == 'definition': (job['cache']/'lung'/'definition.json').write_bytes(b'changed')
    if fault == 'executable': job['exe'].write_bytes(b'changed')
    if fault == 'original_timeout':
        p = job['cache']/'elsepa'/'z20_q1'/'direct'/(job['original'].hex()+'.timed-out')/'dcs_failed.dat'
        p.write_bytes(b'changed')
    if fault == 'migration':
        p = job['cache']/'elsepa'/'z20_q1'/'generation.resume_v6.json'; p.write_bytes(p.read_bytes()+b' ')
    if fault == 'helper':
        implementations = m.implementation_hashes(); implementations['gs_backend.py'] = 'changed'
        monkeypatch.setattr(generate, 'implementation_hashes', lambda: implementations)
    with pytest.raises(ValueError): generate.replace_timeouts(job['out'], job['cache'])
    assert len(job['calls']) == 1
    if fault in ('node', 'gs', 'definition', 'original_timeout', 'migration'):
        with pytest.raises(ValueError, match='hash mismatch'): check.load_replacement_report(job['out'])


def test_appendix10_cli_runs_only_replacements(monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(sys, 'argv', ['generate_elsepa_p8a', '--replace-timeouts'])
    monkeypatch.setattr(generate, 'replace_timeouts', lambda: calls.append('replace'))
    monkeypatch.setattr(generate, 'generate', lambda: pytest.fail('full S4 regeneration is not requested'))
    generate.main()
    assert calls == ['replace']


@pytest.mark.parametrize('mode', ['timeout', 'interpolation'])
def test_appendix10_failure_does_not_start_the_next_replacement(replacement_job, mode):
    job = replacement_job
    name = 'z20_q1.npz'
    work = job['cache']/'elsepa'/'z20_q1'
    ref = job['tables'][name]
    coverage = json.loads((work/'direct_status.json').read_text())
    extra = float(coverage['expected_energy_MeV'][-2])
    raw = work/'direct'/extra.hex()
    identity = saved_elsepa_job((job['exe'].parent/'source', job['exe'], raw), 20, 1, extra)
    (raw/'dcs_failed.dat').write_bytes(b'')
    record = generate._retain_timeout(raw, identity, 'second synthetic original timeout')
    coverage['omitted'].insert(0, dict(status='not-evaluated', reason=record['reason'], Z=20, charge=1,
                                    energy_MeV=extra, timeout_record=str(raw/'timeout.json'), timeout_seconds=600))
    coverage['completed_count'] -= 1
    coverage['not_evaluated_count'] += 1
    with np.load(work/'direct.npz') as f: direct = {k: f[k].copy() for k in f.files}
    keep = direct['energy_MeV'] != extra
    for k in ('energy_MeV', 'dcs', 'error', 'series_xs', 'table_xs', 'warnings', 'notices'):
        direct[k] = direct[k][keep]
    np.savez(work/'direct.npz', **direct)
    m.write(work/'direct_status.json', coverage)
    ref['direct'] = m.reference(work/'direct.npz', 'fixture', 'synthetic direct', '1', 'element')
    ref['direct_status'] = m.reference(work/'direct_status.json', 'fixture', 'synthetic coverage', '1', 'element')
    manifest = json.loads((job['out']/'manifest.json').read_text())
    for material in ('lung', 'bone'):
        path = job['cache']/material/'elastic.json'
        m.write(path, job['tables'])
        manifest['materials'][material]['elastic'] = m.reference(path, 'fixture', 'synthetic elastic', '1', material)
    m.write(job['out']/'manifest.json', manifest)
    job['state']['mode'] = mode
    for _ in range(2):
        with pytest.raises(RuntimeError, match='stop without another shift'):
            generate.replace_timeouts(job['out'], job['cache'])
    assert job['calls'] == [extra*(1+1e-3)]
    report = json.loads((job['out']/'elsepa_replacements_v12.json').read_text())
    assert len(report['groups'][name]['pairs']) == 1
    assert not (work/'replacements'/job['original'].hex()).exists()


@pytest.fixture
def recorded_omission_job(replacement_job):
    job = replacement_job
    es = generate.energies()
    tables = job['tables']
    original = generate.RECORDED_OMISSION['original_energy_MeV']
    replacement = generate.RECORDED_OMISSION['replacement_energy_MeV']
    for charge in (-1, 1):
        work = job['cache']/'elsepa'/f'z17_q{charge}'
        row = synthetic_case_row(17, charge); row.pop('header')
        m.write(work/'placeholder.json', {})
        np.savez(work/f'z17_q{charge}.npz', **generate.stack_rows([row]*len(es), 17, charge, es))
        expected = generate.direct_energies(es, 17)
        completed = expected[expected != original] if charge == 1 else expected
        omitted = []
        if charge == 1:
            assert original in expected
            for raw, t in ((work/'direct'/original.hex(), original),
                           (work/'replacements'/original.hex()/replacement.hex(), replacement)):
                identity = saved_elsepa_job((job['exe'].parent/'source', job['exe'], raw), 17, 1, t)
                (raw/'dcs_failed.dat').write_bytes(b'')
                generate._retain_timeout(raw, identity, 'synthetic saved chlorine timeout')
            omitted = [dict(status='not-evaluated', reason='ELSEPA case timed out', Z=17, charge=1,
                            energy_MeV=original, timeout_record=str(work/'direct'/original.hex()/'timeout.json'), timeout_seconds=600)]
        np.savez(work/'direct.npz', **generate.stack_rows([row]*len(completed), 17, charge, completed))
        m.write(work/'direct_status.json', dict(Z=17, charge=charge, expected_energy_MeV=expected.tolist(),
                completed_count=len(completed), not_evaluated_count=len(omitted), omitted=omitted,
                status='not-evaluated' if omitted else 'pass'))
        m.write(work/'generation.json', dict(Z=17, charge=charge, IHEF=0, energy_MeV=es.tolist(),
                executable_sha256=m.sha(job['exe']), implementations=m.implementation_hashes()))
        tables[f'z17_q{charge}.npz'] = m.reference(work/f'z17_q{charge}.npz', 'fixture', 'synthetic chlorine nodes', '1', 'element',
            Z=17, charge=charge, direct=m.reference(work/'direct.npz', 'fixture', 'synthetic chlorine direct', '1', 'element'),
            direct_status=m.reference(work/'direct_status.json', 'fixture', 'synthetic chlorine coverage', '1', 'element'))
    manifest = json.loads((job['out']/'manifest.json').read_text())
    for material in ('lung', 'bone'):
        folder = job['cache']/material
        d = json.loads((folder/'definition.json').read_text())
        d.update(weights={'17': .5, '20': .5}, atomic_weights={'17': 1., '20': 1.})
        m.write(folder/'definition.json', d)
        m.write(folder/'elastic.json', tables)
        for field in ('definition', 'elastic'):
            manifest['materials'][material][field] = m.reference(folder/(field+'.json'), 'fixture', 'synthetic mixed definition', '1', material)
        m.write(folder/'gs.identity.json', dict(definition_sha256=m.sha(folder/'definition.json'),
                elemental={n: ref['sha256'] for n, ref in tables.items()}, implementations=m.implementation_hashes()))
    m.write(job['out']/'manifest.json', manifest)
    # Model the immutable appendix 10 execution: both migrations were written
    # before chlorine timed out, and calcium has no replacement attempt yet.
    old_impl = dict(m.implementation_hashes(), **{'generate_elsepa_p8a.py': 'v10 generator', 'check_p8a.py': 'v10 checker'})
    old_inputs = {str(job['out']/'manifest.json'): m.sha(job['out']/'manifest.json')}
    for z in (17, 20):
        name = f'z{z}_q1.npz'; ref = tables[name]
        work = job['cache']/'elsepa'/f'z{z}_q1'
        identity = dict(Z=z, charge=1, IHEF=0, energy_MeV=es.tolist(), executable_sha256=m.sha(job['exe']), implementations=old_impl)
        generate.resume_replacements(work/'generation.json', identity,
            dict(nodes=ref['sha256'], direct=ref['direct']['sha256'], coverage=ref['direct_status']['sha256']))
        old_inputs[str(work/'generation.resume_v10.json')] = m.sha(work/'generation.resume_v10.json')
    raw = job['cache']/'elsepa'/'z17_q1'/'replacements'/original.hex()/replacement.hex()/'timeout.json'
    old = dict(authorization='P8a appendix 10', status='not-evaluated', inputs=old_inputs,
               identity=dict(implementations=old_impl, executable_sha256=m.sha(job['exe']), replacement_factor=1+1e-3, timeout_seconds=600),
               groups={'z17_q1.npz': dict(pairs=[dict(original=omitted[0], status='not-evaluated', energy_MeV=replacement,
                      timeout_seconds=600, timeout_record=m.reference(raw, 'fixture', 'v10 timeout', 'MeV', 'element'))])})
    m.write(job['out']/'elsepa_replacements_v10.json', old)
    job['frozen'] = {p: p.read_bytes() for root in (job['out'], job['cache']) for p in root.rglob('*') if p.is_file()}
    return job


def test_appendix11_skips_only_saved_chlorine_and_finishes_calcium(recorded_omission_job):
    job = recorded_omission_job
    report = generate.replace_timeouts(job['out'], job['cache'])
    assert report['status'] == generate.OMISSION_STATUS
    assert job['calls'] == [job['original']*(1+1e-3)]  # Only Ca; Cl is never executed.
    groups = check.load_replacement_report(job['out'])
    chlorine = groups['z17_q1.npz']['pairs'][0]
    assert chlorine['status'] == chlorine['original']['status'] == 'not-evaluated'
    assert groups['z20_q1.npz']['pairs'][0]['status'] == 'pass'
    for _ in ('lung', 'bone'):
        result = check.atomic_checks(job['tables'], groups)
        assert result['status'] == generate.OMISSION_STATUS != 'pass'
        assert result['recorded_omissions'] == [generate.RECORDED_OMISSION]
        omitted = {r['Z']: r for r in result['direct_omissions'] if r['not_evaluated_count']}
        assert omitted[17]['unresolved_count'] == omitted[17]['recorded_omission_count'] == 1
        assert omitted[20]['unresolved_count'] == omitted[20]['recorded_omission_count'] == 0
    generate.replace_timeouts(job['out'], job['cache'])
    assert len(job['calls']) == 1
    assert all(p.read_bytes() == data for p, data in job['frozen'].items())


@pytest.mark.parametrize('mode', ['timeout', 'angular', 'interpolation', 'cdf', 'bad_tst'])
def test_calcium_numerical_failures_stop_but_saved_fixed_timeout_is_allowed(recorded_omission_job, mode):
    job = recorded_omission_job; job['state']['mode'] = mode
    with pytest.raises(RuntimeError, match='stop without another shift'):
        generate.replace_timeouts(job['out'], job['cache'])
    assert len(job['calls']) == 1
    groups = check.load_replacement_report(job['out'])
    assert check.atomic_checks(job['tables'], groups)['status'] == (generate.OMISSION_STATUS if mode == 'timeout' else 'fail')
    if mode == 'timeout':
        assert generate.replace_timeouts(job['out'], job['cache'])['status'] == generate.OMISSION_STATUS
    else:
        with pytest.raises(RuntimeError, match='stop without another shift'):
            generate.replace_timeouts(job['out'], job['cache'])
    assert len(job['calls']) == 1


@pytest.mark.parametrize('fault', ['Z', 'charge', 'original_energy_MeV', 'replacement_energy_MeV'])
def test_appendix11_scope_is_exact_and_other_timeouts_are_not_allowed(fault):
    scope = dict(generate.RECORDED_OMISSION)
    scope[fault] = scope[fault]+1 if fault in ('Z', 'charge') else np.nextafter(scope[fault], np.inf)
    assert not check.fixed_omission(scope['original_energy_MeV'], scope['replacement_energy_MeV'], scope['Z'], scope['charge'])


@pytest.mark.parametrize('fault', ['archive', 'v10_report', 'replacement_identity'])
def test_appendix11_tampered_legacy_evidence_is_rejected(recorded_omission_job, fault):
    job = recorded_omission_job
    original = generate.RECORDED_OMISSION['original_energy_MeV']
    replacement = generate.RECORDED_OMISSION['replacement_energy_MeV']
    raw = job['cache']/'elsepa'/'z17_q1'/'replacements'/original.hex()/replacement.hex()
    if fault == 'archive': (raw.with_name(raw.name+'.timed-out')/'dcs_failed.dat').write_bytes(b'changed')
    if fault == 'replacement_identity':
        record = json.loads((raw/'timeout.json').read_text()); record['identity']['charge'] = -1
        m.write(raw/'timeout.json', record)
    if fault == 'v10_report':
        path = job['out']/'elsepa_replacements_v10.json'
        old = json.loads(path.read_text()); old['identity']['timeout_seconds'] = 601; m.write(path, old)
    with pytest.raises(ValueError): generate.replace_timeouts(job['out'], job['cache'])
    assert not job['calls']


@pytest.fixture
def scoped_s6_snapshot(recorded_omission_job):
    job = recorded_omission_job
    generate.replace_timeouts(job['out'], job['cache'])
    groups = check.load_replacement_report(job['out'])
    full = check.atomic_checks(job['tables'], groups)
    resolved = check.atomic_checks({n: r for n, r in job['tables'].items() if r['Z'] == 20}, groups)
    materials = {}
    for material in m.MATERIALS:
        materials[material] = {k: {'status': 'pass'} for k in ('pair', 'shells', 'mixture', 'pegs_report', 'gs')}
        materials[material]['interpolation'] = {'photoelectric': {'status': 'pass'}}
        materials[material]['atomic'] = copy.deepcopy(full if material == 'lung' else resolved)
    materials['water']['compatibility'] = {'status': 'pass'}
    return dict(status=generate.OMISSION_STATUS, materials=materials,
                recorded_omissions=[dict(generate.RECORDED_OMISSION)], manifest_sha256=m.sha(job['out']/'manifest.json'))


def test_appendix11_s6_aggregation_and_s7_allow_only_fixed_scope(scoped_s6_snapshot):
    snapshot = scoped_s6_snapshot
    assert check.material_checks_allow_s7(snapshot)
    assert check.aggregate_recorded_omission(['pass', generate.OMISSION_STATUS]) == generate.OMISSION_STATUS
    assert check.aggregate_recorded_omission(['fail', generate.OMISSION_STATUS]) == 'fail'
    assert check.aggregate_recorded_omission(['not-evaluated', generate.OMISSION_STATUS]) == 'not-evaluated'
    assert check.aggregate(['not-evaluated']) == 'not-evaluated'  # Ordinary omissions stay rejected.
    assert check.material_checks_allow_s7({'status': 'pass'})  # Existing full-pass gate.


@pytest.mark.parametrize('fault', ['ordinary_status', 'wrong_scope', 'other_failure', 'other_omission', 'atomic_failure',
                                   'unknown_row', 'wrong_energy', 'calcium_failure', 'missing_summary', 'missing_pair'])
def test_appendix11_s7_rejects_other_incomplete_or_failed_checks(scoped_s6_snapshot, fault):
    snapshot = copy.deepcopy(scoped_s6_snapshot)
    lung = snapshot['materials']['lung']
    if fault == 'ordinary_status': snapshot['status'] = 'not-evaluated'
    if fault == 'wrong_scope': snapshot['recorded_omissions'][0]['Z'] = 20
    if fault == 'other_failure': snapshot['materials']['bone']['gs']['status'] = 'fail'
    if fault == 'other_omission': snapshot['materials']['bone']['pegs_report']['status'] = 'not-evaluated'
    if fault == 'atomic_failure': lung['atomic']['rows'][0]['status'] = 'fail'
    if fault == 'unknown_row': lung['atomic']['rows'].append({'status': 'not-evaluated', 'kind': 'other'})
    chlorine = next(g for g in lung['atomic']['direct_omissions'] if g['Z'] == 17 and g['charge'] == 1)
    calcium = next(g for g in lung['atomic']['direct_omissions'] if g['Z'] == 20 and g['charge'] == 1)
    if fault == 'wrong_energy': chlorine['replacement_pairs'][0]['energy_MeV'] = np.nextafter(chlorine['replacement_pairs'][0]['energy_MeV'], np.inf)
    if fault == 'calcium_failure': calcium['replacement_pairs'][0]['checks']['status'] = 'fail'
    if fault == 'missing_summary': lung['atomic']['recorded_omissions'] = []
    if fault == 'missing_pair': chlorine['replacement_pairs'] = []
    assert not check.material_checks_allow_s7(snapshot)


def test_appendix11_estimate_entry_accepts_scoped_status_before_any_calculation(scoped_s6_snapshot, recorded_omission_job, monkeypatch):
    from prototypes.mv_electron import run_p3a
    job = recorded_omission_job
    m.write(job['out']/'checks.json', scoped_s6_snapshot)
    monkeypatch.setattr(estimate, 'verify_protected', lambda: 0)
    monkeypatch.setattr(estimate, 'load_manifest', lambda *a: {})
    def stop_before_calculation(): raise RuntimeError('S7 gate accepted; stop before spectrum or simulation')
    monkeypatch.setattr(run_p3a, 'load_spectrum', stop_before_calculation)
    with pytest.raises(RuntimeError, match='S7 gate accepted'):
        estimate.estimate(job['out'], job['cache'])
    scoped_s6_snapshot['materials']['bone']['gs']['status'] = 'fail'
    m.write(job['out']/'checks.json', scoped_s6_snapshot)
    with pytest.raises(ValueError, match='material checks have not passed'):
        estimate.estimate(job['out'], job['cache'])


def test_appendix11_check_cli_preserves_distinct_accepted_status(monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, 'argv', ['check_p8a'])
    monkeypatch.setattr(check, 'check', lambda: {'status': generate.OMISSION_STATUS})
    check.main()
    assert capsys.readouterr().out.strip() == generate.OMISSION_STATUS
    monkeypatch.setattr(check, 'check', lambda: {'status': 'not-evaluated'})
    with pytest.raises(SystemExit) as error: check.main()
    assert error.value.code == 1


@pytest.fixture
def two_recorded_omissions_job(recorded_omission_job):
    job = recorded_omission_job
    calcium = generate.RECORDED_OMISSIONS[1]
    assert calcium['original_energy_MeV'] == job['original']
    raw = job['cache']/'elsepa'/'z20_q1'/'replacements'/job['original'].hex()/calcium['replacement_energy_MeV'].hex()
    identity = saved_elsepa_job((job['exe'].parent/'source', job['exe'], raw), 20, 1, calcium['replacement_energy_MeV'])
    (raw/'dcs_failed.dat').write_bytes(b'')
    generate._retain_timeout(raw, identity, 'synthetic saved calcium replacement timeout')
    old10_path = job['out']/'elsepa_replacements_v10.json'
    old10 = json.loads(old10_path.read_text())
    old_impl = dict(m.implementation_hashes(), **{'generate_elsepa_p8a.py': 'v11 generator', 'check_p8a.py': 'v11 checker'})
    inputs = dict(old10['inputs'], **{str(old10_path): m.sha(old10_path)})
    groups = {}
    for scope in generate.RECORDED_OMISSIONS:
        z, q = scope['Z'], scope['charge']
        name = f'z{z}_q{q}.npz'; ref = job['tables'][name]
        work = job['cache']/'elsepa'/f'z{z}_q{q}'
        identity = dict(Z=z, charge=q, IHEF=0, energy_MeV=generate.energies().tolist(),
                        executable_sha256=m.sha(job['exe']), implementations=old_impl)
        generate.resume_replacements(work/'generation.json', identity,
            dict(nodes=ref['sha256'], direct=ref['direct']['sha256'], coverage=ref['direct_status']['sha256']), appendix=11)
        inputs[str(work/'generation.resume_v11.json')] = m.sha(work/'generation.resume_v11.json')
        coverage = json.loads((work/'direct_status.json').read_text())
        raw = work/'replacements'/scope['original_energy_MeV'].hex()/scope['replacement_energy_MeV'].hex()/'timeout.json'
        pair = dict(original=coverage['omitted'][0], energy_MeV=scope['replacement_energy_MeV'], timeout_seconds=600,
                    status='not-evaluated', reason='saved appendix 11 timeout',
                    timeout_record=m.reference(raw, 'fixture', 'saved replacement timeout', 'MeV', 'element'))
        groups[name] = dict(Z=z, charge=q, nodes_sha256=ref['sha256'], direct_sha256=ref['direct']['sha256'],
                            coverage_sha256=ref['direct_status']['sha256'], pairs=[pair])
        if z == 17: groups[name]['recorded_omission_policy'] = scope
    m.write(job['out']/'elsepa_replacements_v11.json', dict(authorization='P8a appendix 11', status='not-evaluated',
            identity=dict(implementations=old_impl, executable_sha256=m.sha(job['exe']), replacement_factor=1+1e-3,
                          timeout_seconds=600, recorded_omission=generate.RECORDED_OMISSION), inputs=inputs, groups=groups))
    job['frozen'] = {p: p.read_bytes() for root in (job['out'], job['cache']) for p in root.rglob('*') if p.is_file()}
    return job


def test_appendix12_skips_both_saved_timeouts_without_any_calculation(two_recorded_omissions_job, monkeypatch):
    job = two_recorded_omissions_job
    def forbidden(*a, **kw): pytest.fail('both replacement attempts already timed out; no new calculation')
    monkeypatch.setattr(generate, 'case', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    for _ in range(2):
        report = generate.replace_timeouts(job['out'], job['cache'])
        assert report['status'] == generate.OMISSION_STATUS != 'pass'
        assert report['identity']['recorded_omissions'] == generate.RECORDED_OMISSIONS
    assert not job['calls']
    for scope in generate.RECORDED_OMISSIONS:
        group = report['groups'][f'z{scope["Z"]}_q1.npz']
        assert len(group['pairs']) == 1
        pair = group['pairs'][0]
        assert pair['status'] == pair['original']['status'] == 'not-evaluated'
        assert pair['energy_MeV'] == scope['replacement_energy_MeV']
        assert 'data' not in pair  # No fabricated successful direct calculation.
        assert (job['cache']/'elsepa'/f'z{scope["Z"]}_q1'/'generation.resume_v12.json').exists()
    assert all(p.read_bytes() == data for p, data in job['frozen'].items())


@pytest.fixture
def two_omissions_s6_snapshot(two_recorded_omissions_job):
    job = two_recorded_omissions_job
    generate.replace_timeouts(job['out'], job['cache'])
    groups = check.load_replacement_report(job['out'])
    lung = check.atomic_checks(job['tables'], groups)
    bone = check.atomic_checks({n: r for n, r in job['tables'].items() if r['Z'] == 20}, groups)
    passed = check.atomic_checks({n: r for n, r in job['tables'].items() if r['charge'] == -1}, groups)
    assert lung['recorded_omissions'] == generate.RECORDED_OMISSIONS
    assert bone['recorded_omissions'] == [generate.RECORDED_OMISSIONS[1]]
    materials = {}
    for material in m.MATERIALS:
        materials[material] = {k: {'status': 'pass'} for k in ('pair', 'shells', 'mixture', 'pegs_report', 'gs')}
        materials[material]['interpolation'] = {'photoelectric': {'status': 'pass'}}
        materials[material]['atomic'] = copy.deepcopy(lung if material == 'lung' else bone if material == 'bone' else passed)
    materials['water']['compatibility'] = {'status': 'pass'}
    recorded = check.canonical_recorded_omissions([scope for r in materials.values() for scope in r['atomic']['recorded_omissions']])
    return dict(status=check.aggregate_recorded_omission(r['atomic']['status'] for r in materials.values()),
                materials=materials, recorded_omissions=recorded, manifest_sha256=m.sha(job['out']/'manifest.json'))


def test_appendix12_s6_s7_keep_two_unique_points_with_calcium_shared(two_omissions_s6_snapshot):
    snapshot = two_omissions_s6_snapshot
    assert snapshot['status'] == generate.OMISSION_STATUS
    assert snapshot['recorded_omissions'] == generate.RECORDED_OMISSIONS
    assert len(snapshot['recorded_omissions']) == 2
    assert check.material_checks_allow_s7(snapshot)
    for material in ('lung', 'bone'):
        atomic = snapshot['materials'][material]['atomic']
        ca = next(g for g in atomic['direct_omissions'] if g['Z'] == 20 and g['charge'] == 1)
        assert ca['recorded_omission_count'] == ca['unresolved_count'] == 1
        assert ca['replacement_pairs'][0]['status'] == ca['replacement_pairs'][0]['original']['status'] == 'not-evaluated'


@pytest.mark.parametrize('z', [17, 20])
@pytest.mark.parametrize('field', ['Z', 'charge', 'original_energy_MeV', 'replacement_energy_MeV'])
def test_appendix12_exact_two_point_allowlist(z, field):
    scope = dict(next(r for r in generate.RECORDED_OMISSIONS if r['Z'] == z))
    assert check.fixed_omission(scope['original_energy_MeV'], scope['replacement_energy_MeV'], z, 1)
    scope[field] = scope[field]+1 if field == 'Z' else -1 if field == 'charge' else np.nextafter(scope[field], np.inf)
    assert not check.fixed_omission(scope['original_energy_MeV'], scope['replacement_energy_MeV'], scope['Z'], scope['charge'])


@pytest.mark.parametrize('fault', ['extra_scope', 'duplicate_scope', 'missing_scope', 'other_failure', 'other_omission',
                                   'wrong_energy', 'wrong_charge', 'numerical_failure', 'missing_pair', 'bad_count'])
def test_appendix12_s7_rejects_anything_outside_the_two_scopes(two_omissions_s6_snapshot, fault):
    snapshot = copy.deepcopy(two_omissions_s6_snapshot)
    atomic = snapshot['materials']['lung']['atomic']
    ca = next(g for g in atomic['direct_omissions'] if g['Z'] == 20 and g['charge'] == 1)
    if fault == 'extra_scope': snapshot['recorded_omissions'].append(dict(generate.RECORDED_OMISSION, Z=18))
    if fault == 'duplicate_scope': snapshot['recorded_omissions'].append(dict(generate.RECORDED_OMISSIONS[1]))
    if fault == 'missing_scope': snapshot['recorded_omissions'].pop()
    if fault == 'other_failure': snapshot['materials']['lead']['gs']['status'] = 'fail'
    if fault == 'other_omission': snapshot['materials']['lead']['pegs_report']['status'] = 'not-evaluated'
    if fault == 'wrong_energy': ca['replacement_pairs'][0]['energy_MeV'] *= 1+1e-3
    if fault == 'wrong_charge': ca['charge'] = -1
    if fault == 'numerical_failure': atomic['rows'][0]['status'] = 'fail'
    if fault == 'missing_pair': ca['replacement_pairs'] = []
    if fault == 'bad_count': ca['unresolved_count'] = 0
    assert not check.material_checks_allow_s7(snapshot)


@pytest.mark.parametrize('fault', ['legacy_report', 'legacy_migration', 'original_timeout', 'replacement_timeout'])
def test_appendix12_preserved_history_and_timeout_evidence_are_required(two_recorded_omissions_job, fault):
    job = two_recorded_omissions_job
    generate.replace_timeouts(job['out'], job['cache'])
    ca = generate.RECORDED_OMISSIONS[1]
    work = job['cache']/'elsepa'/'z20_q1'
    if fault == 'legacy_report': path = job['out']/'elsepa_replacements_v11.json'
    elif fault == 'legacy_migration': path = work/'generation.resume_v11.json'
    elif fault == 'original_timeout':
        path = (work/'direct'/ca['original_energy_MeV'].hex()).with_name(ca['original_energy_MeV'].hex()+'.timed-out')/'dcs_failed.dat'
    else:
        path = work/'replacements'/ca['original_energy_MeV'].hex()/(ca['replacement_energy_MeV'].hex()+'.timed-out')/'dcs_failed.dat'
    path.write_bytes(path.read_bytes()+b' ')
    if fault == 'replacement_timeout':
        with pytest.raises(ValueError):
            check.atomic_checks(job['tables'], check.load_replacement_report(job['out']))
    else:
        with pytest.raises(ValueError): check.load_replacement_report(job['out'])
    with pytest.raises(ValueError): generate.replace_timeouts(job['out'], job['cache'])
    assert not job['calls']


def test_appendix12_estimate_entry_accepts_only_valid_two_point_result(two_omissions_s6_snapshot, two_recorded_omissions_job, monkeypatch):
    from prototypes.mv_electron import run_p3a
    job = two_recorded_omissions_job
    m.write(job['out']/'checks.json', two_omissions_s6_snapshot)
    monkeypatch.setattr(estimate, 'verify_protected', lambda: 0)
    monkeypatch.setattr(estimate, 'load_manifest', lambda *a: {})
    def accepted(): raise RuntimeError('accepted before any spectrum/cache/simulation access')
    monkeypatch.setattr(run_p3a, 'load_spectrum', accepted)
    with pytest.raises(RuntimeError, match='accepted before'):
        estimate.estimate(job['out'], job['cache'])
    two_omissions_s6_snapshot['materials']['lead']['gs']['status'] = 'fail'
    m.write(job['out']/'checks.json', two_omissions_s6_snapshot)
    with pytest.raises(ValueError, match='material checks have not passed'):
        estimate.estimate(job['out'], job['cache'])
