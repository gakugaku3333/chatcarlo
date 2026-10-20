"""P8a approximation reports and preregistered fluorescence sensitivity.

Only fluorescence controls the next-stage decision. Pair/positron and binding
fractions are incident-spectrum interaction indicators, never dose upper bounds.
External calculations occur only on explicit invocation of this module.
"""
import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from .materials_p8a import (ROOT, OUT, CACHE, finite, energy_grid, kn_total,
                            photo_interp, pair_interp, load_manifest,
                            load_reference, sha, write, resume, verify_protected, implementation_hashes)
from .pegs_p8a import EGS, pegs_input, shell_probabilities

PILOT_N = 1_000_000
MAX_N = 100_000_000
SEM_TARGET = .001
LIMIT = .003
# Distinct independent streams, including all-water control and pilot/production.
SEEDS = dict(control=202610191, reference=202610192,
             pilot_off=202610193, pilot_on=202610194,
             production_off=202610195, production_on=202610196)
# No measured claims: this paragraph is a literature/code comparison only.
BREMS_REVIEW = (
    'The prototype uses Bethe–Heitler energy sharing with the Butcher–Messel '
    'screening-function approximation. Its electron-electron factor uses a Z-dependent Coulomb correction '
    'in its radiation logarithm (SLAC-R-730, section 2.7; implemented in '
    'bremsstrahlung_p3r.py). Its total is renormalized to ESTAR radiative '
    'stopping power. EGS5 additionally uses its aprime.data correction to the '
    'bremsstrahlung shape; the prototype does not use this research-only table. '
    'Normalization to the same stopping power constrains the energy integral '
    'and does not establish the differential spectrum for Z=82. P8a records '
    'this limitation: its screening branch has no full high-Z Coulomb subtraction. Generating material-specific bremsstrahlung tables and '
    'testing the lead spectrum belong to P8b. Source: '
    'https://www.slac.stanford.edu/pubs/slacreports/reports16/slac-r-730.pdf ; '
    'local source: prototypes/mv_electron/bremsstrahlung_p3r.py and '
    'docs/egs5_crosscheck/egs5/pegs/pegs5.f (BREMDZ, FCOULC, APRIM).'
)


def batch_statistics(dose):
    """Average correlated bins inside each independent batch before SEM."""
    d = finite(dose, 'dose')
    if d.ndim != 2 or d.shape[1] != 55 or d.shape[0] < 2 or np.any(d < 0):
        raise ValueError('expected independent batches by 55 nonnegative bins')
    values = np.column_stack((d[:, 45:50].mean(axis=1),  # A: 9--10 cm water
                              d[:, 50:55].mean(axis=1),  # B: entire lead slab
                              d[:, 10:40].mean(axis=1)))  # fixed Dref: 2--8 cm
    return values.mean(axis=0), values.std(axis=0, ddof=1)/np.sqrt(len(values))


def fluorescence(off, on, seed_off, seed_on):
    if seed_off == seed_on:
        raise ValueError('fluorescence on/off seeds must be independent')
    a, sa = batch_statistics(off)
    b, sb = batch_statistics(on)
    dref = float(a[2])
    if dref <= 0:
        raise ValueError('nonpositive Dref')
    difference = (b[:2]-a[:2])/dref
    sem = np.hypot(sa[:2], sb[:2])/dref
    decisions = []
    for diff, error in zip(difference, sem):
        if error > SEM_TARGET:
            decisions.append('insufficient-statistics')
        elif abs(diff)+2*error < LIMIT:
            decisions.append('continue-without-fluorescence')
        elif abs(diff)-2*error > LIMIT:
            decisions.append('consult-user')
        else:
            decisions.append('undetermined-consult-user')
    return dict(Dref=dref, difference=difference.tolist(), sem=sem.tolist(),
                decisions=decisions, units='fraction of fixed off-run Dref',
                seed_off=seed_off, seed_on=seed_on,
                note='A is a 1 cm average; it does not validate fine P8d interface bins')


def production_size(pilot_sem, pilot_n=PILOT_N):
    sem = finite(pilot_sem)
    if sem.shape != (2,) or np.any(sem < 0) or pilot_n != PILOT_N:
        raise ValueError('invalid preregistered pilot')
    needed = max(PILOT_N, int(math.ceil(pilot_n*(float(sem.max())/SEM_TARGET)**2/100))*100)
    return dict(required_each=needed, histories_each=min(needed, MAX_N),
                status='ready' if needed <= MAX_N else 'insufficient-statistics',
                target_SEM=SEM_TARGET, cap_each=MAX_N)


def parse_egs(text, n, seed, fluor, densities):
    if not re.search(r'^\s*DONE\s*$', text, re.M):
        raise ValueError('EGS output incomplete')
    def unique(tag):
        lines = re.findall(r'^\s*'+tag+r'\s+(.+)$', text, re.M)
        if len(lines) != 1:
            raise ValueError('missing/duplicate EGS tag '+tag)
        return finite([float(v.replace('D', 'E')) for v in lines[0].split()])
    if not np.array_equal(unique('CONFIG'), [n, seed, fluor]):
        raise ValueError('EGS configuration identity mismatch')
    ledger = unique('LEDGER')
    if ledger.shape != (6,) or ledger[0] <= 0 or abs(ledger[0]-ledger[1]-ledger[2])/ledger[0] > 1e-6 or ledger[3] > 1e-6 or ledger[4] != 0 or ledger[5] > 1e-6:
        raise ValueError('EGS energy conservation / unprocessed stack')
    dose = np.zeros((100, 55))
    seen = set()
    for b, j, v in re.findall(r'^\s*DOSE\s+(\d+)\s+(\d+)\s+(\S+)', text, re.M):
        i, k = int(b)-1, int(j)-1
        if not (0 <= i < 100 and 0 <= k < 55) or (i, k) in seen:
            raise ValueError('EGS dose duplicate/index')
        dose[i, k] = float(v.replace('D', 'E'))
        seen.add((i, k))
    if len(seen) != 5500 or not np.all(np.isfinite(dose)) or np.any(dose < 0):
        raise ValueError('EGS dose shape/finite values')
    regions = [finite([float(v) for v in line.split()])
               for line in re.findall(r'^\s*REGION\s+(.+)$', text, re.M)]
    media = [finite([float(v) for v in line.split()])
             for line in re.findall(r'^\s*MEDIUM\s+(.+)$', text, re.M)]
    if len(regions) != 4 or len(media) != 2:
        raise ValueError('EGS region/media output missing')
    for i, row in enumerate(regions):
        # region, medium, bound-Compton/profile/Rayleigh, fluor/Auger,
        # splitting, electron and photon cutoffs.
        expected = [i+1, (0, 1, 2, 0)[i], 0, 0, 0,
                    fluor if i in (1, 2) else 0, fluor if i in (1, 2) else 0, 0]
        if row.shape != (10,) or not np.array_equal(row[:8], expected):
            raise ValueError('EGS per-region physics flags mismatch')
        if i in (1, 2) and not np.allclose(row[8:], [.521, .01], rtol=0, atol=1e-12):
            raise ValueError('EGS region cutoffs mismatch')
    for i, row in enumerate(media):
        if row.shape != (9,) or row[0] != i+1 or not np.isclose(row[1], densities[i], rtol=6e-5, atol=0) or not np.allclose(row[2:4], [.521, .01], rtol=0, atol=1e-12) or not np.array_equal(row[4:], [0, 0, 0, 1, 0]):
            raise ValueError('EGS medium density/cutoffs/flags mismatch')
    widths = np.r_[np.full(50, .2*densities[0]), np.full(5, .1*densities[1])]
    if not np.isclose(np.dot(dose.mean(axis=0), widths)*n, ledger[1], rtol=1e-6, atol=1e-9):
        raise ValueError('EGS dose / deposition ledger mismatch')
    return dose, dict(status='pass', ledger=ledger.tolist(), regions=[r.tolist() for r in regions],
                      media=[r.tolist() for r in media], CPU_seconds=float(unique('CPU')[0]))


def spectrum_text(edges, cum):
    e, c = finite(edges), finite(cum)
    if e.shape != (len(c)+1,) or np.any(np.diff(e) <= 0) or e[0] < 0 or np.any(c <= 0) or np.any(np.diff(c) <= 0) or c[-1] != 1:
        raise ValueError('invalid spectrum')
    return str(len(c))+'\n'+''.join(f'{a:.17g} {b:.17g} {p:.17g}\n' for a, b, p in zip(e[:-1], e[1:], c))


def external_sources():
    return {str(p.relative_to(EGS)): sha(p) for folder in ('egs', 'pegs', 'include', 'pegscommons', 'data')
            for p in sorted((EGS/folder).rglob('*')) if p.is_file()}


def execute(work, user_code, sources, inp, config, spectrum, identity):
    """Seal all executable inputs before work; preserve failed attempts."""
    work = Path(work)
    identity = dict(identity, sources=sources, input=inp, config=config,
                    spectrum_sha256=__import__('hashlib').sha256(spectrum.encode()).hexdigest(),
                    egs_inputs=external_sources())
    resume(work/'identity.json', identity)
    seal = work/'complete.json'
    if seal.exists():
        for name, expected in json.loads(seal.read_text()).items():
            if sha(work/name) != expected:
                raise ValueError('EGS sealed output changed')
        return (work/'egs5job.out').read_text()
    if (work/'execution.json').exists():
        raise RuntimeError('EGS failed/incomplete attempt retained; inspect before retry')
    for name, text in sources.items():
        (work/name).write_text(text)
    (work/(user_code+'.inp')).write_text(inp)
    (work/'config.dat').write_text(config)
    (work/'spectrum.dat').write_text(spectrum)
    p = subprocess.run([str(EGS/'egs5run'), 'comp'], cwd=work, input=user_code+'\n\n\n', capture_output=True, text=True)
    (work/'build.log').write_text(p.stdout+p.stderr)
    p.check_returncode()
    if not (work/'egs5job.exe').exists():
        raise RuntimeError('EGS build failed')
    p = subprocess.run(['./egs5job.exe'], cwd=work, capture_output=True, text=True)
    (work/'run.log').write_text(p.stdout+p.stderr)
    write(work/'execution.json', dict(returncode=p.returncode))
    p.check_returncode()
    if not (work/'egs5job.out').exists():
        raise ValueError('EGS output absent')
    # Parse/health checks are performed before any decision, even on resume.
    names = ['egs5job.out', 'pgs5job.pegs5dat', 'pgs5job.pegs5lst', 'config.dat', 'spectrum.dat', user_code+'.inp', *sources]
    write(seal, {name: sha(work/name) for name in names})
    return (work/'egs5job.out').read_text()


def run_egs(work, n, seed, fluor, definitions, edges, cum, all_water=False):
    if n < 100 or n % 100 or n > MAX_N or fluor not in (0, 1):
        raise ValueError('invalid EGS sensitivity configuration')
    water, lead = definitions['water'], definitions['lead']
    if all_water:
        # Retain two named media, with identical composition/density in control.
        lead = dict(water, material='lead')
    densities = [water['density_g_cm3'], lead['density_g_cm3']]
    code = ROOT/'prototypes/mv_electron/p8a_egs.f'
    text = execute(work, 'p8a_egs', {code.name: code.read_text()},
                   pegs_input([water, lead]), f'{n} {seed} {fluor}\n',
                   spectrum_text(edges, cum), dict(N=n, seed=seed, fluor=fluor,
                   definitions=[water, lead], all_water=all_water))
    return parse_egs(text, n, seed, fluor, densities)


def run_water_reference(work, n, seed, water, edges, cum):
    """P3b single-medium control, isolated copy; only thickness is 10.5 cm.

    Legacy fine bins (.05 cm) are exactly merged to the final .1 cm bins.
    The original source and earlier reference outputs are never changed.
    """
    code = ROOT/'prototypes/mv_electron'
    source = (code/'p3b_egs.f').read_text()
    if source.count('thick=30.d0') != 1:
        raise ValueError('P3b thickness patch anchor changed')
    sources = {p.name: p.read_text() for p in code.glob('p3b_egs*.inc')}
    sources['p3b_egs.f'] = source.replace('thick=30.d0', 'thick=10.5d0')
    text = execute(work, 'p3b_egs', sources,
                   pegs_input([dict(water, medium_name='H2O')]), f'10 {n} {seed} 0\n',
                   spectrum_text(edges, cum), dict(N=n, seed=seed, water=water, thickness_cm=10.5))
    def rows(tag, size):
        a = np.zeros((100, size))
        seen = set()
        for b, j, v in re.findall(r'^\s*'+tag+r'\s+(\d+)\s+(\d+)\s+(\S+)', text, re.M):
            i, k = int(b)-1, int(j)-1
            if (i, k) in seen or not (0 <= i < 100 and 0 <= k < size):
                raise ValueError('single-water reference bin identity')
            a[i, k] = float(v)
            seen.add((i, k))
        if len(seen) != a.size:
            raise ValueError('single-water reference incomplete')
        return finite(a)
    coarse, fine = rows('BATCH', 150), rows('FINE', 600)
    ledgers = re.findall(r'^\s*LEDGER\s+(.+)$', text, re.M)
    if len(ledgers) != 1:
        raise ValueError('reference energy ledger missing')
    ledger = finite(list(map(float, ledgers[0].split())))
    if len(ledger) != 8 or ledger[0] <= 0 or abs(ledger[0]-ledger[1]-ledger[2])/ledger[0] > 1e-6 or ledger[3]/(ledger[0]/n) > 1e-6 or ledger[4] != 0 or ledger[5] != 0:
        raise ValueError('reference energy conservation')
    flags = re.findall(r'^\s*FLAGS\s+(.+)$', text, re.M)
    if len(flags) != 1 or not np.array_equal(finite(list(map(float, flags[0].split()))), [0]*10+[1, 1]):
        raise ValueError('reference physics flags mismatch')
    dose = np.column_stack((coarse[:, :50], fine[:, 200:210].reshape(100, 5, 2).mean(axis=2)))/water['density_g_cm3']
    return dose


def water_control(control, reference):
    from scipy.stats import norm
    a, b = finite(control), finite(reference)
    if a.shape != (100, 55) or b.shape != a.shape:
        raise ValueError('water control shape')
    delta = a.mean(axis=0)-b.mean(axis=0)
    sem = np.sqrt(a.var(axis=0, ddof=1)/100+b.var(axis=0, ddof=1)/100)
    # Familywise 95% check over the predetermined 55 bins; zero variance never
    # conceals a nonzero difference. This is the code check, not P8 dose judging.
    critical = float(norm.isf(.05/(2*55)))
    passed = bool(np.all(abs(delta) <= critical*sem))
    return dict(status='pass' if passed else 'fail', critical_z=critical,
                difference=delta.tolist(), sem=sem.tolist(),
                criterion='55-bin independent comparison, Bonferroni familywise 95%')


def spectrum_indicators(edges, probabilities, photon, shells, ne):
    from numpy.polynomial.legendre import leggauss
    nodes, w = leggauss(32)
    es = energy_grid(photon['E'])
    values = np.zeros(5)
    for lo, hi, weight in zip(edges[:-1], edges[1:], probabilities):
        # Split at every absorption edge and pair threshold. Sub-1keV source
        # contribution is explicitly reported, never extrapolated from XCOM.
        cuts = np.unique(np.r_[max(lo, es[0]), hi,
            photon['edges'][(photon['edges'] > max(lo, es[0])) & (photon['edges'] < hi)],
            photon['thresholds'][(photon['thresholds'] > max(lo, es[0])) & (photon['thresholds'] < hi)]])
        if hi <= es[0]:
            continue
        for a, b in zip(cuts[:-1], cuts[1:]):
            e = (a+b)/2+(b-a)*nodes/2
            pe = photo_interp(e, photon['photo_E'], photon['photo'])
            pair = sum(pair_interp(e, es, photon[k], t) for k, t in
                       zip(('pair_nuclear', 'pair_electron'), photon['thresholds']))
            total = kn_total(e, ne)+pe+pair
            fraction_pair, fraction_pe = pair/total, pe/total
            binding = []
            for v in e:
                selected, local = shell_probabilities(v, shells)
                binding.append(np.sum(selected*shells['binding_MeV'])+local*v)
            binding = np.asarray(binding)
            from .physics import M
            f = w*(b-a)/2*weight/(hi-lo)
            values += [np.sum(f*fraction_pair), np.sum(f*e),
                       np.sum(f*fraction_pair*np.maximum(0., e-2*M)/2),
                       np.sum(f*fraction_pe*e), np.sum(f*fraction_pe*binding)]
    if values[1] <= 0:
        raise ValueError('empty spectrum indicator energy')
    return dict(pair_interaction_fraction=float(values[0]),
                positron_kinetic_energy_fraction=float(values[2]/values[1]),
                photoelectric_energy_fraction=float(values[3]/values[1]),
                binding_energy_fraction=float(values[4]/values[1]),
                sub_table_probability=float(sum(p*max(0., min(b, es[0])-a)/(b-a) for a, b, p in zip(edges[:-1], edges[1:], probabilities) if a < es[0])),
                status='report-only', dose_upper_bound=False,
                definition='incident-spectrum first-interaction process probabilities; symmetric mean pair sharing; tracked range only')


def estimate(out=OUT, cache=CACHE):
    verify_protected()
    m = load_manifest(out/'manifest.json')
    checks = json.loads((out/'checks.json').read_text())
    from .check_p8a import material_checks_allow_s7
    if not material_checks_allow_s7(checks) or checks['manifest_sha256'] != sha(out/'manifest.json'):
        raise ValueError('material checks have not passed')
    from .run_p3a import load_spectrum, SPECTRUM
    edges, prob, cum = load_spectrum()
    definitions = {name: load_reference(row['definition'], name) for name, row in m['materials'].items()}
    resume(out/'estimate_identity.json', dict(manifest_sha256=sha(out/'manifest.json'),
        spectrum_sha256=sha(SPECTRUM), code_sha256=sha(Path(__file__)),
        implementations=implementation_hashes(), seeds=SEEDS,
        pilot_each=PILOT_N, cap_each=MAX_N, sem_target=SEM_TARGET, limit=LIMIT))
    control, health = run_egs(cache/'fluorescence/control', PILOT_N, SEEDS['control'], 0, definitions, edges, cum, all_water=True)
    reference_dose = run_water_reference(cache/'fluorescence/reference', PILOT_N, SEEDS['reference'], definitions['water'], edges, cum)
    control_report = water_control(control, reference_dose)
    write(out/'egs_water_control.json', dict(comparison=control_report, health=health))
    if control_report['status'] != 'pass':
        raise ValueError('P8a all-water control failed')
    pilot = {}
    for mode in ('off', 'on'):
        dose, health = run_egs(cache/f'fluorescence/pilot_{mode}', PILOT_N, SEEDS['pilot_'+mode], int(mode == 'on'), definitions, edges, cum)
        pilot[mode] = dose
        write(out/f'pilot_{mode}_health.json', health)
    pilot_stats = fluorescence(pilot['off'], pilot['on'], SEEDS['pilot_off'], SEEDS['pilot_on'])
    allocation = production_size(pilot_stats['sem'])
    write(out/'fluorescence_allocation.json', dict(pilot=pilot_stats, allocation=allocation))
    if allocation['status'] != 'ready':
        result = dict(status='insufficient-statistics', allocation=allocation, pilot=pilot_stats,
                      required_action='stop and report to user; no fluorescence model decision')
    else:
        production = {}
        for mode in ('off', 'on'):
            dose, health = run_egs(cache/f'fluorescence/production_{mode}', allocation['histories_each'], SEEDS['production_'+mode], int(mode == 'on'), definitions, edges, cum)
            production[mode] = dose
            write(out/f'production_{mode}_health.json', health)
        result = fluorescence(production['off'], production['on'], SEEDS['production_off'], SEEDS['production_on'])
        result['allocation'] = allocation
    lead = m['materials']['lead']
    data = {k: load_reference(lead[k], 'lead') for k in ('photon', 'shells', 'constants')}
    indicators = spectrum_indicators(edges, prob, data['photon'], data['shells'], data['constants']['NE'])
    with np.load(cache/'lead/pegs_stopping.npz') as f:
        ratio = dict(energy_MeV=f['T'].tolist(),
            restricted_stopping=(f['restricted_positron']/f['restricted_collision']).tolist(),
            hard_collision=np.divide(f['bhabha'], f['moller'], out=np.zeros_like(f['moller']), where=f['moller'] != 0).tolist(),
            hard_collision_defined=(f['moller'] != 0).tolist(),
            scattering=(f['G1_positron']/f['G1_electron']).tolist())
    report = dict(fluorescence=result, positron=dict(indicators=indicators, pegs_ratios=ratio, status='report-only', dose_upper_bound=False),
                  binding=dict(photoelectric_energy_fraction=indicators['photoelectric_energy_fraction'],
                               binding_energy_fraction=indicators['binding_energy_fraction'], status='report-only'),
                  bremsstrahlung=dict(paragraph=BREMS_REVIEW, status='report-only'),
                  decision_uses='fluorescence only', protected_files=verify_protected())
    write(out/'estimates.json', report)
    # S8's RESULTS/STATUS and user density/model decisions belong to the operator.
    return report


def main():
    argparse.ArgumentParser(__doc__).parse_args()
    report = estimate()
    print(json.dumps(report['fluorescence'], indent=2))


if __name__ == '__main__':
    main()
