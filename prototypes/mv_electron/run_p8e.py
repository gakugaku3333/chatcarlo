"""P8e sealed EGS5 reference and material-stack prototype commands.

Part 1 parsing/execution remains independent of Part 2 transport/judging.
External execution and material generation require explicit CLI invocation.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from .materials_p8a import ROOT, MATERIALS, load_reference, sha, write, resume
from .pegs_p8a import EGS, pegs_input
from .estimate_p8a import (spectrum_text, external_sources, parse_egs as parse_s7,
                           SEEDS, PILOT_N)

OUT = ROOT/'docs/validation/mv/p8e'
CACHE = Path.home()/'.cache/chatcarlo/p8e/egs'
P8E_EGS_SEED = 820001
PRODUCTION_N = 60_000_000
# Pilot stream distinct from production and S7.
PILOT_SEED = 820000
LAYERS = ((1, 5., .1), (2, 6., .1), (1, 4., .1), (3, 2., .1),
          (1, 4., .1), (4, .5, .05))
S7_LAYERS = ((1, 10., .1), (2, .5, .05))
CODE = Path(__file__).with_name('p8e_egs.f')


def layout(layers, densities):
    """Single layer-table mapping for EGS analysis and future transport/judging.

    Density is supplied from P8a definitions, never a duplicated material table.
    Blocks are geometric 1 cm means from 3 cm, with the final lead layer whole.
    bin_layer is zero-based; block_index=-1 marks report-only build-up bins.
    """
    if not 1 <= len(layers) <= 16:
        raise ValueError('nlayer')
    rho = np.asarray(densities, dtype=float)
    if rho.shape != (4,) or not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError('densities')
    edges, owners, masses, blocks = [0.], [], [], []
    for i, (medium, thickness, width) in enumerate(layers):
        if medium not in (1, 2, 3, 4) or int(medium) != medium:
            raise ValueError('medium')
        if not np.all(np.isfinite([thickness, width])):
            raise ValueError('finite thickness/width')
        if min(thickness, width) <= 0:
            raise ValueError('positive thickness/width')
        ratio = thickness/width
        if not math.isfinite(ratio) or ratio > 1001:
            raise ValueError('bin limit')
        count = round(ratio)
        if count < 1 or abs(ratio-count) > 1e-9*ratio:
            raise ValueError('integer bin count')
        if len(owners)+count > 1000:
            raise ValueError('bin limit')
        start, end = edges[-1], edges[-1]+thickness
        if not math.isfinite(end):
            raise ValueError('finite boundary')
        edges.extend(start+np.arange(1, count+1)*width)
        edges[-1] = end
        owners.extend([i]*count)
        masses.extend([rho[int(medium)-1]*width]*count)
        if medium == 4:
            blocks.append((start, end))
        else:
            lo = max(3., start)
            while lo < end-1e-10:
                hi = min(lo+1., end)
                blocks.append((lo, hi))
                lo = hi
    e = np.asarray(edges)
    # Length weights allow a future caller to use the same mapping for averages.
    weights = np.array([np.maximum(0., np.minimum(e[1:], hi)-
                       np.maximum(e[:-1], lo))/(hi-lo) for lo, hi in blocks])
    weights = weights.reshape(len(blocks), len(owners))
    block_index = np.full(len(owners), -1, dtype=int)
    for b, w in enumerate(weights):
        block_index[w > 1e-10] = b
    return dict(edges_cm=e, bin_layer=np.asarray(owners), mass_g_cm2=np.asarray(masses),
                blocks_cm=np.asarray(blocks).reshape(-1, 2), block_weights=weights,
                block_index=block_index)


def block_statistics(dose, grid):
    d = np.asarray(dose, dtype=float)
    if d.shape != (100, len(grid['mass_g_cm2'])) or not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError('100 independent batches required')
    values = d @ grid['block_weights'].T
    return values, values.mean(axis=0), values.std(axis=0, ddof=1)/10


def media_indices(media_set):
    """PEGS order within the selected EGS media set, in canonical P8a order."""
    if media_set not in (2, 4):
        raise ValueError('media set must be 2 or 4')
    return (0, 3) if media_set == 2 else (0, 1, 2, 3)


def egs_layout(layers, densities, media_set=4):
    indices = media_indices(media_set)
    if any(m not in range(1, len(indices)+1) or int(m) != m for m, _, _ in layers):
        raise ValueError('medium outside selected media set')
    # Keep the common geometry function in canonical water/lung/bone/lead order.
    canonical = [(indices[int(m)-1]+1, t, w) for m, t, w in layers]
    return layout(canonical, densities)


def config_text(n, seed, layers, fluor=1, media_set=4):
    if not isinstance(n, int) or n < 100 or n % 100 or fluor not in (0, 1):
        raise ValueError('ncase/fluor')
    egs_layout(layers, [1.]*4, media_set)  # Geometry validation only.
    return f'{n} {seed} {fluor} {len(layers)} {media_set}\n'+''.join(
        f'{m} {t:.17g} {w:.17g}\n' for m, t, w in layers)


def parse_egs(text, n, seed, layers, densities, fluor=1, media_set=4):
    grid = egs_layout(layers, densities, media_set)
    indices = media_indices(media_set)
    densities = [densities[i] for i in indices]
    def rows(tag):
        return re.findall(r'^\s*'+tag+r'\s+([^\n]+)$', text, re.M)
    def numbers(line):
        a = np.array([float(v.replace('D', 'E')) for v in line.split()])
        if not np.all(np.isfinite(a)):
            raise ValueError('nonfinite output')
        return a
    def unique(tag, expected=None):
        r = rows(tag)
        if len(r) != 1:
            raise ValueError('missing/duplicate '+tag)
        a = numbers(r[0])
        if expected is not None and not np.array_equal(a, expected):
            raise ValueError('mismatch '+tag)
        return a
    if len(re.findall(r'^\s*DONE\s*$', text, re.M)) != 1:
        raise ValueError('incomplete output')
    unique('CONFIG', [n, seed, fluor])
    unique('SETTING', [0, 1])
    lr = rows('LAYER')
    if len(lr) != len(layers):
        raise ValueError('layer count')
    for i, (line, (m, t, w)) in enumerate(zip(lr, layers)):
        if not np.allclose(numbers(line), [i+1, m, t, w, round(t/w)], rtol=1e-12, atol=1e-12):
            raise ValueError('layer mapping')
    rr = rows('REGION')
    media = [0]+[m for m, _, _ in layers]+[0]
    if len(rr) != len(media):
        raise ValueError('region count')
    for i, (line, m) in enumerate(zip(rr, media)):
        # region, med, rhor, ecut, pcut, fluor, Auger, Rayleigh,
        # bound Compton, profile, splitting, estepr, k1h, k1l
        expected = [i+1, m, densities[m-1] if m else 0, .521, .01,
                    fluor if m else 0, fluor if m else 0, 0, 0, 0, 0, 1, 0, 0]
        a = numbers(line)
        if (a.shape != (14,) or not np.isclose(a[2], expected[2], rtol=6e-5, atol=0)
                or not np.allclose(np.delete(a, 2), np.delete(expected, 2), rtol=0, atol=1e-12)):
            raise ValueError('region settings/density')
    mm = rows('MEDIUM')
    if len(mm) != len(indices):
        raise ValueError('medium count')
    for i, line in enumerate(mm):
        parts = line.split()
        if len(parts) != 11 or parts[1] != 'P8E_'+MATERIALS[indices[i]].upper():
            raise ValueError('medium name')
        expected = [i+1, densities[i], .521, .01, 1, .5, 0, 0, 0, 0]
        a = numbers(' '.join([parts[0]]+parts[2:]))
        # PEGS DECK density uses a formatted representation, as in S7.
        if a.shape != (10,) or not np.isclose(a[1], expected[1], rtol=6e-5, atol=0) or not np.allclose(np.delete(a, 1), np.delete(expected, 1), rtol=0, atol=1e-12):
            raise ValueError('medium density/settings')
    ledger = unique('LEDGER')
    if (ledger.shape != (6,) or ledger[0] <= 0 or np.any(ledger < 0)
            or abs(ledger[0]-ledger[1]-ledger[2])/ledger[0] > 1e-6
            or ledger[3] > 1e-6 or ledger[4] != 0 or ledger[5] > 1e-12):
        raise ValueError('energy ledger')
    cpu = unique('CPU')
    if cpu.shape != (1,) or cpu[0] < 0:
        raise ValueError('CPU')
    dose = np.zeros((100, len(grid['mass_g_cm2'])))
    seen = set()
    for line in rows('DOSE'):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('dose record')
        b, j = int(parts[0])-1, int(parts[1])-1
        if not (0 <= b < 100 and 0 <= j < dose.shape[1]) or (b, j) in seen:
            raise ValueError('dose index/duplicate')
        dose[b, j] = numbers(parts[2])[0]
        seen.add((b, j))
    if len(seen) != dose.size or np.any(dose < 0):
        raise ValueError('dose missing/negative')
    if not np.isclose(np.dot(dose.mean(axis=0), grid['mass_g_cm2'])*n, ledger[1], rtol=1e-6, atol=1e-9):
        raise ValueError('dose/deposition ledger')
    return dose, dict(CPU_seconds=float(cpu[0]), ledger=ledger.tolist(),
        energy_balance=dict(global_relative=float(abs(ledger[0]-ledger[1]-ledger[2])/ledger[0]),
                            max_history_relative=float(ledger[3]), nleft=int(ledger[4]),
                            max_score_deposition_MeV=float(ledger[5])),
        regions=rr, media=mm, settings=rows('SETTING'))


def replay_comparison(fine, old):
    fine, old = np.asarray(fine), np.asarray(old)
    if fine.shape != (100, 110) or old.shape != (100, 55):
        raise ValueError('S7 replay shape')
    combined = fine.reshape(100, 55, 2).mean(axis=2)
    if not np.all(np.isfinite([combined, old])) or np.any(old < 0) or np.any(fine < 0):
        raise ValueError('S7 replay finite/nonnegative')
    zero = old == 0
    relative = np.divide(abs(combined-old), old, out=np.zeros_like(old), where=~zero)
    if np.any(combined[zero] != 0) or np.any(relative > 1e-12):
        raise ValueError('S7 replay mismatch: stop; do not relax tolerance (four-media RNG may differ)')
    return dict(status='pass', max_relative=float(relative.max()), tolerance=1e-12)


def verify_seal(work):
    seal = json.loads((work/'complete.json').read_text())
    required = {'egs5job.out', 'pgs5job.pegs5dat', 'pgs5job.pegs5lst', 'config.dat', 'spectrum.dat'}
    if not required <= seal.keys():
        raise ValueError('incomplete seal')
    for name, expected in seal.items():
        if sha(work/name) != expected:
            raise ValueError('sealed output changed: '+name)


def execute(work, inp, config, spectrum, validate):
    """Identity contains executable inputs only; preserve every attempted run."""
    work = Path(work)
    sources = {CODE.name: CODE.read_text()}
    identity = dict(sources=sources, input=inp, config=config,
        spectrum_sha256=hashlib.sha256(spectrum.encode()).hexdigest(),
        egs_inputs=external_sources())
    resume(work/'identity.json', identity)
    seal = work/'complete.json'
    if seal.exists():
        verify_seal(work)
        validate((work/'egs5job.out').read_text())
        return (work/'egs5job.out').read_text(), sha(work/'identity.json')
    if (work/'execution.json').exists():
        raise RuntimeError('failed/incomplete attempt retained; inspect before retry')
    for name, content in {**sources, 'p8e_egs.inp': inp, 'config.dat': config, 'spectrum.dat': spectrum}.items():
        (work/name).write_text(content)
    write(work/'execution.json', dict(status='started'))
    try:
        p = subprocess.run([str(EGS/'egs5run'), 'comp'], cwd=work,
                           input='p8e_egs\n\n\n', capture_output=True, text=True)
        (work/'build.log').write_text(p.stdout+p.stderr)
        p.check_returncode()
        if not (work/'egs5job.exe').exists():
            raise RuntimeError('EGS build failed')
        p = subprocess.run(['./egs5job.exe'], cwd=work, capture_output=True, text=True)
        (work/'run.log').write_text(p.stdout+p.stderr)
        p.check_returncode()
        text = (work/'egs5job.out').read_text()
        validate(text)  # Includes replay comparison, BEFORE sealing.
        names = ['egs5job.out', 'pgs5job.pegs5dat', 'pgs5job.pegs5lst',
                 'egs5job.exe', 'config.dat', 'spectrum.dat', 'p8e_egs.inp', *sources]
        write(seal, {name: sha(work/name) for name in names})
        write(work/'execution.json', dict(status='complete'))
    except Exception as exc:
        write(work/'execution.json', dict(status='failed', error=str(exc)))
        raise
    return text, sha(work/'identity.json')


def projection(dose, grid, cpu, n=PILOT_N):
    values, mean, sem = block_statistics(dose, grid)
    e = grid['edges_cm']
    weights = np.maximum(0., np.minimum(e[1:], 5.)-np.maximum(e[:-1], 3.))/2
    dref = float((dose @ weights).mean())
    if dref <= 0:
        raise ValueError('nonpositive pilot Dref')
    projected = sem*math.sqrt(n/PRODUCTION_N)/dref
    seconds = cpu*PRODUCTION_N/n
    return dict(projected_CPU_seconds=seconds, exceeds_12_hours=seconds > 12*3600,
                pilot_Dref=dref, projected_SEM_over_Dref=projected.tolist(),
                blocks_above_0_0033=np.flatnonzero(projected > .0033).tolist(),
                note='pilot Dref used for forecast only; production Dref is fixed for later judging')


def run(mode, out=OUT, cache=CACHE):
    from .run_p3a import load_spectrum
    from .materials_p8a import OUT as P8A_OUT, CACHE as P8A_CACHE
    manifest = json.loads((P8A_OUT/'manifest.json').read_text())
    definitions = [load_reference(manifest['materials'][m]['definition'], m) for m in MATERIALS]
    densities = [d['density_g_cm3'] for d in definitions]
    edges, _, cum = load_spectrum()
    spectrum = spectrum_text(edges, cum)
    media_set = 4
    old = None
    if mode == 's7_replay':
        layers, n, seed, folder = S7_LAYERS, PILOT_N, SEEDS['pilot_on'], 's7_replay_2media'
        media_set = 2
        ref = P8A_CACHE/'fluorescence/pilot_on'
        verify_seal(ref)
        identity = json.loads((ref/'identity.json').read_text())
        old_input = pegs_input([definitions[0], definitions[3]])
        expected = dict(config=f'{n} {seed} 1\n', input=old_input,
            spectrum_sha256=hashlib.sha256(spectrum.encode()).hexdigest(),
            sources={'p8a_egs.f': CODE.with_name('p8a_egs.f').read_text()}, egs_inputs=external_sources())
        if any(identity.get(k) != v for k, v in expected.items()):
            raise ValueError('S7 sealed reference executable inputs differ')
        # Tie identity to the sealed actual inputs as well.
        for name, content in {'config.dat': expected['config'], 'spectrum.dat': spectrum,
                              'p8a_egs.inp': old_input, **expected['sources']}.items():
            if (ref/name).read_text() != content:
                raise ValueError('S7 actual input differs from identity')
        old, _ = parse_s7((ref/'egs5job.out').read_text(), n, seed, 1, [densities[0], densities[3]])
    elif mode in ('egs-pilot', 'egs'):
        # Always revalidate preregistered prerequisites, including on resume.
        run('s7_replay', out, cache)
        layers = LAYERS
        if mode == 'egs':
            pilot = run('egs-pilot', out, cache)
            if pilot['projection']['exceeds_12_hours']:
                raise RuntimeError('production forecast exceeds 12 hours; stop and consult user')
            n, seed, folder = PRODUCTION_N, P8E_EGS_SEED, 'production'
        else:
            n, seed, folder = PILOT_N, PILOT_SEED, 'pilot'
    else:
        raise ValueError('unknown Part 1 command')
    inp = pegs_input([dict(definitions[i], medium_name='P8E_'+MATERIALS[i].upper())
                      for i in media_indices(media_set)])
    def validate(text):
        d, _ = parse_egs(text, n, seed, layers, densities, media_set=media_set)
        if old is not None:
            replay_comparison(d, old)
    text, identity_hash = execute(cache/folder, inp, config_text(n, seed, layers, media_set=media_set), spectrum, validate)
    dose, health = parse_egs(text, n, seed, layers, densities, media_set=media_set)
    grid = egs_layout(layers, densities, media_set)
    batch_blocks, mean, sem = block_statistics(dose, grid)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    artifact = out/f'egs_{folder}.npz'
    np.savez(artifact, dose=dose, batch_blocks=batch_blocks, **grid)
    report = dict(N=n, seed=seed, fluor=1, media_set=media_set, layers=layers, densities_g_cm3=densities,
        units='MeV cm2/g/history', identity_sha256=identity_hash,
        complete_sha256=sha(cache/folder/'complete.json'), artifact_sha256=sha(artifact),
        block_mean=mean.tolist(), block_SEM=sem.tolist(), **health)
    if old is not None:
        report['s7_replay'] = replay_comparison(dose, old)
    if mode == 'egs-pilot':
        report['projection'] = projection(dose, grid, health['CPU_seconds'])
    write(out/f'egs_{folder}.json', report)
    return report


# Part 2 is deliberately appended: Part 1 geometry, parsing and execution above
# remain unchanged while its independent S7 investigation is in progress.
PROTO_PILOT_SEED = 820002
PROTO_SEED = 820003
PROTO_CAP = 40_000_000


def required_prototype_size(pilot_dose, egs_dose, grid, pilot_n=PILOT_N):
    _, _, sp = block_statistics(pilot_dose, grid)
    _, _, se = block_statistics(egs_dose, grid)
    edges=grid['edges_cm']
    ref_weights=np.maximum(0.,np.minimum(edges[1:],5.)-np.maximum(edges[:-1],3.))/2
    dref=float((egs_dose@ref_weights).mean())
    if dref<=0:raise ValueError('nonpositive fixed EGS Dref')
    sp=sp/dref;se=se/dref;t=.0033
    if np.any(se>=t):raise ValueError('EGS SEM >= 0.33%; stop before prototype allocation')
    needed=pilot_n*sp**2/(t*t-se*se)
    n=max(200,int(math.ceil(1.5*float(needed.max())/200))*200)
    return dict(histories=n,required_by_block=needed.tolist(),margin=1.5,
                pilot_n=pilot_n,Dref=dref,EGS_SEM_over_Dref=se.tolist(),
                prototype_pilot_SEM_over_Dref=sp.tolist(),
                status='ready' if n<=PROTO_CAP else 'stop-consult-user',cap=PROTO_CAP)


def judge_batches(prototype, reference, grid):
    _, pm, ps=block_statistics(prototype,grid)
    _, em, es=block_statistics(reference,grid)
    edges=grid['edges_cm']
    weights=np.maximum(0.,np.minimum(edges[1:],5.)-np.maximum(edges[:-1],3.))/2
    dref=float((reference@weights).mean())
    if dref<=0:raise ValueError('nonpositive fixed EGS Dref')
    diff=(pm-em)/dref;sem=np.hypot(ps,es)/dref
    statuses=classify_blocks(diff,sem)
    return dict(Dref=dref,blocks_cm=grid['blocks_cm'].tolist(),difference=diff.tolist(),
        SEM=sem.tolist(),status=statuses,units='fraction of fixed EGS Dref',
        report_only=dict(bin_difference=((prototype.mean(axis=0)-reference.mean(axis=0))/dref).tolist(),
            buildup_0_3=float(np.dot(prototype.mean(axis=0)-reference.mean(axis=0),
                np.maximum(0.,np.minimum(edges[1:],3.)-np.maximum(edges[:-1],0.))/3)/dref),
            water_20_5_21=float(np.dot(prototype.mean(axis=0)-reference.mean(axis=0),
                np.maximum(0.,np.minimum(edges[1:],21.)-np.maximum(edges[:-1],20.5))/.5)/dref)))


def classify_blocks(differences,errors):
    statuses=[];stopped=False
    for d,s in zip(differences,errors):
        if not np.isfinite(d) or not np.isfinite(s) or s<0:raise ValueError('invalid judgement input')
        if stopped:status='not-judged'
        elif s>.0033:status='insufficient-statistics'
        elif abs(d)>.01:status='fail';stopped=True
        else:status='pass'
        statuses.append(status)
    return statuses


def _read_result(artifact, report, expected_n, expected_seed, grid, densities):
    """Verify hashes and all declared conditions BEFORE computing statistics."""
    if 'artifact' in report and Path(report['artifact']).resolve()!=Path(artifact).resolve():
        raise ValueError('analysis artifact path changed')
    if sha(artifact)!=report['artifact_sha256']:raise ValueError('result hash changed')
    if (report['N']!=expected_n or report['seed']!=expected_seed
        or report['layers']!=[list(x) for x in LAYERS]
        or report['densities_g_cm3']!=list(densities)
        or report['units']!='MeV cm2/g/history'):
        raise ValueError('result N/seed/layers/densities/units mismatch')
    with np.load(artifact,allow_pickle=False) as f:
        for key,value in grid.items():
            if not np.array_equal(f[key],value):raise ValueError('result bin mapping changed')
        d=f['dose'].copy()
    if d.shape!=(100,len(grid['mass_g_cm2'])) or not np.all(np.isfinite(d)) or np.any(d<0):
        raise ValueError('invalid batch dose values')
    return d


def load_egs_result(out,cache,grid,densities):
    report=json.loads((out/'egs_production.json').read_text())
    work=cache/'production'
    verify_seal(work)
    if sha(work/'identity.json')!=report['identity_sha256'] or sha(work/'complete.json')!=report['complete_sha256']:
        raise ValueError('EGS reference identity changed')
    if (work/'config.dat').read_text()!=config_text(PRODUCTION_N,P8E_EGS_SEED,LAYERS):
        raise ValueError('EGS sealed configuration changed')
    dose=_read_result(out/'egs_production.npz',report,PRODUCTION_N,P8E_EGS_SEED,grid,densities)
    raw,_=parse_egs((work/'egs5job.out').read_text(),PRODUCTION_N,P8E_EGS_SEED,LAYERS,densities)
    if not np.array_equal(raw,dose):raise ValueError('EGS analysis differs from sealed raw batches')
    return dose,report


def _seal_prototype(work):
    files=[work/'config.json',work/'result.npz',*sorted(work.glob('batch_*.npz'))]
    if len(files)!=102:raise ValueError('100 prototype batches required before seal')
    resume(work/'complete.json',{p.name:sha(p) for p in files})
    return sha(work/'complete.json')


def _verify_prototype(work,report,params):
    from .photon_transport_p8e import fingerprint
    if report['fingerprint']!=fingerprint(params):raise ValueError('prototype fingerprint changed')
    if sha(work/'complete.json')!=report['complete_sha256']:raise ValueError('prototype seal changed')
    for name,expected in json.loads((work/'complete.json').read_text()).items():
        if sha(work/name)!=expected:raise ValueError('sealed prototype input/output changed')
    config=json.loads((work/'config.json').read_text())
    if (config['kernel']!='p8e_stack' or config['params']!=params or config['N']!=report['N']
            or config['seed']!=report['seed'] or config['nbatch']!=100):
        raise ValueError('prototype saved configuration mismatch')
    with np.load(work/'result.npz') as f, np.load(Path(report['artifact'])) as a:
        if not np.array_equal(f['dose'],a['dose']):
            raise ValueError('prototype analysis differs from sealed raw batches')


def run_csda(bundle, out):
    from scipy.integrate import quad
    from .photon_transport_p8e import load_bundles,material_collision,material_quantities
    from .materials_p8a import OUT as P8A_OUT,load_reference
    materials,identity=load_bundles(bundle);lead=materials[3]
    manifest=json.loads((P8A_OUT/'manifest.json').read_text())
    reference=load_reference(manifest['materials']['lead']['range'],'lead')
    lower=float(lead.stop_T[0])
    anchor=float(np.exp(np.interp(np.log(lower),np.log(reference['T']),np.log(reference['csda_g_cm2']))))
    values=[]
    for T in (.1,1.,10.):
        def inverse(t):
            t=max(t,1e-12)
            return 1/(material_collision(t,lead)+np.interp(np.log(t),lead.log_stop_T,lead.stop_rad))
        # P8a and the default ESTAR response both start at 10 keV. Use the
        # tabulated residual CSDA range as the lower boundary condition rather
        # than inventing unprovided stopping data below the transport cutoff.
        points=lead.stop_T[(lead.stop_T>lower)&(lead.stop_T<T)]
        actual=anchor+quad(inverse,lower,T,points=points,epsabs=1e-10,epsrel=1e-7,limit=len(points)+100)[0]
        expected=float(np.exp(np.interp(np.log(T),np.log(reference['T']),np.log(reference['csda_g_cm2']))))
        values.append(dict(T_MeV=T,actual_g_cm2=actual,ESTAR_g_cm2=expected,relative=actual/expected-1))
    report=dict(status='pass' if all(abs(v['relative'])<=.005 for v in values) else 'fail',
                values=values,bundle_sha256=sha(bundle/'bundles.json'),
                lower_boundary=dict(T_MeV=lower,ESTAR_residual_g_cm2=anchor,
                    source_sha256=manifest['materials']['lead']['range']['sha256']),
                note='stopping-integral consistency above 10 keV with the ESTAR residual-range boundary condition; no validation below cutoff or of lead transport')
    write(out/'csda.json',report)
    if report['status']!='pass':raise ValueError('lead CSDA differs from ESTAR by >0.5%')
    return report


def run_part2(mode,out=OUT,egs_cache=CACHE,reviewed=False):
    from . import photon_transport_p8e as kernel
    from .batch_runner_p3u import run_batches
    base=egs_cache.parent;bundle=base/'materials';out=Path(out)
    if mode=='csda':return run_csda(bundle,out)
    materials,identity=kernel.load_bundles(bundle)
    densities=identity['densities'];grid=layout(LAYERS,densities)
    params=dict(bundle=str(bundle),layers=[list(x) for x in LAYERS],
                densities=densities,spectrum='mohan')
    reference,egs_report=load_egs_result(out,egs_cache,grid,densities)
    if mode=='judge':
        if not reviewed:raise ValueError('Claude must review the judgement code before use; pass --reviewed after review')
        report=json.loads((out/'proto_production.json').read_text())
        allocation=json.loads((out/'proto_allocation.json').read_text())
        pilot_report=json.loads((out/'proto_pilot.json').read_text())
        _verify_prototype(base/'proto/pilot',pilot_report,params)
        pilot=_read_result(out/'proto_pilot.npz',pilot_report,PILOT_N,PROTO_PILOT_SEED,grid,densities)
        expected=required_prototype_size(pilot,reference,grid)
        if (any(allocation.get(k)!=v for k,v in expected.items())
            or allocation['EGS_artifact_sha256']!=egs_report['artifact_sha256']
            or allocation['pilot_artifact_sha256']!=pilot_report['artifact_sha256']):
            raise ValueError('allocation reference changed')
        prototype=_read_result(out/'proto_production.npz',report,allocation['histories'],PROTO_SEED,grid,densities)
        _verify_prototype(base/'proto/production',report,params)
        result=judge_batches(prototype,reference,grid)
        # Locate all preregistered boundary influence intervals intersecting the
        # first failed block; downstream widths use water-equivalent density.
        candidates=[]
        if 'fail' in result['status']:
            index=result['status'].index('fail');lo,hi=grid['blocks_cm'][index]
            z=0.
            for i,(m,t,w) in enumerate(LAYERS[:-1]):
                z+=t;next_m=LAYERS[i+1][0]
                if hi>z-.5/densities[m-1] and lo<z+2/densities[next_m-1]:candidates.append(z)
            result['first_failure_candidates']=dict(boundaries_cm=candidates,
                material_table=not bool(candidates),
                table_material=MATERIALS[LAYERS[int(grid['bin_layer'][np.flatnonzero(grid['block_weights'][index]>1e-10)[0]])][0]-1] if not candidates else None)
        result['input_hashes']=dict(EGS=egs_report['artifact_sha256'],prototype=report['artifact_sha256'])
        write(out/'judgement.json',result)
        return result
    if mode not in ('proto-pilot','proto'):raise ValueError('unknown Part 2 command')
    if mode=='proto':
        allocation=json.loads((out/'proto_allocation.json').read_text())
        pilot_report=json.loads((out/'proto_pilot.json').read_text())
        _verify_prototype(base/'proto/pilot',pilot_report,params)
        pilot=_read_result(out/'proto_pilot.npz',pilot_report,PILOT_N,PROTO_PILOT_SEED,grid,densities)
        expected=required_prototype_size(pilot,reference,grid)
        if any(allocation.get(k)!=v for k,v in expected.items()) or allocation['EGS_artifact_sha256']!=egs_report['artifact_sha256'] or allocation['pilot_artifact_sha256']!=pilot_report['artifact_sha256']:
            raise ValueError('prototype allocation input changed')
        if allocation['status']!='ready':raise RuntimeError('prototype allocation exceeds cap; stop and consult user')
        n,seed,name=allocation['histories'],PROTO_SEED,'production'
    else:n,seed,name=PILOT_N,PROTO_PILOT_SEED,'pilot'
    path=run_batches(base/f'proto/{name}','p8e_stack',params,n,seed,2,nbatch=100)
    with np.load(path) as f:
        values={k:f[k].copy() for k in f.files}
    # Each outer P3u batch is an independent dose mean, not an SEM of sub-bins.
    dose=values['dose'];batch_values,mean,sem=block_statistics(dose,grid)
    work=path.parent;seal_hash=_seal_prototype(work)
    artifact=out/f'proto_{name}.npz';out.mkdir(parents=True,exist_ok=True)
    np.savez(artifact,dose=dose,batch_blocks=batch_values,**grid)
    counts=values['boundary_counts'].sum(axis=0)
    report=dict(N=n,seed=seed,layers=[list(x) for x in LAYERS],densities_g_cm3=densities,
        units='MeV cm2/g/history',artifact=str(artifact),artifact_sha256=sha(artifact),
        complete_sha256=seal_hash,fingerprint=kernel.fingerprint(params),
        block_mean=mean.tolist(),block_SEM=sem.tolist(),wall_seconds=float(values['wall_seconds']),
        energy_totals=values['totals'].sum(axis=0).tolist(),max_history_relative=float(values['maxima'][:,0].max()),
        max_score_deposition_MeV=float(values['p8_max'][:,0].max()),
        boundary_count=counts.tolist(),omitted_small_strength_fraction=float(counts[1]/counts[0]) if counts[0] else 0.,
        boundary_strength_histogram=values['boundary_hist'].sum(axis=0).tolist(),
        boundary_strength_edges=kernel.BOUNDARY_STRENGTH_EDGES.tolist(),
        omitted_boundary_strength_histogram=values['boundary_omitted_hist'].sum(axis=0).tolist(),
        boundary_count_labels=['attempts','omitted_small_s','zero_s','terminal','reflections'],
        fluorescence_counts=values['fluorescence_counts'].sum(axis=0).tolist(),
        fluorescence_count_labels=['emitted','escaped','photoabsorbed','subcutoff_absorbed',
            'K_emitted','K_escaped','K_photoabsorbed','K_subcutoff_absorbed'],
        approximations=['Auger/Coster-Kronig terminate locally without subsequent fluorescence',
                        'boundary scattering below the GS minimum strength is omitted'],speed_status='report-only')
    write(out/f'proto_{name}.json',report)
    if mode=='proto-pilot':
        allocation=required_prototype_size(dose,reference,grid)
        allocation.update(EGS_artifact_sha256=egs_report['artifact_sha256'],pilot_artifact_sha256=sha(artifact))
        write(out/'proto_allocation.json',allocation)
        report['allocation']=allocation
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('s7_replay', 'egs-pilot', 'egs','proto-pilot','proto','csda','judge'))
    parser.add_argument('--reviewed',action='store_true',help='Claude has reviewed the judgement implementation')
    args = parser.parse_args()
    report = (run(args.command) if args.command in ('s7_replay','egs-pilot','egs')
              else run_part2(args.command,reviewed=args.reviewed))
    print(json.dumps(report, indent=2))
    if 'projection' in report:
        p = report['projection']
        if p['blocks_above_0_0033']:
            print('REPORT BEFORE PART 2: projected SEM > 0.33% in blocks', p['blocks_above_0_0033'])
        if p['exceeds_12_hours']:
            print('STOP: production forecast exceeds 12 hours; consult user')


if __name__ == '__main__':
    main()
