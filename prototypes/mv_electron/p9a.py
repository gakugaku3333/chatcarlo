"""P9a first implementation: isolated EGS5 reference driver only.

No external work occurs at import. CLI execution is for the later operator;
offline tests use synthetic output and read-only P8e artefacts.
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
from .pegs_p8a import EGS, pegs_input, NUM
from .estimate_p8a import spectrum_text, external_sources
from . import run_p8e as p8e
from .run_p8e import block_statistics

OUT = ROOT/'docs/validation/mv/p9a'
CACHE = Path.home()/'.cache/chatcarlo/p9a/egs'
CODE = Path(__file__).with_name('p9a_egs.f')
P9A_EGS_SEED = 920001
PILOT_SEED = 920002
PILOT_N = 1_000_000
LAYERS = ((1, 5., .1), (5, .1, .02), (1, 5., .1),
          (5, 1., .1), (1, 10.1, .1))
SLAB_LAYERS = ((5, 2., .1),)
MEDIA_NAMES = ('P8E_WATER', 'P8E_LUNG', 'P8E_BONE', 'P8E_LEAD', 'P9A_SEED')


def seed_definition():
    """Temporary validation I only: pegs_input does not emit I into PEGS input.

    This is not a material-data definition and is never saved as definition.json.
    """
    import xraylib
    return dict(material='seed_agti', medium_name='P9A_SEED', density_g_cm3=2.47,
                weights={'47': .5955, '22': .4045},
                atomic_weights={str(z): xraylib.AtomicWeight(z) for z in (22, 47)},
                I_MeV=1e-4)


def media_indices(media_set):
    if media_set not in (4, 5):
        raise ValueError('media set must be 4 or 5')
    return tuple(range(media_set))


def layout(layers, densities):
    """P9a layer-table mapping for EGS analysis and future transport/judging.

    Density is supplied from P8a definitions, never a duplicated material table.
    Blocks are geometric 1 cm means from 3 cm, with each seed plate whole; the final 0.1 cm is report-only.
    bin_layer is zero-based; block_index=-1 marks report-only build-up bins.
    """
    if not 1 <= len(layers) <= 16:
        raise ValueError('nlayer')
    rho = np.asarray(densities, dtype=float)
    if rho.shape != (5,) or not np.all(np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError('densities')
    edges, owners, masses, blocks = [0.], [], [], []
    for i, (medium, thickness, width) in enumerate(layers):
        if medium not in (1, 2, 3, 4, 5) or int(medium) != medium:
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
        fine_end = end/.05
        if (not math.isfinite(fine_end) or fine_end > 1000+1e-8
                or abs(fine_end-round(fine_end)) > 1e-8
                or round(fine_end) <= round(start/.05)):
            raise ValueError('boundary outside fine lattice/capacity')
        edges.extend(start+np.arange(1, count+1)*width)
        edges[-1] = end
        owners.extend([i]*count)
        masses.extend([rho[int(medium)-1]*width]*count)
        if medium in (4, 5):
            blocks.append((start, end))
        else:
            lo = max(3., start)
            while lo < end-1e-10:
                if end-lo < 1.-1e-10:
                    break
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


def egs_layout(layers, densities, media_set=5):
    media_indices(media_set)
    if any(m not in range(1, media_set+1) for m, _, _ in layers):
        raise ValueError('medium outside selected media set')
    rho = list(densities)
    if len(rho) == 4 and media_set == 4:
        rho.append(2.47)
    return layout(layers, rho)


def fine_layout(layers, densities, media_set=5):
    egs_layout(layers, densities, media_set)
    return egs_layout([(m, t, .05) for m, t, _ in layers], densities, media_set)


def config_text(n, seed, layers, fluor=1, media_set=5):
    if not isinstance(n, int) or n < 100 or n > 2147483600 or n % 100 or fluor not in (0, 1):
        raise ValueError('ncase/fluor')
    if not isinstance(seed, int) or not 1 <= seed <= 2147483647:
        raise ValueError('seed')
    egs_layout(layers, [1.]*5, media_set)  # Geometry validation only.
    return f'{n} {seed} {fluor} {len(layers)} {media_set}\n'+''.join(
        f'{int(m)} {t:.17g} {w:.17g}\n' for m, t, w in layers)


def parse_egs(text, n, seed, layers, densities, fluor=1, media_set=5):
    grid = egs_layout(layers, densities, media_set)
    indices = media_indices(media_set)
    densities_full = densities
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
        if len(parts) != 11 or parts[1] != MEDIA_NAMES[i]:
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
    fine_grid = fine_layout(layers, densities_full, media_set)
    fine = read_fine(text, fine_grid, n, ledger[1])
    return dose, fine, dict(CPU_seconds=float(cpu[0]), ledger=ledger.tolist(),
        energy_balance=dict(global_relative=float(abs(ledger[0]-ledger[1]-ledger[2])/ledger[0]),
                            max_history_relative=float(ledger[3]), nleft=int(ledger[4]),
                            max_score_deposition_MeV=float(ledger[5])),
        regions=rr, media=mm, settings=rows('SETTING'))


def execute(work, inp, config, spectrum, validate):
    """Identity contains executable inputs only; preserve every attempted run."""
    work = Path(work)
    sources = {CODE.name: CODE.read_text()}
    identity = dict(sources=sources, pegs_input_sha256=digest(inp), config=config,
        spectrum_sha256=hashlib.sha256(spectrum.encode()).hexdigest(),
        egs_inputs=external_sources())
    resume(work/'identity.json', identity)
    seal = work/'complete.json'
    if seal.exists():
        verify_seal(work)
        check_actual_inputs(work, sources, inp, config, spectrum)
        validate((work/'egs5job.out').read_text())
        return (work/'egs5job.out').read_text(), sha(work/'identity.json')
    if (work/'execution.json').exists():
        raise RuntimeError('failed/incomplete attempt retained; inspect before retry')
    for name, content in {**sources, 'p9a_egs.inp': inp, 'config.dat': config, 'spectrum.dat': spectrum}.items():
        (work/name).write_text(content)
    write(work/'execution.json', dict(status='started'))
    try:
        p = subprocess.run([str(EGS/'egs5run'), 'comp'], cwd=work,
                           input='p9a_egs\n\n\n', capture_output=True, text=True)
        (work/'build.log').write_text(p.stdout+p.stderr)
        p.check_returncode()
        if not (work/'egs5job.exe').exists():
            raise RuntimeError('EGS build failed')
        p = subprocess.run(['./egs5job.exe'], cwd=work, capture_output=True, text=True)
        (work/'run.log').write_text(p.stdout+p.stderr)
        p.check_returncode()
        text = (work/'egs5job.out').read_text()
        check_actual_inputs(work, sources, inp, config, spectrum)
        validate(text)  # Includes replay comparison, BEFORE sealing.
        names = ['egs5job.out', 'pgs5job.pegs5dat', 'pgs5job.pegs5lst',
                 'egs5job.exe', 'identity.json', 'config.dat', 'spectrum.dat', 'p9a_egs.inp', *sources]
        write(seal, {name: sha(work/name) for name in names})
        write(work/'execution.json', dict(status='complete'))
    except Exception as exc:
        write(work/'execution.json', dict(status='failed', error=str(exc)))
        raise
    return text, sha(work/'identity.json')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def check_actual_inputs(work, sources, inp, config, spectrum):
    for name, content in {**sources, 'p9a_egs.inp': inp,
                          'config.dat': config, 'spectrum.dat': spectrum}.items():
        if (work/name).read_text() != content:
            raise ValueError('sealed executable input differs: '+name)


def verify_seal(work):
    work = Path(work)
    required = {'identity.json', CODE.name, 'p9a_egs.inp', 'egs5job.exe',
                'egs5job.out', 'pgs5job.pegs5dat', 'pgs5job.pegs5lst',
                'config.dat', 'spectrum.dat'}
    seal = json.loads((work/'complete.json').read_text())
    if not required <= seal.keys():
        raise ValueError('incomplete seal')
    for name, expected in seal.items():
        if Path(name).name != name or sha(work/name) != expected:
            raise ValueError('sealed output changed: '+name)


def read_fine(text, grid, n, deposition):
    dose = np.zeros((100, len(grid['mass_g_cm2'])))
    seen = set()
    for line in re.findall(r'^\s*FINE\s+([^\n]+)$', text, re.M):
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('fine record')
        b, j = int(parts[0])-1, int(parts[1])-1
        if not (0 <= b < 100 and 0 <= j < dose.shape[1]) or (b, j) in seen:
            raise ValueError('fine index/duplicate')
        dose[b, j] = float(parts[2].replace('D', 'E'))
        seen.add((b, j))
    if len(seen) != dose.size or not np.all(np.isfinite(dose)) or np.any(dose < 0):
        raise ValueError('fine missing/nonfinite/negative')
    if not np.isclose(np.dot(dose.mean(axis=0), grid['mass_g_cm2'])*n,
                      deposition, rtol=1e-6, atol=1e-9):
        raise ValueError('fine/deposition ledger')
    return dose


def pegs_records(text):
    """Keep complete per-medium DECK records for the four-media comparison."""
    starts = list(re.finditer(r'^\s*MEDIUM=([^,\n]+),', text, re.M))
    result = {}
    for i, match in enumerate(starts):
        name = match[1].strip()
        if name in result:
            raise ValueError('duplicate PEGS medium')
        result[name] = text[match.start():starts[i+1].start() if i+1 < len(starts) else len(text)]
    return result


def check_pegs(text, media_set, reference=None):
    records = pegs_records(text)
    if set(records) != set(MEDIA_NAMES[:media_set]):
        raise ValueError('PEGS medium set')
    if reference is not None:
        old = pegs_records(reference)
        for name in MEDIA_NAMES[:4]:
            if name not in old or records[name].split() != old[name].split():
                raise ValueError('old PEGS record changed: '+name)
    if media_set == 5:
        seed = records['P9A_SEED']
        def number(pattern, line):
            match = re.search(pattern+r'\s*('+NUM+')', line)
            if match is None:
                raise ValueError('PEGS seed header')
            return float(match[1].replace('D', 'E'))
        lines = seed.strip().splitlines()
        if (number(r'RHO=', lines[1]) != 2.47 or number(r'NE=', lines[1]) != 2):
            raise ValueError('PEGS seed density/composition')
        weights = {}
        definition = seed_definition()
        for line in lines[2:4]:
            z = number(r',Z=', line)
            if z != int(z) or str(int(z)) not in definition['weights'] or z in weights:
                raise ValueError('PEGS seed elements')
            weights[z] = number(r'RHOZ=', line)
            if not np.isclose(number(r',A=', line), definition['atomic_weights'][str(int(z))], rtol=6e-5):
                raise ValueError('PEGS seed atomic weight')
        if set(weights) != {22, 47} or any(not np.isclose(weights[z], w, rtol=6e-6, atol=0)
                                          for z, w in ((22, .4045), (47, .5955))):
            raise ValueError('PEGS seed weights')
    return {'status': 'pass', 'media': list(records)}


def replay_comparison(actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    if (a.shape != (100, 220) or b.shape != a.shape or a.dtype != b.dtype
            or not np.all(np.isfinite(a)) or np.any(a < 0) or a.tobytes() != b.tobytes()):
        raise ValueError('P8e four-media replay mismatch; stop')
    return {'status': 'pass', 'comparison': 'all batches/bins byte-identical'}


def compare_settings(health, old_health):
    for key in ('regions', 'settings'):
        if [x.split() for x in health[key]] != [x.split() for x in old_health[key]]:
            raise ValueError('old layout '+key+' changed')
    if [x.split() for x in health['media'][:4]] != [x.split() for x in old_health['media']]:
        raise ValueError('old layout media changed')


def projection(dose, grid, cpu, n=PILOT_N):
    """EGS-only sample size from the pilot; no GPU allocation or judging."""
    _, _, sem = block_statistics(dose, grid)
    edges = grid['edges_cm']
    weights = np.maximum(0., np.minimum(edges[1:], 5.)-np.maximum(edges[:-1], 3.))/2
    dref = float((dose @ weights).mean())
    if not math.isfinite(dref) or dref <= 0:
        raise ValueError('finite positive pilot Dref required')
    if not math.isfinite(cpu) or cpu < 0 or n < 100 or n % 100:
        raise ValueError('pilot CPU/N')
    required = n*float(np.max((sem/dref)**2))/(.0033**2/2)
    if not math.isfinite(required):
        raise ValueError('nonfinite history forecast')
    histories = max(100, math.ceil(1.5*required/100)*100)
    seconds = cpu*histories/n
    return dict(histories=histories, pilot_Dref=dref,
                projected_SEM_over_Dref=(sem/dref*math.sqrt(n/histories)).tolist(),
                projected_CPU_seconds=seconds, exceeds_12_hours=seconds > 12*3600,
                target_SEM_over_Dref=.0033/math.sqrt(2), safety_factor=1.5)


def load_inputs():
    from .materials_p8a import OUT as P8A_OUT
    from .run_p3a import load_spectrum
    manifest = json.loads((P8A_OUT/'manifest.json').read_text())
    definitions = [load_reference(manifest['materials'][m]['definition'], m) for m in MATERIALS]
    definitions = [dict(d, medium_name=MEDIA_NAMES[i]) for i, d in enumerate(definitions)]
    definitions.append(seed_definition())
    edges, _, cum = load_spectrum()
    return definitions, spectrum_text(edges, cum)


def load_replay_reference(definitions, spectrum, reference_out=p8e.OUT, reference_cache=p8e.CACHE):
    """Read the sealed P8e pilot and bind its analysis to actual executable inputs."""
    work = Path(reference_cache)/'pilot'
    p8e.verify_seal(work)
    report = json.loads((Path(reference_out)/'egs_pilot.json').read_text())
    artifact = Path(reference_out)/'egs_pilot.npz'
    if (report['artifact_sha256'] != sha(artifact)
            or report['complete_sha256'] != sha(work/'complete.json')
            or report['identity_sha256'] != sha(work/'identity.json')):
        raise ValueError('P8e pilot reference hash')
    n, seed = p8e.PILOT_N, p8e.PILOT_SEED
    inp = pegs_input(definitions[:4])
    expected = {'p8e_egs.inp': inp, 'config.dat': p8e.config_text(n, seed, p8e.LAYERS),
                'spectrum.dat': spectrum, p8e.CODE.name: p8e.CODE.read_text()}
    for name, content in expected.items():
        if (work/name).read_text() != content:
            raise ValueError('P8e reference input differs: '+name)
    identity = json.loads((work/'identity.json').read_text())
    if (identity['input'] != inp or identity['egs_inputs'] != external_sources()
            or identity['config'] != expected['config.dat']
            or identity['sources'] != {p8e.CODE.name: p8e.CODE.read_text()}
            or identity['spectrum_sha256'] != digest(spectrum)):
        raise ValueError('P8e reference executable identity')
    densities = [d['density_g_cm3'] for d in definitions[:4]]
    dose, health = p8e.parse_egs((work/'egs5job.out').read_text(), n, seed, p8e.LAYERS, densities)
    saved = p8e._read_result(artifact, report, n, seed,
                             p8e.layout(p8e.LAYERS, densities), densities)
    replay_comparison(dose, saved)
    return dose, health, (work/'pgs5job.pegs5dat').read_text()


def run_case(folder, n, seed, layers, media_set, definitions, spectrum, out, cache,
             extra_validate=None):
    densities = [d['density_g_cm3'] for d in definitions]
    inp = pegs_input(definitions[:media_set])
    work = Path(cache)/folder
    def validate(text):
        dose, fine, health = parse_egs(text, n, seed, layers, densities, media_set=media_set)
        check_pegs((work/'pgs5job.pegs5dat').read_text(), media_set)
        if extra_validate is not None:
            extra_validate(dose, health, (work/'pgs5job.pegs5dat').read_text())
    text, identity_hash = execute(work, inp, config_text(n, seed, layers, media_set=media_set), spectrum, validate)
    dose, fine, health = parse_egs(text, n, seed, layers, densities, media_set=media_set)
    grid = egs_layout(layers, densities, media_set)
    fg = fine_layout(layers, densities, media_set)
    batch_blocks, mean, sem = block_statistics(dose, grid)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    artifact = out/f'egs_{folder}.npz'
    np.savez(artifact, dose=dose, fine=fine, batch_blocks=batch_blocks,
             fine_edges_cm=fg['edges_cm'], fine_mass_g_cm2=fg['mass_g_cm2'], **grid)
    report = dict(N=n, seed=seed, fluor=1, media_set=media_set, layers=layers,
                  densities_g_cm3=densities, units='MeV cm2/g/history',
                  identity_sha256=identity_hash, pegs_input_sha256=digest(inp),
                  complete_sha256=sha(work/'complete.json'), artifact_sha256=sha(artifact),
                  block_mean=mean.tolist(), block_SEM=sem.tolist(), **health)
    if folder == 'pilot':
        report['projection'] = projection(dose, grid, health['CPU_seconds'], n)
    write(out/f'egs_{folder}.json', report)
    return report


def run(mode, out=OUT, cache=CACHE):
    if mode not in ('egs-replay', 'egs-pilot', 'egs'):
        raise ValueError('unknown EGS command')
    definitions, spectrum = load_inputs()
    if mode == 'egs-replay':
        old, old_health, old_pegs = load_replay_reference(definitions, spectrum)
        def four(dose, health, pegs):
            replay_comparison(dose, old)
            compare_settings(health, old_health)
            check_pegs(pegs, 4, old_pegs)
        first = run_case('replay_4media', p8e.PILOT_N, p8e.PILOT_SEED, p8e.LAYERS,
                         4, definitions, spectrum, out, cache, four)
        def five(dose, health, pegs):
            compare_settings(health, old_health)
            check_pegs(pegs, 5, old_pegs)
        second = run_case('replay_5media', 100, p8e.PILOT_SEED, p8e.LAYERS,
                          5, definitions, spectrum, out, cache, five)
        return dict(status='pass', four_media=first, five_media=second)
    if mode == 'egs-pilot':
        run('egs-replay', out, cache)
        return run_case('pilot', PILOT_N, PILOT_SEED, LAYERS, 5,
                        definitions, spectrum, out, cache)
    pilot = run('egs-pilot', out, cache)
    forecast = pilot['projection']
    if forecast['exceeds_12_hours']:
        raise RuntimeError('production forecast exceeds 12 hours; stop and consult user')
    return run_case('production', forecast['histories'], P9A_EGS_SEED, LAYERS, 5,
                    definitions, spectrum, out, cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('egs-replay', 'egs-pilot', 'egs', 'protect', 'photon', 'pegs', 'elsepa', 'bundle', 'check', 'check-data', 'allocate-gpu', 'judge'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--cache', type=Path, default=CACHE)
    parser.add_argument('--data-cache', type=Path, default=Path.home()/'.cache/chatcarlo/p9a')
    parser.add_argument('--gpu-root', type=Path)
    parser.add_argument('--pilot-root', type=Path)
    args = parser.parse_args()
    if args.command.startswith('egs'):
        result = run(args.command, args.out, args.cache)
    elif args.command in ('allocate-gpu', 'judge'):
        from . import p9a_judge
        if args.pilot_root is None or (args.command == 'judge' and args.gpu_root is None):
            parser.error('--pilot-root (and --gpu-root for judge) required')
        if args.command == 'judge':
            result = p9a_judge.judge(args.gpu_root,args.pilot_root,args.out,args.data_cache,args.cache)
        else:
            result = p9a_judge.allocate(args.pilot_root,args.out,args.data_cache,args.cache)
    else:
        from .p9a_data import dispatch
        result = dispatch(args.command,args.data_cache,args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
