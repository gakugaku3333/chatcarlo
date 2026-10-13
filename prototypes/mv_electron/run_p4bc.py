"""P4bc fixed, serial checks with immutable stage outputs (plan amendment 1).

ESTAR request assumptions (not verified online in this sandbox): POST to
https://physics.nist.gov/cgi-bin/Star/e_table-t.pl, form fields matno=276,
ShowDefault=on, Energies='' (no user-specified energy grid). Expect a UTF-8 HTML
PRE block identifying CSDA Range, with units g/cm2 (or g/cm^2), and seven numeric
columns: kinetic energy, collision/radiative/total stopping power, CSDA range,
radiation yield, density effect. Required energies must occur exactly in the
returned grid. Five-column custom-energy responses, changed headers/columns,
nonfinite values and missing energies are errors, never guessed/interpolated.
The request fields and response shape still require confirmation by Claude.

No file is deleted or replaced. Existing NPZ outputs require identical config.
fetch writes response/metadata; run writes the lock; production writes NPZ only;
analyse writes checks, figure and report. None of the stage outputs is replaced.
"""
import argparse
import datetime
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import platform
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/validation/mv/p4'
CACHE = Path('~/.cache/chatcarlo/p4').expanduser()
SOURCE_FILES = ('run_p4bc.py', 'photon_transport_p3b.py', 'transport.py',
                'physics.py', 'bremsstrahlung_p3r.py', 'flight_annihilation_p2c.py',
                'photon_data_p2.py', 'run_p3s.py', 'photon_sampling_p2.py',
                'sampling.py', 'backend_sampling.py')
B_ENERGIES = (.5, 1.25, 2., 6., 10.)
C_ENERGIES = (.1, 1., 10.)
X = np.linspace(-.01, .01, 41)
ESTAR_URL = 'https://physics.nist.gov/cgi-bin/Star/e_table-t.pl'
ESTAR_FIELDS = dict(matno='276', ShowDefault='on', Energies='')
LIMITATIONS = (
    'P4bは断面積と放射阻止能の出典の一部が共通で、完全に独立な物理モデルではない。',
    '10 keV未満の光子は負の方向に寄与するが、寄与の大きさは分離していない。'
    '連続制動放射、荷電粒子のカットオフ、原子緩和、束縛のないCompton、'
    '陽電子の近似、NISTのgに含まれる効果もあり、差を一つの原因に帰さない。',
    'ρ_tr−ρ_colは試作のμtrとNISTのμenの差の飛跡積分で、直接数えた放射エネルギーではない。'
    'B3は未実施であり、B1の合格はB3の検証を意味しない。',
    'P4cはP1の核による検査で、現在の核のループそのものの検査ではない。'
    'C1の0.05%刻みの積分誤差は最大0.025%で、SEMとは別。',
)


def encoded(value):
    def convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    return json.dumps(value, default=convert, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)


def provenance(inputs):
    import numba
    from .run_p3a import table_hashes
    return dict(source_hashes={name: hashlib.sha256(
        (Path(__file__).parent / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
        table_hashes=table_hashes('elsepa', inputs),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      numba=numba.__version__))


def load_result(path, config):
    with np.load(path, allow_pickle=False) as data:
        if str(data['config_json']) != encoded(config):
            raise RuntimeError(f'{path}: saved config/provenance mismatch; no overwrite')
        return {k: data[k].copy() for k in data.files if k != 'config_json'}


def saved(path, config, calculate):
    """Single condition checkpoint; only the caller supplies its fixed settings."""
    path = Path(path)
    if path.exists():
        return load_result(path, config)
    result = calculate()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez(stream, config_json=np.array(encoded(config)), **result)
    return load_result(path, config)


def classify(difference, sem, limit, sem_limit, healthy=True):
    if not healthy or not np.all(np.isfinite([difference, sem])) or sem < 0:
        return '判定できない'
    if sem > sem_limit:
        return '統計不足'
    return '不合格' if abs(difference) > limit else '合格'


def finite(result):
    return all(np.all(np.isfinite(v)) for v in result.values())


def photon_healthy(result, n, energy=None):
    if not finite(result):
        return False
    totals = result['totals']
    ok = (totals[0] > 0 and not np.any(result['violations'])
          and not np.any(result['counts'][5:7]))
    if energy is not None:
        ok = ok and totals[0] == n*energy and totals[2] <= totals[0]*1e-6
    return bool(ok)


def b1_statistics(result, n, energy):
    col, tr = result['collision_kerma'], result['kerma']
    ok = (photon_healthy(result, n, energy) and col.shape == tr.shape == (100,)
          and np.all(col > 0) and np.all(tr > 0) and n > 0 and n % 100 == 0)
    if not ok:
        return dict(status='判定できない', difference=None, sem=None)
    rb = col / ((n/100)*energy) - 1
    difference, sem = float(rb.mean()), float(rb.std(ddof=1)/10)
    rho_tr = float(tr.sum()/(n*energy))
    return dict(status=classify(difference, sem, .01, .001),
                difference=difference, sem=sem, rho_col=1+difference,
                rho_tr=rho_tr, tr_minus_col=rho_tr-(1+difference),
                continuous_brems_fraction=float((result['brems'][8]-result['brems'][1])/(n*energy)))


def required_histories(sem):
    if not np.isfinite(sem) or sem < 0:
        raise ValueError('pilot SEM must be finite and nonnegative')
    return max(100000, 100000*math.ceil(1.5*(sem/.001)**2))


def execute_b1(energy, n, seed, inputs, score_kerma=True):
    from . import photon_transport_p3b as kernel, run_p3s
    start = time.perf_counter()
    values = kernel.run_coupled(float(energy), n, seed, .05, .025, 2000., *inputs,
        **run_p3s.rule_kwargs('own'), annihilate_in_flight=True, brems_mode=2,
        score_kerma=score_kerma, score_pair=False,
        source_state=kernel.photon_state(float(energy), 1000., 0., 0., 1., 0))
    # Reduce immediately: never persist the 2000 cm depth arrays.
    return dict(kerma=values[3].sum(axis=1), collision_kerma=values[4].sum(axis=1),
                totals=values[5], maxima=values[6], violations=values[7],
                counts=values[8], brems=values[13], seconds=time.perf_counter()-start)


def execute_b2(inputs):
    from . import photon_transport_p3b as kernel, run_p3s
    from .batch_kernels_p3u import spectrum
    edges, cum = spectrum(dict(spectrum='mohan'))
    start = time.perf_counter()
    values = kernel.run_coupled(10., 2000000, 20291796, .05, .025, 30., *inputs,
        **run_p3s.rule_kwargs('own'), annihilate_in_flight=True, brems_mode=2,
        score_kerma=True, score_pair=False, spectrum_edges=edges, spectrum_cum=cum)
    return dict(fine=values[1], kerma=values[3], collision_kerma=values[4],
                totals=values[5], violations=values[7], counts=values[8],
                seconds=time.perf_counter()-start)


def b2_statistics(result):
    rows = []
    valid = (photon_healthy(result, 2000000)
             and result['fine'].shape == result['collision_kerma'].shape == (100, 600)
             and result['kerma'].shape == (100, 600))
    for lo in range(0, 30, 5):
        row = dict(depth_cm=[lo, lo+5], status='判定できない', ratio=None, sem=None)
        if valid:
            numerator = result['fine'][:, lo*20:(lo+5)*20].sum(axis=1)
            denominator = result['collision_kerma'][:, lo*20:(lo+5)*20].sum(axis=1)
            if np.all(denominator > 0):
                ratios = numerator/denominator
                row.update(status='報告のみ', ratio=float(numerator.sum()/denominator.sum()),
                           sem=float(ratios.std(ddof=1)/10))
        rows.append(row)
    return rows


def integrate_range(T0, stopping):
    def integral(points):
        u = np.linspace(np.log(.01), np.log(T0), points)
        t = np.exp(u)
        s = np.array([stopping(float(v)) for v in t])
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            raise ValueError('stopping power must be finite and positive')
        return float(np.trapezoid(t/s, u))
    if not np.isfinite(T0) or T0 <= .01:
        raise ValueError('T0 must exceed CUT=0.01 MeV')
    r, doubled = integral(20001), integral(40002)
    change = abs(doubled/r-1)
    return dict(range=r, relative_change=change, healthy=change <= 1e-6)


def hinge_moments(T0, stopping):
    mean, variance = 0., 0.
    while T0 > .01:
        next_T = max(.01, .95*T0)
        d = T0-next_T
        v0, v1 = 1/stopping(T0), 1/stopping(next_T)
        mean += d*(v0+v1)/2
        variance += d*d*(v0-v1)**2/12
        T0 = next_T
    return dict(mean=mean, variance=variance)


def c1_point(T0, length, seed, inputs):
    from .transport import run
    start = time.perf_counter()
    result = run(float(T0), 2000, seed, .05, .025, np.array([0., length]),
                 length, *inputs[:4], scoring=False, hard_on=False, mode=2)
    return dict(ledger=result[1], counts=result[2], seconds=time.perf_counter()-start)


def c1_point_healthy(result, T0):
    ledger, counts = result['ledger'], result['counts']
    return bool(finite(result) and ledger.shape == (2000, 7)
                and not np.any(counts[5:7]) and not np.any(counts[1:3])
                and not np.any(ledger[:, 4:6]) and not np.any(ledger[:, 1])
                and np.all(np.abs(ledger[:, 3]) <= T0*1e-9))


def c1_statistics(p, healthy=True):
    p = np.asarray(p)
    valid = (healthy and p.shape == (41,) and np.all(np.isfinite(p))
             and np.all((p >= 0) & (p <= 1)) and p[0] == 1 and p[-1] == 0)
    if not valid:
        return dict(status='判定できない', difference=None, sem=None)
    difference = float(-.01+p.sum()*.0005-.00025)
    sem = float(.0005*np.sqrt(np.sum(p*(1-p))/2000))
    return dict(status=classify(difference, sem, .001, .001/3),
                difference=difference, sem=sem, quadrature_bound=.00025)


class _Pre(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.current = [], None

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'pre':
            self.current = []
        elif tag.lower() == 'br' and self.current is not None:
            self.current.append('\n')

    def handle_endtag(self, tag):
        if tag.lower() == 'pre' and self.current is not None:
            self.blocks.append(''.join(self.current))
            self.current = None

    def handle_data(self, data):
        if self.current is not None:
            self.current.append(data)


def parse_estar(raw):
    def fail(message):
        raise ValueError('ESTAR response differs from assumed seven-column CSDA table: '+message)
    parser = _Pre()
    try:
        parser.feed(raw.decode('utf-8'))
    except UnicodeDecodeError:
        fail('not UTF-8 HTML')
    # Plan amendment 2: the real header splits "CSDA" and "RANGE" over two lines.
    candidates = [b for b in parser.blocks if re.search(r'CSDA', b) and re.search(r'RANGE', b, re.I)]
    if len(candidates) != 1:
        fail('expected exactly one PRE block with CSDA Range header')
    block = candidates[0]
    if not re.search(r'g\s*/\s*cm\s*\^?2', block, re.I):
        fail('missing range units g/cm2')
    rows = []
    # Amendment 2: the density / I line precedes the units line; table rows follow it.
    lines = block.splitlines()
    units = [i for i, line in enumerate(lines) if re.search(r'g\s*/\s*cm\s*\^?2', line, re.I) and 'MeV' in line]
    if len(units) != 1:
        fail('expected exactly one units line')
    for line in lines[units[0]+1:]:
        fields = line.split()
        if not fields or not re.match(r'^[+\-]?(?:\d|\.\d)', fields[0]):
            continue
        try:
            row = [float(v) for v in fields]
        except ValueError:
            fail('nonnumeric table row: '+line)
        if len(row) != 7:
            fail(f'expected 7 columns, got {len(row)}')
        rows.append(row)
    a = np.asarray(rows)
    if a.ndim != 2 or a.shape[0] < 4 or not np.all(np.isfinite(a)):
        fail('missing rows or nonfinite data')
    if np.any(a[:, 0] <= 0) or np.any(a[:, 4] <= 0) or np.any(np.diff(a[:, 0]) <= 0):
        fail('energies must increase strictly and energies/ranges must be positive')
    ranges = {}
    for energy in (.01, .1, 1., 10.):
        matches = a[a[:, 0] == energy, 4]
        if len(matches) != 1:
            fail(f'exact energy {energy:g} MeV missing (no interpolation)')
        ranges[energy] = float(matches[0])
    if np.any(np.diff(list(ranges.values())) <= 0):
        fail('required ranges are not increasing')
    return ranges


def request_estar():
    """Fetch and validate without altering the original response bytes."""
    # Plan amendment 2: urllib's default client is refused (HTTP 403); use curl as fetch_data.py does.
    import subprocess
    args = ['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '2', '--max-time', '60']
    for key, value in ESTAR_FIELDS.items():
        args += ['--data-urlencode', f'{key}={value}']
    done = subprocess.run(args+[ESTAR_URL], capture_output=True)
    if done.returncode:
        raise RuntimeError(done.stderr.decode(errors='replace'))
    raw = done.stdout
    ranges = parse_estar(raw)
    metadata = dict(url=ESTAR_URL, fields=ESTAR_FIELDS,
                    retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    sha256=hashlib.sha256(raw).hexdigest())
    return raw, ranges, metadata


def c2_statistics(T0, integral, ranges=None, reason=None):
    if ranges is None:
        return dict(status='未実施', difference=None, sem=None, reason=reason or 'ESTAR table unavailable')
    denominator = ranges[T0]-ranges[.01]
    if denominator <= 0 or not np.isfinite(denominator):
        return dict(status='判定できない', difference=None, sem=None)
    difference = integral['range']/denominator-1
    return dict(status=classify(difference, 0., .001, 0., integral['healthy']),
                difference=difference, sem=0.)


RULES = dict(backend='elsepa', fe=.05, k=.025, batches=100,
             B_energies=list(B_ENERGIES), C_energies=list(C_ENERGIES),
             pilot_n=100000, max_n=5000000, pilot_seed=20291790,
             production_seed=20291810, b2_n=2000000, b2_seed=20291796,
             c1_n=2000, c1_seed=20291800, x=X.tolist(),
             B1_difference_limit=.01, B1_sem_limit=.001,
             C1_difference_limit=.001, C1_sem_limit=.001/3,
             C2_difference_limit=.001, integral_points=20001,
             integral_check_points=40002)
FIGURE = 'p4bc_summary.png'


def require_absent(*paths):
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f'{path}: stage output already exists; no overwrite')


def write_bytes_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(value)


def write_json_once(path, value):
    write_bytes_once(path, (encoded(value)+'\n').encode('utf-8'))


def fetch_outputs(out, request=request_estar):
    """Testable persistence boundary; reject existing files before any request."""
    out = Path(out)
    html, meta = out/'estar_range_water.html', out/'fetch_p4bc.json'
    require_absent(html, meta)
    raw, _, metadata = request()
    parse_estar(raw)  # Validate even injected responses; never silently guess columns.
    if (metadata['sha256'] != hashlib.sha256(raw).hexdigest()
            or metadata['url'] != ESTAR_URL or metadata['fields'] != ESTAR_FIELDS):
        raise RuntimeError('ESTAR response metadata mismatch')
    write_bytes_once(html, raw)
    write_json_once(meta, metadata)
    return metadata


def reference(out):
    """Missing/unreadable reference means C2 unperformed, not fabricated data."""
    out = Path(out)
    meta_path = out/'fetch_p4bc.json'
    if not meta_path.exists():
        return None, dict(status='未実施', reason='fetch_p4bc.json がない')
    metadata = json.loads(meta_path.read_text())
    if metadata.get('url') != ESTAR_URL or metadata.get('fields') != ESTAR_FIELDS:
        raise RuntimeError('ESTAR fetch settings mismatch')
    try:
        raw = (out/'estar_range_water.html').read_bytes()
    except OSError as exc:
        return None, dict(status='未実施', reason=str(exc), metadata=metadata)
    if hashlib.sha256(raw).hexdigest() != metadata.get('sha256'):
        raise RuntimeError('ESTAR response sha256 mismatch')
    try:
        ranges = parse_estar(raw)
    except ValueError as exc:
        return None, dict(status='未実施', reason=str(exc), metadata=metadata)
    return ranges, dict(status='取得済み', metadata=metadata)


class Workflow:
    """Serial condition checkpoints; CLI wrappers below only wire real inputs.

    Tests replace transport/provenance functions, and use temporary out/cache.
    No worker processes, automatic retries or altered seeds are introduced.
    """
    def __init__(self, inputs, out=OUT, cache=CACHE):
        self.inputs, self.out, self.cache = inputs, Path(out), Path(cache)
        self.current = provenance(inputs)

    def verify(self):
        if provenance(self.inputs) != self.current:
            raise RuntimeError('P4bc source/table/version changed during workflow')

    def config(self, kind, **settings):
        return dict(kind=kind, settings=settings, rules=RULES, provenance=self.current)

    def condition(self, name, config, calculate=None):
        self.verify()
        path = self.cache/(name+'.npz')
        if calculate is None:
            result = load_result(path, config)
        else:
            def checked():
                result = calculate()
                self.verify()
                return result
            result = saved(path, config, checked)
        self.verify()
        return result

    def b1(self, i, n, production=False, read_only=False):
        seed = (20291810 if production else 20291790)+i
        kind = 'B1_production' if production else 'B1_pilot'
        config = self.config(kind, E=B_ENERGIES[i], n=n, seed=seed,
                             thickness=2000., source_z=1000., rule='own',
                             annihilate_in_flight=True, brems_mode=2,
                             score_kerma=True, score_pair=False)
        calc = None if read_only else lambda: execute_b1(B_ENERGIES[i], n, seed, self.inputs)
        return self.condition(f'p4bc_{kind}_{i}', config, calc)

    def b2(self, read_only=False):
        from .run_p3a import SPECTRUM_SHA
        config = self.config('B2', E=10., n=2000000, seed=20291796,
                             thickness=30., rule='own', annihilate_in_flight=True,
                             brems_mode=2, score_kerma=True, score_pair=False,
                             spectrum_sha256=SPECTRUM_SHA)
        return self.condition('p4bc_B2', config,
                              None if read_only else lambda: execute_b2(self.inputs))

    def collect_c(self):
        from .physics import stopping
        ranges, ref = reference(self.out)
        c1, c2 = [], []
        for i, energy in enumerate(C_ENERGIES):
            started = time.perf_counter()
            integral = integrate_range(energy, lambda t: stopping(t, False))
            hinge = hinge_moments(energy, lambda t: stopping(t, False))
            numerical_seconds = time.perf_counter()-started
            points, health, seconds = [], [], 0.
            for j, x in enumerate(X):
                length = integral['range']*(1+x)
                seed = 20291800+100*i+j
                config = self.config('C1', E=energy, n=2000, seed=seed,
                                     x=float(x), thickness=length, scoring=False,
                                     hard_on=False, mode=2)
                result = self.condition(f'p4bc_C1_{i}_{j:02d}', config,
                    lambda: c1_point(energy, length, seed, self.inputs))
                points.append(float(np.mean(result['ledger'][:, 2] > 0)))
                health.append(c1_point_healthy(result, energy))
                seconds += float(result['seconds'])
            row = c1_statistics(points, integral['healthy'] and all(health))
            hinge['difference'] = hinge['mean']/integral['range']-1
            row.update(E=energy, integral=integral, hinge=hinge, p=points,
                       point_healthy=health, seconds=seconds+numerical_seconds,
                       quadrature_bound=.00025)
            c1.append(row)
            c2.append(dict(E=energy, **c2_statistics(energy, integral, ranges, ref.get('reason'))))
        return c1, c2, ref

    def make_lock(self):
        require_absent(self.out/'lock_p4bc.json')
        c1, c2, ref = self.collect_c()
        pilots, runs = [], []
        for i, energy in enumerate(B_ENERGIES):
            result = self.b1(i, 100000)
            stats = b1_statistics(result, 100000, energy)
            # Pilot values size the production, never constitute a B1 verdict.
            healthy = stats['status'] != '判定できない'
            stats['status'] = '報告のみ' if healthy else '判定できない'
            pilots.append(dict(E=energy, n=100000, seed=20291790+i,
                               seconds=float(result['seconds']), **stats))
            n = required_histories(stats['sem']) if healthy else None
            runs.append(dict(E=energy, n=n, seed=20291810+i))
        lock = dict(provenance=self.current, rules=RULES, C1=c1, C2=c2,
                    reference=ref, pilot=pilots, runs=runs)
        self.verify()
        write_json_once(self.out/'lock_p4bc.json', lock)
        return lock

    def read_lock(self):
        self.verify()
        lock = json.loads((self.out/'lock_p4bc.json').read_text())
        if lock.get('provenance') != self.current or lock.get('rules') != RULES:
            raise RuntimeError('P4bc lock source/table/version/settings mismatch')
        if len(lock.get('runs', [])) != 5 or len(lock.get('pilot', [])) != 5:
            raise RuntimeError('P4bc lock must contain five B1 conditions')
        if len(lock.get('C1', [])) != 3 or len(lock.get('C2', [])) != 3:
            raise RuntimeError('P4bc lock must contain three C1 and C2 conditions')
        for i, (spec, pilot) in enumerate(zip(lock['runs'], lock['pilot'])):
            if (spec.get('E') != B_ENERGIES[i] or spec.get('seed') != 20291810+i
                    or pilot.get('E') != B_ENERGIES[i] or pilot.get('n') != 100000
                    or pilot.get('seed') != 20291790+i):
                raise RuntimeError('P4bc lock energy/history/seed mismatch')
            if pilot.get('status') == '判定できない':
                if spec.get('n') is not None:
                    raise RuntimeError('unhealthy pilot must not fix production histories')
                continue
            n = spec.get('n')
            if (type(n) is not int or n != required_histories(pilot['sem'])
                    or n < 100000 or n % 100000):
                raise RuntimeError('P4bc lock history count differs from pilot sizing')
        return lock

    @staticmethod
    def production_allowed(lock):
        if any(spec['n'] is None for spec in lock['runs']):
            raise RuntimeError('B1 pilot 判定できない; production forbidden')
        if any(spec['n'] > 5000000 for spec in lock['runs']):
            raise RuntimeError('B1 requires more than 500万 histories; ユーザーに相談。production forbidden')

    def produce(self):
        lock = self.read_lock()
        self.production_allowed(lock)
        for i, spec in enumerate(lock['runs']):
            if self.read_lock() != lock:
                raise RuntimeError('P4bc lock changed during production')
            self.b1(i, spec['n'], production=True)
        if self.read_lock() != lock:
            raise RuntimeError('P4bc lock changed before B2')
        self.b2()

    def analysis_data(self):
        lock = self.read_lock()
        self.production_allowed(lock)
        # All required NPZ must exist and match before any analysis file is written.
        results = [self.b1(i, spec['n'], production=True, read_only=True)
                   for i, spec in enumerate(lock['runs'])]
        b2 = self.b2(read_only=True)
        b1 = [dict(E=B_ENERGIES[i], n=spec['n'], seed=spec['seed'],
                   seconds=float(result['seconds']),
                   **b1_statistics(result, spec['n'], B_ENERGIES[i]))
              for i, (spec, result) in enumerate(zip(lock['runs'], results))]
        self.verify()
        return dict(provenance=self.current, rules=RULES,
                    lock_sha256=hashlib.sha256((self.out/'lock_p4bc.json').read_bytes()).hexdigest(),
                    B1=b1, B2=b2_statistics(b2), C1=lock['C1'], C2=lock['C2'],
                    pilot=lock['pilot'], reference=lock['reference'],
                    B2_seconds=float(b2['seconds']), limitations=list(LIMITATIONS),
                    seconds=sum(row['seconds'] for row in lock['C1']+lock['pilot']+b1)
                            + float(b2['seconds']))

    def publish_analysis(self):
        paths = [self.out/name for name in ('checks_p4bc.json', FIGURE, 'RESULTS_P4BC.md')]
        require_absent(*paths)
        result = self.analysis_data()
        report = render_report(result)
        figure = render_figure(result)
        self.verify()
        write_json_once(paths[0], result)
        write_bytes_once(paths[1], figure)
        write_bytes_once(paths[2], report.encode('utf-8'))
        return result


def render_report(result):
    def fmt(v, scale=1.):
        return '—' if v is None else f'{v*scale:.8g}'
    lines = ['# P4b・P4c 結果', '', *result['limitations'], '',
             'B1は期待値の検査であり履歴ごとの恒等式ではない。許容1%、SEM上限0.1%。',
             'C1/C2の許容は0.1%、C1のSEM上限は0.1%/3。許容・種・点は変更していない。', '',
             '## B1（本番のみ判定。試し計算は粒子数の決定用）', '',
             '| E MeV | 本番N | 差 % | SEM % | 判定 | 試し差 % | 試しSEM % | 本番秒 | 試し秒 |',
             '|---|---|---|---|---|---|---|---|---|']
    for row, pilot in zip(result['B1'], result['pilot']):
        lines.append(f"| {row['E']} | {row['n']} | {fmt(row['difference'],100)} | "
                     f"{fmt(row['sem'],100)} | {row['status']} | {fmt(pilot['difference'],100)} | "
                     f"{fmt(pilot['sem'],100)} | {fmt(row['seconds'])} | {fmt(pilot['seconds'])} |")
    lines += ['', '### B1 報告のみ', '',
              '| E MeV | ρ_col | ρ_tr | ρ_tr−ρ_col | 連続制動放射 / 入射 |',
              '|---|---|---|---|---|']
    for row in result['B1']:
        lines.append('| '+str(row['E'])+' | '+' | '.join(fmt(row.get(k)) for k in
            ('rho_col', 'rho_tr', 'tr_minus_col', 'continuous_brems_fraction'))+' |')
    lines += ['', '負の差は低エネルギー光子の局所沈着の寄与と符号が整合するが、'
              'その大きさは分離していないため、差全体の説明にはならない。'
              '正の差にはこの負の寄与以外の効果が必要であり、原因は確定しない。', '',
              '## B2（報告のみ）', '',
              '1をわずかに上回りほぼ一定という見込みは仮説であり、合否は判定しない。', '',
              '| 深さ cm | Σ線量 / Σ衝突カーマ | バッチ比のSEM | 状態 |', '|---|---|---|---|']
    for row in result['B2']:
        lo, hi = row['depth_cm']
        lines.append(f"| {lo}–{hi} | {fmt(row['ratio'])} | {fmt(row['sem'])} | {row['status']} |")
    lines += ['', f"B2計算時間: {fmt(result['B2_seconds'])} s", '',
              '## C1・C2', '',
              '| 検査 | E MeV | 差 % | SEM % | 判定 |', '|---|---|---|---|---|']
    for key in ('C1', 'C2'):
        for row in result[key]:
            lines.append(f"| {key} | {row['E']} | {fmt(row['difference'],100)} | "
                         f"{fmt(row['sem'],100)} | {row['status']} |")
    lines += ['', '### 積分・ヒンジ（報告のみ）', '',
              '水の密度1 g/cm³。飛程はcm（質量飛程では数値的にg/cm²と同じ）。', '',
              '| E MeV | R_int cm | 積分倍点数の相対変化 | ヒンジ平均 cm | 分散 cm² | 平均の差 % | C1計算秒 |',
              '|---|---|---|---|---|---|---|']
    for row in result['C1']:
        h, v = row['hinge'], row['integral']
        lines.append(f"| {row['E']} | {fmt(v['range'])} | {fmt(v['relative_change'])} | "
                     f"{fmt(h['mean'])} | {fmt(h['variance'])} | {fmt(h['difference'],100)} | {fmt(row['seconds'])} |")
    lines += ['', 'C1の区間内分布による積分誤差は最大0.025%。上表のSEMには含めない。', '',
              'C2参照: '+encoded(result['reference']), '',
              'C2不合格の場合、阻止能の補間・密度効果の表・放射阻止能が検討対象となる。'
              'この比較だけでは寄与を分離できず、許容は変更しない。', '',
              f"総計算時間（保存済み計算分を含む）: {fmt(result['seconds'])} s", '',
              f'![P4bc summary]({FIGURE})', '']
    return '\n'.join(lines)


def render_figure(result):
    import io
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10, 8), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    for ax, key, limit in zip(axes.flat, ('B1', 'C1', 'C2'), (1., .1, .1)):
        rows = [row for row in result[key] if row['difference'] is not None]
        ax.errorbar([row['E'] for row in rows], [100*row['difference'] for row in rows],
                    yerr=[100*row['sem'] for row in rows], fmt='o', capsize=3)
        ax.axhline(limit, color='gray', linestyle='--')
        ax.axhline(-limit, color='gray', linestyle='--')
        ax.axhline(0, color='gray', linewidth=.5)
        ax.set(xlabel='Energy [MeV]', ylabel='Difference [%]', title=key)
    ax = axes[1, 1]
    rows = [row for row in result['B2'] if row['ratio'] is not None]
    ax.errorbar([sum(row['depth_cm'])/2 for row in rows], [row['ratio'] for row in rows],
                yerr=[row['sem'] for row in rows], fmt='o', capsize=3)
    ax.axhline(1., color='gray', linestyle='--')
    ax.set(xlabel='Depth [cm]', ylabel='Deposit / collision kerma', title='B2 (report only)')
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=150)
    return stream.getvalue()


def fetch():
    return fetch_outputs(OUT)


def run():
    from .run_p3s import inputs
    require_absent(OUT/'lock_p4bc.json')
    workflow = Workflow(inputs('elsepa'))
    lock = workflow.make_lock()
    workflow.production_allowed(lock)
    return lock


def production():
    from .run_p3s import inputs
    return Workflow(inputs('elsepa')).produce()


def analyse():
    from .run_p3s import inputs
    require_absent(*(OUT/name for name in ('checks_p4bc.json', FIGURE, 'RESULTS_P4BC.md')))
    return Workflow(inputs('elsepa')).publish_analysis()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fetch', 'run', 'production', 'analyse'))
    args = parser.parse_args(argv)
    try:
        return dict(fetch=fetch, run=run, production=production, analyse=analyse)[args.command]()
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, str(exc)+'\n')


if __name__ == '__main__':
    main()
