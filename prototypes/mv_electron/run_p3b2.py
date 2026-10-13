"""Fixed P3b-2 reference checks. No dose comparison or seed retries.

Standalone drivers compile in their caller's directory; EGS5 jobs live under
CHATCARLO_P3B_CACHE/egs5. A paused job is the only admissible existing job.
"""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import numpy as np
from scipy.stats import chisquare
from . import run_p3r as r
from .photon_transport_p3b import expand_regions
from .p2_common import ROOT, EGS, sha
from .run_p3b1 import REGIONS

CODE = ROOT/'prototypes/mv_electron'
OUT = ROOT/'docs/validation/mv/p3b'
CACHE = Path(os.environ.get('CHATCARLO_P3B_CACHE', '~/.cache/chatcarlo/p3b')).expanduser()
CHECKS = OUT/'checks_p3b2.json'
SOURCES = ('p3b_egs.f', 'p3b_egs_common.inc', 'p3b_egs_score.inc',
           'p3b_egs_ausgab.inc', 'p3b_score_driver.f', 'p3b_ausgab_driver.f',
           'run_p3b2.py', 'p3r_egs.f', 'p3r_egs_common.inc',
           'photon_transport_p3b.py', 'scoring_p3b.py', 'batch_kernels_p3b.py', 'run_p3r.py')
REGION_KEYS = ('axis_coarse', 'axis_fine', 'slab_coarse', 'profile', 'profile_fine')
LEGACY_KEYS = ('coarse', 'fine', 'totals', 'callbacks', 'nspec', 'batchin',
               'batchcb', 'batch_brems', 'maxrel', 'escapes',
               'cutoffs', 'effective_flags', 'kinetic_cutoff')
EQUALITY_KEYS = (*LEGACY_KEYS, *REGION_KEYS, 'region_ledger', 'sources', 'source_hist')
write, save, load = r.write, r.save, r.load


class NotExecuted(RuntimeError):
    """A required build or execution could not be completed."""

job_lock, spectrum_text, egs_hashes = r.job_lock, r.spectrum_text, r.egs_hashes


def source_hashes():
    return {str((CODE/x).relative_to(ROOT)): sha(CODE/x) for x in SOURCES}


def field_values(field):
    if field is None:
        return [0, 0., 0.]
    a = np.asarray(field, float)
    if a.shape != (2,) or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError('invalid field')
    return [1, *a.tolist()]


def geometry(regions):
    return expand_regions(regions, 30., .25, .1)


def box_rows(g):
    if not g:
        return []
    n = len(g['region_sizes'])
    if not 1 <= n <= 16:
        raise ValueError('box limit')
    rows = []
    for j in range(n):
        row = np.r_[g['region_lo'][j], g['region_hi'][j], g['region_axes'][j]+1,
                    g['region_origins'][j], g['region_widths'][j], g['region_sizes'][j]]
        if (row.shape != (10,) or np.any(np.isnan(row[:6])) or
            not np.all(row[:3] < row[3:6]) or not np.all(np.isfinite(row[6:])) or
            row[6] not in (1, 2, 3) or row[8] <= 0 or
            row[9] != int(row[9]) or not 1 <= row[9] <= 320):
            raise ValueError('invalid box or bin limit')
        rows.append(row)
    return rows


def regions_text(g):
    rows = box_rows(g)
    if not rows:
        return None
    return str(len(rows))+'\n'+''.join(' '.join('%.17g' % x for x in row)+'\n' for row in rows)


def read_regions(text):
    lines = text.splitlines()
    try:
        n = integer(lines[0])
        if not 1 <= n <= 16 or len(lines) != n+1:
            raise ValueError('box limit or row count')
        a = np.array([[float(x) for x in line.split()] for line in lines[1:]])
        if a.shape != (n, 10):
            raise ValueError('box columns')
        g = dict(region_lo=a[:, :3], region_hi=a[:, 3:6], region_axes=a[:, 6]-1,
                 region_origins=a[:, 7], region_widths=a[:, 8], region_sizes=a[:, 9])
        box_rows(g)
        return g
    except (IndexError, RuntimeError) as exc:
        raise ValueError('invalid regions file') from exc


def integer(s):
    # Do not silently truncate fractional counts or accept floating spellings.
    if not s or not s.lstrip('+-').isdigit():
        raise RuntimeError('invalid integer')
    x = int(s)
    if x < 0:
        raise RuntimeError('negative integer')
    return x


def parse_extra(text, N, field, regions):
    """Strict additional-tag parser, independent of legacy PEGS output."""
    g = geometry(regions)
    expected = box_rows(g)
    nr = len(expected)
    sizes = [int(row[-1]) for row in expected]
    raw = np.zeros((100, nr, max(sizes, default=0)))
    ledger = np.zeros(nr, np.int64)
    sources = np.zeros((min(1000, N), 5))
    hist = np.zeros((10, 10), np.int64)
    seen = {tag: set() for tag in ('FIELD', 'NREG', 'REGION', 'REG', 'RLEDGER', 'SRC', 'SRCHIST')}
    columns = dict(FIELD=3, NREG=1, REGION=11, REG=4, RLEDGER=2, SRC=6, SRCHIST=3)
    for line in text.splitlines():
        cells = line.split()
        if not cells or cells[0] not in seen:
            continue
        tag, values = cells[0], cells[1:]
        if len(values) != columns[tag]:
            raise RuntimeError('invalid columns: '+tag)
        if tag in ('FIELD', 'NREG'):
            loc = ()
            if tag == 'FIELD':
                integer(values[0])
                a = np.array([float(x) for x in values])
                if not np.array_equal(a, field_values(field)):
                    raise RuntimeError('FIELD mismatch')
            elif integer(values[0]) != nr:
                raise RuntimeError('NREG mismatch')
        elif tag == 'REGION':
            j = integer(values[0])-1
            loc = (j,)
            if not 0 <= j < nr:
                raise RuntimeError('extra REGION')
            integer(values[7]); integer(values[10])
            a = np.array([float(x) for x in values[1:]])
            if not np.array_equal(a, expected[j]):
                raise RuntimeError('REGION mismatch')
        elif tag == 'REG':
            b, j, k = (integer(x)-1 for x in values[:3])
            loc = (b, j, k)
            if not (0 <= b < 100 and 0 <= j < nr and 0 <= k < sizes[j]):
                raise RuntimeError('REG range')
            value = float(values[3])
            if not np.isfinite(value) or value < 0:
                raise RuntimeError('invalid deposit')
            raw[b, j, k] = value
        elif tag == 'RLEDGER':
            j = integer(values[0])-1
            loc = (j,)
            if not 0 <= j < nr:
                raise RuntimeError('RLEDGER range')
            ledger[j] = integer(values[1])
        elif tag == 'SRC':
            h = integer(values[0])-1
            loc = (h,)
            if not 0 <= h < len(sources):
                raise RuntimeError('SRC range')
            a = np.array([float(x) for x in values[1:]])
            if not np.all(np.isfinite(a)):
                raise RuntimeError('invalid SRC')
            sources[h] = a
        else:
            x, y = (integer(v)-1 for v in values[:2])
            loc = (x, y)
            if not (0 <= x < 10 and 0 <= y < 10):
                raise RuntimeError('SRCHIST range')
            hist[x, y] = integer(values[2])
        if loc in seen[tag]:
            raise RuntimeError('duplicate '+tag)
        seen[tag].add(loc)
    counts = dict(FIELD=1, NREG=1, REGION=nr, REG=100*sum(sizes),
                  RLEDGER=nr, SRC=len(sources), SRCHIST=100)
    if any(len(seen[k]) != n for k, n in counts.items()):
        raise RuntimeError('missing additional rows')
    if int(hist.sum()) != N:
        raise RuntimeError('source histogram total')
    out = dict(region_ledger=ledger, sources=sources, source_hist=hist)
    if nr:
        nl = g['nlayer']
        for j, key in enumerate(REGION_KEYS[:3]):
            out[key] = raw[:, j, :sizes[j]].copy()
        for start, key in ((3, 'profile'), (3+2*nl, 'profile_fine')):
            out[key] = np.stack([raw[:, j, :sizes[j]] for j in range(start, start+2*nl)],
                                axis=1).reshape(100, 2, nl, sizes[start])
    else:
        for key in REGION_KEYS:
            out[key] = np.zeros((100, 0))
    return out


def parse_egs(work, config):
    v, summary = r.parse_egs(work, config)
    v['cutoffs'] = np.asarray(summary['cutoffs'])
    v['effective_flags'] = np.array([summary['effective_flags'][k]
                                     for k in sorted(summary['effective_flags'])])
    v['kinetic_cutoff'] = summary['effective_electron_kinetic_cutoff_keV']
    if config['user_code'] == 'p3b_egs':
        v.update(parse_extra((work/'egs5job.out').read_text(), config['N'],
                             config['field'], config['regions']))
    return v, summary


def equality(a, b, include_extra=True):
    keys = EQUALITY_KEYS if include_extra else LEGACY_KEYS
    return {key: key in a and key in b and bool(np.array_equal(a[key], b[key])) for key in keys}


def refuse_failed():
    if CHECKS.exists():
        previous = json.loads(CHECKS.read_text())
        if any(isinstance(v, dict) and v.get('status') == '不合格' for v in previous.values()):
            raise RuntimeError('previous P3b-2 failure; no rerun permitted')


def verify_config(work, dest, config):
    if source_hashes() != config['source_hashes']:
        raise RuntimeError('source hashes changed')
    for parent in (work, dest):
        path = parent/'config.json'
        if not path.is_file() or json.loads(path.read_text()) != config:
            raise RuntimeError('saved run config mismatch')


def verify_inputs(work, expected):
    for filename in ('field.dat', 'regions.dat', 'spectrum.dat', 'noflight.dat'):
        if filename not in expected and (work/filename).exists():
            raise RuntimeError('unexpected '+filename)
    for filename, value in expected.items():
        if not (work/filename).is_file() or (work/filename).read_text() != value:
            raise RuntimeError('work input mismatch: '+filename)


def checkpoint_records(path):
    data = path.read_bytes()
    rows = []
    offset = 0
    while offset < len(data):
        if offset+4 > len(data):
            raise RuntimeError('truncated checkpoint')
        n = struct.unpack_from('=i', data, offset)[0]
        if n <= 0 or offset+8+n > len(data):
            raise RuntimeError('invalid checkpoint record')
        if struct.unpack_from('=i', data, offset+4+n)[0] != n:
            raise RuntimeError('checkpoint record marker')
        rows.append(data[offset+4:offset+4+n])
        offset += n+8
    return rows


def checkpoint_header(config):
    g = geometry(config['regions'])
    nr = len(box_rows(g))
    lo, hi = np.zeros((3, 16)), np.zeros((3, 16))
    axes, sizes = np.zeros(16, dtype='=i4'), np.zeros(16, dtype='=i4')
    origins, widths = np.zeros(16), np.zeros(16)
    if nr:
        lo[:, :nr] = g['region_lo'].T; hi[:, :nr] = g['region_hi'].T
        axes[:nr] = g['region_axes']+1; sizes[:nr] = g['region_sizes']
        origins[:nr] = g['region_origins']; widths[:nr] = g['region_widths']
    has, half, ssd = field_values(config['field'])
    return (b'P3B_CHECKPOINT  '+struct.pack('=4i2di', 1, config['N'], config['seed'], has, half, ssd, nr)+
            lo.tobytes(order='F')+hi.tobytes(order='F')+axes.tobytes()+origins.tobytes()+
            widths.tobytes()+sizes.tobytes())


def verify_checkpoint(work, config):
    rows = checkpoint_records(work/'checkpoint.bin')
    expected_count = 3 if config['user_code'] == 'p3b_egs' else 1
    if len(rows) != expected_count:
        raise RuntimeError('checkpoint record count')
    if expected_count == 3 and rows[0] != checkpoint_header(config):
        raise RuntimeError('checkpoint config mismatch')
    state = rows[1] if expected_count == 3 else rows[0]
    # Fixed P3r state layout: default integer=4, real*8=8, isdext(25).
    if len(state) != 607600:
        raise RuntimeError('checkpoint state size')
    done = struct.unpack_from('=i', state)[0]
    if not 0 < done <= config['N'] or done % (config['N']//100):
        raise RuntimeError('invalid checkpoint progress')
    if expected_count == 3:
        if len(rows[2]) != 8*320*16*100+4*16+4*100+8*5*1000+4:
            raise RuntimeError('checkpoint region state size')
    return done


def launch(work, fd):
    with (work/'run.log').open('a') as log:
        p = subprocess.Popen(['./egs5job.exe'], cwd=work, stdout=log,
                             stderr=subprocess.STDOUT, pass_fds=(fd,))
        p.wait()
    if p.returncode:
        raise NotExecuted('EGS5 run failed')


def run_egs(name, N, seed, edges, cum, field=None, regions=None, stop_after=None,
            user_code='p3b_egs', resume=False):
    r.name_check(name, N)
    if user_code not in ('p3r_egs', 'p3b_egs'):
        raise ValueError('unknown user code')
    if user_code == 'p3r_egs' and (field is not None or regions is not None):
        raise ValueError('P3r requires legacy source/scoring')
    field_values(field); regions_text(geometry(regions))
    refuse_failed()
    work = CACHE/'egs5'/name; dest = OUT/'egs5'/name
    config = dict(N=N, seed=seed, E=10., IUNRST=0, noflight=0, thickness=30., batches=100,
                  input=r.pegs_input(0), user_code=user_code,
                  field=None if field is None else list(map(float, field)),
                  regions=None if regions is None else list(map(float, regions)),
                  egs_input_hashes=egs_hashes(), source_hashes=source_hashes(),
                  **r.spec_config(edges, cum, r.SPECTRUM_SHA))
    expected = {'config.dat': f'10 {N} {seed} 0\n', user_code+'.inp': config['input']}
    for key, value in (('spectrum.dat', spectrum_text(edges, cum)),
                       ('regions.dat', regions_text(geometry(regions))),
                       ('field.dat', None if field is None else ' '.join('%.17g'%x for x in field)+'\n')):
        if value is not None:
            expected[key] = value
    filenames = ('p3r_egs.f', 'p3r_egs_common.inc') if user_code == 'p3r_egs' else SOURCES[:4]
    if resume:
        verify_config(work, dest, config)
        verify_inputs(work, expected)
        verify_checkpoint(work, config)
        if (dest/'result.npz').exists():
            raise RuntimeError('completed job; no rerun permitted')
        for filename in filenames:
            if not (work/filename).is_file() or sha(work/filename) != sha(CODE/filename):
                raise RuntimeError('work source mismatch')
    elif work.exists() or dest.exists():
        raise RuntimeError('existing job; no rerun permitted')
    with job_lock(work) as fd:
        if not resume:
            if (work/'config.json').exists() or dest.exists():
                raise RuntimeError('existing job; no rerun permitted')
            write(work/'config.json', config); write(dest/'config.json', config)
            for filename in filenames:
                shutil.copy2(CODE/filename, work/filename)
            for filename, value in expected.items():
                (work/filename).write_text(value)
        def verify():
            refuse_failed(); verify_config(work, dest, config); verify_inputs(work, expected)
            r.verify_spec(config, edges, cum)
            if egs_hashes() != config['egs_input_hashes']:
                raise RuntimeError('EGS5 inputs changed')
            for filename in filenames:
                if sha(work/filename) != sha(CODE/filename):
                    raise RuntimeError('work source mismatch')
        verify()
        epstar = ROOT/'docs/validation/mv/p1/egs5/base_2/epstar.dat'
        if (work/'epstar.dat').exists() and sha(work/'epstar.dat') != sha(epstar):
            raise RuntimeError('stopping input mismatch')
        if not (work/'epstar.dat').exists():
            shutil.copy2(epstar, work/'epstar.dat')
        old = ROOT/'docs/validation/mv/p3a/egs5/p3a_egs/config.json'
        if json.loads(old.read_text())['input'] != config['input']:
            raise RuntimeError('P3a PEGS input mismatch')
        if not (work/'egs5job.exe').exists():
            p = subprocess.run([str(EGS/'egs5run'), 'comp'], cwd=work,
                               input=user_code+'\n\n\n', capture_output=True, text=True)
            (work/'build.log').write_text(p.stdout+p.stderr)
            if p.returncode or not (work/'egs5job.exe').exists():
                raise NotExecuted('EGS5 build failed')
        binary = {'sha256': sha(work/'egs5job.exe')}
        if (work/'build_hash.json').exists():
            if json.loads((work/'build_hash.json').read_text()) != binary:
                raise RuntimeError('executable changed')
        else:
            write(work/'build_hash.json', binary)
        if (work/'gsdist.dat').exists() and not (work/'checkpoint.bin').exists():
            raise RuntimeError('gsdist without checkpoint')
        if stop_after is not None:
            if not isinstance(stop_after, int) or not 1 <= stop_after < 100:
                raise ValueError('invalid stopping batch')
            target = work/'stop_batches.dat'; value = str(stop_after)+'\n'
            if target.exists() and target.read_text() != value:
                raise RuntimeError('stop config mismatch')
            if not target.exists():
                target.write_text(value)
        launch(work, fd)
        verify()
        if 'P3R_PAUSE' in (work/'egs5job.out').read_text():
            verify_checkpoint(work, config)
            raise InterruptedError(name)
        if 'TIME_LIMIT' in (work/'egs5job.out').read_text():
            raise NotExecuted('EGS5 time limit')
        v, summary = parse_egs(work, config)
        save(dest/'result.npz', **v); write(dest/'summary.json', summary)
        if summary['status'] != '合格':
            raise RuntimeError('EGS5 legacy checks failed')
        return dest/'result.npz'


def build_driver(work, name):
    compiler = shutil.which('gfortran')
    if compiler is None:
        raise NotExecuted('gfortran is required')
    exe = work/name
    subprocess.run([compiler, '-O2', '-I'+str(CODE),
                    '-I'+str(EGS), str(CODE/(name+'.f')), '-o', str(exe)],
                   check=True, capture_output=True, text=True)
    return exe


def driver_scores(work, exe, g, segments):
    (work/'regions.dat').write_text(regions_text(g))
    np.savetxt(work/'segments.dat', np.asarray(segments), fmt='%.17g')
    out = subprocess.run([str(exe)], cwd=work, check=True, capture_output=True, text=True)
    rows = box_rows(g)
    result = [np.zeros(int(row[-1])) for row in rows]
    for line in out.stdout.splitlines():
        j, k, value = line.split()
        result[int(j)-1][int(k)-1] = float(value)
    return result


def analytic_check(work, exe):
    # Import the pre-existing tables instead of copying numerical expectations.
    from .tests.test_p3b1 import GEOMETRY, test_b4_points
    points = []
    for mark in test_b4_points.pytestmark:
        if mark.name == 'parametrize' and mark.args[0] == 'point,expected':
            points.extend(mark.args[1])
    if len(points) != 5:
        raise RuntimeError('point table count changed')
    maximum = 0.
    for p0, p1, axis, expected, _ in GEOMETRY:
        g = read_regions('1\n0 0 0 2 2 2 %d 0 1 2\n'%(axis+1))
        actual = driver_scores(work, exe, g, [[*p0, *p1, 4.]])[0]
        maximum = max(maximum, float(np.max(abs(actual-expected))))
    for point, expected in points:
        g = read_regions('1\n0 0 0 2 2 2 3 0 1 2\n')
        actual = driver_scores(work, exe, g, [[*point, *point, 4.]])[0]
        maximum = max(maximum, float(np.max(abs(actual-expected))))
    return dict(maximum=maximum, status='合格' if maximum <= 1e-12 else '不合格')


def regroup_check(v, N):
    legacy = v['coarse']/r.MEV_TO_GY_G*(N/100*.2)
    pairs = [(v['slab_coarse'].reshape(100, 30, 4).sum(2), legacy.reshape(100, 30, 5).sum(2)),
             (v['axis_coarse'].reshape(100, 60, 2).sum(2), v['axis_fine'].reshape(100, 60, 5).sum(2)),
             (v['profile'], v['profile_fine'].reshape(*v['profile'].shape, 2).sum(-1))]
    metrics = []
    for a, b in pairs:
        error = abs(a-b).reshape(100, -1).max(1)
        scale = np.maximum(a.reshape(100, -1).max(1), b.reshape(100, -1).max(1))
        metrics.append(bool(np.all(error <= 1e-9*scale)))
    return dict(rebin=metrics, status='合格' if all(metrics) and not np.any(v['region_ledger']) else '不合格')


def source_check(v, N):
    src = v['sources']
    direction = np.column_stack((src[:, :2], np.full(len(src), 100.)))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    error = float(np.max(abs(direction-src[:, 2:])))
    p = float(chisquare(v['source_hist'].ravel(), np.full(100, N/100)).pvalue)
    ok = p > .01 and error <= 1e-14 and np.all(abs(src[:, :2]) <= 5) and v['source_hist'].sum() == N
    return dict(p=p, direction_error=error, status='合格' if ok else '不合格')


def record_checks(work, exe, edges, cum):
    from . import run_p3s as s
    from .photon_transport_p3b import run_coupled
    v = run_coupled(10., 1000, 20291752, .05, .025, 30., *s.inputs('elsepa'),
                    brems_mode=2, annihilate_in_flight=True, spectrum_edges=edges,
                    spectrum_cum=cum, field_half=5., ssd=100., regions=REGIONS,
                    dz=.25, dz_fine=.1, record_deposits=1000000,
                    score_pair=False, score_kerma=False, **s.rule_kwargs('own'))
    if v[22] != 0 or np.any(v[7]) or np.any(v[20]):
        raise RuntimeError('record overflow or ledger failure')
    records = v[21]
    segments = np.column_stack((records[:, 2:8], records[:, 8]))
    g = geometry(REGIONS)
    actual = driver_scores(work, exe, g, segments)
    expected = [v[j].sum(axis=0) for j in (15, 16, 17)]
    for j in (18, 19):
        expected.extend(v[j].sum(axis=0).reshape(6, -1))
    tolerance = 1e-12*float(records[:, 8].sum())
    error = max(float(np.max(abs(a-b))) for a, b in zip(actual, expected))
    e2b = dict(maximum=error, tolerance=tolerance, status='合格' if error <= tolerance else '不合格')
    return e2b, (segments, actual, tolerance)


def partition_check(work, exe, record):
    segments, scores, tolerance = record
    intervals = [(-np.inf, -1.), (-1., 1.), (1., np.inf)]
    maximum = 0.
    for direction in ('x', 'y', 'xy'):
        rows = []
        for xlo, xhi in (intervals if direction != 'y' else [(-np.inf, np.inf)]):
            for ylo, yhi in (intervals if direction != 'x' else [(-np.inf, np.inf)]):
                rows.append([xlo, ylo, 0., xhi, yhi, 30., 3, 0., .25, 120])
        g = read_regions(str(len(rows))+'\n'+''.join(' '.join('%.17g'%x for x in row)+'\n' for row in rows))
        actual = driver_scores(work, exe, g, segments)
        maximum = max(maximum, float(np.max(abs(np.sum(actual, axis=0)-scores[2]))))
        if direction == 'xy':
            maximum = max(maximum, float(np.max(abs(actual[4]-scores[0]))))
    return dict(maximum=maximum, tolerance=tolerance, status='合格' if maximum <= tolerance else '不合格')


# Hand-calculated, asymmetric box: x in [0,2), y in [0,1), z in [0,30).
# Inputs: iarg, charge, region, xyz, uvw, ustep, edep, weight, call-howfar.
AUSGAB_CASES = (
    ((0,-1,2,-1.,.5,.5,1.,0.,1.,4.,8.,.5,0), {1:.5,2:1.,3:.5},4.,0.,0),
    ((0,-1,2,-1.,.5,.5,1.,0.,0.,4.,8.,.5,0), {0:2.},4.,0.,0),
    ((0,-1,2,.5,.5,29.5,0.,0.,1.,4.,2.,.5,1), {29:1.},1.,0.,0),
    ((0,-1,2,.5,.5,.5,0.,0.,-1.,4.,2.,.5,1), {0:1.},1.,0.,0),
    ((1,-1,2,.5,.5,0.,0.,0.,1.,0.,8.,.5,0), {0:4.},4.,0.,0),
    ((2,-1,2,.5,.5,1.,0.,0.,1.,0.,8.,.5,0), {1:4.},4.,0.,0),
    ((4,-1,2,.5,.5,2.,0.,0.,1.,0.,8.,.5,0), {2:4.},4.,0.,0),
    ((4,-1,2,.5,1.5,1.,0.,0.,1.,0.,8.,.5,0), {},4.,0.,0),
    ((3,-1,3,.5,.5,1.,0.,0.,1.,0.,8.,.5,0), {},0.,4.,1),
    ((0,0,2,.5,.5,1.,0.,0.,1.,2.,0.,.5,0), {},0.,0.,0),
    *((((i,-1,2,.5,.5,1.,0.,0.,1.,2.,8.,.5,0)), {},0.,0.,0) for i in range(27,31)),
)


def ausgab_check(work, exe):
    (work/'regions.dat').write_text('1\n0 0 0 2 1 30 3 0 1 30\n')
    maximum = 0.
    for values, contributions, dep, escape, escapes in AUSGAB_CASES:
        result = subprocess.run([str(exe)], cwd=work, input=' '.join(map(str, values))+'\n',
                                capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        totals = list(map(float, lines[0].split()[1:]))
        actual = np.array([float(line.split()[2]) for line in lines[1:]])
        expected = np.zeros(30)
        for index, amount in contributions.items():
            expected[index] = amount
        expected_step = .5 if values[-1] == 1 else values[9]
        maximum = max(maximum, float(np.max(abs(actual-expected))),
                      abs(totals[0]-dep), abs(totals[1]-escape), abs(totals[3]-escapes),
                      abs(totals[2]-expected_step))
    return dict(maximum=maximum, status='合格' if maximum <= 1e-12 else '不合格')


def main():
    # P3b-1 cache may exist; only this stage's egs5 and driver work is guarded.
    if CHECKS.exists() or (OUT/'RESULTS_P3B2.md').exists() or (CACHE/'egs5').exists() or (CACHE/'p3b2_drivers').exists():
        raise SystemExit('P3b-2 results/work exists; no rerun permitted')
    results = dict(source_hashes=source_hashes())
    def gate(name, call):
        try:
            result = call()
        except (NotExecuted, OSError, subprocess.CalledProcessError) as exc:
            result = dict(status='未実行', reason=str(exc))
        except Exception as exc:
            result = dict(status='不合格', reason=str(exc))
        results[name] = result
        write(CHECKS, results)
        if result['status'] != '合格':
            raise SystemExit(name+' stopped; no rerun permitted')
    edges, _, cum = r.base.load_spectrum()
    baseline = {}
    def e1():
        for name, code, regions in (('e1a','p3r_egs',None), ('e1b','p3b_egs',None), ('e1c','p3b_egs',REGIONS)):
            baseline[name] = load(run_egs(name, 200000, 20291750, edges, cum, regions=regions, user_code=code))
        comparisons = {name: equality(baseline['e1a'], baseline[name], include_extra=False) for name in ('e1b','e1c')}
        ok = all(all(row.values()) for row in comparisons.values()) and not np.any(baseline['e1c']['region_ledger'])
        return dict(equality=comparisons, status='合格' if ok else '不合格')
    gate('E1', e1)
    work = CACHE/'p3b2_drivers'; work.mkdir(parents=True)
    driver = []
    def e2a():
        driver.append(build_driver(work, 'p3b_score_driver'))
        return analytic_check(work, driver[0])
    gate('E2a', e2a)
    exe = driver[0]
    records = []
    def e2b():
        result, record = record_checks(work, exe, edges, cum); records.append(record)
        return result
    gate('E2b', e2b)
    gate('E2c', lambda: partition_check(work, exe, records[0]))
    gate('E2d', lambda: ausgab_check(work, build_driver(work, 'p3b_ausgab_driver')))
    full = {}
    def e3():
        full.update(load(run_egs('e3', 200000, 20291751, edges, cum, field=[5.,100.], regions=REGIONS)))
        return regroup_check(full, 200000)
    gate('E3', e3)
    gate('E4', lambda: source_check(full, 200000))
    def e5():
        try:
            run_egs('e5', 200000, 20291751, edges, cum, field=[5.,100.], regions=REGIONS, stop_after=40)
        except InterruptedError:
            pass
        else:
            raise RuntimeError('40-batch pause missing')
        resumed = load(run_egs('e5', 200000, 20291751, edges, cum, field=[5.,100.], regions=REGIONS, resume=True))
        eq = equality(full, resumed)
        return dict(equality=eq, status='合格' if all(eq.values()) else '不合格')
    gate('E5', e5)
    results['E6'] = dict(status='報告のみ', baseline_us=float(baseline['e1a']['seconds'])/200000*1e6,
                         field_us=float(full['seconds'])/200000*1e6,
                         ratio=float(full['seconds'])/float(baseline['e1a']['seconds']))
    write(CHECKS, results)
    (OUT/'RESULTS_P3B2.md').write_text('# P3b-2\n\n'+
        '\n'.join(f"- {key}: {value['status']}" for key, value in results.items() if isinstance(value,dict) and 'status' in value)+
        '\n\nE6 (µs/history): baseline %.6g, field %.6g, ratio %.6g.\n'%
        (results['E6']['baseline_us'], results['E6']['field_us'], results['E6']['ratio'])+
        '\n領域の値はバッチごとのMeVの和。試作との線量比較・対称性検査はP3b-3の対象。\n')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
