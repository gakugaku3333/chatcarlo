"""Approved H6 launcher: prepare locally, run on one owned T4, judge locally.

Private intermediates must stay outside the repository and ~/.cache. No GPU or
CPU calculation is started by import or --help. A lost VM is never replaced.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(HERE)]
from a3_entry_speed import SharedClient, POLL_SECONDS, SITE_CUSTOMIZE, ratio
from a3_t4_gates import writable, reserve, no_cache_writes, sha, verify_old_diagnostic, tar_bytes
from a4u_h6_fixtures import write

FORMAT = 'g5-a4u-h6-v1'
# Retain the shared client's cadence constant; H6 collects on A3's one-minute polls.
# A ten-minute cadence put a 6*600 s floor on the T4 session, almost all of it idle
# (each task sleeps once before its first completion check). User decision 2026-11-07.
H6_POLL_SECONDS = POLL_SECONDS
TASKS = ('A1', 'A2', 'A3', 'B', 'C1', 'C2')


def source_files():
    files = {}
    for folder in ('chatcarlo', 'prototypes/mv_electron', 'docs/validation/gpu', 'docs/validation/mv'):
        for p in (ROOT/folder).rglob('*'):
            if (p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and
                    p.suffix in ('.py', '.h', '.cpp', '.cu', '.json', '.yaml', '.yml', '.csv') and
                    not p.is_relative_to(HERE/'a4u_h6_run')):
                files['source/'+str(p.relative_to(ROOT))] = p
    # Prototype import-time constants, already frozen; read and transfer only.
    for n in ('estar.npz', 'estar_water.html'):
        p = ROOT/'docs/validation/mv/p1/data'/n
        if p.exists(): files['source/'+str(p.relative_to(ROOT))] = p
    return files


def identity():
    return {name: sha(p) for name, p in source_files().items()}


def bundle_quantities(path, sid, device):
    """G1's registered ten paired million-history batches, same sources/seeds."""
    import numpy as np
    from chatcarlo.gpu import bundle, core, validation
    from chatcarlo.gpu.source_host import sample_source_photons
    a, src, spec, settings, _ = bundle.load(path)
    begin = time.perf_counter(); driver = core.Core(a, device)
    initialization_s = time.perf_counter()-begin; timer = None
    if device == 'cuda':
        from a0_speed import EventDelegate
        timer = EventDelegate(driver.fn, driver.cp); driver.fn = timer
    quantities = []; additions = []; keys = None
    for batch in range(10):
        seed = 6000+batch; source_seed = 7000+batch
        rng = np.random.default_rng(source_seed); total = {}; counts = {}
        for start in range(0, 1000000, 200000):
            r = driver.run(*sample_source_photons(src, 200000, rng, spec), seed=seed, start=start,
                           fluorescence=settings['fluorescence'])
            q = validation.quantities(r, settings['material_names'], sid, 1000000)
            k = validation.quantity_additions(r, settings['material_names'], sid, 1000000)
            for name in q:
                total[name] = total.get(name, 0.)+q[name]
                counts[name] = counts.get(name, 0)+k[name]
        keys = list(total); quantities.append(list(total.values())); additions.append([counts[k]+4 for k in keys])
    return dict(values=quantities, additions=additions, keys=keys,
                initialization_s=initialization_s, kernel_s=None if timer is None else sum(timer.seconds))


def voxel_batches(path, grid_record, device, destination):
    import numpy as np
    from chatcarlo.gpu import bundle, g2_core
    from chatcarlo.gpu.source_host import sample_source_photons
    a, src, spec, settings, _ = bundle.load(path)
    driver = g2_core.Core(a, device, grid=g2_core.grid_from(grid_record), count_additions=True)
    rows = {k: [] for k in ('kerma', 'h10', 'grid_additions')}
    for batch in range(10):
        seed = 221100+batch
        ss = int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0])
        r = driver.run(*sample_source_photons(src, 200000, np.random.default_rng(ss), spec),
                       seed=seed, fluorescence=settings['fluorescence'])
        for key in rows: rows[key].append(r[key])
    np.savez(destination, **{k: np.asarray(v) for k, v in rows.items()})


def prepare(args):
    import numpy as np
    import a1_g2gate3, a1_g2vox, a4u_s1_gate1, a4u_h6_fixtures as fixtures
    from a1_launch import create_new
    from a1_speed import load_configs
    from a0_bootstrap import verify_files
    from chatcarlo.gpu import bundle, therapy_inputs
    from chatcarlo.gpu import core as core_module
    from chatcarlo.scene import load_scene
    from chatcarlo.tally import VoxelGrid
    out = writable(args.output)
    if out.exists(): raise ValueError('prepare requires a new output directory')
    for name in ('p7', 'p8g2', 'old_diagnostic', 'therapy_bundle'):
        if getattr(args, name) is None: raise ValueError('prepare requires --'+name.replace('_', '-'))
    verify_old_diagnostic(args.old_diagnostic)
    verify_files(args.p7, json.loads((args.p7/'p7_manifest.json').read_text()))
    therapy_inputs.load(args.therapy_bundle, 'mohan10')
    reserve(out, 512*1024**2); out.mkdir(parents=True)
    base = out/'inputs'; base.mkdir()
    # Small B fixtures are fixed FIRST, before the large CPU baselines.
    fixtures.freeze_inputs(base/'fixtures', args.p8g2); fixtures.run(base/'fixtures')
    for sid in ('s1', 's2', 's3'):
        scene, plane, _ = a4u_s1_gate1.inputs(sid)
        bundle.create(scene, plane, base/'A1'/sid)
        write(base/'A1'/f'{sid}_cpu.json', bundle_quantities(base/'A1'/sid, sid, 'cpu'))
    grids = {}
    for name, path in a1_g2vox.SCENES.items():
        from chatcarlo.geometry import Geometry
        scene = load_scene(str(path)); geom = Geometry(scene.raw['geometry'])
        grid = VoxelGrid.from_bbox(geom.tally_bbox_min, geom.tally_bbox_max, 2.)
        grids[name] = dict(origin_cm=grid.origin_cm.tolist(), shape=list(map(int, grid.shape)),
                           voxel_size_cm=float(grid.voxel_size_cm), layout_version='g2-grid-v1')
        bundle.create(scene, None, base/'A2'/name, g2_tally=True)
        voxel_batches(base/'A2'/name, grids[name], 'cpu', base/'A2'/f'{name}_cpu.npz')
    write(base/'A2/grids.json', grids)
    cfg, _ = load_configs(HERE/'a1_speed_configs.json')
    create_new(base/'diagnostic/new', cfg)
    # C2's measurement path, exercised on the CPU before any VM exists. T4 session 5
    # died in it with KeyError: 'category_count' -- the detector-less g2b config -- and
    # nothing local had ever run it. The timing itself needs a GPU; the record
    # reconstruction and the history count do not, and that is where the bug was.
    from chatcarlo.gpu import g2_core, validation
    from chatcarlo.gpu.source_host import sample_source_photons
    for c in cfg['configurations']:
        ca, csrc, cspec, _, _ = bundle.load(base/'diagnostic/new'/c['id'])
        cls = g2_core.Core if c['grid'] else core_module.Core
        obj = cls(ca, 'cpu', **(dict(grid=g2_core.grid_from(c['grid'])) if c['grid'] else {}))
        probe = 2000
        got = obj.run(*sample_source_photons(csrc, probe, np.random.default_rng(c['source_seed']), cspec),
                      seed=c['seed'], start=c['start'], **dict(c['flags'], record=True, observe=False))
        if isinstance(got, tuple): got = got[0]
        rebuilt = validation.record_reduce(got, obj.a)
        if 'category_count' in got: limits = validation.reconstruction_bounds(got, rebuilt, probe)
        else:
            if 'images' in rebuilt or 'category_count' in rebuilt:
                raise ValueError('detector-less run produced image records: '+c['id'])
            limits = validation.reconstruction_bounds(dict(got, category_count=np.zeros((4, 1, 1))), rebuilt, probe)
        if any(np.any(np.abs(rebuilt[k]-got[k]) > limits[k]) for k in rebuilt):
            raise ValueError('CPU record reconstruction fails for speed config '+c['id'])
        if got['completed'] != probe: raise ValueError('CPU history count for speed config '+c['id'])
    # A3's gate2a rebuilds an artificial sub-5-keV fluorescence table through
    # core.pack, which regenerates the photon tables with xraylib. The VM has no
    # xraylib on purpose (the runbook: the host generates tables, the VM reads
    # them), so T4 session 2 stopped there. Pack it HERE, and only trust the
    # stand-in after checking that the UNEDITED packing reproduces the bundle byte
    # for byte -- that is what makes this the same bytes the VM would have made.
    from chatcarlo.gpu import core as gpu_core
    from chatcarlo.gpu.validation import unpack_tables
    g4 = base/'diagnostic/new/g4'
    ab = bundle.load(g4)[0]

    def packing(edit):
        art = {k: v.copy() for k, v in ab.items()}; at = unpack_tables(art)
        if edit:
            at.k_edge[0, 0] = 1.; at.k_omega[0, 0] = 1.; at.k_frac[0, 0, :] = 1.
            at.n_lines[0, 0] = 1; at.k_line_e[0, 0, :] = 4.
            at.k_line_p[0, 0, :] = 0.; at.k_line_p[0, 0, 0] = 1.
        at.material_names = list(range(len(at.n_elem)))
        return gpu_core.pack(at)

    if not np.array_equal(np.asarray(packing(False)['data']), np.asarray(ab['data'])):
        raise ValueError('host packing does not reproduce the g4 bundle; it cannot stand in for the VM')
    artificial = packing(True)
    changed = int(np.count_nonzero(np.asarray(artificial['data']) != np.asarray(ab['data'])))
    if not changed: raise ValueError('the artificial fluorescence packing changed nothing')
    np.savez(base/'A3_artificial.npz', changed_elements=np.array(changed),
             **{k: np.asarray(v) for k, v in artificial.items()})
    a1_g2gate3.prepare(base/'C1')
    # P7-A input comes from the sealed bundle (private spectrum remains external).
    materials, arrays, fg, _, _ = therapy_inputs.load(args.therapy_bundle, 'mohan10')
    arrays = {k: v for k, v in arrays.items() if not k.startswith('_')}
    np.savez(base/'water_A.npz', **arrays, field_geometry=np.frombuffer(bytes(fg), np.uint8),
             half=np.array(2.5), ssd=np.array(100.))
    # References stay read-only; payload() streams these without staging copies.
    record = dict(format=FORMAT, sources=identity(), poll_seconds=H6_POLL_SECONDS,
                  external={k: str(getattr(args, k).expanduser().resolve()) for k in
                            ('p7', 'old_diagnostic')},
                  fixture_expected_sha256=sha(base/'fixtures/expected.json'),
                  external_hashes={k: {str(p.relative_to(getattr(args, k))): sha(p) for p in getattr(args, k).rglob('*') if p.is_file()} for k in ('p7', 'old_diagnostic')},
                  inputs={str(p.relative_to(base)): sha(p) for p in base.rglob('*')
                          if p.is_file() and not p.is_relative_to(base/'fixtures/cpu')})
    write(out/'prepared.json', record)
    print('CPU baselines and B1--B5 reach counts frozen. run will own one T4; judge stays local.')


def prepared(out):
    record = json.loads((out/'prepared.json').read_text())
    if record['format'] != FORMAT or record['sources'] != identity(): raise ValueError('prepared source identity changed')
    if record['fixture_expected_sha256'] != sha(out/'inputs/fixtures/expected.json'): raise ValueError('CPU fixture changed')
    for name, digest in record['inputs'].items():
        if sha(out/'inputs'/name) != digest: raise ValueError('prepared input changed: '+name)
    for key, files in record['external_hashes'].items():
        for name, digest in files.items():
            if sha(Path(record['external'][key])/name) != digest: raise ValueError('external input changed: '+key+'/'+name)
    verify_old_diagnostic(Path(record['external']['old_diagnostic']))
    return record


def payload(out, record):
    files = source_files()
    for p in (out/'inputs').rglob('*'):
        if not p.is_file(): continue
        if p.is_relative_to(out/'inputs/fixtures/cpu'): continue
        files['inputs/'+str(p.relative_to(out/'inputs'))] = p
    for name, folder in (('bundles/p7', record['external']['p7']),
                         ('diagnostic/old', record['external']['old_diagnostic'])):
        for p in Path(folder).rglob('*'):
            if p.is_file(): files['inputs/'+name+'/'+str(p.relative_to(folder))] = p
    reserve(out, tar_bytes(files)+64*1024**2)
    archive = out/'payload.tar'
    with tarfile.open(archive, 'w') as t:
        for name, p in sorted(files.items()): t.add(p, arcname=name, recursive=False)
    write(out/'transfer.json', {name: sha(p) for name, p in files.items()})
    return archive


class Owner:
    """Every colab invocation is bounded. Never retry creation or a launcher."""
    def __init__(self, session, out): self.session, self.out, self.rows = session, out, []
    def call(self, action, *args, code=None, idempotent=False):
        begin = time.perf_counter(); error = None
        try:
            result = subprocess.run(['colab', action, '-s', self.session, *map(str, args)], input=code,
                                    text=True, capture_output=True, check=True,
                                    timeout={'new': 900, 'upload': 900, 'download': 900, 'stop': 180}.get(action, 300))
            return result.stdout
        except Exception as exc:
            error = str(exc); raise
        finally:
            self.rows.append(dict(operation=action, seconds=time.perf_counter()-begin, error=error))
            write(self.out/'transport_timings.json', self.rows)
    def raw(self, code, **kw): return self.call('exec', code=code, **kw)
    def status(self, code):
        for line in self.raw(code, idempotent=True).splitlines():
            if line.startswith('H6_JSON '): return json.loads(line[8:])
        raise RuntimeError('missing H6 status')


def upload(owner, archive, remote):
    owner.raw(f"from pathlib import Path\nPath({remote!r}).mkdir(parents=True,exist_ok=False)\n")
    parts = []
    with archive.open('rb') as source:
        while block := source.read(20*1024**2):
            p = archive.with_name('upload_part'); reserve(p, len(block)); p.write_bytes(block)
            name = f'part{len(parts):04d}'; owner.call('upload', p, remote+'/'+name); parts.append((name, sha(p)))
    # Assembly/extraction is detached and supervised by its exact PID too.
    code = f'''import pathlib,hashlib,tarfile
r=pathlib.Path({remote!r})
with (r/'payload.tar').open('wb') as dst:
 for name,digest in {parts!r}:
  p=r/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
  with p.open('rb') as src:
   while block:=src.read(1048576):dst.write(block)
h=hashlib.sha256()
with (r/'payload.tar').open('rb') as f:
 while block:=f.read(1048576):h.update(block)
assert h.hexdigest()=={sha(archive)!r}
with tarfile.open(r/'payload.tar') as t:t.extractall(r,filter='data')
(r/'ASSEMBLED').write_text('done')
'''
    launch(owner, remote, 'assemble', code=code)
    while True:
        state = status(owner, remote, 'assemble', 'ASSEMBLED')
        if state['done']: return
        if not state['alive']: raise RuntimeError('payload assembly failed')
        time.sleep(POLL_SECONDS)


def launch(owner, remote, task, code=None):
    source = remote+'/source'; worker = source+'/docs/validation/gpu/g5/a4u_h6.py'
    command = ['python3', worker, '_worker', '--output', remote, '--task', task]
    snippet = f'''import pathlib,subprocess,sys,os
r=pathlib.Path({remote!r});log=(r/{task+'.log'!r}).open('wb')
'''
    if code is not None:
        snippet += f"(r/{task+'.py'!r}).write_text({code!r})\n"
        command = ['python3', remote+'/'+task+'.py']
    snippet += f'''env=dict(os.environ,PYTHONPATH={source!r}+':'+{source+'/docs/validation/gpu/g5'!r},PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR={remote+'/mpl'!r},NUMBA_CACHE_DIR={remote+'/numba'!r})
p=subprocess.Popen({command!r},cwd={remote if code is not None else source!r},env=env,stdout=log,stderr=log,start_new_session=True)
(r/{task+'.pid'!r}).write_text(str(p.pid))
'''
    client = SharedClient(owner, remote, remote+'/cupy_cache')
    owner.raw(f"from pathlib import Path\np=Path({remote+'/instrument'!r});p.mkdir(exist_ok=True)\n(p/'sitecustomize.py').write_text({SITE_CUSTOMIZE!r})\n")
    client.remote_exec(snippet)


def status(owner, remote, task, marker):
    return owner.status(f'''import pathlib,json,os
r=pathlib.Path({remote!r});pid=int((r/{task+'.pid'!r}).read_text())
try:
 os.kill(pid,0)
 alive=(pathlib.Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
except OSError:alive=False
print('H6_JSON '+json.dumps(dict(alive=alive,done=(r/{marker!r}).exists())))
''')


def collect(owner, remote, task, out):
    packed = owner.status(f'''import pathlib,tarfile,json
r=pathlib.Path({remote!r})
with tarfile.open(r/'collected.tar','w') as t:
 for folder in ('results/{task}',):
  p=r/folder
  if p.exists():t.add(p,arcname=folder)
 for name in ({task+'.log'!r},'compile.json'):
  p=r/name
  if p.exists():t.add(p,arcname=name)
print('H6_JSON '+json.dumps(dict(bytes=(r/'collected.tar').stat().st_size)))
''')
    reserve(out, 2*packed['bytes'])
    owner.call('download', remote+'/collected.tar', out/'collected.tar')
    with tarfile.open(out/'collected.tar') as t: t.extractall(out, filter='data')
    if (out/'compile.json').exists(): shutil.copy2(out/'compile.json', out/(task+'_compile.json'))


def run(args):
    out = writable(args.output); record = prepared(out)
    if (out/'STARTED.json').exists(): raise ValueError('one T4 session only; STARTED already exists')
    archive = payload(out, record); owner = Owner('g5-a4u-h6-'+uuid.uuid4().hex[:12], out)
    remote = '/content/'+owner.session
    write(out/'STARTED.json', dict(session=owner.session, remote=remote))
    result = dict(status='running', session=owner.session, tasks={}); begin = time.perf_counter()
    active = None
    try:
        owner.call('new', '--gpu', 'T4'); upload(owner, archive, remote)
        for task in TASKS:
            reserve(out, 64*1024**2); active = task; start = time.perf_counter(); launch(owner, remote, task)
            while True:
                state = status(owner, remote, task, 'results/'+task+'/DONE')
                collect(owner, remote, task, out)
                if state['done']: break
                if not state['alive']: raise RuntimeError(task+' worker failed; collected its log')
                time.sleep(H6_POLL_SECONDS)
            row = json.loads((out/'results'/task/'verdict.json').read_text())
            row['elapsed_including_communication_s'] = time.perf_counter()-start
            result['tasks'][task] = row; write(out/'run.json', result)
            print(json.dumps(dict(task=task, elapsed_s=row['elapsed_including_communication_s'],
                                  compilation_log=str(out/(task+'_compile.json')))), flush=True)
            if row['status'] not in ('pass', 'report-only'): raise RuntimeError(task+' failed; stop')
            if task in ('A1', 'A2'):
                # Judge each pair immediately, so a failed A gate stops this VM.
                checked = judge_pairs(out, task)
                if checked['status'] != 'pass': raise RuntimeError(task+' CPU/T4 comparison failed')
        prepared(out)
        result['status'] = 'complete'  # Only local judge can declare H6 pass.
    except Exception as exc:
        result.update(status='fail', error=str(exc))
        if active is not None:
            try: collect(owner, remote, active, out)
            except Exception as fetch: result['collection_error'] = str(fetch)
        raise
    finally:
        try: owner.call('stop'); result['vm_stopped'] = True
        except Exception as exc: result.update(vm_stopped=False, stop_error=str(exc)); raise
        finally:
            result['total_s'] = time.perf_counter()-begin; write(out/'run.json', result)


def judge_pairs(out, task):
    import numpy as np
    from chatcarlo.gpu import validation
    if task == 'A1':
        rows = {}
        for sid in ('s1', 's2', 's3'):
            a = json.loads((out/'inputs/A1'/f'{sid}_cpu.json').read_text())
            b = json.loads((out/'results/A1'/f'{sid}_cuda.json').read_text())
            if a['keys'] != b['keys']: raise ValueError('paired quantity keys mismatch')
            bounds = (np.asarray(a['additions'])+np.asarray(b['additions']))*2.**-52*(np.abs(a['values'])+np.abs(b['values']))
            rows[sid] = validation.paired_judge(a['values'], b['values'], bounds)
            rows[sid]['keys'] = a['keys']
        # Keep both-zero quantities too, just as the existing G1 gate does.
    else:
        from a1_common import module
        grid_judge = module('h6_g2_judge', HERE.parent/'g2/run_gates.py').grid_paired_judge
        rows = {}
        for name in ('water', 'lead'):
            with np.load(out/'inputs/A2'/f'{name}_cpu.npz') as a, np.load(out/'results/A2'/f'{name}_cuda.npz') as b:
                rows[name] = {q: grid_judge(a[q], b[q], a['grid_additions'], b['grid_additions']) for q in ('kerma', 'h10')}
        rows = {name+':'+q: r for name, values in rows.items() for q, r in values.items()}
    result = dict(status='pass' if all(v['status'] == 'pass' for v in rows.values()) else 'fail', comparisons=rows)
    write(out/('judged_'+task+'.json'), result); return result


def judge(args):
    import a1_g2gate3
    out = writable(args.output); prepared(out); gates = {}
    run_record = json.loads((out/'run.json').read_text())
    for task in ('A1', 'A2'):
        try: gates[task] = judge_pairs(out, task)
        except (OSError, ValueError) as exc: gates[task] = dict(status='not-run', error=str(exc))
    for task in ('A3', 'B', 'C2'):
        p = out/'results'/task/'verdict.json'
        gates[task] = json.loads(p.read_text()) if p.exists() else dict(status='not-run')
    inventory = dict(gates['B'].get('B6_production', {}))
    inventory.update({k+'_legacy': v for k, v in gates['C2'].get('B6_flags', {}).items()})
    gates['B6'] = dict(status='pass' if set(inventory) == {'local', 'electron', 'seed', 'local_legacy', 'electron_legacy'} and all(v.get('backend') == 'nvrtc' and v.get('status') == 'pass' for v in inventory.values()) else 'not-run', modules=inventory)
    c1 = out/'results/C1'
    report = dict(status='not-run')
    if all((c1/(name+'.json')).exists() for name in a1_g2gate3.CONDITIONS):
        # compare() verifies sealed EGS5 hashes; bundles are the new prepared ones.
        for name in a1_g2gate3.CONDITIONS:
            # compare expects bundle directories alongside returned JSON files.
            if not (c1/name).exists(): shutil.copytree(out/'inputs/C1'/name, c1/name)
        a1_g2gate3.compare(c1); report = dict(status='report-only', directory=str(c1))
    complete = (run_record.get('status') == 'complete' and run_record.get('vm_stopped') is True and
                all(g.get('status') == 'pass' for g in gates.values()))
    result = dict(status='pass' if complete else 'fail', gates=gates, C1=report,
                  C4=dict(total_s=run_record.get('total_s'),
                          compilation={task: json.loads((out/(task+'_compile.json')).read_text()) for task in TASKS if (out/(task+'_compile.json')).exists()},
                          worker_timings={task: row.get('worker_total_s') for task, row in run_record.get('tasks', {}).items()},
                          communication=json.loads((out/'transport_timings.json').read_text())),
                  interpretation='事前に指定した分岐・構成で CPU と T4 の一致と健全性を確認した。' if complete else 'H6 は未合格。fail または未実施の関門がある。')
    write(out/'judgement.json', result)
    print(json.dumps(dict(status=result['status'], output=str(out/'judgement.json'))))
    return result


def diagnostic_measure(c, state, n):
    import numpy as np
    from chatcarlo.gpu import validation
    inputs = state['sample'](state['source'], n, np.random.default_rng(c['source_seed']), state['spectrum'])
    state['timer'].seconds.clear(); begin = time.perf_counter()
    r = state['obj'].run(*inputs, seed=c['seed'], start=c['start'], **c['flags'])
    call_s = time.perf_counter()-begin
    if isinstance(r, tuple): r = r[0]
    if len(state['timer'].seconds) != 1: raise ValueError('one timed kernel required')
    kernel_s = state['timer'].seconds[0]
    replay = state['obj'].run(*inputs, seed=c['seed'], start=c['start'], **dict(c['flags'], record=True, observe=False))
    rebuilt = validation.record_reduce(replay, state['obj'].a)
    # record_reduce only produces images/category_count for a scene WITH a detector,
    # while reconstruction_bounds reads r['category_count'] unconditionally to size the
    # image bound. The g2b speed configuration has a grid and no detector, so C2 died
    # there on T4 session 5 with KeyError: 'category_count'. The image bound is used
    # only for the keys rebuilt actually has, so a placeholder cannot change any bound
    # that is returned -- asserted below rather than assumed.
    if 'category_count' in replay:
        bounds = validation.reconstruction_bounds(replay, rebuilt, n)
    else:
        if 'images' in rebuilt or 'category_count' in rebuilt:
            raise ValueError('detector-less run produced image records; the placeholder would matter')
        bounds = validation.reconstruction_bounds(dict(replay, category_count=np.zeros((4, 1, 1))), rebuilt, n)
    if any(np.any(np.abs(rebuilt[k]-replay[k]) > bounds[k]) for k in rebuilt): raise ValueError('timing reconstruction')
    if replay['completed'] != n: raise ValueError('timing history count')
    return dict(kernel_s=kernel_s, call_s=call_s, health=dict(status='pass', histories=n))


def speed(base, out, cp):
    """C2/C3 fixed A1 task orders, and A0 p7A on this same T4."""
    import numpy as np
    import a0_speed, a1_speed, a2_gates
    from chatcarlo.gpu import core, g2_core, bundle
    from chatcarlo.gpu.therapy_cuda import T4
    from a4u_h6_fixtures import load_inputs
    cfg, digest = a1_speed.load_configs(HERE/'a1_speed_configs.json')
    if len(cfg['round_orders']) != 3: raise ValueError('three registered speed orders required')
    # Frozen old loader is installed under a separate package, never over new code.
    old = base/'diagnostic/old/g2b'
    target = out/'old_runtime/chatcarlo_g5_old'; target.mkdir(parents=True)
    for p in old.iterdir():
        if p.suffix in ('.py', '.h', '.cu', '.cpp'): shutil.copy2(p, target/p.name)
    sys.path.insert(0, str(target.parent))
    states = {}; resources = {}; compile_rows = []
    for c in cfg['configurations']:
        for version in ('old', 'new', 'production'):
            begin = time.perf_counter()
            if version == 'old': state = a1_speed.construct(c, 'old', base/'diagnostic')
            else:
                a, src, spec, settings, _ = bundle.load(base/'diagnostic/new'/c['id'])
                cls = g2_core.Core if c['grid'] else core.Core
                kw = dict(grid=g2_core.grid_from(c['grid'])) if c['grid'] else {}
                obj = cls(a, 'cuda', **kw, _test_legacy_physics=(version == 'new'))
                kernel = obj.native if c['grid'] else obj.fn
                timer = a0_speed.EventDelegate(kernel, cp)
                if c['grid']: obj.native = timer
                else: obj.fn = timer
                from chatcarlo.gpu.source_host import sample_source_photons
                state = dict(obj=obj, kernel=kernel, timer=timer, source=src, spectrum=spec, sample=sample_source_photons)
            states[version+'-'+c['id']] = state
            resources[version+'-'+c['id']] = a0_speed.resources(cp, state['kernel'], c)
            compile_rows.append(dict(module=version+'-'+c['id'], initialization_s=time.perf_counter()-begin, backend='nvrtc'))
    rows = []; production = []
    configs = {c['id']: c for c in cfg['configurations']}
    for repeat, order in [(-1, cfg['warmup_order']), *enumerate(cfg['round_orders'])]:
        for task in order:
            key, n = task.split(':'); cid = key.split('-', 1)[1]; state = states[key]; c = configs[cid]
            # Same existing measurement for old/flag builds; its judge is unchanged.
            if key.startswith('old-'):
                row = a1_speed.measure(c, state, int(n))
            else:
                row = diagnostic_measure(c, state, int(n))
            rows.append(dict(row, task=task, repeat=repeat, config_sha256=digest))
            if key.startswith('new-'):
                prod = states['production-'+cid]
                production.append(dict(diagnostic_measure(c, prod, int(n)), task=task, repeat=repeat))
    diagnostic = a1_speed.summarize(rows, cfg, digest)
    # Therapy: same A0 p7A source, field, seed, launch and production_n=200000.
    c = next(c for c in json.loads((HERE/'a0_configs.json').read_text())['configurations'] if c['id'] == 'p7A')
    old_state = a0_speed.construct(cp, c, base/'bundles')
    materials, _, _ = load_inputs(base/'fixtures'); water = a2_gates.load_water('A')
    therapy_states = {v: T4(cp, materials, water['arrays'], water['fg'], water['half'], water['ssd'], _test_legacy_physics=(v == 'new')) for v in ('new', 'production')}
    for v, driver in therapy_states.items(): resources['therapy-'+v] = a0_speed.resources(cp, driver.kernel, c)
    resources['therapy-old'] = a0_speed.resources(cp, old_state['kernel'], c)
    therapy_rows = []
    for repeat in (-1, 0, 1, 2):
        for v in (('old', 'new', 'production') if repeat % 2 else ('new', 'old', 'production')):
            if v == 'old': row = a0_speed.measure(c, old_state, c['production_n'])
            else:
                r = therapy_states[v].run(c['production_n'], c['seed'])
                health = a2_gates.batch_health(r, water['arrays'], water['fg'])
                row = dict(kernel_s=r['kernel_s'], health=health)
            therapy_rows.append(dict(row, version=v, repeat=repeat))
    med = lambda v: statistics.median(r['kernel_s'] for r in therapy_rows if r['version'] == v and r['repeat'] >= 0)
    therapy_ratio = ratio(med('old'), med('new'))
    changes = {}
    for task in ('g2b:500000', 'g2b:100000', 'g4:100000'):
        before = statistics.median(r['kernel_s'] for r in rows if r['task'] == 'old-'+task and r['repeat'] >= 0)
        after = statistics.median(r['kernel_s'] for r in production if r['task'] == 'new-'+task and r['repeat'] >= 0)
        changes[task] = after/before
    changes['p7A'] = med('production')/med('old')
    result = dict(status='pass' if diagnostic['status'] == therapy_ratio['status'] == 'pass' and max(changes.values()) <= 1.3 else 'fail',
                  C2=dict(diagnostic=diagnostic, therapy=therapy_ratio), C3=dict(status='report-only', ratios=changes, stop_for_consultation=max(changes.values()) > 1.3),
                  rows=rows, production_rows=production, therapy_rows=therapy_rows, resources=resources, module_initialization=compile_rows,
                  B6_flags=dict(local=dict(backend='nvrtc', status='pass'), electron=dict(backend='nvrtc', status='pass')))
    write(out/'speed.json', result); return result


def worker(task, root):
    import numpy as np
    import cupy as cp
    import a2_gates, a4u_h6_fixtures as fixtures
    from chatcarlo.gpu import bundle, core, validation
    base = root/'inputs'; out = root/'results'/task; out.mkdir(parents=True, exist_ok=False)
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    if 'T4' not in name or (props['major'], props['minor']) != (7, 5): raise ValueError('T4 sm_75 required')
    begin = time.perf_counter()
    a2_gates.WATER = base
    try:
        if task == 'A1':
            for sid in ('s1', 's2', 's3'): write(out/f'{sid}_cuda.json', bundle_quantities(base/'A1'/sid, sid, 'cuda'))
            result = dict(status='pass')
        elif task == 'A2':
            grids = json.loads((base/'A2/grids.json').read_text())
            for name, grid in grids.items(): voxel_batches(base/'A2'/name, grid, 'cuda', out/f'{name}_cuda.npz')
            result = dict(status='pass')
        elif task == 'A3':
            # Same new bundle on both devices; existing integer/paired/reduction gates.
            from unittest.mock import patch
            path = base/'diagnostic/new/g4'
            # gate2a's one xraylib-dependent call is served by prepare's packing.
            stand_in = dict(np.load(base/'A3_artificial.npz', allow_pickle=False))
            expected_changed = int(stand_in.pop('changed_elements'))
            bundle_data = np.asarray(bundle.load(path)[0]['data'])
            if int(np.count_nonzero(stand_in['data'] != bundle_data)) != expected_changed:
                raise RuntimeError('A3 stand-in packing does not match what prepare recorded')
            calls = []

            def pack_stand_in(tables):
                # Only gate2a's artificial table may take this path, and only once.
                if not (float(tables.k_edge[0, 0]) == 1. and int(tables.n_lines[0, 0]) == 1
                        and float(tables.k_line_e[0, 0, 0]) == 4. and float(tables.k_line_p[0, 0, 0]) == 1.):
                    raise RuntimeError('unexpected core.pack call on the VM; no xraylib here')
                calls.append(1)
                return {k: v for k, v in stand_in.items()}

            with patch.object(core, 'pack', pack_stand_in):
                result = validation.gate2(path, 's3', out, dict(gate='a4u-h6', seed=2040))
            if len(calls) != 1:
                raise RuntimeError(f'the A3 stand-in was used {len(calls)} times, expected once')
            result = dict(result, artificial_stand_in=dict(
                changed_elements=expected_changed, calls=len(calls), source='prepare on the host'))
            if result['gate2a']['status'] != 'pass': result['status'] = 'fail'
            # Registered G4 seed 2040 (gate2 also retains its own probe seeds).
            from chatcarlo.gpu.source_host import sample_source_photons
            a, src, spec, settings, _ = bundle.load(path)
            inputs = sample_source_photons(src, 100000, np.random.default_rng(3712336832425968660), spec)
            images = []
            for device in ('cpu', 'cuda'):
                r = core.Core(a, device).run(*inputs, seed=2040, record=True, fluorescence=settings['fluorescence'])
                images.append(r['images']); np.savez(out/(device+'_2040.npz'), **r)
            # G4's registered seed has its own ten paired batches and the same judge.
            paired = {'cpu': [], 'cuda': [], 'bounds': []}
            rng = np.random.default_rng(3712336832425968660)
            drivers = [core.Core(a, d) for d in ('cpu', 'cuda')]
            for b in range(10):
                inputs = sample_source_photons(src, 10000, rng, spec); qs = []; adds = []
                for driver in drivers:
                    r = driver.run(*inputs, seed=2040, start=b*10000, fluorescence=settings['fluorescence'])
                    qs.append(validation.quantities(r, settings['material_names'], 's3', 10000))
                    adds.append(validation.quantity_additions(r, settings['material_names'], 's3', 10000))
                paired['cpu'].append(list(qs[0].values())); paired['cuda'].append(list(qs[1].values()))
                paired['bounds'].append([(adds[0][k]+adds[1][k])*2.**-52*(abs(qs[0][k])+abs(qs[1][k])) for k in qs[0]])
            result['seed2040'] = validation.paired_judge(paired['cpu'], paired['cuda'], paired['bounds'])
            if result['seed2040']['status'] != 'pass': result['status'] = 'fail'
            # Image floats use the existing paired judge.
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.imsave(out/'g4_T4.png', images[1][0].T, origin='lower', cmap='gray')
        elif task == 'B':
            result = fixtures.run(base/'fixtures', 'cuda', progress=lambda r: write(out/'branches.json', r))
            result['expected_cpu'] = json.loads((base/'fixtures/expected.json').read_text())
            for key in ('seeds', 'inputs', 'fixture_source_sha256'):
                if result[key] != result['expected_cpu'][key]: raise ValueError('CPU/T4 fixture identity mismatch: '+key)
            # Explicit production module inventory; observers/fault builds do not substitute.
            a, *_ = bundle.load(base/'A1/s1'); core.Core(a, 'cuda')
            materials, arrays, _ = fixtures.load_inputs(base/'fixtures')
            from chatcarlo.gpu.therapy_cuda import T4
            T4(cp, materials, arrays)  # exact production options, without check/observer flags
            result['B6_production'] = {n: dict(status='pass', backend='nvrtc') for n in ('local', 'electron', 'seed')}
        elif task == 'C1':
            import a1_g2gate3
            for name in a1_g2gate3.CONDITIONS: a1_g2gate3.legacy.execute(base/'C1'/name, out)
            result = dict(status='report-only')
        elif task == 'C2': result = speed(base, out, cp)
        else: raise ValueError(task)
        result['worker_total_s'] = time.perf_counter()-begin
        write(out/'verdict.json', result)
        if result['status'] not in ('pass', 'report-only'): raise RuntimeError(task+' failed')
        (out/'DONE').write_text('done\n')
    except Exception as exc:
        write(out/'failure.json', dict(status='fail', error=str(exc))); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare', 'run', 'judge', '_worker'))
    p.add_argument('--output', type=Path, required=True)
    for name in ('therapy-bundle', 'p7', 'p8g2', 'old-diagnostic'): p.add_argument('--'+name, type=Path)
    p.add_argument('--task', choices=TASKS, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.command == '_worker': worker(args.task, args.output); return
    out = writable(args.output)
    sys.dont_write_bytecode = True; os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['MPLCONFIGDIR'] = str(out/'library_cache/mpl'); os.environ['NUMBA_CACHE_DIR'] = str(out/'library_cache/numba')
    sys.addaudithook(no_cache_writes)
    if args.command == 'prepare': prepare(args)
    elif args.command == 'run': run(args)
    else: raise SystemExit(0 if judge(args)['status'] == 'pass' else 2)


if __name__ == '__main__': main()
