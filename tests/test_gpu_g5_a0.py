"""A0 orchestration contracts: no CUDA or external/cache data required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / 'docs/validation/gpu/g5'


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


speed = module('a0_speed')
bootstrap = module('a0_bootstrap')
launcher = module('a0_launch')


def rows(cfg, digest, session='one'):
    return [dict(task=task, phase='warmup' if repeat == -1 else 'valid', repeat=repeat,
                 config_sha256=digest, session_id=session, health=dict(status='pass'), kernel_s=1., call_s=2.)
            for repeat, order in [(-1, cfg['warmup_order']), *enumerate(cfg['round_orders'])] for task in order]


def test_frozen_production_configs():
    cfg, digest = speed.load_configs(HERE / 'a0_configs.json')
    assert len(digest) == 64
    cs = cfg['configurations']
    assert [c['production_n'] for c in cs] == [500000, 100000, 200000, 1380000, 260000, 200000]
    from prototypes.mv_electron.gpu import p7_core as p7
    job = json.loads((HERE.parent / 'water60_100M/additional_job.json').read_text())
    for key in ('seed', 'source_seed', 'grid'):
        assert cs[0][key] == job[key]
    for c in cs[2:5]:
        production = p7.contract('production', c['condition'], c['production_n'] * 100)
        assert production['seed'] == c['seed']
        assert c['launch'] == dict(blocks=40, threads=256, capacity=64)
    from prototypes.mv_electron.gpu import p8g3_core
    assert cs[-1]['seed'] == p8g3_core.SEED
    assert cs[-1]['launch'] == dict(blocks=40, threads=256, capacity=18, check=False, fault=0)
    assert len(cfg['warmup_order']) == 11  # G4's 100k measurement serves both roles.


def test_delegate_preserves_exact_arguments_and_return():
    received = []
    marker = object()
    class Event:
        def record(self): pass
        def synchronize(self): pass
    cp = SimpleNamespace(cuda=SimpleNamespace(Event=Event, get_elapsed_time=lambda a, b: 123.))
    def native(*args, **kwargs):
        received.append((args, kwargs))
        return marker
    fn = speed.EventDelegate(native, cp)
    pointer = object()
    args = ((40,), (256,), (pointer, np.uint64(820101)))
    assert fn(*args, stream=pointer) is marker
    assert all(a is b for a, b in zip(received[0][0], args))
    assert received[0][1]['stream'] is pointer
    assert fn.seconds == [.123]


@pytest.mark.parametrize('operation', launcher.TIMEOUTS)
def test_every_colab_call_has_timeout(monkeypatch, tmp_path, operation):
    observed = []
    def fake(command, **kw):
        observed.append((command, kw))
        return SimpleNamespace(stdout='ok')
    monkeypatch.setattr(launcher.subprocess, 'run', fake)
    assert launcher.call([operation, '-s', 'test'], tmp_path / 'heartbeat.json') == 'ok'
    assert observed[0][1]['timeout'] == launcher.TIMEOUTS[operation] > 0
    assert json.loads((tmp_path / 'heartbeat.json').read_text())['operation'] == operation
    with pytest.raises(ValueError, match='unbounded'):
        launcher.call(['unknown'], tmp_path / 'heartbeat.json')


def test_timeout_propagates(monkeypatch, tmp_path):
    def hung(*a, **kw): raise subprocess.TimeoutExpired(a[0], kw['timeout'])
    monkeypatch.setattr(launcher.subprocess, 'run', hung)
    with pytest.raises(subprocess.TimeoutExpired):
        launcher.call(['exec'], tmp_path / 'heartbeat.json')


def test_bundle_manifest_all_members_and_zip_containment(tmp_path):
    p = tmp_path / 'input'
    p.write_bytes(b'input')
    manifest = dict(files={'input': dict(sha256=speed.sha(p), bytes=5)})
    bootstrap.verify_files(tmp_path, manifest)
    p.write_bytes(b'other')
    with pytest.raises(ValueError, match='mismatch'):
        bootstrap.verify_files(tmp_path, manifest)
    for name in ('../outside', '/outside'):
        with pytest.raises(ValueError):
            bootstrap.verify_files(tmp_path, dict(files={name: '0' * 64}))
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escaped.py', 'bad')
    with pytest.raises(ValueError, match='unsafe'):
        bootstrap.extract_sources(archive, tmp_path / 'target')


def test_existing_split_assemble_integrity(tmp_path):
    from chatcarlo.gpu.colab_driver import split_bundle, assemble
    source, parts, target = [tmp_path / x for x in ('source', 'parts', 'target')]
    source.mkdir()
    (source / 'large').write_bytes(b'123456789')
    files = split_bundle(source, parts, part_bytes=3)
    assemble(parts, target, files)
    assert (target / 'large').read_bytes() == b'123456789'
    next(parts.iterdir()).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA256'):
        assemble(parts, target, files)


def test_config_bytes_and_result_loading(tmp_path):
    cfg, digest = speed.load_configs(HERE / 'a0_configs.json')
    measurements = rows(cfg, digest)
    value = dict(status='complete', config_sha256=digest, session_id='one', measurements=measurements,
                 summary=speed.summarize(measurements, cfg, digest, 'one'))
    path = tmp_path / 'measurements.json'
    speed.write_json(path, value)
    assert speed.read_results(path, HERE / 'a0_configs.json') == value
    edited = tmp_path / 'a0_configs.json'
    edited.write_bytes((HERE / 'a0_configs.json').read_bytes() + b'\n')
    with pytest.raises(ValueError, match='mixed'):
        speed.read_results(path, edited)
    value['summary']['tolerance_ratio'] = 1.5
    speed.write_json(path, value)
    with pytest.raises(ValueError, match='summary'):
        speed.read_results(path, HERE / 'a0_configs.json')


@pytest.mark.parametrize('fault', ['session', 'config', 'health', 'missing', 'duplicate'])
def test_no_partial_or_mixed_baselines(fault):
    cfg, digest = speed.load_configs(HERE / 'a0_configs.json')
    data = rows(cfg, digest)
    if fault == 'session': data[0]['session_id'] = 'two'
    elif fault == 'config': data[0]['config_sha256'] = 'wrong'
    elif fault == 'health': data[0]['health']['status'] = 'fail'
    elif fault == 'missing': data.pop()
    else: data.append(data[0])
    with pytest.raises(ValueError):
        speed.summarize(data, cfg, digest, 'one')


@pytest.mark.parametrize('spread,ratio', [(0, 1.05), (.025, 1.05), (.04, 1.08), (.05, 1.10), (.06, None)])
def test_preregistered_tolerance(spread, ratio):
    cfg, digest = speed.load_configs(HERE / 'a0_configs.json')
    data = rows(cfg, digest)
    for row in data:
        if row['task'] == cfg['warmup_order'][0] and row['phase'] == 'valid':
            row['kernel_s'] = [1 - spread / 2, 1, 1 + spread / 2][row['repeat']]
    actual = speed.summarize(data, cfg, digest, 'one')['tolerance_ratio']
    assert actual == pytest.approx(ratio) if ratio else actual is None


def test_turing_register_allocation():
    props = dict(maxThreadsPerMultiProcessor=1024, regsPerMultiprocessor=65536, sharedMemPerMultiprocessor=65536)
    attrs = dict(num_regs=234, shared_size_bytes=0, max_threads_per_block=256)
    assert speed.resident_blocks(attrs, 256, props)['blocks_per_sm'] == 1
    assert speed.resident_blocks(attrs, 128, props)['blocks_per_sm'] == 2
    # 130 rounds up to 136 registers/thread, leaving only one 256-thread block.
    attrs['num_regs'] = 130
    assert speed.resident_blocks(attrs, 256, props)['registers_per_block'] == 34816


def test_diagnostic_energy_fault_detected():
    measured = dict(deposits=np.array([3.]), endings=np.array([1, 0, 0]))
    replay = dict(measured, d=np.array([[3., 3.]]), i=np.zeros((1, 9), dtype=int))
    assert speed.diagnostic_health(measured, replay, np.array([3.]))['status'] == 'pass'
    with pytest.raises(ValueError, match='energy balance'):
        speed.diagnostic_health(measured, replay, np.array([4.]))


def test_bootstrap_refuses_restarting_old_measurements(tmp_path, monkeypatch):
    job = tmp_path / 'job.json'
    speed.write_json(job, dict(control_files={}))
    (tmp_path / 'results').mkdir()
    (tmp_path / 'results/measurements.json').write_text('{}')
    with pytest.raises(ValueError, match='fresh base'):
        bootstrap.launch(tmp_path, speed.sha(job))


def test_diagnostic_balance_with_frozen_cpu_core(tmp_path):
    from chatcarlo.scene import load_scene
    from chatcarlo.gpu import bundle, core
    from chatcarlo.gpu.source_host import sample_source_photons
    bundle.create(load_scene(str(HERE.parent / 'g1/scenes/s1.yaml')), None, tmp_path / 'bundle')
    arrays, source, spectrum, settings, _ = bundle.load(tmp_path / 'bundle')
    engine = core.Core(arrays, build_dir=tmp_path / 'build')
    inputs = sample_source_photons(source, 100, np.random.default_rng(7), spectrum)
    result = engine.run(*inputs, seed=42)
    replay = engine.run(*inputs, seed=42, record=True)
    assert np.any(replay['i'][:, 3] == 0)
    assert speed.diagnostic_health(result, replay, inputs[2])['status'] == 'pass'


def test_fingerprints_separate_drivers_from_physics(tmp_path, monkeypatch):
    from chatcarlo.gpu import bundle
    from prototypes.mv_electron.gpu import p7_colab, p8g2_colab
    monkeypatch.setattr(bundle, 'verify', lambda path: {})
    monkeypatch.setattr(p7_colab, 'load_bundle', lambda path: ({}, None, {}))
    monkeypatch.setattr(p8g2_colab, 'load', lambda path: ({}, None, None))
    for name in ('g2b', 'g4', 'p7', 'p8g2'):
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'driver.py').write_text('old driver')
        (folder / 'physics.h').write_text('frozen physics')
        (folder / 'tables.npz').write_bytes(b'frozen input')
    before = speed.fingerprints(tmp_path)
    (tmp_path / 'g4/driver.py').write_text('new measurement driver')
    after = speed.fingerprints(tmp_path)
    assert before['configuration'] == after['configuration']
    assert before['transfer'] != after['transfer']
    (tmp_path / 'g4/tables.npz').write_bytes(b'changed input')
    assert speed.fingerprints(tmp_path)['configuration'] != after['configuration']


def test_host_orchestration_generates_valid_remote_scripts_and_stops(tmp_path, monkeypatch):
    exports = {}
    for name, filename in [('p7', 'p7_manifest.json'), ('p8g2', 'p8g2_manifest.json')]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'input.npz').write_bytes(b'sealed')
        speed.write_json(folder / filename, dict(files={'input.npz': speed.sha(folder / 'input.npz')}))
        exports[name] = folder
    def diagnostic(destination, configs):
        for name in ('g2b', 'g4'):
            (destination / name).mkdir()
            (destination / name / 'manifest.json').write_text('{}')
    class Watch:
        terminated = False
        def terminate(self): self.terminated = True
        def wait(self, **kw): return 0
    watch = Watch()
    calls = []
    def colab(args, heartbeat, *, input=None, timeout=None):
        calls.append((args, input))
        if input:
            compile(input, '<remote bootstrap/status>', 'exec')
            if 'alive=alive' in input:
                return json.dumps(dict(done=True, alive=False, files=[]))
        return ''
    monkeypatch.setattr(launcher, 'create_diagnostics', diagnostic)
    monkeypatch.setattr(launcher, 'call', colab)
    monkeypatch.setattr(launcher.subprocess, 'Popen', lambda *a, **kw: watch)
    monkeypatch.setattr(launcher, 'read_results', lambda *a: {})
    args = SimpleNamespace(configs=HERE / 'a0_configs.json', session='mock', output=tmp_path / 'out', **exports)
    launcher.run(args)
    assert calls[0][0] == ['new', '-s', 'mock', '--gpu', 'T4']
    assert calls[-1][0] == ['stop', '-s', 'mock']
    assert any(code and 'assembled SHA256' in code for _, code in calls)
    assert watch.terminated
    # A local preflight failure cannot stop an unrelated existing session.
    (exports['p7'] / 'input.npz').write_bytes(b'bad')
    calls.clear()
    with pytest.raises(ValueError, match='mismatch'):
        launcher.run(args)
    assert not calls
