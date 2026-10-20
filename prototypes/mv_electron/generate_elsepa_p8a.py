"""P8a isolated elemental ELSEPA cases and composition-aware GS construction.

H/O are reused read-only from P1f with hashes. New elements have two independent
passes plus direct midpoints and probes. Positron tables are checked, not used.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from .materials_p8a import (ROOT, OUT, CACHE, MATERIALS, MixtureElsepa, sha,
                            write, resume, resume_appendix6, reference, load_reference, load_manifest, verify_protected, implementation_hashes)
from .generate_elsepa_p1f import energies, input_text, parse, prepare

TIMEOUT_SECONDS = 600
REPLACEMENT_FACTOR = 1+1e-3
RECORDED_OMISSION = dict(Z=17, charge=1, original_energy_MeV=float.fromhex('0x1.01de978eb7400p+3'),
                        replacement_energy_MeV=float.fromhex('0x1.01de978eb7400p+3')*REPLACEMENT_FACTOR)
# Exact energies from the sealed S4 coverage (also recorded in the repository's
# elsepa_direct_status.json). This is a two-point allowlist, never a Z-only rule.
RECORDED_OMISSIONS = [RECORDED_OMISSION,
    dict(Z=20, charge=1, original_energy_MeV=float.fromhex('0x1.29c8762af4cf5p+3'),
         replacement_energy_MeV=float.fromhex('0x1.29c8762af4cf5p+3')*REPLACEMENT_FACTOR)]
OMISSION_STATUS = 'pass-with-recorded-omission'
# Appendix 4 explicitly authorizes reuse across this wrapper/checker change.
# Every other implementation, scientific setting and executable must match.
PRE_TIMEOUT_IMPLEMENTATIONS = {
    'generate_elsepa_p8a.py': '3c9ee92ba9efe565f693328167a85991ac4b87640f41a004cface55272fba7fd',
    'check_p8a.py': 'f006065f77649b22df2c68a0916dbb13d8f9d5c4abe45b611e5d5423450baabb',
}
INTERRUPTED_DIRECT = (17, 1, '0x1.01de978eb7400p+3')


class ElsepaCaseTimeout(TimeoutError):
    def __init__(self, record):
        self.record = record
        super().__init__(f'ELSEPA timeout: Z={record["Z"]}, charge={record["charge"]}, '
                         f'E={record["energy_MeV"]} MeV')


def resume_s4(path, identity):
    """Keep the original generation identity and seal the authorized migration."""
    path = Path(path)
    if not path.exists():
        resume(path, identity)
        return
    old = json.loads(path.read_text())
    if path.with_suffix('.resume_v6.json').exists():
        return resume_appendix6(path, identity, 'S4')
    if old == identity:
        return
    baseline = dict(identity)
    baseline['implementations'] = dict(identity['implementations'])
    for name, previous in PRE_TIMEOUT_IMPLEMENTATIONS.items():
        baseline['implementations'][name] = previous
    if old != baseline:
        return resume_appendix6(path, identity, 'S4')
    resume(path.with_suffix('.resume_v4.json'), dict(
        original_sha256=sha(path), current_identity=identity,
        authorization='P8a appendix 4: timeout wrapper and incomplete-check reporting only',
        timeout_seconds=TIMEOUT_SECONDS))


def _read_timeout(work, identity):
    """Verify both the timeout marker and every retained raw file on reuse."""
    marker = work/'timeout.json'
    archive = work.with_name(work.name+'.timed-out')
    # Recover interruption between the atomic directory move and marker write.
    if not marker.exists() and (archive/'timeout_record.json').exists():
        record = json.loads((archive/'timeout_record.json').read_text())
    elif marker.exists():
        record = json.loads(marker.read_text())
    else:
        return None
    if (record['identity'] != identity or record['archive'] != archive.name
            or record['status'] != 'not-evaluated' or record['timeout_seconds'] != TIMEOUT_SECONDS):
        raise ValueError('ELSEPA timeout identity/policy mismatch')
    if json.loads((archive/'timeout_record.json').read_text()) != record:
        raise ValueError('ELSEPA timeout archive marker changed')
    actual = {p.name for p in archive.iterdir() if p.is_file() and p.name != 'timeout_record.json'}
    if actual != set(record['files']) or any(sha(archive/name) != value for name, value in record['files'].items()):
        raise ValueError('ELSEPA timeout archived output changed')
    if not marker.exists():
        write(marker, record)
    return record


def _retain_timeout(work, identity, origin):
    archive = work.with_name(work.name+'.timed-out')
    if archive.exists():
        raise ValueError('ELSEPA timeout archive already exists; no overwrite')
    record = dict(status='not-evaluated', reason='ELSEPA case timed out',
                  Z=identity['Z'], charge=identity['charge'], energy_MeV=identity['energy_MeV'],
                  timeout_seconds=TIMEOUT_SECONDS, origin=origin,
                  identity=identity, archive=archive.name,
                  files={p.name: sha(p) for p in work.iterdir() if p.is_file()})
    write(work/'timeout_record.json', record)
    work.rename(archive)  # Preserve input and partial/zero-byte outputs, never delete.
    write(work/'identity.json', identity)
    write(work/'timeout.json', record)
    return record


def save_verified_table(path, values):
    """Reuse bytes; legacy tables are verified against hash-checked raw replay.

    Call only after reconstructing both matching node passes (or the completed
    direct cases). Old S4 had no npz receipt: array equality supplies the initial
    verification; subsequent resumes also require the unchanged file SHA256.
    """
    path = Path(path)
    seal = path.with_suffix('.sha.json')
    if path.exists():
        before = sha(path)
        if seal.exists() and json.loads(seal.read_text())['sha256'] != before:
            raise ValueError('ELSEPA saved table hash mismatch')
        with np.load(path, allow_pickle=False) as f:
            saved = {k: f[k].copy() for k in f.files}
        if not arrays_identical(saved, values) or sha(path) != before:
            raise ValueError('ELSEPA saved table differs from verified raw cases')
    else:
        if seal.exists():
            raise ValueError('ELSEPA sealed table missing')
        np.savez(path, **values)
    if not seal.exists():
        write(seal, dict(sha256=sha(path)))


def case(source, exe, work, z, charge, T, *, kind='node'):
    if kind not in ('node', 'direct'):
        raise ValueError('unknown ELSEPA case kind')
    work = Path(work)
    text = input_text(z, charge, T, ihef=0)
    identity = dict(Z=z, charge=charge, energy_MeV=T, IHEF=0,
                    input=text, executable_sha256=sha(exe))
    resume(work/'identity.json', identity)
    seal = work/'complete.json'
    if seal.exists():
        receipt = json.loads(seal.read_text())
        for name, expected in receipt['files'].items():
            if sha(work/name) != expected:
                raise ValueError('ELSEPA raw output changed')
        path = work/receipt['dcs_file']
    else:
        timeout = _read_timeout(work, identity)
        if timeout is not None:
            raise ElsepaCaseTimeout(timeout)
        # Only this explicitly stopped legacy point is reclassified. Unknown
        # interrupted jobs remain errors, never automatic omissions or retries.
        if kind == 'direct' and (z, charge, float(T).hex()) == INTERRUPTED_DIRECT:
            outputs = list(work.glob('dcs_*.dat'))
            if (not (work/'execution.json').exists() and (work/'input.in').exists()
                    and (work/'input.in').read_text() == text and outputs
                    and all(p.stat().st_size == 0 for p in outputs)):
                raise ElsepaCaseTimeout(_retain_timeout(work, identity, 'appendix-4 interrupted >2h case'))
        if (work/'execution.json').exists():
            raise RuntimeError('ELSEPA incomplete/failed attempt retained; inspect before retry')
        if any(p.name != 'identity.json' for p in work.iterdir()):
            raise RuntimeError('ELSEPA unfinished attempt retained; inspect before retry')
        (work/'database').symlink_to(source/'database', target_is_directory=True)
        (work/'input.in').write_text(text)
        try:
            p = subprocess.run([str(exe)], cwd=work, input=text, capture_output=True,
                               text=True, timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            # subprocess.run kills and waits for its child on timeout.
            for name, value in (('stdout.log', exc.stdout), ('stderr.log', exc.stderr)):
                (work/name).write_bytes(value if isinstance(value, bytes) else (value or '').encode())
            write(work/'execution.json', dict(status='timeout', timeout_seconds=TIMEOUT_SECONDS))
            raise ElsepaCaseTimeout(_retain_timeout(work, identity, 'subprocess timeout')) from exc
        (work/'stdout.log').write_text(p.stdout)
        (work/'stderr.log').write_text(p.stderr)
        write(work/'execution.json', dict(returncode=p.returncode))
        p.check_returncode()
        paths = list(work.glob('dcs_*.dat'))
        if len(paths) != 1 or 'stop' in p.stderr.lower():
            raise ValueError('ELSEPA incomplete output')
        path = paths[0]
        receipt = dict(dcs_file=path.name, files={p.name: sha(p) for p in work.iterdir() if p.is_file()})
        write(seal, receipt)
    a = parse(path, (work/'stdout.log').read_text())
    import re
    header = a.pop('header')
    if int(re.search(r'Z\s*=\s*(\d+)', header)[1]) != z:
        raise ValueError('ELSEPA output element identity')
    if ('Projectile: electron' in header) != (charge == -1):
        raise ValueError('ELSEPA output charge identity')
    ev = float(re.search(r'Kinetic energy\s*=\s*([-+\d.Ee]+)', header)[1])
    if abs(ev/(T*1e6)-1) > 1e-5:
        raise ValueError('ELSEPA output energy identity')
    if 'High-energy factorization' in header:
        raise ValueError('ELSEPA IHEF=0 unexpectedly factorized')
    required = ['Nuclear model: Fermi distribution', f'./database/z_{z:03}.den',
                'No absorption potential', 'No correlation-polarization potential']
    if charge == -1:
        required.append('Furness-McCarthy exchange potential')
    if any(part not in header for part in required) or any(part in header for part in
            ('LDA-I absorption', 'LDA-II absorption', 'Correlation-polarization potential (', 'Muffin-tin model')):
        raise ValueError('ELSEPA effective model mismatch')
    if charge == 1 and 'exchange potential' in header.lower():
        raise ValueError('ELSEPA positron exchange model mismatch')
    stderr = (work/'stderr.log').read_text().strip()
    if stderr:
        a['warnings'] = np.r_[a['warnings'], 'stderr: '+stderr]
    a['convergence'] = '[]'
    return a


def stack_rows(rows, z, charge, es):
    if any(not np.array_equal(rows[0]['mu'], a['mu']) for a in rows):
        raise ValueError('ELSEPA angular grids differ')
    data = {k: np.stack([a[k] for a in rows]) for k in
            ('dcs', 'error', 'series_xs', 'table_xs')}
    data.update(Z=z, charge=charge, IHEF=0, energy_MeV=es,
                mu=rows[0]['mu'], theta_deg=rows[0]['theta_deg'],
                warnings=np.array([json.dumps(a['warnings'].tolist()) for a in rows]),
                notices=np.array([json.dumps(a['notices'].tolist()) for a in rows]))
    return data


def arrays_identical(a, b):
    return (set(a) == set(b) and all(np.asarray(a[k]).dtype == np.asarray(b[k]).dtype
            and np.asarray(a[k]).shape == np.asarray(b[k]).shape
            and np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() for k in a))


def direct_energies(es, z):
    check = np.unique(np.r_[np.sqrt(es[:-1]*es[1:]), .0201*z*np.array([.99, 1.01])])
    return check[(check >= .01) & (check <= 10)]


def replacement_energy(original, nodes, checks):
    """One deterministic replacement; no chain of progressively shifted points."""
    value = float(original)*REPLACEMENT_FACTOR
    occupied = np.r_[nodes, checks]
    # Also reject collisions after rounding to ELSEPA's actual input precision.
    ev = format(value*1e6, '.6E')
    if (not np.isfinite(value) or not .01 <= value <= 10
            or value in occupied or any(format(float(t)*1e6, '.6E') == ev for t in occupied)):
        raise ValueError('ELSEPA replacement energy overlaps a node/check or leaves the table range')
    return value


def resume_replacements(path, identity, evidence, appendix=10):
    """Appendix 10 migration, preserving the original and appendix 4/6 records."""
    path = Path(path)
    old = json.loads(path.read_text())
    original_sha = sha(path)
    parents = {}
    suffixes = ('.resume_v4.json', '.resume_v6.json')+ (('.resume_v10.json',) if appendix >= 11 else ())
    if appendix >= 12:
        suffixes += ('.resume_v11.json',)
    for suffix in suffixes:
        prior = path.with_suffix(suffix)
        if not prior.exists():
            continue
        record = json.loads(prior.read_text())
        if record['original_sha256'] != original_sha:
            raise ValueError('ELSEPA prior migration original hash mismatch')
        if suffix == '.resume_v6.json' and record['previous_v4_sha256'] != parents.get('.resume_v4.json'):
            raise ValueError('ELSEPA prior migration chain mismatch')
        if suffix in ('.resume_v10.json', '.resume_v11.json') and (record['parents'] != parents or record['evidence'] != evidence):
            raise ValueError('ELSEPA prior replacement migration evidence changed')
        previous = record['current_identity']
        if ({k: v for k, v in previous.items() if k != 'implementations'}
                != {k: v for k, v in old.items() if k != 'implementations'}):
            raise ValueError('ELSEPA prior migration conditions changed')
        parents[suffix] = sha(prior)
        old = previous
    before, after = old['implementations'], identity['implementations']
    allowed = {'fetch_materials_p8a.py', 'check_p8a.py', 'generate_elsepa_p8a.py'}
    if appendix >= 11:
        allowed.add('estimate_p8a.py')  # Only the S7 entry gate changes.
    if appendix >= 12 and '.resume_v11.json' in parents:
        allowed = {'generate_elsepa_p8a.py', 'check_p8a.py'}
    if (set(before) != set(after)
            or any(before[k] != after[k] for k in before if k not in allowed)
            or {k: v for k, v in old.items() if k != 'implementations'}
            != {k: v for k, v in identity.items() if k != 'implementations'}):
        raise ValueError(f'ELSEPA generation conditions changed beyond appendix {appendix}')
    resume(path.with_suffix(f'.resume_v{appendix}.json'), dict(
        authorization=f'P8a appendix {appendix}: replace timed-out direct checks only',
        original_sha256=original_sha, parents=parents, current_identity=identity,
        changes={k: dict(before=before[k], after=after[k]) for k in before if before[k] != after[k]},
        evidence=evidence, replacement_factor=REPLACEMENT_FACTOR, timeout_seconds=TIMEOUT_SECONDS))


def replace_timeouts(out=OUT, cache=CACHE):
    """Run only replacement cases. Never regenerate nodes, GS or completed checks.

    Results are a separate sidecar consumed by S6. The original manifest,
    elemental/direct tables, timeout coverage and raw attempts stay unchanged.
    """
    from .check_p8a import direct_coverage, direct_point_checks
    out, cache = Path(out), Path(cache)
    verify_protected()
    manifest_path = out/'manifest.json'
    manifest = load_manifest(manifest_path, complete=False)
    source, exe = prepare()  # Verifies the existing pristine build; never builds.
    tables, inputs = {}, {str(manifest_path): sha(manifest_path)}
    for material, row in manifest['materials'].items():
        for ref in row.values():
            inputs[ref['path']] = ref['sha256']
        refs = load_reference(row['elastic'], material)
        definition = load_reference(row['definition'], material)
        if material != 'water':
            gs_identity = Path(row['gs']['path']).with_suffix('.identity.json')
            saved = json.loads(gs_identity.read_text())
            if (saved['definition_sha256'] != row['definition']['sha256']
                    or saved['elemental'] != {name: ref['sha256'] for name, ref in refs.items()}):
                raise ValueError('ELSEPA/GS definition or elemental generation conditions changed')
            inputs[str(gs_identity)] = sha(gs_identity)
        if set(refs) != {f'z{z}_q{q}.npz' for z in definition['weights'] for q in (-1, 1)}:
            raise ValueError('ELSEPA missing elemental references')
        for name, ref in refs.items():
            if name in tables and tables[name] != ref:
                raise ValueError('ELSEPA shared elemental references differ')
            tables[name] = ref
    jobs, groups = [], {}
    for name, ref in sorted(tables.items()):
        if 'direct' not in ref:  # H/O use their completed P1f checks.
            continue
        nodes = load_reference(ref, 'element')
        direct = load_reference(ref['direct'], 'element')
        coverage = load_reference(ref['direct_status'], 'element')
        omitted = direct_coverage(nodes, direct, coverage)
        if not omitted:
            continue
        z, charge = int(nodes['Z']), int(nodes['charge'])
        work = cache/'elsepa'/f'z{z}_q{charge}'
        for r in (ref, ref['direct'], ref['direct_status']):
            inputs[r['path']] = r['sha256']
        for original in omitted:
            t = float(original['energy_MeV'])
            raw = work/'direct'/t.hex()
            identity = dict(Z=z, charge=charge, energy_MeV=t, IHEF=0,
                            input=input_text(z, charge, t, ihef=0), executable_sha256=sha(exe))
            timeout = _read_timeout(raw, identity)
            if Path(original['timeout_record']) != raw/'timeout.json' or timeout is None:
                raise ValueError('ELSEPA original timeout evidence missing/mismatched')
            inputs[str(raw/'timeout.json')] = sha(raw/'timeout.json')
            archive = raw.with_name(raw.name+'.timed-out')
            inputs[str(archive/'timeout_record.json')] = sha(archive/'timeout_record.json')
            inputs.update({str(archive/n): value for n, value in timeout['files'].items()})
        expected = direct_energies(nodes['energy_MeV'], z)
        replacements = [replacement_energy(r['energy_MeV'], nodes['energy_MeV'], expected) for r in omitted]
        if (len(set(replacements)) != len(replacements)
                or len({format(t*1e6, '.6E') for t in replacements}) != len(replacements)):
            raise ValueError('ELSEPA duplicate replacement energies')
        generation = work/'generation.json'
        resume_replacements(generation, dict(Z=z, charge=charge, IHEF=0,
            energy_MeV=nodes['energy_MeV'].tolist(), executable_sha256=sha(exe),
            implementations=implementation_hashes()),
            evidence=dict(nodes=ref['sha256'], direct=ref['direct']['sha256'], coverage=ref['direct_status']['sha256']), appendix=12)
        for p in (generation, *(generation.with_suffix(s) for s in
                  ('.resume_v4.json', '.resume_v6.json', '.resume_v10.json', '.resume_v11.json', '.resume_v12.json'))):
            if p.exists():
                inputs[str(p)] = sha(p)
        groups[name] = dict(Z=z, charge=charge, nodes_sha256=ref['sha256'],
                            direct_sha256=ref['direct']['sha256'], coverage_sha256=ref['direct_status']['sha256'], pairs=[])
        for scope in RECORDED_OMISSIONS:
            if z == scope['Z'] and charge == scope['charge']:
                groups[name]['recorded_omission_policy'] = scope
        jobs.extend((name, work, nodes, original, value) for original, value in zip(omitted, replacements))
    for filename, expected_hash in inputs.items():
        if sha(filename) != expected_hash:
            raise ValueError('ELSEPA replacement input hash mismatch')
    path = out/'elsepa_replacements_v12.json'
    identity = dict(implementations=implementation_hashes(), executable_sha256=sha(exe),
                    replacement_factor=REPLACEMENT_FACTOR, timeout_seconds=TIMEOUT_SECONDS)
    identity['recorded_omissions'] = RECORDED_OMISSIONS
    for version in (10, 11):
        old_report = out/f'elsepa_replacements_v{version}.json'
        if not old_report.exists():
            continue
        old = json.loads(old_report.read_text())
        before, after = old['identity']['implementations'], identity['implementations']
        allowed = {'generate_elsepa_p8a.py', 'check_p8a.py'}
        if version == 10:
            allowed.add('estimate_p8a.py')  # The already authorized appendix 11 gate.
        expected = {k: v for k, v in identity.items() if k not in ('implementations', 'recorded_omissions')}
        if version == 11:
            expected['recorded_omission'] = RECORDED_OMISSION
        if (old['authorization'] != f'P8a appendix {version}' or set(before) != set(after)
                or any(before[k] != after[k] for k in before if k not in allowed)
                or {k: v for k, v in old['identity'].items() if k != 'implementations'}
                != expected
                or any(inputs.get(k) != v for k, v in old['inputs'].items())):
            raise ValueError(f'ELSEPA appendix {version} report conditions/evidence changed')
        inputs[str(old_report)] = sha(old_report)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['identity'] != identity or previous['inputs'] != inputs:
            raise ValueError('ELSEPA replacement resume conditions/evidence changed')
    report = dict(authorization='P8a appendix 12', identity=identity, inputs=inputs, groups=groups, status='pending')
    recorded = False
    # Existing raw replacement cases are sealed by case(); a resumed invocation
    # replays them, never reruns them and never invents another shifted energy.
    for name, work, nodes, original, value in jobs:
        folder = work/'replacements'/float(original['energy_MeV']).hex()
        pair = dict(original=original, energy_MeV=value, timeout_seconds=TIMEOUT_SECONDS)
        fixed = dict(Z=int(nodes['Z']), charge=int(nodes['charge']), original_energy_MeV=original['energy_MeV'],
                     replacement_energy_MeV=value) in RECORDED_OMISSIONS
        raw = folder/float(value).hex()
        raw_identity = dict(Z=int(nodes['Z']), charge=int(nodes['charge']), energy_MeV=value, IHEF=0,
                            input=input_text(int(nodes['Z']), int(nodes['charge']), value, ihef=0), executable_sha256=sha(exe))
        # Only already timed-out replacements in the explicit two-point list are
        # skipped. A new timeout (or any other point) still stops immediately.
        skip = fixed and _read_timeout(raw, raw_identity) is not None
        try:
            if skip:
                raise ElsepaCaseTimeout(dict(Z=int(nodes['Z']), charge=int(nodes['charge']), energy_MeV=value))
            row = case(source, exe, folder/float(value).hex(), int(nodes['Z']), int(nodes['charge']), value, kind='direct')
        except ElsepaCaseTimeout:
            pair.update(status='not-evaluated', reason='replacement also timed out; no further shift',
                        timeout_record=reference(folder/float(value).hex()/'timeout.json', 'ELSEPA', 'appendix 10 timeout', 'MeV', 'element'))
            if skip:
                pair['reason'] = 'appendix 12: fixed positron omission; original and replacement timed out'
                recorded = True
        else:
            data = stack_rows([row], int(nodes['Z']), int(nodes['charge']), np.array([value]))
            saved = folder/'direct.npz'
            save_verified_table(saved, data)
            checks = direct_point_checks(nodes, data, 0, kind='replacement')
            pair.update(status=checks['status'], checks=checks,
                        data=reference(saved, str(folder), 'appendix 10 direct replacement', 'cm2/sr; MeV', 'element'))
        groups[name]['pairs'].append(pair)
        report['status'] = 'pending' if pair['status'] == 'pass' or skip else pair['status']
        write(path, report)
        if pair['status'] != 'pass' and not skip:
            raise RuntimeError('ELSEPA replacement timed out or failed; stop without another shift')
    report['status'] = OMISSION_STATUS if recorded else 'pass'
    write(path, report)
    verify_protected()
    return report


def collect_direct(source, exe, work, z, charge, es, nodes):
    """Timeouts omit checks, never nodes; keep a complete coverage receipt."""
    expected = direct_energies(es, z)
    rows, completed, omitted = [], [], []
    for t in expected:
        try:
            a = case(source, exe, work/'direct'/float(t).hex(), z, charge, float(t), kind='direct')
        except ElsepaCaseTimeout as exc:
            omitted.append(dict(status='not-evaluated', reason=exc.record['reason'],
                                Z=z, charge=charge, energy_MeV=float(t),
                                timeout_record=str(work/'direct'/float(t).hex()/'timeout.json'),
                                timeout_seconds=TIMEOUT_SECONDS))
        else:
            rows.append(a)
            completed.append(t)
    if rows:
        data = stack_rows(rows, z, charge, np.asarray(completed))
    else:
        # Even all-timeout direct data has a valid empty shape, not fake values.
        data = {k: np.asarray(v)[:0].copy() if k in
                ('energy_MeV', 'dcs', 'error', 'series_xs', 'table_xs', 'warnings', 'notices')
                else np.asarray(v).copy() for k, v in nodes.items()}
    save_verified_table(work/'direct.npz', data)
    coverage = dict(Z=z, charge=charge, expected_energy_MeV=expected.tolist(),
                    completed_count=len(completed), not_evaluated_count=len(omitted),
                    status='not-evaluated' if omitted else 'pass', omitted=omitted)
    write(work/'direct_status.json', coverage)
    return coverage


def generate(out=OUT, cache=CACHE):
    from .p1f_common import WORK as P1F, OUT as P1F_OUT
    from .gs_backend import build
    verify_protected()
    manifest = load_manifest(out/'manifest.json', complete=False)
    definitions = {m: load_reference(manifest['materials'][m]['definition'], m) for m in MATERIALS}
    elements = sorted({int(z) for d in definitions.values() for z in d['weights']})
    source, exe = prepare()  # read-only verifier of the pristine existing build
    expected = json.loads((P1F_OUT/'t1.json').read_text())['table_hashes']
    receipts = {}
    direct_summary = {}
    for z in elements:
        for charge in (-1, 1):
            name = f'z{z}_q{charge}.npz'
            if z in (1, 8):
                path = P1F/'tables'/name
                if sha(path) != expected[name]:
                    raise ValueError('reused P1f elemental hash mismatch')
                receipts[name] = reference(path, 'P1f T1', 'reused without generation', 'cm2/sr; MeV', 'element', Z=z, charge=charge)
                continue
            es = energies()
            work = cache/'elsepa'/f'z{z}_q{charge}'
            resume_s4(work/'generation.json', dict(Z=z, charge=charge, IHEF=0,
                    energy_MeV=es.tolist(), executable_sha256=sha(exe),
                    implementations=implementation_hashes()))
            passes = []
            path = work/name
            for passno in (1, 2):
                if path.exists() and any(not (work/f'pass{passno}'/float(t).hex()/'complete.json').exists() for t in es):
                    raise ValueError('ELSEPA saved node table lacks completed raw evidence')
                rows = [case(source, exe, work/f'pass{passno}'/float(t).hex(), z, charge, float(t)) for t in es]
                passes.append(stack_rows(rows, z, charge, es))
            if not arrays_identical(*passes):
                raise ValueError('ELSEPA two-pass arrays differ')
            save_verified_table(path, passes[0])
            coverage = collect_direct(source, exe, work, z, charge, es, passes[0])
            direct_summary[name] = coverage
            write(out/'elsepa_direct_status.json', dict(
                status='pending',
                groups=direct_summary, purpose='S8 RESULTS: report omissions per element and charge; not an acceptance verdict'))
            receipts[name] = reference(path, str(work/'generation.json'), 'two independent ELSEPA IHEF=0 passes',
                                      'cm2/sr; MeV', 'element', Z=z, charge=charge,
                                      direct=reference(work/'direct.npz', str(work/'direct'), 'independent midpoints/probes', 'cm2/sr; MeV', 'element'),
                                      direct_status=reference(work/'direct_status.json', str(work/'direct'), 'appendix 4 coverage and timeouts', 'MeV; count', 'element'))
    write(out/'elsepa_direct_status.json', dict(
        status='not-evaluated' if any(r['not_evaluated_count'] for r in direct_summary.values()) else 'pass',
        groups=direct_summary, purpose='S8 RESULTS: omissions per element and charge; coverage only, not data acceptance'))
    write(out/'elemental_elsepa.json', receipts)
    for material, d in definitions.items():
        row = manifest['materials'][material]
        selected = {name: r for name, r in receipts.items() if r['Z'] in set(map(int, d['weights']))}
        path = cache/material/'elastic.json'
        write(path, selected)
        row['elastic'] = reference(path, str(out/'elemental_elsepa.json'), 'composition-sorted ELSEPA mixture', 'cm-1; cm2/sr', material)
        if material == 'water':
            # Explicit water compatibility route; do not recompute legacy GS.
            path = P1F/'gs.npz'
            prior = json.loads((P1F_OUT/'manifest.json').read_text())
            expected_gs = prior['files']['~/.cache/chatcarlo/elsepa/p1f/gs.npz']
            if sha(path) != expected_gs:
                raise ValueError('legacy water GS hash mismatch')
            row['gs'] = reference(path, 'P1f GS', 'byte-preserving legacy water path', 'MeV; dimensionless; cm-1', material)
        else:
            tables = {int(z): load_reference(selected[f'z{z}_q-1.npz'], 'element') for z in d['weights']}
            provider = MixtureElsepa(d, tables)
            path = cache/material/'gs.npz'
            identity = dict(definition_sha256=row['definition']['sha256'],
                            elemental={n: v['sha256'] for n, v in selected.items()},
                            implementations=implementation_hashes())
            resume_s4(path.with_suffix('.identity.json'), identity)
            seal = path.with_suffix('.sha.json')
            if path.exists():
                if not seal.exists() or sha(path) != json.loads(seal.read_text())['sha256']:
                    raise ValueError('GS table changed / incomplete generation')
            else:
                build(provider, path)
                with np.load(path, allow_pickle=False) as f:
                    arrays = {k: f[k].copy() for k in f.files}
                np.savez(path, **arrays, material=np.array(material), density_g_cm3=np.array(d['density_g_cm3']))
                write(seal, dict(sha256=sha(path)))
            row['gs'] = reference(path, list(identity['elemental']), 'gs_backend.build(MixtureElsepa)', 'MeV; dimensionless; cm-1', material)
        write(out/'manifest.json', manifest)
    verify_protected()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--replace-timeouts', action='store_true')
    args = parser.parse_args()
    if args.replace_timeouts:
        replace_timeouts()
    else:
        generate()


if __name__ == '__main__':
    main()
