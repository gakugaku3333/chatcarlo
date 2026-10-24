"""P9a-only material acquisition and gates; external work requires explicit CLI use."""
import json
import re
from pathlib import Path
import numpy as np
from . import materials_p8a as mat
from . import p9a

CACHE = Path.home()/'.cache/chatcarlo/p9a'
OLD = Path.home()/'.cache/chatcarlo/p8e/materials'
NAME = 'seed_agti'


def read(path):
    path = Path(path)
    if path.suffix == '.json':
        return json.loads(path.read_text())
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def protected_paths():
    """Freeze existing shared implementation, old records and old material data."""
    base = Path(__file__).parent
    paths = [p for p in base.rglob('*') if p.is_file() and
             p.suffix in ('.py', '.h', '.cpp', '.cu', '.f') and
             'p9a' not in p.name and '__pycache__' not in p.parts]
    paths.append(base/'p9a_egs.f')
    for folder in (mat.CACHE, OLD, *(mat.ROOT/'docs/validation/mv'/n for n in
                    ('p8a', 'p8e', 'p8g1', 'p8g2', 'p8g3'))):
        paths.extend(p for p in folder.rglob('*') if p.is_file())
    return sorted(set(paths))


def protect(out=p9a.OUT, paths=None):
    paths = protected_paths() if paths is None else paths
    record = {str(p.resolve()): mat.sha(p) for p in paths}
    if not record:
        raise ValueError('empty protection set')
    mat.resume(Path(out)/'protected_start.json', record)
    return verify_protected(out)


def verify_protected(out=p9a.OUT):
    record = read(Path(out)/'protected_start.json')
    if not record or any(mat.sha(Path(p)) != h for p, h in record.items()):
        raise ValueError('protected P8 files changed; stop')
    return dict(status='pass', files=len(record))


def finish(stage, files, out):
    record = {str(Path(p).resolve()): mat.sha(p) for p in files}
    mat.resume(Path(out)/(stage+'.json'), record)
    return dict(status='pass', stage=stage, files=record)


def verified(stage, out):
    record = read(Path(out)/(stage+'.json'))
    for p, expected in record.items():
        if mat.sha(p) != expected:
            raise ValueError('P9a stage hash changed: '+p)
    return record


def photon(cache=CACHE, out=p9a.OUT):
    from .fetch_materials_p8a import photon_table
    folder = Path(cache)/NAME
    folder.mkdir(parents=True, exist_ok=True)
    if (Path(out)/'photon.json').exists():
        verified('photon', out)
        return dict(status='pass', resumed=True)
    # Only composition is consumed here; no provisional I is written.
    d = p9a.seed_definition()
    np.savez(folder/'photon.npz', **photon_table(Path(cache)/'xcom'/NAME, d))
    return finish('photon', [folder/'photon.npz'], out)


def stopping_data(nodes, checks):
    fields = ('T', 'collision', 'radiative', 'delta')
    result = {k: np.asarray(nodes[k]) for k in fields}
    result.update({'check_'+k: np.asarray(checks[k]) for k in fields})
    result['total'] = result['collision']+result['radiative']
    result['check_total'] = result['check_collision']+result['check_radiative']
    return result


def finalized_definition(text):
    from .pegs_p8a import validate_diagnostic_state
    d = dict(p9a.seed_definition(), medium_name='P8A_SEED_AGTI')
    validate_diagnostic_state(text, d)
    rows = re.findall(r'^\s*META\s+([^\n]+)', text, re.M)
    if len(rows) != 1:
        raise ValueError('META count')
    d['I_MeV'] = float(rows[0].split()[-1].replace('D', 'E'))*1e-6
    mat.constants(d)
    d['I_source'] = 'PEGS5 diagnostic META IEV (eV); general mixture formula'
    return d


def estar_report(definition, stopping, elements):
    """Element ESTAR comparator only; electron-weighted logarithmic Bragg I."""
    c = mat.constants(definition)
    ts = np.asarray(stopping['T'])
    keep = (ts >= .01) & (ts <= .1)
    T = ts[keep]
    mixed = sum(definition['weights'][str(z)]*mat.log_interp(T, elements[z]['T'],
                    elements[z]['collision']) for z in c['Z'])
    log_i = sum(n*z*np.log(elements[z]['I_MeV']) for z, n in zip(c['Z'], c['NJ']))/c['NE']
    return dict(status='report-only', T_MeV=T.tolist(),
                PEGS_over_ESTAR_minus_one=(stopping['collision'][keep]/mixed-1).tolist(),
                PEGS_I_MeV=definition['I_MeV'], Bragg_I_MeV=float(np.exp(log_i)))


def acquire_estar(cache, definition, stopping):
    from . import fetch_materials_p8a as f
    import html
    raw = Path(cache)/'estar'
    listing = f.acquire(raw/'materials.html', f.STAR+'compos.pl')
    elements = {}
    for z, name in ((22, 'Titanium'), (47, 'Silver')):
        hits = re.findall(r'<option\s+value=[\'\"]?(\d+)[\'\"]?[^>]*>([^<]+)', listing, re.I)
        hits = [(n, html.unescape(s).strip()) for n, s in hits
                if re.fullmatch(r'\d+: '+name, html.unescape(s).strip(), re.I)]
        if len(hits) != 1:
            raise ValueError('ESTAR element listing '+name)
        number, title = hits[0]
        comp = f.acquire(raw/f'z{z}_composition.html', f.STAR+'compos.pl?matno='+number)
        d = f.parse_composition(comp, 'element', title)
        T = stopping['T'][(stopping['T'] >= .01) & (stopping['T'] <= .1)]
        # ESTAR answers 502 for 364 energies in one request; use P8a's 100-point chunks.
        parts = []
        for k in range(0, len(T), 100):
            chunk = T[k:k+100]
            response = f.acquire(raw/f'z{z}_stopping_{k//100}.html', f.STAR+'e_table-t.pl',
                                dict(matno=number, Energies='\n'.join(format(t, '.17g') for t in chunk)))
            parts.append(f.parse_stopping(response, chunk))
        table = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        elements[z] = dict(table, I_MeV=d['I_MeV'])
    return estar_report(definition, stopping, elements)


def pegs(cache=CACHE, out=p9a.OUT):
    from . import pegs_p8a as pg
    if (Path(out)/'pegs.json').exists():
        verified('pegs', out)
        return dict(status='pass', resumed=True)
    old = mat.load_manifest()
    water = mat.load_reference(old['materials']['water']['stopping'], 'water')
    provisional = dict(p9a.seed_definition(), medium_name='P8A_SEED_AGTI')
    nodes = pg.run_diagnostic(provisional, water['T'], Path(cache)/'pegs/nodes')
    checks = pg.run_diagnostic(provisional, water['check_T'], Path(cache)/'pegs/checks')
    d = finalized_definition(nodes['text'])
    if finalized_definition(checks['text']) != d:
        raise ValueError('PEGS I differs between node and check runs')
    stop = stopping_data(nodes, checks)
    folder = Path(cache)/NAME
    folder.mkdir(parents=True, exist_ok=True)
    mat.write(folder/'definition.json', d)
    mat.write(folder/'constants.json', mat.constants(d))
    np.savez(folder/'stopping.npz', **stop)
    np.savez(folder/'shells.npz', **pg.extract_shells(nodes['text'], d))
    np.savez(folder/'pair.npz', **pg.pair_parameters(nodes['pegs'].read_text(), 'P8A_SEED_AGTI', d))
    mat.write(folder/'estar_comparison.json', acquire_estar(cache, d, stop))
    delta = stop['delta']
    mat.write(folder/'density_report.json', dict(status='report-only',
        negative_indices=np.flatnonzero(delta < 0).tolist(),
        decreasing_indices=np.flatnonzero(np.diff(delta) < 0).tolist()))
    files = [folder/n for n in ('definition.json', 'constants.json', 'stopping.npz',
              'shells.npz', 'pair.npz', 'estar_comparison.json', 'density_report.json')]
    for part in ('nodes', 'checks'):
        files.extend(p for p in (Path(cache)/'pegs'/part).iterdir() if p.is_file())
    return finish('pegs', files, out)


def elsepa(cache=CACHE, out=p9a.OUT):
    from . import generate_elsepa_p8a as el
    from .gs_backend import build
    from .check_p8a import direct_point_checks
    verified('pegs', out)
    if (Path(out)/'elsepa.json').exists():
        verified('elsepa', out)
        return dict(status='pass', resumed=True)
    source, exe = el.prepare()
    es = el.energies()
    refs, atomic, replacements = {}, {}, {}
    for z in (22, 47):
        work = Path(cache)/'elsepa'/f'z{z}_q-1'
        work.mkdir(parents=True, exist_ok=True)
        passes = [el.stack_rows([el.case(source, exe, work/f'pass{j}'/float(t).hex(),
                    z, -1, float(t)) for t in es], z, -1, es) for j in (1, 2)]
        if not el.arrays_identical(*passes):
            raise ValueError('ELSEPA two-pass mismatch')
        path = work/f'z{z}_q-1.npz'
        el.save_verified_table(path, passes[0])
        coverage = el.collect_direct(source, exe, work, z, -1, es, passes[0])
        def ref(p):
            return mat.reference(p, 'ELSEPA IHEF=0', 'P8a case; electron only', 'cm2/sr; MeV', 'element')
        r = ref(path)
        r.update(direct=ref(work/'direct.npz'), direct_status=ref(work/'direct_status.json'))
        name = path.name
        refs[name], atomic[z] = r, passes[0]
        if coverage['omitted']:
            group = dict(Z=z, charge=-1, nodes_sha256=r['sha256'],
                direct_sha256=r['direct']['sha256'], coverage_sha256=r['direct_status']['sha256'], pairs=[])
            for original in coverage['omitted']:
                t = el.replacement_energy(original['energy_MeV'], es, el.direct_energies(es, z))
                # Exactly one substitute; another timeout propagates and stops.
                a = el.case(source, exe, work/'replacements'/float(original['energy_MeV']).hex()/t.hex(),
                            z, -1, t, kind='direct')
                data = el.stack_rows([a], z, -1, np.array([t]))
                dest = work/'replacements'/float(original['energy_MeV']).hex()/'replacement.npz'
                el.save_verified_table(dest, data)
                result = direct_point_checks(passes[0], data, 0, kind='replacement')
                group['pairs'].append(dict(original=original, energy_MeV=t, timeout_seconds=600,
                                           status=result['status'], data=ref(dest)))
            replacements[name] = group
    folder = Path(cache)/NAME
    mat.write(folder/'elastic.json', refs)
    mat.write(folder/'replacements.json', replacements)
    d = read(folder/'definition.json')
    identity = dict(definition=mat.sha(folder/'definition.json'), elements=refs)
    mat.resume(folder/'gs.identity.json', identity)
    gs = folder/'gs.npz'
    if gs.exists():
        if read(folder/'gs.sha.json')['sha256'] != mat.sha(gs):
            raise ValueError('GS changed')
    else:
        build(mat.MixtureElsepa(d, atomic), gs)
        mat.write(folder/'gs.sha.json', dict(sha256=mat.sha(gs)))
    files = [folder/n for n in ('elastic.json', 'replacements.json', 'gs.npz', 'gs.identity.json', 'gs.sha.json')]
    for r in refs.values():
        files += [Path(r['path']), Path(r['direct']['path']), Path(r['direct_status']['path'])]
    return finish('elsepa', files, out)


def bundle(cache=CACHE, out=p9a.OUT):
    from . import photon_transport_p8e as transport
    from .generate_fluorescence_p8e import fluorescence_table
    for stage in ('photon', 'pegs', 'elsepa'):
        verified(stage, out)
    if (Path(out)/'bundle.json').exists():
        verified('bundle', out)
        return dict(status='pass', resumed=True)
    old, identity = transport.load_bundles(OLD)
    folder = Path(cache)/NAME
    c, stop, photon, gs, pair, shells = [read(folder/n) for n in
        ('constants.json', 'stopping.npz', 'photon.npz', 'gs.npz', 'pair.npz', 'shells.npz')]
    fluor = fluorescence_table(shells)
    m = transport.make_material(c, stop, photon, gs, pair, shells, fluor,
            transport.brems_tables(np.array(c['Z'], float), np.array(c['NJ'], float), stop), density=2.47)
    checks = transport.bundle_checks(m)
    dest = Path(cache)/'materials'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/'seed_agti_bundle.npz'
    np.savez(path, **m._asdict())
    np.savez(dest/'seed_agti_fluorescence.npz', **fluor)
    record = dict(names=[*mat.MATERIALS, NAME], densities=[*identity['densities'], 2.47],
                  old_identity=identity, old_manifest_sha256=mat.sha(OLD/'bundles.json'),
                  seed_sha256=mat.sha(path), checks=checks,
                  definition_sha256=mat.sha(folder/'definition.json'))
    mat.write(dest/'bundles.json', record)
    return finish('bundle', [path, dest/'seed_agti_fluorescence.npz', dest/'bundles.json'], out)


def load_bundles(cache=CACHE, out=p9a.OUT):
    from . import photon_transport_p8e as t
    verified('bundle', out)
    record = read(Path(cache)/'materials/bundles.json')
    old, identity = t.load_bundles(OLD)
    if (identity != record['old_identity'] or mat.sha(OLD/'bundles.json') != record['old_manifest_sha256']
            or record['names'] != [*mat.MATERIALS, NAME]
            or mat.sha(Path(cache)/NAME/'definition.json') != record['definition_sha256']):
        raise ValueError('old/seed bundle identity changed')
    values = read(Path(cache)/'materials/seed_agti_bundle.npz')
    seed = t.Material(*(values[k].item() if values[k].ndim == 0 else values[k] for k in t.Material._fields))
    if seed.excitation != read(Path(cache)/NAME/'definition.json')['I_MeV']:
        raise ValueError('bundle I differs from PEGS I')
    return [*old, seed], record


def check_data(cache=CACHE, out=p9a.OUT):
    from . import check_p8a as ck
    for stage in ('photon', 'pegs', 'elsepa'):
        verified(stage, out)
    folder = Path(cache)/NAME
    d = read(folder/'definition.json')
    refs = read(folder/'elastic.json')
    atomic = {int(z): mat.load_reference(refs[f'z{z}_q-1.npz'], 'element') for z in d['weights']}
    photon = read(folder/'photon.npz')
    from .fetch_materials_p8a import check_edge_jumps
    check_edge_jumps(photon['E'], photon['photo'], photon['edges'])
    report = dict(interpolation=ck.interpolation_checks(read(folder/'stopping.npz'), photon, mat.constants(d)['NE']),
                  shells=ck.shell_checks(read(folder/'shells.npz'), photon, d),
                  atomic=ck.atomic_checks(refs, read(folder/'replacements.json')),
                  mixture=ck.mixture_checks(d, atomic),
                  gs=ck.gs_checks(mat.MixtureElsepa(d, atomic), read(folder/'gs.npz')))
    def states(v):
        if isinstance(v, dict):
            if 'status' in v:
                yield v['status']
                return  # The P8a checker already resolves replacement/diagnostic rows.
            for k, item in v.items():
                if k != 'status':
                    yield from states(item)
        elif isinstance(v, list):
            for item in v:
                yield from states(item)
    report['status'] = 'pass' if all(s in ('pass', 'report-only') for s in states(report)) else 'fail'
    mat.write(Path(out)/'check-data.json', report)
    if report['status'] != 'pass':
        raise ValueError('P9a material data gate failed; stop')
    return report


def dispatch(command, cache=CACHE, out=p9a.OUT):
    if command == 'protect':
        return protect(out)
    verify_protected(out)
    try:
        if command == 'check':
            from .gpu.p9a import local
            return local(Path(cache)/'gpu', cache, out)
        return globals()[command.replace('-', '_')](cache, out)
    finally:
        verify_protected(out)
