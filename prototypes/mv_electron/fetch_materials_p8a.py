"""Acquire NIST definitions/ESTAR and isolated XCOM tables, retaining raw receipts.

Executed by the S1/S2 operator, never at import or from offline tests. Arbitrary
ESTAR density is not assumed: --lung explicitly uses PEGS Sternheimer fallback.
"""
import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from .materials_p8a import (ROOT, OUT, CACHE, MATERIALS, energy_grid,
                            finite, sha, write, resume, reference, load_manifest,
                            verify_protected, implementation_hashes, composition_id)
from .fetch_data import request

STAR = 'https://physics.nist.gov/cgi-bin/Star/'
NUMBER = r'[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?'
# Name matching only; no remembered NIST material numbers or physical values.
NAMES = dict(water=r'^Water, Liquid$', lung=r'^Lung \(ICRP\)$',
             bone=r'^Bone, Cortical \(ICRP\)$', lead=r'^\d+: Lead$')


def material_list(raw):
    options = re.findall(r'<option\s+value=[\'"]?(\d+)[\'"]?[^>]*>([^<]+)', raw, re.I)
    result = {}
    for key, pattern in NAMES.items():
        hits = [(n, html.unescape(s).strip()) for n, s in options if re.fullmatch(pattern, html.unescape(s).strip(), re.I)]
        if len(hits) != 1:
            raise ValueError('NIST material name not uniquely found: ' + key)
        result[key] = dict(matno=hits[0][0], name=hits[0][1])
    return result


def parse_composition(raw, material, expected_name):
    from .materials_p8a import composition
    title = re.search(r'Composition of\s+([^<:]+)', raw, re.I)
    if title is None or title[1].strip().casefold() != re.sub(r'^\d+:\s*', '', expected_name).casefold():
        raise ValueError('composition material identity mismatch')
    def scalar(label):
        m = re.search(label+r'.*?</td>\s*<td[^>]*>\s*('+NUMBER+')', raw, re.I | re.S)
        if m is None:
            raise ValueError('missing composition scalar: ' + label)
        return float(m[1])
    density = scalar(r'Density')
    excitation = scalar(r'Mean Excitation Energy') * 1e-6
    weights = {z: float(w) for z, w in re.findall(r'<td>\s*(\d+)\s*</td>\s*<td>\s*('+NUMBER+r')\s*</td>', raw, re.I)}
    import xraylib
    a = {z: xraylib.AtomicWeight(int(z)) for z in weights}
    composition(weights, a)
    return dict(material=material, nist_name=expected_name, density_g_cm3=density,
                standard_density_g_cm3=density, I_MeV=excitation,
                weights=weights, atomic_weights=a,
                sources=[STAR+'compos.pl', 'xraylib.AtomicWeight'])


def parse_stopping(raw, requested):
    text = html.unescape(re.sub(r'<[^>]+>', '\n', raw))
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 5 and all(re.fullmatch(NUMBER, s) for s in fields):
            rows.append([float(s.replace('D', 'E')) for s in fields])
    a = finite(rows)
    requested = finite(requested, 'energy')
    # A final 1-point chunk is legal (1201 nodes in chunks of 100).
    if requested.ndim != 1 or len(requested) < 1 or np.any(requested <= 0) or np.any(np.diff(requested) <= 0):
        raise ValueError('energy must be positive and strictly increasing')
    if a.shape != (len(requested), 5) or not np.allclose(a[:, 0], requested, rtol=6e-4, atol=0):
        raise ValueError('ESTAR five-column row correspondence')
    if np.any(a[:, 1:4] <= 0) or np.any(a[:, 4] < 0):
        raise ValueError('invalid ESTAR stopping values')
    return dict(T=requested, collision=a[:, 1], radiative=a[:, 2], total=a[:, 3], delta=a[:, 4])


def parse_range(raw):
    # Dedicated default-grid response has 7 columns; never treat DELTA as range.
    text = html.unescape(re.sub(r'<[^>]+>', '\n', raw))
    if not re.search(r'CSDA', text, re.I) or not re.search(r'g\s*/\s*cm', text, re.I):
        raise ValueError('CSDA range heading/units absent')
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 7 and all(re.fullmatch(NUMBER, s) for s in fields):
            rows.append([float(s) for s in fields])
    a = finite(rows)
    if a.ndim != 2 or a.shape[1] != 7:
        raise ValueError('ESTAR range must have seven columns')
    energy_grid(a[:, 0])
    if np.any(a[:, 4] <= 0) or np.any(np.diff(a[:, 4]) <= 0):
        raise ValueError('invalid CSDA range')
    return dict(T=a[:, 0], csda_g_cm2=a[:, 4])


def acquire(path, url, fields=None):
    """Keep requests and bytes sealed, including on subsequent invocation."""
    fields = {} if fields is None else fields
    identity = dict(url=url, fields=fields)
    resume(path.with_suffix('.request.json'), identity)
    receipt = path.with_suffix('.receipt.json')
    if path.exists():
        if not receipt.exists() or json.loads(receipt.read_text())['sha256'] != sha(path):
            raise ValueError('raw acquisition hash / incomplete receipt')
    else:
        args = [v for k, value in fields.items() for v in ('--data-urlencode', k+'='+value)]
        raw = request(url, args)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        write(receipt, dict(**identity, sha256=sha(path)))
    return path.read_text()


def xcom_edges(source, elements):
    edges = []
    for z in elements:
        lines = (source/f'MDATX3.{z:03}').read_text().splitlines()
        count = int(lines[1].split()[0])
        if not count:
            continue
        # IDs use 12I6, labels 14(1X,A2), edge energies use 8F9.1.
        start = 2 + (count+11)//12 + 1
        v = [float(x) for line in lines[start:start+(count+7)//8] for x in line.split()]
        if len(v) != count:
            raise ValueError('XCOM absorption edge count')
        edges.extend(e*1e-6 for e in v)
    return np.unique(edges)


def prepare_xcom(work):
    source = ROOT/'docs/validation/mv/p0/data/XCOM'
    files = {p.name: sha(p) for p in source.iterdir() if p.is_file() and p.suffix.lower() != '.exe' and p.name != 'xcom'}
    # Only output formatting changes, in a P8a working copy. Preserve original
    # arithmetic; print enough digits to check the 0.1% requirement.
    original = (source/'XCOM.f').read_text()
    code = original.replace('1PE10.3,1P7E10.3', '1PE23.15,1P7E23.15')
    resume(work/'identity.json', dict(sources=files, format='1PE23.15,1P7E23.15', flags=['-std=legacy', '-O2']))
    if not (work/'xcom').exists():
        shutil.copytree(source, work, dirs_exist_ok=True)
        (work/'XCOM.f').write_text(code)
        p = subprocess.run(['gfortran', '-std=legacy', '-O2', 'XCOM.f', '-o', 'xcom'], cwd=work, capture_output=True, text=True)
        (work/'build.log').write_text(p.stdout+p.stderr)
        p.check_returncode()
    return source


def xcom(work, energies, name, weights):
    """NEGO=3: requested-only energies; edge sides are requested explicitly."""
    result = []
    for offset in range(0, len(energies), 500):
        es = np.asarray(energies[offset:offset+500])
        tag = f'{name}_{offset}'
        request_text = str(len(es))+'\n'+'\n'.join(format(v, '.17g') for v in es)+'\n'
        if len(weights) == 1:
            spec = ['1', str(next(iter(weights))), '3']
        else:
            import xraylib
            spec = ['4', str(len(weights))] + [v for z, w in sorted(weights.items())
                    for v in (xraylib.AtomicNumberToSymbol(z), format(w, '.17g'))] + ['1']
        inp = '\n'.join([name]+spec+['3', '2', 'energies.txt', 'table.txt', '1'])+'\n'
        raw_path = work/(tag+'.txt')
        resume(work/(tag+'.identity.json'), dict(input=inp, energies=request_text, executable_sha256=sha(work/'xcom')))
        if not raw_path.exists():
            (work/'energies.txt').write_text(request_text)
            p = subprocess.run(['./xcom'], input=inp, cwd=work, capture_output=True, text=True, timeout=120)
            (work/(tag+'.log')).write_text(p.stdout+p.stderr)
            p.check_returncode()
            raw_path.write_bytes((work/'table.txt').read_bytes())
            write(work/(tag+'.sha.json'), dict(sha256=sha(raw_path)))
        if sha(raw_path) != json.loads((work/(tag+'.sha.json')).read_text())['sha256']:
            raise ValueError('XCOM raw output changed')
        rows = []
        for line in raw_path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 8:
                try:
                    rows.append([float(v) for v in fields])
                except ValueError:
                    pass
        a = finite(rows)
        if a.shape != (len(es), 8) or not np.allclose(a[:, 0], es, rtol=2e-6, atol=0):
            raise ValueError('XCOM requested energy correspondence')
        result.append(a[:, 1:])
    return np.vstack(result)


def check_edge_jumps(es, photo, edges, *, elemental=False):
    """Rounded XCOM edge labels are not reliable upper-side requests.

    Mixture jumps can be arbitrarily weak for trace elements; allow the smooth
    E**-4 falloff (appendix 7), but require a strict elemental jump.
    """
    es, photo = finite(es), finite(photo)
    if es.shape != photo.shape or np.any(photo <= 0):
        raise ValueError('XCOM edge table shape / positivity')
    for edge in edges:
        sides = []
        for request in (edge*(1-1e-6), edge*(1+1e-6)):
            ids = np.flatnonzero(es == request)
            if len(ids) != 1:
                raise ValueError('XCOM edge side request missing')
            sides.append(photo[ids[0]])
        # Between the two requests (relative 2e-6) the smooth photoelectric
        # falloff is slower than E**-4; a trace-element jump may be smaller
        # than that falloff, so the mixture only must not fall faster.
        floor = sides[0]*((1-1e-6)/(1+1e-6))**4
        if sides[1] < floor or (elemental and sides[1] <= sides[0]):
            raise ValueError('XCOM absorption edge did not jump: '+str(edge))


def refine_edge_grid(es, edges):
    """Appendix 9: quarter log intervals in the 10% below edges <=20 keV.

    Keep every existing node. Clip refinements to each lower-side window and
    never subdivide the explicit two-sided edge gap, including overlapping
    windows of nearby edges. Thus appendix 8 still excludes exactly that gap's
    midpoint; high-energy and pair-threshold nodes are unchanged.
    """
    es, edges = energy_grid(es), finite(edges)
    gaps = {(edge*(1-1e-6), edge*(1+1e-6)) for edge in edges}
    added = []
    for edge in edges[edges <= .02]:
        for a, b in zip(es[:-1], es[1:]):
            if (a, b) in gaps:
                continue
            lower, upper = max(a, .9*edge), min(b, edge*(1-1e-6))
            if lower < upper:
                added.extend(np.geomspace(lower, upper, 5))
    return np.unique(np.r_[es, added])


def photon_table(work, definition):
    from .photon_data_p2 import grid
    source = prepare_xcom(work)
    text = (source/'XCOM.f').read_text()
    thresholds = np.array([float(v.replace('D', 'E'))/1e6 for v in re.search(r'DATA EPAIR1/([^/]+)/,EPAIR2/([^/]+)/', text).groups()])
    weights = {int(z): w for z, w in definition['weights'].items()}
    edges = xcom_edges(source, weights)
    edges = edges[(edges > .001) & (edges < 20)]
    es = grid(thresholds)
    # Each segment is sampled independently; conceptual duplicate edge nodes
    # XCOM labels round edges to 0.1 eV. Query BOTH sides off the label;
    # evaluating exactly at the printed edge can return the lower value.
    es = np.unique(np.r_[es[~np.isin(es, edges)], edges*(1-1e-6), edges*(1+1e-6)])
    es = refine_edge_grid(es, edges)
    values = xcom(work, es, 'mixture', weights)
    check_edge_jumps(es, values[:, 2], edges)
    mid = np.sqrt(es[:-1]*es[1:])
    direct = xcom(work, mid, 'mixture_check', weights)
    photo_e = es.copy()
    for edge in edges:
        photo_e[np.searchsorted(es, edge*(1-1e-6))] = edge
        photo_e[np.searchsorted(es, edge*(1+1e-6))] = edge
    # Also acquire all elemental process tables, once per element across materials.
    for z in weights:
        ew = work.parent/'elements'/f'z{z}'
        prepare_xcom(ew)
        elemental = xcom(ew, es, 'nodes_'+definition['material'], {z: 1.})
        own_edges = xcom_edges(source, [z])
        own_edges = own_edges[(own_edges > .001) & (own_edges < 20)]
        check_edge_jumps(es, elemental[:, 2], own_edges, elemental=True)
    return dict(E=es, photo_E=photo_e, photo=values[:, 2], pair_nuclear=values[:, 3],
                pair_electron=values[:, 4], thresholds=thresholds, edges=edges,
                edge_lower_requests=edges*(1-1e-6), edge_upper_requests=edges*(1+1e-6),
                check_E=mid, check_photo=direct[:, 2], check_total=direct[:, 5],
                check_pair_nuclear=direct[:, 3], check_pair_electron=direct[:, 4],
                xcom_incoherent=values[:, 1])


def fetch(out=OUT, cache=CACHE):
    verify_protected()
    if (out/'lung_state.json').exists():
        raise ValueError('S2 already started; S1 must not overwrite the selected lung state')
    rawdir = out/'raw'
    listing = acquire(rawdir/'materials.html', STAR+'compos.pl')
    selected = material_list(listing)
    write(out/'nist_materials.json', selected)
    nodes = np.geomspace(.01, 20, 1201)
    mid = np.sqrt(nodes[:-1]*nodes[1:])
    for material in MATERIALS:
        item = selected[material]
        folder = cache/material
        folder.mkdir(parents=True, exist_ok=True)
        outputs_receipt = folder/'fetch_outputs.json'
        if outputs_receipt.exists():
            for name, expected in json.loads(outputs_receipt.read_text()).items():
                if sha(folder/name) != expected:
                    raise ValueError('S1 saved output changed: '+material+'/'+name)
        comp = acquire(rawdir/(material+'_composition.html'), STAR+'compos.pl?matno='+item['matno'])
        d = parse_composition(comp, material, item['name'])
        d['matno'] = item['matno']
        d['composition_sha256'] = sha(rawdir/(material+'_composition.html'))
        d['sources'] += [STAR+'e_table-t.pl', 'https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz']
        resume(folder/'fetch_identity.json', dict(definition=d, nodes=nodes.tolist(), check=mid.tolist(), implementations=implementation_hashes()))
        # Separate requests preserve all requested significant digits and avoid
        # NIST response-size limits. No default nodes in interpolation checks.
        def table(es, tag):
            pieces = []
            for i in range(0, len(es), 100):
                chunk = es[i:i+100]
                raw = acquire(rawdir/f'{material}_{tag}_{i}.html', STAR+'e_table-t.pl',
                              dict(matno=item['matno'], Energies='\n'.join(format(e, '.17g') for e in chunk)))
                pieces.append(parse_stopping(raw, chunk))
            return {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        stop, check = table(nodes, 'stopping'), table(mid, 'check')
        stop.update({'check_'+k: v for k, v in check.items()})
        raw = acquire(rawdir/(material+'_range.html'), STAR+'e_table-t.pl', dict(matno=item['matno'], ShowDefault='on', Energies=''))
        ranges = parse_range(raw)
        for table in (stop, ranges):
            table.update(material=np.array(material), composition_sha256=np.array(composition_id(d)),
                         density_g_cm3=np.array(d['density_g_cm3']))
        np.savez(folder/'stopping.npz', **stop)
        np.savez(folder/'range.npz', **ranges)
        np.savez(folder/'photon.npz', **photon_table(cache/'xcom'/material, d), material=np.array(material),
                 composition_sha256=np.array(composition_id(d)))
        write(folder/'definition.json', d)
        # Keep the standard lung inputs for the independently checked S2 path.
        if material == 'lung':
            write(folder/'standard_definition.json', d)
            np.savez(folder/'standard_stopping.npz', **stop)
        names = ['definition.json', 'stopping.npz', 'range.npz', 'photon.npz']
        if material == 'lung':
            names += ['standard_definition.json', 'standard_stopping.npz']
        write(outputs_receipt, {name: sha(folder/name) for name in names})
    write(out/'xcom_responses.json', {str(p): sha(p) for p in sorted((cache/'xcom').rglob('*')) if p.is_file() and p.suffix in ('.txt', '.json', '.log')})


def photon_only(out=OUT, cache=CACHE):
    """Appendix 6: replace only photons; retain and seal all S1/S2/S4/S5 data.

    New XCOM requests use a separate directory: old requested energies and raw
    responses remain immutable. A journal seals the old hashes before work and
    all replacement hashes before installation, allowing interrupted installs
    to resume without treating an arbitrary modified photon table as trusted.
    """
    out, cache = Path(out), Path(cache)
    verify_protected()
    journal = out/'photon_migration_v5.json'
    work_root = cache/'xcom_appendix5'
    staged_manifest = out/'photon_migration_v5.manifest.json'
    manifest_path = out/'manifest.json'
    identity = dict(cache=str(cache.resolve()), materials=list(MATERIALS), implementations=implementation_hashes())
    second = out/'photon_migration_v9_round2.json'
    previous_migration = None
    if second.exists() or (journal.exists() and json.loads(journal.read_text())['identity'] != identity):
        # Appendix 9 authorizes a SECOND migration, not overwriting or relaxing
        # the first journal. Only the edge/refinement and checker edits may
        # differ; all other implementation hashes and identity fields stay fixed.
        previous = json.loads(journal.read_text())
        old_identity = previous['identity']
        before, after = old_identity['implementations'], identity['implementations']
        allowed = {'fetch_materials_p8a.py', 'check_p8a.py'}
        if (previous['status'] != 'complete' or set(before) != set(after)
                or {k: v for k, v in old_identity.items() if k != 'implementations'}
                != {k: v for k, v in identity.items() if k != 'implementations'}
                or any(before[k] != after[k] for k in before if k not in allowed)):
            raise ValueError('photon-only migration conditions changed beyond appendix 9')
        for name, expected in previous['preserved'].items():
            if sha(name) != expected:
                raise ValueError('photon-only previous preserved file changed: '+name)
        for target in previous['targets']:
            if sha(target['staged']) != target['after']:
                raise ValueError('photon-only previous staged output hash mismatch')
            if not second.exists() and sha(target['path']) != target['after']:
                raise ValueError('photon-only previous migration target hash mismatch')
        previous_migration = dict(path=str(journal), sha256=sha(journal))
        journal = second
        work_root = cache/'xcom_appendix9_round2'
        staged_manifest = out/'photon_migration_v9_round2.manifest.json'
    if journal.exists():
        record = json.loads(journal.read_text())
        if record['identity'] != identity or record.get('previous_migration') != previous_migration:
            raise ValueError('photon-only migration conditions changed')
    else:
        manifest = load_manifest(manifest_path, complete=False)
        preserved, receipts = {}, {}
        # Validate every material BEFORE any XCOM request or output change.
        for material in MATERIALS:
            folder = cache/material
            row = manifest['materials'][material]
            for field in ('definition', 'stopping', 'range', 'photon'):
                if Path(row[field]['path']).resolve() != (folder/(field+('.json' if field == 'definition' else '.npz'))).resolve():
                    raise ValueError('photon-only material path mismatch')
            for field, ref in row.items():
                if field != 'photon':
                    preserved[ref['path']] = ref['sha256']
            receipt = folder/'fetch_outputs.json'
            old = json.loads(receipt.read_text())
            if not {'definition.json', 'stopping.npz', 'range.npz', 'photon.npz'} <= set(old):
                raise ValueError('photon-only S1 receipt incomplete')
            selected = {}
            if material == 'lung':
                lung_receipt = folder/'lung_outputs.json'
                selected = json.loads(lung_receipt.read_text())
                if set(selected) != {'definition.json', 'stopping.npz'}:
                    raise ValueError('photon-only S2 receipt incomplete')
                preserved[str(lung_receipt)] = sha(lung_receipt)
                state = json.loads((out/'lung_state.json').read_text())
                if state['status'] != 'pass' or state['density_g_cm3'] != .26:
                    raise ValueError('photon-only requires completed S2 lung state')
                # S2's legacy receipt seals selected definition/stopping, but
                # not its report. Seal the report's current bytes here as well.
                preserved[str(out/'lung_state.json')] = sha(out/'lung_state.json')
            for name, expected in dict(old, **selected).items():
                if sha(folder/name) != expected:
                    raise ValueError('photon-only S1/S2 saved output hash mismatch')
                if name != 'photon.npz':
                    preserved[str(folder/name)] = expected
            if old['photon.npz'] != row['photon']['sha256']:
                raise ValueError('photon-only photon receipt / manifest mismatch')
            receipts[material] = dict(path=str(receipt), sha256=sha(receipt), values=old)
        record = dict(identity=identity, reason='P8a appendix 5 (1): acquire upper absorption-edge values and check jumps',
                      status='pending', original_manifest=manifest, original_manifest_sha256=sha(manifest_path),
                      preserved=preserved, original_receipts=receipts, photons={}, targets=[])
        if previous_migration is not None:
            record.update(round=2, previous_migration=previous_migration,
                          reason='P8a appendix 9: refine the 10% below all absorption edges <=20 keV')
        write(journal, record)

    def verify_unchanged():
        for name, expected in record['preserved'].items():
            if sha(name) != expected:
                raise ValueError('photon-only preserved file changed: '+name)
        targets = {t['path']: t for t in record['targets']}
        existing = {str(manifest_path): record['original_manifest_sha256']}
        for material, receipt in record['original_receipts'].items():
            existing[receipt['path']] = receipt['sha256']
            ref = record['original_manifest']['materials'][material]['photon']
            existing[ref['path']] = ref['sha256']
        for name, expected in existing.items():
            allowed = {expected}
            if name in targets:
                allowed.add(targets[name]['after'])
            if sha(name) not in allowed:
                raise ValueError('photon-only migration target hash mismatch: '+name)
        for t in record['targets']:
            if sha(t['staged']) != t['after']:
                raise ValueError('photon-only staged output hash mismatch')

    verify_unchanged()
    if record['status'] == 'complete':
        if any(sha(t['path']) != t['after'] for t in record['targets']):
            raise ValueError('photon-only completed migration changed')
        load_manifest(manifest_path, complete=False)
        return record
    for material in MATERIALS:
        work = work_root/material
        if material not in record['photons']:
            d = json.loads(Path(record['original_manifest']['materials'][material]['definition']['path']).read_text())
            resume(work/'photon_identity.json', dict(definition=d, implementations=identity['implementations']))
            staged = work/'photon.npz'
            seal = work/'photon.sha.json'
            if staged.exists():
                if not seal.exists() or sha(staged) != json.loads(seal.read_text())['sha256']:
                    raise ValueError('photon-only staged output hash mismatch')
            else:
                if seal.exists():
                    raise ValueError('photon-only staged output missing')
                np.savez(staged, **photon_table(work, d), material=np.array(material),
                         composition_sha256=np.array(composition_id(d)))
                write(seal, dict(sha256=sha(staged)))
            method = 'fetch_materials_p8a --photon-only; '+('appendix 9, round 2' if previous_migration else 'appendix 5 (1)')
            new = reference(staged, d['sources'], method, 'MeV; cm2/g', material)
            old = record['original_manifest']['materials'][material]['photon']
            new['path'] = old['path']
            record['photons'][material] = dict(before=old['sha256'], after=new['sha256'], reference=new, staged=str(staged))
            write(journal, record)
        elif sha(record['photons'][material]['staged']) != record['photons'][material]['after']:
            raise ValueError('photon-only staged output hash mismatch')
    if not record['targets']:
        updated = json.loads(json.dumps(record['original_manifest']))
        for material, change in record['photons'].items():
            updated['materials'][material]['photon'] = change['reference']
            record['targets'].append(dict(path=change['reference']['path'], before=change['before'],
                                          after=change['after'], staged=change['staged']))
            receipt = record['original_receipts'][material]
            staged = work_root/material/'fetch_outputs.v5.json'
            write(staged, dict(receipt['values'], **{'photon.npz': change['after']}))
            record['targets'].append(dict(path=receipt['path'], before=receipt['sha256'], after=sha(staged), staged=str(staged)))
        staged = staged_manifest
        write(staged, updated)
        record['targets'].append(dict(path=str(manifest_path), before=record['original_manifest_sha256'], after=sha(staged), staged=str(staged)))
        record['status'] = 'ready'
        write(journal, record)
    verify_unchanged()
    for target in record['targets']:
        path = Path(target['path'])
        if sha(path) != target['after']:
            tmp = path.with_suffix(path.suffix+'.photon-v5.tmp')
            shutil.copyfile(target['staged'], tmp)
            tmp.replace(path)
    verify_unchanged()
    load_manifest(manifest_path, complete=False)
    record['status'] = 'complete'
    write(journal, record)
    verify_protected()
    return record


def lung_stopping_a(estar, standard, selected):
    """Appendix 3: anchor to ESTAR; use PEGS only for the density change.

    Collision stopping comes from the ratio, not from collision(..., delta).
    Delta is a separately recorded additive correction; radiative stopping is
    unchanged. Evaluate this same recipe on nodes and independent check points.
    """
    es = energy_grid(estar['T'])
    for data in (standard, selected):
        if not np.array_equal(finite(data['T']), es):
            raise ValueError('S2 A PEGS/ESTAR energy grids differ')
    arrays = [finite(data[key], key) for data, key in
              ((estar, 'collision'), (estar, 'radiative'), (estar, 'delta'),
               (standard, 'collision'), (selected, 'collision'),
               (standard, 'delta'), (selected, 'delta'))]
    if any(a.shape != es.shape for a in arrays) or any(np.any(a <= 0) for a in (arrays[0], arrays[1], arrays[3], arrays[4])):
        raise ValueError('S2 A stopping shape / positivity')
    coll, rad, delta, std_coll, low_coll, std_delta, low_delta = arrays
    ratio = low_coll/std_coll
    return dict(T=es, collision=coll*ratio, radiative=rad.copy(),
                delta=delta+(low_delta-std_delta),
                standard_ratio=std_coll/std_coll, density_ratio=ratio)


def check_lung_stopping_a(estar, corrected):
    """The two S2 gates in appendix 3; PEGS/ESTAR agreement is report-only."""
    es = energy_grid(estar['T'])
    ratio = finite(corrected['standard_ratio'])
    actual, reference = finite(corrected['collision']), finite(estar['collision'])
    if (not np.array_equal(corrected['T'], es) or any(a.shape != es.shape for a in (ratio, actual, reference))
            or np.any(reference <= 0) or np.any(actual <= 0)):
        raise ValueError('S2 A check shape / positivity / energy grid')
    low = es <= .2
    if not np.any(low):
        raise ValueError('S2 A low-energy check points missing')
    relative = (actual[low]-reference[low])/reference[low]
    exact = bool(np.all(ratio == 1.))
    maximum = float(np.max(abs(relative)))
    return dict(status='pass' if exact and maximum <= .001 else 'fail',
                standard_ratio_exactly_one=exact, standard_ratio=ratio.tolist(),
                low_energy_limit_MeV=.2, low_energy_relative_limit=.001,
                low_energy_T_MeV=es[low].tolist(), low_energy_relative=relative.tolist(),
                low_energy_max_relative=maximum)


def lung_state(cache=CACHE, out=OUT):
    """User-selected A: ESTAR baseline times PEGS density ratio (appendix 3)."""
    from .pegs_p8a import density_stopping
    folder = cache/'lung'
    s1 = json.loads((folder/'fetch_outputs.json').read_text())
    for name in ('standard_definition.json', 'standard_stopping.npz'):
        if sha(folder/name) != s1[name]:
            raise ValueError('S2 standard-density lung source changed')
    receipt = folder/'lung_outputs.json'
    if receipt.exists():
        for name, expected in json.loads(receipt.read_text()).items():
            if sha(folder/name) != expected:
                raise ValueError('S2 selected lung output changed')
    d = json.loads((folder/'standard_definition.json').read_text())
    with np.load(folder/'standard_stopping.npz') as f:
        old = {k: f[k].copy() for k in f.files}
    low = dict(d, density_g_cm3=.26,
               stopping_method='estar_pegs_density_ratio_A',
               density_method='ESTAR delta + PEGS5 delta(0.26) - delta(standard)')
    # Use direct PEGS collision stopping at BOTH densities, with the same I
    # convention inside PEGS. Do not substitute our NIST-I collision formula.
    standard = density_stopping(d, old['T'], cache/'pegs_density_standard')
    selected = density_stopping(low, old['T'], cache/'pegs_density_lung')
    nodes = lung_stopping_a(old, standard, selected)
    # Neither density's off-grid values are interpolated from the node data.
    check_estar = {k: old['check_'+k] for k in ('T', 'collision', 'radiative', 'delta')}
    check_standard = density_stopping(d, old['check_T'], cache/'pegs_density_standard_check')
    check_selected = density_stopping(low, old['check_T'], cache/'pegs_density_lung_check')
    held_out = lung_stopping_a(check_estar, check_standard, check_selected)
    checks = dict(nodes=check_lung_stopping_a(old, nodes),
                  held_out=check_lung_stopping_a(check_estar, held_out))
    rel = standard['collision']/old['collision']-1
    check_rel = check_standard['collision']/old['check_collision']-1
    result = dict(method='A: ESTAR standard collision * PEGS5 selected/standard collision',
                  density_g_cm3=.26, standard_density_g_cm3=d['density_g_cm3'],
                  delta_method=low['density_method'], radiative_method='unchanged ESTAR',
                  T_MeV=old['T'].tolist(), check_T_MeV=old['check_T'].tolist(),
                  standard_collision_max_relative=float(np.max(abs(rel))),
                  standard_collision_relative=rel.tolist(),
                  check_standard_collision_max_relative=float(np.max(abs(check_rel))),
                  check_standard_collision_relative=check_rel.tolist(),
                  standard_pegs_estar_comparison='report-only; not an S2 gate',
                  pegs_density_collision_ratio=nodes['density_ratio'].tolist(),
                  check_pegs_density_collision_ratio=held_out['density_ratio'].tolist(),
                  selected_over_standard_collision=(nodes['density_ratio']-1).tolist(),
                  checks=checks,
                  status='pass' if all(c['status'] == 'pass' for c in checks.values()) else 'fail',
                  requires_user_density_confirmation=False)
    write(out/'lung_state.json', result)
    if result['status'] != 'pass':
        raise ValueError('S2 A checks failed: standard ratio or low-energy stopping')
    stop = dict(old)
    stop['density_g_cm3'] = np.array(.26)
    for prefix, values in (('', nodes), ('check_', held_out)):
        for key in ('collision', 'radiative', 'delta'):
            stop[prefix+key] = values[key]
        stop[prefix+'total'] = values['collision']+values['radiative']
    write(folder/'definition.json', low)
    np.savez(folder/'stopping.npz', **stop)
    write(receipt, {name: sha(folder/name) for name in ('definition.json', 'stopping.npz')})


def main():
    p = argparse.ArgumentParser(__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--lung', action='store_true')
    group.add_argument('--photon-only', action='store_true')
    args = p.parse_args()
    verify_protected()
    if args.photon_only:
        photon_only()
    elif args.lung:
        lung_state()
    else:
        fetch()


if __name__ == '__main__':
    main()
