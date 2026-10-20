"""P8a material definitions. No transport changes or implicit data generation.

Number densities are per gram; multiply by rho only for macroscopic rates.
Photoelectric edges use the upper-side value at the exact edge.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from .physics import M, RE, NA

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/validation/mv/p8a'
CACHE = Path.home() / '.cache/chatcarlo/p8a'
MATERIALS = ('water', 'lung', 'bone', 'lead')
FIELDS = ('definition', 'constants', 'stopping', 'range', 'photon', 'elastic',
          'gs', 'pair', 'shells')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def implementation_hashes():
    names = ('materials_p8a.py', 'fetch_materials_p8a.py', 'generate_elsepa_p8a.py',
             'pegs_p8a.py', 'check_p8a.py', 'estimate_p8a.py', 'p8a_egs.f',
             'gs_backend.py', 'single_table.py', 'dcs.py', 'physics.py')
    return {name: sha(Path(__file__).with_name(name)) for name in names}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False,
                                default=lambda o: o.item() if hasattr(o, 'item') else o.tolist()) + '\n')
    tmp.replace(path)


def finite(value, name='array'):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError('nonfinite ' + name)
    return a


def energy_grid(value):
    a = finite(value, 'energy')
    if a.ndim != 1 or len(a) < 2 or np.any(a <= 0) or np.any(np.diff(a) <= 0):
        raise ValueError('energy must be positive and strictly increasing')
    return a


def composition(weights, atomic_weights):
    if not weights or set(map(str, weights)) != set(map(str, atomic_weights)):
        raise ValueError('composition / atomic weights mismatch')
    zs = sorted(int(z) for z in weights)
    if len(set(zs)) != len(weights) or any(z < 1 or z > 100 for z in zs):
        raise ValueError('invalid atomic number')
    w = finite([weights.get(str(z), weights.get(z)) for z in zs], 'composition')
    a = finite([atomic_weights.get(str(z), atomic_weights.get(z)) for z in zs])
    if np.any(w <= 0) or np.any(a <= 0) or abs(w.sum() - 1) > 1e-6:
        raise ValueError('composition must sum to one')
    nj = NA * w / a
    return np.array(zs), nj, float(np.dot(zs, nj))


def constants(definition):
    rho, excitation = finite([definition['density_g_cm3'], definition['I_MeV']])
    if rho <= 0 or excitation <= 0:
        raise ValueError('invalid density or excitation energy')
    z, nj, ne = composition(definition['weights'], definition['atomic_weights'])
    return dict(Z=z.tolist(), NJ=nj.tolist(), NE=ne, I_MeV=excitation,
                density_g_cm3=rho, number_density_units='1/g')


def composition_id(definition):
    payload = {str(k): float(v) for k, v in definition['weights'].items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def collision(T, ne, excitation, delta):
    """Unrestricted Berger/Seltzer equation, same algebra as physics.collision."""
    T = finite(T)
    if np.any(T <= 0):
        raise ValueError('nonpositive kinetic energy')
    tau = T / M
    g = 1 + tau
    b2 = 1 - 1 / g**2
    d = tau / 2
    f = (-1 - b2 + np.log((tau-d)*d) + tau/(tau-d)
         + (d*d/2 + (2*tau+1)*np.log1p(-d/tau))/g**2)
    return 2*np.pi*RE**2*M*ne/b2*(np.log(2*(tau+2)/(excitation/M)**2)+f-delta)


def kn_total(E, ne):
    x = finite(E).astype(np.longdouble) / np.longdouble(M)
    if np.any(x <= 0) or not np.isfinite(ne) or ne <= 0:
        raise ValueError('invalid Klein-Nishina arguments')
    l = np.log1p(2*x)
    bracket = 2*((1+x)/x**2*(2*(1+x)/(1+2*x)-l/x)
                 + l/(2*x)-(1+3*x)/(1+2*x)**2)
    # Integrate the convergent q=1/(1+x*(1-mu)) series at small x.
    # Avoid low-energy cancellation even on platforms where longdouble is
    # float64. Coefficients follow KN angular integration, not tabulated data.
    low = np.minimum(x, .05)
    series = np.zeros_like(x)
    for n in range(24):
        coefficient = (-1)**n*(((n+1)*(n+2)/2+1)*2**(n+1)/(n+1)
            - (n+1)*(2*2**(n+2)/(n+2)-2**(n+3)/(n+3)))
        series += coefficient*low**n
    bracket = np.where(x < .05, series, bracket)
    return np.asarray(np.longdouble(str(np.pi))*np.longdouble(RE)**2*ne*bracket, dtype=float)


def log_interp(E, es, values):
    es = energy_grid(es)
    values = finite(values)
    E = finite(E)
    if values.shape != es.shape or np.any(values <= 0):
        raise ValueError('invalid positive interpolation table')
    if np.any(E < es[0]) or np.any(E > es[-1]):
        raise ValueError('energy outside table range')
    return np.exp(np.interp(np.log(E), np.log(es), np.log(values)))


def photo_interp(E, es, values):
    """Duplicate energies encode [below, above] at an absorption edge.

    Interpolate only between adjacent distinct-energy sides. At the exact
    duplicated energy searchsorted(right) selects the above-edge node.
    """
    es, values = finite(es), finite(values)
    if es.ndim != 1 or len(es) < 2 or values.shape != es.shape:
        raise ValueError('photoelectric shape')
    d = np.diff(es)
    if np.any(es <= 0) or np.any(values <= 0) or np.any(d < 0):
        raise ValueError('invalid photoelectric grid')
    if np.any((d[:-1] == 0) & (d[1:] == 0)):
        raise ValueError('more than two edge sides')
    E = finite(E)
    if np.any(E < es[0]) or np.any(E > es[-1]):
        raise ValueError('energy outside photoelectric range')
    i = np.searchsorted(es, E, side='right') - 1
    j = np.minimum(i+1, len(es)-1)
    same = i == j
    den = np.where(same, 1., np.log(es[j]/es[i]))
    f = np.log(E/es[i])/den
    return np.exp(np.log(values[i]) + f*np.log(values[j]/values[i]))


def pair_interp(E, es, values, threshold):
    es = energy_grid(es)
    vs = finite(values)
    if vs.shape != es.shape or threshold <= 0 or not np.isfinite(threshold):
        raise ValueError('pair table shape / threshold')
    if np.any(vs[es <= threshold] != 0) or np.any(vs[es > threshold] <= 0):
        raise ValueError('pair threshold values')
    mask = es > threshold
    if mask.sum() < 2:
        raise ValueError('missing positive pair nodes')
    E = finite(E)
    if np.any(E < es[0]) or np.any(E > es[-1]):
        raise ValueError('energy outside pair range')
    reduced = vs[mask]/(1-threshold/es[mask])**3
    out = np.exp(np.interp(np.log(E), np.log(es[mask]), np.log(reduced)))
    return np.where(E <= threshold, 0., out*np.maximum(0., 1-threshold/E)**3)


def photon_coefficients(E, definition, table):
    """Return KN, photoelectric, nuclear pair, triplet in cm2/g."""
    c = constants(definition)
    return np.stack([kn_total(E, c['NE']),
                     photo_interp(E, table['photo_E'], table['photo']),
                     *[pair_interp(E, table['E'], table[key], threshold)
                       for key, threshold in zip(('pair_nuclear', 'pair_electron'), table['thresholds'])]], axis=-1)


class MixtureElsepa:
    """Composition-aware provider implementing the existing GS provider API."""
    name = 'elsepa'

    def __init__(self, definition, tables, charge=-1, legacy_water=False):
        from .dcs import Dcslib
        self._spline_method = Dcslib.spline.__get__(self)
        self.atomic, self.raw = {}, {}
        c = constants(definition)
        z, nj = np.asarray(c['Z']), np.asarray(c['NJ'])
        if legacy_water:
            from .physics import Z, NJ
            if definition['material'] != 'water' or c['density_g_cm3'] != 1. or not np.array_equal(z, Z):
                raise ValueError('legacy water identity')
            nj = NJ
        for element in z:
            a = tables[int(element)]
            if int(a['Z']) != element or int(a['charge']) != charge or int(a['IHEF']) != 0:
                raise ValueError('ELSEPA identity / IHEF')
            e = energy_grid(a['energy_MeV'])
            mu, ds = finite(a['mu']), finite(a['dcs'])
            if mu.ndim != 1 or mu[0] != 0 or mu[-1] != 1 or np.any(np.diff(mu) <= 0):
                raise ValueError('ELSEPA angular grid')
            if ds.shape != (len(e), len(mu)) or np.any(ds <= 0):
                raise ValueError('ELSEPA DCS shape / positivity')
            if self.atomic and (not np.array_equal(self.ts, e) or not np.array_equal(self.rmu, mu)):
                raise ValueError('ELSEPA grids differ')
            self.ts, self.rmu = e, mu
            self.atomic[int(element)] = (e, ds, a.get('table_xs'))
            self.raw[int(element)] = a
        self.energy_min, self.energy_max = max(.01, self.ts[0]), min(10., self.ts[-1])
        self.native_energies = self.ts[(self.ts >= self.energy_min) & (self.ts <= self.energy_max)]
        # Deterministic sorted summation; legacy scale preserves water roundoff.
        scale_per_gram = nj[1] if legacy_water else float(nj.sum())
        self.scale = scale_per_gram*c['density_g_cm3']
        stoich = nj / scale_per_gram
        combined = sum(stoich[j]*(v+1)/v*self.atomic[int(v)][1] for j, v in enumerate(z))
        self.combined = CubicSpline(np.log(self.ts), np.log(combined), axis=0, bc_type='natural')
        self.atomic_splines = {v: CubicSpline(np.log(e), np.log(ds), axis=0, bc_type='natural')
                               for v, (e, ds, _) in self.atomic.items()}

    def validate_energy(self, T):
        if not self.energy_min <= T <= self.energy_max:
            raise ValueError('ELSEPA supported energy range')

    def spline(self, T, z=None):
        self.validate_energy(T)
        return self._spline_method(T, z)

    def density(self, T, y, z=None):
        return 2*np.pi*np.exp(self.spline(float(T), z)(np.asarray(y)/2))*(self.scale if z is None else 1.)

    def edges(self, T, z=None):
        self.validate_energy(T)
        return 2*self.rmu

    def quadrature(self, T, order=16, z=None):
        from .dcs import Dcslib
        return Dcslib.quadrature(self, T, order, z)


def reference(path, source, method, units, material, **extra):
    path = Path(path).resolve()
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as a:
            shape = {k: list(a[k].shape) for k in a.files}
    else:
        shape = []
    return dict(path=str(path), sha256=sha(path), source=source, method=method,
                units=units, shape=shape, material=material, **extra)


def load_reference(ref, material):
    if ref['material'] != material:
        raise ValueError('material identity mismatch')
    if not all(ref.get(k) is not None for k in ('source', 'method', 'units', 'shape', 'sha256')):
        raise ValueError('incomplete provenance')
    p = Path(ref['path'])
    if sha(p) != ref['sha256']:
        raise ValueError('reference hash mismatch')
    if p.suffix == '.npz':
        with np.load(p, allow_pickle=False) as a:
            data = {k: a[k].copy() for k in a.files}
        if 'material' in data and str(data['material']) != material:
            raise ValueError('embedded material identity mismatch')
        if {k: list(v.shape) for k, v in data.items()} != ref['shape']:
            raise ValueError('reference shape mismatch')
        for key, value in data.items():
            if value.dtype.kind in 'fci':
                finite(value, key)
        return data
    return json.loads(p.read_text())


def resume(path, identity):
    """Persist identity before work; an interrupted run cannot change conditions."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('generation conditions changed on resume')
    else:
        write(path, identity)


def resume_appendix6(path, identity, stage, evidence=None):
    """Authorize only the appendix 5/6 script changes, retaining old identities.

    The generator is included solely for its appendix 4/6 resume wrapper. The
    remaining scientific helpers, key sets and ALL nonimplementation conditions
    must match. The first migration seals both hash sets; subsequent invocations
    must match that exact migration, not accept another rolling code change.
    """
    path = Path(path)
    if not path.exists():
        return resume(path, identity)
    old = json.loads(path.read_text())
    receipt = path.with_suffix('.resume_v6.json')
    if old == identity and not receipt.exists():
        return
    allowed = {'fetch_materials_p8a.py', 'check_p8a.py', 'pegs_p8a.py',
               'materials_p8a.py', 'generate_elsepa_p8a.py'}
    before, after = old['implementations'], identity['implementations']
    changed = {k for k in before if before[k] != after.get(k)}
    if (set(before) != set(after) or not changed <= allowed
            or {k: v for k, v in old.items() if k != 'implementations'}
            != {k: v for k, v in identity.items() if k != 'implementations'}):
        raise ValueError('generation conditions changed beyond appendix 6')
    prior = path.with_suffix('.resume_v4.json')
    prior_sha = None
    if prior.exists():
        record = json.loads(prior.read_text())
        previous = record['current_identity']
        if (record['original_sha256'] != sha(path)
                or {k: v for k, v in previous.items() if k != 'implementations'}
                != {k: v for k, v in old.items() if k != 'implementations'}
                or set(previous['implementations']) != set(before)
                or any(previous['implementations'][k] != after[k] for k in before if k not in allowed)):
            raise ValueError('appendix 4 migration changed / conditions mismatch')
        prior_sha = sha(prior)
    resume(receipt, dict(stage=stage, authorization='P8a appendix 6: apply appendix 5 without regenerating S4/S5',
                         original_sha256=sha(path), previous_v4_sha256=prior_sha,
                         current_identity=identity, changes={k: dict(before=before[k], after=after[k]) for k in sorted(changed)},
                         evidence={} if evidence is None else evidence))


def verify_protected(path=OUT/'protected_start.json'):
    old = json.loads(Path(path).read_text())
    for name, expected in old.items():
        if sha(ROOT/name) != expected:
            raise ValueError('protected file changed: ' + name)
    return len(old)


def build(out=OUT, cache=CACHE):
    """Build references after S1/S2. Later S4/S5 fill their own entries."""
    verify_protected()
    manifest_path = out/'manifest.json'
    manifest = load_manifest(manifest_path, complete=False) if manifest_path.exists() else {'version': 1, 'materials': {}}
    for material in MATERIALS:
        folder = cache/material
        d = json.loads((folder/'definition.json').read_text())
        if d['material'] != material:
            raise ValueError('definition material mismatch')
        c = constants(d)
        write(folder/'constants.json', c)
        row = manifest['materials'].setdefault(material, {})
        if material == 'lung' and (d['density_g_cm3'] != .26 or json.loads((out/'lung_state.json').read_text())['status'] != 'pass'):
            raise ValueError('S2 lung state must be completed before build')
        for field, units in [('definition', 'g/cm3; MeV; mass fraction'), ('constants', '1/g; MeV; g/cm3'),
                             ('stopping', 'MeV; MeV cm2/g; dimensionless'), ('range', 'MeV; g/cm2'),
                             ('photon', 'MeV; cm2/g')]:
            p = folder/(field + ('.json' if field in ('definition', 'constants') else '.npz'))
            method = ('P8a composition-based' if field == 'constants' else
                      'ESTAR standard-density separate CSDA response' if field == 'range' else 'fetch_materials_p8a')
            row[field] = reference(p, d['sources'], method, units, material)
    write(manifest_path, manifest)
    return manifest


def load_manifest(path=OUT/'manifest.json', complete=True):
    m = json.loads(Path(path).read_text())
    if set(m['materials']) != set(MATERIALS):
        raise ValueError('missing material')
    for material, row in m['materials'].items():
        if complete and set(row) != set(FIELDS):
            raise ValueError('missing material tables: ' + material)
        for ref in row.values():
            load_reference(ref, material)
        d = load_reference(row['definition'], material)
        if d['material'] != material:
            raise ValueError('definition material identity')
        constants(d)
        for field in ('stopping', 'photon', 'range'):
            if field in row:
                data = load_reference(row[field], material)
                if str(data.get('material', '')) != material or str(data.get('composition_sha256', '')) != composition_id(d):
                    raise ValueError('table material/composition identity mismatch')
        if 'constants' in row:
            actual = load_reference(row['constants'], material)
            if actual != constants(d):
                raise ValueError('material constants disagree with composition')
        if 'stopping' in row:
            s = load_reference(row['stopping'], material)
            es = energy_grid(s['T'])
            for key in ('collision', 'radiative', 'total', 'delta'):
                # Appendix 3's A records an additive delta correction, not
                # the delta used to recompute collision stopping. Preserve its
                # signed result without clipping; all values remain finite.
                signed_delta = (key == 'delta' and material == 'lung'
                                and d.get('stopping_method') == 'estar_pegs_density_ratio_A')
                if s[key].shape != es.shape or (not signed_delta and np.any(s[key] < 0)):
                    raise ValueError('invalid stopping shape or values')
        if 'range' in row:
            r = load_reference(row['range'], material)
            es = energy_grid(r['T'])
            if r['csda_g_cm2'].shape != es.shape or np.any(r['csda_g_cm2'] <= 0) or np.any(np.diff(r['csda_g_cm2']) <= 0):
                raise ValueError('invalid CSDA range')
        if 'elastic' in row:
            refs = load_reference(row['elastic'], material)
            expected = {f'z{z}_q{q}.npz' for z in d['weights'] for q in (-1, 1)}
            if set(refs) != expected:
                raise ValueError('missing elemental table')
            for name, ref in refs.items():
                atomic = load_reference(ref, 'element')
                if int(atomic['Z']) != ref['Z'] or int(atomic['charge']) != ref['charge'] or name != f'z{int(atomic["Z"])}_q{int(atomic["charge"])}.npz':
                    raise ValueError('elemental table identity')
        if 'gs' in row:
            g = load_reference(row['gs'], material)
            ts, ks, u = energy_grid(g['T']), energy_grid(g['K']), finite(g['u'])
            if u.ndim != 1 or len(u) < 2 or u[0] != 0 or u[-1] != 1 or np.any(np.diff(u) <= 0):
                raise ValueError('GS probability grid')
            if g['y'].shape != (len(ts), len(ks), len(u)) or g['p0'].shape != (len(ts), len(ks)):
                raise ValueError('GS table shape')
            if np.any(g['y'] < 0) or np.any(g['y'] > 2) or np.any(g['p0'] < 0) or np.any(g['p0'] > 1):
                raise ValueError('GS table bounds')
    return m


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('command', choices=['build'])
    p.parse_args()
    build()


if __name__ == '__main__':
    main()


def legacy_water(photon_path=None, gs_path=None, elastic_directory=None):
    """Read the actual water path without regenerating a single array/constant."""
    from . import physics
    from .photon_data_p2 import TABLE
    from .p1f_common import WORK
    def arrays(path):
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k].copy() for k in f.files}
    d = physics.DATA
    return dict(NE=physics.NE, NJ=physics.NJ.copy(), I=physics.I,
                stopping=arrays(d/'estar.npz'),
                photon=arrays(TABLE if photon_path is None else photon_path),
                elastic={f'z{z}_q{q}': arrays((WORK/'tables' if elastic_directory is None else Path(elastic_directory))/f'z{z}_q{q}.npz')
                         for z in (1, 8) for q in (-1, 1)},
                gs=arrays(WORK/'gs.npz' if gs_path is None else gs_path))
