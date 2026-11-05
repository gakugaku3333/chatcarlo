"""H6 B1--B5: freeze small CPU fixtures before exporting them to one T4.

The two B1 reach counters observe copied/in-memory production source only. They
change neither sampling nor scores; CUDA counters always point to global memory.
No work is performed on import.
"""
import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(HERE)]
SEEDS = dict(B1=9504005, B2=20301831, B3=9502101, B4=9504002, B5=9504003)
LOW_N = 256
SEED_N = 2000


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                              default=lambda x: x.tolist() if hasattr(x, 'tolist') else x.item())+'\n')


def diagnostic_arrays():
    import numpy as np
    from chatcarlo.gpu import core
    from chatcarlo.kernel import bake_scene_materials
    a = core.pack(bake_scene_materials(['air', 'lead']))
    a.update(centers=np.array([[0., 0., 1.]]), halves=np.array([[1., 1., .1]]),
             materials=np.array([1], np.int64), background=np.array(0),
             world=np.array([0., 0., 0., 50., 50., 50.]), detector=np.array([]))
    return a


# Observation-only builds; existing hits[73:75] are caller-owned global
# buffers on CUDA. Pass that pointer through the Compton function explicitly.
REJECTION = 'if(rng_uniform(s)>g5_bound(t->photon,element,E*1000,mu))continue;'
COUNTER = r"""
#ifdef P5_GPU
if(h6_counts)atomicAdd(h6_counts+74,1ULL);
#else
if(h6_counts)++h6_counts[74];
#endif
"""


def observe_text(text, name):
    if name == 'photon_physics.h':
        signature = 'P5_FN void p5_compton(double E,double ux,double uy,double uz,const P5Tables *t,RngState *s,double *out)'
        if text.count(REJECTION) != 1 or text.count(signature) != 1:
            raise ValueError('B1 rejection observer site changed')
        text = text.replace(signature, signature[:-1]+',unsigned long long *h6_counts=0)')
        return text.replace(REJECTION, 'if(rng_uniform(s)>g5_bound(t->photon,element,E*1000,mu)){'+COUNTER+'continue;}')
    if name == 't2_kernel.h':
        site = 'if(process==4){double v[3];'
        call = 'p5_compton(E,a[3],a[4],a[5],t,s,p);'
        if text.count(site) != 1 or text.count(call) != 1:
            raise ValueError('B1 transport observer site changed')
        return text.replace(site, 'if(process==4){p8g2_hit(hits,73);double v[3];').replace(call, call[:-2]+',hits);')
    return text


def therapy_driver(device, work, materials, arrays, *, observe=False, **kw):
    from chatcarlo.gpu import therapy
    source = Path(work)/'observed_source'
    if observe:
        source.mkdir(parents=True, exist_ok=True)
        for p in therapy.HERE.iterdir():
            if p.suffix in ('.h', '.cpp', '.cu'):
                (source/p.name).write_text(observe_text(p.read_text(), p.name))
    if device == 'cpu':
        if observe:
            with patch.object(therapy, 'HERE', source): driver = therapy.CPU(work, materials, arrays, **kw)
        else: driver = therapy.CPU(work, materials, arrays, **kw)
        def run(n, seed, energy=1., **args):
            driver.g.energy = energy
            return driver.histories(n, seed, slots=3, **args)
    else:
        import cupy as cp
        from chatcarlo.gpu.therapy_cuda import T4
        if observe:
            with patch.object(therapy, 'HERE', source): expanded = therapy.cuda_source()
            with patch.object(therapy, 'cuda_source', lambda: expanded): driver = T4(cp, materials, arrays, check=1, **kw)
        else: driver = T4(cp, materials, arrays, check=1, **kw)
        def run(n, seed, energy=1., **args):
            return driver.run(n, seed, energy=energy, history=1, **args)
    return driver, run


def load_inputs(path):
    import numpy as np
    from chatcarlo.gpu import catalog
    with np.load(Path(path)/'therapy.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(Path(path)/'materials.npz', allow_pickle=False) as z:
        materials = [SimpleNamespace(**{k: z[f'{i}_{k}'] for k in catalog.P8_DEFINITIONS}) for i in range(5)]
        densities = z['densities']
    return materials, arrays, densities


def freeze_inputs(path, p8g2):
    """Read sealed archives, never regenerate tables or write to their caches."""
    import numpy as np
    from chatcarlo.gpu import catalog
    from prototypes.mv_electron import p9a_data
    import a2_common
    _, _, arrays = a2_common.inputs(p8g2)
    arrays = a2_common.no_spectrum(arrays)
    materials, identity = p9a_data.load_bundles()
    from chatcarlo.gpu.generate_photon_tables import ensure
    materials = [SimpleNamespace(**{k: ensure(m) if k == 'photon' else getattr(m, k)
                                   for k in catalog.P8_DEFINITIONS}) for m in materials]
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    np.savez(path/'therapy.npz', **arrays)
    np.savez_compressed(path/'materials.npz', **{f'{i}_{k}': getattr(m, k) for i, m in enumerate(materials)
                                   for k in catalog.P8_DEFINITIONS}, densities=identity['densities'])
    from prototypes.mv_electron.gpu.p8g2_core import fixed_cases
    write(path/'inputs.json', dict(seeds=SEEDS,
                                  B2_cases=[c for c in fixed_cases() if c['name'] in ('photon-layers', 'rest', 'flight', 'brems')],
                                  B2_pair_n=1024, B4_shell=1, B4_n=6000, B4_all_shell_n=100,
                                  B5_n=96, B5_energy_keV=100., B5_two_photon_seed=9504004,
                                  low_energies_MeV=[.02, .06], low_n=LOW_N,
                                  seed_n=SEED_N, source=identity))


def require_positive(counts):
    if not counts or any(int(v) <= 0 for v in counts.values()):
        raise AssertionError('registered branch not reached: '+str(counts))


def low(device, work, materials, arrays):
    import numpy as np
    from a2_gates import batch_health
    # Inject inside the water layer; include a below-cutoff photon explicitly.
    a = dict(arrays); a['faces'] = np.array([0., 30.]); a['rho'] = np.ones(1)
    a['ids'] = np.zeros(1, np.int64); a['sizes'] = arrays['sizes'].copy(); a['sizes'][0] = 1
    a['bins'] = np.arange(301, dtype=float)*.1
    a['owner'] = np.zeros(300, np.int64); a['sizes'] = np.array([1, 300, 150, 600, 0], np.int64)
    a['mass_g_cm2'] = np.diff(a['bins'])
    _, run = therapy_driver(device, work, materials, a, observe=True)
    rows = {}
    for energy in (.02, .06):
        states = np.zeros((LOW_N, 23)); states[:, 2] = 1.; states[:, 5] = 1.; states[:, 6] = energy
        r = run(LOW_N, SEEDS['B1'], energy=energy, injected=states)
        batch_health(r, a, None)
        reach = dict(rayleigh=int(r['hits'][73]), bound_rejection=int(r['hits'][74]))
        cutoff = states[:1].copy(); cutoff[:, 6] = .007
        cut = run(1, SEEDS['B1'], energy=.007, injected=cutoff)
        batch_health(cut, a, None)
        reach['cutoff_local'] = int(cut['hits'][10])
        require_positive(reach); rows[str(energy)] = dict(status='pass', reached=reach)
    return dict(status='pass', energies=rows)


def high(device, work, materials, arrays):
    import numpy as np
    from prototypes.mv_electron.gpu import p8g2_core as old
    from a2_gates import batch_health
    _, run = therapy_driver(device, work, materials, arrays)
    cases = old.fixed_cases(); rows = {}; pair = 0
    for name in ('photon-layers', 'rest', 'flight', 'brems'):
        case = next(c for c in cases if c['name'] == name)
        n = 1024 if name == 'photon-layers' else 1
        r = run(n, SEEDS['B2'], start=case['start'], injected=np.tile(case['state'], (n, 1)), energy=case['state'][6])
        batch_health(r, arrays, None)
        # Native count layout: violations[3], then counts[16].
        pair += int(r['group_ints'][:, 13:15].sum())
        reach = {old.PATHS[t]: int(r['hits'][t]) for t in case['targets']}
        require_positive(reach); rows[name] = reach
    require_positive(dict(pair_creation=pair))
    return dict(status='pass', reached=dict(pair_creation=pair), cases=rows)


def seed(device, work, materials, arrays, densities):
    import numpy as np
    from chatcarlo.gpu import seed_core, seed_cuda, seed_geometry, seed_geometry_checks
    from chatcarlo.gpu.seed_t4_checks import HostProbe
    grid = seed_geometry.Grid((-20.,)*3, (10.,)*3, (4,)*3)
    kw = dict(constants=arrays['constants'], strength=arrays['strength'], densities=densities,
              grids=(grid, grid), energy=.06, angle=0)
    if device == 'cpu': driver = seed_core.CPU(work, materials, **kw); probe = driver
    else:
        import cupy as cp
        driver = seed_cuda.GPU(cp, materials, **kw); probe = HostProbe(driver)
    boundary = seed_geometry_checks.v6(probe)
    r = driver.run(SEED_N, SEEDS['B3'], field=0)
    if not all(np.isfinite(r[k]).all() for k in ('d', 'i', 'ledger', 'source_counts')):
        raise AssertionError('nonfinite seed output')
    # run() already enforces per-history balance and coarse-grid conservation.
    return dict(status='pass', reached=dict(completed=len(r['d'])), boundary=boundary,
                deposition_MeV=float(r['histories']['dep'].sum()), field=0)


RELAX_KERNEL = '''
extern "C" __global__ void g5_relax_probe_kernel(const double *p,int el,int shell,int n,
 unsigned long long seed,double *out,long long *trace){
 if(blockIdx.x||threadIdx.x)return;
 RngState rng;rng_init(&rng,seed,0);
 for(int j=0;j<n;++j){double *row=out+38*j;int events;row[36]=0.;
 row[37]=g5_relax(p,el,shell,&rng,true,row,row+36,trace+28*j,&events);trace[28*j+27]=events;}
}
'''


def relaxation(driver, packed, shell, n, device):
    import numpy as np
    out = np.zeros((n, 38)); trace = np.zeros((n, 28), np.int64)
    if device == 'cpu':
        fn = driver.lib.g5_relax_probe
        fn.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_uint64, ct.c_void_p, ct.c_void_p]
        fn(packed.ctypes.data, 0, shell, n, SEEDS['B4'], out.ctypes.data, trace.ctypes.data)
    else:
        from chatcarlo.gpu import core
        cp = driver.cp
        fn = cp.RawKernel(core.cuda_source()+RELAX_KERNEL, 'g5_relax_probe_kernel',
                          options=driver.cuda_options, backend='nvrtc')
        o = cp.zeros_like(cp.asarray(out)); t = cp.zeros_like(cp.asarray(trace))
        fn((1,), (1,), (cp.asarray(packed), np.int32(0), np.int32(shell), np.int32(n),
                       np.uint64(SEEDS['B4']), o, t))
        out, trace = cp.asnumpy(o), cp.asnumpy(t)
    return out, trace


def shells(device, work):
    import numpy as np
    from chatcarlo.gpu import core
    from chatcarlo.gpu.generate_photon_tables import material, element
    a = diagnostic_arrays(); packed, _ = material([82], [1.])
    edges, yields, ck = element(82)[3:6]
    driver = core.Core(a, device, build_dir=Path(work)/'good')
    out, tr = relaxation(driver, packed, 1, 6000, device)
    np.testing.assert_allclose(out[:, 36]+out[:, 0:36:4].sum(axis=1), edges[1], atol=1e-10)
    if np.any(out[:, 37] < 0) or not np.isfinite(out).all(): raise AssertionError('relaxation failure')
    radiative = tr[:, 1] >= 0
    if abs(radiative.mean()-yields[1]) > .02: raise AssertionError('radiative probability')
    reach = dict(radiative=int(radiative.sum()))
    for dest in (2, 3):
        branch = (~radiative) & (tr[:, 2] == dest); reach['CK_to_'+str(dest)] = int(branch.sum())
        if abs(branch.mean()-ck[1, dest]) > .02: raise AssertionError('CK probability/destination')
    good = ((~radiative) & (tr[:, 2] == 3)).mean(); faults = {}
    for fault in (1, 2):
        bad = core.Core(a, device, build_dir=Path(work)/str(fault), _test_photon_fault=fault)
        _, bt = relaxation(bad, packed, 1, 6000, device)
        if ((bt[:, 1] < 0) & (bt[:, 2] == 3)).mean() >= good-.4: raise AssertionError('CK fault not detected')
        faults[str(fault)] = 'detected'
    for shell in range(9):
        o, _ = relaxation(driver, packed, shell, 100, device)
        np.testing.assert_allclose(o[:, 36]+o[:, 0:36:4].sum(axis=1), edges[shell], atol=1e-10)
    require_positive(reach)
    return dict(status='pass', reached=reach, faults=faults)


def transport_faults(device, work):
    import copy
    import numpy as np
    from chatcarlo.gpu import core, validation
    a = diagnostic_arrays(); src = (np.zeros((96, 3)), np.tile([0., 0., 1.], (96, 1)), np.full(96, 100.))
    good = core.Core(a, device, build_dir=Path(work)/'good').run(*src, seed=SEEDS['B5'], record=True)
    if good['completed'] != 96 or good['created'] <= 96: raise AssertionError('fluorescence fixture not reached')
    outgoing = good['terminals'][good['terminals'][:, 1] >= -1, 3].sum()
    np.testing.assert_allclose(good['deposits'].sum()+outgoing, 9600., rtol=1e-12)
    detected = {}
    for name, kw in [('capacity', dict(_test_stack_capacity=1)), *[(str(f), dict(_test_photon_fault=f)) for f in (3, 4, 5)]]:
        bad = core.Core(a, device, build_dir=Path(work)/name, **kw)
        try: bad.run(*src, seed=SEEDS['B5'])
        except ValueError as exc:
            if 'invalid history' not in str(exc): raise
            detected[name] = 'detected'
        else: raise AssertionError('transport fault not detected: '+name)
    a = copy.deepcopy(a); a['centers'] = a['halves'] = np.empty((0, 3)); a['materials'] = np.empty(0, np.int64)
    a['world'] = np.array([0., 0., 0., 10., 10., 10.]); a['data'][int(a['offsets'][19])] = 0.
    a['detector'] = np.array([0., 0., 1., 0., 0., -1., 1., 0., 0., 0., 1., 0., 2., 2., 2., 1.])
    for pixels in (2, 1):
        a['detector'][14] = pixels
        driver = core.Core(a, device, build_dir=Path(work)/f'two_{pixels}', _test_two_photons=True)
        r = driver.run(np.zeros((1, 3)), np.array([[0., 0., 1.]]), np.array([100.]), seed=9504004, record=True)
        if r['completed'] != 1 or r['created'] != 2 or r['endings'].tolist() != [0., 0., 2.]: raise AssertionError('two photons')
        np.testing.assert_array_equal(r['images'][4, :, 0], [1., 1.] if pixels == 2 else [2.])
        np.testing.assert_allclose(r['images'][5, :, 0], [50., 50.] if pixels == 2 else [100.])
        np.testing.assert_allclose(r['images'][6, :, 0], [2500., 2500.] if pixels == 2 else [10000.])
        for k, v in validation.record_reduce(r, a).items(): np.testing.assert_allclose(v, r[k], rtol=1e-14, atol=1e-14)
    return dict(status='pass', reached=dict(completed=96, descendants=int(good['created']), two_photons=2), faults=detected)


def run(path, device='cpu', progress=lambda result: None):
    path = Path(path); materials, arrays, densities = load_inputs(path)
    work = path/device; work.mkdir(parents=True, exist_ok=True)
    def digest(p):
        h = hashlib.sha256()
        with Path(p).open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
        return h.hexdigest()
    result = dict(seeds=SEEDS, device=device, gates={},
                  inputs={n: digest(path/n) for n in ('therapy.npz', 'materials.npz', 'inputs.json')},
                  fixture_source_sha256=digest(__file__))
    for name, action in [('B1', lambda: low(device, work/'B1', materials, arrays)),
                         ('B2', lambda: high(device, work/'B2', materials, arrays)),
                         ('B3', lambda: seed(device, work/'B3', materials, arrays, densities)),
                         ('B4', lambda: shells(device, work/'B4')),
                         ('B5', lambda: transport_faults(device, work/'B5'))]:
        result['gates'][name] = action(); write(work/'results.json', result); progress(result)
    result['status'] = 'pass'; write(work/'results.json', result)
    if device == 'cpu': write(path/'expected.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--p8g2', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu'); a = p.parse_args()
    from a3_t4_gates import writable, no_cache_writes, reserve
    a.output = writable(a.output); reserve(a.output, 512*1024**2)
    sys.dont_write_bytecode = True; sys.addaudithook(no_cache_writes)
    if a.p8g2: freeze_inputs(a.output, a.p8g2)
    run(a.output, a.device)


if __name__ == '__main__': main()
