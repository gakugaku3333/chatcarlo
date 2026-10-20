"""Generate P8e xraylib relaxation data and material bundles on explicit CLI use.

No generation or cache writes on import. All binding budgets come from PEGS;
individual radiative line energies come from xraylib, never from differences
of rounded binding energies. Nonradiative vacancies terminate locally.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from .materials_p8a import load_reference, load_manifest, sha, write, OUT

SHELLS = ('K', 'L1', 'L2', 'L3')


def fluorescence_table(shells):
    import xraylib as x
    zs = shells['Z']
    definitions = []
    for name in dir(x):
        match = re.fullmatch(r'(K|L1|L2|L3)(L[123]|M[1-5]|N[1-7]|O[1-7]|P[1-5]|Q[1-3])_LINE', name)
        if match:
            s, child = match.groups()
            definitions.append((SHELLS.index(s), SHELLS.index(child) if child in SHELLS else 4, name, getattr(x, name)))
    entries = [[[] for _ in SHELLS] for z in zs]
    yields = np.zeros((len(zs), 4))
    for i, z in enumerate(zs):
        for s in range(4):
            try:
                yields[i, s] = x.FluorYield(int(z), getattr(x, SHELLS[s]+'_SHELL'))
            except ValueError:
                # Missing/unoccupied shells carry zero radiative probability.
                continue
            for origin, child, name, code in definitions:
                if origin != s:
                    continue
                try:
                    rate = x.RadRate(int(z), code)
                    energy = x.LineEnergy(int(z), code)*.001
                except ValueError:
                    continue
                if rate > 0:
                    if child == 1:
                        raise ValueError('unexpected radiative transition into L1')
                    if energy <= 0 or child <= s:
                        raise ValueError('invalid radiative transition')
                    entries[i][s].append((energy, rate, child, code))
            if yields[i, s] > 0 and not entries[i][s]:
                raise ValueError('missing radiative lines')
    width = max(1, max(len(e) for row in entries for e in row))
    lines = np.zeros((len(zs), 4, width, 4))
    counts = np.zeros((len(zs), 4), dtype=np.int64)
    for i, row in enumerate(entries):
        for s, transitions in enumerate(row):
            counts[i, s] = len(transitions)
            if transitions:
                lines[i, s, :len(transitions)] = transitions
                lines[i, s, :len(transitions), 1] /= sum(t[1] for t in transitions)
    # Check EVERY radiative path, independent of the 10 keV transport cutoff.
    def paths(i, s):
        result = [0.]  # nonradiative termination, or no vacancy
        if s < 4 and yields[i, s] > 0:
            for energy, rate, child, code in lines[i, s, :counts[i, s]]:
                result.extend(energy+e for e in paths(i, int(child)))
        return result
    for i in range(len(zs)):
        for s in range(4):
            if max(paths(i, s)) > shells['binding_MeV'][i, s]:
                raise ValueError(f'negative PEGS relaxation budget: Z={zs[i]}, shell={SHELLS[s]}')
    return dict(yields=yields, lines=lines, line_counts=counts,
                xraylib_version=np.array(x.__version__))


def generate(cache=None):
    from .run_p8e import CACHE
    cache = CACHE.parent/'materials' if cache is None else Path(cache)
    manifest = load_manifest()
    cache.mkdir(parents=True, exist_ok=True)
    for material, row in manifest['materials'].items():
        shells = load_reference(row['shells'], material)
        data = fluorescence_table(shells)
        target = cache/(material+'_fluorescence.npz')
        np.savez(target, **data)
        write(target.with_suffix('.json'), dict(sha256=sha(target),
            shells_sha256=row['shells']['sha256'], generator_sha256=sha(Path(__file__)),
            xraylib_version=str(data['xraylib_version']),
            binding_source='PEGS5 P8a shells', line_source='xraylib LineEnergy/RadRate/FluorYield'))
    from .photon_transport_p8e import prepare_bundles
    return prepare_bundles(cache)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.cache), indent=2))


if __name__ == '__main__':
    main()
