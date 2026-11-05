"""One GPU physics name, with compile-time deposition specializations.

The no-argument declaration is the diagnostic default. Explicit names default
to electron transport for existing therapy callers; runtime paths pass the
specialization explicitly. S2 uses shared all-shell/CK photon tables; legacy builds are test-only.
"""
import hashlib
import json

NAME = 'chatcarlo-photon-electron-v1'
DIAGNOSTIC = NAME
LOCAL = 'local-deposition'
ELECTRON = 'electron-transport'
_LOCAL = dict(components=['photoelectric-all-shell-CK', 'bound-Compton', 'Rayleigh'],
    dose_quantities=['local-energy-deposition', 'track-length-kerma', 'H*(10)'],
    energy_band_keV=[5., 150.])
_ELECTRON = dict(components=['photoelectric-all-shell-CK', 'bound-Compton', 'Rayleigh', 'pair', 'electron', 'positron'],
    dose_quantities=['absorbed-dose'], energy_band_keV=[5., 10000.],
    stopping='ESTAR-table', fluorescence='all-shell-CK', elastic='ELSEPA-GS',
    brems='ESTAR-normalized', annihilation='rest-and-in-flight')
REGISTRY = {NAME: dict(implemented=True, descriptor=_ELECTRON,
    specializations={LOCAL: dict(id=1, descriptor=_LOCAL),
                     ELECTRON: dict(id=2, descriptor=_ELECTRON)})}


def declaration(name=None, *, specialization=None, require_implemented=True):
    specialization = specialization or (LOCAL if name is None else ELECTRON)
    name = NAME if name is None else name
    if name not in REGISTRY:
        raise ValueError('unknown or retired physics_set: '+str(name)+'; regenerate with '+NAME)
    entry = REGISTRY[name]
    if require_implemented and not entry['implemented']:
        raise ValueError('physics_set not implemented: '+name)
    if specialization not in entry['specializations']:
        raise ValueError('unknown physics specialization: '+str(specialization))
    spec = entry['specializations'][specialization]
    canonical = json.dumps(dict(name=name, specialization=specialization,
        stage='S2', descriptor=spec['descriptor']), sort_keys=True,
        separators=(',', ':'), allow_nan=False)
    tag = int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:8], 'big')
    return dict(name=name, specialization=specialization, id=spec['id'], tag=tag)


def observation(name, id, tag):
    # The native specialization ID and its descriptor tag are both observed.
    spec = next((s for s, v in REGISTRY[NAME]['specializations'].items() if v['id']==id), None)
    return dict(name=name, specialization=spec, id=int(id), tag=int(tag))


def check(actual, expected):
    if actual != expected:
        raise ValueError('physics_set declaration mismatch: '+repr(actual))


def require_quantity(name, quantity, *, specialization=ELECTRON):
    declaration(name, specialization=specialization)
    if quantity not in REGISTRY[name]['specializations'][specialization]['descriptor']['dose_quantities']:
        raise ValueError('unsupported dose quantity for '+name+'/'+specialization+': '+quantity)
