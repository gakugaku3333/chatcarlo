"""Compile-time physics identities; canonical descriptors are part of the contract."""
import hashlib
import json

DIAGNOSTIC = 'diagnostic-photon-kerma-v1'
REGISTRY = {
    DIAGNOSTIC: dict(id=1, implemented=True, descriptor=dict(
        components=['photoelectric-K-fluorescence', 'bound-Compton', 'Rayleigh'],
        dose_quantities=['local-energy-deposition', 'track-length-kerma', 'H*(10)'],
        energy_band_keV=[5., 150.])),
    'therapy-electron-v1': dict(id=2, implemented=False, descriptor=dict(
        components=['photon', 'electron', 'positron'], dose_quantities=['absorbed-dose'],
        energy_band_keV=[5., 20000.])),
}


def declaration(name=DIAGNOSTIC, *, require_implemented=True):
    if name not in REGISTRY:
        raise ValueError('unknown physics_set: '+str(name))
    entry = REGISTRY[name]
    if require_implemented and not entry['implemented']:
        raise ValueError('physics_set not implemented: '+name)
    canonical = json.dumps(entry['descriptor'], sort_keys=True, separators=(',', ':'), allow_nan=False)
    tag = int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:8], 'big')
    return dict(name=name, id=entry['id'], tag=tag)


def check(actual, expected):
    if actual != expected:
        raise ValueError('physics_set declaration mismatch: '+repr(actual))
