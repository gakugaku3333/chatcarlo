"""Shared generation seal, approval and receipt contracts for both energy domains."""
import json
from pathlib import Path
from datetime import datetime, timezone
from .condition import canonical, sha256
from .therapy_bundle import sha
from . import physics_sets


def write(path, value):
    Path(path).write_text(canonical(value)+'\n')


def seal(path, cond, members):
    path=Path(path)
    physics_sets.check(cond['physics_set'],physics_sets.declaration(cond['physics_set']['name'],specialization=cond['physics_set'].get('specialization')))
    write(path/'generation.json',dict(condition=cond,condition_sha256=sha256(cond),
          files={name:sha(path/name) for name in members}))


def verify(path, members=None):
    path=Path(path);g=json.loads((path/'generation.json').read_text())
    if sha256(g['condition'])!=g['condition_sha256']:raise ValueError('condition hash mismatch')
    if members is not None and set(g['files'])!=set(members):raise ValueError('generation members mismatch')
    physics_sets.check(g['condition']['physics_set'],physics_sets.declaration(g['condition']['physics_set']['name'],specialization=g['condition']['physics_set'].get('specialization')))
    for name,digest in g['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('invalid seal member')
        if sha(path/name)!=digest:
            label='HTML' if name=='preview.html' else 'bundle' if name.startswith('bundle/') else name
            raise ValueError(label+' SHA256 mismatch')
    return g


def approve(path, verifier=verify):
    path=Path(path);verifier(path)
    write(path/'approval.json',dict(generation_sha256=sha(path/'generation.json'),approved_at=datetime.now(timezone.utc).isoformat()))


def require(path, expected, verifier=verify):
    path=Path(path)
    if not (path/'approval.json').exists():raise ValueError('approval.json missing')
    a=json.loads((path/'approval.json').read_text())
    if a.get('generation_sha256')!=sha(path/'generation.json'):raise ValueError('generation SHA256 mismatch')
    g=verifier(path)
    if g['condition']!=expected:raise ValueError('condition mismatch')
    return g


def check_observation(observed, expected, name):
    if not isinstance(observed,dict):raise ValueError('observation mismatch: '+name+' missing observed')
    for key in sorted(set(observed)|set(expected)):
        if key not in observed or key not in expected or observed[key]!=expected[key]:
            raise ValueError('observation mismatch: '+name+' '+key)


def receipt(actual, cond, observed):
    if actual.get('condition_sha256')!=sha256(cond):raise ValueError('receipt condition mismatch')
    physics_sets.check(actual.get('physics_set'),cond['physics_set'])
    physics_sets.check(observed,cond['physics_set'])
