"""Read-only bundle/T0 verification and canonical P9c approval conditions."""
import json
import os
from pathlib import Path
from .condition import digest
from ..gpu import p9c_core as core, p9c_io as io, p9c_colab as colab

T0_FILES = ('job.json', 'gates.json', 'rectangle_summary.json')
LOCAL_GATES = ('V0', 'V1a', 'V2a', 'V3', 'V5', 'V6', 'V6b')  # all require pass
GPU_GATES = ('V0', 'V1a', 'V1b', 'V3', 'G3_grazing', 'V4_rectangle', 'V4_prostate',
             'V5_GPU', 'V1c', 'V2b')  # pass; V2b also permits insufficient-statistics + fallback V2a


def read(path):
    return json.loads(Path(path).read_text())


def identities(scene):
    export = Path(os.environ.get('CHATCARLO_P9C_EXPORT', '~/.cache/chatcarlo/p9c2/export')).expanduser().resolve()
    t0 = Path(os.environ.get('CHATCARLO_P9C_T0', '~/.cache/chatcarlo/p9c/results_t0')).expanduser().resolve()
    try:
        record, _ = io.load(export)  # manifest, fingerprint, local gates, material structure
        hashes = {name: io.sha(t0/name) for name in T0_FILES}
        colab.require_t0(t0, record)
        prior = read(t0/'job.json')
        n, seed = scene['run']['histories_per_field'], scene['run']['seed']
        job = f'--mode production --histories-per-field {n} --t0 /content/p9c_t0 --beams prostate --seed {seed}'
        cond = dict(scene=scene, export=str(export), t0_path=str(t0),
                    manifest=read(export/'manifest.json'), sources=record['sources'], fingerprint=core.fingerprint(),
                    bundles=record['bundles'], t0=dict(files=hashes, sources=prior['sources'],
                    status=prior['status'], reason=prior.get('reason')), job=job)
        return cond
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        raise ValueError(f'P9c 束/T0: {exc}。既存の export と T0 を確認してください。再作成は gpu/p9c.py export と T0 の手順で行い、入口では作成しません') from exc


def checkpoint_identity(cond):
    from ..gpu.p9c_colab import selections
    return dict(n=cond['scene']['run']['histories_per_field'], seed=cond['scene']['run']['seed'],
                angles=list(core.ANGLES), grids=[g.record() for g in (core.FINE, core.COARSE)],
                beam='prostate', sources=cond['sources'], bundles=cond['bundles'],
                lines_seed_ids=selections(core.FINE)[-1])


def default_out(cond):
    return Path('~/.cache/chatcarlo/mv_entry').expanduser()/('seed_'+digest(cond)[:12])
