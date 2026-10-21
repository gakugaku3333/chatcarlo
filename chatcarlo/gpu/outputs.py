"""Detector arrays, metadata and a spatially oriented count image."""
import json
from pathlib import Path
import numpy as np


def count_figure(tally):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    nu, nv = tally.plane.shape
    image = ax.imshow(tally.photon_count.T, origin='lower', extent=(-.5, nu-.5, -.5, nv-.5), aspect='equal')
    ax.set_xlabel('u pixel index')
    ax.set_ylabel('v pixel index')
    su, sv = tally.plane.size_cm
    top = ax.secondary_xaxis('top', functions=(lambda x: (x+.5)*su/nu-su/2, lambda x: (x+su/2)*nu/su-.5))
    right = ax.secondary_yaxis('right', functions=(lambda x: (x+.5)*sv/nv-sv/2, lambda x: (x+sv/2)*nv/sv-.5))
    top.set_xlabel('u [cm]')
    right.set_ylabel('v [cm]')
    fig.colorbar(image, ax=ax, label='photon count')
    fig.tight_layout()
    return fig


def save(result, path, metadata):
    path = Path(path).with_suffix('.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    t = result.detector
    tracked = t.track_uncertainty
    np.savez_compressed(path, category_fluence=t.category_fluence, photon_count=t.photon_count,
        energy_sum_keV=t.energy_sum_keV, energy_sum2_keV2=t.energy_sum2_keV2,
        total_rel_err=t.total_relative_error() if tracked else np.full(t.plane.shape, np.nan),
        n_batches_hit=t.n_batches_hit if tracked else np.full(t.plane.shape, -1, dtype=np.int32),
        n_histories=result.n_histories, n_batches=t.n_batches, pixel_area_cm2=t.plane.pixel_area_cm2)
    p = t.plane
    meta = dict(metadata, n_histories=result.n_histories, n_batches=t.n_batches,
        track_uncertainty=tracked, detected=float(t.photon_count.sum()),
        detector=dict(center_cm=p.center_cm.tolist(), normal=p.normal.tolist(), u_axis=p.u_axis.tolist(),
                      v_axis=p.v_axis.tolist(), size_cm=p.size_cm, shape=p.shape, roi=t.roi),
        axes=dict(array='[u,v]', png='u horizontal, v vertical, (0,0) lower left', position_unit='cm', count_unit='photons', fluence_unit='cm^-2'),
        energy_deposited_MeV=getattr(result, 'energy_deposited_MeV', {}))
    path.with_suffix('.json').write_text(json.dumps(meta, indent=2, allow_nan=True)+'\n')
    fig = count_figure(t)
    fig.savefig(path.with_suffix('.png'), dpi=140)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return meta
