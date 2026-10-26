"""Summarize and plot the requested water calculation in per-history units."""
from pathlib import Path
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(os.environ.get('CHATCARLO_WATER60_RUN_DIR', str(Path(__file__).resolve().parents[4] / 'runs/diag_water_60kev_gpu/add50M_20261026')))
output = Path(os.environ.get('CHATCARLO_WATER60_OUTPUT_DIR', str(root)))
output.mkdir(parents=True, exist_ok=True)
with np.load(root.parent / 'dose.npz') as z:
    old = {k: z[k] for k in z.files}
with np.load(root / 'dose_100M.npz') as z:
    new = {k: z[k] for k in z.files}
coords = [new['origin_cm'][i] + (np.arange(int(new['shape'][i])) + .5) * float(new['voxel_size_cm']) for i in range(3)]
water = ((np.abs(coords[0])[:,None,None] < 15) & (np.abs(coords[1])[None,:,None] < 15)
         & (coords[2][None,None,:] > 0) & (coords[2][None,None,:] < 20))
field = water & (np.abs(coords[0])[:,None,None] < 5) & (np.abs(coords[1])[None,:,None] < 5)
i = np.argmin(np.abs(coords[0] - .5))
j = np.argmin(np.abs(coords[1] - .5))
k = np.flatnonzero((coords[2] > 0) & (coords[2] < 20))
idx = (int(i), int(j), int(k[0]))
summary = dict(n_histories=100000000, n_batches=int(new['n_batches']),
    central_surface_voxel_cm=[float(coords[d][idx[d]]) for d in range(3)],
    central_surface_dose_Gy_per_history=float(new['dose_per_history_Gy'][idx]),
    central_surface_dose_R=float(new['rel_err_dose'][idx]),
    central_surface_contributing_batches=int(new['n_batches_hit'][idx]),
    field_median_R_before=float(np.median(old['rel_err_dose'][field])),
    field_median_R_after=float(np.median(new['rel_err_dose'][field])),
    field_median_R_ratio=float(np.median(new['rel_err_dose'][field] / old['rel_err_dose'][field])),
    field_min_contributing_batches=int(np.min(new['n_batches_hit'][field])),
    water_max_dose_Gy_per_history=float(new['dose_per_history_Gy'][water].max()),
    total_track_kerma_keV_per_history=float(new['kerma_keV'].sum() / 100000000),
    units='Per incident photon (history); no mAs/CTDI calibration.',
    sampling='1 cm cubed voxel means; surface layer z=0 to 1 cm; x=y=0.5 cm near-axis column.',
    comparison='The 100M estimate contains the original 50M; the plotted estimates are correlated. Error bars are ±1 SEM.')
(output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for dataset, label, color in ((old, '50 million', '#8c8c8c'), (new, '100 million', '#0079b8')):
    depth = coords[2][k]
    dose = dataset['dose_per_history_Gy'][i,j,k]
    sem = dataset['sem_dose_per_history_Gy'][i,j,k]
    axs[0].errorbar(depth, dose, yerr=sem, color=color, label=label, marker='.', capsize=2)
    axs[1].plot(depth, 100 * dataset['rel_err_dose'][i,j,k], color=color, label=label, marker='.')
axs[0].set(xlabel='Water depth (cm)', ylabel='Absorbed dose (Gy/history)', title='Near-axis voxel dose (1 cm grid, ±1 SEM)')
axs[1].set(xlabel='Water depth (cm)', ylabel='Relative standard error (%)', title='Batch uncertainty')
for ax in axs:
    ax.grid(alpha=.25)
    ax.legend()
fig.suptitle('Water, 60 keV, parallel 10 x 10 cm beam; x=y=0.5 cm')
fig.savefig(output / 'comparison_50M_100M.png', dpi=160)
print(json.dumps(summary, indent=2))
