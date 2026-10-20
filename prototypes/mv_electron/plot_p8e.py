"""P8e figure: depth dose of both codes and the judged block differences (report only)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[2]/'docs/validation/mv/p8e'


def main(out=OUT):
    e = np.load(out/'egs_production.npz'); p = np.load(out/'proto_production.npz')
    j = json.loads((out/'judgement.json').read_text()); dref = j['Dref']
    edges = e['edges_cm']; c = (edges[1:]+edges[:-1])/2
    em = e['dose'].mean(0)/dref; pm = p['dose'].mean(0)/dref
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    for z0, z1, label, colour in [(5, 11, 'lung', '#ddeeff'), (15, 17, 'bone', '#f3e6cc'), (21, 21.5, 'Pb', '#cccccc')]:
        for a in ax:
            a.axvspan(z0, z1, color=colour, zorder=0)
        ax[0].text((z0+z1)/2, 1.45, label, ha='center')
    ax[0].plot(c, em, 'k-', lw=1, label=f"EGS5 ({len(e['dose'])} batches)")
    ax[0].plot(c, pm, 'r.', ms=2, label='prototype')
    ax[0].set_ylabel('dose / Dref'); ax[0].set_ylim(0, 1.55); ax[0].legend(loc='lower left')
    b = np.array(j['blocks_cm']); d = np.array(j['difference'])*100; s = np.array(j['SEM'])*100
    ax[1].errorbar(b.mean(1), d, yerr=2*s, xerr=(b[:, 1]-b[:, 0])/2, fmt='o', color='r', ms=3,
                   label='block difference ± 2 SEM')
    for y in (1, -1):
        ax[1].axhline(y, ls='--', c='gray')
    ax[1].axhline(0, c='k', lw=.5)
    ax[1].set_ylabel('(prototype − EGS5)/Dref [%]'); ax[1].set_xlabel('depth [cm]')
    ax[1].set_ylim(-1.5, 1.5); ax[1].legend(loc='lower left')
    passed = j['status'].count('pass')
    fig.suptitle(f'P8e: water–lung–water–bone–water–Pb, 10 MV, {passed}/{len(j["status"])} blocks pass')
    fig.tight_layout(); fig.savefig(out/'p8e_depth_dose.png', dpi=130)


if __name__ == '__main__':
    main()
