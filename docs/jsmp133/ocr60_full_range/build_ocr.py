"""Use the 100M dataset and all available OCR bins, excluding normalization bin."""
from pathlib import Path
import os,json,csv
ROOT=Path(__file__).resolve().parents[3];os.chdir(ROOT)
source=Path('docs/jsmp133/region_analysis/build_fragment.py').read_text().split('R10, R4, K60, ST =')[0]
source=source.replace('runs/diag_water_60kev_gpu/dose.npz','runs/diag_water_60kev_gpu/add50M_20261026/dose_100M.npz')
# All sampled OCR points are shown; blue region is the evaluated |x|=1..8 cm.
source=source.replace("m = p['e'] >= .2","m = np.ones_like(p['e'], dtype=bool)")
source=source.replace('a.axvspan(0, xi.max()+.25 if len(xi) else 0,','a.axvspan(1, 8,')
source=source.replace('b.axvspan(0, xi.max()+.25 if len(xi) else 0,','b.axvspan(1, 8,')
ns={};exec(compile(source,'original_plot_definitions','exec'),ns)
r=ns['kev60']()
for p in r['prof']:p['inner']=p['x']>.5
np=ns['np'];vals=np.concatenate([p['d'][p['inner']] for p in r['prof']])
stats=dict(n=len(vals),mean_absolute_difference_pct=float(100*abs(vals).mean()),max_absolute_difference_pct=float(100*abs(vals).max()),normalization='Each code central |x|=0–1cm; left/right averaged.',range='All available bins |x|=1–8cm, geometric field edge |x|=5cm, outside extension 3cm.',depths_cm=['0–1','9–10'])
# Independent cross-check against the prior exported 100M per-bin CSV.
rows=list(csv.DictReader(open('docs/jsmp133/ocr60_out_of_field_100M/bins.csv')))
v=[float(p['local_difference_pct']) for p in rows if int(p['x_lo_cm'])>0]
assert np.allclose(100*vals,v) and len(vals)==14
out=ROOT/'runs/jsmp133_ocr_full_range';out.mkdir(exist_ok=True)
(out/'ocr.svg').write_text(ns['fig_ocr'](r['prof'],'60 keV 水ファントムの OCR：照射野外 3 cm まで評価',band=False))
(ROOT/'docs/jsmp133/ocr60_full_range/summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(stats)
