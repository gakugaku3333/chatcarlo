"""Regenerate only 60keV figure A using original plotting style, with merged 100M dose."""
from pathlib import Path
import json, hashlib
ROOT=Path(__file__).resolve().parents[3]
import os
os.chdir(ROOT)
source=Path('docs/jsmp133/region_analysis/build_fragment.py').read_text().split('R10, R4, K60, ST =')[0]
source=source.replace('runs/diag_water_60kev_gpu/dose.npz','runs/diag_water_60kev_gpu/add50M_20261026/dose_100M.npz')
ns={};exec(compile(source,'original_plot_definitions','exec'),ns)
r=ns['kev60']()
result={key:float(100*abs(r[arr]).mean() if stat=='mean' else 100*abs(r[arr]).max()) for key,arr,stat in [('pdd_mean','dp','mean'),('pdd_max','dp','max'),('ocr_mean','din','mean'),('ocr_max','din','max')]}
OUT=Path('runs/jsmp133_60kev_100M');OUT.mkdir(exist_ok=True)
(OUT/'pdd.svg').write_text(ns['fig_pdd'](r,'60 keV 水ファントムの PDD',marker=True))
(OUT/'ocr.svg').write_text(ns['fig_ocr'](r['prof'],'60 keV 水ファントムの OCR',band=False))
(OUT/'stats.json').write_text(json.dumps(result,indent=2))
merge=Path('runs/diag_water_60kev_gpu/add50M_20261026/merge_provenance.json');m=json.loads(merge.read_text())
assert hashlib.sha256(Path(m['output']['path']).read_bytes()).hexdigest()==m['output']['sha256']
print(result)
