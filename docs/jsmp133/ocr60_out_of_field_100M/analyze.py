"""Analyze existing public-figure inputs, without running transport or changing the site."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
G=ROOT/'runs/diag_water_60kev_gpu/add50M_20261026/dose_100M.npz';E=ROOT/'docs/egs5_crosscheck/revalidation_2026_10_07/comparison.csv'
d=np.load(G);D=d['dose_per_history_Gy'];S=d['sem_dose_per_history_Gy'];hits=d['n_batches_hit']
assert np.array_equal(d['origin_cm'],[-65,-65,-50]) and float(d['voxel_size_cm'])==1
rows={r['bin']:r for r in csv.DictReader(E.open())};ix=lambda x:x+65;iz=lambda z:z+50;yy=slice(ix(-1),ix(1))
records=[]
for name,z,depth in [('shallow',0,'0–1 cm'),('10cm',9,'9–10 cm')]:
 g=[];gs=[];e=[];es=[];nh=[]
 for x in range(8):
  a=(D[ix(x),yy,iz(z)].mean()+D[ix(-x-1),yy,iz(z)].mean())/2
  sa=(S[ix(x),yy,iz(z)].mean()+S[ix(-x-1),yy,iz(z)].mean())/2
  pair=[rows[f'lat_{name}_x{x}-{x+1}'],rows[f'lat_{name}_x{-x-1}-{-x}']]
  g.append(a);gs.append(sa);e.append(np.mean([float(r['egs5_Gy_history']) for r in pair]));es.append(np.mean([float(r['egs5_SEM_Gy_history']) for r in pair]))
  nh.append(int(min(hits[ix(x),yy,iz(z)].min(),hits[ix(-x-1),yy,iz(z)].min())))
 g,gs,e,es=map(np.array,(g,gs,e,es));gn=g/g[0];en=e/e[0]
 for x in range(8):
  diff=100*(gn[x]/en[x]-1)
  # Covariances of voxels/left-right/normalization are not saved in these inputs.
  # Triangle inequality bounds each code's normalized relative SEM; codes independent.
  upper=100*(gn[x]/en[x])*np.hypot(gs[x]/g[x]+gs[0]/g[0],es[x]/e[x]+es[0]/e[0])
  records.append(dict(depth=depth,x_lo_cm=x,x_hi_cm=x+1,x_mid_cm=x+.5,region='out-of-field' if x>=5 else ('edge-in-field' if x==4 else 'in-field'),gpu_Gy_history=g[x],egs5_Gy_history=e[x],gpu_OCR_pct=100*gn[x],egs5_OCR_pct=100*en[x],local_difference_pct=diff,central_dose_difference_pp=100*(gn[x]-en[x]),gpu_rel_sem_upper_pct=100*gs[x]/g[x],egs5_rel_sem_upper_pct=100*es[x]/e[x],normalized_difference_sem_upper_pct=upper,contributing_batches_min=nh[x],raw_difference_pct=100*(g[x]/e[x]-1)))
with (OUT/'bins.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
summary={}
for depth in ['0–1 cm','9–10 cm','all']:
 rr=[r for r in records if r['region']=='out-of-field' and (depth=='all' or r['depth']==depth)]
 a=np.array([r['local_difference_pct'] for r in rr]);summary[depth]={'n':len(rr),'mean_absolute_local_difference_pct':float(abs(a).mean()),'max_absolute_local_difference_pct':float(abs(a).max()),'mean_signed_local_difference_pct':float(a.mean())}
inner=[r['local_difference_pct'] for r in records if r['x_lo_cm'] in [1,2,3]]
assert np.isclose(np.mean(np.abs(inner)),.27,atol=.005) and round(max(abs(np.array(inner))),1)==.4
meta=dict(summary=summary,inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [G,E]},merge_provenance=json.loads((G.parent/'merge_provenance.json').read_text()),normalization='Each code normalized to its own |x|=0–1 cm bin; x/y convention exactly as public figure; left-right averaged.',range='Geometric field edge |x|=5 cm; out-of-field bins |x|=5–6,6–7,7–8 cm. No EGS5 data beyond 8 cm in the saved comparison.',uncertainty='SEM bounds only, with unknown covariance. No hypothesis test/pass-fail is assigned.')
(OUT/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axs=plt.subplots(2,2,figsize=(10,7),sharex='col',gridspec_kw={'height_ratios':[2,1]})
for j,depth in enumerate(['0–1 cm','9–10 cm']):
 rr=[r for r in records if r['depth']==depth];x=np.array([r['x_mid_cm'] for r in rr]);a,b=axs[:,j]
 for key,lab,c,marker in [('egs5_OCR_pct','EGS5','#c2410c','o'),('gpu_OCR_pct','ChatCarlo GPU','#1f6fb4','x')]:a.plot(x,[r[key] for r in rr],marker=marker,label=lab,c=c)
 a.set_yscale('log');a.set_ylim(.5,130);a.set_title('Depth '+depth);a.legend();a.set_ylabel('OCR (% of central bin)')
 b.errorbar(x[1:],[r['local_difference_pct'] for r in rr[1:]],yerr=[r['normalized_difference_sem_upper_pct'] for r in rr[1:]],fmt='o',capsize=3,c='#1f6fb4',label='SEM upper bound')
 b.axhline(0,c='black',lw=.7);b.set_ylabel('(GPU / EGS5 - 1) (%)');b.set_xlabel('|x| from axis (cm)');b.legend(fontsize=8)
 for ax in [a,b]:ax.axvspan(5,8,color='#fef3c7',alpha=.7,zorder=-10);ax.axvline(5,ls='--',c='gray');ax.set_xlim(0,8);ax.grid(alpha=.2)
fig.suptitle('60 keV water OCR: shaded area is outside 10 x 10 cm field\nExisting 100M GPU / 100M EGS5 histories; 1 cm bins, left-right averaged',fontsize=12)
fig.tight_layout();fig.savefig(OUT/'ocr_outside.png',dpi=180);plt.close(fig)
print(json.dumps(summary,ensure_ascii=False,indent=2))
for r in records:
 if r['region']=='out-of-field':print(r['depth'],r['x_lo_cm'], 'EGS OCR',round(r['egs5_OCR_pct'],4),'GPU OCR',round(r['gpu_OCR_pct'],4),'diff',round(r['local_difference_pct'],4),'SEMbound',round(r['normalized_difference_sem_upper_pct'],4),'hits',r['contributing_batches_min'])
