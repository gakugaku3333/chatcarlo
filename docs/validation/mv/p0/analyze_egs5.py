"""Reproduce P0 J3/J5, PDD, steps and measured performance from saved EGS5 output."""
from pathlib import Path
import json,re,hashlib,csv,os
import numpy as np
BASE=Path(__file__).resolve().parent
PREREG=BASE/'PREREGISTRATION.md'
def parse(name):
 d=BASE/'egs5'/name;p=d/'egs5job.out'
 if not p.exists():return None
 s=p.read_text();meta=json.loads((d/'metadata.json').read_text())
 assert meta['build_exit_code']==0 and meta['run_exit_code']==0
 assert meta['preregistration_sha256']==meta['preregistration_sha256_after']==hashlib.sha256(PREREG.read_bytes()).hexdigest()
 def tagged(tag):return [line.split()[1:] for line in s.splitlines() if line.strip().startswith(tag+' ')]
 ledger=list(map(float,tagged('LEDGER')[0]));res=(ledger[0]-sum(ledger[1:5]))/ledger[0]
 pdd=np.array([list(map(float,v)) for v in tagged('PDD')]);assert pdd.shape==(150,4)
 steps=np.array([list(map(int,v)) for v in tagged('STEPS')]);n=int(tagged('NCASE')[0][0]);assert sum(steps[:,1])==n
 cpu,scoring=map(float,tagged('TIME')[0])
 lst=(d/'pgs5job.pegs5lst').read_text();warnings=[]
 for m in re.finditer(r'Number of allocated intervals[^\n]*\n[^\n]*',lst):warnings.append(m.group(0).strip())
 result={'histories':n,'ledger':dict(zip(['incident_MeV','deposited_MeV','escaped_kinetic_MeV','unprocessed_MeV','net_rest_MeV','max_abs_residual_per_history_MeV','stack_nonempty_histories','abnormal_histories'],ledger)),'relative_residual':res,'J3':'pass' if abs(res)<=1e-4 and ledger[6]==ledger[7]==0 else 'fail','rest_components_MeV':dict(zip(['pair_created','annihilated','escaped_positron_rest'],map(float,tagged('REST_COMPONENTS')[0]))),'cpu_seconds':cpu,'ausgab_instrumented_cpu_seconds':scoring,'wall_seconds':meta['wall_seconds'],'rss_bytes':meta.get('run_peak_rss_bytes'),'charged_steps_mean':float(np.dot(steps[:,0],steps[:,1])/n),'charged_steps_quantiles':{str(q):int(np.repeat(steps[:,0],steps[:,1])[int((n-1)*q)]) for q in [.5,.9,.99,1.]},'photon_steps':int(tagged('PHOTON_STEPS')[0][0]),'warnings':warnings}
 assert abs(result['rest_components_MeV']['pair_created']-result['rest_components_MeV']['annihilated']-ledger[4])<1e-7
 assert abs(result['rest_components_MeV']['escaped_positron_rest']-ledger[4])<1e-7
 return result,pdd,steps

def main():
 coupled=parse('coupled');kerma=parse('kerma')
 if coupled is None or kerma is None:
  result={'J3':'unavailable','J5':'unavailable','reason':'Successful coupled and kerma run output not both available'}
 else:
  c,cp,steps=coupled;k,kp,_=kerma
  combined=float(np.hypot(cp[0,3],kp[0,3]));surface_z=float(abs(cp[0,2]-kp[0,2])/combined);imax=int(np.argmax(cp[:,2]));dmax=float(cp[imax,1])
  deep=[]
  for depth in [5.,10.,20.,29.9]:
   j=min(149,int(depth/.2));ratio=cp[j,2]/kp[j,2] if kp[j,2]>0 else None
   sem=ratio*np.hypot(cp[j,3]/cp[j,2],kp[j,3]/kp[j,2]) if ratio else None
   deep.append({'center_cm':float(cp[j,1]),'dose_to_local_kerma':ratio,'ratio_sem_independent_approximation':sem})
  result={'J3':'pass' if c['J3']==k['J3']=='pass' else 'fail','J5':'pass' if surface_z>2 and imax>0 else 'fail','coupled':c,'kerma':k,'surface_difference_sigma_independent_approximation':surface_z,'dmax_bin_center_cm':dmax,'surface_coupled_MeV_per_history':float(cp[0,2]),'surface_kerma_MeV_per_history':float(kp[0,2]),'peak_coupled_MeV_per_history':float(cp[imax,2]),'beyond_dmax_ratios':deep,'statistical_note':'Each SEM uses per-history first/second moments. Both runs share fixed seed but trajectories diverge; cross-run covariance unmeasured; combined SEM is an approximation. Surface difference is also reported relative to sum of SEMs.','surface_difference_over_sum_SEM':float(abs(cp[0,2]-kp[0,2])/(cp[0,3]+kp[0,3]))}
  with (BASE/'egs5/pdd.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['depth_cm','coupled_MeV_history','coupled_SEM','kerma_MeV_history','kerma_SEM','coupled_Gy_history','kerma_Gy_history'])
   for a,b in zip(cp,kp):w.writerow([a[1],a[2],a[3],b[2],b[3],a[2]*1.602176634e-13/.0008,b[2]*1.602176634e-13/.0008])
  np.savetxt(BASE/'egs5/charged_steps.csv',steps,delimiter=',',fmt='%d',header='charged_steps,history_count',comments='')
  os.environ.setdefault('MPLCONFIGDIR',str(BASE/'plot_cache'))
  os.environ.setdefault('XDG_CACHE_HOME',str(BASE/'plot_cache'))
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(8,4.5))
  scale=1.602176634e-13/.0008
  for data,label in [(cp,'Coupled photon-electron'),(kp,'Electrons locally stopped')]:
   ax.plot(data[:,1],data[:,2]*scale,label=label,lw=1.3)
   ax.fill_between(data[:,1],np.maximum(0,data[:,2]-data[:,3])*scale,(data[:,2]+data[:,3])*scale,alpha=.18)
  ax.set(xlabel='Depth in water (cm)',ylabel='Dose (Gy / incident 10 MeV photon)',title='EGS5 P0: 10 MeV, 10 x 10 cm², central 2 x 2 cm²; 2 mm bins')
  ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(BASE/'egs5/pdd.png',dpi=160);plt.close(fig)

  # Same-input repeats must match scoring exactly, excluding CPU timings.
  a=parse('coupled_repeat_c');b=parse('coupled_repeat_d')
  reproducible=bool(np.array_equal(a[1],b[1]) and a[0]['ledger']==b[0]['ledger'] and np.array_equal(a[2],b[2]))
  result['same_seed_reproducibility']=reproducible;assert reproducible
  result['main_points']=[{'center_cm':float(cp[j,1]),'mean_MeV_history':float(cp[j,2]),'SEM_MeV_history':float(cp[j,3]),'relative_SEM':float(cp[j,3]/cp[j,2]),'N_for_0_5_percent':float(c['histories']*(cp[j,3]/cp[j,2]/.005)**2)} for j in [0,imax,25,50,100,149]]
 (BASE/'egs5_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
