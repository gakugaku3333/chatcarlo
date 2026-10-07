"""Conditional engineering sensitivity, not a bound on an unimplemented code."""
from pathlib import Path
import json,numpy as np,csv
from analyze_egs5 import parse
BASE=Path(__file__).resolve().parent
def main():
 ref=json.loads((BASE/'egs5_check.json').read_text());assert ref['J3']=='pass'
 a,ap,_=parse('spectrum_pilot');b,bp,steps=parse('spectrum_200k');nt,ntp,_=parse('spectrum_no_timer');ns,nsp,_=parse('spectrum_no_pdd')
 assert b['J3']=='pass' and np.array_equal(ap,ntp)
 assert b['ledger']==ns['ledger'] and b['photon_steps']==ns['photon_steps']
 imax=int(np.argmax(bp[:,2]));points=[]
 for j in [0,imax,25,50,100,149]:
  r1=ap[j,3]/ap[j,2];r2=bp[j,3]/bp[j,2]
  nreq=b['histories']*(r2/.005)**2
  points.append({'depth_cm':float(bp[j,1]),'mean_MeV_history':float(bp[j,2]),'SEM_MeV_history':float(bp[j,3]),'R_100k':float(r1),'R_200k':float(r2),'observed_SEM_200k_over_100k':float(bp[j,3]/ap[j,3]),'ideal_SEM_ratio':float(1/np.sqrt(2)),'N_for_R_0_005':float(nreq)})
 nmax=max(v['N_for_R_0_005'] for v in points)
 km=json.loads((BASE/'kernel_measurement.json').read_text())
 units=[r['seconds_per_segment_untallied'] for r in km['rows'] if r['histories']==100000]
 cglo,cghi=min(units),max(units)
 sg=b['photon_steps']/b['histories'];se=b['charged_steps_mean']
 # Actual per-history scoring cost from a matched no-PDD run, excluding timers.
 score=max(0.,(b['cpu_seconds']-ns['cpu_seconds'])/b['histories'])
 # Variation across sample sizes is retained as uncertainty in this small timing subtraction.
 noise=abs(b['cpu_seconds']/b['histories']-nt['cpu_seconds']/nt['histories'])
 scorelo=0.;scorehi=score+noise
 # Unimplemented electron step: 1..20 times current diagnostic photon unit cost;
 # unimplemented secondary stack: 5..50% transport increment;
 # fine scoring: 1..10 times upper observed matched scoring unit.
 low=(sg+se)*cglo*1.05+scorelo
 high=(sg+20*se)*cghi*1.5+10*scorehi
 result={'J6':'conditional; assumed upper exceeds 24h','basis':'Mohan nominal 10 MV full histogram, 10x10 cm2 parallel beam, 30 cm water, central 2x2 cm2, 2 mm depth bins','points':points,'maximum_required_histories':nmax,'reference_spectrum':b,'photon_unit_seconds_range':[cglo,cghi],'charged_steps_per_primary':se,'photon_steps_per_primary_in_segmented_EGS_geometry':sg,'scoring_cpu_seconds_matched_difference':b['cpu_seconds']-ns['cpu_seconds'],'scoring_unit_seconds_point':score,'scoring_unit_seconds_range':[scorelo,scorehi],'scoring_timing_size_variation_seconds_per_history':noise,'matched_scoring_disabled_changes_physics':False,'assumptions':{'electron_unit_multiplier':[1,20],'stack_transport_multiplier':[1.05,1.5],'fine_scoring_multiplier':[1,10],'SEM_scaling':'R(N)=R(N0)*sqrt(N0/N), same estimator, future implementation variance assumed unchanged'},'formula':'C_low=(S_gamma+S_e)*c_gamma_min*1.05; C_high=(S_gamma+20*S_e)*c_gamma_max*1.5+10*c_score_high; T=N_max*C','seconds_per_history_range':[low,high],'hours_range':[nmax*low/3600,nmax*high/3600],'budget_hours':24,'budget_seconds_per_history':24*3600/nmax,'EGS5_wall_extrapolation_hours':nmax*b['wall_seconds']/b['histories']/3600,'scoring_array_bytes':6*150*8+100000*4,'grid_2mm_30cm_cube_dose_and_moments_bytes':3*150**3*8,'limitations':['Pilot SEMs are high; rare-event tails and all-bin maximum bias remain. Scaling is not a guarantee of achieving 0.5% SEM.','Conditional cost range uses assumed unimplemented model costs; measured photon units are 60 keV. Bounds can be exceeded.','EGS photon step count includes 2mm scoring-region boundaries; ChatCarlo may decouple transport from scoring.','Matched scoring time subtraction is one paired measurement and shares machine load with regression tests; range includes size variation but is not a confidence interval.','The transport/PDD is an EGS5 feasibility reference, not independent clinical validation.']}
 (BASE/'performance_check.json').write_text(json.dumps(result,indent=2)+'\n')
 with (BASE/'egs5/spectrum_pdd.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['depth_cm','mean_MeV_history','SEM_MeV_history','Gy_history']);w.writerows([p[1],p[2],p[3],p[2]*1.602176634e-13/.0008] for p in bp)
 np.savetxt(BASE/'egs5/spectrum_charged_steps.csv',steps,delimiter=',',fmt='%d',header='steps,history_count',comments='')
 print(json.dumps({k:v for k,v in result.items() if k!='reference_spectrum'},indent=2))
if __name__=='__main__':main()
