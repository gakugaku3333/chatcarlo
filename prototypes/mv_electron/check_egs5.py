"""Parse EGS5 raw output and use the common curve/uncertainty analysis."""
import re,json
import numpy as np
from .component_checks import OUT
from .analysis import compare,dref
BASE=OUT/'egs5'
def parse(name):
    d=BASE/name;text=(d/'egs5job.out').read_text()
    pdd=re.findall(r'^\s*PDD\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',text,re.M)
    if len(pdd)!=120:raise RuntimeError(f'{name}: incomplete EGS5 run (PDD rows={len(pdd)})')
    z=np.array([float(v[1]) for v in pdd]);q=np.array([float(v[2]) for v in pdd])
    batches=np.zeros((100,120))
    for b,j,v in re.findall(r'^\s*BATCH\s+(\d+)\s+(\d+)\s+(\S+)',text,re.M):batches[int(b)-1,int(j)-1]=float(v)
    ledger=[float(v) for v in re.search(r'^\s*LEDGER\s+(.+)',text,re.M).group(1).split()]
    steps=np.array([[float(x),float(y)] for x,y in re.findall(r'^\s*STEPS\s+(\S+)\s+(\S+)',text,re.M)])
    bins=np.array([[float(x),float(y)] for x,y in re.findall(r'^\s*STEPDIST\s+(\S+)\s+(\S+)',text,re.M)])
    deriv=json.loads((d/'derivation.json').read_text());meta=json.loads((d/'metadata.json').read_text())
    lst=(d/'pgs5job.pegs5lst').read_text()
    warnings=[line.strip() for line in (text+'\n'+lst).splitlines() if any(w in line.lower() for w in ['warning','accuracy','not attained','maximum number','required number','was insufficient'])]
    cross=np.array([[float(a),float(b),float(c)] for a,b,c in re.findall(r'^\s*CROSS\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)',text,re.M)])
    return {'z':z,'q':q,'batches':batches,'ledger':ledger,'steps':steps,'stepbins':bins,'derivation':deriv,'metadata':meta,'warnings':warnings,'cross':cross}
def stopping(name):
    lst=(BASE/name/'pgs5job.pegs5lst').read_text()
    rlc=float(re.search(r'ZE,ZX,RLC\s*\n\s*\S+\s+\S+\s+(\S+)',lst).group(1))
    calls=re.findall(r'Function call:\s*(\S+)\s*= BRMSTM OF\s*(\S+)\s+(\S+)',lst)
    from .physics import ET,ER
    return [{'T':t,'PEGS_radiative_MeV_cm2_g':float(c[0])/rlc,'ESTAR_radiative_MeV_cm2_g':float(np.interp(np.log(t),np.log(ET),ER)),'relative':float(c[0])/rlc/np.interp(np.log(t),np.log(ET),ER)-1} for t,c in zip([2,10],calls)]
def main(output_dir=None):
    result={}
    if not (BASE/'base_2/metadata.json').exists():return
    ref=parse('base_2');rows={};refsteps=ref['steps'];refbins=ref['stepbins']
    for name in ['ms_half','energy_half','fit_alternate']:
        if not (BASE/name/'metadata.json').exists():continue
        a=parse(name);c=compare(ref['batches'],a['batches'],ref['z'],bootstrap=1000,range_indices=(0,))
        c['mean_segments_per_primary']=float(np.sum(a['steps'][:,0]*a['steps'][:,1])/np.sum(a['steps'][:,1]))
        c['actual_step_distribution_changed']=bool(not np.array_equal(refbins,a['stepbins']))
        c['warnings']=a['warnings'];rows[name]=c
    ledger=ref['ledger']
    balance=abs(ledger[0]-sum(ledger[1:4]))/ledger[0]
    # Local radiation is established by AP>=T, zero generated-photon callbacks and closed ledger.
    radiation={'adopted':'normal IUNRST=0, AP=2 MeV (2 MeV source); AP=10 MeV (10 MeV source), UP=21 MeV, UE=20.511 MeV','prototype':'ESTAR radiative stopping power added to continuous loss, locally deposited','reference_balance_relative':balance,'reference_max_history_residual_MeV':ledger[4],'unprocessed_particles':int(ledger[5]),'photon_callbacks':int(ledger[6]),'PEGS_ESTAR':stopping('base_2'),'warnings':ref['warnings']}
    result['radiation']=radiation
    stable=len(rows)==3 and all(r['pass'] and r['actual_step_distribution_changed'] for r in rows.values())
    result['K4']={'status':'pass' if stable else 'incomplete' if len(rows)<3 else 'fail','rows':rows,'reference_mean_segments_per_primary':float(np.sum(refsteps[:,0]*refsteps[:,1])/np.sum(refsteps[:,1]))}
    if (OUT/'runs/base_2.npz').exists():
        p=np.load(OUT/'runs/base_2.npz');mask=ref['q']>=.2*dref(ref['q']);c=compare(ref['batches'],p['batches'],ref['z'],tol=.03,range_tol=.03,mask=mask)
        c['prototype_back_energy_fraction']=float(sum(p['ledger'][:,1])/(float(p['T'])*int(p['N'])))
        c['EGS5_back_energy_fraction']=float(ledger[2]/ledger[0]);c['status']='pass' if stable and c['pass'] else 'pending-K4' if not stable else 'fail'
        result['K6']=c
        egcross=ref['cross'];pn=p['cross_n'];pc=p['cross_sum']/pn
        ec=egcross[:,0]/egcross[:,2]
        result['K9_EGS_reference_only']={'EGS5_cm2':ec.tolist(),'prototype_cm2':pc.tolist(),'relative':(pc/ec-1).tolist(),'EGS5_cross_N':egcross[:,2].astype(int).tolist(),'prototype_cross_N':pn.tolist(),'status':'reference-only; not a K9 acceptance criterion'}
    for name,key in [('fit_changed','ineffective_fit_change'),('moliere_2','USEGSD_model_difference'),('radiation_local_discard_2','radiation_sensitivity')]:
        if (BASE/name/'metadata.json').exists():
            a=parse(name);c=compare(ref['batches'],a['batches'],ref['z'],tol=.03,range_tol=.03,bootstrap=1000);c['ledger']=a['ledger'];result[key]=c
    if (BASE/'base_10/metadata.json').exists() and (OUT/'runs/base_10.npz').exists():
        a=parse('base_10');p=np.load(OUT/'runs/base_10.npz');mask=a['q']>=.2*dref(a['q'])
        c=compare(a['batches'],p['batches'],a['z'],tol=.03,range_tol=.03,mask=mask,bootstrap=1000);c['status']='reference-only';result['K10']=c;result['radiation']['PEGS_10_AP']=stopping('base_10')
    ((OUT if output_dir is None else output_dir)/'egs5_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
