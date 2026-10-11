"""External XCOM generation and photon coefficients; units MeV, cm²/g."""
import json
import re
import shutil
import subprocess
import numpy as np
from numba import njit
from .physics import M, RE, NE
from .p2_common import ROOT,OUT,CACHE,SEED,ENERGIES,save,write,sha,portable
XWORK=CACHE/'xcom'
TABLE=CACHE/'photon.npz'

@njit
def kn_total(E):
    x=E/M;L=np.log1p(2*x)
    return 2*np.pi*RE**2*NE*((1+x)/x**2*(2*(1+x)/(1+2*x)-L/x)+L/(2*x)-(1+3*x)/(1+2*x)**2)

@njit
def kn_transfer(E):
    # q=E'/E; dσ/dq = πr²/x [q + A + B/q + C/q²].
    # A=2/x+1/x², B=1-2/x-2/x², C=1/x²; integrate (1-q).
    x=E/M;q=1/(1+2*x);A=2/x+1/x**2;B=1-2/x-2/x**2;C=1/x**2
    def F(t):
        return t*t/2-t*t*t/3+A*(t-t*t/2)+B*(np.log(t)-t)+C*(-1/t-np.log(t))
    return np.pi*RE**2*NE/x*(F(1.)-F(q))

@njit
def log_interp(E,es,vs):
    return np.exp(np.interp(np.log(E),np.log(es),np.log(vs)))

@njit
def pair_interp(E,es,vs,threshold):
    if E<=threshold:return 0.
    # Transform positive nodes only. Endpoint continuation in transformed space
    # covers the interval between the exact threshold and the first positive node.
    start=np.searchsorted(es,threshold,side='right')
    return np.exp(np.interp(np.log(E),np.log(es[start:]),
                           np.log(vs[start:]/(1-threshold/es[start:])**3)))*(1-threshold/E)**3

@njit
def coefficients(E,es,water,thresholds):
    if E<es[0] or E>es[-1]:raise ValueError('photon energy outside table range')
    return np.array([kn_total(E),log_interp(E,es,water[:,2]),
                    pair_interp(E,es,water[:,3],thresholds[0]),
                    pair_interp(E,es,water[:,4],thresholds[1])])

@njit
def transfer_coefficient(E,es,water,thresholds):
    c=coefficients(E,es,water,thresholds)
    return kn_transfer(E)+c[1]+(c[2]+c[3])*max(0.,1-2*M/E)

def grid(thresholds):
    # 160 intervals per decade, fixed before generating/checking any new values.
    es=np.geomspace(.001,20.,int(np.ceil(np.log10(20/.001)*160))+1)
    return np.unique(np.r_[es,.01,.511,M,1.25,2.,6.,10.,
                           [t*f for t in thresholds for f in (.9999,.99999,1.,1.00001,1.0001,1.001,1.01)]])

def xcom(energies,name,spec):
    result=[]
    # XCOM has a 600 requested-energy bound; chunk without changing source.
    for offset in range(0,len(energies),500):
        e=energies[offset:offset+500]
        (XWORK/'energies.txt').write_text(str(len(e))+'\n'+'\n'.join(format(v,'.17g') for v in e)+'\n')
        inp='\n'.join([name]+spec+['3','2','energies.txt','table.txt','1'])+'\n'
        p=subprocess.run(['./xcom'],input=inp,cwd=XWORK,capture_output=True,text=True,timeout=60);p.check_returncode()
        raw=(XWORK/'table.txt').read_text();rows=[]
        for line in raw.splitlines():
            fields=line.split()
            if len(fields)==8:
                try:rows.append(list(map(float,fields)))
                except ValueError:pass
        if len(rows)!=len(e):raise RuntimeError('XCOM row count mismatch')
        a=np.array(rows)
        if not np.allclose(a[:,0],e,rtol=.0006,atol=0):raise RuntimeError('XCOM energy correspondence failed')
        tag=f'{name}_{offset}'
        (XWORK/(tag+'.txt')).write_text(raw);(XWORK/(tag+'.input')).write_text(inp)
        # Original requested columns, never the printed 4-digit energies.
        np.savetxt(XWORK/(tag+'_requested.txt'),e,fmt='%.17g')
        result.append(a[:,1:])
    return np.vstack(result)

def generate():
    if TABLE.exists():
        if not (OUT/'photon_generation.json').exists():raise RuntimeError('orphan table: inspect before resuming')
        expected=json.loads((OUT/'photon_generation.json').read_text())['table_sha256']
        if sha(TABLE)!=expected:raise RuntimeError('generated photon table changed')
        return
    source=ROOT/'docs/validation/mv/p0/data/XCOM'
    XWORK.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,XWORK,dirs_exist_ok=True)
    p=subprocess.run(['gfortran','-std=legacy','-O2','XCOM.f','-o','xcom'],cwd=XWORK,capture_output=True,text=True);p.check_returncode()
    (XWORK/'build.log').write_text(p.stdout+p.stderr)
    text=(source/'XCOM.f').read_text()
    thresholds=np.array([float(v.replace('D','E'))/1e6 for v in re.search(r'DATA EPAIR1/([^/]+)/,EPAIR2/([^/]+)/',text).groups()])
    atoms=[float(v) for v in re.findall(r'\d+\.\d+',(source/'ATWTS.DAT').read_text())]
    weights=np.array([2*atoms[0],atoms[7]]);weights/=weights.sum()
    es=grid(thresholds)
    element=np.stack([xcom(es,f'Z{z}',['1',str(z),'3']) for z in (1,8)])
    water=np.einsum('i,ijk->jk',weights,element)
    direct=xcom(es,'water',['3','H2O'])
    mid=np.unique(np.r_[np.sqrt(es[:-1]*es[1:]),.511,
                           [t*f for t in thresholds for f in (.999995,1.000005,1.00005,1.0005,1.005)]])
    mid_direct=xcom(mid,'water_mid',['3','H2O'])
    save(TABLE,E=es,water=water,direct=direct,element=element,weights=weights,thresholds=thresholds,mid=mid,mid_direct=mid_direct)
    write('photon_generation.json',dict(table=portable(TABLE),table_sha256=sha(TABLE),source_sha256=sha(source/'XCOM.f'),
          requested_energy_sha256=sha(XWORK/'water_0_requested.txt'),nodes=len(es),midpoints=len(mid),points_per_decade=160,
          thresholds_MeV=thresholds.tolist(),water_weights=weights.tolist(),
          source_url='https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz',
          external_files={portable(p):sha(p) for p in sorted(XWORK.glob('*.txt'))},
          build_command='gfortran -std=legacy -O2 XCOM.f -o xcom',printed_energy_used_as_grid=False))

@njit
def transmission_and_process(E,n,seed,es,w,th):
    np.random.seed(seed);c=coefficients(E,es,w,th);mu=c.sum();trans=0;counts=np.zeros(4,np.int64)
    for i in range(n):
        if -np.log(np.random.random())/mu>=10.:trans+=1
        r=np.random.random()*mu;j=0
        while j<3 and r>=c[j]:r-=c[j];j+=1
        counts[j]+=1
    return trans,counts

def check():
    if (OUT/'q1.json').exists():return json.loads((OUT/'q1.json').read_text())
    g=np.load(TABLE);es=g['E'];w=g['water'];d=g['direct'];th=g['thresholds'];mid=g['mid'];md=g['mid_direct']
    rel=np.divide(w[:,:5]-d[:,:5],d[:,:5],out=np.zeros_like(w[:,:5]),where=d[:,:5]!=0)
    amax=float(np.max(abs(rel)))
    rows=[]
    for j in (1,2,3):
        values=np.array([coefficients(float(e),es,w,th)[j] for e in mid]);ref=md[:,j+1]
        r=np.divide(values-ref,ref,out=np.zeros_like(ref),where=ref!=0)
        rows.append(dict(process=('compton','photoelectric','pair_nuclear','pair_electron')[j],max_relative=float(max(abs(r))),
                         worst_MeV=float(mid[np.argmax(abs(r))]),zero_reference_nonzero=int(np.sum((ref==0)&(values!=0)))))
    b=all(coefficients(float(e),es,w,th)[j+2]==0 for j,t in enumerate(th) for e in (.001,t*(1-1e-10),t))
    stochastic=[];n=10000000
    for j,E in enumerate(ENERGIES):
        trans,counts=transmission_and_process(E,n,SEED+100+j,es,w,th);c=coefficients(E,es,w,th);mu=float(c.sum())
        p=float(np.exp(-mu*10));s=np.sqrt(p*(1-p)/n);prob=c/mu;se=np.sqrt(prob*(1-prob)/n)
        zs=np.divide(counts/n-prob,se,out=np.zeros(4),where=se>0)
        if np.any((se==0)&(counts!=n*prob)):raise RuntimeError("degenerate process probability violated")
        stochastic.append(dict(E=E,N=n,seed=SEED+100+j,mu=mu,expected_transmission=p,measured_transmission=trans/n,
                              transmission_z=float((trans/n-p)/s),counts=counts.tolist(),expected_process_probabilities=prob.tolist(),process_z=zs.tolist()))
    kn=np.array([kn_total(float(e)) for e in es]);ratio=kn/w[:,1]-1
    save(OUT/'kn_xcom_difference.npz',E=es,kn=kn,xcom_incoherent=w[:,1],relative=ratio)
    tests=dict(a=amax<=.001,b=b,c=all(r['max_relative']<=.002 and r['zero_reference_nonzero']==0 for r in rows),
               d=all(abs(r['transmission_z'])<=3 for r in stochastic),e=all(max(abs(np.array(r['process_z'])))<=3 for r in stochastic))
    result=dict(status='合格' if all(tests.values()) else '不合格',tests=tests,synthesis_max_relative=amax,interpolation=rows,stochastic=stochastic,
                kn_xcom_relative_at_10keV=float(kn_total(.01)/np.exp(np.interp(np.log(.01),np.log(es),np.log(w[:,1])))-1),
                kn_difference_file='docs/validation/mv/p2/kn_xcom_difference.npz',note='Q1(a) is path/rounding consistency, not physical accuracy')
    write('q1.json',result);return result
