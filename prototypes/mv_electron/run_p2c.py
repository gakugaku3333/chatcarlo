"""Approved P2c checks, immutable 100-batch runs, and preregistered analysis.
--checks never calls --run or loads EGS5 dose results. --analyse is explicit only.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chisquare
from .physics import M, NE, ET, stopping
from .transport import CUT
from .flight_annihilation_p2c import sigma, differential, zeta_min, cdf, annihilation_rate, samples
from .photon_transport_p2c import run_coupled, initial
from .run_coupled_p2 import inputs, KEYS, MEV_TO_GY_G
from .p2_common import ROOT, TABLES, sha
from .photon_data_p2 import TABLE
from .workflow_p2 import load

OUT = ROOT/'docs/validation/mv/p2c'
CACHE = Path('~/.cache/chatcarlo/p2c').expanduser()
SOURCES = ('flight_annihilation_p2c.py', 'photon_transport_p2c.py', 'run_p2c.py',
           'photon_transport_p2.py', 'photon_sampling_p2.py', 'photon_data_p2.py',
           'physics.py', 'transport.py', 'run_coupled_p2.py', 'sampling.py',
           'backend_sampling.py', 'p2_common.py')
RUNS = {
    'p2c_on_6': dict(E=6., N=12000000, seed=20291310, annihilate_in_flight=True),
    'p2c_off_6': dict(E=6., N=12000000, seed=20291311, annihilate_in_flight=False),
    'p2c_on_10': dict(E=10., N=6938800, seed=20291320, annihilate_in_flight=True),
}


def source_hashes():
    return {str(Path('prototypes/mv_electron')/p): sha(ROOT/'prototypes/mv_electron'/p) for p in SOURCES}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as f:
        # numpy scalars (bool_, int64, float64) are stored as the plain Python value.
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=lambda o: o.item())
        f.write('\n'); temporary = Path(f.name)
    temporary.replace(path)


def save(path, **value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix=path.stem+'.', suffix='.pending.npz', delete=False) as f:
        np.savez(f, **value); temporary = Path(f.name)
    temporary.replace(path)


def finite(r):
    return all(bool(np.all(np.isfinite(v))) for v in r.values())


def execute(E, N, seed, **kwargs):
    backend = kwargs.pop('backend', 'mf')
    r = run_coupled(float(E), N, seed, .05, .025, 30., *inputs(backend), **kwargs)
    return dict(zip(KEYS, r))


def healthy(r):
    return finite(r) and not np.any(r['violations']) and not np.any(r['counts'][5:7])


def balance(r, source_count=0):
    c = r['counts']
    created = int(c[10]+c[11])+source_count
    finished = int(c[14]+c[12]+c[13])
    return dict(created=created, flight=int(c[14]), rest=int(c[12]), escaped=int(c[13]),
                finished=finished, equal=created==finished, finite=finite(r))


def ledger(r):
    return dict(violations=r['violations'].tolist(), failures=r['counts'][5:7].tolist(),
                maxima=r['maxima'].tolist(), finite=finite(r))


def table_hashes(backend, args):
    # Include the actual arrays and the files read to construct them.
    h = hashlib.sha256()
    for value in args:
        a = np.asarray(value)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return dict(arrays_sha256=h.hexdigest(), gs_sha256=sha(TABLES[backend]),
                photon_sha256=sha(TABLE), nist_sha256=sha(ROOT/'chatcarlo/data/nist_xaamdi/water.csv'))


def run_saved(name, E, N, seed, annihilate_in_flight, backend='elsepa', fe=.05, k=.025,
              thickness=30., stop_after=None):
    if N<=0 or N%100:
        raise ValueError('equal 100 batches required')
    if not .01<=E<=10:
        raise ValueError('source energy outside 10 keV–10 MeV')
    if Path(name).name!=name:
        raise ValueError('run name must be a single path component')
    work = CACHE/name; work.mkdir(parents=True, exist_ok=True)
    dest = OUT/'runs'/name
    with (work/'run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('same run is already executing') from exc
        args = inputs(backend)
        config = dict(E=E, N=N, seed=seed, annihilate_in_flight=annihilate_in_flight,
                      backend=backend, fe=fe, k=k, thickness=thickness, batches=100,
                      table_hashes=table_hashes(backend,args), source_hashes=source_hashes())

        def verify_config():
            if source_hashes()!=config['source_hashes'] or table_hashes(backend,args)!=config['table_hashes']:
                raise RuntimeError('run code/input hashes changed; stop')
            for p in (work/'config.json', dest/'config.json'):
                if p.exists() and json.loads(p.read_text())!=config:
                    raise RuntimeError('saved run config mismatch; no rerun permitted')

        verify_config()
        if not (work/'config.json').exists():
            write(work/'config.json', config)
        if (work/'terminal.json').exists():
            raise RuntimeError('run previously failed; no rerun permitted')
        if (dest/'result.npz').exists():
            if not (dest/'config.json').exists():
                raise RuntimeError('completed result lacks config')
            return dest/'result.npz'
        seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(100)]
        for b, bs in enumerate(seeds):
            verify_config()
            target = work/f'batch_{b:03}.npz'
            if target.exists():
                v = load(target)
                if int(v['N'])!=N//100 or int(v['seed'])!=bs or not finite(v):
                    raise RuntimeError('saved batch invalid')
                if not healthy(v):
                    raise RuntimeError('saved batch has ledger violations or failures')
                continue
            t = time.perf_counter()
            v = dict(zip(KEYS,run_coupled(float(E),N//100,bs,fe,k,thickness,*args,
                                        annihilate_in_flight=annihilate_in_flight)))
            seconds = time.perf_counter()-t
            if not finite(v):
                write(work/'terminal.json', dict(reason='non-finite batch', batch=b))
                raise RuntimeError('non-finite batch; stop')
            save(target, **{x:np.sum(v[x],axis=0) for x in KEYS[:5]},
                 **{x:v[x] for x in KEYS[5:]}, seconds=seconds, N=N//100, seed=bs)
            if not healthy(v):
                write(work/'terminal.json', dict(reason='ledger violation or failure',batch=b,**ledger(v)))
                raise RuntimeError('ledger violation or failure; stop')
            print(f'{name}: batch {b+1}/100', flush=True)
            if stop_after is not None and b+1==stop_after:
                return None
        verify_config()
        rows = [load(work/f'batch_{b:03}.npz') for b in range(100)]
        v = {x:np.stack([r[x] for r in rows]) for x in KEYS[:5]}
        v['coarse'] *= MEV_TO_GY_G/(N/100*.2)
        for x in KEYS[1:5]:
            v[x] *= MEV_TO_GY_G/(N/100*.05)
        for x in ('totals','violations','counts'):
            v[x] = np.sum([r[x] for r in rows],axis=0)
        v['maxima'] = np.max([r['maxima'] for r in rows],axis=0)
        v['maxdepth'] = max(int(r['maxdepth']) for r in rows)
        v['seconds'] = sum(float(r['seconds']) for r in rows)
        if not finite(v):
            raise RuntimeError('non-finite aggregate; stop')
        write(dest/'config.json', config)
        save(dest/'result.npz', **v, N=N, seed=seed, E=E, fe=fe, k=k,
             thickness=thickness, unit='Gy cm2', backend=backend, annihilate_in_flight=annihilate_in_flight)
        return dest/'result.npz'


def theoretical_probability(T):
    nodes = np.r_[CUT,ET[(ET>CUT)&(ET<T)],T]
    integral = sum(quad(lambda x: annihilation_rate(x)/stopping(x,False),a,b,
                        epsabs=0,epsrel=1e-12)[0] for a,b in zip(nodes[:-1],nodes[1:]))
    return -np.expm1(-integral)


def cross_section_check():
    integrals = []
    for T in np.geomspace(.01,10,61):
        integral, error = quad(lambda z:differential(T,z)/sigma(T),zeta_min(T),.5,
                               epsabs=0,epsrel=1e-12)
        integrals.append(dict(T=float(T), integral=integral, delta=integral-1, quad_error=error))
    anchors = []
    for T, reference in ((.01,1.27912842995e-24),(1.,1.71929574266e-25),(10.,3.82961894215e-26)):
        value = sigma(T)
        anchors.append(dict(T=T, value=value, reference=reference, relative_delta=value/reference-1))
    expected = NE*1.71929574266e-25
    rate = annihilation_rate(1.)
    passed = (max(abs(r['delta']) for r in integrals)<=1e-9 and
              max(abs(r['relative_delta']) for r in anchors)<=1e-9 and abs(rate/expected-1)<=1e-9)
    return passed, dict(integrals=integrals, anchors=anchors, rate=rate, rate_reference=expected,
                        rate_relative_delta=rate/expected-1)


def sampling_check(N=1000000):
    rows = []; passed = True
    for i,T in enumerate((.05,.5,2.,5.,9.)):
        z,phi,maxima = samples(T,N,20291210+i)
        edges = np.array([zeta_min(T)] + [brentq(lambda x:cdf(T,x)-q,zeta_min(T),.5,
                          xtol=5e-15) for q in np.arange(1,50)/50] + [.5])
        observed = np.histogram(z,edges)[0]
        azimuth = np.histogram(phi,np.linspace(0,2*np.pi,21))[0]
        cp = float(chisquare(observed,np.full(50,N/50)).pvalue)
        ap = float(chisquare(azimuth,np.full(20,N/20)).pvalue)
        ok = cp>=1e-4 and ap>=1e-4 and max(maxima[:2])<=(T+2*M)*1e-9 and maxima[2]<=1e-12
        rows.append(dict(T=T,N=N,seed=20291210+i,zeta_p=cp,azimuth_p=ap,
                         zeta_counts=observed.tolist(),azimuth_counts=azimuth.tolist(),
                         momentum_max=float(maxima[0]),energy_max=float(maxima[1]),
                         unit_length_max=float(maxima[2]),passed=bool(ok)))
        passed &= bool(ok)
    return passed, dict(rows=rows)


def checks():
    report = dict(status='実行中', source_hashes=source_hashes(), checks={})
    records = {}; c4_runs = []

    def record(name, passed, values):
        if source_hashes()!=report['source_hashes']:
            raise RuntimeError('check source hashes changed; stop')
        report['checks'][name] = dict(passed=bool(passed), **values)
        report['status'] = '実行中' if passed else '不合格'
        write(OUT/'checks.json',report)
        print(name+': '+json.dumps(report['checks'][name],ensure_ascii=False,default=lambda o: o.item()),flush=True)
        if not passed:
            raise RuntimeError(name+' failed; stop without changing criteria')

    try:
        rows=[]
        from .photon_transport_p2 import run_coupled as old
        args=inputs('mf')
        for i,E in enumerate((6.,10.)):
            original=dict(zip(KEYS,old(E,20000,20291201+i,.05,.025,30.,*args)))
            r=execute(E,20000,20291201+i,annihilate_in_flight=False)
            equality={x:bool(np.array_equal(original[x],r[x][:14] if x=='counts' else r[x])) for x in KEYS}
            rows.append(dict(E=E,N=20000,seed=20291201+i,equality=equality))
            records[f'C1_original_{E}']=ledger(original);records[f'C1_new_{E}']=ledger(r)
        record('C1',all(all(r['equality'].values()) for r in rows),dict(rows=rows))
        ok,v=cross_section_check();record('C2',ok,v)
        ok,v=sampling_check();record('C3',ok,v)
        rows=[]
        for i,T in enumerate((1.,2.5,5.,9.)):
            r=execute(T,200000,20291220+i,annihilate_in_flight=True,positron_source_T=T,positron_source_z=15.)
            c4_runs.append(r);records[f'C4_{T}']=ledger(r)
            P=int(r['counts'][15])/200000; theory=theoretical_probability(T)
            tolerance=.03*theory+3*np.sqrt(theory*(1-theory)/200000)
            rows.append(dict(T=T,N=200000,seed=20291220+i,source_flight=int(r['counts'][15]),
                             P=P,P_th=theory,delta=P-theory,tolerance=float(tolerance),
                             relative_delta=P/theory-1,passed=abs(P-theory)<=tolerance))
        record('C4',all(r['passed'] for r in rows),dict(rows=rows))
        r=execute(6.,20000,20291230,annihilate_in_flight=True)
        records['C6_photon']=ledger(r)
        balances=[dict(T=T,**balance(v,200000)) for T,v in zip((1.,2.5,5.,9.),c4_runs)]
        photon=balance(r)
        record('C6',all(v['equal'] and v['finite'] for v in balances) and photon['equal'] and photon['finite'] and photon['flight']>0,
               dict(source_rows=balances,photon=photon))
        record('C5',all(v['finite'] and not any(v['violations']) and not any(v['failures']) for v in records.values()),dict(runs=records))
        electron=initial(1.,15.,0.,0.,1.,0,1,.05,.025)
        e=execute(1.,1000,20291240,annihilate_in_flight=True,source_state=electron)
        p=execute(2.,1000,20291241,annihilate_in_flight=True,positron_source_T=2.,annihilation_scale=1000.)
        overflow=execute(2.,1000,20291242,annihilate_in_flight=True,positron_source_T=2.,annihilation_scale=1000.,maxstack=1)
        ancestry=execute(6.,20000,20291243,annihilate_in_flight=True)
        pair_total=float(np.sum(ancestry['pair'])); all_total=float(np.sum(ancestry['fine']))
        strong=balance(p,1000)
        # "ほぼ全部": at least 99% for the specified 1000x diagnostic.
        ok=(e['counts'][14]==0 and healthy(e) and strong['equal'] and p['counts'][15]>=990 and healthy(p)
            and overflow['counts'][6]>0 and finite(overflow) and not np.any(overflow['violations'])
            and max(overflow['maxima'])<=1e-9 and 0<pair_total<=all_total and healthy(ancestry))
        record('C7',ok,dict(electron_flight=int(e['counts'][14]),strong=strong,strong_ledger=ledger(p),
                           overflow=ledger(overflow),overflow_totals=overflow['totals'].tolist(),
                           pair_score=pair_total,all_score=all_total))
        cfg=dict(E=6.,N=2000,seed=20291250,annihilate_in_flight=True,backend='mf')
        work=CACHE/'p2c_resume_test'
        # Preserve evidence of the first interruption on subsequent invocations.
        evidence=work/'interruption.json'
        if not evidence.exists():
            if any(work.glob('batch_*.npz')):
                raise RuntimeError('resume test batches exist without interruption evidence')
            run_saved('p2c_resume_test',**cfg,stop_after=3)
            count=len(list(work.glob('batch_*.npz')))
            write(evidence,dict(stopped_batches=count))
        count=json.loads(evidence.read_text())['stopped_batches']
        resumed=load(run_saved('p2c_resume_test',**cfg))
        continuous=load(run_saved('p2c_resume_continuous',**cfg))
        equality={x:bool(np.array_equal(resumed[x],continuous[x])) for x in KEYS}
        rejected=False
        try:
            run_saved('p2c_resume_test',**{**cfg,'N':2100})
        except RuntimeError as exc:
            rejected='config mismatch' in str(exc)
        record('C8',count==3 and all(equality.values()) and rejected,
               dict(stopped_batches=count,equality=equality,changed_N_rejected=rejected))
        report['status']='合格';write(OUT/'checks.json',report)
    except Exception as exc:
        report['status']='不合格';report['stopped_reason']=str(exc)
        write(OUT/'checks.json',report)
        raise
    return report


def require_checks():
    v=json.loads((OUT/'checks.json').read_text())
    if (v.get('status')!='合格' or v.get('source_hashes')!=source_hashes() or
        set(v.get('checks',{}))!={f'C{i}' for i in range(1,9)} or
        not all(r['passed'] for r in v['checks'].values())):
        raise RuntimeError('C1–C8 pass with current source hashes required')


def region_difference(left, right, key, first, last, denominator):
    a=np.mean(left[key][:,first:last+1],axis=1)
    b=np.mean(right[key][:,first:last+1],axis=1)
    if len(a)!=100 or len(b)!=100:
        raise RuntimeError('analysis requires 100 batches')
    delta=float(np.mean(a)-np.mean(b))
    sem=float(np.sqrt(np.var(a,ddof=1)/100+np.var(b,ddof=1)/100))
    denom=float(np.mean(b)) if denominator is None else denominator
    return dict(key=key,first=first,last=last,delta_Gy_cm2=delta,SEM_Gy_cm2=sem,
                denominator_Gy_cm2=denom,delta_pct=100*delta/denom,SEM_pct=100*sem/denom)


def block_table(left,right,E):
    ranges = ((29,53),(54,78),(79,103),(104,128),(129,149)) if E==6 else ((38,62),(63,87),(88,112),(113,149))
    dref = 1.772659852283039e-11 if E==6 else 2.543223277645552e-11
    rows=[]
    for first,last in ranges:
        r=region_difference(left,right,'coarse',first,last,dref)
        r['status']=('統計不足' if r['SEM_pct']>1/3 else
                     '1%以内' if abs(r['delta_pct'])<=1 else '1%を超える')
        r['abs_delta_plus_1_96_SEM_pct']=abs(r['delta_pct'])+1.96*r['SEM_pct']
        rows.append(r)
    verdict=('固定した深部のブロック平均の差が1%以内' if all(r['status']=='1%以内' for r in rows) else
             '1%を超えるブロックあり' if any(r['status']=='1%を超える' for r in rows) else '統計不足で判定保留')
    build=(6,43) if E==6 else (10,71); near=(11,28) if E==6 else (18,37)
    return dict(E=E,Dref=dref,blocks=rows,verdict=verdict,
                buildup=region_difference(left,right,'fine',*build,dref),
                near=region_difference(left,right,'coarse',*near,dref))


def analyse():
    require_checks()
    native={}
    for name,cfg in RUNS.items():
        path=OUT/'runs'/name
        config=json.loads((path/'config.json').read_text())
        expected=dict(**cfg,backend='elsepa',fe=.05,k=.025,thickness=30.,batches=100)
        if any(config.get(k)!=v for k,v in expected.items()) or config['source_hashes']!=source_hashes():
            raise RuntimeError('production config/hash mismatch')
        r=load(path/'result.npz')
        numeric={k:v for k,v in r.items() if np.issubdtype(np.asarray(v).dtype,np.number)}
        if not healthy(numeric) or any(r[key].shape[0]!=100 for key in KEYS[:5]):
            raise RuntimeError('production has non-finite output, violations, failures or wrong batches; no analysis')
        if any(np.asarray(r[k]).item()!=v for k,v in expected.items() if k in r):
            raise RuntimeError('production result conditions differ from config')
        native[name]=r
    ref6path=ROOT/'docs/validation/mv/p2b/egs5/p2b_Splus'
    ref10path=ROOT/'docs/validation/mv/p2/egs5/main_ref_3_0'
    for path in (ref6path,ref10path):
        if json.loads((path/'summary.json').read_text())['status']!='合格':
            raise RuntimeError('reference did not pass its checks')
    ref6=load(ref6path/'result.npz');ref10=load(ref10path/'result.npz')
    off10=load(ROOT/'docs/validation/mv/p2/runs/main_elsepa_3/result.npz')
    on6=native['p2c_on_6'];off6=native['p2c_off_6'];on10=native['p2c_on_10']
    comparison={
        'on6_minus_EGS5':block_table(on6,ref6,6),
        'on10_minus_EGS5':block_table(on10,ref10,10),
        'off6_minus_EGS5':block_table(off6,ref6,6),
    }
    predictions=[]
    # Measured EGS5 effects are negated Sminus−Splus: the prediction is on−off.
    sm=json.loads((ROOT/'docs/validation/mv/p2b/supplementary/sminus.json').read_text())
    wide={r['region']:r for r in sm['wide'] if r['pair']=='p2b_Sminus - p2b_Splus'}
    for label,a,b,last,prediction,measured in (
        ('6 MeV shallow on−off',on6,off6,53,'約−0.2%',wide['0-10.8cm']),
        ('6 MeV whole on−off',on6,off6,149,'約−0.06%',wide['whole slab']),
        ('6 MeV shallow on−EGS5',on6,ref6,53,'0に近い',None),
        ('10 MeV shallow on−P2 off',on10,off10,62,'約−0.5%',None),
        ('10 MeV whole on−P2 off',on10,off10,149,'約−0.3%（消滅と放射損失の両方の変更からの参考値）',None),
    ):
        r=dict(quantity=label,prediction=prediction,**region_difference(a,b,'coarse',0,last,None))
        if measured is not None:
            refdelta=-measured['delta_pct_of_local_dose'];refsem=measured['SEM_pct_of_local_dose']
            r.update(EGS5_pct=refdelta,EGS5_SEM_pct=refsem,difference_from_EGS5_pct=r['delta_pct']-refdelta,
                     combined_SEM_pct=float(np.hypot(r['SEM_pct'],refsem)))
        predictions.append(r)
    for E,name,path,prediction in ((6,'p2c_on_6',ref6path,'約6.3%'),(10,'p2c_on_10',ref10path,'約8.5%')):
        c=native[name]['counts'];pairs=int(c[10]+c[11]);flight=int(c[14])
        rows=[load(CACHE/name/f'batch_{b:03}.npz')['counts'] for b in range(100)]
        fractions=np.array([v[14]/(v[10]+v[11]) for v in rows])
        sem=float(np.std(fractions,ddof=1)/10*100)
        cb=json.loads((path/'summary.json').read_text())['callbacks']
        refp=cb['flight_annihilation']/cb['pairs']
        refsem=np.sqrt(refp*(1-refp)/cb['pairs'])*100
        predictions.append(dict(quantity=f'{E} MeV flight/pairs',prediction=prediction,
                                delta_pct=100*flight/pairs,SEM_pct=sem,EGS5_pct=100*refp,
                                EGS5_SEM_pct=float(refsem),EGS5_SEM_method='対生成数に対する二項標準誤差（線量SEMとは別）',
                                difference_from_EGS5_pct=100*(flight/pairs-refp),combined_SEM_pct=float(np.hypot(sem,refsem))))
    summaries={}
    for name,r in native.items():
        c=r['counts']
        summaries[name]=dict(deposited_over_incoming=float(r['totals'][1]/r['totals'][0]),
                            pairs=int(c[10]+c[11]),rest=int(c[12]),flight=int(c[14]),escaped_positrons=int(c[13]),
                            violations=r['violations'].tolist(),failures=c[5:7].tolist(),seconds=float(r['seconds']))
    discussion=[
        '基準v2はブロックの点推定を±1%と比較する。真の差が95%の確度で±1%以内という意味ではない。',
        '予想との比較は記述であり、5.1の判定には使わない。陽電子にも電子の阻止能・硬衝突・弾性散乱を使う近似は残る。',
        '10 MeV全体の約−0.3%はEGS5で消滅と放射損失の両方を変えた値に由来し、消滅だけの定量的な予想ではない。',
    ]
    for r in predictions:
        s=f"{r['quantity']}: 予想 {r['prediction']}、結果 {r['delta_pct']:+.6f} ± {r['SEM_pct']:.6f}%（SEM）。"
        if 'difference_from_EGS5_pct' in r:
            s+=f" EGS5測定値との差 {r['difference_from_EGS5_pct']:+.6f} ± {r['combined_SEM_pct']:.6f}%。"
        discussion.append(s+' この数値とSEMを併せて解釈し、機械的な一致判定は付けない。')
    result=dict(source_hashes=source_hashes(),comparisons=comparison,predictions=predictions,runs=summaries,
                discussion=discussion,input_hashes={str(p.relative_to(ROOT)):sha(p) for p in
                [OUT/'runs'/n/'result.npz' for n in RUNS]+[ref6path/'result.npz',ref10path/'result.npz',
                ROOT/'docs/validation/mv/p2/runs/main_elsepa_3/result.npz']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for E,on,off,ref in ((6,on6,off6,ref6),(10,on10,off10,ref10)):
        dref=comparison['on6_minus_EGS5' if E==6 else 'on10_minus_EGS5']['Dref']
        fig,ax=plt.subplots(figsize=(9,5))
        z=(np.arange(150)+.5)*.2
        for label,a in (('on',on),('off' if E==6 else 'off (P2)',off)):
            delta=(a['coarse'].mean(axis=0)-ref['coarse'].mean(axis=0))/dref*100
            sem=np.sqrt(a['coarse'].var(axis=0,ddof=1)/100+ref['coarse'].var(axis=0,ddof=1)/100)/dref*100
            ax.errorbar(z,delta,yerr=sem,label=label,linewidth=1,elinewidth=.5)
        ax.axhline(0,color='black',linewidth=.5);ax.set(xlabel='Depth (cm)',ylabel='Difference / Dref (%)',title=f'{E} MeV: prototype minus EGS5 (±SEM)')
        ax.legend();fig.tight_layout();fig.savefig(OUT/f'difference_{E}MeV.png',dpi=160);plt.close(fig)
    write(OUT/'analysis.json',result)
    lines=['# P2c — 飛行中の陽電子消滅','', '## 予想','',
           '飛行中消滅の割合は6/10 MeVで約6.3/8.5%。線量変化の予想と結果は下表に並べる。','',
           '## 結果','', '差とSEMの単位は%（深部・ビルドアップ・ピーク付近はDref比、対応表の線量は引かれる側の領域平均比）。','']
    for name,v in comparison.items():
        lines += [f"### {name}",'',v['verdict'],'',
                  '| bin | 差 ± SEM (% Dref) | 状態 | |差|+1.96 SEM |',
                  '|---|---:|---|---:|']
        for r in v['blocks']:
            lines.append(f"| {r['first']}–{r['last']} | {r['delta_pct']:+.6f} ± {r['SEM_pct']:.6f} | {r['status']} | {r['abs_delta_plus_1_96_SEM_pct']:.6f} |")
        lines.append('')
        for key in ('buildup','near'):
            r=v[key];lines.append(f"{key} ({r['key']} bin {r['first']}–{r['last']}): {r['delta_pct']:+.6f} ± {r['SEM_pct']:.6f}%（判定なし）。")
        lines.append('')
    lines += ['### 理論との対応','', '| 量 | 予想 | 結果 ± SEM (%) | EGS5値 ± SEM | EGS5との差 ± 合成SEM |', '|---|---|---:|---:|---:|']
    for r in predictions:
        measurement=f"{r['EGS5_pct']:+.6f} ± {r['EGS5_SEM_pct']:.6f}" if 'EGS5_pct' in r else '—'
        diff=f"{r['difference_from_EGS5_pct']:+.6f} ± {r['combined_SEM_pct']:.6f}" if 'difference_from_EGS5_pct' in r else '—'
        lines.append(f"| {r['quantity']} | {r['prediction']} | {r['delta_pct']:+.6f} ± {r['SEM_pct']:.6f} | {measurement} | {diff} |")
    lines += ['', '割合の試作SEMは100バッチの割合の標本標準偏差/√100。EGS5の割合SEMは集計回数からの二項標準誤差。','',
              '### 収支・回数・時間','', '| 計算 | 沈着/入射 | 対生成 | 静止 | 飛行中 | 陽電子脱出 | 違反 | 失敗 | 秒 |', '|---|---:|---:|---:|---:|---:|---|---|---:|']
    for name,r in summaries.items():
        lines.append(f"| {name} | {r['deposited_over_incoming']:.9f} | {r['pairs']} | {r['rest']} | {r['flight']} | {r['escaped_positrons']} | {r['violations']} | {r['failures']} | {r['seconds']:.3f} |")
    lines += ['', '![6 MeV](difference_6MeV.png)','', '![10 MeV](difference_10MeV.png)','', '## 考察','']+discussion
    (OUT/'RESULTS.md').write_text('\n\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checks',action='store_true')
    group.add_argument('--run',choices=tuple(RUNS))
    group.add_argument('--analyse',action='store_true')
    a=p.parse_args()
    if a.checks:
        checks()
    elif a.run:
        require_checks()
        run_saved(a.run,**RUNS[a.run])
    else:
        analyse()


if __name__=='__main__':
    main()
