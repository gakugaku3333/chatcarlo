"""P2b 追補1: 6 MeV EGS5 three-condition runs (S+, C-, S-) and their fixed analysis.

Builds and runs outside the repository (~/.cache/chatcarlo/p2b); only parsed
results are stored under docs/validation/mv/p2b. P2 files are read, never written.
"""
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .reference_p2 import parse,pegs_input

ROOT=Path(__file__).resolve().parents[2]
EGS=ROOT/'docs/egs5_crosscheck/egs5'
CODE=ROOT/'prototypes/mv_electron'
P2=ROOT/'docs/validation/mv/p2'
OUT=ROOT/'docs/validation/mv/p2b'
CACHE=Path(os.environ.get('CHATCARLO_P2B_CACHE','~/.cache/chatcarlo/p2b')).expanduser()
E=6.0
N=12000000
# name: (IUNRST, reject in-flight annihilation, seed)
RUNS={'p2b_Splus':(0,0,20291040),'p2b_Cminus':(4,0,20291041),'p2b_Sminus':(0,1,20291042)}
BUILD=list(range(6,44))  # 0.5 mm bins fixed in P2 pilot_lock.json for 6 MeV


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)


def sources():
    return {f:sha(CODE/f) for f in ('p2b_egs.f','p2b_egs_common.inc','p2_egs.f','reference_p2.py')}


def check_work(work,config):
    """The external work folder must be the job that config.json declares (review 1, point 1)."""
    n,seed,mode,noflight=config['N'],config['seed'],config['IUNRST'],config['noflight']
    ok=((work/'config.dat').read_text()==f'{E:.17g} {n} {seed} {mode}\n' and (work/'p2b_egs.inp').read_text()==config['input']
        and sha(work/'p2b_egs.f')==config['code_sha256'] and sha(work/'p2b_egs_common.inc')==config['include_sha256']
        and (work/'noflight.dat').exists()==bool(noflight))
    if not ok:raise RuntimeError('work folder does not match the declared configuration: '+work.name)


def check_output(work,config):
    """What the Fortran job itself printed must be the declared job."""
    n,seed,mode,noflight=config['N'],config['seed'],config['IUNRST'],config['noflight']
    text=(work/'egs5job.out').read_text()
    if 'TIME_LIMIT' in text:raise RuntimeError('EGS5 time limit: '+work.name)
    def one(tag):
        m=re.findall(r'^\s*'+tag+r'\s+(.+)',text,re.M)
        if len(m)!=1:raise RuntimeError('incomplete EGS5 output: '+tag)
        return m[0].split()
    c=one('CONFIG');ledger=float(one('LEDGER')[0])
    if not (float(c[0])==E and [int(v) for v in c[1:]]==[n,seed,mode] and [int(v) for v in one('NOFLIGHT')]==[noflight]
            and abs(ledger-n*E)<=1e-12*n*E):
        raise RuntimeError('EGS5 output is not the declared job: '+work.name)


def run(name,n,seed,mode,noflight,folder='egs5',stop_after=None):
    """One EGS5 job; a finished job is reused, an interrupted one resumes from its checkpoint.

    stop_after (pre-check only) kills the job once that many batches are checkpointed.
    """
    work=CACHE/folder/name;dest=OUT/folder/name
    config=dict(E=E,N=n,seed=seed,IUNRST=mode,noflight=noflight,code_sha256=sha(CODE/'p2b_egs.f'),
                include_sha256=sha(CODE/'p2b_egs_common.inc'),base_code_sha256=sha(CODE/'p2_egs.f'),input=pegs_input(mode))
    if (dest/'config.json').exists() and json.loads((dest/'config.json').read_text())!=config:
        raise RuntimeError('saved config differs; no rerun permitted: '+name)
    if (dest/'summary.json').exists():return dest
    work.mkdir(parents=True,exist_ok=True)
    # Single writer per work folder. The Fortran child inherits the descriptor, so the
    # lock outlives this Python process for as long as the job itself is alive.
    lock=os.open(work/'run.lock',os.O_CREAT|os.O_RDWR)
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('another process is running this job: '+name)
    try:
        write(dest/'config.json',config)
        if not (work/'egs5job.exe').exists():
            for f in ('p2b_egs.f','p2b_egs_common.inc'):shutil.copy2(CODE/f,work/f)
            (work/'p2b_egs.inp').write_text(config['input'])
            shutil.copy2(ROOT/'docs/validation/mv/p1/egs5/base_2/epstar.dat',work/'epstar.dat')
            (work/'config.dat').write_text(f'{E:.17g} {n} {seed} {mode}\n')
            if noflight:(work/'noflight.dat').write_text('1\n')
            p=subprocess.run([str(EGS/'egs5run'),'comp'],cwd=work,input='p2b_egs\n\n\n',capture_output=True,text=True)
            (work/'build.log').write_text(p.stdout+p.stderr)
            if p.returncode or not (work/'egs5job.exe').exists():raise RuntimeError('EGS5 build failed: '+name)
        check_work(work,config)
        # EGS5 reads gsdist.dat whenever it exists; one left by a job killed before its first
        # checkpoint may be truncated. Nothing is deleted here: stop and let a person remove it.
        if (work/'gsdist.dat').exists() and not (work/'checkpoint.bin').exists():
            raise RuntimeError('gsdist.dat without a checkpoint (possibly truncated); remove the work folder by hand: '+str(work))
        with (work/'run.log').open('a') as log:
            p=subprocess.Popen(['./egs5job.exe'],cwd=work,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock,))
            while p.poll() is None:
                time.sleep(.2 if stop_after else 2)
                f=work/'batch_progress.txt'
                if stop_after and f.exists() and int(f.read_text().split()[1])>=stop_after:
                    p.kill();p.wait();raise InterruptedError(name)
        if p.returncode:raise RuntimeError('EGS5 run failed: '+name)
        check_output(work,config)
        coarse,fine,summary=parse(work,n,E,mode,seed)
        summary['noflight']=noflight
        np.savez(dest/'result.npz',coarse=coarse,fine=fine,E=E,N=n,seed=seed)
        write(dest/'summary.json',summary)
    finally:
        os.close(lock)
    return dest


def precheck():
    """追補1 §3: regression against the P2 self-check and behaviour of the rejection."""
    a=run('precheck_regression',10000,20261524,0,0,'precheck')
    old=np.load(P2/'egs5/selfcheck_d2_2_0/result.npz');new=np.load(a/'result.npz')
    regression=bool(np.array_equal(old['coarse'],new['coarse']) and np.array_equal(old['fine'],new['fine']))
    b=run('precheck_noflight',200000,20291001,0,1,'precheck')
    for d in (a,b):  # also for reused jobs: the saved output must be the declared job
        if (CACHE/'precheck'/d.name).exists():check_output(CACHE/'precheck'/d.name,json.loads((d/'config.json').read_text()))
    # Review 1, point 8: the rejection path must survive a checkpoint restart unchanged.
    try:run('precheck_noflight_resume',200000,20291001,0,1,'precheck',stop_after=40)
    except InterruptedError:pass
    c=run('precheck_noflight_resume',200000,20291001,0,1,'precheck')
    whole=np.load(b/'result.npz');part=np.load(c/'result.npz');sc=json.loads((c/'summary.json').read_text())
    s=json.loads((b/'summary.json').read_text());cb=s['callbacks']
    ratio=cb['rest_annihilation']/cb['pairs']
    tests=dict(regression_identical=regression,parse_all=s['status']=='合格',balance=s['max_history_relative_balance']<=1e-9 and s['ledger'][5]==0,
               flight_calls_positive=cb['flight_annihilation']>0,rest_over_pairs_ge_0_97=ratio>=.97,
               resume_identical=bool(np.array_equal(whole['coarse'],part['coarse']) and np.array_equal(whole['fine'],part['fine'])
                   and s['ledger']==sc['ledger'] and cb==sc['callbacks']))
    result=dict(status='合格' if all(tests.values()) else '不合格',tests=tests,rest_over_pairs=ratio,
                flight_calls=cb['flight_annihilation'],pairs=cb['pairs'],sources=sources(),
                resume_interrupted=(CACHE/'precheck/precheck_noflight_resume/run.log').exists(),regression_summary_status=json.loads((a/'summary.json').read_text())['status'])
    write(OUT/'precheck.json',result);print(json.dumps(result,ensure_ascii=False))
    if result['status']!='合格':raise SystemExit(1)
    return result


def region_stats(left,right,bins,key,dref):
    """Mean difference over a fixed bin list / Dref, SEM from the 100 batch region means."""
    a=left[key][:,bins].mean(axis=1);b=right[key][:,bins].mean(axis=1)
    return float((a.mean()-b.mean())/dref*100),float(np.hypot(a.std(ddof=1),b.std(ddof=1))/np.sqrt(len(a))/dref*100)


def analyse(names):
    lock=json.loads((P2/'pilot_lock.json').read_text())['windows']['6.0'];dref=lock['Dref']
    if lock['build']!=BUILD:raise RuntimeError('build-up bins differ from the P2 lock')
    data={n:dict(np.load(OUT/'egs5'/n/'result.npz')) for n in names}
    summaries={n:json.loads((OUT/'egs5'/n/'summary.json').read_text()) for n in names}
    if any(s['status']!='合格' for s in summaries.values()):raise RuntimeError('a run failed its own checks')
    for n in names:
        mode,noflight,seed=RUNS[n];c=json.loads((OUT/'egs5'/n/'config.json').read_text());v=data[n];q=summaries[n]
        if not ((c['N'],c['seed'],c['IUNRST'],c['noflight'])==(N,seed,mode,noflight)==(q['N'],q['seed'],q['IUNRST'],q['noflight'])
                and (int(v['N']),int(v['seed']),float(v['E']))==(N,seed,E) and v['coarse'].shape==(100,150) and v['fine'].shape==(100,600)
                and abs(q['ledger'][0]-N*E)<=1e-12*N*E and c['code_sha256']==sha(CODE/'p2b_egs.f')):
            raise RuntimeError('saved run is not the registered job: '+n)
    deep=lock['deep'];blocks=[deep[i:i+25] for i in range(0,100,25)]+[deep[100:]]
    regions=[('buildup','fine',BUILD),('window','coarse',lock['near'])]+[(f'deep_{b[0]}_{b[-1]}','coarse',b) for b in blocks]
    pairs=[('rep','p2b_Cminus','p2b_Splus')]
    if 'p2b_Sminus' in names:pairs+=[('ann','p2b_Sminus','p2b_Splus'),('rad','p2b_Cminus','p2b_Sminus')]
    out={}
    for tag,l,r in pairs:
        out[tag]={name:dict(zip(('delta_pct_Dref','SEM_pct_Dref'),region_stats(data[l],data[r],bins,key,dref))) for name,key,bins in regions}
        out[tag]['pair']=f'{l} - {r}'
    d,s=out['rep']['buildup']['delta_pct_Dref'],out['rep']['buildup']['SEM_pct_Dref']
    reproduced=bool(d>0 and d-1.96*s>0)
    result=dict(Dref_Gy_cm2=dref,N=N,E=E,differences=out,reproduced=reproduced,precision_target_met=bool(s<=.15),
        P2_value=dict(delta_pct_Dref=1.052,SEM_pct_Dref=.286,source='posthoc_v2/trial_before_criteria.json'),
        total_deposit_MeV={n:summaries[n]['ledger'][1] for n in names},
        total_deposit_ratio_to_Splus={n:summaries[n]['ledger'][1]/summaries['p2b_Splus']['ledger'][1] for n in names},
        P2_prototype_minus_main={r['pair'].split(' - ')[0]:dict(delta_pct_Dref=r['build_mean_pct'],SEM_pct_Dref=r['build_SEM_pct'])
            for r in json.loads((P2/'posthoc_v2/trial_before_criteria.json').read_text())['rows'] if r['E']==6.0 and not r['pair'].startswith('EGS5')},
        callbacks={n:summaries[n]['callbacks'] for n in names})
    if 'ann' in out:
        for tag in ('ann','rad'):
            v=out[tag]['buildup'];out[tag]['detected']=bool(abs(v['delta_pct_Dref'])-1.96*v['SEM_pct_Dref']>0)
        result['identity_residual']=out['ann']['buildup']['delta_pct_Dref']+out['rad']['buildup']['delta_pct_Dref']-d
    write(OUT/'c_analysis.json',result)
    return result


def report(r):
    f=lambda v:f"{v['delta_pct_Dref']:+.3f} ± {v['SEM_pct_Dref']:.3f}"
    d=r['differences'];lines=['# P2b 追補1: 6 MeVのEGS5の設定間の差の調査','',
        'この調査はP2の判定も、P2bの事後解析の結果も変えない。対象の差は、12の領域の中から事後に見つけたものである。値はすべて % Dref（固定）、±はSEM。','',
        f"- 再現の確認（C− − S＋、ビルドアップ）: {f(d['rep']['buildup'])}。P2の値は +1.052 ± 0.286。判定: {'再現した' if r['reproduced'] else '指定した精度の独立な再計算では差を確認できなかった'}。精度の目標（SEM ≤ 0.15）: {'達成' if r['precision_target_met'] else '未達'}。"]
    if 'ann' in d:
        lines+=[f"- 飛行中の消滅をなくした効果（S− − S＋）: {f(d['ann']['buildup'])}。{'寄与を検出' if d['ann']['detected'] else 'この精度では検出できない'}。",
                f"- 消滅なしの条件で放射損失を連続にした効果（C− − S−。PEGSの表の違いを含む）: {f(d['rad']['buildup'])}。{'寄与を検出' if d['rad']['detected'] else 'この精度では検出できない'}。",
                f"- 恒等式の残差（2つの和 − 再現の差）: {r['identity_residual']:.2e}。"]
    lines+=['- 全沈着エネルギーの比（S＋を1として）: '+'、'.join(f'{n} {v:.5f}' for n,v in r['total_deposit_ratio_to_Splus'].items())+'。',
            '- 参考（P2、ビルドアップ、試作 − EGS5主）: '+'、'.join(f"{n} {f(v)}" for n,v in r['P2_prototype_minus_main'].items())+'。']
    lines+=['','| 領域 | '+' | '.join(d[t]['pair'] for t in d)+' |','|---|'+'---|'*len(d)]
    for name in [k for k in d['rep'] if k not in ('pair','detected')]:
        lines.append(f'| {name} | '+' | '.join(f(d[t][name]) for t in d)+' |')
    lines+=['','「再現」は正の向きの差を確認したという意味で、P2の値の大きさ（約1.05%）の再現を検定したものではない。2つの寄与の判定はそれぞれ名目95%で、2つをまとめた誤検出率は制御していない。2つの差はS−を共有するので独立ではない。',
            '','S−は標準のEGS5にない人工的な設定である。飛行中の消滅の候補を無効な事象にし、その地点でエネルギーヒンジを引き直す（EGS5は離散事象の前にヒンジを打ち切るため、消滅の断面積を0にした輸送と数値的に同一ではない）。S− − S＋にはこの数値的な差が含まれうる。C− − S−にはPEGSの表の当てはめの違いが分離されずに残る。原因が確定したとは主張しない。','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))


def main():
    if '--precheck' in sys.argv:precheck();return
    if '--run' in sys.argv:
        pre=json.loads((OUT/'precheck.json').read_text())
        if pre['status']!='合格' or pre.get('sources')!=sources():raise SystemExit('precheck not passed for the current sources')
        for name in ('p2b_Splus','p2b_Cminus'):
            mode,noflight,seed=RUNS[name];run(name,N,seed,mode,noflight)
        r=analyse(['p2b_Splus','p2b_Cminus'])
        if r['reproduced']:
            mode,noflight,seed=RUNS['p2b_Sminus'];run('p2b_Sminus',N,seed,mode,noflight)
            r=analyse(list(RUNS))
        report(r);print(json.dumps({k:r[k] for k in ('reproduced','precision_target_met')},ensure_ascii=False))

if __name__=='__main__':
    main()
