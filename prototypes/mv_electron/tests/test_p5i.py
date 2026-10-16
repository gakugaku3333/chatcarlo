"""P5i isolated acceptance; the saved 100-batch gate is never run here."""
import json
import re
import subprocess
import sys
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p5b_core as core,p5i,p5i_core as i

@pytest.fixture(scope='module')
def data():
    from prototypes.mv_electron.gpu.p5a import load_inputs
    return load_inputs(full=True)


def test_cpu_local_2000_and_gate0(data,tmp_path):
    r=p5i.local(tmp_path,data)
    assert r['status']=='pass'
    assert all(r['checks'][k]['status']=='pass' for k in ('ordinary','checked','counters'))
    gate=json.loads((tmp_path/'gate0_cpu.json').read_text())
    assert gate['randoms_per_interval']==1000
    assert all(v['inputs']>1000 for v in gate['checks'].values())
    assert not (tmp_path/'gate1.json').exists()


def test_other_rate_branch_and_log_ties(data,tmp_path):
    fixture={k:v.copy() for k,v in data.items()}
    # Frozen non-three-column path takes its energy grid from ts.
    fixture['rates']=np.ascontiguousarray(np.column_stack((np.linspace(1,2,len(data['ts'])),np.linspace(.1,.2,len(data['ts'])))))
    fixture['single']=np.ones((len(data['ts']),len(data['us'])+1))
    # Distinct energies with identical rounded logs: preserve both rows.
    a=10.;b=np.nextafter(a,np.inf)
    assert np.log(a)==np.log(b)
    fixture['pes']=np.array([.01,.1,a,b,20.])
    fixture['water']=np.ones((5,7))
    fixture['thresholds']=np.array([1.,2.])
    cpu=p5i.CPU(tmp_path,fixture,check=True)
    r=p5i.gate0(cpu,tmp_path)
    assert r['status']=='pass'
    assert r['auxiliary']['equal']
    assert r['auxiliary']['intermediates_equal']
    # A damaged precomputed value must fail the frozen-function output comparison.
    cpu.logs[int(cpu.maps['offsets'][6])+1]+=1.
    out=cpu.compare(3,np.array([[.1,.025]]))
    assert not core.bits(out[:,:20].copy(),out[:,20:].copy())


def _functions(text):
    result={}
    for m in re.finditer(r'P5_FN [^\n]*?\b(\w+)\([^;]*?\)\s*\{',text):
        j=m.end();depth=1
        while depth:depth+=(text[j]=='{')-(text[j]=='}');j+=1
        result.setdefault(m[1],[]).append(text[m.start():j])
    return result


def test_reachable_transport_and_transition_tokens():
    new=(core.HERE/'p5i_kernel.h').read_text()
    old=(core.HERE/'p5g_kernel.h').read_text()
    section=new[new.index('enum P5iPhase'):].replace('P5I','P5G').replace('P5i','P5g').replace('p5i_','p5g_')
    for n in ('coefficients','quantities','table_rates','gs_mu_cached','sample_k'):
        section=section.replace('p5g_'+n+'(','p5_'+n+'(')
    section=section.replace('const P5gTables *t','const P5Tables *t')
    assert re.findall(r'\w+|[^\s]',section)==re.findall(r'\w+|[^\s]',old[old.index('enum P5gPhase'):])
    funcs=_functions(new+'\n'+(core.HERE/'p5_kernel.h').read_text()+'\n'+(core.HERE/'p5a0_common.h').read_text())
    seen=set();pending=['p5i_history']
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for body in funcs.get(name,[]):
            for call in re.findall(r'\b(\w+)\s*\(',body[body.index('{')+1:]):
                if call in funcs:pending.append(call)
    assert not seen.intersection({'quantities','p5_quantities','p5_log_interp','p5_pair_interp','p5_pair_value','p5_coefficients','p5_table_rates','p5_gs_mu_cached','p5_sample_k'})
    for name in seen:
        if name.startswith('p5i_'):
            body='\n'.join(funcs.get(name,[]))
            assert not re.search(r'log\((?:es|vs|e|v)\[',body)
            assert 'log(t->ts[i+1]/t->ts[i])' not in body
            assert 'log(e1/e0)' not in body
    source=i.source()
    assert '#include "p5' not in source
    assert source.count('struct P5Tables {')==1
    core.fingerprints()


def test_identity_counters_and_rejection(data,tmp_path):
    identity=i.identity(data)
    assert identity['nspec']==20
    assert not any(k.endswith('.py') for k in identity['sources'])
    path=tmp_path/'gate1.json'
    core.write_json(path,dict(status='pass',identity=identity,n=2000))
    with pytest.raises(core.Stop):i.require_gate1(path,data)
    counts=np.ones((9,2),np.uint64);counts[:,1]=0
    assert i.counters(counts)['status']=='pass'
    counts[0,0]=0
    assert i.counters(counts)['status']=='fail'
    counts[0]=[1,1]
    assert i.counters(counts)['status']=='fail'


def test_notebook_imports_and_export(tmp_path):
    nb=json.loads((core.HERE/'p5i_colab.ipynb').read_text())
    text='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert 'run_all(' in text
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<cell>','exec')
    subprocess.run([sys.executable,'-c','import sys; from prototypes.mv_electron.gpu import p5i_core,p5i_colab; assert "numba" not in sys.modules; assert "cupy" not in sys.modules'],check=True)
    import zipfile
    with zipfile.ZipFile(p5i.export_sources(tmp_path)) as z:
        assert 'prototypes/mv_electron/gpu/p5i_logs.h' in z.namelist()


def test_helper_tokens_only_constant_logs_change():
    """Recreate the allowed substitution; every other arithmetic/search token is fixed."""
    frozen=_functions((core.HERE/'p5a0_common.h').read_text()+'\n'+(core.HERE/'p5_kernel.h').read_text())
    new=_functions((core.HERE/'p5i_kernel.h').read_text())
    names=('quantities','p5_log_interp','p5_pair_value','p5_pair_interp','p5_coefficients',
           'p5_table_rates','p5_gs_mu_cached','p5_quantities','p5_sample_k')
    mapping={n:'p5i_'+(n[3:] if n.startswith('p5_') else n) for n in names}
    for name in names:
        expected=frozen[name][0]
        for a,b in mapping.items():expected=re.sub(r'\b'+a+r'\b',b,expected)
        expected=expected.replace('const P5Tables *t','const P5iTables *t')
        if name=='quantities':
            expected=expected.replace('double *raw) {','double *raw,const P5iTables *ctx) {')
            expected=expected.replace('log(e[i+1])','p5i_read(ctx,0,i+1)').replace('log(e[i])','p5i_read(ctx,0,i)')
            for k in range(3):
                suffix='' if k==0 else '+'+str(k)
                expected=expected.replace(f'log(v[5*i{suffix}])',f'p5i_read(ctx,1,3*i+{k})').replace(f'log(v[5*(i+1){suffix}])',f'p5i_read(ctx,1,3*(i+1)+{k})')
        if name in ('p5_log_interp','p5_pair_interp'):
            expected=expected.replace('int stride)','int stride,const P5iTables *ctx)').replace('double th)','double th,const P5iTables *ctx,int pair)')
            expected=re.sub(r'log\(es\[([^]]+)\]\)',r'p5i_read(ctx,5,\1)',expected)
            expected=re.sub(r'log\(vs\[([^]]+)\]\)',lambda m:'p5i_read(ctx,6,'+m[1].replace('*stride','')+')',expected)
            expected=re.sub(r'p5i_pair_value\(([^,]+),es,vs,stride,th\)',r'p5i_pair_value(\1,es,vs,stride,th,ctx,pair)',expected)
        if name=='p5_pair_value':
            expected=expected.replace('double th)','double th,const P5iTables *ctx,int pair)').replace('double q=1-th/es[j];return log(vs[j*stride]/((q*q)*q));','return p5i_read(ctx,7+pair,j);')
        if name=='p5_coefficients':
            expected=expected.replace('t->watercols);','t->watercols,t);').replace('t->thresholds[0]);','t->thresholds[0],t,0);').replace('t->thresholds[1]);','t->thresholds[1],t,1);')
        if name=='p5_table_rates':
            expected=expected.replace('log(e1/e0)','p5i_read(t,2,i)')
            for k in range(2):
                suffix='' if k==0 else '+1'
                expected=expected.replace(f'log(t->rates[i*t->ratecols+shift{suffix}])',f'p5i_read(t,3,2*i+{k})').replace(f'log(t->rates[(i+1)*t->ratecols+shift{suffix}])',f'p5i_read(t,3,2*(i+1)+{k})')
        if name=='p5_gs_mu_cached':expected=expected.replace('log(t->ts[i+1]/t->ts[i])','p5i_read(t,4,i)')
        if name=='p5_quantities':expected=expected.replace('&idx,raw);','&idx,raw,t);')
        actual=new[mapping[name]][int(name=='p5_quantities')]
        def tokens(s):return re.findall(r'\w+|[^\s]',re.sub(r'//[^\n]*','',s))
        assert tokens(actual)==tokens(expected),name


def test_map_failure_still_captures_both_variants(data,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from prototypes.mv_electron.gpu import p5i_colab as colab
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace())
    monkeypatch.setattr(colab.d,'load_bundle',lambda _: (data,{},{}))
    monkeypatch.setattr(colab.h,'require_gate1',lambda *_:None)
    monkeypatch.setattr(colab,'GPU',lambda *_:SimpleNamespace())
    monkeypatch.setattr(colab,'environment',lambda _: {})
    monkeypatch.setattr(colab,'require_gate2',lambda *_:None)
    monkeypatch.setattr(colab,'require_gate3',lambda *_:None)
    calls=[]
    def run(cmd,**kw):
        calls.append(cmd)
        if cmd[0]=='ncu':return SimpleNamespace(returncode=1,stdout='',stderr='fixture failure')
        raise RuntimeError('fixture line-info failure')
    monkeypatch.setattr(colab.subprocess,'run',run)
    report=colab.divergence_map('bundle',tmp_path,'gate1','gate2','gate3')
    assert report['status']=='unavailable'
    assert [v[v.index('--kernel-name')+1] for v in calls if v[0]=='ncu']==['p5g_run_kernel','p5i_run_kernel']
    assert all(v['capture']=='fail' for v in report['profiles'].values())
