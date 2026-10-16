"""P5j CPU acceptance; no saved-cache batches or GPU execution."""
import json
import re
import subprocess
import sys
import zipfile
from types import SimpleNamespace
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p5b_core as core, p5j, p5j_core as j

@pytest.fixture(scope='module')
def data():
    from prototypes.mv_electron.gpu.p5a import load_inputs
    return load_inputs(full=True)


def test_cpu_gate0_and_p5i_2000(data,tmp_path):
    report=p5j.local(tmp_path,data)
    assert report['status']=='pass'
    for name in ('ordinary','checked','counters'):
        assert report['checks'][name]['status']=='pass'
    for name in ('ordinary','checked'):
        gate=report['gate0'][name]
        assert gate['randoms']==1000000
        assert gate['checks']['2']['inputs']>1000000
        assert gate['identity']['generated']==report['identity']['generated']
    assert all(v>0 for v in report['gate0']['checked']['counters']['sites'].values())
    assert report['checks']['counters']['sites']['C_fallback']==0
    assert not (tmp_path/'gate1.json').exists()


def test_bit_contract_and_damaged_constant(data,tmp_path):
    cpu=p5j.CPU(tmp_path,data,check=True)
    good=cpu.compare(0,np.array([[1.,0.]]))
    assert core.bits(good[:,:20].copy(),good[:,20:].copy())
    cpu.logs[-3]=.25
    bad=cpu.compare(0,np.array([[1.,0.]]))
    assert not core.bits(bad[:,:20].copy(),bad[:,20:].copy())
    # NaN payload/sign are preserved by the special C comparison.
    x=p5j.special_vectors();out=cpu.compare(8,x)
    assert core.bits(out[:,:20].copy(),out[:,20:].copy())
    out=np.zeros((4,40));cpu.lib.p5j_contract(out)
    assert core.bits(out[:,:20].copy(),out[:,20:].copy())


def functions(text):
    result={}
    for m in re.finditer(r'P5_FN [^\n]*?\b(\w+)\([^;]*?\)\s*\{',text):
        i=m.end();depth=1
        while depth:depth+=(text[i]=='{')-(text[i]=='}');i+=1
        result.setdefault(m[1],[]).append(text[m.start():i])
    return result


def tokens(s):
    return re.findall(r'\w+|[^\s]',re.sub(r'//[^\n]*','',s))


def test_only_authorized_transport_sites_change():
    old=(core.HERE/'p5i_kernel.h').read_text()
    new=(core.HERE/'p5j_kernel.h').read_text()
    expected=old[old.index('enum P5iPhase'):].replace('P5I','P5J').replace('P5i','P5j').replace('p5i_','p5j_')
    expected=expected.replace('double q[4];p5j_quantities(ev,t,q);','double Lsrc=ev,Lval=log(ev); double q[4];p5j_quantities(ev,t,q,Lsrc,Lval);')
    expected=expected.replace('p5_hard(ev,t,&hardrate,&hardstop);S=p5_collision(ev,t)','p5j_hard(ev,t,&hardrate,&hardstop);S=p5j_collision(ev,t,Lsrc,Lval)')
    assert tokens(expected)==tokens(new[new.index('enum P5jPhase'):])
    # Preserve original log expressions in non-update quantities callers.
    assert 'p5j_quantities(T,t,q);double envelope' in new
    assert 'p5j_quantities(T,t,q);hn+=' in new
    terms=(core.HERE/'p5j_terms.h').read_text()
    assert 'double arg=-d/tau;' in terms
    assert 'double L=(arg==-0.5)?K2:log1p(arg);' in terms
    assert 'memcpy(&a,&arg,8);memcpy(&b,&Lsrc,8);' in terms
    assert '!isnan(Lsrc) && a==b' in terms
    assert 'reuse?Lval:log(arg)' in terms
    assert 'p5j_primitive0(.5,a,b,t)-p5_primitive0(x,a,b)' in terms
    assert 'p5j_primitive1(.5,a,b,t)-p5_primitive1(x,a,b)' in terms
    frozen=functions((core.HERE/'p5_kernel.h').read_text())
    rewritten=functions(terms)
    for n in ('primitive0','primitive1'):
        expected=frozen['p5_'+n][0].replace('p5_'+n,'p5j_'+n).replace('double b) {','double b,const P5jTables *t) {')
        expected=expected.replace('log(x/(1-x))','p5j_constant(t,0,0)').replace('log(x)','p5j_constant(t,1,1)').replace('log1p(-x)','p5j_constant(t,2,2)')
        assert tokens(expected)==tokens(rewritten['p5j_'+n][0])
    assert 'p5j_log(t>cut?t:cut,Lsrc,Lval,ctx,5)' in new
    assert 'p5j_log(t,Lsrc,Lval,ctx,7)' in new
    assert 'interp(p5j_log(T,Lsrc,Lval,t,9)' in terms
    assert '#include "p5' not in j.source()
    core.fingerprints()


def test_identity_and_gate_rejection(data,tmp_path):
    ident=j.identity(data,generated={'fixture':'hash'})
    assert ident['nspec']==20 and ident['generated']=={'fixture':'hash'}
    assert not any(n.endswith('.py') for n in ident['sources'])
    path=tmp_path/'gate1.json';core.write_json(path,dict(status='pass',identity=ident,n=2000))
    with pytest.raises(core.Stop):j.require_gate1(path,data)
    values=np.zeros((20,2),np.uint64);values[9:,0]=1
    assert j.counters(values,gate0=True)['status']=='pass'
    values[13,0]=0
    assert j.counters(values,gate0=True)['status']=='fail'
    assert j.counters(values)['status']=='pass'
    from prototypes.mv_electron.gpu import p5j_colab as colab
    gpu=SimpleNamespace(launches={},generated={'fixture':'hash'})
    assert colab.gate3_identity(data,{},gpu)['nspec']==20
    assert colab.gate3_identity(data,{},gpu)['generated']==gpu.generated
    source=(core.HERE/'p5j_colab.py').read_text()
    assert 'baseline_i.GPU.__init__' in source
    assert "('P5i','p5i_run_kernel'),('P5j','p5j_run_kernel')" in source
    assert "[('P5i',old),('P5j',new)]" in source
    assert "text=baseline_core.source() if variant=='P5i' else h.source()" in source


def test_notebook_import_export(tmp_path):
    nb=json.loads((core.HERE/'p5j_colab.ipynb').read_text())
    assert 'run_all(' in str(nb)
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<cell>','exec')
    subprocess.run([sys.executable,'-c','import sys; from prototypes.mv_electron.gpu import p5j_core,p5j_colab; assert "numba" not in sys.modules; assert "cupy" not in sys.modules'],check=True)
    with zipfile.ZipFile(p5j.export_sources(tmp_path)) as archive:
        assert 'prototypes/mv_electron/gpu/p5j_terms.h' in archive.namelist()
        assert 'prototypes/mv_electron/gpu/p5i_kernel.h' in archive.namelist()


@pytest.mark.parametrize('failed',['P5i','P5j'])
def test_map_keeps_other_variant(data,tmp_path,monkeypatch,failed):
    from prototypes.mv_electron.gpu import p5j_colab as colab
    monkeypatch.setitem(sys.modules,'cupy',SimpleNamespace())
    monkeypatch.setattr(colab.d,'load_bundle',lambda _: (data,{},{}))
    monkeypatch.setattr(colab.h,'require_gate1',lambda *_:None)
    monkeypatch.setattr(colab,'GPU',lambda *_:SimpleNamespace())
    monkeypatch.setattr(colab,'environment',lambda _: {})
    monkeypatch.setattr(colab,'require_gate2',lambda *_:None)
    monkeypatch.setattr(colab,'require_gate3',lambda *_:None)
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        if cmd[0]!='ncu':raise RuntimeError('fixture line-info failure')
        if '--import' in cmd:return SimpleNamespace(returncode=0,stdout='fixture csv',stderr='')
        variant=cmd[cmd.index('--target')+1]
        return SimpleNamespace(returncode=int(variant==failed),stdout='',stderr='fixture')
    monkeypatch.setattr(colab.subprocess,'run',run)
    monkeypatch.setattr(colab.previous,'parse_source_csv',lambda *a,**k:[])
    monkeypatch.setattr(colab.g,'weighted_map',lambda _: {})
    report=colab.divergence_map('bundle',tmp_path,'gate1','gate2','gate3')
    assert report['status']=='unavailable'
    assert [c[c.index('--kernel-name')+1] for c in calls if '--kernel-name' in c]==['p5i_run_kernel','p5j_run_kernel']
    assert report['profiles'][failed]['capture']=='fail'
    other='P5j' if failed=='P5i' else 'P5i'
    assert report['profiles'][other]['parse']=='pass'
    assert json.loads((tmp_path/'map.json').read_text())==report


def test_remaining_helper_tokens_and_gate0_record(data):
    frozen=functions((core.HERE/'p5_kernel.h').read_text())
    new=functions((core.HERE/'p5j_terms.h').read_text())
    expected=frozen['p5_collision'][0].replace('p5_collision(double T,const P5Tables *t)','p5j_collision(double T,const P5jTables *t,double Lsrc,double Lval)')
    expected=expected.replace('log1p(-d/tau)','p5j_c_log(tau,d,t)').replace('interp(log(T),','interp(p5j_log(T,Lsrc,Lval,t,9),')
    assert tokens(expected)==tokens(new['p5j_collision'][0])
    expected=frozen['p5_hard'][0].replace('p5_hard(double T,const P5Tables *t','p5j_hard(double T,const P5jTables *t')
    expected=expected.replace('p5_primitive0(.5,a,b)','p5j_primitive0(.5,a,b,t)').replace('p5_primitive1(.5,a,b)','p5j_primitive1(.5,a,b,t)')
    assert tokens(expected)==tokens(new['p5j_hard'][0])
    old=functions((core.HERE/'p5i_kernel.h').read_text())
    rewritten=functions((core.HERE/'p5j_kernel.h').read_text())
    for k in range(2):
        expected=old['p5i_quantities'][k].replace('P5i','P5j').replace('p5i','p5j')
        expected=expected.replace('double *raw,const P5jTables *ctx) {','double *raw,const P5jTables *ctx,double Lsrc=NAN,double Lval=NAN) {')
        expected=expected.replace('double *q) {','double *q,double Lsrc=NAN,double Lval=NAN) {')
        expected=expected.replace('log(t>cut?t:cut)','p5j_log(t>cut?t:cut,Lsrc,Lval,ctx,5)').replace('log(t)-p5j_read','p5j_log(t,Lsrc,Lval,ctx,7)-p5j_read')
        expected=expected.replace('&idx,raw,t);','&idx,raw,t,Lsrc,Lval);')
        assert tokens(expected)==tokens(rewritten['p5j_quantities'][k])
    with pytest.raises(core.Stop):j.require_gate0(dict(status='pass'),True)
    record=dict(status='pass',randoms=1000000,
        checks={k:core.verdict(True) for k in ('0','1','2','3','4','5','8','contract')},
        counters=core.verdict(True,sites={k:1 for k in j.TERM_SITES}))
    j.require_gate0(record,True)
    record['counters']['sites']['C_fallback']=0
    with pytest.raises(core.Stop):j.require_gate0(record,True)
