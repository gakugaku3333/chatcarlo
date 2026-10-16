"""P5h acceptance: native boundary searches, exact histories and copied paths."""
import ctypes as ct
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p5b_core as core, p5h_core as h, p5h_tables as maps


@pytest.fixture(scope='module')
def data():
    from prototypes.mv_electron.gpu.p5a import load_inputs
    return load_inputs(full=True)


@pytest.fixture(scope='module')
def checked(data,tmp_path_factory):
    from prototypes.mv_electron.gpu.p5h import CPU
    return CPU(tmp_path_factory.mktemp('p5h_checked'),data,check=True)


def test_gate0_cpu(checked,tmp_path):
    from prototypes.mv_electron.gpu.p5h import gate0
    report=gate0(checked,tmp_path)
    assert report['status']=='pass'
    assert len(report['checks'])==18
    assert all(v['calls']>0 and v['fallbacks']>0 and v['mismatches']==0
               for v in report['counters']['sites'].values())


def test_cpu_2000(data,tmp_path):
    from prototypes.mv_electron.gpu.p5h import local
    report=local(tmp_path,data)
    assert report['status']=='pass'
    assert all(report['checks'][k]['status']=='pass' for k in ('ordinary','checked','counters'))
    assert not (tmp_path/'gate1.json').exists()


@pytest.mark.parametrize('name',maps.NAMES)
def test_grid_validation(data,name):
    fixture={k:v.copy() for k,v in data.items()}
    key='rates' if name=='rates' else name
    grid=fixture[key][:,0] if key=='rates' else fixture[key]
    grid[1]=np.nan
    with pytest.raises(ValueError):maps.generate(fixture)
    grid[1]=grid[0]-1
    with pytest.raises(ValueError):maps.generate(fixture)


def test_duplicates_and_start(checked,data):
    # Explicit duplicate-run starts/ends, with a suffix start inside a duplicate run.
    raw=np.array([.01,.01,.1,.1,.1,1.,1.])
    x=np.array([np.nan,-np.inf,np.inf,-0.,0.,.01,.1,1.,2.])
    for start in (0,1,3,len(raw)):
        out=checked.lookup(raw,x,0,8,start=start,narrow=True)
        assert np.array_equal(out[:,0],out[:,1])
        # Python oracle handles NaN differently; native legacy contract returns start.
        assert out[0,0]==start
        assert out[1,0]==start and out[2,0]==len(raw)


def _functions(text):
    result={}
    for m in re.finditer(r'P5_FN [^\n]*?\b(\w+)\([^;]*?\)\s*\{',text):
        start=m.end();depth=1;i=start
        while depth:
            depth+=(text[i]=='{')-(text[i]=='}');i+=1
        result.setdefault(m[1],[]).append(text[m.start():i])
    return result


def test_reachable_search_paths_and_transition_tokens():
    # Resolve the CPU translation unit's P5h transport graph, including frozen helpers.
    texts={n:(core.HERE/n).read_text() for n in
        ('p5h_cpu.cpp','p5h_kernel.h','p5h_lookup.h','p5_kernel.h','p5a0_common.h','p5_rng.h')}
    functions=_functions('\n'.join(texts.values()))
    pending=['p5h_history'];seen=set();edges=[]
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for body in functions.get(name,[]):
            body=re.sub(r'//[^\n]*','',body)
            for call in re.findall(r'\b(\w+)\s*\(',body[body.index('{')+1:]):
                if call in functions:
                    edges.append((name,call));pending.append(call)
    forbidden={'lower_bound','interp','quantities','p5_log_interp','p5_pair_interp',
               'p5_table_rates','p5_bracket','p5_gs_mu_cached','p5_collision',
               'p5_quantities','p5_sample_k','p5_coefficients'}
    # The sole deliberately retained old path is ks -> bracket -> lower_bound.
    assert {(a,b) for a,b in edges if b in forbidden}=={
        ('p5h_gs_mu_cached','p5_bracket'),('p5_bracket','lower_bound')}
    assert h.OLD_ALLOWLIST==('p5_bracket(t->ks,t->nk,K,&j,&b)',)
    assert all(name in seen for name in ('p5h_interp','p5h_quantities','p5h_find',
               'p5h_coefficients','p5h_gs_mu_cached','p5h_table_rates'))
    # Reverse names in the transition section: every token must equal P5g.
    new=texts['p5h_kernel.h'];new=new[new.index('enum P5hPhase'):]
    new=new.replace('P5H','P5G').replace('P5h','P5g').replace('p5h_','p5g_')
    for name in ('coefficients','quantities','collision','table_rates','gs_mu_cached','sample_k'):
        new=new.replace('p5g_'+name+'(', 'p5_'+name+'(')
    new=new.replace('const P5gTables *t','const P5Tables *t')
    old=(core.HERE/'p5g_kernel.h').read_text();old=old[old.index('enum P5gPhase'):]
    assert re.findall(r'\w+|[^\s]',new)==re.findall(r'\w+|[^\s]',old)
    assert '#include "p5h_kernel.h"' in texts['p5h_cpu.cpp']
    source=h.source()
    assert '#include "p5' not in source
    assert source.count('struct P5Tables {')==1
    gpu=(core.HERE/'p5h_gpu.cu').read_text()
    assert gpu.index('unsigned hm=__ballot_sync')<gpu.index('if(pm&lane)')
    assert '__activemask' not in gpu
    assert 'p5h_find(x,es,n,1,1,start,t,4,4+pair)' in texts['p5h_kernel.h']
    core.fingerprints()


def test_generated_identity_and_gate1_rejection(data,tmp_path):
    identity=h.identity(data)
    assert set(identity['generated'])==set(maps.generate(data))
    assert identity['source_sha256']==core.sha(h.source().encode())
    assert not any(n.endswith('.py') for n in identity['sources'])
    r=dict(status='pass',identity=identity,n=2000)
    path=tmp_path/'local.json';path.write_text(json.dumps(r))
    with pytest.raises(core.Stop):h.require_gate1(path,data)
    r=dict(status='pass',identity=identity,batches=100,checks={str(i):{} for i in range(100)})
    path.write_text(json.dumps(r))
    with pytest.raises((core.Stop,KeyError)):h.require_gate1(path,data)
    changed={k:v.copy() for k,v in data.items()};changed['thresholds'][0]=np.nextafter(changed['thresholds'][0],np.inf)
    assert h.identity(changed)!=identity


def test_counter_rejection():
    a=np.ones((len(maps.SITES),3),np.uint64);a[:,2]=0
    assert h.counters(a)['status']=='pass'
    a[1,2]=1
    assert h.counters(a)['status']=='fail'
    a[1,2]=0;a[2,0]=0
    assert h.counters(a)['status']=='fail'


def test_notebook_and_imports(tmp_path):
    nb=json.loads((core.HERE/'p5h_colab.ipynb').read_text())
    text='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert 'run_all(' in text and 'APPROVE_' not in text and 'CODE_COMMIT' not in text
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<cell>','exec')
    subprocess.run([sys.executable,'-c',
        'import sys; from prototypes.mv_electron.gpu import p5h_core,p5h_tables,p5h_colab; '
        'assert "numba" not in sys.modules; assert "cupy" not in sys.modules'],check=True)
    from prototypes.mv_electron.gpu.p5h import export_sources
    import zipfile
    with zipfile.ZipFile(export_sources(tmp_path)) as z:
        assert 'prototypes/mv_electron/gpu/p5h_kernel.h' in z.namelist()


def test_helper_arithmetic_tokens_unchanged():
    """Mask only permitted search/signature edits, then compare every remaining token."""
    old=_functions((core.HERE/'p5a0_common.h').read_text()+'\n'+(core.HERE/'p5_kernel.h').read_text())
    new=_functions((core.HERE/'p5h_kernel.h').read_text())
    names=('interp','quantities','p5_collision','p5_log_interp','p5_pair_interp',
           'p5_coefficients','p5_table_rates','p5_bracket','p5_gs_mu_cached',
           'p5_quantities','p5_sample_k')
    def tokens(s):
        s=re.sub(r'//[^\n]*','',s)
        return re.findall(r'\w+|[^\s]',s)
    for name in names:
        target='p5h_'+(name[3:] if name.startswith('p5_') else name)
        index=1 if name=='p5_quantities' else 0
        a=old[name][0];b=new[target][index]
        a=a[a.index('{')+1:-1];b=b[b.index('{')+1:-1]
        # Names and context parameters are the only permitted non-search edits.
        for n in names:
            nn='p5h_'+(n[3:] if n.startswith('p5_') else n)
            # Both quantities overloads map to a common token.
            label='quantities' if n in ('quantities','p5_quantities') else n
            a=re.sub(r'\b'+n+r'\b',label,a)
            b=re.sub(r'\b'+nn+r'\b',label,b)
        b=b.replace(',ctx,1)',')').replace(',t,0)',')').replace(',t,1)',')')
        b=b.replace('t->watercols,t)', 't->watercols)').replace('&h,&c,t)', '&h,&c)').replace(',raw,t)', ',raw)')
        # lower_bound expressions become a certified candidate search.
        a=re.sub(r'lower_bound\([^;]*?\)', 'SEARCH', a)
        b=re.sub(r'p5h_find\([^;]*?\)', 'SEARCH', b)
        if name in ('p5_log_interp','p5_pair_interp'):
            a=re.sub(r'int lo=(?:0|start),hi=n;\s*while\(lo<hi\) \{[^}]*\}', 'int lo=SEARCH;', a)
        if name=='p5_pair_interp':
            a=a.replace('int start=0;while(start<n && es[start]<=th)++start;','int start=START;')
            b=b.replace('int start=t->pair_start[pair];','int start=START;')
        if name=='p5_table_rates':
            # shift/n setup is retained verbatim; lo/hi initialization moves into search.
            a=a[:a.index(',lo=0,hi=n;')]+a[a.index('    int i=lo-1;'):]
            b=b[:b.index(';')]+b[b.index('    int i=lo-1;'):]
        assert tokens(a)==tokens(b),name


def test_p5g_reference_wrapper(monkeypatch):
    from prototypes.mv_electron.gpu import p5h_colab as colab
    def init(self,cp,data,**kwargs):
        self.baseline=False
        self.entries=('p5g_run_kernel','p5g_per_history_kernel')
    monkeypatch.setattr(colab.previous.GPU,'__init__',init)
    reference=colab.GPU(None,{},baseline=True)
    assert reference.baseline is True
    assert reference.entries[0]=='p5g_run_kernel'
    assert not hasattr(reference,'counts')
