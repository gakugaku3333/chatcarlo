"""P5f exact derivation, CPU equivalence and rejection gates."""
import ctypes as ct
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from prototypes.mv_electron.gpu import p5f_derive as f, p5b_core as core

EXPECTED = {
 'B':'e156b48ddf834573395531bead82f1b88c25010fa5b5d09b00eb6dc9dc891b17',
 'T':'1987301031308c9841ef4b5d23d35654933b6f490f8a95392aa6ca93d6845cf9',
 'W':'aa1987c215901335b0ae6a929c12ff530375c9a4d786290f54c00783ae33900f',
 'TW':'e78c11dae53617994d461ef0581bdf40d9349dcc4b423721739c2a3308bf821c'}


@pytest.mark.parametrize('c',f.CONDITIONS)
def test_derivation(c):
    text,record=f.derive(c)
    assert record['source_sha256']==EXPECTED[c]
    assert tuple(record['options'])==core.OPTIONS
    assert record['replacements']
    assert 's->p5f_base + ((q) % t->nu)' in text
    if 'W' in c:
        assert 'double hs[150]' not in text
        assert 'double batches[2],totals' in text


@pytest.mark.parametrize('text',['absent','twice twice'])
def test_replace_reject(text):
    with pytest.raises(ValueError):
        f.replace_once(text,'twice','x')


def test_cpu(tmp_path):
    from prototypes.mv_electron.gpu import p5a, p5d_core as d
    data=p5a.load_inputs(full=True)
    baseline=d.CPU(tmp_path,data)
    ref=baseline.run(2000,core.SEED)
    for c in ('B','W'):
        source=tmp_path/(c+'.cpp')
        source.write_text(f.derive(c,cpu=True)[0])
        library=tmp_path/(c+'.so')
        subprocess.run(['clang++',*d.FLAGS,str(source),'-o',str(library)],check=True,capture_output=True)
        cpu=object.__new__(d.CPU)
        cpu.data,cpu.tables=data,baseline.tables
        cpu.lib=ct.CDLL(str(library))
        cpu.lib.p5a_run.argtypes=baseline.lib.p5a_run.argtypes
        cpu.lib.p5a_run.restype=None
        out=cpu.run(2000,core.SEED)
        if c=='B':
            keys=('batches','fine_batches','totals','maxima','violations','counts','nspec','step_batches','brems','maxdepth')
            assert all(core.bits(out[k],ref[k]) for k in keys)
        else:
            assert all(core.bits(out[k],ref[k]) for k in ('counts','nspec','step_batches'))
            assert not out['violations'].any()
            assert np.max(out['maxima'])<=1e-9
            assert np.isclose(out['batches'].ravel()[:200:2].sum(),out['totals'][1],rtol=1e-12)


@pytest.mark.parametrize('bad',['nan','inf','nan_count','step','stack','energy','checksum','spectrum','maxima'])
def test_p3(bad):
    out=core.empty_output(core.run_shapes(20))
    out['nspec'][0,0]=100000
    assert f.health(out,100000)
    if bad in ('nan','inf'): out['totals'][0]=float(bad)
    elif bad=='nan_count': out['nan_count'][...]=1
    elif bad in ('step','stack'): out['counts'][5 if bad=='step' else 6]=1
    elif bad in ('energy','checksum'): out['violations'][0 if bad=='energy' else 1]=1
    elif bad=='spectrum': out['nspec'][0,0]-=1
    else: out['maxima'][1]=1e-8
    assert not f.health(out,100000)


def test_isolation_notebook():
    subprocess.run([sys.executable,'-c',"from prototypes.mv_electron.gpu import p5f_colab; import sys; assert not any(k in ('numba','cupy') or k.endswith('.p5a') for k in sys.modules)"],check=True)
    book=json.loads((f.HERE/'p5f_colab.ipynb').read_text())
    for cell in book['cells']:
        if cell['cell_type']=='code':
            compile(''.join(cell['source']),'<notebook>','exec')
            assert cell['outputs']==[]
    first=''.join(book['cells'][1]['source'])
    # Empty until Claude pins the implementation commit; afterwards exactly 40 hex digits.
    assert re.search(r'CODE_COMMIT = "([0-9a-f]{40})?"', first)
    assert first.index("'fetch'")<first.index("'checkout'")


@pytest.mark.parametrize('copies',[0,2])
def test_derive_rejects_drift(monkeypatch,copies):
    original=Path.read_text
    needle='s->seed=seed;s->history=history;s->calls=0;'
    def read(path,*args,**kwargs):
        text=original(path,*args,**kwargs)
        return text.replace(needle,needle*copies) if path.name=='p5_rng.h' else text
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(ValueError,match='exactly once'):
        f.derive('B')
