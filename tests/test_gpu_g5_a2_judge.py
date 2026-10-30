"""Reject malformed statistical inputs, metadata and ineligible E7 retries."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/validation/gpu/g5'))
import a2_common as common
import a2_judge as j
from chatcarlo.gpu import physics_sets,therapy
from prototypes.mv_electron import analysis_p7 as p7,run_p8e
from prototypes.mv_electron.gpu import p8f_core


def rows():
    return [dict(N=200000,seed=20301831,start=b*200000,physics_set=physics_sets.declaration(therapy.NAME),health={'status':'pass'},sha256='a'*64) for b in range(100)]


@pytest.mark.parametrize('kind',['N','seed','start','health','tag','missing'])
def test_metadata_rejections(kind):
    r=rows();j.metadata(r,'A')
    if kind=='missing':r.pop()
    elif kind=='tag':r[0]['physics_set']['tag']+=1
    elif kind=='health':r[0]['health']['status']='fail'
    else:r[1][kind]+=1
    with pytest.raises(ValueError):j.metadata(r,'A')


@pytest.mark.parametrize('kind',['one','two','mixed','retried','health','none'])
def test_retry(kind):
    s={'difference':[0.]*19,'SEM':[.001]*19}
    if kind in ('one','two','mixed','retried','health'):s['difference'][0]=.011
    if kind=='two':s['difference'][1]=-.011
    if kind=='mixed':s['SEM'][2]=.004
    assert j.retry_eligible(s,kind='stack',healthy=kind!='health',retried=kind=='retried')==(kind=='one')
    p={k:[dict(delta=0.,SEM=.001) for _ in range(n)] for k,n in (('J1',5),('J2',3),('J3',3))}
    if kind in ('one','two','mixed','retried','health'):p['J1'][0]['delta']=1.1
    if kind=='two':p['J3'][1]['delta']=.11
    if kind=='mixed':p['J2'][0]['SEM']=.4
    assert j.retry_eligible(p,kind='p7',healthy=kind!='health',retried=kind=='retried')==(kind=='one')


def stack_grid():
    from prototypes.mv_electron.run_p8e import LAYERS,layout
    return layout(LAYERS,[1.,.26,1.85,11.35])


def test_stack_inputs():
    g=stack_grid();x=np.ones((100,220));mass=g['mass_g_cm2']
    assert j.stack_dose(x,20000000,mass,g).shape==(100,220)
    for n,v,m in ((20000001,x,mass),(20000000,x[:,:19],mass),(20000000,x,mass*2)):
        with pytest.raises(ValueError):j.stack_dose(v,n,m,g)
    with pytest.raises(ValueError,match='Dref'):j.stack_judge(x,20000000,mass,g,x,dref=.1)


def test_p7_rejections():
    with pytest.raises(ValueError):j.p7_expand(np.ones((100,19)),20000000)
    x=np.zeros((100,3420));ref={k:np.ones(s) for k,s in p7.shapes(p7.spec(2.5)).items()};ref['N']=20000000
    for kw in ({'N':20000001},{'volumes':np.ones(15)},{'dref':1.}):
        n=kw.pop('N',20000000)
        with pytest.raises(ValueError):j.p7_judge(x,n,'A',ref,**kw)
    # Mean without crossing and a crossing absent in only some bootstrap resamples.
    for batches in (np.ones((100,80)),np.r_[np.tile(np.r_[np.ones(8),np.full(72,.3)],(50,1)),np.tile(np.r_[np.ones(8),np.full(72,.7)],(50,1))]):
        stats=dict(J3=[dict(proto=p7.bootstrap_position(batches,.25,8),egs={'position':1.,'missing':0})])
        with pytest.raises(ValueError,match='50%'):p7.stop_on_missing(stats)


def test_reference_self_comparisons():
    j.reference_hashes(4);j.reference_hashes(5)
    for condition in 'ABC':
        with np.load(common.ROOT/f'docs/validation/mv/p7/egs5/main_egs_{condition}/result.npz') as z:ref={k:z[k] for k in z.files}
        regions=np.column_stack([ref[k].reshape(100,-1) for k in ('axis_coarse','axis_fine','slab_coarse','profile','profile_fine')])
        r=j.p7_judge(regions,int(ref['N']),condition,ref)
        assert all(v['delta']==0 for key in ('J1','J2','J3') for v in r['statistics'][key])
    g=stack_grid();reference,_=run_p8e.load_egs_result(common.ROOT/'docs/validation/mv/p8e',Path.home()/'.cache/chatcarlo/p8e/egs',g,[1.,.26,1.85,11.35])
    r=j.stack_judge(reference*200000*g['mass_g_cm2'],20000000,g['mass_g_cm2'],g,reference,dref=p8f_core.DREF)
    assert np.max(np.abs(r['difference']))<1e-15


def test_saved_gpu_regression():
    assert set(j.saved_regression())=={'A','B','C','stack'}


def test_retry_invalid_sem():
    p={k:[dict(delta=0.,SEM=.001) for _ in range(n)] for k,n in (('J1',5),('J2',3),('J3',3))}
    p['J1'][0]['delta']=1.1
    for value in (float('nan'),float('inf'),-.1):
        p['J3'][1]['SEM']=value
        assert not j.retry_eligible(p,kind='p7')
