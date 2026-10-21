"""Read-only P8e inputs; all builds use pytest temporary directories."""
import numpy as np
import pytest
from numba import njit, _helperlib
from prototypes.mv_electron import photon_transport_p8e as k
from prototypes.mv_electron.gpu import p8f as p, p8f_core as core

@pytest.fixture(scope='module')
def data():
    import numba
    assert numba.__version__.startswith('0.66.'),'Gate 0b is registered for Numba 0.66'
    return p.inputs()

@pytest.fixture(scope='module')
def cpu(data,tmp_path_factory):
    m,i,e,c=data
    return p.CPU(tmp_path_factory.mktemp('p8f'),m,densities=i['densities'],edges=e,cum=c)

@njit
def reference(which,x,m,seed):
    np.random.seed(seed);out=np.zeros((len(x),128))
    for i in range(len(x)):
        T=x[i,0]
        if which==0:out[i,0]=k.material_collision(T,m)
        elif which==1:
            a,b=k.material_hard(T,m);out[i,0]=a;out[i,1]=b
        elif which==2:out[i,:4]=k.material_coefficients(T,m)
        elif which==3:out[i,:4]=np.asarray(k.material_quantities(T,m))
        elif which==4:out[i,0]=k.material_shape(T,x[i,1],m.z,m.nj)
        elif which==5:out[i,0]=k.material_annihilation_rate(T,m.ne)
        elif which==6:out[i,:len(m.z)*5]=k.shell_distribution(T,m).ravel()
        elif which==7:
            a,b=k.select_shell(T,m);out[i,0]=a;out[i,1]=b
        elif which==8:
            a,b=k.material_sample_k(T,m);out[i,0]=a;out[i,1]=b
        elif which==9:
            photons,local,trace=k.relax_vacancy(int(T),int(x[i,1]),m,bool(x[i,2]));out[i,0]=len(photons);out[i,1]=len(trace);out[i,2]=local
            out[i,3:3+len(photons)*4]=photons.ravel();out[i,35:59]=-1
            out[i,35:35+len(trace)*3]=trace.ravel()
        elif which==10:
            a,b=k.table_rates(T,m.gs_T,m.rates);out[i,0]=a;out[i,1]=b
        elif which==11:
            a,b=k.table_rates(T,m.gs_T,m.rates);out[i,0]=k.gs_mu_cached(T,x[i,1],m.gs_T,m.gs_K,m.gs_u,m.gs_y,a,b)
        elif which==12:out[i,:8]=np.asarray(k.compton(T,0.,0.,1.))
        elif which==13:out[i,:10]=k.pair(T,0.,0.,1.,m.dl,m.bpar,m.delcm,m.delpos)
        elif which==14:out[i,:3]=np.asarray(k.annihilation())
        elif which==15:out[i,:8]=k.photons(T,0.,0.,1.)
        elif which==16:out[i,:23]=k.initial(T,0.,0.,0.,1.,0,0,.05,k.rule_value(T,.025,1,np.array([.02,1.]),np.array([.1,.05,.025])))
        elif which==17:
            a=x[i,:23].copy();out[i,23:28]=k.material_boundary(a,int(x[i,23]),int(x[i,24]),m,x[i,25],x[i,26],x[i,27],x[i,28]);out[i,:23]=a
        elif which==18:out[i,0]=np.sum(x[i])
        elif which==20:out[i,:3]=np.asarray(k.direction(T,0.,0.,1.,x[i,1]))
        elif which==21:out[i,0]=k.new_remaining(bool(T))
        elif which==22:out[i,0]=k.advance_remaining(T,x[i,1],x[i,2])
        elif which==23:
            a=x[i,:23].copy();k.energy_init(a,.05);out[i,:23]=a
        elif which==26:
            touch=np.array([[int(T),int(x[i,1])]])
            out[i,0]=k.touched_sum(x[i,2:],touch,0)
        elif which==25:
            event=6
            for j in range(1,7):
                if x[i,j]<=T+1e-13:event=j-1;break
            out[i,0]=event
        elif which==24:
            r=T;last=-1
            for j in range(1,x.shape[1]):
                if x[i,j]>0:
                    last=j-1
                    if r<x[i,j]:break
                    r-=x[i,j]
            out[i,0]=last//5 if last>=0 else -1;out[i,1]=last%5 if last>=0 else -1
        elif which==19:out[i,:3]=np.asarray(k.rotate(x[i,0],x[i,1],x[i,2],x[i,3],x[i,4]))
    return out

@njit
def next_draws():
    return np.random.random(16)

def compare_probe(cpu,which,x,m,mid):
    x=np.ascontiguousarray(x,float)
    if x.ndim==1:x=x[:,None]
    expected=reference(which,x,m,core.SEED0)
    index,words=_helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr())
    subsequent=next_draws()
    actual,state,next=cpu.probe(which,x,mid)
    core.bitwise(actual,expected,f'function {which} material {mid}')
    core.bitwise(state[:624],np.array(words,np.int64),'MT words')
    assert state[624]==index
    check=np.random.RandomState(core.SEED0);check.random_sample(int(state[625]))
    check_state=check.get_state()
    np.testing.assert_array_equal(check_state[1],words);assert check_state[2]==index
    core.bitwise(next,subsequent,'subsequent MT draws')


def edges(values,lo=.01,hi=20.):
    v=np.unique(values);return np.unique(np.clip(np.r_[v,np.nextafter(v,-np.inf),np.nextafter(v,np.inf)],lo,hi))

@pytest.mark.parametrize('which',range(17))
def test_p8f_0a_material_functions(cpu,data,which):
    rng=np.random.default_rng(820090)
    for mid,m in enumerate(data[0]):
        lo,hi=(.01,10.) if which in (2,6,7,10,11,12,13,15) else (.01,20.)
        if which==13:lo=1.0221
        nodes=np.r_[m.stop_T,m.brems_E,m.gs_T,m.photo_E,m.pair_E,m.bindings.ravel(),.01,.02,.01/(.5-1e-7),20.]
        T=np.r_[np.exp(rng.uniform(np.log(lo),np.log(hi),100000)),edges(nodes,lo,hi)]
        x=T[:,None]
        if which in (6,7):
            bindings=m.bindings[m.bindings>0]
            x=np.r_[T,edges(bindings,float(bindings.min()/2),hi)][:,None]
        if which==4:
            x=np.column_stack([T,T*rng.random(len(T))])
            x=np.r_[x,[[t,k] for t in (.01,.02,1.,20.) for k in (0.,t)]]
        if which==8:x=edges(T,np.nextafter(.01,np.inf),20.)[:,None]
        if which==9:
            # Includes disabled, outer shell, nonradiative and K->L chains.
            x=np.column_stack([rng.integers(len(m.z),size=100000),rng.integers(5,size=100000),np.ones(100000)])
        if which==11:
            x=np.column_stack([T,rng.uniform(m.gs_K[0],m.gs_K[-1],len(T))])
            knots=edges(m.gs_K,float(np.nextafter(m.gs_K[0],0)),float(np.nextafter(m.gs_K[-1],np.inf)))
            x=np.r_[x,[[t,v] for t in edges(m.gs_T,.01,10.) for v in knots]]
        compare_probe(cpu,which,x,m,mid)

@njit
def score_reference(x,edges,owner,faces):
    out=np.zeros((len(x),len(owner)));bound=np.zeros((len(x),2))
    for j in range(len(x)):
        k.score_bins(out[j],x[j,0],x[j,1],int(x[j,2]),x[j,3],edges,owner)
        a,b=k.boundary(x[j,0],x[j,4],int(x[j,2]),faces);bound[j,0]=a;bound[j,1]=b
    return out,bound

def test_p8f_0c_scoring_and_boundaries(cpu):
    rng=np.random.default_rng(820091);a=cpu.arrays;region=rng.integers(cpu.g.nl,size=100000)
    left=a['faces'][region];width=np.diff(a['faces'])[region]
    x=np.column_stack([left+rng.random(len(region))*width,left+rng.random(len(region))*width,region,rng.random(len(region)),rng.uniform(-1,1,len(region))])
    extra=[]
    for r in range(cpu.g.nl):
        for z in a['bins'][a['owner'].tolist().index(r):np.flatnonzero(a['owner']==r)[-1]+2]:
            for v in (z,np.nextafter(z,-np.inf),np.nextafter(z,np.inf)):
                if a['faces'][r]<=v<=a['faces'][r+1]:
                    for w in (-1.,0.,1.):extra.append([v,v,r,.3,w])
    for r in range(cpu.g.nl):
        for z in (a['faces'][r],a['faces'][r+1]):
            for v in (z,np.nextafter(z,-np.inf),np.nextafter(z,np.inf)):
                for w in (-1.,0.,1.):extra.append([v,v,r,0.,w])
    x=np.r_[x,extra];actual,bound=cpu.score(x)
    expected,ref_bound=score_reference(x,a['bins'],a['owner'],a['faces'])
    core.bitwise(actual,expected,'score bins');core.bitwise(bound,ref_bound,'boundary')


def test_p8f_0b_sum_and_power_order(cpu,data):
    # Adversarial magnitudes expose reassociation; explicitly probe 220/5/4 sums.
    rng=np.random.default_rng(820092)
    for width in (4,5,10,220):
        x=np.exp(rng.uniform(-30,30,(1000,width)))
        compare_probe(cpu,18,x,data[0][0],0)
        starts=rng.integers(0,width,len(x));ends=rng.integers(0,width,len(x))
        compare_probe(cpu,26,np.column_stack([np.minimum(starts,ends),np.maximum(starts,ends),x]),data[0][0],0)
    # Integer squares and fractional powers are exercised by hard/shape in 0a.


def test_p8f_0d_boundary_states(cpu,data):
    for mid,m in enumerate(data[0]):
        rows=[]
        for strength in (0.,np.nextafter(m.gs_K[0],0),m.gs_K[0],np.nextafter(m.gs_K[0],np.inf),.01):
            for terminal in (0.,1.):
                for mu in (-1.,1.,2.):
                    a=np.zeros(29);a[5]=1.;a[6]=a[7]=1.;a[14]=strength;a[18]=terminal
                    a[23:]=[0,1,10.,1.,.025,mu];rows.append(a)
        compare_probe(cpu,17,rows,m,mid)
        a=np.zeros(29);a[5]=1.;a[6]=a[7]=1.;a[14]=.01;a[23:]=[0,1,10.,1.,.025,-1.]
        reflected=reference(17,a[None,:],m,core.SEED0)[0,:23]
        again=a.copy();again[:23]=reflected;again[12]=again[13]=0.;again[14]=.01
        compare_probe(cpu,17,[a,again],m,mid)


def test_p8f_0e_real_history_faults(cpu,data,tmp_path):
    from prototypes.mv_electron.run_p8e import LAYERS
    m,i,e,c=data
    good=cpu.run(100,core.SEED0,per_history=True)
    assert k.healthy(good)
    core.check_local_bins(good['histories'],good['dep'])
    # Bound here is deliberately conservative until deposit counters are exported.
    core.check_aggregation(good['p8_batches'],good['histories'],np.repeat(good['counts'][0]+good['counts'][9]+1,good['histories'].size).reshape(good['histories'].shape))
    for fault in (5,6,7):
        bad=p.CPU(tmp_path/f'fault{fault}',m,densities=i['densities'],edges=e,cum=c,fault=fault).run(100,core.SEED0,per_history=True)
        if fault==5:assert bad['p8_max'][0]>1e-12 and not k.healthy(bad)
        elif fault==6:
            with pytest.raises(AssertionError,match='aggregate'):core.check_aggregation(bad['p8_batches'],bad['histories'],np.ones_like(bad['histories'],int))
        else:
            with pytest.raises(AssertionError,match='local bins'):core.check_local_bins(bad['histories'],bad['dep'])


def test_p8f_small_chain0(cpu,data):
    from prototypes.mv_electron.run_p8e import LAYERS
    m,i,e,c=data
    actual=cpu.run(100,core.SEED0)
    expected=k.run_coupled(m,LAYERS,100,core.SEED0,densities=i['densities'],spectrum_edges=e,spectrum_cum=c)
    for key in core.KEYS+('coarse','fine','p8_batches'):
        core.bitwise(actual[key],expected[key],key)


def test_p8f_0d_injected_competition_and_fluorescence(data,tmp_path):
    from prototypes.mv_electron.run_p8e import LAYERS
    m,i,_,_=data
    cpu=p.CPU(tmp_path,m,densities=i['densities'])
    cases=[]
    for flag in (2.,3.):
        a=np.zeros(23);a[2]=21.2;a[5]=1.;a[6]=.007;a[16]=5;a[17]=flag;cases.append(a)
    for offset in (0.,np.nextafter(1e-13,0.),np.nextafter(1e-13,np.inf),2e-13):
        a=np.zeros(23);a[2]=5.;a[5]=1.;a[6]=a[7]=1.;a[9]=.025;a[10]=.025;a[11]=.05
        a[8]=offset;a[12]=0.;a[13]=.025;a[14]=.05;a[15]=0.;a[19]=-1;cases.append(a)
    cases.append(core.fixed_inputs()[1][-1])
    for a in cases:
        cpu.g.energy=a[6]
        expected=k.run_coupled(m,LAYERS,1,core.SEED0,densities=i['densities'],E=a[6],source_state=a,record_events=16)
        if a[6]==10.:assert expected['brems'][0]>0 and expected['event_records'][0,0]==4
        index,words=_helperlib.rnd_get_state(_helperlib.rnd_get_np_state_ptr());next=next_draws()
        actual=cpu.fixed(a,core.SEED0)
        for key in core.KEYS:
            if key=='dose':continue
            core.bitwise(actual[key],expected[key],f'fixed {key}')
        core.bitwise(actual['p8_batches'],expected['p8_batches'],'fixed bins')
        core.bitwise(actual['events'],expected['event_records'],'priority events')
        core.bitwise(actual['mt'][:624],np.array(words,np.int64),'fixed MT array')
        assert actual['mt'][624]==index
        core.bitwise(actual['next'],next,'fixed following draws')


def test_p8f_philox_records(cpu,data,tmp_path):
    m,i,e,c=data
    philox=p.CPU(tmp_path,m,densities=i['densities'],edges=e,cum=c,rng='philox')
    histories=philox.histories(100,core.SEED2)
    run=philox.run(100,core.SEED2)
    assert core.history_health(histories)
    core.check_local_bins(histories['histories'],histories['dep'])
    core.check_aggregation(run['p8_batches'],histories['histories'],histories['additions'])
    assert core.compare_histories(histories,histories)['status']=='pass'


def test_p8f_0a_remaining_direction_energy_init(cpu,data):
    m=data[0][0];rng=np.random.default_rng(820093);T=np.exp(rng.uniform(np.log(.01),np.log(20.),100000))
    compare_probe(cpu,20,np.column_stack([T,rng.uniform(0,2*np.pi,len(T))]),m,0)
    compare_probe(cpu,21,np.ones((100000,1)),m,0)
    compare_probe(cpu,22,rng.random((100000,3)),m,0)
    a=rng.random((100000,23));a[:,6]=T;compare_probe(cpu,23,a,m,0)
    compare_probe(cpu,24,[[.5,.5,.5,0,0,0],[.99,.1,.2,0,0,0],[0.,0,0,0,0,0]],m,0)


def patch_bundle(cpu,mid,m):
    for j,name in enumerate(core.FIELDS):
        off,n,_,_=cpu.meta[mid,j];cpu.data[off:off+n]=np.asarray(getattr(m,name)).ravel()


def test_p8f_0d_forced_relaxation_and_rejection(cpu,data):
    m=data[0][3]
    try:
        # Exactly zero shell probability consumes no random number.
        bindings=np.zeros_like(m.bindings);bindings[:,0]=2.;pm=np.zeros_like(m.pm)
        altered=m._replace(bindings=bindings,pm=pm)
        patch_bundle(cpu,3,altered);compare_probe(cpu,7,[[.1]],altered,3)
        assert cpu.probe(7,[[.1]],3)[1][625]==0
        # A cumulative tie chooses the next positive shell.
        r=np.random.RandomState(core.SEED0).random_sample();bindings[:]=0.;pm[0,0,0]=np.log(r);pm[0,1,0]=np.log(1-r)
        altered=m._replace(bindings=bindings,pm=pm,shell_nj=np.ones_like(m.shell_nj))
        patch_bundle(cpu,3,altered);compare_probe(cpu,7,[[.1]],altered,3)
        assert cpu.probe(7,[[.1]],3)[0][0,1]==1
        # Nonradiative, outer shell, below/exactly/above CUT, K -> L chain.
        for energy in (np.nextafter(.01,0),.01,np.nextafter(.01,np.inf)):
            lines=np.zeros_like(m.lines);counts=np.ones_like(m.line_counts);yields=np.ones_like(m.yields)
            lines[0,0,0,:3]=[energy,1.,1.];lines[0,1,0,:3]=[.01,1.,4.]
            altered=m._replace(lines=lines,line_counts=counts,yields=yields)
            patch_bundle(cpu,3,altered);compare_probe(cpu,9,[[0.,0.,1.],[0.,4.,1.],[0.,0.,0.]],altered,3)
        altered=m._replace(yields=np.zeros_like(m.yields));patch_bundle(cpu,3,altered);compare_probe(cpu,9,[[0.,0.,1.]],altered,3)
    finally:patch_bundle(cpu,3,m)
    m=data[0][0];T=1.;u=np.random.RandomState(core.SEED0).random_sample();photon_k=np.exp(np.log(.01)+np.log(T/.01)*u)
    shape=k.material_shape(T,photon_k,m.z,m.nj)
    try:
        for envelope,expected_calls in ((np.nextafter(shape,np.inf),2),(0.,1)):
            v=m.brems_V.copy();v[:,3]=envelope;altered=m._replace(brems_V=v);patch_bundle(cpu,0,altered)
            compare_probe(cpu,8,[[T]],altered,0)
            assert cpu.probe(8,[[T]],0)[1][625]==expected_calls
        v=m.brems_V.copy();v[:,3]=10*max(k.material_shape(T,x,m.z,m.nj) for x in np.geomspace(.01,T,1025))
        altered=m._replace(brems_V=v);patch_bundle(cpu,0,altered);compare_probe(cpu,8,[[T]],altered,0)
        assert cpu.probe(8,[[T]],0)[1][625]>2
    finally:patch_bundle(cpu,0,m)


def test_p8f_0d_event_priority_nextafter(cpu,data):
    rows=[];step=.25;threshold=step+1e-13
    for j in range(6):
        for candidate in (step,np.nextafter(threshold,-np.inf),threshold,np.nextafter(threshold,np.inf)):
            a=np.full(7,1.e100);a[0]=step;a[j+1:]=candidate;rows.append(a)
    compare_probe(cpu,25,rows,data[0][0],0)
