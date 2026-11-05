"""A4u host-only xraylib table generator. No calls or writes on import.

The packed elemental table is shared by both native specializations. Energies
are keV. Partial cross sections are used ONLY as conditional probabilities.
An exception delimits a measured range; it is never replaced by a zero value.
"""
import argparse
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
import numpy as np

SHELLS=('K','L1','L2','L3','M1','M2','M3','M4','M5')
# Element descriptor: Z, weight, E-count, line-count, then offsets for
# E, photo, compt, rayl, shell probabilities, edges, yields, CK, lines, q,S,x,A.
HEADER=17


def measured(fn, low, high=10000.):
    """Probe the full required low band, then bracket the first upper failure."""
    fn(float(low))  # a missing required lower endpoint is fatal
    good=float(low); failure=None
    for e in np.geomspace(low,high,193)[1:]:
        try: fn(float(e))
        except ValueError as exc:
            failure=str(exc);bad=float(e);break
        good=float(e)
    else:return good,None
    for _ in range(40):
        mid=(good+bad)/2
        try:fn(mid)
        except ValueError:bad=mid
        else:good=mid
    # Stay strictly inside the interpolation domain.
    return good*(1-1e-10),dict(first_invalid=bad,error=failure)


def bracket_edge(fn, edge, others, window=1e-4):
    """Nodes that straddle the actual jump, which need not sit at EdgeEnergy.

    xraylib's CS_Photo steps up to 11 meV away from EdgeEnergy (lead L2 jumps at
    15.199989 keV for an edge of 15.200000). Bracketing the edge itself then puts
    the "below" node above the jump, and the interval reaching down to the previous
    node is interpolated across the step: lead's L2O4 line at 15.178 keV came out
    25% too absorbing. Bisect for the step instead, and keep EdgeEnergy inside the
    straddling pair so the conditional shell probabilities stay consistent with it.
    """
    near=min((abs(edge-o) for o in others if o!=edge),default=edge)
    limit=min(edge*2e-2,near/4)
    half=min(edge*window,limit)
    low,high=edge-half,edge+half
    top=fn(high)
    # Widen the lower end until it is demonstrably below the step: for uranium's
    # L3 the jump sits more than 1.7 eV under EdgeEnergy, so a window chosen from
    # the edge alone can land entirely above it. Count the steps rather than
    # comparing edge-low with the limit: cancellation can pin that difference
    # just under the limit and spin forever (polonium, radon, francium, radium).
    for _ in range(12):
        if fn(low)*1.01<top:break
        half=min(half*4,limit);low=edge-half
    base=fn(low)
    if not top>base*1.01:return [low,high]  # no resolvable step to straddle
    for _ in range(60):
        mid=(low+high)/2
        if fn(mid)>base*1.01:high=mid
        else:low=mid
    return [low,max(high,edge)*(1+1e-12)]


@lru_cache(maxsize=None)
def element(z):
    import xraylib as x
    z=int(z);ranges={};edges=np.zeros(9);occupied=[]
    for sh in range(9):
        try:edges[sh]=x.EdgeEnergy(z,sh)
        except ValueError as exc:
            if 'Invalid shell' not in str(exc):raise
            continue  # structurally unoccupied, not a failed cross-section sample
        occupied.append(sh)
    if not occupied:raise ValueError(f'Z={z}: no occupied shells')
    fns=(x.CS_Photo,x.CS_Compt,x.CS_Rayl)
    upper=[]
    for fn in fns:
        hi,err=measured(lambda e:fn(z,e),5.)
        if hi<150:raise ValueError(f'{z} {fn.__name__}: required 5..150 keV unavailable')
        ranges[fn.__name__]=dict(low=5.,high=hi,termination=err);upper.append(hi)
    partial_hi={}
    for sh in occupied:
        low=max(5.,edges[sh]*(1+1e-7))
        hi,err=measured(lambda e:x.CS_Photo_Partial(z,sh,e),low)
        partial_hi[sh]=hi
        ranges['partial_'+SHELLS[sh]]=dict(low=low,high=hi,termination=err)
    end=max(upper+list(partial_hi.values()))
    brackets=[]
    for sh in occupied:
        if not 5<edges[sh]<end:continue
        pair=bracket_edge(lambda e:x.CS_Photo(z,e),edges[sh],
                          [edges[s] for s in occupied if s!=sh])
        brackets.append(pair)
    nodes=np.unique(np.r_[np.geomspace(5.,end,1025),upper,list(partial_hi.values()),np.ravel(brackets) if brackets else []])
    totals=np.array([[fn(z,min(float(e),upper[j])) for e in nodes] for j,fn in enumerate(fns)])
    partial=np.zeros((len(nodes),9))
    for j,e in enumerate(nodes):
        for sh in occupied:
            if e>edges[sh]:partial[j,sh]=x.CS_Photo_Partial(z,sh,min(float(e),min(partial_hi.values())))
    sums=partial.sum(axis=1)
    if np.any(sums<=0):raise ValueError(f'Z={z}: missing shell probabilities')
    probs=partial/sums[:,None]
    yields=np.zeros(9);lines=[];unavailable={}
    defs=[]
    for name in dir(x):
        # Destinations that xraylib tabulates only as a merged level (KO, L3O45,
        # ...) are the sole representation of those transitions, not aliases of
        # per-level lines, so they belong here. Leaving them out dropped 0.82% of
        # lead's K weight and 1.79% of its L3 weight onto the surviving lines.
        match=re.fullmatch(r'(K|L[123]|M[1-5])(L[123]|M[1-5]|N[1-7]|N67|O[1-7]|O45|P[1-5]|P23|P45|Q[1-3]|O|P)_LINE',name)
        if match:
            a,b=match.groups();defs.append((SHELLS.index(a),SHELLS.index(b) if b in SHELLS else 9,getattr(x,name)))
    for sh in occupied:
        # With a binding budget below 5 keV neither this vacancy nor any
        # descendant can emit a transported photon. Terminate it explicitly;
        # do not ask an unsupported yield API and turn its exception into zero.
        if edges[sh]<5:continue
        yields[sh]=x.FluorYield(z,sh)
        row=[]
        for origin,child,code in defs:
            if origin!=sh:continue
            try:rate=x.RadRate(z,code)
            except ValueError as exc:
                if 'Invalid line' not in str(exc):raise
                continue
            if rate<=0:continue
            try:energy=x.LineEnergy(z,code)
            except ValueError as exc:
                # xraylib knows the branching but not the energy for a few
                # lines; that weight cannot be emitted and must not be
                # renormalised onto the rest in silence.
                if 'Invalid line' not in str(exc):raise
                unavailable[sh]=unavailable.get(sh,0.)+rate;continue
            row.append([sh,energy,rate,child])
        if yields[sh]>0 and not row and edges[sh]>=5:raise ValueError('missing radiative lines')
        if row:
            norm=sum(a[2] for a in row)
            # xraylib's rates for one vacancy sum to 1 over its distinct lines.
            # A shortfall means lines were missed and their weight is about to be
            # renormalised onto the rest, which silently distorts the spectrum.
            if abs(norm+unavailable.get(sh,0.)-1)>1e-3:raise ValueError(f'Z={z} {SHELLS[sh]}: radiative rates sum to {norm}, lines are missing')
            for a in row:a[2]/=norm
            lines.extend(row)
    ck=np.zeros((9,9))
    for sh,dest,code in ((1,2,x.FL12_TRANS),(1,3,x.FL13_TRANS),(2,3,x.FL23_TRANS)):
        if sh not in occupied or dest not in occupied or edges[sh]<5:continue
        ck[sh,dest]=x.CosKronTransProb(z,code)
    if np.any(ck.sum(axis=1)+yields>1+1e-12):raise ValueError('relaxation probability budget')
    q=np.r_[0.,np.geomspace(1e-3,1000.,1999)]
    for name,fn in (('SF_Compt',x.SF_Compt),('FF_Rayl',x.FF_Rayl)):
        hi,err=measured(lambda v:fn(z,v),1e-3,10000.)
        if hi<1000:raise ValueError(f'{z} {name}: required q range unavailable')
        ranges[name]=dict(low=1e-3,high=hi,termination=err)
    sf=np.array([0. if v==0 else x.SF_Compt(z,float(v)) for v in q])
    rx=q*q;ff=np.array([x.FF_Rayl(z,float(v))**2 for v in q])
    area=np.r_[0.,np.cumsum(np.diff(rx)*(ff[1:]+ff[:-1])/2)]
    # A line's escape is set by mu at its own energy, so an edge node placed on
    # the wrong side of the jump quietly changes the spectrum. Lines that fall
    # inside a straddling pair are boundary cases xraylib itself cannot resolve;
    # they are listed instead of asserted.
    at_edges=[]
    for row in lines:
        if row[1]<5:continue
        inside=[list(b) for b in brackets if b[0]<=row[1]<=b[1]]
        if inside:at_edges.append(dict(shell=SHELLS[int(row[0])],keV=float(row[1]),rate=float(row[2]),bracket=inside[0]));continue
        got=float(np.exp(np.interp(np.log(row[1]),np.log(nodes),np.log(totals[0]))))
        if abs(got/x.CS_Photo(z,float(row[1]))-1)>3e-3:
            raise ValueError(f'Z={z} line {row[1]} keV: table photo {got} vs xraylib {x.CS_Photo(z,float(row[1]))}')
    report=dict(Z=z,xraylib=x.__version__,ranges=ranges,shells=[SHELLS[s] for s in occupied],lines_at_edges=at_edges,rates_without_energy={SHELLS[k]:v for k,v in unavailable.items()},
        partial_vs_total=[dict(E=float(e),photo=float(x.CS_Photo(z,e)),partial_sum=float(sum(x.CS_Photo_Partial(z,s,min(e,partial_hi[s])) for s in occupied if e>edges[s]))) for e in (20.,60.,150.)],
        lines_ge5=[a for a in lines if a[1]>=5.])
    return nodes,totals,probs,edges,yields,ck,np.array(lines).reshape(-1,4),q,sf,rx,area,upper,report


def material(zs,weights,*,xcom=None,elemental_xcom=None):
    """xcom=(keV nodes, photo, incoherent, coherent); mandatory above measured domains."""
    zs=np.asarray(zs,int);weights=np.asarray(weights,float);weights=weights/weights.sum()
    blob=[float(len(zs)),0.]+[0.]*(HEADER*len(zs));reports=[]
    for i,(z,w) in enumerate(zip(zs,weights)):
        e,total,prob,edges,yields,ck,lines,q,sf,rx,area,upper,report=element(int(z))
        if xcom is not None:
            if elemental_xcom is None or int(z) not in elemental_xcom:raise ValueError('elemental XCOM fallback is required')
            xe,xp,xc,xr=map(np.asarray,elemental_xcom[int(z)])
            if xe[0]>5 or xe[-1]<10000 or np.any(xp<=0) or np.any(xc<=0) or np.any(xr<=0):raise ValueError('elemental XCOM coverage')
            new_e=np.unique(np.r_[e,xe[xe>min(upper)],np.array(upper)*(1+1e-8)])
            new_total=np.array([np.exp(np.interp(np.log(new_e),np.log(e),np.log(row))) for row in total])
            for j,high in enumerate(upper):
                outside=new_e>high
                new_total[j,outside]=np.exp(np.interp(np.log(new_e[outside]),np.log(xe),np.log((xp,xc,xr)[j])))
            prob=np.array([np.interp(new_e,e,prob[:,sh]) for sh in range(9)]).T
            e,total=new_e,new_total
        fields=[e,*total,prob,edges,yields,ck,lines,q,sf,rx,area]
        desc=[z,w,len(e),len(lines)]
        for a in fields:desc.append(len(blob));blob.extend(np.asarray(a).ravel())
        blob[2+i*HEADER:2+(i+1)*HEADER]=desc
        # Upper bounds are separate from constant extensions used only for interpolation.
        blob.extend(upper);reports.append(report)
    # Trailer descriptor: elemental upper bounds and XCOM fallback. No generated
    # spectrum values are ever stored here.
    head=len(blob);upp=[]
    for z in zs:upp.extend(element(int(z))[-2])
    if xcom is None:
        fallback=np.empty((0,4)) # diagnostic-only table; no extrapolation permitted
    else:
        fallback=np.column_stack(xcom)
        if fallback.shape[1]!=4 or fallback[0,0]>5 or fallback[-1,0]<10000 or np.any(fallback[:,1:]<=0):raise ValueError('missing XCOM 5 keV..10 MeV fallback')
    # Header offset is a final scalar, allowing any number of elements/lines.
    blob.extend([len(fallback),*upp,*fallback.T.ravel()]);blob[1]=head
    return np.asarray(blob),reports


def therapy_material(m, xcom, elemental_xcom):
    import xraylib as x
    weights=np.array([n*x.AtomicWeight(int(z)) for z,n in zip(m.z,m.nj)])
    return material(m.z,weights,xcom=xcom,elemental_xcom=elemental_xcom)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--elements',type=int,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();blob,reports=material(a.elements,np.ones(len(a.elements)))
    np.savez(a.out,photon=blob);a.out.with_suffix('.json').write_text(json.dumps(dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elements=reports),indent=2)+'\n')

if __name__=='__main__':main()


def ensure(m,*,sources=None):
    """Host export adapter for the already sealed P8a/P9a input materials.

    Workers receive `photon` in their sealed bundle and never enter this path.
    Read existing, hash-checked XCOM responses; do not run/fetch XCOM.
    """
    if hasattr(m,'photon'):return validate(m.photon)
    roots=[Path.home()/'.cache/chatcarlo'/p for p in ('p8a/xcom','p8a/xcom_appendix5','p8a/xcom_appendix9_round2','p9a/xcom')]
    for root in roots:
        if not root.exists():continue
        for folder in sorted(root.iterdir()):
            files=sorted(folder.glob('mixture_[0-9]*.txt'),key=lambda p:int(p.stem.split('_')[-1]))
            rows=[];origin=[]
            for path in files:
                pin=json.loads(path.with_suffix('.sha.json').read_text())['sha256']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=pin:raise ValueError('XCOM source SHA256 mismatch')
                origin.append(dict(path=str(path),sha256=pin,kind='material'))
                for line in path.read_text().splitlines():
                    parts=line.split()
                    if len(parts)==8:
                        try:rows.append([float(v) for v in parts])
                        except ValueError:pass
            if not rows:continue
            a=np.asarray(rows)
            if a.shape[0]!=len(m.photo) or not np.allclose(a[:,0],m.photo_E,rtol=2e-6,atol=0) or not np.array_equal(a[:,3],m.photo):continue
            elemental={}
            for z in m.z:
                ef=root/'elements'/('z'+str(int(z)));erows=[]
                for source in sorted(ef.glob('nodes_'+folder.name+'_[0-9]*.txt'),key=lambda p:int(p.stem.split('_')[-1])):
                    pin=json.loads(source.with_suffix('.sha.json').read_text())['sha256']
                    if hashlib.sha256(source.read_bytes()).hexdigest()!=pin:raise ValueError('elemental XCOM SHA256 mismatch')
                    origin.append(dict(path=str(source),sha256=pin,kind='elemental',Z=int(z)))
                    for line in source.read_text().splitlines():
                        parts=line.split()
                        if len(parts)==8:
                            try:erows.append([float(v) for v in parts])
                            except ValueError:pass
                if not erows:raise ValueError(f'missing elemental XCOM Z={int(z)}')
                ea=np.array(erows);elemental[int(z)]=(ea[:,0]*1000,ea[:,3],ea[:,2],ea[:,1])
            value,_=therapy_material(m,(a[:,0]*1000,a[:,3],a[:,2],a[:,1]),elemental)
            if sources is not None:sources.extend(origin)
            return validate(value)
    raise ValueError('A4u: supply generated photon tables with hash-checked XCOM fallback; legacy source not found')


def validate(blob):
    p=np.asarray(blob,float)
    if p.ndim!=1 or len(p)<20 or not np.isfinite(p).all():raise ValueError('invalid A4u photon table')
    nz=int(p[0]);tail=int(p[1])
    if not 1<=nz<=32 or tail<2+17*nz or tail+1+3*nz>len(p):raise ValueError('invalid A4u photon descriptor')
    nf=int(p[tail])
    if nf<0 or len(p)!=tail+1+3*nz+4*nf:raise ValueError('invalid A4u photon fallback')
    for i in range(nz):
        d=p[2+17*i:2+17*(i+1)];n=int(d[2]);nl=int(d[3]);sizes=(n,n,n,n,n*9,9,9,81,nl*4,2000,2000,2000,2000)
        if n<2 or nl<0 or d[1]<=0:raise ValueError('invalid A4u elemental descriptor')
        for off,size in zip(d[4:],sizes):
            if off!=int(off) or off<2+17*nz or off+size>tail:raise ValueError('invalid A4u field range')
        off=int(d[8]);prob=p[off:off+9*n].reshape(n,9)
        if np.any(prob<0) or not np.allclose(prob.sum(axis=1),1,rtol=1e-13,atol=1e-13):raise ValueError('invalid A4u shell probabilities')
    return p
