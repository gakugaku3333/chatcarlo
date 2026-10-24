"""P8g A1 packing, source identity and strict gate comparisons."""
from .p8f_core import *
from . import p8f_core as frozen

SITES=('brems_E','brems_V','rates_ratio','rates_values','gs_ratio','pair_E','pair_n','pair_e',
       'stop_T','stop_collision','photo_E','photo','edge_E_ratio','edge_photo_ratio','edge_photo_left',
       'shape_1194','shape_184','shape_z')

def lengths(m):
    n=len(m.rates) if m.rates.shape[1]==3 else len(m.gs_T)
    return [len(m.brems_E),3*len(m.brems_E),n-1,2*n,len(m.gs_T)-1,*([len(m.pair_E)]*3),
            len(m.stop_T),len(m.stop_T),len(m.photo_E),len(m.photo_E),len(m.photo_E)-1,len(m.photo_E)-1,len(m.photo_E),*([len(m.z)]*3)]

def pack(materials):
    data,old=frozen.pack(materials);meta=np.zeros((len(materials),136),np.int64)
    meta[:,:132]=old.reshape(len(materials),132)
    chunks=[data];offset=len(data)
    for mid,m in enumerate(materials):
        size=sum(lengths(m));meta[mid,132:135]=[offset,offset+size,mid]
        chunks.extend([np.full(size,np.nan),np.zeros(37)]);offset+=size+37
    # f2: material 0, shape_z only, first pointer redirected to material 1.
    if len(materials)>1:meta[0,135]=meta[1,132]+sum(lengths(materials[1])[:17])
    return np.ascontiguousarray(np.concatenate(chunks)),meta

def fingerprint():
    names=set(frozen.fingerprint())|{'p8g_logs.h','p8g_reference.h','p8g_functions.h','p8g_materials.h','p8g_kernel.h','p8g_cpu.cpp','p8g_gpu.cu','p8g_core.py'}
    return {n:sha(HERE/n) for n in sorted(names)}

def cuda_source():
    def expand(name,seen):
        lines=[]
        for line in (HERE/name).read_text().splitlines():
            if line.startswith('#include "'):
                child=line.split('"')[1]
                if child not in seen:seen.add(child);lines.append(expand(child,seen))
            else:lines.append(line)
        return '\n'.join(lines)
    probe=(HERE/'p8g_cpu.cpp').read_text()
    probe=probe[probe.index('extern "C" void p8g_probe('):probe.index('extern "C" void p8g_score_probe(')]
    probe=probe.replace('extern "C" void p8g_probe(', 'extern "C" __global__ void p8g_probe_kernel(')
    probe=probe.replace('P8gMaterial m=p8g_material','if(blockIdx.x || threadIdx.x)return;P8gMaterial m=p8g_material',1)
    synthetic=(HERE/'p8g_cpu.cpp').read_text().split('// Synthetic rounded-log')[1]
    synthetic=synthetic[synthetic.index('extern "C"'):].replace('extern "C" void p8g_interp_probe(', 'extern "C" __global__ void p8g_interp_kernel(')
    synthetic=synthetic.replace('double logs[512];', 'if(blockIdx.x || threadIdx.x)return;double logs[512];')
    values=(HERE/'p8g_cpu.cpp').read_text().split('extern "C" void p8g_values',1)[1].split('// Direct comparison',1)[0]
    values='extern "C" __global__ void p8g_values'+values
    values=values.replace('{P8gMaterial m=', '{if(blockIdx.x || threadIdx.x)return;P8gMaterial m=',1)
    parts=(HERE/'p8g_cpu.cpp').read_text().split('extern "C" void p8g_parts',1)[1].split('// Synthetic rounded-log',1)[0]
    parts='extern "C" __global__ void p8g_parts'+parts
    parts=parts.replace(' P8gMaterial m=', ' if(blockIdx.x || threadIdx.x)return;P8gMaterial m=',1)
    return (HERE/'p8g_gpu.cu').read_text().replace('//@@COMMON@@',expand('p8g_kernel.h',set())).replace('//@@PROBE@@',probe)+'\n'+expand('p8g_reference.h',set())+'\n'+values+'\n'+parts+'\n'+synthetic

def exact_histories(actual,expected,label='histories'):
    if actual.keys()!=expected.keys():raise AssertionError(label+' keys')
    for key in expected:bitwise(actual[key],expected[key],label+' '+key)

def check_records(aggregate,h,n):
    """All additive and maximum fields, with a bound for signed sums."""
    group=np.minimum(99,np.arange(n)*100//n)
    sums={'p8_batches':'histories','coarse':'coarse','fine':'fine','totals':'totals','brems':'brems',
          'violations':'violations','counts':'counts','boundary_counts':'boundary_counts','boundary_hist':'boundary_hist',
          'boundary_omitted_hist':'boundary_omitted_hist','fluorescence_counts':'fluorescence_counts','branches':'branches'}
    for dest,src in sums.items():
        values=h[src];expected=values.sum(axis=0)
        actual=aggregate[dest].sum(axis=0) if dest in ('p8_batches','coarse','fine') else aggregate[dest]
        if values.dtype.kind in 'iu':bitwise(actual.astype(np.int64),expected.astype(np.int64),dest)
        else:
            # Deposit order inside each history plus history/group reductions.
            additions=int(h['counts'][:,0].sum()+h['counts'][:,9].sum()+n*4)
            bound=(additions+3*n)*2.**-52*np.abs(values).sum(axis=0)
            if not np.isfinite(actual).all() or np.any(np.abs(actual-expected)>bound):raise AssertionError('aggregate '+dest)
        if dest in ('p8_batches','coarse','fine'):
            for b in range(100):
                v=values[group==b];a=aggregate[dest][b];bound=(int(h['counts'][group==b,0].sum())+4*n)*2.**-52*np.abs(v).sum(axis=0)
                if np.any(np.abs(a-v.sum(axis=0))>bound):raise AssertionError('group '+dest)
    for key in ('maxima','p8_max'):bitwise(aggregate[key],h[key].max(axis=0),key)
    bitwise(aggregate['maxdepth'],np.array(h['maxdepth'].max(),np.int64),'maxdepth')
    bitwise(aggregate['steps'],np.array([h['steps'][group==b].sum() for b in range(100)],np.int64),'steps')

def nodes(v):
    v=np.asarray(v);v=v[v>0]
    return np.unique(np.r_[v,np.nextafter(v,0),np.nextafter(v,np.inf),v.min()/2,v.max()*2])

def component_cases(materials):
    for mid,m in enumerate(materials):
        for which,v in ((0,m.stop_T),(2,np.r_[m.photo_E,m.pair_E,m.thresholds]),(3,m.brems_E),(10,np.r_[m.gs_T,m.rates[:,0]])):
            yield which,mid,nodes(v)[:,None]
        t=nodes(np.r_[m.brems_E,.01,.02,1.,10.])
        yield 4,mid,np.array([[a,k] for a in t for k in (0.,.01,a/2,a)])
        t=nodes(m.gs_T)
        yield 11,mid,np.array([[a,k] for a in t for k in (m.gs_K[0],m.gs_K[-1])])


def check_group_records(aggregate,h,n):
    """Reconstruct each of the 100 actual raw atomic records, not only totals.

    Nonaggregated fields: source, dep, diagnostic, calls, diagnostic_count,
    nspec_index. Maxima have no summation tolerance. All integer sums are exact.
    """
    d=np.column_stack([h[k] for k in ('histories','coarse','fine','totals','maxima','brems','p8_max')])
    ints=np.column_stack([h[k] for k in ('violations','counts','steps','boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts','branches')])
    f=h['histories'].shape[1]+h['coarse'].shape[1]+h['fine'].shape[1]
    maxcols=[*range(f+6,f+9),f+19,f+20]
    group=np.minimum(99,np.arange(n)*100//n)
    raw=aggregate['_groups_d'];rawi=aggregate['_groups_i']
    if not np.isfinite(raw).all():raise AssertionError('nonfinite groups')
    for b in range(100):
        mask=group==b;v=d[mask];hi=ints[mask];num=int(mask.sum())
        expected=v.sum(axis=0)
        for col in maxcols:expected[col]=v[:,col].max(initial=0.)
        # Each history's exported brems[1] contains several internal additions.
        # counts[0] includes charged events; counts[9] bounds photo deposits;
        # source/escape/rest counters bound the other internal additions.
        inner=int(np.abs(h['counts'][mask]).sum()+h['steps'][mask].sum()+num*8)
        bound=(inner+2*num)*2.**-52*np.abs(v).sum(axis=0)
        nb=h['histories'].shape[1]
        bound[:nb]=(h['additions'][mask].sum(axis=0)+2*num)*2.**-52*np.abs(v[:,:nb]).sum(axis=0)
        bound[maxcols]=0.
        if np.any(np.abs(raw[b,:f+21]-expected)>bound):raise AssertionError(f'group {b} floating sum/max')
        bitwise(rawi[b,:75].astype(np.int64),hi.sum(axis=0).astype(np.int64),f'group {b} integer sums')
        bitwise(rawi[b,79:].astype(np.int64),h['additions'][mask].sum(axis=0).astype(np.int64),f'group {b} additions')
        if int(rawi[b,77])!=int(h['maxdepth'][mask].max(initial=0)):raise AssertionError(f'group {b} maxdepth')
    check_records(aggregate,h,n)
