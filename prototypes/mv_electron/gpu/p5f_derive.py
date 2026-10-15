"""P5f-0 exact, audited source transformations. Standard library and NumPy only."""
import hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CONDITIONS = ('B', 'T', 'W', 'TW')
OPTIONS = ('-DP5_GPU', '-DP5_RNG_PHILOX', '--fmad=false',
           '-DP5_TRANSPORT_FAULT=0', '-DP5_STACK_CAP=64')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('replacement must occur exactly once: '+old)
    return text.replace(old, new)


def derive(condition, cpu=False):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    records = []
    def edit(text, old, new):
        records.append(dict(old=old, new=new))
        return replace_once(text, old, new)
    common = (HERE/'p5a0_common.h').read_text()
    rng = (HERE/'p5_rng.h').read_text()
    rng = edit(rng, '#include "p5a0_common.h"', '')
    rng = edit(rng, 'struct RngState {uint64_t seed,history,calls;};',
               'struct RngState {uint64_t seed,history,calls,reads;int p5f_base;};')
    rng = edit(rng, 's->seed=seed;s->history=history;s->calls=0;',
               's->seed=seed;s->history=history;s->calls=0;s->reads=0;s->p5f_base=0;')
    kernel = (HERE/'p5_kernel.h').read_text()
    for name in ('p5_rng.h',):
        kernel = edit(kernel, '#include "'+name+'"', '')
    kernel = edit(kernel, 'double u=rng_uniform(s);int i=lower_bound(T,t->ts,t->nt)-1;',
                  '++s->reads;double u=rng_uniform(s);int i=lower_bound(T,t->ts,t->nt)-1;')
    # Each of the eight expressions is a separate exact replacement.
    if 'T' in condition:
        for q in ('q00', 'q00+1', 'q01', 'q01+1', 'q10', 'q10+1', 'q11', 'q11+1'):
            kernel = edit(kernel, 't->ys['+q+']', 'P5F_YS('+q+')')
    prefix = '#define P5F_YS(q) (t->ys[s->p5f_base + ((q) % t->nu)])\n'
    if 'W' in condition:
        kernel = edit(kernel, 'double hs[150]={0.},fine[600]={0.};', 'double checksum[2]={0.,0.};')
        # Replace complete call pairs; repeated pairs use their unique surrounding line.
        old = 'p5_local(hs,150,a[2],E,.2);p5_local(fine,600,a[2],E,.05);'
        kernel = edit(kernel, old, 'p5f_loss(checksum,a[2],E);')
        pair = 'p5_track_local(hs,150,a[2],a[6],.2,touch);p5_track_local(fine,600,a[2],a[6],.05,touch+2);'
        # Anchor each identical call pair to its exact indentation and newline.
        for indent in (16, 20, 12):
            kernel = edit(kernel, '\n'+' '*indent+pair,
                          '\n'+' '*indent+'p5f_track(checksum,a[2],a[2],a[6],touch,false);')
        seg = 'p5_track_segment(hs,150,z0,a[2],loss,.2,touch);p5_track_segment(fine,600,z0,a[2],loss,.05,touch+2);'
        kernel = edit(kernel, seg, 'p5f_track(checksum,z0,a[2],loss,touch,true);')
        kernel = edit(kernel, 'fabs(p5_touched_sum(hs,touch)-dep),fine_error=fabs(p5_touched_sum(fine,touch+2)-dep)',
                      'fabs(checksum[0]-dep),fine_error=fabs(checksum[0]-dep)')
        old = ('for(int z=touch[0];z<=touch[1];++z)o->batches[batch*150+z]+=hs[z];\n'
               '    for(int z=touch[2];z<=touch[3];++z)o->fine_batches[batch*600+z]+=fine[z];')
        kernel = edit(kernel, old, 'o->batches[batch*2]+=checksum[0];o->batches[batch*2+1]+=checksum[1];')
    helpers = '''
P5_FN void p5f_loss(double *c,double z,double loss){c[0]+=loss;c[1]+=z*loss;}
P5_FN void p5f_track(double *c,double z0,double z1,double loss,int *touch,bool segment){
    p5f_loss(c,(z0+z1)*.5,loss);
    if(loss==0 && segment)return;
    int a=p5_clip_bin(int(p5_min(z0,z1)/.2),150),b=p5_clip_bin(int(p5_max(z0,z1)/.2),150);
    if(a<touch[0])touch[0]=a;if(b>touch[1])touch[1]=b;
    a=p5_clip_bin(int(p5_min(z0,z1)/.05),600);b=p5_clip_bin(int(p5_max(z0,z1)/.05),600);
    if(a<touch[2])touch[2]=a;if(b>touch[3])touch[3]=b;
}
'''
    # Helpers need p5_clip_bin, defined before the transport header.
    kernel = edit(kernel, 'P5_FN void history(', helpers+'\nP5_FN void history(')
    text = common+'\n'+rng+'\n'+prefix+kernel
    if cpu:
        entry = (HERE/'p5_cpu.cpp').read_text()
        entry = edit(entry, '#include "p5_kernel.h"', '')
    else:
        wrap = (HERE/'p5b_wrap.h').read_text()
        wrap = edit(wrap, '#include "p5_kernel.h"', '')
        wrap = edit(wrap, 'struct P5bLocal {', 'struct P5fLocal {')
        wrap = edit(wrap, 'P5_FN int p5b_row(', 'typedef P5fLocal P5bLocal;\nP5_FN int p5b_row(')
        wrap = edit(wrap, 'uint64_t calls;', 'uint64_t calls,reads;')
        wrap = edit(wrap, 'int maxsteps,P5bLocal *L)', 'int maxsteps,int p5f_base,P5bLocal *L)')
        wrap = edit(wrap, 'RngState s;rng_init(&s,seed,global_h);',
                    'RngState s;rng_init(&s,seed,global_h);s.p5f_base=p5f_base;')
        wrap = edit(wrap, 'L->calls=s.calls;', 'L->reads=s.reads;L->calls=s.calls;')
        if 'W' in condition:
            wrap = edit(wrap, 'double batches[150],fine_batches[600],totals[6],maxima[3],brems[10];',
                        'double batches[2],totals[6],maxima[3],brems[10];')
            wrap = edit(wrap, 'L->batches,L->fine_batches,L->totals', 'L->batches,0,L->totals')
            wrap = edit(wrap, 'for(int j=0;j<150;++j)n+=p5b_nan(L->batches[j]);\n    for(int j=0;j<600;++j)n+=p5b_nan(L->fine_batches[j]);',
                        'for(int j=0;j<2;++j)n+=!isfinite(L->batches[j]);')
            wrap = edit(wrap, 'for(int j=0;j<150;++j)batches[uint64_t(i)*150+j]=L->batches[j];\n    for(int j=0;j<600;++j)fine_batches[uint64_t(i)*600+j]=L->fine_batches[j];',
                        'for(int j=0;j<2;++j)batches[uint64_t(i)*2+j]=L->batches[j];')
        # Nonfinite detection, including infinity, in every local output.
        wrap = edit(wrap, 'return v!=v;', 'return !isfinite(v);')
        entry = (HERE/'p5f_gpu.cu').read_text()
        text += '\n'+wrap
    text += '\n'+entry
    text = ('#ifdef P5_GPU\n#ifndef INFINITY\n#define INFINITY (__longlong_as_double(0x7ff0000000000000ULL))\n#endif\n#endif\n#define P5F_WRITE '+str(int('W' in condition))+'\n'+text)
    return text, dict(condition=condition, replacements=records,
                      source_sha256=hashlib.sha256(text.encode()).hexdigest(), options=list(OPTIONS))


def health(out, n):
    """P3: applies to each timed run, including changed-transport controls."""
    try:
        return bool(all(np.isfinite(v).all() for v in out.values())
                    and not np.any(out['nan_count'])
                    and out['counts'][5] == out['counts'][6] == 0
                    and not np.any(out['violations'])
                    and np.max(out['maxima']) <= 1e-9
                    and out['nspec'].sum() == n)
    except (KeyError, ValueError, TypeError):
        return False
