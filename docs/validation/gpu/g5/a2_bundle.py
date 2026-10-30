"""A2 sealed input equality: catalogue -> disk -> native input; no physics gate."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import a2_common as common
from chatcarlo.gpu import catalog,therapy,therapy_bundle,physics_sets


def equal(a,b,label):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        raise AssertionError('input mismatch: '+label)
    return dict(shape=list(a.shape),sha256=catalog.digest(a),equal=True)


def run(source=common.SOURCE,out=common.CACHE/'bundle',summary=common.SUMMARY/'bundle.json'):
    from prototypes.mv_electron.gpu import p8g2_core as old,p7_core
    manifest,materials,arrays=common.inputs(source)
    generators={d['generator_path']:catalog.generator(d['generator_path'],common.ROOT) for d in catalog.P8_DEFINITIONS.values()}
    # Read private spectra at their sealed paths; save hashes only in repo evidence.
    spectra={}
    for name in p7_core.SPECTRA:
        e,p,c=p7_core.spectrum(name);spectra[name]=(e,p,c)
    all_inputs={**arrays}
    for name,(e,p,c) in spectra.items():
        all_inputs.update({name+'_edges':e,name+'_prob':p,name+'_cum':c})
    fg,ga=p7_core.geometry('p7');L=p7_core.layout('p7')
    all_inputs.update({'field_'+n:v for n,v in ga.items()})
    all_inputs['field_volumes']=np.array(L['volumes'])
    # Also verify the separately sealed fluorescence and slab spectrum files.
    extra={}
    for filename in ('p8f_spectrum.npz',*(n+'_fluorescence.npz' for n in therapy_bundle.NAMES)):
        with np.load(Path(source)/filename,allow_pickle=False) as z:
            for n in z.files:
                all_inputs[filename+'.'+n]=z[n];extra[filename+'.'+n]=z[n]
    for mid,name in enumerate(therapy_bundle.NAMES):
        for field in ('yields','lines','line_counts'):
            equal(np.asarray(getattr(materials[mid],field)),extra[name+'_fluorescence.npz.'+field],name+' fluorescence '+field)
    for field in ('edges','cum'):equal(arrays[field],extra['p8f_spectrum.npz.'+field],'slab spectrum '+field)
    therapy_bundle.create(out,materials,all_inputs,generators)
    restored,a,settings,_=therapy_bundle.load(out)
    rows={}
    for name,l,r in zip(therapy_bundle.NAMES,materials,restored):
        for f in catalog.P8_DEFINITIONS:rows[name+'.'+f]=equal(np.asarray(getattr(l,f),np.float64),getattr(r,f),name+'.'+f)
    for name in all_inputs:rows['input.'+name]=equal(all_inputs[name],a[name],name)
    data,meta=therapy.pack(restored);old_data,old_meta=old.pack(materials)
    rows['native.data']=equal(data,old_data,'native data (uninitialized log NaNs)')
    rows['native.meta']=equal(meta,old_meta,'native descriptors')
    native_field=therapy.FieldGeometry()
    for key in ('lo','hi','origins','widths','axes','sizes','offsets'):
        getattr(native_field,key)[:]=a['field_'+key]
    rows['native.field_geometry']=equal(np.frombuffer(bytes(native_field),np.uint8),np.frombuffer(bytes(fg),np.uint8),'native field geometry ABI')
    c=therapy.CPU(Path(out).parent/'build',restored,a,field_geometry=native_field)
    for key in ('faces','rho','bins','strength','edges','cum','ids','owner'):
        rows['native.'+key]=equal(c.arrays[key],arrays[key],'native '+key)
    from prototypes.mv_electron.gpu.p8g2 import CPU as OldCPU
    ref=OldCPU(Path(out).parent/'old-build',materials)
    # Constants must be the sealed constants in both initializations.
    equal(c.constants,arrays['constants'],'constants')
    for mid,q in enumerate(c.meta):
        rows['logs.'+str(mid)]=equal(c.data[q[132]:q[133]],ref.data[ref.meta[mid,132]:ref.meta[mid,133]],'initialized logs')
    report=dict(status='pass',fields_per_material=33,rows=rows,observed=c.observed,source_manifest=therapy_bundle.sha(Path(source)/'p8g2_manifest.json'))
    common.write(summary,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=common.SOURCE);p.add_argument('--out',type=Path);p.add_argument('--summary',type=Path,default=common.SUMMARY/'bundle.json');args=p.parse_args()
    if args.out is None:
        # Sandbox-safe small-lived copy, retained (no deletion) for inspection.
        args.out=Path(tempfile.mkdtemp(prefix='chatcarlo-a2-',dir='/private/tmp'))/'bundle'
    print(run(args.source,args.out,args.summary)['status'])
