"""P8g-2's 33 fields/material round-trip; no therapy engine is constructed."""
import argparse
import json
from pathlib import Path
import numpy as np
import a1_common
from chatcarlo.gpu import catalog

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def roundtrip(material,name,root=ROOT):
    generators={d['generator_path']:catalog.generator(d['generator_path'],root)
                for d in catalog.P8_DEFINITIONS.values()}
    cat,blob=catalog.p8_material(material,name,generators)
    restored=catalog.reconstruct(cat,{'p8':blob})
    for field in catalog.P8_DEFINITIONS:
        a=np.asarray(getattr(material,field),np.float64);b=restored[name+'.'+field]
        if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
            raise ValueError(name+'.'+field+': roundtrip mismatch')
    return cat,blob


def run(source,out,summary):
    from a1_common import space
    space(out)
    from prototypes.mv_electron.gpu.p8g2_colab import load
    _,materials,_=load(source)
    if len(materials)!=4:raise ValueError('four P8 materials required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    reports={}
    for name,material in zip(('water','lung','bone','lead'),materials):
        space(out)
        cat,blob=roundtrip(material,name)
        (out/(name+'_catalog.json')).write_text(json.dumps(cat,indent=2)+'\n')
        np.savez(out/(name+'_catalog_blob.npz'),p8=blob)
        # Read the persisted representation, not merely an in-memory reshape.
        with np.load(out/(name+'_catalog_blob.npz')) as z:
            restored=catalog.reconstruct(json.loads((out/(name+'_catalog.json')).read_text()),{'p8':z['p8']})
        for f in catalog.P8_DEFINITIONS:
            if restored[name+'.'+f].tobytes()!=np.asarray(getattr(material,f),np.float64).tobytes():
                raise ValueError(name+'.'+f+': persisted roundtrip mismatch')
        reports[name]=dict(fields=len(cat['fields']),sha256=catalog.digest(blob),
                           explicit_shapes={f:list(np.shape(getattr(material,f))) for f in ('ne','gs_y','lines')})
    Path(summary).parent.mkdir(parents=True,exist_ok=True)
    Path(summary).write_text(json.dumps(dict(status='pass',materials=reports),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path.home()/'.cache/chatcarlo/p8g2/export')
    p.add_argument('--out',type=Path,default=Path.home()/'.cache/chatcarlo/g5/a1/catalog_p8')
    p.add_argument('--summary',type=Path,default=HERE/'a1_run/catalog_p8.json')
    a=p.parse_args();run(a.source,a.out,a.summary)
