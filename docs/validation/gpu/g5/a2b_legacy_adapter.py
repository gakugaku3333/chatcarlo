"""Validation-only old-kernel CLI adapter: 50 batches, 0.5 mm, common recording.

Never imported by package runtime. Baseline conditions are rejected by the normal
entry and B3 judge. Host-added history counters/physics labels on the OLD side
are provenance only and do not claim independent native observation.
"""
from contextlib import ExitStack
import ctypes as ct
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from chatcarlo.gpu import seed_entry,seed_condition,seed_colab,seed_cuda,seed_core,physics_sets,therapy,contracts

REAL_CONDITION=seed_condition.condition


def condition(*args):
    from prototypes.mv_electron.gpu import p9c_core
    c=REAL_CONDITION(*args);c['baseline_only']='frozen-p9c-0.5mm-50-batches';c['legacy_sources']=p9c_core.fingerprint();return c


class GPU:
    def __init__(self,cp,materials,*,constants,strength,**kw):
        from prototypes.mv_electron.gpu.p9c_colab import GPU as Old
        from chatcarlo.gpu import seed_geometry
        kw.setdefault('grids',(seed_geometry.FINE,seed_geometry.COARSE))
        self.e=Old(cp,materials,**kw);self.cp=cp;self.constants=constants;self.strength=strength
    @property
    def g(self):return self.e.g
    def upload(self):self.e.upload()
    def run(self,n,seed,start=0,field=0,**kw):
        r=self.e.run(n,seed,start=start,field=field,**kw);h=r['histories']
        d=np.column_stack([h[k] for k in ('totals','maxima','brems','p8_max','sources','dep')]+[h['diagnostic'].reshape(n,14)])
        i=np.column_stack([h[k] for k in ('violations','counts','steps','boundary_counts','boundary_hist','boundary_omitted_hist','fluorescence_counts','branches','calls','diagnostic_count','maxdepth','nspec_index')])
        # OLD code has no independent history completion record or set ID kernel.
        # This adapter is restricted to B5 timing, never B1/B3 evidence.
        counts=np.column_stack((r['source_counts'],np.arange(start,start+n),np.ones(n,np.int64)))
        cp=self.cp;arrays={k:cp.asnumpy(v) for k,v in self.e.device_arrays.items()};arrays['strength']=self.strength
        arrays['bins']=np.empty(0);arrays['owner']=np.empty(0,np.int64)
        obs=seed_core.observation(self.e.g,arrays,tuple(cp.asnumpy(v) for v in self.e.args),seed,start,n,field,physics_sets.declaration(therapy.NAME))
        obs['scoring']=__import__('hashlib').sha256(bytes(therapy.FieldGeometry())).hexdigest()
        return dict(d=d,i=i,ledger=r['ledger'],source_counts=counts,grids=[cp.asnumpy(v) for v in r['grids']],seconds=r['seconds'],observed=obs)


def worker(root):
    # Load independent NEW-kernel gates before replacing only production driver.
    from chatcarlo.gpu import seed_t4_checks
    # seed_gates imports seed_cuda.GPU lazily, so gates must run with the real NEW GPU restored.
    real_gpu=seed_cuda.GPU;real_checks=seed_t4_checks.run
    def checks(*a,**kw):
        with patch.object(seed_cuda,'GPU',real_gpu):return real_checks(*a,**kw)
    with patch.object(seed_condition,'condition',condition),patch.object(seed_cuda,'GPU',GPU),patch.object(seed_t4_checks,'run',checks):
        seed_colab.worker(root)
    p=Path(root)/'results/summary.json';r=json.loads(p.read_text());r['baseline_only']=True;contracts.write(p,r)
    seed_colab.publish(Path(root)/'results')


def main(argv=None,*,client=None):
    real_build=seed_colab.build;real_bootstrap=seed_colab.bootstrap;real_run=seed_colab.run
    def build(out):
        import zipfile
        transfer,_=real_build(out)
        with zipfile.ZipFile(transfer/'source.zip','a',zipfile.ZIP_DEFLATED) as z:
            for name in ('a2b_legacy_adapter.py','a2b_speed.py','a3_entry_speed.py'):
                z.write(Path(__file__).with_name(name),name)
            from prototypes.mv_electron.gpu import p9c_core
            root=p9c_core.HERE.parents[2]
            for folder in (root/'prototypes',):
                for p in sorted(folder.rglob('*')):
                    if p.suffix in ('.py','.h','.cpp','.cu'):z.write(p,str(p.relative_to(root)))
            for n in ('metadata.json','estar_water.html','estar.npz'):
                p=root/'docs/validation/mv/p1/data'/n;z.write(p,str(p.relative_to(root)))
        from chatcarlo.gpu import colab_driver
        return transfer,colab_driver.split_bundle(transfer,transfer/'parts_seed')
    def bootstrap(remote,files):
        code=real_bootstrap(remote,files)
        # Condition verifier also needs the baseline declaration before launching.
        code=code.replace('from chatcarlo.gpu.seed_condition import verify',"import a2b_legacy_adapter as legacy\nfrom chatcarlo.gpu import seed_condition\nseed_condition.condition=legacy.condition\nfrom chatcarlo.gpu.seed_condition import verify")
        return code.replace("'chatcarlo.gpu.seed_colab','worker'","'a2b_legacy_adapter','worker'")
    with ExitStack() as stack:
        stack.enter_context(patch.object(seed_condition,'condition',condition));stack.enter_context(patch.object(seed_colab,'build',build));stack.enter_context(patch.object(seed_colab,'bootstrap',bootstrap))
        if client is not None:stack.enter_context(patch.object(seed_colab,'run',lambda *a,**kw:real_run(*a,**kw,client=client)))
        return seed_entry.main(argv)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(Path(sys.argv[2]))
    else:raise SystemExit(main())
