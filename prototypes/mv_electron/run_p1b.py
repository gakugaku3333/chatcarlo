"""Reproducible gated calculations. Preregistration and pre-change baseline must exist.
No EGS5 executions, no p1 writes, no matching/fitting loops.
"""
import argparse,sys,json,contextlib,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prototypes.mv_electron.run_m1_p1b import main as m1,verify_p1
from prototypes.mv_electron.dcs import get_backend
P=ROOT/'docs/validation/mv/p1b'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['dcslib','eedl','both'],default='both');parser.add_argument('--benchmark-only',action='store_true');parser.add_argument('--resume-benchmark-log',type=Path);a=parser.parse_args()
    if not (P/'PREREGISTRATION.md').exists() or not (P/'m1_baseline.npz').exists():raise RuntimeError('freeze rules and pre-change run first')
    verify_p1();m1()
    from prototypes.mv_electron.run_checks_p1b import reanalyse_p1
    reanalyse_p1()
    from prototypes.mv_electron.check_dcs_p1b import inputs,scattering
    from prototypes.mv_electron.gs_backend import build
    from prototypes.mv_electron.check_gs_p1b import main as m4
    from prototypes.mv_electron.check_m5_p1b import main as m5
    from prototypes.mv_electron.run_transport import execute
    from prototypes.mv_electron.benchmark_p1b import main as benchmark
    import numpy as np
    names=['dcslib','eedl'] if a.backend=='both' else [a.backend]
    if not a.benchmark_only:(P/'m3.json').write_text(json.dumps(scattering(),indent=2)+'\n')
    for name in names:
        out=P/name;(out/'data').mkdir(parents=True,exist_ok=True)
        if a.benchmark_only:benchmark(name,out,a.resume_benchmark_log);continue
        inp=inputs(name);(P/f'm2_{name}.json').write_text(json.dumps(inp,indent=2)+'\n')
        if inp['status']!='pass':continue
        provider=get_backend(name)
        try:
            if not (out/'data/gs.npz').exists():build(provider,out/'data/gs.npz')
            r=m4(provider,np.load(out/'data/gs.npz'));r['rutherford_call_isolation']='requires passing tests/test_dcs_p1b.py (included in final checks)'
        except Exception as e:r={'status':'fail','reason':str(e),'traceback':traceback.format_exc()}
        (P/f'm4_{name}.json').write_text(json.dumps(r,indent=2)+'\n')
        if r['status']!='pass':continue
        r=m5(name,out);(P/f'm5_{name}.json').write_text(json.dumps(r,indent=2)+'\n')
        if r['status']!='pass':continue
        if not (out/'runs/base_10.npz').exists():execute('base_10',T=10.,n=1000000,output_dir=out,backend=name)
    verify_p1()
if __name__=='__main__':main()
