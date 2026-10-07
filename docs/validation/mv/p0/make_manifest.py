"""Freeze provenance and validate preregistration and unmodified EGS5 tree."""
from pathlib import Path
import hashlib,json,subprocess,datetime,sys
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
EGS=ROOT/'docs/egs5_crosscheck/egs5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 previous=json.loads((BASE/'manifest.json').read_text())
 pre=previous['preregistration_sha256'];assert sha(BASE/'PREREGISTRATION.md')==pre
 assert sha(ROOT/'docs/ai/plans/2026-10-07-mv-p0-feasibility.md')==previous['plan_sha256']
 tree=hashlib.sha256(b''.join(str(p.relative_to(EGS)).encode()+p.read_bytes() for folder in ('egs','pegs','auxcode','include','pegscommons','auxcommons','data') for p in sorted((EGS/folder).rglob('*')) if p.is_file())).hexdigest()
 runs={}
 for p in sorted((BASE/'egs5').glob('*/metadata.json')):
  m=json.loads(p.read_text());assert m['egs5_tree_sha256']==tree
  assert m['code_sha256']==sha(p.parent/f'{p.parent.name}.f')
  assert m['input_sha256']==sha(p.parent/f'{p.parent.name}.inp')
  assert m['preregistration_sha256']==m['preregistration_sha256_after']==pre
  runs[p.parent.name]=m
 records=[]
 excluded={'numba_cache','plot_cache','__pycache__'}
 for p in sorted(BASE.rglob('*')):
  rel=p.relative_to(BASE)
  if not p.is_file() or p.name=='manifest.json' or any(v in excluded for v in rel.parts):continue
  records.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':sha(p)})
 fetches={name:json.loads((BASE/'data'/name).read_text()) for name in ['photon_fetch.json','estar_fetch.json','spectrum_fetch.json']}
 refs=[ROOT/'chatcarlo/data/nist_xaamdi/water.csv',ROOT/'chatcarlo/kernel.py',ROOT/'chatcarlo/materials.py',ROOT/'docs/egs5_crosscheck/revalidation_2026_10_07/egs5/pdd_phantom/pdd_phantom.f',EGS/'data/density_corrections/compounds/water_liquid.density',ROOT/'.claude/agents/egs5-operator.md']
 result={'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preregistration_sha256':pre,'plan_sha256':previous['plan_sha256'],'seed':20261007,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'python':sys.version,'egs5_tree_sha256_after':tree,'runs':runs,'acquisition_records':fetches,'read_only_reference_files':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in refs],'files':records,'redistribution_note':'Raw SRD/spectrum/third-party reference source/density fixtures are locally ignored; no redistribution permission is asserted; no git commit performed.'}
 (BASE/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');print('Preregistration intact; EGS5 tree unchanged across all builds/runs;',len(records),'artifact hashes recorded')
if __name__=='__main__':main()
