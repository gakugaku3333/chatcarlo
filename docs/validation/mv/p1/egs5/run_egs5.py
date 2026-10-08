"""Fresh scratch build/run based on revalidation recipe. EGS5 tree is read-only."""
from pathlib import Path
import subprocess,tempfile,shutil,json,hashlib,time,datetime,sys,resource,os
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[4]
EGS=ROOT/'docs/egs5_crosscheck/egs5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name):
 dest=BASE/name
 if (dest/'metadata.json').exists():raise RuntimeError('Existing run: use a new name')
 prereg=BASE.parent/'PREREGISTRATION.md';pre=sha(prereg)
 meta={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preregistration_sha256':pre,'code_sha256':sha(dest/f'{name}.f'),'input_sha256':sha(dest/f'{name}.inp'),'compiler':subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],'egs5run_sha256':sha(EGS/'egs5run')}
 meta['egs5_tree_sha256']=hashlib.sha256(b''.join(str(p.relative_to(EGS)).encode()+p.read_bytes() for folder in ('egs','pegs','auxcode','include','pegscommons','auxcommons','data') for p in sorted((EGS/folder).rglob('*')) if p.is_file())).hexdigest()
 with tempfile.TemporaryDirectory(prefix='mv_p1_') as tmp:
  d=Path(tmp)
  for ext in ['f','inp']:shutil.copy2(dest/f'{name}.{ext}',d)
  if (dest/'epstar.dat').exists():shutil.copy2(dest/'epstar.dat',d)
  with (dest/'build.log').open('w') as f:
   p=subprocess.run([str(EGS/'egs5run'),'comp'],cwd=d,input=name+'\n\n\n',text=True,stdout=f,stderr=subprocess.STDOUT)
  meta['build_exit_code']=p.returncode
  if p.returncode==0 and (d/'egs5job.exe').exists():
   start=time.perf_counter()
   with (dest/'run.log').open('w') as f:
    child=subprocess.Popen([str(d/'egs5job.exe')],cwd=d,stdout=f,stderr=subprocess.STDOUT)
    _,status,usage=os.wait4(child.pid,0)
    child.returncode=os.waitstatus_to_exitcode(status)
    meta['run_peak_rss_bytes']=usage.ru_maxrss
    p=child
   meta['child_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
   meta['rss_note']='Darwin child high-water includes build children; not isolated run RSS'
   meta['run_exit_code']=p.returncode;meta['wall_seconds']=time.perf_counter()-start
  for filename in ['egs5job.out','pgs5job.pegs5lst','egs5job.dummy','pgs5job.pegs5dat','egs5job.out99']:
   if (d/filename).exists():shutil.copy2(d/filename,dest)
 meta['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();meta['preregistration_sha256_after']=sha(prereg)
 (dest/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 assert pre==sha(prereg)
 print(name,json.dumps(meta))
 if meta.get('run_exit_code',1)!=0:raise RuntimeError('Build/run failure: see logs')
if __name__=='__main__':run(sys.argv[1])
