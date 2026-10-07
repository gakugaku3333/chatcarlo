"""Fetch Mohan 10 MV data and its bin-format definition from NRC's public repository.
Only spectrum/reference files are fetched; no second transport code is installed.
"""
from pathlib import Path
import subprocess,json,hashlib,datetime
BASE=Path(__file__).resolve().parent
REPO='https://raw.githubusercontent.com/nrc-cnrc/EGSnrc/master/'
URLS={
'mohan10.spectrum':REPO+'HEN_HOUSE/spectra/egsnrc/mohan10.spectrum',
'spectra_readme.txt':REPO+'HEN_HOUSE/spectra/egsnrc/README',
'egsnrc_license.txt':REPO+'LICENCE',
'egs_spectra.cpp':REPO+'HEN_HOUSE/egs++/egs_spectra.cpp',
'egs_spectra.h':REPO+'HEN_HOUSE/egs++/egs_spectra.h',
}
def main():
 records=[]
 for name,url in URLS.items():
  p=subprocess.run(['curl','--fail','--silent','--show-error','--location','--max-time','60',url],capture_output=True)
  r={'name':name,'url':url,'exit_code':p.returncode,'stderr':p.stderr.decode()}
  if p.returncode==0:
   (BASE/'data'/name).write_bytes(p.stdout);r.update(sha256=hashlib.sha256(p.stdout).hexdigest(),bytes=len(p.stdout))
  records.append(r)
 (BASE/'data/spectrum_fetch.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records},indent=2)+'\n');print(json.dumps(records,indent=2))
if __name__=='__main__':main()
