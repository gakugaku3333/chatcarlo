"""Fetch official NIST XCOM 3.1 archive and provenance (no reference values typed)."""
from pathlib import Path
import subprocess, hashlib, json, datetime, tarfile
BASE=Path(__file__).resolve().parent
URLS={
 'XCOM.tar.gz':'https://physics.nist.gov/PhysRefData/Xcom/XCOM.tar.gz',
 'xcom_download.html':'https://physics.nist.gov/PhysRefData/Xcom/Text/download.html',
 'xcom_version.html':'https://physics.nist.gov/PhysRefData/Xcom/Text/version.shtml',
 'nist_license.html':'https://www.nist.gov/open/license',
 'water_composition.html':'https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=276',
 'air_composition.html':'https://physics.nist.gov/cgi-bin/Star/compos.pl?matno=104',
}
def fetch():
 records=[]
 for name,url in URLS.items():
  p=BASE/'data'/name
  try:
   raw=subprocess.check_output(['curl','--fail','--location','--silent','--show-error','--max-time','60',url]); p.write_bytes(raw)
   records.append(dict(file=str(p.relative_to(BASE)),url=url,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
  except Exception as exc: records.append(dict(file=str(p.relative_to(BASE)),url=url,error=str(exc)))
 (BASE/'data/photon_fetch.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':records},indent=2)+'\n')
 archive=BASE/'data/XCOM.tar.gz'
 if archive.exists():
  with tarfile.open(archive) as t: t.extractall(BASE/'data',filter='data')
 print(json.dumps(records,indent=2))
if __name__=='__main__': fetch()
