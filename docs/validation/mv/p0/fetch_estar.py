"""Fetch ESTAR reference through NIST's documented text form, keep response/errors."""
from pathlib import Path
import subprocess,json,hashlib,datetime
BASE=Path(__file__).resolve().parent
URL='https://physics.nist.gov/cgi-bin/Star/e_table-t.pl'
def main():
 args=['curl','--fail','--silent','--show-error','--location','--max-time','60','--data-urlencode','matno=276','--data-urlencode','Energies=0.1\n1\n10\n',URL]
 p=subprocess.run(args,capture_output=True)
 record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'url':URL,'method':'POST','parameters':{'matno':'276','Energies':'0.1\n1\n10\n'},'exit_code':p.returncode,'stderr':p.stderr.decode()}
 if p.returncode==0:
  out=BASE/'data/estar_water.html';out.write_bytes(p.stdout);record['sha256']=hashlib.sha256(p.stdout).hexdigest()
 (BASE/'data/estar_fetch.json').write_text(json.dumps(record,indent=2)+'\n');print(record)
if __name__=='__main__':main()
