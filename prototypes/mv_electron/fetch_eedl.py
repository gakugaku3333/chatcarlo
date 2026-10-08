"""Fetch EPICS2025 bytes and verify preregistered hashes before exposing inputs."""
import hashlib,json,urllib.request
from pathlib import Path
from .physics import ROOT
DATA=ROOT/'docs/validation/mv/p1b/data'
BASE='https://nuclear.llnl.gov/EPICS/'
EXPECTED={'ZA001000':(101409,'814c909e819e185b96511e47ff0d5e4c1ba06650cf28a45a9769ef95ceb8dfd7'),'ZA008000':(132671,'87b39aed3276f1c2db52b6cb2878871025245890dca80368b2c4b7a85292eb1b')}
def main():
    DATA.mkdir(exist_ok=True);rows={}
    for name,(size,digest) in EXPECTED.items():
        url=BASE+'ENDF2025/EEDL.ELEMENTS/'+name;b=urllib.request.urlopen(url,timeout=60).read();actual=hashlib.sha256(b).hexdigest()
        if len(b)!=size or actual!=digest:raise ValueError(f'{name}: unregistered input bytes')
        (DATA/name).write_bytes(b);rows[name]={'url':url,'bytes':len(b),'sha256':actual}
    (DATA/'eedl_provenance.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
