"""Upload a sealed export and launch once; never run by Codex."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def call(args,timeout=180,input=None):
    return subprocess.run(['colab',*args],input=input,text=True,capture_output=True,check=True,timeout=timeout).stdout


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--export',type=Path,default=Path.home()/'.cache/chatcarlo/p8g2/export');parser.add_argument('--session',default='p8g2');args=parser.parse_args()
    root=args.export;manifest=json.loads((root/'p8g2_manifest.json').read_text())
    if manifest['status']!='pass':raise ValueError('passed export required')
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('export hash '+name)
    print(call(['new','-s',args.session,'--gpu','T4']),flush=True)
    print(call(['exec','-s',args.session],input='from pathlib import Path\nfor p in ("/content/p8g2","/content/p8g2/results","/content/parts"):Path(p).mkdir(parents=True,exist_ok=True)\n'))
    for name in (*manifest['files'],'p8g2_manifest.json'):
        source=root/name
        if source.stat().st_size<=150000000:call(['upload','-s',args.session,str(source),'/content/p8g2/'+name],timeout=600)
        else:
            parts=root.parent/'cli/parts';parts.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as stream:
                index=0
                while chunk:=stream.read(50*1024*1024):
                    part=parts/(name+f'.part.{index:05}');part.write_bytes(chunk)
                    call(['upload','-s',args.session,str(part),'/content/parts/'+part.name],timeout=600);index+=1
    print(call(['exec','-s',args.session,'-f',str(Path(__file__).with_name('assemble.py'))]),flush=True)
    print('Poll/download every 10 minutes: python -m prototypes.mv_electron.gpu.p8g2_cli.watch --session '+args.session)

if __name__=='__main__':main()
