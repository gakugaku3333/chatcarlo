"""Claude-only upload/re-send/start. Never executed by Codex."""
import argparse
from pathlib import Path
import subprocess
import tempfile
from .. import p8g3_core as core


def call(args,timeout=180,input=None):
    return subprocess.run(['colab',*args],input=input,text=True,capture_output=True,
                          check=True,timeout=timeout).stdout


def check_export(root):
    manifest=core.read_json(Path(root)/'p8g3_manifest.json')
    if manifest['status']!='pass':raise ValueError('export status')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or core.kernel.sha(Path(root)/name)!=digest:raise ValueError('export hash '+name)
    from ..p8g2_colab import load
    sealed,_,_=load(root)
    if sealed['identity']!=manifest['identity']:raise ValueError('export identity')
    return manifest


def resend_files(output,identity):
    files=[];output=Path(output)
    for group,schedule in (('trial',core.spec(4,2000)),('results',core.spec())):
        root=output/group
        calls=core.available(root,identity,schedule)
        sessions={r['session_id'] for r in calls.values()}
        files.extend(core.session_path(root,sid) for sid in sorted(sessions))
        # trial-resume may have a session that seals no new call (already full).
        if (root/'trial_qualified.json').exists():
            from ..p8g3_colab import require_trial
            record=require_trial(root,identity)
            path=core.session_path(root,record['new_session'])
            if path not in files:files.append(path)
        for j in sorted(calls):
            path=core.call_path(root,j);files.extend([path,path.with_suffix('.json')])
        for name in ('complete.json','trial_qualified.json'):
            path=root/name
            if path.exists():
                if name=='complete.json':core.verify(root,identity,schedule)
                files.append(path)
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,default=Path.home()/'.cache/chatcarlo/p8g3/export')
    parser.add_argument('--output',type=Path,default=Path.home()/'.cache/chatcarlo/p8g3/colab')
    parser.add_argument('--session',default='p8g3')
    parser.add_argument('--mode',choices=('production','trial-save','trial-resume'),default='production')
    args=parser.parse_args();manifest=check_export(args.export)
    files=resend_files(args.output,manifest['identity']) # reject before VM creation
    if args.mode=='production':
        from ..p8g3_colab import require_trial
        require_trial(args.output/'trial',manifest['identity'])
    created=False
    try:
        print(call(['new','-s',args.session,'--gpu','T4']),flush=True);created=True
        call(['exec','-s',args.session],input='from pathlib import Path\nfor n in ("", "results", "trial", "parts"):Path("/content/p8g3",n).mkdir(parents=True,exist_ok=True)\n')
        with tempfile.TemporaryDirectory(prefix='p8g3-upload-') as tmp:
            for name in (*manifest['files'],'p8g3_manifest.json'):
                source=args.export/name
                if source.stat().st_size<=150000000:
                    call(['upload','-s',args.session,str(source),'/content/p8g3/'+name],timeout=600)
                else:
                    with source.open('rb') as stream:
                        index=0
                        while chunk:=stream.read(50*1024*1024):
                            part=Path(tmp)/(name+f'.part.{index:05}');part.write_bytes(chunk)
                            call(['upload','-s',args.session,str(part),'/content/p8g3/parts/'+part.name],timeout=600);index+=1
        # Every referenced session is uploaded before its calls.
        for source in files:
            call(['upload','-s',args.session,str(source),'/content/p8g3/'+str(source.relative_to(args.output))],timeout=600)
        script=Path(__file__).with_name('assemble.py').read_text()
        print(call(['exec','-s',args.session],input='MODE='+repr(args.mode)+'\n'+script),flush=True)
    except BaseException:
        if created:
            from .watch import poll_collect
            try:poll_collect(args.session,args.output,manifest['identity'])
            finally:call(['stop','-s',args.session])
        raise
    print('Run python -m prototypes.mv_electron.gpu.p8g3_cli.watch --session '+args.session)

if __name__=='__main__':main()
