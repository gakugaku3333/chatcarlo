"""Claude-only ten-minute watch; salvage and stop even after polling failure."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from .start import call,check_export
from .collect import collect


def poll_collect(session, output, identity):
    last=None
    for attempt in range(3):
        try:
            status=json.loads(call(['exec','-s',session,'-f',str(Path(__file__).with_name('status.py'))]).strip().splitlines()[-1])
            break
        except (subprocess.SubprocessError,ValueError,IndexError) as error:last=error
    else:raise RuntimeError('final status retrieval failed') from last
    def download(name,path):
        call(['download','-s',session,'/content/p8g3/'+name,str(path)],timeout=600)
    collect(status,output,identity,download)
    # Log is diagnostic; it does not overwrite any sealed artifact.
    call(['download','-s',session,'/content/p8g3/run.log',str(Path(output)/'run.log.pending')],timeout=600)
    (Path(output)/'run.log.pending').replace(Path(output)/'run.log')
    print(status['state'],status['tail'],flush=True)
    return status


def watch(session,output,identity):
    try:
        while True:
            status=poll_collect(session,output,identity)
            if status['state']=='failed':raise RuntimeError('VM driver failed or PID disappeared')
            if status['state']=='done':return
            for _ in range(10):time.sleep(60)
    finally:
        # Also attempted after a failed download, exhausted poll or interrupt.
        try:poll_collect(session,output,identity)
        finally:call(['stop','-s',session])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',default='p8g3')
    parser.add_argument('--export',type=Path,default=Path.home()/'.cache/chatcarlo/p8g3/export')
    parser.add_argument('--output',type=Path,default=Path.home()/'.cache/chatcarlo/p8g3/colab')
    args=parser.parse_args();manifest=check_export(args.export)
    watch(args.session,args.output,manifest['identity'])

if __name__=='__main__':main()
