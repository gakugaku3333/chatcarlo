"""Claude-only ten-minute polling with collection at every poll."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from .start import call


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',default='p8g2');parser.add_argument('--output',type=Path,default=Path.home()/'.cache/chatcarlo/p8g2/colab');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    while True:
        for attempt in range(3):
            try:
                status=json.loads(call(['exec','-s',args.session,'-f',str(Path(__file__).with_name('status.py'))]).strip().splitlines()[-1]);break
            except (subprocess.SubprocessError,ValueError):
                if attempt==2:raise
        print(status['state'],status['tail'],flush=True)
        for name in status['files']:
            destination=args.output/name;destination.parent.mkdir(parents=True,exist_ok=True)
            call(['download','-s',args.session,'/content/p8g2/'+name,str(destination)],timeout=600)
        if status['state'] in ('done','failed'):
            print(call(['stop','-s',args.session]),flush=True)
            if status['state']=='failed':raise RuntimeError('VM driver failed; results collected')
            return
        # Poll schedule is 10 minutes; short sleeps allow interruption.
        for _ in range(10):time.sleep(60)

if __name__=='__main__':main()
