"""Report-only ncu capture of the P5k drivers F, Q1 and Q8 (why Q is slower). Never parses CSV on Colab.

No gate is run or required: the variants passed gates 0-2 in P5k; this only records raw profiles.
"""
from pathlib import Path
import subprocess
import sys
from . import p5b_core as core

VARIANTS=('F','Q1','Q8')
SECTIONS=('SpeedOfLight','SourceCounters','MemoryWorkloadAnalysis','MemoryWorkloadAnalysis_Tables',
          'WarpStateStats','Occupancy','LaunchStats','SchedulerStats','InstructionStats','ComputeWorkloadAnalysis')


def run_all(bundle,results):
    root=Path(results);root.mkdir(parents=True,exist_ok=True);report=dict(status='incomplete',profiles={},parse='local')
    for variant in VARIANTS:
        folder=root/variant;folder.mkdir(parents=True,exist_ok=True)
        cmd=['ncu','--replay-mode','kernel','--cache-control','all','--kernel-name','p5k_kernel','--launch-skip','1','--launch-count','1']
        for section in SECTIONS:cmd+=['--section',section]
        cmd+=['--export',str(folder/'profile'),sys.executable,'-m','prototypes.mv_electron.gpu.p5k_colab',
              '--target',variant,'--bundle',str(bundle),'--results',str(folder)]
        capture=subprocess.run(cmd,capture_output=True,text=True);(folder/'capture.log').write_text(capture.stdout+capture.stderr)
        state=dict(capture_returncode=capture.returncode,exports={});report['profiles'][variant]=state
        if capture.returncode==0:
            for page in ('source','details','raw'):
                cmd=['ncu','--import',str(folder/'profile.ncu-rep'),'--page',page,'--csv']
                if page=='source':cmd+=['--print-source','cuda,sass','--resolve-source-file',str(folder/'compiled.cu')]
                out=subprocess.run(cmd,capture_output=True,text=True)
                (folder/(page+'.csv')).write_text(out.stdout);(folder/(page+'.log')).write_text(out.stderr)
                state['exports'][page]=out.returncode
        core.write_json(root/'map.json',report)
    ok=all(v['capture_returncode']==0 and len(v['exports'])==3 and not any(v['exports'].values()) for v in report['profiles'].values())
    report['status']='captured' if ok else 'unavailable'
    core.write_json(root/'map.json',report);return report
