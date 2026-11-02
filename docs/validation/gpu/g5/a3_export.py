"""Claude-side export only. Reads private/cache inputs; never run in Codex H1/H2.

The runtime pins the A2-verified array contents, not NPZ container timestamps.
"""
import argparse
from pathlib import Path
from a2_bundle import run
from chatcarlo.gpu import therapy_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,default=therapy_inputs.DEFAULT_BUNDLE)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    run(source=a.source,out=a.out,summary=a.report)
    for spectrum in ('mohan10','mohan4'):therapy_inputs.load(a.out,spectrum)


if __name__=='__main__':main()
