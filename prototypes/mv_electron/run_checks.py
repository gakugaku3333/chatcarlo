"""Reproduce the report solely from saved raw data (no transport, no network)."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from prototypes.mv_electron.component_checks import OUT
def main():
    p=argparse.ArgumentParser();p.add_argument('--from-saved',action='store_true',required=True);p.parse_args()
    from prototypes.mv_electron.check_transport import main as transport
    from prototypes.mv_electron.check_egs5 import main as egs
    from prototypes.mv_electron.check_diagnostics import main as diagnostic
    transport();egs();diagnostic()
    from prototypes.mv_electron.report import main as report
    report()
if __name__=='__main__':main()
