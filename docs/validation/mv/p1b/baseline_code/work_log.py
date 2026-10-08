"""Observable phase intervals, not invented human labor hours."""
from pathlib import Path
import json,datetime
from .component_checks import ROOT,OUT
def main():
    groups=[
      ('data','docs/validation/mv/p1/PREREGISTRATION.md','docs/validation/mv/p1/data/metadata.json'),
      ('tables','prototypes/mv_electron/physics.py','docs/validation/mv/p1/gs_checks.json'),
      ('transport','prototypes/mv_electron/transport.py','docs/validation/mv/p1/prototype_tests_v2.log'),
      ('boundary','prototypes/mv_electron/transport.py','docs/validation/mv/p1/runs/split_shifted.json'),
      ('validation','prototypes/mv_electron/check_fixed.py','docs/validation/mv/p1/diagnostics.json')]
    rows=[]
    for name,a,b in groups:
        pa=ROOT/a;pb=ROOT/b
        if not pb.exists():continue
        start=pa.stat().st_birthtime;end=pb.stat().st_mtime
        rows.append({'phase':name,'first_artifact':a,'final_artifact':b,'started_utc':datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),'finished_utc':datetime.datetime.fromtimestamp(end,datetime.timezone.utc).isoformat(),'elapsed_seconds':max(0,end-start)})
    benchmark=json.loads((OUT/'benchmark.json').read_text())
    ans={'rows':rows,'benchmark_interval':{'started_utc':benchmark['started_utc'],'finished_utc':benchmark['finished_utc'],'operator':'User reports completing the unchanged benchmark script during interruption'},'interruption_note':'Usage-limit interruption reported by user; precise pause boundaries and active work durations unavailable. Wall intervals must not be treated as labor hours.','measurement':'Filesystem creation/final-output timestamps (wall-clock intervals, overlapping). Includes waiting, concurrent computation and rework; not active human labor and not additive. No standalone labor stopwatch was recorded.','rework':[{'phase':'class-II','fact':'Near-threshold dimensionless subtraction lost relative precision; fixed with physical energy interval. K2 analytic/numeric agreement rechecked.'},{'phase':'GS tables','fact':'SciPy Q_l derivative returned NaN at large order; replaced unstable evaluation with mathematically identical backward continued fraction, validated by independent quadrature.'},{'phase':'K9(i)','fact':'200k histories missed SEM requirement; increased to 1M without changing criteria.'},{'phase':'transport','fact':'Removed repeated logarithm/array allocation and cached unchanged evaluated-energy coefficients; tests rerun.'},{'phase':'EGS5','fact':'GS ignores K1 scaling; sensitivity uses CHARD. EPE/NIPE perturbation left final fit unchanged; tested NALE=100.'}],'remaining_estimates_work_weeks':{'explicit_bremsstrahlung':['1','2–3','4–5'],'pair_production':['1','2–3','4–5'],'positrons':['1','2–3','3–5'],'coupled_stack_scoring_validation':['2–3','4–6','8–12']},'unmeasured':'Remaining widths are P0 engineering estimates, not P1 measurements. Full coupled necessary history count and active human work remain unmeasured; do not sum overlapping phases or multiply P0 EGS5 step counts by prototype costs.'}
    (OUT/'work_log.json').write_text(json.dumps(ans,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
