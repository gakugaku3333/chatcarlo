"""関門 D（Claude、2026-10-23）: 承認済みの gate_d で、VM の最終の引数に故障を入れ、照合で止まるかを見る。
使い方: run_fault.py RUN_DIR FAULT OUT.json"""
import json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
from chatcarlo.gpu import approval, bundle
from chatcarlo.gpu.colab_driver import launch

run, fault, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
job = json.loads((run/'job.json').read_text())
results = run/f'fault_{fault}'; results.mkdir()
gates = ROOT/'docs/validation/gpu/g1'
with tempfile.TemporaryDirectory() as t:
    fp = Path(t)/'fault.json'; fp.write_text(json.dumps(dict(fault=fault)))
    launch(run/'bundle', job, results, gate1=gates/'gate1.json', gate2=gates/'gate2.json',
           session='chatcarlo-g2b-'+fault.replace('_', '-'), fault_path=fp)
try:
    approval.verify_batches(run/'bundle', job, results, set())
    rep = dict(fault=fault, stopped=False)
except ValueError as e:
    rep = dict(fault=fault, stopped=True, message=str(e))
rep['npz_written'] = any(results.glob('dose*.npz'))
rep['status'] = 'pass' if rep['stopped'] and not rep['npz_written'] and ('observation mismatch' in rep.get('message', '')) else 'fail'
out.write_text(json.dumps(rep, ensure_ascii=False, indent=1)+'\n'); print(rep)
