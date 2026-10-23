"""Strict MV parser, reached before the diagnostic CLI parses its arguments."""
import argparse
from pathlib import Path
import sys
import time
from . import scene, condition, preview, bundle
from ..gpu import p5b_core as core


def default_out(cond):
    return Path('~/.cache/chatcarlo/mv_entry').expanduser()/condition.digest(cond)[:12]


def main(argv=None):
    p = argparse.ArgumentParser(prog='chatcarlo (therapy_mv)')
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('validate', 'preview', 'approve', 'run', 'trace', 'animate', 'plot'):
        command = sub.add_parser(name)
        command.add_argument('scene')
        if name in ('preview', 'run'):
            command.add_argument('--out', type=Path)
        if name == 'approve':
            command.add_argument('--preview', type=Path, required=True)
    raw = list(argv if argv is not None else sys.argv[1:])
    if raw and raw[0] in ('trace', 'animate', 'plot'):
        print('[エラー] 治療領域では未対応', file=sys.stderr)
        return 1
    args = p.parse_args(raw)
    try:
        s = scene.load(args.scene)
        if args.command == 'validate':
            print(f'OK: {args.scene} は有効な治療領域シーンです（{preview.LABELS[s["source"]["spectrum"]][2]}、水30 cm、10×10 cm²、SSD 100 cm）')
            return 0
        cond, mat, data = condition.identities(s)
        if args.command == 'approve':
            record = preview.approve(s, args.preview, data)
            print('承認記録: '+str(args.preview/'approval.json'))
            return 0
        out = args.out or default_out(cond)
        if args.command == 'preview':
            preview.write(s, out, data)
            print('3D表示を確認して承認してください: '+str(out/'preview.html'))
            return 0
        preview.require_approval(out, cond, mat)  # must precede any bundle/output creation
        timings = {}; before = time.perf_counter()
        bundle.build(out/'bundle', cond, mat, data, out)
        timings['bundle_creation'] = time.perf_counter()-before
        if cond['engine'] == 'cpu_local':
            from .vm_driver import run
            report = run(out/'bundle', out/'results')
        else:
            from .colab_cli import run
            run(out/'bundle', out/'results')
            report = bundle.read_json(out/'results/driver_status.json')
        before = time.perf_counter()
        bundle.collect(out/'results', cond, mat)
        timings['verification'] = time.perf_counter()-before
        result = dict(status='CPU local provenance verified')
        if cond['engine'] == 'colab_t4':
            from .compare import analyse, profiles
            before = time.perf_counter()
            result = (profiles if cond['spectrum'] == 'mohan4' else analyse)(out/'bundle', out/'results')
            timings['analysis'] = time.perf_counter()-before
        timings.update(report.get('timings', {}))
        colab_timing = out/'colab_timings.json'
        if colab_timing.exists():
            timings.update(bundle.read_json(colab_timing))
        record = dict(histories=cond['histories'], condition_hash=condition.digest(cond), materials_hash=condition.digest(mat),
                      timings=timings, gates=report.get('gates'), fault=report.get('fault'), comparison=result)
        core.write_json(out/'run_record.json', record)
        if result.get('user_attention_required'):
            print('[注意] 本番の対称性で偏りを検出。抄録に数値を渡す前に使用者へ報告が必要です。')
        print('結果: '+str(out/'run_record.json'))
        return 0
    except (OSError, ValueError, core.Stop, KeyError) as exc:
        print('[エラー] '+str(exc), file=sys.stderr)
        return 1
