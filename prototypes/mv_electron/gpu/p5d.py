"""P5d local A/B measurements. Retains every output; no deletion."""
import argparse
from pathlib import Path
import platform
import subprocess
import time
from . import p5d_core as d
from . import p5b_core as core


def local(work, saved):
    from . import p5a, p5b
    work = Path(work)
    report = dict(overall='incomplete', checks={}, measurements=[],
                  environment=dict(machine=platform.machine(), processor=platform.processor()),
                  p4_boundary='run_coupled plus result formatting; one process, MT')
    try:
        if platform.system() == 'Darwin':
            report['environment']['cpu_model'] = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        data, config = p5b.load_local_inputs(saved)
        mt = p5a.CPU(work/'build', data, rng='mt')
        cpu = d.CPU(work/'build', data)
        if cpu.compiler != config['compiler']:
            raise core.Stop('saved compiler mismatch')
        report['environment'].update(compiler=cpu.compiler, flags=cpu.flags)
        warm = p5a.p4_run(data, 100, core.SEED)
        core.require(d.cpu_health(warm, 100), 'P4 warmup health')
        a = p5a.p4_run(data, 20000, 20291951, history_rng=False)
        b = mt.run(20000, 20291951, history_rng=False)
        for r in (a, b):
            core.require(d.cpu_health(r, 20000), 'A-g health')
        report['checks']['A-g'] = dict(**core.require(p5a.compare_outputs(a, b), 'A-g'),
                                     configuration=d.metadata('A-g', 20000, seed=20291951))
        b, _ = d.timed_cpu(cpu, 100000)
        report['checks']['B-g'] = dict(**core.require(p5a.compare_outputs(p5b.saved_batch(saved, 0, config), b), 'B-g'),
                                     configuration=d.metadata('B-g', 100000))
        for label in ('A', 'B'):
            for repeat in range(3):
                meta = d.metadata(label, 50000)
                if label == 'A':
                    begin = time.perf_counter()
                    r = p5a.p4_run(data, 50000, core.SEED, history_rng=False)
                    times = dict(call_s=time.perf_counter()-begin, inner_s=float(r['seconds']))
                    core.require(d.cpu_health(r, 50000), 'A health')
                else:
                    r, times = d.timed_cpu(cpu, 50000)
                core.save_npz(work/f'{label}_{repeat}.npz', r, meta)
                report['measurements'].append(dict(**meta, repeat=repeat, **times))
                core.write_json(work/'local.json', report)
        report['overall'] = 'pass'
    except Exception as exc:
        report.update(overall='stopped', stop=str(exc))
        raise
    finally:
        core.write_json(work/'local.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['local'])
    parser.add_argument('--work', type=Path, default=Path('~/.cache/chatcarlo/p5/p5d').expanduser())
    parser.add_argument('--saved', type=Path, default=Path('~/.cache/chatcarlo/p5/p5a/philox').expanduser())
    args = parser.parse_args()
    local(args.work, args.saved)


if __name__ == '__main__':
    main()
