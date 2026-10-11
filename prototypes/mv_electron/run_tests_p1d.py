"""Run the plan's pytest and diff checks; sanitize before saving any output."""
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from prototypes.mv_electron.p1d_common import ROOT, OUT, write

def main():
    commands = [
        (['.venv/bin/python', '-m', 'pytest', 'tests/', '-q'], 'regression_tests.log'),
        (['git', 'diff', '--stat', '--', 'chatcarlo', 'tests', 'scripts', 'examples',
          'docs/validation/mv/p0', 'docs/validation/mv/p1', 'docs/validation/mv/p1b', 'docs/validation/mv/p1c'], 'diff_stat.log'),
        (['.venv/bin/python', '-m', 'pytest', 'prototypes/mv_electron/tests', '-q'], 'prototype_tests.log'),
    ]
    rows = []
    for command, log in commands:
        print(' '.join(command), flush=True)
        r = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        output = (r.stdout + r.stderr).replace(str(ROOT), '.').replace(str(Path.home()), '~')
        (OUT / log).write_text(output)
        rows.append(dict(command=' '.join(command), exit_code=r.returncode, log=log,
                         output=output, matches_initial_diff=output == (OUT / 'initial_diff_stat.log').read_text() if log == 'diff_stat.log' else None))
        write('test_commands.json', dict(commands=rows))
        print(log, r.returncode, output[-500:], flush=True)
        if r.returncode: raise RuntimeError('acceptance test failed; see ' + log)
    if not rows[1]['matches_initial_diff']: raise RuntimeError('protected diff changed since initial freeze')

if __name__ == '__main__': main()
