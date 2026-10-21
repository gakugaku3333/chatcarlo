"""Regression against the unchanged diagnostic main at HEAD, plus MV routing."""
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
import yaml
from chatcarlo import __main__ as cli

ROOT = Path(__file__).resolve().parents[1]
MV = str(ROOT/'examples/therapy/water_10mv_10x10.yaml')
DX = str(ROOT/'examples/chest_room.yaml')


@pytest.mark.parametrize('argv,code', [(['validate', MV], 0), (['run', MV, '--workers', '1'], 2),
    (['run', '--workers', '1', MV], 2),
    (['trace', MV], 1), (['animate', MV], 1), (['plot', 'missing.npz', '--scene', MV], 1),
    (['plot', 'missing.npz', '--scene='+MV], 1), (['approve', DX, '--preview', '/tmp/no_preview'], 1), (['approve', DX], 1)])
def test_mv_dispatch(argv, code, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['chatcarlo', *argv])
    try:
        actual = cli.main()
    except SystemExit as exc:
        actual = exc.code
    assert actual == code


def test_mv_diagnostic_run_options(monkeypatch):
    def check(args):
        assert args.dose_grid and args.workers == 1 and args.n_histories == 100
        return 7
    monkeypatch.setattr(cli, 'cmd_run', check)
    monkeypatch.setattr(sys, 'argv', ['chatcarlo', 'run', DX, '--dose-grid', '--workers', '1', '-n', '100'])
    assert cli.main() == 7


@pytest.mark.parametrize('example', sorted((ROOT/'examples').glob('*.yaml')))
def test_mv_diagnostic_output_unchanged(example, tmp_path, capsys, monkeypatch):
    if yaml.safe_load(example.read_text()).get('domain'):
        pytest.skip('diagnostic regression only')
    # Import the previous version under the chatcarlo package, preserving relative imports.
    previous = subprocess.check_output(['git', 'show', 'HEAD:chatcarlo/__main__.py'], cwd=ROOT, text=True)
    module_path = tmp_path/'previous_main.py'; module_path.write_text(previous)
    spec = importlib.util.spec_from_file_location('chatcarlo._previous_main', module_path)
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    for command in ('validate', 'preview'):
        out = tmp_path/'preview.html'
        argv = ['chatcarlo', command, str(example)]
        if command == 'preview':
            argv += ['--out', str(out)]
        monkeypatch.setattr(sys, 'argv', argv)
        old = baseline.main(); old_output = capsys.readouterr()
        old_html = out.read_bytes() if out.exists() else None
        new = cli.main(); new_output = capsys.readouterr()
        assert (new, new_output.out, new_output.err) == (old, old_output.out, old_output.err)
        if old_html is not None:
            assert out.read_bytes() == old_html


def test_mv_dispatch_options_before_scene(monkeypatch, tmp_path):
    from prototypes.mv_electron.entry import cli as mv
    argv = ['preview', '--out', str(tmp_path/'output'), MV]
    monkeypatch.setattr(mv, 'main', lambda got: 9 if got == argv else 0)
    assert cli._mv_dispatch(argv) == 9
