"""G5 A4: EGS5 P7-field runs with Rayleigh + bound Compton ON (PEGS5 data and user code).

Runs the unchanged prototypes/mv_electron/run_p3b2.run_egs with scoped paths, a user-code directory whose
p3b_egs.f differs only by iraylr(2)=1 / incohr(2)=1, a PEGS5 input with IRAYL=1,IBOUND=1,INCOH=1, and one
removed check (equality with the P3a PEGS input, which is physics OFF by design). Nothing under prototypes/ is edited.
"""
import hashlib
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT)]
from prototypes.mv_electron import run_p3b2 as egs  # noqa: E402
from prototypes.mv_electron.gpu import p7_core as h  # noqa: E402

RUN = HERE/'a4_run'
CODE = RUN/'egs_code'
CACHE = Path.home()/'.cache/chatcarlo/g5/a4'
OFF = 'IBOUND=0,INCOH=0,ICPROF=0,IRAYL=0'
ON = 'IBOUND=1,INCOH=1,ICPROF=0,IRAYL=1'
WANT_ON = dict(IBOUND=1, INCOH=1, ICPROF=0, IRAYL=1, incohm=1, incohr=1, iraylm=1, iraylr=1, iprofm=0, iprofr=0, iedgfl=0, iauger=0, iprdst=0, ibrspl=0,
               iausfl8=1, usegsd=1)
FLAGS_OFF = '      iraylr=0\n      incohr=0\n'
CHECK = ("        old = ROOT/'docs/validation/mv/p3a/egs5/p3a_egs/config.json'\n"
         "        if json.loads(old.read_text())['input'] != config['input']:\n"
         "            raise RuntimeError('P3a PEGS input mismatch')\n")


def mono_edges(energy_keV):
    """One 0.2 keV bin centred on the line, in MeV (plan A4-D8). The engine side must use the same numbers."""
    return (np.array([round((energy_keV-0.1)/1000, 9), round((energy_keV+0.1)/1000, 9)]), np.array([1.0]))


def build_code(physics_on=True):
    CODE.mkdir(parents=True, exist_ok=True)
    for name in egs.SOURCES:
        target = CODE/name
        if name == 'p3b_egs.f':
            text = (egs.CODE/name).read_text()
            if text.count(FLAGS_OFF) != 1:
                raise RuntimeError('unexpected p3b_egs.f flag block')
            if physics_on:
                text = text.replace(FLAGS_OFF, FLAGS_OFF+'      iraylr(2)=1\n      incohr(2)=1\n')
            if not target.exists():
                target.write_text(text)
            elif target.read_text() != text:
                raise RuntimeError('egs_code/p3b_egs.f differs from the expected patch')
        elif not target.exists():
            target.symlink_to(egs.CODE/name)


@contextmanager
def scope(spectrum_sha, physics_on=True):
    build_code(physics_on)
    original = egs.r.pegs_input

    def pegs_input(mode):
        text = original(mode)
        if text.count(OFF) != 1:
            raise RuntimeError('unexpected PEGS5 input')
        return text.replace(OFF, ON) if physics_on else text
    parse_original = egs.r.parse_egs

    def parse_egs(work, config):
        """The legacy 'flags' test requires every physics flag 0; for ON runs require exactly WANT_ON instead."""
        v, summary = parse_original(work, config)
        if physics_on:
            tests = summary['tests']
            tests['flags'] = {k: int(x) for k, x in summary['effective_flags'].items()} == WANT_ON
            summary['status'] = '合格' if all(tests.values()) else '不合格'
        return v, summary
    source = inspect.getsource(egs.run_egs)
    if source.count(CHECK) != 1:
        raise RuntimeError('unexpected run_egs source')
    saved = egs.CODE, egs.CACHE, egs.OUT, egs.CHECKS, egs.run_egs, egs.r.SPECTRUM_SHA, egs.r.pegs_input, egs.r.parse_egs
    namespace = egs.__dict__
    try:
        egs.CODE, egs.CACHE, egs.OUT, egs.CHECKS = CODE, CACHE, RUN, RUN/'checks_none.json'
        egs.r.SPECTRUM_SHA, egs.r.pegs_input, egs.r.parse_egs = spectrum_sha, pegs_input, parse_egs
        exec(compile(source.replace(CHECK, ''), '<a4_run_egs>', 'exec'), namespace)
        yield
    finally:
        egs.CODE, egs.CACHE, egs.OUT, egs.CHECKS, egs.run_egs, egs.r.SPECTRUM_SHA, egs.r.pegs_input, egs.r.parse_egs = saved


def run(label, N, seed, energy_keV, *, half=5.0, ssd=100.0, physics_on=True, resume=False):
    edges, cum = mono_edges(energy_keV)
    sha = hashlib.sha256(egs.spectrum_text(edges, cum).encode()).hexdigest()
    with scope(sha, physics_on):
        return egs.run_egs(label, N, seed, edges, cum, field=[half, ssd],
                           regions=h.layout('p7')['regions'], resume=resume)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('label')
    p.add_argument('--N', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--energy-keV', type=float, required=True)
    p.add_argument('--physics-off', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    print(run(a.label, a.N, a.seed, a.energy_keV, physics_on=not a.physics_off, resume=a.resume))
