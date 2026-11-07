"""NVRTC compiles the GPU modules without the host C++ library.

Added 2026-11-07 after a T4 session was spent on a single compile error: A4u's
photon_physics.h returns ``NAN``, the diagnostic entry g1_gpu.cu has no guard for
it (t2_gpu.cu and t2_seed_gpu.cu do), and NVRTC refused the module with
``identifier "NAN" is undefined``. Nothing on the CPU side can see this: the CPU
build takes the ``#else`` branch and gets the macro from <cmath>.

This is the fourth time a GPU-only defect reached the real machine, so the lesson
is enforced here rather than written down again. The check is a preprocessor pass
over the same sources the launchers hand to CuPy, looking for host-only spellings
that survive into the ``P5_GPU`` branch.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Macros the host <cmath>/<climits> provide and NVRTC does not.
HOST_MACROS = ('NAN', 'INFINITY', 'HUGE_VAL', 'HUGE_VALF', 'M_PI', 'M_E', 'M_SQRT2',
               'DBL_MAX', 'DBL_MIN', 'DBL_EPSILON', 'FLT_MAX', 'FLT_MIN', 'FLT_EPSILON',
               'INT_MAX', 'INT_MIN', 'UINT_MAX', 'LLONG_MAX', 'SIZE_MAX', 'RAND_MAX')
PREPROCESSOR = shutil.which('clang') or shutil.which('gcc') or shutil.which('cc')


def modules():
    """(label, source, options) for every module a launcher compiles with NVRTC."""
    from chatcarlo.gpu import core, therapy, seed_core
    common = ('-DP5_GPU', '-DP5_RNG_PHILOX')
    local = core.CUDA_OPTIONS + ('-DG5_SET_ID=1', '-DG5_SET_TAG=1ULL', '-DG5_PHOTON_STACK_CAP=18',
                                 '-DG5_PHOTON_FAULT=0', '-DG5_TEST_TWO_PHOTONS=0')
    electron = common + ('-DT2_FIELD=1', '-DT2_SOURCE_CHECK=0', '-DG5_SET_ID=2',
                         '-DG5_SET_TAG=1ULL', '-DP8G_STACK_CAP=18')
    seed = common + ('-DT2_SEED=1', '-DP8G_STACK_CAP=18', '-DT2S_CHECK=0',
                     '-DT2_SEED_FAULT=0', '-DG5_SET_ID=2', '-DG5_SET_TAG=1ULL')
    out = []
    for legacy in (0, 1):
        out.append((f'local-deposition(legacy={legacy})', core.cuda_source(),
                    local + (f'-DG5_TEST_LEGACY_LOCAL={legacy}',)))
        out.append((f'electron-transport(legacy={legacy})', therapy.cuda_source(),
                    electron + (f'-DG5_TEST_LEGACY_ELECTRON={legacy}',)))
        out.append((f'seed(legacy={legacy})', seed_core.cuda_source(),
                    seed + (f'-DG5_TEST_LEGACY_ELECTRON={legacy}',)))
    return out


def preprocess(source, options):
    with tempfile.NamedTemporaryFile('w', suffix='.cu', delete=False) as handle:
        handle.write(source)
        path = handle.name
    try:
        done = subprocess.run([PREPROCESSOR, '-E', '-P', '-x', 'c++', '-std=c++14',
                               *[o for o in options if o.startswith('-D')], path],
                              capture_output=True, text=True)
    finally:
        Path(path).unlink()
    assert done.returncode == 0, done.stderr[-2000:]
    # String and character literals may legitimately contain any spelling.
    body = re.sub(r'"(\\.|[^"\\])*"', '""', done.stdout)
    return re.sub(r"'(\\.|[^'\\])'", "' '", body)


@pytest.mark.skipif(PREPROCESSOR is None, reason='no C preprocessor available')
@pytest.mark.parametrize('label,source,options', modules(), ids=lambda v: v if isinstance(v, str) else '')
def test_no_host_only_spelling_reaches_nvrtc(label, source, options):
    body = preprocess(source, options)
    undefined = {m: len(re.findall(r'\b' + m + r'\b', body)) for m in HOST_MACROS}
    undefined = {m: n for m, n in undefined.items() if n}
    assert not undefined, f'{label}: host-only macro(s) survive preprocessing: {undefined}'
    assert '#include <' not in body, f'{label}: a host header reaches the GPU branch'
    assert 'std::' not in body, f'{label}: a std:: name reaches the GPU branch'


@pytest.mark.skipif(PREPROCESSOR is None, reason='no C preprocessor available')
def test_check_detects_a_missing_guard():
    """Without the g5_base.h NAN guard the diagnostic module must fail this check.

    Guards the test itself: a check that cannot fail would have let the real
    defect through just as silently.
    """
    guard = ('#if defined(P5_GPU) && !defined(NAN)\n'
             '#define NAN (__longlong_as_double(0x7ff8000000000000ULL))\n'
             '#endif\n')
    label, source, options = next(m for m in modules() if m[0].startswith('local-deposition'))
    # g5_base.h is expanded once per including header, so the guard appears several
    # times; only the first takes effect. Remove every copy.
    assert source.count(guard) >= 1, 'g5_base.h NAN guard not found in the assembled source'
    assert re.search(r'\bNAN\b', preprocess(source.replace(guard, ''), options)), \
        'removing the guard did not make NAN undefined; the check has no power'
