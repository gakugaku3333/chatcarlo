"""Validation-only shared-session clients and whole-CLI timing (A3 §2e.2–3)."""
from contextlib import ExitStack
import json
import math
import re
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

LIMIT = 1.0571
HISTORIES = 4_600_000
POLL_SECONDS = 60


def ratio(old, new):
    if any(not math.isfinite(x) or x <= 0 for x in (old, new)):
        raise ValueError('positive finite timings required')
    value = new / old
    return dict(ratio=value, limit=LIMIT, status='pass' if value <= LIMIT else 'fail')


# Loaded in each fresh VM worker before CuPy. Logs actual compiler calls, including
# whether NVRTC compiled a nonempty source; no change to any engine source/options.
SITE_CUSTOMIZE = r'''
import atexit,json,os,pathlib,time
p=pathlib.Path(os.environ['A3_COMPILE_LOG']);rows=[]
def save():
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(dict(instrumented=True,rows=rows)))
try:
 import cupy as cp
 from cupy.cuda import compiler
 real=compiler.compile_using_nvrtc
 def compile_using_nvrtc(*args,**kw):
  begin=time.perf_counter();source=kw.get('source',args[0] if args else '')
  try:return real(*args,**kw)
  finally:
   rows.append(dict(kind='nvrtc',source_nonempty=bool(source.strip()),seconds=time.perf_counter()-begin));save()
 compiler.compile_using_nvrtc=compile_using_nvrtc
 raw=cp.RawKernel
 class TimedKernel(raw):
  def compile(self,*args,**kw):
   begin=time.perf_counter()
   try:return super().compile(*args,**kw)
   finally:
    rows.append(dict(kind='RawKernel.compile',seconds=time.perf_counter()-begin));save()
 cp.RawKernel=TimedKernel
 save();atexit.register(save)
except Exception as exc:
 p.write_text(json.dumps(dict(instrumented=False,error=str(exc))))
'''


def cache_used(first, second):
    """Conservative evidence: fixed cache, same binaries, measured second compile,
    and no nonempty-source NVRTC call in the fresh second worker. Missing -> unknown.
    """
    a, b = first.get('compiler', {}), second.get('compiler', {})
    cold, warm = first.get('cache_after', {}), second.get('cache_after', {})
    compiled = [r['seconds'] for r in b.get('rows', []) if r.get('kind') == 'RawKernel.compile']
    nvrtc = lambda d: [r for r in d.get('rows', []) if r.get('kind') == 'nvrtc' and r.get('source_nonempty')]
    verified = (a.get('instrumented') is True and b.get('instrumented') is True
                and bool(nvrtc(a)) and not nvrtc(b) and bool(compiled)
                and all(math.isfinite(x) and x >= 0 for x in compiled)
                and first.get('cache_dir') == second.get('cache_dir')
                and bool(cold) and cold == second.get('cache_before') == warm)
    return dict(status='pass' if verified else 'undetermined', second_compile_s=sum(compiled),
                reason='cache binaries + second compile + NVRTC trace' if verified else 'cache use not proven')


class SharedClient:
    """One already-owned T4. Child new/stop are leases, never VM lifecycle calls.
    The old client protocol returns MV_ENTRY_JSON, the new one G5_ENTRY.
    """
    def __init__(self, owner, remote, cache=None):
        self.owner, self.remote, self.cache = owner, remote, cache
        self.timings = []
        self.worker_roots = set()

    def call(self, *args, code=None):
        action = args[0]
        if action in ('new', 'stop'):
            return ''
        if action == 'exec':
            return self.remote_exec(code)
        paths = args[3:]  # action, -s, frozen session name
        return self.owner.call(action, *paths)

    def status_code(self, code):
        # Match only the frozen read-only status templates, before instrumentation.
        from prototypes.mv_electron.entry import colab_cli
        from chatcarlo.gpu import therapy_colab
        return code in (colab_cli.poll_code(), therapy_colab.listing(self.remote))

    def remote_exec(self, code):
        if self.status_code(code):
            return self.owner.raw(self.instrument(code),idempotent=True)
        return self.owner.raw(self.instrument(code))

    def instrument(self, code):
        self.worker_roots.update(re.findall(r"(?:root|r)=pathlib\.Path\(['\"]([^'\"]+)['\"]\)", code))
        if self.cache is None or 'subprocess.Popen(' not in code:
            return code
        # The frozen bootstrap's cwd is source. Set env on the launched worker;
        # modifying os.environ in colab exec alone would not survive another exec.
        prefix = f'''import os,subprocess
_a3_popen=subprocess.Popen
def _a3_launch(*args,**kwargs):
 env=dict(os.environ,**kwargs.pop('env',{{}}))
 env.update(CUPY_CACHE_DIR={self.cache!r},A3_COMPILE_LOG={self.remote+'/compile.json'!r},PYTHONPATH={self.remote+'/instrument'!r}+os.pathsep+env.get('PYTHONPATH',''))
 kwargs['env']=env
 return _a3_popen(*args,**kwargs)
'''
        return prefix + code.replace('subprocess.Popen(', '_a3_launch(')

    def execute(self, code):
        begin = time.perf_counter()
        try:
            text = self.remote_exec(code)
            for line in text.splitlines():
                for marker in ('MV_ENTRY_JSON ', 'G5_ENTRY '):
                    if line.startswith(marker):
                        return json.loads(line[len(marker):])
            raise RuntimeError('missing shared-session status')
        finally:
            self.timings.append(dict(operation='exec', seconds=time.perf_counter()-begin))

    def new(self): pass
    def stop(self): pass

    def upload(self, local, remote):
        return self.owner.call('upload', str(local), remote)

    def download(self, remote, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        return self.owner.call('download', remote, str(local))


def timed_main(main, argv, *, clock=time.perf_counter):
    begin = clock()
    rc = main(argv)
    elapsed = clock() - begin
    if rc != 0:
        raise RuntimeError('CLI main failed: '+str(rc))
    return elapsed


def cli_measure(version, scene, out, client, *, old_cli=None, old_colab=None,
                new_cli=None, new_colab=None):
    """Time the real main, with only client/poll injection; include analysis below."""
    from prototypes.mv_electron.entry import cli, colab_cli
    from chatcarlo.gpu import therapy_entry, therapy_colab
    old_cli, old_colab = old_cli or cli, old_colab or colab_cli
    new_cli, new_colab = new_cli or therapy_entry, new_colab or therapy_colab
    with ExitStack() as stack:
        if version == 'old':
            real = old_colab.run
            stack.enter_context(patch.object(old_colab, 'REMOTE', client.remote))
            stack.enter_context(patch.object(old_colab, 'POLL_SECONDS', POLL_SECONDS))
            stack.enter_context(patch.object(old_colab, 'run', lambda d, r: real(d, r, client=client)))
            return timed_main(old_cli.main, ['run', str(scene), '--out', str(out)])
        real = new_colab.run
        stack.enter_context(patch.object(new_colab, 'run',
            lambda *a, **kw: real(*a, **dict(kw, client=client, poll_seconds=POLL_SECONDS))))
        return timed_main(new_cli.main, ['run', str(scene), '--out', str(out)])
