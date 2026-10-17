"""Opt-in pytest retention for the user's no-deletion execution requirement.

Load with PYTEST_ADDOPTS='-p prototypes.mv_electron.gpu.p5k_pytest_keep'.
Pytest's automatic temporary-directory and dead-symlink cleanup are disabled.
Any other attempted filesystem deletion raises before the system call.
"""
import sys
import _pytest.pathlib
import _pytest.tmpdir


def _keep(*args,**kwargs):
    return None


_pytest.pathlib.rm_rf=_keep
_pytest.pathlib.cleanup_dead_symlinks=_keep
_pytest.tmpdir.rm_rf=_keep
_pytest.tmpdir.cleanup_dead_symlinks=_keep


def _audit(event,args):
    if event in ('os.remove','os.rmdir'):
        raise RuntimeError('P5k no-deletion execution policy: '+event+' '+str(args))


sys.addaudithook(_audit)


def pytest_configure(config):
    config.inicfg['tmp_path_retention_policy']='all'
    config.inicfg['tmp_path_retention_count']='1000000'

# tempfile's default-directory probe and anonymous capture files normally delete
# their probe/backing names. Use retained named files and a known writable root.
import tempfile
import os
import time
_retained_root='/private/tmp/p5k-retained-'+str(os.getpid())+'-'+str(time.time_ns())
os.makedirs(_retained_root)
tempfile.tempdir=_retained_root
_named_temporary_file=tempfile.NamedTemporaryFile


def _retained_file(*args,**kwargs):
    kwargs['delete']=False
    return _named_temporary_file(*args,**kwargs)


def _retained_directory_cleanup(self):
    self._finalizer.detach()


tempfile.TemporaryFile=_retained_file
tempfile.TemporaryDirectory.cleanup=_retained_directory_cleanup
_pytest.pathlib.register_cleanup_lock_removal=_keep
_pytest.pathlib._force_symlink=_keep
import _pytest.capture
_pytest.capture.TemporaryFile=_retained_file
# Keep caches in this retained run directory. The unique matplotlib directory is
# private to this test process, so its font-cache lock needs no filesystem lock.
os.environ['MPLCONFIGDIR']=_retained_root+'/matplotlib'
os.environ['XDG_CACHE_HOME']=_retained_root+'/cache'
import contextlib
import matplotlib.cbook
matplotlib.cbook._lock_path=lambda *args,**kwargs:contextlib.nullcontext()
tempfile.NamedTemporaryFile=_retained_file
_directory_init=tempfile.TemporaryDirectory.__init__


def _retained_directory_init(self,*args,**kwargs):
    _directory_init(self,*args,**kwargs)
    self._finalizer.detach()


tempfile.TemporaryDirectory.__init__=_retained_directory_init
