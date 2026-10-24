"""Incremental, session-first transfer with validation before publication."""
from pathlib import Path
import shutil
import tempfile
from .. import p8g3_core as core


def install_file(source, target, digest):
    if core.kernel.sha(source)!=digest:raise ValueError('download hash '+str(target))
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if core.kernel.sha(target)!=digest:raise ValueError('immutable download conflict '+str(target))
        return
    pending=target.with_name(target.name+'.pending')
    shutil.copyfile(source,pending);pending.replace(target)


def collect(status, output, identity, download):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # A staging area on the same filesystem; interrupted downloads never appear
    # under a public call name. Sessions always precede call data.
    with tempfile.TemporaryDirectory(prefix='.p8g3-download-',dir=output) as tmp:
        stage=Path(tmp)
        for group in status['groups']:
            name=group['name'];schedule=group['schedule']
            if name not in ('results','trial'):raise ValueError('remote group')
            dest=output/name;dest.mkdir(exist_ok=True)
            staging=stage/name;staging.mkdir()
            for sid,digest in group['sessions'].items():
                target=core.session_path(dest,sid)
                if target.exists():
                    core.require_session(dest,sid,identity,schedule,digest)
                else:
                    candidate=core.session_path(staging,sid)
                    download(name+'/'+candidate.name,candidate)
                    core.require_session(staging,sid,identity,schedule,digest)
                    install_file(candidate,target,digest)
                shutil.copyfile(target,core.session_path(staging,sid))
            local=core.available(dest,identity,schedule)
            for key,seal in group['calls'].items():
                j=int(key);core.invocation(j,schedule)
                if j in local:
                    if (local[j]['sha256']!=seal['sha256'] or
                        core.kernel.sha(core.call_path(dest,j).with_suffix('.json'))!=seal['receipt_sha256']):
                        raise ValueError('collected call conflict '+key)
                    continue
                path=core.call_path(staging,j)
                download(name+'/'+path.with_suffix('.json').name,path.with_suffix('.json'))
                download(name+'/'+path.name,path)
                _,receipt=core.load_call(staging,j,identity,schedule)
                if receipt['sha256']!=seal['sha256']:raise ValueError('remote call seal')
                if core.kernel.sha(path.with_suffix('.json'))!=seal['receipt_sha256']:raise ValueError('remote receipt seal')
                # Orphan public data from an interrupted publication is replaceable.
                target=core.call_path(dest,j)
                target.with_suffix('.json').unlink(missing_ok=True)
                shutil.copyfile(path,target.with_name(target.name+'.pending'))
                target.with_name(target.name+'.pending').replace(target)
                install_file(path.with_suffix('.json'),target.with_suffix('.json'),seal['receipt_sha256'])
            for filename,digest in group['records'].items():
                if filename not in ('complete.json','trial_qualified.json'):raise ValueError('record name')
                path=staging/filename
                download(name+'/'+filename,path)
                if core.kernel.sha(path)!=digest:raise ValueError('record hash')
                if filename=='complete.json':
                    if core.read_json(path)!=core.completion(dest,identity,schedule):raise ValueError('download completion')
                else:
                    from ..p8g3_colab import require_trial
                    # Qualification is checked in staging with all collected pairs.
                    for source in dest.iterdir():
                        if source.is_file() and source.name!=filename:shutil.copyfile(source,staging/source.name)
                    require_trial(staging,identity)
                install_file(path,dest/filename,digest)
