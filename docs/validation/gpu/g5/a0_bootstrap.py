"""A0 VM bootstrap: verified transfer, frozen loaders, launch lock and PID."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from a0_speed import sha, write_json, fingerprints


def verify_files(root, manifest):
    """All existing manifest members, with containment and optional size checks."""
    root = Path(root).resolve()
    for name, item in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('invalid/missing bundle member: ' + name)
        digest = item['sha256'] if isinstance(item, dict) else item
        if sha(path) != digest or isinstance(item, dict) and path.stat().st_size != item['bytes']:
            raise ValueError('bundle size/hash mismatch: ' + name)


def extract_sources(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            rel = Path(info.filename)
            target = (destination / rel).resolve()
            if rel.is_absolute() or '..' in rel.parts or not target.is_relative_to(destination):
                raise ValueError('unsafe source zip member')
            if info.is_dir():
                continue
            content = z.read(info)
            if target.exists() and target.read_bytes() != content:
                # Export packages carry equivalent empty package initializers.
                if target.name != '__init__.py' or target.read_bytes().strip() or content.strip():
                    raise ValueError('source bundles disagree: ' + str(rel))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


def prepare(base):
    base = Path(base).resolve()
    bundles = base / 'bundles'
    for name, filename in [('g2b', 'manifest.json'), ('g4', 'manifest.json'),
                           ('p7', 'p7_manifest.json'), ('p8g2', 'p8g2_manifest.json')]:
        verify_files(bundles / name, json.loads((bundles / name / filename).read_text()))
    for name, archive in [('p7', 'p7_sources.zip'), ('p8g2', 'p8g2_source.zip')]:
        extract_sources(bundles / name / archive, base)
    gpu = base / 'chatcarlo/gpu'
    gpu.mkdir(parents=True, exist_ok=True)
    (base / 'chatcarlo/__init__.py').write_text('')
    from_names = json.loads((bundles / 'g2b/manifest.json').read_text())['files']
    for name in from_names:
        if Path(name).suffix in ('.py', '.h', '.cpp', '.cu'):
            source = bundles / 'g2b' / name
            other = bundles / 'g4' / name
            if other.exists() and sha(source) != sha(other):
                raise ValueError('diagnostic runtime bundles disagree: ' + name)
            (gpu / name).write_bytes(source.read_bytes())
    sys.path.insert(0, str(base))
    # All four existing loaders verify their native manifests and identities.
    return fingerprints(bundles)


def alive(pid):
    try:
        os.kill(pid, 0)
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, ValueError, IndexError):
        return False


def launch(base, expected_job_sha):
    base = Path(base).resolve()
    lock = (base / 'launch.lock').open('a')
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if sha(base / 'job.json') != expected_job_sha:
            raise ValueError('job SHA256 mismatch')
        job = json.loads((base / 'job.json').read_text())
        # Verify source/config transfer before invoking uploaded engine loaders.
        for name, digest in job['control_files'].items():
            if Path(name).name != name or sha(base / name) != digest:
                raise ValueError('control SHA256 mismatch: ' + name)
        pidpath = base / 'worker.pid'
        if pidpath.exists() and alive(int(pidpath.read_text())):
            if (base / 'worker.job.sha256').read_text() != expected_job_sha:
                raise ValueError('active worker job mismatch')
            return int(pidpath.read_text())
        if (base / 'results/measurements.json').exists():
            raise ValueError('ended/interrupted session; use a fresh base, repeat all six')
        identity = prepare(base)
        write_json(base / 'bundle_fingerprints.json', identity)
        env = dict(os.environ, PYTHONPATH=str(base))
        with (base / 'worker.log').open('ab') as log:
            process = subprocess.Popen(['nohup', sys.executable, str(base / 'a0_speed.py'),
                                        '--configs', str(base / 'a0_configs.json'), '--bundles', str(base / 'bundles'),
                                        '--output', str(base / 'results')], cwd=base, env=env, stdout=log,
                                       stderr=log, start_new_session=True)
        pidpath.write_text(str(process.pid))
        (base / 'worker.job.sha256').write_text(expected_job_sha)
        return process.pid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--job-sha', required=True)
    args = p.parse_args()
    print('G5A0_LAUNCHED', launch(args.base, args.job_sha))


if __name__ == '__main__':
    main()
