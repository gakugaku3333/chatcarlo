"""Fresh-scratch, foreground EGS5 revalidation; never overwrites old runs."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
OLD = ROOT / 'docs/egs5_crosscheck'
EGS = OLD / 'egs5'
PREREG = OUT.parent / 'PREREGISTRATION.md'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def run(name, original, n, seed, prereg_sha):
    dest = OUT / name
    dest.mkdir(exist_ok=False)
    source = OLD / original / (original + '.f')
    inp = OLD / original / (original + '.inp')
    code = source.read_text()
    code = code.replace('RHO=1.001', 'RHO=1.000')
    code = re.sub(r'(?m)^(      ncase=)\d+', rf'\g<1>{n}', code)
    code = re.sub(r'(?m)^(      inseed=)\d+', rf'\g<1>{seed}', code)
    code = code.replace('      pcut(2)=0.010', '      pcut(2)=0.001')
    code = code.replace('      pcut(3)=0.010', '      pcut(3)=0.001')
    code = code.replace('      zin=0.0', '      zin=-0.0001d0')
    code = code.replace('      irin=2', '      irin=1')
    code = code.replace('          ustep=0.0\n          irnew=2',
                        '          if (-z(np)/w(np).gt.ustep) return\n'
                        '          ustep=max(0.d0,-z(np)/w(np))\n          irnew=2')
    code = code.replace('      call hatch\n',
                        '      call hatch\n'
                        '      write(6,*) "REVALIDATION PHYSICS: WATER REGIONS"\n'
                        '      do j=1,nreg\n'
                        '        if (med(j).eq.1) then\n'
                        '          write(6,*) "REGION MED ECUT PCUT IRAYLR INCOHR",\n'
                        '     *      j,med(j),ecut(j),pcut(j),iraylr(j),incohr(j)\n'
                        '        end if\n'
                        '      end do\n'
                        '      write(6,*) "MEDIUM RHO AE AP UE UP",\n'
                        '     *  rhom(1),ae(1),ap(1),ue(1),up(1)\n')
    code = ('! Revalidation 2026-10-07: RHO=1.000, AP/PCUT=1 keV.\n'
            '! Source z=-0.0001 cm, vacuum entry tracked to z=0.\n' + code)
    (dest / (name + '.f')).write_text(code)
    (dest / (name + '.inp')).write_text(
        inp.read_text().replace('RHO=1.001', 'RHO=1.000').replace('AP=0.0100', 'AP=0.0010'))
    metadata = {'started_utc': now(), 'preregistration_sha256_before': prereg_sha,
                'history_count': n, 'inseed': seed, 'original_code': str(source.relative_to(ROOT)),
                'original_code_sha256': sha(source), 'original_input_sha256': sha(inp),
                'code_sha256': sha(dest / (name + '.f')), 'input_sha256': sha(dest / (name + '.inp')),
                'runner_sha256': sha(Path(__file__)),
                'compiler_version': subprocess.check_output(['gfortran', '--version'], text=True).splitlines()[0],
                'egs5run_sha256': sha(EGS / 'egs5run'),
                'egs5_source_tree_sha256': hashlib.sha256(b''.join(
                    str(p.relative_to(EGS)).encode() + p.read_bytes() for folder in ('egs','pegs','auxcode','include','pegscommons','auxcommons','data')
                    for p in sorted((EGS / folder).rglob('*')) if p.is_file())).hexdigest(),
                'physics': {'rho':1.000,'AP_MeV':.001,'PCUT_MeV':.001,'ECUT_MeV':1.5,
                            'IBOUND':1,'INCOH':1,'IRAYL':1,'ICPROF':0,'IMPACT':0,'IEDGFL':0},
                'commands': [f"printf '{name}\\n\\n\\n' | {EGS / 'egs5run'} comp", './egs5job.exe']}
    (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='egs5_revalidation_') as scratch_s:
        scratch = Path(scratch_s)
        shutil.copy2(dest / (name + '.f'), scratch)
        shutil.copy2(dest / (name + '.inp'), scratch)
        start = time.monotonic()
        with (dest / 'build.log').open('w') as log:
            proc = subprocess.run([str(EGS / 'egs5run'), 'comp'], cwd=scratch,
                                  input=name+'\n\n\n', text=True, stdout=log, stderr=subprocess.STDOUT)
        metadata['build_exit_code'] = proc.returncode
        if proc.returncode != 0 or not (scratch / 'egs5job.exe').is_file():
            raise RuntimeError(f'Build failed: {name}')
        with (dest / 'run.log').open('w') as log:
            proc = subprocess.run([str(scratch / 'egs5job.exe')], cwd=scratch, stdout=log, stderr=subprocess.STDOUT)
        metadata['run_exit_code'] = proc.returncode
        metadata['elapsed_seconds_build_and_run'] = time.monotonic() - start
        for filename in ('egs5job.out','pgs5job.pegs5lst','egs5job.dummy'):
            shutil.copy2(scratch / filename, dest)
        metadata['finished_utc'] = now()
        metadata['preregistration_sha256_after'] = sha(PREREG)
        metadata['output_sha256'] = sha(dest / 'egs5job.out')
        (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if proc.returncode != 0 or sha(PREREG) != prereg_sha:
            raise RuntimeError(f'Execution or preregistration integrity failed: {name}')
    print(name, 'complete', metadata['elapsed_seconds_build_and_run'], flush=True)

def main():
    prereg_sha = sha(PREREG)
    run('pdd_repeat_a', 'pdd60_phantom_incoh1', 10000, 1, prereg_sha)
    run('pdd_repeat_b', 'pdd60_phantom_incoh1', 10000, 1, prereg_sha)
    a = (OUT / 'pdd_repeat_a/egs5job.out').read_bytes()
    b = (OUT / 'pdd_repeat_b/egs5job.out').read_bytes()
    # PEGS5 CPU timings are nondeterministic, whereas all scoring output must match.
    def scoring(data):
        text = data.decode()
        return text[text.index(' Central-axis'): ] if ' Central-axis' in text else text[text.index(' ncase='):]
    aa, bb = scoring(a), scoring(b)
    if aa != bb:
        raise RuntimeError('Same-seed scoring not identical')
    (OUT / 'reproducibility.json').write_text(json.dumps(
        {'histories':10000,'seed':1,'scoring_text_identical':True,
         'scoring_sha256':hashlib.sha256(aa.encode()).hexdigest(),
         'preregistration_sha256':prereg_sha}, indent=2)+'\n')
    run('bsf_thinslab', 'bsf60_thinslab_incoh1', 8000000, 6, prereg_sha)
    run('bsf_phantom', 'bsf60_phantom_8M_incoh1', 8000000, 7, prereg_sha)
    run('pdd_phantom', 'pdd60_phantom_incoh1', 100000000, 1, prereg_sha)

if __name__ == '__main__':
    main()
