"""P8a research-only PEGS execution, format-aware pair and shell extraction.

All external work is isolated in P8a cache. No existing PEGS input is edited.
Pair offsets follow PEGS DECK and EGS5 HATCH, not a water-specific line number.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from .materials_p8a import (ROOT, OUT, CACHE, MATERIALS, constants, energy_grid,
                            finite, sha, write, resume_appendix6, reference, load_reference,
                            verify_protected, implementation_hashes, load_manifest)
EGS = ROOT/'docs/egs5_crosscheck/egs5'
NUM = r'[-+]?\d*\.?\d+(?:[EDed][-+]?\d+)?'


def pegs_input(definitions, unrestricted=0):
    """Use exactly the NIST weight fractions and selected mass density."""
    import xraylib
    result = []
    for d in definitions:
        c = constants(d)
        zs = c['Z']
        name = d.get('medium_name', 'P8A_'+d['material'].upper())
        if not re.fullmatch(r'[A-Z0-9_]{1,24}', name):
            raise ValueError('invalid PEGS medium name')
        w = ','.join(format(d['weights'][str(z)], '.17g') for z in zs)
        a = ','.join(format(d['atomic_weights'][str(z)], '.17g') for z in zs)
        # EPSTFL=0 and an unmatched ID invoke Sternheimer general formula.
        # For the common comparison use the same density procedure at all rho.
        sternheimer_id = 'P8A_'+d['material'].upper()
        # PEGS5 rejects MIXT with NE=1 ("improperly defined for a mixture").
        head = (f'ELEM\n &INP RHO={d["density_g_cm3"]:.17g},EPSTFL=0 &END\n'
                if len(zs) == 1 else
                f'MIXT\n &INP NE={len(zs)},RHO={d["density_g_cm3"]:.17g},\n'
                f' RHOZ={w},\n WA={a},EPSTFL=0 &END\n')
        result.append(head +
                      f'{name:<30}{sternheimer_id}\n'
                      + ''.join(f'{xraylib.AtomicNumberToSymbol(z).upper():<3}' for z in zs)+'\n'
                      'ENER\n &INP AE=0.521,AP=0.01,UE=20.511,UP=21,\n'
                      f' IUNRST={unrestricted},IBOUND=0,INCOH=0,ICPROF=0,\n'
                      ' IRAYL=0 &END\nPWLF\n &INP EPE=0.01,NIPE=20 &END\n'
                      'DECK\n &INP &END\n')
    return ''.join(result)


def pair_parameters(text, medium, definition=None, expected_start=None):
    """Parse the 36 DL + 7 tail fields after the per-medium header.

    Return both pair/triplet tails; transport currently consumes the first.
    Record zero-based line offsets and source hash in the caller's receipt.
    """
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines)
            if re.match(r'\s*MEDIUM='+re.escape(medium)+r'\s*,', line)]
    if len(hits) != 1:
        raise ValueError('pair material identity / duplicate medium')
    start = hits[0]
    if expected_start is not None and expected_start != start:
        raise ValueError('pair reading position changed')
    header = lines[start+1]
    ne = int(re.search(r'NE=\s*(\d+)', header)[1])
    atom_rows = lines[start+2:start+2+ne]
    if len(atom_rows) != ne or any('ASYM=' not in line for line in atom_rows):
        raise ValueError('pair composition header shape')
    if definition is not None:
        if medium != 'P8A_'+definition['material'].upper() and not (definition['material'] == 'water' and medium == 'H2O'):
            raise ValueError('pair material does not match definition')
        rho = float(re.search(r'RHO=\s*('+NUM+')', header)[1])
        if not np.isclose(rho, definition['density_g_cm3'], rtol=6e-5, atol=0):
            raise ValueError('pair material density mismatch')
        weights = {int(float(re.search(r',Z=\s*('+NUM+')', line)[1])):
                   float(re.search(r'RHOZ=\s*('+NUM+')', line)[1]) for line in atom_rows}
        # ELEM writes RHOZ as the atomic weight; compare mass fractions.
        total = sum(weights.values())
        weights = {z: w/total for z, w in weights.items()}
        if set(weights) != set(map(int, definition['weights'])) or any(
                not np.isclose(w, definition['weights'][str(z)], rtol=6e-6, atol=0) for z, w in weights.items()):
            raise ValueError('pair material composition mismatch')
    bounds_line = start+2+ne
    bounds = finite([float(v) for v in lines[bounds_line].split()])
    flags = finite([float(v) for v in lines[bounds_line+1].split()])
    if bounds.shape != (5,) or flags.shape != (12,):
        raise ValueError('pair header shape / reading position')
    # Formatted DECK writes 36 entries in 8 records (5 per line, last 1).
    dl_start = bounds_line+2
    dl_rows = lines[dl_start:dl_start+8]
    if [len(line.split()) for line in dl_rows] != [5]*7+[1]:
        raise ValueError('pair DL shape / reading position')
    dl = finite([float(v.replace('D', 'E')) for line in dl_rows for v in line.split()]).reshape(6, 6)
    tail_rows = lines[dl_start+8:dl_start+10]
    if [len(line.split()) for line in tail_rows] != [5, 2]:
        raise ValueError('pair tail shape / reading position')
    tail = finite([float(v.replace('D', 'E')) for line in tail_rows for v in line.split()])
    if tail[0] <= 0 or np.any(tail[[3, 6]] <= 0) or np.any(tail[[2, 5]] < 0) or np.any(tail[[2, 5]] > 1):
        raise ValueError('invalid pair constants')
    return dict(dl=dl, delcm=tail[0], alphi=tail[[1, 4]], bpar=tail[[2, 5]],
                delpos=tail[[3, 6]], medium=np.array(medium),
                medium_line=np.array(start), dl_line=np.array(dl_start),
                AE=bounds[1], AP=bounds[2])


def diagnostic_source():
    """Direct unrestricted PEGS functions + density params + EGS atomic data.

    No shower is called. block_set links EGS5's block_data_atom shell constants.

    pegs5.f label 1220 (DECK) restores the initial LSPION coefficients/IEV,
    and label 1040 clears RHO. Each run_diagnostic supplies exactly ONE medium.
    Its MIX results (EDEN, RLC, ZC, Z/PZ and IDSTRN) survive DECK and EOF;
    prelastino only writes scattering metadata. After restoring RHO, rerun
    SPINIT with the same EPSTFL=0/ISSB=0 general-formula inputs as pegs_input.
    SPINIT rebuilds ALL of LSPION, including SPC1/SPC2/TOLN10, at full precision.
    This avoids rounded listing-file reconstruction and changes no EGS5 source.
    Do not call MIX here: label 1040 has also cleared WA at EOF.
    """
    return '''! P8a direct PEGS diagnostic; no transport or dose fitting.
      implicit none
      call block_set
      call pegs5
! pegs5 closes unit 22 and restores IAPRFL=0; reopen for direct brem calls.
      open(22,file='data/aprime.data',status='old')
! SPINIT and APRIM write to the listing (unit 26 was closed at EOF).
      open(26,file='pgs5job.pegs5lst',status='old',
     *     position='append')
      call p8adiag
      close(26)
      stop
      end
      subroutine p8adiag
      use, intrinsic :: ieee_arithmetic
      implicit none
      include 'include/egs5_h.f'
      include 'pegscommons/dercon.f'
      include 'pegscommons/thres2.f'
      include 'pegscommons/molvar.f'
      include 'pegscommons/lspion.f'
      include 'pegscommons/epstar.f'
      include 'pegscommons/mixdat.f'
      include 'include/egs5_edge.f'
      real*8 t,en,spione,spionp,brmstm,x,delta
      real*8 amoltm,bhabtm,g1e
      integer i,j,k,n,iz0
      open(43,file='energies.dat',status='old')
! pegs5 resets RHO to 0 at end of input; restore the medium density.
      read(43,*) n,rho
! DECK (pegs5.f:1220) restored LSPION defaults, not this medium.
! The single medium's MIX outputs survive EOF; SPINIT reconstructs
! its density effect AND stopping constants using the original routine.
      if(ne.le.0.or.rho.le.0.d0.or.eden.le.0.d0.or.
     *   rlc.le.0.d0) stop 31
      issb=0
      epstfl=0
      afact=0.d0
      sk=0.d0
      x0=0.d0
      x1=0.d0
      cbar=0.d0
      iev=0.d0
      call spinit
! A reset/invalid state must stop BEFORE any dependent direct function.
! DELTA0=0 is legitimate for the general formula; X0 may also be zero.
      if(.not.all(ieee_is_finite((/afact,sk,x0,x1,cbar,
     *   delta0,iev,spc1,spc2,toln10/)))) stop 32
      if(afact.le.0.d0.or.sk.le.0.d0.or.cbar.le.0.d0.or.
     *   iev.le.0.d0.or.x1.le.x0.or.delta0.lt.0.d0.or.
     *   spc1.le.0.d0.or.toln10.le.0.d0) stop 33
      open(42,file='direct.txt',status='replace')
      write(42,*) 'META',rm,ae,ap,rlc,rho,iev
      write(42,*) 'DENSITY',afact,sk,x0,x1,cbar,delta0
      write(42,*) 'SPION',spc1,spc2,toln10
      do i=1,n
        read(43,*) t
        en=t+rm
        x=.21715d0*log((en/rm)**2-1.d0)
        if(x.le.x0) then
          delta=delta0*10.d0**(2.d0*(x-x0))
        else if(x.lt.x1) then
          delta=toln10*x-cbar+afact*(x1-x)**sk
        else
          delta=toln10*x-cbar
        endif
        write(42,'(A,1X,11ES25.16)') 'STOP',t,
     *    spione(en,en)/(rlc*rho),brmstm(en,en)/(rlc*rho),
     *    spionp(en,en)/(rlc*rho),delta,spione(en,ae)/(rlc*rho),
     *    spionp(en,ae)/(rlc*rho),amoltm(en)/(rlc*rho),
     *    bhabtm(en)/(rlc*rho),
     *    g1e(-1,en,0)/(rlc*rho),g1e(1,en,0)/(rlc*rho)
      enddo
      do j=1,ne
        iz0=int(z(j))
        write(42,*) 'ATOM',iz0,(eedge(k,iz0),k=1,4),
     *    embind(iz0)
        do k=1,5
          write(42,*) 'PM',iz0,k,pm0(k,iz0),pm1(k,iz0),
     *      pm2(k,iz0),pm3(k,iz0)
        enddo
      enddo
      write(42,*) 'DONE'
      close(42)
      close(43)
      return
      end
      subroutine ausgab(iarg)
      integer iarg
      return
      end
      subroutine howfar
      return
      end
'''


def validate_diagnostic_state(text, definition):
    """Reject reset common blocks even in sealed, finite-looking old output."""
    def record(tag, size):
        rows = [line.split()[1:] for line in text.splitlines()
                if line.split()[:1] == [tag]]
        if len(rows) != 1 or len(rows[0]) != size:
            raise ValueError('PEGS diagnostic state missing/duplicate '+tag)
        return finite([float(v.replace('D', 'E')) for v in rows[0]], tag)
    meta = record('META', 6)
    a, sk, x0, x1, cbar, delta0 = record('DENSITY', 6)
    spc1, spc2, toln10 = record('SPION', 3)
    if (np.any(meta <= 0) or a <= 0 or sk <= 0 or cbar <= 0 or x1 <= x0
            or delta0 < 0 or spc1 <= 0 or not np.isclose(toln10, 2*np.log(10), rtol=1e-12)
            or not np.isclose(meta[4], definition['density_g_cm3'], rtol=1e-12)):
        raise ValueError('PEGS diagnostic state reset/invalid Sternheimer coefficients or IEV')


def run_diagnostic(definition, energies, work, unrestricted=4):
    es = energy_grid(energies)
    work = Path(work)
    source = diagnostic_source()
    inp = pegs_input([definition], unrestricted=unrestricted)
    identity = dict(definition=definition, energies=es.tolist(), source=source,
                    implementations=implementation_hashes(),
                    pegs_input=inp, egs_sources={str(p.relative_to(EGS)): sha(p)
                    for folder in ('pegs', 'pegscommons', 'egs', 'include')
                    for p in sorted((EGS/folder).glob('*.f'))})
    seal = work/'complete.json'
    if seal.exists():
        for name, expected in json.loads(seal.read_text()).items():
            if sha(work/name) != expected:
                raise ValueError('PEGS saved output hash mismatch')
    # Verify completed evidence before authorizing script-only migration. Legacy
    # seals omitted the binary: record its hash in the migration as well, so it
    # cannot change on a subsequent resume. Source/input/energy conditions above
    # still have to match exactly; no PEGS run is needed for this migration.
    if (not seal.exists() and (work/'identity.json').exists()
            and json.loads((work/'identity.json').read_text()) != identity):
        raise ValueError('S5 migration requires completed diagnostic evidence')
    evidence = dict(executable_sha256=sha(work/'egs5job.exe'), complete_sha256=sha(seal)) if seal.exists() else {}
    resume_appendix6(work/'identity.json', identity, 'S5', evidence)
    if not seal.exists():
        (work/'p8adiag.f').write_text(source)
        (work/'p8adiag.inp').write_text(inp)
        (work/'energies.dat').write_text(str(len(es))+' '+format(float(definition['density_g_cm3']), '.17g')+'\n'+'\n'.join(format(e, '.17g') for e in es)+'\n')
        p = subprocess.run([str(EGS/'egs5run'), 'comp'], cwd=work,
                           input='p8adiag\n\n\n', capture_output=True, text=True)
        (work/'build.log').write_text(p.stdout+p.stderr)
        p.check_returncode()
        if not (work/'egs5job.exe').exists():
            raise RuntimeError('PEGS diagnostic build failed')
        p = subprocess.run(['./egs5job.exe'], cwd=work, capture_output=True, text=True)
        (work/'run.log').write_text(p.stdout+p.stderr)
        p.check_returncode()
        if not (work/'direct.txt').exists() or 'DONE' not in (work/'direct.txt').read_text():
            raise ValueError('PEGS direct diagnostic incomplete')
    text = (work/'direct.txt').read_text()
    validate_diagnostic_state(text, definition)
    rows = finite([[float(v.replace('D', 'E')) for v in line.split()[1:]]
                   for line in text.splitlines() if line.split() and line.split()[0] == 'STOP'])
    if rows.shape != (len(es), 11) or not np.allclose(rows[:, 0], es, rtol=1e-14, atol=0):
        raise ValueError('PEGS diagnostic energy shape')
    if not seal.exists():
        write(seal, {name: sha(work/name) for name in
                     ('direct.txt', 'pgs5job.pegs5dat', 'pgs5job.pegs5lst', 'p8adiag.f', 'p8adiag.inp', 'egs5job.exe')})
    return dict(T=es, collision=rows[:, 1], radiative=rows[:, 2],
                positron_collision=rows[:, 3], delta=rows[:, 4],
                restricted_collision=rows[:, 5], restricted_positron=rows[:, 6],
                moller=rows[:, 7], bhabha=rows[:, 8], G1_electron=rows[:, 9],
                G1_positron=rows[:, 10], text=text,
                raw=work/'direct.txt', pegs=work/'pgs5job.pegs5dat')


def density_stopping(definition, energies, work):
    return run_diagnostic(definition, energies, work)


def extract_shells(text, definition):
    c = constants(definition)
    z = np.array(c['Z'])
    binding, pm = [], []
    for element in z:
        atoms = [line.split()[2:] for line in text.splitlines()
                 if line.split()[:2] == ['ATOM', str(element)]]
        if len(atoms) != 1 or len(atoms[0]) != 5:
            raise ValueError('missing atomic shell binding data')
        binding.append([float(v) * 1e-3 for v in atoms[0]])
        rows = [line.split() for line in text.splitlines()
                if line.split()[:2] == ['PM', str(element)]]
        if len(rows) != 5 or [int(row[2]) for row in rows] != list(range(1, 6)):
            raise ValueError('shell polynomial shape')
        pm.append([[float(v) for v in row[3:]] for row in rows])
    return dict(Z=z, NJ=np.array(c['NJ']), binding_MeV=finite(binding), pm=finite(pm),
                shell_labels=np.array(['K', 'L1', 'L2', 'L3', 'M_effective']))


def shell_probabilities(E, shells):
    """Joint element/shell probabilities reproducing egs5_photo.f.

    EGS5's effective M binding energy is distinct from an individual M shell.
    Below all bindings the photon deposits locally (no selected shell).
    """
    E = float(E)
    if not np.isfinite(E) or E <= 0:
        raise ValueError('invalid photoelectric energy')
    z, nj = shells['Z'], finite(shells['NJ'])
    b, pm = finite(shells['binding_MeV']), finite(shells['pm'])
    if b.shape != (len(z), 5) or pm.shape != (len(z), 5, 4) or nj.shape != (len(z),):
        raise ValueError('shell data shape')
    if np.any(b < 0) or np.any(nj <= 0):
        raise ValueError('invalid shell binding / number density')
    l = np.log(E*1000)
    xs = np.exp(pm[..., 0]+pm[..., 1]*l+pm[..., 2]*l*l+pm[..., 3]*l*l*l)
    xs[(E <= b) | (pm[..., 0] == 0)] = 0
    # K's polynomial is not guarded by pm0==0 in EGS5.
    xs[:, 0] = np.where(E > b[:, 0], np.exp(pm[:, 0, 0]+pm[:, 0, 1]*l+pm[:, 0, 2]*l*l+pm[:, 0, 3]*l*l*l), 0)
    xs[E <= b[:, 4]] = 0
    xs *= nj[:, None]  # pz is proportional to atoms per gram.
    finite(xs, 'shell cross sections')
    # Below L3 EGS5 selects effective M regardless of polynomial shell ratio.
    for i in range(len(z)):
        if E <= b[i, 3]:
            total = xs[i].sum()
            xs[i] = 0
            xs[i, 4] = total
    total = xs.sum()
    if total == 0:
        return np.zeros_like(xs), 1.
    return xs/total, 0.


def generate(out=OUT, cache=CACHE):
    verify_protected()
    m = load_manifest(out/'manifest.json', complete=False)
    for material in MATERIALS:
        row = m['materials'][material]
        d = load_reference(row['definition'], material)
        stopping = load_reference(row['stopping'], material)
        data = run_diagnostic(d, stopping['T'], cache/'pegs'/material)
        params = pair_parameters(data['pegs'].read_text(), 'P8A_'+material.upper(), d)
        params.update(source_sha256=np.array(sha(data['pegs'])), material=np.array(material))
        p = cache/material/'pair.npz'
        np.savez(p, **params)
        row['pair'] = reference(p, str(data['pegs']), 'PEGS DECK/HATCH 36+7 fields; research only', 'dimensionless; MeV', material,
                                source_sha256=sha(data['pegs']), medium_line=int(params['medium_line']), dl_line=int(params['dl_line']))
        shells = extract_shells(data['text'], d)
        np.savez(cache/material/'shells.npz', **shells)
        row['shells'] = reference(cache/material/'shells.npz', str(data['raw']), 'EGS5 block_data_atom; research only', 'MeV; atoms/g; log(keV) polynomial', material, source_sha256=sha(data['raw']))
        np.savez(cache/material/'pegs_stopping.npz', **{k: data[k] for k in ('T', 'collision', 'radiative', 'positron_collision', 'delta',
             'restricted_collision', 'restricted_positron', 'moller', 'bhabha',
             'G1_electron', 'G1_positron')})
        # Report unrestricted direct functions, never fitted transport EDEDX.
        report = dict(IUNRST=4, AE=.521, AP=.01, units='MeV cm2/g', raw=str(data['raw']), raw_sha256=sha(data['raw']),
                      T_MeV=data['T'].tolist(),
                      unrestricted=True, density_g_cm3=d['density_g_cm3'],
                      I_NIST_MeV=d['I_MeV'],
                      PEGS_META=[float(v) for line in data['text'].splitlines() if line.split()[:1] == ['META'] for v in line.split()[1:]],
                      collision_relative=(data['collision']/stopping['collision']-1).tolist(),
                      radiative_relative=(data['radiative']/stopping['radiative']-1).tolist(),
                      positron_over_electron=(data['positron_collision']/data['collision']).tolist(),
                      status='report-only', exceeding_1pct_requires_explanation=True)
        write(out/(material+'_pegs_stopping.json'), report)
        write(out/'manifest.json', m)
    verify_protected()


def main():
    argparse.ArgumentParser(__doc__).parse_args()
    generate()


if __name__ == '__main__':
    main()
