"""PEGS-only calls for unrestricted and restricted collision stopping power.
SPIONE is evaluated directly with EE=E0 for unrestricted and EE=AE for restricted.
No unrestricted data deck is fed to transport.
"""
from pathlib import Path
import re,json
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[4]
def main():
 name='stopping_v3';d=BASE/name;d.mkdir(exist_ok=False)
 original=ROOT/'docs/egs5_crosscheck/revalidation_2026_10_07/egs5/pdd_phantom/pdd_phantom.f'
 s=original.read_text();a=s.index('      implicit none');b=s.index('      call pegs5',a)+len('      call pegs5')
 s=s[a:b]+'\n      stop\n      end\n      subroutine ausgab(iarg)\n      integer iarg\n      return\n      end\n      subroutine howfar\n      return\n      end\n'
 # No need for old labels and initialization; PEGS-only main.
 a=s.index('!     ------------------------------------------------------------\n!     Bin labels');b=s.index('!     ----------\n!     Open files',a);s=s[:a]+s[b:]
 (d/f'{name}.f').write_text('! PEGS-only, derived from pdd_phantom MAIN initialization.\n'+s)
 html=(BASE.parent/'data/water_composition.html').read_text()
 weights={int(z):float(w) for z,w in re.findall(r'<td>\s*(\d+)\s*</td>\s*<td>\s*([\d.]+)\s*</td>',html)}
 inp=f'''MIXT
 &INP NE=2,RHO=1.000,RHOZ={weights[1]},{weights[8]},EPSTFL=1,
      IRAYL=1,IBOUND=1,INCOH=1,ICPROF=0,IMPACT=0 &END
H2O                           H2O
H  O
ENER
 &INP AE=0.521,AP=0.001,UE=20.511,UP=20.0 &END
PWLF
 &INP &END
DECK
 &INP &END
'''
 # Evaluate PEGS DERCON formula using constants parsed from the installed source.
 constants=(ROOT/'docs/egs5_crosscheck/egs5/pegs/pegs5.f').read_text()
 def constant(key):
  match=re.search(r'^      '+key+r'=([0-9.eEdD+\-]+)\s*$',constants,re.M)
  if not match:raise RuntimeError('Constant not found: '+key)
  return float(match[1].replace('D','E').replace('d','e'))
 rm=constant('RME')*constant('C')**2/(1e6*constant('EMKS')*1e7)
 for kinetic in [.1,1.,10.]:
  total=kinetic+rm
  for cutoff in [total,.521]:
   inp+=f'CALL\n &INP XP(1)={total:.12g},XP(2)={cutoff:.12g} &END\nSPIONE\n'
 inp=inp.replace('DECK\n &INP &END\n','')+'DECK\n &INP &END\n'
 (d/f'{name}.inp').write_text(inp)
 (d/'epstar.dat').write_bytes((ROOT/'docs/egs5_crosscheck/egs5/data/density_corrections/compounds/water_liquid.density').read_bytes())
 (d/'conditions.json').write_text(json.dumps({'RM_MeV':rm,'mass_fractions':weights,'density_g_cm3':1.,'EPSTFL':1,'AE_MeV_total':.521,'unrestricted_cutoff':'EE=E0','restricted_cutoff':'EE=AE'},indent=2)+'\n')
if __name__=='__main__':main()
