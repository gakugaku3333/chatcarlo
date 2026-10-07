"""Compare actual PEGS SPIONE calls with fetched ESTAR, same composition/density effect."""
from pathlib import Path
import re,json,math
BASE=Path(__file__).resolve().parent
def main():
 refpath=BASE/'data/estar_water.html';lst=BASE/'egs5/stopping_v3/pgs5job.pegs5lst'
 if not refpath.exists() or not lst.exists():
  result={'J4':'unavailable','reason':'ESTAR response or PEGS listing missing'}
 else:
  html=refpath.read_text(); rows=re.findall(r'([\d.]+E[+\-]\d+)\s+([\d.]+E[+\-]\d+)\s+([\d.]+E[+\-]\d+)\s+([\d.]+E[+\-]\d+)\s+([\d.]+E[+\-]\d+)',html)
  refs={float(r[0]):[float(v) for v in r[1:]] for r in rows}
  text=lst.read_text();rho=float(re.search(r'density=\s*([\d.E+\-]+)',text).group(1))
  rlc=float(re.search(r'ZE,ZX,RLC\s*\n\s*\S+\s+\S+\s+(\S+)',text).group(1))
  calls=re.findall(r'Function call:\s*(\S+)\s*= SPIONE OF\s*(\S+)\s+(\S+)',text)
  assert len(calls)==6 and set(refs)=={.1,1,10}
  conditions=json.loads((BASE/'egs5/stopping_v3/conditions.json').read_text())
  epstar=(BASE/'egs5/stopping_v3/epstar.dat').read_text().splitlines()
  density_pairs=re.findall(r'([\d.]+e[+\-]\d+),([\d.]+e[+\-]\d+)', '\n'.join(epstar[3:]),re.I)
  delta={float(e):float(d) for e,d in density_pairs}
  comp_html=(BASE/'data/water_composition.html').read_text()
  weights={int(z):float(w) for z,w in re.findall(r'<td>\s*(\d+)\s*</td>\s*<td>\s*([\d.]+)\s*</td>',comp_html)}
  assert {int(k):v for k,v in conditions['mass_fractions'].items()}==weights
  matches=all(math.isclose(delta[e],refs[e][3],abs_tol=1e-8) for e in refs)
  values=[]
  for i,e in enumerate([.1,1.,10.]):
   unrestricted=float(calls[2*i][0])/(rlc*rho);restricted=float(calls[2*i+1][0])/(rlc*rho)
   values.append({'kinetic_MeV':e,'ESTAR_unrestricted_MeV_cm2_g':refs[e][0],'PEGS_unrestricted_MeV_cm2_g':unrestricted,'relative_difference':unrestricted/refs[e][0]-1,'PEGS_restricted_MeV_cm2_g':restricted,'ESTAR_delta':refs[e][3],'PEGS_tabulated_delta':delta[e]})
  result={'J4':'pass' if matches and max(abs(v['relative_difference']) for v in values)<=.02 else 'fail','density_effect_matches_at_comparison_energies':matches,'density_g_cm3':rho,'radiation_length_cm':rlc,'conditions':conditions,'rows':values,'note':'CALL output MeV/radiation length converted by dividing RLC(cm)*rho(g/cm3); CALL listing rounding retained'}
 (BASE/'estar_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
