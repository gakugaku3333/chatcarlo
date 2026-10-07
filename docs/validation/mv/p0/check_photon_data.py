"""Run unmodified official XCOM in scratch and compare its output; never extrapolate."""
from pathlib import Path
import subprocess,tempfile,shutil,re,json,csv
import numpy as np
import xraylib
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
COLS=['rayleigh','compton','photoelectric','pair_nuclear','pair_electron','total','total_without_coherent']
def composition(name):
 s=(BASE/f'data/{name}_composition.html').read_text()
 return {int(z):float(w) for z,w in re.findall(r'<td>\s*(\d+)\s*</td>\s*<td>\s*([\d.]+)\s*</td>',s)}
def main():
 src=BASE/'data/XCOM'
 if not (src/'XCOM.f').exists():
  result={'J1a':'unavailable','J1b':'unavailable','reason':'Official archive unavailable'}
  (BASE/'photon_check.json').write_text(json.dumps(result,indent=2)+'\n');print(result);return
 nist=np.loadtxt(ROOT/'chatcarlo/data/nist_xaamdi/water.csv',delimiter=',',comments='#')
 hi=nist[(nist[:,0]>=1000)&(nist[:,0]<=20000)]
 assert len(hi)==12
 energies=sorted(set([.001,.01,.02]+[i/10 for i in range(1,9)]+list(hi[:,0]/1000)+[1.0219,1.0221,2.0439,2.0441]))
 water=composition('water');air=composition('air');assert set(water)=={1,8} and set(air)=={6,7,8,18}
 tables={}
 with tempfile.TemporaryDirectory(prefix='p0_xcom_') as tmp:
  d=Path(tmp);shutil.copytree(src,d,dirs_exist_ok=True)
  p=subprocess.run(['gfortran','-std=legacy','-O2','XCOM.f','-o','xcom'],cwd=d,capture_output=True,text=True)
  (BASE/'data/xcom_build.log').write_text(p.stdout+p.stderr);p.check_returncode()
  (d/'energies.txt').write_text(str(len(energies))+'\n'+'\n'.join(map(str,energies))+'\n')
  materials={f'Z{z}':['1',str(z),'3'] for z in sorted(set(water)|set(air))}
  materials['water']=['3','H2O']
  symbols={1:'H',6:'C',7:'N',8:'O',18:'Ar'}
  materials['air']=['4',str(len(air))]+[v for z,w in air.items() for v in (symbols[z],str(w))]+['1']
  for name,spec in materials.items():
   inp='\n'.join([name]+spec+['3','2','energies.txt','table.txt','1'])+'\n'
   p=subprocess.run([str(d/'xcom')],input=inp,cwd=d,capture_output=True,text=True);p.check_returncode()
   (BASE/f'data/xcom_{name}.input').write_text(inp)
   raw=(d/'table.txt').read_text();(BASE/f'data/xcom_{name}.txt').write_text(raw)
   rows=[]
   for line in raw.splitlines():
    parts=line.split()
    if len(parts)==8:
     try: row=list(map(float,parts))
     except ValueError:continue
     rows.append(row)
   assert len(rows)==len(energies),(name,len(rows))
   tables[name]=np.array(rows)
 def row(name,e):
  a=tables[name];j=energies.index(e);assert abs(a[j,0]-e)<5e-4;return a[j,1:]
 comparisons=[]
 for e,mu,_ in hi:
  got=row('water',e/1000)[5]; comparisons.append({'reference':'bundled_NIST','keV':e,'got':got,'reference_value':mu,'relative_difference':got/mu-1})
 low=[]
 for e in range(100,801,100):
  got=row('water',e/1000)
  for k,func in [('total',xraylib.CS_Total_CP),('photoelectric',xraylib.CS_Photo_CP),('compton',xraylib.CS_Compt_CP),('rayleigh',xraylib.CS_Rayl_CP)]:
   ref=func('H2O',e);r={'reference':'xraylib','process':k,'keV':e,'got':got[COLS.index(k)],'reference_value':ref,'relative_difference':got[COLS.index(k)]/ref-1}
   low.append(r)
   if k=='total': comparisons.append(r)
 synthesis=[]
 # Water direct formula uses XCOM atomic weights. Use printed direct weights,
 # rather than NIST-rounded weight fractions, for the same-composition check.
 # XCOM FORM weights from downloaded ATWTS.DAT, parsed numeric DATA payload.
 attext=(src/'ATWTS.DAT').read_text(); atoms=[float(v) for v in re.findall(r'\d+\.\d+',attext)]
 waterweights={1:2*atoms[0]/(2*atoms[0]+atoms[7]),8:atoms[7]/(2*atoms[0]+atoms[7])}
 for name,weights in [('water',waterweights),('air',{z:w/sum(air.values()) for z,w in air.items()})]:
  for e in energies:
   synth=sum(w*row(f'Z{z}',e) for z,w in weights.items()); direct=row(name,e)
   diff=np.divide(synth-direct,direct,out=np.zeros_like(direct),where=direct!=0)
   synthesis.append({'material':name,'MeV':e,'max_relative_difference':float(max(abs(diff)))})
 pair={}
 for label,e,k in [('nuclear_below',1.0219,3),('nuclear_above',1.0221,3),('electron_below',2.0439,4),('electron_above',2.0441,4)]:pair[label]=float(row('water',e)[k])
 result={'J1a':'pass' if max(abs(r['relative_difference']) for r in comparisons)<=.01 else 'fail','J1b':'see numerical differences; no preregistered partial tolerance','total_comparisons':comparisons,'partial_comparisons':low,'pair_thresholds':pair,'synthesis':synthesis,'water_weights':waterweights,'air_weights':air,'xraylib_version':xraylib.__version__}
 (BASE/'photon_check.json').write_text(json.dumps(result,indent=2)+'\n')
 with (BASE/'photon_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['reference','process','keV','got','reference_value','relative_difference']);w.writeheader();w.writerows(comparisons+[r for r in low if r['process']!='total'])
 print(json.dumps({'J1a':result['J1a'],'total_max_difference':max(abs(r['relative_difference']) for r in comparisons),'partial_max_difference':{k:max(abs(r['relative_difference']) for r in low if r['process']==k) for k in COLS[:3]},'pair_thresholds':pair,'synthesis_max':max(r['max_relative_difference'] for r in synthesis)},indent=2))
if __name__=='__main__':main()
