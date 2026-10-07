"""Convert saved EGS5 output to absolute Gy/history; preserve raw statistics."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

BASE = Path(__file__).resolve().parent
MEV_TO_J = 1.602176634e-13

def load(name):
    p = BASE / name / 'egs5job.out'
    return p, p.read_text()

def bsf(name):
    p, text = load(name)
    mean = float(re.search(r'\(MeV\)\s*=\s*([\d.E+-]+)', text).group(1))
    sem = float(re.search(r'Standard error of the mean \(MeV\)\s*=\s*([\d.E+-]+)', text).group(1))
    # 10 x 10 x .2 cm^3 * rho 1 g/cm^3 = 20 g = .020 kg.
    return {'mean_MeV_per_history':mean,'sem_MeV_per_history':sem,
            'mass_g':20.,'mean_Gy_per_history':mean*MEV_TO_J/.020,
            'sem_Gy_per_history':sem*MEV_TO_J/.020,'relative_sem':sem/mean,
            'output_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    thin, phantom = bsf('bsf_thinslab'), bsf('bsf_phantom')
    ratio = phantom['mean_MeV_per_history']/thin['mean_MeV_per_history']
    rel = math.hypot(thin['relative_sem'], phantom['relative_sem'])
    p, text = load('pdd_phantom')
    bins = []
    for match in re.finditer(r'^\s*(\S+)\s+mean\(MeV\)=\s*([\d.E+-]+)\s+sem\(MeV\)=\s*([\d.E+-]+)\s+relerr\(%\)=\s*([\d.E+-]+)', text, re.M):
        label, m, s, reported = match.groups()
        m, s = float(m),float(s)
        mass = 4. if label.startswith('pdd') else 2.
        bins.append({'label':label,'mean_MeV_per_history':m,'sem_MeV_per_history':s,
                     'reported_relative_sem_percent':float(reported),'mass_g':mass,
                     'mean_Gy_per_history':m*MEV_TO_J/(mass*.001),
                     'sem_Gy_per_history':s*MEV_TO_J/(mass*.001),'relative_sem':s/m})
    assert len(bins)==47
    warning_records = []
    for name in ('pdd_repeat_a','pdd_repeat_b','bsf_thinslab','bsf_phantom','pdd_phantom'):
        for fname in ('build.log','run.log','egs5job.out','pgs5job.pegs5lst'):
            lines = (BASE/name/fname).read_text().splitlines()
            hits = []
            for i,line in enumerate(lines):
                if re.search(r'warning|error:|insufficient|stopped|fatal',line,re.I):
                    hits.append({'line':i+1,'text':'\n'.join(lines[i:min(i+2,len(lines))])})
            warning_records.append({'run':name,'file':fname,'hits':hits})
    result = {'units':'Gy/history','density_g_cm3':1.,
              'conversion':'mean(MeV/history) * 1.602176634e-13 J/MeV / mass(kg)',
              'normalization':'Absolute per-history values; PDD surface/OCR centre normalization is for plots only, not acceptance.',
              'bsf':{'thin':thin,'phantom':phantom,'value':ratio,'sem':ratio*rel,'relative_sem':rel},
              'pdd':{'histories':100000000,'inseed':1,'bins':bins,
                     'output_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},
              'warnings':warning_records}
    (BASE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (BASE/'pdd_bins.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)
    print(json.dumps({'bsf_value':ratio,'bsf_sem':ratio*rel,'bsf_relative_sem':rel,
                      'thin':thin,'phantom':phantom,'pdd_first':bins[0],
                      'pdd_last':bins[14],'pdd_statistically_eligible':sum(b['relative_sem']<.01 for b in bins)},indent=2))

if __name__ == '__main__':
    main()
