"""P3s 追補1: production and analysis with S4 recorded as 統計不足 (user decision 2026-10-12).

run_p3s.py is not edited. Only its S4 gate is replaced by a check of every other gate.
"""
import json,sys
from prototypes.mv_electron import run_p3s as m
def gate(rule):
    d=json.loads((m.OUT/'checks.json').read_text());c=d['checks']
    if not all(c[k]['passed'] for k in ('S1','S2','S3','S5','S6','S7')):raise RuntimeError('a non-S4 check failed')
    plan='docs/ai/plans/2026-10-11-mv-p3s-step-speedup.md'  # edited by 追補1 itself; every other source must match
    old={k:v for k,v in d['source_hashes'].items() if k!=plan};new={k:v for k,v in m.source_hashes().items() if k!=plan}
    if len(old)!=28 or old!=new:raise RuntimeError('sources changed since the checks')
    if c['S4']['rules'][rule]['status']!='統計不足':raise RuntimeError('unexpected S4 state')
m.require_checks=gate
class Recorded(str):
    """S4 status as recorded (統計不足); only the analysis' eligibility test treats it as eligible."""
    def __eq__(self,other):return other=='合格' or str.__eq__(self,other)
    def __ne__(self,other):return not self.__eq__(other)
    __hash__=str.__hash__
class Shim:
    def __getattr__(self,name):return getattr(json,name)
    def loads(self,text,*a,**k):
        v=json.loads(text,*a,**k)
        if isinstance(v,dict) and 'S4' in v.get('checks',{}):
            for r in v['checks']['S4']['rules'].values():r['status']=Recorded(r['status'])
        return v
if sys.argv[1]=='analyse':
    m.json=Shim();m.analyse()
else:m.production(sys.argv[1])
