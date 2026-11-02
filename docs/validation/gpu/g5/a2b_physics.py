"""Read-only B0 provenance: shared scalar physics and Philox match P9c's helpers."""
from pathlib import Path
import re
import hashlib

ROOT=Path(__file__).resolve().parents[4]


def functions(path):
    text=Path(path).read_text();result={}
    for match in re.finditer(r'P5_FN\s+[^;{}]+?\s+(\w+)\([^;{}]*\)\s*\{',text):
        start=match.start();pos=match.end();depth=1
        while depth:
            if text[pos]=='{':depth+=1
            elif text[pos]=='}':depth-=1
            pos+=1
        body=re.sub(r'//[^\n]*','',text[start:pos]);body=re.sub(r'\s+','',body)
        result[match.group(1)]=hashlib.sha256(body.encode()).hexdigest()
    return result


def check():
    old=ROOT/'prototypes/mv_electron/gpu';new=ROOT/'chatcarlo/gpu'
    pairs=(('p5a0_common.h','g5_base.h'),('p5_rng.h','t2_rng.h'),('p5_kernel.h','t2_physics.h'),('p8g_logs.h','t2_logs.h'),('p8g_functions.h','t2_functions.h'),('p8g_materials.h','t2_materials.h'),('p8g_kernel.h','t2_transport.h'))
    rows={}
    for a,b in pairs:
        x,y=functions(old/a),functions(new/b)
        for name,digest in x.items():
            if name in ('toy','p8g_history','p8g_record_history','p8g_clear'):continue
            if name not in y:
                # A2 moved base interpolation/quantities to t2_rng.h.
                y.update(functions(new/'t2_rng.h'))
            if name not in y or y[name]!=digest:raise ValueError('shared physics function differs: '+name)
            rows[name]=digest
    return dict(status='pass',functions=rows)

if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
