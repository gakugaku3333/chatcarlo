"""P4a single-process entry points; immutable outputs and frozen provenance.

A1 explicit-layer charged-source cases conflict with detailed-review item 5.
Plan amendment 1 removes those two cases from A1; they are recorded as out of scope.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import numba
from . import analysis_p4a as a
from . import photon_transport_p4 as kernel
from . import photon_transport_p3b as old
from . import run_p3s as s, run_p3a
from .physics import collision, hard

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/validation/mv/p4'
CACHE = Path('~/.cache/chatcarlo/p4').expanduser()
SOURCE_FILES = ('photon_transport_p4.py','run_p4a.py','analysis_p4a.py',
    'photon_transport_p3b.py','transport.py','physics.py','sampling.py',
    'backend_sampling.py','bremsstrahlung_p3r.py','flight_annihilation_p2c.py',
    'photon_data_p2.py','photon_sampling_p2.py','run_p3s.py')
REGIONS = [0.,0.,1.,1.,9.,.25,.125,1.,5.,10.,20.]
RULES = dict(fe=.05, k=.025, rule='own', backend='elsepa',
             annihilate_in_flight=True, score_kerma=False, score_pair=False,
             energies=list(a.ENERGIES), ranges=list(a.RANGES), batches=100,
             difference_limit=a.DIFF_LIMIT, sem_limit=a.SEM_LIMIT)
A1_CONFLICT = ('本文A1の明示単層＋charged_source_T／positron_source_Tの比較は、'
               '優先する詳細レビュー5の併用禁止と両立しない。代替検査は未実装。')
LIMITATION = ('この試験は密度のかけ方と境界を越える処理を確かめる。'
              'Fano試験はこの核の作りのうえで成り立つ見込みであり、'
              'ヒンジ方式そのものの近似の良し悪しを確かめられない。')


def plain(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):return [plain(v) for v in value]
    return value


def encoded(value):
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(encoded(value)+'\n')


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:f.write(value)


def provenance(inputs=None):
    inputs = s.inputs('elsepa') if inputs is None else inputs
    code = ROOT/'prototypes/mv_electron'
    return dict(source_hashes={name:hashlib.sha256((code/name).read_bytes()).hexdigest()
                               for name in SOURCE_FILES},
                table_hashes=run_p3a.table_hashes('elsepa', inputs),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              numba=numba.__version__))


def validate_lock(lock, current):
    if lock.get('provenance') != current or lock.get('rules') != RULES:
        raise RuntimeError('P4a lock hash, table, version or rule mismatch')
    runs = lock.get('runs', {})
    expected = {f'{i}_{system}' for i in range(3) for system in ('A','B')}
    if set(runs) != expected:raise RuntimeError('invalid locked conditions')
    for i in range(3):
        for j, system in enumerate(('A','B'), 1):
            row = runs[f'{i}_{system}']
            n = row.get('n', 0)
            if (not isinstance(n, int) or n < a.PILOT_N or n > a.MAX_N
                or n % a.PILOT_N or row.get('seed') != 20291783+1000*i+100*j):
                raise RuntimeError('invalid locked histories or seed')


def load_lock(current):
    lock = json.loads((OUT/'lock_p4a.json').read_text())
    validate_lock(lock, current)
    return lock


def require_checks(current):
    checks = json.loads((OUT/'checks_p4a.json').read_text())
    if checks.get('provenance') != current:
        raise RuntimeError('checks provenance differs from current inputs')
    for name in ('A0','A1','A4'):
        if checks.get(name, {}).get('status') != '合格':
            raise RuntimeError(f'{name} not passed; pilot forbidden')
    return checks


def save_result(path, config, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez(f, config_json=np.array(encoded(config)), **result)


def load_result(path, config):
    with np.load(path, allow_pickle=False) as f:
        if str(f['config_json']) != encoded(config):
            raise RuntimeError('saved P4a config differs; no rerun permitted')
        return {key:f[key].copy() for key in f.files if key != 'config_json'}


def saved(name, E, n, seed, *, current=None, inputs=None, kernel_name='p4', **options):
    inputs = s.inputs('elsepa') if inputs is None else inputs
    current = provenance(inputs) if current is None else current
    if current != provenance(inputs):raise RuntimeError('provenance changed before run')
    config = dict(E=E, n=n, seed=seed, kernel=kernel_name,
                  options=plain(options), rules=RULES, provenance=current)
    path = CACHE/(name+'.npz')
    if path.exists():return load_result(path, config)
    thickness = options.pop('thickness', 30.)
    kw = dict(annihilate_in_flight=True, score_kerma=False, score_pair=False,
              **s.rule_kwargs('own'))
    kw.update(options)
    started = time.perf_counter()
    transport = old.run_coupled if kernel_name == 'p3b' else kernel.run_coupled
    outputs = kernel.OUTPUTS[:25] if kernel_name == 'p3b' else kernel.OUTPUTS
    result = dict(zip(outputs, transport(float(E), n, seed, .05, .025,
                                        thickness, *inputs, **kw)))
    result['seconds'] = time.perf_counter()-started
    result['n'] = n
    if current != provenance(inputs):raise RuntimeError('provenance changed during run')
    save_result(path, config, result)
    return result


def fano_options(i, system, mode=1, margin=2., charge=-1, history=False, fault=0):
    geom = a.geometry(i, system, margin)
    return dict(thickness=geom['thickness'], layer_faces=geom['layer_faces'],
                layer_rho=geom['layer_rho'], mass_bin=geom['mass_bin'],
                fano_source_T=a.ENERGIES[i], fano_source_charge=charge,
                brems_mode=mode, history_rng=history, record_histories=history, fault=fault)


def stats(result, i, system, margin=2.):
    geom = a.geometry(i, system, margin)
    return a.layer_statistics(result['mass_batches'], int(result['n']),
                              a.ENERGIES[i], geom['mass_faces'], geom['mass_bin'], a.healthy(result))


def margin_check():
    rows = []
    for T, R in zip(a.ENERGIES, a.RANGES):
        grid = np.geomspace(.01, T, 2001)
        stopping = np.array([collision(x)-hard(x)[1] for x in grid])
        upper = float(T/np.min(stopping))
        integral = float(np.trapezoid(1/stopping, grid))
        rows.append(dict(T=T, upper=upper, integral=integral, margin=2*R,
                         status='合格' if upper < 2*R else '不合格'))
    return dict(rows=rows, status='合格' if all(r['status']=='合格' for r in rows) else '不合格')


def byte_equal(left, right):
    return all(np.asarray(x).shape == np.asarray(y).shape and
               np.asarray(x).dtype == np.asarray(y).dtype and
               np.asarray(x).tobytes() == np.asarray(y).tobytes() for x,y in zip(left,right))


def legacy_check(inputs):
    cases = [dict(), dict(charged_source_T=2., brems_mode=2), dict(positron_source_T=5.),
             dict(field_half=5., ssd=100., regions=REGIONS)]
    rows = []
    current = provenance(inputs)
    for i, case in enumerate(cases):
        common = dict(inputs=inputs, current=current)
        n = 2000 if i < 3 else 500
        legacy = saved(f'A1_{i}_old',6.,n,20291780,kernel_name='p3b',**common,**case)
        new = saved(f'A1_{i}_new',6.,n,20291780,**common,**case)
        exact = byte_equal([legacy[k] for k in kernel.OUTPUTS[:25]],
                           [new[k] for k in kernel.OUTPUTS[:25]])
        rows.append(dict(case=i+1, status='合格' if exact else '不合格',
                         seconds=float(legacy['seconds']+new['seconds'])))
        if i == 0:
            explicit = saved('A1_i_explicit',6.,n,20291780,layer_faces=[0.,30.],layer_rho=[1.],**common)
            exact = byte_equal([legacy[k] for k in ('totals','counts')],
                               [explicit[k] for k in ('totals','counts')])
            rows.append(dict(case='i explicit',status='合格' if exact else '不合格',
                             seconds=float(explicit['seconds'])))
    # Plan amendment 1: explicit-layer charged-source cases are out of scope (item 5 forbids them).
    rows.extend(dict(case=case,status='対象外',reason=A1_CONFLICT) for case in ('ii explicit','iii explicit'))
    status = '不合格' if any(r['status']=='不合格' for r in rows) else '合格'
    return dict(rows=rows, status=status, reason=A1_CONFLICT)


def history_pair(i, system, mode, seed, n=20000, fault=0, current=None, inputs=None, prefix='A2'):
    common = dict(current=current, inputs=inputs)
    left = saved(f'{prefix}_{i}_{system}_{mode}', a.ENERGIES[i], n, seed,
                 **common, **fano_options(i,system,mode,history=True,fault=fault))
    right = saved(f'{prefix}_{i}_H_{system}_{mode}', a.ENERGIES[i], n, seed,
                  **common, **fano_options(i,'H_'+system,mode,history=True))
    row = a.compare_histories(left['history_mass'],right['history_mass'],a.ENERGIES[i])
    row.update(E=a.ENERGIES[i], system=system, mode=mode, seed=seed,
               seconds=float(left['seconds']+right['seconds']))
    return row


def sensitivity(current, inputs):
    pair = history_pair(1,'A',1,20291784,fault=1,current=current,inputs=inputs,prefix='A4a')
    result = saved('A4b',1.,200000,20291784,current=current,inputs=inputs,
                   **fano_options(1,'A',fault=2))
    result['n'] = 200000
    rows = stats(result,1,'A')[1:4]
    passed = pair['matched'] < 19980 and any(abs(r['difference']) > .01 and
                                            abs(r['difference']) > 3*r['sem'] for r in rows)
    return dict(status='合格' if passed else '不合格', a=pair, b=rows,
                seconds=float(pair['seconds']+result['seconds']))


def stop_report(checks, reason):
    lines = ['# P4a 結果', '', '停止理由: '+reason, '', LIMITATION, '',
             '| 検査 | 状態 |', '|---|---|']
    for name in ('A0','A1','A4','A2'):
        lines.append(f"| {name} | {checks.get(name,{}).get('status','未実施')} |")
    lines += ['', 'A3（18量）、報告のみの量、粒子数固定、本番、解析: 未実施。'+reason,
              '', f"計算時間: {checks.get('seconds',0):.3f} s", '',
              '10 MeV固定。9 MeVへの変更は行わない。', '']
    write_text(OUT/'RESULTS_P4A.md', '\n'.join(lines))


def checks():
    for target in (OUT/'checks_p4a.json',OUT/'RESULTS_P4A.md'):
        if target.exists():raise FileExistsError(target)
    inputs = s.inputs('elsepa');current = provenance(inputs);started = time.perf_counter()
    value = dict(provenance=current, A0=margin_check())
    reason = None
    if value['A0']['status'] != '合格':reason = 'A0不成立'
    else:
        value['A1'] = legacy_check(inputs)
        if value['A1']['status'] != '合格':reason = value['A1']['reason']
    if reason is None:
        value['A4'] = sensitivity(current, inputs)
        if value['A4']['status'] != '合格':reason = 'A4不成立'
    if reason is None:
        pairs = [history_pair(i,system,mode,20291781+1000*i+100*j+10*(mode-1),
                              current=current,inputs=inputs)
                 for i in range(3) for j,system in enumerate(('A','B'),1) for mode in (1,2)]
        value['A2'] = dict(rows=pairs, status='合格' if all(r['status']=='合格' for r in pairs) else '不合格')
    if provenance(inputs) != current:raise RuntimeError('provenance changed during checks')
    value['seconds'] = time.perf_counter()-started
    write_json(OUT/'checks_p4a.json',value)
    if reason:
        stop_report(value,reason)
        raise RuntimeError(reason)
    return value


def pilot():
    inputs = s.inputs('elsepa');current = provenance(inputs);require_checks(current)
    if (OUT/'lock_p4a.json').exists():raise FileExistsError(OUT/'lock_p4a.json')
    runs = {}
    for i in range(3):
        for j,system in enumerate(('A','B'),1):
            r = saved(f'pilot_{i}_{system}',a.ENERGIES[i],a.PILOT_N,20291782+1000*i+100*j,
                      current=current,inputs=inputs,**fano_options(i,system))
            r['n'] = a.PILOT_N
            rows = stats(r,i,system)
            if not a.healthy(r):raise RuntimeError('pilot ledger/cutoff violation; no lock')
            n = a.required_histories([row['sem'] for row in rows[1:4]])
            runs[f'{i}_{system}'] = dict(n=n,seed=20291783+1000*i+100*j,pilot=rows,
                                         pilot_seed=20291782+1000*i+100*j)
    if provenance(inputs) != current:raise RuntimeError('provenance changed before lock')
    write_json(OUT/'lock_p4a.json',dict(provenance=current,rules=RULES,runs=runs))


def production():
    inputs = s.inputs('elsepa');current = provenance(inputs);lock = load_lock(current)
    for i in range(3):
        for system in ('A','B'):
            load_lock(provenance(inputs))
            spec = lock['runs'][f'{i}_{system}']
            saved(f'production_{i}_{system}',a.ENERGIES[i],spec['n'],spec['seed'],
                  current=current,inputs=inputs,**fano_options(i,system))
    saved('report_4R',1.,200000,20291786,current=current,inputs=inputs,**fano_options(1,'A',margin=4.))
    for i in range(3):
        for j,system in enumerate(('A','B'),1):
            load_lock(provenance(inputs))
            saved(f'report_brems_{i}_{system}',a.ENERGIES[i],200000,20291785+1000*i+100*j,
                  current=current,inputs=inputs,**fano_options(i,system,mode=2))
    saved('report_positron',1.,200000,20291785+1000+100,current=current,inputs=inputs,
          **fano_options(1,'A',mode=2,charge=1))


def existing(name):
    with np.load(CACHE/(name+'.npz'),allow_pickle=False) as f:
        config = json.loads(str(f['config_json']))
    if config['provenance'] != provenance() or config['rules'] != RULES:
        raise RuntimeError('analysis input provenance mismatch')
    return load_result(CACHE/(name+'.npz'),config),config


def analyse():
    current = provenance();lock = load_lock(current);checked = require_checks(current)
    path = OUT/'RESULTS_P4A.md'
    if path.exists():raise FileExistsError(path)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = ['# P4a 結果','','10 MeV固定。9 MeVへの切り替えなし。', '', LIMITATION, '',
             '中央の判定する範囲には外側の端の影響が届かない。体系全体では端から電子が出る。', '',
             '## A0・A1・A4・A2', '', '```json', encoded(checked), '```', '',
             '## A3（本番の18量。試し計算は判定に使用しない）','',
             '| E MeV | 体系 | 層 | 本番N | 差 % | SEM % | 状態 | 試しの差 % | 試しSEM % |',
             '|---|---|---|---|---|---|---|---|---|']
    seconds = float(checked['seconds'])
    for i in range(3):
        for system in ('A','B'):
            r,c = existing(f'production_{i}_{system}');r['n'] = c['n']
            spec = lock['runs'][f'{i}_{system}']
            if c['n'] != spec['n'] or c['seed'] != spec['seed'] or c['options'] != plain(fano_options(i,system)):
                raise RuntimeError('production data differs from lock')
            rows = stats(r,i,system)
            for row,trial in zip(rows[1:4],spec['pilot'][1:4]):
                lines.append(f"| {a.ENERGIES[i]} | {system} | {row['layer']} | {c['n']} | "
                    f"{100*row['difference']:.6f} | {100*row['sem']:.6f} | {row['status']} | "
                    f"{100*trial['difference']:.6f} | {100*trial['sem']:.6f} |")
            seconds += float(r['seconds'])
            g = a.geometry(i,system)
            expectation = c['n']*a.ENERGIES[i]*g['mass_bin']/g['mass_faces'][-1]
            fig,ax = plt.subplots()
            ax.plot((np.arange(r['mass_batches'].shape[1])+.5)*g['mass_bin'],
                    r['mass_batches'].sum(axis=0)/expectation)
            ax.set(xlabel='Mass depth [g/cm²]',ylabel='Deposit / expectation',title=f'{system}, {a.ENERGIES[i]} MeV')
            dest = OUT/f'mass_{i}_{system}.png'
            with dest.open('xb') as f:fig.savefig(f,format='png',dpi=150)
            plt.close(fig)
    lines += ['', '## 報告のみ','']
    for name in ['report_4R','report_positron']+[f'report_brems_{i}_{system}' for i in range(3) for system in ('A','B')]:
        r,c = existing(name);seconds += float(r['seconds'])
        opts = c['options'];mf = np.r_[0.,np.cumsum(np.diff(opts['layer_faces'])*opts['layer_rho'])]
        if name.startswith('report_brems'):
            value = a.report_only_doses(r['mass_batches'],mf,opts['mass_bin'])
        elif name == 'report_4R':
            value = a.layer_statistics(r['mass_batches'],c['n'],1.,mf,opts['mass_bin'],a.healthy(r))[1:4]
        else:
            edges = np.rint(mf/opts['mass_bin']).astype(int)
            value = [dict(layer=j,deposit=float(r['mass_batches'][:,edges[j]:edges[j+1]].sum())) for j in range(5)]
        lines += [name, '', '```json', encoded(value), '```', '']
    seconds += sum(float(existing(f'pilot_{i}_{system}')[0]['seconds']) for i in range(3) for system in ('A','B'))
    lines += [f'計算時間（保存された核の実行時間＋checks）: {seconds:.3f} s','']
    load_lock(provenance())
    write_text(path,'\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('checks','pilot','production','analyse'))
    args = parser.parse_args()
    globals()[args.stage]()


if __name__ == '__main__':main()
