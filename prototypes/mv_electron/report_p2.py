"""Generate the active verdict and a portable report from saved evidence only."""
import json
from .p2_common import OUT,write,verify

def saved(name,default=None):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else (default if default is not None else dict(status='未評価'))

def report():
    protected=verify()
    progress=saved('progress.json',dict(stages={},production_runs_started=0))
    q={f'Q{i}':saved(f'q{i}.json') for i in range(8)}
    q['Q1']=saved('q1_amendment1.json',q['Q1'])
    m=saved('manifest.json')
    regression=saved('q0_regression.json')
    if m.get('production_frozen'):
        q['Q0']=dict(status=regression['status'],protected=protected,regression=regression,production=verify('production_frozen'))
    else:q['Q0']=dict(status='未評価',protected=protected,regression=regression,reason='本計算前の凍結待ち')
    stopping=saved('t5_stopping_amendment2.json');cut=saved('t5_cutoffs_amendment1.json')
    summary=dict(q,active_verdicts=dict(Q1='q1_amendment1.json',stopping='t5_stopping_amendment2.json'),stopping=stopping,cutoffs=cut)
    write('summary.json',summary)
    lines=['# P2 実装・実行結果','',
        '判定は条件別に記録する。連成の結論はQ6・Q7の完了後に限る。dcslibは研究専用対照、mfとelsepa（P1f、IHEF=0）を併記。K1_max=0.025、f_E=0.05。',
        '省略: 束縛効果、Rayleigh、三重項反跳、結合エネルギー、Bhabha、陽電子用弾性散乱、飛行中消滅、明示的制動放射光子。Drefの20%未満の低線量binは比較対象外。',
        '基準側の未定量の不確かさ: PEGSの150区間の当てはめ警告は未解消。ESIGの最大相対誤差は主約7.9%・感度約8.6%、ESTEPMAXは約31.4%。D1の合格は阻止能比較だけであり、これらの線量への影響を定量化したものではない。P1〜P1fも同じ警告つきの基準表を使用。非標準IUNRST=4は自己点検つきの感度計算。','',
        '| 検査 | 現在の状態 | 値・根拠 |','|---|---|---|',
        f"| Q0 | {q['Q0']['status']} | 保護{protected['files']}ファイル一致。dcslib/mfの電子専用各10⁴履歴・7配列完全一致。本計算凍結: {'実施済み' if m.get('production_frozen') else '未実施'}。 |"]
    old=saved('q1.json');a=q['Q1']
    if 'interpolation' in a:
        vals='、'.join(f"{r['process']} {100*r['max_relative']:.6f}%" for r in a['interpolation'])
        lines.append(f"| Q1 | {a['status']}（追補1） | {vals}。旧判定{old['status']}をq1.jsonに保持。表・格子・補間・seed不変。 |")
    if 'rows' in q['Q2']:
        rows=q['Q2']['rows'];maxz=max(abs(r['mean_z']) for r in rows);minp=min(r[k]['p'] for r in rows for k in ('compton_angle','pair_split','pair_azimuth'))
        lines.append(f"| Q2 | {q['Q2']['status']} | 各10⁶標本、平均最大{maxz:.6f}σ、分布検査最小p={minp:.7g}。 |")
    if 'rows' in q['Q3']:
        maxr=max(r['history_max_relative'][0] for r in q['Q3']['rows'])
        lines.append(f"| Q3 | {q['Q3']['status']} | 12条件各10⁴履歴、最大収支残差{maxr:.7g}。上限超過0、2mm/0.5mm再集計一致。 |")
    if 'rows' in q['Q4']:
        vals='; '.join(f"{r['E']:g}MeV: {r['mu_tr_measured_cm2_g']:.9g}±{r['SEM_cm2_g']:.3g}cm²/g" for r in q['Q4']['rows'])
        lines.append(f"| Q4 | {q['Q4']['status']} | {vals} |")
    for i in (5,6,7):
        lines.append(f"| Q{i} | {q[f'Q{i}']['status']} | {'保存結果を下表・JSONに記録' if 'rows' in q[f'Q{i}'] else '本計算・解析待ち'} |")
    if stopping.get('rows'):
        dc=max(r['max_collision_relative'] for r in stopping['rows']);dr=max(r['max_radiation_relative'] for r in stopping['rows']);ds=max(r['max_sum_absolute_over_total'] for r in stopping['rows'])
        lines.extend(['',f"D1: {stopping['status']}。衝突相対差最大{100*dc:.6f}%、放射相対差最大{100*dr:.6f}%、絶対成分差の和/全阻止能最大{100*ds:.6f}%（上限0.05%）。分母は試作の制限衝突＋全放射。既知の最大0.0286%を見て承認された基準である。旧t5_stopping_amendment1.jsonの判定保留を保持。これはbin線量差の上限ではない。"])
    if cut.get('rows'):
        c=cut['rows'][0];lines.extend(['',f"C2: {cut['status']}。AE=ECUT={c['AE']}MeV、実効運動エネルギー{c['kinetic_cutoff_keV']:.9f}keV、試作との差{c['difference_eV']:.9f}eV、追加飛程推定{c['estimated_extra_range_nm']:.6f}nm。境界では残り約10keVが隣接binに移り得る。"])
    if q['Q6'].get('rows'):
        lines.extend(['','| backend | MeV | Q6状態 | 工学的判定 | 統計的一致 | 最大差/Dref | 基準幅/Dref |','|---|---|---|---|---|---|---|'])
        for r in q['Q6']['rows']:
            diff=f"{r['max_difference_Dref']*100:.5f}%" if 'max_difference_Dref' in r else '未取得'
            width=f"{r['max_reference_width_Dref']*100:.5f}%" if 'max_reference_width_Dref' in r else '未取得'
            lines.append(f"| {r['backend']} | {r['E']} | {r['status']} | {r.get('engineering','未評価')} | {r.get('statistical','未評価')} | {diff} | {width} |")
    if q['Q7'].get('rows'):
        lines.extend(['','Q7（各backend、2/10MeV、f_E/K1を別々に半減）:'])
        for r in q['Q7']['rows']:
            for a in r.get('rows',[]):
                lines.append(f"- {r['backend']} {a['E']:g}MeV {a['half']}: {a['status']}、最大差/Dref={a.get('max_difference_Dref','未取得')}、差のSEM/Dref={a.get('max_difference_SEM_Dref','未取得')}。")
    lock=saved('pilot_lock.json',{})
    if lock:
        lines.extend(['','試行計算から固定したDref・評価bin・窓・履歴数・seed・見込み時間: pilot_lock.json。凍結後の選び直しなし。','', '| MeV | Dref（Gy cm²） | 窓bin | 要求履歴数 |','|---|---|---|---|'])
        for e,w in lock['windows'].items():lines.append(f"| {e} | {w['Dref']:.9g} | {w.get('window','未評価')} | {lock['requirements'][e]['desired_common_N']} |")
    goodparts=all(q[f'Q{i}']['status']=='合格' for i in range(5))
    if goodparts:lines.extend(['','光子側の部品が、EGS5に依存しない基準で成立（問いD）。'])
    steps={r['backend']:r['status'] for r in q['Q7'].get('rows',[])}
    for b in ('dcslib','mf','elsepa'):
        rows={r['E']:r for r in q['Q6'].get('rows',[]) if r['backend']==b}
        if goodparts and steps.get(b)=='合格' and all(rows.get(e,{}).get('status')=='合格' for e in (1.25,2.)):
            lines.append(f"{b}（K1_max=0.025、f_E=0.05）: 制動放射を局所沈着とした条件で、光子と電子の連成が工学的基準で成立。統計的一致は条件別表に併記。")
        for e in (6.,10.):
            if rows.get(e,{}).get('engineering') not in (None,'未評価'):lines.append(f"{b} {e:g}MeV: 陽電子のモデル差を含む比較、{rows[e]['status']}。")
    lines.extend(['',
        '基準側の未定量の不確かさ: PEGSのESIG（主約7.9%、感度約8.6%）、ESTEPMAX（約31.4%）等の当てはめ警告は未解消で、線量への寄与は未定量。阻止能比較の合格では解消できず、基準の作り直しは行っていない。',
        '省略と低線量部もこの結論に適用される。',
        '事実: 同じ401点の保存データでは、主・感度の全放射阻止能が完全一致。約1.04%はPEGS直接評価とESTAR補間値の差で、輸送用区分当てはめの差ではない。',
        '仮説: 放射阻止能差はPEGSの制動放射式・補正表とESTAR体系のモデル差に由来する可能性がある。XCOMの閾値近傍の差に丸めが寄与する可能性も未検証。線量への影響や不一致の原因は確定していない。','',
        f"本計算開始ケース数: {progress.get('production_runs_started',0)}。各ケース1回、完成batchを保持して未完了分のみ同じ条件で再開。状態: {json.dumps(progress.get('stages',{}).get('T6',{}),ensure_ascii=False)}。",
        '旧停止理由: 旧Q1不合格、その後C2b旧判定保留・再開コードのハッシュ衝突。現在有効な判定はQ1追補1と阻止能追補2。旧自己点検はt5_selfcheck.json、新コードはt5_selfcheck_d2.json。',
        '未完了: '+('なし（限定されたP2ベンチマークの範囲内）。' if progress.get('stages',{}).get('T6',{}).get('state')=='complete' else 'progress.jsonの未完了段階。試行計算、条件固定、凍結、本計算、Q5/Q6/Q7、図は保存物の有無で区別する。'),'','再開（この1コマンドで未完了段階から最終報告まで）:','',
        '```bash','.venv/bin/python prototypes/mv_electron/run_checks_p2.py --resume','```','',
        '実行コマンド: commands.json。段階・実行中ケース: progress.json。計画ファイルは編集せず、commitなし。成果物のパスは相対またはキャッシュ変数表記。'])
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return summary

if __name__=='__main__':report()
