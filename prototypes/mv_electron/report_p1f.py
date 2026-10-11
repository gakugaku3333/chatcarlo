"""Portable Japanese evidence report; no fitted conclusions or hidden warnings."""
import json
import numpy as np
from .p1f_common import *

def report(s):
    g=all(s[k]['status']=='pass' for k in ('T1','T2'));h=all(s[k]['status']=='pass' for k in ('T0','T1','T2','T4','T5','T6'))
    lines=['# P1f 結果','',"問いG': "+('ELSEPAによる水素・酸素の表の自前生成が、再現可能な手順で成立。' if g else '必要なT1/T2の全条件を満たしておらず、成立を主張できない。'),
        "問いH': "+('2 MeV・水・指定条件で、自前生成の表による電子輸送の試作が成立。' if h else '成立の判定に必要な検査が不合格または未評価。未評価を物理的不合格と扱わない。'),
        '', 'P1e（`IHEF=1`）は表の検査で不合格だった。P1fは、その結果を見たあとに作り方を変えたものである。変更の根拠はEGS5に依存しない検査である。',
        '', '事前登録: [PREREGISTRATION.md](PREREGISTRATION.md)。2つのseedは20261013/20261014。やり直し・増統計・seed選び直し・合わせるための調整なし。警告は事前登録の分類で全件を保存。',
        '', '|項目|状態|値・根拠|','|---|---|---|']
    for k in ('T0','T1','T2','T3','T4','T5','T6','T7'):
        r=s[k];value=r.get('reason','詳細は summary.json')
        if k=='T0':value=f"既存4 backend各10000履歴の全配列完全一致。凍結 {r['freeze']['files']} ファイル。追加・削除・変更0。"
        if k=='T1' and 'bundled' in r:value=f"同梱DCS最大相対差 {r['bundled']['max_DCS_relative']:g}、断面積差 {r['bundled']['cross_sections_relative']}。4条件×{r['table_energies']}点×独立2巡、全配列一致。"
        if k=='T2' and 'maxima' in r:value=json.dumps(r['maxima'],ensure_ascii=False)+f"。受入整合警告 {r['accepted_consistency_warning_count']}、不受入警告 {r['rejected_warning_count']}、無効置換 {r['invalid_replacements']}。"
        if k in ('T6','T7') and 'max_dose_difference_over_Dref' in r:value=f"最大線量差/Dref={r['max_dose_difference_over_Dref']:.8g}、R50/Rp相対差={r['range_difference_over_reference_R50']}、合成SEM/Dref={r['max_dose_difference_SEM_over_Dref']:.8g}。"+('参考のみ、合否規則なし。' if k=='T7' else '')
        lines.append(f"|{k}|{LABELS[r['status']]}|{value}|")
    lines+=['','T3/T7は合否規則のない記録項目。計算済みかどうかを状態とは別に示す。',
        '', '警告の受入上限（全断面積0.5%、第1輸送断面積0.1%）は、既知の全断面積差約0.25%を見たうえで事前に置いた。全断面積の上限はELSEPA内部閾値の2.5倍である。結果を見たあとで除外していない。']
    if 'rows' in s['T2']:
        bad=[r for r in s['T2']['rows']+s['T2']['interpolation'] if r['status']=='fail']
        lines+=['','T2不合格の全内訳（警告の全件は [warnings.json](warnings.json)、数値の全件は [t2.json](t2.json)）:','', '|Z|電荷|MeV|種類|不合格の量|差 [全,G1,G2] / CDF|','|---|---|---|---|---|---|']
        for r in bad:lines.append(f"|{r['Z']}|{r['charge']}|{r['T']:.10g}|{r.get('kind','energy interpolation')}|{', '.join(r['failure_quantities'])}|{r.get('relative_table',r.get('relative'))} / {r.get('CDF_max_difference','—')}|")
        lines.append(f'\n不合格 {len(bad)} 件。格子点 {sum(r["kind"]=="node" for r in s["T2"]["rows"])}、直接中間点 {len(s["T2"]["interpolation"])}、旧切り替え前後の記録点 {sum(r["kind"]=="probes" for r in s["T2"]["rows"])}。±1%の2点比較で連続性は主張しない。')
    if 'R1' in s['T5']:
        lines+=['','K9(i)（差は[横変位二乗平均, 深さ変位平均, 補助量]、後ろの量は判定対象外）:','', '|道のり cm|差|合成SEM|状態|','|---|---|---|---|']
        for r in s['T5']['R1']['rows']:lines.append(f"|{r['path_cm']}|{r['relative']}|{r['difference_SEM_relative']}|{LABELS[r['status']]}|")
        lines+=['','K5/K5b/K7/K8/K9(ii): '+json.dumps(s['T5']['R2'],ensure_ascii=False)]
    if 'rows' in s['T3']:
        lines+=['','dcslib照合（計算済み、判定対象外）:','', '|Z|電荷|全断面積比|G1比|G2比|DCS角度比|','|---|---|---|---|---|---|']
        for z in (1,8):
            for q in (-1,1):
                rs=[r for r in s['T3']['rows'] if r['Z']==z and r['charge']==q]
                spans=[f"{min(r['xs_ratio'][i] for r in rs):.8g}–{max(r['xs_ratio'][i] for r in rs):.8g}" for i in range(3)]
                lines.append(f'|{z}|{q}|'+ '|'.join(spans)+f"|{min(r['DCS_ratio_min'] for r in rs):.8g}–{max(r['DCS_ratio_max'] for r in rs):.8g}|")
        lines+=['','水の合成G1/G2/後方半球積分、IHEF=0/1の比、旧切り替え前後の2点は t3.json / t2.json に全件保存。']
    lines+=['','補助指標（合否に使わない）:','']
    aux=s['auxiliary']
    if aux.get('status'):lines.append(aux['reason'])
    else:
        for k,r in aux.items():
            small={a:b for a,b in r.items() if a not in ('difference_over_Dref','combined_SEM_over_Dref')};lines.append('- '+k+': '+json.dumps(small,ensure_ascii=False))
    timing=OUT/'elsepa/benchmark.json'
    if timing.exists():lines+=['','輸送単価: '+json.dumps(json.loads(timing.read_text()),ensure_ascii=False)]
    lines+=['','生成時間・表サイズ:','', '|巡/Z/電荷|ケース数|秒|bytes|','|---|---|---|---|']
    for path in sorted(OUT.glob('generation_pass*.json')):
        r=json.loads(path.read_text());lines.append(f"|{path.stem}|{r['cases']}|{r['seconds']:.4f}|{r['table_bytes']}|")
    if (OUT/'generation_estimate.json').exists():lines+=['','生成前の代表条件による見込み: '+json.dumps(json.loads((OUT/'generation_estimate.json').read_text()),ensure_ascii=False)]
    lines+=['','確かめた事実:','', '- アーカイブSHA256と展開全ファイルの内容を照合。ELSEPAソースは変更していない。同梱再現はP1eの保存出力を読み取りだけで確認した。',
        '- ELSEPAのstdoutの第2輸送断面積はDCS積分由来。継承フィールドseries_xsの第3欄は独立な級数値ではなく、警告受入判定には使わない。',
        '- IHEF=0の格子点・直接計算点の警告、ERROR、級数/表断面積、収束の出力行、計算時間を保存した。生成表と生出力はP1f専用のリポジトリ外の保存先にある。',
        '- 水素とdcslibの差は上の数値として確認した。電子密度・核分布・電子交換はEGS5説明と共通だが、EGS5のMCPOL/MABS/IHEF/陽電子交換の指定は確定できていない。',
        '- T2で検査した表のハッシュをGS表と各runのレシートに結び付ける。陽電子の表は検査だけで、輸送には未使用。',
        '', '仮説: 水素差の由来として旧版の原子データや実装版の違いが考えられるが、今回原因は調べていない。P1eの不合格が切り替えの跳びを連続スプラインでまたいだためという説明も、原因を分離した実証ではない。',
        '', '利用条件: 計画/P1eの取得時に確認したMendeley掲載表記は「CC BY NC 3.0」。同梱ソース/readmeには別のライセンス記述がないことをP1eが確認した。今回はネットワークを使っていない。CC BY-NC 3.0の条文そのものと生成表への適用は未確認。',
        '', '未完了の項目:']
    pending=[k for k in ('T4','T5','T6','T7') if s[k]['status']=='not-evaluated']
    if pending:lines.append('、'.join(pending)+'。停止規則または未完了のrunによる。統計不足・未評価を合格と数えない。')
    else:lines.append('T0〜T6の判定は完了。T7は参考計算済みで合否規則がないため未評価。')
    if not (OUT/'elsepa/runs/base_2.npz').exists():lines.append('新規線量/R50/Rp/後方散乱・elsepa−dcslib線量差・輸送単価・深さ線量図は未実行。')
    lines+=['','実行コマンド・終了コード: [commands.json](commands.json)。データ出典・保存先・再生成手順: [DATA.md](DATA.md)。保存物のハッシュ: [manifest.json](manifest.json)。EGS5再実行・commit・計画編集なし。','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    plots(s)
    manifest()

def manifest():
    files={display(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json' and not p.name.endswith('.tmp')}
    files.update({display(p):sha(p) for p in WORK.rglob('*') if p.is_file()})
    write('manifest.json',dict(files=files,generated_tables_external=True,ELSEPA_sources_external=True,EGS5_rerun=False))

def plots(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if 'DCS_ratios' in s['T3']:
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,(z,q) in zip(axes.flat,((1,-1),(1,1),(8,-1),(8,1))):
            for r in s['T3']['DCS_ratios']:
                if r['Z']==z and r['charge']==q:ax.semilogx(r['theta_deg'][1:],r['ratio'][1:],label=f"{r['T']:g} MeV")
            ax.set(title=f'Z={z}, charge={q}',xlabel='Angle (degree)',ylabel='ELSEPA / dcslib DCS');ax.axhline(1,color='k',lw=.5);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(OUT/'dcs_ratios.png',dpi=160);plt.close(fig)
    if (OUT/'elsepa/runs/base_2.npz').exists() and s['T2']['status']=='pass':
        from .check_egs5 import parse
        from .analysis import dref
        ref=parse('base_2');D=dref(ref['q']);fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
        axes[0].plot(ref['z'],ref['q']/D,label='EGS5 saved reference',color='k')
        for b in ('elsepa','dcslib','mf'):
            path=OUT/'elsepa/runs/base_2.npz' if b=='elsepa' else ROOT/f'docs/validation/mv/p1d/{b}/runs/base_2.npz'
            with np.load(path) as data:q=data['batches'].mean(0)
            axes[0].plot(ref['z'],q/D,label=b);axes[1].plot(ref['z'],(q-ref['q'])/D,label=b)
        axes[0].set(ylabel='Dose / Dref');axes[1].set(xlabel='Depth (cm)',ylabel='Difference / Dref')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUT/'dose_comparison.png',dpi=160);plt.close(fig)
