"""Japanese P1e evidence report, including negative/unevaluated outcomes."""
import json
from .p1e_common import *

def report(s):
    lines=['# P1e 結果','', '事前登録: [PREREGISTRATION.md](PREREGISTRATION.md)。生成表と取得物はリポジトリ外に保存。EGS5再実行・commit・計画変更は行っていない。','', '|項目|状態|値・根拠|','|---|---|---|']
    for key in ('S0','S1','S2','S3','S4','S5','S6','S7'):
        r=s[key];value=''
        if key=='S0':value=f"既存4 backend各10000履歴、全配列完全一致。凍結 {r['freeze']['files']} ファイル、追加・削除・変更なし。"
        elif key=='S1' and 'bundled' in r:value=f"ZIP SHA256一致。同梱再現DCS最大相対差 {r['bundled']['max_DCS_relative']:.8g}、断面積差 {r['bundled']['cross_sections_relative']}。独立2巡、各元素・電荷 {r['table_energies']} 点、全配列完全一致。"
        elif key=='S2' and 'maxima' in r:value='最大差: '+json.dumps(r['maxima'],ensure_ascii=False)+f"。数値警告 {r['numerical_warning_count']}、因子分解通知 {r['factorization_notice_count']}。"
        else:value=r.get('reason','詳細は summary.json')
        lines.append(f"|{key}|{LABELS[r['status']]}|{value}|")
    lines+=['','S3とS7は記録項目で合否規則がないため、計算済みでも状態は未評価とし、計算の有無を別に記す。']
    if 'rows' in s['S2']:
        bad=[r for r in s['S2']['rows']+s['S2']['interpolation'] if r['status']=='fail']
        lines+=['','S2不合格ケース（全件は [s2.json](s2.json)）:','', '|Z|電荷|MeV|表積分差または補間差 [全,G1,G2]|CDF差|','|---|---|---|---|---|']
        for r in bad[:16]:lines.append(f"|{r['Z']}|{r['charge']}|{r['T']:.9g}|{r.get('relative_table',r.get('relative'))}|{r.get('CDF_max_difference','—')}|")
        lines.append(f'\n不合格ケース計 {len(bad)}。格子・モデル・補間・倍率・許容幅の調整、seed選択、増統計は行っていない。')
    if 'rows' in s['S3']:
        lines+=['','dcslibとの照合（合否には使用しない。全角度の比と水の合成値は [s3.json](s3.json)）:','', '|Z|電荷|全断面積比の範囲|G1比の範囲|G2比の範囲|DCS角度比の範囲|','|---|---|---|---|---|---|']
        for z in (1,8):
            for q in (-1,1):
                rows=[r for r in s['S3']['rows'] if r['Z']==z and r['charge']==q]
                if rows:
                    spans=[f"{min(r['xs_ratio'][i] for r in rows):.8g}–{max(r['xs_ratio'][i] for r in rows):.8g}" for i in range(3)]
                    lines.append(f"|{z}|{q}|"+'|'.join(spans)+f"|{min(r['DCS_ratio_min'] for r in rows):.8g}–{max(r['DCS_ratio_max'] for r in rows):.8g}|")
    lines+=['','補助指標: '+json.dumps(s['auxiliary'],ensure_ascii=False) if s['auxiliary'].get('status') else '\n補助指標は summary.json に記録。']
    estimates=OUT/'generation_estimate.json'
    if estimates.exists():lines+=['','生成前の代表条件2 MeVの時間・見込み: '+json.dumps(json.loads(estimates.read_text()),ensure_ascii=False)]
    lines+=['','生成実績（初回のH電子1 keVの1ケースは出力検査中断前に時間レシートを作っていない。時間不明のケースを0秒と物理解釈せず、集計は既知分のみ）:','', '|巡|Z|電荷|ケース数|既知の秒数|表 bytes|','|---|---|---|---|---|---|']
    for path in sorted(OUT.glob('generation_pass*.json')):
        r=json.loads(path.read_text());lines.append(f"|{path.stem.split('_')[1]}|{path.stem.split('_')[2]}|{path.stem.split('_')[3]}|{r['cases']}|{r['seconds']:.4f}|{r['table_bytes']}|")
    lines+=['','確かめた事実:','', '- 同梱出力を再現し、アーカイブの各ファイルと展開物の内容一致を確認した。ELSEPAソースは変更していない。', '- IHEF=1は指定閾値の上で因子分解に切り替わり、交換が無効になる。因子分解ケースのstdoutには級数側の3断面積が印字されない。表ではseries_available=false、series_xs=NaNとして「印字なし」を保存し、値を補っていない。', '- 出力検査の実装中に、級数側の印字、無効ポテンシャルの表記、陽電子で交換の説明行が省略されることの扱いを修正した。元の生出力を保存し、完了したELSEPA計算は再実行せず変換を再開した。物理設定は変えていない。', '- 水素差の数値はIHEF差・dcslib比として記録した。生成元の電子密度・核分布・交換はEGS5説明と共通。EGS5のMCPOL/MABS/IHEF/陽電子交換指定は確認できていない。','', '仮説: 水素の差が旧版の原子データ、実装版、IHEFの選択に由来する可能性はあるが、原因の確定には至っていない。今回の比較だけで同じ断面積や輸送の成立を主張しない。','', '利用条件: 計画に記録されたMendeley掲載表記は「CC BY NC 3.0」。今回の展開ソース/readmeにも別ライセンスの記述は見つからなかった。CC BY-NC 3.0条文そのものと生成表への適用は未確認。陽電子表は検査のみで、輸送には未使用。','']
    accepted_g=all(s[k]['status']=='pass' for k in ('S1','S2'));accepted_h=all(s[k]['status']=='pass' for k in ('S0','S1','S2','S4','S5','S6'))
    lines.append('問いG: '+('ELSEPAによる水素・酸素の表の自前生成が、再現可能な手順で成立。' if accepted_g else '成立の判定に必要なS1/S2の全条件を満たしていない。'))
    lines.append('問いH: '+('2 MeV・水・指定条件で、自前生成の表による電子輸送の試作が成立。' if accepted_h else '未成立の判定ではなく、必要な検査が未評価または不合格のため成立を主張できない。'))
    if s['S2']['status']=='fail':lines+=['','未完了: S4のGS一貫性、S5のK9(i)/K5/K5b/K7/K8/K9(ii)、S6の2 MeV比較、S7の10 MeV参考、輸送単価、elsepa線量/R50/Rp/後方散乱、elsepa−dcslib線量差、3 backend/EGS5の深さ線量図。S2停止規則に従い実行していない。']
    lines+=['','実行コマンドと終了コードは [commands.json](commands.json)。再評価:','', '```bash','.venv/bin/python prototypes/mv_electron/run_checks_p1e.py --from-saved','.venv/bin/python prototypes/mv_electron/run_checks_p1e.py --verify-untouched','```','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    if 'DCS_ratios' in s['S3']:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,(z,q) in zip(axes.flat,((1,-1),(1,1),(8,-1),(8,1))):
            for r in s['S3']['DCS_ratios']:
                if r['Z']==z and r['charge']==q:ax.semilogx(r['theta_deg'][1:],r['ratio'][1:],label=f"{r['T']:g} MeV")
            ax.set(title=f'Z={z}, charge={q}',xlabel='Angle (degree)',ylabel='ELSEPA / dcslib DCS');ax.axhline(1,color='k',lw=.5);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(OUT/'dcs_ratios.png',dpi=160);plt.close(fig)
    if (OUT/'elsepa/runs/base_2.npz').exists() and s['S2']['status']=='pass':
        from .check_egs5 import parse
        from .analysis import dref
        import matplotlib.pyplot as plt
        ref=parse('base_2');D=dref(ref['q']);fig,axes=plt.subplots(2,1,figsize=(8,7),sharex=True)
        axes[0].plot(ref['z'],ref['q']/D,label='EGS5 saved reference',color='k')
        for b in ('elsepa','dcslib','mf'):
            p=OUT/'elsepa/runs/base_2.npz' if b=='elsepa' else ROOT/f'docs/validation/mv/p1d/{b}/runs/base_2.npz'
            with np.load(p) as data:q=data['batches'].mean(0)
            axes[0].plot(ref['z'],q/D,label=b);axes[1].plot(ref['z'],(q-ref['q'])/D,label=b)
        axes[0].set(ylabel='Dose / Dref');axes[1].set(xlabel='Depth (cm)',ylabel='Difference / Dref')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUT/'dose_comparison.png',dpi=160);plt.close(fig)
    files={display(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json' and not p.name.endswith('.tmp')}
    files.update({display(p):sha(p) for p in WORK.rglob('*') if p.is_file() and 'source' not in p.parts})
    write('manifest.json',dict(files=files,generated_tables_external=True,ELSEPA_sources_external=True,EGS5_rerun=False))
