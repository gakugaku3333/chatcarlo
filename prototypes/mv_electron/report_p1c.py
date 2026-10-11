"""Japanese P1c report from saved values; no hand-entered result numbers."""
import hashlib,json
import numpy as np
from .regression_p1c import ROOT,P

def report(s):
    p1=json.loads((ROOT/'docs/validation/mv/p1/egs5_checks.json').read_text())['K6']
    dc=json.loads((ROOT/'docs/validation/mv/p1b/comparison_dcslib.json').read_text())
    lines=['# P1c — McKinley–Feshbach解析補正の結果','',
           '[承認計画](../../../ai/plans/2026-10-08-mv-p1c-mott-correction.md)に従い、事前登録をコード変更前に固定し、変更前N1を保存した。電子・水（H/O）のみ。η、B、Z(Z+1)、損失モデル、輸送状態機械、採点、f_E=0.05、K1_max=0.05、共通解析と基準EGS5 runを固定。係数・倍率・刻み・閾値の合わせ込みは行っていない。EGS5再実行なし。commitなし。','',
           '|項目|値・検査|判定|','|---|---|---|']
    n=s['N1'];count=sum(len(v) for v in n['checks'].values());lines.append(f"|N1|3 backend × 2 MeV・10⁴履歴、再生成表を含む{count}配列の完全一致。追跡対象{n['untouched']['tracked_files']}ファイル不変|{n['status']}|")
    n=s['N2'];lines.append(f"|N2|{len(n['rows'])}エネルギー点（10–20 keV実表中点を含む）。Gauss16→32最大相対変化 {max(r['doubling_max_relative'] for r in n['rows']):.10g}、Born G1閉形式との最大相対差 {max(r['born_relative'] for r in n['rows']):.10g}、最小DCS {min(r['minimum_density'] for r in n['rows']):.10g} cm⁻¹/dμ|{n['status']}|")
    n=s['N3'];checks=n['precalculation_crosscheck'];lines.append(f"|N3|5モデル×6エネルギーのΣ0/G1/G2/後方積分。Claude事前値の照合 {sum(r['within_1e3'] for r in checks)}/{len(checks)}件が相対10⁻³以内|記録のみ|")
    n=s['N4']
    if 'K3_rows' in n:
        lines.append(f"|N4|{len(n['K3_rows'])}条件のGSモーメント。最大相対差 {max(max(abs(v) for v in r['relative']) for r in n['K3_rows']):.10g}、規格化最大誤差 {n['normalization_max_error']:.10g}、次数/積分倍増の最大変化 {max(r['moment_change'] for r in n['refinement']):.10g}、逆CDF最大変化 {max(r['inverse_table_moment_change'] for r in n['refinement']):.10g}、χ²最小p {min(r['chi_p'] for r in n['cdf_tests']):.10g}。実表でRutherford呼出禁止を検査|{n['status']}|")
    else:lines.append(f"|N4|{n.get('reason','未実施')}|{n['status']}|")
    n=s['N5'];lines.append('|N5|'+('；'.join(f"{k}={n[k]['status']}" for k in ['K5','K5b','K7','K8','K9i','K9ii']) if 'K5' in n else '未実施')+'|'+n['status']+'|')
    for key in ['N6','N7']:
        c=s[key];text=(f"最大線量差/Dref {c['max_dose_difference_over_Dref']:.10g}（SEM {c['max_dose_difference_SEM_over_Dref']:.10g}）；R50/Rp差/R50 {c['range_difference_over_reference_R50']}；統計要求 {c['statistics_ok']}" if 'Dref' in c else c['reason'])
        lines.append(f"|{key}|{text}|{c['status']}|")
    lines+=['','図depth_2.pngは保存rawの重ね描きと差。N5未達の場合も表示するが、N6の合否を判定した図とは扱わない。','', '## 問いC','']
    if all(s[k]['status']=='pass' for k in ['N1','N2','N4','N5','N6']):
        lines.append('2 MeV・水・指定条件で、外部データを使わない解析式の有効DCS（既存の電子–電子の近似と明示Møllerは固定）で電子輸送の試作が成立。適用範囲は低Zで、今回確かめたのはHとOだけ。補正式そのものの物理的な妥当性を独立に検証したことを意味しない。')
    elif s['N6']['status']=='fail':lines.append('解析的な補正ではK6を満たせなかった。調整は行わず、以下の残差とdcslibとの差を報告する。')
    else:lines.append('前提検査の未達・未実施とN6の物理的不合格を区別する。N6は未判定であり、問いCにはまだ結論を出せない。N6/N7・残差低減率・EGS5に対するmfのR50/Rp差は、計画の停止規則により未評価。DCSと試作同士の補助指標は記録のみ。')
    lines+=['','## P1・P1bとの対比（2 MeV）','','|backend|最大線量差/Dref|R50差/R50|Rp差/R50|後方エネルギー割合|','|---|---|---|---|---|']
    curves=[('P1 Rutherford',p1),('P1b dcslib',dc['M6'])]
    if 'Dref' in s['N6']:curves.append(('P1c mf',s['N6']))
    for name,c in curves:lines.append(f"|{name}|{c['max_dose_difference_over_Dref']:.10g}|{c['range_difference_over_reference_R50'][0]:.10g}|{c['range_difference_over_reference_R50'][1]:.10g}|{c['prototype_back_energy_fraction']:.10g}|")
    if 'Dref' not in s['N6'] and (P/'mf/runs/base_2.npz').exists():
        raw=np.load(P/'mf/runs/base_2.npz');back=float(raw['ledger'][:,1].sum()/(float(raw['T'])*int(raw['N'])))
        lines.append(f'|P1c mf|未判定|未判定|未判定|{back:.10g}|')
    lines.append('\nP1の未達とP1bの結論は変更しない。基準EGS5の後方エネルギー割合は '+str(p1['EGS5_back_energy_fraction'])+'。')
    if 'reduction' in s:
        r=s['reduction'];lines.append(f"\n残差低減率 r={r['point']:.10g}、95%区間={r['CI95']}（1000回、seed={r['seed']}、共有EGS5 realization、評価mask固定）。P1b dcslib: r={dc['reduction']['point']:.10g}、95%区間={dc['reduction']['CI95']}。mf残差自体の95%区間={r['d_new_CI95']}。")
        c=s['N6'];lines.append(f"\nmf後方割合SEM={c['prototype_back_fraction_SEM']:.10g}、EGS5保守的SEM上限={c['EGS5_back_fraction_SEM_upper_bound']:.10g}。")
    if 'auxiliary' in s:
        a=s['auxiliary']['mf_minus_dcslib'];lines.append(f"\nmf−dcslib（共通EGS5 Drefで正規化）: 評価対象binの最大絶対差={a['max_difference_over_Dref']:.10g}、符号付き差={a['signed_difference_over_Dref']:.10g}、深さ={a['depth_cm']} cm。R50={a['reference_R50_cm']} cmより深い全binの最大差={a['beyond_R50_all_bins_max_over_Dref']:.10g}、同じ深さ域かつ評価対象binでは={a['beyond_R50_eligible_bins_max_over_Dref']:.10g}。差分自体は2試作の直接比較で、EGS5は共通尺度・評価maskだけに使用。有限標本の変動も含み、数値表の収束値から線量の数値誤差上限は算出していない。")
        lines+=['','無散乱確率（合否には使わない）:','','|backend|T MeV|K|Σ0|G1|−log p0 = KΣ0/G1|p0|','|---|---|---|---|---|---|---|']
        for r in s['auxiliary']['p0']:lines.append('|'+ '|'.join(str(r[k]) for k in ['backend','T','K','total','G1','negative_log_p0','p0'])+'|')
        lines.append('\np0=0は指数の浮動小数点underflowであり、Σ0の差だけから線量の差を推定しない。')
    lines+=['','## N3 散乱係数（cm⁻¹）','','|backend|T MeV|Σ0|G1|G2|後方半球|','|---|---|---|---|---|---|']
    for r in s['N3']['rows']:lines.append('|'+ '|'.join(str(r[k]) for k in ['backend','T','total_cm1','G1_cm1','G2_cm1','back_cm1'])+'|')
    lines.append('\n後方積分は必ずy=1で分割し[1,2]を別に積分。P1b M3は全角度Gauss点のy>1を足す方法で、記録値は書き換えていない。変種（核部分のみ、Bornのみ）は記録専用で輸送の判定には使っていない。')
    lines+=['','事前計算値の照合（合否に使わない）:','','|T|量|事前の丸め値|今回の比|相対差|10⁻³以内|','|---|---|---|---|---|---|']
    for r in checks:lines.append('|'+ '|'.join(str(r[k]) for k in ['T','quantity','preregistered','actual','relative','within_1e3'])+'|')
    lines+=['','## N5 詳細','']
    n=s['N5']
    if 'K5' in n:
        lines.append('K5: '+json.dumps(n['K5'],ensure_ascii=False))
        lines.append('\nK5b: 実際のmf輸送表で、既存の全境界・hinge・cutoff・parent復帰・明示failureのassertionを実行。結果のexit codeをN5に直接結び付けた。無条件passを採用しない。')
        for key in ['K7','K8']:
            for name,r in n[key]['rows'].items():lines.append(f"\n{key} {name}: dose差/Dref={r['max_dose_difference_over_Dref']}, R50差/R50={r['range_difference_over_reference_R50'][0]}, dose SEM/Dref={r['max_dose_difference_SEM_over_Dref']}, pass={r['pass']}。")
        lines.append('\nK9(i): '+json.dumps(n['K9i'],ensure_ascii=False))
        lines.append('\nK9(ii): '+json.dumps(n['K9ii'],ensure_ascii=False))
    lines+=['','## 計測','']
    if 'benchmark' in s:
        b=s['benchmark'];lines+=['P1bと同じ単一process、各条件2万履歴×3回の中央値。JIT除外。計測中に別の検証processを走らせない。表生成は別に計測し、検証表との全数値配列一致を確認。','','|T|f_E|K|scoring|µs/一次|区間/一次|GS/一次|Møller/一次|','|---|---|---|---|---|---|---|---|']
        for r in b['rows']:lines.append('|'+ '|'.join(str(v) for v in [r['T'],r['fe'],r['K'],r['scoring'],r['us_per_primary'],*r['counts_per_primary'][:3]])+'|')
        lines.append(f"\n表生成 {b['table_build_seconds']} s、load/初期化 {b['table_load_initialization_seconds']} s、表 {b['table_bytes']} bytes、stack {b['stack_allocated_bytes']} bytes。個別replicate時間・warmup・ledgerはbenchmark.json。")
    else:lines.append('未実施。再開コマンドは下記。')
    if 'K9i' in s['N5'] and s['N5']['K9i']['status']=='fail':
        failed=[r for r in s['N5']['K9i']['rows'] if not r['pass']]
        lines.append('\n確認した検査失敗: '+json.dumps(failed,ensure_ascii=False)+'。規定の統計要求を満たしても、点推定の許容幅超過は合格に置き換えない。原因は今回特定していない。SEMは有限標本の不確かさであり、閾値やseedを変更する根拠には使わない。')
        control=json.loads((ROOT/'docs/validation/mv/p1b/m5_dcslib.json').read_text())['K9i']['rows']
        matched=[r for r in control if any(r['path_cm']==x['path_cm'] and r['K']==x['K'] for x in failed)]
        lines.append('\n同条件P1b dcslibのK9(i)記録: '+json.dumps(matched,ensure_ascii=False)+'。この対照の値をmfの合否へ流用しない。既存の有限刻み近似と有限標本の変動が差へ寄与する可能性は仮説で、今回の数値だけから原因を確定しない。')

    lines+=['','## 確かめた事実','',
            '- 補正Rはθ=0、πと内部角度についてH/O×複数エネルギーで独立に式を書き下して単体照合。有限Tの電子についてR≥1−β²>0。P1のeta/Bはそのまま使用。',
            '- mfは専用のdensity/edges/quadratureを持ち、固有のエネルギー点は空集合。既存の全次数GS数値積分・率・単一CDFの処理を利用。輸送中と逐次参照は同じmf表を使う。',
            '- GS/輸送の数値的な検査と指定条件のEGS5比較は、完全なMott断面積、電子–電子の交換・反跳、分子・凝縮相効果を独立検証するものではない。',
            '- 低エネルギーの散乱係数差はN3に実測記録。数値積分誤差はN2、GS数値表の誤差はN4で別に評価。',
            '- P1/P1b追跡対象の開始時hash不変を確認。基準runを読み取りのみで使った。既存3 backendの配列完全一致を確認。','',
            '## 仮説と未完了','',
            '- mfとdcslibの低エネルギー散乱係数の差が深いbinの線量差へ寄与する可能性は仮説。エネルギー別の対照介入をしておらず、因果を確定しない。',
            '- 共通の損失・軟らかい電子–電子近似と明示Møllerの重複、EGS5基準のfit警告の寄与も今回だけでは分離していない。',
            '- 高Z、陽電子、水以外、光子連成、完全Mott、ELSEPAやGeant4データの取得・照合、本番経路の選定は範囲外。','']
    incomplete=[k for k in ['N1','N2','N4','N5','N6','N7'] if s[k]['status']=='not-evaluated']
    if 'benchmark' not in s:incomplete.append('T5計測')
    lines.append('対象範囲内の未完了: '+('、'.join(incomplete) if incomplete else 'なし')+'。検査失敗があれば上の判定をそのまま保持する。')
    lines+=['','経路選定の比較材料（推薦なし）: 解析式は外部DCSファイル不要でH/Oのみを今回検査。ELSEPAの自前生成は今回未実施、掲載条件は非営利限定で出力表への適用未確認。Geant4 dpwaデータは今回未取得、データファイルの利用条件は未確認。これらの条件は承認計画の前提事実6/7に基づく。','',
            '## 実行・再開・再現コマンド','',
            '```bash','.venv/bin/python -m prototypes.mv_electron.regression_p1c --before','.venv/bin/python -m prototypes.mv_electron.regression_p1c','.venv/bin/python prototypes/mv_electron/run_p1c.py','.venv/bin/python prototypes/mv_electron/run_p1c.py --benchmark-only','.venv/bin/python -m pytest tests/ -q','git diff --stat -- chatcarlo tests scripts examples docs/validation/mv/p0 docs/validation/mv/p1','.venv/bin/python -m pytest prototypes/mv_electron/tests -q','.venv/bin/python prototypes/mv_electron/run_checks_p1c.py --from-saved','.venv/bin/python prototypes/mv_electron/run_checks_p1c.py --verify-untouched','```','',
            'N5の固定エネルギーK9(i)は独立したprocessでも先行計算した（`fixed_checks()`）。この過程のwall timeは性能比較には使わず、T5を検証process終了後に単独実行する。',
            'regression --beforeは保存済みbaselineを上書きしない。GSはエネルギー行単位、N4は検査条件単位、輸送は完了run単位、K9(i)はreference/prototype配列単位、計測はreplicate単位で保存。中断時の実行中のrun/replicateだけ固定seedから再実行し、完了単位を再利用する。N1後に輸送コードと共通解析は変更していない。詳細値・統計・hashはsummary.jsonとmanifest.json。ログにも利用者の絶対パスを含めない。']
    if (P/'test_commands.json').exists():lines+=['','テスト実行結果:','```json',(P/'test_commands.json').read_text().strip(),'```']
    (P/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def plots(s):
    if not (P/'mf/runs/base_2.npz').exists():return
    import os
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/p1c-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .check_egs5 import parse
    from .analysis import dref
    ref=parse('base_2');q=ref['batches'].mean(0);z=ref['z'];D=dref(q)
    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True);axs[0].plot(z,q,color='k',label='EGS5 saved')
    for label,path in [('Rutherford',ROOT/'docs/validation/mv/p1/runs/base_2.npz'),('dcslib',ROOT/'docs/validation/mv/p1b/dcslib/runs/base_2.npz'),('mf',P/'mf/runs/base_2.npz')]:
        r=np.load(path)['batches'].mean(0);axs[0].plot(z,r,label=label);axs[1].plot(z,100*(r-q)/D,label=label)
    axs[0].set_ylabel('Dose [MeV cm²/g / primary]');axs[1].set_ylabel('Difference / EGS5 Dref [%]');axs[1].set_xlabel('Depth [cm]');axs[1].axhline(0,color='k',lw=.5)
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.suptitle('2 MeV electrons in water; '+('N6 judged' if 'Dref' in s['N6'] else 'N6 not evaluated (prerequisite gate)'));fig.tight_layout();fig.savefig(P/'depth_2.png',dpi=160);plt.close(fig)

def manifest():
    hashes=json.loads((P/'initial_hashes.json').read_text())
    inputs=['docs/validation/mv/p1/data/gs.npz',
            'docs/validation/mv/p1/runs/base_2.npz',
            'docs/validation/mv/p1b/m1_baseline.npz',
            'docs/validation/mv/p1b/dcslib/data/gs.npz',
            'docs/validation/mv/p1b/eedl/data/gs.npz',
            'docs/validation/mv/p1b/dcslib/runs/base_2.npz']
    for name in inputs:
        f=ROOT/name
        hashes[name]=hashlib.sha256(f.read_bytes()).hexdigest()
    for directory in [ROOT/'prototypes/mv_electron',P]:
        for f in sorted(directory.rglob('*')):
            if not f.is_file() or f==P/'manifest.json' or '__pycache__' in f.parts:continue
            hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    for f in (ROOT/'docs/egs5_crosscheck/egs5/data/dcslib').glob('eeldx*.tab'):
        if f.name in ['eeldx001.tab','eeldx008.tab']:hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    hashes['.gitignore']=hashlib.sha256((ROOT/'.gitignore').read_bytes()).hexdigest()
    plan=ROOT/'docs/ai/plans/2026-10-08-mv-p1c-mott-correction.md';hashes[str(plan.relative_to(ROOT))]=hashlib.sha256(plan.read_bytes()).hexdigest()
    (P/'manifest.json').write_text(json.dumps({'seed':20261007,'bootstrap_seed':20261008,'hashes':hashes,'note':'Repository-relative paths only. Ignored raw npz kept locally. EGS5/dcslib read-only; no redistribution of original dcslib. Manifest excludes itself/cache.'},indent=2)+'\n')
