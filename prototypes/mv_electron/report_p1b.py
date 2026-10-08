"""P1b report from measured saved data; facts and hypotheses kept separate."""
import hashlib,json
from pathlib import Path
import numpy as np
from .physics import ROOT
from .component_checks import OUT as P1
P=ROOT/'docs/validation/mv/p1b'

def report(s):
    lines=['# P1b RESULTS','', '計画: docs/ai/plans/2026-10-08-mv-p1b-elastic-dcs.md。結果から判定閾値・物理パラメータを変更していない。P1のRutherford K6 fail判定は維持。','',
    '基準: 2 MeV・水1 g/cm³・12 mm平板・0.1 mm scoring bin、10 MeVは60 mm・0.5 mm。各本計算100万一次電子、100 batch。f_E=K1_max=.05、seed=20261007。輸送状態機械、損失、採点、analysis.pyの定義を維持。EGS5はP1 USEGSD=1の保存runのみ使用し、再計算していない。','',
    'M1基準ソースはP1 manifestと一致した版をbaseline_code/へ変更前に保存。変更前runはその時点の未変更コードを実行（コピー内のROOTは相対配置に依存するため、コピーをその配置で直接importしない）。同じP1 GS表・同じ環境・引数で実行した。','',
    '|項目|値|判定|','|---|---|---|',f"|M1|10⁴履歴の全7保存配列の完全一致、P1の輸送/EGS比較JSON完全一致、p1全{s['M1']['p1_untouched']['file_count']}ファイルのhash不変|{s['M1']['status']}|"]
    for name,r in s['backends'].items():
        m2=r['M2'];rows=m2.get('rows',[])
        if name=='dcslib':
            vals=[a['independent_2MeV_relative'] for a in rows if a['T']==2]
            text=f"Gauss16→32最大相対変化 {max(a['doubling_max_relative'] for a in rows):.8g}；2 MeV H/O独立積分相対差 {vals}；格子点G1/G2ヘッダー最大相対差 {max(max(abs(v) for v in a['header_relative'][1:]) for a in rows if a['native_energy']):.8g}"
        else:text=f"全角度積分/23-526の最大相対差 {max(abs(a['total_relative']) for a in rows):.8g}；26-525積分最大誤差 {max(abs(a['angular_integral']-1) for a in rows):.8g}"
        lines.append(f"|M2 {name}|{text}|{m2['status']}|")
        m4=r['M4'];ref=m4.get('refinement',[])
        text=(f"20条件K3最大モーメント相対差 {max(max(abs(v) for v in a['relative']) for a in m4['K3_rows']):.8g}；CDF規格化最大誤差 {m4['normalization_max_error']:.8g}；次数/積分格子2倍のCDFモーメント最大変化 {max(a['moment_change'] for a in ref):.8g}；逆CDF表最大変化 {max(a['inverse_table_moment_change'] for a in ref):.8g}；χ² p {[a['chi_p'] for a in m4['cdf_tests']]}" if ref else m4.get('reason','未評価'))
        lines.append(f"|M4 {name}|{text}|{m4['status']}|")
        m5=r['M5'];text='；'.join(f"{k}={m5[k]['status']}" for k in ['K5','K5b','K7','K8','K9i','K9ii'] if k in m5)
        lines.append(f"|M5 {name}|{text or '未実施'}|{m5['status']}|")
        for key in ['M6','M7']:
            c=r[key]
            text=(f"最大線量差/Dref {c['max_dose_difference_over_Dref']:.10g}（SEM {c['max_dose_difference_SEM_over_Dref']:.8g}）；R50/Rp差/R50 {c['range_difference_over_reference_R50']}；統計要求 {c['statistics_ok']}" if 'Dref' in c else c.get('reason','未実施'))
            lines.append(f"|{key} {name}|{text}|{c['status']}|")
    lines+=['','M3（合否に使わない）:','', '|backend|T MeV|Σ0 cm⁻¹|G1 cm⁻¹|G2 cm⁻¹|後方半球 cm⁻¹|','|---|---|---|---|---|---|']
    for r in s['M3']['rows']:lines.append('|'+ '|'.join(str(r[k]) for k in ['backend','T','total_cm-1','G1_cm-1','G2_cm-1','back_cm-1'])+'|')
    gscheck=json.loads((P/'p1_reanalysis/gs_value_verification.json').read_text())
    lines.append('\nM1補足: P1 GS再検査の報告値・判定は同じ。一部SEMに1 ULP相当の丸め差（最大絶対差 '+str(gscheck['SEM_max_absolute_difference'])+'）がある。M1の輸送配列・履歴収支はビット単位で一致。')
    lines+=['','## 問いA（dcslibだけの結論）','']
    a=s['backends']['dcslib'];red=a['reduction']
    if 'point' in red:
        lines.append(f"{red['status']}。低減率r={red['point']:.10g}、95%区間={red['CI95']}。")
        lines.append('K6の残差の主因はDCSの差と判断する。' if red['CI95'][0]>=.5 else 'DCSの差だけで残差の大部分を説明できたとは確認できない。これだけでDCS以外の要因が残っているとは判断しない。')
        if red['majority_possible']:lines.append('大部分が消えた可能性を排除できない。')
        lines.append(f"d_newのbootstrap95%区間={red['d_new_CI95']}。区間はmax-bin推定の標本変動であり、残差原因の分解を証明しない。数値誤差を超えた残差の原因分解はこの計画では行わない。")
    else:lines.append('DCSの差だけで残差の大部分を説明できたとは確認できない。前提ゲート未達によりM6・低減率は未評価。DCS以外の要因が残っているとは判断しない。')
    lines+=['','## 問いB（EEDLだけの結論）','']
    b=s['backends']['eedl'];red=b['reduction']
    if 'point' in red:
        lines.append(f"{red['status']}。EEDLへの差し替えで残差が"+('低減した。' if red['CI95'][0]>0 else '低減したとは確認できない。')+f" r={red['point']:.10g}、95%区間={red['CI95']}。")
        if red['majority_possible']:lines.append('大部分が消えた可能性を排除できない。')
    else:lines.append('EEDLへの差し替えで残差が低減したとは確認できない。前提ゲート未達によりM6・低減率は未評価。')
    for name,r in s['backends'].items():
        established=all(r[key]['status']=='pass' for key in ['M4','M5','M6'])
        lines.append(f"\n{name}: "+(f"2 MeV・水・指定条件で電子輸送の試作が成立（DCSは{'EGS5と同じ、研究専用の対照' if name=='dcslib' else 'EEDL'}）。" if established else '2 MeV・水・指定条件での試作成立を確認できない。'))
    baseline=json.loads((P1/'egs5_checks.json').read_text())['K6'];lines+=['','## P1との補助指標対比','', '|backend|最大線量差/Dref|R50差/R50|Rp差/R50|後方エネルギー割合|EGS5との差|','|---|---|---|---|---|---|']
    for name,c in [('P1 Rutherford',baseline)]+[(name,r['M6']) for name,r in s['backends'].items() if 'Dref' in r['M6']]:
        f=c['prototype_back_energy_fraction'];lines.append(f"|{name}|{c['max_dose_difference_over_Dref']}|{c['range_difference_over_reference_R50'][0]}|{c['range_difference_over_reference_R50'][1]}|{f}|{f-c['EGS5_back_energy_fraction']}|")
    for name,r in s['backends'].items():
        if 'Dref' in r['M6']:
            c=r['M6'];lines.append(f"\n{name}: 後方割合SEM prototype={c['prototype_back_fraction_SEM']}、EGS5保守的SEM上限={c['EGS5_back_fraction_SEM_upper_bound']}（各一次履歴割合0≤f≤1からsqrt(mean*(1-mean)/(N-1))）。")
        if 'refinement' in r['M4']:
            a=next(x for x in r['M4']['refinement'] if x['T']==2 and x['K']==.05);lines.append(f"\n{name}: 2 MeV・sΣ1=.05でCDF最大変化={a['cdf_max_change']}、後方半球積分変化={a['back_change']}。Rutherford呼び出し禁止は実際のbackend表でmode0/mode1/固定エネルギー逐次参照をcompileして検査。")
    check=json.loads((P/'eedl_independent_verification.json').read_text())
    oxygen=check['atoms']['8'];critical=[v for v in oxygen['checks'] if v['T_MeV'] in [1.,2.]]
    maxpdf=max(v['PDF_max_relative_difference'] for a in check['atoms'].values() for v in a['checks'])
    maxmoment=max(abs(x) for a in check['atoms'].values() for v in a['checks'] for x in v['production_relative_difference'])
    lines+=['','## EEDL不一致：格子・補間の事実と実装検証の範囲','',
    '(a) 確認したファイルの事実: 酸素ZA008000のMF26/MT525には16エネルギー点があり、256 keVの直後が10 MeVである。実際の格子 [MeV] は '+str(oxygen['angular_energy_grid_MeV'])+'。水素も同じ格子を持つ。MF26 TAB2の指定は '+str(oxygen['MF26_regions'])+'（INT=2、入射エネルギーに対して線形）、角度LISTはLANG=12（cosθに対して線形）。MF23/525・526もINT=2。これらをそのまま使い、異なる角度格子を和集合上で線形評価して条件付き確率を補間した。',
    '1/2 MeVの補間区間・上側点の重みは '+str([(v['T_MeV'],v['bracket_MeV'],v['upper_energy_weight']) for v in critical])+'。従って2 MeVの角度分布はファイルに実在する精密な部分波評価点ではない。今回の大きな線量不一致は、この格子と指定補間を保った入力DCSで計算した結果である。格子や補間則を合わせるために変更していない。',
    '(b) 確認できた実装検証: 別の固定幅読取器（NumPy genfromtxt）でH/OのMF23/525・526の全断面積表値、およびMF26/525の全角度表値・角度エネルギー格子を読み、製品パーサの配列との完全一致を確認。製品補間器を使わないスカラー線形補間で0.02/0.1/0.256/1/2/10 MeVの角度確率を照合した最大相対差は '+str(maxpdf)+'。大角度の線形区間を多項式として積分し、前方枝を解析積分する別経路による元素別Σ0/G1/G2と製品の数値積分の最大相対差は '+str(maxmoment)+'。記録はeedl_independent_verification.json、再現コマンドは `.venv/bin/python -m prototypes.mv_electron.check_eedl_independent`。',
    '取得ハッシュ、M2の非負性・規格化・枝別積分、参照表の中点自己収束、M4の実輸送用CDF・全次数GS検査、M5の同じbackendによる逐次参照と損失・境界・刻み検査も通過。初期の粗い参照表の実装誤差は修正して旧試行を不採用にした。',
    '排除できた範囲は、照合した全表値・エネルギー格子の読取値の相違、および検査エネルギー・角度での指定線形補間、単位、元素別モーメント積分の相違である。全コード経路の無誤り、補間の物理的妥当性、共通の輸送・損失近似やEGS5基準の誤差までは証明していない。この別経路照合も独立した確立済み輸送コードとの検証ではない。',
    '仮説: 広い256 keV–10 MeV区間で条件付き角度分布を線形接続したことが大きな散乱係数・線量差へ寄与している可能性がある。ただし、別の補間則・高密度の評価データによる対照実験を行っていないため、その接続だけが線量不一致の全原因であるとは確定しない。']
    lines+=['','## M5の計測値','']
    for name,r in s['backends'].items():
        c=r['M5'];lines.append(f'### {name}')
        if 'K5' not in c:lines.append('前提ゲート未達。');continue
        lines.append('K5: '+json.dumps(c['K5'],ensure_ascii=False))
        for key in ['K7','K8']:
            for n,v in c[key]['rows'].items():lines.append(f"{key} {n}: 線量差={v['max_dose_difference_over_Dref']}, R50差={v['range_difference_over_reference_R50'][0]}, SEM={v['max_dose_difference_SEM_over_Dref']}, statistics_ok={v['statistics_ok']}, pass={v['pass']}。")
        lines.append('K9(i): '+json.dumps(c['K9i'],ensure_ascii=False));lines.append('K9(ii): '+json.dumps(c['K9ii'],ensure_ascii=False))
    lines+=['','## 計測（T6）','', 'P1と同じ単一process、各条件2万一次×3回の中央値、warmup/JIT除外。全履歴ledgerと100 batch tallyを含む。表生成は輸送検証後に単独で再計測し、検証に使った表の全数値配列と完全一致を確認。表生成は生成時間として別に記録し、輸送単価へ隠していない。','', '|backend|T|f_E|K|scoring|µs/一次|区間/一次|GS/一次|Møller/一次|','|---|---|---|---|---|---|---|---|---|']
    for name,r in s['backends'].items():
        bm=r.get('benchmark')
        if not bm:continue
        for v in bm['rows']:
            counts=v['counts_per_primary'] if v['counts_per_primary'] is not None else ['未保存']*3
            lines.append(f"|{name}|{v['T']}|{v['fe']}|{v['K']}|{v['scoring']}|{v['us_per_primary']}|{counts[0]}|{counts[1]}|{counts[2]}|")
        lines.append(f"\n{name}: 表生成={bm['table_build_seconds']} s、表load/初期化={bm['table_load_initialization_seconds']} s、表={bm['table_bytes']} bytes、stack={bm['stack_allocated_bytes']} bytes。JITは各条件のwarmup_seconds_excludedに別記。")
        if 'interruption_note' in bm:lines.append('\n計測中断の制限: EEDLの完了済み13条件は3回×2万履歴の中央値をstdoutログから回収。個別3回の時間、counts、warmup、ledgerの計測補助値はプロセスメモリだけにあり未保存。残り3条件を同じ方法で測定しcheckpointへ保存。表生成時間は保存済みbenchmark_gs.npzから回収、load/初期化は再開時に再測定。未保存値を推測で補っていない。再開コマンド: `.venv/bin/python prototypes/mv_electron/run_p1b.py --backend eedl --benchmark-only --resume-benchmark-log docs/validation/mv/p1b/benchmark_final.log`。')
    lines+=['','## 確かめた事実','',
    '- dcslibは元配布ファイルを読み取りのみで使い、コピーしていない。H/O別に(Z+1)/Zを掛け、P1数密度を化学量論比へ換算して合成した後、logエネルギー→RMU方向のlog DCS自然三次スプラインを適用。ヘッダーへの再規格化はしない。',
    '- EEDL原子データのサイズ・SHA256は計画と一致。ENDF指定INT=2、LANG=12を使用。mu_c=.999999は表から読んだ。枝別積分・継ぎ目の両側値・前方G1/G2はm2_eedl.jsonに全検査点を保存し、継ぎ目の連続性を強制していない。',
    '- EEDLライセンス本文の再配布許諾と帰属条件を確認し、data/ATTRIBUTION.mdに出典と無変更の表示を付けた。遮蔽式の係数はEEDLが引用するSeltzer式を載せたLANL原資料から確認し、P1のetaを流用していない。原EEDL報告が引用するSeltzer原著のp.160式7.8を直接読み、MCNP5 manual p.2-77の再掲式とも一致した。出典PDFのhashはdata/reference_provenance.json。',
    '- GS全次数の係数を選択した有効DCSから数値積分。単一散乱項p0*s*Fを級数から厳密に分離して明示する同一GS展開により、入力DCSの細かい構造を有限級数で切り捨てる振動を抑えた。無散乱確率は同じbackendのΣ0とG1を使用。EEDLではnative節点・枝境界の左右極限もCDF積分格子に含めた。',
    '- 初期参照表は49点GS格子をrate/単一CDFにも流用し、実表ではなく直接生成CDFを単一散乱検査した不足があった。単体検査で粗いenergy補間と確率尾部の誤差を確認し、数値参照表の収束と実表χ²検査を追加して再計算した。旧試行はinitial_numerical_trials/に保存し、M6/M7判定には使用しない。',
    '- 初期の単一散乱項未分離GS生成はdcslibで負密度、EEDLでは枝境界を含まないCDF積分で規格化不足となり不採用。数値表の作り方を単体で調べ、定数・補間指定・刻み・倍率・cutoffを変えずに修正した。',
    '- rates/単一散乱CDF/GS CDFは同じbackendから生成。rateと単一CDFの参照格子はnativeエネルギー点を含め、入力DCSに対する検査中点の相対差1e-4で数値収束させる。確率端は浮動小数点epsから解像度を決める。これは補間則/物理係数の変更でもM4の閾値変更でもない。M4は実輸送用の単一表を5エネルギーで検査し、期待度数だけを使って100未満のχ²binを併合する。有限GS energy/確率格子の誤差をM4とK8で検査。','',
    '出典とライセンス: [EPICS2025](https://nuclear.llnl.gov/EPICS/)、[CC BY 4.0本文](https://nuclear.llnl.gov/EPICS/CC%20Attribution%204.0%20Intl%20Public%20License.pdf)、[ENDF-102](https://nds.iaea.org/exfor/x4guide/manuals/endf-manual.pdf)、[Seltzer式を記載するLANL原資料](https://mcnp.lanl.gov/pdf_files/TechReport_2003_LANL_LA-UR-03-1987Revised212008_SweezyBoothEtAl.pdf)。','',
    '## 仮説・未解決事項','',
    '- DCS以外の輸送近似やEGS5の電子fit警告が残差へ寄与する可能性は、今回の介入だけでは因果分解していない。',
    '- EEDL 1/2 MeVは角度分布の補間点であり、精密な部分波データとは呼ばない。元DCSと異なるエネルギー接続の影響があり得るが、合わせるために補間規則を選び直していない。',
    '- 分子・凝縮相、明示Møllerと軟らかい電子–電子補正の重複はP1から固定。水以外、Fano、光子連成、実際の放射光子輸送、最終データ選定は範囲外。',
    '- dcslib由来の表はローカルのみで.gitignoreに指定。dcslibは本番/公開用データ経路にはならない。',
    '- 本計算・診断・計測の詳細未丸め値はsummary.json、m2/m4/m5 JSON、comparison JSON、benchmark JSON、raw npzを参照。','',
    '## 再現・テスト','',
    '計算: `.venv/bin/python prototypes/mv_electron/run_p1b.py --backend both`。既存raw runは再利用。性能計測は他の計算を終了後に `.venv/bin/python prototypes/mv_electron/run_p1b.py --backend both --benchmark-only`。','',
    '指定最終チェック（実行結果はtest_commands.json）:','',
    '```bash','.venv/bin/python -m pytest tests/ -q','git diff --stat -- chatcarlo tests scripts examples docs/validation/mv/p0','.venv/bin/python -m pytest prototypes/mv_electron/tests -q','.venv/bin/python prototypes/mv_electron/run_checks_p1b.py --from-saved','.venv/bin/python prototypes/mv_electron/run_checks_p1b.py --verify-p1-untouched','```','',
    '開始前から存在する計画/P1/prototypesの未commit変更を保持。今回の変更は許可されたprototype/backend関連とp1bのみ。commitなし。']
    if (P/'test_commands.json').exists():lines.append('\n'+json.dumps(json.loads((P/'test_commands.json').read_text()),ensure_ascii=False,indent=2))
    (P/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def plots(s,data_root=None,energies=(2,10)):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/p1b-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .check_egs5 import parse
    from .analysis import dref
    root=P if data_root is None else Path(data_root)
    for T in energies:
        ref=parse(f'base_{T}');rq=ref['batches'].mean(0);D=dref(rq)
        fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
        for col in [0,1]:axes[0,col].plot(ref['z'],rq,label='EGS5 saved USEGSD=1',color='k')
        for name,path in [('P1 Rutherford',P1/'runs'/f'base_{T}.npz')]+[(name,root/name/'runs'/f'base_{T}.npz') for name in s['backends']]:
            if not path.exists():continue
            a=np.load(path);q=a['batches'].mean(0)
            for col in ([0] if name=='eedl' else [0,1]):
                axes[0,col].plot(ref['z'],q,label=name)
                axes[1,col].plot(ref['z'],100*(q-rq)/D,label=name)
        axes[0,0].set_title('All three DCS providers');axes[0,1].set_title('Question A control and P1 baseline')
        for col in [0,1]:
            axes[0,col].set_ylabel('MeV cm²/g per primary');axes[1,col].set_ylabel('(prototype - EGS5) / Dref [%]');axes[1,col].set_xlabel('depth [cm]')
            axes[1,col].axhline(0,color='k',lw=.5)
        for ax in axes.flat:ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f'{T} MeV water; saved mean curves (acceptance in RESULTS.md)');fig.tight_layout();fig.savefig(P/f'depth_{T}.png',dpi=160);plt.close(fig)

def manifest():
    hashes={}
    for directory in [ROOT/'prototypes/mv_electron',P]:
        for f in sorted(directory.rglob('*')):
            if not f.is_file() or f==P/'manifest.json' or any(x in f.parts for x in ['__pycache__','.pytest_cache']):continue
            hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    hashes.update(json.loads((P/'p1_initial_hashes.json').read_text()))
    egs=ROOT/'docs/egs5_crosscheck/egs5'
    for f in list((egs/'data/dcslib').glob('*'))+[egs/'pegs'/f for f in ['elastino.f','elinit.f','dcstab.f','dcsel.f','gscoef.f','inigrd.f','spline.f']]:
        if f.is_file():hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    plan=ROOT/'docs/ai/plans/2026-10-08-mv-p1b-elastic-dcs.md';hashes[str(plan.relative_to(ROOT))]=hashlib.sha256(plan.read_bytes()).hexdigest()
    (P/'manifest.json').write_text(json.dumps({'seed':20261007,'bootstrap_seed':20261008,'hashes':hashes,'note':'All input/source/saved output files including ignored local EGS5-derived tables; manifest itself excluded to avoid self-reference. Cache bytecode excluded. p1 files verified against initial snapshot. dcslib files are read-only originals, not redistributed.'},indent=2)+'\n')
