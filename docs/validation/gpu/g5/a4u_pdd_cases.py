# A4U_PDD_PREREG.md 追補 1 の表の値だけを置く。
# 60 keV の CPU_N/CPU_seed は **実行していない**: 追補 2（ユーザー決定 2026-11-05）で
# 60 keV は 1.0 cm bin・既存データで判定したため、この条件での計算は不要になった。
# EGS5 の pdd_60kev は回したが 1.0 cm の判定には使わず、**ユーザー決定 2026-11-05 で消した**
# （~/.Trash へ退避。空にするのはユーザー）。CASES の 60 keV の行を残すのは追補 1 の登録値の記録
# としてで、この条件を使うなら EGS5 から回し直す必要がある。
CASES={20:dict(label='pdd_20kev',EGS5_N=52727700,EGS5_seed=20201,CPU_N=47953300,CPU_seed=20211),
       60:dict(label='pdd_60kev',EGS5_N=150767400,EGS5_seed=60201,CPU_N=146722900,CPU_seed=60211)}
