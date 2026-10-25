# P9c の図と数値（前立腺の形の照射野、シード 58 本のあり／なし）

チェックポイント（`~/.cache/chatcarlo/p9c2/prod_prostate*/prostate.npz`、git の外）から図と数値を作る。
`combined.py` が選んだ計算のモーメントを足し合わせる（バッチの大きさが同じなので和で合算できる。各計算は 99 バッチ以上・0 から連番・種がすべて違うことを検査）。

```bash
export P9C_RUNS=prod_prostate,prod_prostate2,prod_prostate3,prod_prostate4,prod_prostate5,prod_prostate6   # 600 バッチ
export P9C_TAG=_600 P9C_LABEL='各側 3.0×10⁸ 光子（600 バッチ）' P9C_OUT=runs/prostate_seed_7field
for f in make_slices make_profiles make_radial make_05mm make_top3; do PYTHONPATH=. .venv/bin/python prototypes/mv_electron/p9c_figures/$f.py; done
```

- `make_radial.py`: シードの軸からの距離ごとの比（58 本の平均、シードの長さの範囲）。標準出力に表の値。
- `make_top3.py`: 資料用（シードの長さで平均した横断面のあり・なし・比）。
- 誤差は、セルを独立とみなした値（セル間の相関は保存していない）。`make_profiles`・`make_05mm` の帯は相関 1 の上限も描く。
