#!/bin/zsh
# 5 段階を Colab で 1 本ずつ。各段階の標準出力は n<N>/run_stdout.txt
cd "$(git rev-parse --show-toplevel)"
D=runs/xray_portrait_noise_20261024
i=0
for N in 1e4 1e5 1e6 1e7 1e8; do
  S=$((20261024+i)); i=$((i+1))
  echo "$(date '+%F %T') start $N seed $S" >> $D/run_all.log
  .venv/bin/python -m chatcarlo run $D/scene.yaml --engine gpu --gpu-device colab -n $N --seed $S --run-dir $D/n$N > $D/n$N/run_stdout.txt 2>&1
  rc=$?  # 2026-10-24 の実行時は $? をコマンド置換の後に読んでいて、記録の exit は常に 0（ステージ B 所見 3）
  echo "$(date '+%F %T') end $N exit $rc" >> $D/run_all.log
done
echo "$(date '+%F %T') all done" >> $D/run_all.log
