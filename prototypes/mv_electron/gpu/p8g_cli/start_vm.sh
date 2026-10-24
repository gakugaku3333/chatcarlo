#!/bin/zsh
# P8g-1 gate 2 + timing on a Colab T4 (same pattern as p8f_cli): upload the export (>150 MB in 50 MB parts),
# reassemble and hash-check on the VM, launch p8g_colab with nohup. Then run watch.sh.
set -u
HERE=${0:A:h}; EXPORT=$HOME/.cache/chatcarlo/p8g1/export; WORK=$HOME/.cache/chatcarlo/p8g1/cli
mkdir -p $WORK/parts
cd /tmp && colab new -s p8g --gpu T4 | tail -1
printf 'import os\nfor d in ["/content/p8g","/content/parts","/content/p8g/results"]:os.makedirs(d,exist_ok=True)\n' | colab exec -s p8g >/dev/null
cd $EXPORT
for f in *; do
  if [ $(stat -f%z "$f") -gt 150000000 ]; then [ -n "$(ls $WORK/parts/$f.part.* 2>/dev/null)" ] || split -b 50m "$f" "$WORK/parts/$f.part."; continue; fi
  colab upload -s p8g "$f" "/content/p8g/$f" >/dev/null 2>&1 || colab upload -s p8g "$f" "/content/p8g/$f" >/dev/null 2>&1 || echo FAIL $f
done
cd $WORK/parts && for p in *; do colab upload -s p8g "$p" "/content/parts/$p" >/dev/null 2>&1 || echo FAIL $p; done
colab exec -s p8g -f $HERE/assemble.py | tail -4
