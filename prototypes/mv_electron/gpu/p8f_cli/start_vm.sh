#!/bin/zsh
# P8f on a Colab T4 through the Colab CLI (used 2026-10-21; see docs/validation/mv/p8f/HANDOFF.md).
# Creates session "p8f", uploads the export (files over 150 MB in 50 MB parts; whole-file
# uploads of that size failed), re-uploads results already pulled back (resume), verifies every
# manifest hash on the VM and launches the driver with nohup. Run watch.sh right after.
set -u
HERE=${0:A:h}
EXPORT=${P8F_EXPORT:-$HOME/.cache/chatcarlo/p8f/export}
WORK=${P8F_CLI_WORK:-$HOME/.cache/chatcarlo/p8f/cli}
mkdir -p $WORK/parts $WORK/results
cd /tmp && colab new -s p8f --gpu T4 | tail -1
printf 'import os\nfor d in ["/content/p8f","/content/parts","/content/p8f/results"]:os.makedirs(d,exist_ok=True)\n' | colab exec -s p8f >/dev/null
cd $EXPORT
for f in *; do
  if [ $(stat -f%z "$f") -gt 150000000 ]; then split -b 50m "$f" "$WORK/parts/$f.part."; continue; fi
  colab upload -s p8f "$f" "/content/p8f/$f" >/dev/null 2>&1 || colab upload -s p8f "$f" "/content/p8f/$f" >/dev/null 2>&1 || echo FAIL $f
done
cd $WORK/parts && for p in *(N); do colab upload -s p8f "$p" "/content/parts/$p" >/dev/null 2>&1 || echo FAIL $p; done
cd $WORK/results && for f in $(find . -type f); do colab upload -s p8f "$f" "/content/p8f/results/${f#./}" >/dev/null 2>&1 || echo FAIL $f; done
colab exec -s p8f -f $HERE/assemble.py | tail -3
