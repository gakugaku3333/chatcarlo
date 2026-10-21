#!/bin/zsh
# Poll every 20 minutes (user decision 2026-10-21; 4 minutes held for the 2.5 h P8f run, 20 is untested) with colab exec and pull finished result files into $P8F_CLI_WORK/results.
# The first P8f VM, left without polling, was gone after about 40 minutes (cause inferred: idle
# kernel; n=1). Stops when the driver ends or the session disappears; then pull, colab stop.
HERE=${0:A:h}
WORK=${P8F_CLI_WORK:-$HOME/.cache/chatcarlo/p8f/cli}
mkdir -p $WORK/results
while true; do
  o=$(colab exec -s p8f -f $HERE/listres.py 2>&1)
  if echo "$o" | grep -q "not found"; then echo "SESSION LOST $(date)"; break; fi
  for f in $(echo "$o" | awk '/^FILE/{print $2}'); do
    [ -f "$WORK/results/$f" ] || { mkdir -p "$WORK/results/$(dirname $f)"; colab download -s p8f "/content/p8f/results/$f" "$WORK/results/$f" >/dev/null 2>&1; }
  done
  echo "$(date +%H:%M) $(ls $WORK/results/gate3 2>/dev/null | grep -c npz) gate3, $(ls $WORK/results/production 2>/dev/null | grep -c npz) production" > $WORK/watch_status.txt
  if echo "$o" | grep -q "STATE DEAD"; then echo "$o" | tail -4; echo "DRIVER ENDED $(date)"; break; fi
  sleep 1200
done
