#!/bin/zsh
# Claude only. All service calls have time limits in start.py.
set -eu
TASK_ROOT=${0:A:h:h:h:h:h}
cd "$TASK_ROOT"
.venv/bin/python -m prototypes.mv_electron.gpu.p8g2_cli.start "$@"
