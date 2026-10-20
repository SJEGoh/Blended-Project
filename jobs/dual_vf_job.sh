#!/usr/bin/env bash
# One value-function dual run: jobs/dual_vf_job.sh "<name>|<dual_value.py args>"   (see dual_jobs_vf.txt)
set -u
cd "$(dirname "$0")/.."
NAME="${1%%|*}"; ARGS="${1#*|}"
LOG=results/logs/stdout_dualvf_${NAME}.txt
{
  echo "== start $(date -Is) $NAME"
  python3 src/dual_value.py $ARGS --name "$NAME" --threads 1
  echo "== end $(date -Is) status=$?"
} > "$LOG" 2>&1
