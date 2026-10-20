#!/usr/bin/env bash
# One dual-bound run: jobs/dual_job.sh "<name>|<dual.py args>"   (see dual_jobs.txt)
set -u
cd "$(dirname "$0")/.."
NAME="${1%%|*}"; ARGS="${1#*|}"
LOG=results/logs/stdout_dual_${NAME}.txt
{
  echo "== start $(date -Is) $NAME"
  python3 src/dual.py $ARGS --name "$NAME" --batch-log2 12 --threads 1
  echo "== end $(date -Is) status=$?"
} > "$LOG" 2>&1
