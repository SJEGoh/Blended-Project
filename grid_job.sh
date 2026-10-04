#!/usr/bin/env bash
# One grid cell: adversarial (GDA) training + fresh-adversary exploitability attack.
# Usage: grid_job.sh <lam> <alpha>     (run from the repo root; see run_grid.sh)
set -u
cd "$(dirname "$0")"
L=$1; A=$2
COMMON="tag=grid train.batch_log2=12 train.outer_iters=600 threads=1 liquidity.enabled=true costs.unwind=cash"
EXTRA=""
if [ "$A" = "0.99" ]; then EXTRA="train.batch_log2=13"; fi   # >= 82 tail samples per batch at alpha = 0.99
X="$COMMON market.lam=$L train.alpha=$A $EXTRA"
LOG=results/logs/stdout_grid_L${L}_A${A}.txt
{
  echo "== start $(date -Is) lam=$L alpha=$A"
  python3 train.py --mode gda --set $X && \
  python3 train.py --mode exploit --source gda --set $X
  echo "== end $(date -Is) status=$?"
} > "$LOG" 2>&1
