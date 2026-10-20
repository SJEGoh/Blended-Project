#!/usr/bin/env bash
# LSM-trained hedges (and a fresh attacker against each) at the grid's other alpha levels, lam = 1, liquidity 'base',
# cash unwind. Mirrors grid_job.sh: batch 2^12, 600 outer iterations (2^13 at alpha = 0.99), tag=grid, so each
# hedge sits beside the GDA grid hedge at the same alpha. Usage: jobs/run_alpha_lsm.sh [alphas...]  (default: 0 0.5 0.95 0.99)
set -u
cd "$(dirname "$0")/.."
ALPHAS="${*:-0 0.5 0.95 0.99}"
one_alpha() {
  A=$1
  X="tag=grid train.batch_log2=12 train.outer_iters=600 threads=1 liquidity.enabled=true costs.unwind=cash market.lam=1.0 train.alpha=$A"
  if [ "$A" = "0.99" ]; then X="$X train.batch_log2=13"; fi
  {
    echo "== start $(date -Is) alpha=$A"
    python3 src/train.py --mode fixed --rule lsm --set $X && \
    python3 src/train.py --mode exploit --source fixed-lsm --set $X
    echo "== end $(date -Is) status=$?"
  } > results/logs/stdout_alpha_lsm_A${A}.txt 2>&1
}
export -f one_alpha
printf '%s\n' $ALPHAS | xargs -P 2 -I{} bash -c 'one_alpha {}'
date -Is > results/logs/alpha_lsm_done.txt
