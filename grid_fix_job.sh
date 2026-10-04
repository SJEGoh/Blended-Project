#!/usr/bin/env bash
# Diagnostic re-run of one grid cell with adversary restarts (train.adv_restart_every > 0), the proposed
# fix for adversary collapse. Writes under tag=gridrs, so the main grid's checkpoints are untouched.
# Usage: grid_fix_job.sh <lam> <alpha>
set -u
cd "$(dirname "$0")"
L=$1; A=$2
X="tag=gridrs train.batch_log2=13 train.outer_iters=600 threads=1 liquidity.enabled=true costs.unwind=cash \
market.lam=$L train.alpha=$A train.adv_restart_every=150 train.adv_restart_steps=300"
LOG=results/logs/stdout_gridrs_L${L}_A${A}.txt
{
  echo "== start $(date -Is) lam=$L alpha=$A (adversary restarts)"
  python3 train.py --mode gda --set $X && \
  python3 train.py --mode exploit --source gda --set $X
  echo "== end $(date -Is) status=$?"
} > "$LOG" 2>&1
