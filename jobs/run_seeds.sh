#!/usr/bin/env bash
# Seed robustness of the liquidity headline (lam = 1, alpha = 0.9, liquidity 'base', cash unwind) with dual
# certificates for each new GDA and LSM-trained hedge; then dual certificates for the grid's alpha = 0.9 / 0.95 cells.
# Results for seeds other than 1234 go to results/seeds/. Usage: jobs/run_seeds.sh [seeds...]   (default: 2 3)
set -u
cd "$(dirname "$0")/.."
SEEDS="${*:-2 3}"
export X0="tag=reduced train.batch_log2=12 train.outer_iters=1500 threads=1 market.lam=1.0 train.alpha=0.9 liquidity.enabled=true costs.unwind=cash"
one_seed() {
  S=$1; X="$X0 seed=$S"
  {
    echo "== start $(date -Is) seed $S"
    python3 src/train.py --mode gda --set $X && python3 src/train.py --mode exploit --source gda --set $X
    for R in lsm maturity; do
      python3 src/train.py --mode fixed --rule $R --set $X && python3 src/train.py --mode exploit --source fixed-$R --set $X
    done
    python3 src/evaluate.py --mode headline --set $X
    for SRC in gda fixed-lsm; do
      python3 src/dual_value.py --ckpt checkpoints/${SRC}_lam1_a0p9_s${S}_heuristic_liqbase_uwcash_reduced.pt --alpha 0.9 \
        --threads 1 --name v5_${SRC}_liqbase_s$S
    done
    echo "== end $(date -Is) seed $S"
  } > results/logs/stdout_seed_s$S.txt 2>&1
}
grid_dual() {
  L=$1; A=$2
  python3 src/dual_value.py --ckpt checkpoints/gda_lam${L}_a${A}_s1234_heuristic_liqbase_uwcash_grid.pt --threads 1 \
    --name grid_lam${L}_a${A} > results/logs/stdout_dualvf_grid_lam${L}_a${A}.txt 2>&1
}
export -f one_seed grid_dual
printf '%s\n' $SEEDS | xargs -P 2 -I{} bash -c 'one_seed {}'
printf '1 0p9\n0p5 0p9\n0 0p9\n1 0p95\n0p5 0p95\n0 0p95\n' | xargs -P 2 -L 1 bash -c 'grid_dual "$0" "$1"'
python3 src/evaluate.py --mode md
date -Is > results/logs/seeds_done.txt
