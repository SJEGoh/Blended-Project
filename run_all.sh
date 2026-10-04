#!/usr/bin/env bash
# End-to-end pipeline. Stages are added as the build order progresses:
#   [x] 1  simulator / benchmarks / unit tests
#   [x] 2  stopper only, zero hedge, alpha = 0  vs LSM (and CRR when lam = 0)
#   [x] 3  hedger + GDA at alpha = 0.9 (and 0), exploitability   (reduced compute profile below)
#   [x] 4  LSM-fixed-stopper (headline) experiment at lam = 1, alpha = 0.9 for c = 0, c = 0.005, c = 0.005 with
#          cash-settled unwind cost, and liquidity 'base' + cash unwind; lam x alpha grid under liquidity 'base'
#          + cash unwind (run_grid.sh), plus the alpha = 0.99 column re-run with adversary restarts (grid_fix_job.sh)
set -euo pipefail
cd "$(dirname "$0")"
LAMS="0 0.5 1.0"
# Compute profile for GDA / exploitability. Spec: batch 2^14, 3000 outer iterations (set PROFILE="").
PROFILE="${PROFILE-tag=reduced train.batch_log2=12 train.outer_iters=1500}"

echo "== unit tests =="
mkdir -p results
python -m pytest -q tests | tee results/pytest_summary.txt

echo "== test sets + benchmarks =="
for L in $LAMS; do python sim.py --set market.lam=$L; done
python benchmarks.py --lams $LAMS

echo "== step 2: stopper only, zero hedge, alpha = 0 =="
# As specified (train.stopper_init=random) the stopper collapses to never-exercise; see results/RESULTS.md.
# The runs below use the heuristic warm start (adopted).
for L in $LAMS; do
  python train.py --mode stopper_only --set market.lam=$L train.alpha=0 train.stopper_init=heuristic
done
python evaluate.py --mode step2 --lams $LAMS --init heuristic

echo "== step 3: GDA + exploitability (lam = 1) =="
for A in 0.9 0; do
  python train.py --mode gda     --set $PROFILE market.lam=1.0 train.alpha=$A
  python train.py --mode exploit --set $PROFILE market.lam=1.0 train.alpha=$A
done
python evaluate.py --mode step3 --alphas 0.9 0 --set $PROFILE market.lam=1.0

echo "== headline: hedgers trained against fixed rules vs the adversarial hedger (lam = 1, alpha = 0.9) =="
for SETTING in "0 none" "0.005 none" "0.005 cash"; do
  set -- $SETTING; C=$1; UW=$2
  X="$PROFILE market.lam=1.0 train.alpha=0.9 costs.c=$C costs.unwind=$UW"
  if [ "$C" != "0" ]; then   # the c = 0 GDA run and its attack are produced in step 3
    python train.py --mode gda     --set $X
    python train.py --mode exploit --source gda --set $X
  fi
  for R in lsm maturity; do
    python train.py --mode fixed   --rule $R          --set $X
    python train.py --mode exploit --source fixed-$R  --set $X
  done
  python evaluate.py --mode headline --set $X
done

echo "== liquidity extension: sanity checks (no training) =="
python liqcheck.py --out liqcheck_base.json --set market.lam=1.0 train.alpha=0.9
python liqcheck.py --out liqcheck_mild.json --set market.lam=1.0 train.alpha=0.9 liquidity.name=mild liquidity.ell_max=5 liquidity.sig_ratio_max=3
echo "== headline under liquidity risk ('base' calibration, cash unwind) =="
X="$PROFILE market.lam=1.0 train.alpha=0.9 liquidity.enabled=true costs.unwind=cash"
python train.py --mode gda     --set $X
python train.py --mode exploit --source gda --set $X
for R in lsm maturity; do
  python train.py --mode fixed   --rule $R          --set $X
  python train.py --mode exploit --source fixed-$R  --set $X
done
python evaluate.py --mode headline --set $X

echo "== step-4 grid: lam x alpha under liquidity 'base' + cash unwind (2 parallel workers) =="
./run_grid.sh
echo "== alpha = 0.99 column re-run with adversary restarts (fixes the lam = 1 adversary collapse) =="
printf '1.0 0.99\n0.5 0.99\n0 0.99\n' | xargs -P 2 -L 1 ./grid_fix_job.sh
python evaluate.py --mode grid --lams 0 0.5 1.0 --alphas 0.99 --set tag=gridrs train.outer_iters=600 liquidity.enabled=true \
  costs.unwind=cash train.adv_restart_every=150 train.adv_restart_steps=300
