#!/usr/bin/env bash
# Step-4 grid under the liquidity model ('base', cash unwind): 2 workers pull cells from grid_jobs.txt in order.
cd "$(dirname "$0")"
xargs -P 2 -L 1 ./grid_job.sh < grid_jobs.txt
python3 evaluate.py --mode grid --set tag=grid train.outer_iters=600 liquidity.enabled=true costs.unwind=cash
