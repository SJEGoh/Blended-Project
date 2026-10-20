#!/usr/bin/env bash
# One pricing case: jobs/pricing_job.sh "<name>|<pricing.py run args>"   (see pricing_jobs.txt)
set -u
cd "$(dirname "$0")/.."
NAME="${1%%|*}"; ARGS="${1#*|}"
python3 src/pricing.py run --name "$NAME" --eval-log2 18 --threads 1 $ARGS > results/logs/stdout_pricing_${NAME}.txt 2>&1
