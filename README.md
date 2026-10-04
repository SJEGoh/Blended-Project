# Adversarial deep hedging of a Bermudan put (MVP)

Seller's risk-based price of a Bermudan put (N = 50 dates) under Merton jump-diffusion,

    p* = inf_hedge sup_stopping CVaR_alpha( Z~_tau - G~_tau + C_tau ),

computed with two networks trained against each other: a hedger (minimiser) and an adversarial
stopper (maximiser). Spec: `adversarial_american_hedging_mvp.md`.

## Status (build order)

| Step | What | Status |
|---|---|---|
| 1 | Simulator + Merton closed form + test; LSM, CRR; LSM vs CRR at lam = 0 | done |
| 2 | Stopper only, zero hedge, alpha = 0 → recovers LSM | fails as specified (random init collapses to never-exercise); passes with a heuristic warm start of the stopper (adopted) — see `results/RESULTS.md` |
| 3 | Hedger + GDA at alpha = 0.9, exploitability | done with reduced compute (batch 2^12, 1500 outer iterations) — see `results/RESULTS.md` |
| 4 | Full grid, LSM-fixed-stopper experiment, figures | headline (LSM-fixed vs adversarial hedger) done at lam = 1, alpha = 0.9 for c = 0, c = 0.5%, c = 0.5% + cash unwind, and liquidity 'base' + cash unwind; grid lam x alpha (3 x 5) done under liquidity 'base' + cash unwind (batch 2^12, 600 iterations): monotone, 14/15 cells pass exploitability, lam = 1 / alpha = 0.99 fails with plain GDA (adversary collapse). The alpha = 0.99 column re-run with adversary restarts (`train.adv_restart_every=150`) passes everywhere and is the one to report; see `results/RESULTS.md` |
| ext | Liquidity / volume risk (`LIQUIDITY.md`) | model, tests, sanity checks done; headline under the 'base' calibration done (adversarial hedge beats fixed-rule hedges under attack by 1.2% / 5.1%); 'mild' calibration pending |

## Layout

    config.yaml      all parameters (override with --set section.key=value)
    config.py        typed config loader, seeds, device
    sim.py           exact Merton simulator (discounted units), test set, Merton/BS closed forms
    benchmarks.py    LSM (deg-3 in moneyness, separate regression/pricing sets), CRR
    models.py        Hedger, Stopper, constrain_trade hook (identity in the MVP)
    losses.py        rollout, relaxed/hard stopping, empirical CVaR (+ SE)
    train.py         stopper-only (step 2), GDA (step 3), fixed-rule hedgers (headline), exploitability attack
    evaluate.py      hard-rule test-set evaluation, figures, results/RESULTS.md
    bandcheck.py     fixed-cost pre-check with band-rule hedgers (no training)
    liqcheck.py      liquidity-extension sanity checks (no training)
    LIQUIDITY.md     liquidity / volume-risk model, calibration and evidence
    tests/           simulator, non-anticipativity, CVaR, LSM vs CRR
    run_all.sh       pipeline for the stages implemented so far
    run_grid.sh      step-4 grid (grid_jobs.txt, grid_job.sh: one cell = GDA + fresh-adversary attack)
    grid_fix_job.sh  re-run of a cell with adversary restarts (train.adv_restart_every, off by default)
    frontend/        dashboard + interactive path demo: adversarial vs LSM-trained hedge under attack (see below)

## Quick start

    pip install torch numpy scipy matplotlib pyyaml pytest
    python -m pytest -q tests
    python benchmarks.py --lams 0 0.5 1.0
    python train.py --mode stopper_only --set market.lam=0 train.alpha=0 train.stopper_init=heuristic
    python evaluate.py --mode step2 --lams 0 --init heuristic
    python train.py --mode gda     --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python train.py --mode exploit --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python evaluate.py --mode step3 --alphas 0.9 --set tag=reduced train.batch_log2=12 train.outer_iters=1500
    python train.py --mode fixed   --rule lsm        --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python train.py --mode exploit --source fixed-lsm --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python evaluate.py --mode headline --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python liqcheck.py --out liqcheck_base.json --set market.lam=1.0 train.alpha=0.9
    # headline under liquidity: add  liquidity.enabled=true costs.unwind=cash  to the train/evaluate --set lists

The held-out test sets (2^18 paths per lam, ~107 MB each) are generated on first use into `data/`
from a fixed seed and are not committed.

## Dashboard

`frontend/index.html` compares the adversarial (GDA) hedger with the LSM-trained baseline under attack,
for each headline setting (frictionless, c = 0.5%, c = 0.5% + unwind, liquidity 'base' + unwind). It reads
`frontend/data.js`, which is generated from `results/headline*.json` and `results/logs/` (stdlib only, no torch):

    python frontend/build_data.py      # re-run after evaluate.py --mode headline
    open frontend/index.html           # or: python -m http.server -d frontend

`frontend/demo.html` is the interactive version: it simulates a path in the browser, runs the trained
adversarial and LSM-trained hedgers, each attacked by the fresh adversary trained against it, by one shared
adversary (to hold the attacker fixed), or by the LSM exercise rule,
animates prices, hedge positions, the seller's running loss and each adversary's exercise probability, and can
simulate thousands of paths to show the CVaR_0.9 tail. It also shows the two prices: the classical LSM price
(risk-neutral expected payoff under the LSM exercise rule) and the GDA price p* (CVaR_0.9 of the adversarial
hedge's loss against its co-trained adversary), as published and as live estimates from the simulated paths. The networks are exported from `checkpoints/` without torch:

    python frontend/export_models.py   # checkpoints/ -> frontend/models.js
    python frontend/verify_demo.py     # numpy re-implementation vs published CVaRs (all within ~1 SE)
    python frontend/verify_demo.py --log2 12 --fixture /tmp/fx.json && node frontend/test_engine.js /tmp/fx.json 20000

## Conventions and implementation choices

- Everything in discounted units. Exercise dates t_1..t_N, forced at t_N, none at t_0.
- Random numbers are drawn in a fixed order with inverse-CDF Poisson jump counts, so all lam
  share common random numbers (jump counts are pathwise monotone in lam) — this tightens the
  lam-monotonicity comparisons.
- Network inputs are rescaled by fixed constants (log-moneyness by sigma*sqrt(T), money amounts by
  K*sigma*sqrt(T)); information sets are exactly as in the spec.
- Temperature anneal is geometric 1.0 → 0.1. Evaluation always uses the hard rule (logit > 0).
- Denormal floats are flushed to zero (low temperatures otherwise make CPU training ~2x slower).
- `train.stopper_init: heuristic` (default, adopted after step 2): 300 BCE steps towards "exercise iff
  S < K and K − S > Merton European value" before the spec'd ascent. Needed because a randomly
  initialised stopper collapses to never-exercise (details in `results/RESULTS.md`). Applies to the
  GDA adversary and the fresh exploitability attacker too.
- Transaction costs follow the spec: C_n = sum_{k<n} c |delta_k − delta_{k−1}| S~_k. Optional unwind cost at
  exercise (`costs.unwind`: none = spec, cash = c |delta_{tau−1}| S~_tau, physical = c |delta_{tau−1} + 1{S_tau<K}| S~_tau),
  included in L_n. Runs with c > 0 get a `_c<c>` suffix, plus `_uw<mode>` when an unwind cost is on.
- Hedger gradient flows through the stopper's inputs (running P&L, delta_{n-1}): the hedger
  best-responds to the stopper's feedback rule (`train.hedger_grad_through_stopper`).
- `train.hedger_pnl_input` (off by default): ablation flag that also feeds running P&L to the hedger.
- Exploitability: besides the spec's fresh stopper, simple supplementary attacks are reported
  (never-early, LSM rule, best running-P&L threshold fitted on separate validation paths).
- CVaR standard errors use the influence function psi(x) = VaR + (x − VaR)^+/(1 − alpha).
- Stopper-vs-LSM comparisons are also reported *paired* on the same test paths (much smaller SE).
- CRR: besides the American price (spec), a Bermudan-on-grid CRR (exercise only at the 50 dates) is
  reported; the American–Bermudan gap is 0.08% here, so either reference works for the 1% check.
# Blended-Project
