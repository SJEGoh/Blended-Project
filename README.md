# Adversarial deep hedging of a Bermudan put (MVP)

Seller's risk-based price of a Bermudan put (N = 50 dates) under Merton jump-diffusion,

    p* = inf_hedge sup_stopping CVaR_alpha( Z~_tau - G~_tau + C_tau ),

computed with two networks trained against each other: a hedger (minimiser) and an adversarial
stopper (maximiser). Spec: `docs/adversarial_american_hedging_mvp.md`.

## Status (build order)

| Step | What | Status |
|---|---|---|
| 1 | Simulator + Merton closed form + test; LSM, CRR; LSM vs CRR at lam = 0 | done |
| 2 | Stopper only, zero hedge, alpha = 0 → recovers LSM | fails as specified (random init collapses to never-exercise); passes with a heuristic warm start of the stopper (adopted) — see `results/RESULTS.md` |
| 3 | Hedger + GDA at alpha = 0.9, exploitability | done with reduced compute (batch 2^12, 1500 outer iterations) — see `results/RESULTS.md` |
| 4 | Full grid, LSM-fixed-stopper experiment, figures | headline (LSM-fixed vs adversarial hedger) done at lam = 1, alpha = 0.9 for c = 0, c = 0.5%, c = 0.5% + cash unwind, and liquidity 'base' + cash unwind; grid lam x alpha (3 x 5) done under liquidity 'base' + cash unwind (batch 2^12, 600 iterations): monotone, 14/15 cells pass exploitability, lam = 1 / alpha = 0.99 fails with plain GDA (adversary collapse). The alpha = 0.99 column re-run with adversary restarts (`train.adv_restart_every=150`) passes everywhere and is the one to report; see `results/RESULTS.md` |
| ext | Liquidity / volume risk (`docs/LIQUIDITY.md`) | model, tests, sanity checks done; headline under the 'base' calibration done (adversarial hedge beats fixed-rule hedges under attack by 1.2% / 5.1%); 'mild' calibration pending |
| ext | Dual upper bounds (`dual.py`, `dual_value.py`, `docs/DUAL_CERTIFICATE_PLAN.md`) | martingale-duality certificate D(h) >= worst case of a frozen hedge; shocks exposed in `sim.simulate(..., return_shocks=True)`; validation (alpha = 0 at lam = 0 and 1, monotonicity at alpha = 0.9) in `results/RESULTS.md`; headline brackets pending |
| ext | Volume caps (`volume_cap.enabled`) | participation limit on rebalancing (unwind exempt), off by default; implemented and tested, binding check for existing hedges in `results/RESULTS.md` (`capcheck.py`); no capped training yet |
| P1 | Pricing (Part 1, `pricing.py`) | risk-neutral value with cash vs physical (liquidity-cost) settlement; 38-case sensitivity study over jump intensity/size/volatility, moneyness and liquidity severity, each a certified primal-dual bracket; see `results/RESULTS.md` |

## Layout

    config.yaml      all parameters (override with --set section.key=value)
    run_all.sh       pipeline for the stages implemented so far
    src/             all Python modules; run from the repo root as  python src/<module>.py
      config.py          typed config loader, seeds, device; outputs go to data/, results/, checkpoints/ at the repo root
      sim.py             exact Merton simulator (discounted units), test set, Merton/BS closed forms
      benchmarks.py      LSM (deg-3 in moneyness, separate regression/pricing sets), CRR
      models.py          Hedger, Stopper, constrain_trade hook (identity in the MVP)
      losses.py          rollout, relaxed/hard stopping, empirical CVaR (+ SE)
      train.py           stopper-only (step 2), GDA (step 3), fixed-rule hedgers (headline), exploitability attack
      evaluate.py        hard-rule test-set evaluation, figures, results/RESULTS.md
      dual.py            martingale-duality upper bound D(h) for a frozen hedge (CVaR via Rockafellar-Uryasev)
      dual_value.py      same bound with a value-function (Andersen-Broadie) martingale and one-step inner sampling
      pricing.py         Part 1: risk-neutral Bermudan put value under jumps and volume risk, certified brackets
      pricing_report.py  summary, figures and RESULTS.md section for the pricing study (jobs/pricing_jobs.txt)
      bandcheck.py       fixed-cost pre-check with band-rule hedgers (no training)
      liqcheck.py        liquidity-extension sanity checks (no training)
      capcheck.py        how often volume caps would bind for the existing hedges (no training)
    tests/           simulator, non-anticipativity, CVaR, LSM vs CRR (conftest.py puts src/ on the path)
    dashboard/       interactive seller-vs-holder dashboard (see dashboard/README.md)
    jobs/            batch runners called by run_all.sh; each cd's to the repo root, logs to results/logs/
      run_grid.sh        step-4 grid (grid_jobs.txt, grid_job.sh: one cell = GDA + fresh-adversary attack)
      grid_fix_job.sh    re-run of a cell with adversary restarts (train.adv_restart_every, off by default)
      dual_job.sh        one dual.py run (dual_jobs.txt); dual_vf_job.sh: one dual_value.py run (dual_jobs_vf.txt)
      run_seeds.sh       extra seeds of the liquidity headline (+ dual certificates) and grid dual certificates
      run_alpha_lsm.sh   LSM-trained hedges + attackers at the grid's other alpha levels (for the dashboard)
      pricing_job.sh     one pricing.py case (pricing_jobs.txt)
    docs/
      adversarial_american_hedging_mvp.md  original spec
      LIQUIDITY.md                         liquidity / volume-risk model, calibration and evidence
      DUAL_CERTIFICATE_PLAN.md             plan for the martingale-duality upper bounds

## Quick start

    pip install torch numpy scipy matplotlib pyyaml pytest
    python -m pytest -q tests
    python src/benchmarks.py --lams 0 0.5 1.0
    python src/train.py --mode stopper_only --set market.lam=0 train.alpha=0 train.stopper_init=heuristic
    python src/evaluate.py --mode step2 --lams 0 --init heuristic
    python src/train.py --mode gda     --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python src/train.py --mode exploit --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python src/evaluate.py --mode step3 --alphas 0.9 --set tag=reduced train.batch_log2=12 train.outer_iters=1500
    python src/train.py --mode fixed   --rule lsm        --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python src/train.py --mode exploit --source fixed-lsm --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python src/evaluate.py --mode headline --set tag=reduced train.batch_log2=12 train.outer_iters=1500 train.alpha=0.9
    python src/liqcheck.py --out liqcheck_base.json --set market.lam=1.0 train.alpha=0.9
    # headline under liquidity: add  liquidity.enabled=true costs.unwind=cash  to the train/evaluate --set lists
    python src/dual.py --ckpt none --alpha 0 --set market.lam=0                    # classical Bermudan dual
    python src/dual.py --ckpt checkpoints/gda_lam1_a0p9_s1234_heuristic_reduced.pt   # D(h) for a frozen hedge
    python src/capcheck.py

The held-out test sets (2^18 paths per lam, ~107 MB each) are generated on first use into `data/`
from a fixed seed and are not committed.

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
