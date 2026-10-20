# Task brief: certified upper bounds for the seller's worst-case price (martingale duality)

This brief is self-contained: it is written for a new session that has not seen the earlier discussion.
Read it fully, then read `README.md`, `results/RESULTS.md`, `sim.py`, `losses.py` and `config.yaml` before
writing code.

## 1. Project context

**Repository:** adversarial deep hedging of a Bermudan put (MVP for a research track question:
*"Advanced valuation methods for American options under jump and volume risks"*).

**Contract:** Bermudan put, S0 = K = 100, T = 0.5 years, N = 50 exercise and rebalancing dates
(dt = 0.01), r = 3%, no dividends, exercise at t_1..t_N (forced at t_N, none at t_0), cash-settled.

**Stock:** Merton jump-diffusion, simulated exactly (`sim.simulate`): sigma = 20%, lam = 1 jump/year,
mean log jump m = -0.10, jump std delta = 0.15, mu = r (so the simulation measure is a pricing measure).
Everything is in discounted units.

**Optional liquidity extension** (`liquidity.enabled`, see `LIQUIDITY.md`, `sim.liquidity_state`):
non-tradable illiquidity ell, relative volume v and EWMA volatility sig, driven by the returns plus two
extra Gaussian noises (eps, eps_v). Trades pay half-spread + square-root impact (`losses.liquidity_cost`);
`costs.unwind = cash` charges the cost of closing the hedge at exercise.

**Seller's loss if exercised at date n:** `L_n = Z~_n - G~_n + C_n (+ U_n)`, computed by
`losses.hedge_rollout` as `Rollout.L` with shape (B, N) for dates 1..N.

**Seller's price (what the project computes):**

```
p* = inf over hedges  sup over exercise rules tau  CVaR_alpha( L_tau )
```

solved by two networks trained with gradient descent-ascent (`train.py --mode gda`): a hedger (seller)
and an adversarial stopper (holder) that maximises the seller's CVaR. Exercise rules and hedges are
non-anticipative (see `tests/test_nonanticipativity.py`, `tests/test_liquidity.py`).

**Headline finding (lam = 1, alpha = 0.9, one seed, reduced compute):** assuming the holder follows the
risk-neutral Longstaff-Schwartz (LSM) exercise rule underprices the hedged seller's risk.

| Setting | price assuming LSM exercise | p* | underpricing |
|---|---|---|---|
| frictionless | 13.410 | 13.527 | +0.116 (0.9%) |
| 0.5% cost | 14.192 | 14.401 | +0.210 (1.5%) |
| 0.5% cost + cash unwind | 14.330 | 14.599 | +0.268 (1.9%) |
| liquidity 'base' + cash unwind | 14.650 | 15.123 | +0.474 (3.2%) |

Sources: `results/headline*.json`, `results/benchmarks_lam1.json` (LSM price 6.546 +/- 0.017).

**The weakness this task fixes:** p* is only as high as the strongest attack found. A trained adversary is
one feasible exercise rule, so its CVaR is a *lower* estimate of the true worst case over realistic
(non-anticipative) holders. `RESULTS.md` shows a fresh adversary falling 0.03-0.09 short of the co-trained
one, and simple rule-based attacks beating p* slightly in a few grid cells. We need a certified **upper**
bound so every headline number becomes an interval.

## 2. Goal

For a **fixed, frozen hedge** h, bracket the realistic worst case

```
W(h) = sup over non-anticipative tau  CVaR_alpha( L_tau(h) )
```

as

```
CVaR under the trained adversary  <=  W(h)  <=  dual upper bound  D(h)
```

and report D(h) for the adversarial (GDA) hedge and the LSM-trained hedge in each headline setting, so the
underpricing can be stated with a certified lower and upper bound.

Secondary (validation): in the frictionless case with alpha = 0 the same machinery gives the classical
primal-dual bracket of the risk-neutral Bermudan price (lower bound from LSM, upper bound from duality).

## 3. The mathematics

### 3.1 Classical martingale duality (Rogers 2002; Haugh & Kogan 2004)

For payoffs Z_n and any martingale M with M_0 = 0:

1. Optional stopping: for any stopping time tau <= N, E[M_tau] = 0 (the indicator 1{tau > n} is
   F_n-measurable, so each increment averages to zero). This is exactly where non-anticipativity is used.
2. Weak duality: E[Z_tau] = E[Z_tau - M_tau] <= E[max_n (Z_n - M_n)], so
   `sup_tau E[Z_tau] <= E[max_n (Z_n - M_n)]` for every martingale M.
3. Strong duality: with the Snell envelope U and its Doob decomposition U_n = U_0 + M*_n - A_n,
   Z_n - M*_n <= U_0 pathwise, so the bound is exact (and has zero variance) at M = M*.
4. In a complete market M* is the replicating hedge's gains; with jumps, stock-only martingales cannot
   reach M*, so stock-hedge penalties give valid but loose bounds.

Already measured with the JS engine (frontend scenario 'frictionless', 40k fresh paths): LSM lower bound
6.55 +/- 0.04; E[max_n Z_n] = 10.95; E[max_n (Z_n - G_n)] with the trained GDA hedge = 7.55 +/- 0.02.

### 3.2 Extension to CVaR (the new part)

Use the Rockafellar-Uryasev representation `CVaR_alpha(X) = min_c { c + E[(X - c)^+] / (1 - alpha) }`.
Since sup min <= min sup:

```
W(h) = sup_tau min_c { c + E[(L_tau - c)^+]/(1-alpha) }
     <= min_c { c + sup_tau E[(L_tau - c)^+] / (1-alpha) }
```

For fixed c the inner problem is an ordinary optimal stopping problem with payoff (L_n - c)^+, so weak
duality applies:

```
D(h) = min_c { c + E[ max_{n=1..N} ( (L_n - c)^+ - M^c_n ) ] / (1-alpha) }      for any martingales M^c
```

D(h) is a valid upper bound for **any** martingales; optimising them only tightens it. Caveat: the
sup/min swap can leave a gap even with perfect martingales (CVaR is not time-consistent). Measure and
report that gap honestly; do not claim tightness.

Note: for alpha = 0 this collapses to the classical dual (set c very negative, (L - c)^+ = L - c).

## 4. Implementation plan

### 4.1 Expose the driving shocks (`sim.py`)

`simulate` currently draws, per step n: Z (price Gaussian), U (uniform -> jump count N_n by inverse CDF
of Poisson(lam*dt)), Zj (jump-size Gaussian); `liquidity_state` then draws eps and eps_v. Return these
(e.g. in `PathBatch.extras["shocks"]` or a separate return value behind a flag) **without changing the
draw order or values**, so all existing results and common random numbers are unchanged. Add a test
that paths are bit-identical with and without the flag.

### 4.2 Martingale increments from known shocks (new module, e.g. `dual.py`)

Any increment `Delta M_{n+1} = sum_k phi_k(F_n) * xi_k(n+1)` with `E[xi_k | F_n] = 0` is a martingale
increment. Use shocks with known zero mean, for example:

- Z_{n+1}, Z_{n+1}^2 - 1
- N_{n+1} - lam*dt (compensated jump count)
- Zj_{n+1} * 1{N_{n+1} > 0} (and/or the full jump size minus its mean)
- with liquidity: eps_{n+1}, eps_v_{n+1}, eps^2 - 1

The coefficients phi_k(F_n) come from a small network whose inputs are F_n-measurable features (the same
features the hedger and stopper use: t_n/T, log-moneyness, delta_{n-1}, running loss L_n / scale,
ITM flag, liquidity features). M^c_0 = 0 and M^c_n = cumulative sum. Make c an input (or train one network
per c on a grid).

Check: the discounted stock increment itself, delta*(S~_{n+1} - S~_n), is one admissible martingale, so
"penalty = hedge gains" is a special case and must reproduce the 7.55 number above.

### 4.3 Training and evaluation

- Freeze the hedge (load from `checkpoints/`, e.g. `gda_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced.pt`
  and `fixed-lsm_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced.pt`; loading code in
  `evaluate.load_policy`, config from the checkpoint's `cfg`).
- For a grid of c around the VaR level of L_tau (roughly 8-18 for the liquidity setting), train the
  martingale network on fresh simulated batches to minimise E[max_n ((L_n - c)^+ - M^c_n)] (plain
  gradient descent; the max is piecewise differentiable).
- Evaluate on **independent** paths (the fixed test set via `sim.get_test_set`, or fresh paths with a
  different seed) to avoid optimistic bias. Report D(h) with a Monte Carlo standard error; take the min
  over the c grid (a coarse grid plus local refinement is enough; note that taking a min over c on the
  same paths introduces a small downward bias - either use separate paths to choose c and to evaluate, or
  report it).
- Report alongside: the adversary's CVaR (lower), the hindsight CVaR with M = 0 (trivial upper), and D(h).

### 4.4 Validation (must pass before reporting headline brackets)

1. alpha = 0, frictionless, lam = 0: the dual upper bound should approach the CRR Bermudan price 5.006
   (`results/benchmarks_lam0.json`); the LSM lower bound is 4.994.
2. alpha = 0, frictionless, lam = 1: bracket LSM 6.546 from above; report how much the jump shocks tighten
   it relative to the hedge-only penalty (7.55).
3. Monotonicity: D(h) >= adversary CVaR for every hedge and setting (if violated, it is a bug).
4. Tests: martingale increments have sample mean ~0 (within 3 SE) at every n; D(h) unchanged under
   permutation of future shocks after date n for the F_n-measurable parts; existing test suite still
   passes (`python -m pytest -q tests`).

### 4.5 Deliverables

- `dual.py` (or similar) with a CLI, e.g. `python dual.py --ckpt <hedge ckpt> --alpha 0.9 --set ...`
- `results/dual_*.json` and a short section appended to `results/RESULTS.md` via the existing reporting
  pattern in `evaluate.py`.
- A table: for each headline setting and each hedge, [adversary CVaR, D(h)] and the resulting certified
  interval for the underpricing (lower end: p* minus price assuming LSM exercise; upper end: the dual
  bound of the adversarial hedge minus price assuming LSM exercise).

## 5. Constraints and conventions

- Do not change existing results, checkpoints or default behaviour; new behaviour goes behind flags.
- Keep everything non-anticipative; the dual is the one place where hindsight (the pathwise max) is
  allowed, and only on simulated paths.
- Follow the repo's style: single config (`config.yaml`, `--set section.key=value`), fixed seeds,
  standard errors on every reported number, results computed on held-out paths.
- Environment note: in the cloud sandbox `pip install torch` works from PyPI; the pytorch.org wheel index
  is blocked. numpy is required.
- Budget: about 1.5-2 weeks for one person; it is one part of an 8-week plan that also adds volume caps
  (hard participation limits on |delta_n - delta_{n-1}|), a recurrent hedger/stopper supplied by the
  supervisor, multi-seed robustness and a change of measure.

## 6. Key references

- L.C.G. Rogers (2002), Monte Carlo valuation of American options, Mathematical Finance 12(3), 271-286.
- M. Haugh and L. Kogan (2004), Pricing American options: a duality approach, Operations Research 52(2),
  258-270 (free PDF on Martin Haugh's website).
- L. Andersen and M. Broadie (2004), Primal-dual simulation algorithm for pricing multidimensional American
  options, Management Science 50(9), 1222-1234.
- S. Becker, P. Cheridito and A. Jentzen (2020), Pricing and hedging American-style options with deep
  learning, JRFM 13(7), 158 (neural lower/upper bounds; arXiv 1912.11060).
- V. Krätschmer and J. Schoenmakers (2010), Representations for optimal stopping under dynamic monetary
  utility functionals, SIAM J. Financial Math. 1(1), 811-832 (duality beyond expectations).
- R.T. Rockafellar and S. Uryasev (2000), Optimization of conditional value-at-risk, J. Risk 2(3), 21-41.
- A.F. Roch (2022), Hedging of American options in illiquid markets with price impacts, IJTAF (closest
  prior work on worst-case exercise for a hedging seller; position against it).
