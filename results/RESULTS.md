# Results — adversarial deep hedging of a Bermudan put (MVP)

All numbers: hard exercise rule on the fixed held-out test set (2^18 paths, same seed for every lam),
± Monte Carlo standard error. Build-order status: steps 1–3 done (reduced compute for step 3); headline experiment (item 4) done at lam = 1, alpha = 0.9 (several cost settings); step-4 grid done (grid_liqbase_uwcash_grid 15/15 cells; reduced compute).

## Benchmarks (step 1)

| lam | LSM price (indep. pricing paths) | Merton European | CRR Bermudan-on-grid | CRR American |
|---|---|---|---|---|
| 0 | 4.9936 ± 0.0118 | 4.8822 | 5.0059 | 5.0097 |
| 0.5 | 5.7860 ± 0.0150 | 5.6776 | — (lam > 0) | — (lam > 0) |
| 1 | 6.5460 ± 0.0174 | 6.4314 | — (lam > 0) | — (lam > 0) |

LSM vs CRR (lam = 0): LSM − CRR Bermudan = -0.0123 (-0.25%, -1.0 SE). CRR American − Bermudan-on-grid = 0.0038.

## Validation

| # | Check | Status | Numbers |
|---|---|---|---|
| 1 | alpha = 0 recovers LSM | Step 2 as written: **FAIL** (random-init stopper collapses to never-exercise: 4.8715 at lam=0, -2.44%). With warm start (adopted): **PASS**. Full GDA at alpha = 0 (lam=1, reduced): p* = 6.6014 ± 0.0321 vs LSM 6.5460 (+0.85%) → **PASS** | lam=0: NN 5.0033±0.0118 vs LSM 4.9936 (+0.20%); lam=0.5: NN 5.8317±0.0152 vs LSM 5.7860 (+0.79%); lam=1: NN 6.5885±0.0176 vs LSM 6.5460 (+0.65%) |
| 2 | Monotonicity in alpha and lam | step-4 grid (see below) | grid_liqbase_uwcash_grid (15/15 cells): all of 22 adjacent pairs non-decreasing within 2 SE |
| 3 | Exploitability gap (< 2% of p*) | step 3 (alpha = 0.9, 0; lam = 1; reduced) — see below | alpha=0.9: gap -0.0035 ± 0.0016 (-0.03% of p*) → PASS; best supplementary attack -0.77% (none beats p*); alpha=0: gap +0.0018 ± 0.0010 (+0.03% of p*) → PASS; best supplementary attack -0.02% (none beats p*) |
| 4 | Adversary vs LSM-fixed hedger | reported (lam = 1, alpha = 0.9, reduced) — see below | c = 0 + liquidity 'base' + cash unwind: trained vs LSM rule: rise +4.5% under attack, worse than adversarial hedger by +0.266 ± 0.006; trained vs hold-to-maturity: rise +11.7% under attack, worse than adversarial hedger by +0.857 ± 0.009 / c = 0: trained vs LSM rule: rise +1.0% under attack, worse than adversarial hedger by +0.017 ± 0.003; trained vs hold-to-maturity: rise +2.5% under attack, worse than adversarial hedger by +0.025 ± 0.003 / c = 0.005 + cash unwind: trained vs LSM rule: rise +1.9% under attack, worse than adversarial hedger by +0.029 ± 0.005; trained vs hold-to-maturity: rise +1.9% under attack, worse than adversarial hedger by +0.025 ± 0.003 / c = 0.005: trained vs LSM rule: rise +1.5% under attack, worse than adversarial hedger by +0.038 ± 0.004; trained vs hold-to-maturity: rise +1.9% under attack, worse than adversarial hedger by +0.010 ± 0.002 |
| 5 | Non-anticipativity unit test | PASS | 65 passed, 1 warning in 28.07s |

## Step-4 grid under liquidity model 'base', cash unwind (15/15 cells done)

Each cell: adversarial (GDA) training, 600 outer iterations, batch 2^12 (2^13 at alpha = 0.99),
then a fresh warm-started adversary (2000 steps) against the frozen hedger. Hard-rule CVaR on the 2^18 test paths
of that lam. 'Strongest simple attack' is the best of never early, LSM rule, running-loss threshold, illiquidity threshold, post-decline (fitted on validation paths).

Seller's price p* (± SE); last column: frictionless LSM price for reference.

| lam | alpha = 0 | alpha = 0.5 | alpha = 0.9 | alpha = 0.95 | alpha = 0.99 | LSM |
|---|---|---|---|---|---|---|
| 0 | 5.002 ± 0.012 | 6.229 ± 0.003 | 7.433 ± 0.004 | 7.764 ± 0.004 | 8.350 ± 0.008 | 4.994 |
| 0.5 | 5.830 ± 0.015 | 7.931 ± 0.011 | 12.956 ± 0.029 | 15.029 ± 0.036 | 19.559 ± 0.075 | 5.786 |
| 1 | 6.590 ± 0.017 | 9.460 ± 0.014 | 15.166 ± 0.027 | 17.281 ± 0.037 | 21.002 ± 0.070 | 6.546 |

Exploitability gap (fresh adversary − trained adversary, paired; PASS if < 2% of p*) and strongest simple attack vs p*:

| lam | alpha = 0 | alpha = 0.5 | alpha = 0.9 | alpha = 0.95 | alpha = 0.99 |
|---|---|---|---|---|---|
| 0 | +0.03% PASS; simple +0.1% | -0.03% PASS; simple -0.7% | +0.00% PASS; simple +0.1% | +0.00% PASS; simple +0.2% | +0.00% PASS; simple +0.3% |
| 0.5 | -0.05% PASS; simple -0.2% | +0.22% PASS; simple -0.8% | +0.15% PASS; simple -0.1% | -0.04% PASS; simple -0.8% | +0.03% PASS; simple -1.4% |
| 1 | -0.02% PASS; simple -0.1% | +0.60% PASS; simple -0.0% | +0.02% PASS; simple -0.5% | +0.01% PASS; simple -0.7% | +8.75% FAIL; simple +1.8% |

Trained adversary: mean exercise time / early-exercise fraction; hedge mean (std):

| lam | alpha = 0 | alpha = 0.5 | alpha = 0.9 | alpha = 0.95 | alpha = 0.99 |
|---|---|---|---|---|---|
| 0 | 0.440 / 0.37; 0.00 (0.00) | 0.464 / 0.25; -0.48 (0.30) | 0.500 / 0.00; -0.47 (0.28) | 0.500 / 0.00; -0.47 (0.27) | 0.500 / 0.00; -0.46 (0.26) |
| 0.5 | 0.443 / 0.33; -0.00 (0.00) | 0.467 / 0.19; -0.46 (0.29) | 0.481 / 0.11; -0.56 (0.27) | 0.487 / 0.07; -0.58 (0.21) | 0.496 / 0.02; -0.55 (0.14) |
| 1 | 0.435 / 0.35; -0.00 (0.00) | 0.461 / 0.18; -0.45 (0.29) | 0.480 / 0.10; -0.57 (0.22) | 0.487 / 0.07; -0.57 (0.19) | 0.500 / 0.00; -0.52 (0.15) |

Convergence check (lam = 1, alpha = 0.9): p* = 15.1662 ± 0.0269 at 600 iterations vs 15.1234 ± 0.0275 at 1500 iterations (+0.28%).

Reading:
- **Shape.** p* rises with alpha and with lam everywhere: 22 of 22 adjacent pairs are non-decreasing (validation
  item 2). Jumps drive the tail. At alpha = 0.9, going from lam = 0 to 0.5 adds 5.5 (+74%), and 0.5 to 1 adds 2.2.
  Without jumps the price is nearly flat beyond alpha = 0.9 (7.43 -> 8.35). With lam = 1 it keeps climbing
  (15.2 -> 21.0, and the 21.0 is itself an understatement, see below).
- **alpha = 0 sanity check.** The hedger learns not to trade (hedge identically 0). Under the pricing measure, hedging
  gains have zero mean and every trade costs, so this is the right answer. p* is within 0.2-0.8% of the frictionless
  LSM price at every lam (validation item 1 tolerance: 1%). The test and LSM pricing paths are different samples.
- **Adversary behaviour.** The early-exercise fraction falls as alpha rises: about 0.35 at alpha = 0, and 0.02-0.10
  at alpha >= 0.9 with lam > 0. In the tail the worst losses are jump losses accumulated by holding, so the adversary
  mostly holds. At lam = 0 and alpha >= 0.9, both the trained and the fresh adversary never exercise early.
- **Exploitability.** 14 of 15 cells pass (|gap| <= 0.6% of p*). The trained adversary beats every simple attack except
  at lam = 0, alpha >= 0.9. There, a running-loss threshold rule (it exercises on about 0.6% of paths) is higher by
  0.11% / 0.19% / 0.33% of p* (paired z about 10 / 10 / 5; `results/diagnostics/grid_pnl_attack_paired.json`).
  The gap is small but real, so at lam = 0 the gradient-trained adversaries stop slightly short of the best response.
- **lam = 1, alpha = 0.99 fails.** The trained adversary collapsed to never-early within 50 iterations. At batch 2^13
  the 1% tail is about 82 paths. Once the stopping logits saturate negative, the relaxed-stopping gradient vanishes,
  and the adversary pool only holds snapshots of the collapsed stopper. A fresh adversary reaches 22.84 (+8.75%) and
  the running-loss rule 21.37 (+1.75%, paired z = 14.5). So 21.00 is this hedger's CVaR under hold-to-maturity, not
  a saddle value; its worst case is at least 22.84. The lam = 0.5, alpha = 0.99 run collapsed too (iteration 100) but
  recovered by iteration 200. The alpha = 0.99 column was re-run with adversary restarts (below): this fixes the cell
  and leaves the other two unchanged, so that column is the one to report for alpha = 0.99.
- **Convergence.** At (lam = 1, alpha = 0.9), p* at 600 iterations is 0.28% above the 1500-iteration run. The grid
  prices are probably biased up by a few tenths of a percent (the hedger is slightly under-trained). Compute is
  reduced throughout: batch 2^12 and 600 iterations, versus 2^14 and 3000 in the spec.

Figure: `figures/grid_price_vs_alpha_liqbase_uwcash_grid.png`.

### Re-run with adversary restarts (fix for adversary collapse; the grid above is plain GDA)

Same setting, seed and compute as the grid cell, plus: every 150 outer iterations a fresh
warm-started stopper is trained 300 ascent steps against the frozen hedger and swapped in
if its hard-rule CVaR on a common batch is higher (`train.adv_restart_every`, off by default). Evaluated
exactly like the grid cells (fresh warm-started adversary, 2000 steps).

| lam | alpha | p* (restarts) | grid p* (no restarts) | fresh-adversary gap | strongest simple attack | adversary early-exercise fraction | restarts (it: current → fresh, swap?) |
|---|---|---|---|---|---|---|---|
| 0 | 0.99 | 8.349 ± 0.008 | 8.350 ± 0.008 (gap +0.00% PASS; fresh 8.350) | +0.00% PASS (fresh 8.349) | P&L threshold +0.3% | 0.000 | 150: 8.50 → 8.50 keep; 300: 8.39 → 8.39 keep; 450: 8.38 → 8.38 keep |
| 0.5 | 0.99 | 19.561 ± 0.075 | 19.559 ± 0.075 (gap +0.03% PASS; fresh 19.565) | -0.01% PASS (fresh 19.559) | P&L threshold -1.7% | 0.019 | 150: 19.95 → 20.28 swap; 300: 19.62 → 19.53 keep; 450: 19.83 → 19.76 keep |
| 1 | 0.99 | 21.879 ± 0.070 | 21.002 ± 0.070 (gap +8.75% FAIL; fresh 22.839) | -0.11% PASS (fresh 21.856) | P&L threshold -2.0% | 0.021 | 150: 20.87 → 22.98 swap; 300: 22.15 → 22.01 keep; 450: 22.17 → 21.99 keep |

Monotonicity in lam within the re-run: alpha = 0.99, lam 0 -> 0.5: +11.213 (ok); alpha = 0.99, lam 0.5 -> 1: +2.318 (ok).

Reading:
- **No effect where plain GDA worked.** At lam = 0, every fresh stopper tied the GDA stopper (both never exercise
  early), so nothing was swapped and p* is unchanged (8.349 vs 8.350). At lam = 0.5, which collapsed and recovered
  without restarts, one swap happened at iteration 150. The final p* (19.561 vs 19.559) and the hedge are unchanged.
- **Fixes the failed cell.** At lam = 1, the restart at iteration 150 caught the collapse (22.98 vs 20.87). Later
  restarts kept the GDA adversary, which beat fresh ones by 0.15-0.18. p* = 21.879 ± 0.070, gap -0.11% (PASS), and
  every simple attack is below p* (best -2.0%).
- **The hedge changes, not only the number.** Against a working adversary, the lam = 1 hedger accepts more
  hold-to-maturity risk (never-early CVaR 21.41 vs 21.00) in exchange for much less early-exercise exposure. Its
  worst case found is 21.88, versus at least 22.84 for the collapsed run's hedger (about 4% lower).
- **Reporting.** Use this column for alpha = 0.99 so the column has one training protocol. All three cells pass,
  and p* is monotone in lam and, against the alpha = 0.95 column, in alpha. The alpha <= 0.95 cells all passed
  without restarts and are left as run. Restarts changed nothing wherever GDA had not collapsed, so re-running those
  cells is not expected to move them (untested). Cost: about +55% training time (94-98 vs 61-63 min per cell).
- **Not fixed by restarts.** At lam = 0 the running-loss threshold edge remains (+0.30%). The fresh stoppers are
  gradient-trained from the same warm start and also settle on never-early. Limitation: gradient-trained
  adversaries can miss a small, rare exercise region (about 0.6% of paths here), worth a few tenths of a percent.


## Headline experiment: does the adversary matter? (lam = 1, alpha = 0.9, transaction cost c = 0, liquidity model 'base', cash-settled unwind cost at exercise)

Three hedgers, identical initial weights, batch 2^12, 1500 hedger updates, same lr schedule:
the adversarial (GDA) hedger, one trained against the fixed risk-neutral LSM exercise rule, and one trained
against hold-to-maturity. Each frozen hedger is then attacked by a fresh warm-started adversary (identical
initialisation and recipe for all three, 2000 steps) and by the simple rules. All CVaRs on the same test paths.

| Hedger | CVaR vs the rule it was trained against | Fresh adversary | never early | LSM rule | P&L threshold | Worst case | Rise (paired) |
|---|---|---|---|---|---|---|---|
| adversarial (GDA) | 15.1234 ± 0.0275 | 15.0363 | 14.7703 | 14.7212 | 14.8908 | **15.1234** | +0.0000 ± 0.0000 (+0.0%) |
| trained vs LSM rule | 14.6499 ± 0.0260 | 15.3025 | 14.5343 | 14.6499 | 14.9684 | **15.3025** | +0.6527 ± 0.0098 (+4.5%) |
| trained vs hold-to-maturity | 14.2320 ± 0.0282 | 15.8933 | 14.2320 | 14.7967 | 15.2781 | **15.8933** | +1.6613 ± 0.0119 (+11.7%) |
| no hedge | — | — | 30.4019 | 25.4535 | 30.5975 | 30.5975 | — |

- trained vs LSM rule minus adversarial hedger, each under its own fresh adversary: +0.2662 ± 0.0063 (paired over test paths).
- trained vs hold-to-maturity minus adversarial hedger, each under its own fresh adversary: +0.8570 ± 0.0091 (paired over test paths).

Exercise behaviour of each fresh adversary (E[tau] / early-exercise fraction): adversarial (GDA): 0.476 / 0.111; trained vs LSM rule: 0.481 / 0.081; trained vs hold-to-maturity: 0.475 / 0.098.

Mean hedge position delta_n in the money (S_n < 0.9 K) by illiquidity ell_n when it is set:

| Hedger | ell <1.25 | ell 1.25-2 | ell 2-4 | ell 4-8 | ell >=8 |
|---|---|---|---|---|---|
| adversarial (GDA) | -0.928 (n=1370029) | -0.880 (n=414587) | -0.661 (n=149677) | -0.475 (n=77241) | -0.412 (n=36006) |
| trained vs LSM rule | -0.813 (n=1370029) | -0.788 (n=414587) | -0.660 (n=149677) | -0.533 (n=77241) | -0.504 (n=36006) |
| trained vs hold-to-maturity | -0.880 (n=1370029) | -0.863 (n=414587) | -0.750 (n=149677) | -0.662 (n=77241) | -0.628 (n=36006) |

Reading:
- **Cost model:** liquidity extension (docs/LIQUIDITY.md): every trade pays a volatility-proportional half-spread plus
  square-root impact at the current (stochastic, non-tradable) liquidity state; the hedge is unwound at exercise
  under the liquidity state of that date.
- **The adversary now matters for the hedge.** Taking the strongest attack found for each hedger: adversarial
  15.123, LSM-trained 15.303 (+0.18, +1.2%), maturity-trained 15.893 (+0.77, +5.1%). Under each hedger's own fresh
  adversary the gaps are 0.266 ± 0.006 (LSM-trained) and 0.857 ± 0.009 (maturity-trained). Attacker strength is
  uncertain by about 0.09 (the fresh adversary falls 0.087 ± 0.006 short of the co-trained one on the adversarial
  hedge), so the maturity gap is far outside that range and the LSM gap about 2-3x it. In every frictionless and
  proportional-cost setting before, these gaps were 0.01-0.04 and within the uncertainty.
- **Pricing understatement is also larger:** the fixed-rule hedges look 4.5% (LSM-trained) and 11.7% (maturity-
  trained) safer under their own assumed rule than under attack, versus 1-2.5% without liquidity risk.
- **Mechanism:** the unwind at exercise is paid at that date's liquidity, so the holder can force a costly unwind by
  exercising during stress. The adversarial hedge is the most liquidity-sensitive: close to fully hedged in calm
  markets (mean delta -0.93 in the money) and cutting its position hardest when illiquid (-0.41 when ell >= 8).
  The maturity-trained hedge, which only expects to unwind at T, keeps large positions in stress (-0.63) and is
  hit hardest; the LSM-trained hedge is under-hedged in calm markets (-0.81), and its attacker exercises far less
  often than the LSM rule assumes (8% vs 34% early).
- **Price of the liquidity risk:** p* rises from 13.53 (frictionless) to 15.12 (+11.8%).
- **Calibration matters:** 'base' allows stress spreads up to 40x normal; the 'mild' calibration (up to 15x) is not
  run yet and the sanity checks suggest a smaller effect there.
- **Caveats:** one seed, lam = 1, alpha = 0.9, reduced compute; the worst case is only as strong as the attackers.

Figures: `figures/headline_loss_hist_liqbase_uwcash.png`, `figures/headline_hedge_ratio_liqbase_uwcash.png`, `figures/headline_hedge_vs_liquidity_liqbase_uwcash.png`.

## Headline experiment: does the adversary matter? (lam = 1, alpha = 0.9, transaction cost c = 0)

Three hedgers, identical initial weights, batch 2^12, 1500 hedger updates, same lr schedule:
the adversarial (GDA) hedger, one trained against the fixed risk-neutral LSM exercise rule, and one trained
against hold-to-maturity. Each frozen hedger is then attacked by a fresh warm-started adversary (identical
initialisation and recipe for all three, 2000 steps) and by the simple rules. All CVaRs on the same test paths.

| Hedger | CVaR vs the rule it was trained against | Fresh adversary | never early | LSM rule | P&L threshold | Worst case | Rise (paired) |
|---|---|---|---|---|---|---|---|
| adversarial (GDA) | 13.5265 ± 0.0265 | 13.5230 | 13.2399 | 13.4218 | 13.3515 | **13.5265** | +0.0000 ± 0.0000 (+0.0%) |
| trained vs LSM rule | 13.4102 ± 0.0261 | 13.5398 | 13.2520 | 13.4102 | 13.3896 | **13.5398** | +0.1297 ± 0.0056 (+1.0%) |
| trained vs hold-to-maturity | 13.2155 ± 0.0262 | 13.5478 | 13.2155 | 13.4386 | 13.3733 | **13.5478** | +0.3323 ± 0.0047 (+2.5%) |
| no hedge | — | — | 30.4019 | 25.4535 | 30.5975 | 30.5975 | — |

- trained vs LSM rule minus adversarial hedger, each under its own fresh adversary: +0.0168 ± 0.0028 (paired over test paths).
- trained vs hold-to-maturity minus adversarial hedger, each under its own fresh adversary: +0.0248 ± 0.0027 (paired over test paths).

Exercise behaviour of each fresh adversary (E[tau] / early-exercise fraction): adversarial (GDA): 0.474 / 0.137; trained vs LSM rule: 0.478 / 0.121; trained vs hold-to-maturity: 0.473 / 0.139.

Reading:
- **Pricing:** evaluating the seller's risk under an assumed exercise rule understates the worst case. The LSM-trained
  hedge looks like 13.410 under the LSM rule but is 13.540 under attack (+0.130 ± 0.006, +1.0%);
  for hold-to-maturity the understatement is +0.332 ± 0.005 (+2.5%).
- **Hedging:** the adversarially trained hedge lowers the attacked CVaR by only 0.017 ± 0.003 vs the LSM-trained hedge and 0.025 ± 0.003
  vs the maturity-trained hedge (0.12% / 0.18%): statistically clear, economically small.
- **Mechanism:** the LSM-trained hedge differs from the others only below the LSM exercise boundary (it is less
  short there, because in training those paths had already been exercised); the worst-case holder rarely
  visits that region, so the exploit is small. In this frictionless setting the three hedges are nearly identical.
- **Caveats:** one seed, lam = 1, alpha = 0.9, reduced compute; the worst case is only as strong as the attackers.

Figures: `figures/headline_loss_hist.png`, `figures/headline_hedge_ratio.png`.

## Headline experiment: does the adversary matter? (lam = 1, alpha = 0.9, transaction cost c = 0.005, cash-settled unwind cost at exercise)

Three hedgers, identical initial weights, batch 2^12, 1500 hedger updates, same lr schedule:
the adversarial (GDA) hedger, one trained against the fixed risk-neutral LSM exercise rule, and one trained
against hold-to-maturity. Each frozen hedger is then attacked by a fresh warm-started adversary (identical
initialisation and recipe for all three, 2000 steps) and by the simple rules. All CVaRs on the same test paths.

| Hedger | CVaR vs the rule it was trained against | Fresh adversary | never early | LSM rule | P&L threshold | Worst case | Rise (paired) |
|---|---|---|---|---|---|---|---|
| adversarial (GDA) | 14.5986 ± 0.0258 | 14.5718 | 14.3575 | 14.3655 | 14.3575 | **14.5986** | +0.0000 ± 0.0000 (+0.0%) |
| trained vs LSM rule | 14.3302 ± 0.0251 | 14.6009 | 14.4333 | 14.3302 | 14.4941 | **14.6009** | +0.2706 ± 0.0077 (+1.9%) |
| trained vs hold-to-maturity | 14.3262 ± 0.0255 | 14.5969 | 14.3262 | 14.3625 | 14.3502 | **14.5969** | +0.2707 ± 0.0041 (+1.9%) |
| no hedge | — | — | 30.4019 | 25.4535 | 30.5975 | 30.5975 | — |

- trained vs LSM rule minus adversarial hedger, each under its own fresh adversary: +0.0290 ± 0.0048 (paired over test paths).
- trained vs hold-to-maturity minus adversarial hedger, each under its own fresh adversary: +0.0250 ± 0.0025 (paired over test paths).

Exercise behaviour of each fresh adversary (E[tau] / early-exercise fraction): adversarial (GDA): 0.477 / 0.147; trained vs LSM rule: 0.484 / 0.090; trained vs hold-to-maturity: 0.479 / 0.124.

Reading:
- **Cost model:** liquidity extension (docs/LIQUIDITY.md): every trade pays a volatility-proportional half-spread plus
  square-root impact at the current (stochastic, non-tradable) liquidity state; the hedge is unwound at exercise
- **Cost model:** C_n = sum_{k<n} c |delta_k - delta_{k-1}| S~_k (initial purchase charged), plus the cash-settled unwind cost at exercise c |delta_{tau-1}| S~_tau.
- **Unwind cost raises the price further:** p* = 14.599 vs 14.401 without the unwind charge (+1.4%) and 13.527
  frictionless (+7.9%).
- **Pricing understatement:** both fixed-rule hedgers look like ~14.33 under the rule they were trained for and
  ~14.60 under attack (+1.9% each).
- **Hedging, no benefit:** the strongest attack found gives 14.599 (adversarial), 14.601 (LSM-trained) and 14.597
  (maturity-trained) — identical within noise. Under each hedger's own fresh adversary the adversarial hedge is
  better by 0.029 ± 0.005 / 0.025 ± 0.003, but the fresh adversary falls 0.027 ± 0.003 short of the co-trained one
  on the adversarial hedge, so that edge is within attacker-strength uncertainty.
- **Conclusion across c = 0, c = 0.5% and c = 0.5% + cash unwind:** the worst-case exercise assumption moves the
  seller's risk number by ~1–2.5%, but a hedge trained for a fixed exercise rule (even hold-to-maturity) is as
  robust to adversarial exercise as the adversarially trained hedge.
- **Caveats:** one seed, lam = 1, alpha = 0.9, reduced compute; the worst case is only as strong as the attackers.

Figures: `figures/headline_loss_hist_c0.005_uwcash.png`, `figures/headline_hedge_ratio_c0.005_uwcash.png`.

## Headline experiment: does the adversary matter? (lam = 1, alpha = 0.9, transaction cost c = 0.005)

Three hedgers, identical initial weights, batch 2^12, 1500 hedger updates, same lr schedule:
the adversarial (GDA) hedger, one trained against the fixed risk-neutral LSM exercise rule, and one trained
against hold-to-maturity. Each frozen hedger is then attacked by a fresh warm-started adversary (identical
initialisation and recipe for all three, 2000 steps) and by the simple rules. All CVaRs on the same test paths.

| Hedger | CVaR vs the rule it was trained against | Fresh adversary | never early | LSM rule | P&L threshold | Worst case | Rise (paired) |
|---|---|---|---|---|---|---|---|
| adversarial (GDA) | 14.4010 ± 0.0258 | 14.3704 | 14.1369 | 14.2176 | 14.1614 | **14.4010** | +0.0000 ± 0.0000 (+0.0%) |
| trained vs LSM rule | 14.1915 ± 0.0252 | 14.4083 | 14.1800 | 14.1915 | 14.2615 | **14.4083** | +0.2169 ± 0.0070 (+1.5%) |
| trained vs hold-to-maturity | 14.1141 ± 0.0255 | 14.3800 | 14.1141 | 14.2180 | 14.1692 | **14.3800** | +0.2660 ± 0.0041 (+1.9%) |
| no hedge | — | — | 30.4019 | 25.4535 | 30.5975 | 30.5975 | — |

- trained vs LSM rule minus adversarial hedger, each under its own fresh adversary: +0.0379 ± 0.0039 (paired over test paths).
- trained vs hold-to-maturity minus adversarial hedger, each under its own fresh adversary: +0.0096 ± 0.0024 (paired over test paths).

Exercise behaviour of each fresh adversary (E[tau] / early-exercise fraction): adversarial (GDA): 0.477 / 0.130; trained vs LSM rule: 0.481 / 0.112; trained vs hold-to-maturity: 0.476 / 0.138.

Reading:
- **Cost model:** liquidity extension (docs/LIQUIDITY.md): every trade pays a volatility-proportional half-spread plus
  square-root impact at the current (stochastic, non-tradable) liquidity state; the hedge is unwound at exercise
- **Cost model:** C_n = sum_{k<n} c |delta_k - delta_{k-1}| S~_k (initial purchase charged), unwinding the hedge at exercise not charged (spec).
- **Costs raise the price:** p* goes from 13.527 (c = 0) to 14.401 (+6.5%).
- **Pricing understatement persists:** assuming the LSM rule now understates the attacked risk by 1.5% (was 1.0%),
  assuming hold-to-maturity by 1.9% (was 2.5%).
- **Hedging, still small:** under each hedger's own fresh adversary the adversarial hedge is better by 0.038 ± 0.004
  (vs LSM-trained) and 0.010 ± 0.002 (vs maturity-trained), i.e. 0.26% / 0.07%. But on the adversarial hedge the
  fresh adversary falls 0.031 ± 0.003 short of the co-trained GDA adversary, so attacker strength is uncertain by about
  as much as these differences. Taking the strongest attack found for each hedger, the maturity-trained hedge
  (14.380) is not worse than the adversarial one (14.401). No robust evidence that adversarial training improves
  the hedge at c = 0.5%.
- **What costs change:** the hedge becomes path-dependent (visible no-trade band in the hedge-ratio scatter) and the
  LSM-trained hedge departs further from the others below the LSM boundary, but that region is rarely visited by
  the worst-case holder.
- **Caveats:** one seed, lam = 1, alpha = 0.9, reduced compute; the worst case is only as strong as the attackers.

Figures: `figures/headline_loss_hist_c0.005.png`, `figures/headline_hedge_ratio_c0.005.png`.

## Liquidity (volume-risk) extension: sanity checks (`liqcheck.py`, lam = 1, alpha = 0.9)

Model in `docs/LIQUIDITY.md`: stochastic illiquidity (worsens after large declines), volume rising with |return|, EWMA
volatility; each trade pays a volatility-proportional half-spread plus square-root impact. No training here: the
frozen frictionless adversarial hedge (which ignores liquidity) is evaluated under liquidity costs against simple
exercise rules fitted on validation paths (never early, LSM, spot boundary, running-loss threshold, illiquidity
threshold, post-decline).

- Regression: liquidity off -> p* = 13.5265 ± 0.0265 (step-3 value 13.5265): unchanged.

Liquidity model **base** (ell_max = 10, sig_ratio_max = 4): CVaR_0.9 of the frictionless hedge

| Unwind at exercise | cost scale | never early | LSM rule | strongest simple attack | vs best of never-early / LSM |
|---|---|---|---|---|---|
| none | 0 | 13.240 | 13.422 | spot boundary 13.492 | +0.070 (+0.5%) |
| none | 0.5 | 13.920 | 13.803 | spot boundary 14.049 | +0.129 (+0.9%) |
| none | 1 | 14.653 | 14.218 | spot boundary 14.721 | +0.068 (+0.5%) |
| none | 2 | 16.237 | 15.139 | spot boundary 16.270 | +0.033 (+0.2%) |
| none | 4 | 19.659 | 17.235 | spot boundary 19.660 | +0.001 (+0.0%) |
| cash | 0 | 13.240 | 13.422 | spot boundary 13.492 | +0.070 (+0.5%) |
| cash | 0.5 | 14.179 | 14.378 | P&L threshold 14.951 | +0.573 (+4.0%) |
| cash | 1 | 15.268 | 15.517 | P&L threshold 18.198 | +2.681 (+17.3%) |
| cash | 2 | 17.676 | 18.005 | P&L threshold 25.453 | +7.448 (+41.4%) |
| cash | 4 | 22.878 | 23.419 | P&L threshold 41.164 | +17.745 (+75.8%) |

Liquidity model **mild** (ell_max = 5, sig_ratio_max = 3): CVaR_0.9 of the frictionless hedge

| Unwind at exercise | cost scale | never early | LSM rule | strongest simple attack | vs best of never-early / LSM |
|---|---|---|---|---|---|
| none | 0 | 13.240 | 13.422 | spot boundary 13.492 | +0.070 (+0.5%) |
| none | 0.5 | 13.611 | 13.674 | spot boundary 13.799 | +0.125 (+0.9%) |
| none | 1 | 13.990 | 13.932 | spot boundary 14.133 | +0.143 (+1.0%) |
| none | 2 | 14.774 | 14.468 | spot boundary 14.858 | +0.084 (+0.6%) |
| none | 4 | 16.421 | 15.613 | spot boundary 16.459 | +0.038 (+0.2%) |
| cash | 0 | 13.240 | 13.422 | spot boundary 13.492 | +0.070 (+0.5%) |
| cash | 0.5 | 13.763 | 13.899 | spot boundary 14.026 | +0.127 (+0.9%) |
| cash | 1 | 14.329 | 14.421 | P&L threshold 14.821 | +0.400 (+2.8%) |
| cash | 2 | 15.573 | 15.585 | P&L threshold 17.390 | +1.804 (+11.6%) |
| cash | 4 | 18.329 | 18.178 | P&L threshold 24.143 | +5.814 (+31.7%) |

Reading:
- **Plumbing checks pass:** liquidity off reproduces p* exactly; cost scale 0 reproduces the frictionless numbers;
  CVaR rises monotonically with the cost scale.
- **Without an unwind charge, liquidity does not help the holder:** exercising during illiquidity ends the seller's
  costly re-hedging, so the strongest simple attack stays the spot boundary and the liquidity-timed rules are weak.
- **With a cash-settled unwind, the holder gains a real lever** against a hedge that ignores liquidity: the strongest
  attack exercises when the seller's loss-if-exercised-now (which includes the unwind at current liquidity) is high.
  Its size depends on how severe stress liquidity is: about +17% over never-early/LSM with the base caps (spreads up
  to 40x normal) at cost scale 1, about +3% with the milder caps (up to 15x), growing quickly with the cost scale.
  At cost scale 0 (no liquidity cost) the same comparison gives +0.5%.
- **What this does not show yet:** whether a liquidity-aware hedger trained for a fixed exercise rule stays exploitable,
  and whether adversarial training removes the gap. That is the headline comparison, still to run.


## Pre-check: band-rule hedgers under fixed vs proportional costs (lam = 1, alpha = 0.9)

No networks trained. A band hedger follows a target hedge and trades only when |delta_{n-1} - Delta_n| > b:
fixed cost kappa per trade -> trade back to the target; proportional cost c -> trade to the band edge (control).
b is tuned on 2^17 validation paths under three exercise assumptions (hold to maturity, LSM rule, worst of a
simple attack family: never early, LSM, running-loss threshold, spot boundary, hedge staleness), then the tuned
hedgers are cross-evaluated on the 2^18 test paths. 'Loss' = extra worst-case CVaR from tuning b for the wrong
assumption (paired over test paths). Simple attacks only: a lower bound on a learned adversary.

Target: **Merton delta**

| Costs | b* maturity / LSM / worst | trades at b* (maturity / worst) | worst-case CVaR at b*_worst | loss if tuned for maturity | loss if tuned for LSM | strongest attack |
|---|---|---|---|---|---|---|
| fixed kappa = 0.005 | 0.36 / 0.36 / 0.36 | 2.1 / 2.1 | 15.5981 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |
| fixed kappa = 0.01 | 0.36 / 0.36 / 0.36 | 2.1 / 2.1 | 15.6070 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |
| fixed kappa = 0.02 | 0.36 / 0.36 / 0.36 | 2.1 / 2.1 | 15.6202 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |
| fixed kappa = 0.05 | 0.36 / 0.36 / 0.36 | 2.1 / 2.1 | 15.6758 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |
| proportional c = 0.005 | 0.10 / 0.10 / 0.10 | 21.2 / 21.2 | 16.2338 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |

Target: **frictionless NN hedge**

| Costs | b* maturity / LSM / worst | trades at b* (maturity / worst) | worst-case CVaR at b*_worst | loss if tuned for maturity | loss if tuned for LSM | strongest attack |
|---|---|---|---|---|---|---|
| fixed kappa = 0.005 | 0.04 / 0.04 / 0.04 | 17.4 / 17.4 | 13.5901 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |
| fixed kappa = 0.01 | 0.04 / 0.04 / 0.04 | 17.4 / 17.4 | 13.6594 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |
| fixed kappa = 0.02 | 0.06 / 0.06 / 0.06 | 11.1 / 11.1 | 13.7594 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |
| fixed kappa = 0.05 | 0.08 / 0.08 / 0.08 | 7.8 / 7.8 | 13.9907 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |
| proportional c = 0.005 | 0.04 / 0.04 / 0.04 | 23.8 / 23.8 | 14.2433 | +0.0000 ± 0.0000 (+0.00%) | +0.0000 ± 0.0000 (+0.00%) | spot boundary |

Time-varying band b_n = b0 (1 + beta t_n / T), target = frictionless NN hedge; grid b0 in [0.01, 0.3] x
beta in {-0.75, 0, 1, 3, 9} (beta > 0: the band widens towards maturity, i.e. trade less as the remaining life shrinks).

| Costs | (b0, beta)* maturity | (b0, beta)* LSM | (b0, beta)* worst | trades at worst-tuned | loss if tuned for maturity | loss if tuned for LSM |
|---|---|---|---|---|---|---|
| fixed kappa = 0.02 | (0.055, 0) | (0.055, 0) | (0.055, 0) | 12.4 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| fixed kappa = 0.05 | (0.089, 0) | (0.089, 0) | (0.089, 0) | 6.8 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |

Reading:
- **No horizon effect within these families.** With the NN target, the tuned band width is identical under hold-to-
  maturity, the LSM rule and the worst simple attack at every cost level (parabolic interpolation puts the three
  optima within 0.003 of each other), so tuning for the wrong assumption costs nothing measurable.
- **The test has power.** The same procedure moves b* with the cost level (0.04 -> 0.09 as kappa goes 0.005 -> 0.05,
  trades 17 -> 8), and the CVaR curves have clear minima (0.02-0.08 CVaR within one grid step).
- **The assumption shifts the level, not the shape.** The three CVaR-vs-b curves are near-parallel vertical shifts
  (LSM about +0.1-0.15, worst attack about +0.2 above maturity): the exercise assumption changes the risk number
  but not where the best band sits - the same pattern as the neural experiments.
- **Time-varying bands do not change this.** Allowing the band to widen or shrink towards maturity, the best beta is 0
  (constant band) under all three assumptions, and all three rank beta identically.
- **Staleness is not what the adversary exploits:** the strongest simple attack is always the spot boundary
  (exercise once S falls below a level), never 'exercise when the band hedge is most out of date'.
- **Merton-delta target is degenerate** (b* about 0.35, ~2 trades, regardless of kappa): the Merton delta is a poor
  CVaR target, so the rule prefers an almost static hedge; it is uninformative about horizon effects.
- **Proportional control** behaves the same way, consistent with the neural null result.
- **Caveats:** simple attacks only; band families are time- but not state-dependent; one target hedge; lam = 1,
  alpha = 0.9. A richer (e.g. neural, gated) fixed-cost hedger could still differ, but these families give no sign of it.

Figures: `figures/bandcheck_merton.png`, `figures/bandcheck_nn.png`.

## Dual upper bounds for a frozen hedge (martingale duality, `dual.py`) — validation

For a frozen hedge h: CVaR under any exercise rule <= W(h) = sup_tau CVaR_alpha(L_tau) <= D(h) =
min_c { c + E[max_n ((L_n - c)^+ - M^c_n)] / (1 - alpha) } (alpha = 0: E[max_n (L_n - M_n)]). M^c is built from
the simulator's shocks with coefficients from a network of F_{n-1}-measurable features; c* is chosen on
independent selection paths and D(h) is evaluated on the 2^18 test paths, so it is an unbiased estimate of a
valid upper bound. 'Hindsight' is the same bound with M = 0, i.e. CVaR_alpha(max_n L_n).

Martingale: 'linear' = network coefficients on the shock basis (`dual.py`); 'value fn' = Andersen-Broadie
increments of a fitted Snell-envelope approximation with stratified antithetic one-step inner draws
(`dual_value.py`). The fitted rule at c* (Longstaff-Schwartz on (L - c*)^+) is included among the lower bounds.

| Run | martingale | hedge | lam | alpha | best lower bound (rule) | D(h) | hindsight (M = 0) | D − lower | c* | max \|z\| of dM | reference |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dual_v1_nohedge_lam0_a0 | linear | none | 0 | 0 | 5.0061 ± 0.0118 (LSM rule) | **5.0557 ± 0.0045** | 9.0352 | +0.0495 (+0.99%) | — | 1.73 | LSM price 4.994, CRR Bermudan 5.006 |
| dual_v1_nohedge_lam0_a0_vf | value fn | none | 0 | 0 | 5.0061 ± 0.0118 (LSM rule) | **5.0608 ± 0.0008** | 9.0352 | +0.0547 (+1.09%) | — | 1.85 | LSM price 4.994, CRR Bermudan 5.006 |
| dual_v2_gdahedge_lam1_a0 | linear | gda_lam1_a0p9_s1234_heuristic_reduced | 1 | 0 | 6.5574 ± 0.0069 (LSM rule) | **6.6269 ± 0.0067** | 7.5654 | +0.0696 (+1.06%) | — | 2.81 |  |
| dual_v2_nohedge_lam1_a0 | linear | none | 1 | 0 | 6.5805 ± 0.0174 (LSM rule) | **6.6227 ± 0.0070** | 10.9488 | +0.0422 (+0.64%) | — | 1.88 | LSM price 6.546 |
| dual_v2_nohedge_lam1_a0_vf | value fn | none | 1 | 0 | 6.5805 ± 0.0174 (LSM rule) | **6.5929 ± 0.0009** | 10.9488 | +0.0124 (+0.19%) | — | 1.55 | LSM price 6.546 |
| dual_grid_lam0_a0p9_a0p9_vf | value fn | gda_lam0_a0p9_s1234_heuristic_liqbase_uwcash_grid | 0 | 0.9 | 7.4597 ± 0.0035 (fitted rule at c* (LSM on (L - c*)^+)) | **7.8182 ± 0.0014** | 7.6171 | +0.3586 (+4.81%) | 7.251 | 2.82 |  |
| dual_grid_lam0p5_a0p9_a0p9_vf | value fn | gda_lam0p5_a0p9_s1234_heuristic_liqbase_uwcash_grid | 0.5 | 0.9 | 13.0651 ± 0.0305 (fitted rule at c* (LSM on (L - c*)^+)) | **13.4023 ± 0.0039** | 14.1030 | +0.3372 (+2.58%) | 9.847 | 2.19 |  |
| dual_v4_lsm_liqbase_a0p9_vf | value fn | fixed-lsm_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced | 1 | 0.9 | 15.3288 ± 0.0279 (fitted rule at c* (LSM on (L - c*)^+)) | **15.6769 ± 0.0039** | 16.1613 | +0.3481 (+2.27%) | 12.038 | 3.10 |  |
| dual_v4_lsm_frictionless_a0p9_vf | value fn | fixed-lsm_lam1_a0p9_s1234_heuristic_reduced | 1 | 0.9 | 13.5398 ± 0.0265 (fresh adversary) | **13.7524 ± 0.0037** | 13.9293 | +0.2126 (+1.57%) | 10.840 | 2.51 |  |
| dual_v5_fixed-lsm_liqbase_s2_a0p9_vf | value fn | fixed-lsm_lam1_a0p9_s2_heuristic_liqbase_uwcash_reduced | 1 | 0.9 | 15.4490 ± 0.0281 (fitted rule at c* (LSM on (L - c*)^+)) | **15.8541 ± 0.0043** | 16.3653 | +0.4051 (+2.62%) | 11.708 | 3.41 |  |
| dual_v5_fixed-lsm_liqbase_s3_a0p9_vf | value fn | fixed-lsm_lam1_a0p9_s3_heuristic_liqbase_uwcash_reduced | 1 | 0.9 | 15.3397 ± 0.0283 (fitted rule at c* (LSM on (L - c*)^+)) | **15.7173 ± 0.0040** | 16.1436 | +0.3776 (+2.46%) | 12.150 | 3.36 |  |
| dual_grid_lam1_a0p9_a0p9_vf | value fn | gda_lam1_a0p9_s1234_heuristic_liqbase_uwcash_grid | 1 | 0.9 | 15.2304 ± 0.0281 (fitted rule at c* (LSM on (L - c*)^+)) | **15.6814 ± 0.0041** | 16.2059 | +0.4510 (+2.96%) | 12.233 | 2.82 |  |
| dual_v3_gda_liqbase_a0p9 | linear | gda_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced | 1 | 0.9 | 15.1234 ± 0.0275 (co-trained adversary) | **15.5499 ± 0.0242** | 16.0065 | +0.4265 (+2.82%) | 12.117 | 3.38 |  |
| dual_v3_gda_liqbase_a0p9_vf | value fn | gda_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced | 1 | 0.9 | 15.1234 ± 0.0275 (co-trained adversary) | **15.5215 ± 0.0041** | 16.0065 | +0.3981 (+2.63%) | 12.031 | 2.80 |  |
| dual_v3_gda_liqbase_w64_a0p9_vf | value fn | gda_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced | 1 | 0.9 | 15.1234 ± 0.0275 (co-trained adversary) | **15.6590 ± 0.0044** | 16.0065 | +0.5356 (+3.54%) | 12.031 | 2.71 |  |
| dual_v3_gda_frictionless_a0p9 | linear | gda_lam1_a0p9_s1234_heuristic_reduced | 1 | 0.9 | 13.5265 ± 0.0265 (co-trained adversary) | **13.7965 ± 0.0251** | 13.8393 | +0.2700 (+2.00%) | 10.777 | 2.35 |  |
| dual_v3_gda_frictionless_a0p9_vf | value fn | gda_lam1_a0p9_s1234_heuristic_reduced | 1 | 0.9 | 13.5265 ± 0.0265 (co-trained adversary) | **13.7450 ± 0.0038** | 13.8393 | +0.2185 (+1.62%) | 10.823 | 2.51 |  |
| dual_v3_gda_frictionless_w64_a0p9_vf | value fn | gda_lam1_a0p9_s1234_heuristic_reduced | 1 | 0.9 | 13.5265 ± 0.0265 (co-trained adversary) | **13.7344 ± 0.0028** | 13.8393 | +0.2079 (+1.54%) | 10.823 | 2.57 |  |
| dual_v5_gda_liqbase_s2_a0p9_vf | value fn | gda_lam1_a0p9_s2_heuristic_liqbase_uwcash_reduced | 1 | 0.9 | 15.1730 ± 0.0270 (co-trained adversary) | **15.5953 ± 0.0041** | 16.0018 | +0.4223 (+2.78%) | 12.147 | 2.90 |  |
| dual_v5_gda_liqbase_s3_a0p9_vf | value fn | gda_lam1_a0p9_s3_heuristic_liqbase_uwcash_reduced | 1 | 0.9 | 15.1251 ± 0.0273 (co-trained adversary) | **15.5443 ± 0.0041** | 15.9983 | +0.4191 (+2.77%) | 12.132 | 3.07 |  |
| dual_grid_lam0_a0p95_a0p95_vf | value fn | gda_lam0_a0p95_s1234_heuristic_liqbase_uwcash_grid | 0 | 0.95 | 7.7929 ± 0.0039 (fitted rule at c* (LSM on (L - c*)^+)) | **8.2042 ± 0.0019** | 7.9290 | +0.4114 (+5.28%) | 7.618 | 3.22 |  |
| dual_grid_lam0p5_a0p95_a0p95_vf | value fn | gda_lam0p5_a0p95_s1234_heuristic_liqbase_uwcash_grid | 0.5 | 0.95 | 15.0882 ± 0.0372 (fitted rule at c* (LSM on (L - c*)^+)) | **15.7238 ± 0.0052** | 15.9427 | +0.6355 (+4.21%) | 12.299 | 1.85 |  |
| dual_grid_lam1_a0p95_a0p95_vf | value fn | gda_lam1_a0p95_s1234_heuristic_liqbase_uwcash_grid | 1 | 0.95 | 17.3524 ± 0.0383 (fitted rule at c* (LSM on (L - c*)^+)) | **18.1000 ± 0.0058** | 18.2454 | +0.7476 (+4.31%) | 14.475 | 3.10 |  |

Per-hedge certificates (alpha > 0). 'Own estimate' is what the seller reports: p* under the co-trained
adversary for the GDA hedge, CVaR under the LSM exercise rule for the LSM-trained hedge. The worst-case range
takes the strongest lower bound and the lowest D over the runs for that hedge (a mild selection effect).

| Hedge | alpha | own estimate | certified worst case W(h) | understatement of own estimate |
|---|---|---|---|---|
| fixed-lsm_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced | 0.9 | 14.650 | [15.329, 15.677] (lower: fitted rule at c* (LSM on (L - c*)^+)) | +4.63% to +7.01% |
| fixed-lsm_lam1_a0p9_s1234_heuristic_reduced | 0.9 | 13.410 | [13.540, 13.752] (lower: fresh adversary) | +0.97% to +2.55% |
| fixed-lsm_lam1_a0p9_s2_heuristic_liqbase_uwcash_reduced | 0.9 | 14.659 | [15.449, 15.854] (lower: fitted rule at c* (LSM on (L - c*)^+)) | +5.39% to +8.15% |
| fixed-lsm_lam1_a0p9_s3_heuristic_liqbase_uwcash_reduced | 0.9 | 14.638 | [15.340, 15.717] (lower: fitted rule at c* (LSM on (L - c*)^+)) | +4.79% to +7.37% |
| gda_lam0_a0p95_s1234_heuristic_liqbase_uwcash_grid | 0.95 | 7.764 | [7.793, 8.204] (lower: fitted rule at c* (LSM on (L - c*)^+)) | +0.37% to +5.67% |
| gda_lam0_a0p9_s1234_heuristic_liqbase_uwcash_grid | 0.9 | 7.433 | [7.460, 7.818] (lower: fitted rule at c* (LSM on (L - c*)^+)) | +0.35% to +5.18% |
| gda_lam0p5_a0p95_s1234_heuristic_liqbase_uwcash_grid | 0.95 | 15.029 | [15.088, 15.724] (lower: fitted rule at c* (LSM on (L - c*)^+)) | +0.40% to +4.62% |
| gda_lam0p5_a0p9_s1234_heuristic_liqbase_uwcash_grid | 0.9 | 12.956 | [13.065, 13.402] (lower: fitted rule at c* (LSM on (L - c*)^+)) | +0.84% to +3.45% |
| gda_lam1_a0p95_s1234_heuristic_liqbase_uwcash_grid | 0.95 | 17.281 | [17.352, 18.100] (lower: fitted rule at c* (LSM on (L - c*)^+)) | +0.41% to +4.74% |
| gda_lam1_a0p9_s1234_heuristic_liqbase_uwcash_grid | 0.9 | 15.166 | [15.230, 15.681] (lower: fitted rule at c* (LSM on (L - c*)^+)) | +0.42% to +3.40% |
| gda_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced | 0.9 | 15.123 | [15.123, 15.521] (lower: co-trained adversary) | +0.00% to +2.63% |
| gda_lam1_a0p9_s1234_heuristic_reduced | 0.9 | 13.527 | [13.527, 13.734] (lower: co-trained adversary) | +0.00% to +1.54% |
| gda_lam1_a0p9_s2_heuristic_liqbase_uwcash_reduced | 0.9 | 15.173 | [15.173, 15.595] (lower: co-trained adversary) | +0.00% to +2.78% |
| gda_lam1_a0p9_s3_heuristic_liqbase_uwcash_reduced | 0.9 | 15.125 | [15.125, 15.544] (lower: co-trained adversary) | +0.00% to +2.77% |

Reading:
- **Item 1 (alpha = 0, lam = 0).** The bound comes within 1.0-1.1% of the CRR Bermudan price (5.056 linear, 5.061
  value fn, vs 5.006). The diagnostics in `results/diagnostics/dual/` show where the gap comes from. With 64 instead
  of 16 antithetic pairs the value-fn bound barely moves (5.058). With a wider, longer-trained value network it
  drops to 5.035 (+0.6%). So the remaining gap is the quality of the value-function fit, not inner-sampling noise.
- **Item 2 (alpha = 0, lam = 1).** The hedge-only penalty (M = the GDA hedge's gains, i.e. hindsight with that
  hedge) is 7.565, which reproduces the brief's 7.55. Learned martingales that use the jump shocks bring this down to
  6.62 (linear) and 6.593 (value fn). That is 0.2% above the LSM rule on the same paths (6.580) and 0.7% above the
  LSM price on independent paths (6.546).
- **Item 3 (monotonicity).** D(h) is above every lower bound in every run.
- **Martingale property.** All per-date means of the increments are within 3.4 SE of zero (50 dates per run).
  The unit tests check exactness by construction.
- **New CVaR certificate (alpha = 0.9, GDA hedges).** For the frozen hedge, W(h) lies in [13.527, 13.734]
  frictionless (width 1.5%) and in [15.123, 15.522] under liquidity 'base' + cash unwind (width 2.6%). Both
  improve on the hindsight bound (13.839, 16.007).
- **Where the width comes from.** The rule implied by the fitted value function at c* gives 13.51 and 15.09-15.10.
  That is close to, but below, the GDA adversaries, so nothing suggests the adversaries leave value on the table.
  As at alpha = 0, most of the width is probably the value fit.
- **Fit-to-fit variability.** It is about 0.1 at alpha = 0.9 under liquidity: the wider network did worse there
  (15.659). Each D is a valid bound on its own; choosing among them on the test paths adds a selection effect, but
  the differences are far larger than the SEs (0.003-0.025).
- **Theory.** The sup/min swap is exact for pure stopping times here (Belomestny & Kraetschmer 2016), so the
  bracket can close completely with better martingales.
- **LSM-trained hedges (the seller who assumes LSM exercise).** Her own estimate is understated by +4.6% to +7.0%
  under liquidity (W in [15.329, 15.677] vs 14.650 believed) and by +1.0% to +2.6% frictionless ([13.540, 13.752]
  vs 13.410). Under liquidity the fitted rule at c* (Longstaff-Schwartz on (L - c*)^+) is a new, slightly stronger
  attack on this hedge than the fresh adversary (15.329 vs 15.303).
- **Certified contrast (liquidity).** The LSM seller's estimate is off by at least 4.6%; the GDA seller's by at most
  2.6%. The intervals do not overlap, so this holds whatever the remaining slack.
- **Not certified yet: the hedge ranking.** The worst-case ranges overlap: under liquidity GDA [15.123, 15.522] vs
  LSM-trained [15.329, 15.677], and frictionless [13.527, 13.734] vs [13.540, 13.752]. Certifying 'GDA hedge is
  better' under liquidity needs D(h_GDA) < 15.329. Frictionless, the two hedges are statistically indistinguishable,
  as the saddle-point argument predicts.
- **Grid cells (alpha = 0.9 and 0.95; GDA hedges trained 600 iterations).** The rule fitted at c* beats the grid's
  co-trained adversary in all six cells, by +0.35% to +0.84% (largest at lam = 0.5, alpha = 0.9). So the grid p*
  values are slightly understated, within the 2% exploitability tolerance. The headline runs (1500 iterations) are
  not beaten. The certified ceilings are wider in the grid: D is +3.4% to +5.7% above p*, widest at lam = 0 and at
  alpha = 0.95. Grid p* values are therefore known to be at least 0.4-0.8% and at most 3.4-5.7% below each hedge's
  true worst case.


## Volume caps (participation limit, `capcheck.py`): exposure of the existing hedges (no training)

Cap: |delta_n − delta_(n−1)| <= 0.1 × v_n × dt × 252 / psi (unwind at exercise exempt);
65536 validation paths. The hedges were trained without caps, so the CVaR columns show the mechanical
effect of imposing the cap (and, separately, of the larger position's impact costs), not a capped optimum.

| Hedge | trades over cap, psi = 1 (initial / later) | psi = 0.05 | CVaR scenario | never early | LSM rule | co-trained adversary |
|---|---|---|---|---|---|---|
| GDA (liquidity base) | 100% / 0.00% | 0.00% | psi=0.05, no cap | 14.826 | 14.772 | 15.191 |
|  |  |  | psi=1, no cap | 17.623 | 17.921 | 20.070 |
|  |  |  | psi=1, cap | 17.697 | 17.999 | 20.111 |
| LSM-trained (liquidity base) | 100% / 0.00% | 0.00% | psi=0.05, no cap | 14.595 | 14.692 | — |
|  |  |  | psi=1, no cap | 16.796 | 17.842 | — |
|  |  |  | psi=1, cap | 16.849 | 17.915 | — |
| maturity-trained (liquidity base) | 100% / 0.00% | 0.00% | psi=0.05, no cap | 14.276 | 14.835 | — |
|  |  |  | psi=1, no cap | 16.541 | 18.386 | — |
|  |  |  | psi=1, cap | 16.654 | 18.477 | — |
| GDA (frictionless) | 100% / 0.36% | 0.00% | psi=0.05, no cap | 13.314 | 13.483 | 13.597 |
|  |  |  | psi=1, no cap | 13.314 | 13.483 | 13.597 |
|  |  |  | psi=1, cap | 13.492 | 13.678 | 13.797 |

Reading:
- **Default position size: the cap never binds.** With psi = 0.05, a 10% participation cap allows about 5 shares
  per option per step, far above any trade the hedges make (0.00% of trades).
- **Large position (psi = 1): the cap binds only on the initial hedge.** The cap is about 0.25 shares per option per
  step at normal volume. Building the initial hedge (about -0.5 for the at-the-money put) exceeds it on every path,
  so it takes two dates. Later trades almost never hit it (under 0.001% for the liquidity-trained hedges, 0.36% for
  the frictionless GDA hedge). Volume rises with |returns|, so the cap loosens exactly when the hedge moves most.
- **Impact costs matter far more than the cap.** Imposing the cap on these frozen hedges adds only 0.04-0.20 CVaR.
  Moving from psi = 0.05 to psi = 1 without a cap adds 2.2-3.6 under fixed rules and 4.9 under the GDA hedge's
  own adversary, because square-root impact scales with sqrt(psi).
- **Implication.** With volume-linked caps in this calibration, a large seller is constrained through impact costs.
  Caps would bite only at much lower participation (a few percent) or if caps tightened in stress, which is the
  opposite of what the volume evidence says. These hedges were trained without caps; capped training is not run yet.


## Seed robustness of the liquidity headline (lam = 1, alpha = 0.9, liquidity 'base', cash unwind)

Same settings and compute for every seed; the seed changes network initialisation and training paths (the test
set is fixed). 'Worst found' is the strongest attack on that hedge (fresh adversary or simple rules); 'surprise'
is the LSM-trained seller's worst found / own estimate − 1. 'Edge' columns are paired differences under each
hedge's own fresh adversary (LSM-trained minus GDA, maturity-trained minus GDA). Dual: certified range of the
worst case W(h) from `dual_value.py` where run.

| seed | GDA p* (fresh attack) | LSM-trained: own / worst found | surprise | maturity-trained: own / worst found | edge vs LSM-trained | edge vs maturity-trained | dual W(h_GDA) | dual W(h_LSM) | certified surprise |
|---|---|---|---|---|---|---|---|---|---|
| 1234 | 15.123 (15.036) | 14.650 / 15.303 | +4.5% | 14.232 / 15.893 | +0.266 ± 0.006 | +0.857 ± 0.009 | [15.123, 15.521] | [15.329, 15.677] | +4.6% to +7.0% |
| 2 | 15.173 (15.078) | 14.659 / 15.410 | +5.1% | 14.258 / 15.846 | +0.333 ± 0.010 | +0.768 ± 0.009 | [15.173, 15.595] | [15.449, 15.854] | +5.4% to +8.2% |
| 3 | 15.125 (15.027) | 14.638 / 15.312 | +4.6% | 14.234 / 15.920 | +0.285 ± 0.007 | +0.893 ± 0.009 | [15.125, 15.544] | [15.340, 15.717] | +4.8% to +7.4% |
| mean ± sd (3 seeds) | 15.140 ± 0.028 | 14.649 ± 0.011 / 15.342 ± 0.060 | +4.7% ± 0.3% | 14.242 ± 0.015 / 15.886 ± 0.038 | 0.295 ± 0.034 | 0.839 ± 0.064 | | | |

Reading:
- **The headline is stable across seeds.** Every risk number moves by at most 0.11 (0.7%) between seeds. For
  example, GDA p* is 15.12-15.17 (sd 0.03) and the LSM-trained seller's own estimate is 14.64-14.66.
- **The LSM-trained seller's surprise holds in every seed.** Her worst found case is 4.5-5.1% above her own
  estimate. The certified range is 4.6-7.0%, 5.4-8.2% and 4.8-7.4%. The GDA seller's estimate is within 2.6-2.8% of
  its certified worst case in every seed, so the certified contrast holds for all three seeds.
- **The hedge ranking is robust to the seed, though still not certified.** Under each hedge's own fresh adversary
  the LSM-trained hedge is worse by 0.27-0.33 (paired SE about 0.01). By worst case found it is worse by 1.2-1.6%
  in every seed. Both are far larger than the seed-to-seed spread (sd about 0.03). The dual brackets still overlap
  (D(h_GDA) 15.52-15.60 vs certified lower bounds 15.33-15.45 for the LSM-trained hedge).
- **The maturity-trained hedge is worse than GDA by 0.77-0.89 in every seed.**
- **A stronger attack.** In every seed, the rule fitted at c* (Longstaff-Schwartz on (L - c*)^+) beats the fresh
  adversary on the LSM-trained hedge. It does not beat the co-trained adversary on the GDA hedge.


## Part 1 — pricing: Bermudan put value under jumps and volume risk (`pricing.py`)

Risk-neutral value sup_tau E^Q[e^{-r tau} E_tau] with E_n = (K - S_n - kappa_n)^+. 'cash' = cash settlement
(kappa = 0, the classical price); 'liquidity' = physical settlement, the holder buying the delivered share at that
date's liquidity (kappa = per-share half-spread + square-root impact). Each value is a certified bracket on the
same 2^17 independent paths: lower = exercise rule fitted by Longstaff-Schwartz (per-date networks), upper =
Andersen-Broadie dual with an exact value-function martingale. Base: S0 = K = 100, T = 0.5, N = 50, r = 3%,
sigma = 20%, lam = 1, m = -0.10, delta = 0.15; liquidity 'base', psi = 0.05.

Lower end = max(fitted rule, European), since never exercising early is also a rule; the European value is
the Merton closed form (cash) or the closed form plus a 2^20-path control-variate estimate of the liquidity
cost at maturity (physical; SE ~0.0005). Ranges for the early-exercise premium (price − European) and for the
liquidity discount (cash price − physical price) follow from the brackets.

| case | settlement | S0 | lam | m | delta | psi / cost scale / ell_max | price bracket | width | European | early-exercise premium | early-exercise frac | liquidity discount vs cash | reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cash_S0110 | cash | 110 | 1 | -0.1 | 0.15 | — | [3.475, 3.492] | 0.017 | 3.416 | [0.059, 0.076] | 0.20 |  | Merton Eur. 3.416 |
| cash_S090 | cash | 90 | 1 | -0.1 | 0.15 | — | [11.940, 12.000] | 0.060 | 11.663 | [0.277, 0.337] | 0.63 |  | Merton Eur. 11.663 |
| cash_delta0p05 | cash | 100 | 1 | -0.1 | 0.05 | — | [5.758, 5.786] | 0.028 | 5.621 | [0.137, 0.165] | 0.39 |  | Merton Eur. 5.621 |
| cash_delta0p25 | cash | 100 | 1 | -0.1 | 0.25 | — | [7.733, 7.758] | 0.025 | 7.655 | [0.078, 0.104] | 0.36 |  | Merton Eur. 7.655 |
| cash_delta0p35 | cash | 100 | 1 | -0.1 | 0.35 | — | [9.109, 9.137] | 0.029 | 9.074 | [0.035, 0.064] | 0.33 |  | Merton Eur. 9.074 |
| cash_lam0 | cash | 100 | 0 | -0.1 | 0.15 | — | [4.998, 5.043] | 0.046 | 4.882 | [0.115, 0.161] | 0.39 |  | CRR 5.006; Merton Eur. 4.882 |
| cash_lam0p5 | cash | 100 | 0.5 | -0.1 | 0.15 | — | [5.804, 5.828] | 0.024 | 5.678 | [0.127, 0.150] | 0.38 |  | Merton Eur. 5.678 |
| cash_lam1 | cash | 100 | 1 | -0.1 | 0.15 | — | [6.564, 6.586] | 0.022 | 6.431 | [0.133, 0.155] | 0.38 |  | Merton Eur. 6.431 |
| cash_lam2 | cash | 100 | 2 | -0.1 | 0.15 | — | [7.952, 7.979] | 0.026 | 7.812 | [0.140, 0.167] | 0.38 |  | Merton Eur. 7.812 |
| cash_lam4 | cash | 100 | 4 | -0.1 | 0.15 | — | [10.284, 10.304] | 0.020 | 10.132 | [0.152, 0.172] | 0.41 |  | Merton Eur. 10.132 |
| cash_m0 | cash | 100 | 1 | 0 | 0.15 | — | [6.166, 6.190] | 0.024 | 6.088 | [0.078, 0.102] | 0.38 |  | Merton Eur. 6.088 |
| cash_mneg0p05 | cash | 100 | 1 | -0.05 | 0.15 | — | [6.258, 6.280] | 0.023 | 6.149 | [0.108, 0.131] | 0.38 |  | Merton Eur. 6.149 |
| cash_mneg0p2 | cash | 100 | 1 | -0.2 | 0.15 | — | [7.698, 7.720] | 0.022 | 7.527 | [0.171, 0.193] | 0.38 |  | Merton Eur. 7.527 |
| cash_mneg0p3 | cash | 100 | 1 | -0.3 | 0.15 | — | [9.299, 9.318] | 0.019 | 9.097 | [0.202, 0.221] | 0.37 |  | Merton Eur. 9.097 |
| phys_S0110 | physical | 110 | 1 | -0.1 | 0.15 | 0.05 / 1 / 10 | [3.309, 3.405] | 0.096 | 3.266 | [0.043, 0.139] | 0.17 | [0.070, 0.183] (3.6% at mid) |  |
| phys_S090 | physical | 90 | 1 | -0.1 | 0.15 | 0.05 / 1 / 10 | [11.647, 11.827] | 0.180 | 11.387 | [0.260, 0.440] | 0.61 | [0.113, 0.353] (1.9% at mid) |  |
| phys_delta0p05 | physical | 100 | 1 | -0.1 | 0.05 | 0.05 / 1 / 10 | [5.550, 5.635] | 0.085 | 5.431 | [0.119, 0.204] | 0.36 | [0.123, 0.236] (3.1% at mid) |  |
| phys_delta0p25 | physical | 100 | 1 | -0.1 | 0.25 | 0.05 / 1 / 10 | [7.479, 7.609] | 0.130 | 7.435 | [0.044, 0.174] | 0.33 | [0.124, 0.279] (2.6% at mid) |  |
| phys_delta0p35 | physical | 100 | 1 | -0.1 | 0.35 | 0.05 / 1 / 10 | [8.851, 8.987] | 0.135 | 8.851 | [0.000, 0.135] | 0.30 | [0.122, 0.286] (2.2% at mid) |  |
| phys_lam0 | physical | 100 | 0 | -0.1 | 0.15 | 0.05 / 1 / 10 | [4.842, 4.928] | 0.086 | 4.741 | [0.101, 0.186] | 0.37 | [0.070, 0.201] (2.7% at mid) |  |
| phys_lam0p5 | physical | 100 | 0.5 | -0.1 | 0.15 | 0.05 / 1 / 10 | [5.603, 5.679] | 0.076 | 5.500 | [0.103, 0.179] | 0.35 | [0.125, 0.225] (3.0% at mid) |  |
| phys_lam1 | physical | 100 | 1 | -0.1 | 0.15 | 0.05 / 1 / 10 | [6.328, 6.405] | 0.077 | 6.218 | [0.110, 0.187] | 0.35 | [0.159, 0.258] (3.2% at mid) |  |
| phys_lam2 | physical | 100 | 2 | -0.1 | 0.15 | 0.05 / 1 / 10 | [7.654, 7.750] | 0.095 | 7.531 | [0.123, 0.219] | 0.34 | [0.203, 0.325] (3.3% at mid) |  |
| phys_lam4 | physical | 100 | 4 | -0.1 | 0.15 | 0.05 / 1 / 10 | [9.870, 9.995] | 0.125 | 9.732 | [0.137, 0.263] | 0.36 | [0.289, 0.434] (3.5% at mid) |  |
| phys_m0 | physical | 100 | 1 | 0 | 0.15 | 0.05 / 1 / 10 | [5.971, 6.061] | 0.090 | 5.907 | [0.065, 0.155] | 0.35 | [0.105, 0.219] (2.6% at mid) |  |
| phys_mild | physical | 100 | 1 | -0.1 | 0.15 | 0.05 / 1 / 5 | [6.344, 6.431] | 0.087 | 6.251 | [0.093, 0.180] | 0.35 | [0.133, 0.242] (2.9% at mid) |  |
| phys_mneg0p05 | physical | 100 | 1 | -0.05 | 0.15 | 0.05 / 1 / 10 | [6.045, 6.126] | 0.081 | 5.954 | [0.091, 0.172] | 0.35 | [0.132, 0.235] (2.9% at mid) |  |
| phys_mneg0p2 | physical | 100 | 1 | -0.2 | 0.15 | 0.05 / 1 / 10 | [7.397, 7.510] | 0.113 | 7.274 | [0.123, 0.236] | 0.34 | [0.188, 0.323] (3.3% at mid) |  |
| phys_mneg0p3 | physical | 100 | 1 | -0.3 | 0.15 | 0.05 / 1 / 10 | [8.928, 9.089] | 0.160 | 8.814 | [0.114, 0.274] | 0.32 | [0.210, 0.390] (3.2% at mid) |  |
| phys_psi0p01 | physical | 100 | 1 | -0.1 | 0.15 | 0.01 / 1 / 10 | [6.410, 6.491] | 0.081 | 6.315 | [0.094, 0.175] | 0.35 | [0.074, 0.176] (1.9% at mid) |  |
| phys_psi0p25 | physical | 100 | 1 | -0.1 | 0.15 | 0.25 / 1 / 10 | [6.158, 6.240] | 0.082 | 6.008 | [0.151, 0.233] | 0.34 | [0.324, 0.428] (5.7% at mid) |  |
| phys_psi1 | physical | 100 | 1 | -0.1 | 0.15 | 1 / 1 / 10 | [5.876, 5.956] | 0.081 | 5.656 | [0.220, 0.300] | 0.34 | [0.608, 0.710] (10.0% at mid) |  |
| phys_psi1_lam0 | physical | 100 | 0 | -0.1 | 0.15 | 1 / 1 / 10 | [4.493, 4.575] | 0.081 | 4.345 | [0.148, 0.229] | 0.34 | [0.423, 0.550] (9.7% at mid) |  |
| phys_psi1_lam2 | physical | 100 | 2 | -0.1 | 0.15 | 1 / 1 / 10 | [7.107, 7.198] | 0.091 | 6.816 | [0.291, 0.382] | 0.34 | [0.755, 0.872] (10.2% at mid) |  |
| phys_scale0 | physical | 100 | 1 | -0.1 | 0.15 | 0.05 / 0 / 10 | [6.554, 6.621] | 0.067 | 6.431 | [0.123, 0.189] | 0.38 | [-0.057, 0.032] (-0.2% at mid) |  |
| phys_scale0p5 | physical | 100 | 1 | -0.1 | 0.15 | 0.05 / 0.5 / 10 | [6.419, 6.501] | 0.082 | 6.324 | [0.096, 0.177] | 0.35 | [0.063, 0.167] (1.7% at mid) |  |
| phys_scale2 | physical | 100 | 1 | -0.1 | 0.15 | 0.05 / 2 / 10 | [6.158, 6.240] | 0.082 | 6.014 | [0.144, 0.226] | 0.34 | [0.324, 0.428] (5.7% at mid) |  |
| phys_scale4 | physical | 100 | 1 | -0.1 | 0.15 | 0.05 / 4 / 10 | [5.851, 5.930] | 0.079 | 5.640 | [0.211, 0.290] | 0.34 | [0.634, 0.735] (10.4% at mid) |  |

Exercise propensity of the fitted rule (phys_lam1): share of alive states, at dates before maturity,
where the holder exercises, by moneyness and illiquidity (cells with < 200 visited states left blank).

| moneyness | ell < 1.25 | 1.25-2 | 2-4 | > 4 |
|---|---|---|---|---|
| S/K 0.70-0.80 | 26.6% (n = 17,990) | 5.2% (n = 34,663) | 0.2% (n = 23,669) | 0.9% (n = 20,716) |
| S/K 0.80-0.90 | 9.4% (n = 243,969) | 4.3% (n = 87,692) | 0.3% (n = 31,202) | 1.1% (n = 19,458) |
| S/K 0.90-0.97 | 0.8% (n = 925,126) | 0.5% (n = 145,509) | 0.3% (n = 13,621) | 1.3% (n = 5,531) |

Reading:
- **Validation.** Cash, lam = 0: [4.998, 5.043] contains the exact CRR Bermudan value 5.006. Physical with cost
  scale 0 reproduces the cash price (discount range [-0.057, 0.032] contains 0).
- **Precision.** Cash brackets are 0.2-0.9% wide. Physical ones are 1.2-2.9%: the liquidity state adds three
  dimensions to the value function, and the fit is the limiting factor (the bounds' Monte Carlo SEs are about
  0.001).
- **Jumps raise the price, as expected.** At lam = 1 the cash price is 6.56-6.59, against 5.00-5.04 without jumps.
  The early-exercise premium grows with jump intensity and with downside jump size: [0.15, 0.17] at lam = 4,
  [0.20, 0.22] at m = -0.30. It shrinks with jump volatility: [0.035, 0.064] at delta = 0.35. Jump volatility
  adds time value, which favours waiting.
- **Volume risk lowers the price by about 3%** for a small holder (psi = 0.05). At lam = 1 the certified
  discount is [0.16, 0.26]. In absolute terms it grows with jump intensity, from [0.07, 0.20] at lam = 0 to
  [0.29, 0.43] at lam = 4: jumps cause the illiquidity spikes that make exercise costly.
- **Position size dominates.** The discount is about 3% at psi = 0.05, 5.7% at psi = 0.25 and 10% at psi = 1, the
  same as scaling costs by 2 and 4 (impact grows with sqrt(psi)). The milder stress calibration changes little
  (2.9% vs 3.2%), because illiquidity rarely reaches the caps on the dates when holders exercise.
- **Holders wait out illiquidity.** Deep in the money (S/K 0.7-0.8), the fitted rule exercises in 27% of alive
  states when liquidity is normal (ell < 1.25), 5% at ell 1.25-2, and under 1% above 2. Early exercise happens
  almost only in normal liquidity (mean illiquidity at early exercise about 1.2).
- **When liquid, physical holders exercise earlier.** Their effective boundary at normal liquidity sits about 2-4
  above the cash holder's for t < 0.4 (figure; 99% quantile of exercised spots). They exercise while liquidity is
  good, because further declines tend to bring illiquidity. This boundary is read off visited states, so it is
  approximate.
- **The early-exercise right includes a liquidity-timing option.** With large positions the premium rises:
  [0.22, 0.30] at psi = 1, against [0.11, 0.19] at psi = 0.05. The European holder must trade at maturity
  whatever the liquidity; the Bermudan holder can choose a liquid date.

Figures: `figures/pricing_sweeps.png`, `figures/pricing_boundary.png`.

## Step 3: GDA and exploitability (lam = 1)

Reduced compute (agreed): training batch 2^12, 1500 outer iterations (spec: 2^14, 3000); k_adv = 5, adversary pool on, temperature 1 → 0.1, warm-started stopper.
Exploitability attack: fresh stopper (new random init + the same warm start) trained 2000 steps at the same batch
against the frozen hedger. Supplementary attacks (not in the spec) use simple rules on the same frozen hedger;
the P&L-threshold rule is fitted on separate validation paths. Gaps are paired on the test paths.

### alpha = 0.9

p* (trained hedger vs trained stopper) = **13.5265 ± 0.0265**; trained stopper E[tau] = 0.477, early-exercise fraction 0.118; training 92.8 min.
Unhedged seller, same CVaR: 25.4535 under the LSM rule, 30.4019 if never exercised early. LSM price (risk-neutral) = 6.5460.

| Attack on frozen hedger | CVaR | gap vs p* (paired) | gap / p* | E[tau] | early frac |
|---|---|---|---|---|---|
| fresh stopper (spec gap) | 13.5230 ± 0.0265 | -0.0035 ± 0.0016 | -0.03% | 0.474 | 0.137 |
| never early (tau = T) | 13.2399 ± 0.0259 | -0.2866 ± 0.0046 | -2.12% | 0.500 | 0.000 |
| LSM rule | 13.4218 ± 0.0261 | -0.1047 ± 0.0053 | -0.77% | 0.434 | 0.336 |
| P&L threshold (c=11.95, fit on validation) | 13.3515 ± 0.0255 | -0.1750 ± 0.0068 | -1.29% | 0.485 | 0.063 |

Reference hedge (not in spec): Merton European delta, same simple stopping rules, same CVaR:

| Stopping rule | NN hedger | Merton-delta hedger | NN − Merton (paired) |
|---|---|---|---|
| never early | 13.2399 | 15.6271 | -2.3872 ± 0.0271 |
| LSM rule | 13.4218 | 16.0442 | -2.6224 ± 0.0261 |
| P&L threshold | 13.3515 | 16.0892 | -2.7376 ± 0.0272 |

### alpha = 0

p* (trained hedger vs trained stopper) = **6.6014 ± 0.0321**; trained stopper E[tau] = 0.435, early-exercise fraction 0.362; training 86.8 min.
Unhedged seller, same CVaR: 6.5805 under the LSM rule, 6.4443 if never exercised early. LSM price (risk-neutral) = 6.5460.

| Attack on frozen hedger | CVaR | gap vs p* (paired) | gap / p* | E[tau] | early frac |
|---|---|---|---|---|---|
| fresh stopper (spec gap) | 6.6033 ± 0.0321 | +0.0018 ± 0.0010 | +0.03% | 0.436 | 0.351 |
| never early (tau = T) | 6.4733 ± 0.0325 | -0.1281 ± 0.0062 | -1.94% | 0.500 | 0.000 |
| LSM rule | 6.6001 ± 0.0321 | -0.0013 ± 0.0027 | -0.02% | 0.434 | 0.336 |
| P&L threshold (c=19.61, fit on validation) | 6.5053 ± 0.0321 | -0.0961 ± 0.0047 | -1.46% | 0.464 | 0.158 |

Reference hedge (not in spec): Merton European delta, same simple stopping rules, same CVaR:

| Stopping rule | NN hedger | Merton-delta hedger | NN − Merton (paired) |
|---|---|---|---|
| never early | 6.4733 | 6.4373 | +0.0360 ± 0.0292 |
| LSM rule | 6.6001 | 6.5666 | +0.0335 ± 0.0286 |
| P&L threshold | 6.5053 | 6.4901 | +0.0152 ± 0.0287 |

Notes:
- The fresh attacker uses the same warm start and recipe as the GDA adversary, so a small gap says the trained
  stopper is a best response *within this recipe*. The supplementary attacks (different rule families) are all
  weaker than the trained adversary, which is mild extra evidence that it is not a weak adversary.
- alpha = 0: the hedger is irrelevant in expectation (E[G_tau] = 0), so its objective is flat and its positions
  drift (std ~0.75 by the end); this only adds variance (SE 0.032 vs 0.018 unhedged). Paired against the LSM rule
  on the same paths and hedger, p* − LSM rule ≈ 0, i.e. the GDA stopper matches LSM.
- alpha = 0.9: the hedge is flatter than the Merton European delta and nearly time-homogeneous (more short
  stock out of the money, i.e. protection against down-jumps). It beats the Merton-delta hedge by 2.4–2.7 CVaR
  points under every simple stopping rule, so the shape is not obviously an under-training artefact.
- alpha = 0.9 training plateaus within ~50 outer iterations; the adversary briefly stopped exercising early
  (iterations ~50–100) and recovered to ~12% early exercise (LSM: 34%): the seller's worst case is mostly
  holding to maturity, with early exercise used selectively.

Figures: `figures/step3_training.png`, `figures/step3_hedge_ratio.png`.

## Step 2 finding: the spec'd stopper training collapses from a random init

With the stopper randomly initialised (spec), joint training of the relaxed objective
X = sum_n f_n prod_{k<n}(1 - f_k) L_n drives every logit negative within ~50–100 steps: the hard rule never
exercises early and the value is the European price. Mechanism: at init f_n ~ 0.5, so the relaxed rule
stops at t_1–t_3, where stopping is worse than waiting; the shared network lowers all logits; once
f_n(1 - f_n) is tiny everywhere, the (small, rare) positive signal from deep-ITM late states is
exponentially down-weighted and cannot recover (max f over the test set fell to ~1e-30 by step 600).
Reproduced at full scale (batch 2^14, `diagnostics/fullscale_random_init_lam0_partial.csv`, stopped at
step 850 with zero early exercise) and in five small-scale variants (`diagnostics/step2_collapse_smallscale.txt`):
lr 1e-3 / 3e-4 / 1e-4, output-bias init at logit(1/N), and Becker et al.'s per-date objective with hard
continuation applied to all dates at once — all collapse. A supervised warm start does not.

**Fix (adopted 2026-10-02):** `train.stopper_init: heuristic` — 300 Adam steps of BCE towards the rule
"exercise iff S < K and K - S > Merton European put value" (does not use LSM; it is a sub-optimal rule),
then the spec'd relaxed ascent unchanged. Used for every stopper (step 2, GDA adversary, exploitability attacker).

## Step 2 detail (alpha = 0, zero hedge, warm-started stopper, spec compute)

| lam | NN (hard, test) | LSM rule on same paths | NN − LSM (paired) | rel. vs LSM price | rel. vs CRR Berm. | warm-start rule alone | never-exercise (collapse) | E[tau] NN / LSM | early-ex NN / LSM | same tau |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5.0033 ± 0.0118 | 5.0061 ± 0.0118 | -0.0028 ± 0.0019 | +0.20% | -0.05% | 4.9421 | 4.8715 | 0.440 / 0.439 | 0.378 / 0.377 | 0.906 |
| 0.5 | 5.8317 ± 0.0152 | 5.8186 ± 0.0150 | +0.0131 ± 0.0037 | +0.79% | — | 5.7810 | 5.6822 | 0.440 / 0.435 | 0.363 / 0.345 | 0.789 |
| 1 | 6.5885 ± 0.0176 | 6.5805 ± 0.0174 | +0.0080 ± 0.0038 | +0.65% | — | 6.5574 | 6.4443 | 0.437 / 0.434 | 0.354 / 0.336 | 0.826 |

With a zero hedge and alpha = 0 the hedger plays no role (E[G_tau] = 0 for any stopping time under mu = r),
so this isolates the stopper. The paired column compares both rules on identical test paths (small SE).
Both NN and LSM values are lower-bound estimators of the Bermudan price.
Figures: `figures/step2_exercise_region.png`, `figures/step2_training.png`.
