# Build Spec: Adversarial Deep Hedging of an American Put (MVP)

## Goal

Build a small, reproducible PyTorch research codebase that computes the **seller's risk-based price** of a Bermudan/American put under a Merton jump-diffusion. It uses two networks trained against each other:

- **Hedger (seller, minimizer):** chooses the hedge position at each rebalancing date.
- **Stopper (adversarial holder, maximizer):** chooses when the option is exercised, so as to *maximize* the seller's risk.

The seller's price is

```
p* = inf_{hedge} sup_{stopping rule} CVaR_alpha( Z_tau - G_tau + C_tau )
```

where `Z_tau` is the discounted payoff at exercise, `G_tau` the discounted hedging gains up to exercise, and `C_tau` transaction costs (zero in the MVP baseline).

This is an MVP. **Do not** implement volume caps, amortized/parametric pricing, or latency testing. Leave clean hooks for volume caps only (see "Extension hooks").

## Ground rules

- Python 3.11+, PyTorch, NumPy, SciPy, Matplotlib. CPU must be enough; use GPU if available.
- Everything is driven by one config file (YAML or a dataclass), with fixed seeds.
- Every reported number comes from a **fixed held-out test set** of simulated paths, never from training batches.
- Do not claim a result works unless the corresponding check in "Validation" passes. Report actual numbers with Monte Carlo standard errors.
- If something in this spec is ambiguous or seems wrong, stop and ask rather than silently changing the design.

## 1. Market simulator (`sim.py`)

Merton jump-diffusion under the simulation measure, using exact simulation of log-prices on a uniform grid `t_0 = 0 < t_1 < ... < t_N = T`:

```
log S_{n+1} = log S_n + (mu - lam*kappa - 0.5*sigma^2) dt + sigma*sqrt(dt)*Z_n + sum_{j=1}^{N_n} Y_j
Z_n ~ N(0,1),  N_n ~ Poisson(lam*dt),  Y_j ~ N(m, delta^2),  kappa = exp(m + 0.5*delta^2) - 1
```

Default parameters:

| Param | Value | Param | Value |
|---|---|---|---|
| S0 | 100 | K | 100 |
| T | 0.5 | N (dates) | 50 |
| r | 0.03 | mu | = r (default) |
| sigma | 0.20 | lam | 1.0 |
| m | -0.10 | delta | 0.15 |

- Work in **discounted units** throughout: `S~_n = exp(-r t_n) S_n`, discounted payoff `Z~_n = exp(-r t_n) (K - S_n)^+`.
- `mu = r` is the default so the simulation measure is a pricing measure. This is what makes the alpha = 0 sanity check work (see Validation). Keep `mu` configurable.
- Set `lam = 0` for the pure Black-Scholes case.
- Test set: 2^18 paths, generated once with a fixed seed and saved to disk.

**Simulator unit test:** the Monte Carlo European put price must match the Merton closed-form series (sum over the Poisson jump count of Black-Scholes prices) within 3 standard errors.

## 2. Exercise and hedging conventions

- Exercise dates are `t_1, ..., t_N`. Exercise is forced at `t_N` if it has not happened earlier. No exercise at `t_0`.
- The rebalancing dates are the same grid. The hedge position `delta_n` is held over `[t_n, t_{n+1})`.
- Discounted hedging gains up to date n: `G~_n = sum_{k<n} delta_k (S~_{k+1} - S~_k)`.
- Transaction costs (off by default, `c = 0`): `C_n = sum_{k<n} c * |delta_k - delta_{k-1}| * S~_k`, with `delta_{-1} = 0`. Implement it, but run the baseline with `c = 0`.
- Seller's per-path loss if exercised at n: `L_n = Z~_n - G~_n + C_n`.

## 3. Networks (`models.py`)

Both networks are MLPs **shared across time** with time as an input: 2 hidden layers of width 64, SiLU activations.

**Hedger** `H_theta`
- Inputs: `t_n / T`, `log(S_n / K)`, `delta_{n-1}`.
- Output: `delta_n`, an unbounded linear output.
- Extension hook: route the output through a `constrain_trade(delta_prev, delta_raw)` function that is the identity in the MVP.

**Stopper** `F_phi` (adversarial holder)
- Inputs: `t_n / T`, `log(S_n / K)`, current hedge `delta_{n-1}`, running hedged P&L `Z~_n - G~_n + C_n`, and an "in the money" indicator.
- Output: stopping probability `f_n = sigmoid(logit / temperature)` in (0, 1). Force `f_N = 1`.
- **Non-anticipativity is mandatory.** The stopper's input at date n may only use information available at t_n. Write a unit test that permutes future increments of a path and checks that decisions at earlier dates are unchanged.

**Relaxed stopping (training only)**, following Becker, Cheridito and Jentzen (2019):

```
p_n = f_n * prod_{k<n} (1 - f_k)          # probability of stopping at n; sums to 1
X_path = sum_n p_n * L_n                  # smoothed per-path loss used in training
```

Anneal the temperature from 1.0 down to 0.1 over training so the rule becomes near-deterministic.

**Evaluation always uses the hard rule:** stop at the first n with `f_n > 0.5`. Report hard-rule numbers only.

## 4. Objective and training (`train.py`)

Use the empirical CVaR, which is differentiable:

```
CVaR_alpha(X) = mean of the top ceil((1 - alpha) * B) values of X in the batch
```

For `alpha = 0`, this is just the mean.

**Alternating gradient descent–ascent:**
- Each outer iteration runs `k_adv = 5` stopper steps (gradient **ascent** on CVaR), then 1 hedger step (gradient **descent** on CVaR).
- Batch size 2^14, fresh paths for every step.
- Adam optimizers: stopper lr 1e-3, hedger lr 1e-3, with cosine decay.
- 3000 outer iterations by default.
- **Optional adversary pool (flag, default on):** every 200 iterations, snapshot the stopper. The hedger loss is the maximum of the CVaR against the current stopper and the CVaR against one randomly drawn past snapshot. This guards against cycling.
- Log every 50 iterations: train CVaR, mean exercise time, fraction exercised early, and the mean and standard deviation of the hedge.

**Price:** `p*` is the hard-rule test-set CVaR at the end of training.

## 5. Benchmarks (`benchmarks.py`)

1. **Longstaff–Schwartz (LSM)** on the same Bermudan grid under `mu = r`, with a polynomial basis in moneyness (degree 3). Use separate regression and pricing path sets. Report the price ± standard error and store the implied exercise boundary.
2. **CRR binomial American put** for the `lam = 0` case, with ≥ 2000 steps, as ground truth.
3. **Merton European put** closed form (used by the simulator test).

## 6. Validation (all must be run and reported)

1. **alpha = 0 recovers LSM.** With `mu = r` and `alpha = 0`, the hedging gains have zero mean, so the price reduces to the optimal stopping value. The adversarial price must match LSM within about 1% (and CRR when `lam = 0`). The hedger is irrelevant in this case; note that.
2. **Monotonicity.** The price must be non-decreasing in alpha over `{0, 0.5, 0.9, 0.95, 0.99}` and in jump intensity over `lam ∈ {0, 0.5, 1.0}`.
3. **Exploitability gap (main convergence metric).** Freeze the trained hedger. Train a *fresh* stopper from random initialization against it for 2000 steps. Report:
   ```
   gap = CVaR(fresh stopper) - CVaR(trained stopper)
   ```
   This should be small relative to `p*` (target < 2%). Report it for every alpha.
4. **Why the adversary matters (headline experiment).** Train a hedger against a *fixed* stopper equal to the LSM risk-neutral boundary, not an adversary. Then attack it with a fresh adversarial stopper. Report how much its CVaR rises, compared with the adversarially trained hedger under the same attack.
5. **Non-anticipativity unit test** passes (see Section 3).

## 7. Outputs

Save all figures to `results/figures/` and tables as CSV in `results/`:

- Price vs alpha, one line per `lam`, with the LSM price as a horizontal reference.
- Learned exercise region: a heatmap of hard stop decisions over (t, S), with the LSM boundary overlaid. Use a representative hedge state, and also show the boundary's dependence on running P&L, since this is where the adversary differs from LSM.
- Learned hedge ratio vs time and spot, compared with the Merton European delta as a rough reference.
- Histograms of the seller's terminal loss: adversarial hedger vs no hedge vs the LSM-trained hedger.
- Training curves (CVaR, exercise statistics).
- Exploitability table across alpha.
- `results/RESULTS.md`: a short summary listing every Validation item with pass/fail and its numbers. Report honestly. If something fails, say so and give the likely cause.

## 8. Repo layout

```
adv_american/
  config.yaml
  sim.py          # Merton simulator + Merton European closed form
  models.py       # Hedger, Stopper, constrain_trade hook
  losses.py       # empirical CVaR, relaxed stopping aggregation
  train.py        # GDA loop, adversary pool, checkpoints
  benchmarks.py   # LSM, CRR
  evaluate.py     # hard-rule eval, exploitability, figures, RESULTS.md
  tests/          # simulator, non-anticipativity, CVaR, LSM vs CRR
  run_all.sh      # full experiment grid end to end
  README.md
```

## Extension hooks (do not implement now)

- **Volume caps:** `constrain_trade` will enforce `|delta_n - delta_{n-1}| <= v_n` with stochastic `v_n`. The simulator should be able to return extra per-step state, such as `v_n`, without refactoring.
- **Amortization:** contract parameters (K, T) will later become network inputs. Keep `K` and `T` flowing through the config rather than hard-coded.

## Build order

1. Simulator and its test; LSM and CRR, with a check that LSM matches CRR when `lam = 0`.
2. Stopper only, with a zero hedge and `alpha = 0`: check that it recovers LSM.
3. Add the hedger and GDA at `alpha = 0.9`, then run the exploitability check.
4. Full grid, the LSM-fixed-stopper experiment, figures, and `RESULTS.md`.

Stop and report after steps 2 and 3 before moving on.
