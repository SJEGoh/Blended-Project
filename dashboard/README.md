# Seller vs Holder dashboard

Open `seller_vs_holder_standalone.html` in a browser. Everything is inside the file: the trained networks' weights,
the payoff matrix and the training logs. Paths are simulated live in the browser.

Three views (keys 1, 2, 3):

- **Game**: the seller's hedge and the holder's exercise rule play out on the same path, with the payoff matrix
  (CVaR 90% of the seller's hedged loss on the 2^18-path test set), live score and the GDA training replay.
- **Seller**: market, hedge, tail risk, hedge policy.
- **Holder**: market, exercise decision, certified price brackets, where and when the holder exercised.

Controls: Space play or pause, → step a day, N new path, B play a batch of rounds, S speed.

**Play the holder** (key 5): you exercise against two sellers at once, the GDA hedge and the LSM-trained hedge
(both at CVaR 90%). Each round compares your damage with the attacker trained against that seller on the same path.
The rule builder scores a threshold rule (seller owes at least X, illiquidity at most Y, from day D, optionally only
in the money) on 4,000 fresh paths; "Find the best rule" fits on half of them and scores on the other half.

## Files

| File | What it does |
|---|---|
| `export_data.py` | Exports weights, parameters, the payoff matrix, histograms, logs and parity samples to `dashboard_data.json` (about 3 minutes) |
| `export_alpha_sets.py` | Adds the other risk levels (models, payoff matrices, histograms, logs) to `dashboard_data.json` |
| `engine.js` | JavaScript port of the simulator, networks, rollout and exercise rules |
| `test_parity.mjs` | Checks `engine.js` against the Python outputs on shared shocks (`node dashboard/test_parity.mjs`) |
| `page.html`, `app.js` | Page markup, styles and the dashboard logic |
| `build.py` | Inlines everything into `seller_vs_holder.html` (published page body) and `seller_vs_holder_standalone.html` |

Rebuild after retraining:

```
python dashboard/export_data.py          # alpha = 0.9 models, pricing nets, parity samples
python dashboard/export_alpha_sets.py    # alpha = 0, 0.5, 0.95, 0.99 (needs jobs/run_alpha_lsm.sh first)
node dashboard/test_parity.mjs
python dashboard/build.py
```

The risk-measure bar (game and seller views) switches every trained model to the chosen alpha. At 0.9 these are
the headline models (1,500 iterations). At the other levels the GDA hedge and its attacker come from the grid run
(600 iterations; alpha = 0.99 uses the run with attacker restarts), and `jobs/run_alpha_lsm.sh` trains an LSM-trained
hedge with the same settings plus an attacker against it. The maturity-trained hedge exists only at 0.9.

Models shown: the liquidity headline (lam = 1, alpha = 0.9, base liquidity, cash unwind, seed 1234). Hedges: GDA,
LSM-trained, maturity-trained (checkpoints), Merton delta and no hedge. Holders: the attacker vs GDA (co-trained with the
GDA hedge), the attackers vs the LSM-trained and maturity-trained hedges (each trained against that frozen hedge), the LSM rule and never early. Pricing nets:
`pricing_cash_lam1`, `pricing_phys_lam1`, `pricing_phys_psi1`.
