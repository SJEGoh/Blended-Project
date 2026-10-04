# Liquidity / volume-risk extension

Switch on with `--set liquidity.enabled=true` (and usually `costs.unwind=cash`). With it off, the
MVP is unchanged and every earlier number reproduces exactly (checked: p* = 13.5265).

## Why liquidity, not a falling volume cap

The evidence goes the other way from "volume dries up in a crash":

- Trading volume rises with the size of price moves (Karpoff 1987).
- What deteriorates after declines is liquidity: spreads widen and depth falls, asymmetrically after
  market drops (Hameed, Kang & Viswanathan 2010; Chordia, Roll & Subrahmanyam 2001). Funding and market
  liquidity can spiral together (Brunnermeier & Pedersen 2009).
- Spreads track volatility per trade (Wyart, Bouchaud et al. 2008).
- Price impact follows a square-root law, I ≈ Y σ √(Q/V) with Y ≈ 1 (Tóth et al. 2011).

So the friction is a stochastic **cost** of trading, not a hard cap on trade size.

## State (non-tradable, observed at each date t_n; simulated with Euler)

| Variable | Dynamics |
|---|---|
| illiquidity ℓ_n = e^{x_n} | x_{n+1} = min(x_n − κ x_n Δt + η √Δt ε_n + β (−r_n − h)⁺, log ℓ_max), x_0 = 0 |
| relative volume v_n | log v_n = γ (min(\|z_n\|, z_cap) − √(2/π)) + ζ ε'_n − ζ²/2, with z_n = r_n / (σ√Δt) |
| volatility estimate σ̂_n | σ̂²_n = (1 − w) σ̂²_{n−1} + w r_n² / Δt |

Here r_n is the log return into t_n and h = h_mult σ√Δt. Only large declines move illiquidity, so
normal-times liquidity stays at ℓ ≈ 1. Liquidity noise is drawn after the price draws, so price paths
keep common random numbers with the MVP. The saved test set regenerates the liquidity state
deterministically from its prices (seed test_seed + seed_offset).

## Costs

Trading q shares per option at t_n costs, in discounted units:

    [ s0 · ℓ_n · (σ̂_n/σ) · q   +   Y · ℓ_n · (σ̂_n/√252) · √(ψ / v_n) · q^1.5 ] · S̃_n
       half-spread ∝ volatility       square-root impact (ψ = position size / normal daily volume)

The σ̂/σ ratio is capped at sig_ratio_max. Every rebalance pays this cost. With `costs.unwind=cash`,
the hedge is closed at exercise at that date's liquidity (physical settlement is also supported). The
cost is convex in q, so the seller's problem stays convex. The open question is whether the worst-case
exercise rule now depends on the liquidity state.

The hedger and the stopper both observe (log ℓ_n, log v_n, log σ̂_n/σ) as extra inputs.

## Default parameters (illustrative, not estimated)

| Parameter | Default | Meaning |
|---|---|---|
| s0 | 0.0005 | normal half-spread (5 bp) |
| Y | 1.0 | impact prefactor |
| psi | 0.05 | position = 5% of normal daily volume |
| kappa, eta | 20, 1.0 | liquidity half-life about 9 trading days |
| beta_down, h_mult | 15, 2 | a −10% jump raises illiquidity about 2.5×; −20% or worse hits the cap |
| gamma_v, zeta | 0.3, 0.2 | volume rises about 3.5× after a −10% jump |
| ell_max, sig_ratio_max | 10, 4 | stress caps: spread at most 40× normal (`mild`: 5, 3 → 15×) |
| cost_scale | 1 | multiplies s0 and Y |

## Sanity checks (`liqcheck.py`)

- With liquidity off, results are unchanged.
- At cost_scale = 0 the frictionless numbers are recovered.
- Risk rises monotonically with cost_scale.
- Simple liquidity-timed exercise rules are tested against a hedge that ignores liquidity.

Results are in `results/RESULTS.md`.
