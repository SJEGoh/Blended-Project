"""Merton jump-diffusion simulator (exact on the grid) and Merton European closed form.

All path quantities are also returned in *discounted* units:
    S~_n = exp(-r t_n) S_n,      Z~_n = exp(-r t_n) (K - S_n)^+.

Exact log-price step (simulation measure, drift mu):
    log S_{n+1} = log S_n + (mu - lam*kappa - sigma^2/2) dt + sigma sqrt(dt) Z_n + sum_{j<=N_n} Y_j
    N_n ~ Poisson(lam dt),  Y_j ~ N(m, delta^2),  kappa = exp(m + delta^2/2) - 1.
Given N_n, the jump sum is exactly N(N_n m, N_n delta^2), so we draw it as N_n m + delta sqrt(N_n) Z'_n.

Random numbers are always drawn in the fixed order (Z, U, Z') with N_n = F^{-1}_{Poisson(lam dt)}(U),
so two configs that differ only in (mu, lam, m, delta, sigma) share common random numbers and the
jump counts are pathwise monotone in lam.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
from scipy.stats import norm

from config import Config, LiquidityCfg, MarketCfg, add_config_args, config_from_args, lam_tag, liq_of


# --------------------------------------------------------------------------------------
# Path container
# --------------------------------------------------------------------------------------
@dataclass
class PathBatch:
    t: torch.Tensor            # (N+1,) time grid
    S: torch.Tensor            # (B, N+1) spot
    S_disc: torch.Tensor       # (B, N+1) discounted spot
    Z_disc: torch.Tensor       # (B, N+1) discounted put payoff
    # Extension hook: extra per-step state (e.g. stochastic volume caps "v": (B, N)).
    extras: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def B(self) -> int:
        return self.S.shape[0]

    @property
    def N(self) -> int:
        return self.S.shape[1] - 1

    def to(self, device=None, dtype=None) -> "PathBatch":
        cv = lambda x: x.to(device=device, dtype=dtype if x.is_floating_point() else None)
        return PathBatch(cv(self.t), cv(self.S), cv(self.S_disc), cv(self.Z_disc),
                         {k: cv(v) for k, v in self.extras.items()})

    def subset(self, idx) -> "PathBatch":
        return PathBatch(self.t, self.S[idx], self.S_disc[idx], self.Z_disc[idx],
                         {k: v[idx] for k, v in self.extras.items()})


def _poisson_inv_cdf(U: torch.Tensor, rate: float) -> torch.Tensor:
    """Inverse-CDF Poisson draw: N = min{k : U <= F(k)} (exact up to ~1e-30 truncation)."""
    if rate <= 0.0:
        return torch.zeros_like(U)
    kmax = int(math.ceil(rate + 12.0 * math.sqrt(rate) + 12.0))
    pmf = math.exp(-rate)
    cdf = pmf
    count = (U > cdf).to(U.dtype)
    for k in range(1, kmax + 1):
        pmf *= rate / k
        cdf += pmf
        count += (U > cdf).to(U.dtype)
    return count


def build_paths(mc: MarketCfg, logS: torch.Tensor, extras: dict | None = None,
                dtype=torch.float32) -> PathBatch:
    """Assemble a PathBatch from log-spot paths (B, N+1) computed in float64."""
    t = torch.linspace(0.0, mc.T, mc.N + 1, dtype=torch.float64, device=logS.device)
    S = torch.exp(logS)
    disc = torch.exp(-mc.r * t)
    S_disc = S * disc
    Z_disc = torch.clamp(mc.K - S, min=0.0) * disc
    extras = {k: v.to(dtype) for k, v in (extras or {}).items()}
    return PathBatch(t.to(dtype), S.to(dtype), S_disc.to(dtype), Z_disc.to(dtype), extras)


def liquidity_state(lq: LiquidityCfg, mc: MarketCfg, logS: torch.Tensor, gen: torch.Generator) -> dict:
    """Liquidity state at dates 0..N (each (B, N+1)), F_{t_n}-measurable (uses returns up to t_n only).
      x_{n+1} = min(x_n - kappa x_n dt + eta sqrt(dt) eps_n + beta_down (-r_n - h)^+, log ell_max)   (Euler; ell = exp(x))
      log v_n = gamma_v (min(|z_n|, zcap) - sqrt(2/pi)) + zeta eps'_n - zeta^2/2,      z_n = r_n / (sigma sqrt(dt))
      sig_n^2 = (1 - w) sig_{n-1}^2 + w r_n^2 / dt                                     (EWMA volatility)
    with r_n = log S_n - log S_{n-1} and h = h_mult sigma sqrt(dt); x_0 = 0, v_0 = 1, sig_0 = sigma."""
    B, N1 = logS.shape
    N, dt = N1 - 1, mc.dt
    sd = mc.sigma * math.sqrt(dt)
    r = logS.diff(dim=1)
    eps = torch.randn(B, N, generator=gen, dtype=torch.float64).to(logS.device)
    eps_v = torch.randn(B, N, generator=gen, dtype=torch.float64).to(logS.device)
    x = torch.zeros(B, N1, dtype=torch.float64, device=logS.device)
    s2 = torch.full((B, N1), mc.sigma ** 2, dtype=torch.float64, device=logS.device)
    drive = torch.clamp(-r - lq.h_mult * sd, min=0.0)
    x_max = math.log(lq.ell_max)
    for n in range(N):
        x[:, n + 1] = torch.clamp(x[:, n] - lq.kappa * x[:, n] * dt + lq.eta * math.sqrt(dt) * eps[:, n]
                                  + lq.beta_down * drive[:, n], max=x_max)
        s2[:, n + 1] = (1.0 - lq.ewma_w) * s2[:, n] + lq.ewma_w * r[:, n] ** 2 / dt
    logv = torch.zeros(B, N1, dtype=torch.float64, device=logS.device)
    zabs = torch.clamp((r / sd).abs(), max=lq.zcap)
    logv[:, 1:] = lq.gamma_v * (zabs - math.sqrt(2.0 / math.pi)) + lq.zeta * eps_v - 0.5 * lq.zeta ** 2
    return {"ell": torch.exp(x), "vol": torch.exp(logv), "sig": torch.sqrt(s2)}


def simulate(mc: MarketCfg, n_paths: int, generator: torch.Generator | int,
             device="cpu", dtype=torch.float32, liq: LiquidityCfg | None = None) -> PathBatch:
    """Simulate n_paths Merton paths on the uniform grid t_0..t_N (computed in float64)."""
    if isinstance(generator, int):
        generator = torch.Generator(device="cpu").manual_seed(generator)
    N, dt = mc.N, mc.dt
    f64 = torch.float64
    Z = torch.randn(n_paths, N, generator=generator, dtype=f64)
    U = torch.rand(n_paths, N, generator=generator, dtype=f64)
    Zj = torch.randn(n_paths, N, generator=generator, dtype=f64)
    Nj = _poisson_inv_cdf(U, mc.lam * dt)
    jumps = Nj * mc.m + mc.delta * torch.sqrt(Nj) * Zj
    drift = (mc.mu_eff - mc.lam * mc.kappa - 0.5 * mc.sigma**2) * dt
    incr = drift + mc.sigma * math.sqrt(dt) * Z + jumps
    logS = math.log(mc.S0) + torch.cat([torch.zeros(n_paths, 1, dtype=f64), incr.cumsum(1)], 1)
    # Liquidity state is drawn *after* the price draws, so price paths keep common random numbers.
    extras = liquidity_state(liq, mc, logS, generator) if liq is not None else {}
    return build_paths(mc, logS, extras, dtype).to(device)


def simulate_cfg(cfg: Config, n_paths: int, generator, device="cpu", dtype=torch.float32) -> PathBatch:
    """simulate() with the liquidity extension switched on iff cfg.liquidity.enabled."""
    return simulate(cfg.market, n_paths, generator, device, dtype, liq_of(cfg))


# --------------------------------------------------------------------------------------
# Held-out test set (generated once, saved to disk)
# --------------------------------------------------------------------------------------
def _test_key(cfg: Config) -> str:
    mc = cfg.market
    key = dict(asdict(mc), mu_eff=mc.mu_eff, test_log2=cfg.eval.test_log2, test_seed=cfg.eval.test_seed)
    key.pop("mu")
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]


def test_set_path(cfg: Config):
    return cfg.dir("data") / f"test_{lam_tag(cfg.market.lam)}_{_test_key(cfg)}.pt"


def get_test_set(cfg: Config, device="cpu", dtype=torch.float32) -> PathBatch:
    path = test_set_path(cfg)
    if path.exists():
        blob = torch.load(path)
        logS = blob["logS"]
    else:
        gen = torch.Generator(device="cpu").manual_seed(cfg.eval.test_seed)
        pb = simulate(cfg.market, 2 ** cfg.eval.test_log2, gen, "cpu", torch.float64)
        logS = torch.log(pb.S)
        torch.save({"logS": logS, "market": asdict(cfg.market), "seed": cfg.eval.test_seed}, path)
    extras = None
    if cfg.liquidity.enabled:     # liquidity state is regenerated deterministically from the saved prices
        gen = torch.Generator(device="cpu").manual_seed(cfg.eval.test_seed + cfg.liquidity.seed_offset)
        extras = liquidity_state(cfg.liquidity, cfg.market, logS.double(), gen)
    return build_paths(cfg.market, logS, extras, dtype).to(device)


# --------------------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------------------
def bs_put(S, K, tau, r, sigma):
    S, tau = np.asarray(S, float), np.asarray(tau, float)
    tau = np.maximum(tau, 1e-300)
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / vol
    d2 = d1 - vol
    return K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_delta(S, K, tau, r, sigma):
    S, tau = np.asarray(S, float), np.asarray(tau, float)
    tau = np.maximum(tau, 1e-300)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1) - 1.0


def _merton_terms(tau, r, sigma, lam, m, delta, tol=1e-14, n_cap=200):
    """Yield (weight, r_n, sigma_n) of the Merton Poisson-mixture representation."""
    kappa = math.exp(m + 0.5 * delta**2) - 1.0
    lam_p = lam * (1.0 + kappa)
    if lam == 0.0:
        yield 1.0, r, sigma
        return
    cum = 0.0
    for n in range(n_cap):
        w = math.exp(-lam_p * tau + n * math.log(lam_p * tau) - math.lgamma(n + 1))
        sig_n = math.sqrt(sigma**2 + n * delta**2 / tau)
        r_n = r - lam * kappa + n * math.log1p(kappa) / tau
        yield w, r_n, sig_n
        cum += w
        if 1.0 - cum < tol and n > lam_p * tau:
            return


def merton_put(S, K, tau, r, sigma, lam, m, delta):
    """Merton (1976) European put: sum over jump count n of BS puts with (r_n, sigma_n)."""
    if np.ndim(tau) == 0 and float(tau) <= 0.0:
        return np.maximum(K - np.asarray(S, float), 0.0)
    return sum(w * bs_put(S, K, tau, r_n, s_n) for w, r_n, s_n in _merton_terms(float(tau), r, sigma, lam, m, delta))


def merton_put_delta(S, K, tau, r, sigma, lam, m, delta):
    """dP/dS of the Merton European put (same mixture; weights do not depend on S)."""
    return sum(w * bs_put_delta(S, K, tau, r_n, s_n) for w, r_n, s_n in _merton_terms(float(tau), r, sigma, lam, m, delta))


def merton_european_put(mc: MarketCfg) -> float:
    return float(merton_put(mc.S0, mc.K, mc.T, mc.r, mc.sigma, mc.lam, mc.m, mc.delta))


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = add_config_args(argparse.ArgumentParser(description="Generate/check the held-out test set"))
    cfg = config_from_args(ap.parse_args())
    pb = get_test_set(cfg, dtype=torch.float64)
    mc = cfg.market
    disc_T = pb.Z_disc[:, -1].numpy()
    est, se = disc_T.mean(), disc_T.std(ddof=1) / math.sqrt(len(disc_T))
    cf = merton_european_put(mc)
    print(f"test set {test_set_path(cfg).name}: {pb.B} paths")
    print(f"European put  MC {est:.4f} +- {se:.4f}   closed form {cf:.4f}   z = {(est - cf) / se:+.2f}")
