"""Benchmarks: Longstaff-Schwartz (Bermudan, same grid) and CRR binomial (lam = 0).

LSM works in discounted units under mu = r: regress the realised discounted cash flow on a
degree-`deg` polynomial in moneyness x = S_n/K - 1 (in-the-money paths only), exercise when the
discounted payoff exceeds the fitted continuation value. Coefficients are fitted on one path set
and the price is estimated on an independent set (=> low-biased estimator of the Bermudan price).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np
import torch

from config import Config, MarketCfg, add_config_args, config_from_args, lam_tag
from sim import PathBatch, simulate, merton_european_put


# --------------------------------------------------------------------------------------
# CRR binomial tree (Black-Scholes, lam = 0)
# --------------------------------------------------------------------------------------
def crr_put(S0, K, T, r, sigma, steps, style="american", n_dates=None, return_boundary=False):
    """CRR put. style: 'american' (exercise every step), 'bermudan' (exercise only at the
    n_dates uniform dates t_1..t_N, i.e. every steps/n_dates steps), or 'european'.
    If return_boundary, also returns the critical spot S*_n (largest node where exercise is
    strictly optimal) at each exercise date (bermudan) — NaN where never optimal."""
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    q = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    if style == "bermudan":
        assert n_dates and steps % n_dates == 0
        every = steps // n_dates
    j = np.arange(steps + 1)
    S = S0 * u ** (steps - j) * d ** j
    V = np.maximum(K - S, 0.0)
    boundary = {}
    for i in range(steps - 1, -1, -1):
        V = disc * (q * V[:-1] + (1 - q) * V[1:])
        if i == 0:
            break
        can_ex = style == "american" or (style == "bermudan" and i % every == 0)
        if can_ex:
            j = np.arange(i + 1)
            S = S0 * u ** (i - j) * d ** j
            ex = K - S
            if return_boundary:
                strictly = ex > V + 1e-12
                boundary[i] = S[strictly].max() if strictly.any() else np.nan
            V = np.maximum(V, ex)
    price = float(V[0])
    if return_boundary:
        return price, boundary
    return price


def crr_bermudan_boundary(mc: MarketCfg, steps: int) -> np.ndarray:
    """Critical spot S*_n at the grid dates n = 1..N-1 (index n), NaN elsewhere."""
    _, b = crr_put(mc.S0, mc.K, mc.T, mc.r, mc.sigma, steps, "bermudan", mc.N, return_boundary=True)
    every = steps // mc.N
    out = np.full(mc.N + 1, np.nan)
    for i, s in b.items():
        out[i // every] = s
    return out


# --------------------------------------------------------------------------------------
# Longstaff-Schwartz
# --------------------------------------------------------------------------------------
def _basis(S: np.ndarray, K: float, deg: int) -> np.ndarray:
    x = S / K - 1.0
    return np.stack([x**p for p in range(deg + 1)], axis=-1)


@dataclass
class LSMRule:
    """Fitted LSM exercise rule: exercise at n (1 <= n < N) iff payoff_n > 0 and
    payoff_n > continuation_n(S_n); forced at N. Decisions only use S_n (non-anticipative)."""
    coefs: np.ndarray   # (N+1, deg+1); rows 0 and N unused
    K: float
    r: float
    t: np.ndarray       # (N+1,)
    deg: int

    @property
    def N(self) -> int:
        return len(self.t) - 1

    def continuation(self, n: int, S: np.ndarray) -> np.ndarray:
        return _basis(S, self.K, self.deg) @ self.coefs[n]

    def exercise(self, n: int, S: np.ndarray) -> np.ndarray:
        S = np.asarray(S, float)
        if n >= self.N:
            return np.ones_like(S, dtype=bool)
        if n <= 0:
            return np.zeros_like(S, dtype=bool)
        payoff = np.exp(-self.r * self.t[n]) * np.maximum(self.K - S, 0.0)
        return (payoff > 0) & (payoff > self.continuation(n, S))

    def stop_matrix(self, S: np.ndarray) -> np.ndarray:
        """(B, N) boolean decisions at dates 1..N for spot paths S (B, N+1)."""
        return np.stack([self.exercise(n, S[:, n]) for n in range(1, self.N + 1)], axis=1)

    def stop_index(self, S) -> np.ndarray:
        """Exercise date index in 1..N (first True)."""
        if isinstance(S, torch.Tensor):
            S = S.detach().cpu().double().numpy()
        dec = self.stop_matrix(S)
        return dec.argmax(axis=1) + 1

    def boundary(self, grid=None) -> np.ndarray:
        """Largest S on a fine grid in [0.6K, K] where the rule exercises, per date (NaN if none)."""
        grid = np.linspace(0.6 * self.K, self.K, 4001) if grid is None else grid
        out = np.full(self.N + 1, np.nan)
        for n in range(1, self.N):
            ex = self.exercise(n, grid)
            if ex.any():
                out[n] = grid[ex].max()
        return out

    def is_threshold_rule(self, grid=None) -> bool:
        """True if at every date the exercise set within [0.7K, K] (where the regression has data)
        is a single interval, i.e. the rule is a threshold rule there."""
        grid = np.linspace(0.7 * self.K, self.K, 3001) if grid is None else grid
        for n in range(1, self.N):
            ex = self.exercise(n, grid).astype(int)
            if np.abs(np.diff(ex)).sum() > 1:
                return False
        return True

    def to_dict(self) -> dict:
        return dict(coefs=self.coefs.tolist(), K=self.K, r=self.r, t=self.t.tolist(), deg=self.deg)

    @classmethod
    def from_dict(cls, d: dict) -> "LSMRule":
        return cls(np.array(d["coefs"]), d["K"], d["r"], np.array(d["t"]), d["deg"])


def fit_lsm(paths: PathBatch, mc: MarketCfg, deg: int = 3) -> LSMRule:
    S = paths.S.double().cpu().numpy()
    Zd = paths.Z_disc.double().cpu().numpy()
    N = mc.N
    coefs = np.zeros((N + 1, deg + 1))
    cash = Zd[:, N].copy()
    for n in range(N - 1, 0, -1):
        itm = Zd[:, n] > 0
        if itm.sum() <= deg + 1:
            continue
        X = _basis(S[itm, n], mc.K, deg)
        beta, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
        coefs[n] = beta
        cont = X @ beta
        ex = Zd[itm, n] > cont
        idx = np.flatnonzero(itm)[ex]
        cash[idx] = Zd[idx, n]
    return LSMRule(coefs, mc.K, mc.r, mc.t_grid(), deg)


def price_with_rule(paths: PathBatch, stop_idx: np.ndarray) -> tuple[float, float, np.ndarray]:
    Zd = paths.Z_disc.double().cpu().numpy()
    vals = Zd[np.arange(len(Zd)), stop_idx]
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(len(vals))), vals


@dataclass
class LSMResult:
    price: float
    se: float
    rule: LSMRule
    mean_ex_time: float
    frac_early: float


def lsm(cfg: Config) -> LSMResult:
    mc = cfg.market.risk_neutral()     # LSM is always run under mu = r
    reg = simulate(mc, 2 ** cfg.lsm.n_reg_log2, cfg.lsm.seed_reg, dtype=torch.float64)
    rule = fit_lsm(reg, mc, cfg.lsm.degree)
    pr = simulate(mc, 2 ** cfg.lsm.n_price_log2, cfg.lsm.seed_price, dtype=torch.float64)
    tau = rule.stop_index(pr.S)
    price, se, _ = price_with_rule(pr, tau)
    return LSMResult(price, se, rule, float(mc.t_grid()[tau].mean()), float((tau < mc.N).mean()))


def run_benchmarks(cfg: Config, verbose=True) -> dict:
    """LSM (+ CRR when lam = 0) for cfg.market; writes results/benchmarks_<lam>.json."""
    mc = cfg.market
    res = lsm(cfg)
    out = dict(
        lam=mc.lam, mu_eff=mc.mu_eff,
        lsm_price=res.price, lsm_se=res.se,
        lsm_mean_ex_time=res.mean_ex_time, lsm_frac_early=res.frac_early,
        lsm_threshold_rule=res.rule.is_threshold_rule(),
        lsm_boundary=[None if np.isnan(b) else float(b) for b in res.rule.boundary()],
        lsm_rule=res.rule.to_dict(),
        merton_european=merton_european_put(mc.risk_neutral()),
    )
    if mc.lam == 0.0:
        s = cfg.crr.steps
        out["crr_american"] = crr_put(mc.S0, mc.K, mc.T, mc.r, mc.sigma, s, "american")
        out["crr_bermudan"] = crr_put(mc.S0, mc.K, mc.T, mc.r, mc.sigma, s, "bermudan", mc.N)
        out["crr_european"] = crr_put(mc.S0, mc.K, mc.T, mc.r, mc.sigma, s, "european")
        out["crr_steps"] = s
        out["crr_bermudan_boundary"] = [None if np.isnan(b) else float(b) for b in crr_bermudan_boundary(mc, s)]
    path = cfg.dir("results") / f"benchmarks_{lam_tag(mc.lam)}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    if verbose:
        print(f"[{lam_tag(mc.lam)}] LSM {res.price:.4f} +- {res.se:.4f}  (E[tau]={res.mean_ex_time:.3f}, "
              f"early={res.frac_early:.3f})  European {out['merton_european']:.4f}")
        if mc.lam == 0.0:
            print(f"        CRR({s}) American {out['crr_american']:.4f}  Bermudan-on-grid {out['crr_bermudan']:.4f}  "
                  f"European {out['crr_european']:.4f}")
        print(f"        -> {path}")
    return out


def load_benchmarks(cfg: Config) -> dict:
    path = cfg.dir("results") / f"benchmarks_{lam_tag(cfg.market.lam)}.json"
    if not path.exists():
        return run_benchmarks(cfg, verbose=False)
    with open(path) as fh:
        return json.load(fh)


if __name__ == "__main__":
    ap = add_config_args(argparse.ArgumentParser(description="LSM / CRR benchmarks"))
    ap.add_argument("--lams", nargs="*", type=float, default=None, help="run for several jump intensities")
    args = ap.parse_args()
    lams = args.lams if args.lams is not None else [None]
    for lam in lams:
        extra = [] if lam is None else [f"market.lam={lam}"]
        cfg = config_from_args(argparse.Namespace(config=args.config, set=list(args.set) + extra))
        run_benchmarks(cfg)
