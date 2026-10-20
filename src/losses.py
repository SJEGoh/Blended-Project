"""Seller's per-path losses, relaxed / hard stopping aggregation, empirical CVaR."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from models import stop_probs
from sim import PathBatch


# --------------------------------------------------------------------------------------
# CVaR
# --------------------------------------------------------------------------------------
def _k_tail(alpha: float, B: int) -> int:
    return max(1, min(B, math.ceil((1.0 - alpha) * B - 1e-9)))


def cvar(X: torch.Tensor, alpha: float) -> torch.Tensor:
    """Empirical CVaR_alpha: mean of the top ceil((1-alpha) B) values (alpha = 0 -> mean).
    Differentiable (sub-gradient through top-k)."""
    if alpha <= 0.0:
        return X.mean()
    return torch.topk(X, _k_tail(alpha, X.numel()), sorted=False).values.mean()


def cvar_estimate(X, alpha: float) -> tuple[float, float]:
    """(CVaR estimate, Monte Carlo standard error) for a sample X (no gradient).
    SE from the CVaR influence function  psi(x) = VaR + (x - VaR)^+ / (1 - alpha)."""
    X = X.detach().double().cpu().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, float)
    B = X.size
    if alpha <= 0.0:
        return float(X.mean()), float(X.std(ddof=1) / math.sqrt(B))
    k = _k_tail(alpha, B)
    part = np.partition(X, B - k)
    var = part[B - k]
    est = float(part[B - k:].mean())
    psi = var + np.maximum(X - var, 0.0) / (1.0 - alpha)
    return est, float(psi.std(ddof=1) / math.sqrt(B))


def paired_cvar_diff(XA, XB, alpha: float) -> tuple[float, float]:
    """CVaR(XA) - CVaR(XB) on the same paths with an influence-function SE for the difference."""
    XA = XA.detach().double().cpu().numpy() if isinstance(XA, torch.Tensor) else np.asarray(XA, float)
    XB = XB.detach().double().cpu().numpy() if isinstance(XB, torch.Tensor) else np.asarray(XB, float)
    B = XA.size
    if alpha <= 0.0:
        d = XA - XB
        return float(d.mean()), float(d.std(ddof=1) / math.sqrt(B))

    def psi(X):
        k = _k_tail(alpha, B)
        part = np.partition(X, B - k)
        var = part[B - k]
        return part[B - k:].mean(), var + np.maximum(X - var, 0.0) / (1.0 - alpha)

    a, pa = psi(XA)
    b, pb = psi(XB)
    return float(a - b), float((pa - pb).std(ddof=1) / math.sqrt(B))


# --------------------------------------------------------------------------------------
# Stopping aggregation
# --------------------------------------------------------------------------------------
def relaxed_stop_weights(f: torch.Tensor) -> torch.Tensor:
    """p_n = f_n prod_{k<n} (1 - f_k); sums to 1 along the last axis when f_N = 1."""
    surv = torch.cumprod(1.0 - f[..., :-1], dim=-1)
    surv = torch.cat([torch.ones_like(f[..., :1]), surv], dim=-1)
    return f * surv


def hard_stop_index(logits: torch.Tensor) -> torch.Tensor:
    """Index (0-based over dates 1..N) of the first date with f_n > 0.5 (<=> logit > 0); forced at N."""
    stop = logits > 0
    stop[..., -1] = True
    return stop.to(torch.uint8).argmax(dim=-1)


# --------------------------------------------------------------------------------------
# Rollout of the game on a batch of paths
# --------------------------------------------------------------------------------------
@dataclass
class Rollout:
    L: torch.Tensor          # (B, N) seller loss L_n = Z~_n - G~_n + C_n at dates 1..N
    delta_prev: torch.Tensor  # (B, N) delta_{n-1} at dates 1..N (i.e. deltas held over [t_{n-1}, t_n))
    G: torch.Tensor          # (B, N) discounted hedging gains G~_n, dates 1..N
    C: torch.Tensor          # (B, N) transaction costs C_n, dates 1..N
    U: torch.Tensor | None = None  # (B, N) cost of unwinding the hedge if exercised at n (0 unless enabled)


def liquidity_cost(q: torch.Tensor, Sd: torch.Tensor, ell, vol, sig, lq) -> torch.Tensor:
    """Cost (discounted) of trading q >= 0 shares per option under the liquidity state (ell, vol, sig):
        [ s0 ell (sig/sigma_ref) q  +  Y ell (sig/sqrt(252)) sqrt(psi / vol) q^1.5 ] S~
    = volatility-proportional half-spread + square-root market impact (convex in q)."""
    sr = torch.clamp(sig / lq.sigma_ref, max=lq.sig_ratio_max)
    half_spread = lq.cost_scale * lq.s0 * ell * sr
    impact = lq.cost_scale * lq.Y * ell * (sr * lq.sigma_ref / math.sqrt(252.0)) * torch.sqrt(lq.psi / vol)
    return (half_spread * q + impact * q.clamp(min=0.0).pow(1.5)) * Sd


def unwind_cost(paths: PathBatch, D: torch.Tensor, K: float, c: float, unwind: str, liq=None) -> torch.Tensor:
    """Cost of closing the hedge if exercised at date n (1..N), given delta_{n-1} = D[:, n-1].
    cash: the position to close is delta_{n-1}.  physical: the holder delivers one share when in the
    money, so the position to close is delta_{n-1} + 1{S_n < K}. Proportional part c * q * S~_n, plus the
    liquidity cost of trading q at t_n when the liquidity extension is on."""
    Sd = paths.S_disc[:, 1:]
    if unwind == "none" or (c <= 0.0 and liq is None):
        return torch.zeros_like(Sd)
    if unwind == "cash":
        q = D.abs()
    elif unwind == "physical":
        q = (D + (paths.S[:, 1:] < K).to(Sd.dtype)).abs()
    else:
        raise ValueError(f"unknown unwind mode {unwind!r}")
    U = c * q * Sd
    if liq is not None:
        e = paths.extras
        U = U + liquidity_cost(q, Sd, e["ell"][:, 1:], e["vol"][:, 1:], e["sig"][:, 1:], liq)
    return U


def hedge_rollout(paths: PathBatch, hedger, K: float, T: float, c: float = 0.0, unwind: str = "none",
                  liq=None) -> Rollout:
    """Run the (recurrent) hedger along the grid; delta_n is held over [t_n, t_{n+1}).
    L_n = Z~_n - G~_n + C_n + U_n, the seller's loss if the option is exercised at t_n, where U_n is the
    optional cost of unwinding the hedge at exercise (zero in the spec's baseline). With the liquidity
    extension (liq = LiquidityCfg), every trade also pays liquidity_cost at the current state."""
    B, N = paths.B, paths.N
    Zd = paths.Z_disc[:, 1:]
    if getattr(hedger, "is_zero", False):
        z = torch.zeros_like(Zd)
        U = unwind_cost(paths, z, K, c, unwind, liq)
        return Rollout(Zd + U, z, z, z, U)
    logm = torch.log(paths.S / K)
    t_frac = paths.t / T
    Sd = paths.S_disc
    delta_prev = torch.zeros(B, dtype=Sd.dtype, device=Sd.device)
    G = torch.zeros_like(delta_prev)
    C = torch.zeros_like(delta_prev)
    deltas, Gs, Cs = [], [], []
    for n in range(N):
        step_state = {k: v[:, n] for k, v in paths.extras.items()} if paths.extras else None
        pnl = paths.Z_disc[:, n] - G + C          # running hedged P&L at t_n (F_{t_n}-measurable)
        delta = hedger(t_frac[n], logm[:, n], delta_prev, step_state, pnl)
        if c > 0.0:
            C = C + c * (delta - delta_prev).abs() * Sd[:, n]
        if liq is not None:
            C = C + liquidity_cost((delta - delta_prev).abs(), Sd[:, n], step_state["ell"], step_state["vol"],
                                   step_state["sig"], liq)
        G = G + delta * (Sd[:, n + 1] - Sd[:, n])
        deltas.append(delta)
        Gs.append(G)
        Cs.append(C)
        delta_prev = delta
    D, G, C = torch.stack(deltas, 1), torch.stack(Gs, 1), torch.stack(Cs, 1)
    U = unwind_cost(paths, D, K, c, unwind, liq)
    return Rollout(Zd - G + C + U, D, G, C, U)


def stopper_logits(paths: PathBatch, ro: Rollout, stopper, K: float, T: float,
                   detach_inputs: bool = False) -> torch.Tensor:
    """Stopper logits at dates 1..N from F_{t_n}-measurable inputs only.
    detach_inputs=True blocks the hedger's gradient through the stopper's inputs."""
    S = paths.S[:, 1:]
    t_frac = (paths.t[1:] / T).unsqueeze(0)
    dp, L = (ro.delta_prev.detach(), ro.L.detach()) if detach_inputs else (ro.delta_prev, ro.L)
    liq_feats = None
    if getattr(stopper, "use_liq", False):
        e = paths.extras
        liq_feats = {k: e[k][:, 1:] for k in ("ell", "vol", "sig")}
    return stopper(t_frac, torch.log(S / K), dp, L, (S < K).to(S.dtype), liq_feats)


def aggregate(ro: Rollout, logits: torch.Tensor, temperature: float, hard: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-path loss X and stopping-date distribution.
    hard=True: X = L_tau with tau the first date with f_n > 0.5; returns (X, tau_index).
    hard=False: X = sum_n p_n L_n (relaxed); returns (X, p)."""
    if hard:
        idx = hard_stop_index(logits)
        return ro.L.gather(1, idx.unsqueeze(1)).squeeze(1), idx
    p = relaxed_stop_weights(stop_probs(logits, temperature))
    return (p * ro.L).sum(1), p
