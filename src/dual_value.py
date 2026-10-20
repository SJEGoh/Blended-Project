"""Value-function martingales for the dual bound (the 'upgrade' in docs/DUAL_CERTIFICATE_PLAN.md, Section 4.2).

Same certificate as dual.py,
    W(h) <= D(h) = min_c { c + E[ max_n ( (L_n - c)^+ - M^c_n ) ] / (1 - alpha) },
but M^c is built Andersen-Broadie style from an approximate Snell envelope V_k of the payoff Y_k = (L_k - c)^+
(L_k for alpha = 0), with the one-step conditional expectation replaced by an average over m fresh inner draws:

    Delta M_k = V_k(X_k) - (1/m) sum_j V_k(X_k^(j)),     X_k^(j) = one step from X_{k-1} with fresh shocks.

The inner draws are independent of everything else, so E[Delta M_k | F_{k-1} v inner draws up to k-1] = 0
exactly: M is a martingale in a filtration that contains every stopping time of the original problem, and
the bound stays valid whatever V is (a poor V only loosens it; inner-sampling noise adds a small upward bias).

V_k = max(Y_k, C_k) with the continuation value C_k fitted by Longstaff-Schwartz backward induction (one small
network per date, warm-started from the next date) on training paths. The fitted exercise rule
    tau_c = first k < N with Y_k > 0 and Y_k >= C_k(X_k), else N
is also reported as a lower-bound attack (the worst-case rule for fixed c is an ordinary stopping rule).

The state X_k = (t_k, S_k, delta_{k-1}, L_k, liquidity state) is Markov for the frozen hedge, so one-step
simulation only needs the outer path's state at k-1 (`inner_step` reproduces the rollout exactly when fed
the outer path's own shocks; see tests/test_dual.py).

Usage:
    python src/dual_value.py --ckpt checkpoints/<hedge>.pt [--alpha 0.9] [--set key=val ...]
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import Config, add_config_args, get_device, liq_of, seed_everything
from dual import _rollout, bound_from_samples, load_hedge, lower_bounds, payoff
from losses import cvar_estimate, liquidity_cost
from models import liquidity_features, mlp
from sim import PathBatch, _poisson_inv_cdf, get_test_set, simulate_cfg

CHUNK = 2 ** 13


# --------------------------------------------------------------------------------------
# State, features and one-step simulation
# --------------------------------------------------------------------------------------
def outer_state(cfg: Config, paths: PathBatch, ro) -> dict:
    """Per-path state needed to step from date k-1 to k (each (B, N), column k-1), in float64."""
    B, N = paths.B, paths.N
    zero = torch.zeros(B, 1, dtype=torch.float64, device=paths.S.device)
    st = dict(logS=torch.log(paths.S[:, :-1].double()), Sd=paths.S_disc[:, :-1].double(),
              delta=ro.delta_prev.double(),                                  # position held over step k
              G=torch.cat([zero, ro.G[:, :-1].double()], 1),                  # gains up to t_{k-1}
              C=ro.C.double())                                                # costs incl. the trade at t_{k-1}
    if cfg.liquidity.enabled:
        e = paths.extras
        st["x"] = torch.log(e["ell"][:, :-1].double())
        st["s2"] = e["sig"][:, :-1].double() ** 2
    return st


def features(cfg: Config, t_frac, S, dprev, L, c, liq=None) -> torch.Tensor:
    """Value-net features at a date: t/T, log-moneyness, delta_{k-1}, (L - c) and its positive part (scaled),
    ITM flag, [liquidity state]. Works for any leading shape."""
    mc = cfg.market
    x_scale = mc.sigma * math.sqrt(mc.T)
    s = mc.K * mc.sigma * math.sqrt(mc.T)
    z = (L - c) / s
    f = [torch.as_tensor(t_frac, dtype=S.dtype, device=S.device).expand_as(S), torch.log(S / mc.K) / x_scale,
         dprev, z, torch.relu(z), (S < mc.K).to(S.dtype)]
    if liq is not None:
        f += liquidity_features(liq["ell"], liq["vol"], liq["sig"], cfg.liquidity.sigma_ref or mc.sigma)
    return torch.stack(f, -1)


def draw_inner(cfg: Config, B: int, m: int, gen: torch.Generator, device) -> tuple[dict, torch.Tensor]:
    """Inner one-step shocks with exact weights for E[V(X_k) | F_{k-1}] (each (B, J)), plus weights (J,).

    Stratified on the jump count (N = 0, N = 1, N >= 2, with exact Poisson weights) and antithetic within each
    stratum (every Gaussian draw is paired with its negative, which removes the odd part of the response, e.g.
    the delta term). sum_j w_j V(X^(j)) is therefore an exactly unbiased estimate of the conditional mean.
    m = number of antithetic pairs in the N = 0 stratum; the jump strata get max(2, m // 4) and 1 pairs."""
    mc = cfg.market
    rate = mc.lam * mc.dt
    p0 = math.exp(-rate)
    p1 = rate * p0
    strata = [(0, p0, m)] if rate <= 0 else [(0, p0, m), (1, p1, max(2, m // 4)), (2, 1.0 - p0 - p1, 1)]
    parts, weights = {}, []
    names = ["Z", "Zj"] + (["eps", "eps_v"] if cfg.liquidity.enabled else [])
    for n, w, k in strata:
        g = {q: torch.randn(B, k, generator=gen, dtype=torch.float64) for q in names}
        if n < 2:
            Nj = torch.full((B, k), float(n), dtype=torch.float64)
        else:                                              # N | N >= 2 by inverse CDF on (F(1), 1)
            u = p0 + p1 + (1.0 - p0 - p1) * torch.rand(B, k, generator=gen, dtype=torch.float64)
            Nj = torch.clamp(_poisson_inv_cdf(u, rate), min=2.0)
        for q in names:
            parts.setdefault(q, []).append(torch.cat([g[q], -g[q]], 1))
        parts.setdefault("Nj", []).append(torch.cat([Nj, Nj], 1))
        weights.append(torch.full((2 * k,), w / (2 * k), dtype=torch.float64))
    sh = {q: torch.cat(v, 1).to(device) for q, v in parts.items()}
    return sh, torch.cat(weights).to(device)


def inner_step(cfg: Config, st: dict, k: int, sh: dict, c) -> tuple[torch.Tensor, torch.Tensor]:
    """Step every path from t_{k-1} to t_k with shocks sh (each (B, m)); returns (L_k, features), (B, m)."""
    mc, lq = cfg.market, cfg.liquidity
    col = lambda v: v[:, k - 1].unsqueeze(1)
    dt, sd = mc.dt, mc.sigma * math.sqrt(mc.dt)
    incr = ((mc.mu_eff - mc.lam * mc.kappa - 0.5 * mc.sigma ** 2) * dt + sd * sh["Z"]
            + sh["Nj"] * mc.m + mc.delta * torch.sqrt(sh["Nj"]) * sh["Zj"])
    t_k = mc.T * k / mc.N
    S = torch.exp(col(st["logS"]) + incr)
    disc = math.exp(-mc.r * t_k)
    Sd = S * disc
    Zd = torch.clamp(mc.K - S, min=0.0) * disc
    d = col(st["delta"])
    G = col(st["G"]) + d * (Sd - col(st["Sd"]))
    liq = None
    if cfg.liquidity.enabled:
        x0 = col(st["x"])
        x = torch.clamp(x0 - lq.kappa * x0 * dt + lq.eta * math.sqrt(dt) * sh["eps"]
                        + lq.beta_down * torch.clamp(-incr - lq.h_mult * sd, min=0.0), max=math.log(lq.ell_max))
        s2 = (1.0 - lq.ewma_w) * col(st["s2"]) + lq.ewma_w * incr ** 2 / dt
        logv = (lq.gamma_v * (torch.clamp((incr / sd).abs(), max=lq.zcap) - math.sqrt(2.0 / math.pi))
                + lq.zeta * sh["eps_v"] - 0.5 * lq.zeta ** 2)
        liq = dict(ell=torch.exp(x), vol=torch.exp(logv), sig=torch.sqrt(s2))
    U = torch.zeros_like(Sd)
    if cfg.costs.unwind != "none" and (cfg.costs.c > 0.0 or liq is not None):
        q = d.abs() if cfg.costs.unwind == "cash" else (d + (S < mc.K).to(S.dtype)).abs()
        U = cfg.costs.c * q * Sd
        if liq is not None:
            U = U + liquidity_cost(q.expand_as(Sd), Sd, liq["ell"], liq["vol"], liq["sig"], lq)
    L = Zd - G + col(st["C"]) + U
    return L, features(cfg, k / mc.N, S, d.expand_as(S), L, c, liq)


def outer_features(cfg: Config, paths: PathBatch, ro, c) -> torch.Tensor:
    """Features of the actual states at dates 1..N, (B, N, F), float64."""
    mc = cfg.market
    liq = None
    if cfg.liquidity.enabled:
        e = paths.extras
        liq = {k: e[k][:, 1:].double() for k in ("ell", "vol", "sig")}
    t = (paths.t[1:].double() / mc.T).unsqueeze(0)
    return features(cfg, t, paths.S[:, 1:].double(), ro.delta_prev.double(), ro.L.double(), c, liq)


# --------------------------------------------------------------------------------------
# Continuation values by Longstaff-Schwartz with small per-date networks
# --------------------------------------------------------------------------------------
class ContinuationNets(nn.Module):
    """C_k for dates k = 1..N-1 (index k-1); outputs in units of K sigma sqrt(T)."""

    def __init__(self, cfg: Config, n_feat: int, hidden: int = 32):
        super().__init__()
        mc = cfg.market
        self.scale = mc.K * mc.sigma * math.sqrt(mc.T)
        self.nets = nn.ModuleList([mlp(n_feat, 1, hidden, 2) for _ in range(mc.N - 1)])

    def forward(self, k: int, f: torch.Tensor) -> torch.Tensor:
        return self.nets[k - 1](f.float()).squeeze(-1).double() * self.scale


def fit_continuation(cfg: Config, hedger, alpha: float, c: float, log2: int, seed: int, device,
                     steps_first: int = 400, steps_next: int = 120, lr: float = 3e-3, hidden: int = 32,
                     target: str = "lsm", m_fit: int = 8) -> ContinuationNets:
    """Backward induction for C_k, k = N-1..1, on 2^log2 training paths. target='lsm': regress the realised
    payoff of the current exercise rule (Longstaff-Schwartz). target='onestep': regress the one-step
    conditional expectation E[max(Y_{k+1}, C_{k+1}(X_{k+1})) | X_k], estimated per path with the same
    stratified antithetic inner draws as the martingale (a low-noise target)."""
    mc = cfg.market
    seed_everything(seed)
    paths = simulate_cfg(cfg, 2 ** log2, seed, device, torch.float32)
    ro = _rollout(cfg, paths, hedger)
    F = outer_features(cfg, paths, ro, c).float()
    Y = payoff(ro.L.double(), c, alpha)
    nets = ContinuationNets(cfg, F.shape[-1], hidden).to(device)
    st = outer_state(cfg, paths, ro) if target == "onestep" else None
    gen = torch.Generator().manual_seed(seed + 5)
    cf = Y[:, -1].clone()                            # realised payoff of the current rule from date k+1 on
    for k in range(mc.N - 1, 0, -1):
        net = nets.nets[k - 1]
        if k < mc.N - 1:
            net.load_state_dict(nets.nets[k].state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        if target == "onestep":
            tg = torch.empty(paths.B, dtype=torch.float64, device=F.device)
            with torch.no_grad():
                for a in range(0, paths.B, CHUNK):
                    sl = slice(a, a + CHUNK)
                    sub_st = {q: v[sl] for q, v in st.items()}
                    sh, w = draw_inner(cfg, sub_st["logS"].shape[0], m_fit, gen, F.device)
                    Lin, Fin = inner_step(cfg, sub_st, k + 1, sh, c)
                    yin = payoff(Lin, c, alpha)
                    vin = torch.maximum(yin, nets(k + 1, Fin)) if k + 1 < mc.N else yin
                    tg[sl] = (vin * w).sum(1)
            target_k = (tg / nets.scale).float()
        else:
            target_k = (cf / nets.scale).float()
        target = target_k
        for _ in range(steps_first if k == mc.N - 1 else steps_next):
            loss = ((net(F[:, k - 1]).squeeze(-1) - target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            C = nets(k, F[:, k - 1])
            yk = Y[:, k - 1]
            ex = (yk >= C) & ((yk > 0) if alpha > 0 else torch.ones_like(yk, dtype=torch.bool))
            cf = torch.where(ex, yk, cf)
    for p in nets.parameters():
        p.requires_grad_(False)
    return nets.eval()


# --------------------------------------------------------------------------------------
# Dual and primal on given paths
# --------------------------------------------------------------------------------------
@torch.no_grad()
def ab_samples(cfg: Config, paths: PathBatch, shocks: dict, hedger, nets: ContinuationNets, alpha: float, c: float,
               m: int, seed: int, device) -> dict:
    """Per-path max_n (Y_n - M_n), the hindsight max_n Y_n, the fitted rule's L_tau, and martingale
    increment diagnostics. The outer state X_k is rebuilt with `inner_step` from the path's own shocks, so
    outer and inner states come from the same map of (X_{k-1}, shock): Delta M_k has conditional mean zero
    exactly, not just up to float rounding."""
    mc = cfg.market
    N = mc.N
    gen = torch.Generator().manual_seed(seed)
    mx, mx0, Ltau, sums, sq, cnt = [], [], [], torch.zeros(N, dtype=torch.float64), torch.zeros(N, dtype=torch.float64), 0
    for s in range(0, paths.B, CHUNK):
        sl = slice(s, s + CHUNK)
        sub = paths.subset(sl).to(device, torch.float32)
        sh_out = {k: v[sl].to(device) for k, v in shocks.items()}
        ro = _rollout(cfg, sub, hedger)
        B = sub.B
        st = outer_state(cfg, sub, ro)
        L = ro.L.double()
        Y = payoff(L, c, alpha)
        M = torch.zeros(B, dtype=torch.float64, device=L.device)
        best = torch.full((B,), -math.inf, dtype=torch.float64, device=L.device)
        tau = torch.full((B,), N, dtype=torch.long, device=L.device)
        for k in range(1, N + 1):
            yk = Y[:, k - 1]
            L_o, F_o = inner_step(cfg, st, k, {q: v[:, k - 1:k] for q, v in sh_out.items()}, c)
            y_o = payoff(L_o[:, 0], c, alpha)
            if k < N:
                Ck = nets(k, F_o[:, 0])
                v_out = torch.maximum(y_o, Ck)
                ex = (yk >= Ck) & ((yk > 0) if alpha > 0 else torch.ones_like(yk, dtype=torch.bool)) & (tau == N)
                tau = torch.where(ex, torch.full_like(tau, k), tau)
            else:
                v_out = y_o
            sh_in, w_in = draw_inner(cfg, B, m, gen, L.device)
            Lin, Fin = inner_step(cfg, st, k, sh_in, c)
            yin = payoff(Lin, c, alpha)
            v_in = torch.maximum(yin, nets(k, Fin)) if k < N else yin
            dM = v_out - (v_in * w_in).sum(1)
            M = M + dM
            best = torch.maximum(best, yk - M)
            sums[k - 1] += dM.sum().cpu()
            sq[k - 1] += (dM ** 2).sum().cpu()
        cnt += B
        mx.append(best.cpu())
        mx0.append(Y.max(1).values.cpu())
        Ltau.append(L.gather(1, (tau - 1).unsqueeze(1)).squeeze(1).cpu())
    mean = sums / cnt
    se = torch.sqrt(torch.clamp(sq / cnt - mean ** 2, min=0.0) / cnt)
    return dict(max=torch.cat(mx).numpy(), max0=torch.cat(mx0).numpy(), L_tau=torch.cat(Ltau).numpy(),
                inc_mean=mean.numpy(), inc_se=se.numpy())


def fit_kw(args) -> tuple:
    return args.fit_steps_first, args.fit_steps_next, 3e-3, args.hidden, args.fit_target, args.m_fit


def run(args) -> dict:
    cfg, hedger, stopper, _ = load_hedge(args.ckpt, args.set)
    if args.threads:
        cfg.threads = args.threads
    device = get_device(cfg)
    alpha = cfg.train.alpha if args.alpha is None else float(args.alpha)
    mc = cfg.market
    tag = args.name or (Path(args.ckpt).stem if args.ckpt and args.ckpt != "none" else
                        f"nohedge_lam{mc.lam:g}".replace(".", "p"))
    name = f"dual_{tag}_a{alpha:g}".replace(".", "p") + "_vf"
    print(f"== {name}: value-function martingale, alpha = {alpha:g}, lam = {mc.lam:g}, c = {cfg.costs.c:g}, "
          f"unwind = {cfg.costs.unwind}, liquidity = {cfg.liquidity.enabled}, m = {args.m}", flush=True)
    hedger = hedger.to(device)
    t0 = time.time()
    test = get_test_set(cfg, dtype=torch.float64)
    lows = lower_bounds(cfg, test, hedger, stopper, args.ckpt, alpha, device)
    del test
    gc.collect()
    print("  lower bounds: " + ", ".join(f"{k} {v['cvar']:.4f}" for k, v in lows.items()), flush=True)

    # candidate c values (alpha > 0): a small grid between VaR_alpha(L_N) and VaR_alpha(max_n L_n) on a pilot
    seed = cfg.seed + 4242
    if alpha > 0:
        if args.c_grid:
            cands = list(args.c_grid)
        else:
            pilot = simulate_cfg(cfg, 2 ** 15, cfg.seed + 977, dtype=torch.float32)
            Lp = _rollout(cfg, pilot, hedger).L.double()
            q_lo = float(torch.quantile(Lp[:, -1], alpha))
            q_hi = float(torch.quantile(Lp.max(1).values, alpha))
            cands = list(np.linspace(q_lo - 0.15 * (q_hi - q_lo) - 0.2, q_hi + 0.15 * (q_hi - q_lo) + 0.2, args.n_c))
        if len(cands) == 1:                  # c fixed in advance (e.g. chosen on selection paths in an earlier run)
            cands = cands * 0
            curve, c_star = [], float(args.c_grid[0])
        sel, sel_sh = (simulate_cfg(cfg, 2 ** args.select_log2, cfg.eval.test_seed + 2, dtype=torch.float64,
                                    return_shocks=True) if cands else (None, None))
        curve = []
        for j, c in enumerate(cands):
            nets = fit_continuation(cfg, hedger, alpha, c, args.fit_log2, seed + j, device, *fit_kw(args))
            r = ab_samples(cfg, sel, sel_sh, hedger, nets, alpha, c, args.m, seed + 100 + j, device)
            D_sel = bound_from_samples(r["max"], c, alpha)[0]
            curve.append(dict(c=float(c), D=D_sel))
            print(f"  c = {c:.3f}: D on selection paths = {D_sel:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        c_star = min(curve, key=lambda q: q["D"])["c"] if curve else float(args.c_grid[0])
        del sel, sel_sh
        gc.collect()
    else:
        c_star, curve = 0.0, None
    # refit at c* with a fresh seed (independent of the selection), evaluate on the held-out test set
    nets = fit_continuation(cfg, hedger, alpha, c_star, args.fit_log2, seed + 999, device, *fit_kw(args))
    test, test_sh = get_test_set(cfg, dtype=torch.float64, return_shocks=True)
    if args.test_log2 < cfg.eval.test_log2:
        n = 2 ** args.test_log2
        test, test_sh = test.subset(slice(0, n)), {k: v[:n] for k, v in test_sh.items()}
    r = ab_samples(cfg, test, test_sh, hedger, nets, alpha, c_star, args.m, seed + 2024, device)
    D, D_se = bound_from_samples(r["max"], c_star, alpha)
    D0, D0_se = bound_from_samples(r["max0"], c_star, alpha)
    rule_cvar = cvar_estimate(r["L_tau"], alpha)
    lows["fitted rule at c* (LSM on (L - c*)^+)"] = dict(cvar=rule_cvar[0], se=rule_cvar[1])
    best_low = max((k for k in lows if not k.startswith("hindsight")), key=lambda k: lows[k]["cvar"])
    low = lows[best_low]
    gap = D - low["cvar"]
    z = r["inc_mean"] / np.maximum(r["inc_se"], 1e-300)
    out = dict(name=name, method="value function (Andersen-Broadie, one-step inner sampling)", ckpt=args.ckpt,
               alpha=alpha, lam=mc.lam, c=cfg.costs.c, unwind=cfg.costs.unwind,
               liquidity=cfg.liquidity.name if cfg.liquidity.enabled else None, m=args.m, fit_log2=args.fit_log2,
               fit_steps=[args.fit_steps_first, args.fit_steps_next], hidden=args.hidden,
               fit_target=args.fit_target, m_fit=args.m_fit,
               test_paths=int(r["max"].size), c_star=c_star if alpha > 0 else None, selection_curve=curve,
               D=D, D_se=D_se, D_M0_at_cstar=D0, D_M0_at_cstar_se=D0_se,
               lower=lows, best_lower=best_low, gap=gap, gap_rel=gap / low["cvar"],
               monotone_ok=bool(D + 2 * math.hypot(D_se, low["se"]) >= low["cvar"]),
               increments_mean=r["inc_mean"].tolist(), increments_se=r["inc_se"].tolist(),
               increments_max_abs_z=float(np.abs(z).max()), minutes=round((time.time() - t0) / 60, 1))
    print(f"  D(h) = {D:.4f} ± {D_se:.4f} | fitted rule at c* {rule_cvar[0]:.4f} | best lower ({best_low}) "
          f"{low['cvar']:.4f} | gap {gap:+.4f} ({gap / low['cvar']:+.2%}) | max |z| {out['increments_max_abs_z']:.2f}",
          flush=True)
    (cfg.dir("results") / f"{name}.json").write_text(json.dumps(out, indent=1))
    torch.save(dict(nets=nets.state_dict(), alpha=alpha, c_star=c_star, cfg=cfg.to_dict()), cfg.dir("ckpt") / f"{name}.pt")
    print(f"wrote results/{name}.json  ({out['minutes']} min)")
    return out


def main():
    ap = add_config_args(argparse.ArgumentParser(description="Dual bound with a value-function martingale"))
    ap.add_argument("--ckpt", default="none")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--m", type=int, default=16, help="antithetic pairs (no-jump stratum) per date for the one-step mean")
    ap.add_argument("--fit-log2", type=int, default=16, help="paths for the Longstaff-Schwartz fit")
    ap.add_argument("--fit-steps-first", type=int, default=400)
    ap.add_argument("--fit-steps-next", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--fit-target", choices=["lsm", "onestep"], default="lsm")
    ap.add_argument("--m-fit", type=int, default=8, help="antithetic pairs for the one-step regression targets")
    ap.add_argument("--select-log2", type=int, default=15)
    ap.add_argument("--test-log2", type=int, default=18)
    ap.add_argument("--n-c", type=int, default=5)
    ap.add_argument("--c-grid", type=float, nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--name", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
