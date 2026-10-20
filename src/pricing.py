"""Part 1 (pricing): risk-neutral Bermudan put values under jumps and volume risk, with certified brackets.

    V_0 = sup_tau E^Q[ e^{-r tau} E_tau ],      E_n = (K - S_n - kappa_n)^+

The pricing measure Q is the simulation measure (mu = r, jump law unchanged: jump risk unpriced, as in Merton).
Volume risk enters through the exercise value: with physical settlement the holder delivers one share per option,
and a holder without the stock buys it at t_n, paying kappa_n per share on top of S_n (losses.liquidity_cost with
q = 1 at that date's liquidity state). exercise_cost = 'none' is cash settlement (the classical price).

Bracket for each case, both ends evaluated on the same independent paths (never used for fitting):
  lower: the exercise rule fitted by Longstaff-Schwartz backward induction (one small network per date),
         tau = first n < N with E_n > 0 and E_n >= C_n(X_n), else N; estimated as E[Y_tau - M_tau] with the
         dual's martingale as a control variate (E[M_tau] = 0 by optional stopping; pathwise <= the dual's max);
  upper: Andersen-Broadie dual E[max_n (Y_n - M_n)] with the value-function martingale
         Delta M_k = V_k(X_k) - sum_j w_j V_k(X_k^(j)), V_k = max(Y_k, C_k), stratified antithetic one-step inner
         draws (dual_value.draw_inner): an exact martingale, so the bound is valid whatever the fit.
Also reported: the European value E[Y_N], the early-exercise premium, exercise statistics, and the exercise
boundary S*(t_n, ell) implied by the fitted continuation values.

Usage:
    python src/pricing.py run --name base_liq --exercise-cost liquidity --set market.lam=1.0
    python src/pricing.py report            # summary JSON, figures, RESULTS.md section
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

from config import ROOT, Config, get_device, load_config, seed_everything
from dual_value import draw_inner
from losses import liquidity_cost
from models import liquidity_features, mlp
from sim import PathBatch, merton_put, simulate_cfg

CHUNK = 2 ** 13
OUT = ROOT / "results" / "pricing"


# --------------------------------------------------------------------------------------
# Exercise value, state, features, one-step simulation
# --------------------------------------------------------------------------------------
def exercise_value(cfg: Config, mode: str, Sd: torch.Tensor, Kd, liq: dict | None) -> torch.Tensor:
    """Discounted exercise value (K~ - S~ - kappa~)^+ ; Kd = K e^{-r t} (broadcastable)."""
    pay = Kd - Sd
    if mode == "liquidity":
        pay = pay - liquidity_cost(torch.ones_like(Sd), Sd, liq["ell"], liq["vol"], liq["sig"], cfg.liquidity)
    return torch.clamp(pay, min=0.0)


def features(cfg: Config, t_frac, S, Y, liq=None) -> torch.Tensor:
    """t/T, log-moneyness, ITM flag, exercise value (scaled), [log ell, log vol, log sig/sigma]."""
    mc = cfg.market
    f = [torch.as_tensor(t_frac, dtype=S.dtype, device=S.device).expand_as(S),
         torch.log(S / mc.K) / (mc.sigma * math.sqrt(mc.T)), (S < mc.K).to(S.dtype),
         Y / (mc.K * mc.sigma * math.sqrt(mc.T))]
    if liq is not None:
        f += liquidity_features(liq["ell"], liq["vol"], liq["sig"], cfg.liquidity.sigma_ref or mc.sigma)
    return torch.stack(f, -1)


def outer(cfg: Config, mode: str, paths: PathBatch):
    """Exercise values Y (B, N), features F (B, N, F) at dates 1..N, and the state at dates 0..N-1."""
    mc = cfg.market
    t = paths.t.double()
    disc = torch.exp(-mc.r * t)
    liq = None
    st = dict(logS=torch.log(paths.S[:, :-1].double()))
    if cfg.liquidity.enabled:
        e = paths.extras
        liq = {k: e[k][:, 1:].double() for k in ("ell", "vol", "sig")}
        st["x"] = torch.log(e["ell"][:, :-1].double())
        st["s2"] = e["sig"][:, :-1].double() ** 2
    S = paths.S[:, 1:].double()
    Y = exercise_value(cfg, mode, S * disc[1:], mc.K * disc[1:], liq)
    F = features(cfg, (t[1:] / mc.T).unsqueeze(0), S, Y, liq)
    return Y, F, st


def inner_step(cfg: Config, mode: str, st: dict, k: int, sh: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """State at t_{k-1} -> exercise value and features at t_k for shocks sh (each (B, J)). Same maps as
    sim.simulate / sim.liquidity_state, so it reproduces the outer path when fed that path's own shocks."""
    mc, lq = cfg.market, cfg.liquidity
    col = lambda v: v[:, k - 1].unsqueeze(1)
    dt, sd = mc.dt, mc.sigma * math.sqrt(mc.dt)
    incr = ((mc.mu_eff - mc.lam * mc.kappa - 0.5 * mc.sigma ** 2) * dt + sd * sh["Z"]
            + sh["Nj"] * mc.m + mc.delta * torch.sqrt(sh["Nj"]) * sh["Zj"])
    S = torch.exp(col(st["logS"]) + incr)
    disc = math.exp(-mc.r * mc.T * k / mc.N)
    liq = None
    if cfg.liquidity.enabled:
        x0 = col(st["x"])
        x = torch.clamp(x0 - lq.kappa * x0 * dt + lq.eta * math.sqrt(dt) * sh["eps"]
                        + lq.beta_down * torch.clamp(-incr - lq.h_mult * sd, min=0.0), max=math.log(lq.ell_max))
        s2 = (1.0 - lq.ewma_w) * col(st["s2"]) + lq.ewma_w * incr ** 2 / dt
        logv = (lq.gamma_v * (torch.clamp((incr / sd).abs(), max=lq.zcap) - math.sqrt(2.0 / math.pi))
                + lq.zeta * sh["eps_v"] - 0.5 * lq.zeta ** 2)
        liq = dict(ell=torch.exp(x), vol=torch.exp(logv), sig=torch.sqrt(s2))
    Y = exercise_value(cfg, mode, S * disc, mc.K * disc, liq)
    return Y, features(cfg, k / mc.N, S, Y, liq)


# --------------------------------------------------------------------------------------
# Continuation values (Longstaff-Schwartz, one network per date)
# --------------------------------------------------------------------------------------
class ContinuationNets(nn.Module):
    def __init__(self, cfg: Config, n_feat: int, hidden: int):
        super().__init__()
        mc = cfg.market
        self.scale = mc.K * mc.sigma * math.sqrt(mc.T)
        self.nets = nn.ModuleList([mlp(n_feat, 1, hidden, 2) for _ in range(mc.N - 1)])

    def forward(self, k: int, f: torch.Tensor) -> torch.Tensor:
        return self.nets[k - 1](f.float()).squeeze(-1).double() * self.scale


def fit_continuation(cfg: Config, mode: str, log2: int, seed: int, device, hidden: int, steps_first: int,
                     steps_next: int, lr: float = 3e-3) -> ContinuationNets:
    mc = cfg.market
    seed_everything(seed)
    paths = simulate_cfg(cfg, 2 ** log2, seed, device, torch.float64)
    Y, F, _ = outer(cfg, mode, paths)
    F = F.float()
    nets = ContinuationNets(cfg, F.shape[-1], hidden).to(device)
    cf = Y[:, -1].clone()                    # realised payoff of the current rule from date k+1 on
    for k in range(mc.N - 1, 0, -1):
        net = nets.nets[k - 1]
        if k < mc.N - 1:
            net.load_state_dict(nets.nets[k].state_dict())
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        target = (cf / nets.scale).float()
        for _ in range(steps_first if k == mc.N - 1 else steps_next):
            loss = ((net(F[:, k - 1]).squeeze(-1) - target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            yk = Y[:, k - 1]
            ex = (yk > 0) & (yk >= nets(k, F[:, k - 1]))
            cf = torch.where(ex, yk, cf)
    for p in nets.parameters():
        p.requires_grad_(False)
    return nets.eval()


# --------------------------------------------------------------------------------------
# Bracket on independent paths
# --------------------------------------------------------------------------------------
@torch.no_grad()
def bracket_samples(cfg: Config, mode: str, paths: PathBatch, shocks: dict, nets: ContinuationNets, m: int,
                    seed: int) -> dict:
    mc = cfg.market
    N = mc.N
    gen = torch.Generator().manual_seed(seed)
    acc = {k: [] for k in ("primal", "primal_cv", "dual", "hind", "euro", "tau", "ell_ex")}
    sums, sq, cnt = torch.zeros(N, dtype=torch.float64), torch.zeros(N, dtype=torch.float64), 0
    for s in range(0, paths.B, CHUNK):
        sl = slice(s, s + CHUNK)
        sub = paths.subset(sl)
        sh_out = {q: v[sl] for q, v in shocks.items()}
        Y, F, st = outer(cfg, mode, sub)
        B = Y.shape[0]
        M = torch.zeros(B, dtype=torch.float64)
        M_tau = torch.zeros(B, dtype=torch.float64)          # M at the fitted rule's exercise date
        best = torch.full((B,), -math.inf, dtype=torch.float64)
        tau = torch.full((B,), N, dtype=torch.long)
        for k in range(1, N + 1):
            yk = Y[:, k - 1]
            y_o, f_o = inner_step(cfg, mode, st, k, {q: v[:, k - 1:k] for q, v in sh_out.items()})
            if k < N:
                c_o = nets(k, f_o[:, 0])
                v_out = torch.maximum(y_o[:, 0], c_o)
                ex = (yk > 0) & (yk >= nets(k, F[:, k - 1])) & (tau == N)
                tau = torch.where(ex, torch.full_like(tau, k), tau)
            else:
                v_out = y_o[:, 0]
            sh_in, w = draw_inner(cfg, B, m, gen, "cpu")
            y_in, f_in = inner_step(cfg, mode, st, k, sh_in)
            v_in = torch.maximum(y_in, nets(k, f_in)) if k < N else y_in
            dM = v_out - (v_in * w).sum(1)
            M = M + dM
            M_tau = torch.where(tau == k, M, M_tau)
            best = torch.maximum(best, yk - M)
            sums[k - 1] += dM.sum()
            sq[k - 1] += (dM ** 2).sum()
        cnt += B
        y_tau = Y.gather(1, (tau - 1).unsqueeze(1)).squeeze(1)
        acc["primal"].append(y_tau)
        acc["primal_cv"].append(y_tau - M_tau)    # same mean (optional stopping, E[M_tau] = 0), far lower variance
        acc["dual"].append(best)
        acc["hind"].append(Y.max(1).values)
        acc["euro"].append(Y[:, -1])
        acc["tau"].append(tau)
        if cfg.liquidity.enabled:
            acc["ell_ex"].append(sub.extras["ell"][:, 1:].double().gather(1, (tau - 1).unsqueeze(1)).squeeze(1))
    out = {k: torch.cat(v).numpy() for k, v in acc.items() if v}
    mean = sums / cnt
    out["inc_z"] = (mean / torch.sqrt(torch.clamp(sq / cnt - mean ** 2, min=1e-300) / cnt)).numpy()
    return out


@torch.no_grad()
def exercise_boundary(cfg: Config, mode: str, nets: ContinuationNets, ells=(1.0, 2.0, 5.0, 10.0)) -> dict:
    """S*(t_n, ell): the largest spot on a grid where exercising beats continuing (E > 0 and E >= C), at
    normal volume and volatility (v = 1, sig = sigma) and the given illiquidity. NaN = no exercise at that date."""
    mc = cfg.market
    grid = torch.linspace(0.5 * mc.K, mc.K, 501, dtype=torch.float64)
    out = {}
    for ell in (ells if cfg.liquidity.enabled else (1.0,)):
        Sstar = []
        for k in range(1, mc.N):
            disc = math.exp(-mc.r * mc.T * k / mc.N)
            liq = None
            if cfg.liquidity.enabled:
                one = torch.ones_like(grid)
                liq = dict(ell=ell * one, vol=one, sig=mc.sigma * one)
            Y = exercise_value(cfg, mode, grid * disc, mc.K * disc, liq)
            ex = (Y > 0) & (Y >= nets(k, features(cfg, k / mc.N, grid, Y, liq)))
            Sstar.append(float(grid[ex].max()) if ex.any() else float("nan"))
        out[f"{ell:g}"] = Sstar
    return out


def se(x) -> float:
    return float(np.std(x, ddof=1) / math.sqrt(len(x)))


def run(args) -> dict:
    cfg = load_config(None, args.set)
    cfg.threads = args.threads
    device = get_device(cfg)
    mode = args.exercise_cost
    if mode == "liquidity" and not cfg.liquidity.enabled:
        cfg.liquidity.enabled = True
    if abs(cfg.market.mu_eff - cfg.market.r) > 1e-12:
        raise ValueError("pricing needs mu = r (Q = simulation measure)")
    mc = cfg.market
    t0 = time.time()
    print(f"== {args.name}: exercise cost = {mode}, S0 = {mc.S0:g}, lam = {mc.lam:g}, m = {mc.m:g}, delta = {mc.delta:g}, "
          f"sigma = {mc.sigma:g}" + (f", psi = {cfg.liquidity.psi:g}, cost scale = {cfg.liquidity.cost_scale:g}, "
                                      f"ell_max = {cfg.liquidity.ell_max:g}" if mode == "liquidity" else ""), flush=True)
    nets = fit_continuation(cfg, mode, args.fit_log2, cfg.seed + 8191, device, args.hidden, args.steps_first,
                            args.steps_next)
    t_fit = time.time() - t0
    # independent evaluation paths: fixed seed for every case (common random numbers across parameters)
    paths, shocks = simulate_cfg(cfg, 2 ** args.eval_log2, cfg.eval.test_seed + 11, "cpu", torch.float64,
                                 return_shocks=True)
    r = bracket_samples(cfg, mode, paths, shocks, nets, args.m, cfg.seed + 4099)
    del paths, shocks
    gc.collect()
    N = mc.N
    tau = r["tau"]
    out = dict(name=args.name, exercise_cost=mode, market=dict(S0=mc.S0, K=mc.K, T=mc.T, N=N, r=mc.r, sigma=mc.sigma,
               lam=mc.lam, m=mc.m, delta=mc.delta),
               liquidity=None if mode != "liquidity" else dict(name=cfg.liquidity.name, psi=cfg.liquidity.psi,
                   cost_scale=cfg.liquidity.cost_scale, ell_max=cfg.liquidity.ell_max,
                   sig_ratio_max=cfg.liquidity.sig_ratio_max, s0=cfg.liquidity.s0, Y=cfg.liquidity.Y),
               lower=float(r["primal_cv"].mean()), lower_se=se(r["primal_cv"]),
               lower_plain=float(r["primal"].mean()), lower_plain_se=se(r["primal"]),
               upper=float(r["dual"].mean()), upper_se=se(r["dual"]),
               hindsight=float(r["hind"].mean()), european=float(r["euro"].mean()), european_se=se(r["euro"]),
               frac_early=float((tau < N).mean()), mean_ex_time=float((tau * mc.dt).mean()),
               inc_max_abs_z=float(np.abs(r["inc_z"]).max()),
               fit=dict(log2=args.fit_log2, hidden=args.hidden, steps=[args.steps_first, args.steps_next]),
               eval_paths=int(len(tau)), m=args.m)
    out["mid"] = 0.5 * (out["lower"] + out["upper"])
    out["width"] = out["upper"] - out["lower"]
    out["early_premium"] = out["lower"] - out["european"]
    if mode == "liquidity":
        ell_ex = r["ell_ex"][tau < N]
        out["early_ex_mean_ell"] = float(ell_ex.mean()) if ell_ex.size else float("nan")
        out["early_ex_frac_illiquid"] = float((ell_ex > 2.0).mean()) if ell_ex.size else float("nan")
    else:
        out["european_closed_form"] = float(merton_put(mc.S0, mc.K, mc.T, mc.r, mc.sigma, mc.lam, mc.m, mc.delta))
        if mc.lam == 0:
            from benchmarks import crr_put
            out["crr_bermudan"] = float(crr_put(mc.S0, mc.K, mc.T, mc.r, mc.sigma, cfg.crr.steps, style="bermudan",
                                                n_dates=N))
    out["boundary"] = exercise_boundary(cfg, mode, nets)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    out["fit_minutes"] = round(t_fit / 60, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(out, indent=1))
    torch.save(dict(nets=nets.state_dict(), hidden=args.hidden, mode=mode, cfg=cfg.to_dict()),
               cfg.dir("ckpt") / f"pricing_{args.name}.pt")
    print(f"  price in [{out['lower']:.4f} ± {out['lower_se']:.4f}, {out['upper']:.4f} ± {out['upper_se']:.4f}] "
          f"(width {out['width']:.4f}); European {out['european']:.4f}; early premium {out['early_premium']:+.4f}; "
          f"early exercise {out['frac_early']:.3f}; max |z| {out['inc_max_abs_z']:.2f}; {out['minutes']} min", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="Risk-neutral Bermudan put pricing under jumps and volume risk")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--name", required=True)
    a.add_argument("--exercise-cost", choices=["none", "liquidity"], default="none")
    a.add_argument("--set", nargs="*", default=[])
    a.add_argument("--fit-log2", type=int, default=16)
    a.add_argument("--eval-log2", type=int, default=17)
    a.add_argument("--hidden", type=int, default=48)
    a.add_argument("--steps-first", type=int, default=600)
    a.add_argument("--steps-next", type=int, default=250)
    a.add_argument("--m", type=int, default=16)
    a.add_argument("--threads", type=int, default=1)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        from pricing_report import report
        report()


if __name__ == "__main__":
    main()
