"""Martingale-duality upper bounds for the worst case of a frozen hedge (docs/DUAL_CERTIFICATE_PLAN.md).

For a fixed hedge h with seller loss L_n (dates 1..N) the realistic worst case is

    W(h) = sup over non-anticipative tau  CVaR_alpha(L_tau).

Rockafellar-Uryasev plus weak duality for each c gives, for ANY martingales M^c with M^c_0 = 0,

    W(h) <= D(h) = min_c { c + E[ max_{n=1..N} ( (L_n - c)^+ - M^c_n ) ] / (1 - alpha) }       (alpha > 0)
    W(h) <= D(h) = E[ max_n ( L_n - M_n ) ]                                                    (alpha = 0)

Belomestny & Kraetschmer (AAP 2016, Thm 3.1 / 3.9) show the sup/min swap is exact for pure stopping times
when the information after t_0 is atomless (true here), so any gap comes from the martingale family, the
c grid and Monte Carlo error, not from time inconsistency.

Martingales are built from the simulator's driving shocks (sim.simulate(..., return_shocks=True)):
    Delta M_k = sum_i phi_i(F_{k-1}; c) * xi_i(k),     E[xi_i(k) | F_{k-1}] = 0,
with xi from `basis()` (Hermite polynomials of the diffusion shock, the discounted stock return, the
compensated jump count, jump-indicator x Hermite of the jump-size shock, and the liquidity noises) and phi
a small network of F_{k-1}-measurable features (the same information the hedger and stopper use).

Usage:
    python src/dual.py --ckpt checkpoints/<hedge>.pt [--alpha 0.9] [--set key=val ...]
    python src/dual.py --ckpt none --alpha 0 --set market.lam=0          # no hedge (classical Bermudan dual)
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

from config import Config, ROOT, add_config_args, config_from_dict, get_device, liq_of, load_config, seed_everything
from losses import cvar_estimate, hedge_rollout
from models import Hedger, Stopper, ZeroHedger, liquidity_features, mlp
from sim import PathBatch, get_test_set, simulate_cfg

CHUNK = 2 ** 14


# --------------------------------------------------------------------------------------
# Martingale basis: zero conditional mean shocks for steps 1..N
# --------------------------------------------------------------------------------------
def basis_names(cfg: Config) -> list[str]:
    names = ["H1(Z)", "H2(Z)", "H3(Z)", "dS/S"]
    if cfg.market.lam > 0:
        names += ["N-lam*dt", "1{N>0}H1(Zj)", "1{N>0}H2(Zj)", "1{N>0}H3(Zj)"]
    if cfg.liquidity.enabled:
        names += ["eps", "H2(eps)", "eps_v"]
    return names


def basis(cfg: Config, paths: PathBatch, shocks: dict) -> torch.Tensor:
    """(B, N, K) float64; column k-1 is the shock of step t_{k-1} -> t_k. Each entry has conditional mean
    exactly 0 given F_{k-1} (and unit variance up to the stock-return term), under the simulation measure."""
    mc = cfg.market
    if abs(mc.mu_eff - mc.r) > 1e-12:
        raise ValueError("dual.py needs mu = r: the discounted stock is a martingale only under the pricing measure")
    Z = shocks["Z"]
    out = [Z, (Z ** 2 - 1) / math.sqrt(2), (Z ** 3 - 3 * Z) / math.sqrt(6)]
    # discounted stock return S~_k / S~_{k-1} - 1, rebuilt in float64 from the shocks (exactly mean zero)
    jumps = shocks["Nj"] * mc.m + mc.delta * torch.sqrt(shocks["Nj"]) * shocks["Zj"]
    log_ret = -(mc.lam * mc.kappa + 0.5 * mc.sigma ** 2) * mc.dt + mc.sigma * math.sqrt(mc.dt) * Z + jumps
    out.append(torch.expm1(log_ret) / (mc.sigma * math.sqrt(mc.dt)))
    if mc.lam > 0:
        rate = mc.lam * mc.dt
        p = 1.0 - math.exp(-rate)
        Nj, Zj = shocks["Nj"], shocks["Zj"]
        ind = (Nj > 0).to(Z.dtype) / math.sqrt(p)
        out += [(Nj - rate) / math.sqrt(rate), ind * Zj, ind * (Zj ** 2 - 1) / math.sqrt(2),
                ind * (Zj ** 3 - 3 * Zj) / math.sqrt(6)]
    if cfg.liquidity.enabled:
        e, ev = shocks["eps"], shocks["eps_v"]
        out += [e, (e ** 2 - 1) / math.sqrt(2), ev]
    return torch.stack(out, dim=-1)


# --------------------------------------------------------------------------------------
# Coefficient network phi(F_{k-1}; c)
# --------------------------------------------------------------------------------------
class MartingaleNet(nn.Module):
    """Coefficients of the basis shocks for step k from F_{k-1}-measurable features:
    t/T, log-moneyness, the position held over the step, running loss (minus c) and its positive part,
    ITM flag, [c], [liquidity state]. Zero-initialised output layer, so training starts from M = 0."""

    def __init__(self, cfg: Config, n_basis: int, use_c: bool):
        super().__init__()
        mc = cfg.market
        self.use_c, self.use_liq = use_c, bool(cfg.liquidity.enabled)
        self.x_scale = mc.sigma * math.sqrt(mc.T)
        self.loss_scale = mc.K * mc.sigma * math.sqrt(mc.T)
        self.out_scale = mc.K * mc.sigma * math.sqrt(mc.dt)
        self.sigma_ref = cfg.liquidity.sigma_ref or mc.sigma
        n_in = 6 + int(use_c) + 3 * int(self.use_liq)
        self.net = mlp(n_in, n_basis, cfg.model.hidden, cfg.model.depth)
        last = [m for m in self.net.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def features(self, cfg: Config, paths: PathBatch, ro, c: torch.Tensor) -> torch.Tensor:
        """(B, N, F) features at dates 0..N-1 (row k-1 is used for step k). c: (B,) or scalar tensor."""
        mc = cfg.market
        B, N = paths.B, paths.N
        S = paths.S[:, :-1]
        t = (paths.t[:-1] / mc.T).unsqueeze(0).expand(B, N)
        # running loss if exercised at t_j: L_j for j >= 1; at t_0 the intrinsic value plus set-up cost
        L0 = paths.Z_disc[:, :1] + ro.C[:, :1]
        Lrun = torch.cat([L0, ro.L[:, :-1]], dim=1)
        c = c.reshape(-1, 1) if c.dim() else c
        z = (Lrun - c) / self.loss_scale
        feats = [t, torch.log(S / mc.K) / self.x_scale, ro.delta_prev, z, torch.relu(z), (S < mc.K).to(S.dtype)]
        if self.use_c:
            feats.append((c / self.loss_scale).expand(B, N))
        if self.use_liq:
            e = paths.extras
            feats += liquidity_features(e["ell"][:, :-1], e["vol"][:, :-1], e["sig"][:, :-1], self.sigma_ref)
        return torch.stack(feats, dim=-1)

    def forward(self, feats: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Martingale M_n at dates 1..N, (B, N)."""
        phi = self.net(feats) * self.out_scale
        return (phi * xi).sum(-1).cumsum(1)


def payoff(L: torch.Tensor, c, alpha: float) -> torch.Tensor:
    """Stopping payoff of the inner problem: (L_n - c)^+ for alpha > 0, L_n for alpha = 0."""
    if alpha <= 0.0:
        return L
    c = c.reshape(-1, 1) if isinstance(c, torch.Tensor) and c.dim() else c
    return torch.relu(L - c)


# --------------------------------------------------------------------------------------
# Hedge loading
# --------------------------------------------------------------------------------------
def load_hedge(ckpt: str | None, overrides: list[str]):
    """(cfg, hedger, stopper-or-None, blob). ckpt None/'none' -> no hedge, config from config.yaml."""
    if not ckpt or ckpt == "none":
        cfg = load_config(None, overrides)
        return cfg, ZeroHedger(), None, None
    p = Path(ckpt)
    if not p.is_absolute() and not p.exists():
        p = ROOT / ckpt
    blob = torch.load(p, map_location="cpu")
    cfg = config_from_dict(blob["cfg"], overrides)
    hedger = Hedger(cfg)
    hedger.load_state_dict(blob["hedger"])
    stopper = None
    if blob.get("stopper") is not None:
        stopper = Stopper(cfg)
        stopper.load_state_dict(blob["stopper"])
        stopper.eval()
    for m in (hedger,) + ((stopper,) if stopper is not None else ()):
        for q in m.parameters():
            q.requires_grad_(False)
    return cfg, hedger.eval(), stopper, blob


# --------------------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _rollout(cfg: Config, paths: PathBatch, hedger):
    mc = cfg.market
    return hedge_rollout(paths, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))


def train_martingale(cfg: Config, hedger, alpha: float, c_range: tuple[float, float] | None, steps: int,
                     batch_log2: int, lr: float, seed: int, c_fixed: float | None = None,
                     net: MartingaleNet | None = None, log_every: int = 250) -> tuple[MartingaleNet, list]:
    """Minimise E[max_n (payoff_n(c) - M^c_n)] on fresh paths. alpha > 0: c ~ U(c_range) per path
    (c-conditional net), or c = c_fixed for fine-tuning."""
    device = get_device(cfg)
    names = basis_names(cfg)
    use_c = alpha > 0.0
    if net is None:
        seed_everything(seed)
        net = MartingaleNet(cfg, len(names), use_c).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed)
    gen_c = torch.Generator().manual_seed(seed + 1)
    B = 2 ** batch_log2
    hist, t0 = [], time.time()
    for it in range(steps):
        paths, sh = simulate_cfg(cfg, B, gen, device, torch.float32, return_shocks=True)
        ro = _rollout(cfg, paths, hedger)
        if not use_c:
            c = torch.zeros((), device=device)
        elif c_fixed is not None:
            c = torch.full((B,), float(c_fixed), device=device)
        else:
            c = c_range[0] + (c_range[1] - c_range[0]) * torch.rand(B, generator=gen_c).to(device)
        xi = basis(cfg, paths, sh).float()
        M = net(net.features(cfg, paths, ro, c), xi)
        loss = (payoff(ro.L, c, alpha) - M).max(dim=1).values.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == steps - 1:
            hist.append(dict(it=it, loss=loss.item(), sec=round(time.time() - t0, 1)))
            print(f"  [dual train] it={it:5d}  E[max(Y - M)] = {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return net.eval(), hist


@torch.no_grad()
def dual_samples(cfg: Config, paths: PathBatch, shocks: dict, hedger, net: MartingaleNet | None, alpha: float,
                 cs: list[float], device) -> tuple[np.ndarray, np.ndarray]:
    """Per-path pathwise maxima max_n (payoff_n(c) - M^c_n) for each c in cs -> (len(cs), B), plus the
    martingale increments' per-date means and SEs (for c = cs[0]) as (2, N)."""
    out = np.zeros((len(cs), paths.B))
    sums, sq, cnt = None, None, 0
    for s in range(0, paths.B, CHUNK):
        sl = slice(s, s + CHUNK)
        sub = paths.subset(sl).to(device, torch.float32)
        sh = {k: v[sl].to(device) for k, v in shocks.items()}
        ro = _rollout(cfg, sub, hedger)
        xi = basis(cfg, sub, sh).float()
        for j, c in enumerate(cs):
            ct = torch.tensor(float(c), device=device)
            M = torch.zeros_like(ro.L) if net is None else net(net.features(cfg, sub, ro, ct), xi)
            out[j, sl] = (payoff(ro.L, ct, alpha) - M).max(dim=1).values.double().cpu().numpy()
            if j == 0 and net is not None:
                dM = torch.diff(torch.cat([torch.zeros_like(M[:, :1]), M], 1), dim=1).double()
                sums = dM.sum(0) if sums is None else sums + dM.sum(0)
                sq = (dM ** 2).sum(0) if sq is None else sq + (dM ** 2).sum(0)
                cnt += dM.shape[0]
    inc = None
    if sums is not None:
        mean = sums / cnt
        se = torch.sqrt(torch.clamp(sq / cnt - mean ** 2, min=0.0) / cnt)
        inc = torch.stack([mean, se]).cpu().numpy()
    return out, inc


def bound_from_samples(Y: np.ndarray, c: float, alpha: float) -> tuple[float, float]:
    """(D(c), SE) from per-path maxima Y."""
    if alpha <= 0.0:
        return float(Y.mean()), float(Y.std(ddof=1) / math.sqrt(Y.size))
    return float(c + Y.mean() / (1 - alpha)), float(Y.std(ddof=1) / (1 - alpha) / math.sqrt(Y.size))


@torch.no_grad()
def lower_bounds(cfg: Config, test: PathBatch, hedger, stopper, ckpt: str | None, alpha: float, device) -> dict:
    """CVaR_alpha of L_tau on the test set for the exercise rules at hand (each is a lower bound on W(h))."""
    from benchmarks import LSMRule, load_benchmarks
    from evaluate import load_policy, rollout_np, stopped

    mc = cfg.market
    L, _, tau = rollout_np(cfg, test, hedger, stopper, device)
    out = {"never early": cvar_estimate(L[:, -1], alpha)}
    try:
        rule = LSMRule.from_dict(load_benchmarks(cfg)["lsm_rule"])
        out["LSM rule"] = cvar_estimate(stopped(L, rule.stop_index(test.S)), alpha)
    except FileNotFoundError:
        pass
    if tau is not None:
        out["co-trained adversary"] = cvar_estimate(stopped(L, tau), alpha)
    if ckpt and ckpt != "none":
        name = Path(ckpt).name
        ex = name.replace("gda_", "exploit_", 1) if name.startswith("gda_") else "exploit-" + name
        ex_path = cfg.dir("ckpt") / ex
        if ex_path.exists():
            _, fresh, _ = load_policy(ex_path, cfg, device)
            _, _, tau_f = rollout_np(cfg, test, hedger, fresh, device)
            out["fresh adversary"] = cvar_estimate(stopped(L, tau_f), alpha)
    out["hindsight max (M = 0)"] = cvar_estimate(L.max(1), alpha)    # trivial UPPER bound, reported alongside
    return {k: dict(cvar=v[0], se=v[1]) for k, v in out.items()}


def run(args) -> dict:
    cfg, hedger, stopper, blob = load_hedge(args.ckpt, args.set)
    if args.threads:
        cfg.threads = args.threads
    device = get_device(cfg)
    alpha = cfg.train.alpha if args.alpha is None else float(args.alpha)
    mc = cfg.market
    tag = args.name or (Path(args.ckpt).stem if args.ckpt and args.ckpt != "none" else
                        f"nohedge_lam{mc.lam:g}".replace(".", "p"))
    name = f"dual_{tag}_a{alpha:g}".replace(".", "p")
    print(f"== {name}: alpha = {alpha:g}, lam = {mc.lam:g}, c = {cfg.costs.c:g}, unwind = {cfg.costs.unwind}, "
          f"liquidity = {cfg.liquidity.enabled}", flush=True)
    hedger = hedger.to(device)
    t0 = time.time()
    test = get_test_set(cfg, dtype=torch.float64)
    lows = lower_bounds(cfg, test, hedger, stopper, args.ckpt, alpha, device)
    best_low = max((k for k in lows if not k.startswith("hindsight")), key=lambda k: lows[k]["cvar"])
    print("  lower bounds: " + ", ".join(f"{k} {v['cvar']:.4f}" for k, v in lows.items()), flush=True)
    del test                                  # the test set (with shocks) is reloaded for the final evaluation
    gc.collect()

    # c range from a pilot batch: between the VaR of the best lower-bound rule's loss and of the hindsight max
    c_range = None
    if alpha > 0:
        if args.c_range:
            c_range = tuple(args.c_range)
        else:
            pilot = simulate_cfg(cfg, 2 ** 15, cfg.seed + 977, dtype=torch.float32)
            Lp = _rollout(cfg, pilot, hedger).L
            q_lo = float(torch.quantile(Lp[:, -1].double(), alpha))
            q_hi = float(torch.quantile(Lp.max(1).values.double(), alpha))
            c_range = (q_lo - 0.25 * (q_hi - q_lo) - 0.5, q_hi + 0.25 * (q_hi - q_lo) + 0.5)
        print(f"  c range {c_range[0]:.3f} .. {c_range[1]:.3f}", flush=True)

    seed = cfg.seed + 31337
    net, hist = train_martingale(cfg, hedger, alpha, c_range, args.steps, args.batch_log2, args.lr, seed)
    # choose c on independent selection paths (never the test set), then optionally fine-tune at c*
    c_star, sel_curve = None, None
    if alpha > 0:
        sel, sel_sh = simulate_cfg(cfg, 2 ** args.select_log2, cfg.eval.test_seed + 2, dtype=torch.float64,
                                   return_shocks=True)
        grid = list(np.linspace(c_range[0], c_range[1], 25))
        Ys, _ = dual_samples(cfg, sel, sel_sh, hedger, net, alpha, grid, device)
        vals = [bound_from_samples(Ys[j], c, alpha)[0] for j, c in enumerate(grid)]
        j = int(np.argmin(vals))
        step = grid[1] - grid[0]
        fine = list(np.linspace(grid[j] - step, grid[j] + step, 11))
        Yf, _ = dual_samples(cfg, sel, sel_sh, hedger, net, alpha, fine, device)
        fvals = [bound_from_samples(Yf[i], c, alpha)[0] for i, c in enumerate(fine)]
        c_star = float(fine[int(np.argmin(fvals))])
        sel_curve = dict(c=[float(c) for c in grid], D=vals)
        del sel, sel_sh, Ys, Yf
        gc.collect()
        print(f"  c* = {c_star:.4f} (selection paths: D = {min(fvals):.4f})", flush=True)
        if args.finetune_steps > 0:
            net.train()
            net, hist_ft = train_martingale(cfg, hedger, alpha, None, args.finetune_steps, args.batch_log2,
                                            args.lr * 0.3, seed + 7, c_fixed=c_star, net=net)
            hist += [dict(h, phase="finetune") for h in hist_ft]
    # evaluate on the held-out test set at the pre-chosen c*
    cs = [c_star] if alpha > 0 else [0.0]
    test, test_sh = get_test_set(cfg, dtype=torch.float64, return_shocks=True)
    Y, inc = dual_samples(cfg, test, test_sh, hedger, net, alpha, cs, device)
    D, D_se = bound_from_samples(Y[0], cs[0], alpha)
    Y0, _ = dual_samples(cfg, test, test_sh, hedger, None, alpha, cs, device)
    D0, D0_se = bound_from_samples(Y0[0], cs[0], alpha)
    z_inc = inc[0] / np.maximum(inc[1], 1e-300)
    low = lows[best_low]
    gap = D - low["cvar"]
    out = dict(name=name, method="linear (shock basis)", ckpt=args.ckpt, alpha=alpha, lam=mc.lam, c=cfg.costs.c, unwind=cfg.costs.unwind,
               liquidity=cfg.liquidity.name if cfg.liquidity.enabled else None,
               basis=basis_names(cfg), steps=args.steps, finetune_steps=args.finetune_steps if alpha > 0 else 0,
               batch_log2=args.batch_log2, c_range=c_range, c_star=c_star, selection_curve=sel_curve,
               D=D, D_se=D_se, D_M0_at_cstar=D0, D_M0_at_cstar_se=D0_se,
               lower=lows, best_lower=best_low, gap=gap, gap_rel=gap / low["cvar"],
               monotone_ok=bool(D + 2 * math.hypot(D_se, low["se"]) >= low["cvar"]),
               increments_mean=inc[0].tolist(), increments_se=inc[1].tolist(), increments_max_abs_z=float(np.abs(z_inc).max()),
               train_hist=hist, minutes=round((time.time() - t0) / 60, 1))
    print(f"  D(h) = {D:.4f} ± {D_se:.4f} | hindsight (M = 0) {lows['hindsight max (M = 0)']['cvar']:.4f} | "
          f"best lower ({best_low}) {low['cvar']:.4f} | gap {gap:+.4f} ({gap / low['cvar']:+.2%}) | "
          f"max |z| of martingale increments {out['increments_max_abs_z']:.2f}", flush=True)
    res = cfg.dir("results")
    (res / f"{name}.json").write_text(json.dumps(out, indent=1))
    torch.save(dict(net=net.state_dict(), basis=basis_names(cfg), alpha=alpha, c_star=c_star, cfg=cfg.to_dict()),
               cfg.dir("ckpt") / f"{name}.pt")
    print(f"wrote results/{name}.json  ({out['minutes']} min)")
    return out


def main():
    ap = add_config_args(argparse.ArgumentParser(description="Dual (martingale) upper bound for a frozen hedge"))
    ap.add_argument("--ckpt", default="none", help="hedge checkpoint (gda_*, fixed-*), or 'none' for no hedge")
    ap.add_argument("--alpha", type=float, default=None, help="CVaR level (default: the checkpoint's train.alpha)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--finetune-steps", type=int, default=1000)
    ap.add_argument("--batch-log2", type=int, default=13)
    ap.add_argument("--select-log2", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--c-range", type=float, nargs=2, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--name", default=None, help="label for the output files")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
