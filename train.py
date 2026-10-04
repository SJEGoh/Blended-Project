"""Training loops.

  stopper_only : zero hedge, stopper trained alone by gradient ascent on CVaR (build-order step 2).
  gda          : alternating gradient descent-ascent, k_adv stopper ascent steps per hedger
                 descent step, optional adversary pool (build-order step 3).
  fixed        : hedger trained against a FIXED, non-adversarial stopping rule ('lsm' = the
                 risk-neutral LSM exercise rule, 'maturity' = hold to T) — headline experiment.
  exploit      : freeze a trained hedger (from gda or fixed), train a *fresh* stopper against it
                 (same routine as stopper_only) — the exploitability attack.
All stoppers are warm-started (cfg.train.stopper_init = "heuristic") unless configured otherwise.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, add_config_args, alpha_tag, config_from_args, get_device, lam_tag, liq_of, seed_everything
from losses import aggregate, cvar, hedge_rollout, stopper_logits
from models import Hedger, Stopper, ZeroHedger
from sim import merton_put, simulate, simulate_cfg


def anneal_temperature(it: int, total: int, t0: float, t1: float) -> float:
    """Geometric anneal from t0 (it = 0) to t1 (it = total - 1)."""
    if total <= 1:
        return t1
    return t0 * (t1 / t0) ** (it / (total - 1))


def run_name(cfg: Config, mode: str, alpha: float | None = None) -> str:
    a = cfg.train.alpha if alpha is None else alpha
    tr = cfg.train
    name = f"{mode}_{lam_tag(cfg.market.lam)}_{alpha_tag(a)}_s{cfg.seed}"
    if tr.stopper_init != "random":
        name += f"_{tr.stopper_init}"
    if tr.hedger_pnl_input:
        name += "_hpnl"
    if not tr.hedger_grad_through_stopper:
        name += "_detach"
    if mode in ("gda", "exploit"):          # adversary restarts only change the GDA run (and its attack)
        if tr.adv_restart_every > 0:
            name += f"_rs{tr.adv_restart_every}"
    if cfg.costs.c > 0:
        name += f"_c{cfg.costs.c:g}"
        if cfg.costs.unwind != "none":
            name += f"_uw{cfg.costs.unwind}"
    if cfg.liquidity.enabled:
        name += f"_liq{cfg.liquidity.name}"
        if cfg.liquidity.cost_scale != 1.0:
            name += f"x{cfg.liquidity.cost_scale:g}"
        if cfg.costs.c <= 0 and cfg.costs.unwind != "none":
            name += f"_uw{cfg.costs.unwind}"
    if cfg.tag:
        name += f"_{cfg.tag}"
    return name


def set_trainable(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


# --------------------------------------------------------------------------------------
# Optional stopper warm start (proposed fix, cfg.train.stopper_init = "heuristic")
# --------------------------------------------------------------------------------------
class EuropeanHeuristic:
    """Exercise at t_n (n < N) iff S_n < K and (K - S_n) > P_Eur(t_n, S_n), with P_Eur the Merton
    European put under mu = r. A simple, sub-optimal rule that does NOT use LSM; it only gives the
    stopper logits that already separate deep-ITM from OTM states before the ascent starts."""

    def __init__(self, cfg: Config, n_grid: int = 3001):
        mc = cfg.market
        self.K, self.N = mc.K, mc.N
        self.S_grid = np.exp(np.linspace(np.log(0.3 * mc.K), np.log(3.0 * mc.K), n_grid))
        t = mc.t_grid()
        self.table = np.stack([merton_put(self.S_grid, mc.K, mc.T - t[n], mc.r, mc.sigma, mc.lam, mc.m, mc.delta)
                               if n < mc.N else np.maximum(mc.K - self.S_grid, 0.0) for n in range(1, mc.N + 1)])

    def targets(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, N) spot at dates 1..N -> (B, N-1) 0/1 targets at dates 1..N-1."""
        Sn = S.detach().double().cpu().numpy()
        eu = np.stack([np.interp(Sn[:, n], self.S_grid, self.table[n]) for n in range(self.N - 1)], 1)
        tgt = (Sn[:, :-1] < self.K) & (self.K - Sn[:, :-1] > eu + 1e-9)
        return torch.from_numpy(tgt).to(S.device, S.dtype)


def warm_start_stopper(cfg: Config, stopper: Stopper, hedger, gen: torch.Generator, device) -> dict:
    mc, tr = cfg.market, cfg.train
    heur = EuropeanHeuristic(cfg)
    opt = torch.optim.Adam(stopper.parameters(), lr=tr.lr_stopper)
    B = 2 ** tr.batch_log2
    for it in range(tr.warm_steps):
        paths = simulate_cfg(cfg, B, gen, device)
        with torch.no_grad():
            ro = hedge_rollout(paths, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
        logits = stopper_logits(paths, ro, stopper, mc.K, mc.T)
        tgt = heur.targets(paths.S[:, 1:])
        loss = F.binary_cross_entropy_with_logits(logits[:, :-1], tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float(((logits[:, :-1] > 0).to(tgt.dtype) == tgt).float().mean())
    print(f"warm start ({tr.warm_steps} steps): BCE {float(loss):.4f}, agreement with heuristic {acc:.4f}", flush=True)
    return dict(warm_bce=float(loss), warm_agreement=acc)


class Logger:
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "w", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=fields)
        self.w.writeheader()

    def log(self, row: dict, echo: bool = True):
        self.w.writerow(row)
        self.fh.flush()
        if echo:
            print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)

    def close(self):
        self.fh.close()


@torch.no_grad()
def batch_stats(paths, ro, logits, alpha, T) -> dict:
    X_hard, idx = aggregate(ro, logits, 1.0, hard=True)
    t_ex = paths.t[1:][idx]
    return dict(
        hard_cvar=float(cvar(X_hard, alpha)),
        mean_ex_time=float(t_ex.mean()),
        frac_early=float((idx < paths.N - 1).float().mean()),
        hedge_mean=float(ro.delta_prev.mean()),
        hedge_std=float(ro.delta_prev.std()),
    )


def train_stopper(cfg: Config, hedger: torch.nn.Module, steps: int, alpha: float, name: str,
                  stopper: Stopper | None = None, seed_offset: int = 0) -> tuple[Stopper, Path]:
    """Gradient ascent on CVaR_alpha of the relaxed stopped loss, with the hedger frozen."""
    device = get_device(cfg)
    seed_everything(cfg.seed + seed_offset)
    mc, tr = cfg.market, cfg.train
    stopper = (stopper or Stopper(cfg)).to(device)
    hedger = hedger.to(device).eval()
    for p in hedger.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(stopper.parameters(), lr=tr.lr_stopper)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(cfg.seed + 7919 + seed_offset)
    B = 2 ** tr.batch_log2
    t_start = time.time()
    warm = warm_start_stopper(cfg, stopper, hedger, gen, device) if tr.stopper_init == "heuristic" else {}
    log = Logger(cfg.dir("results") / "logs" / f"{name}.csv",
                 ["step", "temp", "train_cvar", "hard_cvar", "mean_ex_time", "frac_early",
                  "hedge_mean", "hedge_std", "sec"])
    for it in range(steps):
        temp = anneal_temperature(it, steps, tr.temp_start, tr.temp_end)
        paths = simulate_cfg(cfg, B, gen, device)
        with torch.no_grad():
            ro = hedge_rollout(paths, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
        logits = stopper_logits(paths, ro, stopper, mc.K, mc.T)
        X, _ = aggregate(ro, logits, temp, hard=False)
        obj = cvar(X, alpha)
        opt.zero_grad(set_to_none=True)
        (-obj).backward()          # ascent
        opt.step()
        sched.step()
        if it % tr.log_every == 0 or it == steps - 1:
            st = batch_stats(paths, ro, logits.detach(), alpha, mc.T)
            log.log(dict(step=it, temp=temp, train_cvar=float(obj.detach()), **st, sec=round(time.time() - t_start, 1)))
    log.close()
    ckpt = cfg.dir("ckpt") / f"{name}.pt"
    torch.save(dict(stopper=stopper.state_dict(),
                    hedger=None if getattr(hedger, "is_zero", False) else hedger.state_dict(),
                    cfg=cfg.to_dict(), alpha=alpha, mode="stopper_only", steps=steps,
                    stopper_init=tr.stopper_init, **warm,
                    train_seconds=time.time() - t_start), ckpt)
    print(f"saved {ckpt}  ({time.time() - t_start:.0f}s)")
    return stopper, ckpt


def train_stopper_only(cfg: Config) -> Path:
    """Build-order step 2: zero hedge, alpha as configured (spec: alpha = 0)."""
    alpha = cfg.train.alpha
    name = run_name(cfg, "stopper_only", alpha)
    _, ckpt = train_stopper(cfg, ZeroHedger(), cfg.stopper_only.steps, alpha, name)
    return ckpt


def restart_adversary(cfg: Config, stopper: Stopper, hedger: torch.nn.Module, gen: torch.Generator, device,
                      temp_now: float, alpha: float) -> dict:
    """Proposed fix for adversary collapse (cfg.train.adv_restart_every > 0). Warm-start a fresh
    stopper, train it by ascent against the frozen hedger (temperature annealed from temp_start to the
    current GDA temperature), then compare hard-rule CVaR with the current stopper on a common batch of
    4x the training batch. If the candidate is higher, copy its weights into `stopper` (in place, so the
    caller's optimiser keeps pointing at the same parameters)."""
    mc, tr = cfg.market, cfg.train
    set_trainable(hedger, False)
    cand = Stopper(cfg).to(device)
    warm_start_stopper(cfg, cand, hedger, gen, device)
    opt = torch.optim.Adam(cand.parameters(), lr=tr.lr_stopper)
    B = 2 ** tr.batch_log2
    for k in range(tr.adv_restart_steps):
        temp = anneal_temperature(k, tr.adv_restart_steps, tr.temp_start, temp_now)
        paths = simulate_cfg(cfg, B, gen, device)
        with torch.no_grad():
            ro = hedge_rollout(paths, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
        X, _ = aggregate(ro, stopper_logits(paths, ro, cand, mc.K, mc.T), temp, hard=False)
        obj = cvar(X, alpha)
        opt.zero_grad(set_to_none=True)
        (-obj).backward()
        opt.step()
    with torch.no_grad():
        paths = simulate_cfg(cfg, 4 * B, gen, device)
        ro = hedge_rollout(paths, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
        lc = stopper_logits(paths, ro, stopper, mc.K, mc.T)
        ln = stopper_logits(paths, ro, cand, mc.K, mc.T)
        v_cur = float(cvar(aggregate(ro, lc, 1.0, hard=True)[0], alpha))
        v_new = float(cvar(aggregate(ro, ln, 1.0, hard=True)[0], alpha))
        st_new = batch_stats(paths, ro, ln, alpha, mc.T)
    swapped = v_new > v_cur
    if swapped:
        stopper.load_state_dict(cand.state_dict())
    print(f"  adversary restart: current {v_cur:.4f} vs fresh {v_new:.4f} (fresh frac_early {st_new['frac_early']:.3f})"
          f" -> {'swap' if swapped else 'keep'}", flush=True)
    return dict(current=v_cur, fresh=v_new, fresh_frac_early=st_new["frac_early"], swapped=bool(swapped))


def train_gda(cfg: Config) -> Path:
    """Alternating gradient descent-ascent on CVaR_alpha of the relaxed stopped loss.
    Each outer iteration: k_adv stopper ascent steps (hedger frozen), then one hedger descent step
    against the current stopper (and, with the adversary pool, the max with a random past snapshot).
    The hedger gradient flows through the stopper's inputs (running P&L, delta_{n-1}) unless
    cfg.train.hedger_grad_through_stopper is False. Fresh paths for every step."""
    device = get_device(cfg)
    seed_everything(cfg.seed)
    mc, tr = cfg.market, cfg.train
    alpha = tr.alpha
    name = run_name(cfg, "gda", alpha)
    hedger, stopper = Hedger(cfg).to(device), Stopper(cfg).to(device)
    gen = torch.Generator().manual_seed(cfg.seed + 104729)
    rng = random.Random(cfg.seed)
    B = 2 ** tr.batch_log2
    t_start = time.time()
    warm = {}
    if tr.stopper_init == "heuristic":
        set_trainable(hedger, False)
        warm = warm_start_stopper(cfg, stopper, hedger, gen, device)
    opt_s = torch.optim.Adam(stopper.parameters(), lr=tr.lr_stopper)
    opt_h = torch.optim.Adam(hedger.parameters(), lr=tr.lr_hedger)
    sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=tr.outer_iters * tr.k_adv)
    sch_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=tr.outer_iters)
    pool: list[Stopper] = []
    restarts: list[dict] = []
    detach = not tr.hedger_grad_through_stopper
    log = Logger(cfg.dir("results") / "logs" / f"{name}.csv",
                 ["it", "temp", "stopper_obj", "hedger_loss", "pool_loss", "hard_cvar", "mean_ex_time",
                  "frac_early", "hedge_mean", "hedge_std", "pool_size", "sec"])
    for it in range(tr.outer_iters):
        temp = anneal_temperature(it, tr.outer_iters, tr.temp_start, tr.temp_end)
        # ---- adversary: k_adv ascent steps, hedger frozen ----
        set_trainable(stopper, True)
        set_trainable(hedger, False)
        for _ in range(tr.k_adv):
            paths = simulate_cfg(cfg, B, gen, device)
            with torch.no_grad():
                ro = hedge_rollout(paths, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
            X, _ = aggregate(ro, stopper_logits(paths, ro, stopper, mc.K, mc.T), temp, hard=False)
            s_obj = cvar(X, alpha)
            opt_s.zero_grad(set_to_none=True)
            (-s_obj).backward()
            opt_s.step()
            sch_s.step()
        # ---- hedger: one descent step against the (frozen) current stopper [and a pool snapshot] ----
        set_trainable(stopper, False)
        set_trainable(hedger, True)
        paths = simulate_cfg(cfg, B, gen, device)
        ro = hedge_rollout(paths, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
        logits = stopper_logits(paths, ro, stopper, mc.K, mc.T, detach_inputs=detach)
        X, _ = aggregate(ro, logits, temp, hard=False)
        h_loss = cvar(X, alpha)
        loss, p_loss = h_loss, float("nan")
        if tr.adversary_pool and pool:
            snap = pool[rng.randrange(len(pool))]
            X2, _ = aggregate(ro, stopper_logits(paths, ro, snap, mc.K, mc.T, detach_inputs=detach), temp, hard=False)
            pl = cvar(X2, alpha)
            p_loss = float(pl.detach())
            loss = torch.maximum(h_loss, pl)
        opt_h.zero_grad(set_to_none=True)
        loss.backward()
        opt_h.step()
        sch_h.step()
        if tr.adversary_pool and (it + 1) % tr.pool_every == 0:
            snap = copy.deepcopy(stopper).eval()
            set_trainable(snap, False)
            pool.append(snap)
        if tr.adv_restart_every > 0 and (it + 1) % tr.adv_restart_every == 0 and it + 1 < tr.outer_iters:
            r = restart_adversary(cfg, stopper, hedger, gen, device, temp, alpha)
            restarts.append(dict(it=it, **r))
            if r["swapped"]:
                opt_s.state.clear()          # fresh Adam moments for the swapped-in weights
        if it % tr.log_every == 0 or it == tr.outer_iters - 1:
            st = batch_stats(paths, ro, logits.detach(), alpha, mc.T)
            log.log(dict(it=it, temp=temp, stopper_obj=float(s_obj.detach()), hedger_loss=float(h_loss.detach()),
                         pool_loss=p_loss, hard_cvar=st["hard_cvar"], mean_ex_time=st["mean_ex_time"],
                         frac_early=st["frac_early"], hedge_mean=st["hedge_mean"], hedge_std=st["hedge_std"],
                         pool_size=len(pool), sec=round(time.time() - t_start, 1)))
    log.close()
    ckpt = cfg.dir("ckpt") / f"{name}.pt"
    torch.save(dict(stopper=stopper.state_dict(), hedger=hedger.state_dict(), cfg=cfg.to_dict(), alpha=alpha,
                    mode="gda", outer_iters=tr.outer_iters, stopper_init=tr.stopper_init, **warm,
                    restarts=restarts, train_seconds=time.time() - t_start), ckpt)
    print(f"saved {ckpt}  ({time.time() - t_start:.0f}s)")
    return ckpt


class FixedRule:
    """Non-learned stopping rule on spot paths. 'lsm': the LSM exercise rule fitted under mu = r
    (benchmarks.py, same lam); 'maturity': never exercise early (tau = T)."""

    def __init__(self, cfg: Config, rule: str):
        from benchmarks import LSMRule, load_benchmarks
        self.rule = rule
        self.N = cfg.market.N
        self.lsm = LSMRule.from_dict(load_benchmarks(cfg)["lsm_rule"]) if rule == "lsm" else None
        if rule not in ("lsm", "maturity"):
            raise ValueError(rule)

    def stop_index(self, S: torch.Tensor) -> torch.Tensor:
        """Exercise date in 1..N for spot paths S (B, N+1)."""
        if self.rule == "maturity":
            return torch.full((S.shape[0],), self.N, dtype=torch.long, device=S.device)
        return torch.from_numpy(self.lsm.stop_index(S)).to(S.device)


def train_fixed(cfg: Config, rule: str) -> Path:
    """Hedger trained by descent on CVaR_alpha(L_tau) with tau from a fixed rule (no adversary).
    Same initial weights, batch, learning rate, schedule and number of hedger updates as GDA."""
    device = get_device(cfg)
    seed_everything(cfg.seed)              # same hedger initialisation as train_gda
    mc, tr = cfg.market, cfg.train
    alpha = tr.alpha
    name = run_name(cfg, f"fixed-{rule}", alpha)
    hedger = Hedger(cfg).to(device)
    fixed = FixedRule(cfg, rule)
    opt = torch.optim.Adam(hedger.parameters(), lr=tr.lr_hedger)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tr.outer_iters)
    gen = torch.Generator().manual_seed(cfg.seed + 104729)
    B = 2 ** tr.batch_log2
    log = Logger(cfg.dir("results") / "logs" / f"{name}.csv",
                 ["it", "loss", "mean_ex_time", "frac_early", "hedge_mean", "hedge_std", "sec"])
    t_start = time.time()
    for it in range(tr.outer_iters):
        paths = simulate_cfg(cfg, B, gen, device)
        tau = fixed.stop_index(paths.S)
        ro = hedge_rollout(paths, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
        X = ro.L.gather(1, (tau - 1).unsqueeze(1)).squeeze(1)
        loss = cvar(X, alpha)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if it % tr.log_every == 0 or it == tr.outer_iters - 1:
            log.log(dict(it=it, loss=float(loss.detach()), mean_ex_time=float(paths.t[tau].mean()),
                         frac_early=float((tau < mc.N).float().mean()), hedge_mean=float(ro.delta_prev.mean()),
                         hedge_std=float(ro.delta_prev.std()), sec=round(time.time() - t_start, 1)))
    log.close()
    ckpt = cfg.dir("ckpt") / f"{name}.pt"
    torch.save(dict(stopper=None, hedger=hedger.state_dict(), cfg=cfg.to_dict(), alpha=alpha, mode="fixed",
                    rule=rule, outer_iters=tr.outer_iters, train_seconds=time.time() - t_start), ckpt)
    print(f"saved {ckpt}  ({time.time() - t_start:.0f}s)")
    return ckpt


def train_exploit(cfg: Config, source: str = "gda") -> Path:
    """Exploitability attack: freeze the hedger of `source` ('gda', 'fixed-lsm', 'fixed-maturity'),
    train a fresh (re-initialised, warm-started) stopper against it for cfg.exploit.steps steps."""
    alpha = cfg.train.alpha
    src_ckpt = cfg.dir("ckpt") / f"{run_name(cfg, source, alpha)}.pt"
    blob = torch.load(src_ckpt, map_location="cpu")
    hedger = Hedger(cfg)
    hedger.load_state_dict(blob["hedger"])
    name = run_name(cfg, "exploit" if source == "gda" else f"exploit-{source}", alpha)
    # different seed offset -> different initial weights and training paths from the GDA stopper;
    # the same offset for every source, so every hedger faces an identically initialised attacker
    _, ckpt = train_stopper(cfg, hedger, cfg.exploit.steps, alpha, name, seed_offset=50_000)
    blob_e = torch.load(ckpt)
    blob_e.update(mode="exploit", source_ckpt=str(src_ckpt.name))
    torch.save(blob_e, ckpt)
    return ckpt


if __name__ == "__main__":
    ap = add_config_args(argparse.ArgumentParser(description="Train hedger/stopper"))
    ap.add_argument("--mode", choices=["stopper_only", "gda", "fixed", "exploit"], required=True)
    ap.add_argument("--rule", choices=["lsm", "maturity"], default="lsm", help="fixed rule for --mode fixed")
    ap.add_argument("--source", default="gda", help="hedger to attack in --mode exploit: gda | fixed-lsm | fixed-maturity")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if args.mode == "stopper_only":
        train_stopper_only(cfg)
    elif args.mode == "gda":
        train_gda(cfg)
    elif args.mode == "fixed":
        train_fixed(cfg, args.rule)
    else:
        train_exploit(cfg, args.source)
