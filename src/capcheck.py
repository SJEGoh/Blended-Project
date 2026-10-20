"""Volume caps: how often would they bind for the existing (uncapped) hedges? -- no training.

For each frozen hedge, on validation paths (2^16, seed test_seed + 1):
  * how often the uncapped trade |delta_n - delta_{n-1}| exceeds the participation cap
        q_max_n = participation * v_n * dt * 252 / psi
    for the default position size (psi = 0.05) and the large-position scenario (psi = 1);
  * the mechanical effect of imposing the cap on the frozen hedge (psi = 1 also raises impact costs, so
    the uncapped psi = 1 row separates the cost of size from the cap itself): CVaR_alpha under never-early,
    the LSM rule and the hedge's own co-trained adversary (if any).
The hedges were trained without caps, so this measures exposure to the constraint, not a capped optimum.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from benchmarks import LSMRule, load_benchmarks
from config import config_from_dict, get_device
from evaluate import load_policy, rollout_np, stopped
from losses import cvar_estimate
from models import Hedger
from sim import simulate_cfg

HEDGES = [
    ("GDA (liquidity base)", "gda_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced.pt"),
    ("LSM-trained (liquidity base)", "fixed-lsm_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced.pt"),
    ("maturity-trained (liquidity base)", "fixed-maturity_lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced.pt"),
    ("GDA (frictionless)", "gda_lam1_a0p9_s1234_heuristic_reduced.pt"),
]


def _load(ck, blob, cfg, device):
    """Frozen hedger (and co-trained stopper, if the checkpoint has one) under config cfg."""
    if blob.get("stopper") is not None:
        h, st, _ = load_policy(ck, cfg, device)
        return h, st
    h = Hedger(cfg)
    h.load_state_dict(blob["hedger"])
    return h.eval(), None


def main(out_name: str = "capcheck.json", participation: float = 0.10):
    from config import ROOT
    rows = []
    for label, fname in HEDGES:
        ck = ROOT / "checkpoints" / fname
        blob = torch.load(ck, map_location="cpu")
        base = config_from_dict(blob["cfg"])
        device = get_device(base)
        alpha, mc = base.train.alpha, base.market
        val = simulate_cfg(base, 2 ** 16, base.eval.test_seed + 1, dtype=torch.float64)
        rule = LSMRule.from_dict(load_benchmarks(base)["lsm_rule"])
        tau_lsm = rule.stop_index(val.S)
        # 1. binding frequency of the uncapped hedge
        hedger, stopper = _load(ck, blob, base, device)
        L, D, tau_adv = rollout_np(base, val, hedger, stopper, device)
        trades = np.abs(np.concatenate([D[:, :1], np.diff(D, axis=1)], 1))          # (B, N), first from 0
        vol = val.extras["vol"].double().numpy()[:, :-1] if base.liquidity.enabled else np.ones_like(trades)
        bind = {}
        for psi in (0.05, 1.0):
            q = participation * mc.dt * 252.0 / psi * vol
            over = trades > q
            bind[f"psi={psi:g}"] = dict(q_normal=participation * mc.dt * 252.0 / psi,
                                        frac_trades_over=float(over.mean()), frac_paths_any=float(over.any(1).mean()),
                                        frac_initial_over=float(over[:, 0].mean()),
                                        frac_later_over=float(over[:, 1:].mean()))
        # 2. mechanical effect of the cap (psi = 1) on CVaR
        cvars = {}
        for scen, ovs in (("psi=0.05, no cap", []),
                          ("psi=1, no cap", ["liquidity.psi=1.0"]),
                          ("psi=1, cap", ["liquidity.psi=1.0", "volume_cap.enabled=true",
                                          f"volume_cap.participation={participation}"])):
            cfg = config_from_dict(blob["cfg"], ovs)       # psi enters costs and caps, not the liquidity state
            h, st = _load(ck, blob, cfg, device)
            Lc, Dc, tau_c = rollout_np(cfg, val, h, st, device)
            r = {"never early": cvar_estimate(Lc[:, -1], alpha), "LSM rule": cvar_estimate(stopped(Lc, tau_lsm), alpha)}
            if tau_c is not None:
                r["co-trained adversary"] = cvar_estimate(stopped(Lc, tau_c), alpha)
            cvars[scen] = {k: dict(cvar=v[0], se=v[1]) for k, v in r.items()}
            cvars[scen]["mean_abs_position"] = float(np.abs(Dc).mean())
        rows.append(dict(hedge=label, ckpt=fname, alpha=alpha, liquidity=base.liquidity.enabled,
                         binding=bind, cvar=cvars))
        print(f"[{label}] psi=1: {bind['psi=1']['frac_trades_over']:.1%} of trades over the cap "
              f"({bind['psi=1']['frac_paths_any']:.0%} of paths, initial {bind['psi=1']['frac_initial_over']:.0%}); "
              f"psi=0.05: {bind['psi=0.05']['frac_trades_over']:.2%} | "
              + " | ".join(f"{s}: " + ", ".join(f"{k} {v['cvar']:.3f}" for k, v in c.items() if isinstance(v, dict))
                           for s, c in cvars.items()), flush=True)
    out = dict(participation=participation, paths=2 ** 16, rows=rows)
    (ROOT / "results" / out_name).write_text(json.dumps(out, indent=1))
    print(f"wrote results/{out_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Volume-cap binding check for existing hedges (no training)")
    ap.add_argument("--participation", type=float, default=0.10)
    ap.add_argument("--out", default="capcheck.json")
    a = ap.parse_args()
    main(a.out, a.participation)
