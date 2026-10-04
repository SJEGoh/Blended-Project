"""Sanity checks for the liquidity (volume-risk) extension -- no training.

1. Regression: with liquidity off, the frictionless GDA model reproduces its reported p*.
2. Cost scaling: a frozen hedger's CVaR under fixed exercise rules rises monotonically with
   liquidity.cost_scale, and cost_scale = 0 recovers the frictionless numbers.
3. Does liquidity give the holder a new lever? Simple exercise rules fitted on validation paths and
   evaluated on test paths, including two liquidity-timed rules:
     - illiquidity threshold:  exercise at the first n with ell_n >= theta,
     - post-decline:           exercise at the first n with log-return r_n <= -d.
   If a liquidity-timed rule is the strongest and clearly beats never-early / LSM, the adversary has
   something new to exploit, and the full adversarial comparison is worth running.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch

from benchmarks import LSMRule, load_benchmarks
from config import add_config_args, config_from_args, load_config
from evaluate import rollout_np
from losses import cvar_estimate, paired_cvar_diff
from models import Hedger, Stopper, ZeroHedger
from sim import get_test_set, simulate_cfg
from train import run_name


def stopped(L, tau):
    return L[np.arange(len(L)), tau - 1]


def first_hit(mask, N):
    hit = mask.any(1)
    return np.where(hit, mask.argmax(1) + 1, N)


def fit_family(X, grid, op, L, N, alpha):
    best = (None, -np.inf)
    for g in grid:
        tau = first_hit(X >= g if op == ">=" else X <= g, N)
        v = cvar_estimate(stopped(L, tau), alpha)[0]
        if v > best[1]:
            best = (float(g), v)
    return best[0]


def rules(paths, L, lsm_tau, N, params=None, alpha=0.9):
    """Stopping rules on (paths, L). params=None -> fit parametric rules on these paths."""
    S = paths.S.double().numpy()
    r = np.diff(np.log(S), axis=1)                                 # r[:, n-1] = return into date n
    ex = paths.extras
    fam = {"spot boundary": (S[:, 1:N], np.linspace(60, 100, 41), "<="),
           "P&L threshold": (L[:, :-1], np.quantile(L[:, :-1], np.linspace(0.5, 0.999, 40)), ">=")}
    if ex:
        ell = ex["ell"].double().numpy()[:, 1:N]
        fam["illiquidity threshold"] = (ell, np.quantile(ell, np.linspace(0.5, 0.999, 40)), ">=")
        fam["post-decline"] = (-r[:, :N - 1], np.linspace(0.01, 0.30, 30), ">=")
    fitted = {} if params is None else params
    out = {"never early": np.full(len(L), N), "LSM rule": lsm_tau}
    for k, (X, grid, op) in fam.items():
        if params is None:
            fitted[k] = fit_family(X, grid, op, L, N, alpha)
        out[k] = first_hit(X >= fitted[k] if op == ">=" else X <= fitted[k], N)
    return out, fitted


def load_hedger(cfg_train, src="gda"):
    blob = torch.load(cfg_train.dir("ckpt") / f"{run_name(cfg_train, src, cfg_train.train.alpha)}.pt", map_location="cpu")
    h = Hedger(cfg_train)
    h.load_state_dict(blob["hedger"])
    st = None
    if blob.get("stopper") is not None:
        st = Stopper(cfg_train)
        st.load_state_dict(blob["stopper"])
        st.eval()
    return h.eval(), st


def main(base, out_name="liqcheck.json"):
    alpha = base.train.alpha
    # frictionless adversarial model from step 3 (trained without liquidity -> 3-input hedger)
    cfg_fr = load_config(None, ["market.lam=1.0", f"train.alpha={alpha}", "tag=reduced",
                                "train.batch_log2=12", "train.outer_iters=1500"])
    hedger, stopper = load_hedger(cfg_fr)
    out = {"alpha": alpha, "liquidity": {k: getattr(base.liquidity, k) for k in ("name", "ell_max", "sig_ratio_max", "s0", "Y", "psi")}}

    # 1. regression with liquidity off
    test0 = get_test_set(cfg_fr, dtype=torch.float64)
    L0, _, tau0 = rollout_np(cfg_fr, test0, hedger, stopper)
    out["regression_p_star"] = cvar_estimate(stopped(L0, tau0), alpha)
    print(f"[regression] liquidity off: p* = {out['regression_p_star'][0]:.4f} (reported 13.5265)")

    rule = LSMRule.from_dict(load_benchmarks(cfg_fr)["lsm_rule"])
    rows = []
    for unwind in ("none", "cash"):
        for scale in (0.0, 0.5, 1.0, 2.0, 4.0):
            cfg = copy.deepcopy(base)
            cfg.liquidity.enabled = True
            cfg.liquidity.cost_scale = scale
            cfg.costs.c, cfg.costs.unwind = 0.0, unwind
            test = get_test_set(cfg, dtype=torch.float64)
            val = simulate_cfg(cfg, 2 ** 16, cfg.eval.test_seed + 1, dtype=torch.float64)
            N = cfg.market.N
            for hname, h in (("frictionless NN hedge", hedger), ("no hedge", ZeroHedger())):
                Lt, _, _ = rollout_np(cfg, test, h)
                Lv, _, _ = rollout_np(cfg, val, h)
                _, fitted = rules(val, Lv, rule.stop_index(val.S), N, alpha=alpha)
                taus, _ = rules(test, Lt, rule.stop_index(test.S), N, params=fitted, alpha=alpha)
                vals = {k: cvar_estimate(stopped(Lt, t), alpha) for k, t in taus.items()}
                strongest = max(vals, key=lambda k: vals[k][0])
                liq_best = max(("illiquidity threshold", "post-decline"), key=lambda k: vals[k][0])
                base_best = max(("never early", "LSM rule", "spot boundary", "P&L threshold"), key=lambda k: vals[k][0])
                d, d_se = paired_cvar_diff(stopped(Lt, taus[liq_best]), stopped(Lt, taus[base_best]), alpha)
                row = dict(unwind=unwind, cost_scale=scale, hedger=hname,
                           cvar={k: v[0] for k, v in vals.items()}, se={k: v[1] for k, v in vals.items()},
                           strongest=strongest, liq_rule_edge=d, liq_rule_edge_se=d_se,
                           liq_rule=liq_best, base_rule=base_best, fitted=fitted,
                           early_frac={k: float((t < N).mean()) for k, t in taus.items()})
                rows.append(row)
                print(f"[unwind={unwind:<4s} scale={scale:<3g} {hname:<22s}] "
                      + "  ".join(f"{k}={v[0]:.3f}" for k, v in vals.items())
                      + f" | strongest: {strongest} | best liquidity rule vs best other: {d:+.3f} ± {d_se:.3f}", flush=True)
    out["rows"] = rows
    (base.dir("results") / out_name).write_text(json.dumps(out, indent=1))
    print(f"wrote results/{out_name}")


if __name__ == "__main__":
    ap = add_config_args(argparse.ArgumentParser(description="Liquidity-extension sanity checks"))
    ap.add_argument("--out", default="liqcheck.json")
    a = ap.parse_args()
    main(config_from_args(a), a.out)
