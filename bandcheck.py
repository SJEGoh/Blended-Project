"""Band-rule pre-check: does the exercise assumption change the optimal hedge under fixed costs?

No neural networks are trained here. A band hedger follows a target hedge Delta_n and trades only
when it has drifted too far:

    fixed costs ('target' mode):        if |delta_{n-1} - Delta_n| > b: delta_n = Delta_n   (pay kappa)
    proportional costs ('edge' mode):   delta_n = clip(delta_{n-1}, Delta_n - b, Delta_n + b)

Targets: the Merton European put delta, and the frictionless adversarial NN hedge (GDA, c = 0).
For each cost setting and target, the band width b is tuned on validation paths under three exercise
assumptions -- hold to maturity, the risk-neutral LSM rule, and the worst of a family of simple
attacks -- and the three tuned hedgers are cross-evaluated on the test paths. If tuning for the wrong
exercise assumption costs little worst-case CVaR, the horizon effect is weak.

Attack family (each fitted on validation paths, evaluated on test): never early, LSM rule, running-loss
threshold (stop when L_n >= c), spot boundary (stop when S_n <= s), hedge staleness (stop when
|delta_{n-1} - Delta_n| >= d). The worst case is only a lower bound on what a learned adversary finds.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks import LSMRule, load_benchmarks
from config import add_config_args, config_from_args, load_config
from evaluate import rollout_np
from losses import cvar_estimate, paired_cvar_diff
from models import Hedger
from sim import get_test_set, merton_put_delta, simulate

VAL_LOG2 = 17


# --------------------------------------------------------------------------------------
def first_hit(mask: np.ndarray, N: int) -> np.ndarray:
    """mask (B, N-1) for dates 1..N-1 -> stop index in 1..N (N if never hit)."""
    hit = mask.any(1)
    return np.where(hit, mask.argmax(1) + 1, N)


def stopped(L: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return L[np.arange(len(L)), tau - 1]


def merton_targets(mc, S: np.ndarray) -> np.ndarray:
    t = mc.t_grid()
    return np.stack([merton_put_delta(S[:, n], mc.K, mc.T - t[n], mc.r, mc.sigma, mc.lam, mc.m, mc.delta)
                     for n in range(mc.N)], 1)


def nn_targets(cfg_nn, paths) -> np.ndarray:
    from train import run_name
    blob = torch.load(cfg_nn.dir("ckpt") / f"{run_name(cfg_nn, 'gda', cfg_nn.train.alpha)}.pt", map_location="cpu")
    h = Hedger(cfg_nn)
    h.load_state_dict(blob["hedger"])
    h.eval()
    _, D, _ = rollout_np(cfg_nn, paths, h)
    return D.astype(np.float64)                     # D[:, n] = delta_n, n = 0..N-1


def band_rollout(target, S, Sd, Zd, t, r, b, kappa, c_prop, mode):
    """Returns L (B, N) for dates 1..N, delta_prev (B, N) = delta_{n-1}, cumulative trades (B, N)."""
    B, N = target.shape
    d_prev = np.zeros(B)
    G = np.zeros(B)
    C = np.zeros(B)
    ntr = np.zeros(B)
    Ls, Ds, Ts = [], [], []
    for n in range(N):
        tgt = target[:, n]
        bn = b[n] if np.ndim(b) else b          # optional time-varying band
        if mode == "target":
            trade = np.abs(d_prev - tgt) > bn
            d = np.where(trade, tgt, d_prev)
        else:  # edge: minimal trade back into the band
            d = np.clip(d_prev, tgt - bn, tgt + bn)
            trade = d != d_prev
        C = C + kappa * math.exp(-r * t[n]) * trade + c_prop * np.abs(d - d_prev) * Sd[:, n]
        G = G + d * (Sd[:, n + 1] - Sd[:, n])
        ntr = ntr + trade
        Ls.append(Zd[:, n + 1] - G + C)
        Ds.append(d)
        Ts.append(ntr.copy())
        d_prev = d
    return np.stack(Ls, 1), np.stack(Ds, 1), np.stack(Ts, 1)


class Attacks:
    """Simple stopping-rule family; parameters are fitted on validation, then applied to test."""

    def __init__(self, alpha, N, lsm_rule):
        self.alpha, self.N, self.lsm = alpha, N, lsm_rule

    def taus(self, L, D, target, S, params=None):
        """All candidate rules. With params=None, fit each parametric family on (L, D, ...) and return
        (taus, params); otherwise apply the given params."""
        N, a = self.N, self.alpha
        mism = np.abs(D[:, :-1] - target[:, 1:])          # |delta_{n-1} - Delta_n|, dates 1..N-1
        fams = {
            "P&L threshold": (L[:, :-1], np.quantile(L[:, :-1], np.linspace(0.5, 0.999, 40)), ">="),
            "spot boundary": (S[:, 1:N], np.linspace(60.0, 100.0, 41), "<="),
            "hedge staleness": (mism, np.quantile(mism, np.linspace(0.5, 0.999, 40)), ">="),
        }
        out = {"never early": np.full(len(L), N), "LSM rule": self.lsm_tau}
        fitted = {}
        for name, (X, grid, op) in fams.items():
            cand = grid if params is None else [params[name]]
            best = (None, -np.inf, None)
            for g in cand:
                mask = X >= g if op == ">=" else X <= g
                tau = first_hit(mask, N)
                v = cvar_estimate(stopped(L, tau), a)[0] if params is None else 0.0
                if v > best[1]:
                    best = (g, v, tau)
            fitted[name] = float(best[0])
            out[name] = best[2]
        return out, fitted


def run(base, kappas, c_props, targets_kinds, b_grid):
    alpha = base.train.alpha
    cfg = load_config(None, [f"market.lam={base.market.lam}", f"train.alpha={alpha}"])
    mc = cfg.market
    test = get_test_set(cfg, dtype=torch.float64)
    val = simulate(mc, 2 ** VAL_LOG2, cfg.eval.test_seed + 2, dtype=torch.float64)
    rule = LSMRule.from_dict(load_benchmarks(cfg)["lsm_rule"])
    t = mc.t_grid()
    data = {}
    for nm, p in (("val", val), ("test", test)):
        S = p.S.numpy()
        data[nm] = dict(S=S, Sd=p.S_disc.numpy(), Zd=p.Z_disc.numpy(), lsm_tau=rule.stop_index(S), paths=p)
    # targets
    tg = {}
    if "merton" in targets_kinds:
        tg["Merton delta"] = {nm: merton_targets(mc, d["S"]) for nm, d in data.items()}
    if "nn" in targets_kinds:
        cfg_nn = load_config(None, [f"market.lam={mc.lam}", f"train.alpha={alpha}", "tag=reduced",
                                    "train.batch_log2=12", "train.outer_iters=1500"])
        tg["frictionless NN hedge"] = {nm: nn_targets(cfg_nn, d["paths"]) for nm, d in data.items()}
    settings = [dict(kind="fixed", kappa=k, c=0.0, mode="target") for k in kappas] + \
               [dict(kind="proportional", kappa=0.0, c=c, mode="edge") for c in c_props]
    results = []
    for st in settings:
        for tname, tdict in tg.items():
            t0 = time.time()
            curves = {k: {"val": [], "test": []} for k in ("maturity", "LSM rule", "worst attack")}
            trades = {"val": [], "test": []}
            Xs = {}          # test per-path losses under each rule, per b index
            att_names = []
            for bi, b in enumerate(b_grid):
                if np.ndim(b):                    # (b0, beta): b_n = b0 * (1 + beta * t_n / T)
                    b = b[0] * (1.0 + b[1] * t[:-1] / mc.T)
                per = {}
                for nm in ("val", "test"):
                    d = data[nm]
                    L, D, NT = band_rollout(tdict[nm], d["S"], d["Sd"], d["Zd"], t, mc.r, b, st["kappa"], st["c"], st["mode"])
                    per[nm] = (L, D, NT)
                att = Attacks(alpha, mc.N, None)
                att.lsm_tau = data["val"]["lsm_tau"]
                Lv, Dv, NTv = per["val"]
                taus_v, fitted = att.taus(Lv, Dv, tdict["val"], data["val"]["S"])
                att.lsm_tau = data["test"]["lsm_tau"]
                Lt, Dt, NTt = per["test"]
                taus_t, _ = att.taus(Lt, Dt, tdict["test"], data["test"]["S"], params=fitted)
                att_names = list(taus_t)
                for nm, (L, taus) in (("val", (Lv, taus_v)), ("test", (Lt, taus_t))):
                    vals = {k: cvar_estimate(stopped(L, tau), alpha)[0] for k, tau in taus.items()}
                    curves["maturity"][nm].append(vals["never early"])
                    curves["LSM rule"][nm].append(vals["LSM rule"])
                    curves["worst attack"][nm].append(max(vals.values()))
                trades["val"].append(float(NTv[:, -1].mean()))
                trades["test"].append(float(NTt[:, -1].mean()))
                # keep test losses for paired comparisons: per rule, and worst attack (by test value)
                tv = {k: stopped(Lt, tau) for k, tau in taus_t.items()}
                worst_k = max(tv, key=lambda k: cvar_estimate(tv[k], alpha)[0])
                Xs[bi] = {"maturity": tv["never early"], "LSM rule": tv["LSM rule"], "worst attack": tv[worst_k],
                          "worst_name": worst_k}
            # tuned band widths (selected on validation)
            chosen = {k: int(np.argmin(curves[k]["val"])) for k in curves}
            cross = {}
            for tuned_for, bi in chosen.items():
                cross[tuned_for] = {ev: cvar_estimate(Xs[bi][ev], alpha)[0] for ev in ("maturity", "LSM rule", "worst attack")}
            bw = chosen["worst attack"]
            loss = {}
            for tuned_for in ("maturity", "LSM rule"):
                bi = chosen[tuned_for]
                dlt, se = paired_cvar_diff(Xs[bi]["worst attack"], Xs[bw]["worst attack"], alpha)
                loss[tuned_for] = dict(diff=dlt, se=se, rel=dlt / cross["worst attack"]["worst attack"])
            enc = (lambda x: [float(v) for v in x]) if np.ndim(b_grid[0]) else float
            res = dict(setting=st, target=tname, b_grid=[enc(x) for x in b_grid], curves=curves, trades=trades,
                       chosen_b={k: enc(b_grid[v]) for k, v in chosen.items()},
                       chosen_trades={k: trades["test"][v] for k, v in chosen.items()},
                       worst_attack_at_chosen={k: Xs[v]["worst_name"] for k, v in chosen.items()},
                       cross=cross, robustness_loss=loss, seconds=round(time.time() - t0, 1))
            results.append(res)
            lab = f"kappa={st['kappa']:g}" if st["kind"] == "fixed" else f"c={st['c']:g}"
            fb = lambda v: f"{v:.2f}" if not isinstance(v, list) else f"(b0={v[0]:.3f}, beta={v[1]:g})"
            print(f"[{st['kind']} {lab} | {tname}] b* maturity={fb(res['chosen_b']['maturity'])} "
                  f"LSM={fb(res['chosen_b']['LSM rule'])} worst={fb(res['chosen_b']['worst attack'])} | "
                  f"worst-case loss from tuning for maturity {loss['maturity']['diff']:+.4f} ± {loss['maturity']['se']:.4f} "
                  f"({loss['maturity']['rel']:+.2%}), for LSM {loss['LSM rule']['diff']:+.4f} ± {loss['LSM rule']['se']:.4f} "
                  f"({loss['LSM rule']['rel']:+.2%}) | {res['seconds']}s", flush=True)
    return results


def figures(results, figdir):
    figdir.mkdir(parents=True, exist_ok=True)
    tnames = sorted({r["target"] for r in results})
    for tname in tnames:
        rs = [r for r in results if r["target"] == tname]
        fig, axes = plt.subplots(1, len(rs), figsize=(4.6 * len(rs), 3.9), squeeze=False)
        for ax, r in zip(axes[0], rs):
            b = np.array(r["b_grid"])
            for k, col in (("maturity", "#e09f3e"), ("LSM rule", "#d1495b"), ("worst attack", "#1f4e9c")):
                y = np.array(r["curves"][k]["test"])
                ax.plot(b, y, color=col, lw=1.5, label=f"{k} (b* = {r['chosen_b'][k]:.2f})")
                ax.axvline(r["chosen_b"][k], color=col, lw=0.8, ls=":")
            st = r["setting"]
            lab = f"fixed kappa = {st['kappa']:g}" if st["kind"] == "fixed" else f"proportional c = {st['c']:g}"
            ax.set_title(lab, fontsize=10)
            ax.set_xlabel("band width b (delta units)")
            ax.set_ylabel("CVaR_0.9 on test paths")
            ax.legend(fontsize=7)
        fig.suptitle(f"Band hedger around {tname}: CVaR vs band width under three exercise assumptions", fontsize=11)
        fig.tight_layout()
        tag = "merton" if tname.startswith("Merton") else "nn"
        fig.savefig(figdir / f"bandcheck_{tag}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    ap = add_config_args(argparse.ArgumentParser(description="Band-rule pre-check for fixed transaction costs"))
    ap.add_argument("--kappas", nargs="*", type=float, default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--props", nargs="*", type=float, default=[0.005])
    ap.add_argument("--targets", nargs="*", default=["merton", "nn"])
    ap.add_argument("--nb", type=int, default=31)
    ap.add_argument("--out", default="bandcheck.json", help="output file name in results/")
    ap.add_argument("--betas", nargs="*", type=float, default=None,
                    help="time-varying band b_n = b0 (1 + beta t_n/T): grid over b0 x beta")
    args = ap.parse_args()
    base = config_from_args(args)
    if args.betas:
        b0s = np.geomspace(0.01, 0.3, args.nb)
        b_grid = [np.array([b0, be]) for b0 in b0s for be in args.betas]
    else:
        b_grid = np.concatenate([[0.0], np.linspace(0.02, 0.6, args.nb - 1)])
    out = run(base, args.kappas, args.props, args.targets, b_grid)
    res_dir = base.dir("results")
    (res_dir / args.out).write_text(json.dumps(out, indent=1))
    if not args.betas:
        figures(out, res_dir / "figures")
    print(f"wrote results/{args.out} and figures/bandcheck_*.png")
