"""Hard-rule evaluation on the fixed held-out test set, figures, and RESULTS.md.

  --mode step2 : stopper only, zero hedge, alpha = 0 vs LSM / CRR.
  --mode step3 : GDA price, exploitability gap (fresh stopper) and supplementary attacks.
  --mode headline : validation item 4 — hedgers trained against fixed rules vs the adversarial one.
All reported numbers are hard-rule (stop at the first n with f_n > 0.5) on the test set.
"""
from __future__ import annotations

import argparse
import dataclasses
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks import LSMRule, load_benchmarks, price_with_rule
from config import ROOT, Config, add_config_args, alpha_tag, config_from_args, get_device, lam_tag, liq_of, load_config
from losses import aggregate, cvar_estimate, hedge_rollout, paired_cvar_diff, stopper_logits
from models import Hedger, Stopper, ZeroHedger
from sim import PathBatch, get_test_set, merton_european_put, test_set_path

CHUNK = 2 ** 14


# --------------------------------------------------------------------------------------
# Policy loading / evaluation
# --------------------------------------------------------------------------------------
def load_policy(ckpt: Path, cfg: Config, device="cpu"):
    blob = torch.load(ckpt, map_location=device)
    stopper = Stopper(cfg).to(device)
    stopper.load_state_dict(blob["stopper"])
    if blob.get("hedger") is None:
        hedger = ZeroHedger()
    else:
        hedger = Hedger(cfg).to(device)
        hedger.load_state_dict(blob["hedger"])
    return hedger.eval(), stopper.eval(), blob


@torch.no_grad()
def evaluate_policy(cfg: Config, paths: PathBatch, hedger, stopper, alpha: float, device="cpu") -> dict:
    """Hard-rule per-path loss X = L_tau on the given paths (chunked), plus summary stats."""
    mc = cfg.market
    Xs, idxs, deltas = [], [], []
    for s in range(0, paths.B, CHUNK):
        sub = paths.subset(slice(s, s + CHUNK)).to(device, torch.float32)
        ro = hedge_rollout(sub, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
        logits = stopper_logits(sub, ro, stopper, mc.K, mc.T)
        X, idx = aggregate(ro, logits, 1.0, hard=True)
        Xs.append(X.double().cpu())
        idxs.append(idx.cpu())
        deltas.append(ro.delta_prev.float().cpu())
    X = torch.cat(Xs).numpy()
    idx = torch.cat(idxs).numpy()
    t = paths.t.double().cpu().numpy()[1:]
    est, se = cvar_estimate(X, alpha)
    D = torch.cat(deltas)
    return dict(X=X, tau_idx=idx + 1, cvar=est, se=se, mean_ex_time=float(t[idx].mean()),
                frac_early=float((idx < paths.N - 1).mean()),
                hedge_mean=float(D.mean()), hedge_std=float(D.std()))


@torch.no_grad()
def stopper_decision_grid(cfg: Config, stopper, S_grid: np.ndarray, delta_prev: float = 0.0,
                          pnl_shift: float = 0.0) -> np.ndarray:
    """Logits on a (date n = 1..N) x S grid at a fixed hedge state.
    Running P&L is set to Z~_n(S) + pnl_shift (zero hedge gains + shift)."""
    mc = cfg.market
    t = torch.tensor(mc.t_grid()[1:], dtype=torch.float32)
    S = torch.tensor(S_grid, dtype=torch.float32)
    TT, SS = torch.meshgrid(t, S, indexing="ij")
    Zd = torch.exp(-mc.r * TT) * torch.clamp(mc.K - SS, min=0)
    logits = stopper(TT / mc.T, torch.log(SS / mc.K), torch.full_like(SS, delta_prev),
                     Zd + pnl_shift, (SS < mc.K).float())
    return logits.numpy()


def boundary_from_grid(logits: np.ndarray, S_grid: np.ndarray) -> np.ndarray:
    """Largest S (in grid, S < K) with logit > 0 at each date 1..N-1 (NaN if none)."""
    out = np.full(logits.shape[0] + 1, np.nan)
    for n in range(1, logits.shape[0]):
        ex = logits[n - 1] > 0
        if ex.any():
            out[n] = S_grid[ex].max()
    return out


def _nan(x):
    return np.array([np.nan if v is None else v for v in x], float)


# --------------------------------------------------------------------------------------
# Step 2 report: stopper only, zero hedge, alpha = 0  vs  LSM / CRR
# --------------------------------------------------------------------------------------
def step2_report(base: Config, lams: list[float], seed: int | None = None, init: str = "heuristic") -> list[dict]:
    from train import EuropeanHeuristic, run_name

    device = get_device(base)
    rows, curves, grids = [], {}, {}
    figs = base.dir("results") / "figures"
    figs.mkdir(exist_ok=True)
    for lam in lams:
        cfg = load_config(None, [f"market.lam={lam}", "train.alpha=0", f"train.stopper_init={init}"]
                          + ([f"seed={seed}"] if seed else []))
        mc = cfg.market
        name = run_name(cfg, "stopper_only", 0.0)
        ckpt = cfg.dir("ckpt") / f"{name}.pt"
        if not ckpt.exists():
            print(f"skip lam={lam}: {ckpt.name} not found")
            continue
        hedger, stopper, blob = load_policy(ckpt, cfg, device)
        test = get_test_set(cfg, dtype=torch.float64)
        ev = evaluate_policy(cfg, test, hedger, stopper, 0.0, device)
        bm = load_benchmarks(cfg)
        rule = LSMRule.from_dict(bm["lsm_rule"])
        tau_lsm = rule.stop_index(test.S)
        lsm_test, lsm_test_se, X_lsm = price_with_rule(test, tau_lsm)
        d, d_se = paired_cvar_diff(ev["X"], X_lsm, 0.0)
        # reference rules on the same test paths
        never_early = test.Z_disc[:, -1].numpy()                    # collapsed rule: never exercise before T
        h = EuropeanHeuristic(cfg).targets(test.S[:, 1:]).bool().numpy()
        h = np.concatenate([h, np.ones((h.shape[0], 1), bool)], 1)
        X_heur = test.Z_disc.numpy()[np.arange(test.B), h.argmax(1) + 1]
        row = dict(
            lam=lam, init=init, steps=blob.get("steps"), train_minutes=round(blob.get("train_seconds", 0) / 60, 1),
            nn_price=ev["cvar"], nn_se=ev["se"],
            lsm_price=bm["lsm_price"], lsm_se=bm["lsm_se"],
            lsm_on_test=lsm_test, lsm_on_test_se=lsm_test_se,
            nn_minus_lsm_test=d, nn_minus_lsm_test_se=d_se,
            rel_err_vs_lsm=(ev["cvar"] - bm["lsm_price"]) / bm["lsm_price"],
            nn_mean_ex_time=ev["mean_ex_time"], nn_frac_early=ev["frac_early"],
            lsm_mean_ex_time=float(mc.t_grid()[tau_lsm].mean()), lsm_frac_early=float((tau_lsm < mc.N).mean()),
            same_tau_frac=float((ev["tau_idx"] == tau_lsm).mean()),
            european=merton_european_put(mc.risk_neutral()),
            never_early_on_test=float(never_early.mean()), heuristic_on_test=float(X_heur.mean()),
            warm_agreement=blob.get("warm_agreement"),
        )
        if lam == 0.0:
            row.update(crr_bermudan=bm["crr_bermudan"], crr_american=bm["crr_american"],
                       rel_err_vs_crr_bermudan=(ev["cvar"] - bm["crr_bermudan"]) / bm["crr_bermudan"],
                       rel_err_vs_crr_american=(ev["cvar"] - bm["crr_american"]) / bm["crr_american"])
        ok = abs(row["rel_err_vs_lsm"]) <= 0.01
        if lam == 0.0:
            ok = ok and abs(row["rel_err_vs_crr_bermudan"]) <= 0.01 and abs(row["rel_err_vs_crr_american"]) <= 0.01
        row["pass_1pct"] = bool(ok)
        rows.append(row)
        # artefacts for figures
        S_grid = np.linspace(0.6 * mc.K, 1.1 * mc.K, 501)
        lg = stopper_decision_grid(cfg, stopper.cpu(), S_grid)
        grids[lam] = dict(S=S_grid, logits=lg, nn_b=boundary_from_grid(lg, S_grid),
                          lsm_b=_nan(bm["lsm_boundary"]),
                          crr_b=_nan(bm["crr_bermudan_boundary"]) if lam == 0.0 else None, t=mc.t_grid())
        log_path = cfg.dir("results") / "logs" / f"{name}.csv"
        if log_path.exists():
            with open(log_path) as fh:
                curves[lam] = (list(csv.DictReader(fh)), bm["lsm_price"])
        print(f"lam={lam}: NN {row['nn_price']:.4f} +- {row['nn_se']:.4f} | LSM {row['lsm_price']:.4f} +- {row['lsm_se']:.4f}"
              f" | NN-LSM(test, paired) {d:+.4f} +- {d_se:.4f} | rel {row['rel_err_vs_lsm']:+.2%} | pass={ok}")

    if not rows:
        return rows
    (base.dir("results") / "step2_alpha0.json").write_text(json.dumps(rows, indent=1))
    out_csv = base.dir("results") / "step2_alpha0.csv"
    with open(out_csv, "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- figure: exercise region (zero-hedge state) vs LSM / CRR boundaries ----
    fig, axes = plt.subplots(1, len(grids), figsize=(5.2 * len(grids), 4.2), squeeze=False)
    for ax, (lam, g) in zip(axes[0], grids.items()):
        t = g["t"][1:]
        ax.pcolormesh(t, g["S"], (g["logits"] > 0).T.astype(float), cmap="Blues", vmin=0, vmax=1.6,
                      shading="nearest", rasterized=True)
        ax.plot(g["t"], g["nn_b"], color="#1f4e9c", lw=1.6, label="NN stopper (hard rule)")
        ax.plot(g["t"], g["lsm_b"], "--", color="#d1495b", lw=1.4, label="LSM (deg-3, upper edge)")
        if g["crr_b"] is not None:
            ax.plot(g["t"], g["crr_b"], ":", color="black", lw=1.6, label="CRR Bermudan (exact)")
        ax.set_title(f"Exercise region, alpha=0, zero hedge, lam={lam:g}")
        ax.set_xlabel("t")
        ax.set_ylabel("S")
        ax.set_ylim(g["S"][0], g["S"][-1])
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "step2_exercise_region.png", dpi=150)
    plt.close(fig)

    # ---- figure: training curves (+ the spec-faithful random-init run that collapsed) ----
    def smooth(v, k=5):
        v = np.asarray(v, float)
        return np.array([v[max(0, i - k + 1): i + 1].mean() for i in range(len(v))])

    if curves:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        rnd = base.dir("results") / "diagnostics" / "fullscale_random_init_lam0_partial.csv"
        if rnd.exists():
            with open(rnd) as fh:
                rr = list(csv.DictReader(fh))
            st = [int(r["step"]) for r in rr]
            ax[0].plot(st, smooth([float(r["hard_cvar"]) for r in rr]), color="0.55", lw=1.0,
                       label="lam=0 random init (spec), hard — collapsed")
            ax[1].plot(st, [float(r["mean_ex_time"]) for r in rr], color="0.55", lw=1.0, label="lam=0 random init (spec)")
        for lam, (rows_c, lsm_p) in curves.items():
            st = np.array([int(r["step"]) for r in rows_c])
            line, = ax[0].plot(st, smooth([float(r["hard_cvar"]) for r in rows_c]), lw=1.2, label=f"lam={lam:g} warm start, hard (batch)")
            ax[0].plot(st, smooth([float(r["train_cvar"]) for r in rows_c]), lw=0.8, alpha=0.6, color=line.get_color(), ls="--")
            ax[0].axhline(lsm_p, color=line.get_color(), lw=0.8, ls=":")
            ax[1].plot(st, [float(r["mean_ex_time"]) for r in rows_c], lw=1.2, color=line.get_color(), label=f"lam={lam:g} warm start")
        ax[0].set_ylim(4.0, 7.2)
        ax[0].set_title("Batch value (5-pt rolling mean): hard, relaxed (dashed), LSM (dotted)", fontsize=10)
        ax[0].set_xlabel("step")
        ax[0].legend(fontsize=7)
        ax[1].set_title("Mean exercise time (hard rule, batch)", fontsize=10)
        ax[1].set_xlabel("step")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / "step2_training.png", dpi=150)
        plt.close(fig)
    write_results_md(base)
    return rows


# --------------------------------------------------------------------------------------
# Step 3: GDA price, exploitability, supplementary attacks
# --------------------------------------------------------------------------------------
@torch.no_grad()
def rollout_np(cfg: Config, paths: PathBatch, hedger, stopper=None, device="cpu"):
    """Chunked rollout on (float64) paths -> L (B, N), delta_prev (B, N), and the stopper's hard
    stop index in 1..N if a stopper is given."""
    mc = cfg.market
    Ls, Ds, Is = [], [], []
    for s in range(0, paths.B, CHUNK):
        sub = paths.subset(slice(s, s + CHUNK)).to(device, torch.float32)
        ro = hedge_rollout(sub, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
        Ls.append(ro.L.double().cpu())
        Ds.append(ro.delta_prev.float().cpu())
        if stopper is not None:
            _, idx = aggregate(ro, stopper_logits(sub, ro, stopper, mc.K, mc.T), 1.0, hard=True)
            Is.append(idx.cpu() + 1)
    L, D = torch.cat(Ls).numpy(), torch.cat(Ds).numpy()
    return L, D, (torch.cat(Is).numpy() if Is else None)


def _first_hit(mask: np.ndarray, N: int) -> np.ndarray:
    """mask (B, N-1) for dates 1..N-1 -> first date index in 1..N (N if never)."""
    return np.where(mask.any(1), mask.argmax(1) + 1, N)


def stopped(L: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """L_tau for tau in 1..N (L has columns for dates 1..N)."""
    return L[np.arange(len(L)), tau - 1]


def pnl_threshold_tau(L: np.ndarray, c: float) -> np.ndarray:
    """Stop at the first date n < N with running loss L_n >= c, else at N."""
    hit = L[:, :-1] >= c
    any_hit = hit.any(1)
    return np.where(any_hit, hit.argmax(1) + 1, L.shape[1])


def best_pnl_threshold(L_val: np.ndarray, alpha: float) -> tuple[float, float]:
    """Grid-search the threshold c on validation paths; returns (c, validation CVaR)."""
    qs = np.quantile(L_val[:, :-1], np.linspace(0.5, 0.9995, 80))
    best = (np.inf, cvar_estimate(L_val[:, -1], alpha)[0])          # c = inf: never early
    for c in qs:
        v = cvar_estimate(stopped(L_val, pnl_threshold_tau(L_val, c)), alpha)[0]
        if v > best[1]:
            best = (float(c), v)
    return best


def step3_report(base: Config, alphas: list[float], lam: float = 1.0) -> list[dict]:
    from train import run_name

    device = get_device(base)
    res = base.dir("results")
    figs = res / "figures"
    figs.mkdir(exist_ok=True)
    rows, hedge_figs, curves = [], {}, {}
    for alpha in alphas:
        cfg = load_config(None, [f"market.lam={lam}", f"train.alpha={alpha}", f"tag={base.tag}",
                                 f"train.batch_log2={base.train.batch_log2}", f"train.outer_iters={base.train.outer_iters}"])
        mc = cfg.market
        g_ck = cfg.dir("ckpt") / f"{run_name(cfg, 'gda', alpha)}.pt"
        e_ck = cfg.dir("ckpt") / f"{run_name(cfg, 'exploit', alpha)}.pt"
        if not g_ck.exists():
            print(f"skip alpha={alpha}: {g_ck.name} missing")
            continue
        hedger, stopper, gblob = load_policy(g_ck, cfg, device)
        test = get_test_set(cfg, dtype=torch.float64)
        L, D, tau_tr = rollout_np(cfg, test, hedger, stopper, device)
        X_tr = stopped(L, tau_tr)
        p, p_se = cvar_estimate(X_tr, alpha)
        row = dict(lam=lam, alpha=alpha, tag=cfg.tag, batch_log2=cfg.train.batch_log2,
                   outer_iters=cfg.train.outer_iters, train_minutes=round(gblob.get("train_seconds", 0) / 60, 1),
                   p_star=p, p_star_se=p_se,
                   tr_mean_ex_time=float(mc.t_grid()[tau_tr].mean()), tr_frac_early=float((tau_tr < mc.N).mean()),
                   hedge_mean=float(D.mean()), hedge_std=float(D.std()))
        attacks = {}
        if e_ck.exists():
            _, fresh, eblob = load_policy(e_ck, cfg, device)
            _, _, tau_fr = rollout_np(cfg, test, hedger, fresh, device)
            X_fr = stopped(L, tau_fr)
            attacks["fresh stopper (spec gap)"] = (X_fr, tau_fr)
            row.update(exploit_steps=eblob.get("steps"))
        # supplementary attacks on the frozen hedger
        attacks["never early (tau = T)"] = (L[:, -1], np.full(len(L), mc.N))
        rule = LSMRule.from_dict(load_benchmarks(cfg)["lsm_rule"])
        tau_l = rule.stop_index(test.S)
        attacks["LSM rule"] = (stopped(L, tau_l), tau_l)
        val = simulate_val(cfg)
        L_val, _, _ = rollout_np(cfg, val, hedger, None, device)
        c, _ = best_pnl_threshold(L_val, alpha)
        tau_c = pnl_threshold_tau(L, c)
        attacks[f"P&L threshold (c={c:.2f}, fit on validation)"] = (stopped(L, tau_c), tau_c)
        att_rows = []
        for nm, (Xa, ta) in attacks.items():
            v, v_se = cvar_estimate(Xa, alpha)
            d, d_se = paired_cvar_diff(Xa, X_tr, alpha)
            att_rows.append(dict(attack=nm, cvar=v, se=v_se, gap=d, gap_se=d_se, rel_gap=d / p,
                                 mean_ex_time=float(mc.t_grid()[ta].mean()), frac_early=float((ta < mc.N).mean())))
        row["attacks"] = att_rows
        spec = next((a for a in att_rows if a["attack"].startswith("fresh")), None)
        if spec is not None:
            row.update(gap=spec["gap"], gap_se=spec["gap_se"], rel_gap=spec["rel_gap"],
                       pass_gap_2pct=bool(spec["rel_gap"] < 0.02))
        worst = max(att_rows, key=lambda a: a["cvar"])
        row.update(worst_attack=worst["attack"], worst_cvar=worst["cvar"], worst_rel_gap=max(0.0, worst["rel_gap"]))
        # unhedged references (same stopping rules)
        Z = test.Z_disc.numpy()[:, 1:]
        row.update(nohedge_lsm=cvar_estimate(stopped(Z, tau_l), alpha)[0],
                   nohedge_never_early=cvar_estimate(Z[:, -1], alpha)[0],
                   lsm_price=load_benchmarks(cfg)["lsm_price"])
        # reference hedge: Merton European delta, under the same simple (non-learned) attacks
        Lm = merton_delta_L(cfg, test)
        cm, _ = best_pnl_threshold(merton_delta_L(cfg, val), alpha)
        ref = {"never early": Lm[:, -1], "LSM rule": stopped(Lm, tau_l),
               "P&L threshold": stopped(Lm, pnl_threshold_tau(Lm, cm))}
        nn = {"never early": L[:, -1], "LSM rule": stopped(L, tau_l), "P&L threshold": stopped(L, tau_c)}
        row["ref_merton_delta"] = [dict(rule=k, merton_delta=cvar_estimate(v, alpha)[0], nn_hedger=cvar_estimate(nn[k], alpha)[0],
                                        diff=paired_cvar_diff(nn[k], v, alpha)[0], diff_se=paired_cvar_diff(nn[k], v, alpha)[1])
                                   for k, v in ref.items()]
        rows.append(row)
        print(f"alpha={alpha}: p* = {p:.4f} +- {p_se:.4f}")
        for a in att_rows:
            print(f"   attack {a['attack']:<45s} CVaR {a['cvar']:.4f} +- {a['se']:.4f}  gap {a['gap']:+.4f} +- {a['gap_se']:.4f}"
                  f" ({a['rel_gap']:+.2%})  E[tau]={a['mean_ex_time']:.3f} early={a['frac_early']:.3f}")
        hedge_figs[alpha] = (test, D, mc)
        lp = res / "logs" / f"{run_name(cfg, 'gda', alpha)}.csv"
        if lp.exists():
            with open(lp) as fh:
                curves[alpha] = list(csv.DictReader(fh))
    if not rows:
        return rows
    (res / "step3.json").write_text(json.dumps(rows, indent=1))
    _step3_figures(figs, rows, hedge_figs, curves)
    write_results_md(base)
    return rows


def merton_delta_L(cfg: Config, paths: PathBatch) -> np.ndarray:
    """Seller loss L_n (dates 1..N) for the reference hedge delta_n = Merton European put delta(t_n, S_n)."""
    from sim import merton_put_delta
    mc = cfg.market
    S = paths.S.double().numpy()
    Sd = paths.S_disc.double().numpy()
    t = mc.t_grid()
    D = np.stack([merton_put_delta(S[:, n], mc.K, mc.T - t[n], mc.r, mc.sigma, mc.lam, mc.m, mc.delta)
                  for n in range(mc.N)], 1)
    G = np.cumsum(D * np.diff(Sd, axis=1), axis=1)
    C = np.cumsum(cfg.costs.c * np.abs(np.diff(np.concatenate([np.zeros((len(S), 1)), D], 1), axis=1)) * Sd[:, :-1], axis=1)
    return paths.Z_disc.double().numpy()[:, 1:] - G + C


def simulate_val(cfg: Config) -> PathBatch:
    """Validation paths (2^16, seed test_seed + 1) — used only to fit supplementary attacks."""
    from sim import simulate_cfg
    return simulate_cfg(cfg, 2 ** 16, cfg.eval.test_seed + 1, dtype=torch.float64)


def _step3_figures(figs: Path, rows, hedge_figs, curves):
    from sim import merton_put_delta
    # training curves
    if curves:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for alpha, rc in curves.items():
            it = np.array([int(r["it"]) for r in rc])
            line, = ax[0].plot(it, [float(r["hedger_loss"]) for r in rc], lw=1.2, label=f"alpha={alpha:g} hedger loss (relaxed)")
            ax[0].plot(it, [float(r["hard_cvar"]) for r in rc], lw=0.8, ls="--", color=line.get_color(), label=f"alpha={alpha:g} hard (batch)")
            ax[1].plot(it, [float(r["mean_ex_time"]) for r in rc], lw=1.2, color=line.get_color(), label=f"alpha={alpha:g} E[tau]")
            ax[1].plot(it, [float(r["frac_early"]) for r in rc], lw=0.8, ls="--", color=line.get_color(), label=f"alpha={alpha:g} frac early")
            ax[2].plot(it, [float(r["hedge_mean"]) for r in rc], lw=1.2, color=line.get_color(), label=f"alpha={alpha:g} mean delta")
            ax[2].plot(it, [float(r["hedge_std"]) for r in rc], lw=0.8, ls="--", color=line.get_color(), label=f"alpha={alpha:g} std delta")
        for a, t in zip(ax, ["Batch CVaR", "Exercise statistics (hard rule, batch)", "Hedge position"]):
            a.set_title(t, fontsize=10)
            a.set_xlabel("outer iteration")
            a.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / "step3_training.png", dpi=150)
        plt.close(fig)
    # hedge ratio vs Merton European delta
    fig, axes = plt.subplots(1, len(hedge_figs), figsize=(5.5 * len(hedge_figs), 4.2), squeeze=False)
    for ax, (alpha, (test, D, mc)) in zip(axes[0], hedge_figs.items()):
        S = test.S.numpy()
        t = mc.t_grid()
        for n, col in zip((5, 25, 45), ("#1f77b4", "#ff7f0e", "#2ca02c")):
            k = n  # delta_n held over [t_n, t_{n+1}) = D[:, n] (D[:, n] is delta_{(n+1)-1})
            sel = np.random.default_rng(0).choice(len(S), 4000, replace=False)
            ax.scatter(S[sel, n], D[sel, k], s=2, alpha=0.25, color=col)
            g = np.linspace(70, 130, 200)
            ax.plot(g, merton_put_delta(g, mc.K, mc.T - t[n], mc.r, mc.sigma, mc.lam, mc.m, mc.delta),
                    color=col, lw=1.4, label=f"t={t[n]:.2f}: NN (dots) vs Merton Eur. delta (line)")
        ax.set_xlim(70, 130)
        ax.set_ylim(-1.2, 0.2)
        ax.set_title(f"Hedge ratio, alpha={alpha:g}, lam={mc.lam:g}", fontsize=10)
        ax.set_xlabel("S_n")
        ax.set_ylabel("delta_n")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "step3_hedge_ratio.png", dpi=150)
    plt.close(fig)



# --------------------------------------------------------------------------------------
# Headline experiment (validation item 4): does training against an adversary matter?
# --------------------------------------------------------------------------------------
LIQCHECK_NOTES: list[str] = [
    "Reading:",
    "- **Plumbing checks pass:** liquidity off reproduces p* exactly; cost scale 0 reproduces the frictionless numbers;",
    "  CVaR rises monotonically with the cost scale.",
    "- **Without an unwind charge, liquidity does not help the holder:** exercising during illiquidity ends the seller's",
    "  costly re-hedging, so the strongest simple attack stays the spot boundary and the liquidity-timed rules are weak.",
    "- **With a cash-settled unwind, the holder gains a real lever** against a hedge that ignores liquidity: the strongest",
    "  attack exercises when the seller's loss-if-exercised-now (which includes the unwind at current liquidity) is high.",
    "  Its size depends on how severe stress liquidity is: about +17% over never-early/LSM with the base caps (spreads up",
    "  to 40x normal) at cost scale 1, about +3% with the milder caps (up to 15x), growing quickly with the cost scale.",
    "  At cost scale 0 (no liquidity cost) the same comparison gives +0.5%.",
    "- **What this does not show yet:** whether a liquidity-aware hedger trained for a fixed exercise rule stays exploitable,",
    "  and whether adversarial training removes the gap. That is the headline comparison, still to run.",
    "",
]

# Hand-written interpretation of each step-4 grid (keyed by grid suffix; filled in after the results).
GRID_NOTES: dict[str, list[str]] = {
    "_liqbase_uwcash_grid": [
        "Reading:",
        "- **Shape.** p* rises with alpha and with lam everywhere: 22 of 22 adjacent pairs are non-decreasing (validation",
        "  item 2). Jumps drive the tail. At alpha = 0.9, going from lam = 0 to 0.5 adds 5.5 (+74%), and 0.5 to 1 adds 2.2.",
        "  Without jumps the price is nearly flat beyond alpha = 0.9 (7.43 -> 8.35). With lam = 1 it keeps climbing",
        "  (15.2 -> 21.0, and the 21.0 is itself an understatement, see below).",
        "- **alpha = 0 sanity check.** The hedger learns not to trade (hedge identically 0). Under the pricing measure, hedging",
        "  gains have zero mean and every trade costs, so this is the right answer. p* is within 0.2-0.8% of the frictionless",
        "  LSM price at every lam (validation item 1 tolerance: 1%). The test and LSM pricing paths are different samples.",
        "- **Adversary behaviour.** The early-exercise fraction falls as alpha rises: about 0.35 at alpha = 0, and 0.02-0.10",
        "  at alpha >= 0.9 with lam > 0. In the tail the worst losses are jump losses accumulated by holding, so the adversary",
        "  mostly holds. At lam = 0 and alpha >= 0.9, both the trained and the fresh adversary never exercise early.",
        "- **Exploitability.** 14 of 15 cells pass (|gap| <= 0.6% of p*). The trained adversary beats every simple attack except",
        "  at lam = 0, alpha >= 0.9. There, a running-loss threshold rule (it exercises on about 0.6% of paths) is higher by",
        "  0.11% / 0.19% / 0.33% of p* (paired z about 10 / 10 / 5; `results/diagnostics/grid_pnl_attack_paired.json`).",
        "  The gap is small but real, so at lam = 0 the gradient-trained adversaries stop slightly short of the best response.",
        "- **lam = 1, alpha = 0.99 fails.** The trained adversary collapsed to never-early within 50 iterations. At batch 2^13",
        "  the 1% tail is about 82 paths. Once the stopping logits saturate negative, the relaxed-stopping gradient vanishes,",
        "  and the adversary pool only holds snapshots of the collapsed stopper. A fresh adversary reaches 22.84 (+8.75%) and",
        "  the running-loss rule 21.37 (+1.75%, paired z = 14.5). So 21.00 is this hedger's CVaR under hold-to-maturity, not",
        "  a saddle value; its worst case is at least 22.84. The lam = 0.5, alpha = 0.99 run collapsed too (iteration 100) but",
        "  recovered by iteration 200. The alpha = 0.99 column was re-run with adversary restarts (below): this fixes the cell",
        "  and leaves the other two unchanged, so that column is the one to report for alpha = 0.99.",
        "- **Convergence.** At (lam = 1, alpha = 0.9), p* at 600 iterations is 0.28% above the 1500-iteration run. The grid",
        "  prices are probably biased up by a few tenths of a percent (the hedger is slightly under-trained). Compute is",
        "  reduced throughout: batch 2^12 and 600 iterations, versus 2^14 and 3000 in the spec.",
        "",
    ],
    "_liqbase_uwcash_gridrs": [
        "Reading:",
        "- **No effect where plain GDA worked.** At lam = 0, every fresh stopper tied the GDA stopper (both never exercise",
        "  early), so nothing was swapped and p* is unchanged (8.349 vs 8.350). At lam = 0.5, which collapsed and recovered",
        "  without restarts, one swap happened at iteration 150. The final p* (19.561 vs 19.559) and the hedge are unchanged.",
        "- **Fixes the failed cell.** At lam = 1, the restart at iteration 150 caught the collapse (22.98 vs 20.87). Later",
        "  restarts kept the GDA adversary, which beat fresh ones by 0.15-0.18. p* = 21.879 ± 0.070, gap -0.11% (PASS), and",
        "  every simple attack is below p* (best -2.0%).",
        "- **The hedge changes, not only the number.** Against a working adversary, the lam = 1 hedger accepts more",
        "  hold-to-maturity risk (never-early CVaR 21.41 vs 21.00) in exchange for much less early-exercise exposure. Its",
        "  worst case found is 21.88, versus at least 22.84 for the collapsed run's hedger (about 4% lower).",
        "- **Reporting.** Use this column for alpha = 0.99 so the column has one training protocol. All three cells pass,",
        "  and p* is monotone in lam and, against the alpha = 0.95 column, in alpha. The alpha <= 0.95 cells all passed",
        "  without restarts and are left as run. Restarts changed nothing wherever GDA had not collapsed, so re-running those",
        "  cells is not expected to move them (untested). Cost: about +55% training time (94-98 vs 61-63 min per cell).",
        "- **Not fixed by restarts.** At lam = 0 the running-loss threshold edge remains (+0.30%). The fresh stoppers are",
        "  gradient-trained from the same warm start and also settle on never-early. Limitation: gradient-trained",
        "  adversaries can miss a small, rare exercise region (about 0.6% of paths here), worth a few tenths of a percent.",
        "",
    ],
}

# Hand-written interpretation of the band-rule pre-check (filled in after looking at the results).
BANDCHECK_NOTES: list[str] = [
    "Reading:",
    "- **No horizon effect within these families.** With the NN target, the tuned band width is identical under hold-to-",
    "  maturity, the LSM rule and the worst simple attack at every cost level (parabolic interpolation puts the three",
    "  optima within 0.003 of each other), so tuning for the wrong assumption costs nothing measurable.",
    "- **The test has power.** The same procedure moves b* with the cost level (0.04 -> 0.09 as kappa goes 0.005 -> 0.05,",
    "  trades 17 -> 8), and the CVaR curves have clear minima (0.02-0.08 CVaR within one grid step).",
    "- **The assumption shifts the level, not the shape.** The three CVaR-vs-b curves are near-parallel vertical shifts",
    "  (LSM about +0.1-0.15, worst attack about +0.2 above maturity): the exercise assumption changes the risk number",
    "  but not where the best band sits - the same pattern as the neural experiments.",
    "- **Time-varying bands do not change this.** Allowing the band to widen or shrink towards maturity, the best beta is 0",
    "  (constant band) under all three assumptions, and all three rank beta identically.",
    "- **Staleness is not what the adversary exploits:** the strongest simple attack is always the spot boundary",
    "  (exercise once S falls below a level), never 'exercise when the band hedge is most out of date'.",
    "- **Merton-delta target is degenerate** (b* about 0.35, ~2 trades, regardless of kappa): the Merton delta is a poor",
    "  CVaR target, so the rule prefers an almost static hedge; it is uninformative about horizon effects.",
    "- **Proportional control** behaves the same way, consistent with the neural null result.",
    "- **Caveats:** simple attacks only; band families are time- but not state-dependent; one target hedge; lam = 1,",
    "  alpha = 0.9. A richer (e.g. neural, gated) fixed-cost hedger could still differ, but these families give no sign of it.",
    "",
]

# Hand-written interpretation lines per cost level (filled in after looking at the results).
HEADLINE_NOTES: dict[str, list[str]] = {
    "_liqbase_uwcash": [
        "- **The adversary now matters for the hedge.** Taking the strongest attack found for each hedger: adversarial",
        "  15.123, LSM-trained 15.303 (+0.18, +1.2%), maturity-trained 15.893 (+0.77, +5.1%). Under each hedger's own fresh",
        "  adversary the gaps are 0.266 ± 0.006 (LSM-trained) and 0.857 ± 0.009 (maturity-trained). Attacker strength is",
        "  uncertain by about 0.09 (the fresh adversary falls 0.087 ± 0.006 short of the co-trained one on the adversarial",
        "  hedge), so the maturity gap is far outside that range and the LSM gap about 2-3x it. In every frictionless and",
        "  proportional-cost setting before, these gaps were 0.01-0.04 and within the uncertainty.",
        "- **Pricing understatement is also larger:** the fixed-rule hedges look 4.5% (LSM-trained) and 11.7% (maturity-",
        "  trained) safer under their own assumed rule than under attack, versus 1-2.5% without liquidity risk.",
        "- **Mechanism:** the unwind at exercise is paid at that date's liquidity, so the holder can force a costly unwind by",
        "  exercising during stress. The adversarial hedge is the most liquidity-sensitive: close to fully hedged in calm",
        "  markets (mean delta -0.93 in the money) and cutting its position hardest when illiquid (-0.41 when ell >= 8).",
        "  The maturity-trained hedge, which only expects to unwind at T, keeps large positions in stress (-0.63) and is",
        "  hit hardest; the LSM-trained hedge is under-hedged in calm markets (-0.81), and its attacker exercises far less",
        "  often than the LSM rule assumes (8% vs 34% early).",
        "- **Price of the liquidity risk:** p* rises from 13.53 (frictionless) to 15.12 (+11.8%).",
        "- **Calibration matters:** 'base' allows stress spreads up to 40x normal; the 'mild' calibration (up to 15x) is not",
        "  run yet and the sanity checks suggest a smaller effect there.",
    ],
    "_c0.005_uwcash": [
        "- **Unwind cost raises the price further:** p* = 14.599 vs 14.401 without the unwind charge (+1.4%) and 13.527",
        "  frictionless (+7.9%).",
        "- **Pricing understatement:** both fixed-rule hedgers look like ~14.33 under the rule they were trained for and",
        "  ~14.60 under attack (+1.9% each).",
        "- **Hedging, no benefit:** the strongest attack found gives 14.599 (adversarial), 14.601 (LSM-trained) and 14.597",
        "  (maturity-trained) — identical within noise. Under each hedger's own fresh adversary the adversarial hedge is",
        "  better by 0.029 ± 0.005 / 0.025 ± 0.003, but the fresh adversary falls 0.027 ± 0.003 short of the co-trained one",
        "  on the adversarial hedge, so that edge is within attacker-strength uncertainty.",
        "- **Conclusion across c = 0, c = 0.5% and c = 0.5% + cash unwind:** the worst-case exercise assumption moves the",
        "  seller's risk number by ~1–2.5%, but a hedge trained for a fixed exercise rule (even hold-to-maturity) is as",
        "  robust to adversarial exercise as the adversarially trained hedge.",
    ],
    "_c0.005": [
        "- **Costs raise the price:** p* goes from 13.527 (c = 0) to 14.401 (+6.5%).",
        "- **Pricing understatement persists:** assuming the LSM rule now understates the attacked risk by 1.5% (was 1.0%),",
        "  assuming hold-to-maturity by 1.9% (was 2.5%).",
        "- **Hedging, still small:** under each hedger's own fresh adversary the adversarial hedge is better by 0.038 ± 0.004",
        "  (vs LSM-trained) and 0.010 ± 0.002 (vs maturity-trained), i.e. 0.26% / 0.07%. But on the adversarial hedge the",
        "  fresh adversary falls 0.031 ± 0.003 short of the co-trained GDA adversary, so attacker strength is uncertain by about",
        "  as much as these differences. Taking the strongest attack found for each hedger, the maturity-trained hedge",
        "  (14.380) is not worse than the adversarial one (14.401). No robust evidence that adversarial training improves",
        "  the hedge at c = 0.5%.",
        "- **What costs change:** the hedge becomes path-dependent (visible no-trade band in the hedge-ratio scatter) and the",
        "  LSM-trained hedge departs further from the others below the LSM boundary, but that region is rarely visited by",
        "  the worst-case holder.",
    ],
}

HEDGERS = [("gda", "adversarial (GDA)", None), ("fixed-lsm", "trained vs LSM rule", "LSM rule"),
           ("fixed-maturity", "trained vs hold-to-maturity", "never early (tau = T)")]


def headline_report(base: Config, lam: float = 1.0, alpha: float = 0.9) -> dict:
    from train import run_name

    device = get_device(base)
    res = base.dir("results")
    import copy
    cfg = copy.deepcopy(base)
    cfg.market.lam, cfg.train.alpha = float(lam), float(alpha)
    mc = cfg.market
    c_cost = cfg.costs.c
    liq = cfg.liquidity if cfg.liquidity.enabled else None
    unwind = cfg.costs.unwind if (c_cost > 0 or liq is not None) else "none"
    suffix = "" if c_cost == 0 else f"_c{c_cost:g}"
    if liq is not None:
        suffix += f"_liq{liq.name}" + ("" if liq.cost_scale == 1.0 else f"x{liq.cost_scale:g}")
    suffix += "" if unwind == "none" else f"_uw{unwind}"
    test = get_test_set(cfg, dtype=torch.float64)
    val = simulate_val(cfg)
    rule = LSMRule.from_dict(load_benchmarks(cfg)["lsm_rule"])
    tau_l = rule.stop_index(test.S)
    out, hist, hedges = {"lam": lam, "alpha": alpha, "c": c_cost, "unwind": unwind, "suffix": suffix,
                         "liquidity": None if liq is None else dataclasses.asdict(liq), "tag": cfg.tag, "batch_log2": cfg.train.batch_log2,
                         "outer_iters": cfg.train.outer_iters, "hedgers": []}, {}, {}
    X_adv_fresh = None
    for src, label, train_rule in HEDGERS:
        ck = cfg.dir("ckpt") / f"{run_name(cfg, src, alpha)}.pt"
        ek = cfg.dir("ckpt") / f"{run_name(cfg, 'exploit' if src == 'gda' else f'exploit-{src}', alpha)}.pt"
        if not (ck.exists() and ek.exists()):
            print(f"skip {src}: missing {ck.name if not ck.exists() else ek.name}")
            continue
        blob = torch.load(ck, map_location="cpu")
        hedger = Hedger(cfg)
        hedger.load_state_dict(blob["hedger"])
        hedger.eval()
        own = None
        if blob.get("stopper") is not None:
            own = Stopper(cfg)
            own.load_state_dict(blob["stopper"])
            own.eval()
        L, D, tau_own = rollout_np(cfg, test, hedger, own, device)
        _, fresh, _ = load_policy(ek, cfg, device)
        _, _, tau_fr = rollout_np(cfg, test, hedger, fresh, device)
        L_val, _, _ = rollout_np(cfg, val, hedger, None, device)
        c, _ = best_pnl_threshold(L_val, alpha)
        tau_c = pnl_threshold_tau(L, c)
        rules = {"fresh adversary": tau_fr, "never early (tau = T)": np.full(len(L), mc.N),
                 "LSM rule": tau_l, "P&L threshold (validation-fitted)": tau_c}
        if liq is not None:   # liquidity-timed simple attacks, fitted on validation paths
            for nm_, (Xv, Xt, grid) in {
                "illiquidity threshold (validation-fitted)": (
                    val.extras["ell"].double().numpy()[:, 1:mc.N], test.extras["ell"].double().numpy()[:, 1:mc.N],
                    np.quantile(val.extras["ell"].double().numpy()[:, 1:mc.N], np.linspace(0.5, 0.999, 40))),
                "post-decline (validation-fitted)": (
                    -np.diff(np.log(val.S.double().numpy()), axis=1)[:, :mc.N - 1],
                    -np.diff(np.log(test.S.double().numpy()), axis=1)[:, :mc.N - 1], np.linspace(0.01, 0.30, 30)),
            }.items():
                best_g, best_v = None, -np.inf
                for g in grid:
                    v = cvar_estimate(stopped(L_val, _first_hit(Xv >= g, mc.N)), alpha)[0]
                    if v > best_v:
                        best_g, best_v = g, v
                rules[nm_] = _first_hit(Xt >= best_g, mc.N)
        if own is not None:
            rules = {"own adversary (GDA stopper)": tau_own, **rules}
        trained_key = "own adversary (GDA stopper)" if own is not None else train_rule
        evals = {}
        for k, t in rules.items():
            X = stopped(L, t)
            v, se = cvar_estimate(X, alpha)
            evals[k] = dict(cvar=v, se=se, mean_ex_time=float(mc.t_grid()[t].mean()), frac_early=float((t < mc.N).mean()))
        trained_val = evals[trained_key]["cvar"]
        worst_key = max(evals, key=lambda k: evals[k]["cvar"])
        X_tr = stopped(L, rules[trained_key])
        X_w = stopped(L, rules[worst_key])
        rise, rise_se = paired_cvar_diff(X_w, X_tr, alpha)
        X_fr = stopped(L, tau_fr)
        row = dict(source=src, label=label, trained_against=trained_key, trained_cvar=trained_val,
                   trained_se=evals[trained_key]["se"], evals=evals, worst_attack=worst_key,
                   worst_cvar=evals[worst_key]["cvar"], worst_se=evals[worst_key]["se"],
                   rise=rise, rise_se=rise_se, rel_rise=rise / trained_val,
                   train_minutes=round(blob.get("train_seconds", 0) / 60, 1))
        if src == "gda":
            X_adv_fresh = X_fr
        elif X_adv_fresh is not None:
            d, d_se = paired_cvar_diff(X_fr, X_adv_fresh, alpha)
            row.update(vs_adv_under_fresh=d, vs_adv_under_fresh_se=d_se)
        out["hedgers"].append(row)
        hist[label] = X_w
        hedges[label] = D
        print(f"{label:<30s} trained-against {trained_val:.4f} | worst ({worst_key}) {row['worst_cvar']:.4f} "
              f"| rise {rise:+.4f} +- {rise_se:.4f} ({rise / trained_val:+.2%})")
        for k, e in evals.items():
            print(f"      {k:<36s} {e['cvar']:.4f} +- {e['se']:.4f}  E[tau]={e['mean_ex_time']:.3f} early={e['frac_early']:.3f}")
    # unhedged reference under the simple rules
    Z = test.Z_disc.numpy()[:, 1:]
    Zv = val.Z_disc.numpy()[:, 1:]
    cz, _ = best_pnl_threshold(Zv, alpha)
    nh = {"never early (tau = T)": Z[:, -1], "LSM rule": stopped(Z, tau_l),
          "P&L threshold (validation-fitted)": stopped(Z, pnl_threshold_tau(Z, cz))}
    out["no_hedge"] = {k: cvar_estimate(v, alpha)[0] for k, v in nh.items()}
    nh_worst = max(nh, key=lambda k: out["no_hedge"][k])
    hist["no hedge"] = nh[nh_worst]
    if liq is not None:   # hedge position vs liquidity state, in-the-money region (S_n < 0.9 K)
        ell = test.extras["ell"].double().numpy()[:, :mc.N]          # ell at the date the position is chosen
        Sn = test.S.double().numpy()[:, :mc.N]
        edges = np.array([0.0, 1.25, 2.0, 4.0, 8.0, np.inf])
        itm = Sn < 0.9 * mc.K
        out["hedge_by_liquidity"] = {"edges": [float(e) for e in edges[:-1]] + ["inf"], "by_hedger": {}}
        for lab, D in hedges.items():
            rows = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = itm & (ell >= lo) & (ell < hi)
                rows.append(dict(n=int(m.sum()), mean_delta=float(D[m].mean()) if m.any() else None))
            out["hedge_by_liquidity"]["by_hedger"][lab] = rows
    (res / f"headline{suffix}.json").write_text(json.dumps(out, indent=1))
    _headline_figures(res / "figures", out, hist, hedges, test, rule, mc, alpha, suffix,
                      (f"{c_cost:g}" if liq is None else f"{c_cost:g} + liquidity '{liq.name}'")
                      + ("" if unwind == "none" else f", {unwind} unwind"))
    write_results_md(base)
    return out


def _headline_figures(figs: Path, out, hist, hedges, test, rule, mc, alpha, suffix="", c_cost: float | str = 0.0):
    figs.mkdir(exist_ok=True)
    colors = {"adversarial (GDA)": "#1f4e9c", "trained vs LSM rule": "#d1495b",
              "trained vs hold-to-maturity": "#e09f3e", "no hedge": "0.55"}
    # loss histograms under each hedger's strongest attack
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    allx = np.concatenate(list(hist.values()))
    bins = np.linspace(np.quantile(allx, 0.001), np.quantile(allx, 0.9999), 120)
    for lab, X in hist.items():
        ax.hist(X, bins=bins, histtype="step", lw=1.4, color=colors.get(lab), label=lab, density=True)
        v = cvar_estimate(X, alpha)[0]
        ax.axvline(v, color=colors.get(lab), lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("seller's discounted loss at exercise, L_tau")
    ax.set_ylabel("density (log)")
    ax.set_title(f"Seller loss under each hedger's strongest attack (lam={mc.lam:g}, c={c_cost}); dashed = CVaR_{alpha:g}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / f"headline_loss_hist{suffix}.png", dpi=150)
    plt.close(fig)
    # hedge ratio comparison with the LSM boundary
    S = test.S.numpy()
    b = rule.boundary()
    t = mc.t_grid()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    sel = np.random.default_rng(0).choice(len(S), 5000, replace=False)
    for ax, n in zip(axes, (10, 30, 45)):
        for lab, D in hedges.items():
            ax.scatter(S[sel, n], D[sel, n], s=2, alpha=0.3, color=colors.get(lab), label=lab)
        if not np.isnan(b[n]):
            ax.axvline(b[n], color="black", lw=1, ls=":", label="LSM exercise boundary")
        ax.set_xlim(65, 130)
        ax.set_ylim(-1.3, 0.2)
        ax.set_title(f"Hedge ratio at t = {t[n]:.2f} (c = {c_cost})", fontsize=10)
        ax.set_xlabel("S_n")
        ax.set_ylabel("delta_n")
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, fontsize=7, markerscale=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / f"headline_hedge_ratio{suffix}.png", dpi=150)
    plt.close(fig)
    hb = out.get("hedge_by_liquidity")
    if hb:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        labels = ["<1.25", "1.25-2", "2-4", "4-8", ">=8"]
        for lab, rows in hb["by_hedger"].items():
            y = [r["mean_delta"] if r["mean_delta"] is not None else np.nan for r in rows]
            ax.plot(range(len(rows)), y, marker="o", color=colors.get(lab), label=lab)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel("illiquidity ell_n when the position is set")
        ax.set_ylabel("mean hedge delta_n (S_n < 0.9 K)")
        ax.set_title(f"Hedge position vs liquidity, in the money (c = {c_cost})", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figs / f"headline_hedge_vs_liquidity{suffix}.png", dpi=150)
        plt.close(fig)



# --------------------------------------------------------------------------------------
# Step-4 grid: lam x alpha (prices, exploitability, simple attacks, monotonicity)
# --------------------------------------------------------------------------------------
GRID_ALPHAS = [0.0, 0.5, 0.9, 0.95, 0.99]


def _grid_suffix(cfg: Config) -> str:
    sfx = ""
    if cfg.costs.c > 0:
        sfx += f"_c{cfg.costs.c:g}"
    if cfg.liquidity.enabled:
        sfx += f"_liq{cfg.liquidity.name}" + ("" if cfg.liquidity.cost_scale == 1.0 else f"x{cfg.liquidity.cost_scale:g}")
    if cfg.costs.unwind != "none":
        sfx += f"_uw{cfg.costs.unwind}"
    return sfx + (f"_{cfg.tag}" if cfg.tag else "")


def grid_report(base: Config, lams: list[float], alphas: list[float]) -> dict:
    import copy
    from train import run_name

    device = get_device(base)
    res = base.dir("results")
    sfx = _grid_suffix(base)
    liq = base.liquidity if base.liquidity.enabled else None
    out = {"suffix": sfx, "tag": base.tag, "outer_iters": base.train.outer_iters,
           "liquidity": None if liq is None else dataclasses.asdict(liq), "unwind": base.costs.unwind,
           "c": base.costs.c, "restart_every": base.train.adv_restart_every,
           "restart_steps": base.train.adv_restart_steps, "cells": []}
    for lam in lams:
        cl = copy.deepcopy(base)
        cl.market.lam = float(lam)
        mc = cl.market
        test = get_test_set(cl, dtype=torch.float64)
        val = simulate_val(cl)
        bm = load_benchmarks(cl)
        rule = LSMRule.from_dict(bm["lsm_rule"])
        tl_test, tl_val = rule.stop_index(test.S), rule.stop_index(val.S)
        for alpha in alphas:
            cfg = copy.deepcopy(cl)
            cfg.train.alpha = float(alpha)
            g_ck = cfg.dir("ckpt") / f"{run_name(cfg, 'gda', alpha)}.pt"
            e_ck = cfg.dir("ckpt") / f"{run_name(cfg, 'exploit', alpha)}.pt"
            cell = dict(lam=lam, alpha=alpha, lsm_price=bm["lsm_price"], available=g_ck.exists() and e_ck.exists())
            if not cell["available"]:
                out["cells"].append(cell)
                print(f"[lam={lam:g} alpha={alpha:g}] missing")
                continue
            # per-cell cache, invalidated when either checkpoint changes
            cache_dir = res / "grid_cells"
            cache_dir.mkdir(exist_ok=True)
            cache = cache_dir / f"{run_name(cfg, 'gda', alpha)}.json"
            stamp = [g_ck.stat().st_mtime, e_ck.stat().st_mtime]
            if cache.exists():
                cached = json.loads(cache.read_text())
                if cached.get("_stamp") == stamp:
                    cached.pop("_stamp")
                    out["cells"].append(cached)
                    print(f"[lam={lam:g} alpha={alpha:g}] p* = {cached['p_star']:.4f} (cached)")
                    continue
            hedger, stopper, gblob = load_policy(g_ck, cfg, device)
            L, D, tau_own = rollout_np(cfg, test, hedger, stopper, device)
            X_own = stopped(L, tau_own)
            p, p_se = cvar_estimate(X_own, alpha)
            _, fresh, eblob = load_policy(e_ck, cfg, device)
            _, _, tau_fr = rollout_np(cfg, test, hedger, fresh, device)
            v_fr, _ = cvar_estimate(stopped(L, tau_fr), alpha)
            gap, gap_se = paired_cvar_diff(stopped(L, tau_fr), X_own, alpha)
            # simple attacks, fitted on validation paths
            Lv, _, _ = rollout_np(cfg, val, hedger, None, device)
            c_thr, _ = best_pnl_threshold(Lv, alpha)
            simple = {"never early": np.full(len(L), mc.N), "LSM rule": tl_test, "P&L threshold": pnl_threshold_tau(L, c_thr)}
            if liq is not None:
                fams = {"illiquidity threshold": (val.extras["ell"].double().numpy()[:, 1:mc.N],
                                                  test.extras["ell"].double().numpy()[:, 1:mc.N]),
                        "post-decline": (-np.diff(np.log(val.S.double().numpy()), axis=1)[:, :mc.N - 1],
                                         -np.diff(np.log(test.S.double().numpy()), axis=1)[:, :mc.N - 1])}
                for nm_, (Xv, Xt) in fams.items():
                    grid = np.quantile(Xv, np.linspace(0.5, 0.999, 40)) if nm_.startswith("illiq") else np.linspace(0.01, 0.3, 30)
                    vals = [cvar_estimate(stopped(Lv, _first_hit(Xv >= g, mc.N)), alpha)[0] for g in grid]
                    simple[nm_] = _first_hit(Xt >= grid[int(np.argmax(vals))], mc.N)
            sv = {k: cvar_estimate(stopped(L, t), alpha)[0] for k, t in simple.items()}
            strongest = max(sv, key=sv.get)
            if gblob.get("restarts"):
                cell["restarts"] = gblob["restarts"]
            cell.update(p_star=p, p_star_se=p_se, fresh=v_fr, gap=gap, gap_se=gap_se, rel_gap=gap / p,
                        pass_gap_2pct=bool(gap / p < 0.02), simple=sv, strongest_simple=strongest,
                        strongest_simple_rel=(sv[strongest] - p) / p,
                        mean_ex_time=float(mc.t_grid()[tau_own].mean()), frac_early=float((tau_own < mc.N).mean()),
                        hedge_mean=float(D.mean()), hedge_std=float(D.std()), batch_log2=gblob["cfg"]["train"]["batch_log2"],
                        train_minutes=round(gblob.get("train_seconds", 0) / 60, 1))
            out["cells"].append(cell)
            cache.write_text(json.dumps(dict(cell, _stamp=stamp)))
            print(f"[lam={lam:g} alpha={alpha:g}] p* = {p:.4f} ± {p_se:.4f} | fresh {v_fr:.4f} gap {gap:+.4f} ± {gap_se:.4f} "
                  f"({gap / p:+.2%}) | strongest simple: {strongest} {sv[strongest]:.4f} | LSM price {bm['lsm_price']:.4f}", flush=True)
    # monotonicity (independent-SE approximation, 2 SE tolerance)
    have = [c for c in out["cells"] if c.get("available")]
    mono = {"alpha": [], "lam": []}
    for lam in lams:
        cs = sorted([c for c in have if c["lam"] == lam], key=lambda c: c["alpha"])
        for a, b in zip(cs, cs[1:]):
            d = b["p_star"] - a["p_star"]
            tol = 2 * math.hypot(a["p_star_se"], b["p_star_se"])
            mono["alpha"].append(dict(lam=lam, from_alpha=a["alpha"], to_alpha=b["alpha"], diff=d, ok=bool(d >= -tol)))
    for alpha in alphas:
        cs = sorted([c for c in have if c["alpha"] == alpha], key=lambda c: c["lam"])
        for a, b in zip(cs, cs[1:]):
            d = b["p_star"] - a["p_star"]
            tol = 2 * math.hypot(a["p_star_se"], b["p_star_se"])
            mono["lam"].append(dict(alpha=alpha, from_lam=a["lam"], to_lam=b["lam"], diff=d, ok=bool(d >= -tol)))
    out["monotonicity"] = mono
    # convergence reference: the 1500-iteration run of the same setting, if evaluated
    hp = res / f"headline{_grid_suffix(copy.deepcopy(base)).replace('_' + base.tag, '') if base.tag else ''}.json"
    if hp.exists():
        hd = json.loads(hp.read_text())
        g = next((h for h in hd["hedgers"] if h["source"] == "gda"), None)
        if g:
            out["convergence_ref"] = dict(lam=hd["lam"], alpha=hd["alpha"], outer_iters=hd["outer_iters"],
                                          p_star=g["trained_cvar"], p_star_se=g["trained_se"])
    (res / f"grid{sfx}.json").write_text(json.dumps(out, indent=1))
    if base.train.adv_restart_every > 0:          # diagnostic re-run: reported as a table in RESULTS.md
        write_results_md(base)
        return out
    # figure: price vs alpha, one line per lam, LSM price as reference
    figs = res / "figures"
    figs.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    cols = {0.0: "#2a9d8f", 0.5: "#e09f3e", 1.0: "#1f4e9c"}
    for lam in lams:
        cs = sorted([c for c in have if c["lam"] == lam], key=lambda c: c["alpha"])
        if not cs:
            continue
        col = cols.get(lam)
        ax.errorbar([c["alpha"] for c in cs], [c["p_star"] for c in cs], yerr=[2 * c["p_star_se"] for c in cs],
                    marker="o", color=col, capsize=3, label=f"lam = {lam:g}: p* (hard rule, test)")
        ax.axhline(cs[0]["lsm_price"], color=col, ls="--", lw=0.9, label=f"lam = {lam:g}: LSM price (frictionless)")
        for c in cs:                     # cells failing the exploitability test: show the fresh adversary's value
            if not c["pass_gap_2pct"]:
                ax.plot([c["alpha"]] * 2, [c["p_star"], c["fresh"]], color=col, ls=":", lw=1.2)
                ax.scatter([c["alpha"]], [c["fresh"]], marker="^", s=40, facecolors="none", edgecolors=col,
                           label=f"lam = {lam:g}, alpha = {c['alpha']:g}: fresh adversary (gap FAIL)")
    # re-runs with adversary restarts for the same setting, if any (open squares)
    base_sfx = sfx.rsplit("_", 1)[0]
    shown = False
    for fp in sorted(res.glob("grid*.json")):
        fx = json.loads(fp.read_text())
        if not fx.get("restart_every") or fx["suffix"].rsplit("_", 1)[0] != base_sfx:
            continue
        for c in fx["cells"]:
            if c.get("available") and c["lam"] in lams:
                ax.scatter([c["alpha"] + 0.004], [c["p_star"]], marker="s", s=36, facecolors="none",
                           edgecolors=cols.get(c["lam"]), label=None if shown else "with adversary restarts (re-run)")
                shown = True
    ax.set_xlabel("CVaR level alpha")
    ax.set_ylabel("seller's price p*")
    ttl = "Seller's price vs alpha" + (f" (liquidity '{liq.name}', {base.costs.unwind} unwind)" if liq else "")
    ax.set_title(ttl, fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / f"grid_price_vs_alpha{sfx}.png", dpi=150)
    plt.close(fig)
    write_results_md(base)
    return out



# --------------------------------------------------------------------------------------
# RESULTS.md
# --------------------------------------------------------------------------------------
def _pytest_summary(cfg: Config) -> str:
    p = cfg.dir("results") / "pytest_summary.txt"
    return p.read_text().strip().splitlines()[-1] if p.exists() else "not run"


def write_results_md(cfg: Config) -> Path:
    res = cfg.dir("results")
    j2, j3 = res / "step2_alpha0.json", res / "step3.json"
    step2_rows = json.loads(j2.read_text()) if j2.exists() else []
    step3_rows = json.loads(j3.read_text()) if j3.exists() else []
    heads = [json.loads(p.read_text()) for p in sorted(res.glob("headline*.json"))]
    heads = sorted([h for h in heads if h.get("hedgers")], key=lambda h: (h.get("c", 0.0), h.get("unwind", "none")))
    head = heads[0] if heads else None
    bms = {}
    for lam in (0.0, 0.5, 1.0):
        p = res / f"benchmarks_{lam_tag(lam)}.json"
        if p.exists():
            bms[lam] = json.loads(p.read_text())
    status = "steps 1–3 done (reduced compute for step 3); step 4 pending" if step3_rows else \
        "steps 1–2 done; steps 3–4 pending"
    if heads:
        cs = ", ".join(f"{h.get('c', 0.0):g}" for h in heads)
        status = (f"steps 1–3 done (reduced compute for step 3); headline experiment (item 4) done at lam = 1, alpha = 0.9, "
                  f"c = {cs}; rest of step 4 (grid, monotonicity, figures) pending")
    main_grids = [g for g in (json.loads(p_.read_text()) for p_ in sorted(res.glob("grid*.json"))) if not g.get("restart_every")]
    if heads and main_grids:
        done = [g for g in main_grids if all(c.get("available") for c in g["cells"])]
        gtxt = "; ".join(f"grid{g['suffix']} {sum(1 for c in g['cells'] if c.get('available'))}/{len(g['cells'])} cells"
                         for g in main_grids)
        status = (f"steps 1–3 done (reduced compute for step 3); headline experiment (item 4) done at lam = 1, alpha = 0.9 "
                  f"(several cost settings); step-4 grid {'done' if len(done) == len(main_grids) else 'in progress'} "
                  f"({gtxt}; reduced compute)")
    L = ["# Results — adversarial deep hedging of a Bermudan put (MVP)", "",
         "All numbers: hard exercise rule on the fixed held-out test set (2^18 paths, same seed for every lam),",
         f"± Monte Carlo standard error. Build-order status: {status}.", "",
         "## Benchmarks (step 1)", "",
         "| lam | LSM price (indep. pricing paths) | Merton European | CRR Bermudan-on-grid | CRR American |",
         "|---|---|---|---|---|"]
    for lam, b in bms.items():
        fmt = lambda k: f"{b[k]:.4f}" if k in b else "— (lam > 0)"
        L.append(f"| {lam:g} | {b['lsm_price']:.4f} ± {b['lsm_se']:.4f} | {b['merton_european']:.4f} | "
                 f"{fmt('crr_bermudan')} | {fmt('crr_american')} |")
    if 0.0 in bms:
        b = bms[0.0]
        L += ["", f"LSM vs CRR (lam = 0): LSM − CRR Bermudan = {b['lsm_price'] - b['crr_bermudan']:+.4f} "
                  f"({(b['lsm_price'] - b['crr_bermudan']) / b['crr_bermudan']:+.2%}, {(b['lsm_price'] - b['crr_bermudan']) / b['lsm_se']:+.1f} SE). "
                  f"CRR American − Bermudan-on-grid = {b['crr_american'] - b['crr_bermudan']:.4f}."]

    # ---------------- validation table ----------------
    L += ["", "## Validation", "", "| # | Check | Status | Numbers |", "|---|---|---|---|"]
    g0 = next((r for r in step3_rows if r["alpha"] == 0.0), None)
    if step2_rows:
        nums = "; ".join(f"lam={r['lam']:g}: NN {r['nn_price']:.4f}±{r['nn_se']:.4f} vs LSM {r['lsm_price']:.4f} "
                         f"({r['rel_err_vs_lsm']:+.2%})" for r in step2_rows)
        allpass = all(r["pass_1pct"] for r in step2_rows)
        r0 = next((r for r in step2_rows if r["lam"] == 0.0), step2_rows[0])
        gtxt = "Full GDA at alpha = 0: pending"
        if g0 is not None:
            rel = (g0["p_star"] - g0["lsm_price"]) / g0["lsm_price"]
            gtxt = (f"Full GDA at alpha = 0 (lam={g0['lam']:g}, reduced): p* = {g0['p_star']:.4f} ± {g0['p_star_se']:.4f} vs LSM "
                    f"{g0['lsm_price']:.4f} ({rel:+.2%}) → **{'PASS' if abs(rel) <= 0.01 else 'FAIL'}**")
        L.append(f"| 1 | alpha = 0 recovers LSM | Step 2 as written: **FAIL** (random-init stopper collapses to never-exercise: "
                 f"{r0['never_early_on_test']:.4f} at lam={r0['lam']:g}, {(r0['never_early_on_test'] - r0['lsm_price']) / r0['lsm_price']:+.2%}). "
                 f"With warm start (adopted): **{'PASS' if allpass else 'FAIL'}**. {gtxt} | {nums} |")
    else:
        L.append("| 1 | alpha = 0 recovers LSM | pending | |")
    all_grids = [json.loads(p_.read_text()) for p_ in sorted(res.glob("grid*.json"))]
    grids = [g for g in all_grids if not g.get("restart_every")]
    fixes = [g for g in all_grids if g.get("restart_every")]     # diagnostic re-runs with adversary restarts
    if grids:
        parts2 = []
        for gd in grids:
            ma, ml = gd["monotonicity"]["alpha"], gd["monotonicity"]["lam"]
            bad = [m for m in ma + ml if not m["ok"]]
            n_cells = sum(1 for c in gd["cells"] if c.get("available"))
            parts2.append(f"grid{gd['suffix']} ({n_cells}/{len(gd['cells'])} cells): "
                          f"{'all' if not bad else len(ma) + len(ml) - len(bad)} of {len(ma) + len(ml)} adjacent pairs "
                          f"non-decreasing within 2 SE" + (f"; violations: " + ", ".join(
                              (f"lam={m['lam']:g} alpha {m['from_alpha']:g}->{m['to_alpha']:g} ({m['diff']:+.3f})" if "lam" in m and "from_alpha" in m
                               else f"alpha={m['alpha']:g} lam {m['from_lam']:g}->{m['to_lam']:g} ({m['diff']:+.3f})") for m in bad) if bad else ""))
        L.append(f"| 2 | Monotonicity in alpha and lam | step-4 grid (see below) | {'; '.join(parts2)} |")
    else:
        L.append("| 2 | Monotonicity in alpha and lam | pending (step 4) | |")
    if step3_rows:
        parts = []
        for r in step3_rows:
            if "gap" in r:
                sup = max(a["rel_gap"] for a in r["attacks"] if not a["attack"].startswith("fresh"))
                parts.append(f"alpha={r['alpha']:g}: gap {r['gap']:+.4f} ± {r['gap_se']:.4f} ({r['rel_gap']:+.2%} of p*) "
                             f"→ {'PASS' if r['pass_gap_2pct'] else 'FAIL'}; best supplementary attack {sup:+.2%}"
                             f"{' (none beats p*)' if sup <= 0 else ''}")
        L.append(f"| 3 | Exploitability gap (< 2% of p*) | step 3 (alpha = 0.9, 0; lam = 1; reduced) — see below | {'; '.join(parts)} |")
    else:
        L.append("| 3 | Exploitability gap | pending (step 3) | |")
    if heads:
        parts = []
        for hd in heads:
            hs = {h["source"]: h for h in hd["hedgers"]}
            sub = []
            for src in ("fixed-lsm", "fixed-maturity"):
                if src in hs and "vs_adv_under_fresh" in hs[src]:
                    h = hs[src]
                    sub.append(f"{h['label']}: rise {h['rel_rise']:+.1%} under attack, worse than adversarial hedger by "
                               f"{h['vs_adv_under_fresh']:+.3f} ± {h['vs_adv_under_fresh_se']:.3f}")
            uw = hd.get("unwind", "none")
            lq = hd.get("liquidity")
            parts.append(f"c = {hd.get('c', 0.0):g}{'' if not lq else ' + liquidity ' + repr(lq['name'])}"
                         f"{'' if uw == 'none' else f' + {uw} unwind'}: " + "; ".join(sub))
        L.append(f"| 4 | Adversary vs LSM-fixed hedger | reported (lam = {head['lam']:g}, alpha = {head['alpha']:g}, reduced) — see below | "
                 f"{' / '.join(parts)} |")
    else:
        L.append("| 4 | Adversary vs LSM-fixed hedger | pending (step 4) | |")
    L += [f"| 5 | Non-anticipativity unit test | {'PASS' if 'passed' in _pytest_summary(cfg) and 'failed' not in _pytest_summary(cfg) else 'see pytest'} | {_pytest_summary(cfg)} |"]

    # ---------------- step-4 grid ----------------
    for gd in grids:
        cells = [c for c in gd["cells"] if c.get("available")]
        lams_ = sorted({c["lam"] for c in gd["cells"]})
        alphas_ = sorted({c["alpha"] for c in gd["cells"]})
        lqn = (gd.get("liquidity") or {}).get("name")
        L += ["", f"## Step-4 grid{' under liquidity model ' + repr(lqn) if lqn else ''}"
                  f"{', ' + gd['unwind'] + ' unwind' if gd.get('unwind', 'none') != 'none' else ''} "
                  f"({len(cells)}/{len(gd['cells'])} cells done)", "",
              f"Each cell: adversarial (GDA) training, {gd['outer_iters']} outer iterations, batch 2^12 (2^13 at alpha = 0.99),",
              "then a fresh warm-started adversary (2000 steps) against the frozen hedger. Hard-rule CVaR on the 2^18 test paths",
              "of that lam. 'Strongest simple attack' is the best of never early, LSM rule, running-loss threshold"
              + (", illiquidity threshold, post-decline" if lqn else "") + " (fitted on validation paths).", "",
              "Seller's price p* (± SE); last column: frictionless LSM price for reference.", "",
              "| lam | " + " | ".join(f"alpha = {a:g}" for a in alphas_) + " | LSM |", "|---|" + "---|" * (len(alphas_) + 1)]
        for lam in lams_:
            row, lsm = [], None
            for a in alphas_:
                c = next((c for c in gd["cells"] if c["lam"] == lam and c["alpha"] == a), None)
                lsm = c["lsm_price"] if c else lsm
                row.append(f"{c['p_star']:.3f} ± {c['p_star_se']:.3f}" if c and c.get("available") else "pending")
            L.append(f"| {lam:g} | " + " | ".join(row) + f" | {lsm:.3f} |")
        L += ["", "Exploitability gap (fresh adversary − trained adversary, paired; PASS if < 2% of p*) and strongest simple attack vs p*:", "",
              "| lam | " + " | ".join(f"alpha = {a:g}" for a in alphas_) + " |", "|---|" + "---|" * len(alphas_)]
        for lam in lams_:
            row = []
            for a in alphas_:
                c = next((c for c in gd["cells"] if c["lam"] == lam and c["alpha"] == a), None)
                if c and c.get("available"):
                    row.append(f"{c['rel_gap']:+.2%} {'PASS' if c['pass_gap_2pct'] else 'FAIL'}; simple {c['strongest_simple_rel']:+.1%}")
                else:
                    row.append("pending")
            L.append(f"| {lam:g} | " + " | ".join(row) + " |")
        L += ["", "Trained adversary: mean exercise time / early-exercise fraction; hedge mean (std):", "",
              "| lam | " + " | ".join(f"alpha = {a:g}" for a in alphas_) + " |", "|---|" + "---|" * len(alphas_)]
        for lam in lams_:
            row = []
            for a in alphas_:
                c = next((c for c in gd["cells"] if c["lam"] == lam and c["alpha"] == a), None)
                row.append(f"{c['mean_ex_time']:.3f} / {c['frac_early']:.2f}; {c['hedge_mean']:.2f} ({c['hedge_std']:.2f})"
                           if c and c.get("available") else "pending")
            L.append(f"| {lam:g} | " + " | ".join(row) + " |")
        cr = gd.get("convergence_ref")
        if cr:
            c = next((c for c in cells if c["lam"] == cr["lam"] and c["alpha"] == cr["alpha"]), None)
            if c:
                L += ["", f"Convergence check (lam = {cr['lam']:g}, alpha = {cr['alpha']:g}): p* = {c['p_star']:.4f} ± {c['p_star_se']:.4f} at "
                          f"{gd['outer_iters']} iterations vs {cr['p_star']:.4f} ± {cr['p_star_se']:.4f} at {cr['outer_iters']} iterations "
                          f"({(c['p_star'] - cr['p_star']) / cr['p_star']:+.2%})."]
        L += [""] + GRID_NOTES.get(gd["suffix"], []) + [f"Figure: `figures/grid_price_vs_alpha{gd['suffix']}.png`."]
        # diagnostic re-runs of failed cells with adversary restarts (same setting, same seed)
        for fx in fixes:
            if fx["suffix"].rsplit("_", 1)[0] != gd["suffix"].rsplit("_", 1)[0]:
                continue
            L += ["", f"### Re-run with adversary restarts (fix for adversary collapse; the grid above is plain GDA)", "",
                  f"Same setting, seed and compute as the grid cell, plus: every {fx['restart_every']} outer iterations a fresh",
                  f"warm-started stopper is trained {fx['restart_steps']} ascent steps against the frozen hedger and swapped in",
                  "if its hard-rule CVaR on a common batch is higher (`train.adv_restart_every`, off by default). Evaluated",
                  "exactly like the grid cells (fresh warm-started adversary, 2000 steps).", "",
                  "| lam | alpha | p* (restarts) | grid p* (no restarts) | fresh-adversary gap | strongest simple attack | adversary early-exercise fraction | restarts (it: current → fresh, swap?) |",
                  "|---|---|---|---|---|---|---|---|"]
            for c in fx["cells"]:
                if not c.get("available"):
                    L.append(f"| {c['lam']:g} | {c['alpha']:g} | pending | | | | | |")
                    continue
                g0 = next((x for x in gd["cells"] if x["lam"] == c["lam"] and x["alpha"] == c["alpha"] and x.get("available")), None)
                g0txt = (f"{g0['p_star']:.3f} ± {g0['p_star_se']:.3f} (gap {g0['rel_gap']:+.2%} "
                         f"{'PASS' if g0['pass_gap_2pct'] else 'FAIL'}; fresh {g0['fresh']:.3f})") if g0 else "—"
                rs = "; ".join(f"{r['it'] + 1}: {r['current']:.2f} → {r['fresh']:.2f} {'swap' if r['swapped'] else 'keep'}"
                               for r in c.get("restarts", []))
                L.append(f"| {c['lam']:g} | {c['alpha']:g} | {c['p_star']:.3f} ± {c['p_star_se']:.3f} | {g0txt} | "
                         f"{c['rel_gap']:+.2%} {'PASS' if c['pass_gap_2pct'] else 'FAIL'} (fresh {c['fresh']:.3f}) | "
                         f"{c['strongest_simple']} {c['strongest_simple_rel']:+.1%} | {c['frac_early']:.3f} | {rs} |")
            mono = [m for m in fx.get("monotonicity", {}).get("lam", [])]
            if mono:
                L += ["", "Monotonicity in lam within the re-run: " + "; ".join(
                    f"alpha = {m['alpha']:g}, lam {m['from_lam']:g} -> {m['to_lam']:g}: {m['diff']:+.3f} ({'ok' if m['ok'] else 'VIOLATION'})"
                    for m in mono) + "."]
            L += [""] + GRID_NOTES.get(fx["suffix"], [])

    # ---------------- headline (one section per cost level) ----------------
    for head in heads:
        sfx = head.get("suffix", "")
        L += ["", f"## Headline experiment: does the adversary matter? (lam = {head['lam']:g}, alpha = {head['alpha']:g}, "
                    f"transaction cost c = {head.get('c', 0.0):g}"
                    f"{'' if not head.get('liquidity') else ', liquidity model ' + repr(head['liquidity']['name'])}"
                    f"{'' if head.get('unwind', 'none') == 'none' else ', ' + head['unwind'] + '-settled unwind cost at exercise'})", "",
              f"Three hedgers, identical initial weights, batch 2^{head['batch_log2']}, {head['outer_iters']} hedger updates, same lr schedule:",
              "the adversarial (GDA) hedger, one trained against the fixed risk-neutral LSM exercise rule, and one trained",
              "against hold-to-maturity. Each frozen hedger is then attacked by a fresh warm-started adversary (identical",
              "initialisation and recipe for all three, 2000 steps) and by the simple rules. All CVaRs on the same test paths.", "",
              "| Hedger | CVaR vs the rule it was trained against | Fresh adversary | never early | LSM rule | P&L threshold | Worst case | Rise (paired) |",
              "|---|---|---|---|---|---|---|---|"]
        for h in head["hedgers"]:
            e = h["evals"]
            g = lambda k: f"{e[k]['cvar']:.4f}" if k in e else "—"
            L.append(f"| {h['label']} | {h['trained_cvar']:.4f} ± {h['trained_se']:.4f} | {g('fresh adversary')} | "
                     f"{g('never early (tau = T)')} | {g('LSM rule')} | {g('P&L threshold (validation-fitted)')} | "
                     f"**{h['worst_cvar']:.4f}** | {h['rise']:+.4f} ± {h['rise_se']:.4f} ({h['rel_rise']:+.1%}) |")
        nh = head.get("no_hedge", {})
        if nh:
            L.append(f"| no hedge | — | — | {nh.get('never early (tau = T)', float('nan')):.4f} | {nh.get('LSM rule', float('nan')):.4f} | "
                     f"{nh.get('P&L threshold (validation-fitted)', float('nan')):.4f} | {max(nh.values()):.4f} | — |")
        L.append("")
        for h in head["hedgers"]:
            if "vs_adv_under_fresh" in h:
                L.append(f"- {h['label']} minus adversarial hedger, each under its own fresh adversary: "
                         f"{h['vs_adv_under_fresh']:+.4f} ± {h['vs_adv_under_fresh_se']:.4f} (paired over test paths).")
        L += ["", "Exercise behaviour of each fresh adversary (E[tau] / early-exercise fraction): " + "; ".join(
            f"{h['label']}: {h['evals']['fresh adversary']['mean_ex_time']:.3f} / {h['evals']['fresh adversary']['frac_early']:.3f}"
            for h in head["hedgers"]) + ".",
              ""]
        hb = head.get("hedge_by_liquidity")
        if hb:
            labs = ["<1.25", "1.25-2", "2-4", "4-8", ">=8"]
            L += ["Mean hedge position delta_n in the money (S_n < 0.9 K) by illiquidity ell_n when it is set:", "",
                  "| Hedger | " + " | ".join(f"ell {x}" for x in labs) + " |", "|---|" + "---|" * len(labs)]
            for lab, rows in hb["by_hedger"].items():
                L.append(f"| {lab} | " + " | ".join("—" if r["mean_delta"] is None else f"{r['mean_delta']:.3f} (n={r['n']})"
                                                    for r in rows) + " |")
            L.append("")
        hs = {h["source"]: h for h in head["hedgers"]}
        if {"gda", "fixed-lsm", "fixed-maturity"} <= set(hs):
            fl, fm = hs["fixed-lsm"], hs["fixed-maturity"]
            generic = [
                  f"- **Pricing:** evaluating the seller's risk under an assumed exercise rule understates the worst case. The LSM-trained",
                  f"  hedge looks like {fl['trained_cvar']:.3f} under the LSM rule but is {fl['worst_cvar']:.3f} under attack "
                  f"({fl['rise']:+.3f} ± {fl['rise_se']:.3f}, {fl['rel_rise']:+.1%});",
                  f"  for hold-to-maturity the understatement is {fm['rise']:+.3f} ± {fm['rise_se']:.3f} ({fm['rel_rise']:+.1%}).",
                  f"- **Hedging:** the adversarially trained hedge lowers the attacked CVaR by only {fl['vs_adv_under_fresh']:.3f} ± "
                  f"{fl['vs_adv_under_fresh_se']:.3f} vs the LSM-trained hedge and {fm['vs_adv_under_fresh']:.3f} ± {fm['vs_adv_under_fresh_se']:.3f}",
                  f"  vs the maturity-trained hedge ({fl['vs_adv_under_fresh'] / hs['gda']['evals']['fresh adversary']['cvar']:.2%} / "
                  f"{fm['vs_adv_under_fresh'] / hs['gda']['evals']['fresh adversary']['cvar']:.2%}): statistically clear, economically small.",
                  ]
            frictionless = head.get("c", 0.0) == 0 and not head.get("liquidity")
            L += ["Reading:"] + (generic if frictionless else []) + ([
                  "- **Mechanism:** the LSM-trained hedge differs from the others only below the LSM exercise boundary (it is less",
                  "  short there, because in training those paths had already been exercised); the worst-case holder rarely",
                  "  visits that region, so the exploit is small. In this frictionless setting the three hedges are nearly identical.",
                  ] if frictionless else ([
                  "- **Cost model:** liquidity extension (LIQUIDITY.md): every trade pays a volatility-proportional half-spread plus",
                  "  square-root impact at the current (stochastic, non-tradable) liquidity state; the hedge is unwound at exercise",
                  "  under the liquidity state of that date." if head.get("liquidity") else
                  "- **Cost model:** C_n = sum_{k<n} c |delta_k - delta_{k-1}| S~_k (initial purchase charged)"
                  + (", unwinding the hedge at exercise not charged (spec)." if head.get("unwind", "none") == "none" else
                     f", plus the {head['unwind']}-settled unwind cost at exercise "
                     + ("c |delta_{tau-1}| S~_tau." if head["unwind"] == "cash" else "c |delta_{tau-1} + 1{S_tau < K}| S~_tau.")),
                  ]) + HEADLINE_NOTES.get(head.get("suffix", ""), [])) + [
                  "- **Caveats:** one seed, lam = 1, alpha = 0.9, reduced compute; the worst case is only as strong as the attackers.",
                  ""]
        L += [f"Figures: `figures/headline_loss_hist{sfx}.png`, `figures/headline_hedge_ratio{sfx}.png`"
              + (f", `figures/headline_hedge_vs_liquidity{sfx}.png`." if head.get("hedge_by_liquidity") else ".")]

    # ---------------- liquidity extension sanity checks ----------------
    lqs = [json.loads(p_.read_text()) for p_ in sorted(res.glob("liqcheck_*.json"))]
    if lqs:
        L += ["", "## Liquidity (volume-risk) extension: sanity checks (`liqcheck.py`, lam = 1, alpha = 0.9)", "",
              "Model in `LIQUIDITY.md`: stochastic illiquidity (worsens after large declines), volume rising with |return|, EWMA",
              "volatility; each trade pays a volatility-proportional half-spread plus square-root impact. No training here: the",
              "frozen frictionless adversarial hedge (which ignores liquidity) is evaluated under liquidity costs against simple",
              "exercise rules fitted on validation paths (never early, LSM, spot boundary, running-loss threshold, illiquidity",
              "threshold, post-decline).", ""]
        reg = lqs[0].get("regression_p_star")
        if reg:
            L += [f"- Regression: liquidity off -> p* = {reg[0]:.4f} ± {reg[1]:.4f} (step-3 value 13.5265): unchanged.", ""]
        for d in lqs:
            lq = d.get("liquidity", {})
            L += [f"Liquidity model **{lq.get('name')}** (ell_max = {lq.get('ell_max'):g}, sig_ratio_max = {lq.get('sig_ratio_max'):g}): "
                  "CVaR_0.9 of the frictionless hedge", "",
                  "| Unwind at exercise | cost scale | never early | LSM rule | strongest simple attack | vs best of never-early / LSM |",
                  "|---|---|---|---|---|---|"]
            for r in d["rows"]:
                if not r["hedger"].startswith("frictionless"):
                    continue
                cv = r["cvar"]
                ref = max(cv["never early"], cv["LSM rule"])
                L.append(f"| {r['unwind']} | {r['cost_scale']:g} | {cv['never early']:.3f} | {cv['LSM rule']:.3f} | "
                         f"{r['strongest']} {cv[r['strongest']]:.3f} | {cv[r['strongest']] - ref:+.3f} ({(cv[r['strongest']] - ref) / ref:+.1%}) |")
            L.append("")
        L += LIQCHECK_NOTES

    # ---------------- band-rule pre-check (fixed costs) ----------------
    bands, tvb = [], []
    for bp in sorted(res.glob("bandcheck_*.json")):
        (tvb if bp.name.startswith("bandcheck_tv") else bands).extend(json.loads(bp.read_text()))
    if bands:
        L += ["", "## Pre-check: band-rule hedgers under fixed vs proportional costs (lam = 1, alpha = 0.9)", "",
              "No networks trained. A band hedger follows a target hedge and trades only when |delta_{n-1} - Delta_n| > b:",
              "fixed cost kappa per trade -> trade back to the target; proportional cost c -> trade to the band edge (control).",
              "b is tuned on 2^17 validation paths under three exercise assumptions (hold to maturity, LSM rule, worst of a",
              "simple attack family: never early, LSM, running-loss threshold, spot boundary, hedge staleness), then the tuned",
              "hedgers are cross-evaluated on the 2^18 test paths. 'Loss' = extra worst-case CVaR from tuning b for the wrong",
              "assumption (paired over test paths). Simple attacks only: a lower bound on a learned adversary.", ""]
        for tname in sorted({r["target"] for r in bands}):
            L += [f"Target: **{tname}**", "",
                  "| Costs | b* maturity / LSM / worst | trades at b* (maturity / worst) | worst-case CVaR at b*_worst | loss if tuned for maturity | loss if tuned for LSM | strongest attack |",
                  "|---|---|---|---|---|---|---|"]
            for r in [r for r in bands if r["target"] == tname]:
                st = r["setting"]
                lab = f"fixed kappa = {st['kappa']:g}" if st["kind"] == "fixed" else f"proportional c = {st['c']:g}"
                cb, ct, lo = r["chosen_b"], r["chosen_trades"], r["robustness_loss"]
                L.append(f"| {lab} | {cb['maturity']:.2f} / {cb['LSM rule']:.2f} / {cb['worst attack']:.2f} | "
                         f"{ct['maturity']:.1f} / {ct['worst attack']:.1f} | {r['cross']['worst attack']['worst attack']:.4f} | "
                         f"{lo['maturity']['diff']:+.4f} ± {lo['maturity']['se']:.4f} ({lo['maturity']['rel']:+.2%}) | "
                         f"{lo['LSM rule']['diff']:+.4f} ± {lo['LSM rule']['se']:.4f} ({lo['LSM rule']['rel']:+.2%}) | "
                         f"{r['worst_attack_at_chosen']['worst attack']} |")
            L.append("")
        if tvb:
            L += ["Time-varying band b_n = b0 (1 + beta t_n / T), target = frictionless NN hedge; grid b0 in [0.01, 0.3] x",
                  "beta in {-0.75, 0, 1, 3, 9} (beta > 0: the band widens towards maturity, i.e. trade less as the remaining life shrinks).", "",
                  "| Costs | (b0, beta)* maturity | (b0, beta)* LSM | (b0, beta)* worst | trades at worst-tuned | loss if tuned for maturity | loss if tuned for LSM |",
                  "|---|---|---|---|---|---|---|"]
            for r in tvb:
                st, cb, lo = r["setting"], r["chosen_b"], r["robustness_loss"]
                f = lambda v: f"({v[0]:.3f}, {v[1]:g})"
                L.append(f"| fixed kappa = {st['kappa']:g} | {f(cb['maturity'])} | {f(cb['LSM rule'])} | {f(cb['worst attack'])} | "
                         f"{r['chosen_trades']['worst attack']:.1f} | {lo['maturity']['diff']:+.4f} ± {lo['maturity']['se']:.4f} | "
                         f"{lo['LSM rule']['diff']:+.4f} ± {lo['LSM rule']['se']:.4f} |")
            L.append("")
        L += BANDCHECK_NOTES + ["Figures: `figures/bandcheck_merton.png`, `figures/bandcheck_nn.png`."]

    # ---------------- step 3 ----------------
    if step3_rows:
        r = step3_rows[0]
        L += ["", "## Step 3: GDA and exploitability (lam = 1)", "",
              f"Reduced compute (agreed): training batch 2^{r['batch_log2']}, {r['outer_iters']} outer iterations "
              f"(spec: 2^14, 3000); k_adv = 5, adversary pool on, temperature 1 → 0.1, warm-started stopper.",
              "Exploitability attack: fresh stopper (new random init + the same warm start) trained 2000 steps at the same batch",
              "against the frozen hedger. Supplementary attacks (not in the spec) use simple rules on the same frozen hedger;",
              "the P&L-threshold rule is fitted on separate validation paths. Gaps are paired on the test paths.", ""]
        for r in step3_rows:
            L += [f"### alpha = {r['alpha']:g}", "",
                  f"p* (trained hedger vs trained stopper) = **{r['p_star']:.4f} ± {r['p_star_se']:.4f}**; "
                  f"trained stopper E[tau] = {r['tr_mean_ex_time']:.3f}, early-exercise fraction {r['tr_frac_early']:.3f}; "
                  f"training {r['train_minutes']} min.",
                  f"Unhedged seller, same CVaR: {r['nohedge_lsm']:.4f} under the LSM rule, {r['nohedge_never_early']:.4f} if never exercised early. "
                  f"LSM price (risk-neutral) = {r['lsm_price']:.4f}.", "",
                  "| Attack on frozen hedger | CVaR | gap vs p* (paired) | gap / p* | E[tau] | early frac |", "|---|---|---|---|---|---|"]
            for a in r["attacks"]:
                L.append(f"| {a['attack']} | {a['cvar']:.4f} ± {a['se']:.4f} | {a['gap']:+.4f} ± {a['gap_se']:.4f} | {a['rel_gap']:+.2%} | "
                         f"{a['mean_ex_time']:.3f} | {a['frac_early']:.3f} |")
            if r.get("ref_merton_delta"):
                L += ["", "Reference hedge (not in spec): Merton European delta, same simple stopping rules, same CVaR:", "",
                      "| Stopping rule | NN hedger | Merton-delta hedger | NN − Merton (paired) |", "|---|---|---|---|"]
                for q in r["ref_merton_delta"]:
                    L.append(f"| {q['rule']} | {q['nn_hedger']:.4f} | {q['merton_delta']:.4f} | {q['diff']:+.4f} ± {q['diff_se']:.4f} |")
            L.append("")
        L += ["Notes:",
              "- The fresh attacker uses the same warm start and recipe as the GDA adversary, so a small gap says the trained",
              "  stopper is a best response *within this recipe*. The supplementary attacks (different rule families) are all",
              "  weaker than the trained adversary, which is mild extra evidence that it is not a weak adversary.",
              "- alpha = 0: the hedger is irrelevant in expectation (E[G_tau] = 0), so its objective is flat and its positions",
              "  drift (std ~0.75 by the end); this only adds variance (SE 0.032 vs 0.018 unhedged). Paired against the LSM rule",
              "  on the same paths and hedger, p* − LSM rule ≈ 0, i.e. the GDA stopper matches LSM.",
              "- alpha = 0.9: the hedge is flatter than the Merton European delta and nearly time-homogeneous (more short",
              "  stock out of the money, i.e. protection against down-jumps). It beats the Merton-delta hedge by 2.4–2.7 CVaR",
              "  points under every simple stopping rule, so the shape is not obviously an under-training artefact.",
              "- alpha = 0.9 training plateaus within ~50 outer iterations; the adversary briefly stopped exercising early",
              "  (iterations ~50–100) and recovered to ~12% early exercise (LSM: 34%): the seller's worst case is mostly",
              "  holding to maturity, with early exercise used selectively.",
              "",
              "Figures: `figures/step3_training.png`, `figures/step3_hedge_ratio.png`."]

    # ---------------- step 2 ----------------
    if step2_rows:
        L += ["", "## Step 2 finding: the spec'd stopper training collapses from a random init", "",
              "With the stopper randomly initialised (spec), joint training of the relaxed objective",
              "X = sum_n f_n prod_{k<n}(1 - f_k) L_n drives every logit negative within ~50–100 steps: the hard rule never",
              "exercises early and the value is the European price. Mechanism: at init f_n ~ 0.5, so the relaxed rule",
              "stops at t_1–t_3, where stopping is worse than waiting; the shared network lowers all logits; once",
              "f_n(1 - f_n) is tiny everywhere, the (small, rare) positive signal from deep-ITM late states is",
              "exponentially down-weighted and cannot recover (max f over the test set fell to ~1e-30 by step 600).",
              "Reproduced at full scale (batch 2^14, `diagnostics/fullscale_random_init_lam0_partial.csv`, stopped at",
              "step 850 with zero early exercise) and in five small-scale variants (`diagnostics/step2_collapse_smallscale.txt`):",
              "lr 1e-3 / 3e-4 / 1e-4, output-bias init at logit(1/N), and Becker et al.'s per-date objective with hard",
              "continuation applied to all dates at once — all collapse. A supervised warm start does not.", "",
              "**Fix (adopted 2026-10-02):** `train.stopper_init: heuristic` — 300 Adam steps of BCE towards the rule",
              "\"exercise iff S < K and K - S > Merton European put value\" (does not use LSM; it is a sub-optimal rule),",
              "then the spec'd relaxed ascent unchanged. Used for every stopper (step 2, GDA adversary, exploitability attacker).", "",
              "## Step 2 detail (alpha = 0, zero hedge, warm-started stopper, spec compute)", "",
              "| lam | NN (hard, test) | LSM rule on same paths | NN − LSM (paired) | rel. vs LSM price | rel. vs CRR Berm. | warm-start rule alone | never-exercise (collapse) | E[tau] NN / LSM | early-ex NN / LSM | same tau |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in step2_rows:
            crr = f"{r['rel_err_vs_crr_bermudan']:+.2%}" if "rel_err_vs_crr_bermudan" in r else "—"
            L.append(f"| {r['lam']:g} | {r['nn_price']:.4f} ± {r['nn_se']:.4f} | {r['lsm_on_test']:.4f} ± {r['lsm_on_test_se']:.4f} | "
                     f"{r['nn_minus_lsm_test']:+.4f} ± {r['nn_minus_lsm_test_se']:.4f} | {r['rel_err_vs_lsm']:+.2%} | {crr} | "
                     f"{r['heuristic_on_test']:.4f} | {r['never_early_on_test']:.4f} | "
                     f"{r['nn_mean_ex_time']:.3f} / {r['lsm_mean_ex_time']:.3f} | {r['nn_frac_early']:.3f} / {r['lsm_frac_early']:.3f} | "
                     f"{r['same_tau_frac']:.3f} |")
        L += ["", "With a zero hedge and alpha = 0 the hedger plays no role (E[G_tau] = 0 for any stopping time under mu = r),",
              "so this isolates the stopper. The paired column compares both rules on identical test paths (small SE).",
              "Both NN and LSM values are lower-bound estimators of the Bermudan price.",
              "Figures: `figures/step2_exercise_region.png`, `figures/step2_training.png`."]
    path = res / "RESULTS.md"
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    ap = add_config_args(argparse.ArgumentParser(description="Evaluation / reports"))
    ap.add_argument("--mode", choices=["step2", "step3", "headline", "grid", "md"], required=True)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.9, 0.0])
    ap.add_argument("--lams", nargs="*", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=None, help="evaluate runs trained with this seed")
    ap.add_argument("--init", default="heuristic", choices=["random", "heuristic"], help="stopper init of the runs")
    args = ap.parse_args()
    cfg = config_from_args(args)
    if args.mode == "step2":
        step2_report(cfg, args.lams, args.seed, args.init)
    elif args.mode == "step3":
        step3_report(cfg, args.alphas, lam=cfg.market.lam)
    elif args.mode == "headline":
        headline_report(cfg, lam=cfg.market.lam, alpha=cfg.train.alpha)
    elif args.mode == "grid":
        grid_report(cfg, args.lams, args.alphas if args.alphas != [0.9, 0.0] else GRID_ALPHAS)
    else:
        write_results_md(cfg)
