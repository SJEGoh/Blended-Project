"""Summary, figures and RESULTS.md section for the Part 1 pricing study (results/pricing/*.json)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import ROOT

RES = ROOT / "results"
OUT = RES / "pricing"
FIG = RES / "figures"
BASE = dict(S0=100.0, lam=1.0, m=-0.10, delta=0.15, sigma=0.20)
BASE_LIQ = dict(psi=0.05, cost_scale=1.0, ell_max=10.0)
GREY, BLUE = "#8c8c8c", "#1f4e9c"
PRICING_NOTES: list[str] = [      # hand-written reading of the 38-case run
    "Reading:",
    "- **Validation.** Cash, lam = 0: [4.998, 5.043] contains the exact CRR Bermudan value 5.006. Physical with cost",
    "  scale 0 reproduces the cash price (discount range [-0.057, 0.032] contains 0).",
    "- **Precision.** Cash brackets are 0.2-0.9% wide. Physical ones are 1.2-2.9%: the liquidity state adds three",
    "  dimensions to the value function, and the fit is the limiting factor (the bounds' Monte Carlo SEs are about",
    "  0.001).",
    "- **Jumps raise the price, as expected.** At lam = 1 the cash price is 6.56-6.59, against 5.00-5.04 without jumps.",
    "  The early-exercise premium grows with jump intensity and with downside jump size: [0.15, 0.17] at lam = 4,",
    "  [0.20, 0.22] at m = -0.30. It shrinks with jump volatility: [0.035, 0.064] at delta = 0.35. Jump volatility",
    "  adds time value, which favours waiting.",
    "- **Volume risk lowers the price by about 3%** for a small holder (psi = 0.05). At lam = 1 the certified",
    "  discount is [0.16, 0.26]. In absolute terms it grows with jump intensity, from [0.07, 0.20] at lam = 0 to",
    "  [0.29, 0.43] at lam = 4: jumps cause the illiquidity spikes that make exercise costly.",
    "- **Position size dominates.** The discount is about 3% at psi = 0.05, 5.7% at psi = 0.25 and 10% at psi = 1, the",
    "  same as scaling costs by 2 and 4 (impact grows with sqrt(psi)). The milder stress calibration changes little",
    "  (2.9% vs 3.2%), because illiquidity rarely reaches the caps on the dates when holders exercise.",
    "- **Holders wait out illiquidity.** Deep in the money (S/K 0.7-0.8), the fitted rule exercises in 27% of alive",
    "  states when liquidity is normal (ell < 1.25), 5% at ell 1.25-2, and under 1% above 2. Early exercise happens",
    "  almost only in normal liquidity (mean illiquidity at early exercise about 1.2).",
    "- **When liquid, physical holders exercise earlier.** Their effective boundary at normal liquidity sits about 2-4",
    "  above the cash holder's for t < 0.4 (figure; 99% quantile of exercised spots). They exercise while liquidity is",
    "  good, because further declines tend to bring illiquidity. This boundary is read off visited states, so it is",
    "  approximate.",
    "- **The early-exercise right includes a liquidity-timing option.** With large positions the premium rises:",
    "  [0.22, 0.30] at psi = 1, against [0.11, 0.19] at psi = 0.05. The European holder must trade at maturity",
    "  whatever the liquidity; the Bermudan holder can choose a liquid date.",
    "",
]


def load() -> list[dict]:
    cases = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))
             if p.name not in ("summary.json", "european_cache.json")]
    cache_p = OUT / "european_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    for c in cases:
        key = json.dumps([c["exercise_cost"], c["market"], c.get("liquidity")], sort_keys=True)
        if key not in cache:
            cache[key] = european_precise(c)
            cache_p.write_text(json.dumps(cache, indent=1))
        e, e_se = cache[key]
        c["european_precise"], c["european_precise_se"] = e, e_se
        c["lower_rule"] = c["lower"]
        c["lower"] = max(c["lower"], e)                    # never-early is also a valid exercise rule
        c["mid"] = 0.5 * (c["lower"] + c["upper"])
        c["width"] = c["upper"] - c["lower"]
        c["early_premium"] = c["mid"] - e
    return cases


def european_precise(c: dict, log2: int = 20, seed: int = 777) -> tuple[float, float]:
    """E[e^{-rT} E_T] to high precision: the Merton closed form for cash settlement; for physical settlement the
    closed form plus a Monte Carlo estimate of E[E_T(phys) - E_T(cash)] (highly correlated, so the SE is tiny)."""
    import torch
    from config import load_config
    from pricing import exercise_value
    from sim import merton_put, simulate_cfg
    mk = c["market"]
    cf = float(merton_put(mk["S0"], mk["K"], mk["T"], mk["r"], mk["sigma"], mk["lam"], mk["m"], mk["delta"]))
    if c["exercise_cost"] == "none":
        return cf, 0.0
    lq = c["liquidity"]
    cfg = load_config(None, [f"market.{k}={mk[k]}" for k in ("S0", "K", "T", "N", "r", "sigma", "lam", "m", "delta")]
                      + ["liquidity.enabled=true", f"liquidity.name={lq['name']}"]
                      + [f"liquidity.{k}={lq[k]}" for k in ("psi", "cost_scale", "ell_max", "sig_ratio_max", "s0", "Y")])
    diffs = []
    for i in range(2 ** max(0, log2 - 16)):
        p = simulate_cfg(cfg, 2 ** 16, seed + i, dtype=torch.float64)
        d = math.exp(-mk["r"] * mk["T"])
        liq = {k: p.extras[k][:, -1] for k in ("ell", "vol", "sig")}
        Sd = p.S_disc[:, -1]
        diffs.append((exercise_value(cfg, "liquidity", Sd, mk["K"] * d, liq) - exercise_value(cfg, "none", Sd, mk["K"] * d, None)).numpy())
    x = np.concatenate(diffs)
    return cf + float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


def _is(c: dict, **kw) -> bool:
    mk = c["market"]
    for k, v in kw.items():
        if k in mk:
            if not math.isclose(mk[k], v, abs_tol=1e-12):
                return False
        elif k == "mode":
            if c["exercise_cost"] != v:
                return False
        elif c.get("liquidity") is None or not math.isclose(c["liquidity"][k], v, abs_tol=1e-12):
            return False
    return True


def sweep(cases, key, mode, extra=None):
    """Cases varying `key` around the base, all other market parameters at base (and base liquidity)."""
    fixed = {k: v for k, v in BASE.items() if k != key}
    out = [c for c in cases if _is(c, mode=mode, **fixed) and (mode == "none" or _is(c, **BASE_LIQ))]
    return sorted(out, key=lambda c: c["market"][key])


def _fig_sweeps(cases):
    panels = [("lam", "jump intensity lam (per year)"), ("m", "mean log jump m"), ("delta", "jump volatility delta")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, (key, label) in zip(axes, panels):
        for mode, col, name in (("none", GREY, "cash-settled (classical)"), ("liquidity", BLUE, "physical, liquidity cost")):
            cs = sweep(cases, key, mode)
            if not cs:
                continue
            x = [c["market"][key] for c in cs]
            mid = np.array([c["mid"] for c in cs])
            lo = np.maximum(mid - np.array([c["lower"] for c in cs]), 0.0)
            hi = np.maximum(np.array([c["upper"] for c in cs]) - mid, 0.0)
            ax.errorbar(x, mid, yerr=[lo, hi], color=col, marker="o", ms=4, capsize=3, lw=1.4, label=name)
            ax.plot(x, [c["european_precise"] for c in cs], color=col, ls=":", lw=1, label=f"{name}: European")
        ax.axvline(BASE[key], color="#cccccc", lw=0.8, zorder=0)
        ax.set_xlabel(label)
        ax.set_ylabel("Bermudan put value (bracket)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Bermudan put value under jumps: cash vs physical settlement with liquidity cost "
                 "(whiskers = certified bracket)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "pricing_sweeps.png", dpi=150)
    plt.close(fig)


def _rolling_median(y, w=5):   # kept for ad-hoc use
    y = np.array(y, float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        win = y[max(0, i - w // 2): i + w // 2 + 1]
        win = win[~np.isnan(win)]
        if win.size >= 3:
            out[i] = np.median(win)
    return out


def empirical_boundary(name: str, log2: int = 17, q: float = 0.99) -> dict | None:
    """Effective exercise boundary of the fitted rule, read off the states where it actually exercises on the
    evaluation paths: per date, the q-quantile of S among exercised states, split by illiquidity. Unlike the
    boundary of the fitted continuation function, every point is supported by visited states."""
    import torch
    from config import config_from_dict
    from pricing import ContinuationNets, outer
    from sim import simulate_cfg
    ck = ROOT / "checkpoints" / f"pricing_{name}.pt"
    if not ck.exists():
        return None
    blob = torch.load(ck, map_location="cpu")
    cfg = config_from_dict(blob["cfg"])
    mc = cfg.market
    paths = simulate_cfg(cfg, 2 ** log2, cfg.eval.test_seed + 11, "cpu", torch.float64)
    Y, F, _ = outer(cfg, blob["mode"], paths)
    nets = ContinuationNets(cfg, F.shape[-1], blob["hidden"])
    nets.load_state_dict(blob["nets"])
    groups = {"all": (0.0, 1e9)} if not cfg.liquidity.enabled else {"ell < 1.25": (0.0, 1.25), "ell 1.25-2": (1.25, 2.0)}
    out = {g: [] for g in groups}
    alive = torch.ones(paths.B, dtype=torch.bool)
    with torch.no_grad():
        for k in range(1, mc.N):
            yk = Y[:, k - 1]
            ex = ((yk > 0) & (yk >= nets(k, F[:, k - 1])) & alive).numpy()
            S = paths.S[:, k].numpy()
            ell = paths.extras["ell"][:, k].numpy() if cfg.liquidity.enabled else np.ones_like(S)
            for g, (a, b) in groups.items():
                sel = ex & (ell >= a) & (ell < b)
                out[g].append(float(np.quantile(S[sel], q)) if sel.sum() >= 50 else float("nan"))
            alive = alive & ~torch.from_numpy(ex)
    return dict(case=name, t=[mc.T * k / mc.N for k in range(1, mc.N)], q=q, boundary=out)


def _fig_boundary(cases):
    """Effective exercise boundaries (cash vs physical by illiquidity) from the states where the rule exercises."""
    cash, phys = empirical_boundary("cash_lam1"), empirical_boundary("phys_lam1")
    if not (cash and phys):
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(cash["t"], cash["boundary"]["all"], color=GREY, lw=1.8, label="cash-settled")
    for g, a in (("ell < 1.25", 1.0), ("ell 1.25-2", 0.5)):
        ax.plot(phys["t"], phys["boundary"][g], color=BLUE, alpha=a, lw=1.8, label=f"physical, illiquidity {g}")
    ax.set_xlabel("time t (years)")
    ax.set_ylabel(f"spot below which the rule exercises ({cash['q']:.0%} quantile)")
    ax.set_title("Effective exercise boundary, lam = 1 (no exercise at higher illiquidity is visible: too rare)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "pricing_boundary.png", dpi=150)
    plt.close(fig)


def exercise_propensity(name: str = "phys_lam1", log2: int = 17) -> dict | None:
    """Share of alive in-the-money states where the fitted rule exercises, by moneyness and illiquidity
    (states visited by the evaluation paths, so every cell is supported by data)."""
    import torch
    from config import config_from_dict
    from pricing import ContinuationNets, outer
    from sim import simulate_cfg
    ck = ROOT / "checkpoints" / f"pricing_{name}.pt"
    if not ck.exists():
        return None
    blob = torch.load(ck, map_location="cpu")
    cfg = config_from_dict(blob["cfg"])
    mc = cfg.market
    paths = simulate_cfg(cfg, 2 ** log2, cfg.eval.test_seed + 11, "cpu", torch.float64)
    Y, F, _ = outer(cfg, blob["mode"], paths)
    nets = ContinuationNets(cfg, F.shape[-1], blob["hidden"])
    nets.load_state_dict(blob["nets"])
    m_edges, l_edges = [0.70, 0.80, 0.90, 0.97], [0.0, 1.25, 2.0, 4.0, 1e9]
    l_names = ["ell < 1.25", "1.25-2", "2-4", "> 4"]
    cnt = np.zeros((len(m_edges) - 1, len(l_edges) - 1))
    exn = np.zeros_like(cnt)
    alive = torch.ones(paths.B, dtype=torch.bool)
    with torch.no_grad():
        for k in range(1, mc.N):
            yk = Y[:, k - 1]
            ex = (yk > 0) & (yk >= nets(k, F[:, k - 1])) & alive
            mny = (paths.S[:, k] / mc.K).numpy()
            ell = paths.extras["ell"][:, k].numpy()
            a = alive.numpy()
            for i in range(len(m_edges) - 1):
                for j in range(len(l_edges) - 1):
                    sel = a & (mny >= m_edges[i]) & (mny < m_edges[i + 1]) & (ell >= l_edges[j]) & (ell < l_edges[j + 1])
                    cnt[i, j] += sel.sum()
                    exn[i, j] += (sel & ex.numpy()).sum()
            alive = alive & ~ex
    rows = [dict(moneyness=f"S/K {m_edges[i]:.2f}-{m_edges[i + 1]:.2f}",
                 cells=[dict(ell=l_names[j], states=int(cnt[i, j]),
                             exercise_rate=(float(exn[i, j] / cnt[i, j]) if cnt[i, j] >= 200 else None))
                        for j in range(len(l_names))]) for i in range(len(m_edges) - 1)]
    return dict(case=name, rows=rows)


def summarise(cases) -> dict:
    by = {c["name"]: c for c in cases}
    rows = []
    for c in sorted(cases, key=lambda c: (c["exercise_cost"] != "none", c["name"])):
        mk, lq = c["market"], c.get("liquidity") or {}
        twin = [d for d in cases if d["exercise_cost"] == "none" and all(
            math.isclose(d["market"][k], mk[k], abs_tol=1e-12) for k in ("S0", "lam", "m", "delta", "sigma"))]
        rows.append(dict(name=c["name"], cost=c["exercise_cost"], S0=mk["S0"], lam=mk["lam"], m=mk["m"],
                         delta=mk["delta"], psi=lq.get("psi"), cost_scale=lq.get("cost_scale"), ell_max=lq.get("ell_max"),
                         lower=c["lower"], upper=c["upper"], width=c["width"], mid=c["mid"], european=c["european_precise"],
                         lower_rule=c["lower_rule"],
                         early_premium=c["early_premium"], frac_early=c["frac_early"],
                         liquidity_discount=(twin[0]["mid"] - c["mid"]) if (c["exercise_cost"] == "liquidity" and twin) else None,
                         liquidity_discount_range=([twin[0]["lower"] - c["upper"], twin[0]["upper"] - c["lower"]]
                                                   if (c["exercise_cost"] == "liquidity" and twin) else None),
                         premium_range=[c["lower"] - c["european_precise"], c["upper"] - c["european_precise"]],
                         crr=c.get("crr_bermudan"), euro_cf=c.get("european_closed_form")))
    return dict(rows=rows, n=len(rows))


def section(cases) -> list[str]:
    if not cases:
        return []
    s = summarise(cases)
    L = ["", "## Part 1 — pricing: Bermudan put value under jumps and volume risk (`pricing.py`)", "",
         "Risk-neutral value sup_tau E^Q[e^{-r tau} E_tau] with E_n = (K - S_n - kappa_n)^+. 'cash' = cash settlement",
         "(kappa = 0, the classical price); 'liquidity' = physical settlement, the holder buying the delivered share at that",
         "date's liquidity (kappa = per-share half-spread + square-root impact). Each value is a certified bracket on the",
         "same 2^17 independent paths: lower = exercise rule fitted by Longstaff-Schwartz (per-date networks), upper =",
         "Andersen-Broadie dual with an exact value-function martingale. Base: S0 = K = 100, T = 0.5, N = 50, r = 3%,",
         "sigma = 20%, lam = 1, m = -0.10, delta = 0.15; liquidity 'base', psi = 0.05.", "",
         "Lower end = max(fitted rule, European), since never exercising early is also a rule; the European value is",
         "the Merton closed form (cash) or the closed form plus a 2^20-path control-variate estimate of the liquidity",
         "cost at maturity (physical; SE ~0.0005). Ranges for the early-exercise premium (price − European) and for the",
         "liquidity discount (cash price − physical price) follow from the brackets.", "",
         "| case | settlement | S0 | lam | m | delta | psi / cost scale / ell_max | price bracket | width | European | early-exercise premium | early-exercise frac | liquidity discount vs cash | reference |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["rows"]:
        liq = "—" if r["cost"] == "none" else f"{r['psi']:g} / {r['cost_scale']:g} / {r['ell_max']:g}"
        ref = (f"CRR {r['crr']:.3f}" if r["crr"] is not None else "") + \
              (f"{'; ' if r['crr'] is not None else ''}Merton Eur. {r['euro_cf']:.3f}" if r["euro_cf"] is not None else "")
        disc = "" if r["liquidity_discount_range"] is None else (
            f"[{r['liquidity_discount_range'][0]:.3f}, {r['liquidity_discount_range'][1]:.3f}] "
            f"({r['liquidity_discount'] / (r['mid'] + r['liquidity_discount']):.1%} at mid)")
        prem = f"[{r['premium_range'][0]:.3f}, {r['premium_range'][1]:.3f}]"
        L.append(f"| {r['name']} | {'cash' if r['cost'] == 'none' else 'physical'} | {r['S0']:g} | {r['lam']:g} | {r['m']:g} | "
                 f"{r['delta']:g} | {liq} | [{r['lower']:.3f}, {r['upper']:.3f}] | {r['width']:.3f} | {r['european']:.3f} | "
                 f"{prem} | {r['frac_early']:.2f} | {disc} | {ref} |")
    sp = OUT / "summary.json"
    prop = json.loads(sp.read_text()).get("exercise_propensity") if sp.exists() else None
    if prop:
        heads = [c["ell"] for c in prop["rows"][0]["cells"]]
        L += ["", f"Exercise propensity of the fitted rule ({prop['case']}): share of alive states, at dates before maturity,",
              "where the holder exercises, by moneyness and illiquidity (cells with < 200 visited states left blank).", "",
              "| moneyness | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
        for r in prop["rows"]:
            L.append(f"| {r['moneyness']} | " + " | ".join(
                "" if c["exercise_rate"] is None else f"{c['exercise_rate']:.1%} (n = {c['states']:,})" for c in r["cells"]) + " |")
    L += [""] + PRICING_NOTES + ["Figures: `figures/pricing_sweeps.png`, `figures/pricing_boundary.png`."]
    return L


def report():
    cases = load()
    FIG.mkdir(parents=True, exist_ok=True)
    summ = summarise(cases)
    prop = exercise_propensity()
    if prop is not None:
        summ["exercise_propensity"] = prop
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    _fig_sweeps(cases)
    _fig_boundary(cases)
    from config import load_config
    from evaluate import write_results_md
    write_results_md(load_config(None, ["tag=grid"]))
    print(f"pricing report: {len(cases)} cases")


if __name__ == "__main__":
    report()
