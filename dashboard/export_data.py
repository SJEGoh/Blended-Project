"""Export everything the Seller vs Holder dashboard needs into dashboard/dashboard_data.json.

  * model parameters (market, liquidity, costs) of the liquidity headline (lam = 1, alpha = 0.9, cash unwind);
  * network weights (float32, base64): GDA / LSM-trained / maturity-trained hedgers, three trained adversaries,
    and the pricing continuation networks (cash, physical psi = 0.05, physical psi = 1);
  * the LSM exercise rule (polynomial coefficients);
  * the game payoff matrix: CVaR_0.9 of the seller's hedged loss for every (hedge, holder strategy) pair on the
    fixed 2^18-path test set, with reference histograms of the loss;
  * training logs (GDA and the LSM-trained hedge), certified dual brackets, pricing brackets;
  * a few reference paths with their driving shocks and Python outputs, for the JavaScript parity test.

Usage:  python dashboard/export_data.py            (about 5-10 minutes on 2 CPUs)
"""
from __future__ import annotations

import base64
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks import LSMRule, load_benchmarks  # noqa: E402
from config import config_from_dict, liq_of  # noqa: E402
from losses import aggregate, cvar_estimate, hedge_rollout, stopper_logits  # noqa: E402
from models import Hedger, Stopper, ZeroHedger  # noqa: E402
from pricing import exercise_value, features, outer  # noqa: E402
from sim import get_test_set, merton_put_delta, simulate_cfg  # noqa: E402

OUT = Path(__file__).resolve().parent / "dashboard_data.json"
CK = ROOT / "checkpoints"
RES = ROOT / "results"
TAG = "lam1_a0p9_s1234_heuristic_liqbase_uwcash_reduced"
CHUNK = 2 ** 14
ALPHA = 0.9

torch.set_num_threads(2)


def b64(arrs) -> str:
    flat = np.concatenate([np.asarray(a, np.float32).ravel() for a in arrs])
    return base64.b64encode(flat.astype("<f4").tobytes()).decode()


def mlp_blob(sd: dict, prefix: str = "net.") -> dict:
    """Linear layers of an nn.Sequential MLP (SiLU between) -> {shapes, data(base64)}."""
    ws, shapes = [], []
    i = 0
    while f"{prefix}{i}.weight" in sd:
        W, b = sd[f"{prefix}{i}.weight"].float().numpy(), sd[f"{prefix}{i}.bias"].float().numpy()
        ws += [W, b]
        shapes.append(list(W.shape))
        i += 2
    return {"shapes": shapes, "data": b64(ws)}


class MertonDeltaHedger(torch.nn.Module):
    """Reference hedge: delta_n = Merton European put delta at (t_n, S_n)."""

    is_zero = False

    def __init__(self, mc):
        super().__init__()
        self.mc = mc

    def forward(self, t_frac, logm, delta_prev, step_state=None, pnl=None):
        mc = self.mc
        tau = mc.T * (1.0 - float(t_frac))
        S = (mc.K * torch.exp(logm.double())).numpy()
        d = merton_put_delta(S, mc.K, max(tau, 1e-12), mc.r, mc.sigma, mc.lam, mc.m, mc.delta)
        return torch.as_tensor(d, dtype=logm.dtype)


def lsm_stop_logits(rule: LSMRule, S: torch.Tensor) -> torch.Tensor:
    """+1 where the LSM rule exercises, -1 elsewhere (dates 1..N), as stopper-style logits."""
    dec = rule.stop_matrix(S.double().numpy())
    return torch.as_tensor(np.where(dec, 1.0, -1.0), dtype=torch.float32)


def main():
    t0 = time.time()
    gda = torch.load(CK / f"gda_{TAG}.pt", weights_only=False)
    cfg = config_from_dict(gda["cfg"])
    mc, lq = cfg.market, cfg.liquidity
    blobs = {
        "gda": gda,
        "lsm": torch.load(CK / f"fixed-lsm_{TAG}.pt", weights_only=False),
        "mat": torch.load(CK / f"fixed-maturity_{TAG}.pt", weights_only=False),
        "advA": gda,
        "advB": torch.load(CK / f"exploit-fixed-lsm_{TAG}.pt", weights_only=False),
        "advC": torch.load(CK / f"exploit-fixed-maturity_{TAG}.pt", weights_only=False),
    }
    hedgers = {}
    for k in ("gda", "lsm", "mat"):
        h = Hedger(cfg)
        h.load_state_dict(blobs[k]["hedger"])
        hedgers[k] = h.eval()
    hedgers["delta"] = MertonDeltaHedger(mc)
    hedgers["none"] = ZeroHedger()
    stoppers = {}
    for k in ("advA", "advB", "advC"):
        s = Stopper(cfg)
        s.load_state_dict(blobs[k]["stopper"])
        stoppers[k] = s.eval()
    rule = LSMRule.from_dict(load_benchmarks(cfg)["lsm_rule"])

    # ---------------- payoff matrix on the fixed test set ----------------
    test = get_test_set(cfg, dtype=torch.float32)
    print(f"test set {test.B} paths ({time.time() - t0:.0f}s)", flush=True)
    H_KEYS = ["gda", "lsm", "mat", "delta", "none"]
    S_KEYS = ["advA", "advB", "advC", "lsm_rule", "never"]
    X = {(h, s): [] for h in H_KEYS for s in S_KEYS}
    TAU = {(h, s): [] for h in H_KEYS for s in S_KEYS}
    with torch.no_grad():
        for st in range(0, test.B, CHUNK):
            sub = test.subset(slice(st, st + CHUNK))
            for h in H_KEYS:
                ro = hedge_rollout(sub, hedgers[h], mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
                for s in S_KEYS:
                    if s in stoppers:
                        lg = stopper_logits(sub, ro, stoppers[s], mc.K, mc.T)
                    elif s == "lsm_rule":
                        lg = lsm_stop_logits(rule, sub.S)
                    else:
                        lg = -torch.ones_like(ro.L)
                    x, idx = aggregate(ro, lg, 1.0, hard=True)
                    X[(h, s)].append(x.double())
                    TAU[(h, s)].append(idx)
            print(f"  chunk {st // CHUNK + 1}/{test.B // CHUNK} ({time.time() - t0:.0f}s)", flush=True)
    matrix, hists = {}, {}
    edges = np.linspace(-10.0, 40.0, 101)          # loss histogram bins (discounted $ per option)
    for (h, s), xs in X.items():
        x = torch.cat(xs).numpy()
        idx = torch.cat(TAU[(h, s)]).numpy()
        est, se = cvar_estimate(x, ALPHA)
        var = float(np.quantile(x, ALPHA))
        matrix[f"{h}|{s}"] = dict(cvar=est, se=se, mean=float(x.mean()), var=var,
                                  frac_early=float((idx < mc.N - 1).mean()),
                                  mean_ex_day=float(((idx + 1) * mc.dt * 252).mean()))
        cnt, _ = np.histogram(np.clip(x, edges[0], edges[-1] - 1e-9), edges)
        hists[f"{h}|{s}"] = (cnt / cnt.sum()).round(6).tolist()
    print("matrix done", flush=True)
    for h in H_KEYS:
        print(h, " ".join(f"{matrix[f'{h}|{s}']['cvar']:7.3f}" for s in S_KEYS))

    # ---------------- pricing networks and results ----------------
    import pricing_report
    report_rows = {r["name"]: r for r in pricing_report.load()}
    pricing = {}
    for name in ("cash_lam1", "phys_lam1", "phys_psi1"):
        pb = torch.load(CK / f"pricing_{name}.pt", weights_only=False)
        nets = []
        for k in range(mc.N - 1):
            sd = {kk.split(f"nets.{k}.", 1)[1]: v for kk, v in pb["nets"].items() if kk.startswith(f"nets.{k}.")}
            nets.append(mlp_blob(sd, prefix=""))
        res = report_rows[name]          # lower = max(fitted rule, European); precise European (pricing_report)
        pricing[name] = dict(mode=pb["mode"], psi=pb["cfg"]["liquidity"]["psi"],
                             n_feat=int(pb["nets"]["nets.0.0.weight"].shape[1]), nets=nets,
                             lower=res["lower"], upper=res["upper"], european=res["european_precise"],
                             frac_early=res["frac_early"], mean_ex_time=res["mean_ex_time"])

    # ---------------- logs, duals ----------------
    def read_csv(p):
        with open(p) as fh:
            return [{k: float(v) if v not in ("", "nan") else None for k, v in row.items()} for row in csv.DictReader(fh)]

    logs = {"gda": read_csv(RES / "logs" / f"gda_{TAG}.csv"), "lsm": read_csv(RES / "logs" / f"fixed-lsm_{TAG}.csv")}
    duals = {}
    for key, fn in (("gda", "dual_v3_gda_liqbase_a0p9_vf.json"), ("lsm", "dual_v4_lsm_liqbase_a0p9_vf.json")):
        p = RES / fn
        if p.exists():
            d = json.loads(p.read_text())
            duals[key] = {k: d[k] for k in d if isinstance(d[k], (int, float))}

    # ---------------- parity samples for the JS port ----------------
    paths, sh = simulate_cfg(cfg, 4, 99, dtype=torch.float64, return_shocks=True)
    p32 = paths.to(dtype=torch.float32)
    with torch.no_grad():
        ro = hedge_rollout(p32, hedgers["gda"], mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
        lgA = stopper_logits(p32, ro, stoppers["advA"], mc.K, mc.T)
        ro_d = hedge_rollout(p32, hedgers["delta"], mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
    from pricing import ContinuationNets
    par_price = {}
    for name in ("cash_lam1", "phys_lam1"):
        pb = torch.load(CK / f"pricing_{name}.pt", weights_only=False)
        pcfg = config_from_dict(pb["cfg"])
        nets = ContinuationNets(pcfg, pricing[name]["n_feat"], pb["hidden"])
        nets.load_state_dict(pb["nets"])
        pp = paths if pcfg.liquidity.enabled else paths.__class__(paths.t, paths.S, paths.S_disc, paths.Z_disc, {})
        Y, F, _ = outer(pcfg, pb["mode"], pp)
        with torch.no_grad():
            C = torch.stack([nets(k, F[:, k - 1]) for k in range(1, mc.N)], 1)
        par_price[name] = dict(Y=Y.tolist(), C=C.tolist())
    parity = dict(
        shocks={k: v.tolist() for k, v in sh.items()},
        S=paths.S.tolist(), ell=paths.extras["ell"].tolist(), vol=paths.extras["vol"].tolist(),
        sig=paths.extras["sig"].tolist(), delta=ro.delta_prev.tolist(), L=ro.L.tolist(), logitA=lgA.tolist(),
        L_delta=ro_d.L.tolist(), delta_merton=ro_d.delta_prev.tolist(), pricing=par_price,
        lsm_stop=rule.stop_matrix(paths.S.numpy()).astype(int).tolist(),
    )

    out = dict(
        generated=time.strftime("%Y-%m-%d"),
        market=dict(S0=mc.S0, K=mc.K, T=mc.T, N=mc.N, r=mc.r, sigma=mc.sigma, lam=mc.lam, m=mc.m, delta=mc.delta),
        liquidity={k: getattr(lq, k) for k in ("cost_scale", "s0", "Y", "psi", "kappa", "eta", "beta_down", "h_mult",
                                                "gamma_v", "zeta", "ewma_w", "zcap", "ell_max", "sig_ratio_max")}
        | {"sigma_ref": lq.sigma_ref or mc.sigma},
        costs=dict(c=cfg.costs.c, unwind=cfg.costs.unwind), alpha=ALPHA,
        hedgers={k: mlp_blob(blobs[k]["hedger"]) for k in ("gda", "lsm", "mat")},
        stoppers={k: mlp_blob(blobs[k]["stopper"]) for k in ("advA", "advB", "advC")},
        lsm_rule=dict(coefs=np.asarray(rule.coefs).tolist(), deg=rule.deg),
        matrix=matrix, hist_edges=edges.tolist(), hists=hists, h_keys=H_KEYS, s_keys=S_KEYS,
        logs=logs, duals=duals, pricing=pricing,
        test_paths=int(test.B), parity=parity,
    )
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
