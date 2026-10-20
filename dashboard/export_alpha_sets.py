"""Add the other tail levels (alpha = 0, 0.5, 0.95, 0.99) to dashboard/dashboard_data.json.

Run after export_data.py and after jobs/run_alpha_lsm.sh has trained the LSM-trained hedges and their attackers.
For each alpha it exports the GDA grid hedge with its co-trained attacker, the LSM-trained hedge with the attacker
trained against it, and computes the payoff matrix at that alpha (rows: GDA, LSM-trained, Merton delta, no hedge;
columns: the two attackers, the LSM rule, never early) on the same 2^18-path test set.

The alpha = 0.9 set is the one export_data.py already wrote (the headline models, 1,500 iterations). Networks for
other levels are stored as hedgers["gda@0.95"], stoppers["advA@0.95"], and so on.

Usage:  python dashboard/export_alpha_sets.py      (about 2 minutes per alpha on 2 CPUs)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks import LSMRule, load_benchmarks  # noqa: E402
from config import config_from_dict, liq_of  # noqa: E402
from export_data import MertonDeltaHedger, lsm_stop_logits, mlp_blob  # noqa: E402
from losses import aggregate, cvar_estimate, hedge_rollout, stopper_logits  # noqa: E402
from models import Hedger, Stopper, ZeroHedger  # noqa: E402
from sim import get_test_set  # noqa: E402

D = Path(__file__).resolve().parent
OUT = D / "dashboard_data.json"
CK, RES = ROOT / "checkpoints", ROOT / "results"
CHUNK = 2 ** 14
LEVELS = {"0": "a0", "0.5": "a0p5", "0.95": "a0p95", "0.99": "a0p99"}
torch.set_num_threads(2)


def read_csv(p: Path):
    import csv
    with open(p) as fh:
        return [{k: float(v) if v not in ("", "nan") else None for k, v in row.items()} for row in csv.DictReader(fh)]


def names(akey: str) -> dict:
    t = LEVELS[akey]
    gda = (f"gda_lam1_{t}_s1234_heuristic_rs150_liqbase_uwcash_gridrs" if akey == "0.99"
           else f"gda_lam1_{t}_s1234_heuristic_liqbase_uwcash_grid")
    return dict(gda=gda, lsm=f"fixed-lsm_lam1_{t}_s1234_heuristic_liqbase_uwcash_grid",
                advB=f"exploit-fixed-lsm_lam1_{t}_s1234_heuristic_liqbase_uwcash_grid",
                dual=f"dual_grid_lam1_{t}_{t}_vf.json")


def main():
    t0 = time.time()
    data = json.loads(OUT.read_text())
    edges = np.asarray(data["hist_edges"])
    sets = {"0.9": dict(alpha=0.9, h_keys=data["h_keys"], s_keys=data["s_keys"], matrix=data["matrix"],
                        hists=data["hists"], logs=data["logs"], duals=data["duals"],
                        iters=dict(gda=1500, lsm=1500, mat=1500), restarts=False)}
    test = None
    for akey in LEVELS:
        nm = names(akey)
        missing = [n for k, n in nm.items() if k != "dual" and not (CK / f"{n}.pt").exists()]
        if missing:
            print(f"alpha {akey}: skipped, missing {missing}")
            continue
        alpha = float(akey)
        gda = torch.load(CK / f"{nm['gda']}.pt", weights_only=False)
        lsm = torch.load(CK / f"{nm['lsm']}.pt", weights_only=False)
        advB = torch.load(CK / f"{nm['advB']}.pt", weights_only=False)
        cfg = config_from_dict(gda["cfg"])
        mc = cfg.market
        hedgers = {}
        for k, blob in (("gda", gda), ("lsm", lsm)):
            h = Hedger(cfg); h.load_state_dict(blob["hedger"]); hedgers[k] = h.eval()
        hedgers["delta"] = MertonDeltaHedger(mc)
        hedgers["none"] = ZeroHedger()
        stoppers = {}
        for k, blob in (("advA", gda), ("advB", advB)):
            s = Stopper(cfg); s.load_state_dict(blob["stopper"]); stoppers[k] = s.eval()
        rule = LSMRule.from_dict(load_benchmarks(cfg)["lsm_rule"])
        if test is None:
            test = get_test_set(cfg, dtype=torch.float32)
        H, Sk = ["gda", "lsm", "delta", "none"], ["advA", "advB", "lsm_rule", "never"]
        X = {(h, s): [] for h in H for s in Sk}
        TAU = {(h, s): [] for h in H for s in Sk}
        with torch.no_grad():
            for st in range(0, test.B, CHUNK):
                sub = test.subset(slice(st, st + CHUNK))
                for h in H:
                    ro = hedge_rollout(sub, hedgers[h], mc.K, mc.T, cfg.costs.c, cfg.costs.unwind, liq_of(cfg))
                    for s in Sk:
                        if s in stoppers:
                            lg = stopper_logits(sub, ro, stoppers[s], mc.K, mc.T)
                        elif s == "lsm_rule":
                            lg = lsm_stop_logits(rule, sub.S)
                        else:
                            lg = -torch.ones_like(ro.L)
                        x, idx = aggregate(ro, lg, 1.0, hard=True)
                        X[(h, s)].append(x.double()); TAU[(h, s)].append(idx)
        matrix, hists = {}, {}
        for (h, s), xs in X.items():
            x = torch.cat(xs).numpy(); idx = torch.cat(TAU[(h, s)]).numpy()
            est, se = cvar_estimate(x, alpha)
            matrix[f"{h}|{s}"] = dict(cvar=est, se=se, mean=float(x.mean()), var=float(np.quantile(x, alpha)),
                                      frac_early=float((idx < mc.N - 1).mean()),
                                      mean_ex_day=float(((idx + 1) * mc.dt * 252).mean()))
            cnt, _ = np.histogram(np.clip(x, edges[0], edges[-1] - 1e-9), edges)
            hists[f"{h}|{s}"] = (cnt / cnt.sum()).round(6).tolist()
        duals = {}
        p = RES / nm["dual"]
        if p.exists():
            d = json.loads(p.read_text())
            duals["gda"] = {k: d[k] for k in d if isinstance(d[k], (int, float))}
        logs = {"gda": read_csv(RES / "logs" / f"{nm['gda']}.csv"), "lsm": read_csv(RES / "logs" / f"{nm['lsm']}.csv")}
        sets[akey] = dict(alpha=alpha, h_keys=H, s_keys=Sk, matrix=matrix, hists=hists, logs=logs, duals=duals,
                          iters=dict(gda=int(gda["outer_iters"]), lsm=int(lsm["outer_iters"])),
                          restarts=akey == "0.99")
        data["hedgers"][f"gda@{akey}"] = mlp_blob(gda["hedger"])
        data["hedgers"][f"lsm@{akey}"] = mlp_blob(lsm["hedger"])
        data["stoppers"][f"advA@{akey}"] = mlp_blob(gda["stopper"])
        data["stoppers"][f"advB@{akey}"] = mlp_blob(advB["stopper"])
        print(f"alpha {akey} ({time.time() - t0:.0f}s)")
        for h in H:
            print("  ", h.ljust(6), " ".join(f"{matrix[f'{h}|{s}']['cvar']:7.3f}" for s in Sk))
    data["sets"] = sets
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB), levels {sorted(sets, key=float)}")


if __name__ == "__main__":
    main()
