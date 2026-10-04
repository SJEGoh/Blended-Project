"""Check the exported networks (frontend/models.js) against the published headline results.

numpy re-implementation of sim.simulate, losses.hedge_rollout / stopper_logits and the hard stopping
rule, driven only by models.js. Simulates fresh paths, attacks each hedger with its fresh adversary
and with the LSM rule, and compares CVaR_0.9 with results/headline*.json (agreement within a few
standard errors means the weights, network inputs and cost model were ported correctly).

  python frontend/verify_demo.py                 # 2^16 paths per scenario
  python frontend/verify_demo.py --fixture out.json   # also dump one path + outputs for the JS check
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HEADLINE = {"liqbase_uwcash": "headline_liqbase_uwcash.json", "c0.005_uwcash": "headline_c0.005_uwcash.json",
            "c0.005": "headline_c0.005.json", "frictionless": "headline.json"}


def load_models():
    txt = (HERE / "models.js").read_text()
    return json.loads(txt[txt.index("=") + 1:].strip().rstrip(";"))


def mlp(layers, x):
    for i, l in enumerate(layers):
        x = x @ np.asarray(l["W"], np.float64).T + np.asarray(l["b"], np.float64)
        if i < len(layers) - 1:
            x = x / (1.0 + np.exp(-x))           # SiLU
    return x[..., 0]


def poisson_inv(U, rate):
    kmax = int(math.ceil(rate + 12 * math.sqrt(rate) + 12))
    pmf = math.exp(-rate); cdf = pmf; n = (U > cdf).astype(float)
    for k in range(1, kmax + 1):
        pmf *= rate / k; cdf += pmf; n += U > cdf
    return n


def simulate(mk, liq, B, rng):
    N, T = mk["N"], mk["T"]; dt = T / N
    kappa = math.exp(mk["m"] + 0.5 * mk["delta"] ** 2) - 1
    Z, U, Zj = rng.standard_normal((B, N)), rng.random((B, N)), rng.standard_normal((B, N))
    Nj = poisson_inv(U, mk["lam"] * dt)
    incr = (mk["r"] - mk["lam"] * kappa - 0.5 * mk["sigma"] ** 2) * dt + mk["sigma"] * math.sqrt(dt) * Z \
        + Nj * mk["m"] + mk["delta"] * np.sqrt(Nj) * Zj
    logS = math.log(mk["S0"]) + np.concatenate([np.zeros((B, 1)), incr.cumsum(1)], 1)
    ex = liquidity_state(mk, liq, logS, rng) if liq else None
    return logS, ex


def liquidity_state(mk, lq, logS, rng):
    B, N1 = logS.shape; N = N1 - 1; dt = mk["T"] / mk["N"]; sd = mk["sigma"] * math.sqrt(dt)
    r = np.diff(logS, axis=1)
    eps, eps_v = rng.standard_normal((B, N)), rng.standard_normal((B, N))
    x = np.zeros((B, N1)); s2 = np.full((B, N1), mk["sigma"] ** 2)
    drive = np.maximum(-r - lq["h_mult"] * sd, 0.0); xmax = math.log(lq["ell_max"])
    for n in range(N):
        x[:, n + 1] = np.minimum(x[:, n] - lq["kappa"] * x[:, n] * dt + lq["eta"] * math.sqrt(dt) * eps[:, n]
                                 + lq["beta_down"] * drive[:, n], xmax)
        s2[:, n + 1] = (1 - lq["ewma_w"]) * s2[:, n] + lq["ewma_w"] * r[:, n] ** 2 / dt
    logv = np.zeros((B, N1))
    zabs = np.minimum(np.abs(r / sd), lq["zcap"])
    logv[:, 1:] = lq["gamma_v"] * (zabs - math.sqrt(2 / math.pi)) + lq["zeta"] * eps_v - 0.5 * lq["zeta"] ** 2
    return dict(ell=np.exp(x), vol=np.exp(logv), sig=np.sqrt(s2))


def liq_cost(q, Sd, ell, vol, sig, lq, sigma):
    ref = lq.get("sigma_ref") or sigma
    sr = np.minimum(sig / ref, lq["sig_ratio_max"])
    hs = lq["cost_scale"] * lq["s0"] * ell * sr
    imp = lq["cost_scale"] * lq["Y"] * ell * (sr * ref / math.sqrt(252)) * np.sqrt(lq["psi"] / vol)
    return (hs * q + imp * np.maximum(q, 0) ** 1.5) * Sd


def liq_feats(ex, sl, lq, sigma):
    ref = lq.get("sigma_ref") or sigma
    return [np.log(ex["ell"][sl]), np.log(ex["vol"][sl]), np.log(ex["sig"][sl] / ref)]


def rollout(sc, hedger, logS, ex):
    """Seller loss L (B, N) at dates 1..N and deltas D (B, N): D[:, n] decided at t_n, held to t_{n+1}."""
    mk, liq = sc["market"], sc["liquidity"]
    c, unwind = sc["costs"].get("c", 0.0), sc["costs"].get("unwind", "none")
    N, T, K = mk["N"], mk["T"], mk["K"]
    B = logS.shape[0]
    t = np.linspace(0, T, N + 1); S = np.exp(logS); Sd = S * np.exp(-mk["r"] * t); Zd = np.maximum(K - S, 0) * np.exp(-mk["r"] * t)
    xs = mk["sigma"] * math.sqrt(T)
    dp = np.zeros(B); G = np.zeros(B); C = np.zeros(B); D, Gs, Cs = [], [], []
    for n in range(N):
        feats = [np.full(B, t[n] / T), np.log(S[:, n] / K) / xs, dp]
        if liq:
            feats += liq_feats(ex, (slice(None), n), liq, mk["sigma"])
        d = mlp(hedger, np.stack(feats, -1))
        C = C + c * np.abs(d - dp) * Sd[:, n]
        if liq:
            C = C + liq_cost(np.abs(d - dp), Sd[:, n], ex["ell"][:, n], ex["vol"][:, n], ex["sig"][:, n], liq, mk["sigma"])
        G = G + d * (Sd[:, n + 1] - Sd[:, n])
        D.append(d); Gs.append(G); Cs.append(C); dp = d
    D, G, C = np.stack(D, 1), np.stack(Gs, 1), np.stack(Cs, 1)
    U = np.zeros_like(D)
    if unwind == "cash" and (c > 0 or liq):
        q = np.abs(D)
        U = c * q * Sd[:, 1:]
        if liq:
            U = U + liq_cost(q, Sd[:, 1:], ex["ell"][:, 1:], ex["vol"][:, 1:], ex["sig"][:, 1:], liq, mk["sigma"])
    elif unwind not in ("none", "cash"):
        raise ValueError(unwind)
    return dict(L=Zd[:, 1:] - G + C + U, D=D, G=G, C=C, U=U, Zd=Zd[:, 1:])


def stopper_logits(sc, stopper, logS, ex, ro):
    mk, liq = sc["market"], sc["liquidity"]
    N, T, K = mk["N"], mk["T"], mk["K"]
    B = logS.shape[0]
    t = np.linspace(0, T, N + 1)[1:]; S = np.exp(logS[:, 1:])
    xs, ps = mk["sigma"] * math.sqrt(T), K * mk["sigma"] * math.sqrt(T)
    feats = [np.broadcast_to(t / T, (B, N)), np.log(S / K) / xs, ro["D"], ro["L"] / ps, (S < K).astype(float)]
    if liq:
        feats += liq_feats(ex, (slice(None), slice(1, None)), liq, mk["sigma"])
    return mlp(stopper, np.stack(feats, -1))


def first_true(mask):
    mask = mask.copy(); mask[:, -1] = True
    return mask.argmax(1)


def lsm_stop(rule, logS):
    """0-based index over dates 1..N of the LSM rule's exercise date."""
    K, r, t, coefs = rule["K"], rule["r"], np.asarray(rule["t"]), np.asarray(rule["coefs"])
    S = np.exp(logS); N = S.shape[1] - 1
    x = S / K - 1
    cont = sum(coefs[:, p][None, :] * x ** p for p in range(rule["deg"] + 1))
    pay = np.exp(-r * t)[None, :] * np.maximum(K - S, 0)
    ex = (pay > 0) & (pay > cont)
    ex[:, 0] = False
    return first_true(ex[:, 1:])


def cvar(X, alpha=0.9):
    k = max(1, math.ceil((1 - alpha) * X.size - 1e-9))
    part = np.partition(X, X.size - k); var = part[X.size - k]
    psi = var + np.maximum(X - var, 0) / (1 - alpha)
    return part[X.size - k:].mean(), psi.std(ddof=1) / math.sqrt(X.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log2", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fixture", type=str, default=None)
    a = ap.parse_args()
    M = load_models()
    fixture = []
    worst_z = 0.0
    for sc in M["scenarios"]:
        rng = np.random.default_rng(a.seed)
        logS, ex = simulate(sc["market"], sc["liquidity"], 2 ** a.log2, rng)
        pub = {h["source"]: h for h in json.loads((ROOT / "results" / HEADLINE[sc["id"]]).read_text())["hedgers"]}
        tau_lsm = lsm_stop(M["lsm_rule"], logS)
        print(f"\n{sc['label']}  ({2 ** a.log2} fresh paths; published = 2^18 test paths)")
        for key, src, adv in (("hedger_gda", "gda", "adv_vs_gda"), ("hedger_lsm", "fixed-lsm", "adv_vs_lsm")):
            ro = rollout(sc, sc["nets"][key], logS, ex)
            tau = first_true(stopper_logits(sc, sc["nets"][adv], logS, ex, ro) > 0)
            for name, idx, pub_key in (("fresh adversary", tau, "fresh adversary"), ("LSM rule", tau_lsm, "LSM rule")):
                X = ro["L"][np.arange(len(idx)), idx]
                v, se = cvar(X)
                p = pub[src]["evals"][pub_key]
                z = (v - p["cvar"]) / math.hypot(se, p["se"])
                worst_z = max(worst_z, abs(z))
                print(f"  {src:10s} vs {name:16s} ours {v:8.4f} ± {se:.4f}   published {p['cvar']:8.4f} ± {p['se']:.4f}"
                      f"   z = {z:+.2f}   early {np.mean(idx < sc['market']['N'] - 1):.3f} (pub {p['frac_early']:.3f})")
        if a.fixture:
            i = 3
            one = {k: v[i:i + 1] for k, v in ex.items()} if ex else None
            out = dict(id=sc["id"], logS=logS[i].tolist(), extras={k: v[0].tolist() for k, v in one.items()} if one else None)
            for key, adv in (("hedger_gda", "adv_vs_gda"), ("hedger_lsm", "adv_vs_lsm")):
                ro = rollout(sc, sc["nets"][key], logS[i:i + 1], one)
                out[key] = dict(L=ro["L"][0].tolist(), D=ro["D"][0].tolist(),
                                logits=stopper_logits(sc, sc["nets"][adv], logS[i:i + 1], one, ro)[0].tolist())
            out["tau_lsm"] = int(lsm_stop(M["lsm_rule"], logS[i:i + 1])[0])
            fixture.append(out)
    print(f"\nlargest |z| = {worst_z:.2f}")
    if a.fixture:
        Path(a.fixture).write_text(json.dumps(fixture))
        print(f"wrote fixture {a.fixture}")


if __name__ == "__main__":
    main()
