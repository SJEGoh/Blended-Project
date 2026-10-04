"""Liquidity (volume-risk) extension: state dynamics, cost function, limits, non-anticipativity."""
import glob
import math
from pathlib import Path

import pytest
import torch
from torch import nn

from config import load_config
from losses import hedge_rollout, liquidity_cost, stopper_logits
from models import Hedger, Stopper
from sim import build_paths, liquidity_state, simulate, simulate_cfg

ON = ["liquidity.enabled=true", "market.lam=1.0"]


class ConstHedger(nn.Module):
    is_zero = False

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t_frac, logm, delta_prev, step_state=None, pnl=None):
        return torch.full_like(delta_prev, self.d)


def _permute_future(logS, m, seed=5):
    g = torch.Generator().manual_seed(seed)
    incr = logS.diff(dim=1).clone()
    for i in range(incr.shape[0]):
        tail = incr[i, m:]
        incr[i, m:] = tail[torch.randperm(len(tail), generator=g)]
    return torch.cat([logS[:, :1], logS[:, :1] + incr.cumsum(1)], 1)


def test_state_shapes_and_initial_values():
    cfg = load_config(None, ON)
    p = simulate_cfg(cfg, 256, 1, dtype=torch.float64)
    for k in ("ell", "vol", "sig"):
        assert p.extras[k].shape == (256, cfg.market.N + 1)
        assert (p.extras[k] > 0).all()
    assert torch.allclose(p.extras["ell"][:, 0], torch.ones(256, dtype=torch.float64))
    assert torch.allclose(p.extras["sig"][:, 0], torch.full((256,), cfg.market.sigma, dtype=torch.float64))
    assert (p.extras["ell"] <= cfg.liquidity.ell_max + 1e-9).all()


def test_price_paths_keep_common_random_numbers():
    on, off = load_config(None, ON), load_config(None, ["market.lam=1.0"])
    assert torch.equal(simulate_cfg(on, 512, 2, dtype=torch.float64).S, simulate_cfg(off, 512, 2, dtype=torch.float64).S)


def test_illiquidity_and_volume_respond_to_large_declines():
    cfg = load_config(None, ON)
    p = simulate_cfg(cfg, 2 ** 14, 3, dtype=torch.float64)
    r = torch.log(p.S).diff(dim=1)
    big, calm = r < -0.12, r.abs() < 0.01
    ell, vol = p.extras["ell"][:, 1:], p.extras["vol"][:, 1:]
    assert ell[big].mean() > 3.0 * ell[calm].mean()
    assert vol[big].mean() > 3.0 * vol[calm].mean()
    assert 0.8 < float(ell[calm].median()) < 1.25          # normal times ~ normal liquidity


def test_liquidity_state_is_non_anticipative():
    cfg = load_config(None, ON)
    mc = cfg.market
    logS = torch.log(simulate(mc, 128, 4, dtype=torch.float64).S)
    m = 17
    a = liquidity_state(cfg.liquidity, mc, logS, torch.Generator().manual_seed(9))
    b = liquidity_state(cfg.liquidity, mc, _permute_future(logS, m), torch.Generator().manual_seed(9))
    for k in a:
        assert torch.equal(a[k][:, : m + 1], b[k][:, : m + 1])
        assert not torch.allclose(a[k][:, m + 1:], b[k][:, m + 1:])


def test_cost_function_properties():
    lq = load_config(None, ON).liquidity
    S = torch.full((5,), 100.0, dtype=torch.float64)
    one = torch.ones(5, dtype=torch.float64)
    q = torch.tensor([0.0, 0.05, 0.1, 0.5, 1.0], dtype=torch.float64)
    c = liquidity_cost(q, S, one, one, 0.2 * one, lq)
    assert c[0] == 0 and (c.diff() > 0).all()
    # convex in q: cost at midpoint below the chord
    mid = liquidity_cost(torch.tensor([0.5], dtype=torch.float64), S[:1], one[:1], one[:1], 0.2 * one[:1], lq)
    assert mid < 0.5 * (c[0] + c[4])
    # more expensive when illiquid / volatile, cheaper when volume is high
    base = liquidity_cost(q[3:4], S[:1], one[:1], one[:1], 0.2 * one[:1], lq)
    assert liquidity_cost(q[3:4], S[:1], 3 * one[:1], one[:1], 0.2 * one[:1], lq) > base
    assert liquidity_cost(q[3:4], S[:1], one[:1], one[:1], 0.4 * one[:1], lq) > base
    assert liquidity_cost(q[3:4], S[:1], one[:1], 4 * one[:1], 0.2 * one[:1], lq) < base
    # normal-times magnitudes: 5 bp half-spread; full unwind of one share costs well under 1% of S
    assert abs(float(liquidity_cost(torch.tensor([1.0], dtype=torch.float64), S[:1], one[:1], one[:1], 0.2 * one[:1], lq))
               - 100 * (lq.s0 + lq.Y * 0.2 / math.sqrt(252) * math.sqrt(lq.psi))) < 1e-9


def test_zero_cost_scale_recovers_frictionless_rollout():
    on = load_config(None, ON + ["liquidity.cost_scale=0"])
    p = simulate_cfg(on, 256, 6, dtype=torch.float64)
    mc = on.market
    ro_liq = hedge_rollout(p, ConstHedger(-0.4), mc.K, mc.T, 0.0, "cash", on.liquidity)
    ro_off = hedge_rollout(p, ConstHedger(-0.4), mc.K, mc.T, 0.0, "none", None)
    assert torch.allclose(ro_liq.L, ro_off.L) and torch.allclose(ro_liq.C, torch.zeros_like(ro_liq.C))


def test_trading_and_cash_unwind_costs_in_rollout():
    on = load_config(None, ON + ["costs.unwind=cash"])
    p = simulate_cfg(on, 256, 7, dtype=torch.float64)
    mc, lq, e = on.market, on.liquidity, p.extras
    ro = hedge_rollout(p, ConstHedger(-0.4), mc.K, mc.T, 0.0, "cash", lq)
    c0 = liquidity_cost(torch.full((256,), 0.4, dtype=torch.float64), p.S_disc[:, 0], e["ell"][:, 0], e["vol"][:, 0],
                        e["sig"][:, 0], lq)
    assert torch.allclose(ro.C, c0.unsqueeze(1).expand_as(ro.C))        # only the initial purchase trades
    U = liquidity_cost(torch.full_like(ro.L, 0.4), p.S_disc[:, 1:], e["ell"][:, 1:], e["vol"][:, 1:], e["sig"][:, 1:], lq)
    assert torch.allclose(ro.U, U)
    assert torch.allclose(ro.L, p.Z_disc[:, 1:] - ro.G + ro.C + ro.U)


def test_networks_with_liquidity_inputs_are_non_anticipative():
    cfg = load_config(None, ON + ["costs.unwind=cash"])
    mc, lq = cfg.market, cfg.liquidity
    logS = torch.log(simulate(mc, 256, 8, dtype=torch.float64).S)
    torch.manual_seed(0)
    hedger, stopper = Hedger(cfg).double(), Stopper(cfg).double()
    out = []
    for ls in (logS, _permute_future(logS, 23)):
        ext = liquidity_state(lq, mc, ls, torch.Generator().manual_seed(11))
        p = build_paths(mc, ls, ext, dtype=torch.float64)
        with torch.no_grad():
            ro = hedge_rollout(p, hedger, mc.K, mc.T, 0.0, "cash", lq)
            out.append((ro.delta_prev, stopper_logits(p, ro, stopper, mc.K, mc.T)))
    (d0, l0), (d1, l1) = out
    m = 23
    assert torch.equal(d0[:, : m + 1], d1[:, : m + 1])      # delta_0..delta_m
    assert torch.equal(l0[:, :m], l1[:, :m])                # stopping logits at dates 1..m
    assert not torch.allclose(l0[:, m:], l1[:, m:])


LIQ_CKPTS = sorted(p for p in glob.glob(str(Path(__file__).resolve().parents[1] / "checkpoints" / "*_liq*.pt"))
                   if Path(p).name.startswith("gda"))


@pytest.mark.parametrize("ckpt", LIQ_CKPTS[:2])
def test_trained_liquidity_models_are_non_anticipative(ckpt):
    cfg = load_config(None, ON + ["costs.unwind=cash"])
    mc, lq = cfg.market, cfg.liquidity
    blob = torch.load(ckpt, map_location="cpu")
    hedger, stopper = Hedger(cfg).double(), Stopper(cfg).double()
    hedger.load_state_dict(blob["hedger"])
    stopper.load_state_dict(blob["stopper"])
    logS = torch.log(simulate(mc, 256, 12, dtype=torch.float64).S)
    m, out = 31, []
    for ls in (logS, _permute_future(logS, 31)):
        p = build_paths(mc, ls, liquidity_state(lq, mc, ls, torch.Generator().manual_seed(13)), dtype=torch.float64)
        with torch.no_grad():
            ro = hedge_rollout(p, hedger, mc.K, mc.T, 0.0, "cash", lq)
            out.append((ro.delta_prev, stopper_logits(p, ro, stopper, mc.K, mc.T)))
    (d0, l0), (d1, l1) = out
    assert torch.equal(d0[:, : m + 1], d1[:, : m + 1])
    assert torch.equal(l0[:, :m], l1[:, :m])

