"""Volume caps (hard participation limit on rebalancing): off by default, bound respected, scales with
volume, unwind exempt, non-anticipative, separate run names."""
import math

import torch

from config import load_config
from losses import hedge_rollout, liquidity_cost
from models import Hedger, VolumeCap, constrain_trade
from sim import build_paths, liquidity_state, simulate, simulate_cfg
from train import run_name

LIQ = ["market.lam=1.0", "liquidity.enabled=true", "costs.unwind=cash", "liquidity.psi=1.0"]
CAP = LIQ + ["volume_cap.enabled=true"]


def _wild_hedger(cfg, seed=0):
    """Hedger with large, fast-moving outputs so the cap binds often."""
    torch.manual_seed(seed)
    h = Hedger(cfg).double()
    for p in h.parameters():
        torch.nn.init.normal_(p, std=1.0)
    return h


def test_off_by_default_and_run_names():
    cfg = load_config(None, LIQ)
    assert not cfg.volume_cap.enabled and Hedger(cfg).cap is None
    assert "vcap" not in run_name(cfg, "gda")
    on = load_config(None, CAP)
    assert run_name(on, "gda").endswith("_vcap0.1psi1") and Hedger(on).cap is not None


def test_cap_is_identity_when_absent():
    d0, raw = torch.zeros(5), torch.linspace(-3, 3, 5)
    assert torch.equal(constrain_trade(d0, raw, None, None), raw)


def test_trades_respect_the_cap_including_the_first_one():
    for mode in ("clamp", "tanh"):
        cfg = load_config(None, CAP + [f"volume_cap.mode={mode}"])
        mc = cfg.market
        p = simulate_cfg(cfg, 2000, 21, dtype=torch.float64)
        with torch.no_grad():
            ro = hedge_rollout(p, _wild_hedger(cfg), mc.K, mc.T, 0.0, "cash", cfg.liquidity)
        D = ro.delta_prev
        trades = torch.cat([D[:, :1], D.diff(dim=1)], 1).abs()          # first trade is from delta_{-1} = 0
        q = VolumeCap(cfg).q_normal * p.extras["vol"][:, :-1]
        assert (trades <= q * (1 + 1e-9)).all()
        assert (trades > 0.999 * q).float().mean() > 0.05 if mode == "clamp" else True   # it does bind


def test_cap_scales_with_volume_and_defaults_to_normal_volume():
    cfg = load_config(None, CAP)
    cap = VolumeCap(cfg)
    assert math.isclose(cap.q_normal, 0.10 * 0.01 * 252 / 1.0)
    vol = torch.tensor([0.5, 1.0, 3.0])
    assert torch.allclose(cap.q_max({"vol": vol}, vol), cap.q_normal * vol)
    off = load_config(None, ["market.lam=1.0", "volume_cap.enabled=true"])         # liquidity off: v = 1
    like = torch.zeros(4)
    assert torch.allclose(VolumeCap(off).q_max(None, like), torch.full((4,), 0.10 * 2.52 / 0.05))


def test_tanh_mode_keeps_gradients_where_clamp_does_not():
    cfg = load_config(None, CAP + ["volume_cap.mode=tanh"])
    cap = VolumeCap(cfg)
    raw = torch.tensor([2.0], requires_grad=True)
    constrain_trade(torch.zeros(1), raw, None, cap).sum().backward()
    assert raw.grad.abs().item() > 0
    cfg_c = load_config(None, CAP)
    raw_c = torch.tensor([2.0], requires_grad=True)
    constrain_trade(torch.zeros(1), raw_c, None, VolumeCap(cfg_c)).sum().backward()
    assert raw_c.grad.abs().item() == 0


def test_unwind_at_exercise_is_not_capped():
    cfg = load_config(None, CAP)
    mc, lq = cfg.market, cfg.liquidity
    p = simulate_cfg(cfg, 500, 22, dtype=torch.float64)
    with torch.no_grad():
        ro = hedge_rollout(p, _wild_hedger(cfg, 1), mc.K, mc.T, 0.0, "cash", lq)
    e = p.extras
    U = liquidity_cost(ro.delta_prev.abs(), p.S_disc[:, 1:], e["ell"][:, 1:], e["vol"][:, 1:], e["sig"][:, 1:], lq)
    assert torch.allclose(ro.U, U)
    assert (ro.delta_prev.abs() > VolumeCap(cfg).q_normal * e["vol"][:, 1:]).any()    # positions above one step's cap


def test_capped_hedge_is_non_anticipative():
    cfg = load_config(None, CAP)
    mc, lq = cfg.market, cfg.liquidity
    logS = torch.log(simulate(mc, 256, 23, dtype=torch.float64).S)
    g = torch.Generator().manual_seed(5)
    incr = logS.diff(dim=1).clone()
    m = 19
    for i in range(incr.shape[0]):
        incr[i, m:] = incr[i, m:][torch.randperm(mc.N - m, generator=g)]
    logS2 = torch.cat([logS[:, :1], logS[:, :1] + incr.cumsum(1)], 1)
    h = _wild_hedger(cfg, 2)
    out = []
    for ls in (logS, logS2):
        p = build_paths(mc, ls, liquidity_state(lq, mc, ls, torch.Generator().manual_seed(24)), dtype=torch.float64)
        with torch.no_grad():
            out.append(hedge_rollout(p, h, mc.K, mc.T, 0.0, "cash", lq).delta_prev)
    assert torch.equal(out[0][:, : m + 1], out[1][:, : m + 1])
