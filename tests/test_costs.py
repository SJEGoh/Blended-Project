"""Transaction-cost bookkeeping: C_n (spec) and the optional unwind cost at exercise."""
import torch
from torch import nn

from config import load_config
from losses import hedge_rollout
from sim import simulate


class ConstHedger(nn.Module):
    is_zero = False

    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t_frac, logm, delta_prev, step_state=None, pnl=None):
        return torch.full_like(delta_prev, self.d)


def _setup(unwind):
    cfg = load_config(None, ["market.lam=1.0", "costs.c=0.005", f"costs.unwind={unwind}"])
    mc = cfg.market
    p = simulate(mc, 256, 21, dtype=torch.float64)
    return cfg, mc, p


def test_trading_cost_charges_only_the_initial_purchase_for_a_constant_hedge():
    cfg, mc, p = _setup("none")
    ro = hedge_rollout(p, ConstHedger(-0.5), mc.K, mc.T, cfg.costs.c, cfg.costs.unwind)
    expected = cfg.costs.c * 0.5 * p.S_disc[:, 0]
    assert torch.allclose(ro.C, expected.unsqueeze(1).expand_as(ro.C))
    assert torch.allclose(ro.U, torch.zeros_like(ro.U))
    G = (-0.5 * (p.S_disc[:, 1:] - p.S_disc[:, 0:1]))
    assert torch.allclose(ro.L, p.Z_disc[:, 1:] - G + ro.C)


def test_cash_unwind_cost():
    cfg, mc, p = _setup("cash")
    ro = hedge_rollout(p, ConstHedger(-0.5), mc.K, mc.T, cfg.costs.c, cfg.costs.unwind)
    assert torch.allclose(ro.U, cfg.costs.c * 0.5 * p.S_disc[:, 1:])
    assert torch.allclose(ro.L, p.Z_disc[:, 1:] - ro.G + ro.C + ro.U)


def test_physical_unwind_cost():
    cfg, mc, p = _setup("physical")
    ro = hedge_rollout(p, ConstHedger(-0.5), mc.K, mc.T, cfg.costs.c, cfg.costs.unwind)
    itm = (p.S[:, 1:] < mc.K).double()
    assert torch.allclose(ro.U, cfg.costs.c * (-0.5 + itm).abs() * p.S_disc[:, 1:])
