"""Part 1 pricing module: liquidity-adjusted exercise value, exact one-step reproduction, smoke run."""
import math

import torch

from config import load_config
from pricing import ContinuationNets, bracket_samples, exercise_value, inner_step, outer
from sim import simulate_cfg

BASE = ["market.lam=1.0"]
LIQ = BASE + ["liquidity.enabled=true"]


def test_exercise_value_cash_vs_liquidity():
    cfg = load_config(None, LIQ)
    Sd = torch.tensor([80.0, 95.0, 105.0], dtype=torch.float64)
    one = torch.ones(3, dtype=torch.float64)
    liq = dict(ell=one, vol=one, sig=0.2 * one)
    cash = exercise_value(cfg, "none", Sd, 100.0, None)
    assert torch.equal(cash, torch.tensor([20.0, 5.0, 0.0], dtype=torch.float64))
    phys = exercise_value(cfg, "liquidity", Sd, 100.0, liq)
    assert (phys <= cash).all() and (phys[:2] < cash[:2]).all()
    worse = exercise_value(cfg, "liquidity", Sd, 100.0, dict(ell=5 * one, vol=one, sig=0.4 * one))
    assert (worse <= phys).all() and (worse[:2] < phys[:2]).all()
    free = load_config(None, LIQ + ["liquidity.cost_scale=0"])
    assert torch.equal(exercise_value(free, "liquidity", Sd, 100.0, liq), cash)
    # normal-times cost of one share ~ 0.33% of S at psi = 0.05 (5 bp spread + square-root impact)
    kappa = cash[0] - phys[0]
    assert math.isclose(float(kappa) / 80.0, 0.0005 + 0.2 / math.sqrt(252) * math.sqrt(0.05), rel_tol=1e-9)


def test_inner_step_reproduces_outer_paths():
    for ov, mode in ((BASE, "none"), (LIQ, "liquidity")):
        cfg = load_config(None, ov)
        p, sh = simulate_cfg(cfg, 300, 31, dtype=torch.float64, return_shocks=True)
        Y, F, st = outer(cfg, mode, p)
        for k in (1, 7, cfg.market.N):
            y, f = inner_step(cfg, mode, st, k, {q: v[:, k - 1:k] for q, v in sh.items()})
            assert torch.allclose(y[:, 0], Y[:, k - 1], atol=1e-10)
            assert torch.allclose(f[:, 0], F[:, k - 1], atol=1e-10)


def test_bracket_smoke_runs_and_dual_dominates_pathwise():
    cfg = load_config(None, LIQ)
    p, sh = simulate_cfg(cfg, 512, 32, dtype=torch.float64, return_shocks=True)
    torch.manual_seed(0)
    nets = ContinuationNets(cfg, outer(cfg, "liquidity", p)[1].shape[-1], 8)
    r = bracket_samples(cfg, "liquidity", p, sh, nets, 2, 33)
    assert all(math.isfinite(float(v.mean())) for k, v in r.items() if k != "inc_z")
    assert (r["hind"] >= r["primal"] - 1e-12).all() and (r["hind"] >= r["euro"] - 1e-12).all()


def test_control_variate_lower_bound_is_below_dual_pathwise():
    """Y_tau - M_tau <= max_n (Y_n - M_n) on every path, so the control-variate lower bound never exceeds the dual."""
    cfg = load_config(None, BASE)
    p, sh = simulate_cfg(cfg, 256, 34, dtype=torch.float64, return_shocks=True)
    torch.manual_seed(1)
    nets = ContinuationNets(cfg, outer(cfg, "none", p)[1].shape[-1], 8)
    r = bracket_samples(cfg, "none", p, sh, nets, 2, 35)
    assert (r["dual"] >= r["primal_cv"] - 1e-10).all()
