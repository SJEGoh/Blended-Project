"""Simulator tests: MC European put vs Merton closed-form series (within 3 SE)."""
import math

import numpy as np
import pytest
import torch

from config import load_config
from sim import bs_put, merton_european_put, merton_put, simulate

N_PATHS = 2 ** 18


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_european_put_matches_merton_series(lam):
    cfg = load_config(None, [f"market.lam={lam}"])
    mc = cfg.market.risk_neutral()
    pb = simulate(mc, N_PATHS, 7, dtype=torch.float64)
    x = pb.Z_disc[:, -1].numpy()
    est, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    cf = merton_european_put(mc)
    assert abs(est - cf) < 3 * se, f"lam={lam}: MC {est:.4f} +- {se:.4f} vs closed form {cf:.4f}"


def test_discounted_spot_is_martingale_under_mu_eq_r():
    cfg = load_config(None, ["market.lam=1.0"])
    pb = simulate(cfg.market, N_PATHS, 8, dtype=torch.float64)
    for n in (10, 25, 50):
        x = pb.S_disc[:, n].numpy()
        se = x.std(ddof=1) / math.sqrt(len(x))
        assert abs(x.mean() - cfg.market.S0) < 3 * se


def test_merton_series_reduces_to_black_scholes():
    assert merton_put(100, 100, 0.5, 0.03, 0.2, 0.0, -0.1, 0.15) == pytest.approx(bs_put(100, 100, 0.5, 0.03, 0.2))


def test_common_random_numbers_across_lam():
    """Same seed, different lam: the diffusive part is shared, so lam=0 and lam=1 paths are
    identical wherever no jump occurred up to that date (up to the compensator drift)."""
    c0 = load_config(None, ["market.lam=0"]).market
    c1 = load_config(None, ["market.lam=1"]).market
    p0 = simulate(c0, 4096, 9, dtype=torch.float64)
    p1 = simulate(c1, 4096, 9, dtype=torch.float64)
    d = torch.log(p1.S) - torch.log(p0.S)
    comp = -c1.lam * c1.kappa * torch.tensor(c1.t_grid())
    no_jump = (d - comp).abs().max(1).values < 1e-9
    assert no_jump.float().mean() > 0.5   # P(no jump on [0, T]) = exp(-0.5) ~ 0.61
