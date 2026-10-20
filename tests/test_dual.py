"""Dual (martingale) upper bounds: exposed shocks, basis martingale property, non-anticipativity, and the
hedge-gains special case (penalty = hedge gains)."""
import math

import numpy as np
import torch

from config import config_from_dict, load_config
from dual import MartingaleNet, basis, basis_names, payoff
from losses import hedge_rollout
from models import Hedger, ZeroHedger
from sim import build_paths, liquidity_state, simulate, simulate_cfg

LIQ = ["market.lam=1.0", "liquidity.enabled=true", "costs.unwind=cash"]


def _permute_future(logS, m, seed=5):
    g = torch.Generator().manual_seed(seed)
    incr = logS.diff(dim=1).clone()
    for i in range(incr.shape[0]):
        tail = incr[i, m:]
        incr[i, m:] = tail[torch.randperm(len(tail), generator=g)]
    return torch.cat([logS[:, :1], logS[:, :1] + incr.cumsum(1)], 1)


def test_shock_flag_leaves_paths_bit_identical():
    for ov in (["market.lam=1.0"], LIQ):
        cfg = load_config(None, ov)
        a = simulate_cfg(cfg, 300, 11, dtype=torch.float64)
        b, sh = simulate_cfg(cfg, 300, 11, dtype=torch.float64, return_shocks=True)
        assert torch.equal(a.S, b.S) and torch.equal(a.Z_disc, b.Z_disc)
        assert a.extras.keys() == b.extras.keys() and all(torch.equal(a.extras[k], b.extras[k]) for k in a.extras)
        assert all(v.shape == (300, cfg.market.N) for v in sh.values())
        assert ("eps" in sh) == cfg.liquidity.enabled


def test_shocks_rebuild_the_price_paths():
    cfg = load_config(None, ["market.lam=1.0"])
    mc = cfg.market
    p, sh = simulate_cfg(cfg, 500, 12, dtype=torch.float64, return_shocks=True)
    incr = ((mc.mu_eff - mc.lam * mc.kappa - 0.5 * mc.sigma ** 2) * mc.dt + mc.sigma * math.sqrt(mc.dt) * sh["Z"]
            + sh["Nj"] * mc.m + mc.delta * torch.sqrt(sh["Nj"]) * sh["Zj"])
    assert torch.allclose(torch.log(p.S).diff(dim=1), incr, atol=1e-12)


def test_stock_return_basis_term_matches_paths():
    cfg = load_config(None, ["market.lam=1.0"])
    mc = cfg.market
    p, sh = simulate_cfg(cfg, 500, 13, dtype=torch.float64, return_shocks=True)
    xi = basis(cfg, p, sh)
    j = basis_names(cfg).index("dS/S")
    ret = p.S_disc[:, 1:] / p.S_disc[:, :-1] - 1.0
    assert torch.allclose(xi[..., j] * mc.sigma * math.sqrt(mc.dt), ret, atol=1e-12)


def test_basis_terms_have_mean_zero():
    cfg = load_config(None, LIQ)
    p, sh = simulate_cfg(cfg, 2 ** 16, 14, dtype=torch.float64, return_shocks=True)
    xi = basis(cfg, p, sh).reshape(-1, len(basis_names(cfg)))
    z = xi.mean(0) / (xi.std(0) / math.sqrt(xi.shape[0]))
    assert (z.abs() < 4.0).all(), dict(zip(basis_names(cfg), z.tolist()))


def test_martingale_coefficients_are_non_anticipative():
    """phi for step k+1 uses F_k only: rows 0..m of the features (steps 1..m+1) do not change when the
    price path after t_m is permuted (liquidity noise held fixed)."""
    cfg = load_config(None, LIQ)
    mc, lq = cfg.market, cfg.liquidity
    logS = torch.log(simulate(mc, 256, 15, dtype=torch.float64).S)
    torch.manual_seed(0)
    hedger = Hedger(cfg).double()
    net = MartingaleNet(cfg, len(basis_names(cfg)), use_c=True).double()
    for prm in net.parameters():                        # non-zero output layer
        torch.nn.init.normal_(prm, std=0.3)
    m, out = 23, []
    for ls in (logS, _permute_future(logS, m)):
        p = build_paths(mc, ls, liquidity_state(lq, mc, ls, torch.Generator().manual_seed(16)), dtype=torch.float64)
        with torch.no_grad():
            ro = hedge_rollout(p, hedger, mc.K, mc.T, 0.0, "cash", lq)
            out.append(net.net(net.features(cfg, p, ro, torch.tensor(12.0, dtype=torch.float64))))
    a, b = out
    assert torch.equal(a[:, : m + 1], b[:, : m + 1])
    assert not torch.allclose(a[:, m + 1:], b[:, m + 1:])


def test_hedge_gains_are_a_special_case():
    """With coefficient delta_{k-1} S~_{k-1} sigma sqrt(dt) on the stock-return term, M equals the hedge's
    gains G, so the unhedged dual max_n (Z~_n - M_n) equals the hedged hindsight max_n L_n (frictionless)."""
    cfg = load_config(None, ["market.lam=1.0"])
    mc = cfg.market
    p, sh = simulate_cfg(cfg, 2000, 17, dtype=torch.float64, return_shocks=True)
    torch.manual_seed(1)
    hedger = Hedger(cfg).double()
    with torch.no_grad():
        ro = hedge_rollout(p, hedger, mc.K, mc.T)
        ro0 = hedge_rollout(p, ZeroHedger(), mc.K, mc.T)
    xi = basis(cfg, p, sh)
    j = basis_names(cfg).index("dS/S")
    M = (ro.delta_prev * p.S_disc[:, :-1] * mc.sigma * math.sqrt(mc.dt) * xi[..., j]).cumsum(1)
    assert torch.allclose(M, ro.G, atol=1e-9)
    lhs = (payoff(ro0.L, 0.0, 0.0) - M).max(1).values
    assert torch.allclose(lhs, ro.L.max(1).values, atol=1e-9)


def test_payoff_and_zero_init():
    cfg = load_config(None, ["market.lam=1.0"])
    L = torch.tensor([[1.0, 5.0, -2.0]])
    assert torch.equal(payoff(L, 2.0, 0.9), torch.tensor([[0.0, 3.0, 0.0]]))
    assert torch.equal(payoff(L, 2.0, 0.0), L)
    net = MartingaleNet(cfg, len(basis_names(cfg)), use_c=True)
    p, sh = simulate_cfg(cfg, 64, 18, return_shocks=True)
    with torch.no_grad():
        ro = hedge_rollout(p, ZeroHedger(), cfg.market.K, cfg.market.T)
        M = net(net.features(cfg, p, ro, torch.tensor(5.0)), basis(cfg, p, sh).float())
    assert torch.equal(M, torch.zeros_like(M))          # training starts from the hindsight bound


def test_config_round_trip_through_checkpoint_dict():
    cfg = load_config(None, LIQ + ["train.alpha=0.95"])
    assert config_from_dict(cfg.to_dict()) == cfg
    assert config_from_dict(cfg.to_dict(), ["train.alpha=0.5"]).train.alpha == 0.5


def test_inner_step_reproduces_the_rollout_with_the_paths_own_shocks():
    """dual_value.inner_step fed the outer path's shocks rebuilds L_k and the state features exactly, in every
    cost setting; this is what makes the value-function martingale's outer and inner states comparable."""
    from dual_value import inner_step, outer_features, outer_state
    for ov in (["market.lam=1.0"], ["market.lam=1.0", "costs.c=0.005", "costs.unwind=physical"], LIQ):
        cfg = load_config(None, ov)
        mc = cfg.market
        p, sh = simulate_cfg(cfg, 400, 19, dtype=torch.float64, return_shocks=True)
        torch.manual_seed(3)
        hedger = Hedger(cfg).double()
        with torch.no_grad():
            ro = hedge_rollout(p, hedger, mc.K, mc.T, cfg.costs.c, cfg.costs.unwind,
                               cfg.liquidity if cfg.liquidity.enabled else None)
        st = outer_state(cfg, p, ro)
        Fo = outer_features(cfg, p, ro, 7.0)
        for k in (1, 2, 17, mc.N):
            L, F = inner_step(cfg, st, k, {q: v[:, k - 1:k] for q, v in sh.items()}, 7.0)
            assert torch.allclose(L[:, 0], ro.L[:, k - 1], atol=1e-9), (ov, k)
            assert torch.allclose(F[:, 0], Fo[:, k - 1], atol=1e-9), (ov, k)


def test_inner_draws_give_exactly_unbiased_conditional_means():
    """Stratified + antithetic inner draws: weights sum to 1, odd functions average to exactly 0 per path, and
    the jump count and the discounted stock return have the right means (within Monte Carlo error)."""
    from dual_value import draw_inner
    cfg = load_config(None, LIQ)
    mc = cfg.market
    sh, w = draw_inner(cfg, 4000, 8, torch.Generator().manual_seed(20), "cpu")
    assert math.isclose(float(w.sum()), 1.0, rel_tol=1e-12)
    for q in ("Z", "Zj", "eps", "eps_v"):
        assert torch.allclose((sh[q] * w).sum(1), torch.zeros(4000, dtype=torch.float64), atol=1e-12)
    n_hat = (sh["Nj"] * w).sum(1)                       # per-path estimates of E[N] = lam dt
    assert abs(float(n_hat.mean()) - mc.lam * mc.dt) < 4 * float(n_hat.std()) / math.sqrt(4000) + 1e-12
    incr = (-(mc.lam * mc.kappa + 0.5 * mc.sigma ** 2) * mc.dt + mc.sigma * math.sqrt(mc.dt) * sh["Z"]
            + sh["Nj"] * mc.m + mc.delta * torch.sqrt(sh["Nj"]) * sh["Zj"])
    r_hat = (torch.expm1(incr) * w).sum(1)              # E[S~_k / S~_(k-1) - 1] = 0
    assert abs(float(r_hat.mean())) < 4 * float(r_hat.std()) / math.sqrt(4000)
