"""Benchmark tests: CRR sanity and LSM vs CRR when lam = 0."""
import pytest

from benchmarks import crr_put, lsm
from config import load_config
from sim import bs_put


def test_crr_european_matches_black_scholes():
    assert crr_put(100, 100, 0.5, 0.03, 0.2, 5000, "european") == pytest.approx(bs_put(100, 100, 0.5, 0.03, 0.2), abs=2e-3)


def test_crr_ordering_american_bermudan_european():
    am = crr_put(100, 100, 0.5, 0.03, 0.2, 2000, "american")
    be = crr_put(100, 100, 0.5, 0.03, 0.2, 2000, "bermudan", 50)
    eu = crr_put(100, 100, 0.5, 0.03, 0.2, 2000, "european")
    assert am >= be >= eu


def test_lsm_matches_crr_bermudan_when_no_jumps():
    cfg = load_config(None, ["market.lam=0"])
    mc = cfg.market
    res = lsm(cfg)
    crr_b = crr_put(mc.S0, mc.K, mc.T, mc.r, mc.sigma, cfg.crr.steps, "bermudan", mc.N)
    crr_a = crr_put(mc.S0, mc.K, mc.T, mc.r, mc.sigma, cfg.crr.steps, "american")
    # LSM (independent pricing paths) is low-biased: allow 3 SE below plus 1% tolerance
    assert abs(res.price - crr_b) / crr_b < 0.01, (res.price, res.se, crr_b)
    assert abs(res.price - crr_a) / crr_a < 0.01
    assert res.price < crr_b + 3 * res.se
