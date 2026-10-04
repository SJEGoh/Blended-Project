"""Empirical CVaR and stopping-aggregation tests."""
import math

import numpy as np
import pytest
import torch
from scipy.stats import norm

from losses import cvar, cvar_estimate, hard_stop_index, relaxed_stop_weights
from models import stop_probs


def test_cvar_top_k_mean():
    X = torch.arange(1.0, 11.0)  # 1..10
    assert cvar(X, 0.0).item() == pytest.approx(5.5)
    assert cvar(X, 0.8).item() == pytest.approx(9.5)       # top 2
    assert cvar(X, 0.75).item() == pytest.approx(9.0)      # ceil(2.5) = top 3
    assert cvar(X, 0.9).item() == pytest.approx(10.0)


def test_cvar_gradient_only_on_tail():
    X = torch.arange(1.0, 11.0, requires_grad=True)
    cvar(X, 0.8).backward()
    assert torch.allclose(X.grad, torch.tensor([0.0] * 8 + [0.5, 0.5]))


@pytest.mark.parametrize("alpha", [0.5, 0.9, 0.99])
def test_cvar_estimate_and_se_on_normal(alpha):
    exact = norm.pdf(norm.ppf(alpha)) / (1 - alpha)
    rng = np.random.default_rng(0)
    ests, ses = [], []
    for _ in range(40):
        e, s = cvar_estimate(rng.standard_normal(2 ** 14), alpha)
        ests.append(e)
        ses.append(s)
    ests = np.array(ests)
    assert abs(ests.mean() - exact) < 4 * ests.std(ddof=1) / math.sqrt(len(ests)) + 2e-3
    assert np.mean(ses) == pytest.approx(ests.std(ddof=1), rel=0.35)  # SE formula is calibrated


def test_relaxed_weights_sum_to_one_and_hard_rule():
    torch.manual_seed(0)
    logits = torch.randn(1000, 50) * 3
    for temp in (1.0, 0.1):
        p = relaxed_stop_weights(stop_probs(logits, temp))
        assert torch.allclose(p.sum(1), torch.ones(1000), atol=1e-6)
        assert (p >= 0).all()
    idx = hard_stop_index(logits)
    first = torch.tensor([next((n for n in range(50) if logits[i, n] > 0), 49) for i in range(1000)])
    assert torch.equal(idx, first)


def test_relaxed_converges_to_hard_as_temperature_to_zero():
    torch.manual_seed(1)
    logits = torch.randn(500, 50) * 3
    p = relaxed_stop_weights(stop_probs(logits, 1e-4))
    onehot = torch.nn.functional.one_hot(hard_stop_index(logits), 50).float()
    assert torch.allclose(p, onehot, atol=1e-3)
