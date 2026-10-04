"""Non-anticipativity: permuting a path's increments after date m must not change any decision
(hedge delta_n or stopping prob f_n) at dates n <= m."""
import glob
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks import LSMRule, fit_lsm
from config import load_config
from losses import hedge_rollout, stopper_logits, hard_stop_index
from models import Hedger, Stopper
from sim import build_paths, simulate

ROOT = Path(__file__).resolve().parents[1]


def _permute_future(logS: torch.Tensor, m: int, gen: torch.Generator) -> torch.Tensor:
    incr = logS.diff(dim=1)
    out = incr.clone()
    for i in range(incr.shape[0]):
        tail = incr[i, m:]
        out[i, m:] = tail[torch.randperm(len(tail), generator=gen)]
    return torch.cat([logS[:, :1], logS[:, :1] + out.cumsum(1)], 1)


def _models(cfg, ckpt=None):
    torch.manual_seed(3)
    hedger, stopper = Hedger(cfg).double(), Stopper(cfg).double()
    if ckpt is not None:
        blob = torch.load(ckpt, map_location="cpu")
        if "hedger" in blob and blob["hedger"] is not None:
            hedger.load_state_dict(blob["hedger"])
        stopper.load_state_dict(blob["stopper"])
    else:
        # make the random hedger produce non-trivial positions
        with torch.no_grad():
            hedger.net[-1].bias.fill_(-0.4)
    return hedger, stopper


def _decisions(cfg, paths, hedger, stopper):
    with torch.no_grad():
        ro = hedge_rollout(paths, hedger, cfg.market.K, cfg.market.T, cfg.costs.c)
        logits = stopper_logits(paths, ro, stopper, cfg.market.K, cfg.market.T)
    return ro.delta_prev, logits   # delta_prev[:, n-1] = delta_{n-1};  logits[:, n-1] = date n


# checkpoints trained without the liquidity extension (liquidity runs are covered in test_liquidity.py)
CKPTS = sorted(p for p in glob.glob(str(ROOT / "checkpoints" / "*.pt")) if "_liq" not in Path(p).name)


@pytest.mark.parametrize("ckpt", [None] + CKPTS[:3])
@pytest.mark.parametrize("m", [1, 10, 37])
def test_decisions_do_not_depend_on_future_increments(m, ckpt):
    cfg = load_config(None, ["market.lam=1.0", "costs.c=0.001"])
    mc = cfg.market
    pb = simulate(mc, 512, 11, dtype=torch.float64)
    logS = torch.log(pb.S)
    gen = torch.Generator().manual_seed(12)
    logS_perm = _permute_future(logS, m, gen)
    assert torch.allclose(logS_perm[:, : m + 1], logS[:, : m + 1])
    pa = build_paths(mc, logS, dtype=torch.float64)
    pp = build_paths(mc, logS_perm, dtype=torch.float64)
    hedger, stopper = _models(cfg, ckpt)
    d0, l0 = _decisions(cfg, pa, hedger, stopper)
    d1, l1 = _decisions(cfg, pp, hedger, stopper)
    # hedges delta_0..delta_m  (d[:, k] = delta_k for k = 0..N-1)
    assert torch.allclose(d0[:, : m + 1], d1[:, : m + 1], atol=0, rtol=0)
    # stopping logits at dates 1..m  (l[:, n-1] = date n)
    assert torch.allclose(l0[:, :m], l1[:, :m], atol=0, rtol=0)
    # hard stopping: if stopped by date m on the original path, same date on the permuted one
    i0, i1 = hard_stop_index(l0), hard_stop_index(l1)
    early = i0 < m
    assert torch.equal(i0[early], i1[early])
    # the test is not vacuous: decisions after m do change
    assert not torch.allclose(l0[:, m:], l1[:, m:])


def test_lsm_rule_is_non_anticipative():
    cfg = load_config(None, ["market.lam=1.0"])
    mc = cfg.market
    rule = fit_lsm(simulate(mc, 2 ** 14, 13, dtype=torch.float64), mc, 3)
    pb = simulate(mc, 512, 14, dtype=torch.float64)
    logS = torch.log(pb.S)
    m = 20
    pp = build_paths(mc, _permute_future(logS, m, torch.Generator().manual_seed(1)), dtype=torch.float64)
    a = rule.stop_matrix(pb.S.numpy())
    b = rule.stop_matrix(pp.S.numpy())
    assert np.array_equal(a[:, :m], b[:, :m])
