"""Adversary restarts (train.adv_restart_every): off by default, opt-in run names, swap only when better."""
import copy

import torch

from config import load_config
from models import Hedger, Stopper
from train import restart_adversary, run_name

BASE = ["market.lam=1.0", "train.alpha=0.9", "liquidity.enabled=true", "costs.unwind=cash"]


def test_off_by_default_and_run_names():
    cfg = load_config(None, BASE)
    assert cfg.train.adv_restart_every == 0
    assert "_rs" not in run_name(cfg, "gda") and "_rs" not in run_name(cfg, "exploit")
    on = load_config(None, BASE + ["train.adv_restart_every=150"])
    assert "_rs150" in run_name(on, "gda") and "_rs150" in run_name(on, "exploit")
    assert "_rs" not in run_name(on, "fixed-lsm")          # fixed-rule hedgers have no adversary to restart


def test_restart_swaps_only_when_fresh_is_better():
    cfg = load_config(None, BASE + ["train.batch_log2=8", "train.warm_steps=3", "train.adv_restart_steps=2"])
    torch.manual_seed(0)
    hedger, stopper = Hedger(cfg), Stopper(cfg)
    before = copy.deepcopy(stopper.state_dict())
    r = restart_adversary(cfg, stopper, hedger, torch.Generator().manual_seed(1), "cpu", 0.5, cfg.train.alpha)
    assert set(r) == {"current", "fresh", "fresh_frac_early", "swapped"}
    assert r["swapped"] == (r["fresh"] > r["current"])
    same = all(torch.equal(before[k], v) for k, v in stopper.state_dict().items())
    assert same != r["swapped"]
