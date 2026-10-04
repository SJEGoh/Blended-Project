"""Typed configuration loaded from config.yaml (+ optional --set overrides)."""
from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import os
import random
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


@dataclass
class MarketCfg:
    S0: float = 100.0
    K: float = 100.0
    T: float = 0.5
    N: int = 50
    r: float = 0.03
    mu: float | None = None
    sigma: float = 0.20
    lam: float = 1.0
    m: float = -0.10
    delta: float = 0.15

    @property
    def mu_eff(self) -> float:
        return self.r if self.mu is None else float(self.mu)

    @property
    def dt(self) -> float:
        return self.T / self.N

    @property
    def kappa(self) -> float:
        return math.exp(self.m + 0.5 * self.delta**2) - 1.0

    def t_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.T, self.N + 1)

    def risk_neutral(self) -> "MarketCfg":
        out = copy.copy(self)
        out.mu = None
        return out


@dataclass
class CostCfg:
    c: float = 0.0
    unwind: str = "none"   # cost of closing the hedge at exercise: none (spec) | cash | physical


@dataclass
class LiquidityCfg:
    """Stochastic liquidity (volume-risk extension). Off by default -> MVP behaviour unchanged.
    State (non-tradable, observed at each date): log-illiquidity x (Euler-simulated, mean-reverting,
    jumps up after large declines), relative market volume v (rises with |return|), and an EWMA
    volatility estimate sig. Trading q shares (per option) at t_n costs, in discounted units,
        [ s0 * ell_n * (sig_n / sigma_ref) * q  +  Y * ell_n * (sig_n / sqrt(252)) * sqrt(psi / v_n) * q^1.5 ] * S~_n
    (half-spread proportional to volatility; square-root market impact). See LIQUIDITY.md."""
    enabled: bool = False
    name: str = "base"           # label used in run names
    cost_scale: float = 1.0      # multiplies s0 and Y (for scaling checks)
    s0: float = 0.0005           # normal-times half-spread, fraction of price (5 bp)
    Y: float = 1.0               # square-root impact prefactor (~1 empirically)
    psi: float = 0.05            # position size / normal daily volume (shares per option x options sold)
    kappa: float = 20.0          # mean reversion of log-illiquidity (per year): half-life ~ 9 trading days
    eta: float = 1.0             # vol of log-illiquidity (per sqrt(year))
    beta_down: float = 15.0      # illiquidity jump per unit of log-decline beyond the threshold
    h_mult: float = 2.0          # decline threshold, in units of sigma*sqrt(dt)
    gamma_v: float = 0.3         # log-volume sensitivity to |standardised return|
    zeta: float = 0.2            # idiosyncratic log-volume noise
    ewma_w: float = 0.3          # EWMA weight for the volatility estimate sig
    zcap: float = 10.0           # cap on |standardised return| in the volume equation
    ell_max: float = 10.0        # cap on illiquidity (spreads/impact at most 10x normal from this factor)
    sig_ratio_max: float = 4.0   # cap on sig / sigma_ref in the cost function
    sigma_ref: float | None = None  # reference volatility for the spread (None -> market.sigma)
    seed_offset: int = 7777      # liquidity noise for the saved test set uses test_seed + seed_offset


@dataclass
class ModelCfg:
    hidden: int = 64
    depth: int = 2


@dataclass
class TrainCfg:
    alpha: float = 0.9
    outer_iters: int = 3000
    k_adv: int = 5
    batch_log2: int = 14
    lr_hedger: float = 1e-3
    lr_stopper: float = 1e-3
    temp_start: float = 1.0
    temp_end: float = 0.1
    adversary_pool: bool = True
    pool_every: int = 200
    log_every: int = 50
    stopper_init: str = "heuristic"  # heuristic (adopted after step 2) | random (spec as written)
    warm_steps: int = 300
    hedger_pnl_input: bool = False    # ablation flag: also feed running P&L to the hedger
    hedger_grad_through_stopper: bool = True  # hedger gradient flows through the stopper's inputs
    # Proposed fix for adversary collapse (off by default): every adv_restart_every outer iterations,
    # warm-start a fresh stopper, train it adv_restart_steps ascent steps against the frozen hedger, and
    # swap it in if its hard-rule CVaR on a common batch beats the current stopper's.
    adv_restart_every: int = 0
    adv_restart_steps: int = 200


@dataclass
class StopperOnlyCfg:
    steps: int = 3000


@dataclass
class ExploitCfg:
    steps: int = 2000


@dataclass
class EvalCfg:
    test_log2: int = 18
    test_seed: int = 20261002


@dataclass
class LSMCfg:
    degree: int = 3
    n_reg_log2: int = 18
    n_price_log2: int = 18
    seed_reg: int = 101
    seed_price: int = 202


@dataclass
class CRRCfg:
    steps: int = 5000


@dataclass
class PathsCfg:
    data_dir: str = "data"
    results_dir: str = "results"
    ckpt_dir: str = "checkpoints"


@dataclass
class Config:
    tag: str = ""          # appended to run names (e.g. "reduced")
    seed: int = 1234
    device: str = "auto"
    threads: int = 0
    market: MarketCfg = field(default_factory=MarketCfg)
    costs: CostCfg = field(default_factory=CostCfg)
    liquidity: LiquidityCfg = field(default_factory=LiquidityCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    stopper_only: StopperOnlyCfg = field(default_factory=StopperOnlyCfg)
    exploit: ExploitCfg = field(default_factory=ExploitCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    lsm: LSMCfg = field(default_factory=LSMCfg)
    crr: CRRCfg = field(default_factory=CRRCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)

    # ---- directories (resolved relative to the repo root) ----
    def dir(self, name: str) -> Path:
        p = ROOT / getattr(self.paths, f"{name}_dir")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, data: dict):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        ftype = f.default_factory if f.default_factory is not dataclasses.MISSING else None
        if ftype is not None and is_dataclass(ftype):
            kwargs[f.name] = _build(ftype, v or {})
        else:
            kwargs[f.name] = v
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


def _apply_override(d: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"Override must look like section.key=value, got {assignment!r}")
    parts = key.strip().split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    for ov in overrides or []:
        _apply_override(data, ov)
    cfg = _build(Config, data)
    # numeric coercion (yaml may give ints where floats are expected)
    for sect in (cfg.market, cfg.costs, cfg.train):
        for f in fields(sect):
            v = getattr(sect, f.name)
            if f.type in ("float", float) and v is not None:
                setattr(sect, f.name, float(v))
    for f in fields(cfg.liquidity):
        v = getattr(cfg.liquidity, f.name)
        if f.type in ("float", float) and v is not None:
            setattr(cfg.liquidity, f.name, float(v))
    if cfg.liquidity.sigma_ref is None:
        cfg.liquidity.sigma_ref = cfg.market.sigma
    if cfg.crr.steps % cfg.market.N != 0:
        raise ValueError("crr.steps must be a multiple of market.N")
    return cfg


def liq_of(cfg: Config):
    """The liquidity config if the extension is enabled, else None."""
    return cfg.liquidity if cfg.liquidity.enabled else None


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="path to YAML config (default: config.yaml)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="config overrides")
    return parser


def config_from_args(args) -> Config:
    return load_config(args.config, args.set)


def get_device(cfg: Config):
    import torch

    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    # Low temperatures push sigmoid/cumprod outputs into float32 denormals, which are ~10x slower on
    # CPU; flushing them to zero changes values only below ~1e-38.
    torch.set_flush_denormal(True)
    if cfg.device != "auto":
        return torch.device(cfg.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lam_tag(lam: float) -> str:
    return f"lam{lam:g}".replace(".", "p")


def alpha_tag(alpha: float) -> str:
    return f"a{alpha:g}".replace(".", "p")
