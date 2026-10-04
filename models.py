"""Hedger (seller, minimiser) and Stopper (adversarial holder, maximiser).

Both are MLPs shared across time with time as an input (depth x width SiLU layers).
Inputs are passed raw (as in the spec) and rescaled internally by *fixed* constants
(sigma*sqrt(T) for log-moneyness, K*sigma*sqrt(T) for money amounts); this is a fixed
affine reparametrisation of the first layer, not a change of the information set.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from config import Config


def mlp(in_dim: int, out_dim: int, hidden: int = 64, depth: int = 2) -> nn.Sequential:
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def constrain_trade(delta_prev: torch.Tensor, delta_raw: torch.Tensor, step_state: dict | None = None) -> torch.Tensor:
    """Extension hook for trading constraints. MVP: identity.
    Volume caps will clip |delta_n - delta_{n-1}| <= v_n with v_n taken from step_state["v"]."""
    return delta_raw


def liquidity_features(ell, vol, sig, sigma_ref) -> list:
    """Observed liquidity state -> network features (log scale)."""
    return [torch.log(ell), torch.log(vol), torch.log(sig / sigma_ref)]


class Hedger(nn.Module):
    """delta_n = constrain_trade(delta_{n-1}, H_theta(t_n/T, log(S_n/K), delta_{n-1})).
    With cfg.train.hedger_pnl_input (ablation, off by default) the running P&L
    Z~_n - G~_n + C_n is a fourth input."""

    is_zero = False

    def __init__(self, cfg: Config):
        super().__init__()
        mc = cfg.market
        self.x_scale = mc.sigma * math.sqrt(mc.T)
        self.pnl_scale = mc.K * mc.sigma * math.sqrt(mc.T)
        self.use_pnl = bool(cfg.train.hedger_pnl_input)
        self.use_liq = bool(cfg.liquidity.enabled)       # observe (ell, vol, sig) at t_n
        self.sigma_ref = cfg.liquidity.sigma_ref or mc.sigma
        n_in = 3 + int(self.use_pnl) + 3 * int(self.use_liq)
        self.net = mlp(n_in, 1, cfg.model.hidden, cfg.model.depth)

    def forward(self, t_frac, logm, delta_prev, step_state: dict | None = None, pnl=None):
        feats = [t_frac.expand_as(logm), logm / self.x_scale, delta_prev]
        if self.use_pnl:
            feats.append(pnl / self.pnl_scale)
        if self.use_liq:
            feats += liquidity_features(step_state["ell"], step_state["vol"], step_state["sig"], self.sigma_ref)
        raw = self.net(torch.stack(feats, dim=-1)).squeeze(-1)
        return constrain_trade(delta_prev, raw, step_state)


class ZeroHedger(nn.Module):
    """No hedge (delta = 0); used for build-order step 2 and as the 'no hedge' reference."""

    is_zero = True

    def forward(self, t_frac, logm, delta_prev, step_state=None, pnl=None):
        return torch.zeros_like(delta_prev)


class Stopper(nn.Module):
    """Adversarial holder. At date n (1..N) sees only F_{t_n}-measurable inputs:
    t_n/T, log(S_n/K), delta_{n-1}, running hedged P&L L_n = Z~_n - G~_n + C_n, 1{S_n < K},
    and, with the liquidity extension, the liquidity state (ell_n, vol_n, sig_n).
    Returns logits; stopping probability f_n = sigmoid(logit / temperature), f_N := 1."""

    def __init__(self, cfg: Config):
        super().__init__()
        mc = cfg.market
        self.x_scale = mc.sigma * math.sqrt(mc.T)
        self.pnl_scale = mc.K * mc.sigma * math.sqrt(mc.T)
        self.use_liq = bool(cfg.liquidity.enabled)
        self.sigma_ref = cfg.liquidity.sigma_ref or mc.sigma
        self.net = mlp(5 + 3 * int(self.use_liq), 1, cfg.model.hidden, cfg.model.depth)

    def forward(self, t_frac, logm, delta_prev, pnl, itm, liq_feats: dict | None = None):
        feats = [t_frac.expand_as(logm), logm / self.x_scale, delta_prev, pnl / self.pnl_scale, itm]
        if self.use_liq:
            feats += liquidity_features(liq_feats["ell"], liq_feats["vol"], liq_feats["sig"], self.sigma_ref)
        return self.net(torch.stack(feats, dim=-1)).squeeze(-1)


def stop_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """f_n = sigmoid(logit_n / temperature) for dates 1..N, with f_N forced to 1."""
    f = torch.sigmoid(logits / temperature)
    return torch.cat([f[..., :-1], torch.ones_like(f[..., -1:])], dim=-1)
