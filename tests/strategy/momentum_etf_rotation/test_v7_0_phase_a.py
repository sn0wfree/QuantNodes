"""v7.0 Phase A 单元测试: 交易成本 + 流动性 cap.

[Stage 30.5 Phase A] 验证:
    - apply_turnover_cost 不破坏 sum=1
    - apply_max_weight_cap 不破坏 sum=1
    - apply_turnover_cap 单 ETF 月度换手受限
    - portfolio_drag 估算正确
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    compute_turnover,
    apply_turnover_cost,
    portfolio_drag,
    apply_max_weight_cap,
    apply_turnover_cap,
)


# ====== transaction_cost tests ======

def test_compute_turnover_first_time():
    """第一次建仓 turnover = sum(w)/2."""
    w = {"510300": 0.3, "510500": 0.3, "159915": 0.4}
    t = compute_turnover(w, weights_old=None)
    assert abs(t - 0.5) < 1e-9


def test_compute_turnover_no_change():
    """权重不变 turnover = 0."""
    w = {"510300": 0.3, "510500": 0.3, "159915": 0.4}
    t = compute_turnover(w, weights_old=w)
    assert abs(t - 0.0) < 1e-9


def test_apply_turnover_cost_sum_one():
    """apply_turnover_cost 后 sum 仍 = 1."""
    w_new = {"510300": 0.3, "510500": 0.3, "159915": 0.4}
    w_old = {"510300": 0.5, "510500": 0.3, "159915": 0.2}
    w_adj = apply_turnover_cost(w_new, w_old, fee_bps=10.0)
    assert abs(sum(w_adj.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in w_adj.values())


def test_apply_turnover_cost_first_time():
    """第一次建仓时, turnover = sum/2, cost 仍应用."""
    w_new = {"510300": 0.3, "510500": 0.3, "159915": 0.4}
    w_adj = apply_turnover_cost(w_new, None, fee_bps=10.0)
    assert abs(sum(w_adj.values()) - 1.0) < 1e-9
    assert w_adj["159915"] <= 0.4  # 因 cost 略减或不变


def test_portfolio_drag_basic():
    """portfolio_drag = fee_bps * rebal_freq / 10000."""
    drag = portfolio_drag(fee_bps=10.0, rebal_freq_per_year=12)
    assert abs(drag - 0.012) < 1e-9
    drag2 = portfolio_drag(fee_bps=10.0, rebal_freq_per_year=4)
    assert abs(drag2 - 0.004) < 1e-9


# ====== liquidity_cap tests ======

def test_apply_max_weight_cap_basic():
    """单 ETF 30% cap (4+ ETF 池, 可满足 sum=1)."""
    w = {"510300": 0.5, "510500": 0.3, "159915": 0.1, "518880": 0.1}
    capped = apply_max_weight_cap(w, max_weight=0.30)
    assert all(v <= 0.30 + 1e-9 for v in capped.values())
    assert abs(sum(capped.values()) - 1.0) < 1e-9


def test_apply_max_weight_cap_below_limit():
    """权重均 < 30% 时不变."""
    w = {"510300": 0.2, "510500": 0.2, "159915": 0.2, "518880": 0.2, "513100": 0.2}
    capped = apply_max_weight_cap(w, max_weight=0.30)
    assert capped == w


def test_apply_max_weight_cap_empty():
    """空 dict 输入."""
    capped = apply_max_weight_cap({})
    assert capped == {}


def test_apply_turnover_cap_basic():
    """单 ETF 月度换手受限 (4+ ETF 池, normalize 后可保持 sum=1)."""
    w_new = {"510300": 0.5, "510500": 0.3, "159915": 0.1, "518880": 0.1}
    w_old = {"510300": 0.1, "510500": 0.3, "159915": 0.1, "518880": 0.1}
    capped = apply_turnover_cap(w_new, w_old, max_turnover=0.30)
    # 510300 旧 0.1, 新 0.5, delta=0.4 > 0.03 (30% × 0.1), 应被 cap
    # raw capped 510300 = 0.13
    # normalize 后应 < 0.30 (因为其他 ETF 保持原值, sum 变化不大)
    assert capped["510300"] <= 0.30 + 1e-9
    assert abs(sum(capped.values()) - 1.0) < 1e-9


def test_apply_turnover_cap_no_old():
    """无 old 时直接返回 new."""
    w_new = {"510300": 0.5, "510500": 0.3, "159915": 0.2}
    capped = apply_turnover_cap(w_new, None, max_turnover=0.30)
    assert capped == w_new


# ====== 集成测试 ======

def test_cost_then_cap_compose():
    """先 cost 后 cap, sum 仍 = 1 (4 ETF 池, 满足 0.30 cap + sum=1)."""
    w_new = {"510300": 0.5, "510500": 0.2, "159915": 0.2, "518880": 0.1}
    w_old = {"510300": 0.2, "510500": 0.2, "159915": 0.5, "518880": 0.1}
    w_cost = apply_turnover_cost(w_new, w_old, fee_bps=10.0)
    w_cap = apply_max_weight_cap(w_cost, max_weight=0.30)
    assert abs(sum(w_cap.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in w_cap.values())
    assert all(v <= 0.30 + 1e-9 for v in w_cap.values())
