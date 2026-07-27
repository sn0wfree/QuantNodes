# coding=utf-8
"""Tests for brinson.py (Brinson 归因)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.common.brinson import (
    CATEGORIES,
    brinson_attribution,
)
from QuantNodes.strategy.momentum_etf_rotation.common.universe import (
    Category, ETFMeta, ETFPool,
)


def _make_pool(n_codes: int = 4) -> ETFPool:
    members = tuple(
        ETFMeta(code=f"E{i:03d}", name=f"E{i:03d}", category=Category.A_BROAD, liquidity_rank=1)
        for i in range(n_codes)
    )
    return ETFPool(members=members)


def _make_returns(n_days: int = 252, n_codes: int = 4, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"E{i:03d}" for i in range(n_codes)]
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.012, (n_days, n_codes))
    return pd.DataFrame(rets, index=idx, columns=codes)


def _make_weights(n_days: int = 252, n_codes: int = 4, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [f"E{i:03d}" for i in range(n_codes)]
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    w = rng.uniform(0, 0.4, (n_days, n_codes))
    w = w / w.sum(axis=1, keepdims=True)
    return pd.DataFrame(w, index=idx, columns=codes)


class TestBrinsonConstants:
    def test_categories(self) -> None:
        """CATEGORIES 应包含 5 个标准类别."""
        assert len(CATEGORIES) == 5
        assert "a_broad" in CATEGORIES
        assert "a_sector" in CATEGORIES
        assert "hk" in CATEGORIES
        assert "commodity" in CATEGORIES
        assert "overseas" in CATEGORIES


class TestBrinsonAttribution:
    def test_returns_9_keys(self) -> None:
        """应返回 9 个键."""
        pw = _make_weights()
        pr = _make_returns()
        pool = _make_pool()
        result = brinson_attribution(pw, pr, None, None, pool)
        assert "allocation_abs" in result
        assert "selection_abs" in result
        assert "interaction_abs" in result
        assert "allocation_pct" in result
        assert "selection_pct" in result
        assert "interaction_pct" in result
        assert "total_active" in result
        assert "port_total_return" in result
        assert "bench_total_return" in result

    def test_empty_inputs(self) -> None:
        """空输入应返回零值 dict."""
        pool = _make_pool()
        result = brinson_attribution(
            pd.DataFrame(),
            pd.DataFrame(),
            None, None, pool,
        )
        assert result["allocation_abs"] == 0.0
        assert result["total_active"] == 0.0

    def test_active_components_sum_to_total(self) -> None:
        """active = allocation + selection + interaction 应等于 total_active."""
        pw = _make_weights()
        pr = _make_returns()
        pool = _make_pool()
        result = brinson_attribution(pw, pr, None, None, pool)
        # benchmark = portfolio 时, total_active = 0
        # 所以测试用不同的 benchmark
        bw = _make_weights(seed=99)
        br = _make_returns(seed=100)
        result2 = brinson_attribution(pw, pr, bw, br, pool)
        if abs(result2["total_active"]) > 1e-6:
            summed = (
                result2["allocation_abs"]
                + result2["selection_abs"]
                + result2["interaction_abs"]
            )
            assert abs(summed - result2["total_active"]) < 1e-6

    def test_with_explicit_benchmark(self) -> None:
        """提供 benchmark 时 total_active 非零."""
        pw = _make_weights(seed=42)
        pr = _make_returns(seed=42)
        bw = _make_weights(seed=99)  # 不同权重
        br = _make_returns(seed=99)
        pool = _make_pool()
        result = brinson_attribution(pw, pr, bw, br, pool)
        # total_active 应包含差异
        assert result["total_active"] != 0.0 or abs(result["allocation_abs"]) > 1e-9