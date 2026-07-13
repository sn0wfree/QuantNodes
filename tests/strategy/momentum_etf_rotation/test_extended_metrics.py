# coding=utf-8
"""Tests for extended_metrics.py (17 个业绩指标)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.extended_metrics import (
    extended_metrics,
    format_metrics_table,
)


def _make_nav(n_days: int = 500, seed: int = 42, drift: float = 0.0003) -> pd.Series:
    """合成单调上涨 nav."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n_days)
    prices = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.Series(prices, index=idx, name="nav")


class TestExtendedMetrics:
    def test_returns_17_keys(self) -> None:
        """应返回 17 个指标的 dict."""
        nav = _make_nav()
        m = extended_metrics(nav)
        expected_keys = {
            "ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar",
            "sortino", "downside_dev", "info_ratio", "win_rate",
            "profit_loss_ratio", "max_dd_duration", "calmar_avg_dd",
            "var_95", "cvar_95", "ann_turnover", "max_monthly_loss",
            "profit_months_ratio",
        }
        assert set(m.keys()) == expected_keys

    def test_synthetic_positive_returns(self) -> None:
        """合成单调上涨 nav 应有正年化 + 正夏普."""
        nav = _make_nav()
        m = extended_metrics(nav)
        assert m["ann_return"] > 0
        assert m["sharpe"] > 0
        assert m["calmar"] > 0

    def test_empty_nav(self) -> None:
        """空 nav 应返回空 dict."""
        m = extended_metrics(pd.Series(dtype=float))
        assert m == {}

    def test_short_nav(self) -> None:
        """少于 2 个数据点应返回空 dict."""
        nav = pd.Series([100.0], index=pd.bdate_range("2024-01-01", periods=1))
        m = extended_metrics(nav)
        assert m == {}

    def test_max_drawdown_negative(self) -> None:
        """max_drawdown 应 ≤ 0."""
        nav = _make_nav()
        m = extended_metrics(nav)
        assert m["max_drawdown"] <= 0

    def test_win_rate_between_0_and_1(self) -> None:
        """win_rate 应在 [0, 1] 区间."""
        nav = _make_nav()
        m = extended_metrics(nav)
        assert 0.0 <= m["win_rate"] <= 1.0
        assert 0.0 <= m["profit_months_ratio"] <= 1.0

    def test_var_cvar_consistent(self) -> None:
        """CVaR 应 ≤ VaR (更负)."""
        nav = _make_nav()
        m = extended_metrics(nav)
        # VaR/CVaR 都是负数, CVaR 应更负
        assert m["var_95"] <= 0
        assert m["cvar_95"] <= 0
        assert m["cvar_95"] <= m["var_95"]


class TestFormatMetricsTable:
    def test_returns_markdown_string(self) -> None:
        """应返回 markdown 格式字符串."""
        nav = _make_nav()
        m = extended_metrics(nav)
        md = format_metrics_table(m)
        assert isinstance(md, str)
        assert "| # | 指标 |" in md or "| # |指标 |" in md
        assert "Calmar" in md

    def test_includes_17_rows(self) -> None:
        """markdown 表应有 17 行数据 (不含 header)."""
        nav = _make_nav()
        m = extended_metrics(nav)
        md = format_metrics_table(m)
        # 数据行数 = 表行数 - header 行 - 分隔行
        rows = [line for line in md.split("\n") if line.startswith("|") and "|" in line[1:]]
        data_rows = [r for r in rows if not r.startswith("|---") and not r.startswith("| #")]
        assert len(data_rows) == 17

    def test_with_equal_weight_comparison(self) -> None:
        """提供等权对照时应生成差异列."""
        nav1 = _make_nav(seed=42)
        nav2 = _make_nav(seed=43, drift=0.0002)
        m1 = extended_metrics(nav1)
        m2 = extended_metrics(nav2)
        md = format_metrics_table(m1, m2)
        assert "等权" in md
        assert "差异" in md


class TestReportsAlignment:
    """与 reports/extended_metrics.json 字段对齐检查."""

    def test_keys_match_reports(self) -> None:
        """应包含 reports/extended_metrics.json 中的所有字段."""
        import json
        from pathlib import Path
        json_path = Path("reports/momentum_etf_rotation/extended_metrics.json")
        if not json_path.exists():
            pytest.skip("reports/extended_metrics.json 不存在")
        with open(json_path) as f:
            report_data = json.load(f)
        # 取任一 strategy 的指标列表
        first_key = next(iter(report_data))
        first_metrics = report_data[first_key]
        # 我们实现的字段应该是超集
        nav = _make_nav(n_days=1000)
        m = extended_metrics(nav)
        for key in first_metrics:
            assert key in m, f"Missing key: {key}"