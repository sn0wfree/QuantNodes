# coding: utf-8
"""K9 (2026-06-21): SamplePoolFilter + TradabilityFilter 详尽测试.

SamplePoolFilter: index='all'/'industry'='all' 路径.
TradabilityFilter: ST/停牌/涨跌停/IPO 各 flag 独立验证 + 组合.
"""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.nodes.sample_pool_filter_node import (
    SamplePoolFilterNode,
)
from QuantNodes.research.factor_test.nodes.tradability_filter_node import (
    TradabilityFilterNode,
)


def _make_ctx(n_days=10, n_stocks=5, with_st=False, with_suspend=False,
              with_ud=False, ipo_days_val=500):
    """构造 LoadData ctx."""
    dates = list(range(20260101, 20260101 + n_days))
    stocks = list(range(100001, 100001 + n_stocks))
    ones = np.ones((n_days, n_stocks), dtype=int)
    zeros = np.zeros((n_days, n_stocks), dtype=int)
    st = zeros.copy()
    suspend = zeros.copy()
    ud_limit = zeros.copy()
    if with_st:
        st[:, 0] = 1   # 第 0 只股票全部 ST
    if with_suspend:
        suspend[2:5, 1] = 1   # 第 1 只股票 day 2-4 停牌
    if with_ud:
        ud_limit[3, 2] = 1   # 第 2 只股票 day 3 涨停
    ipo_days = ones * ipo_days_val
    return {
        "LoadData": {
            "stklist": pd.DataFrame(stocks, columns=[0]),
            "trade_dt": pd.DataFrame(dates, columns=[0]),
            "st": pd.DataFrame(st, index=dates, columns=stocks),
            "suspend": pd.DataFrame(suspend, index=dates, columns=stocks),
            "ud_limit": pd.DataFrame(ud_limit, index=dates, columns=stocks),
            "ipo_days": pd.DataFrame(ipo_days, index=dates, columns=stocks),
            "_loader": None,
        }
    }


# ── SamplePoolFilterNode ──


class TestSamplePoolFilterNode:
    def test_all_returns_ones(self):
        """sample_index='all' + industry='all' → 全 1 矩阵."""
        ctx = _make_ctx()
        result = SamplePoolFilterNode(config={
            "sample_index": "all",
            "sample_industry": "all",
        }).execute(context=ctx)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (10, 5)
        # 全 1.0 (替换 0→NaN 后无 NaN, 因初始就是全 1)
        assert (result.values == 1.0).all()

    def test_unsupported_index_raises(self):
        """不支持的 index 名 → raise (需要 loader)."""
        ctx = _make_ctx()
        with pytest.raises(Exception):
            SamplePoolFilterNode(config={
                "sample_index": "UNKNOWN_INDEX",
                "sample_industry": "all",
            }).execute(context=ctx)

    def test_industry_without_loader_raises(self):
        """指定具体 industry 但无 loader → raise."""
        ctx = _make_ctx()
        with pytest.raises(Exception):
            SamplePoolFilterNode(config={
                "sample_index": "all",
                "sample_industry": "银行",
            }).execute(context=ctx)


# ── TradabilityFilterNode ──


class TestTradabilityFilterNode:
    def test_no_filters_all_tradable(self):
        """所有 flag 都关 → 全 1."""
        ctx = _make_ctx()
        result = TradabilityFilterNode(config={
            "tradable": {
                "no_st": False, "no_suspended": False,
                "no_up_down_limit": False, "min_ipo_days": 0,
            }
        }).execute(context=ctx)
        assert isinstance(result, pd.DataFrame)
        # 默认初始矩阵全 1
        assert (result.values == 1.0).all()

    def test_no_st_filters_st_stocks(self):
        """no_st=True 时, ST 股票应为 NaN."""
        ctx = _make_ctx(with_st=True)
        result = TradabilityFilterNode(config={
            "tradable": {"no_st": True},
        }).execute(context=ctx)
        # 第 0 只 (stocks[0]=100001) 全部为 NaN
        stk0 = result.iloc[:, 0]
        assert stk0.isna().all()
        # 其它列全 1
        stk_rest = result.iloc[:, 1:]
        assert (stk_rest.values == 1.0).all()

    def test_no_suspended_filters_suspended(self):
        """no_suspended=True, 停牌日 → NaN."""
        ctx = _make_ctx(with_suspend=True)
        result = TradabilityFilterNode(config={
            "tradable": {"no_suspended": True},
        }).execute(context=ctx)
        # 第 1 只股票 day 2-4 NaN
        assert result.iloc[2:5, 1].isna().all()
        # 同股票其它日期 == 1
        assert (result.iloc[0:2, 1].values == 1.0).all()
        assert (result.iloc[5:, 1].values == 1.0).all()

    def test_no_up_down_limit_filters(self):
        """no_up_down_limit=True, 涨跌停日 → NaN."""
        ctx = _make_ctx(with_ud=True)
        result = TradabilityFilterNode(config={
            "tradable": {"no_up_down_limit": True},
        }).execute(context=ctx)
        # 第 2 只股票 day 3 NaN
        assert pd.isna(result.iloc[3, 2])
        # 其它都是 1
        cnt_nan = result.isna().sum().sum()
        assert cnt_nan == 1

    def test_min_ipo_days_filters_new_listings(self):
        """min_ipo_days=600 时, ipo_days=500 全部 < 600 → 全 NaN."""
        ctx = _make_ctx(ipo_days_val=500)
        result = TradabilityFilterNode(config={
            "tradable": {"min_ipo_days": 600},
        }).execute(context=ctx)
        assert result.isna().all().all()

    def test_min_ipo_days_passthrough_when_satisfied(self):
        """min_ipo_days=100, ipo_days=500 满足 → 全 1."""
        ctx = _make_ctx(ipo_days_val=500)
        result = TradabilityFilterNode(config={
            "tradable": {"min_ipo_days": 100},
        }).execute(context=ctx)
        assert (result.values == 1.0).all()

    def test_combined_filters(self):
        """多 flag 组合: ST + 停牌 → NaN 累加 (≥3 个独立 NaN 来源)."""
        ctx = _make_ctx(with_st=True, with_suspend=True, with_ud=True)
        result = TradabilityFilterNode(config={
            "tradable": {
                "no_st": True, "no_suspended": True, "no_up_down_limit": True,
            },
        }).execute(context=ctx)
        # 至少 10 (stock 0 ST all day) + 3 (stock 1 suspend) + 1 (stock 2 ud)
        nan_count = result.isna().sum().sum()
        assert nan_count >= 14

    def test_output_shape(self):
        """输出 shape == (n_dt, n_stk)."""
        ctx = _make_ctx(n_days=20, n_stocks=8)
        result = TradabilityFilterNode(config={
            "tradable": {"no_st": True},
        }).execute(context=ctx)
        assert result.shape == (20, 8)

    def test_output_index_is_trade_dt(self):
        """输出 index == trade_dt.iloc[:, 0]."""
        ctx = _make_ctx()
        result = TradabilityFilterNode(config={
            "tradable": {"no_st": True},
        }).execute(context=ctx)
        expected_idx = ctx["LoadData"]["trade_dt"].iloc[:, 0].values
        assert (result.index.values == expected_idx).all()

    def test_output_columns_is_stklist(self):
        """输出 columns == stklist.iloc[:, 0]."""
        ctx = _make_ctx()
        result = TradabilityFilterNode(config={
            "tradable": {"no_st": True},
        }).execute(context=ctx)
        expected_cols = ctx["LoadData"]["stklist"].iloc[:, 0].values
        assert (result.columns.values == expected_cols).all()
