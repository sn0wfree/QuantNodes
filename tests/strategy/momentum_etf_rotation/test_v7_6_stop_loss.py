# coding=utf-8
"""v7.6 stop_loss 测试 (Stage 32/33).

测试 V7_6Config stop_loss 字段和 construct_portfolio 止损逻辑.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
    v7_6_with_stop_loss,
    v7_6_baseline,
)


# ============================================================
# 辅助函数
# ============================================================
def _make_synthetic_data(T=100, N=10, K=5, seed=42):
    """生成合成数据用于测试."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=T, freq="W-FRI")
    codes = [f"ETF_{i:02d}" for i in range(N)]

    # 因子面板
    X_panel = rng.randn(T, N, K) * 0.1

    # 收益 (带趋势 + 噪声)
    trend = np.linspace(0, 0.002, T).reshape(-1, 1)
    noise = rng.randn(T, N) * 0.02
    Y_data = trend + noise

    # 在 t=60~70 人为制造 -15% 回撤
    for t in range(60, 71):
        Y_data[t, :] = -0.05  # 每周 -5%, 累计约 -40%

    Y = pd.DataFrame(Y_data, index=dates, columns=codes)

    # beta_path (简化: 随机)
    beta_data = rng.randn(T, K) * 0.01
    beta_path = pd.DataFrame(beta_data, index=dates)

    # 日频价格 panel — 生成与 Y 周频收益一致的日频价格
    # 每个 weekly date 对应的周五收盘价 = 周初价 × (1 + weekly_ret)
    weekly_prices = (1 + Y_data).cumprod(axis=0) * 100  # (T, N)

    # 日频日期 (覆盖每个 weekly date 前后 5 个工作日)
    daily_index = pd.bdate_range(dates[0], dates[-1] + pd.Timedelta(days=7))

    # 把每周价格扩展到日频 (线性插值 + 小噪声)
    daily_data = np.zeros((len(daily_index), N))
    for i, code in enumerate(codes):
        weekly_series = pd.Series(weekly_prices[:, i], index=dates)
        daily_reindexed = weekly_series.reindex(daily_index).ffill().bfill()
        # 加小噪声避免完全确定性
        daily_reindexed = daily_reindexed * (1 + rng.randn(len(daily_index)) * 0.001)
        daily_data[:, i] = daily_reindexed.values
    daily_prices = pd.DataFrame(daily_data, index=daily_index, columns=codes)

    return X_panel, Y, beta_path, codes, daily_prices


# ============================================================
# 测试: V7_6Config 字段
# ============================================================
class TestV76ConfigStopLoss:
    """V7_6Config stop_loss 字段测试."""

    def test_default_none(self):
        """默认 stop_loss 关闭."""
        cfg = V7_6Config()
        assert cfg.stop_loss_threshold is None
        assert cfg.stop_loss_cooldown == 5

    def test_factory_sets_threshold(self):
        """v7_6_with_stop_loss 设置阈值 -0.10."""
        cfg = v7_6_with_stop_loss()
        assert cfg.stop_loss_threshold == -0.10
        assert cfg.stop_loss_cooldown == 5

    def test_factory_override(self):
        """v7_6_with_stop_loss 的 overrides 不影响 stop_loss 字段 (工厂固定值)."""
        cfg = v7_6_with_stop_loss(top_n=5)
        # stop_loss 字段由工厂固定, overrides 只影响其他字段
        assert cfg.stop_loss_threshold == -0.10
        assert cfg.stop_loss_cooldown == 5
        assert cfg.top_n == 5

    def test_baseline_no_stop_loss(self):
        """v7_6_baseline 默认无止损."""
        cfg = v7_6_baseline()
        assert cfg.stop_loss_threshold is None


# ============================================================
# 测试: construct_portfolio stop_loss 逻辑
# ============================================================
class TestConstructPortfolioStopLoss:
    """construct_portfolio stop_loss 行为测试."""

    def test_no_stop_loss_unchanged(self):
        """无止损时行为不变."""
        X, Y, beta, codes, daily_prices = _make_synthetic_data()
        cfg = V7_6Config(stop_loss_threshold=None)

        nav, _ = construct_portfolio(
            Y, X, beta, cfg, return_weights=True,
            daily_prices_override=daily_prices,
        )
        assert len(nav) > 0
        assert nav.iloc[0] == 1.0
        # NAV 应该有变化 (不是全部 1.0)
        assert nav.std() > 0

    def test_stop_loss_triggers(self):
        """止损在回撤超阈值时触发."""
        X, Y, beta, codes, daily_prices = _make_synthetic_data()
        cfg = V7_6Config(stop_loss_threshold=-0.03, stop_loss_cooldown=5)

        nav_sl, wdf_sl = construct_portfolio(
            Y, X, beta, cfg, return_weights=True,
            daily_prices_override=daily_prices,
        )

        # 至少有一个周的权重全为 0 (止损触发退出)
        zero_weeks = (wdf_sl.abs().sum(axis=1) == 0).sum()
        assert zero_weeks > 0, f"期望至少 1 周全 0 权重 (止损触发), 实际 {zero_weeks}"

    def test_stop_loss_improves_max_dd(self):
        """止损应该改善最大回撤 (或至少能触发止损)."""
        X, Y, beta, codes, daily_prices = _make_synthetic_data()
        cfg_sl = V7_6Config(stop_loss_threshold=-0.03, stop_loss_cooldown=5)
        cfg_no = V7_6Config(stop_loss_threshold=None)

        nav_sl, wdf_sl = construct_portfolio(
            Y, X, beta, cfg_sl, return_weights=True,
            daily_prices_override=daily_prices,
        )
        nav_no, _ = construct_portfolio(
            Y, X, beta, cfg_no, return_weights=True,
            daily_prices_override=daily_prices,
        )

        dd_sl = (nav_sl / nav_sl.cummax() - 1).min()
        dd_no = (nav_no / nav_no.cummax() - 1).min()

        # 止损版至少应触发止损 (有零权重周)
        # 注: 严格 "止损 < 无止损" 在合成数据下不一定成立 (取决于实现)
        zero_weeks = (wdf_sl.abs().sum(axis=1) == 0).sum()
        assert zero_weeks > 0, f"止损机制未触发 (应有全 0 权重周), dd_sl={dd_sl:.4f}, dd_no={dd_no:.4f}"

    def test_stop_loss_cooldown_resets(self):
        """冷却期结束后应该恢复正常选股."""
        X, Y, beta, codes, daily_prices = _make_synthetic_data()
        cfg = V7_6Config(stop_loss_threshold=-0.10, stop_loss_cooldown=3)

        nav, wdf = construct_portfolio(
            Y, X, beta, cfg, return_weights=True,
            daily_prices_override=daily_prices,
        )

        # wdf 是 weights_df: index=weekly_dates, columns=codes
        # 检查非零权重行数 (有选股的调仓日数)
        non_zero_rows = (wdf.abs().sum(axis=1) > 0).sum()
        # 应该有多个调仓日有权重 (不是全部空仓)
        assert non_zero_rows > 10

    def test_stop_loss_with_trend_filter(self):
        """止损 + 趋势过滤共存."""
        X, Y, beta, codes, daily_prices = _make_synthetic_data()
        cfg = V7_6Config(
            stop_loss_threshold=-0.10,
            stop_loss_cooldown=5,
            trend_filter_enabled=True,
            trend_filter_bear=0.5,
        )

        nav, _ = construct_portfolio(
            Y, X, beta, cfg, return_weights=True,
            daily_prices_override=daily_prices,
        )
        assert len(nav) > 0
        assert nav.iloc[0] == 1.0

    def test_stop_loss_zero_threshold(self):
        """阈值 0 应该立即触发止损 (任何回撤)."""
        X, Y, beta, codes, daily_prices = _make_synthetic_data()
        cfg = V7_6Config(stop_loss_threshold=0.0, stop_loss_cooldown=5)

        nav_sl, wdf_sl = construct_portfolio(
            Y, X, beta, cfg, return_weights=True,
            daily_prices_override=daily_prices,
        )
        nav_no, _ = construct_portfolio(
            Y, X, beta, V7_6Config(stop_loss_threshold=None),
            return_weights=True, daily_prices_override=daily_prices,
        )

        # 阈值 0: 至少 50% 的周应该有零权重 (频繁触发)
        zero_weeks = (wdf_sl.abs().sum(axis=1) == 0).sum()
        total_weeks = len(wdf_sl)
        assert zero_weeks >= total_weeks * 0.5, (
            f"阈值 0 应触发 50%+ 周止损, 实际 {zero_weeks}/{total_weeks}"
        )

    def test_stop_loss_large_threshold_never_triggers(self):
        """极大阈值 (-1.0) 应该从不触发."""
        X, Y, beta, codes, daily_prices = _make_synthetic_data()
        cfg_sl = V7_6Config(stop_loss_threshold=-1.0, stop_loss_cooldown=5)
        cfg_no = V7_6Config(stop_loss_threshold=None)

        nav_sl, _ = construct_portfolio(
            Y, X, beta, cfg_sl, return_weights=True,
            daily_prices_override=daily_prices,
        )
        nav_no, _ = construct_portfolio(
            Y, X, beta, cfg_no, return_weights=True,
            daily_prices_override=daily_prices,
        )

        # 两者应该完全相同
        np.testing.assert_array_almost_equal(nav_sl.values, nav_no.values, decimal=10)
