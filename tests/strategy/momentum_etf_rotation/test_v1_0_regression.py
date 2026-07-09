# coding=utf-8
"""v1.0 回归测试 - 确保版本锁定的配置可重现.

任何对 v1.0 配置的修改都必须先更新 strategy_versions.py,
然后更新本测试的预期值 (这是契约).
"""
from __future__ import annotations

import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation import (
    v1_0, v0_0_baseline, v0_1_vt_only, v0_4_hybrid, VERSIONS, get_version,
    run_rotation_backtest, BacktestConfig, DEFAULT_POOL, performance_metrics,
)


# v1.0 锁定配置预期值 (变更时需同步更新)
V1_0_LOOKBACK = 90
V1_0_TOP_N = 10
V1_0_MOMENTUM_TYPE = "hybrid"
V1_0_MOMENTUM_FUSED_WEIGHT = 0.5
V1_0_VT_ENABLED = True
V1_0_VT_TARGET = 0.15
V1_0_COST_ENABLED = True
V1_0_COST_COMM_BP = 5.0
V1_0_COST_SLIP_BP = 10.0


class TestV10ConfigLock:
    """v1.0 配置锁定的契约测试."""

    def test_v1_0_lookback(self):
        cfg = v1_0()
        assert cfg.lookback == V1_0_LOOKBACK

    def test_v1_0_top_n(self):
        cfg = v1_0()
        assert cfg.top_n == V1_0_TOP_N

    def test_v1_0_momentum_type(self):
        cfg = v1_0()
        assert cfg.momentum_type == V1_0_MOMENTUM_TYPE

    def test_v1_0_momentum_fused_weight(self):
        cfg = v1_0()
        assert cfg.momentum_fused_weight == V1_0_MOMENTUM_FUSED_WEIGHT

    def test_v1_0_vt_enabled(self):
        cfg = v1_0()
        assert cfg.vol_targeting.enabled == V1_0_VT_ENABLED

    def test_v1_0_vt_target(self):
        cfg = v1_0()
        assert cfg.vol_targeting.target_vol == V1_0_VT_TARGET

    def test_v1_0_cost_enabled(self):
        cfg = v1_0()
        assert cfg.cost_model.enabled == V1_0_COST_ENABLED

    def test_v1_0_cost_commission(self):
        cfg = v1_0()
        assert cfg.cost_model.commission_bp == V1_0_COST_COMM_BP

    def test_v1_0_cost_slippage(self):
        cfg = v1_0()
        assert cfg.cost_model.slippage_bp == V1_0_COST_SLIP_BP

    def test_v1_0_returns_new_instance(self):
        """每次调用应返回新实例, 避免修改污染."""
        cfg1 = v1_0()
        cfg2 = v1_0()
        assert cfg1 is not cfg2
        # 修改 cfg1 不应影响 cfg2
        cfg1.momentum_type = "price"
        cfg3 = v1_0()
        assert cfg3.momentum_type == V1_0_MOMENTUM_TYPE


class TestVersionRegistry:
    """版本注册表测试."""

    def test_get_version_default(self):
        """默认应返回 LATEST."""
        cfg = get_version()
        cfg_latest = v1_0()
        assert cfg.momentum_type == cfg_latest.momentum_type
        assert cfg.vol_targeting.enabled == cfg_latest.vol_targeting.enabled

    def test_get_version_specific(self):
        cfg = get_version("0.0")
        assert cfg.momentum_type == "price"  # baseline
        assert not cfg.vol_targeting.enabled

    def test_get_version_unknown_raises(self):
        with pytest.raises(ValueError, match="未知版本"):
            get_version("99.0")

    def test_all_versions_in_registry(self):
        """所有版本函数都应在 VERSIONS dict 中."""
        assert "1.0" in VERSIONS
        assert "0.4" in VERSIONS
        assert "0.3" in VERSIONS
        assert "0.2" in VERSIONS
        assert "0.1" in VERSIONS
        assert "0.0" in VERSIONS

    def test_version_registry_callable(self):
        """VERSIONS dict 中每个值都应可调用."""
        for ver, factory in VERSIONS.items():
            cfg = factory()
            assert cfg.lookback > 0


class TestV10Regression:
    """v1.0 真实数据回测 - 防止指标意外退化.

    指标基线 (2019-2026):
        Calmar ~1.60, DD ~-4%, Ann ~6%

    阈值设宽 (允许小幅波动), 主要检测灾难性退化.
    """

    @pytest.fixture(scope="class")
    def panel(self):
        from pathlib import Path
        path = Path("data/real/etf_nav_2018-01-01_2026-06-30.parquet")
        if not path.exists():
            pytest.skip("真实数据未找到, 跳过回归测试")
        panel = pd.read_parquet(path)
        panel.index = pd.to_datetime(panel.index).tz_localize(None)
        return panel.loc["2019-01-01":]

    def test_v1_0_calmar_not_below_1_0(self, panel):
        """v1.0 Calmar 应 > 1.0 (设置宽阈值防回归)."""
        cfg = v1_0()
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        assert m["calmar"] > 1.0, f"v1.0 Calmar 退化: {m['calmar']:.2f}"

    def test_v1_0_dd_not_below_minus_10pct(self, panel):
        """v1.0 DD 应 > -10%."""
        cfg = v1_0()
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        assert m["max_drawdown"] > -0.10, f"v1.0 DD 退化: {m['max_drawdown']:.2%}"

    def test_v0_0_baseline_calmar(self, panel):
        """v0.0 baseline Calmar 应 > 0.5 (宽松阈值防回归)."""
        cfg = v0_0_baseline()
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        assert m["calmar"] > 0.5, f"v0.0 Calmar 退化: {m['calmar']:.2f}"

    def test_v0_1_vt_better_than_v0_0(self, panel):
        """v0.1 (有VT) 应优于 v0.0 (无VT)."""
        r0 = run_rotation_backtest(panel, DEFAULT_POOL,
                                  BacktestConfig(rotation=v0_0_baseline()))
        r1 = run_rotation_backtest(panel, DEFAULT_POOL,
                                  BacktestConfig(rotation=v0_1_vt_only()))
        m0 = performance_metrics(r0.nav)
        m1 = performance_metrics(r1.nav)
        assert m1["calmar"] > m0["calmar"], (
            f"v0.1 ({m1['calmar']:.2f}) 应 > v0.0 ({m0['calmar']:.2f})"
        )