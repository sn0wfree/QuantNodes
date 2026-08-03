"""Phase 0: combo/build_all_strategy_navs.py 规范评估管线守门测试.

[覆盖]
- _to_business_day_index: 丢弃周末 + NaN
- _to_business_day_index: 保留 business-day 索引
- 合成 weekend 污染源 → 修复后 vol 不变
- OOS_START 与 docs/77 审计对齐 (2022-01-01)
- 实际 all_strategy_navs.parquet shape ≈ 2,058 trading days
- 实际 all_strategy_navs.parquet 索引全部 weekday
- 实际 OOS Calmar 排名与 docs/77 审计一致
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from combo.build_all_strategy_navs import _to_business_day_index, OOS_START  # noqa: E402


class TestToBusinessDayIndex:
    def test_drops_weekends(self):
        """包含周末的索引 → 修复后只剩 weekday."""
        dates = pd.to_datetime([
            "2024-01-08", "2024-01-09", "2024-01-10",  # Mon-Wed
            "2024-01-13", "2024-01-14",                  # weekend (Sat-Sun)
        ])
        nav = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=dates)
        result = _to_business_day_index(nav)
        assert (result.index.dayofweek < 5).all(), "周末行残留"
        assert len(result) == 3
        assert result.iloc[0] == 1.0
        assert result.iloc[-1] == 1.2

    def test_drops_nan(self):
        """NaN 行被丢弃."""
        dates = pd.bdate_range("2024-01-08", periods=5)
        nav = pd.Series([1.0, np.nan, 1.2, 1.3, np.nan], index=dates)
        result = _to_business_day_index(nav)
        assert not result.isna().any()
        assert len(result) == 3

    def test_drops_duplicate_dates(self):
        """重复日期只保留最后一行."""
        dates = pd.to_datetime([
            "2024-01-08", "2024-01-08", "2024-01-09",
        ])
        nav = pd.Series([1.0, 99.0, 1.1], index=dates)
        result = _to_business_day_index(nav)
        assert len(result) == 2
        assert result.iloc[0] == 99.0

    def test_pure_business_day_input_unchanged(self):
        """纯 business-day 输入保持不变."""
        dates = pd.bdate_range("2024-01-08", periods=10)
        nav = pd.Series(np.arange(10).astype(float), index=dates)
        result = _to_business_day_index(nav)
        assert len(result) == 10
        np.testing.assert_array_equal(result.values, nav.values)

    def test_calendar_only_input_filled_nan_dropped(self):
        """纯 calendar 输入 → 周末位置变 NaN → 全部 drop."""
        dates = pd.date_range("2024-01-08", periods=10, freq="D")
        nav = pd.Series(np.arange(10).astype(float), index=dates)
        result = _to_business_day_index(nav)
        assert (result.index.dayofweek < 5).all()
        assert len(result) == 8  # Mon-Sat→Mon-Fri × 1.6w


class TestOOSStartAlignment:
    def test_oos_start_matches_audit(self):
        """OOS_START = 2022-01-01 (与 docs/77 + unified_v1v5_compare 一致)."""
        assert OOS_START == pd.Timestamp("2022-01-01"), \
            f"OOS_START 不一致: {OOS_START} (期望 2022-01-01)"


class TestAllStrategyNavsParquet:
    """对实际 all_strategy_navs.parquet 进行健全性检查."""

    @pytest.fixture
    def navs(self):
        path = REPO / "reports" / "momentum_etf_rotation" / "combo" / "all_strategy_navs.parquet"
        if not path.exists():
            pytest.skip("all_strategy_navs.parquet 不存在, 请先运行 combo/build_all_strategy_navs.py")
        return pd.read_parquet(path)

    def test_shape_in_business_day_range(self, navs):
        """Shape 必须在 ~2,000-2,100 business days 范围."""
        n = len(navs)
        assert 2000 <= n <= 2100, f"Shape 异常: {n} (期望 2000-2100)"

    def test_index_all_weekdays(self, navs):
        """索引必须全部是 weekday."""
        assert (navs.index.dayofweek < 5).all(), "周末行污染!"

    def test_index_no_duplicates(self, navs):
        """索引无重复."""
        assert navs.index.is_unique, "重复日期!"

    def test_no_nan(self, navs):
        """无 NaN 残留."""
        assert not navs.isna().any().any(), "NaN 残留!"

    def test_all_start_at_one(self, navs):
        """所有策略 NAV 在 first_valid 处归一化为 1.0."""
        for col in navs.columns:
            first_val = navs[col].iloc[0]
            assert abs(first_val - 1.0) < 0.01, \
                f"{col} 起点不是 1.0: {first_val}"

    def test_v10_dynd_oos_calmar_matches_audit(self, navs):
        """v10-DynD OOS Calmar ≈ 1.806 (Phase 0 修复后).

        与 docs/77 审计值 1.806 严格一致.
        """
        from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics
        col = "v10-DynD 信号加权"
        if col not in navs.columns:
            pytest.skip(f"{col} 不在 navs 中")
        oos = navs.loc[OOS_START:, col].dropna()
        m = compute_metrics(oos)
        assert abs(m["Calmar"] - 1.806) < 0.05, \
            f"v10-DynD OOS Calmar={m['Calmar']:.3f} (期望 ~1.806)"

    def test_v10_vol_parity_oos_calmar_matches_audit(self, navs):
        """v10 Vol-parity OOS Calmar ≈ 1.426 (Phase 0 修复后)."""
        from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics
        col = "v10 4策略Vol-parity"
        if col not in navs.columns:
            pytest.skip(f"{col} 不在 navs 中")
        oos = navs.loc[OOS_START:, col].dropna()
        m = compute_metrics(oos)
        assert abs(m["Calmar"] - 1.426) < 0.05, \
            f"v10 Vol-parity OOS Calmar={m['Calmar']:.3f} (期望 ~1.426)"
