# coding: utf-8
"""Phase 3.2 M11: ANNUAL_DAYS 注入 evaluation()"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.utils.performance_metrics import evaluation


class TestAnnualDaysParameter:
    """M11: evaluation() 接受 annual_days 参数，默认全局 ANNUAL_DAYS=250"""

    @staticmethod
    def _generate_dates(start_year: int, n_days: int) -> list[int]:
        """Generate consecutive calendar dates as 8-digit int (YYYYMMDD)."""
        dates = pd.date_range(
            start=f"{start_year}-01-01",
            periods=n_days,
            freq='D',
        )
        # 转换为 int YYYYMMDD
        return [int(d.strftime('%Y%m%d')) for d in dates]

    def test_default_250(self):
        """默认 250 (A股 trading days), use 2026 calendar year (~252 days, close enough)"""
        # 构造 2026 年完整日历
        dates = self._generate_dates(2026, 365)
        # take first 250 trading-like days (approx)
        dates = dates[:250]
        n_days = len(dates)
        net = pd.Series(np.linspace(1, 1.1, n_days), index=dates)
        adj_dates = dates[::20]  # 每月调仓 (~12 次/year)
        res = evaluation(net, adj_dates)
        # accum return 0.1 (1 → 1.1) → annual_rt = (0.1 / 1y) * 250 = 0.1
        annual_rt = res[res['Year'] == 'all']['AnnualRt'].iloc[0]
        assert abs(annual_rt - 0.1) < 0.015  # allow small diff

    def test_custom_252(self):
        """自定义 252 (美股), 2026 calendar year"""
        dates = self._generate_dates(2026, 365)
        dates = dates[:252]
        n_days = len(dates)
        net = pd.Series(np.linspace(1, 1.1008, n_days), index=dates)
        adj_dates = dates[::20]
        res = evaluation(net, adj_dates, annual_days=252)
        annual_rt = res[res['Year'] == 'all']['AnnualRt'].iloc[0]
        # accum return 0.1008 over 1y → annual_rt = 0.1008
        assert abs(annual_rt - 0.1008) < 0.002

    def test_custom_365(self):
        """自定义 365 (24h 加密货币 market), 2026 full calendar"""
        dates = self._generate_dates(2026, 365)
        n_days = len(dates)
        net = pd.Series(np.linspace(1, 1.1, n_days), index=dates)
        adj_dates = dates[::30]
        res = evaluation(net, adj_dates, annual_days=365)
        annual_rt = res[res['Year'] == 'all']['AnnualRt'].iloc[0]
        assert abs(annual_rt - 0.1) < 0.015

    def test_annual_days_in_yearly_calculation(self):
        """分年计算也使用自定义 annual_days"""
        # 2025-2026 full calendar → 365 × 2 = 730 calendar days
        dates_2025 = self._generate_dates(2025, 365)
        dates_2026 = self._generate_dates(2026, 365)
        dates = dates_2025 + dates_2026
        n_days = len(dates)
        net = pd.Series(np.linspace(1, 1.2016, n_days), index=dates)
        adj_dates = dates[::21]  # weekly rebalance
        res = evaluation(net, adj_dates, annual_days=252)
        # total accum 0.2016 over 2 full calendar years (730 days), annualized with annual_days=252:
        #   adj_cycle = 730 / (len(dates[::21]) - 1) ≈ 730 / 34 ≈ 21.47
        #   annualized = 0.2016 / 2 → 0.1008, expected ~0.1008 with tolerance for discrete sampling
        annual_all = res[res['Year'] == 'all']['AnnualRt'].iloc[0]
        assert abs(annual_all - 0.1008) < 0.05
        # each year ~ 0.1 annualized
        for y in [2025, 2026]:
            row = res[res['Year'] == y]
            assert abs(row['AnnualRt'].iloc[0] - 0.1008) < 0.05
