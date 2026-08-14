# coding=utf-8
"""期权移植模块测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.utils import (
    cal_pnl,
    cal_portfolio_greek,
    find_third_friday,
    required_share,
    ym_to_expire_day,
)
from QuantNodes.strategy.cta.options.contract_marking import (
    compute_f_k_diff,
    filter_by_main_mark,
    mark_main_contracts,
    mark_otm,
)
from QuantNodes.strategy.cta.options.match_engine import DequeMatcher
from QuantNodes.strategy.cta.options.optimizer import OptBundle, OptionPortfolioWithDT
from QuantNodes.strategy.cta.options.backtest_bridge import (
    compute_pnl_from_transactions,
    status_to_transactions,
)


class TestUtils:
    def test_required_share(self):
        # hedge=5M, delta=0.5, f=5500 → 5M/0.5/5500/100 ≈ 18.18
        result = required_share(5_000_000, 0.5, 5500)
        assert abs(result - 18.18) < 0.01

    def test_cal_pnl_call_itm(self):
        # Call, k=5400, f=5500, weight=1 → PnL = (5500-5400)*1 = 100
        selected = pd.DataFrame({"k": [5400], "cp": ["C"], "weight": [1]})
        assert cal_pnl(5500, selected) == 100

    def test_cal_pnl_call_otm(self):
        # Call, k=5600, f=5500 → OTM, PnL = 0
        selected = pd.DataFrame({"k": [5600], "cp": ["C"], "weight": [1]})
        assert cal_pnl(5500, selected) == 0

    def test_cal_pnl_put_itm(self):
        # Put, k=5600, f=5500 → ITM, PnL = (5600-5500)*1 = 100
        selected = pd.DataFrame({"k": [5600], "cp": ["P"], "weight": [1]})
        assert cal_pnl(5500, selected) == 100

    def test_cal_pnl_multi(self):
        selected = pd.DataFrame({
            "k": [5400, 5600],
            "cp": ["C", "P"],
            "weight": [1, 1],
        })
        # Call ITM: 100, Put ITM: 100 → total 200
        assert cal_pnl(5500, selected) == 200

    def test_find_third_friday(self):
        # 2025-01: 第三个周五是 1/17
        result = find_third_friday(2025, 1)
        assert result.day == 17
        assert result.month == 1

    def test_find_third_friday_march(self):
        # 2025-03: 第三个周五是 3/14
        result = find_third_friday(2025, 3)
        assert result.day == 14


class TestContractMarking:
    def _make_data(self):
        return pd.DataFrame({
            "dt": ["2025-01-10"] * 6,
            "ym": ["2503"] * 6,
            "cp": ["C", "C", "C", "P", "P", "P"],
            "k": [5300, 5400, 5500, 5300, 5400, 5500],
            "f": [5450] * 6,
            "fee": [100, 50, 20, 20, 50, 100],
        })

    def test_compute_f_k_diff(self):
        df = self._make_data()
        result = compute_f_k_diff(df)
        assert "f_k_diff" in result.columns
        assert result["f_k_diff"].iloc[0] == 150  # 5450-5300

    def test_mark_main_contracts(self):
        df = self._make_data()
        df = compute_f_k_diff(df)
        result = mark_main_contracts(df)
        assert "c1" in result.columns
        assert "main_mark" in result.columns
        # k=5400 is closest to f=5450, should get highest main_mark
        row_5400 = result[result["k"] == 5400].iloc[0]
        row_5300 = result[result["k"] == 5300].iloc[0]
        assert row_5400["main_mark"] >= row_5300["main_mark"]

    def test_mark_otm(self):
        df = self._make_data()
        result = mark_otm(df)
        assert "OTM" in result.columns
        # Call k=5500, f=5450 → OTM (k >= f for call)
        call_5500 = result[(result["cp"] == "C") & (result["k"] == 5500)].iloc[0]
        assert call_5500["OTM"] == True
        # Put k=5300, f=5450 → OTM (f >= k for put)
        put_5300 = result[(result["cp"] == "P") & (result["k"] == 5300)].iloc[0]
        assert put_5300["OTM"] == True

    def test_filter_by_main_mark(self):
        df = self._make_data()
        df = compute_f_k_diff(df)
        df = mark_main_contracts(df)
        filtered = filter_by_main_mark(df, min_mark=3)
        assert len(filtered) <= len(df)


class TestDequeMatcher:
    def test_fifo_match(self):
        transactions = pd.DataFrame({
            "contract_code": ["A"] * 3,
            "手数": [10, 5, 5],
            "成交均价": [100, 110, 120],
            "trade_type": [1, 1, -1],  # buy, buy, sell
            "日期": [1, 2, 3],
        })
        avg_price, matched = DequeMatcher.single_contract_recurs_match(
            transactions, share_col="手数", price_col="成交均价",
            trade_type="trade_type", dt_col="日期",
        )
        assert avg_price is not None
        assert len(matched) == 1  # 1 sell matched with 1 buy


class TestOptimizer:
    def test_optbundle_delta(self):
        Delta = pd.DataFrame({"A": [0.5], "B": [-0.3]}, index=[5500])
        weights = np.array([1, -1])
        result = OptBundle.delta(weights, Delta, 5500)
        # (0.5 - (-0.3)) * 100 * 5500 = 0.8 * 550000 = 440000
        assert abs(result - 440000) < 1


class TestBacktestBridge:
    def test_status_to_transactions(self):
        dd = pd.DataFrame({
            "dt": ["2025-01-10"],
            "contract_code": ["rb2503C5400"],
            "fee": [50.0],
            "weight": [2],
        })
        result = status_to_transactions(dd)
        assert len(result) == 1
        assert result.iloc[0]["买卖"] == "买"
        assert result.iloc[0]["手数"] == 2
        assert result.iloc[0]["开平"] == "开"
