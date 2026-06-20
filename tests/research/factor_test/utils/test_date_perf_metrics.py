# coding: utf-8
"""Unit tests: date_utils + performance_metrics.

历史来源: 迁移自 ``QuantNodes/research/factor_test/tests/test_utils.py`` (C2 收敛).
"""

import pandas as pd

from QuantNodes.research.factor_test.utils.date_utils import (
    valid_date,
    resample_trade_date,
    get_adjust_date,
    datenum_to_datetime,
    datetime_to_datenum,
)
from QuantNodes.research.factor_test.utils.performance_metrics import (
    calc_max_drawdown,
    evaluation,
    cal_net_simple,
)


class TestDateUtils:

    def test_valid_date_int(self):
        df = pd.DataFrame([20170101, 20170102])
        assert valid_date(df) is True

    def test_valid_date_str(self):
        df = pd.DataFrame(['2017-01-01'])
        assert valid_date(df) is False

    def test_valid_date_series(self):
        s = pd.Series([20170101, 20170102])
        assert valid_date(s) is True

    def test_resample_monthly_end(self):
        trade_dt = pd.DataFrame([
            20170103, 20170104, 20170105,
            20170131, 20170201, 20170202, 20170228,
            20170301, 20170302, 20170331,
        ])
        result = resample_trade_date(trade_dt, ('M', 'end'))
        assert len(result) == 3
        assert result.iloc[0, 0] == 20170131

    def test_get_adjust_date_monthly(self):
        trade_dt = pd.DataFrame(range(20170101, 20170110))
        result = get_adjust_date(trade_dt, 20170101, 20170109, ('M', 'end'))
        assert len(result) >= 1

    def test_datenum_conversion(self):
        df = pd.DataFrame([20170101, 20170102])
        dt = datenum_to_datetime(df)
        back = datetime_to_datenum(dt)
        assert back.iloc[0, 0] == 20170101


class TestPerformanceMetrics:

    def test_calc_max_drawdown_simple(self):
        net = pd.Series([1.0, 1.1, 1.05, 1.2, 1.15, 1.3])
        result = calc_max_drawdown(net)
        assert 'MDD' in result
        assert 0 < result['MDD'] < 1

    def test_calc_max_drawdown_no_drawdown(self):
        net = pd.Series([1.0, 1.1, 1.2, 1.3])
        result = calc_max_drawdown(net)
        assert result['MDD'] == 0

    def test_evaluation_basic(self):
        dates = [20170101, 20170201, 20170301]
        net = pd.Series([1.0, 1.05, 1.1], index=dates)
        result = evaluation(net, dates)
        assert isinstance(result, pd.DataFrame)
        assert 'AnnualRt' in result.columns
        assert 'SR' in result.columns
        assert 'MDD' in result.columns

    def test_cal_net_simple(self):
        net = pd.Series([1.0, 1.02, 1.04, 1.03, 1.06], index=range(5))
        adj_dates = [1, 3]
        result = cal_net_simple(net, adj_dates)
        assert isinstance(result, pd.Series)
        assert len(result) == 5
