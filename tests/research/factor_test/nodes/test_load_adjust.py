# coding: utf-8
"""K7+K8 (2026-06-21): LoadDataNode + AdjustDateNode 详尽测试.

LoadDataNode: data_path 校验, H5 round-trip, axis 载入.
AdjustDateNode: M/W/D 频率, beg/end 边界, 多 mode.
"""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
from QuantNodes.research.factor_test.nodes.load_data_node import LoadDataNode


# ── LoadDataNode 验证 ──


class TestLoadDataNode:
    def test_empty_data_path_raises(self):
        """空 data_path 抛 raise (P-2 启动报错)."""
        n = LoadDataNode(config={"data_path": ""})
        with pytest.raises(Exception):
            n.execute(context={})

    def test_missing_data_path_raises(self):
        """缺失 data_path 抛 raise."""
        with pytest.raises(Exception):
            LoadDataNode(config={}).execute(context={})

    def test_invalid_data_path_raises(self):
        """不存在的路径 → raise."""
        n = LoadDataNode(config={"data_path": "/nonexistent/path/xyz"})
        with pytest.raises(Exception):
            n.execute(context={})

    def test_load_h5_round_trip(self, tmp_path):
        """H5 round-trip: 写入 stk_daily.h5 后能读出 cp 面板."""
        n_days, n_stocks = 30, 10
        dates = [
            int(d.strftime("%Y%m%d"))
            for d in pd.bdate_range("2026-01-04", periods=n_days)
        ]
        stocks = list(range(100001, 100001 + n_stocks))
        cp = pd.DataFrame(
            np.cumprod(
                1 + np.random.RandomState(0).randn(n_days, n_stocks) * 0.01,
                axis=0,
            ) * 100,
            index=dates,
            columns=stocks,
        )
        h5_path = tmp_path / "stk_daily.h5"
        cp.to_hdf(h5_path, key="cp", mode="w")
        pd.DataFrame(stocks, columns=[0]).to_hdf(h5_path, key="stklist", mode="a")
        pd.DataFrame(dates, columns=[0]).to_hdf(h5_path, key="trade_dt", mode="a")

        n = LoadDataNode(config={
            "data_path": str(tmp_path) + "/",
            "load_keys": [],
            "factor": None,
        })
        result = n.execute(context={})
        assert "price" in result
        assert "_loader" in result
        assert "stklist" in result
        assert "trade_dt" in result
        pd.testing.assert_frame_equal(
            result["price"], cp, check_dtype=False
        )


# ── AdjustDateNode 详尽测试 ──


@pytest.fixture
def trade_dt():
    """2025-01-01 到 2026-12-31 的工作日 trade_dt."""
    dates = [
        int(d.strftime("%Y%m%d"))
        for d in pd.bdate_range("2025-01-01", "2026-12-31")
    ]
    return pd.DataFrame(dates, columns=[0])


class TestAdjustDateNode:
    def test_missing_beg_raises(self, trade_dt):
        """adj_date_beg=None → raise."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        with pytest.raises(Exception):
            AdjustDateNode(config={
                "adj_date_beg": None,
                "adj_date_end": 20251231,
                "adj_mode": ["M", "end"],
            }).execute(context=ctx)

    def test_missing_end_raises(self, trade_dt):
        """adj_date_end=None → raise."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        with pytest.raises(Exception):
            AdjustDateNode(config={
                "adj_date_beg": 20250101,
                "adj_date_end": None,
                "adj_mode": ["M", "end"],
            }).execute(context=ctx)

    def test_monthly_end(self, trade_dt):
        """M-end: 1 年 ≈ 12 个月末."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        result = AdjustDateNode(config={
            "adj_date_beg": 20250101,
            "adj_date_end": 20251231,
            "adj_mode": ["M", "end"],
        }).execute(context=ctx)
        assert isinstance(result, pd.DataFrame)
        assert 10 <= len(result) <= 13

    def test_monthly_begin(self, trade_dt):
        """M-begin: 1 年 ≈ 12 个月初 (注意是 'begin' 不是 'beg')."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        result = AdjustDateNode(config={
            "adj_date_beg": 20250101,
            "adj_date_end": 20251231,
            "adj_mode": ["M", "begin"],
        }).execute(context=ctx)
        assert 10 <= len(result) <= 13

    def test_weekly_end_returns_dates(self, trade_dt):
        """W-end 返回交易日序列 (现有实现按 weekday-change 切分, 数量较密)."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        result = AdjustDateNode(config={
            "adj_date_beg": 20250101,
            "adj_date_end": 20251231,
            "adj_mode": ["W", "end"],
        }).execute(context=ctx)
        # 锁定输出非空 + 在范围内 + 严格递增 (现有实现细节验证)
        assert len(result) > 0
        vals = result.iloc[:, 0].values
        assert vals.min() >= 20250101
        assert vals.max() <= 20251231
        assert (np.diff(vals) > 0).all()

    def test_daily_step5(self, trade_dt):
        """D step=5: 250 交易日 / 5 ≈ 50."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        result = AdjustDateNode(config={
            "adj_date_beg": 20250101,
            "adj_date_end": 20251231,
            "adj_mode": ["D", 5],
        }).execute(context=ctx)
        assert 40 <= len(result) <= 60

    def test_adj_dates_within_bounds(self, trade_dt):
        """调仓日 ∈ [beg, end]."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        result = AdjustDateNode(config={
            "adj_date_beg": 20250601,
            "adj_date_end": 20251231,
            "adj_mode": ["M", "end"],
        }).execute(context=ctx)
        vals = result.iloc[:, 0]
        assert vals.min() >= 20250601
        assert vals.max() <= 20251231

    def test_adj_dates_sorted_ascending(self, trade_dt):
        """调仓日严格递增."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        result = AdjustDateNode(config={
            "adj_date_beg": 20250101,
            "adj_date_end": 20251231,
            "adj_mode": ["M", "end"],
        }).execute(context=ctx)
        vals = result.iloc[:, 0].values
        assert (np.diff(vals) > 0).all()

    def test_reverse_range_empty_or_raises(self, trade_dt):
        """beg > end → 抛 raise 或返回空 (实现可选)."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        try:
            result = AdjustDateNode(config={
                "adj_date_beg": 20251231,
                "adj_date_end": 20250101,
                "adj_mode": ["M", "end"],
            }).execute(context=ctx)
            # 若不报错, 应返回空 DataFrame
            assert len(result) == 0
        except Exception:
            pass  # raise 也是合理的

    def test_invalid_mode_raises(self, trade_dt):
        """不支持的 mode → raise."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        with pytest.raises(Exception):
            AdjustDateNode(config={
                "adj_date_beg": 20250101,
                "adj_date_end": 20251231,
                "adj_mode": ["X", "end"],
            }).execute(context=ctx)

    @pytest.mark.parametrize("mode", [["M", "end"], ["M", "begin"], ["W", "end"]])
    def test_dtype_preserved(self, trade_dt, mode):
        """输出 dtype 与 trade_dt 一致 (int64)."""
        ctx = {"LoadData": {"trade_dt": trade_dt}}
        result = AdjustDateNode(config={
            "adj_date_beg": 20250101,
            "adj_date_end": 20251231,
            "adj_mode": mode,
        }).execute(context=ctx)
        assert result.iloc[:, 0].dtype == trade_dt.iloc[:, 0].dtype
