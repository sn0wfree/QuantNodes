"""factor_test 节点边界测试 (15 tests, 用真实 H5 数据集)。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
from QuantNodes.research.factor_test.nodes.load_data_node import LoadDataNode
from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
from QuantNodes.research.factor_test.utils.data_loader import DataLoader


@pytest.fixture
def data_ctx(tmp_path):
    """完整 H5 数据集 + LoadData context。"""
    d = tmp_path
    n_days, n_stocks = 28, 5  # 1 月 (28 天, 避免 20250132 等无效日期)
    dates = [20250101 + i for i in range(n_days)]
    stks = [f"00000{i}.SZ" for i in range(n_stocks)]

    stklist = pd.DataFrame({0: stks})
    trade_dt = pd.DataFrame({0: dates})

    cp = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.randn(n_days, n_stocks) * 0.01, axis=0)),
        index=dates, columns=stks,
    )

    with pd.HDFStore(d / "stk_daily.h5", mode="w") as store:
        store.put("stklist", stklist, format="table")
        store.put("trade_dt", trade_dt, format="table")
        store.put("cp", cp, format="table")
        store.put("st", pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stks), format="table")
        store.put("suspend", pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stks), format="table")
        store.put("ud_limit", pd.DataFrame(np.zeros((n_days, n_stocks), dtype=int), index=dates, columns=stks), format="table")
        store.put("ipo_days", pd.DataFrame(np.full((n_days, n_stocks), 500, dtype=int), index=dates, columns=stks), format="table")
        store.put("id_citic1", pd.DataFrame(np.random.randint(1, 5, (n_days, n_stocks)), index=dates, columns=stks), format="table")
        store.put("mv_float", pd.DataFrame(np.random.uniform(1e8, 1e10, (n_days, n_stocks)), index=dates, columns=stks), format="table")

    with pd.HDFStore(d / "index_daily.h5", mode="w") as store:
        store.put("indexlist", pd.DataFrame({0: ["000300.SH", "000905.SH"]}), format="table")
        store.put("trade_dt", trade_dt, format="table")
        store.put("index_cp", pd.DataFrame(
            np.cumsum(np.random.randn(n_days, 2) * 0.01, axis=0) + 100,
            index=dates, columns=["000300.SH", "000905.SH"]), format="table")

    with pd.HDFStore(d / "factor.h5", mode="w") as store:
        factor = pd.DataFrame(np.random.randn(n_days, n_stocks), index=dates, columns=stks)
        store.put("data", factor, format="table")

    # LoadDataNode 加载
    loader_node = LoadDataNode(config={
        "data_path": str(d) + "/",
        "load_keys": ["stklist", "trade_dt", "cp", "st", "suspend", "ud_limit", "ipo_days", "id_citic1", "mv_float"],
        "factor": {
            "factor_dir": "factor.h5",
            "name": "data",
        },
    })
    load_data = loader_node._execute()
    return {"LoadData": load_data, "SamplePoolFilter": None}


class TestAdjustDateNode:
    def test_month_end(self, data_ctx):
        node = AdjustDateNode(config={"adj_date_beg": 20250101, "adj_date_end": 20250228, "adj_mode": ["M", "end"]})
        out = node._execute(context=data_ctx)
        assert isinstance(out, pd.DataFrame)
        assert out.iloc[-1, 0] <= 20250228

    def test_default_config_out_of_range(self, data_ctx):
        """默认 config 范围 (2017) 与数据 (2025) 不匹配, 抛 ValueError。"""
        node = AdjustDateNode()
        with pytest.raises(ValueError):
            node._execute(context=data_ctx)

    def test_daily(self, data_ctx):
        node = AdjustDateNode(config={"adj_date_beg": 20250101, "adj_date_end": 20250110, "adj_mode": ["D", 1]})
        out = node._execute(context=data_ctx)
        assert len(out) <= 10


class TestTradabilityFilter:
    def test_no_constraints_all_tradable(self, data_ctx):
        node = TradabilityFilterNode(config={"tradable": {}})
        out = node._execute(context=data_ctx)
        assert out.shape == (28, 5)
        # 全可交易 → 全 1
        assert (out == 1).all().all()

    def test_no_st_excludes_st_stocks(self, data_ctx):
        node = TradabilityFilterNode(config={"tradable": {"no_st": True}})
        out = node._execute(context=data_ctx)
        # ST 全为 0, 不排除任何
        assert out.notna().sum().sum() == 28 * 5

    def test_no_ipo_days_excludes_new(self, data_ctx):
        node = TradabilityFilterNode(config={"tradable": {"min_ipo_days": 600}})
        out = node._execute(context=data_ctx)
        # 默认 ipo=500 < 600, 全部排除
        assert out.isna().all().all()

    def test_trace_unknown_raises(self, data_ctx):
        node = TradabilityFilterNode(config={"tradable": {"trace": {"foo": (1, 1)}}})
        with pytest.raises(ValueError, match="不支持的追踪条件"):
            node._execute(context=data_ctx)

    def test_no_suspended_with_data(self, data_ctx):
        ld = data_ctx["LoadData"]
        # stk 0 每天 suspend=1, 其余正常
        suspend = pd.DataFrame(
            np.zeros((28, 5), dtype=int),
            index=ld["trade_dt"].iloc[:, 0].values,
            columns=ld["stklist"].iloc[:, 0].values,
        )
        suspend.iloc[:, 0] = 1
        ld["suspend"] = suspend
        node = TradabilityFilterNode(config={"tradable": {"no_suspended": True}})
        out = node._execute(context=data_ctx)
        # 第 1 列全部不可交易
        assert out.iloc[:, 0].isna().all()


class TestFactorPreprocess:
    def test_missing_factor_raises(self, data_ctx):
        node = FactorPreprocessNode(config={"missing": "", "extreme": "", "norm": ""})
        data_ctx_no_factor = {"LoadData": {}}
        with pytest.raises(ValueError, match="因子数据缺失"):
            node._execute(context=data_ctx_no_factor)

    def test_basic_preprocess(self, data_ctx):
        node = FactorPreprocessNode(config={"missing": "ind_avg", "extreme": "median", "norm": "zscore"})
        # 需要 adj_dates 与 tradable
        data_ctx["TradabilityFilter"] = data_ctx["LoadData"]["cp"].copy() * 0 + 1
        data_ctx["TradabilityFilter"].index = data_ctx["LoadData"]["trade_dt"].iloc[:, 0].values
        data_ctx["TradabilityFilter"].columns = data_ctx["LoadData"]["stklist"].iloc[:, 0].values
        data_ctx["AdjustDate"] = pd.DataFrame(
            [data_ctx["LoadData"]["trade_dt"].iloc[0, 0], data_ctx["LoadData"]["trade_dt"].iloc[14, 0]]
        )
        out = node._execute(context=data_ctx)
        # 输出 index 是调仓日
        assert len(out) == 2

    def test_pct_shrink(self, data_ctx):
        node = FactorPreprocessNode(config={"extreme": "pct_shrink", "norm": "norm"})
        data_ctx["TradabilityFilter"] = data_ctx["LoadData"]["cp"].copy() * 0 + 1
        data_ctx["TradabilityFilter"].index = data_ctx["LoadData"]["trade_dt"].iloc[:, 0].values
        data_ctx["TradabilityFilter"].columns = data_ctx["LoadData"]["stklist"].iloc[:, 0].values
        data_ctx["AdjustDate"] = pd.DataFrame(
            [data_ctx["LoadData"]["trade_dt"].iloc[0, 0], data_ctx["LoadData"]["trade_dt"].iloc[14, 0]]
        )
        out = node._execute(context=data_ctx)
        assert len(out) == 2

    def test_empty_tradable_returns_unchanged(self, data_ctx):
        """全 NaN tradable → 返回原值。"""
        node = FactorPreprocessNode(config={"missing": "", "extreme": "median", "norm": ""})
        data_ctx["TradabilityFilter"] = None
        data_ctx["AdjustDate"] = pd.DataFrame(
            [data_ctx["LoadData"]["trade_dt"].iloc[0, 0]]
        )
        out = node._execute(context=data_ctx)
        assert len(out) == 1
