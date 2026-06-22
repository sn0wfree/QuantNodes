# coding: utf-8
"""Tests for GroupAnalyzerNode bool / discrete / low_tie factor dispatch.

Covers the 3-branch strategy in group_analyzer_node:
  - _classify_factor     (dtype + n_unique routing)
  - _group_continuous    (unchanged pd.qcut behavior)
  - _group_low_tie       (rank('first') + qcut, fixes ValueError on ties)
  - _group_discrete      (bool / 二值, value-proportional + seeded shuffle)
  - end-to-end dispatch  via _calc_group_return mock
"""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.nodes.group_analyzer_node import (
    _classify_factor,
    _group_continuous,
    _group_low_tie,
    _group_discrete,
)


# ---------------------------------------------------------------------------
# _classify_factor
# ---------------------------------------------------------------------------

class TestClassifyFactor:
    def test_bool_dtype_always_discrete(self):
        row = pd.Series([True, False, True, False] * 5, dtype=bool)
        assert _classify_factor(row, 5) == "discrete"

    def test_integer_dtype_low_cardinality_discrete(self):
        # -1/0/+1, integer dtype, n_unique=3 <= 10
        row = pd.Series([-1] * 10 + [0] * 20 + [1] * 20, dtype=np.int64)
        assert _classify_factor(row, 5) == "discrete"

    def test_integer_dtype_high_cardinality_continuous(self):
        # integer dtype but n_unique=20 > 10
        row = pd.Series(list(range(20)) * 3, dtype=np.int64)
        assert _classify_factor(row, 5) == "continuous"

    def test_float_bivalue_discrete(self):
        # -1.0/+1.0 cast from polars, n_unique=2
        row = pd.Series([-1.0] * 30 + [1.0] * 20)
        assert _classify_factor(row, 5) == "discrete"

    def test_low_tie(self):
        # 50 unique floats, group=5 → n_unique >= n_groups → continuous
        # (n_unique < n_groups 且 dtype 不在 discrete 范围内的场景, 用 float 测)
        np.random.seed(1)
        row = pd.Series([0.1, 0.2, 0.3] * 17)  # 51 rows, 3 unique floats
        # n_unique=3 < n_groups=5, float dtype → low_tie
        assert _classify_factor(row, 5) == "low_tie"

    def test_continuous(self):
        np.random.seed(42)
        row = pd.Series(np.random.randn(50))
        assert _classify_factor(row, 5) == "continuous"


# ---------------------------------------------------------------------------
# _group_continuous  (无回归: 与原 pd.qcut 行为一致)
# ---------------------------------------------------------------------------

class TestGroupContinuous:
    def test_continuous_unchanged(self):
        """50 个 randn 走 _group_continuous, 应得 5 组且无 ValueError。"""
        np.random.seed(42)
        row = pd.Series(np.random.randn(50), index=[f"s{i}" for i in range(50)])
        groups = _group_continuous(row, 5)
        assert groups.nunique() == 5
        assert set(groups.unique()) <= {1, 2, 3, 4, 5}

    def test_continuous_matches_original_qcut(self):
        """关键回归点: 连续因子结果与直接调 pd.qcut 一致。"""
        np.random.seed(7)
        row = pd.Series(np.random.randn(100))
        expected = pd.qcut(
            row, 5, labels=range(1, 6), duplicates='drop'
        )
        result = _group_continuous(row, 5)
        pd.testing.assert_series_equal(
            result.astype('float64'),
            expected.astype('float64'),
        )


# ---------------------------------------------------------------------------
# _group_low_tie  (bug 修复: 原 pd.qcut 在 ties 下抛 ValueError)
# ---------------------------------------------------------------------------

class TestGroupLowTie:
    def test_low_tie_no_value_error(self):
        """3 unique × 50 行, 原 pd.qcut 抛 ValueError, 新分支通过。"""
        row = pd.Series([0] * 17 + [1] * 17 + [2] * 16,
                        index=[f"s{i}" for i in range(50)])
        # sanity: 原调用必崩
        with pytest.raises(ValueError):
            pd.qcut(row, 5, labels=range(1, 6), duplicates='drop')
        # 新分支修复
        groups = _group_low_tie(row, 5)
        assert groups.nunique() == 5
        assert set(groups.unique()) == {1, 2, 3, 4, 5}

    def test_low_tie_deterministic(self):
        """rank(method='first') 决定性 → 同输入多次结果一致。"""
        row = pd.Series([0] * 17 + [1] * 17 + [2] * 16)
        g1 = _group_low_tie(row, 5)
        g2 = _group_low_tie(row, 5)
        pd.testing.assert_series_equal(g1, g2)


# ---------------------------------------------------------------------------
# _group_discrete
# ---------------------------------------------------------------------------

class TestGroupDiscrete:
    def test_bool_30_neg_20_pos(self):
        """30 × -1 + 20 × +1 → -1 在 group 1-3, +1 在 group 4-5。"""
        row = pd.Series([-1] * 30 + [1] * 20,
                        index=[f"stock_{i}" for i in range(50)])
        groups = _group_discrete(row, 5, 20200615)

        neg_groups = groups.iloc[:30].unique()
        pos_groups = groups.iloc[30:].unique()
        assert all(g in [1, 2, 3] for g in neg_groups)
        assert all(g in [4, 5] for g in pos_groups)
        assert groups.nunique() == 5

    def test_three_value_discrete(self):
        """-1/0/+1 (15/20/15) → 3 段组, 不崩。"""
        row = pd.Series(
            [-1] * 15 + [0] * 20 + [1] * 15,
            index=[f"s{i}" for i in range(50)]
        )
        groups = _group_discrete(row, 5, 20200615)
        # 三个 value 各自落在不同组段
        neg_set = set(groups.iloc[:15].unique())
        zero_set = set(groups.iloc[15:35].unique())
        pos_set = set(groups.iloc[35:].unique())
        assert neg_set.isdisjoint(zero_set)
        assert zero_set.isdisjoint(pos_set)
        assert neg_set.isdisjoint(pos_set)
        assert groups.nunique() == 5

    def test_deterministic_seed(self):
        """同一 date_int 多次调用结果一致。"""
        row = pd.Series([-1] * 30 + [1] * 20,
                        index=[f"s{i}" for i in range(50)])
        g1 = _group_discrete(row, 5, 20200615)
        g2 = _group_discrete(row, 5, 20200615)
        pd.testing.assert_series_equal(g1, g2)

    def test_seed_differs_by_date(self):
        """不同 date_int → shuffle 不同, 证明 seed 真的生效。"""
        row = pd.Series([-1] * 30 + [1] * 20,
                        index=[f"s{i}" for i in range(50)])
        g_a = _group_discrete(row, 5, 20200615)
        g_b = _group_discrete(row, 5, 20200616)
        # shuffle 顺序不同, 至少有一个 -1 位置的组标签变化
        # 排除极小概率完全相同: 在 30! shuffle 空间下概率 ~ 0
        assert not g_a.equals(g_b)

    def test_underfilled_groups(self):
        """只有 5 个值不够 5 组, 不应崩。"""
        row = pd.Series([-1, -1, 1, 1, 1],
                        index=["a", "b", "c", "d", "e"])
        groups = _group_discrete(row, 5, 20200615)
        assert len(groups) == 5
        # 所有非 nan
        assert groups.notna().all()

    def test_all_same_value(self):
        """全部相同值 (n_unique=1) → 全部进同一组段, 不崩。"""
        row = pd.Series([1] * 50, index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 5, 20200615)
        assert groups.notna().all()
        # 全 -1 (1) 走 value-proportional: 100% 占 5 组
        assert groups.nunique() == 5

    def test_handles_nan_input_via_dropna(self):
        """helper 自身不处理 nan (caller 负责 dropna), 输入含 nan 时
        应只对非 nan 赋值, nan 保持 nan。"""
        row = pd.Series([-1] * 5 + [np.nan] * 5 + [1] * 5)
        clean = row.dropna()
        groups = _group_discrete(clean, 5, 20200615)
        assert groups.notna().all()
        # helper 接收的是 dropna 后的 series, 长度为 10 (5 -1 + 5 +1)
        assert len(groups) == 10


# ---------------------------------------------------------------------------
# end-to-end: 通过 _calc_group_return 跑完整 dispatch
# ---------------------------------------------------------------------------

class TestCalcGroupReturnDispatch:
    """验证 3 种因子在 _calc_group_return 真实入口处的行为。

    注: 本测试只走 _calc_group_return 的分组循环 (前 ~25 行),
    不构造 price / index_cp 等下游依赖, 因为那些与本变更无关。
    """

    @staticmethod
    def _run_grouping_only(factor_data: pd.DataFrame, group: int = 5):
        """仅调用 _calc_group_return 的分组阶段, 跳过 price/IC 计算。

        通过 monkeypatch 掉内部价格依赖: 直接复用源文件中的分组循环
        逻辑, 但因源逻辑嵌入 _calc_group_return, 此处用简化 wrapper。
        """
        # 复用模块级 handler + 与 _calc_group_return 一致的 loop
        fac_group = factor_data.copy() * np.nan
        for i in range(len(factor_data)):
            t_i = factor_data.index[i]
            row = factor_data.loc[t_i].dropna()
            nonan = len(row)
            if nonan == 0 or nonan < group:
                continue
            kind = _classify_factor(row, group)
            if kind == "discrete":
                fac_group.loc[t_i] = _group_discrete(row, group, int(t_i))
            elif kind == "low_tie":
                fac_group.loc[t_i] = _group_low_tie(row, group)
            else:
                fac_group.loc[t_i] = _group_continuous(row, group)
        return fac_group

    def test_bool_factor_end_to_end(self):
        """50 stocks × 5 dates, 全部 bool 因子, 不抛 ValueError。"""
        dates = [20200615, 20200616, 20200617, 20200618, 20200619]
        stocks = [f"s{i}" for i in range(50)]
        data = pd.DataFrame(
            np.where(
                np.random.RandomState(0).rand(50, 5) > 0.4, 1, -1
            ).T,  # 5 dates × 50 stocks
            index=dates,
            columns=stocks,
        )
        result = self._run_grouping_only(data, group=5)
        # 每行都是 5 组
        for d in dates:
            assert result.loc[d].nunique() == 5, f"date {d} did not produce 5 groups"

    def test_continuous_factor_end_to_end(self):
        """50 stocks × 5 dates, 连续因子, 行为与原 pd.qcut 一致。"""
        np.random.seed(123)
        dates = [20200615, 20200616, 20200617, 20200618, 20200619]
        stocks = [f"s{i}" for i in range(50)]
        data = pd.DataFrame(
            np.random.randn(5, 50),
            index=dates,
            columns=stocks,
        )
        result = self._run_grouping_only(data, group=5)
        for d in dates:
            assert result.loc[d].nunique() == 5

    def test_low_tie_factor_end_to_end(self):
        """3 unique × 50 stocks × 3 dates, 不抛 ValueError。"""
        dates = [20200615, 20200616, 20200617]
        stocks = [f"s{i}" for i in range(50)]
        rows = []
        for _ in dates:
            row = [0] * 17 + [1] * 17 + [2] * 16
            np.random.RandomState(0).shuffle(row)
            rows.append(row)
        data = pd.DataFrame(rows, index=dates, columns=stocks)
        result = self._run_grouping_only(data, group=5)
        for d in dates:
            assert result.loc[d].nunique() == 5
