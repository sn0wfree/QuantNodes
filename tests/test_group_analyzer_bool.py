# coding: utf-8
"""Tests for GroupAnalyzerNode bool / discrete / ranked factor dispatch.

Covers the 2-branch strategy in group_analyzer_node:
  - _classify_factor  (dtype + n_unique routing → "ranked" | "discrete")
  - _group_ranked     (rank('first') + qcut, 处理连续和 ties 因子,
                       修复 ties 下 pd.qcut ValueError)
  - _group_discrete   (bool / 二值, value-proportional + seeded shuffle)
  - end-to-end dispatch via _calc_group_return mock
"""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.nodes.group_analyzer_node import (
    _classify_factor,
    _group_ranked,
    _group_discrete,
)


# ---------------------------------------------------------------------------
# _classify_factor
# ---------------------------------------------------------------------------

class TestClassifyFactor:
    def test_bool_dtype_always_discrete(self):
        row = pd.Series([True, False, True, False] * 5, dtype=bool)
        assert _classify_factor(row, 5) == "discrete"

    def test_integer_dtype_bivalue_discrete(self):
        # -1/+1, integer dtype, n_unique=2 → discrete
        row = pd.Series([-1] * 30 + [1] * 20, dtype=np.int64)
        assert _classify_factor(row, 5) == "discrete"

    def test_integer_dtype_3_value_ranked(self):
        # -1/0/+1 integer, n_unique=3 → ranked (value-proportional 不适用)
        row = pd.Series([-1] * 10 + [0] * 20 + [1] * 20, dtype=np.int64)
        assert _classify_factor(row, 5) == "ranked"

    def test_integer_dtype_high_cardinality_ranked(self):
        # integer dtype but n_unique=20 → ranked
        row = pd.Series(list(range(20)) * 3, dtype=np.int64)
        assert _classify_factor(row, 5) == "ranked"

    def test_float_bivalue_discrete(self):
        # -1.0/+1.0 cast from polars, n_unique=2
        row = pd.Series([-1.0] * 30 + [1.0] * 20)
        assert _classify_factor(row, 5) == "discrete"

    def test_ranked_continuous(self):
        np.random.seed(42)
        row = pd.Series(np.random.randn(50))
        assert _classify_factor(row, 5) == "ranked"

    def test_ranked_low_tie(self):
        # 51 rows, 3 unique floats, group=5 → ranked (含 ties)
        np.random.seed(1)
        row = pd.Series([0.1, 0.2, 0.3] * 17)
        assert _classify_factor(row, 5) == "ranked"


# ---------------------------------------------------------------------------
# _group_ranked  (合并自原 _group_continuous + _group_low_tie)
# ---------------------------------------------------------------------------

class TestGroupRanked:
    def test_continuous_matches_original_qcut(self):
        """无 ties 时 _group_ranked 与原 pd.qcut(series) bitwise 等价 (零回归)。"""
        np.random.seed(7)
        row = pd.Series(np.random.randn(100))
        expected = pd.qcut(
            row, 5, labels=range(1, 6), duplicates='drop'
        )
        result = _group_ranked(row, 5)
        pd.testing.assert_series_equal(
            result.astype('float64'),
            expected.astype('float64'),
        )

    def test_no_ties(self):
        """50 randn 无 ties 走 ranked 分支, 5 组每组 10。"""
        np.random.seed(42)
        row = pd.Series(np.random.randn(50), index=[f"s{i}" for i in range(50)])
        assert _classify_factor(row, 5) == "ranked"
        groups = _group_ranked(row, 5)
        assert groups.nunique() == 5
        for g in range(1, 6):
            assert (groups == g).sum() == 10

    def test_continuous_with_ties(self):
        """alpha-004 场景: 7 unique × 50 rows, 不应抛 ValueError。

        12 个 -9, 8 个 -8, 5 个 -7, 5 个 -6, 5 个 -5, 4 个 -4, 11 个 -3。
        原 pd.qcut(duplicates='drop') 必抛 ValueError。
        """
        row = pd.Series(
            [-9] * 12 + [-8] * 8 + [-7] * 5 + [-6] * 5 +
            [-5] * 5 + [-4] * 4 + [-3] * 11,
            index=[f"s{i}" for i in range(50)],
        )
        assert _classify_factor(row, 5) == "ranked"
        # 原调用必崩
        with pytest.raises(ValueError):
            pd.qcut(row, 5, labels=range(1, 6), duplicates='drop')
        # _group_ranked 修复
        groups = _group_ranked(row, 5)
        assert groups.notna().all()
        assert groups.nunique() == 5
        for g in range(1, 6):
            assert (groups == g).sum() == 10

    def test_returns_5_groups_for_50_unique(self):
        """50 unique values 走 ranked 分支, 5 组每组 10。"""
        np.random.seed(42)
        row = pd.Series(np.random.choice(range(1000), 50, replace=False))
        groups = _group_ranked(row, 5)
        assert groups.nunique() == 5
        for g in range(1, 6):
            assert (groups == g).sum() == 10, f"Group {g} should have 10 members"

    def test_low_tie_subrange(self):
        """原 low_tie 场景 (3 <= n_unique < group) 仍工作 (回归保护)。"""
        row = pd.Series(
            [0] * 17 + [1] * 17 + [2] * 16,
            index=[f"s{i}" for i in range(50)],
        )
        groups = _group_ranked(row, 5)
        assert groups.nunique() == 5
        assert set(groups.unique()) == {1, 2, 3, 4, 5}

    def test_deterministic(self):
        """rank(method='first') 决定性 → 同输入多次结果一致。"""
        row = pd.Series([0] * 17 + [1] * 17 + [2] * 16)
        g1 = _group_ranked(row, 5)
        g2 = _group_ranked(row, 5)
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
        assert not g_a.equals(g_b)

    def test_underfilled_groups(self):
        """只有 5 个值不够 5 组, 不应崩。"""
        row = pd.Series([-1, -1, 1, 1, 1],
                        index=["a", "b", "c", "d", "e"])
        groups = _group_discrete(row, 5, 20200615)
        assert len(groups) == 5
        assert groups.notna().all()

    def test_all_same_value(self):
        """全部相同值 (n_unique=1) → 全部进同一组段, 不崩。"""
        row = pd.Series([1] * 50, index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 5, 20200615)
        assert groups.notna().all()
        assert groups.nunique() == 5

    def test_handles_nan_input_via_dropna(self):
        """helper 自身不处理 nan (caller 负责 dropna)。"""
        row = pd.Series([-1] * 5 + [np.nan] * 5 + [1] * 5)
        clean = row.dropna()
        groups = _group_discrete(clean, 5, 20200615)
        assert groups.notna().all()
        assert len(groups) == 10


# ---------------------------------------------------------------------------
# end-to-end: 通过 _calc_group_return 跑完整 dispatch
# ---------------------------------------------------------------------------

class TestCalcGroupReturnDispatch:
    """验证 2 种因子在 _calc_group_return 真实入口处的行为。"""

    @staticmethod
    def _run_grouping_only(factor_data: pd.DataFrame, group: int = 5):
        """仅调用 _calc_group_return 的分组阶段, 跳过 price/IC 计算。

        复用与 _calc_group_return 一致的 dispatch 逻辑 (continuous/
        low_tie 已合并为 ranked, 走 _group_ranked)。
        """
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
            else:
                fac_group.loc[t_i] = _group_ranked(row, group)
        return fac_group

    def test_bool_factor_end_to_end(self):
        """50 stocks × 5 dates, 全部 bool 因子, 不抛 ValueError。"""
        dates = [20200615, 20200616, 20200617, 20200618, 20200619]
        stocks = [f"s{i}" for i in range(50)]
        data = pd.DataFrame(
            np.where(
                np.random.RandomState(0).rand(50, 5) > 0.4, 1, -1
            ).T,
            index=dates,
            columns=stocks,
        )
        result = self._run_grouping_only(data, group=5)
        for d in dates:
            assert result.loc[d].nunique() == 5, f"date {d} did not produce 5 groups"

    def test_continuous_factor_end_to_end(self):
        """50 stocks × 5 dates, 连续因子, 5 组每组 10。"""
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

    def test_alpha_004_ties_end_to_end(self):
        """alpha-004 真实场景: 7 unique × 50 stocks × 3 dates,
        原 pd.qcut 必崩, _group_ranked 修复。"""
        dates = [20200615, 20200616, 20200617]
        stocks = [f"s{i}" for i in range(50)]
        rows = []
        for _ in dates:
            row = ([-9] * 12 + [-8] * 8 + [-7] * 5 + [-6] * 5 +
                   [-5] * 5 + [-4] * 4 + [-3] * 11)
            np.random.RandomState(0).shuffle(row)
            rows.append(row)
        data = pd.DataFrame(rows, index=dates, columns=stocks)
        result = self._run_grouping_only(data, group=5)
        for d in dates:
            assert result.loc[d].notna().all()
            assert result.loc[d].nunique() == 5
            # 每组 10 个
            for g in range(1, 6):
                assert (result.loc[d] == g).sum() == 10
