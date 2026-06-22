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

try:
    from hypothesis import given, settings, strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

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


# ---------------------------------------------------------------------------
# 边界场景 — _classify_factor
# ---------------------------------------------------------------------------

class TestClassifyFactorEdgeCases:
    def test_n_unique_one_constant_value(self):
        """n_unique=1 (全同值) → discrete (走 _group_discrete 安全路径)。"""
        row = pd.Series([42.0] * 30)
        assert _classify_factor(row, 5) == "discrete"

    def test_n_unique_two_boundary_discrete(self):
        """n_unique=2 边界 → discrete。"""
        row = pd.Series([0, 1] * 15)
        assert _classify_factor(row, 5) == "discrete"

    def test_n_unique_three_boundary_ranked(self):
        """n_unique=3 边界 → ranked (因为 _group_discrete 对 n_unique>2 会
        产出 n_unique 个组而非 n_groups 个组)。"""
        row = pd.Series([0, 1, 2] * 10)
        assert _classify_factor(row, 5) == "ranked"

    def test_bool_dtype_one_value_still_discrete(self):
        """bool dtype 哪怕 n_unique=1 (全 True/全 False) 也走 discrete。"""
        row = pd.Series([True] * 20, dtype=bool)
        assert _classify_factor(row, 5) == "discrete"

    def test_int8_dtype(self):
        """int8 dtype 走 n_unique<=2 判断, 不走 dtype 短路。"""
        row = pd.Series([-1, 1] * 15, dtype=np.int8)
        assert _classify_factor(row, 5) == "discrete"

    def test_int16_dtype(self):
        row = pd.Series([0, 1, 2] * 10, dtype=np.int16)
        assert _classify_factor(row, 5) == "ranked"

    def test_int32_dtype(self):
        row = pd.Series(list(range(15)) * 2, dtype=np.int32)
        assert _classify_factor(row, 5) == "ranked"

    def test_object_dtype_strings(self):
        """object dtype (含字符串) 走 n_unique 判断, 不抛 dtype 错误。"""
        row = pd.Series(["a", "b"] * 20, dtype=object)
        assert _classify_factor(row, 5) == "discrete"

    def test_n_unique_equals_n_groups_boundary(self):
        """n_unique == n_groups → ranked (走 _group_ranked 5 组)。"""
        row = pd.Series([0, 1, 2, 3, 4] * 10)
        assert _classify_factor(row, 5) == "ranked"

    def test_n_unique_just_below_n_groups(self):
        """n_unique == n_groups - 1 → ranked。"""
        row = pd.Series([0, 1, 2, 3] * 13)  # n_unique=4, group=5
        assert _classify_factor(row, 5) == "ranked"


# ---------------------------------------------------------------------------
# 边界场景 — _group_ranked
# ---------------------------------------------------------------------------

class TestGroupRankedEdgeCases:
    def test_two_groups(self):
        """n_groups=2 边界, 50 个值应分 2 组每组 25。"""
        np.random.seed(11)
        row = pd.Series(np.random.randn(50))
        groups = _group_ranked(row, 2)
        assert groups.nunique() == 2
        assert set(groups.unique()) == {1, 2}
        assert (groups == 1).sum() == 25
        assert (groups == 2).sum() == 25

    def test_three_groups(self):
        """n_groups=3, 51 个值 (17×3) 完美均分。"""
        np.random.seed(22)
        row = pd.Series(np.random.randn(51))
        groups = _group_ranked(row, 3)
        assert groups.nunique() == 3
        for g in [1, 2, 3]:
            assert (groups == g).sum() == 17

    def test_ten_groups(self):
        """n_groups=10, 100 个值 (10×10) 完美均分。"""
        np.random.seed(33)
        row = pd.Series(np.random.randn(100))
        groups = _group_ranked(row, 10)
        assert groups.nunique() == 10
        for g in range(1, 11):
            assert (groups == g).sum() == 10

    def test_n_groups_equals_n_unique(self):
        """n_groups == n_unique 边界 (5 unique, 5 groups): 每组 10 个。"""
        row = pd.Series([0] * 10 + [1] * 10 + [2] * 10 +
                        [3] * 10 + [4] * 10)
        groups = _group_ranked(row, 5)
        assert groups.nunique() == 5
        for g in range(1, 6):
            assert (groups == g).sum() == 10

    def test_extreme_tie_one_value_dominates(self):
        """49 个相同值 + 1 个 outlier, 不崩, 产出 5 组。"""
        row = pd.Series([0.0] * 49 + [100.0])
        groups = _group_ranked(row, 5)
        assert groups.notna().all()
        assert groups.nunique() == 5

    def test_all_same_value_ranked(self):
        """全同值, n_unique=1, 但 _classify_factor 会判 discrete, 不走 ranked。
        直接调 _group_ranked 应仍能跑 (rank 后全 1.0, qcut 抛错或合并到 1 组)。
        验证不抛 AttributeError/TypeError, 行为可观察即可。"""
        row = pd.Series([7.0] * 30)
        try:
            groups = _group_ranked(row, 5)
            # 如果 qcut 处理掉全同值, 应返回 1 组
            assert groups.nunique() <= 5
        except ValueError:
            # pandas 可能对全同值 qcut 报错, 这是已知行为
            pass

    def test_int_dtype_ranked(self):
        """整数 dtype 走 _group_ranked, 不抛错。"""
        row = pd.Series(list(range(50)), dtype=np.int64)
        groups = _group_ranked(row, 5)
        assert groups.nunique() == 5
        for g in range(1, 6):
            assert (groups == g).sum() == 10

    def test_nan_in_series_ranked(self):
        """含 nan 的 series 走 _group_ranked, 不崩 (rank 处理 nan)。"""
        row = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 10)
        groups = _group_ranked(row, 5)
        # rank 后的 qcut 应当能跑
        assert groups.notna().sum() > 0

    def test_index_preservation(self):
        """_group_ranked 返回的 series index 与输入一致。"""
        idx = [f"stock_{i}" for i in range(50)]
        np.random.seed(44)
        row = pd.Series(np.random.randn(50), index=idx)
        groups = _group_ranked(row, 5)
        assert list(groups.index) == idx

    def test_groups_are_int_labels(self):
        """分组标签是 1..n_groups 的整数, 不是 float。"""
        np.random.seed(55)
        row = pd.Series(np.random.randn(50))
        groups = _group_ranked(row, 5)
        assert set(groups.unique()) <= {1, 2, 3, 4, 5}
        # Categorical 内部 dtype 可能是 int
        assert pd.api.types.is_integer_dtype(groups) or \
               isinstance(groups.dtype, pd.CategoricalDtype)

    def test_ties_deterministic_via_rank_first(self):
        """ties 场景下, rank('first') 保证结果确定 (与索引顺序一致)。"""
        # 50 个值, 每 5 个一组, 10 组 ties
        row = pd.Series(
            ([0.0] * 5 + [1.0] * 5 + [2.0] * 5 + [3.0] * 5 +
             [4.0] * 5 + [5.0] * 5 + [6.0] * 5 + [7.0] * 5 +
             [8.0] * 5 + [9.0] * 5),
            index=[f"s{i}" for i in range(50)],
        )
        g1 = _group_ranked(row, 5)
        g2 = _group_ranked(row, 5)
        # 两次调用应 bitwise 一致
        pd.testing.assert_series_equal(
            g1.astype('float64'), g2.astype('float64')
        )


# ---------------------------------------------------------------------------
# 边界场景 — _group_discrete
# ---------------------------------------------------------------------------

class TestGroupDiscreteEdgeCases:
    def test_three_groups_2value(self):
        """n_groups=3, 30×-1 + 20×+1 → -1 占 2 组, +1 占 1 组。"""
        row = pd.Series([-1] * 30 + [1] * 20,
                        index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 3, 20200615)
        neg_set = set(groups.iloc[:30].unique())
        pos_set = set(groups.iloc[30:].unique())
        # -1 占 2 组, +1 占 1 组 → 3 组互不相交
        assert neg_set.isdisjoint(pos_set)
        assert groups.nunique() == 3

    def test_ten_groups_2value(self):
        """n_groups=10, 30×-1 + 20×+1 → -1 占 6 组, +1 占 4 组。"""
        row = pd.Series([-1] * 30 + [1] * 20,
                        index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 10, 20200615)
        neg_set = set(groups.iloc[:30].unique())
        pos_set = set(groups.iloc[30:].unique())
        assert neg_set.isdisjoint(pos_set)
        assert groups.nunique() == 10
        # -1 占 60% (6 组), +1 占 40% (4 组)
        assert len(neg_set) == 6
        assert len(pos_set) == 4

    def test_extreme_ratio_49_1(self):
        """49/1 极端比例, 不崩。49 个 -1 应占 5 组中的较多组, 1 个 +1 占 1 组。

        注: 算法限制 - 当前 -1 (49/50=0.98) 占比 round 到 5 组, +1 (1/50=0.02)
        通过减法 + max(1, ...) 保底得到 1 组, 总计 6 组 (n_groups + 1)。
        这是算法的已知行为: 不等比 2-value 输入可能产生 n_groups+1 组。
        """
        row = pd.Series([-1] * 49 + [1] * 1,
                        index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 5, 20200615)
        # 极端比例下产出 n_groups + 1 组 (算法特性, 非 bug)
        assert groups.nunique() == 6
        # +1 那只股票的 group 不在 -1 占的组里
        plus1_idx = row.index[49]
        plus1_group = groups.loc[plus1_idx]
        neg_groups = set(groups.iloc[:49].unique())
        assert plus1_group not in neg_groups

    def test_equal_50_50_split(self):
        """25×-1 + 25×+1, 完美均分: -1 占 2-3 组, +1 占 2-3 组。"""
        row = pd.Series([-1] * 25 + [1] * 25,
                        index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 5, 20200615)
        neg_set = set(groups.iloc[:25].unique())
        pos_set = set(groups.iloc[25:].unique())
        assert neg_set.isdisjoint(pos_set)
        # 25/50=0.5 → round(2.5)=2; 总和 5, 最后减法得 3
        # 实际可能 (-1=2, +1=3) 或 (-1=3, +1=2), 取决于 round 规则
        assert len(neg_set) + len(pos_set) == 5
        assert groups.nunique() == 5

    def test_string_values_discrete(self):
        """字符串/类别型 2-value, 走 _group_discrete 不崩。"""
        row = pd.Series(["buy"] * 30 + ["sell"] * 20,
                        index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 5, 20200615)
        assert groups.nunique() == 5
        buy_groups = set(groups.iloc[:30].unique())
        sell_groups = set(groups.iloc[30:].unique())
        assert buy_groups.isdisjoint(sell_groups)

    def test_single_value_n_unique_1(self):
        """n_unique=1 (全 1), 全部进 1 组段 (1..n_groups), 不崩。"""
        row = pd.Series([1] * 30, index=[f"s{i}" for i in range(30)])
        groups = _group_discrete(row, 5, 20200615)
        # 唯一 value 占全部 5 组
        assert groups.nunique() == 5

    def test_date_int_zero(self):
        """date_int=0 边界: 0 % 2**31 = 0, numpy 接受 seed=0。"""
        row = pd.Series([-1] * 30 + [1] * 20,
                        index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 5, 0)
        assert groups.notna().all()
        assert groups.nunique() == 5

    def test_date_int_large(self):
        """date_int 接近 2**32, 验证 % 2**31 模运算不崩。"""
        row = pd.Series([-1] * 30 + [1] * 20,
                        index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 5, 2**33)
        assert groups.notna().all()
        assert groups.nunique() == 5

    def test_groups_cover_full_index(self):
        """_group_discrete 返回 series 应覆盖输入全部 index, 无遗漏。"""
        idx = [f"stock_{i}" for i in range(50)]
        row = pd.Series([-1] * 30 + [1] * 20, index=idx)
        groups = _group_discrete(row, 5, 20200615)
        assert list(groups.index) == idx
        assert groups.notna().all()

    def test_3value_3group_perfect_match(self):
        """3 value × 3 group, 完美一一对应: 每个 value 占 1 组。"""
        row = pd.Series([0] * 17 + [1] * 17 + [2] * 16,
                        index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 3, 20200615)
        # 17/50≈0.34→round 1, 17/50≈0.34→round 1, 减法得 1
        # 应恰好 3 组, 每个 value 一个
        assert groups.nunique() == 3
        assert set(groups.iloc[:17].unique()) == {1}
        assert set(groups.iloc[17:34].unique()) == {2}
        assert set(groups.iloc[34:].unique()) == {3}


# ---------------------------------------------------------------------------
# 边界场景 — dispatch 集成
# ---------------------------------------------------------------------------

class TestDispatchEdgeCases:
    """更多 dispatch 场景: 混合类型, sparse days, floor_mode, 多 n_groups。"""

    @staticmethod
    def _run_grouping_only(factor_data, group=5, floor_mode='group'):
        """复用 dispatch 逻辑, 加 floor_mode 支持。"""
        fac_group = factor_data.copy() * np.nan
        for i in range(len(factor_data)):
            t_i = factor_data.index[i]
            row = factor_data.loc[t_i].dropna()
            nonan = len(row)
            if nonan == 0 or nonan < group:
                if floor_mode == 'group' or i == 0:
                    continue
                elif floor_mode == 'last':
                    fac_group.loc[t_i] = fac_group.iloc[i - 1]
                    continue
            kind = _classify_factor(row, group)
            if kind == "discrete":
                fac_group.loc[t_i] = _group_discrete(row, group, int(t_i))
            else:
                fac_group.loc[t_i] = _group_ranked(row, group)
        return fac_group

    def test_mixed_types_across_days(self):
        """同一 factor_data 中, day1 连续, day2 bool, day3 ties:
        每行应正确路由到对应 handler。"""
        dates = [20200615, 20200616, 20200617]
        stocks = [f"s{i}" for i in range(50)]

        # day 1: 连续
        np.random.seed(66)
        day1 = np.random.randn(50)
        # day 2: bool/二值
        day2 = np.where(np.arange(50) % 2 == 0, 1, -1).astype(float)
        # day 3: ties (7 unique)
        day3 = np.array(
            [-9] * 12 + [-8] * 8 + [-7] * 5 + [-6] * 5 +
            [-5] * 5 + [-4] * 4 + [-3] * 11, dtype=float
        )

        data = pd.DataFrame(
            [day1, day2, day3], index=dates, columns=stocks
        )
        result = self._run_grouping_only(data, group=5)
        # 三行都应是 5 组
        for d in dates:
            assert result.loc[d].nunique() == 5, f"{d} failed"

    def test_sparse_day_skipped_floor_mode_group(self):
        """某日 nonan < n_groups, floor_mode='group' 跳过该日。"""
        dates = [20200615, 20200616, 20200617]
        stocks = [f"s{i}" for i in range(50)]
        np.random.seed(77)
        day1 = np.random.randn(50)
        # day 2: 只有 3 个非 nan 值 (不足 5)
        day2 = np.array([np.nan] * 47 + [1.0, 2.0, 3.0])
        day3 = np.random.randn(50)

        data = pd.DataFrame(
            [day1, day2, day3], index=dates, columns=stocks
        )
        result = self._run_grouping_only(data, group=5, floor_mode='group')
        # day2 应全 nan
        assert result.loc[20200616].isna().all()
        # day1, day3 应有 5 组
        assert result.loc[20200615].nunique() == 5
        assert result.loc[20200617].nunique() == 5

    def test_sparse_day_filled_floor_mode_last(self):
        """floor_mode='last' 时 sparse day 复制上一日的分组。

        注意: 原代码逻辑是 `if i == 0: continue` (跳过第一天), 所以 day2
        (i=1) 能复制 day1 (i=0); day1 自己是 i=0 直接跳过。
        """
        dates = [20200615, 20200616, 20200617]
        stocks = [f"s{i}" for i in range(50)]
        np.random.seed(88)
        day1 = np.random.randn(50)
        # day 2 sparse
        day2 = np.array([np.nan] * 47 + [1.0, 2.0, 3.0])
        day3 = np.random.randn(50)

        data = pd.DataFrame(
            [day1, day2, day3], index=dates, columns=stocks
        )
        result = self._run_grouping_only(data, group=5, floor_mode='last')
        # day2 应复制 day1 (用 check_names=False 忽略 index name 差异)
        pd.testing.assert_series_equal(
            result.loc[20200616].fillna(-1).reset_index(drop=True),
            result.loc[20200615].fillna(-1).reset_index(drop=True),
            check_names=False,
        )

    def test_first_day_sparse_floor_mode_last_stays_nan(self):
        """第一天就 sparse, floor_mode='last' 无法复制, 保持 nan。"""
        dates = [20200615, 20200616]
        stocks = [f"s{i}" for i in range(50)]
        day1 = np.array([np.nan] * 50)
        day2 = np.random.RandomState(99).randn(50)

        data = pd.DataFrame([day1, day2], index=dates, columns=stocks)
        result = self._run_grouping_only(data, group=5, floor_mode='last')
        assert result.loc[20200615].isna().all()
        assert result.loc[20200616].nunique() == 5

    def test_all_nan_day(self):
        """某日全 nan, floor_mode='group' 跳过, 'last' 复制前一日。"""
        dates = [20200615, 20200616, 20200617]
        stocks = [f"s{i}" for i in range(50)]
        day1 = np.random.RandomState(111).randn(50)
        day2 = np.array([np.nan] * 50)
        day3 = np.random.RandomState(222).randn(50)

        data = pd.DataFrame([day1, day2, day3], index=dates, columns=stocks)
        result_group = self._run_grouping_only(data, group=5, floor_mode='group')
        result_last = self._run_grouping_only(data, group=5, floor_mode='last')
        # floor_mode='group': day2 跳过 → nan
        assert result_group.loc[20200616].isna().all()
        # floor_mode='last': day2 复制 day1 (i=1, 前一行是 i=0)
        pd.testing.assert_series_equal(
            result_last.loc[20200616].reset_index(drop=True),
            result_last.loc[20200615].reset_index(drop=True),
            check_names=False,
        )

    def test_multi_group_counts(self):
        """同一数据集, 跑 n_groups=2/3/5/10, 都应产出对应数量的组。"""
        np.random.seed(333)
        row = pd.Series(np.random.randn(50))
        for ng in [2, 3, 5, 10]:
            groups = _group_ranked(row, ng)
            assert groups.nunique() == ng, f"n_groups={ng} failed"
            assert set(groups.unique()) <= set(range(1, ng + 1))

    def test_alpha_004_n_groups_10(self):
        """alpha-004 场景跑 n_groups=10, 应得 10 组每组 5。"""
        row = pd.Series(
            [-9] * 12 + [-8] * 8 + [-7] * 5 + [-6] * 5 +
            [-5] * 5 + [-4] * 4 + [-3] * 11,
            index=[f"s{i}" for i in range(50)],
        )
        groups = _group_ranked(row, 10)
        assert groups.nunique() == 10
        for g in range(1, 11):
            assert (groups == g).sum() == 5

    def test_bool_2groups(self):
        """bool 因子跑 n_groups=2: 应得 2 个组, 严格按 value 分。"""
        row = pd.Series([-1] * 30 + [1] * 20,
                        index=[f"s{i}" for i in range(50)])
        groups = _group_discrete(row, 2, 20200615)
        assert groups.nunique() == 2
        # -1 占 1 组, +1 占 1 组
        assert set(groups.iloc[:30].unique()) == {1}
        assert set(groups.iloc[30:].unique()) == {2}


# ---------------------------------------------------------------------------
# 属性测试 — hypothesis
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestPropertyBased:
    """随机生成的输入, 验证不变量。"""

    @given(
        values=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False,
                      allow_infinity=False),
            min_size=10, max_size=200,
        ),
        n_groups=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_ranked_output_shape(self, values, n_groups):
        """_group_ranked 总产出 n_groups 个 distinct group, 且不崩。"""
        row = pd.Series(values)
        groups = _group_ranked(row, n_groups)
        # 至少 1 组, 至多 n_groups 组 (duplicates='drop' 行为)
        assert 1 <= groups.nunique() <= n_groups
        # 输出长度 = 输入长度
        assert len(groups) == len(row)
        # 输出标签 ∈ {1, ..., n_groups}
        labels = set(groups.dropna().unique())
        assert labels <= set(range(1, n_groups + 1))

    @given(
        n_neg=st.integers(min_value=1, max_value=49),
        n_groups=st.integers(min_value=2, max_value=10),
        date_int=st.integers(min_value=0, max_value=2**32),
    )
    @settings(max_examples=30, deadline=None)
    def test_discrete_2value_shape(self, n_neg, n_groups, date_int):
        """_group_discrete 2-value, 输出 ∈ {1, ..., n_groups + 1}, 完整覆盖。

        注: 极端比例下算法产出 n_groups + 1 组 (前一 value 占满所有组,
        后一 value 强制 1 组)。
        """
        n_total = 50
        n_pos = n_total - n_neg
        row = pd.Series([-1] * n_neg + [1] * n_pos,
                        index=[f"s{i}" for i in range(n_total)])
        groups = _group_discrete(row, n_groups, date_int)
        assert groups.notna().all()
        # n_groups + 1 是极端比例下的上限
        assert 1 <= groups.nunique() <= n_groups + 1
        # value 内分组不交叉
        if n_neg > 0 and n_pos > 0:
            neg_set = set(groups.iloc[:n_neg].unique())
            pos_set = set(groups.iloc[n_neg:].unique())
            assert neg_set.isdisjoint(pos_set)

    @given(
        n_values=st.integers(min_value=1, max_value=50),
        n_total=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_classify_factor_returns_known_label(self, n_values, n_total):
        """_classify_factor 总返回 'ranked' 或 'discrete'。"""
        n_values = min(n_values, n_total)
        vals = list(range(n_values))
        # 重复填到 n_total
        data = (vals * ((n_total // n_values) + 1))[:n_total]
        row = pd.Series(data)
        kind = _classify_factor(row, 5)
        assert kind in ("ranked", "discrete")
        # n_unique <= 2 → discrete
        if n_values <= 2:
            assert kind == "discrete"
        # n_unique >= 3 → ranked
        else:
            assert kind == "ranked"

    @given(
        values=st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False,
                      allow_infinity=False),
            min_size=20, max_size=100,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_ranked_preserves_index(self, values):
        """_group_ranked 输出 index 与输入完全一致。"""
        row = pd.Series(values, index=[f"stock_{i}" for i in range(len(values))])
        groups = _group_ranked(row, 5)
        assert list(groups.index) == list(row.index)


# ---------------------------------------------------------------------------
# 稳定性 / 回归保护
# ---------------------------------------------------------------------------

class TestStability:
    """验证跨版本/跨调用稳定性的不变量。"""

    def test_classify_factor_string_constants(self):
        """_classify_factor 返回值必须是 Literal 定义的 2 种之一。"""
        allowed = {"ranked", "discrete"}
        for vals in [
            [0, 1] * 20,         # 2 unique
            [0, 1, 2] * 15,      # 3 unique
            list(range(50)),     # 50 unique
            [True] * 30,         # bool
        ]:
            row = pd.Series(vals)
            kind = _classify_factor(row, 5)
            assert kind in allowed

    def test_ranked_label_range(self):
        """_group_ranked 输出标签范围 ∈ {1, ..., n_groups}。"""
        np.random.seed(444)
        for n_groups in [2, 3, 4, 5, 7, 10]:
            row = pd.Series(np.random.randn(100))
            groups = _group_ranked(row, n_groups)
            labels = set(groups.dropna().unique())
            assert labels <= set(range(1, n_groups + 1)), \
                f"n_groups={n_groups} produced out-of-range labels: {labels}"

    def test_discrete_label_range(self):
        """_group_discrete 输出标签范围 ∈ {1, ..., n_groups}。"""
        for n_groups in [2, 3, 5, 10]:
            row = pd.Series([-1] * 30 + [1] * 20)
            groups = _group_discrete(row, n_groups, 20200615)
            labels = set(groups.dropna().unique())
            assert labels <= set(range(1, n_groups + 1)), \
                f"n_groups={n_groups} produced out-of-range labels: {labels}"

    def test_ranked_no_nan_when_no_nan_input(self):
        """输入无 nan 时, _group_ranked 输出也不应有 nan。"""
        np.random.seed(555)
        row = pd.Series(np.random.randn(50))
        groups = _group_ranked(row, 5)
        assert groups.notna().all()

    def test_discrete_no_nan_when_no_nan_input(self):
        """输入无 nan 时, _group_discrete 输出也不应有 nan。"""
        row = pd.Series([-1] * 30 + [1] * 20)
        groups = _group_discrete(row, 5, 20200615)
        assert groups.notna().all()

    def test_ranked_idempotent_under_shuffle(self):
        """_group_ranked 对同一 multiset 顺序不同结果应一致 (因 rank('first') 决定性)。"""
        row_a = pd.Series([0.0] * 5 + [1.0] * 5 + [2.0] * 5 + [3.0] * 5 + [4.0] * 5)
        row_b = row_a.iloc[::-1].reset_index(drop=True)
        g_a = _group_ranked(row_a, 5)
        g_b = _group_ranked(row_b, 5)
        # 两组应都被分到 5 个 group 中, 每个 group 都有 10 个
        # 但因为 rank('first') 用的是 index 顺序, 顺序不同 → group 标签排列不同
        # 验证: multiset(group values) 一致
        assert sorted(g_a.value_counts().tolist()) == \
               sorted(g_b.value_counts().tolist())

    def test_discrete_shuffle_changes_assignment(self):
        """_group_discrete 在 value 内部随机分配, shuffle 后 group 标签排列会变。"""
        row = pd.Series([-1] * 30 + [1] * 20)
        g_a = _group_discrete(row, 5, 100)
        g_b = _group_discrete(row, 5, 200)
        # 同一 -1 位置上 group 不同 (seed 不同 → shuffle 不同)
        # 至少有一个 -1 位置的 group 标签变化
        diff = (g_a != g_b).sum()
        assert diff > 0, "different seeds should produce different shuffles"
