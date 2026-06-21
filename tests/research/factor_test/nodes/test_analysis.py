# coding: utf-8
"""Analysis-layer node tests: ICAnalyzer, GroupAnalyzer, LongShort.

历史来源: 迁移自 ``QuantNodes/research/factor_test/tests/test_nodes/test_analysis.py`` (C2 收敛).
Phase K1-K4 (2026-06-21): 补 GroupAnalyzer/LongShort 数值验证 + ICAnalyzer 边界.
"""

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
from QuantNodes.research.factor_test.nodes.group_analyzer_node import GroupAnalyzerNode
from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode
from QuantNodes.research.factor_test.nodes.long_short_node import LongShortNode
from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
from QuantNodes.research.factor_test.utils.labels import L_S_COLS, NET_COLS


def _build_analysis_context(synthetic_data, adj_mode=None):
    """构建分析层 context (通过前 5 个节点)."""
    ctx = dict(synthetic_data)
    ctx['LoadData'] = {
        'factor': ctx['factor'],
        'price': ctx['price'],
        'id_citic1': ctx['id_citic1'],
        'mv_float': ctx['mv_float'],
        'st': ctx['st'],
        'suspend': ctx['suspend'],
        'ud_limit': ctx['ud_limit'],
        'ipo_days': ctx['ipo_days'],
        'index_cp': ctx['index_cp'],
        'stklist': ctx['stklist'],
        'trade_dt': ctx['trade_dt'],
        '_loader': None,
    }
    n3 = TradabilityFilterNode(config={
        'tradable': {'no_st': True, 'no_suspended': True},
    })
    ctx['TradabilityFilter'] = n3.execute(context=ctx)

    mode = adj_mode or ['M', 'end']
    n4 = AdjustDateNode(config={
        'adj_date_beg': 20260101, 'adj_date_end': 20260630,
        'adj_mode': mode,
    })
    ctx['AdjustDate'] = n4.execute(context=ctx)

    n5 = FactorPreprocessNode(config={
        'missing': '', 'extreme': 'median', 'norm': 'zscore',
    })
    ctx['FactorPreprocess'] = n5.execute(context=ctx)
    ctx['FactorNeutralize'] = ctx['FactorPreprocess']
    return ctx


# ── ICAnalyzerNode ─────────────────────────────────────────────

class TestICAnalyzerNode:

    def test_basic_ic(self, synthetic_data):
        """基础 IC 计算."""
        ctx = _build_analysis_context(synthetic_data)
        n = ICAnalyzerNode(config={'min_group_size': 5})
        result = n.execute(context=ctx)
        assert 'ic' in result
        assert 'ic_result' in result
        assert isinstance(result['ic'], pd.Series)

    def test_ic_result_has_metrics(self, synthetic_data):
        """IC 结果包含标准指标."""
        ctx = _build_analysis_context(synthetic_data)
        n = ICAnalyzerNode(config={'min_group_size': 5})
        result = n.execute(context=ctx)
        ic_result = result['ic_result']
        assert 'IC均值' in ic_result
        assert 'ICIR' in ic_result

    def test_rank_ic_exists(self, synthetic_data):
        """Rank IC 存在."""
        ctx = _build_analysis_context(synthetic_data)
        n = ICAnalyzerNode(config={'min_group_size': 5})
        result = n.execute(context=ctx)
        assert 'rank_ic' in result
        assert isinstance(result['rank_ic'], pd.Series)

    def test_ic_analyzer_fallback_to_preprocess(self, synthetic_data):
        """当 FactorNeutralize 不存在时回退到 FactorPreprocess."""
        ctx = _build_analysis_context(synthetic_data)
        del ctx['FactorNeutralize']
        n = ICAnalyzerNode(config={'min_group_size': 5})
        result = n.execute(context=ctx)
        assert 'ic' in result

    def test_ic_analyzer_no_factor_raises(self, synthetic_data):
        """无因子数据时抛出."""
        ctx = _build_analysis_context(synthetic_data)
        ctx['FactorPreprocess'] = None
        ctx['FactorNeutralize'] = None
        n = ICAnalyzerNode(config={'min_group_size': 5})
        with pytest.raises(Exception):
            n.execute(context=ctx)

    # ── K4: IC 边界 (2026-06-21) ──

    def test_ic_values_in_range(self, synthetic_data):
        """IC 值必须在 [-1, 1] 内 (correlation 数学界限)."""
        ctx = _build_analysis_context(synthetic_data)
        result = ICAnalyzerNode(config={'min_group_size': 5}).execute(context=ctx)
        for key in ('ic', 'rank_ic'):
            s = result[key].dropna()
            if not s.empty:
                assert s.min() >= -1.0 - 1e-9
                assert s.max() <= 1.0 + 1e-9

    def test_constant_factor_ic_is_nan(self, synthetic_data):
        """常数因子的 IC 应为 NaN (std=0, corr undefined)."""
        ctx = _build_analysis_context(synthetic_data)
        factor = ctx['FactorPreprocess'].copy()
        factor.values[~np.isnan(factor.values)] = 1.0
        ctx['FactorPreprocess'] = factor
        ctx['FactorNeutralize'] = factor
        result = ICAnalyzerNode(config={'min_group_size': 5}).execute(context=ctx)
        ic = result['ic']
        assert ic.isna().sum() >= len(ic) - 1

    def test_min_group_size_filters_sparse_dates(self, synthetic_data):
        """min_group_size > universe → IC 全 NaN."""
        ctx = _build_analysis_context(synthetic_data)
        r_small = ICAnalyzerNode(config={'min_group_size': 5}).execute(context=ctx)
        r_big = ICAnalyzerNode(config={'min_group_size': 100}).execute(context=ctx)
        assert r_big['ic'].dropna().shape[0] <= r_small['ic'].dropna().shape[0]
        assert r_big['ic'].dropna().empty


# ── GroupAnalyzerNode ──────────────────────────────────────────

class TestGroupAnalyzerNode:

    def test_group_analyzer_no_factor_raises(self, synthetic_data):
        """无因子数据时抛出."""
        ctx = _build_analysis_context(synthetic_data)
        ctx['FactorPreprocess'] = None
        ctx['FactorNeutralize'] = None
        n = GroupAnalyzerNode(config={
            'groups': 5, 'factor_direction': 1,
            'floor_mode': 'group', 'hedge': 'equal',
        })
        with pytest.raises(Exception):
            n.execute(context=ctx)

    # ── K1: 分组数值验证 (2026-06-21) ──

    def _run(self, synthetic_data, n_groups=5, direction=1, hedge='equal'):
        ctx = _build_analysis_context(synthetic_data)
        result = GroupAnalyzerNode(config={
            'groups': n_groups, 'factor_direction': direction,
            'floor_mode': 'group', 'hedge': hedge,
        }).execute(context=ctx)
        return result, ctx

    def test_output_keys_complete(self, synthetic_data):
        """16 个 output key 必须全部存在 (回归保护)."""
        result, _ = self._run(synthetic_data)
        expected = {
            'adjust_dates', 'fac_group', 'group_num', 'group_ret',
            'group_winratio', 'group_winloss',
            'daily_net_simp', 'daily_net_cmp',
            'daily_excnet_simp', 'daily_excnet_cmp',
            'group_eva_abs', 'group_eva_exc',
            'group_eva_abs_yearly', 'group_eva_exc_yearly',
            'turnover', 'n_groups',
        }
        assert set(result.keys()) == expected

    @pytest.mark.parametrize('n_groups', [3, 5, 10])
    def test_group_shapes(self, synthetic_data, n_groups):
        """分组数对应输出列数 + n_groups."""
        result, _ = self._run(synthetic_data, n_groups=n_groups)
        assert result['n_groups'] == n_groups
        assert result['group_ret'].shape[1] == n_groups
        assert result['group_num'].shape[1] == n_groups
        assert list(result['group_ret'].columns) == list(range(1, n_groups + 1))

    def test_group_labels_within_range(self, synthetic_data):
        """fac_group 标签必须在 [1, n_groups]."""
        result, _ = self._run(synthetic_data, n_groups=5)
        labels = result['fac_group'].stack().dropna().unique()
        assert set(labels).issubset({1, 2, 3, 4, 5})

    def test_group_num_bounded_by_universe(self, synthetic_data):
        """每个调仓期 group_num 之和 ≤ 当期有效因子数."""
        result, ctx = self._run(synthetic_data, n_groups=5)
        factor = ctx['FactorPreprocess']
        gn = result['group_num'].dropna(how='all')
        if gn.empty:
            pytest.skip('无有效分组日')
        date = gn.index[0]
        assert gn.loc[date].sum() <= factor.loc[date].notna().sum()

    def test_net_starts_at_one(self, synthetic_data):
        """日度净值首日 == 1.0."""
        result, _ = self._run(synthetic_data)
        net = result['daily_net_simp'].dropna(how='all')
        if net.empty:
            pytest.skip('无净值')
        np.testing.assert_allclose(net.iloc[0].values, 1.0, atol=1e-10)

    def test_excess_net_starts_at_one(self, synthetic_data):
        """超额净值首日 == 1.0."""
        result, _ = self._run(synthetic_data, hedge='equal')
        exc = result['daily_excnet_simp'].dropna(how='all')
        if exc.empty:
            pytest.skip('无超额净值')
        np.testing.assert_allclose(exc.iloc[0].values, 1.0, atol=1e-9)

    def test_eva_abs_required_metrics(self, synthetic_data):
        """group_eva_abs 必须含 11 个标准指标."""
        result, _ = self._run(synthetic_data)
        required = {'AnnualRt', 'AccumRt', 'SR', 'MDD', 'WinRatio',
                    'WinLossRatio', 'Calmar', 'MDD_date',
                    'MDD_lastdays', 'MDD_recoverdays', 'Periods'}
        assert required.issubset(set(result['group_eva_abs'].index))

    def test_hedge_empty_equals_abs_net(self, synthetic_data):
        """hedge='' (走 None 分支) 时 daily_excnet_simp == daily_net_simp."""
        ctx = _build_analysis_context(synthetic_data)
        result = GroupAnalyzerNode(config={
            'groups': 5, 'factor_direction': 1,
            'floor_mode': 'group', 'hedge': '',
        }).execute(context=ctx)
        pd.testing.assert_frame_equal(
            result['daily_excnet_simp'], result['daily_net_simp']
        )

    def test_direction_does_not_affect_group_ret(self, synthetic_data):
        """factor_direction 不改变 group_ret (LongShort 才区分多空)."""
        ctx = _build_analysis_context(synthetic_data)
        r_pos = GroupAnalyzerNode(config={
            'groups': 5, 'factor_direction': 1,
            'floor_mode': 'group', 'hedge': 'equal',
        }).execute(context=dict(ctx))
        r_neg = GroupAnalyzerNode(config={
            'groups': 5, 'factor_direction': -1,
            'floor_mode': 'group', 'hedge': 'equal',
        }).execute(context=dict(ctx))
        pd.testing.assert_frame_equal(r_pos['group_ret'], r_neg['group_ret'])


# ── LongShortNode ──────────────────────────────────────────────

class TestLongShortNode:

    def test_longshort_no_group_raises(self, synthetic_data):
        """无分组数据时抛出."""
        ctx = _build_analysis_context(synthetic_data)
        ctx['GroupAnalyzer'] = None
        n9 = LongShortNode(config={'factor_direction': 1})
        with pytest.raises(Exception):
            n9.execute(context=ctx)

    # ── K2: 多空输出 6 列验证 (2026-06-21) ──

    def _run(self, synthetic_data, direction=1, n_groups=5):
        ctx = _build_analysis_context(synthetic_data)
        ctx['GroupAnalyzer'] = GroupAnalyzerNode(config={
            'groups': n_groups, 'factor_direction': direction,
            'floor_mode': 'group', 'hedge': 'equal',
        }).execute(context=ctx)
        result = LongShortNode(config={
            'factor_direction': direction,
        }).execute(context=ctx)
        return result, ctx

    def test_output_keys(self, synthetic_data):
        """LongShort 输出 5 keys."""
        result, _ = self._run(synthetic_data)
        assert set(result.keys()) == {
            'net', 'eva_total', 'eva_yearly',
            'period_ret', 'longshort_ret',
        }

    def test_net_columns_match_labels(self, synthetic_data):
        """net 5 列 == NET_COLS (多头/空头/多头超额/空头超额/多空)."""
        result, _ = self._run(synthetic_data)
        assert list(result['net'].columns) == NET_COLS
        assert len(NET_COLS) == 5

    def test_period_ret_columns_match_labels(self, synthetic_data):
        """period_ret 3 列 == L_S_COLS."""
        result, _ = self._run(synthetic_data)
        assert list(result['period_ret'].columns) == L_S_COLS
        assert len(L_S_COLS) == 3

    def test_period_ret_diff_equals_longshort(self, synthetic_data):
        """period_ret 第 3 列 (多空) 应 == longshort_ret."""
        result, _ = self._run(synthetic_data)
        ls = result['longshort_ret'].dropna()
        diff = result['period_ret'].iloc[:, 2].dropna()
        common = ls.index.intersection(diff.index)
        if len(common) == 0:
            pytest.skip('无 overlap')
        np.testing.assert_allclose(
            ls.loc[common].astype(float).values,
            diff.loc[common].astype(float).values,
            atol=1e-12,
        )

    def test_direction_swaps_long_short(self, synthetic_data):
        """direction=1 与 direction=-1 时 longshort_ret 互为相反数."""
        r_pos, _ = self._run(synthetic_data, direction=1)
        r_neg, _ = self._run(synthetic_data, direction=-1)
        ls_pos = r_pos['longshort_ret'].dropna()
        ls_neg = r_neg['longshort_ret'].dropna()
        common = ls_pos.index.intersection(ls_neg.index)
        if len(common) == 0:
            pytest.skip('无 overlap')
        np.testing.assert_allclose(
            ls_pos.loc[common].values, -ls_neg.loc[common].values,
            atol=1e-12,
        )

    def test_eva_total_three_columns(self, synthetic_data):
        """eva_total 三列 == L_S_COLS."""
        result, _ = self._run(synthetic_data)
        assert list(result['eva_total'].columns) == L_S_COLS

    def test_net_first_row_one(self, synthetic_data):
        """5 列净值首日均 ≈ 1.0 (绝对/超额/多空净值同起点).

        daily_net_longshort = long - short + 1, 首日 long=short=1 → ls_net=1.
        """
        result, _ = self._run(synthetic_data)
        net = result['net'].dropna(how='all')
        if net.empty:
            pytest.skip('无净值')
        np.testing.assert_allclose(net.iloc[0].values, 1.0, atol=1e-9)

    def test_eva_yearly_has_three_buckets(self, synthetic_data):
        """eva_yearly 含 多头超额/空头超额/多空 三键."""
        result, _ = self._run(synthetic_data)
        assert set(result['eva_yearly'].keys()) == {'多头超额', '空头超额', '多空'}
