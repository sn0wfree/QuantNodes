# coding: utf-8
"""Node 8: 分组分析 / Group Analyzer Node

Migrated from factor_performance.py:361-560 cal_group_ret()
"""


import logging
from typing import Literal

import numpy as np
import pandas as pd

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import GroupAnalyzerNodeConfig
from QuantNodes.research.factor_test.utils.performance_metrics import (
    evaluation, cal_net_simple
)
from QuantNodes.research.factor_test.utils.constants import INDEX_CP_MAPPING

logger = logging.getLogger(__name__)


FactorKind = Literal["ranked", "discrete"]


def _classify_factor(row: pd.Series, n_groups: int) -> FactorKind:
    """按 dtype + n_unique 判别因子分桶策略。

    Returns:
        "discrete" — bool dtype, 或 n_unique <= 2（捕获 float-cast 二值）
        "ranked"   — 其余（连续或轻度 ties, 走 rank('first') + qcut）

    注: n_unique >= 3 的整数 dtype 因子走 ranked 分支, 因为
    _group_discrete 按 value 比例分配组段, 对 n_unique > n_groups 的
    输入无法产出恰好 n_groups 个组 (会产生 n_unique 个组)。
    """
    n_unique = row.nunique()
    if pd.api.types.is_bool_dtype(row):
        return "discrete"
    if n_unique <= 2:
        return "discrete"
    return "ranked"


def _group_ranked(series: pd.Series, n_groups: int) -> pd.Series:
    """通用 rank('first') + qcut: 处理连续和轻度 ties 因子。

    修复 pd.qcut(duplicates='drop') 在 ties 下抛 "Bin labels must be
    one fewer than the number of bin edges" 的 bug (alpha-004 等场景:
    7 unique × 50 行有大量 ties)。

    对无 ties 的纯连续因子, rank('first') 产出 1..n 唯一序号,
    qcut 在序号上的分位点与原 qcut(series) 上的分位点对应同一组
    元素 (单调变换保序), 行为 bitwise 等价 (零回归)。
    """
    return pd.qcut(
        series.rank(method='first'),
        n_groups,
        labels=range(1, n_groups + 1),
        duplicates='drop'
    )


def _group_discrete(
    series: pd.Series, n_groups: int, date_int: int
) -> pd.Series:
    """bool / 离散因子 (n_unique <= 2): 按 value 比例分配组段 + 内部 seeded shuffle。

    算法:
      1. sorted unique values, 按 count/len*n_groups 比例 round 分配组数
      2. 最后一个 value 用减法 (n_groups - sum(prev)) 强制补齐, 避免累计漂移
      3. np.random.seed(date_int % 2**31) 每日同 seed → 可复现
      4. value 内部 shuffle 后用整数除法 j * n_g // n_v 均分到所属组段

    Args:
        series: 单日 50 只股票的 factor values (dropna 后)
        n_groups: 总组数 (默认 5)
        date_int: yyyymmdd 整数, 用作 shuffle seed

    Returns:
        pd.Series: index=series.index, values ∈ {1, ..., n_groups}
    """
    unique_vals = sorted(series.unique())
    n_total = len(series)
    counts = {v: int((series == v).sum()) for v in unique_vals}

    n_g_list: list[int] = []
    for v in unique_vals[:-1]:
        n_g_list.append(max(1, round(counts[v] / n_total * n_groups)))
    n_g_list.append(max(1, n_groups - sum(n_g_list)))

    starts: list[int] = [1]
    for ng in n_g_list[:-1]:
        starts.append(starts[-1] + ng)

    np.random.seed(int(date_int) % (2**31))
    groups = pd.Series(np.nan, index=series.index)
    for v, ng, g_start in zip(unique_vals, n_g_list, starts):
        if ng <= 0:
            continue
        val_idx = list(series[series == v].index)
        np.random.shuffle(val_idx)
        n_v = len(val_idx)
        for j, idx in enumerate(val_idx):
            g_within = min(ng - 1, j * ng // n_v) if n_v > 0 else 0
            groups[idx] = g_start + g_within
    return groups


class GroupAnalyzerNode(PydanticConfigNode):
    """N 分位分组 + 各组收益/净值/评价

    输入: factor_neutral, price, index_cp
    输出: {fac_group, group_num, group_ret, daily_net_simp, daily_excnet_simp,
            group_eva_abs, group_eva_exc, turnover, ...}
    """

    ConfigSchema = GroupAnalyzerNodeConfig
    _ALIASES = {
        "_groups": "groups",
        "_factor_direction": "factor_direction",
        "_floor_mode": "floor_mode",
        "_hedge": "hedge",
        "_hedge_path": "hedge_path",
    }

    def _execute(self, input_data=None, **kwargs) -> dict:
        context = kwargs.get('context', {})
        factor_data = self._factor_data(context)
        price = self._ctx_load(context, 'price')
        index_cp = self._ctx_load(context, 'index_cp')

        if factor_data is None or price is None:
            raise ValueError("因子或价格数据缺失")

        return self._calc_group_return(
            factor_data, price, index_cp,
            self._groups, self._factor_direction,
            self._floor_mode, self._hedge, self._hedge_path
        )

    def _calc_group_return(self, factor_data, price, index_cp,
                           group, factor_ori, floor_mode, hedge, hedge_path):
        """计算分组收益"""
        adj_dates = factor_data.index.tolist()
        n_groups = group

        # 分组标签
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

        # 各期每组收益
        price_adj = price.loc[price.index.isin(adj_dates)]
        stock_cycle_ret = price_adj.pct_change(fill_method=None).shift(-1)

        group_num = pd.DataFrame(np.nan, index=adj_dates, columns=range(1, n_groups + 1))
        group_ret = group_num.copy()
        group_winratio = group_num.copy()
        group_winloss = group_num.copy()

        for t_i in adj_dates[:-1]:
            if t_i not in fac_group.index or t_i not in stock_cycle_ret.index:
                continue
            fg = fac_group.loc[t_i].dropna()
            if fg.empty:
                continue
            ret_i = stock_cycle_ret.loc[t_i]
            temp = ret_i.groupby(fg).agg(['count', 'mean'])
            for g in range(1, n_groups + 1):
                if g in temp.index:
                    group_num.loc[t_i, g] = temp.loc[g, 'count']
                    group_ret.loc[t_i, g] = temp.loc[g, 'mean']
                    mask = fg[fg == g].index
                    vals = ret_i.reindex(mask).dropna()
                    if len(vals) > 0:
                        group_winratio.loc[t_i, g] = (vals > 0).mean()
                        pos = vals[vals > 0].mean()
                        neg = vals[vals < 0].mean()
                        has_valid_neg = pd.notna(neg) and neg != 0
                        group_winloss.loc[t_i, g] = pos / neg * -1 if has_valid_neg else np.nan

        # 各组日度净值 (单利)
        price_full = price.loc[adj_dates[0]:adj_dates[-1]]
        group_daily_ret = pd.DataFrame(
            np.nan, index=price_full.index, columns=range(1, n_groups + 1)
        )
        group_daily_ret.iloc[0] = 0

        for i in range(len(adj_dates) - 1):
            t_i = adj_dates[i]
            t_ii = adj_dates[i + 1]
            if t_i not in fac_group.index:
                continue
            fg = fac_group.loc[t_i].dropna()
            if fg.empty:
                continue
            cycle_net = price_full.loc[t_i:t_ii] / price_full.loc[t_i]
            for g in range(1, n_groups + 1):
                stocks_g = fg[fg == g].index
                if len(stocks_g) > 0:
                    valid_stocks = [s for s in stocks_g if s in cycle_net.columns]
                    if valid_stocks:
                        group_net = cycle_net[valid_stocks].mean(axis=1)
                        pct_change_values = group_net.pct_change(fill_method=None).iloc[1:]
                        group_daily_ret.loc[group_net.index[1:], g] = pct_change_values

        group_daily_net_cmp = (group_daily_ret + 1).cumprod()
        group_daily_net_simp = group_daily_net_cmp.copy() * np.nan
        for g in range(n_groups):
            group_daily_net_simp.iloc[:, g] = cal_net_simple(
                group_daily_net_cmp.iloc[:, g], adj_dates
            )

        # 对冲基准
        benchmark = self._get_benchmark(hedge, hedge_path, index_cp, price_full,
                                        factor_data, adj_dates)
        if benchmark is not None:
            benchmark_cmp = benchmark / benchmark.iloc[0]
            benchmark_simp = cal_net_simple(benchmark_cmp, adj_dates)

            # 超额净值
            group_daily_excnet_simp = group_daily_net_simp.sub(benchmark_simp.values, axis=0) + 1
            group_daily_excnet_cmp = group_daily_net_cmp.sub(benchmark_cmp.values, axis=0) + 1
        else:
            group_daily_excnet_simp = group_daily_net_simp.copy()
            group_daily_excnet_cmp = group_daily_net_cmp.copy()

        # 评价
        group_eva_abs = pd.DataFrame()
        group_eva_exc = pd.DataFrame()
        group_eva_abs_yearly = {}
        group_eva_exc_yearly = {}

        for g in range(1, n_groups + 1):
            # 换手率
            turn = ((fac_group == g) & (fac_group.diff(-1) != 0)).sum(axis=1) / (
                (fac_group == g).sum(axis=1) * 2
            )
            turn.name = g

            result_g = evaluation(group_daily_net_simp[g], adj_dates)
            result_exc_g = evaluation(group_daily_excnet_simp[g], adj_dates)

            group_eva_abs = pd.concat([group_eva_abs, result_g.iloc[0, 1:].to_frame()], axis=1)
            group_eva_exc = pd.concat([group_eva_exc, result_exc_g.iloc[0, 1:].to_frame()], axis=1)
            group_eva_abs_yearly[g] = result_g.iloc[1:]
            group_eva_exc_yearly[g] = result_exc_g.iloc[1:]

        group_eva_abs.columns = range(1, n_groups + 1)
        group_eva_exc.columns = range(1, n_groups + 1)

        return {
            'adjust_dates': adj_dates,
            'fac_group': fac_group,
            'group_num': group_num,
            'group_ret': group_ret,
            'group_winratio': group_winratio,
            'group_winloss': group_winloss,
            'daily_net_simp': group_daily_net_simp,
            'daily_net_cmp': group_daily_net_cmp,
            'daily_excnet_simp': group_daily_excnet_simp,
            'daily_excnet_cmp': group_daily_excnet_cmp,
            'group_eva_abs': group_eva_abs,
            'group_eva_exc': group_eva_exc,
            'group_eva_abs_yearly': group_eva_abs_yearly,
            'group_eva_exc_yearly': group_eva_exc_yearly,
            'turnover': pd.DataFrame() if 'turn' not in dir() else turn,
            'n_groups': n_groups,
        }

    def _get_benchmark(self, hedge, hedge_path, index_cp, price_full, factor_data, adj_dates):
        """获取对冲基准净值"""
        if hedge == 'equal':
            # 等权基准: 所有有因子值的股票收益等权
            benchmark = pd.Series(np.nan, index=price_full.index)
            benchmark.iloc[0] = 0
            for i in range(len(adj_dates) - 1):
                t_i = adj_dates[i]
                t_ii = adj_dates[i + 1]
                if t_i not in factor_data.index:
                    continue
                stocks_with_factor = factor_data.loc[t_i].dropna().index
                cycle = price_full.loc[t_i:t_ii] / price_full.loc[t_i]
                valid = [s for s in stocks_with_factor if s in cycle.columns]
                if valid:
                    benchmark.loc[cycle.index[1:]] = (
                        cycle[valid].mean(axis=1).pct_change(fill_method=None).iloc[1:]
                    )
            return (benchmark + 1).cumprod()

        elif hedge in INDEX_CP_MAPPING and index_cp is not None:
            key = INDEX_CP_MAPPING[hedge]
            if key in index_cp.columns:
                return index_cp.loc[price_full.index, key]

        elif hedge == 'custom' and hedge_path:
            try:
                from QuantNodes.research.factor_test.utils.data_loader import DataLoader
                loader = DataLoader()
                custom = loader.load_custom(('', hedge_path))
                return custom.iloc[:, 0]
            except Exception as e:
                logger.warning("GroupAnalyzerNode: 自定义对冲加载失败 (%s): %s", hedge_path, e)

        return None
