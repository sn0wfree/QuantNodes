# coding: utf-8
"""Node 8: 分组分析 / Group Analyzer Node

Migrated from factor_performance.py:361-560 cal_group_ret()
"""

from typing import Union

import numpy as np
import pandas as pd

from QuantNodes.core.node import BaseNode
from QuantNodes.research.factor_test.nodes.configs import GroupAnalyzerNodeConfig
from QuantNodes.research.factor_test.utils.performance_metrics import (
    evaluation, cal_net_simple
)
from QuantNodes.research.factor_test.utils.constants import INDEX_CP_MAPPING


class GroupAnalyzerNode(BaseNode):
    """N 分位分组 + 各组收益/净值/评价

    输入: factor_neutral, price, index_cp
    输出: {fac_group, group_num, group_ret, daily_net_simp, daily_excnet_simp,
            group_eva_abs, group_eva_exc, turnover, ...}
    """

    def __init__(self, name: str = "GroupAnalyzer",
                 config: Union[dict, GroupAnalyzerNodeConfig, None] = None, **kwargs):
        # T0-4: 预先 Union 化
        if isinstance(config, GroupAnalyzerNodeConfig):
            cfg = config
            super().__init__(name, cfg.model_dump(), **kwargs)
        elif isinstance(config, dict) or config is None:
            cfg = GroupAnalyzerNodeConfig.model_validate(config or {})
            super().__init__(name, config, **kwargs)
        else:
            raise TypeError(
                f"config must be dict/None/GroupAnalyzerNodeConfig, got {type(config).__name__}"
            )
        self._groups = cfg.groups
        self._factor_direction = cfg.factor_direction
        self._floor_mode = cfg.floor_mode
        self._hedge = cfg.hedge
        self._hedge_path = cfg.hedge_path

    def _execute(self, input_data=None, **kwargs) -> dict:
        context = kwargs.get('context', {})
        factor_data = context.get('FactorNeutralize') if context.get('FactorNeutralize') is not None else context.get('FactorPreprocess')
        price = context.get('LoadData', {}).get('price')
        index_cp = context.get('LoadData', {}).get('index_cp')

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
            nonan = factor_data.loc[t_i].notna().sum()
            if nonan == 0 or nonan < group:
                if floor_mode == 'group' or i == 0:
                    continue
                elif floor_mode == 'last':
                    fac_group.loc[t_i] = fac_group.iloc[i - 1]
                    continue
            fac_group.loc[t_i] = pd.qcut(
                factor_data.loc[t_i], group,
                labels=range(1, group + 1),
                duplicates='drop'
            )

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
                        group_winloss.loc[t_i, g] = pos / neg * -1 if pd.notna(neg) and neg != 0 else np.nan

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
                        group_daily_ret.loc[group_net.index[1:], g] = group_net.pct_change(fill_method=None).iloc[1:]

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
                    benchmark.loc[cycle.index[1:]] = cycle[valid].mean(axis=1).pct_change(fill_method=None).iloc[1:]
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
            except Exception:
                pass

        return None
