# coding: utf-8
"""Node 10: 市值行业分层打分 / Factor Score Node

Migrated from factor_performance.py:730-877 score_by_size_ind()
"""

from typing import Union

import numpy as np
import pandas as pd

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import ScoreNodeConfig
from QuantNodes.research.factor_test.utils.performance_metrics import evaluation, cal_net_simple


class FactorScoreNode(PydanticConfigNode):
    """市值行业分层打分 (3 市值组 × 29 中信行业 × N 分位)

    输入: factor_neutral, mv, industry, price
    输出: {fac_group, daily_net, eva, ...}
    """

    ConfigSchema = ScoreNodeConfig

    def __init__(self, name: str = "FactorScore",
                 config: Union[dict, ScoreNodeConfig, None] = None, **kwargs):
        super().__init__(name, config, **kwargs)
        # T0-2: 3 隐式默认从 Pydantic 字段读取
        # (n_industries=29, n_size_groups=3, n_quantile_groups=5)
        self._enabled = self.cfg.enabled
        self._n_industries = self.cfg.n_industries
        self._n_size_groups = self.cfg.n_size_groups
        self._n_quantile_groups = self.cfg.n_quantile_groups

    def _execute(self, input_data=None, **kwargs) -> dict:
        if not self._enabled:
            return {}

        context = kwargs.get('context', {})
        neutralized = context.get('FactorNeutralize')
        factor_data = (
            neutralized if neutralized is not None
            else context.get('FactorPreprocess')
        )
        mv = context.get('LoadData', {}).get('mv_float')
        industry = context.get('LoadData', {}).get('id_citic1')
        price = context.get('LoadData', {}).get('price')
        factor_ori = context.get('GroupAnalyzer', {}).get('factor_direction', 1)

        if factor_data is None or mv is None or industry is None or price is None:
            raise ValueError("市值行业分层打分需要因子、市值、行业、价格数据")

        return self._score_by_size_ind(factor_data, mv, industry, price, factor_ori)

    def _score_by_size_ind(self, factor_data, mv, industry, price, factor_ori):
        """市值行业分层打分"""
        # H15: 全部可配置 (config.get 已在 __init__)
        group = self._n_quantile_groups
        n_size = self._n_size_groups
        n_ind = self._n_industries
        adj_dates = factor_data.index.tolist()

        # 对齐
        mv_adj = mv.loc[mv.index.isin(adj_dates)]
        ind_adj = industry.loc[industry.index.isin(adj_dates)]
        ind_adj = ind_adj.astype('int')

        # 因子分组
        fac_group = factor_data.copy() * np.nan
        mv_group = mv.copy() * np.nan

        def my_qcut(x, n):
            if len(x.dropna().unique()) >= (n - 1):
                return pd.qcut(x.rank(method='first'), n, labels=range(1, n + 1), duplicates='drop')
            return x * np.nan

        for i in range(len(factor_data)):
            t_i = factor_data.index[i]
            nonan = factor_data.loc[t_i].notna().sum()
            # H15: 最低有效股票数 = 市值组数 × 行业数 × 分位组数
            if nonan == 0 or nonan < n_size * n_ind * group:
                if i > 0:
                    fac_group.loc[t_i] = fac_group.iloc[i - 1]
                continue

            mv_group.loc[t_i] = pd.qcut(
                mv_adj.loc[t_i], n_size, labels=range(1, n_size + 1),
            )
            fac_group.loc[t_i] = factor_data.loc[t_i].groupby(
                [mv_group.loc[t_i], ind_adj.loc[t_i]]
            ).apply(lambda x: my_qcut(x, group))

        # 计算各组净值
        price_full = price.loc[adj_dates[0]:adj_dates[-1]]
        group_daily_ret = pd.DataFrame(np.nan, index=price_full.index, columns=range(1, group + 1))
        group_daily_ret.iloc[0] = 0

        for i in range(len(adj_dates) - 1):
            t_i = adj_dates[i]
            t_ii = adj_dates[i + 1]
            if t_i not in fac_group.index:
                continue
            fg = fac_group.loc[t_i].dropna()
            if fg.empty:
                continue
            cycle = price_full.loc[t_i:t_ii] / price_full.loc[t_i]
            for g in range(1, group + 1):
                stocks = fg[fg == g].index
                valid = [s for s in stocks if s in cycle.columns]
                if valid:
                    gn = cycle[valid].mean(axis=1)
                    group_daily_ret.loc[gn.index[1:], g] = gn.pct_change(fill_method=None).iloc[1:]

        group_daily_net_cmp = (group_daily_ret + 1).cumprod()
        group_daily_net_simp = group_daily_net_cmp.copy() * np.nan
        for g in range(group):
            group_daily_net_simp.iloc[:, g] = cal_net_simple(
                group_daily_net_cmp.iloc[:, g], adj_dates
            )

        # 多空净值
        if factor_ori == 1:
            long_n, short_n = group, 1
        else:
            long_n, short_n = 1, group

        group_daily_net_simp['longshort'] = (
            group_daily_net_simp[long_n] - group_daily_net_simp[short_n] + 1
        )

        # 评价
        eva = pd.DataFrame()
        eva_yearly = {}
        for col in group_daily_net_simp.columns:
            eva_g = evaluation(group_daily_net_simp[col], adj_dates)
            eva = pd.concat([eva, eva_g.iloc[0, :].to_frame()], axis=1)
            eva_yearly[col] = eva_g.iloc[1:] if len(eva_g) > 1 else pd.DataFrame()

        eva.columns = group_daily_net_simp.columns.tolist()
        if 'Year' in eva.index:
            eva = eva.drop('Year')

        return {
            'fac_group': fac_group,
            'daily_net_simp': group_daily_net_simp,
            'daily_net_cmp': group_daily_net_cmp,
            'eva': eva,
            'eva_yearly': eva_yearly,
        }
