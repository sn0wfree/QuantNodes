# coding: utf-8
"""Node 7: IC 分析 / IC Analyzer Node

Migrated from factor_performance.py:111-158 cal_ic()
"""

from typing import Union

import numpy as np
import pandas as pd

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import ICAnalyzerNodeConfig


class ICAnalyzerNode(PydanticConfigNode):
    """计算 IC / Rank IC / ICIR / 因子 rank 自相关性

    输入: factor_neutral, price
    输出: {ic, rank_ic, ic_result, rank_ic_result, factor_rank_autocorr}
    """

    ConfigSchema = ICAnalyzerNodeConfig

    def __init__(self, name: str = "ICAnalyzer",
                 config: Union[dict, ICAnalyzerNodeConfig, None] = None, **kwargs):
        super().__init__(name, config, **kwargs)
        self._min_group_size = self.cfg.min_group_size

    def _execute(self, input_data=None, **kwargs) -> dict:
        context = kwargs.get('context', {})
        neutralized = context.get('FactorNeutralize')
        factor_data = (
            neutralized if neutralized is not None
            else context.get('FactorPreprocess')
        )
        price = context.get('LoadData', {}).get('price')

        if factor_data is None or price is None:
            raise ValueError("因子或价格数据缺失")

        return self._calc_ic(factor_data, price, self._min_group_size)

    def _calc_ic(self, factor_data, price, group):
        """计算 IC 系列"""
        adj_dates = factor_data.index.tolist()

        # 对齐价格到调仓日
        price_adj = price.loc[price.index.isin(adj_dates)]

        # 下一期收益
        stock_cycle_ret = price_adj.pct_change(fill_method=None).shift(-1)

        ic = pd.Series(np.nan, index=adj_dates, dtype=float)
        rank_ic = pd.Series(np.nan, index=adj_dates, dtype=float)
        factor_rank_autocorr = pd.Series(np.nan, index=adj_dates, dtype=float)

        factor_rank = factor_data.rank(axis=1)
        factor_rank_next = factor_rank.shift(-1)

        for t_i in adj_dates:
            nonan = factor_data.loc[t_i].notna().sum()
            if nonan == 0 or nonan < group:
                continue

            # Pearson IC
            if t_i in stock_cycle_ret.index:
                ic.loc[t_i] = factor_data.loc[t_i].corr(stock_cycle_ret.loc[t_i])

            # Spearman Rank IC
            rank_ic.loc[t_i] = factor_data.loc[t_i].corr(
                stock_cycle_ret.loc[t_i] if t_i in stock_cycle_ret.index else pd.Series(),
                method='spearman'
            )

            # 因子 rank 自相关
            if t_i in factor_rank_next.index:
                factor_rank_autocorr.loc[t_i] = factor_rank.loc[t_i].corr(
                    factor_rank_next.loc[t_i], method='spearman'
                )

        # 评价指标
        ic_result = pd.Series([
            ic.mean(), ic.std(ddof=1),
            ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) != 0 else np.nan,
            ic.mean() / ic.std(ddof=1) * np.sqrt(ic.notna().sum() - 1)
            if ic.std(ddof=1) != 0 else np.nan,
            ((ic > 0).sum() / ic.count()) if ic.count() > 0 else np.nan,
            ((ic < 0).sum() / ic.count()) if ic.count() > 0 else np.nan,
        ], index=['IC均值', 'IC标准差', 'ICIR', 'IC_T值', 'IC为正比例', 'IC为负比例'])

        rank_ic_result = pd.Series([
            rank_ic.mean(), rank_ic.std(ddof=1),
            rank_ic.mean() / rank_ic.std(ddof=1) if rank_ic.std(ddof=1) != 0 else np.nan,
            rank_ic.mean() / rank_ic.std(ddof=1) * np.sqrt(rank_ic.notna().sum() - 1)
            if rank_ic.std(ddof=1) != 0 else np.nan,
            ((rank_ic > 0).sum() / rank_ic.count()) if rank_ic.count() > 0 else np.nan,
            ((rank_ic < 0).sum() / rank_ic.count()) if rank_ic.count() > 0 else np.nan,
        ], index=[
            'rankIC均值', 'rankIC标准差', 'rankICIR',
            'rankIC_T值', 'rankIC为正比例', 'rankIC为负比例',
        ])

        return {
            'ic': ic,
            'rank_ic': rank_ic,
            'ic_result': ic_result,
            'rank_ic_result': rank_ic_result,
            'factor_rank_autocorr': factor_rank_autocorr,
        }
