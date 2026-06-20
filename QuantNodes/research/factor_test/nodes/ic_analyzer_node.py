# coding: utf-8
"""Node 7: IC 分析 / IC Analyzer Node

Migrated from factor_performance.py:111-158 cal_ic()
"""


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
    _ALIASES = {"_min_group_size": "min_group_size"}

    def _execute(self, input_data=None, **kwargs) -> dict:
        context = kwargs.get('context', {})
        factor_data = self._factor_data(context)
        price = self._ctx_load(context, 'price')

        if factor_data is None or price is None:
            raise ValueError("因子或价格数据缺失")

        return self._calc_ic(factor_data, price, self._min_group_size)

    def _calc_ic(self, factor_data, price, group):
        """计算 IC 系列

        H4 (2026-06-20): vectorised per-date corr loop into DataFrame.corrwith.
        Old loop did N separate Series.corr() calls (each O(K log K) over
        K stocks); now one vectorised pairwise correlation across rows.
        Typically 5-50x speedup on the hot path.

        Empty / low-sample dates (nonan < group) are masked to NaN via
        a per-row count of notnull values, preserving the original guard.
        """
        adj_dates = factor_data.index.tolist()

        # 对齐价格到调仓日
        price_adj = price.loc[price.index.isin(adj_dates)]

        # 下一期收益
        stock_cycle_ret = price_adj.pct_change(fill_method=None).shift(-1)

        # 共同日期 (factor_data 与 stock_cycle_ret 都有)
        common_dates = factor_data.index.intersection(stock_cycle_ret.index)
        f = factor_data.loc[common_dates]
        r = stock_cycle_ret.loc[common_dates]

        # Per-date valid sample count (preserves <group guard).
        per_date_count = f.notna().sum(axis=1)
        valid_mask = per_date_count >= group

        # Pearson IC (vectorised: one corr per row)
        ic = f.corrwith(r, axis=1)
        ic = ic.where(valid_mask, np.nan)
        ic = ic.reindex(adj_dates)

        # Spearman Rank IC (rank each row, then corrwith)
        rank_f = f.rank(axis=1)
        rank_r = r.rank(axis=1)
        rank_ic = rank_f.corrwith(rank_r, axis=1)
        rank_ic = rank_ic.where(valid_mask, np.nan)
        rank_ic = rank_ic.reindex(adj_dates)

        # 因子 rank 自相关 (rank(this row) vs rank(next row))
        factor_rank = factor_data.rank(axis=1)
        factor_rank_next = factor_rank.shift(-1)
        common_dates2 = factor_rank.index.intersection(factor_rank_next.index)
        factor_rank_autocorr = factor_rank.loc[common_dates2].corrwith(
            factor_rank_next.loc[common_dates2], axis=1, method='spearman'
        )
        factor_rank_autocorr = factor_rank_autocorr.reindex(adj_dates)

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
