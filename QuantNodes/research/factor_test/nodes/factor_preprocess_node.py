# coding: utf-8
"""Node 5: 因子预处理 / Factor Preprocess Node

Migrated from factor_utils.py:310-532 preprocess_onePeriod + preprocess_factor
Phase 1: 原样迁移保持行为一致性
Phase 2: 逐步替换为 QuantNodes section_ops 算子
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from QuantNodes.core.node import BaseNode
from QuantNodes.research.factor_test.nodes.configs import PreprocessNodeConfig


class FactorPreprocessNode(BaseNode):
    """因子预处理: 缺失值填充 + 去极值 + 标准化

    输入: factor, tradable, adj_dates, industry
    输出: factor_std (预处理后的因子, 仅调仓日)
    """

    def __init__(self, name: str = "FactorPreprocess",
                 config: Union[dict, PreprocessNodeConfig, None] = None, **kwargs):
        # T0-4: 预先 Union 化
        if isinstance(config, PreprocessNodeConfig):
            cfg = config
            super().__init__(name, cfg.model_dump(), **kwargs)
        elif isinstance(config, dict) or config is None:
            cfg = PreprocessNodeConfig.model_validate(config or {})
            super().__init__(name, config, **kwargs)
        else:
            raise TypeError(
                f"config must be dict/None/PreprocessNodeConfig, got {type(config).__name__}"
            )
        # T0-2: 3 隐式默认从 Pydantic 字段读取 (mad_n=5.0, pct_low=0.025, pct_high=0.975)
        self._missing = cfg.missing
        self._extreme = cfg.extreme
        self._norm = cfg.norm
        self._mad_n = cfg.mad_n
        self._pct_low = cfg.pct_low
        self._pct_high = cfg.pct_high
        # M12: 自定义行业名称映射 (覆盖全局默认)
        self._i18n_name_map = cfg.i18n_name_map if cfg.i18n_name_map is not None else None

    def _execute(self, input_data=None, **kwargs) -> pd.DataFrame:
        context = kwargs.get('context', {})
        factor = context.get('LoadData', {}).get('factor')
        tradable = context.get('TradabilityFilter')
        adj_dates = context.get('AdjustDate')
        sample = context.get('SamplePoolFilter')

        if factor is None:
            raise ValueError("因子数据缺失")
        if tradable is None:
            tradable = pd.DataFrame(np.ones_like(factor.values))

        # 合并可交易性
        tradable_factor = factor * tradable

        # 提取调仓日的因子值
        adj_date_values = adj_dates.iloc[:, 0].values if isinstance(adj_dates, pd.DataFrame) else adj_dates
        tradable_factor_adj = tradable_factor.loc[tradable_factor.index.isin(adj_date_values)]
        tradable_adj = tradable.loc[tradable.index.isin(adj_date_values)]

        # 行业数据
        industry = context.get('LoadData', {}).get('id_citic1')
        if industry is not None:
            industry_adj = industry.loc[industry.index.isin(adj_date_values)]
        else:
            industry_adj = None

        # 逐日预处理
        method = {'missing': self._missing, 'extreme': self._extreme, 'norm': self._norm}
        result = tradable_factor_adj.apply(
            lambda x: self._preprocess_one_period(
                x, tradable_adj, industry_adj, method
            ),
            axis=1
        )

        # 清理索引
        if hasattr(result.index, 'get_level_values'):
            result.index = result.index.get_level_values(0).values
        if hasattr(result.columns, 'get_level_values'):
            result.columns = result.columns.get_level_values(0).values

        return result

    def _preprocess_one_period(self, factor_i, tradable_adj, industry_adj, method):
        """对单日因子值进行预处理"""
        factor_i_new_all = factor_i.copy()

        # 获取当日可交易和行业数据
        date_key = factor_i.name[0] if hasattr(factor_i.name, '__len__') else factor_i.name

        if tradable_adj is not None and date_key in tradable_adj.index:
            tradable_i = tradable_adj.loc[date_key]
        else:
            return factor_i_new_all

        if industry_adj is not None and date_key in industry_adj.index:
            ind_i = industry_adj.loc[date_key]
        else:
            ind_i = pd.Series(np.nan, index=factor_i.index)

        # 合并数据
        df = pd.DataFrame({
            'factor': factor_i,
            'ind': ind_i,
            'tradable': tradable_i,
        }, index=factor_i.index)

        # 剔除不可交易
        df = df.loc[df['tradable'].notna() & (df['tradable'] != 0)]
        if df.empty:
            return factor_i_new_all

        df['ind'] = df['ind'].replace(np.nan, 0)
        df['factor_filled'] = df['factor'].copy()

        # 1. 缺失值处理
        if method['missing'] == 'ind_avg':
            has_ind = df['ind'] > 0
            if has_ind.any():
                df.loc[has_ind, 'factor_filled'] = (
                    df.loc[has_ind].groupby('ind')['factor']
                    .transform(lambda x: x.fillna(x.mean()))
                )

        # 2. 去极值
        if method['extreme'] == 'median':
            n = self._mad_n  # M5: 可调
            d_m = df['factor_filled'].dropna().median()
            d_mad = (df['factor_filled'] - d_m).abs().dropna().median()
            df['factor_filled'] = df['factor_filled'].clip(d_m - n * d_mad, d_m + n * d_mad)
        elif method['extreme'] == 'pct_shrink':
            q1 = df['factor_filled'].quantile(self._pct_low)   # M5: 可调
            q2 = df['factor_filled'].quantile(self._pct_high)  # M5: 可调
            df['factor_filled'] = df['factor_filled'].clip(q1, q2)

        # 3. 标准化
        if method['norm'] == 'zscore':
            f_mean = df['factor_filled'].dropna().mean()
            f_std = df['factor_filled'].dropna().std(ddof=1)
            if f_std > 0:
                df['factor_filled'] = (df['factor_filled'] - f_mean) / f_std
        elif method['norm'] == 'norm':
            valid = df['factor_filled'].notna()
            if valid.sum() > 1:
                df.loc[valid, 'rank'] = df.loc[valid, 'factor_filled'].rank(pct=True)
                # 处理 0 和 1 的边界
                ranks = df['rank'].dropna()
                df['rank'] = df['rank'].clip(
                    lower=ranks[ranks > 0].min() * 0.5 if (ranks > 0).any() else 0.01,
                    upper=(ranks[ranks < 1].max() + 1) * 0.5 if (ranks < 1).any() else 0.99
                )
                df['factor_filled'] = scipy_norm.ppf(df['rank'], 0, 1)

        # 写回
        factor_i_new_all.loc[df.index] = df['factor_filled'].values
        return factor_i_new_all
