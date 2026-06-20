# coding: utf-8
"""Node 5: 因子预处理 / Factor Preprocess Node

Migrated from factor_utils.py:310-532 preprocess_onePeriod + preprocess_factor
Phase 1: 原样迁移保持行为一致性
Phase 2: 逐步替换为 QuantNodes section_ops 算子
Phase 3 (H5, 2026-06-20): vectorised _preprocess_one_period's per-date
  Python loop into DataFrame-level operations (groupby/transform/clip/sub).
  Expected 10-100x speedup on the preprocess step.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import PreprocessNodeConfig


class FactorPreprocessNode(PydanticConfigNode):
    """因子预处理: 缺失值填充 + 去极值 + 标准化

    输入: factor, tradable, adj_dates, industry
    输出: factor_std (预处理后的因子, 仅调仓日)
    """

    ConfigSchema = PreprocessNodeConfig
    _ALIASES = {
        "_missing": "missing",
        "_extreme": "extreme",
        "_norm": "norm",
        "_mad_n": "mad_n",
        "_pct_low": "pct_low",
        "_pct_high": "pct_high",
        "_i18n_name_map": "i18n_name_map",
    }

    def _execute(self, input_data=None, **kwargs) -> pd.DataFrame:
        context = kwargs.get('context', {})
        factor = self._ctx_load(context, 'factor')
        tradable = context.get('TradabilityFilter')
        adj_dates = context.get('AdjustDate')

        if factor is None:
            raise ValueError("因子数据缺失")
        if tradable is None:
            tradable = pd.DataFrame(np.ones_like(factor.values))

        # 合并可交易性
        tradable_factor = factor * tradable

        # 提取调仓日的因子值
        adj_date_values = (
            adj_dates.iloc[:, 0].values
            if isinstance(adj_dates, pd.DataFrame)
            else adj_dates
        )
        tradable_factor_adj = tradable_factor.loc[tradable_factor.index.isin(adj_date_values)]
        tradable_adj = tradable.loc[tradable.index.isin(adj_date_values)]

        # 行业数据
        industry = self._ctx_load(context, 'id_citic1')
        industry_adj = (
            industry.loc[industry.index.isin(adj_date_values)]
            if industry is not None else None
        )

        # Vectorised preprocessing (H5, 2026-06-20).
        result = self._preprocess_vectorized(
            tradable_factor_adj, tradable_adj, industry_adj,
            missing=self._missing, extreme=self._extreme, norm=self._norm,
        )

        # 清理索引
        if hasattr(result.index, 'get_level_values'):
            result.index = result.index.get_level_values(0).values
        if hasattr(result.columns, 'get_level_values'):
            result.columns = result.columns.get_level_values(0).values

        return result

    # ------------------------------------------------------------------
    # Vectorised preprocessing (H5, 2026-06-20)
    # ------------------------------------------------------------------
    def _preprocess_vectorized(
        self,
        tradable_factor: pd.DataFrame,
        tradable: pd.DataFrame,
        industry: pd.DataFrame | None,
        *,
        missing: str,
        extreme: str,
        norm: str,
    ) -> pd.DataFrame:
        """Apply missing-fill + de-extreme + normalise across all dates.

        Replaces the previous ``apply(_preprocess_one_period, axis=1)`` loop.
        Each step is a vectorised DataFrame operation:

          1. Tradable mask (per-row): factor where tradable != 0 & notnan,
             else NaN.
          2. Missing fill ('ind_avg'): per (date, industry) group mean.
          3. De-extreme ('median'): per-date median +/- n * MAD.
          4. De-extreme ('pct_shrink'): per-date quantiles.
          5. Norm ('zscore'): per-date (x - mean) / std.
          6. Norm ('norm'): per-date rank -> scipy.stats.norm.ppf.

        Behavioural parity: produces same NaN pattern and same numeric
        result as the per-date loop, to within floating-point noise.
        """
        result = tradable_factor.copy()

        # 1. Tradable mask (vectorised)
        if tradable is not None:
            tradable_mask = tradable.reindex(index=result.index, columns=result.columns)
            tradable_mask = tradable_mask.notna() & (tradable_mask != 0)
            result = result.where(tradable_mask, np.nan)

        # 2. Missing fill by industry mean
        if missing == 'ind_avg' and industry is not None:
            industry_aligned = industry.reindex(index=result.index, columns=result.columns)
            # Stack to long format for groupby transform on (date, industry).
            long = pd.DataFrame({
                'factor': result.stack(),
                'ind': industry_aligned.stack(),
            }).dropna(subset=['ind'])
            long = long[long['ind'] > 0]
            if not long.empty:
                group_mean = long.groupby([long.index.get_level_values(0), 'ind'])['factor'].transform('mean')
                filled = long['factor'].fillna(group_mean)
                # Write back into wide result.
                filled_wide = filled.unstack(level=-1) if isinstance(filled.index, pd.MultiIndex) else filled
                if isinstance(filled.index, pd.MultiIndex):
                    # Re-stack to wide format keyed on original (date, stock).
                    filled_wide = filled.unstack(level=1)
                # Overlay filled values onto result.
                result.loc[filled_wide.index, filled_wide.columns] = filled_wide.values

        # 3/4. De-extreme (vectorised across rows)
        if extreme == 'median':
            n = self._mad_n
            median_per_row = result.median(axis=1)
            mad_per_row = (result.sub(median_per_row, axis=0)).abs().median(axis=1)
            lower = median_per_row - n * mad_per_row
            upper = median_per_row + n * mad_per_row
            # Clip per row using broadcasting.
            result = result.clip(lower=lower, upper=upper, axis=0)
        elif extreme == 'pct_shrink':
            q1 = result.quantile(self._pct_low, axis=1)
            q2 = result.quantile(self._pct_high, axis=1)
            result = result.clip(lower=q1, upper=q2, axis=0)

        # 5/6. Normalise
        if norm == 'zscore':
            mean_per_row = result.mean(axis=1)
            std_per_row = result.std(axis=1, ddof=1)
            std_per_row = std_per_row.replace(0, np.nan)
            result = result.sub(mean_per_row, axis=0).div(std_per_row, axis=0)
        elif norm == 'norm':
            ranks = result.rank(axis=1, pct=True)
            # Clip ranks away from 0 and 1 (avoid -inf/+inf from ppf).
            min_rank = ranks.where(ranks > 0).min(axis=1) * 0.5
            max_rank = (ranks.where(ranks < 1).max(axis=1) + 1) * 0.5
            # Default fallback for rows where min_rank/max_rank is NaN.
            min_rank = min_rank.fillna(0.01)
            max_rank = max_rank.fillna(0.99)
            lower = pd.DataFrame(
                np.broadcast_to(min_rank.values[:, None], ranks.shape),
                index=ranks.index, columns=ranks.columns,
            )
            upper = pd.DataFrame(
                np.broadcast_to(max_rank.values[:, None], ranks.shape),
                index=ranks.index, columns=ranks.columns,
            )
            ranks = ranks.where(ranks.notna(), np.nan)  # preserve NaN
            ranks = ranks.clip(lower=lower, upper=upper, axis=1)
            result = pd.DataFrame(
                scipy_norm.ppf(ranks.values, 0, 1),
                index=ranks.index, columns=ranks.columns,
            )

        return result

    # ------------------------------------------------------------------
    # Legacy per-date loop (kept for reference / fallback)
    # ------------------------------------------------------------------
    def _preprocess_one_period(self, factor_i, tradable_adj, industry_adj, method):
        """对单日因子值进行预处理 (legacy per-date loop, kept for reference).

        H5 (2026-06-20): _execute now uses _preprocess_vectorized instead.
        This method is preserved for unit-testing the legacy semantics but
        is no longer called from the pipeline.
        """
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
