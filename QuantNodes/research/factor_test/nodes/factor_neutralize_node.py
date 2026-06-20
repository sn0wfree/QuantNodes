# coding: utf-8
"""Node 6: 因子中性化 / Factor Neutralize Node

Migrated from factor_utils.py:534-625 neutralize()
"""

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import NeutralizeNodeConfig

logger = logging.getLogger(__name__)


class FactorNeutralizeNode(PydanticConfigNode):
    """行业/风险因子中性化 (OLS 残差)

    输入: factor_std, industry, risk_factors
    输出: factor_neutral
    """

    ConfigSchema = NeutralizeNodeConfig
    _ALIASES = {
        "_if_industry": "industry_neutral",
        "_if_risk": "risk_neutral",
        "_risk_factor_specs": "risk_factors",
    }

    def _execute(self, input_data=None, **kwargs) -> pd.DataFrame:
        context = kwargs.get('context', {})
        factor_std = context.get('FactorPreprocess')
        if factor_std is None:
            raise ValueError("因子预处理数据缺失")

        if not self._if_industry and not self._if_risk:
            return factor_std

        industry = self._ctx_load(context, 'id_citic1')
        if industry is None and self._if_industry:
            raise ValueError("行业数据缺失")

        # 加载风险因子
        risk_data = []
        if self._if_risk and self._risk_factor_specs:
            loader = self._ctx_load(context, '_loader')
            for file_key, factor_key in self._risk_factor_specs:
                try:
                    if file_key == 'risk_factor.h5':
                        rf = loader.load_h5(file_key, factor_key)
                    else:
                        rf = loader.load_custom((file_key, factor_key))
                    if loader.valid_shape(rf):
                        rf = loader.add_index(rf)
                    risk_data.append(rf)
                except Exception as e:
                    logger.warning(f"加载风险因子 {factor_key} 失败: {e}")

        return self._neutralize(factor_std, self._if_industry, industry,
                                self._if_risk, risk_data)

    def _neutralize(self, factor_i, if_industry, industry, if_risk, risk_data):
        """中性化处理"""
        factor_i = factor_i.copy()
        if industry is not None:
            industry = industry.copy().replace(np.nan, 0)

        factor_neut = factor_i * np.nan
        date_factor_i = factor_i.index.values

        if if_industry and if_risk:
            # 行业 + 风险中性
            for date_j in date_factor_i:
                if factor_i.loc[date_j].notna().sum() > 0:
                    ind_j = industry.loc[date_j]
                    dum_ind = pd.get_dummies(ind_j)
                    dum_ind = dum_ind.loc[:, dum_ind.sum() > 0]
                    X = dum_ind.copy()
                    for rf in risk_data:
                        if date_j in rf.index:
                            X = pd.merge(X, rf.loc[date_j].to_frame().T,
                                         left_index=True, right_index=True,
                                         suffixes=('', '_rf'))
                    lm_data = pd.merge(
                        factor_i.loc[date_j].to_frame(), X,
                        left_index=True, right_index=True,
                        suffixes=('_y', '_x')
                    ).dropna()
                    if len(lm_data) > X.shape[1]:
                        model = sm.OLS(lm_data.iloc[:, 0].values,
                                       sm.add_constant(lm_data.iloc[:, 1:].values))
                        resid = model.fit().resid
                        factor_neut.loc[date_j, lm_data.index.values] = resid

        elif if_industry:
            # 仅行业中性
            for date_j in date_factor_i:
                if factor_i.loc[date_j].notna().sum() > 0:
                    ind_j = industry.loc[date_j]
                    dum_ind = pd.get_dummies(ind_j)
                    dum_ind = dum_ind.loc[:, dum_ind.sum() > 0]
                    lm_data = pd.merge(
                        factor_i.loc[date_j].to_frame(), dum_ind,
                        left_index=True, right_index=True,
                        suffixes=('_y', '_x')
                    ).dropna()
                    if len(lm_data) > dum_ind.shape[1]:
                        model = sm.OLS(lm_data.iloc[:, 0].values,
                                       sm.add_constant(lm_data.iloc[:, 1:].values))
                        resid = model.fit().resid
                        factor_neut.loc[date_j, lm_data.index.values] = resid

        elif if_risk:
            # 仅风险中性
            for date_j in date_factor_i:
                if factor_i.loc[date_j].notna().sum() > 0:
                    X = pd.DataFrame()
                    for rf in risk_data:
                        if date_j in rf.index:
                            X = pd.concat([X, rf.loc[date_j].to_frame().T], axis=1)
                    if not X.empty:
                        X = sm.add_constant(X)
                        lm_data = pd.merge(
                            factor_i.loc[date_j].to_frame(), X,
                            left_index=True, right_index=True,
                            suffixes=('_y', '_x')
                        ).dropna()
                        if len(lm_data) > X.shape[1]:
                            model = sm.OLS(lm_data.iloc[:, 0].values,
                                           lm_data.iloc[:, 1:].values)
                            resid = model.fit().resid
                            factor_neut.loc[date_j, lm_data.index.values] = resid

        return factor_neut
