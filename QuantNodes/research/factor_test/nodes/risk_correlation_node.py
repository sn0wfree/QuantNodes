# coding: utf-8
"""Node 11: 风险因子相关性 / Risk Correlation Node

Migrated from factor_performance.py:879-937 corr_riskfactor()
Security: exec()/eval() eliminated.
"""

import logging

import numpy as np
import pandas as pd

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import RiskCorrelationNodeConfig

logger = logging.getLogger(__name__)


class RiskCorrelationNode(PydanticConfigNode):
    """与风险因子的 Spearman 秩相关 + 稳定系数

    输入: factor_neutral, risk_factors
    输出: {mean: DataFrame, stability: DataFrame}
    """

    ConfigSchema = RiskCorrelationNodeConfig
    _ALIASES = {"_factors": "factors"}

    def _execute(self, input_data=None, **kwargs) -> dict:
        context = kwargs.get('context', {})
        neutralized = context.get('FactorNeutralize')
        factor_data = (
            neutralized if neutralized is not None
            else context.get('FactorPreprocess')
        )
        if factor_data is None:
            raise ValueError("因子数据缺失")

        loader = context.get('LoadData', {}).get('_loader')
        if loader is None:
            raise ValueError("数据加载器缺失")

        return self._calc_risk_correlation(factor_data, loader, self._factors)

    def _calc_risk_correlation(self, factor_data, loader, risk_factors_config):
        """计算与风险因子的相关性"""
        adj_dates = factor_data.index.tolist()

        # 确定风险因子列表
        if risk_factors_config == 'all':
            try:
                risk_keys = loader.get_apikeys('risk_factor.h5')
                risk_keys = [k[1:] for k in risk_keys]  # 去掉前导 /
                risk_factors = [('risk_factor.h5', k) for k in risk_keys]
            except Exception:
                risk_factors = []
        else:
            risk_factors = risk_factors_config

        if not risk_factors:
            return {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}

        # 加载风险因子 (安全版本: 字典查找替代 exec)
        risk_data_dict = {}
        for file_key, factor_key in risk_factors:
            try:
                if file_key == 'risk_factor.h5':
                    rf = loader.load_h5(file_key, factor_key)
                else:
                    rf = loader.load_custom((file_key, factor_key))
                if loader.valid_shape(rf):
                    rf = loader.add_index(rf)
                risk_data_dict[factor_key] = rf
            except Exception as e:
                logger.warning(f"加载风险因子 {factor_key} 失败: {e}")

        if not risk_data_dict:
            return {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}

        risk_names = list(risk_data_dict.keys())

        # 计算每期相关系数
        corr_all_list = []
        for date_i in adj_dates:
            risk_i = pd.DataFrame()
            for name in risk_names:
                rf = risk_data_dict[name]
                if date_i in rf.index:
                    risk_i[name] = rf.loc[date_i]

            factor_i = factor_data.loc[date_i]
            merged = pd.merge(
                factor_i.to_frame('factor'), risk_i,
                left_index=True, right_index=True
            )
            if len(merged) > 2:
                corr_i = merged.corr(method='spearman')
                corr_i['date'] = date_i
                corr_all_list.append(corr_i)

        if not corr_all_list:
            return {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}

        corr_all = pd.concat(corr_all_list)
        corr_all['group'] = corr_all['date']

        corr_mean = corr_all.groupby('group').mean()
        corr_std = corr_all.groupby('group').std(ddof=1)
        corr_stab = corr_mean / corr_std
        corr_stab = corr_stab.reindex(index=corr_mean.columns.tolist())
        corr_stab.replace(np.inf, np.nan, inplace=True)

        return {
            'mean': corr_mean,
            'stability': corr_stab,
        }
