# coding=utf-8
"""Node 6: 因子中性化 / Factor Neutralize Node

Migrated from factor_utils.py:534-625 neutralize()

Phase 2.1 (Chain of Responsibility):
  - 原 70 行 _neutralize 三个 if/elif 分支 (industry only / risk only / both)
    替换为 chain dispatch.
  - 中性化逻辑 (设计矩阵 X 组装) 抽到 nodes/neutralizers.py:
      Neutralizer (ABC) / IndustryNeutralizer / RiskNeutralizer
      build_neutralizer_chain() / apply_neutralizer_chain()
  - 新增中性化类型 (如 StyleNeutralizer) 只需新增一个 Neutralizer 子类,
    _execute 无需修改.
"""

import logging

import pandas as pd

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import NeutralizeNodeConfig
from QuantNodes.research.factor_test.nodes.neutralizers import (
    apply_neutralizer_chain,
    build_neutralizer_chain,
)

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
        """中性化处理 (Phase 2.1: 委托给 chain).

        Phase 2.1 行为完全等价于旧实现:
          - chain 为空 → 返回 factor_i (全 nan, 与原 _neutralize 入口一致)
          - 4 种 flag 组合的输出与原 branch 1/2/3 bitwise 一致
        """
        chain = build_neutralizer_chain(if_industry, if_risk, industry, risk_data)
        if not chain:
            # 无 neutralizer 时返回原 factor (保留 nan 模式), 与旧 line 40 一致
            return factor_i
        return apply_neutralizer_chain(factor_i, chain)
