# coding: utf-8
"""Node 2: 样本池筛选 / Sample Pool Filter Node

Migrated from factor_utils.py:155-234 select_range()
"""

import numpy as np
import pandas as pd

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import SamplePoolNodeConfig
from QuantNodes.research.factor_test.utils.constants import (
    resolve_index_mapping, resolve_industry_map,
)


class SamplePoolFilterNode(PydanticConfigNode):
    """根据指数范围和行业筛选样本池

    输入: context["LoadData"] 的输出
    输出: stock_sample (1=选中, nan=剔除)

    M9: index_mapping 可自定义，合并全局默认 + 节点自定义覆盖
    M12: i18n_name_map 可自定义行业代码→名称映射，合并全局默认 + 节点自定义覆盖
    """

    ConfigSchema = SamplePoolNodeConfig
    _ALIASES = {
        "_sample_index": "sample_index",
        "_sample_industry": "sample_industry",
        "_sample_customdir": "sample_index_customdir",
    }

    def __init__(self, name="SamplePoolFilter", config=None, **kwargs):
        super().__init__(name, config, **kwargs)
        # M9: 合并全局默认 INDEX_MAPPING + 节点自定义覆盖
        self._index_mapping = resolve_index_mapping({
            "INDEX_MAPPING": self.cfg.index_mapping
        } if self.cfg.index_mapping else None)
        # M12: 合并全局默认 INDUSTRY_MAPPING + 节点自定义覆盖
        self._i18n_name_map = resolve_industry_map({
            "INDUSTRY_MAP": self.cfg.i18n_name_map
        } if self.cfg.i18n_name_map else None)

    def _execute(self, input_data=None, **kwargs) -> pd.DataFrame:
        context = kwargs.get('context', {})
        load_data = context.get('LoadData', input_data)

        stklist = load_data['stklist']
        trade_dt = load_data['trade_dt']
        loader = load_data['_loader']

        n_dt = len(trade_dt)
        n_stk = len(stklist)

        # 指数筛选
        index_filt = np.ones((n_dt, n_stk))
        if self._sample_index != 'all':
            if self._sample_index in self._index_mapping:
                h5_file, key = self._index_mapping[self._sample_index]
                if_index = loader.load_h5(h5_file, key)
                if_index = if_index.replace(np.nan, 0)
                index_filt = index_filt * if_index.values
            elif self._sample_index == 'ZZ800':
                if_300 = loader.load_h5('stk_daily.h5', 'id_300').replace(np.nan, 0)
                if_500 = loader.load_h5('stk_daily.h5', 'id_500').replace(np.nan, 0)
                index_filt = index_filt * (if_300 + if_500).values
            elif self._sample_index == 'custom' and self._sample_customdir:
                if_index = loader.load_custom(self._sample_customdir)
                if_index = if_index.replace(np.nan, 0)
                index_filt = index_filt * if_index.values
            else:
                raise ValueError(f"不支持的指数: {self._sample_index}")

         # 行业筛选
        industry_filt = np.ones((n_dt, n_stk))
        if self._sample_industry != 'all':
            if isinstance(self._sample_industry, tuple):
                ind_key, ind_name = self._sample_industry
            else:
                ind_key = 'id_citic1'
                ind_name = self._sample_industry

            id_industry = loader.load_h5('stk_daily.h5', ind_key)
            # M12: i18n 映射: self._i18n_name_map
            if self._i18n_name_map and ind_key in self._i18n_name_map:
                ind_name = self._i18n_name_map[ind_key]
            id_name_data = loader.load_h5(
                'stk_daily.h5', f'ind_name_{ind_key.replace("id_", "").upper()}'
            )

            # 找到行业名称对应的编号
            ind_idx = np.where(id_name_data.values == ind_name)[0]
            if len(ind_idx) > 0:
                ind_num = ind_idx[0] + 1  # 行业编号从 1 开始
                industry_filt[id_industry.values != ind_num] = 0
            else:
                raise ValueError(f"行业 '{ind_name}' 未找到")

        # 合并筛选
        stock_sample = index_filt * industry_filt
        stock_sample[stock_sample > 0] = 1
        stock_sample = pd.DataFrame(stock_sample)
        stock_sample = stock_sample.replace(0, np.nan)

        return stock_sample
