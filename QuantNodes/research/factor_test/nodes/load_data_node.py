# coding: utf-8
"""Node 1: 加载数据 / Load Data Node"""

import logging
from typing import Dict

import pandas as pd

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.utils.data_loader import DataLoader
from QuantNodes.research.factor_test.utils.safe_load import (
    safe_load_factor,
    safe_load_h5,
    try_load_panels,
)
from QuantNodes.research.factor_test.nodes.configs import LoadDataNodeConfig

logger = logging.getLogger(__name__)


class LoadDataNode(PydanticConfigNode):
    """加载因子数据、价格、行业、市值等

    输入: config (LoadDataNodeConfig: data_path + load_keys + factor)
    输出: Dict[str, pd.DataFrame]
    """

    ConfigSchema = LoadDataNodeConfig
    _ALIASES = {
        "_data_path": "data_path",
        "_load_keys": "load_keys",
        "_factor_config": "factor",
    }

    def _execute(self, input_data=None, **kwargs) -> Dict[str, pd.DataFrame]:
        # P-2: 空字符串校验 (Pydantic Field(...) 不挡空串, 需显式检查)
        if not self._data_path:
            raise ValueError(
                "data_path required (P-2: 启动报错, 防止 None 数据目录导致下游谜之失败)"
            )

        loader = DataLoader(self._data_path)
        result = {}

        # 加载因子
        if self._factor_config:
            factor = safe_load_factor(
                loader, self._factor_config.factor_dir, self._factor_config.name
            )
            if factor is None:
                raise ValueError(
                    f"因子加载失败: dir={self._factor_config.factor_dir}, "
                    f"name={self._factor_config.name}"
                )
            # 检查是否有索引, 没有则添加
            if hasattr(factor, 'columns') and factor.columns.dtype == 'int64':
                if loader.valid_shape(factor):
                    factor = loader.add_index(factor)
                else:
                    raise ValueError(
                        f"因子 {self._factor_config.name} shape 不一致: {factor.shape}"
                    )
            else:
                stklist, trade_dt = loader.get_stock_axis()
                factor = factor.reindex(index=trade_dt.iloc[:, 0], columns=stklist.iloc[:, 0])
            result['factor'] = factor

        # 加载价格 (price 几乎所有下游节点都需要, 强制加载)
        price = safe_load_h5(loader, 'stk_daily.h5', 'cp')
        if price is not None:
            result['price'] = price
        else:
            logger.warning("LoadDataNode: 未找到 price 数据 (stk_daily.h5/cp)")

        # 加载其他数据
        for key in self._load_keys:
            if key in ('stklist', 'trade_dt'):
                continue  # 已通过 get_axis 处理
            if key in result:
                continue  # 已加载
            data = try_load_panels(loader, key)
            if data is not None:
                result[key] = data
            else:
                logger.warning("LoadDataNode: 跳过 %s (stk_daily.h5/index_daily.h5 都不存在)", key)

        # 加载指数收盘价 (用于对冲基准)
        index_cp = safe_load_h5(loader, 'index_daily.h5', 'index_cp', axis_type='index')
        if index_cp is not None:
            result['index_cp'] = index_cp
        else:
            logger.debug("LoadDataNode: 未找到 index_cp 数据 (index_daily.h5/index_cp)")

        # 保存 loader 和轴数据供下游使用
        result['_loader'] = loader
        stklist, trade_dt = loader.get_stock_axis()
        result['stklist'] = stklist
        result['trade_dt'] = trade_dt

        return result
