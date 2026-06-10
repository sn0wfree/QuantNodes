# coding: utf-8
"""Node 1: 加载数据 / Load Data Node"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# 将项目根目录加入 sys.path 以支持直接导入
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from QuantNodes.core.node import BaseNode
from QuantNodes.research.factor_test.utils.data_loader import DataLoader
from QuantNodes.research.factor_test.config import FactorSetting


class LoadDataNode(BaseNode):
    """加载因子数据、价格、行业、市值等

    输入: config (FactorSetting + data_path + load_keys)
    输出: Dict[str, pd.DataFrame]
    """

    def __init__(self, name: str = "LoadData", config: dict = None, **kwargs):
        super().__init__(name, config, **kwargs)
        self._factor_config = None
        self._data_path = config.get('data_path', './testdata/test_h5_new/') if config else './testdata/test_h5_new/'
        self._load_keys = config.get('load_keys', ['stklist', 'trade_dt', 'cp', 'id_citic1', 'mv_float']) if config else []
        if config and 'factor' in config:
            self._factor_config = FactorSetting(**config['factor'])

    def _execute(self, input_data=None, **kwargs) -> Dict[str, pd.DataFrame]:
        loader = DataLoader(self._data_path)
        result = {}

        # 加载因子
        if self._factor_config:
            factor = loader.load_factor(self._factor_config.factor_dir, self._factor_config.name)
            # 检查是否有索引, 没有则添加
            if hasattr(factor, 'columns') and factor.columns.dtype == 'int64':
                if loader.valid_shape(factor):
                    factor = loader.add_index(factor)
                else:
                    raise ValueError(f"因子 {self._factor_config.name} shape 不一致: {factor.shape}")
            else:
                stklist, trade_dt = loader.get_stock_axis()
                factor = factor.reindex(index=trade_dt.iloc[:, 0], columns=stklist.iloc[:, 0])
            result['factor'] = factor

        # 加载价格
        if 'cp' in self._load_keys or 'cp' not in self._load_keys:
            try:
                cp = loader.load_h5('stk_daily.h5', 'cp')
                result['price'] = loader.add_index(cp)
            except Exception:
                pass

        # 加载其他数据
        for key in self._load_keys:
            if key in ('stklist', 'trade_dt'):
                continue  # 已通过 get_axis 处理
            if key in result:
                continue  # 已加载
            try:
                data = loader.load_h5('stk_daily.h5', key)
                result[key] = loader.add_index(data)
            except Exception:
                try:
                    data = loader.load_h5('index_daily.h5', key)
                    result[key] = loader.add_index(data, axis_type='index')
                except Exception:
                    pass

        # 加载指数收盘价 (用于对冲基准)
        try:
            index_cp = loader.load_h5('index_daily.h5', 'index_cp')
            result['index_cp'] = loader.add_index(index_cp, axis_type='index')
        except Exception:
            pass

        # 保存 loader 和轴数据供下游使用
        result['_loader'] = loader
        stklist, trade_dt = loader.get_stock_axis()
        result['stklist'] = stklist
        result['trade_dt'] = trade_dt

        return result
