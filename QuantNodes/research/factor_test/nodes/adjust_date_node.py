# coding: utf-8
"""Node 4: 调仓日生成 / Adjust Date Node

Migrated from date_utils.py:134-191 get_adjust_date()
"""

import sys
from pathlib import Path
from typing import Union

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from QuantNodes.core.node import BaseNode
from QuantNodes.research.factor_test.nodes.configs import AdjustDateNodeConfig
from QuantNodes.research.factor_test.utils.date_utils import get_adjust_date


class AdjustDateNode(BaseNode):
    """根据起始日、截止日、调仓模式生成调仓日序列

    输入: context["LoadData"] 的输出
    输出: adj_dates (yyyymmdd int DataFrame)
    """

    def __init__(self, name: str = "AdjustDate",
                 config: Union[dict, AdjustDateNodeConfig, None] = None, **kwargs):
        # T0-4: 预先 Union 化
        if isinstance(config, AdjustDateNodeConfig):
            cfg = config
            super().__init__(name, cfg.model_dump(), **kwargs)
        elif isinstance(config, dict) or config is None:
            cfg = AdjustDateNodeConfig.model_validate(config or {})
            super().__init__(name, config, **kwargs)
        else:
            raise TypeError(
                f"config must be dict/None/AdjustDateNodeConfig, got {type(config).__name__}"
            )
        # T0-3: H10 兼容, 默认 None → _execute 启动校验抛错
        self._adj_date_beg = cfg.adj_date_beg
        self._adj_date_end = cfg.adj_date_end
        self._adj_mode = list(cfg.adj_mode)

    def _execute(self, input_data=None, **kwargs) -> pd.DataFrame:
        # T0-3: H10 启动校验 (避免静默跑废日期)
        if self._adj_date_beg is None or self._adj_date_end is None:
            raise ValueError(
                f"AdjustDateNode 需要 adj_date_beg 和 adj_date_end 字段 "
                f"(H10: 默认 None, 当前 beg={self._adj_date_beg}, end={self._adj_date_end})"
            )

        context = kwargs.get('context', {})
        load_data = context.get('LoadData', input_data)

        trade_dt = load_data['trade_dt']

        adj_dates = get_adjust_date(
            trade_dt,
            self._adj_date_beg,
            self._adj_date_end,
            tuple(self._adj_mode)
        )

        return adj_dates
