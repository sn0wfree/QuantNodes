# coding: utf-8
"""Node 4: 调仓日生成 / Adjust Date Node

Migrated from date_utils.py:134-191 get_adjust_date()
"""

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from QuantNodes.core.node import BaseNode
from QuantNodes.research.factor_test.utils.date_utils import get_adjust_date


class AdjustDateNode(BaseNode):
    """根据起始日、截止日、调仓模式生成调仓日序列

    输入: context["LoadData"] 的输出
    输出: adj_dates (yyyymmdd int DataFrame)
    """

    def __init__(self, name: str = "AdjustDate", config: dict = None, **kwargs):
        super().__init__(name, config, **kwargs)
        # H10: 不再硬编码 20170801/20171231, 默认 None → 启动校验时必填
        self._adj_date_beg = config.get('adj_date_beg', None) if config else None
        self._adj_date_end = config.get('adj_date_end', None) if config else None
        self._adj_mode = config.get('adj_mode', ['M', 'end']) if config else ['M', 'end']

    def _execute(self, input_data=None, **kwargs) -> pd.DataFrame:
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
