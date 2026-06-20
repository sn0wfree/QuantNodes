# coding: utf-8
"""Node 3: 可交易性筛选 / Tradability Filter Node

Migrated from factor_utils.py:250-308 valid_tradable()
Security: exec() replaced with dict lookup.
"""

import numpy as np
import pandas as pd

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import TradabilityNodeConfig


class TradabilityFilterNode(PydanticConfigNode):
    """验证股票可交易性 (ST/停牌/涨跌停/新股/自定义追踪)

    输入: context["LoadData"] 的输出 + context["SamplePoolFilter"] 的输出
    输出: tradable (1=可交易, nan=不可)
    """

    ConfigSchema = TradabilityNodeConfig
    _ALIASES = {"_tradable_setting": "tradable"}

    def _execute(self, input_data=None, **kwargs) -> pd.DataFrame:
        context = kwargs.get('context', {})
        load_data = context.get('LoadData', {})
        sample = context.get('SamplePoolFilter')

        loader = load_data['_loader']
        stklist = load_data['stklist']
        trade_dt = load_data['trade_dt']

        n_dt = len(trade_dt)
        n_stk = len(stklist)

        # 加载可交易性数据 (优先从 context, 回退到 loader)
        def _get(key):
            if key in load_data:
                return load_data[key]
            return loader.load_h5('stk_daily.h5', key) if loader else None

        st = _get('st')
        suspend = _get('suspend')
        ud_limit = _get('ud_limit')
        if ud_limit is not None:
            ud_limit = ud_limit.abs()
        ipo_days = _get('ipo_days')

        # 初始化: 全部可交易
        if_tradable = pd.DataFrame(
            np.ones((n_dt, n_stk)),
            index=trade_dt.iloc[:, 0].values,
            columns=stklist.iloc[:, 0].values
        )

        s = self._tradable_setting

        if s.no_st:
            if_tradable[st == 1] = np.nan

        if s.no_suspended:
            if_tradable[suspend == 1] = np.nan

        if s.no_up_down_limit:
            if_tradable[ud_limit == 1.0] = np.nan

        if s.min_ipo_days:
            if_tradable[ipo_days < s.min_ipo_days] = np.nan

        # 自定义追踪条件 (安全版本: 字典查找替代 exec)
        if s.trace:
            trace_data_map = {
                'suspend': suspend,
                'st': st,
                'ud_limit': ud_limit,
            }
            for trace_key, (m, n) in s.trace.items():
                if trace_key not in trace_data_map:
                    raise ValueError(f"不支持的追踪条件: {trace_key}")
                trace_data = trace_data_map[trace_key]
                if m > 0:
                    trace_sum = trace_data.rolling(m).sum().shift(1)
                else:
                    trace_sum = trace_data.rolling(-m).sum().shift(m)
                if_tradable[trace_sum >= n] = np.nan

        # 合并样本池筛选
        if sample is not None:
            if_tradable = if_tradable * sample

        return if_tradable
