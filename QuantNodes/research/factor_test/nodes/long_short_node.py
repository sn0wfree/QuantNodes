# coding: utf-8
"""Node 9: 多空组合 / Long-Short Node

Migrated from factor_performance.py:562-617 cal_longshort_ret()
"""

from typing import Union

import pandas as pd

from QuantNodes.research.factor_test.nodes._base import PydanticConfigNode
from QuantNodes.research.factor_test.nodes.configs import LongShortNodeConfig
from QuantNodes.research.factor_test.utils.performance_metrics import evaluation


class LongShortNode(PydanticConfigNode):
    """多空组合构建 + 净值 + 评价

    输入: GroupAnalyzerNode 的输出
    输出: {net, eva_total, eva_yearly, period_ret}
    """

    ConfigSchema = LongShortNodeConfig

    def __init__(self, name: str = "LongShort",
                 config: Union[dict, LongShortNodeConfig, None] = None, **kwargs):
        super().__init__(name, config, **kwargs)
        self._factor_direction = self.cfg.factor_direction

    def _execute(self, input_data=None, **kwargs) -> dict:
        context = kwargs.get('context', {})
        group_result = context.get('GroupAnalyzer')
        if group_result is None:
            raise ValueError("分组分析数据缺失")

        return self._calc_longshort(group_result, self._factor_direction)

    def _calc_longshort(self, group_result, factor_ori):
        """计算多空净值"""
        n_groups = group_result['n_groups']
        adj_dates = group_result['adjust_dates']

        if factor_ori == 1:
            long_n = n_groups
            short_n = 1
        else:
            long_n = 1
            short_n = n_groups

        # 各期收益
        long_ret = group_result['group_ret'][long_n]
        short_ret = group_result['group_ret'][short_n]
        longshort_ret = long_ret - short_ret

        # 净值
        daily_net_long = group_result['daily_net_simp'][long_n]
        daily_net_short = group_result['daily_net_simp'][short_n]
        daily_exc_long = group_result['daily_excnet_simp'][long_n]
        daily_exc_short = group_result['daily_excnet_simp'][short_n]

        # 多空净值 (单利)
        daily_net_longshort = daily_net_long - daily_net_short + 1

        # 评价
        eva_longshort = evaluation(daily_net_longshort, adj_dates)

        # 合并结果
        eva_l_s_ls = pd.concat([
            group_result['group_eva_exc'][long_n],
            group_result['group_eva_exc'][short_n],
            eva_longshort.iloc[0, 1:],
        ], axis=1)
        eva_l_s_ls.columns = ['多头超额', '空头超额', '多空']

        period_ret = pd.concat([long_ret, short_ret, longshort_ret], axis=1)
        period_ret.columns = ['多头超额', '空头超额', '多空']

        net = pd.concat([
            daily_net_long, daily_net_short,
            daily_exc_long, daily_exc_short,
            daily_net_longshort,
        ], axis=1)
        net.columns = ['多头', '空头', '多头超额', '空头超额', '多空']

        eva_yearly = {
            '多头超额': group_result['group_eva_exc_yearly'].get(long_n),
            '空头超额': group_result['group_eva_exc_yearly'].get(short_n),
            '多空': eva_longshort.iloc[1:] if len(eva_longshort) > 1 else pd.DataFrame(),
        }

        return {
            'net': net,
            'eva_total': eva_l_s_ls,
            'eva_yearly': eva_yearly,
            'period_ret': period_ret,
            'longshort_ret': longshort_ret,
        }
