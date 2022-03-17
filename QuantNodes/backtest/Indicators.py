#coding=utf-8
import numpy as np
import pandas as pd

"""
策略业绩评估指标
年化收益率
年化波动率
年化夏普率
最大化回撤率
M2测度
TreynorRatio
JensenRatio
InformationRatio
CalmarRatio
SortinoRatio
SterlingRatio
BurkeRatio
KappaRatio
OmegaRatio
-----
最大亏损，最大盈利，胜率，超额胜率，平均组合大小
"""

class Statistics(object):
    @staticmethod
    def cash(obj, code):
        for trade in obj:
            if trade.code == code and trade.side == 'sell':
                yield trade.trade_result_trade_size

    @staticmethod
    def cash_values(obj):
        return sum([trade.trade_result_trade_size for trade in obj if trade.side == 'sell'])

    @classmethod
    def securities(cls, obj):

        return {code: sum(cls.share(code)) for code in obj.codes()}

    @staticmethod
    def security_cost(obj):
        return sum([trade.trade_result_cost_size for trade in obj if trade.side == 'buy'])

    @staticmethod
    def security_cost_without_fee(obj):
        return sum([trade.trade_result_trade_size for trade in obj if trade.side == 'buy'])

    @staticmethod
    def share(obj, code):
        for trade in obj:
            if trade.code == code and trade.side == 'buy':
                yield trade.traded_size


class Indicators(object):
    """
    计算回测结果的指标信息
    """

    def __init__(self, quote):
        self.quote = quote

    @staticmethod
    def filter_dt_pandas(traded, dt, dt_col='dt'):
        data = traded[traded[dt_col] <= dt]
        return data

    def cal_traded_df(self, traded, dt, dt_col='dt', code_col='code'):
        traded2 = self.filter_dt_pandas(traded, dt, dt_col=dt_col)
        res = traded2.groupby(code_col)[
            ['traded_size', 'fee', 'trade_result_cost_size', 'trade_result_trade_size']].sum()
        # self.quote.opr_filter('@Code='GooG'))
        return res

    pass


class IndicatorsFromNetValue(object):
    @staticmethod
    def ann_ret_period(net_value, period):
        return net_value.rolling(period).apply(lambda x: np.power(x[-1] / x[0], 250 / period) - 1)

    @staticmethod
    def ann_ret(net_value):
        return np.power(net_value[-1] / net_value[0], 250 / len(net_value)) - 1

    @staticmethod
    def max_drawdown(net_value, return_dt=True):
        max_value = np.maximum.accumulate(net_value)
        i = np.argmax((max_value - net_value) / max_value)  # 最小的净值
        if i == 0:
            return None, None, 0
        j = np.argmax(net_value[:i])
        if return_dt:
            return net_value.index[i].strftime("%Y%m%d"), net_value.index[j].strftime("%Y%m%d"), (
                    - net_value[j] + net_value[i]) / net_value[j]
        else:
            return i, j, (net_value[i] - net_value[j]) / net_value[j]

    @staticmethod
    def ann_vol(daily_yield):
        return daily_yield.std() * np.sqrt(252)

    @staticmethod
    def ann_sharpe(ann_ret: float, ann_vol: float, rf: float):
        return (ann_ret - rf) / ann_vol

    @classmethod
    def cal(cls, name: str, net_value: pd.Series, rf: float = 1.5 / 100, return_dt=True,
            period_num=(20, 40, 60, 90, 180)):
        info = {}
        daily_yield = net_value.pct_change(1)
        info['name'] = name
        info['AnnRet'] = cls.ann_ret(net_value)
        info['Vol'] = cls.ann_vol(daily_yield)
        info['Sharpe'] = cls.ann_sharpe(info['AnnRet'], info['Vol'], rf)
        end, start, r = cls.max_drawdown(net_value, return_dt=return_dt)
        info['MaxDrawDown_start'] = start
        info['MaxDrawDown_end'] = end
        info['MaxDrawDown_Rate'] = r
        for num in period_num:
            r_ann_ret = cls.ann_ret_period(net_value, num)
            r_vol = daily_yield.rolling(num).std() * np.sqrt(250)
            r_sharpe = (r_ann_ret - rf) / r_vol
            rolling_stats = pd.concat([r_ann_ret, r_vol, r_sharpe], axis=1)
            rolling_stats.columns = ['r_ann_ret', 'r_vol', 'r_sharpe']
            info[f'rolling_{num}'] = rolling_stats
        return info

        pass
