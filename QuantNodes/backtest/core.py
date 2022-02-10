# coding=utf-8

import numpy as np
# import random
# from QuantNodes.test import GOOG
import pandas as pd

# from QuantNodes.utils_node.file_cache import file_cache
from QuantNodes.backtest.Broker import Broker
from QuantNodes.backtest.Indicators import IndicatorsFromNetValue
from QuantNodes.backtest.Positions import Positions
from QuantNodes.backtest.Quote import QuoteData
from QuantNodes.utils_node.generate_str_node import randon_str_hash


# from collections import OrderedDict


class ScriptsBackTest(object):
    __slots__ = ['scripts', 'broker', 'orders', 'quote', 'trades', 'closed_trades', 'positions']

    def __init__(self, scripts, data, cash, commission, margin, trade_on_close, hedging, exclusive_orders,
                 default_limit=None, default_stop=-np.inf, default_sl=None, default_tp=None):
        self.scripts = scripts
        self.quote = QuoteData.create_quote(data)
        self.broker = Broker(self.quote, cash, commission, margin, trade_on_close, hedging, exclusive_orders)

        self.orders = self.broker.create_orders(scripts, self.quote, default_stop=default_stop,
                                                default_sl=default_sl, default_tp=default_tp)

        self.positions = Positions(self.quote)

    """
    主程序, run
    """

    def run(self, reduce=True, position_check_func=None):
        """
        the run function to begin back testing



        :return:
        """

        for dt, df, order_list in self.broker(self.orders, reduce=reduce,
                                              position_check_func=position_check_func):
            trade_list = [order.deal(df[df['Code'] == order._attr.code]['Close'].values.ravel()[0]) for order in
                          order_list]

            self.positions.trade_extend(trade_list, reduce=reduce)

            # last_position = self.positions.last_position(dt, code)
            # current_position = self.positions.current_position(dt, code)
            # res = self.operate(o, single_dt_quote, current_position, last_position)

        c3 = self.positions.to_pandas()
        c32 = self.positions._indicators.cal_traded_df(c3, '2007-01-01')
        print(1)

        # else:
        #     dt = o.create_date
        #     price = self.broker._data[self.broker._data['date'] == dt]['price']
        #     o.oper(None, price)

        # pass
    #
    # @staticmethod
    # def get_data(*args, **kwargs):
    #     """
    #     prepare required data
    #     :param args:
    #     :param kwargs:
    #     :return:
    #     """
    #     pass


def test_ScriptsBackTest():
    from QuantNodes.test import GOOG

    # ['size', 'limit', 'stop', 'sl', 'tp']

    GOOG['Code'] = 'GOOG'
    # GOOG = GOOG.reset_index().rename(columns={'index': 'date'})
    # GOOG['Code'] = 'GOOG'

    # QD = QuoteData(GOOG)
    # print(len(QD))

    np.random.seed(1)
    price = pd.DataFrame(np.random.random(size=(GOOG.shape[0], 1)), columns=['GOOG'])
    orders_df = (price > 0.75) * 1
    orders_df['date'] = GOOG.index
    scripts = orders_df.set_index('date').stack().reset_index()
    scripts.columns = ['date', 'code', 'size']

    GOOG = GOOG.reset_index().rename(columns={'index': "date"})

    QD = QuoteData(GOOG)
    ## 先手动生成scripts
    "A limit order is a type of order to purchase or sell a security at a specified price or better. " \
    "For buy limit orders, the order will be executed only at the limit price or a lower one, " \
    "while for sell limit orders, the order will be executed only at the limit price or a higher one. " \
    "This stipulation allows traders to better control the prices they trade."

    # class BuyandHoldStrategy(Strategy):
    #     def init(self, scripts, cols=['date', 'code', 'size']):
    #         code_list = scripts['code'].unique().tolist()
    cash = 100000
    commission = 0.01
    margin = 0.01
    trade_on_close = True
    hedging = 0
    exclusive_orders = []
    # quote_data = GOOG
    sbt = ScriptsBackTest(scripts, QD, cash, commission, margin, trade_on_close, hedging, exclusive_orders)
    sbt.run()


def index_merge(idx1: list, idx2: list):
    idx = sorted(set(idx1 + idx2))
    return idx


class FastBackTestWeight(object):
    """
    1. directly give stock list and dt, | weight or share number
    2. fast calc net value and daily holdings
    3. calc indicators
    4 draw plot

    """

    __slots__ = ['scripts', 'broker', 'orders', 'quote', 'trades', 'closed_trades', 'positions', 'default_cols', 'sid']

    def __init__(self, scripts, data, cash, commission, margin, trade_on_close, hedging, exclusive_orders,
                 default_limit=None, default_stop=-np.inf, default_sl=None, default_tp=None,
                 default_cols={'date': 'date', 'code': 'code', 'weight': 'weight'}):
        self.scripts = scripts
        self.quote = QuoteData.create_quote(data)
        self.default_cols = default_cols
        self.sid = randon_str_hash(num=12)

    def _matrix_scripts(self, re_weight=True):
        dc = self.default_cols
        res = self.scripts.pivot_table(columns=dc['code'], index=dc['date'], values=dc['weight'], aggfunc="sum")
        cols = res.columns.tolist()
        if re_weight:
            res_sum = res.sum(axis=1)
            r4 = res.apply(lambda x: x / res_sum).fillna(0)
            return r4, cols
        else:
            return res, cols

    @staticmethod
    def _fast_cal_nv_pct(scripts_trim, pct_trim):
        port_info = scripts_trim * pct_trim
        port_info['port_pct'] = port_info.sum(axis=1)
        port_info['port_nv'] = np.cumprod(port_info['port_pct'] + 1)
        return port_info

    def run(self, re_weight=True, idx_list=None, rf=1.5 / 100, return_dt=True,
            period_num=(20, 40, 60, 90, 180)):
        scripts, stk_list = self._matrix_scripts(re_weight=re_weight)
        # scripts : index = dt , column : stock
        data = self.quote.data_filter_matrix(stk_list, )

        pct = data.get_one_data('Close').pct_change()
        # idx_list = pct.index.tolist()
        # idx = index_merge(pct.index.tolist(), scripts.index.tolist())
        cols = index_merge(pct.columns.tolist(), scripts.columns.tolist())
        scripts_trim = scripts.reindex(index=idx_list, columns=cols)
        pct_trim = pct.reindex(index=idx_list, columns=cols)
        port_info = self._fast_cal_nv_pct(scripts_trim, pct_trim)
        info = IndicatorsFromNetValue.cal(self.sid, port_info['port_nv'], rf=rf, return_dt=return_dt,
                                          period_num=period_num)
        # port_info contain stock_pct, port_pct, port_nv
        # port_nv.plot()

        # trim
        return port_info, info


if __name__ == '__main__':
    # prepare quote data
    from QuantNodes.test import GOOG

    # ['size', 'limit', 'stop', 'sl', 'tp']

    QD = QuoteData(GOOG, code='Code', date='date')

    # prepare scipts
    np.random.seed(1)
    price = pd.DataFrame(np.random.random(size=(GOOG.shape[0], 1)), columns=['GOOG'])
    orders_df = (price > 0.75) * 1
    orders_df['date'] = GOOG['date']
    scripts = orders_df.set_index('date').stack().reset_index()
    scripts.columns = ['date', 'code', 'weight']
    scripts['weight'] = scripts['weight'].shift(1)

    config = {}
    config['cash'] = 100000
    config['commission'] = 0.01
    config['margin'] = 0.01
    config['trade_on_close'] = True
    config['hedging'] = 0
    idx_list = scripts['date'].tolist()[-300:]
    exclusive_orders = []
    # quote_data = GOOG
    sbt = FastBackTestWeight(scripts, QD, **config, exclusive_orders=exclusive_orders)
    port_info, indicators = sbt.run(re_weight=True, idx_list=idx_list)

    print(1)


