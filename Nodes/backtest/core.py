# coding=utf-8
from abc import abstractmethod, ABCMeta
from collections import Iterable

# import random
# from Nodes.test import GOOG
import numpy as np
import pandas as pd

# from Nodes.utils_node.file_cache import file_cache
from Nodes.backtest.Broker import Broker
from Nodes.backtest.Positions import Positions
from Nodes.backtest.Quote import QuoteData


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

        self.positions = Positions()

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


if __name__ == '__main__':
    from Nodes.test import GOOG

    ['size', 'limit', 'stop', 'sl', 'tp']

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
    print(1)

pass
