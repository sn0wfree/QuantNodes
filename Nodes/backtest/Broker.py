# coding=utf-8
# import random
# from Nodes.test import GOOG
import numpy as np
import pandas as pd

from Nodes.backtest.Orders import Orders, Order
from Nodes.backtest.Quote import QuoteData


# from Nodes.utils_node.file_cache import file_cache


class Broker(object):
    """
    提供用户策略和Orders转换工具,不提供存储功能
    """
    __slots__ = ['_data', '_cash', '_commission', '_leverage', '_trade_on_close', '_hedging', '_exclusive_orders']

    def __init__(self, data, cash, commission, margin, trade_on_close, hedging, exclusive_orders):
        assert 0 < cash, f"cash should be >0, is {cash}"
        assert 0 <= commission < .1, f"commission should be between 0-10%, is {commission}"
        assert 0 < margin <= 1, f"margin should be between 0 and 1, is {margin}"
        self._data = QuoteData.create_quote(data)
        self._cash = cash
        self._commission = commission
        self._leverage = 1 / margin
        self._trade_on_close = trade_on_close
        self._hedging = hedging
        self._exclusive_orders = exclusive_orders

        # self._equity = np.tile(np.nan, len(index))
        # self.orders = []
        # self.trades = []
        # self.closed_trades = []

    def create_orders(self, scripts: pd.DataFrame, quote,
                      default_stop=-np.inf, default_sl=None, default_tp=None, ):
        """create all orders at begin"""
        ## TODO limit price should be set!!!

        a = list(self._create_orders(scripts, self._commission, quote, self._trade_on_close, default_stop=default_stop,
                                     default_sl=default_sl, default_tp=default_tp))

        return Orders(*a)

    @staticmethod
    def _create_orders(scripts: pd.DataFrame, _commission, quota, trade_on_close, default_stop=-np.inf, default_sl=None,
                       default_tp=None):
        default_dict = {'stop': default_stop, 'sl': default_sl, 'tp': default_tp}
        cols = scripts.columns.tolist()
        must = ['date', 'code', 'size', 'limit']
        reqired = ['stop', 'sl', 'tp']

        if 'limit' in cols:
            pass
        else:
            if trade_on_close:
                col = 'Close'
            else:
                col = 'Open'
            scripts = scripts.merge(
                quota._data[['date', 'Code', col]].rename(columns={col: 'limit', 'Code': 'code'}),
                on=['date', 'code'])
            cols = scripts.columns.tolist()
        missed = set(must) - set(cols)
        if len(missed) != 0:
            raise ValueError(f"{','.join(missed)} are missing")

        for c in filter(lambda x: x not in cols, reqired):
            scripts[c] = default_dict[c]
        # container = OrderedDict()
        for dt, df in scripts.sort_values(must[0]).groupby(must[0]):
            day = []
            for code, size_df in df.groupby(must[1]):
                for date, code, size, limit, stop, sl, tp in size_df[must + reqired].values:
                    # = size_df[must + reqired].values.tolist()
                    order = Order(_commission, size, code,
                                  limit_price=limit,
                                  stop_price=stop,
                                  sl_price=sl,
                                  tp_price=tp,
                                  order_id=None,
                                  create_date=date,
                                  parent_trade=None)
                    day.append(order)
            yield dt.strftime('%Y-%m-%d'), day

    def __call__(self, orders: Orders, reduce=False):
        # dt_list =
        # dt_col = self._data.date_col

        dt_list = list(orders.reduce_keys(exclude_value=0)) if reduce else list(orders.keys())

        filtered_quote = self._data[dt_list]  # reduce quote data by select required date only
        # h = []
        ## todo test the perfermance of df with group by and the perfrtmance of dataframe with slice
        for dt, df in iter(filtered_quote):
            dt = pd.to_datetime(dt).strftime('%Y-%m-%d')
            order_list = orders.get(dt)
            yield dt, QuoteData.create_quote(df), order_list
        # for dt, order_list in orders.items():
        #     single_dt_quote = QuoteData.create_quote(filtered_quote[filtered_quote[dt_col] == dt])
        #     # res = list(map(lambda x: x.operate(single_dt_quote), order_list))
        #     # h.append(res)
        #     yield dt, single_dt_quote, order_list
        # return h
