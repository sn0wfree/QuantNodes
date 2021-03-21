# coding=utf-8
# import random
# from Nodes.test import GOOG
import numpy as np
import pandas as pd

from Nodes.backtest.Orders import Orders, Order
from Nodes.backtest.Quote import QuoteData
from collections import Iterable, Iterator


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
                      default_stop=-np.inf, default_sl=None, default_tp=None, iterable=False):
        """create all orders at begin"""
        ## TODO limit price should be set!!!

        a_iter = self._create_orders(scripts, self._commission, quote, self._trade_on_close, default_stop=default_stop,
                                     default_sl=default_sl, default_tp=default_tp)
        if iterable:
            return a_iter
        else:
            return Orders(*list(a_iter))

    @staticmethod
    def _create_orders(scripts: pd.DataFrame, _commission, quota, trade_on_close, default_stop=-np.inf, default_sl=None,
                       default_tp=None):
        default_dict = {'stop': default_stop, 'sl': default_sl, 'tp': default_tp}
        cols = scripts.columns.tolist()
        must = ['date', 'code', 'size', 'limit']
        reqired = list(default_dict.keys())  # ['stop', 'sl', 'tp']

        if 'limit' in cols:
            pass
        else:
            if trade_on_close:
                col = 'Close'
            else:
                col = 'Open'
            q = quota._data[['date', 'Code', col]].rename(columns={col: 'limit', 'Code': 'code'})
            scripts = scripts.merge(q, on=['date', 'code'])
            cols = scripts.columns.tolist()
        missed = set(must) - set(cols)
        if len(missed) != 0:
            raise ValueError(f"{','.join(missed)} are missing")

        for c in filter(lambda x: x not in cols, reqired):
            scripts[c] = default_dict[c]
        # container = OrderedDict()
        date_col, code_col = must[:2]
        scripts_groupby = scripts.sort_values([date_col, code_col])[must + reqired].groupby(date_col)
        for dt, df in scripts_groupby:
            day = [Order(_commission, size, code,
                         limit_price=limit, stop_price=stop,
                         sl_price=sl, tp_price=tp,
                         order_id=None, create_date=date,
                         parent_trade=None) for date, code, size, limit, stop, sl, tp in df.values]
            # = size_df[must + reqired].values.tolist()
            # order =
            # day.append(order)
            yield dt.strftime('%Y-%m-%d'), day

    def __call__(self, orders: (Orders, Iterable), reduce=True, position_check_func=None):
        # dt_list =
        # dt_col = self._data.date_col
        if position_check_func is None:
            position_check_func = lambda x, y, z: (x, y, z)

        if isinstance(orders, Orders):

            dt_list = list(orders.reduce_keys(exclude_value=0)) if reduce else list(orders.keys())

            filtered_quote = self._data[dt_list]  # reduce quote data by select required date only

            ## todo test the perfermance of df with group by and the perfrtmance of dataframe with slice
            for dt, df in iter(filtered_quote):
                dt = pd.to_datetime(dt).strftime('%Y-%m-%d')
                order_list = orders.get(dt)
                # q = QuoteData.create_quote(df)
                ## reduce quote data initetime

                dt, df, order_list = position_check_func(dt, df, order_list)
                yield dt, df, order_list
        elif isinstance(orders, Iterable):  # iter prepare to d to check current position
            dt_col = self._data.date_col
            for dt, order_list in orders:

                if reduce:
                    filtered_order_list = list(filter(lambda x: x.size != 0, order_list))
                    if len(filtered_order_list) == 0:
                        pass
                    else:
                        q = self._data[self._data[dt_col] == dt]
                        # res = list(map(lambda x: x.operate(single_dt_quote), order_list))
                        # h.append(res)
                        dt, q, order_list = position_check_func(dt, q, filtered_order_list)
                        yield dt, q, order_list
                else:
                    q = self._data[self._data[dt_col] == dt]
                    # res = list(map(lambda x: x.operate(single_dt_quote), order_list))
                    # h.append(res)
                    dt, q, order_list = position_check_func(dt, q, order_list)
                    yield dt, q, order_list
        else:
            raise ValueError('unknown orders!')

        # return h
