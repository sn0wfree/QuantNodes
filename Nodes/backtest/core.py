# coding=utf-8
from abc import ABCMeta, abstractmethod
from collections import Iterable
import multiprocessing as mp
import os
import sys
import warnings
from abc import abstractmethod, ABCMeta

from copy import copy
from functools import lru_cache, partial
from itertools import repeat, product, chain, compress
from math import copysign
from numbers import Number
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import pandas as pd
from collections import OrderedDict, namedtuple
import datetime

OHLCV_AGG = OrderedDict((
    ('Open', 'first'),
    ('High', 'max'),
    ('Low', 'min'),
    ('Close', 'last'),
    ('Volume', 'sum'),
))

import random
import uuid


def random_str(num=6):
    uln = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    rs = random.sample(uln, num)  # 生成一个 指定位数的随机字符串
    a = uuid.uuid1()  # 根据 时间戳生成 uuid , 保证全球唯一
    b = ''.join(rs + str(a).split("-"))  # 生成将随机字符串 与 uuid拼接
    return b  # 返回随机字符串


class QuoteData(object):
    def __init__(self, df: (pd.DataFrame,), code='Code', date='date', target_cols=None, start=None, end=None):
        if start is None:
            start = '2000-01-01'
        if end is None:
            end = datetime.datetime.now().strftime('%Y-%m-%d')
        self._start = start
        self._end = end
        if target_cols is None:
            target_cols = list(OHLCV_AGG.keys())
        self._data = df[(df[date] >= self._start) and (df[date] <= self._end)].sort_values(date, ascending=True)
        self.target_cols = target_cols

        self._length = len(self._data)
        self._data_cols = self._data.columns.tolist()
        if code in self._data_cols and date in self._data_cols:
            self._general_cols = [date, code]
        else:
            raise AttributeError(f'Column {code} or {date} not in data')

        self._setup()

    def __len__(self):
        return self._length

    @lru_cache(maxsize=100)
    def _obtain_data(self, key):
        if key in self._data_cols:

            cols = self._general_cols + [key]
            if len(set(cols)) <= 2:
                return self._data[key]
            else:
                return self._data[cols].set_index(self._general_cols)
        else:
            raise AttributeError(f"Column '{key}' not in data")

    def __getitem__(self, key):
        return self._obtain_data(key)

    def __getattr__(self, key):
        try:
            return self._obtain_data(key)
        except KeyError:
            raise AttributeError(f"Column '{key}' not in data")

    def __iter__(self):
        for dt, data in self._data.groupby(self._general_cols[0]):
            yield dt, data

    def _setup(self):
        for col in self._data_cols:
            setattr(self, col, self._obtain_data(col))


class Strategy(metaclass=ABCMeta):
    """
    提供用户自定义界面
    """

    # def __str__(self):
    #     params = ','.join(f'{i[0]}={i[1]}' for i in zip(self._params.keys(),
    #                                                     map(_as_str, self._params.values())))
    #     if params:
    #         params = '(' + params + ')'
    #     return f'{self.__class__.__name__}{params}'

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    @abstractmethod
    def init(self, *args, **kwargs):
        """
        初始化
        :return:
        """
        pass

    @abstractmethod
    def next(self) -> Iterable:
        """
        iter operate strategy for each day
        :return:
        """
        pass


class Broker(object):
    """
    提供用户策略和Orders转换工具
    """

    def __init__(self, *, data, cash, commission, margin, trade_on_close, hedging, exclusive_orders, index):
        assert 0 < cash, f"cash should be >0, is {cash}"
        assert 0 <= commission < .1, f"commission should be between 0-10%, is {commission}"
        assert 0 < margin <= 1, f"margin should be between 0 and 1, is {margin}"
        self._data = data
        self._cash = cash
        self._commission = commission
        self._leverage = 1 / margin
        self._trade_on_close = trade_on_close
        self._hedging = hedging
        self._exclusive_orders = exclusive_orders

        # self._equity = np.tile(np.nan, len(index))
        self.orders = []
        self.trades = []
        self.closed_trades = []

    @staticmethod
    def set_order(size: float,
                  limit_price: float = None,
                  stop_price: float = None,
                  sl_price: float = None,
                  tp_price: float = None,
                  order_id=None,
                  parent_trade=None):
        return Orders(size,
                      limit_price=limit_price,
                      stop_price=stop_price,
                      sl_price=sl_price,
                      tp_price=tp_price,
                      order_id=order_id,
                      parent_trade=parent_trade)
        pass

    def sell(self, orders):
        pass

    pass


class BackTest(object):
    """
    主程序, run
    """

    def run(self):
        """
        the run function to begin back testing
        :return:
        """
        pass

    @staticmethod
    def get_data(*args, **kwargs):
        """
        prepare required data
        :param args:
        :param kwargs:
        :return:
        """
        pass

    pass


class Orders(object):
    """
    订单执行系统
    """

    def __init__(self,
                 size: float,
                 limit_price: float = None,
                 stop_price: float = None,
                 sl_price: float = None,
                 tp_price: float = None,
                 order_id=None,
                 create_date=None,
                 parent_trade=None):
        ORDERSCreator = namedtuple("ORDER", ('order_id', 'size', 'limit_price', 'stop_price', 'sl_price', 'tp_price'))
        self.create_date = create_date
        self._order_id = _order_id = order_id if order_id is not None else 'order_' + random_str(num=19)
        self._parent_trade = parent_trade
        self._attr = ORDERSCreator(_order_id, size, limit_price, stop_price, sl_price, tp_price)

    def __repr__(self):
        attr = (('order_id', self._order_id),
                ('size', self._attr.size),
                ('limit', self._attr.limit_price),
                ('stop', self._attr.stop_price),
                ('sl', self._attr.sl_price),
                ('tp', self._attr.tp_price),
                ('create_date', self.create_date),
                )
        settings = ','.join([f'{name}={value}' for name, value in attr])

        return f'<Order {settings}>'

    @property
    def size(self):
        return self._attr.size

    @property
    def limit_price(self):
        return self._attr.limit_price

    @property
    def stop_price(self):
        return self._attr.stop_price

    @property
    def sl_price(self):
        return self._attr.sl_price

    @property
    def tp_price(self):
        return self._attr.tp_price

    @property
    def is_long(self):
        return self.size > 0

    @property
    def is_short(self):
        return not self.is_long

    pass


class Positions(object):
    """
    计算每个时间点的仓位信息
    """
    pass


class Indicators(object):
    """
    计算回测结果的指标信息
    """
    pass


if __name__ == '__main__':
    from Nodes.test import GOOG

    GOOG = GOOG.reset_index().rename(columns={'index': 'date'})
    GOOG['Code'] = 'GOOG'

    QD = QuoteData(GOOG)
    for dt, s in iter(QD):
        print(dt)
    pass
