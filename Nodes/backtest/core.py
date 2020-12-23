# coding=utf-8
from abc import abstractmethod, ABCMeta
from collections import Iterable
from collections import OrderedDict, namedtuple
from Nodes.test import GOOG
import datetime
import pandas as pd
from functools import lru_cache
import numpy as np

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
        self._data = df[(df[date] >= self._start) & (df[date] <= self._end)].sort_values(date, ascending=True)
        self.target_cols = target_cols

        self._length = len(self._data)
        self._data_cols = self._data.columns.tolist()
        if code in self._data_cols and date in self._data_cols:
            self._general_cols = [date, code]
        else:
            raise AttributeError(f'Column {code} or {date} not in data')

        self._setup()

    def shape(self):
        return self._data

    def length(self):
        return len(self._data[self._general_cols[0]].unique())

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
    @staticmethod
    def set_order(size: float,
                  limit_price: float = None,
                  stop_price: float = None,
                  sl_price: float = None,
                  tp_price: float = None,
                  order_id=None,
                  parent_trade=None):
        return Order(size,
                     limit_price=limit_price,
                     stop_price=stop_price,
                     sl_price=sl_price,
                     tp_price=tp_price,
                     order_id=order_id,
                     parent_trade=parent_trade)

    @classmethod
    def create_orders(cls, scripts,

                      default_limit=None,
                      default_stop=-np.inf, default_sl=-np.inf,
                      default_tp=np.inf):
        default_dict = {'limit': default_limit, 'stop': default_stop, 'sl': default_sl, 'tp': default_tp}
        cols = scripts.columns
        must = ['date', 'code', 'size']
        reqired = ['limit', 'stop', 'sl', 'tp']
        missed = set(must) - set(cols)
        if len(missed) != 0:
            raise ValueError(f"{','.join(missed)} are missing")

        for c in reqired:
            if c in cols:
                pass
            else:
                scripts[c] = default_dict[c]


class Order(object):
    """
    订单执行系统
    """

    def __init__(self,
                 size: float,
                 code: str,
                 limit_price: float = None,
                 stop_price: float = None,
                 sl_price: float = None,
                 tp_price: float = None,
                 order_id=None,
                 create_date=None,
                 parent_trade=None):
        ORDERSCreator = namedtuple("ORDER",
                                   ('order_id', 'code', 'size', 'limit_price', 'stop_price', 'sl_price', 'tp_price'))
        self.create_date = create_date
        self._order_id = _order_id = order_id if order_id is not None else 'order_' + random_str(num=19)
        self._parent_trade = parent_trade
        self._attr = ORDERSCreator(_order_id, code, size, limit_price, stop_price, sl_price, tp_price)

    def __repr__(self):
        attr = (('order_id', self._order_id),
                ('code', self._attr.code),
                ('size', self._attr.size),
                ('limit', self._attr.limit_price),  # 限价单或者市场单
                ('stop', self._attr.stop_price),  # 止盈止损市场单
                ('sl', self._attr.sl_price),  # 止损限价单
                ('tp', self._attr.tp_price),  # 止盈限价单
                ('create_date', self.create_date))
        settings = ','.join([f'{name}={value}' for name, value in attr])

        return f'<Order {settings}>'


    def _adjusted_price(self,size, commission, size, price) -> float:
        """
        Long/short `price`, adjusted for commisions.
        In long positions, the adjusted price is a fraction higher, and vice versa.
        """
        return (price or self.last_price) * (1 + copysign(self._commission, size))

    def oper(self, quotedata):
        code_data = quotedata[quotedata['code'] == self._attr.code]

        """
        default_stop=-np.inf, default_sl=-np.inf,default_tp=np.inf
        """

        if self.is_long:

            if not (self.sl_price or -np.inf) < (self.limit_price or self.stop_price or adjusted_price) < (self.tp_price or np.inf):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({self.sl_price}) < LIMIT ({self.limit_price or self.stop_price or adjusted_price}) < TP ({self.tp_price})")
        else:
            if not (self.tp_price or -np.inf) < (self.limit_price or self.stop_price or adjusted_price) < (self.sl_price or np.inf):
                raise ValueError(
                    "Short orders require: "
                    f"TP ({self.tp_price}) < LIMIT ({self.limit_price or self.stop_price or adjusted_price}) < SL ({self.sl_price})")


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
    import numpy as np

    # GOOG = GOOG.reset_index().rename(columns={'index': 'date'})
    # GOOG['Code'] = 'GOOG'

    # QD = QuoteData(GOOG)
    # print(len(QD))
    price = pd.DataFrame(np.random.random(size=(GOOG.shape[0], 1)), columns=['GOOG'])
    orders = (price > 0.5) * 1
    price['date'] = GOOG.index
    orders['date'] = GOOG.index
    scripts = orders.set_index('date').stack().reset_index()
    scripts.columns = ['date', 'code', 'size']
    from Nodes.test import GOOG

    ['size', 'limit', 'stop', 'sl', 'tp']
    QD = QuoteData(GOOG)

    "A limit order is a type of order to purchase or sell a security at a specified price or better. " \
    "For buy limit orders, the order will be executed only at the limit price or a lower one, " \
    "while for sell limit orders, the order will be executed only at the limit price or a higher one. " \
    "This stipulation allows traders to better control the prices they trade."

    # class BuyandHoldStrategy(Strategy):
    #     def init(self, scripts, cols=['date', 'code', 'size']):
    #         code_list = scripts['code'].unique().tolist()
    pass

pass
