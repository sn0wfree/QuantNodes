# coding=utf-8
from collections import OrderedDict, namedtuple, deque
import pandas as pd
from functools import singledispatch, lru_cache

from Nodes.backtest.Orders import Trade
from Nodes.backtest.Indicators import Indicators,Statistics

class cached_property(object):
    """
    Decorator that converts a method with a single self argument into a
    property cached on the instance.
    Optional ``name`` argument allows you to make cached properties of other
    methods. (e.g.  url = cached_property(get_absolute_url, name='url') )
    """

    def __init__(self, func, name=None):
        self.func = func
        self.__doc__ = getattr(func, '__doc__')
        self.name = name or func.__name__

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res


# import random
# from Nodes.test import GOOG
def create_element():
    name = 'element'
    cols = ['code', 'trade_amount', 'deal_price', 'fee']
    element_creator = namedtuple(name, cols)  # 股票代码, 交易金额，交易价格 ，交易费用

    def share(self):
        return self.trade_amount / self.deal_price

    cls_obj = type('ele', (element_creator,), {'share': property(share)})
    return cls_obj


cls_obj = create_element()


class EmptyPositions(Exception):
    pass


class PositionSectionTrade(deque):
    def __init__(self, dt, *orders, maxlen=100000):
        super(PositionSectionTrade, self).__init__(orders, maxlen=maxlen)
        self.dt = dt
        self.stats = Statistics()
        # self.traded = {dt: [] for dt in dt_list}

    def transform(self):
        for trade in self:
            value = trade.trade_result_trade_size  # 成交金额
            cost = trade.deal_price
            code = trade.code
            fee = trade.fee
            yield cls_obj(code, value, cost, fee)

    def codes(self):
        return set(map(lambda x: x.code, self))

    def get_code_trade(self, code):
        for trade in self:
            if trade.code == code:
                yield trade

    @lru_cache(maxsize=100)
    def get_code_shares(self, code):
        return sum(map(lambda x: x.order_size, self.get_code_trade(code)))

    @lru_cache(maxsize=100)
    def get_code_values(self, code, current_price):
        return self.get_code_shares(code) * current_price

    def tolist(self):
        return list(map(lambda x: x.tolist(), self))



class Positions(object):
    """
    计算每个时间点的仓位信息
    """

    def last_position(self, dt, code):
        last_dt = self.get_last_dt(dt, self.dt_list, raiseerror=False)
        return self.current_position(last_dt, code)

    def position_share(self, dt, code):
        exists = self.__getitem__(dt)
        shares = exists.get_code_shares(code)
        # res = list(filter(lambda x: x.code == code, exists))
        # length = len(res)
        # if length == 0:
        #     return 0
        # elif length == 1:
        #     return res[0].value
        return shares
        # else:
        #     raise ValueError('POSITION HOLD MULTI VALUE FOR SAME CODE！')

    @staticmethod
    def get_last_dt(dt, dt_list, raiseerror=True):
        # dt_list = self.date_list()
        if dt in dt_list:
            index_code = dt_list.index(dt)
            if index_code == 0:
                return dt
            else:
                return dt_list[index_code - 1]
        else:
            msg = f'cannot found {dt}'
            if raiseerror:
                raise EmptyPositions(msg)
            else:
                print(msg)
                return None

    def keys(self):
        return self._trades.keys()

    def values(self):
        return self._trades.values()

    def to_list(self):
        h = []
        for e in self.values():
            h.extend(e.tolist())
        return h

    def to_pandas(self):
        return pd.DataFrame(self.to_list())

    def items(self):
        return self._trades.items()

    def sorted(self):
        return OrderedDict(sorted(self.items(), key=lambda x: x[0]))

    __slots__ = ['_position_sign_', '_trades']

    def __init__(self):
        # self._obj = {}
        self._position_sign_ = 'positions'

        # self.traded = {dt: [] for dt in dt_list}
        self._trades = {}

    @property
    def dt_list(self):
        return list(self._trades.keys())

    # def get_trade(self, key):
    #     return self._trades.get(key, PositionSection(key))
    #
    # def set_trade(self, key, value):
    #     exists = self.get_trade(key)
    #     exists.append(value)
    #     self._trades[key] = exists

    def __setitem__(self, key, value):
        exists = self.__getitem__(key)
        if isinstance(value, (tuple, PositionSectionTrade, Trade)):
            exists.append(value)
        else:
            raise ValueError(f'{value} must be tuple or namedtuple')
        self._trades[key] = exists

    def __getitem__(self, key):
        return self._trades.get(key, PositionSectionTrade(key))

    # def update(self, item):
    #     if hasattr(self, '_position_sign_') and self._position_sign_ == 'position':
    #         # keys = self.keys()
    #         for dt, ele in item.items():
    #             self.__setitem__(dt, ele)
    #     else:
    #         raise ValueError(f'update function only accept position sign class to update!')

    @singledispatch
    def append(self, trade: Trade, reduce=False, check_duplicate=True):

        self.__setitem__(trade.dt, trade)

    @property
    def _obj(self):
        return {dt: list(trade_list.transform()) for dt, trade_list in self.items()}

    # # self.set_trade(dt, trade)
    # value = trade.trade_result_trade_size  # 成交金额
    # # dt_trade = self[dt]  # 获取对应dt，如果没有这返回默认的PositionSection(dt)
    # # dt_trade.append(trade)
    # cost_price = trade.deal_price
    # code = trade.code
    # fee = trade.fee
    # self.raw_append(dt, code, value, cost_price, fee, check_duplicate=check_duplicate)

    # @append.register
    # def append(self,trade):
    #     # exists = False
    #
    #     self.raw_append(dt, code, value, cost, fee)

    # filter(lambda x: x.code ,exists)
    def check_duplicate_code(self, dt, code: str):
        ## todo 可能的性能点,可能存在重复计算的可能性
        for exists in self.__getitem__(dt):
            for e in exists:
                if e.code == code:
                    # 当天存在多笔交易
                    raise ValueError(f'detect duplicates code:{code}')

    # def raw_append(self, dt, code: str, value: float, cost: float, fee: float, check_duplicate=True):
    #     """
    #
    #     :param check_duplicate:
    #     :param dt:  trade date
    #     :param code:  股票代码
    #     :param value:   交易金额
    #     :param cost:  交易价格
    #     :param fee:  交易费用
    #     :return:
    #     """
    #     if check_duplicate:
    #         self.check_duplicate_code(dt, code)
    #     # element_creator = namedtuple('element', ['code', 'trade_amount', 'deal_price', 'fee'])  # 股票代码, 交易金额，交易价格 ，交易费用
    #
    #     self.__setitem__(dt, cls_obj(code, value, cost, fee))

    # @trade_append.register(Iterable)
    # @trade_append.register(tuple)
    # @trade_append.register(list)
    def trade_extend(self, trades, reduce=False):
        for trade in trades:
            self.append(trade, reduce=reduce)

    # def create_element(self, element_code: str, value: float, cost_price: float):
    #     return element_creator(element_code, value, cost_price)
