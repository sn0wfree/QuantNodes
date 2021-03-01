# coding=utf-8
from collections import OrderedDict, namedtuple, deque

from functools import singledispatch

from Nodes.backtest.Orders import Trade


# import random
# from Nodes.test import GOOG


class EmptyPositions(Exception):
    pass


class PositionSection(deque):
    def __init__(self, dt, *orders, maxlen=100000):
        super(PositionSection, self).__init__(orders, maxlen=maxlen)
        self.dt = dt
        # self.traded = {dt: [] for dt in dt_list}


class Positions(object):
    """
    计算每个时间点的仓位信息
    """

    def last_position(self, dt, code):
        last_dt = self.get_last_dt(dt, self.dt_list, raiseerror=False)
        return self.current_position(last_dt, code)

    def current_position(self, dt, code):
        exists = self.__getitem__(dt)
        res = list(filter(lambda x: x.code == code, exists))
        length = len(res)
        if length == 0:
            return 0
        elif length == 1:
            return res[0].value
        else:
            raise ValueError('POSITION HOLD MULTI VALUE FOR SAME CODE！')

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
        return self._obj.keys()

    def values(self):
        return self._obj.values()

    def items(self):
        return self._obj.items()

    def sorted(self):
        return OrderedDict(sorted(self.items(), key=lambda x: x[0]))

    def __init__(self, dt_list):
        self._obj = {}
        # self.traded = {dt: [] for dt in dt_list}
        self.dt_list = dt_list
        self._position_sign_ = 'position'
        # self.raw = {}

    def __setitem__(self, key, value):
        exists = self.__getitem__(key)
        if isinstance(value, (tuple, PositionSection)):
            exists.append(value)
        else:
            raise ValueError(f'{value} must be tuple or namedtuple')
        self._obj[key] = exists

    def __getitem__(self, key):
        return self._obj.get(key, PositionSection(key))

    def update(self, item):
        if hasattr(self, '_position_sign_') and self._position_sign_ == 'position':
            # keys = self.keys()
            for dt, ele in item.items():
                self.__setitem__(dt, ele)
        else:
            raise ValueError(f'update function only accept position sign class to update!')

    @singledispatch
    def append(self, trade: Trade):
        dt = trade.dt
        cost_price = trade.deal_price
        code = trade.code
        value = trade.trade_result_trade_side  # 交易金额
        fee = trade.fee
        # dt_trade = self[dt]  # 获取对应dt，如果没有这返回默认的PositionSection(dt)
        # dt_trade.append(trade)

        self.raw_append(dt, code, value, cost_price, fee)

    # @append.register
    # def append(self,trade):
    #     # exists = False
    #
    #     self.raw_append(dt, code, value, cost, fee)

    # filter(lambda x: x.code ,exists)

    def raw_append(self, dt, code: str, value: float, cost: float, fee: float):
        for exists in self.__getitem__(dt):
            for e in exists:
                if e.code == code:
                    raise ValueError(f'detect duplicates code:{code}')
        else:
            element_creator = namedtuple('element', ['code', 'value', 'cost_price', 'fee'])  # 股票代码, 交易金额，  交易价格 ，交易费用
            ele = element_creator(code, value, cost, fee)
            self.__setitem__(dt, ele)

    # @trade_append.register(Iterable)
    # @trade_append.register(tuple)
    # @trade_append.register(list)
    def trade_extend(self, trades):
        for trade in trades:
            self.append(trade)

    # def create_element(self, element_code: str, value: float, cost_price: float):
    #     return element_creator(element_code, value, cost_price)
