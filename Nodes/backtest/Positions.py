# coding=utf-8
from collections import OrderedDict, namedtuple, deque

from functools import singledispatch

from Nodes.backtest.Orders import Trade


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

    __slots__ = ['_obj', 'dt_list', '_position_sign_']

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
    def append(self, trade: Trade, reduce=False):
        value = trade.trade_result_trade_side  # 成交金额
        # dt_trade = self[dt]  # 获取对应dt，如果没有这返回默认的PositionSection(dt)
        # dt_trade.append(trade)
        if reduce and value == 0:
            pass
        else:
            dt = trade.dt
            cost_price = trade.deal_price
            code = trade.code
            fee = trade.fee
            self.raw_append(dt, code, value, cost_price, fee)

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

    def raw_append(self, dt, code: str, value: float, cost: float, fee: float, check_duplicate=True):
        """

        :param check_duplicate:
        :param dt:  trade date
        :param code:  股票代码
        :param value:   交易金额
        :param cost:  交易价格
        :param fee:  交易费用
        :return:
        """
        if check_duplicate:
            self.check_duplicate_code(dt, code)
        # element_creator = namedtuple('element', ['code', 'trade_amount', 'deal_price', 'fee'])  # 股票代码, 交易金额，交易价格 ，交易费用

        self.__setitem__(dt, cls_obj(code, value, cost, fee))

    # @trade_append.register(Iterable)
    # @trade_append.register(tuple)
    # @trade_append.register(list)
    def trade_extend(self, trades, reduce=False):
        for trade in trades:
            self.append(trade, reduce=reduce)

    # def create_element(self, element_code: str, value: float, cost_price: float):
    #     return element_creator(element_code, value, cost_price)
