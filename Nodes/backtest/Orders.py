# coding=utf-8
# import random
import copy
import warnings
from collections import OrderedDict, namedtuple

# from Nodes.test import GOOG
import numpy as np

from Nodes.backtest.bt_utils import random_str


# from Nodes.utils_node.file_cache import file_cache
class Trade(object):
    __slots__ = ['correspond_order', 'side', 'deal_price', 'adjusted_price', '_status']

    def __init__(self, order, deal_price, adjusted_price, side, status='completed'):
        self.correspond_order = order  # 对应的order 实例
        self.side = side
        self.deal_price = deal_price
        self.adjusted_price = adjusted_price
        self._status = status

    @property
    def code(self):
        return self.correspond_order._attr.code

    @property
    def dt(self):
        return self.correspond_order.create_date

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, stau):
        self._status = stau

    @property
    def order_size(self):
        return self.correspond_order._attr.size

    @property
    def traded_size(self):
        if self.status == 'completed':
            return self.order_size
        else:
            return 0

    @property
    def trade_result_cost_side(self):
        return self.adjusted_price * self.traded_size

    @property
    def trade_result_trade_side(self):
        return self.deal_price * self.traded_size

    @property
    def fee(self):
        return abs(self.trade_result_trade_side - self.trade_result_cost_side)


class Order(object):
    """
    订单执行系统
    """
    __slots__ = ['commission', 'create_date', '_order_id', '_parent_trade', '_attr', '_is_cancel', 'adj_is_long']

    def __init__(self,
                 commission: float,
                 size: float,  # 买入份数
                 code: str,  # 证券code
                 limit_price: float = None,  # 限价单或者市场单
                 stop_price: float = None,  # 止盈止损市场单
                 sl_price: float = None,  # 止损限价单
                 tp_price: float = None,  # 止盈限价单
                 order_id=None,
                 create_date=None,
                 parent_trade=None, ):
        self.commission = commission  ## TODO 通过创建float类进行分析commission多样化计算
        ORDERSCreator = namedtuple("ORDER",
                                   ('order_id', 'code', 'size', 'limit_price', 'stop_price', 'sl_price', 'tp_price'))
        self.create_date = create_date
        self._order_id = _order_id = order_id if order_id is not None else 'order_' + random_str(num=6)
        self._parent_trade = parent_trade
        self.adj_is_long = pow(1, (size <= 0) * 1)
        if sl_price is None:
            sl_price = -np.inf * self.adj_is_long
        if tp_price is None:
            tp_price = np.inf * self.adj_is_long
        self._attr = ORDERSCreator(_order_id, code, size, limit_price, stop_price, sl_price, tp_price)
        self._is_cancel = False

    def cancel(self):
        self._is_cancel = True

    def __repr__(self):
        attr = (('order_id', self._order_id),
                ('code', self._attr.code),  # 证券code
                ('size', self._attr.size),  # 买入份数
                ('limit', self._attr.limit_price),  # 限价单或者市场单
                ('stop', self._attr.stop_price),  # 止盈止损市场单
                ('sl', self._attr.sl_price),  # 止损限价单
                ('tp', self._attr.tp_price),  # 止盈限价单
                ('create_date', self.create_date))
        settings = ','.join([f'{name}={value}' for name, value in attr])

        return f'<Order {settings}>'

    def copy(self):
        new_order = copy.deepcopy(self)
        return new_order

    def check_buy(self, adjusted_price, raiseError=False):
        """

        :param adjusted_price:
        :param raiseError:
        :return:
        """
        if not (self._attr.sl_price or -np.inf) < (
                self._attr.limit_price or self._attr.stop_price or adjusted_price) < (
                       self.tp_price or np.inf):
            msg = f"Long orders require: SL ({self._attr.sl_price}) < LIMIT ({self._attr.limit_price or self._attr.stop_price or adjusted_price}) < TP ({self._attr.tp_price})"
            if raiseError:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return False
        else:
            return True

    def check_sell(self, adjusted_price, raiseError=False):
        if not (self._attr.tp_price or -np.inf) < (
                self._attr.limit_price or self._attr.stop_price or adjusted_price) < (
                       self._attr.sl_price or np.inf):
            msg = f"Short orders require: TP ({self._attr.tp_price}) < LIMIT ({self._attr.limit_price or self._attr.stop_price or adjusted_price}) < SL ({self._attr.sl_price})"
            if raiseError:
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return False
        else:
            return True

    def check_available(self, adjusted_price, raiseError=False):
        ## TODO not consider situation when size > volume
        if self.adj_is_long:
            return self.check_buy(adjusted_price, raiseError=raiseError), 'buy'
        else:
            return self.check_sell(adjusted_price, raiseError=raiseError), 'sell'

    @staticmethod
    def _adjusted_price(last_price, price, commission, size) -> float:
        """
        Long/short `price`, adjusted for commisions.
        In long positions, the adjusted price is a fraction higher, and vice versa.
        """
        return (price or last_price) * (1 + np.copysign(commission, size))  # price * commission fee

    # @classmethod
    # def _detect_trade_side_valid(cls, size, sl_price, limit_price, stop_price, tp_price, adjusted_price):
    #     """
    #     detect trade side whether valid
    #     :param adjusted_price:
    #     :return:
    #     """
    #     # adjusted_price = cls._adjusted_price(last_price, price, commission, size)
    #     is_long = size > 0
    #     if is_long:
    #         if not (sl_price or -np.inf) < (limit_price or stop_price or adjusted_price) < (
    #                 tp_price or np.inf):
    #             raise ValueError(
    #                 "Long orders require: "
    #                 f"SL ({sl_price}) < LIMIT ({limit_price or stop_price or adjusted_price}) < TP ({tp_price})")
    #
    #     else:
    #         if not (tp_price or -np.inf) < (limit_price or stop_price or adjusted_price) < (
    #                 sl_price or np.inf):
    #             raise ValueError(
    #                 "Short orders require: "
    #                 f"TP ({tp_price}) < LIMIT ({limit_price or stop_price or adjusted_price}) < SL ({sl_price})")

    def deal(self, quote):
        """
        清算order 然后决定是否能够成交
        :param quote: QuotaData
        :return:
        """
        data = quote.filter(self, to_quote=False)
        price = data['Close'].values.ravel()[0]
        adjusted_price = self._adjusted_price(None, price, self.commission, self.size)

        available, side = self.check_available(adjusted_price, raiseError=True)
        ## todo consider connot deal situation
        ## TODO not consider situation when size > volume
        # if self.is_long:
        #     self.check_buy(adjusted_price)
        # else:
        #     self.check_sell(adjusted_price)

        return Trade(self.copy(), price, adjusted_price, side, status='completed')

        #     # msg = 'status completed! will do nothing!'
        #     # print(msg)
        #     # return current_p
        #     else:
        #         if current_p >
        #
        #     return order, quote
        #
        # price = quote['Close'].values[0]
        #
        # if current_position == self.size:
        #     msg = 'have some order traded before, status has been completed! will do nothing!'
        #     print(msg)
        #     return Trade()
        # else:
        #     if current_position > self.size:
        #         msg = f'will sell {self.size - current_position}'

    # def operate(self, quote, code_col='Code', last_price=None):
    #     code = self._attr.code
    #
    #     price = quote[(quote[code_col] == code)]['Close'].values[0]
    #     adjusted_price = self._adjusted_price(last_price, price)
    #     self._detect_trade_side_valid(adjusted_price)
    #
    #     return

    # def oper(self, last_price, price):
    #     # code_data = quotedata[quotedata['code'] == self._attr.code]
    #
    #     """
    #     default_stop=-np.inf, default_sl=-np.inf,default_tp=np.inf
    #     """
    #
    #     trade_side = self.is_long

    # return None

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


class Orders(object):
    __slots__ = ['orders', 'trades', 'closed_trades']

    def __init__(self, *order_info):
        filtered_order = sorted(
            filter(lambda s: len(s) == 2 and isinstance(s[0], str) and isinstance(s[1], list), order_info),
            key=lambda x: x[0])
        # for s in order_info :
        #     print( len(s) == 2,isinstance(s[0], str) , isinstance(s[1], Order))
        #     filtered_order.append(s)

        self.orders = OrderedDict(filtered_order)
        # self.trades = []
        # self.closed_trades = []

    def get(self, dt, default=[]):
        d = self.orders.get(dt, default)
        return d

    def sort(self):
        filtered_order = self.items()
        self.orders = OrderedDict(sorted(filtered_order, key=lambda x: x[0]))

    def update(self, item: dict):
        for k, v in item.items():
            if isinstance(k, str):
                d = self.orders.get(k, [])
            else:
                raise ValueError(f'{k} is not str')
            if isinstance(v, (list,)):
                d.extend(v)
            else:
                raise ValueError(f'{v} is not list')

    def keys(self):
        return self.orders.keys()

    def reduce_keys(self, exclude_value=0):
        for key, items in self.items():
            c = list(filter(lambda x: x.size != exclude_value, items))
            if len(c) != 0:
                yield key

    def items(self):
        return self.orders.items()

    def values(self):
        return self.orders.values()
