# coding=utf-8
from abc import abstractmethod, ABCMeta
from collections import OrderedDict, namedtuple, Iterable
# from Nodes.test import GOOG
import datetime
import pandas as pd
from functools import lru_cache
import numpy as np
# import random
import uuid

# from Nodes.utils_node.file_cache import file_cache

# from collections import OrderedDict


OHLCV_AGG = OrderedDict((
    ('Open', 'first'),
    ('High', 'max'),
    ('Low', 'min'),
    ('Close', 'last'),
    ('Volume', 'sum'),
))


def random_str(num=6):
    # uln = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    # rs = ''.join(random.sample(uln, num))  # 生成一个 指定位数的随机字符串
    rs = ''
    a = uuid.uuid1()  # 根据 时间戳生成 uuid , 保证全球唯一
    b = rs + a.hex  # 生成将随机字符串 与 uuid拼接
    return b  # 返回随机字符串


# @file_cache
def create_quote(data):
    if isinstance(data, pd.DataFrame):
        QD = QuoteData(data)
        return QD
    elif isinstance(data, QuoteData):
        return data
    else:
        raise ValueError('quote data is not pd.DataFrame or QuoteData!')


class Order(object):
    """
    订单执行系统
    """
    __slots__ = ['commission', 'create_date', '_order_id', '_parent_trade', '_attr', '_is_cancel']

    def __init__(self,
                 commission: float,
                 size: float,
                 code: str,
                 limit_price: float = None,
                 stop_price: float = None,
                 sl_price: float = None,
                 tp_price: float = None,
                 order_id=None,
                 create_date=None,
                 parent_trade=None, ):
        self.commission = commission
        ORDERSCreator = namedtuple("ORDER",
                                   ('order_id', 'code', 'size', 'limit_price', 'stop_price', 'sl_price', 'tp_price'))
        self.create_date = create_date
        self._order_id = _order_id = order_id if order_id is not None else 'order_' + random_str(num=6)
        self._parent_trade = parent_trade
        self._attr = ORDERSCreator(_order_id, code, size, limit_price, stop_price, sl_price, tp_price)
        self._is_cancel = False

    def cancel(self):
        self._is_cancel = True

    def __repr__(self):
        attr = (('order_id', self._order_id),
                ('code', self._attr.code),  # 证券code
                ('size', self._attr.size),  # 买入金额
                ('limit', self._attr.limit_price),  # 限价单或者市场单
                ('stop', self._attr.stop_price),  # 止盈止损市场单
                ('sl', self._attr.sl_price),  # 止损限价单
                ('tp', self._attr.tp_price),  # 止盈限价单
                ('create_date', self.create_date))
        settings = ','.join([f'{name}={value}' for name, value in attr])

        return f'<Order {settings}>'

    def _adjusted_price(self, last_price, price) -> float:
        """
        Long/short `price`, adjusted for commisions.
        In long positions, the adjusted price is a fraction higher, and vice versa.
        """
        return (price or last_price) * (1 + np.copysign(self.commission, self.size))  # price * commission fee

    def operate(self, quote, code_col='Code', last_price=None):
        code = self._attr.code

        price = quote[(quote[code_col] == code)]['Close'].values[0]
        adjusted_price = self._adjusted_price(last_price, price)
        self._detect_trade_side_valid(adjusted_price)

        return

    def _detect_trade_side_valid(self, adjusted_price):
        """
        detect trade side whether valid
        :param adjusted_price:
        :return:
        """
        if self.is_long:
            if not (self.sl_price or -np.inf) < (self.limit_price or self.stop_price or adjusted_price) < (
                    self.tp_price or np.inf):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({self.sl_price}) < LIMIT ({self.limit_price or self.stop_price or adjusted_price}) < TP ({self.tp_price})")

        else:
            if not (self.tp_price or -np.inf) < (self.limit_price or self.stop_price or adjusted_price) < (
                    self.sl_price or np.inf):
                raise ValueError(
                    "Short orders require: "
                    f"TP ({self.tp_price}) < LIMIT ({self.limit_price or self.stop_price or adjusted_price}) < SL ({self.sl_price})")

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

    def sort(self):
        filtered_order = self.items()
        self.orders = OrderedDict(sorted(filtered_order, key=lambda x: x[0]))

    def update(self, item: dict):
        for k, v in item.items():
            if isinstance(k, str):
                d = self.orders.get(k, default=[])
            else:
                raise ValueError(f'{k} is not str')
            if isinstance(v, (list,)):
                d.extend(v)
            else:
                raise ValueError(f'{v} is not list')

    def keys(self):
        return self.orders.keys()

    def items(self):
        return self.orders.items()

    def values(self):
        return self.orders.values()


class QuoteData(object):
    # __slots__ =

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
        # self.__slots__ = ['_start', '_end', '_data', '_length', '_data_cols', '_general_cols', 'date_list', 'shape',
        #                   'length',  'target_cols'] + self._data_cols
        self._setup()

    # def date_list(self):
    #     return getattr(self, self._general_cols[0]).unique()

    @property
    def shape(self):
        return self._data.shape

    @property
    def length(self):
        return self.shape[0]

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

    def _getitem_bool_array(self, indexer):

        return self._data[indexer]

    def where(self, dt: (str, list), col=None):
        if isinstance(dt, str):
            dt = [dt]
        elif isinstance(dt, list):
            pass
        else:
            raise ValueError('dt must be str or list')
        if col is None:
            col = self._general_cols[0]
        indexer = self._data[col].isin(dt)
        return self._getitem_bool_array(indexer)

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


class Positions(object):
    """
    计算每个时间点的仓位信息
    """

    def last_position(self, dt, code):
        last_dt = self.get_last_dt(dt, self.dt_list)
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

    def get_last_dt(self, dt, dt_list):
        # dt_list = self.date_list()
        if dt in dt_list:
            index_code = dt_list.index(dt)
            if index_code == 0:
                return dt
            else:
                return dt_list[index_code - 1]
        else:
            raise ValueError(f'cannot found {dt}')

    def date_list(self):
        return sorted(self._obj.keys())

    def sorted(self):
        return OrderedDict(sorted(self._obj.items(), key=lambda x: x[0]))

    def __init__(self, dt_list):
        self._obj = {}
        # self.traded = {dt: [] for dt in dt_list}
        self.dt_list = dt_list

    def __setitem__(self, key, value):
        exists = self.__getitem__(key)
        if isinstance(value, tuple):
            exists.append(value)

        # elif isinstance(value, list):
        #     exists.extend(value)

        else:
            raise ValueError(f'{value} must be tuple or namedtuple')

        self._obj[key] = exists

    def __getitem__(self, key):
        return self._obj.get(key, default=[])

    #     if item not in self._obj.keys():
    #         return []
    #     else:
    #         return self._obj[item]

    def append(self, dt, code: str, value: float, cost: float):
        # exists = False
        for exists in self.__getitem__(dt):
            for e in exists:
                if e.code == code:
                    raise ValueError(f'detect duplicates code:{code}')
        else:
            self.raw_append(dt, code, value, cost)

        # filter(lambda x: x.code ,exists)

    def raw_append(self, dt, code: str, value: float, cost: float):
        element_creator = namedtuple('element', ['code', 'value', 'cost'])
        ele = element_creator(code, value, cost)
        self.__setitem__(dt, ele)

    # def create_element(self, element_code: str, value: float, cost_price: float):
    #     return element_creator(element_code, value, cost_price)


class Indicators(object):
    """
    计算回测结果的指标信息
    """
    pass


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
    提供用户策略和Orders转换工具,不提供存储功能
    """
    __slots__ = ['_data', '_cash', '_commission', '_leverage', '_trade_on_close', '_hedging', '_exclusive_orders']

    def __init__(self, data, cash, commission, margin, trade_on_close, hedging, exclusive_orders):
        assert 0 < cash, f"cash should be >0, is {cash}"
        assert 0 <= commission < .1, f"commission should be between 0-10%, is {commission}"
        assert 0 < margin <= 1, f"margin should be between 0 and 1, is {margin}"
        self._data = create_quote(data)
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

    def create_orders(self, scripts: pd.DataFrame, default_limit=None,
                      default_stop=-np.inf, default_sl=-np.inf, default_tp=np.inf, ):
        """create all orders at begin"""
        a = list(self._create_orders(scripts, self._commission, default_limit=default_limit, default_stop=default_stop,
                                     default_sl=default_sl, default_tp=default_tp))

        return Orders(*a)

    @staticmethod
    def _create_orders(scripts: pd.DataFrame, _commission, default_limit=None,
                       default_stop=-np.inf, default_sl=-np.inf, default_tp=np.inf, ):
        default_dict = {'limit': default_limit, 'stop': default_stop, 'sl': default_sl, 'tp': default_tp}
        cols = scripts.columns.tolist()
        must = ['date', 'code', 'size']
        reqired = ['limit', 'stop', 'sl', 'tp']

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

    def __call__(self, orders: Orders, date_col='date'):
        dt_list = list(orders.keys())
        filtered_quote = self._data.where(dt_list)
        # h = []
        for dt, order_list in orders.items():
            single_dt_quote = filtered_quote[filtered_quote[date_col] == dt]
            # res = list(map(lambda x: x.operate(single_dt_quote), order_list))
            # h.append(res)
            yield dt, single_dt_quote, order_list
        # return h


class BackTest(object):
    __slots__ = ['scripts', 'broker', 'orders', 'quote', 'trades', 'closed_trades', 'positions']

    def __init__(self, scripts, data, cash, commission, margin, trade_on_close, hedging, exclusive_orders,
                 default_limit=None, default_stop=-np.inf, default_sl=-np.inf, default_tp=np.inf):
        self.scripts = scripts
        self.quote = create_quote(data)
        self.broker = Broker(self.quote, cash, commission, margin, trade_on_close, hedging, exclusive_orders)
        self.orders = self.broker.create_orders(scripts, default_limit=default_limit, default_stop=default_stop,
                                                default_sl=default_sl, default_tp=default_tp)
        dt_list = sorted(self.quote.date.unique())
        self.positions = Positions(dt_list)

    """
    主程序, run
    """

    @staticmethod
    def operate(order, quote, current_position):
        return order, quote

    def run(self):
        """
        the run function to begin back testing
        :return:
        """

        for dt, single_dt_quote, order_list in self.broker(self.orders):
            for o in order_list:
                code = o._attr.code

                last_position = self.positions.last_position(dt, code)
                current_position = self.positions.current_position(dt, code)
                res = self.operate(o, single_dt_quote, current_position)

                # else:
                #     dt = o.create_date
                #     price = self.broker._data[self.broker._data['date'] == dt]['price']
                #     o.oper(None, price)

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


# class Orders(object):
#     def __init__(self, *order):
#         self._orders = order
#
#     @staticmethod
#     def set_order(size: float,
#                   limit_price: float = None,
#                   stop_price: float = None,
#                   sl_price: float = None,
#                   tp_price: float = None,
#                   order_id=None,
#                   parent_trade=None):
#         return Order(size,
#                      limit_price=limit_price,
#                      stop_price=stop_price,
#                      sl_price=sl_price,
#                      tp_price=tp_price,
#                      order_id=order_id,
#                      parent_trade=parent_trade)


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
    orders_df = (price > 0.5) * 1
    orders_df['date'] = GOOG.index
    scripts = orders_df.set_index('date').stack().reset_index()
    scripts.columns = ['date', 'code', 'size']

    GOOG = GOOG.reset_index().rename(columns={'index': "date"})

    QD = QuoteData(GOOG)

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
    bt = BackTest(scripts, QD, cash, commission, margin, trade_on_close, hedging, exclusive_orders)
    bt.run()
    print(1)

pass
