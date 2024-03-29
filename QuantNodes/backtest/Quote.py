# coding=utf-8
from collections import OrderedDict

# import random
# from QuantNodes.test import GOOG
import datetime
import pandas as pd
from functools import lru_cache, partial

# @file_cache


OHLCV_AGG = OrderedDict((
    ('Open', 'first'),
    ('High', 'max'),
    ('Low', 'min'),
    ('Close', 'last'),
    ('Volume', 'sum'),
))


class QuoteData(object):
    __slots__ = ['_start', '_end', '_data', '_length', '_data_cols', '_general_cols', 'date_list', 'target_cols']

    @staticmethod
    def create_quote(data):
        if isinstance(data, pd.DataFrame):
            return QuoteData(data)
        elif isinstance(data, QuoteData):
            return data
        else:
            raise ValueError('quote data is not pd.DataFrame or QuoteData!')

    @property
    def _recreate_quote_func(self):
        return partial(QuoteData, code=self.code_col, date=self.date_col, target_cols=self.target_cols,
                       start=self._start, end=self._end)

    def __init__(self, df: (pd.DataFrame,), code='Code', date='date', target_cols=None, start=None, end=None):
        self._start = '1990-01-01' if start is None else start
        self._end = datetime.datetime.now().strftime('%Y-%m-%d') if end is None else end
        self._data = df[(df[date] >= self._start) & (df[date] <= self._end)].sort_values(date, ascending=True)
        self.target_cols = list(OHLCV_AGG.keys()) if target_cols is None else target_cols
        self._length = len(self._data)
        self._data_cols = self._data.columns.tolist()
        if code in self._data_cols:  # date columns have been checked
            self._general_cols = [date, code]
        else:
            raise AttributeError(f'Column {code} or {date} not in data')
        # self._setup()

    # def date_list(self):
    #     return getattr(self, self._general_cols[0]).unique()
    @property
    def date_col(self):
        return self._general_cols[0]

    @property
    def code_col(self):
        return self._general_cols[1]

    @property
    def shape(self):
        return self._data.shape

    @property
    def length(self):
        return self.shape[0]

    # def __len__(self):
    #     return self._length

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

        return self._recreate_quote_func(self._data[indexer])

    # def __le__(self,key:str):

    def opr_filter(self, opr: str, split='&&', prefix='@'):
        if isinstance(opr, str) and opr.startswith(prefix):
            pass
        else:
            raise NotImplementedError('opr definition is wild! !')
        filters = list(map(lambda x: x.strip()[1:], opr.split(split)))
        cp = "(" + ") & (".join(filters) + ')'
        return self._data.query(cp)

    def filter(self, order, to_quote=True):
        code = [order._attr.code]
        dt = [order.create_date]
        data_indexer = (self._data[self.code_col].isin(code)) & (self._data[self.date_col].isin(dt))
        # data = self._data[(self._data[self.code_col].isin(code)) & (self._data[self.date_col].isin(dt))]
        if to_quote:
            return self._getitem_bool_array(data_indexer)
        else:
            return self._data[data_indexer]

    def __getitem__(self, dt):
        col = self.date_col
        if isinstance(dt, str):
            dt = [dt]
        elif isinstance(dt, list):
            pass
        else:
            raise ValueError('dt must be str or list')
        dt = pd.to_datetime(dt)
        indexer = self._data[col].isin(dt)
        return self._getitem_bool_array(indexer)

    def __getattr__(self, key):
        try:
            return self._obtain_data(key)
        except KeyError:
            raise AttributeError(f"Column '{key}' not in data")

    def __iter__(self):
        for dt, data in self._data.groupby(self.date_col):
            yield dt, data

    # def _setup(self):
    #     ## todo 性能点,会消耗过度资源
    #     for col in self._data_cols:
    #         setattr(self, col, self._obtain_data(col))


if __name__ == '__main__':
    from QuantNodes.test import GOOG
    import numpy as np

    # ['size', 'limit', 'stop', 'sl', 'tp']
    #
    # GOOG['Code'] = 'GOOG'
    #
    # np.random.seed(1)
    # price = pd.DataFrame(np.random.random(size=(GOOG.shape[0], 1)), columns=['GOOG'])
    # orders_df = (price > 0.75) * 1
    # orders_df['date'] = GOOG.index
    # scripts = orders_df.set_index('date').stack().reset_index()
    # scripts.columns = ['date', 'code', 'size']
    #
    # GOOG = GOOG.reset_index().rename(columns={'index': "date"})
    #
    # QD = QuoteData(GOOG)
    # res = QD.opr_filter("@date<='2007-01-01' && @Code=='GOOG' ", split='&&')
    # print(res)
    # pass
