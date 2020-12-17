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


class QuoteData(object):
    def __init__(self, df: (pd.DataFrame,), code='Code', date='date'):
        self._data = df
        self._length = len(df)
        self._data_cols = df.columns.tolist()
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

    GOOG = GOOG.reset_index().rename(columns={'index':'date'})
    GOOG['Code'] = 'GOOG'

    QD = QuoteData(GOOG)
    pass
