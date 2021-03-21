# coding=utf-8
import warnings
from typing import Dict, List, Optional, Sequence, Union
from numbers import Number

import numpy as np
import pandas as pd
import uuid


def random_str(seed=1, **kwargs):
    # uln = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    # rs = ''.join(random.sample(uln, num))  # 生成一个 指定位数的随机字符串
    rs = ''
    a = uuid.uuid1()  # 根据 时间戳生成 uuid , 保证全球唯一
    b = rs + a.hex  # 生成将随机字符串 与 uuid拼接
    return b  # 返回随机字符串


def try_(lazy_func, default=None, exception=Exception):
    try:
        return lazy_func()
    except exception:
        return default


def _as_str(value) -> str:
    if isinstance(value, (Number, str)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return 'df'
    name = str(getattr(value, 'name', '') or '')
    if name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        return name[:1]
    if callable(value):
        name = getattr(value, '__name__', value.__class__.__name__).replace('<lambda>', 'λ')
    if len(name) > 10:
        name = name[:9] + '…'
    return name
