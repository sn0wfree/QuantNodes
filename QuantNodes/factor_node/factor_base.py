# coding=utf-8
"""
因子基类模块 - 已简化

原有的 Factor 类已合并到 factor.py 中，此文件仅保留 _default_operator 函数
供 factor_operation.py 使用。

注意：新代码应直接从 factor.py 导入 Factor 类。
"""

from typing import Any

import numpy as np


def _default_operator(f, idt, iid, x, args):
    """默认算子，返回 NaN"""
    return np.nan
