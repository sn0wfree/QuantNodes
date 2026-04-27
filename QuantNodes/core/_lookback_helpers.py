# -*- coding: utf-8 -*-
"""LookBack 窗口计算的共用辅助函数"""
import numpy as np


def compute_lookback_params(lookback, lookback_mode):
    """计算 LookBack 窗口参数
    
    Args:
        lookback: list of lookback values for descriptors
        lookback_mode: list of lookback modes for descriptors
    
    Returns:
        tuple: (StartIndAndLen, MaxLookBack, MaxLen)
            - StartIndAndLen: list of (start_ind, length) tuples
            - MaxLookBack: maximum lookback value
            - MaxLen: maximum window length
    """
    StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1
    for i, iLookBack in enumerate(lookback):
        if lookback_mode[i] == "滚动窗口":
            StartIndAndLen.append((iLookBack, iLookBack + 1))
            MaxLen = max(MaxLen, iLookBack + 1)
        else:
            StartIndAndLen.append((iLookBack, np.inf))
            MaxLen = np.inf
        MaxLookBack = max(MaxLookBack, iLookBack)
    return StartIndAndLen, MaxLookBack, MaxLen


def extend_dt_ruler(dt_ruler, dts, max_lookback):
    """扩展时间标尺向前回溯
    
    如果 start_ind >= max_lookback，直接返回 dt_ruler[start_ind-max_lookback:]
    否则在前面补 None 凑够 max_lookback 个。
    
    Args:
        dt_ruler: 时间标尺
        dts: 当前时间序列
        max_lookback: 最大回溯值
    
    Returns:
        list: 扩展后的时间标尺
    """
    start_ind = dt_ruler.index(dts[0]) if dts[0] in dt_ruler else 0
    if start_ind >= max_lookback:
        return list(dt_ruler[start_ind - max_lookback:])
    else:
        return [None] * (max_lookback - start_ind) + list(dt_ruler)