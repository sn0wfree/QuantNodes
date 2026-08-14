# coding=utf-8
"""主力合约标记 (c1-c5) + OTM 标记 — 从 期权工作平台 移植.

c1-c5 标记算法:
    对每个 (date, ym, cp) 分组:
    c1: |f-k| 最小的 2 个合约
    c2: |f-k| 最小的 4 个合约
    ...
    c5: |f-k| 最小的 10 个合约
    main_mark = c1 + c2 + c3 + c4 + c5 (范围 0~5)

OTM 标记:
    (f - k) * sign(cp) >= 0
    Call: f >= k → OTM (价外)
    Put:  f <= k → OTM (价外)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mark_main_contracts(
    df: pd.DataFrame,
    max_level: int = 5,
    f_col: str = "f",
    k_col: str = "k",
    dt_col: str = "dt",
    ym_col: str = "ym",
    cp_col: str = "cp",
) -> pd.DataFrame:
    """c1-c5 主力合约标记.

    Args:
        df: 期权数据, 需含 f/k/dt/ym/cp 列
        max_level: 最大标记层级

    Returns:
        原 DataFrame, 新增 c1~c{max_level} + main_mark 列
    """
    df = df.copy()
    for lvl in range(1, max_level + 1):
        df[f"c{lvl}"] = 0

    for (dt, ym, cp), group in df.groupby([dt_col, ym_col, cp_col]):
        if len(group) == 0:
            continue
        idx = group.index
        # f_k_diff 需要预先计算
        if "f_k_diff" not in group.columns:
            df.loc[idx, "f_k_diff"] = group[f_col] - group[k_col]
        diffs = df.loc[idx, "f_k_diff"].abs()
        sorted_idx = diffs.sort_values().index

        for lvl in range(1, max_level + 1):
            top_n = lvl * 2
            marked = sorted_idx[:min(top_n, len(sorted_idx))]
            df.loc[marked, f"c{lvl}"] = 1

    # main_mark = sum of all c-levels
    c_cols = [f"c{lvl}" for lvl in range(1, max_level + 1)]
    df["main_mark"] = df[c_cols].sum(axis=1).astype(int)

    return df


def mark_otm(
    df: pd.DataFrame,
    f_col: str = "f",
    k_col: str = "k",
    cp_col: str = "cp",
) -> pd.DataFrame:
    """OTM (Out-of-the-Money) 标记.

    Call: f >= k → OTM (价外)
    Put:  f <= k → OTM (价外)

    Args:
        df: 期权数据, 需含 f/k/cp 列

    Returns:
        原 DataFrame, 新增 OTM 列 (bool)
    """
    df = df.copy()
    cp_sign = df[cp_col].replace({"C": -1, "P": 1}).astype(int)
    df["OTM"] = ((df[f_col] - df[k_col]) * cp_sign) >= 0
    return df


def compute_f_k_diff(
    df: pd.DataFrame,
    f_col: str = "f",
    k_col: str = "k",
) -> pd.DataFrame:
    """计算 f-k 差值 (用于排序和 OTM 判断).

    Args:
        df: 期权数据

    Returns:
        原 DataFrame, 新增 f_k_diff 列
    """
    df = df.copy()
    df["f_k_diff"] = df[f_col] - df[k_col]
    return df


def filter_by_main_mark(
    df: pd.DataFrame,
    min_mark: int = 1,
    main_mark_col: str = "main_mark",
) -> pd.DataFrame:
    """按 main_mark 过滤流动性最好的合约.

    Args:
        df: 已标记的期权数据
        min_mark: 最小 main_mark (1=最宽松, 5=最严格 ATM)

    Returns:
        过滤后的 DataFrame
    """
    return df[df[main_mark_col] >= min_mark].copy()
