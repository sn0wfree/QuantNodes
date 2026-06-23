# coding: utf-8
"""日期工具 / Date Utilities

Migrated from ~/Public/单因子回测/date_utils.py
"""

import pandas as pd
import numpy as np
from datetime import datetime


def valid_date(trade_dts) -> bool:
    """验证日期格式, 必须为 yyyymmdd 数值型

    v3.0.0 fix: use ``.iloc[0]`` instead of ``[0]`` for indexing the
    ``dtypes`` Series. The previous code assumed integer column labels
    (e.g. ``pd.DataFrame(dates, columns=[0])``), but after
    ``resample_trade_date`` renames columns to strings the index
    becomes string-keyed, and ``dtypes[0]`` raises ``KeyError: 0``.
    """
    if isinstance(trade_dts, pd.DataFrame):
        num_lens = len(str(trade_dts.iloc[0, 0]))
        data_type = trade_dts.dtypes.iloc[0]
    elif isinstance(trade_dts, pd.Series):
        num_lens = len(str(trade_dts.iloc[0]))
        data_type = trade_dts.dtypes
    else:
        return False
    return (num_lens == 8) and (data_type == 'int64')


def datenum_to_datetime(trade_dt: pd.DataFrame) -> pd.DataFrame:
    """将 yyyymmdd int 转换为 datetime"""
    trade_dt = trade_dt.copy()
    if isinstance(trade_dt, pd.Series):
        trade_dt = trade_dt.to_frame()
    # v3.0.0 fix: ``DataFrame.applymap`` was removed in pandas 3.0; use ``.map``.
    return trade_dt.map(lambda x: datetime.strptime(str(int(x)), '%Y%m%d'))


def datetime_to_datenum(trade_dt: pd.DataFrame) -> pd.DataFrame:
    """将 datetime 转换为 yyyymmdd int"""
    trade_dt = trade_dt.copy()
    # v3.0.0 fix: ``DataFrame.applymap`` was removed in pandas 3.0; use ``.map``.
    return trade_dt.map(lambda x: int(datetime.strftime(x, '%Y%m%d')))


def chg_idx_to_datestr(data):
    """将 Series/DataFrame 的 int index 转换为 'yyyy/mm/dd' 字符串"""
    data = data.copy()
    date_idx = data.index.tolist()
    date_idx_str = list(map(
        lambda x: str(x)[:4] + '/' + str(x)[4:6] + '/' + str(x)[6:], date_idx
    ))
    data.index = date_idx_str
    return data


def resample_trade_date(trade_dt: pd.DataFrame, rule=('M', 'end')) -> pd.DataFrame:
    """将交易日重采样为周/月/季

    Args:
        trade_dt: yyyymmdd int DataFrame, 单列.
        rule: ``(mode, position)`` 两元组.

            - ``mode``: ``'W'`` (周) / ``'M'`` (月) / ``'Q'`` (季)
            - ``position``: ``'end'`` 或 ``'begin'``

              L1 (2026-06-21) 新增 alias::

                  'beg' / 'start' / 'first' → 'begin'
                  'last'                    → 'end'

              用户不再因为写 ``'beg'`` 这类自然简写而踩 ``不支持的 position``
              错误. 未在 alias 表内的值仍抛 ``ValueError``.

    Returns:
        DataFrame: 重采样后的 yyyymmdd int 日期序列.

    Raises:
        ValueError: 日期格式非法 / rule 不是 2 元 tuple / mode 不在 W/M/Q /
            position 既不在 alias 也不是 begin/end.
    """
    trade_date = trade_dt.copy()
    trade_date.columns = ['trade_dt']
    if not valid_date(trade_date):
        raise ValueError("日期格式有误, 请输入 yyyymmdd int 型")

    trade_date = datenum_to_datetime(trade_date)

    if not isinstance(rule, tuple) or len(rule) != 2:
        raise ValueError("请输入正确的调仓模式 ('M', 'end')")

    mode, position = rule

    # L1 (2026-06-21): 友好 alias 表, 用户写 'beg' 等同 'begin'.
    _POSITION_ALIASES = {
        'beg': 'begin',
        'start': 'begin',
        'first': 'begin',
        'last': 'end',
    }
    position = _POSITION_ALIASES.get(position, position)

    if mode in ('W', 'M', 'Q'):
        period_func = {
            'W': lambda x: x.weekday(),
            'M': lambda x: x.month,
            'Q': lambda x: x.quarter,
        }[mode]
        trade_date['period'] = trade_date.iloc[:, 0].apply(period_func)

        if position == 'end':
            trade_date['diff'] = trade_date['period'].diff(-1)
            trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
        elif position == 'begin':
            trade_date['diff'] = trade_date['period'].diff(1)
            trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
        else:
            raise ValueError(f"不支持的 position: {position}")
    else:
        raise ValueError(f"不支持的 resample mode: {mode}")

    return datetime_to_datenum(trade_date)


def get_adjust_date(trade_dt: pd.DataFrame, beg_date: int, end_date: int,
                    adj_mode=('M', 'end')) -> pd.DataFrame:
    """根据起始日、截止日、调仓模式确定调仓日"""
    trade_dt = trade_dt.copy()

    if type(beg_date) is not type(end_date) or not isinstance(beg_date, int):
        raise ValueError("起始日与截止日格式不一致或非 int 型")

    if not isinstance(adj_mode, tuple) or len(adj_mode) != 2:
        raise ValueError("请输入正确的调仓模式 ('M', 'end')")

    if not valid_date(trade_dt):
        raise ValueError("trade_dt 格式有误")

    if isinstance(trade_dt, pd.Series):
        trade_dt = trade_dt.to_frame()

    # 调整起止时间到实际交易日
    try:
        beg_date_new = trade_dt[trade_dt >= beg_date].dropna().iloc[0, 0]
        end_date_new = trade_dt[trade_dt <= end_date].dropna().iloc[-1, 0]
    except Exception:
        raise ValueError(
            f"获取测试起止日期出错, 现有数据: {trade_dt.iloc[0, 0]} ~ {trade_dt.iloc[-1, 0]}"
        )

    mode = adj_mode[0]
    if mode in ('M', 'W'):
        adj_date = resample_trade_date(trade_dt, adj_mode)
        adj_date = adj_date[(adj_date >= beg_date) & (adj_date <= end_date)].dropna()
        adj_date = adj_date.astype(trade_dt.iloc[0].values)
    elif mode == 'D':
        beg_idx = np.where(trade_dt.iloc[:, 0] == beg_date_new)[0][0]
        end_idx = np.where(trade_dt.iloc[:, 0] == end_date_new)[0][0]
        adj_date = trade_dt.iloc[beg_idx:end_idx + 1:adj_mode[1]]
    elif mode == 'custom':
        adj_date = adj_mode[1]
        if not valid_date(adj_date):
            raise ValueError("自定义调仓日格式有误")
    else:
        raise ValueError(f"不支持的调仓模式: {mode}")

    return adj_date


def offset_date(date_input, trade_dt_all, n, mode='D', if_modify=False):
    """日期偏移

    H6: replaced ``arr[arr <= x][-1]`` (O(M) boolean mask per query) with
    ``arr[idx-1]`` where ``idx = np.searchsorted(arr, x, side='right')``
    (O(log M) per query). Cumulative gain is K * (M -> log M) where K is
    the size of date_input and M is the size of trade_dt_all.

    L2 (2026-06-21): 显式越界检查.

        修复: 原 ``adj_date.iloc[adj_date_idx + n]`` 在负索引时**不会** raise
        IndexError — pandas/numpy 负索引会 wrap-around 到末尾, 导致 n=-1
        (回退 1 天) 错误返回最后一个调仓日. 现改为显式计算 final_idx, 越界
        元素按 ``if_modify`` 决定:

        - ``if_modify=False`` (默认): 显式 raise ``IndexError``, 错误信息
          列出越界 idx.
        - ``if_modify=True``: ``np.clip`` 到 [0, len(adj_date)-1] 边界.

        行为变化: 之前 silent 错误 (返回末尾日期) → 现在明确报错. 任何依赖
        wrap-around 的上游代码 (经 grep 确认无) 会受影响.
    """
    if isinstance(trade_dt_all, pd.DataFrame):
        trade_dt_all = trade_dt_all.iloc[:, 0]

    if mode == 'D':
        adj_date = trade_dt_all
        adj_values = np.asarray(trade_dt_all.values).ravel()
    elif mode in ('W', 'M', 'Q'):
        adj_date = resample_trade_date(trade_dt_all.to_frame(), (mode, 'end'))
        # resample_trade_date returns a DataFrame; flatten to 1D for searchsorted.
        adj_values = np.asarray(adj_date.iloc[:, 0].values).ravel()
    else:
        raise ValueError(f"不支持的偏移模式: {mode}")

    date_input_arr = np.asarray(list(date_input))
    # For each x, find largest idx where adj_values[idx] <= x (searchsorted).
    insert_idx = np.searchsorted(adj_values, date_input_arr, side='right')
    date_last_idx = np.clip(insert_idx - 1, 0, len(adj_values) - 1)
    date_last = adj_values[date_last_idx]
    # Use the same 1D representation for the lookup index.
    if isinstance(adj_date, pd.DataFrame):
        adj_index = pd.Index(adj_date.iloc[:, 0])
    else:
        adj_index = pd.Index(adj_date)
    adj_date_idx = adj_index.get_indexer(date_last)

    # L2 (2026-06-21): 显式越界检查 (修复 pandas 负索引 wrap-around silent bug).
    final_idx = adj_date_idx + n
    out_of_bounds = (final_idx < 0) | (final_idx >= len(adj_date))
    if out_of_bounds.any():
        if if_modify:
            final_idx = np.clip(final_idx, 0, len(adj_date) - 1)
        else:
            raise IndexError(
                f"offset_date 越界 (n={n}): "
                f"final_idx={final_idx[out_of_bounds].tolist()} "
                f"不在 [0, {len(adj_date) - 1}] 范围. "
                f"设置 if_modify=True 可裁剪到边界."
            )
    return np.array(adj_date.iloc[final_idx])
