# coding: utf-8
"""日期工具 / Date Utilities

Migrated from ~/Public/单因子回测/date_utils.py
"""

import pandas as pd
import numpy as np
from datetime import datetime


def valid_date(trade_dts) -> bool:
    """验证日期格式, 必须为 yyyymmdd 数值型"""
    if isinstance(trade_dts, pd.DataFrame):
        num_lens = len(str(trade_dts.iloc[0, 0]))
        data_type = trade_dts.dtypes[0]
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
    return trade_dt.applymap(lambda x: datetime.strptime(str(int(x)), '%Y%m%d'))


def datetime_to_datenum(trade_dt: pd.DataFrame) -> pd.DataFrame:
    """将 datetime 转换为 yyyymmdd int"""
    trade_dt = trade_dt.copy()
    return trade_dt.applymap(lambda x: int(datetime.strftime(x, '%Y%m%d')))


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
    """将交易日重采样为周/月/季"""
    trade_date = trade_dt.copy()
    trade_date.columns = ['trade_dt']
    if not valid_date(trade_date):
        raise ValueError("日期格式有误, 请输入 yyyymmdd int 型")

    trade_date = datenum_to_datetime(trade_date)

    if not isinstance(rule, tuple) or len(rule) != 2:
        raise ValueError("请输入正确的调仓模式 ('M', 'end')")

    mode, position = rule

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
    """日期偏移"""
    if isinstance(trade_dt_all, pd.DataFrame):
        trade_dt_all = trade_dt_all.iloc[:, 0]

    if mode == 'D':
        adj_date = trade_dt_all
        date_last = list(map(
            lambda x: trade_dt_all.values[trade_dt_all.values <= x][-1],
            list(date_input)
        ))
        date_last_idx = pd.Index(trade_dt_all).get_indexer(date_last)
    elif mode in ('W', 'M', 'Q'):
        adj_date = resample_trade_date(trade_dt_all.to_frame(), (mode, 'end'))
        date_last = list(map(
            lambda x: adj_date.values[adj_date.values <= x][-1],
            list(date_input)
        ))
        date_last_idx = pd.Index(adj_date.iloc[:, 0]).get_indexer(date_last)
    else:
        raise ValueError(f"不支持的偏移模式: {mode}")

    try:
        return np.array(adj_date.iloc[date_last_idx + n])
    except IndexError:
        if if_modify:
            new_idx = np.clip(date_last_idx + n, 0, len(trade_dt_all) - 1)
            return np.array(trade_dt_all.iloc[new_idx])
        raise
