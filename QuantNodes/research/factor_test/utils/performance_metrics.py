# coding: utf-8
"""评价指标工具 / Performance Metrics

Migrated from ~/Public/单因子回测/factor_performance.py
"""

import pandas as pd
import numpy as np
from .constants import ANNUAL_DAYS


def calc_max_drawdown(net_day: pd.Series) -> dict:
    """计算最大回撤 (复利净值曲线)

    Returns:
        dict: MDD, MDD_date, Lastingtime, Endingtime, DD (每日回撤序列)
    """
    DD = 1 - net_day / net_day.cummax()
    DD = DD.dropna()
    if DD.empty:
        return {'MDD': 0, 'MDD_date': None, 'Lastingtime': 0, 'Endingtime': 0, 'DD': DD}

    MDD = DD.max()
    maxdraw_idx = DD.idxmax()

    # 最大回撤开始时间: 回撤为 0 的最后一天
    try:
        index_bg = DD.loc[:maxdraw_idx][DD.loc[:maxdraw_idx] == 0].index[-1]
    except (IndexError, KeyError):
        index_bg = DD.index[0]

    lasting_time = (pd.Index(DD.index).get_indexer([maxdraw_idx])[0]
                    - pd.Index(DD.index).get_indexer([index_bg])[0] + 1)

    # 最大回撤结束时间: 回撤恢复为 0 的第一天
    try:
        index_end = DD.loc[maxdraw_idx:][DD.loc[maxdraw_idx:] == 0].idxmax()
    except (ValueError, KeyError):
        index_end = DD.index[-1]

    ending_time = (pd.Index(DD.index).get_indexer([index_end])[0]
                   - pd.Index(DD.index).get_indexer([maxdraw_idx])[0] + 1)

    return {
        'MDD': MDD,
        'MDD_date': maxdraw_idx,
        'Lastingtime': lasting_time,
        'Endingtime': ending_time,
        'DD': DD,
    }


def evaluation(account_net: pd.Series, adj_dates: list, annual_days: int = ANNUAL_DAYS) -> pd.DataFrame:
    """输入净值曲线返回评价结果 (全期 + 分年)

    Args:
        account_net: 单利净值曲线
        adj_dates: 调仓日列表
        annual_days: M11 年化天数 (默认全局 ANNUAL_DAYS=250, 美股 252, 24h 365)

    Returns:
        DataFrame: Year, AnnualRt, AccumRt, SR, MDD, WinRatio, WinLossRatio, Calmar, ...
    """
    result = []
    account_net = account_net.copy()
    account_net.name = 'net'

    if isinstance(account_net, pd.DataFrame):
        account_net = account_net.iloc[:, 0]

    # 日收益率
    net = account_net.to_frame()
    net['ret_daily'] = 0.0
    if adj_dates[0] != net.index[0]:
        all_dates = [net.index[0]] + adj_dates
    else:
        all_dates = adj_dates

    for i in range(len(all_dates)):
        current_date = all_dates[i]
        next_date = all_dates[i + 1] if i < len(all_dates) - 1 else all_dates[-1]
        current_net = net.loc[current_date:next_date, 'net']
        current_net = current_net - current_net.iloc[0] + 1
        net.loc[current_net.index[1:], 'ret_daily'] = current_net.pct_change()[1:]

    daily_ret = net['ret_daily']

    # 每期收益
    every_return = account_net.loc[adj_dates].to_frame().diff(1)
    adj_cycle = len(account_net.loc[adj_dates[0]:adj_dates[-1]]) / (len(adj_dates) - 1)

    # 全期指标
    accum_rt = account_net.iloc[-1] / account_net.iloc[0] - 1
    annual_rt = every_return.mean().iloc[0] / adj_cycle * annual_days
    annu_std = daily_ret.std(ddof=1) * np.sqrt(annual_days)
    SR = np.nan if annu_std == 0 else annual_rt / annu_std
    mdd = calc_max_drawdown(account_net)
    winRatio = (every_return.dropna() > 0).mean().iloc[0]
    winlossRatio = (every_return[every_return > 0].mean().iloc[0]
                    / every_return[every_return < 0].mean().iloc[0] * -1)
    calmar = np.nan if mdd['MDD'] == 0 else annual_rt / mdd['MDD']
    trade_times = every_return.notna().sum().iloc[0]

    result.append([
        'all', annual_rt, accum_rt, SR, mdd['MDD'], winRatio, winlossRatio,
        calmar, mdd['MDD_date'], mdd['Lastingtime'], mdd['Endingtime'], trade_times
    ])

    # 分年
    account_net_df = account_net.to_frame()
    account_net_df['trade_dt'] = account_net_df.index.values
    account_net_df['year'] = pd.to_datetime(
        account_net_df['trade_dt'].astype(str), format='%Y%m%d'
    ).dt.year

    every_return_cp = every_return.copy()
    every_return_cp['year'] = list(map(
        lambda x: pd.to_datetime(str(x), format='%Y%m%d').year,
        every_return.index.tolist()
    ))

    for year_i in account_net_df['year'].unique():
        account_net_i = account_net_df[account_net_df['year'] == year_i]['net']
        every_return_i = every_return[every_return_cp['year'] == year_i]

        accum_rt_i = every_return_i.sum().iloc[0]
        annual_rt_i = every_return_i.mean().iloc[0] / adj_cycle * annual_days
        annu_std_i = daily_ret[account_net_df['year'] == year_i].std(ddof=1) * np.sqrt(annual_days)
        SR_i = np.nan if annu_std_i == 0 else annual_rt_i / annu_std_i

        year_dates = every_return_cp.index[every_return_cp['year'] == year_i].tolist()
        mdd_i = calc_max_drawdown(account_net_i) if len(account_net_i) > 1 else {'MDD': 0}

        winRatio_i = (every_return_i.dropna() > 0).mean().iloc[0]
        wl_i = every_return_i[every_return_i > 0].mean().iloc[0]
        wl_j = every_return_i[every_return_i < 0].mean().iloc[0]
        winlossRatio_i = wl_i / wl_j * -1 if pd.notna(wl_j) and wl_j != 0 else np.nan
        calmar_i = np.nan if mdd_i['MDD'] == 0 else annual_rt_i / mdd_i['MDD']
        trade_times_i = every_return_i.notna().sum().iloc[0]

        result.append([
            year_i, annual_rt_i, accum_rt_i, SR_i, mdd_i['MDD'], winRatio_i,
            winlossRatio_i, calmar_i, mdd_i.get('MDD_date'), mdd_i.get('Lastingtime', 0),
            mdd_i.get('Endingtime', 0), trade_times_i
        ])

    columns = [
        'Year', 'AnnualRt', 'AccumRt', 'SR', 'MDD', 'WinRatio', 'WinLossRatio',
        'Calmar', 'MDD_date', 'MDD_lastdays', 'MDD_recoverdays', 'Periods'
    ]
    return pd.DataFrame(result, columns=columns)


def cal_net_simple(net: pd.Series, adj_dates: list) -> pd.Series:
    """将复利净值转换为单利净值"""
    data_net = net.to_frame()
    data_net.columns = ['account']
    data_net['simp'] = np.nan

    # 第一个调仓日之前
    data_net.loc[:adj_dates[1], 'simp'] = data_net.loc[:adj_dates[1], 'account']
    benchmark_i = data_net.loc[adj_dates[1], 'account']

    for i in range(1, len(adj_dates) - 1):
        adj_i = adj_dates[i]
        net_i = net.loc[adj_i:adj_dates[i + 1]]
        ret_i = net_i / net_i.iloc[0] - 1
        net_i_update = ret_i + benchmark_i
        data_net.loc[adj_i:adj_dates[i + 1], 'simp'] = net_i_update.values
        benchmark_i = data_net.loc[adj_dates[i + 1], 'simp']

    result = data_net['simp'].to_frame()
    result.columns = ['net']
    return result['net']
