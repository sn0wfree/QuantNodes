# coding=utf-8
import pandas as pd
import numpy as np


class IC(object):
    pass

def cal_mdd(net_day: pd.Series, adj_dates: list, if_cmp=0):
    """
    计算复利的最大回撤
    :param net_day:
    :param adj_dates:
    :param if_cmp:
    :return:
    """
    mdd = {}
    if if_cmp:
        ret_daily = net_day.pct_change()
        ret_daily.iloc[0] = 0
    else:
        ret_daily = simpnet2ret(net_day, adj_dates)

    net_day = (ret_daily + 1).cumprod()  # 复利的净值曲线
    DD = 1 - net_day / net_day.cummax()  # 每日回撤
    MDD = max(DD.dropna())
    maxdrawIndex = DD.dropna().idxmax()  # 最大回撤时间
    try:
        index_bg = DD.loc[:maxdrawIndex].index[np.where(DD.loc[:maxdrawIndex] == 0)[0][-1]]  # 最大回撤开始时间

    except Exception:
        index_bg = DD
        # withdrawLastingtime = np.nan
    withdrawLastingtime = pd.Index(DD.index).get_indexer([maxdrawIndex])[0] - \
                          pd.Index(DD.index).get_indexer([index_bg])[0] + 1  # 最大回撤持续时间

    try:
        index_end = (DD.loc[maxdrawIndex:] == 0).idxmax()  # 最大回撤结束时间
    except Exception:
        index_end = DD.index[-1]
    withdrawEndingtime = pd.Index(DD.index).get_indexer([index_end])[0] - \
                         pd.Index(DD.index).get_indexer([maxdrawIndex])[0] + 1  # 最大回撤回复时间

    mdd['DD'] = DD  # 每日回撤
    mdd['MDD'] = MDD  # 最大回撤
    mdd['MDD_date'] = maxdrawIndex  # 最大回撤时间
    # mdd['MDD_date_beg'] = index_bg
    mdd['Lastingtime'] = withdrawLastingtime  # 最大回撤持续时间
    mdd['Endingtime'] = withdrawEndingtime  # 最大回撤回复时间

    return mdd


def cal_ic(factor_data: pd.DataFrame, group=np.inf) -> tuple:
    """
    计算因子ic，rank ic 以及评价结果
    :param factor_data:
    :param group: 计算ic最少需要多少个因子值, 如果因子数少于组数，ic为nan
    :return:
    """
    print('计算IC...')
    ic = pd.Series(np.nan * np.ones(len(factor_data)), index=factor_data.index.to_list())
    rank_ic = pd.Series(np.nan * np.ones(len(factor_data)), index=factor_data.index.to_list())
    factor_rank_autocorr = ic.copy()  # 因子rank的自相关性（滚动一期)

    # 导入收盘价,计算各期股票收益
    cp = Factor.add_index(Factor.get_apidata(('stk_daily.h5', 'cp')))
    stock_cycle_ret = cp.loc[factor_data.index.to_list(), :].pct_change().shift(-1)  # 下一期的股票收益
    # stock_cycle_ret[factor_data.isnull()]=np.nan
    fator_rank = factor_data.rank(axis=1)  # 当期因子排序
    fator_rank_next = fator_rank.shift(-1)  # 下一期

    for t_i in factor_data.index.to_list():
        nonan = factor_data.loc[t_i, :].notnull().sum()
        if (nonan == 0) | (nonan < group):
            continue
        else:
            # 当期因子值与下期收益的相关系数
            ic.loc[t_i] = factor_data.loc[t_i, :].corr(stock_cycle_ret.loc[t_i, :])
            rank_ic.loc[t_i] = factor_data.loc[t_i, :].corr(stock_cycle_ret.loc[t_i, :], 'spearman')
            factor_rank_autocorr.loc[t_i] = fator_rank.loc[t_i, :].corr(fator_rank_next.loc[t_i, :], 'spearman')

    # 评价
    ic_result = pd.Series([ic.mean(), ic.std(ddof=1), ic.mean() / ic.std(ddof=1),
                           ic.mean() / ic.std(ddof=1) * np.sqrt(ic.notnull().sum() - 1),
                           ((ic > 0) * 1).sum() / ic.count(), ((ic < 0) * 1).sum() / ic.count()],
                          index=['IC均值', 'IC标准差', 'ICIR',
                                 'IC_T值', 'IC为正比例', 'IC为负比例'])
    rank_ic_result = pd.Series([rank_ic.mean(), rank_ic.std(ddof=1), rank_ic.mean() / rank_ic.std(ddof=1),
                                rank_ic.mean() / rank_ic.std(ddof=1) * np.sqrt(rank_ic.notnull().sum() - 1),
                                ((rank_ic > 0) * 1).sum() / ic.count(), ((rank_ic < 0) * 1).sum() / ic.count()],
                               index=['rankIC均值', 'rankIC标准差', 'rankICIR',
                                      'rankIC_T值', 'rankIC为正比例', 'rankIC为负比例'])

    ic_inf = {}
    ic_inf['ic'] = ic
    ic_inf['rank_ic'] = rank_ic
    ic_inf['ic_result'] = ic_result
    ic_inf['rank_ic_result'] = rank_ic_result
    ic_inf['factor_rank_autocorr'] = factor_rank_autocorr

    return ic_inf


def cal_fac_ret(factor_data: pd.DataFrame, group=np.inf):
    print('计算因子收益...')
    # 导入收盘价,计算各期股票收益
    cp = Factor.add_index(Factor.get_apidata(('stk_daily.h5', 'cp')))
    stock_cycle_ret = cp.loc[factor_data.index.to_list(), :].pct_change().shift(-1)  # 下一期的股票收益
    fac_return = pd.DataFrame(np.nan * np.ones((len(factor_data), 3)), index=factor_data.index.to_list(),
                              columns=['fac_ret', 'T_value', 'R2'])

    for t_i in factor_data.index.to_list()[:-1]:
        # print(t_i)
        nonan = factor_data.loc[t_i, :].notnull().sum()
        if (nonan == 0) | (nonan < group):
            continue
        else:
            x = factor_data.loc[t_i, :];
            x.name = 'fact_data'
            X = pd.concat([pd.Series(np.ones(len(x)), index=x.index.to_list()).to_frame(),
                           x], axis=1)
            X.dropna(inplace=True)
            # X = sm.add_constant(x)
            y = stock_cycle_ret.loc[t_i, :];
            y.name = 'ret'
            lm_data = pd.merge(y.to_frame(), X, left_index=True, right_index=True)
            lm_data.dropna(inplace=True)
            model = sm.OLS(lm_data.iloc[:, 0], lm_data.iloc[:, 1:])
            fit = model.fit()
            fac_return.loc[t_i, 'fac_ret'] = fit.params.iloc[1]  # 回归系数
            fac_return.loc[t_i, 'T_value'] = fit.tvalues.iloc[1]  # T值
            fac_return.loc[t_i, 'R2'] = fit.rsquared_adj  # 回归调整R方
    fac_return_result = pd.Series([fac_return.loc[:, 'fac_ret'].mean(), fac_return.loc[:, 'fac_ret'].std(ddof=1),
                                   fac_return.loc[:, 'fac_ret'].mean() / fac_return.loc[:, 'fac_ret'].std(ddof=1),
                                   fac_return.loc[:, 'fac_ret'].mean() / fac_return.loc[:, 'fac_ret'].std(ddof=1) *
                                   np.sqrt(fac_return.loc[:, 'fac_ret'].notnull().sum() - 1),
                                   ((fac_return.loc[:, 'T_value'] > 2) * 1).sum() / fac_return.loc[:,
                                                                                    'T_value'].count(),
                                   ((fac_return.loc[:, 'T_value'] < -2) * 1).sum() / fac_return.loc[:,
                                                                                     'T_value'].count()],
                                  index=[['因子收益的均值', '因子收益的标准差', 'IR', 'T值', '因子收益显著为正的比例',
                                          '因子收益显著为负的比例']])
    fac_return_inf = {}
    fac_return_inf['fac_return_result'] = fac_return_result
    fac_return_inf['fac_return_value'] = fac_return
    return fac_return_inf


def cal_event_evaluation(ret_list):
    # 根据分组情况计算每组收益
    # 计算样本量、收益均值、胜率、盈亏比
    out = list([ret_list.count(), ret_list.mean(), ((ret_list > 0) * 1).sum() / ret_list.count(),
                ret_list[ret_list > 0].mean() / ret_list[ret_list < 0].mean() * -1])
    return out


# def evaluation(account_net: pd.Series, adj_dates: list, if_cmp=0):
#     """
#     输入净值曲线返回评价结果（单利）
#     :param account_net:
#     :param adj_dates:
#     :param if_cmp: 是否是复利，默认为复利，需要转为单利计算
#     :return:
#     """
#     N = 250
#     result = []
#     account_net = account_net.copy()
#     account_net.name = 'net'
#     if if_cmp == 1:
#         account_net= cal_net_simple(account_net, adj_dates)  # 如果是复利转为单利
#         # daily_ret = account_net['net'].pct_change().dropna() # 日收益率通过相除
#
#     if isinstance(account_net, pd.Series):
#         account_net = account_net.to_frame()
#
#     daily_ret = simpnet2ret(account_net.iloc[:,0], adj_dates) #通过单利净值曲线计算的每日收益
#     every_return = account_net.loc[adj_dates,:].diff(1).shift(-1) #每一期的收益，复利为相除，单利为相减
#     adj_cycle = len(account_net.loc[adj_dates[0]:adj_dates[-1],:] - 1) / (len(adjust_dates)-1) #平均每期持仓时间
#     # 累计收益、年化收益，按照252天年化
#     accum_rt = account_net['net'].iloc[-1]/account_net['net'].iloc[0] - 1
#     # annual_rt = (account_net['net'].iloc[-1]/account_net['net'].iloc[0]) ** (N / account_net.shape[0]) - 1
#     annual_rt = every_return.mean() / adj_cycle * N #年化收益通过每个持仓期的收益均值计算
#     # 年化波动率
#     annu_std = daily_ret.std(ddof=1) * np.sqrt(N)
#     #    Sharpe比率
#     SR = annual_rt / annu_std
#     # 最大回撤
#     # DD = 1 - account_net['net'] / account_net['net'].cummax()
#     # MDD = max(DD.dropna())
#     mdd = cal_mdd(account_net['net'], adj_dates, if_cmp = 0)
#
#     # 胜率、盈亏比、收益风险比
#     #根据调仓日来提取交易流水
#     transactions = account_net.loc[adj_dates + [account_net.index[-1]], :]
#     transactions['ret'] = transactions['net'].pct_change(1)
#
#     winRatio = ((transactions['ret'].dropna()) > 0).mean()
#     winlossRatio = (transactions['ret'][transactions['ret'] > 0].mean() /
#                    transactions['ret'][transactions['ret'] < 0].mean()) * -1
#     calmar = annual_rt / MDD
#     # 分年度评价
#     account_net['trade_dt'] = account_net.index.values
#     account_net['trade_dt'] = account_net['trade_dt'].apply(lambda x: pd.datetime.strptime(str(x), '%Y%m%d'))
#     account_net['year'] = account_net['trade_dt'].apply(lambda x: x.year)
#     year_all = account_net['year'].unique()
#     result.append(['all', SR, accum_rt, annual_rt, MDD, winRatio, winlossRatio, calmar])
#     for year_i in year_all:
#         # print(year_i)
#         # 取出当年净值并且归一化
#         account_net_i = account_net[account_net['year'] == year_i]
#         account_net_i.loc[:,'net'] = account_net_i.loc[:,'net'].values / account_net_i['net'].iloc[0]
#         # 指标计算
#         accum_rt_i = account_net_i['net'].iloc[-1] - 1
#         annual_rt_i = account_net_i['net'].iloc[-1] ** (N / account_net_i.shape[0]) - 1
#         try:
#             SR_i = account_net_i['net'].dropna().pct_change().mean() / (
#                 account_net_i['net'].dropna().pct_change().std()) * np.sqrt(N)
#         except:
#             # 当年全部空仓
#             SR_i = np.nan
#
#         DD_i = 1 - account_net_i['net'] / account_net_i['net'].cummax()
#         MDD_i = max(DD_i.dropna())
#         # 交易流水,按照开仓日期来
#         transactions_i = transactions[np.in1d(list(map(lambda x: int(str(x)[:4]),list(transactions.index))),year_i)]
#         winRatio_i = ((transactions_i['ret'].dropna()) > 0).mean()
#         winlossRatio_i = -(transactions_i['ret'][transactions_i['ret'] > 0].mean() /
#                          transactions_i['ret'][transactions_i['ret'] < 0].mean())
#         calmar_i = annual_rt_i / MDD_i
#
#         result.append([year_i, SR_i, accum_rt_i, annual_rt_i, MDD_i, winRatio_i, winlossRatio_i, calmar_i])
#     result = pd.DataFrame(result, columns=['Year', 'SR', 'AccumRt', 'AnnualRt',
#                                            'MDD', 'WinRatio', 'WinLossRatio', 'Calmar'])
#     return result

def evaluation(account_net: pd.Series, adj_dates: list, if_cmp=0):
    """
    输入净值曲线返回评价结果（单利）
    :param account_net:
    :param adj_dates: adjust_dates
    :param if_cmp: 是否是复利，默认为复利，需要转为单利计算
    :return:
    """
    N = 250
    result = []
    account_net = account_net.copy()
    account_net.name = 'net'
    if if_cmp == 1:
        account_net = cal_net_simple(account_net, adj_dates)  # 如果是复利转为单利

    if isinstance(account_net, pd.Series):
        account_net = account_net.to_frame()

    daily_ret = simpnet2ret(account_net.iloc[:, 0], adj_dates)  # 通过单利净值曲线计算的每日收益
    every_return = account_net.loc[adj_dates, :].diff(1)  # .shift(-1)# #每一期的收益，复利为相除，单利为相减
    adj_cycle = len(account_net.loc[adj_dates[0]:adj_dates[-1], :] - 1) / (len(adj_dates) - 1)  # 平均每期持仓时间
    # 累计收益、年化收益，按照252天年化
    accum_rt = account_net['net'].iloc[-1] / account_net['net'].iloc[0] - 1
    annual_rt = every_return.mean()[0] / adj_cycle * N  # 年化收益通过每个持仓期的收益均值计算
    # 年化波动率
    annu_std = daily_ret.std(ddof=1) * np.sqrt(N)
    #    Sharpe比率
    SR = annual_rt / annu_std
    # 最大回撤
    mdd = cal_mdd(account_net['net'], adj_dates, if_cmp=0)
    # 胜率、盈亏比、收益风险比
    winRatio = (every_return.dropna() > 0).mean()[0]
    winlossRatio = every_return[every_return > 0].mean()[0] / every_return[every_return < 0].mean()[0] * -1
    calmar = annual_rt / mdd['MDD']
    trade_times = every_return.notnull().sum()[0]  # 交易期数
    # 分年度评价
    account_net['trade_dt'] = account_net.index.values
    account_net['trade_dt'] = account_net['trade_dt'].apply(lambda x: pd.datetime.strptime(str(x), '%Y%m%d'))
    account_net['year'] = account_net['trade_dt'].apply(lambda x: x.year)
    year_all = account_net['year'].unique()
    result.append(['all', annual_rt, accum_rt, SR, mdd['MDD'], winRatio, winlossRatio, calmar,
                   mdd['MDD_date'], mdd['Lastingtime'], mdd['Endingtime'], trade_times])
    every_return_cp = every_return.copy()
    every_return_cp['year'] = list(
        map(lambda x: pd.datetime.strptime(str(x), '%Y%m%d').year, every_return.index.tolist()))

    for year_i in year_all:
        # print(year_i)
        # 取出当年净值(用于计算最大回撤)
        account_net_i = account_net[account_net['year'] == year_i]  # 当年的单利净值
        # 当前年份的每期数据，年度划分以【卖出】日期为准
        every_return_i = every_return[every_return_cp['year'] == year_i]
        # 累计收益、年化收益，按照252天年化
        accum_rt_i = every_return_i.sum()[0]
        annual_rt_i = every_return_i.mean()[0] / adj_cycle * N  # 年化收益通过每个持仓期的收益均值计算
        # 年化波动率
        annu_std_i = daily_ret[account_net['year'] == year_i].std(ddof=1) * np.sqrt(N)
        # annu_std_i = np.nan if annu_std_i == 0 else annu_std_i #年化波动率为0的情况
        #    Sharpe比率
        SR_i = np.nan if annu_std_i == 0 else annual_rt_i / annu_std_i
        # 最大回撤
        mdd_i = cal_mdd(account_net_i['net'], every_return_cp.index[every_return_cp['year'] == year_i].to_list(),
                        if_cmp=0)
        # 胜率、盈亏比、收益风险比
        winRatio_i = (every_return_i.dropna() > 0).mean()[0]
        winlossRatio_i = every_return_i[every_return_i > 0].mean()[0] / every_return_i[every_return_i < 0].mean()[
            0] * -1
        calmar_i = np.nan if (mdd_i['MDD'] == 0) else (annual_rt_i / mdd_i['MDD'])
        trade_times_i = every_return_i.notnull().sum()[0]

        result.append([year_i, annual_rt_i, accum_rt_i, SR_i, mdd_i['MDD'], winRatio_i, winlossRatio_i, calmar_i,
                       mdd_i['MDD_date'], mdd_i['Lastingtime'], mdd_i['Endingtime'], trade_times_i])
    result = pd.DataFrame(result, columns=['Year', 'AnnualRt', 'AccumRt', 'SR', 'MDD',
                                           'WinRatio', 'WinLossRatio', 'Calmar', 'MDD_date',
                                           'MDD_lastdays', 'MDD_recoverdays', 'Periods'])
    return result


def cal_group_ret(factor_data: pd.DataFrame, group: int, sampleind, factor_floor: str, hedge: str, hedge_dir: str):
    """
    计算分组收益
    :param factor_data:
    :param group:
    :param sampleind: 选取对冲基准时，如果基准为行业指数时，需要提取对应行业
    :param factor_floor:
    :param hedge:
    :param hedge_dir:
    :return:
    """
    print('计算分组收益...')
    fac_group = factor_data.copy() * np.nan
    adjust_dates = factor_data.index.to_list()
    # 导入收盘价,计算各期股票收益
    cp = Factor.add_index(Factor.get_apidata(('stk_daily.h5', 'cp')))
    stock_cycle_ret = cp.loc[factor_data.index.to_list(), :].pct_change().shift(-1)  # 下一期的股票收益

    print('计算分组收益：根据因子值进行分组')
    # 根据因子值给各期因子分组
    for i in range(0, len(factor_data)):
        t_i = factor_data.index.to_list()[i]
        nonan = factor_data.loc[t_i, :].notnull().sum()
        print(t_i)
        if (nonan == 0) | (nonan < group):
            if (factor_floor == 'group') | (i == 0):
                print("计算分组有效性： {} 数据过少，分组失败".format(t_i))
                continue
            elif factor_floor == 'last':
                print("计算分组有效性： {} 数据过少，采用上一期分组".format(t_i))
                fac_group.loc[t_i, :] = fac_group.iloc[i - 1, :]
                continue
        else:
            # 因子值最小的为组1,最大的为组group
            fac_group.loc[t_i, :] = pd.qcut(factor_data.loc[t_i, :], group, labels=range(1, group + 1))

    print('计算分组收益：各期指标')
    # 各组样本量、收益均值、胜率、盈亏比
    group_num = pd.DataFrame(np.nan * np.ones((len(factor_data), group)), index=factor_data.index.to_list(),
                             columns=list(range(1, group + 1)))
    group_ret = group_num.copy()
    group_winratio = group_num.copy()
    group_winloss = group_num.copy()

    for t_i in adjust_dates[:-1]:
        # stock_cycle_ret.loc[t_i, :].groupby(fac_group.loc[t_i, :]).count()
        # stock_cycle_ret.loc[t_i,:].groupby(fac_group.loc[t_i, :]).mean()
        temp = stock_cycle_ret.loc[t_i, :].groupby(fac_group.loc[t_i, :]).agg(cal_event_evaluation)
        group_num.loc[t_i, :] = list(map(lambda x: x[0], temp))
        group_ret.loc[t_i, :] = list(map(lambda x: x[1], temp))
        group_winratio.loc[t_i, :] = list(map(lambda x: x[2], temp))
        group_winloss.loc[t_i, :] = list(map(lambda x: x[3], temp))
    group_rank = group_ret.rank(axis=1)  # 各组排序，收益排序最小的为1

    print('计算分组收益：各组收益曲线')
    # 根据因子分组情况计算各组收益率曲线
    group_daily_ret = pd.DataFrame(np.nan * np.ones((len(cp.loc[factor_data.index[0]: factor_data.index[-1], :].index),
                                                     group)),
                                   index=cp.loc[adjust_dates[0]: adjust_dates[-1], :].index.to_list(),
                                   columns=list(range(1, group + 1)))
    group_daily_ret.iloc[0, :] = 0
    for i in range(0, len(adjust_dates) - 1):
        t_i = adjust_dates[i]
        t_ii = adjust_dates[i + 1]
        # print(t_i)
        cycle_net_i = cp.loc[t_i:t_ii, :] / cp.loc[t_i, :]
        group_net_i = pd.DataFrame()
        for g_i in range(1, group + 1):
            group_net_i = pd.concat([group_net_i,
                                     cycle_net_i.loc[:, fac_group.loc[t_i, :] == g_i].mean(axis=1).to_frame()], axis=1)
        group_net_i.columns = list(range(1, group + 1))
        # group_net_i = cycle_net_i.apply(lambda x: x.groupby(fac_group.loc[t_i, :]).mean(),axis=1) #分组净值
        group_daily_ret_i = group_net_i.pct_change()
        group_daily_ret.loc[group_daily_ret_i.index[1:], :] = group_daily_ret_i.iloc[1:, :]

    group_daily_net_cmp = (group_daily_ret + 1).cumprod()  # 复利
    group_daily_net_simp = group_daily_net_cmp.copy() * np.nan  # 单利
    for i in range(0, group):
        group_daily_net_simp.iloc[:, i] = cal_net_simple(group_daily_net_cmp.iloc[:, i], adjust_dates)

    print('计算分组收益：获取对冲基准')
    # 对冲基准
    # 可选范围
    index_range = ['custom', 'SZ50', 'HS300', 'ZZ500', 'equal']
    index_mapping = {'HS300': '000300.SH',
                     'ZZ500': '000905.SH',
                     'SZ50': '000016.SH',
                     'custom': hedge_dir}
    index_cp = Factor.add_index(Factor.get_apidata(('index_daily.h5', 'index_cp')), type='index')

    if hedge in index_range:
        if hedge in ['SZ50', 'HS300', 'ZZ500']:
            benchmark_index = index_cp.loc[group_daily_ret.index, index_mapping[hedge]]
        elif hedge == 'equal':
            # 有因子值的股票收益等权重
            benchmark_index = pd.Series(
                np.nan * np.ones(len(cp.loc[factor_data.index[0]: factor_data.index[-1], :].index)),
                index=cp.loc[adjust_dates[0]:adjust_dates[-1], :].index.to_list())
            benchmark_index.iloc[0] = 0
            for i in range(0, len(adjust_dates) - 1):
                t_i = adjust_dates[i]
                t_ii = adjust_dates[i + 1]
                # print(t_i)
                cycle_net_i = cp.loc[t_i:t_ii, :] / cp.loc[t_i, :]  # 该期所有股票净值
                if factor_data.loc[t_i, :].notnull().sum() == 0:
                    benchmark_index.loc[cycle_net_i.index[1:]] = 0
                else:
                    cycle_net_i.loc[:, factor_data.loc[t_i, :].isnull()] = np.nan  # 因子值为空的设为nan
                    benchmark_index.loc[cycle_net_i.index[1:]] = cycle_net_i.mean(axis=1).pct_change()
            benchmark_index = (benchmark_index + 1).cumprod()
        elif hedge == 'custom':
            try:
                benchmark_index = Factor.get_customdata(index_mapping[hedge])  # 列数只能为1
                stklist, trade_dt = Factor.get_axis(type='stock')
                # 如果没有index，默认索引为标准trade_dt,否则按照因子索引进行reindex
                if benchmark_index.columns.dtype == 'int64':  # 没有索引的话，默认索引为trade_dt，并转换为series
                    benchmark_index.index = trade_dt.iloc[:, 0]
                    benchmark_index = benchmark_index.loc[group_daily_ret.index, 0]
                else:
                    benchmark_index = benchmark_index.reindex(index=group_daily_ret.index)
                    benchmark_index = benchmark_index.iloc[:, 0]
            except Exception:
                raise ValueError('计算分组收益: 获取自定义对冲基准出错')
    else:
        raise ValueError("计算分组收益，请输入正确的对冲基准：index_range {}".format(index_range))

    # 基准净值曲线：复利、单利
    benchmark_index_cmp = benchmark_index.loc[adjust_dates[0]:adjust_dates[-1]] / benchmark_index.loc[adjust_dates[0]]
    benchmark_index_simp = cal_net_simple(benchmark_index_cmp, adjust_dates)  # 单利

    # 分组超额净值：复利、单利
    group_daily_excnet_simp = group_daily_net_simp - np.array(benchmark_index_simp) + 1
    group_daily_excnet_cmp = group_daily_net_cmp - np.array(benchmark_index_cmp.to_frame()) + 1

    print('计算分组收益：进行分组收益评价')
    # 分组收益评价
    group_eva_abs = pd.DataFrame()  # 各组收益评价-单利（不对冲和对冲）
    group_eva_exc = pd.DataFrame()  # 各组收益评价-单利（不对冲和对冲）
    group_eva_abs_yearly = {}
    group_eva_exc_yearly = {}
    turn_fac = pd.DataFrame()
    for i in range(1, group + 1):
        print(i)
        # 换手率
        turn_fac_i = ((fac_group == i) & (fac_group.diff(-1) != 0)).sum(axis=1) / ((fac_group == i).sum(axis=1) * 2)
        turn_fac_i.name = i
        turn_fac = pd.concat([turn_fac, turn_fac_i], axis=1)
        # 收益评价(单利)
        result_group_i = evaluation(group_daily_net_simp.loc[:, i], adjust_dates, if_cmp=0)  # 不对冲
        result_exc_group_i = evaluation(group_daily_excnet_simp.loc[:, i], adjust_dates, if_cmp=0)  # 对冲
        # 收益评价(单利)-累计
        group_eva_abs = pd.concat([group_eva_abs, result_group_i.iloc[0, 1:].to_frame()], axis=1)  # 不对冲
        group_eva_exc = pd.concat([group_eva_exc, result_exc_group_i.iloc[0, 1:].to_frame()], axis=1)  # 对冲
        # 收益评价(单利)-分年
        group_eva_abs_yearly = group_eva_abs_yearly.copy()
        group_eva_exc_yearly = group_eva_exc_yearly.copy()
        group_eva_abs_yearly[i] = result_group_i.loc[1:, :]
        group_eva_exc_yearly[i] = result_exc_group_i.loc[1:, :]
        # group_eva_abs_yearly[i] = result_group_i.loc[1:,['Year', 'AnnualRt', 'AccumRt', 'SR', 'MDD', 'WinRatio',
        #                                                  'WinLossRatio', 'Calmar','MDD_date','MDD_lastdays',
        #                                                  'MDD_recoverdays', 'Periods']]
        # group_eva_exc_yearly[i] = result_exc_group_i.loc[1:,['Year', 'AnnualRt', 'AccumRt', 'SR', 'MDD', 'WinRatio',
        #                                                  'WinLossRatio', 'Calmar','MDD_date','MDD_lastdays',
        #                                                  'MDD_recoverdays', 'Periods']]

    group_eva_abs.columns = list(range(1, group + 1));  # group_eva_abs = group_eva_abs.copy()
    # group_eva_abs = group_eva_abs.loc[['AnnualRt', 'AccumRt', 'SR', 'MDD', 'WinRatio', 'WinLossRatio', 'Calmar',
    #                                    'MDD_date', 'MDD_lastdays',
    #                                    'MDD_recoverdays', 'Periods'
    #                                    ], :]
    group_eva_exc.columns = list(range(1, group + 1));  # group_eva_exc = group_eva_exc.copy()
    # group_eva_exc = group_eva_exc.loc[['AnnualRt','AccumRt', 'SR', 'MDD', 'WinRatio', 'WinLossRatio', 'Calmar',
    #                                    'MDD_date', 'MDD_lastdays',
    #                                    'MDD_recoverdays', 'Periods'
    #                                    ], :]

    print('返回计算结果')
    # 输出结果
    group_ret_result = {}
    group_ret_result['adjust_dates'] = adjust_dates

    group_ret_result['fac_group'] = fac_group
    group_ret_result['group_num'] = group_num
    group_ret_result['group_ret'] = group_ret  # 分组每期收益均值（均为不对冲）
    group_ret_result['group_winratio'] = group_winratio  # 分组每期胜率
    group_ret_result['group_winloss'] = group_winloss  # 分组每期盈亏比
    group_ret_result['group_rank'] = group_rank  # 取值最小的为1

    group_ret_result['daily_net_simp'] = group_daily_net_simp  # 单利净值
    group_ret_result['daily_net_cmp'] = group_daily_net_cmp  # 复利净值

    group_ret_result['daily_excnet_simp'] = group_daily_excnet_simp  # 超额单利净值
    group_ret_result['daily_excnet_cmp'] = group_daily_excnet_cmp  # 超额复利净值

    group_ret_result['group_eva_abs'] = group_eva_abs  # 分组评价（不对冲，单利）
    group_ret_result['group_eva_exc'] = group_eva_exc  # 分组评价（对冲，单利）
    group_ret_result['group_eva_abs_yearly'] = group_eva_abs_yearly  # 不对冲，单利-分年
    group_ret_result['group_eva_exc_yearly'] = group_eva_exc_yearly  # 对冲，单利-分年

    group_ret_result['turn_fac'] = turn_fac  # 换手率

    return group_ret_result


def cal_longshort_ret(group_ret_inf, group, factor_ori):
    print('计算多空净值...')
    group_ret_result = group_ret_inf.copy()
    longshort_inf = {}
    if factor_ori == 1:  # 越大越好,则第一组为空头
        long_n = group
        short_n = 1
    elif factor_ori == -1:  # 第一组为多头
        long_n = 1
        short_n = group

    # 每期收益
    longshort_inf['long_ret'] = group_ret_result['group_ret'].loc[:, long_n]  # 每期多头收益均值
    longshort_inf['short_ret'] = group_ret_result['group_ret'].loc[:, short_n]  # 每期空头收益均值
    longshort_inf['longshort_ret'] = longshort_inf['long_ret'] - longshort_inf['short_ret']  # 每期多空组合收益
    # 净值曲线
    ##不对冲
    longshort_inf['daily_net_simp_long'] = group_ret_result['daily_net_simp'].loc[:, long_n]  # 日度多头净值曲线-单利
    longshort_inf['daily_net_simp_short'] = group_ret_result['daily_net_simp'].loc[:, short_n]  # 日度空头净值曲线-单利
    ##对冲
    longshort_inf['daily_excnet_simp_long'] = group_ret_result['daily_excnet_simp'].loc[:, long_n]  # 日度多头对冲净值-单利
    longshort_inf['daily_excnet_simp_short'] = group_ret_result['daily_excnet_simp'].loc[:, short_n]  # 日度空头对冲净值-单利
    ##多空净值
    longshort_inf['daily_net_simp_longshort'] = longshort_inf['daily_net_simp_long'] - longshort_inf[
        'daily_net_simp_short'] + 1  # 日度多空净值

    # 多空净值评价
    eva_longshort = evaluation(longshort_inf['daily_net_simp_longshort'], group_ret_result['adjust_dates'])
    # eva_longshort = eva_longshort.loc[:, ['Year', 'AnnualRt', 'AccumRt', 'SR', 'MDD', 'WinRatio', 'WinLossRatio', 'Calmar']]

    # 存储结果
    # 1. 对冲后收益评价（多头对冲、空头对冲、多空）
    eva_l_s_ls = pd.concat([group_ret_result['group_eva_exc'][long_n], group_ret_result['group_eva_exc'][short_n],
                            eva_longshort.iloc[0, 1:]], axis=1)
    eva_l_s_ls.columns = ['多头超额', '空头超额', '多空']
    longshort_inf['eva_l_s_ls'] = eva_l_s_ls

    # 2. 每期收益（多头对冲、空头对冲、多空）
    period_ret_l_s_ls = pd.concat(
        [longshort_inf['long_ret'], longshort_inf['short_ret'], longshort_inf['longshort_ret']], axis=1)
    period_ret_l_s_ls.columns = ['多头超额', '空头超额', '多空']
    longshort_inf['period_ret_l_s_ls'] = period_ret_l_s_ls

    # 3. 净值曲线-单利（多头、空头、多头对冲、空头对冲、多空）
    net_l_s_ls = pd.concat([longshort_inf['daily_net_simp_long'], longshort_inf['daily_net_simp_short'],
                            longshort_inf['daily_excnet_simp_long'], longshort_inf['daily_excnet_simp_short'],
                            longshort_inf['daily_net_simp_longshort']], axis=1)
    net_l_s_ls.columns = ['多头', '空头', '多头超额', '空头超额', '多空']
    longshort_inf['net_l_s_ls'] = net_l_s_ls

    # 4. 分年评价单利（多头对冲、空头对冲、多空）
    eva_yearly_l_s_ls = {}
    eva_yearly_l_s_ls['多头超额'] = group_ret_result['group_eva_exc_yearly'][long_n]
    eva_yearly_l_s_ls['空头超额'] = group_ret_result['group_eva_exc_yearly'][short_n]
    eva_yearly_l_s_ls['多空'] = eva_longshort.iloc[1:, :]
    longshort_inf['eva_yearly_l_s_ls'] = eva_yearly_l_s_ls

    return longshort_inf


def analyse_effect(factor_std: pd.DataFrame, prepro_setting: dict, execute_setting: dict):
    """
    因子有效性分析
    :param factor_std:
    :param prepro_setting:
    :param execute_setting:
    :return:
    """
    factor_data = factor_std.copy()
    adjust_dates = factor_data.index.to_list()
    sample_ind = prepro_setting["sample_ind"]
    group = execute_setting["EffectAnalysis_group"]
    factor_ori = execute_setting["EffectAnalysis_fac_ori"]
    factor_floor = execute_setting["EffectAnalysis_floor"]
    hedge = execute_setting["EffectAnalysis_hedge"]
    if 'EffectAnalysis_hedgedir' in execute_setting.keys():
        hedge_dir = execute_setting['EffectAnalysis_hedgedir']
    else:
        hedge_dir = ''

    # 如果因子值太少报错
    if factor_floor == 'group':
        if (factor_data.notnull().sum(axis=1) < group).sum() > 0:
            print('以下期因子值少于组数')
            print(factor_data.index[(factor_data.notnull().sum(axis=1) < group)])
            raise ('因子值太少不进行分组分析!')
    """
    IC分析
    """
    ic_inf = cal_ic(factor_data, group)
    """
    因子收益分析
    """
    fac_return_inf = cal_fac_ret(factor_data, group)
    """
    因子分组收益情况
    """
    group_ret_inf = cal_group_ret(factor_data, group, sample_ind, factor_floor, hedge, hedge_dir)
    """
    因子多空收益
    """
    longshort_inf = cal_longshort_ret(group_ret_inf, group, factor_ori)

    """
    返回有效性分析结果
    """
    result_analyse_effect = {}
    result_analyse_effect['ic_inf'] = ic_inf
    result_analyse_effect['fac_return_inf'] = fac_return_inf
    result_analyse_effect['group_ret_inf'] = group_ret_inf
    result_analyse_effect['longshort_inf'] = longshort_inf
    return result_analyse_effect


def analys_effect_bysample(factor_std: pd.DataFrame, prepro_setting: dict, execute_setting: dict):
    """
    分样本分析因子有效性, 在每个样本内进行一次完整的有效性分析
    :param factor_std:
    :param prepro_setting:
    :param execute_setting:
    :return:
    """

    result_analyse_effect_bysample = {}
    execute_setting_i = execute_setting.copy()
    execute_setting_i['EffectAnalysis_fac_ori'] = execute_setting['EffectBySample_fac_ori']
    execute_setting_i['EffectAnalysis_group'] = execute_setting['EffectBySample_group']
    execute_setting_i['EffectAnalysis_floor'] = execute_setting['EffectBySample_floor']
    execute_setting_i['EffectAnalysis_hedge'] = execute_setting['EffectBySample_hedge']
    prepro_setting_i = prepro_setting.copy()

    if execute_setting["EffectBySample_Sample"] == 'by_ind':
        ind_of_stock = Factor.add_index(Factor.get_apidata(('stk_daily.h5', 'id_citic1')))
        ind_of_stock = ind_of_stock.loc[factor_std.index, :]
        ind_of_stock = ind_of_stock.astype('int')

        ind_name = Factor.get_apidata(('stk_daily.h5', 'ind_name_CITIC_1'))
        max_num = ind_of_stock.max().max()
        for i in range(1, max_num + 1):
            # 无行业分类的（0，nan）不做统计
            print("分样本有效性分析: {}/{}".format(i, max_num))
            print(ind_name.iloc[i - 1][0])
            factor_std_i = factor_std.copy()
            factor_std_i[ind_of_stock != i] = np.nan
            prepro_setting_i['sample_ind'] = ind_name.iloc[i - 1][0]  # 转换为索引需要-1
            result_i = analyse_effect(factor_std_i, prepro_setting_i, execute_setting_i)
            result_analyse_effect_bysample[ind_name.iloc[i - 1][0]] = result_i  # 存储计算结果，key为行业名称
    elif execute_setting["EffectBySample_Sample"] == 'custom':
        # 自定义的样本
        try:
            # 提取数据（支持格式：h5,csv,bp, 形状必须为标准矩阵
            # ('E:/Data/temp_data/testdata/test_h5_new/stk_daily.h5', 'id_citic1')
            # ('E:/Data/temp_data/testdata/', 'id_300.csv')
            # ('E:/Data/temp_data/testdata/', 'bp.npy')
            sample = Factor.add_index(Factor.get_customdata(execute_setting["EffectBySample_Sampledir"]))
        except Exception:
            raise ('分样本有效性分析: 提取自定义样本数据出错')

        sample_unique = np.unique(sample.values).tolist()
        max_num = len(sample_unique)
        for g_i in range(0, max_num):
            i = sample_unique[g_i]  # 样本取值
            print("分样本有效性分析: {}/{}".format(g_i, max_num))
            factor_std_i = factor_std.copy()
            factor_std_i[sample != i] = np.nan
            if execute_setting["EffectBySample_hedge"] == 'custom':  # 分样本的对冲基准
                execute_setting_i['EffectAnalysis_hedgedir'] = execute_setting["EffectBySample_hedgedir"]
            result_i = analyse_effect(factor_std_i, prepro_setting_i, execute_setting_i)
            result_analyse_effect_bysample[i] = result_i  # 存储计算结果，key为行业名称

    return result_analyse_effect_bysample


def score_by_size_ind(factor_std: pd.DataFrame, prepro_setting: dict, execute_setting: dict):
    """
    市值行业分层打分: 将股票池按照市值划分为三组，再按照中信29个行业分为29组，共87组，每组按照因子值等分为5分
    注：因子不需要进行市值行业中性处理
    :param factor_std:
    :param prepro_setting:
    :param execute_setting:
    :return:
    """

    def my_qcut(x, group):
        # 87个组内每个组不相等的元素至少超过4个才进行分组
        if len(x.dropna().unique()) >= (group - 1):
            return pd.qcut(x.rank(method='first'), group, labels=range(1, group + 1), duplicates='drop')
        else:
            return x * np.nan

    factor_data = factor_std.copy()
    fac_group = factor_data.copy() * np.nan
    adjust_dates = factor_data.index.to_list()
    group_all = 3 * 29 * 5  # 共87组->435组
    group = 5
    factor_floor = 'last'
    factor_ori = execute_setting["EffectAnalysis_fac_ori"]
    if factor_ori == 1:  # 越大越好,则第一组为空头
        long_n = group
        short_n = 1
    elif factor_ori == -1:  # 第一组为多头
        long_n = 1
        short_n = group

    # 加载市值、行业、收盘价数据
    ind_of_stock = Factor.add_index(Factor.get_apidata(('stk_daily.h5', 'id_citic1')))
    ind_of_stock = ind_of_stock.loc[adjust_dates, :]
    ind_of_stock[factor_data.isnull()] = np.nan
    mv = Factor.add_index(Factor.get_apidata(('stk_daily.h5', 'mv_float')))  # 此处取流通市值
    mv = mv.loc[adjust_dates, :]
    mv[factor_data.isnull()] = np.nan
    mv_group = mv.copy() * np.nan  # 存储市值分组信息

    cp = Factor.add_index(Factor.get_apidata(('stk_daily.h5', 'cp')))
    stock_cycle_ret = cp.loc[factor_data.index.to_list(), :].pct_change().shift(-1)  # 下一期的股票收益

    print('市值行业分层打分：因子分组')
    for i in range(0, len(factor_data)):
        t_i = adjust_dates[i]
        # print(t_i)
        nonan = factor_data.loc[t_i, :].notnull().sum()
        print(t_i)
        if (nonan == 0) | (nonan < group_all):
            if (factor_floor == 'group') | (i == 0):
                print("市值行业分层打分： {} 数据过少，分组失败".format(t_i))
                continue
            elif factor_floor == 'last':
                print("市值行业分层打分： {} 数据过少，采用上一期分组".format(t_i))
                fac_group.loc[t_i, :] = fac_group.iloc[i - 1, :]
                continue
        else:
            # 因子值最小的为组1,最大的为组5
            mv_group.loc[t_i, :] = pd.qcut(mv.loc[t_i, :], 3, labels=range(1, 4))  # 按照市值从小到大分为3组
            fac_group.loc[t_i, :] = factor_data.loc[t_i, :].groupby(
                [mv_group.loc[t_i, :], ind_of_stock.loc[t_i, :]]).apply(
                lambda x: my_qcut(x, group))

    print("市值行业分层打分：根据因子分组情况计算各期指标")
    # 各组样本量、收益均值、胜率、盈亏比
    group_num = pd.DataFrame(np.nan * np.ones((len(factor_data), group)), index=factor_data.index.to_list(),
                             columns=list(range(1, group + 1)))
    group_ret = group_num.copy()
    group_winratio = group_num.copy()
    group_winloss = group_num.copy()

    for t_i in adjust_dates[:-1]:
        # print(t_i)
        # stock_cycle_ret.loc[t_i, :].groupby(fac_group.loc[t_i, :]).count()
        # stock_cycle_ret.loc[t_i,:].groupby(fac_group.loc[t_i, :]).mean()
        temp = stock_cycle_ret.loc[t_i, :].groupby(fac_group.loc[t_i, :]).agg(cal_event_evaluation)
        group_num.loc[t_i, :] = list(map(lambda x: x[0], temp))
        group_ret.loc[t_i, :] = list(map(lambda x: x[1], temp))
        group_winratio.loc[t_i, :] = list(map(lambda x: x[2], temp))
        group_winloss.loc[t_i, :] = list(map(lambda x: x[3], temp))
    group_rank = group_ret.rank(axis=1)  # 各组排序，收益排序最小的为1

    print("市值行业分层打分：根据因子分组情况计算各组净值曲线")
    group_daily_ret = pd.DataFrame(np.nan * np.ones((len(cp.loc[factor_data.index[0]: factor_data.index[-1], :].index),
                                                     group)),
                                   index=cp.loc[adjust_dates[0]: adjust_dates[-1], :].index.to_list(),
                                   columns=list(range(1, group + 1)))
    group_daily_ret.iloc[0, :] = 0
    for i in range(0, len(adjust_dates) - 1):
        t_i = adjust_dates[i]
        t_ii = adjust_dates[i + 1]
        print(t_i)
        cycle_net_i = cp.loc[t_i:t_ii, :] / cp.loc[t_i, :]
        group_net_i = pd.DataFrame()
        for g_i in range(1, group + 1):
            group_net_i = pd.concat([group_net_i,
                                     cycle_net_i.loc[:, fac_group.loc[t_i, :] == g_i].mean(axis=1).to_frame()], axis=1)
        group_net_i.columns = list(range(1, group + 1))
        # group_net_i = cycle_net_i.apply(lambda x: x.groupby(fac_group.loc[t_i, :]).mean(),axis=1) #分组净值
        group_daily_ret_i = group_net_i.pct_change()
        group_daily_ret.loc[group_daily_ret_i.index[1:], :] = group_daily_ret_i.iloc[1:, :]

    group_daily_net_cmp = (group_daily_ret + 1).cumprod()  # 复利
    group_daily_net_simp = group_daily_net_cmp.copy() * np.nan  # 单利
    for i in range(0, group):
        group_daily_net_simp.iloc[:, i] = cal_net_simple(group_daily_net_cmp.iloc[:, i], adjust_dates)

    print("市值行业分层打分：计算多空净值")
    group_daily_net_simp.loc[:, 'longshort'] = group_daily_net_simp.loc[:, long_n] - group_daily_net_simp.loc[:,
                                                                                     short_n] + 1
    group_daily_net_cmp.loc[:, 'longshort'] = group_daily_net_cmp.loc[:, long_n] - group_daily_net_cmp.loc[:,
                                                                                   short_n] + 1

    # 多空净值评价(单利)
    eva_longshort_simp = evaluation(group_daily_net_simp.loc[:, 'longshort'], adjust_dates, if_cmp=0)
    # eva_longshort_simp = eva_longshort_simp.loc[:, ['Year', 'AnnualRt', 'SR', 'MDD', 'WinRatio', 'WinLossRatio', 'Calmar']]
    # 复利
    # eva_longshort_cmp  = evaluation(group_daily_net_cmp.loc[:, 'longshort'], adjust_dates, if_cmp=0 )
    # eva_longshort_cmp = eva_longshort_cmp.loc[:, ['Year', 'AnnualRt', 'SR', 'MDD', 'WinRatio', 'WinLossRatio', 'Calmar']]

    # 每组评价
    print('每组净值评价')
    eva_longshort_simp = pd.DataFrame()
    eva_longshort_simp_yearly = {}
    for i in group_daily_net_simp.columns.tolist():
        temp = evaluation(group_daily_net_simp.loc[:, i], adjust_dates, if_cmp=0)
        # temp.name = i
        eva_longshort_simp = pd.concat([eva_longshort_simp, temp.iloc[0, :].to_frame()], axis=1)
        eva_longshort_simp_yearly[i] = temp.iloc[1:, :]

    eva_longshort_simp.columns = group_daily_net_simp.columns.tolist()
    eva_longshort_simp.drop('Year', inplace=True)  # 不对冲的每组收益

    print("市值行业分层打分：返回计算结果")
    score_by_size_ind_inf = {}
    score_by_size_ind_inf['adjust_dates'] = adjust_dates

    score_by_size_ind_inf['fac_group'] = fac_group
    score_by_size_ind_inf['group_num'] = group_num
    score_by_size_ind_inf['group_ret'] = group_ret  # 分组每期收益均值（均为不对冲）
    score_by_size_ind_inf['group_winratio'] = group_winratio  # 分组每期胜率
    score_by_size_ind_inf['group_winloss'] = group_winloss  # 分组每期盈亏比
    score_by_size_ind_inf['group_rank'] = group_rank  # 取值最小的为1

    score_by_size_ind_inf['daily_net_simp'] = group_daily_net_simp  # 单利净值（不对冲）
    score_by_size_ind_inf['daily_net_cmp'] = group_daily_net_cmp  # 复利净值（不对冲）

    score_by_size_ind_inf['eva_longshort_simp'] = eva_longshort_simp  # 多空评价（单利）

    return score_by_size_ind_inf


def corr_riskfactor(factor_std: pd.DataFrame, execute_setting: dict):
    factor_data = factor_std.copy()
    adj_date = factor_data.index.to_list()
    riskfactors = execute_setting["RiskAnalysis_factors"]
    # riskfactors = [('risk_factor.h5','Beta'), ('E:/Data/temp_data/testdata/', 'mom21.npy'), ('E:/Data/temp_data/testdata/', 'size.csv')]

    if riskfactors == 'all':
        riskfactor_name = Factor.get_apikeys('risk_factor.h5')
        riskfactor_name = [x[1:] for x in riskfactor_name]
        # riskfactor_inf = pd.DataFrame({'riskfactor_name': riskfactor_name})
        # riskfactor_inf['file_dir'] = 'risk_factor.h5'
        riskfactors = [('risk_factor.h5', x) for x in riskfactor_name]
    else:
        riskfactor_name = [x[1].split('.')[0] for x in riskfactors]

    # 导入风险因子数据
    risk_data = []
    print("计算与风险因子的相关性：风险因子数据导入...")
    for i, dir_i in enumerate(riskfactors):
        print(i, dir_i)
        risk_name_i = dir_i[1].split('.')[0]
        if dir_i[0] == 'risk_factor.h5':  # 提取风险因子
            ## todo exec warning
            exec("{}=Factor.get_apidata(dir_i)".format(dir_i[1]))
        else:  # 提取alpha因子或自定义因子
            exec("{}=Factor.get_customdata(dir_i)".format(risk_name_i))

        if eval("Factor.valid_shape({})".format(risk_name_i)):
            exec("{} = Factor.add_index({})".format(risk_name_i, risk_name_i))  # 检查风险因子形状，加上索引
        else:
            raise ValueError('风险因子形状不一致:' + risk_name_i)
        exec("risk_data.append({})".format(risk_name_i))

    print("计算与风险因子的相关性：计算与风险因子的相关系数...")
    corr_all = pd.Series()
    corr_all_sum = pd.DataFrame()
    for i in adj_date:
        risk_data_i = pd.DataFrame()
        for r_i in range(0, len(riskfactor_name)):
            temp = risk_data[r_i].loc[i, :]
            temp.name = riskfactor_name[r_i]
            risk_data_i = pd.concat([risk_data_i, temp.to_frame()], axis=1)
        # 计算秩相关系数
        factor_data_i = factor_data.loc[i, :]
        factor_data_i.name = 'factor'
        corr_i = pd.merge(factor_data_i, risk_data_i, left_index=True, right_index=True).corr('spearman')
        corr_all.loc[i] = corr_i
        corr_all_sum = pd.concat([corr_all_sum, corr_i])
    corr_all_sum.loc[:, 'group'] = corr_all_sum.index.to_list()
    corr_all_mean = corr_all_sum.groupby('group').mean()
    corr_all_std = corr_all_sum.groupby('group').std(ddof=1)
    corr_all_stab = corr_all_mean / corr_all_std
    corr_all_mean = corr_all_mean.reindex(index=corr_all_mean.columns.tolist())
    corr_all_stab = corr_all_stab.reindex(index=corr_all_stab.columns.tolist())
    corr_all_stab.replace(np.inf, np.nan, inplace=True)

    corr_inf = {}
    corr_inf['mean'] = corr_all_mean
    corr_inf['stability'] = corr_all_stab
    return corr_inf



if __name__ == '__main__':
    pass
