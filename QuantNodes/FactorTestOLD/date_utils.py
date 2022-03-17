# coding = utf-8
"""
Author: DaisyZhou

date: 2019/11/6 16:17
"""

import pandas as pd
import numpy as np
from datetime import datetime

def valid_date(trade_dts: pd.DataFrame) -> bool:
    """
    验证日期格式,必须为yyyymmdd数值型
    :param trade_dts: Series 或 Dataframe, 一列
    :return: 1：格式正确
    """



    if (isinstance(trade_dts, pd.DataFrame)| isinstance(trade_dts, pd.Series)):
        """是否为Series或Dataframe"""
        """内部是否只有1种类型且为yyyymmdd数值型"""
        if len(trade_dts.shape) == 2:
            #Dataframe
            num_lens = len(str(trade_dts.iloc[0,0]))
            data_type = trade_dts.dtypes[0]
        else:
            num_lens = len(str(trade_dts.iloc[0]))
            data_type = trade_dts.dtypes

            # if (num_lens==8) & (trade_dts.dtypes == 'int64'):
        if (num_lens == 8) & (data_type  == 'int64'):
            # print(1)
            return 1
        else:
            # print(0)
            return 0
    else:
        print("Fun: valid_date 数据类型错误，请输入pd.DataFrame ,pd.Series")
        return 0

def datenum_2date(trade_dt: pd.DataFrame, type='datetime') -> pd.DataFrame:
    """
    将trade_date转化成不同类型
    :param trade_dt:
    :param type: 'datetime','str'
    :return: Dataframe
    """
    trade_dt = trade_dt.copy()
    if not valid_date(trade_dt):
        raise ValueError("请输入dataframe或series")

    if isinstance(trade_dt, pd.Series):
        trade_dt = trade_dt.to_frame()

    if type == 'datetime':
        trade_dt_dates = trade_dt.applymap(lambda x: datetime.strptime(str(int(x)), '%Y%m%d'))
    elif type == 'str':
        trade_dt_dates = trade_dt.applymap(lambda x: str(int(x)))
    else:
        raise ValueError("请输入正确的转换类型")
    return trade_dt_dates

def date_2datenum(trade_dt: pd.DataFrame) -> pd.DataFrame:
    """
    将datetime转化成trade_dt的yyyymmdd int64型
    :param trade_dt:
    :param type: 'datetime','str'
    :return: Dataframe
    """
    trade_date = trade_dt.copy()
    if not (len(trade_date.dtypes) == 1) & (trade_date.dtypes[0] == 'datetime64[ns]'):
        raise ValueError("请输入datetime64[ns]型")
    trade_dt_dates = trade_date.applymap(lambda x: int(datetime.strftime(x, '%Y%m%d')))
    return trade_dt_dates

def chg_idx2datestr(data):
    #将Series或DataFrame的index转换为datestr 'yyyy/mm/dd'型
    data = data.copy()
    date_idx = data.index.tolist()
    date_idx_str = list(map(lambda x: str(x)[:4] + '/' + str(x)[4:6] + '/' + str(x)[6:], date_idx))
    data.index = date_idx_str
    return data

def resample_tradedate(trade_dt: pd.DataFrame, rule=('W','end')) -> pd.DataFrame:
    """
    将交易日转换成周度、月度等
    :param trade_dt:
    :param rule:('W','begin')->turple: 'W','M' + 'begin', 'end'
    :return:
    """
    trade_date = trade_dt.copy()
    trade_date.columns = ['trade_dt']
    if not valid_date(trade_date):
        raise ValueError("输入日期格式有误,请输入正确的日期格式")
    trade_date = datenum_2date(trade_date)

    if not isinstance(rule, tuple):
        raise ValueError("请输入正确的调仓模式 ('M','end')")

    if rule[0] == 'W':
        #获取每周最后一个交易日
        trade_date['period'] = trade_date.iloc[:,0].apply(lambda x: x.weekday())
        if rule[1] == 'end':
            trade_date['diff'] = trade_date['period'].diff(-1)
            trade_date = trade_date.loc[trade_date['diff'] > 0, 'trade_dt'].to_frame()
        elif rule[1] == 'begin':
            trade_date['diff'] = trade_date['period'].diff(1)
            trade_date = trade_date.loc[trade_date['diff'] < 0, 'trade_dt'].to_frame()
    elif rule[0] == 'M':
        trade_date['period'] = trade_date.iloc[:,0].apply(lambda x: x.month)
        if rule[1] == 'end':
            trade_date['diff'] = trade_date['period'].diff(-1)
            trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
        elif rule[1] == 'begin':
            trade_date['diff'] = trade_date['period'].diff(1)
            trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
    elif rule[0] == 'Q':
        trade_date['period'] = trade_date.iloc[:,0].apply(lambda x: x.quarter)
        if rule[1] == 'end':
            trade_date['diff'] = trade_date['period'].diff(-1)
            trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
        elif rule[1] == 'begin':
            trade_date['diff'] = trade_date['period'].diff(1)
            trade_date = trade_date.loc[trade_date['diff'] != 0, 'trade_dt'].to_frame()
    else:
        raise ValueError("请输入正确resample参数")
    #转换成yyyymmdd int型
    trade_date = date_2datenum(trade_date)
    return trade_date

def get_adjustdate(trade_dt: pd.DataFrame, beg_date: int, end_date: int,
                   adj_mode=('M', 'end')) -> pd.DataFrame :
    """
    根据起始日，截至日，调仓模式确定调仓日
    :param trade_dt: yyyymmdd数值型
    :param beg_date: yyyymmdd数值型
    :param end_date: yyyymmdd数值型
    :param adj_mode:
                     ('M','end'),('M','begin')
                     ('D', int),
                     ('W','end'),('W','begin')
                     ('custom', adj_date_arg)
    :return: adj_date -> Dataframe
    """
    trade_dt = trade_dt.copy()
    """判断输入的类型"""
    if type(beg_date) != type(end_date):
        raise ValueError("起始日与截止日格式不一致！")
    else:
        if not isinstance(beg_date, int):
            raise ValueError("请输入正确的调仓日期格式: yyyymmdd数值型")

    if not ((isinstance(adj_mode, tuple) & (len(adj_mode) == 2))):
        raise ValueError("请输入正确的调仓模式 ('M','end')")

    if not valid_date(trade_dt):
        raise ValueError("请输入正确的trade_dt格式，pd.DataFrame, pd.Series")

    if type(trade_dt) is pd.core.series.Series:
        trade_dt = trade_dt.to_frame()

    """调整测试起止时间"""
    try:
        beg_date_new = trade_dt[trade_dt>=beg_date].dropna().iloc[0,0]
        end_date_new = trade_dt[trade_dt <= end_date].dropna().iloc[-1, 0]

        beg_date_newidx = np.where(trade_dt.iloc[:,0] == beg_date_new)[0][0]
        end_date_newidx = np.where(trade_dt.iloc[:,0] == end_date_new)[0][0]
    except Exception:
        raise ("获取测试起止日期出错, 现有数据起始日:{}，截止日:{}".format(trade_dt.iloc[0,0], trade_dt.iloc[-1,0]))


    """确定调仓日"""
    if (adj_mode[0] == 'M') | (adj_mode[0] == 'W'):
        # trade_dt = trade_dt.iloc[beg_date_newidx: end_date_newidx+1]
        adj_date = resample_tradedate(trade_dt, adj_mode)
        adj_date = adj_date[(adj_date>=beg_date) & (adj_date<=end_date)].dropna()
        adj_date = adj_date.astype(trade_dt.iloc[0].values)

    elif adj_mode[0] == 'D':
        adj_date = trade_dt.iloc[beg_date_newidx:end_date_newidx+1:adj_mode[1]]
    elif adj_mode[0] == 'custom':
        adj_date = adj_mode[1]
        if not valid_date(adj_date):
            raise ValueError("自定义调仓日格式有误：请输入pd.Dataframe pd.Series yyyymmdd int型")
    # adj_date =  adj_date[beg_date: end_date]

    return adj_date

def offset_date(date_input: np.array, trade_dt_all: pd.Series,
                n: int, mode='D', type='trade_date', if_modify = False) -> np.array:
    """
    对日期进行偏移
    :param date_input: 需要偏移的日期序列 array或Series, yyyymmdd int型,
    :param trade_dt_all: 所有交易日,dataframe或series
    :param n: 偏移量，0表示取离输入日期最近的一个交易日
    :param mode: 'D','W','M'
    :param type: trade_date, calendar, 按交易日、日历日
    :param if_modify: 若日期超出索引范围是否进行修正
    :return:
    """
    date_offseted = []
    # date_input = np.array([20050207,20050710])
    #格式转换成Series
    if isinstance(trade_dt_all, pd.DataFrame):
        trade_dt_all = trade_dt_all.iloc[:,0]


    #日期偏移
    if type == 'trade_date':
        if mode =='D':
            # 离输入日期最近的一个交易日
            date_last = list(map(lambda x: trade_dt_all.values[trade_dt_all.values <= x][-1], list(date_input)))
            date_last_idx = pd.Index(trade_dt_all).get_indexer(date_last)
            adj_date = trade_dt_all.copy()
        elif (mode == 'W') | (mode == 'M') | (mode == 'Q') :
            #resample成对应周期
            adj_date = resample_tradedate(trade_dt_all, (mode,'end'))
            try:
                date_last = list(map(lambda x:  adj_date.values[ adj_date.values <= x][-1], list(date_input)))
                date_last_idx = pd.Index(adj_date.iloc[:,0]).get_indexer(date_last)
            except IndexError:
                print("日期偏移超过索引范围!")

        try:
             # date_offseted = np.array(trade_dt_all.iloc[date_last_idx + n])
             date_offseted = np.array(adj_date.iloc[date_last_idx + n])
        except IndexError as e:
            print("日期偏移超过索引范围!")
            new_idx = date_last_idx + n
            if np.where(new_idx > len(trade_dt_all)-1)[0] > 0:
                 print('index: {} 超过最大日期!'.format(np.where(new_idx > len(trade_dt_all)-1)[0][0]))

            if len(np.where(new_idx < 0 )[0]) > 0:
                 print('index: {} 小于最小日期!'.format(np.where(new_idx < 0 )[0][0]))

            if if_modify:
                print("将进行日期修正！")
                new_idx = np.where(new_idx > len(trade_dt_all) - 1, len(trade_dt_all) - 1, np.where(new_idx < 0, 0, new_idx))
                # new_idx = np.where(new_idx < 0, 0, new_idx)
                date_offseted = np.array(trade_dt_all.iloc[new_idx])

    return date_offseted

def count_date(start_dates: pd.Series, end_dates: pd.Series, trade_dt_all: pd.Series, mode='D', type='trade_date'):
    """
    两个日期之间有几个交易日
    :param start_dates: pd.Series或int
    :param end_dates: pd.Series或int
    :param trade_dt_all: pd.Series或pd.DataFrame
    :param mode: 'D'
    :param type:
    :return:
    """
    #检查格式
    if not valid_date(trade_dt_all):
        raise ValueError('function count_date:输入参数trade_dt_all日期格式有误')

    if isinstance(start_dates, int) & isinstance(end_dates, int):
        start_dates = pd.Series(start_dates)
        end_dates = pd.Series(end_dates)

    if isinstance(start_dates, pd.Series) | isinstance(start_dates, pd.DataFrame):
        if not (valid_date(start_dates) & valid_date(end_dates)):
            raise ValueError('function count_date:输入参数start_dates end_dates日期格式有误')

        # 格式转换成Series
        if isinstance(trade_dt_all, pd.DataFrame):
            trade_dt_all = trade_dt_all.iloc[:,0]

        start_dates_new = offset_date(start_dates, trade_dt_all, 0, mode, type = 'trade_date', if_modify = False)
        end_dates_new = offset_date(end_dates, trade_dt_all, 0, mode, type = 'trade_date', if_modify = False)

        start_dates_new_idx = pd.Index(trade_dt_all.values).get_indexer(start_dates_new )
        end_dates_new_idx = pd.Index(trade_dt_all.values).get_indexer(end_dates_new)

    else:
        raise ValueError('function count_date:输入参数start_dates end_dates日期格式有误')






    days_between = end_dates_new_idx-start_dates_new_idx
    return days_between


if __name__ == '__main__':
    """
    import factor_utils
    from factor_utils import Factor
    stklist, trade_dt = Factor.get_axis(type='stock')

    date_input = trade_dt.iloc[[56, 76],0] # 20050401, 20050429
    trade_dt_all = trade_dt.iloc[:,0].copy()
    temp = offset_date(date_input, trade_dt_all, 1, 'Q')
    print(temp)
    """

