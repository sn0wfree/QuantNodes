# coding = utf-8
"""
Author: DaisyZhou

date: 2019/11/4 9:43
"""
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import statsmodels.api as sm
from pathlib import Path
import date_utils
# from . import date_utils #封装成包

# 当前文件目录
DIR_OF_THIS_FILE = Path(__file__).resolve()
# 主程序目录
DIR_OF_MAIN_PROG = DIR_OF_THIS_FILE.parent


class Factor():
    #TODO： 需要根据本地目录修改文件地址,用于提取标准格式文件
    # api = 'E:/Data/temp_data/testdata/test_h5_new/'
    # api = str(DIR_OF_MAIN_PROG) + '\\testdata\\test_h5_new\\'
    api = './testdata/test_h5_new/'

    # def __init__(self, datadir=(api,''), **kwargs):
    def __init__(self, datadir=(api, ''), api = api):
        """拆包文件地址: 文件夹+hd5名"""
        # kwargs = {'api':'E:/Data/temp_data/testdata/'}
        self.dir = datadir[0]
        self.filename = datadir[1]
        self.api = api
        # if kwargs is not None:
        #     for key, value in kwargs.items():
        # if (len(kwargs)!=0) & ('api' in kwargs.keys()):
        #     exec('self.api = "{api}"'.format(**kwargs) )

    # def get_apidata(self, apidata_dir=('stk_daily.h5','cp')):
    #     """
    #     h5+因子名称
    #     提取API数据,参数只需要API文件目录下的h5文件名和地址，不需要完整路径
    #     """
    #     h5_store = pd.HDFStore(self.api + apidata_dir[0], mode='r')
    #     try:
    #         apidata = h5_store.get(apidata_dir[1] + '.csv')
    #     except Exception:
    #         raise("获取api因子数据出错!")
    #     finally:
    #         h5_store.close()
    #
    #     print("实例")
    #     return apidata

    @classmethod
    def get_apidata(cls, apidata_dir=('stk_daily.h5','cp')) -> pd.DataFrame:
        """
        提取API数据,参数只需要API文件目录下的h5文件名和地址，不需要完整路径

        :param apidata_dir: turple, (文件名.h5, 因子名)

        :return api 目录下的因子值
        """
        h5_store = pd.HDFStore(cls.api + apidata_dir[0], mode='r')
        try:
            # if len(apidata_dir[1].split('.')) == 1: #没有文件格式
            #     apidata = h5_store.get(apidata_dir[1] + '.csv')
            # else:
            #检查是否存在
            if eval("\'/{}\' in h5_store.keys()".format(apidata_dir[1])):
                apidata = h5_store.get(apidata_dir[1])
            else:
                raise (apidata_dir[1] + "数据不存在, h5_store.keys: ", h5_store.keys())

        except Exception:
            raise("获取api因子数据出错!")
        finally:
            h5_store.close()
        # print("类")
        return apidata

    @classmethod
    def get_apikeys(cls, apidata_dir = 'stk_daily.h5'):
        h5_store = pd.HDFStore(cls.api + apidata_dir, mode='r')
        k = h5_store.keys().copy()
        h5_store.close()
        return k

    @classmethod
    def get_axis(cls, type = 'stock') -> tuple:
        """
        提取api下的indexlist和trade_dt
        :return: assetlist, trade_dt
        """
        if type == 'stock':
            h5_store = pd.HDFStore(cls.api + 'stk_daily.h5', mode='r')
            trade_dt = cls.get_apidata(('stk_daily.h5', 'trade_dt'))
            assetlist = cls.get_apidata(('stk_daily.h5', 'stklist'))
        elif type == 'index':
            h5_store = pd.HDFStore(cls.api + 'index_daily.h5', mode='r')
            trade_dt = cls.get_apidata(('index_daily.h5', 'trade_dt'))
            assetlist = cls.get_apidata(('index_daily.h5', 'indexlist'))

        h5_store.close()
        return assetlist, trade_dt

    @classmethod
    def get_customdata(cls, data_dir=(api,'')) -> pd.DataFrame:
        """
        从输入的自定义目录加载因子(自定义格式)
        :param data_dir: 文件路径+文件名
        ('E:/Data/temp_data/testdata/', 'bp.csv')
        ('E:/Data/temp_data/testdata/riskfactor.h5', 'size.csv')

        :returns factor : pd.DataFrame
        """
        factor = []
        if (data_dir[0].split('/')[-1] == "") & (data_dir[1].split('.')[-1] == 'csv'):
            factor = pd.read_csv(data_dir[0] + data_dir[1], index_col=0)
        elif data_dir[1].split('.')[-1] == 'npy':
            factor = pd.DataFrame(np.load(data_dir[0] + data_dir[1],allow_pickle = True))
        elif data_dir[0].split('.')[-1] == 'h5': #.h5视作文件夹
            h5_store = pd.HDFStore(data_dir[0], mode='r')
            factor = h5_store.get(data_dir[1])
            h5_store.close()
        else:
            raise ("请输入支持的数据类型！")
        return factor

    @classmethod
    def add_index(cls, factor: pd.DataFrame, type='stock', inplace=True) ->pd.DataFrame:
        """
        给因子加标签
        :param factor: 不带标签的因子
        :param type: 数据类型：股票和指数分别加标签 type = 'stock' 'index'
        :return: 带stklist和trade_dt标准化因子
        """
        if inplace == False:
            factor_copy = factor.copy()
        else:
            factor_copy = factor
        # if type == 'stock':
            # assetlist = Factor().get_apidata(('stk_daily.h5', 'stklist'))
            # trade_dt = Factor().get_apidata(('stk_daily.h5', 'trade_dt'))
        # elif type == 'index':
            # assetlist = Factor().get_apidata(('index_daily.h5', 'indexlist'))
            # trade_dt = Factor().get_apidata(('index_daily.h5', 'trade_dt'))
        assetlist, trade_dt = cls.get_axis(type)
        factor_copy.index = trade_dt.iloc[:,0].values
        factor_copy.columns = assetlist.iloc[:,0].values
        return factor_copy

    @classmethod
    def select_range(cls, index: str, industry: tuple,
                     index_customdir=('E:/Data/temp_data/testdata/', 'id_300.csv')) ->pd.DataFrame:
        """
        根据样本池限定样本范围，
        :param index: 指数，'all', 'HS300'、'ZZ500'、'ZZ800'、'SZ50'、'custom'(需要填customdir)
        :param industry=('id_citic1', '房地产')
        :param index_customdir: 自定义index的目录 ,只要>0就在样本池中 index_customdir=('E:/Data/temp_data/testdata/', 'id_300.csv')
        :returns: stock_sample：pd.DataFrame, 1表示选入，nan反之
        """

        stklist, trade_dt = cls.get_axis()
        """指数范围"""
        index_range = ['all', 'custom', 'SZ50','HS300', 'ZZ500', 'ZZ800']
        index_mapping = {'HS300': ('stk_daily.h5', 'id_300'),
                        'ZZ500': ('stk_daily.h5', 'id_500'),
                        'SZ50': ('stk_daily.h5', 'id_50'),
                        'custom': index_customdir}
        """
        指数筛选：选中的为1，否则为0
        """
        if index in index_range:
            # index_filt = pd.DataFrame(np.nan * np.zeros((len(trade_dt), len(stklist))))
            index_filt = np.ones((len(trade_dt), len(stklist))) # 初始化为array 取值为1
            if index == 'all':
                pass
                # index_filt = pd.DataFrame(np.ones((len(trade_dt), len(stklist))))
            elif index in ['SZ50','HS300', 'ZZ500', 'custom']:
                if index == 'custom':
                    if_index = cls.get_customdata(index_mapping[index])
                else:
                    if_index = cls.get_apidata(index_mapping[index])
                if_index = if_index.replace(np.nan, 0)
                # index_filt[if_index>0]=1
                index_filt = index_filt * if_index.values

            elif index == 'ZZ800':
                if_index_300 = cls.get_apidata(index_mapping['HS300']); if_index_300 = if_index_300.replace(np.nan, 0)
                if_index_500 = cls.get_apidata(index_mapping['ZZ500']); if_index_500 = if_index_500.replace(np.nan, 0)
                if_index = if_index_300 + if_index_500
                index_filt = index_filt * if_index.values
                # index_filt[if_index>0]=1
            else:
                raise ValueError("指数范围筛选出错！")
        else:
            raise ValueError("Unsupportable index, please input:" + str(index_range))



        #中信31个
        #TODO: 测试行业
        """行业范围 初始化为1 选中的为1，否则为0"""
        industry_range = ['all', 'id_citic1A', 'id_citic1']
        industry_mapping = {'id_citic1A': 'ind_name_CITIC_1A',
                            'id_citic1': 'ind_name_CITIC_1'
                            }

        if (industry == 'all') | (industry[0] in industry_range):
            # industry_filt = pd.DataFrame(np.zeros((len(trade_dt), len(stklist))))
            industry_filt = np.ones((len(trade_dt), len(stklist))) # 初始化为array 取值为1
            if industry == 'all':
                pass
                # industry_filt = pd.DataFrame(np.ones((len(trade_dt), len(stklist))))
            elif industry[0] in industry_range:
                id_industry = cls.get_apidata(('stk_daily.h5', industry[0]))
                ind_name = cls.get_apidata(('stk_daily.h5', industry_mapping[industry[0]])) #提取行业名称
                ind_idx = np.where(ind_name == industry[1])[0][0] + 1 #行业名称对应的行业编号（索引需要+1）
                industry_filt[id_industry != ind_idx] = 0
            else:
                raise ValueError("行业范围筛选出错！")
        else:
            raise ValueError("Unsupportable index, please input:" + str(index_range) )


        """合并筛选"""
        stock_sample = index_filt * industry_filt
        stock_sample[stock_sample > 0] = 1
        stock_sample = pd.DataFrame(stock_sample)
        #用nan替换0
        stock_sample = stock_sample.replace(0, np.nan)
        return stock_sample

    @classmethod
    def valid_shape(cls, factor:pd.DataFrame) ->bool :
        """
        检查因子shape是否为标准矩阵
        :param factor: 因子输入
        :returns 1,0
        """
        stklist, trade_dt = cls.get_axis()
        if not (factor.shape == (len(trade_dt), len(stklist))):
            print("请检查因子shape!")
            return 0
        else:
            return 1

    @classmethod
    def valid_tradable(cls, ifnoST:bool, ifnoSusp:bool, ifnoUpDownLimit:bool, ifnoNewStock:int, if_trace=None):
        """
        验证股票可交易性
        :param ifnoST: 1代表剔除ST
        :param ifnoSusp: 1代表剔除停牌
        :param ifnoUpDownLimit: 1代表剔除涨跌停
        :param ifnoNewStock: n代表剔除上市小于n日的新股,
        :param if_trace:某种状态过去m日出现过n次的 if_trace = {'suspend': (25, 1), 'st': (250,1), 'ud_limit': (-1,1)}
        :return: Dataframe : if_tradable,可交易为1，否则为np.nan
        """
        stklist, trade_dt = cls.get_axis()
        st = cls.get_apidata(('stk_daily.h5','st')) #st的为1，否则为0
        suspend = cls.get_apidata(('stk_daily.h5', 'suspend')) #停牌的为1
        ud_limit = cls.get_apidata(('stk_daily.h5', 'ud_limit')) #涨跌停为+-1
        ud_limit = ud_limit.abs()
        ipo_days = cls.get_apidata(('stk_daily.h5', 'ipo_days')) #上市时间，0表示未上市，-1表示退市

        if_tradable = pd.DataFrame(np.ones((len(trade_dt), len(stklist))))
        if ifnoST:
            if_tradable[st==1] = np.nan

        if ifnoSusp:
            if_tradable[suspend==1] = np.nan

        if ifnoUpDownLimit:
            # if_tradable[(ud_limit)==1.0 | (ud_limit)==-1.0] = np.nan
            if_tradable[ud_limit == 1.0] = np.nan
            if_tradable[ud_limit == -1.0] = np.nan

        if ifnoNewStock:
            if_tradable[ipo_days<ifnoNewStock] = np.nan

        if if_trace is not None:
            trace_data=[]
            trace_list = list(if_trace.keys())
            for trace_i in trace_list:
                # print(trace_i)
                m = if_trace[trace_i][0] #过去m日
                n = if_trace[trace_i][1] #出现n次
                # print(trace_data)
                ldict={}
                exec("trace_data = " + trace_i + ".copy()", locals(), ldict)
                trace_data = ldict['trace_data']
                # print(trace_data)
                # if trace_i == 'suspend':
                #     trace_data = suspend.copy()
                # elif trace_i == 'st':
                #     trace_data = st.copy()
                # elif trace_i == 'ud_limit':
                #     trace_data = ud_limit.copy()
                # exec("trace_data = " +  trace_i + ".copy()")
                # print(trace_data)
                if m > 0:
                    trace_sum = trace_data.rolling(m).sum().shift(1) #不含当天
                else:
                    trace_sum = trace_data.rolling(-m).sum().shift(m)
                if_tradable[trace_sum >= n] = np.nan
        return  if_tradable

    @staticmethod
    def preprocess_onePeriod(factor_i: pd.Series,stock_tradable_adjdate: pd.DataFrame,
                            ind_of_stock_adjdate: pd.DataFrame, method: dict) -> pd.Series:
            """
            输入某一日的因子值和index为包含当日因子的日期的行业分类,可交易性。只对有分类可交易的股票做处理
            :param factor_i: 该日因子值，factor_i.name = yyyymmdd
            :param stock_tradable_adjdate :可交易性（只对tradable的做填充）
            :param ind_of_stock_adjdate: 行业分类
            :param method:预处理方法，{'missing': 'ind_avg', 'extreme': 'median', 'norm': 'zscore'}
            :return: 行业均值填充后的因子值
            """
            factor_i_new_all = factor_i.copy()
            #检查数据
            # print(factor_i.name)
            assert (([factor_i.name[0]] in ind_of_stock_adjdate.index.values) &
                    ([factor_i.name[0]] in stock_tradable_adjdate.index.values)), \
                    "行业均值填充：因子日期不在行业因子或stock_trabdable矩阵内"

            #将调仓当日因子值、行业和股票可交易性合并
            ind_of_stock_adjdate_i = ind_of_stock_adjdate.loc[factor_i.name[0],:]
            stock_tradable_adjdate_i = stock_tradable_adjdate.loc[factor_i.name[0], :]
            #若全都不可交易，则直接返回原始因子值
            if stock_tradable_adjdate_i.count() == 0:
                print("该期股票全都不可交易，不进行因子预处理：", factor_i.name)
                return factor_i_new_all

            factor_i_new = pd.DataFrame(zip(factor_i,ind_of_stock_adjdate_i, stock_tradable_adjdate_i),
                                        index =ind_of_stock_adjdate.columns.tolist(),
                                        columns=['factor','ind','stock_tradable'])

            # #剔除不可交易和无行业分类的(只对有行业分类且可交易（非nan非0）的做填充)
            # factor_i_new = factor_i_new.loc[~(factor_i_new['ind']*factor_i_new['stock_tradable']).isna(),:]
            # factor_i_new = factor_i_new.loc[~(factor_i_new['ind'] * factor_i_new['stock_tradable']==0), :]
            #剔除不可交易的(无行业分类的，还是需要去极值和标准化)
            factor_i_new = factor_i_new.loc[~(factor_i_new['stock_tradable']).isna(),:]
            factor_i_new = factor_i_new.loc[~(factor_i_new['stock_tradable']==0), :]
            factor_i_new['ind'] = factor_i_new['ind'].replace(np.nan, 0) #用0替换行业分类nan的股票的行业数据,避免均值填充出错
            factor_i_new['factor_filled'] = factor_i_new['factor'].copy()
            """缺失值：None, 'ind_avg'(业均值填充)"""
            if (method["missing"] is None) | (method["missing"] == ""):
                pass
            elif method['missing'] == 'ind_avg':
                # factor_i_new.loc[:,'factor_filled'] = factor_i_new.groupby('ind')['factor'].transform(
                #                                 lambda x: x.fillna(x.mean()))
                #只对有行业分类的用行业均值填充
                factor_i_new.loc[factor_i_new['ind']>0,'factor_filled'] = \
                    factor_i_new.loc[factor_i_new['ind']>0,:].groupby('ind')['factor'].transform(lambda x: x.fillna(x.mean()))
            else:
                raise ("因子预处理-缺失值填充：请输入有效的填充方法！")

            """ 去极值： None, 'median', 'pct_shrink'"""
            if (method["extreme"] is None) | (method["extreme"] == ""):
                pass
            elif method["extreme"] == "median": #中位数去极值
                n = 5
                d_m = factor_i_new['factor_filled'].dropna().median() #中位值
                d_mad = (factor_i_new['factor_filled'] - d_m).abs().dropna().median() #偏离中位值的中位值
                factor_i_new = factor_i_new.copy()
                # factor_i_new['factor_filled'][factor_i_new['factor_filled'] > (d_m+n*d_mad)] = d_m+n*d_mad
                factor_i_new.loc[factor_i_new['factor_filled'] > (d_m + n * d_mad),'factor_filled' ]= d_m + n * d_mad
                # factor_i_new['factor_filled'][factor_i_new['factor_filled'] < (d_m-n*d_mad)] = d_m-n*d_mad
                factor_i_new.loc[factor_i_new['factor_filled'] < (d_m - n * d_mad),'factor_filled'] = d_m - n * d_mad
            elif method["extreme"] == "pct_shrink": #分位数缩尾：2.5%和97.5%的数据缩尾
                q1 = 0.025
                q2 = 0.975
                factor_i_q1 = factor_i_new['factor_filled'].quantile(q1)
                factor_i_q2 = factor_i_new['factor_filled'].quantile(q2)
                factor_i_new = factor_i_new.copy()
                # factor_i_new.loc['factor_filled'][factor_i_new['factor_filled'] <= factor_i_q1] = factor_i_q1
                # factor_i_new.loc['factor_filled'][factor_i_new['factor_filled'] >= factor_i_q2] = factor_i_q2
                factor_i_new.loc[factor_i_new['factor_filled'] <= factor_i_q1, 'factor_filled'] = factor_i_q1
                factor_i_new.loc[factor_i_new['factor_filled'] >= factor_i_q2, 'factor_filled'] = factor_i_q2
            else:
                raise ValueError("因子预处理-去极值：请输入有效的去极值方法！")

            """标准化：None, 'zscore', 'norm'"""
            if (method["norm"] is None) | (method["norm"] == ""):
                pass
            elif method["norm"] == 'zscore':
                f_mean = factor_i_new['factor_filled'].dropna().mean()
                f_std = factor_i_new['factor_filled'].dropna().std(ddof=1)
                factor_i_new['factor_filled'] = (factor_i_new['factor_filled'] - f_mean)/f_std
            elif method["norm"] == 'norm':
                if factor_i_new['factor_filled'].notna().sum() <=1:
                    raise ValueError('正态化数据全为nan')
                #分位数
                factor_i_new.loc[factor_i_new['factor_filled'].notnull(),'rank'] = \
                    factor_i_new.loc[factor_i_new['factor_filled'].notnull(),'factor_filled'].rank(pct=True)
                #把等于0和1的处理成剩余数据最小、最大值，否则正态逆函数会产生inf
                factor_i_new = factor_i_new.copy()
                factor_i_new['rank'].loc[factor_i_new['rank']==1] = (factor_i_new['rank'][factor_i_new['rank']!=1].max() + 1) * 0.5
                factor_i_new['rank'].loc[factor_i_new['rank']==0] = (factor_i_new['rank'][factor_i_new['rank'] != 0].min()) * 0.5
                factor_i_new['factor_filled'] = norm.ppf(factor_i_new['rank'], 0, 1)
            else:
                raise ValueError("因子预处理-标准化：请输入有效的准化法！")

            #将处理好的数据更新到factor中
            factor_i_new_all.loc[factor_i_new.index] = factor_i_new['factor_filled'].values
            return factor_i_new_all

    @staticmethod
    def preprocess_onePeriod1(factor_i: pd.Series, stock_tradable_adjdate_i : pd.Series,
                             ind_of_stock_adjdate_i : pd.Series, method: dict) -> pd.Series:
        """
        输入某一日的因子值和index为包含当日因子的日期的行业分类,可交易性。只对有分类可交易的股票做处理
        :param factor_i: 该日因子值，factor_i.name = yyyymmdd
        :param stock_tradable_adjdate :可交易性（只对tradable的做填充）
        :param ind_of_stock_adjdate: 行业分类
        :param method:预处理方法，{'missing': 'ind_avg', 'extreme': 'median', 'norm': 'zscore'}
        :return: 行业均值填充后的因子值
        """

        # assert (([factor_i.name[0]] in ind_of_stock_adjdate.index.values) &
        #         ([factor_i.name[0]] in stock_tradable.index.values)), \
        #     "行业均值填充：因子日期不在行业因子或stock_trabdable矩阵内"

        # 将调仓当日因子值、行业和股票可交易性合并
        # ind_of_stock_adjdate_i = ind_of_stock_adjdate.loc[factor_i.name[0], :]
        # stock_tradable_adjdate_i = stock_tradable_adjdate.loc[factor_i.name[0], :]
        factor_i_new = pd.DataFrame(zip(factor_i, ind_of_stock_adjdate_i, stock_tradable_adjdate_i),
                                    index=ind_of_stock_adjdate.columns.tolist(),
                                    columns=['factor', 'ind', 'stock_tradable'])

        # 剔除不可交易和无行业分类的(只对有行业分类且可交易的做填充)
        factor_i_new = factor_i_new.loc[~(factor_i_new['ind'] * factor_i_new['stock_tradable']).isna(), :]
        factor_i_new = factor_i_new.loc[~(factor_i_new['ind'] * factor_i_new['stock_tradable'] == 0), :]
        factor_i_new['factor_filled'] = factor_i_new['factor'].copy()

        """缺失值：None, 'ind_avg'(行业均值填充)"""
        if (method["missing"] is None) | (method["missing"] == ""):
            pass
        elif method['missing'] == 'ind_avg':
            factor_i_new.loc[:, 'factor_filled'] = factor_i_new.groupby('ind')['factor'].transform(
                lambda x: x.fillna(x.mean()))
        else:
            raise ("因子预处理-缺失值填充：请输入有效的填充方法！")

        """ 去极值： None, 'median', 'pct_shrink'"""
        if (method["extreme"] is None) | (method["extreme"] == ""):
            pass
        elif method["extreme"] == "median":  # 中位数去极值
            n = 5
            d_m = factor_i_new['factor_filled'].dropna().median()  # 中位值
            d_mad = (factor_i_new['factor_filled'] - d_m).abs().dropna().median()  # 偏离中位值的中位值
            factor_i_new['factor_filled'][factor_i_new['factor_filled'] > (d_m + n * d_mad)] = d_m + n * d_mad
            factor_i_new['factor_filled'][factor_i_new['factor_filled'] < (d_m - n * d_mad)] = d_m - n * d_mad
        elif method["extreme"] == "pct_shrink":  # 分位数缩尾：2.5%和97.5%的数据缩尾
            q1 = 0.025
            q2 = 0.975
            factor_i_q1 = factor_i_new['factor_filled'].quantile(q1)
            factor_i_q2 = factor_i_new['factor_filled'].quantile(q2)
            factor_i_new['factor_filled'][factor_i_new['factor_filled'] <= factor_i_q1] = factor_i_q1
            factor_i_new['factor_filled'][factor_i_new['factor_filled'] >= factor_i_q2] = factor_i_q2
        else:
            raise ValueError("因子预处理-去极值：请输入有效的去极值方法！")

        """标准化：None, 'zscore', 'norm'"""
        if (method["norm"] is None) | (method["norm"] == ""):
            pass
        elif method["norm"] == 'zscore':
            f_mean = factor_i_new['factor_filled'].dropna().mean()
            f_std = factor_i_new['factor_filled'].dropna().std()
            factor_i_new['factor_filled'] = (factor_i_new['factor_filled'] - f_mean) / f_std
        elif method["norm"] == 'norm':
            if factor_i_new['factor_filled'].notna().sum() <= 1:
                raise ValueError('正态化数据全为nan')
            factor_i_new['rank'] = factor_i_new['factor_filled'].rank()
            # 归一化
            factor_i_new['rank'] = factor_i_new['rank'] / factor_i_new['rank'].dropna().sum()
            # #把等于1的处理成剩余数据最大值，否则正态逆函数会产生inf
            # factor_i_new['rank'][factor_i_new['rank']==1] = factor_i_new['rank'][factor_i_new['rank']!=1].max()
            #
            factor_i_new['factor_filled'] = norm.ppf(factor_i_new['rank'], 0, 1)
        else:
            raise ValueError("因子预处理-标准化：请输入有效的准化法！")

        # 将处理好的数据更新到factor中
        factor_i_new_all = factor_i.copy()
        factor_i_new_all.loc[factor_i_new.index] = factor_i_new['factor_filled'].values
        return factor_i_new_all

    @staticmethod
    def preprocess_factor(adj_date: pd.DataFrame, factor_data: pd.DataFrame, stock_tradable: pd.DataFrame,
                          method: dict, ind_of_stock: pd.DataFrame) -> pd.DataFrame:
        """
        输入因子，返回预处理之后的因子值
        :param adj_date: 调仓期 pd.DataFrame, yyyymmdd int型
        :param factor_data: 因子原始值, 必须带日期索引
        :param stock_tradable: 可交易的为1，否则为0
        :param method: 预处理方法，{'missing': 'ind_avg', 'extreme': 'median', 'norm': 'zscore'}
        :param'missing': None, 'ind_avg'
        :param'extreme': None, 'median', 'pct_shrink', 'sigma'
        :param'norm': None, 'zscore', 'norm'
        :param ind_of_stock: pd.DataFrame
        :return: factor_sd：预处理好的因子（只有调仓日的）
        """

        assert date_utils.valid_date(adj_date), '因子预处理：请输入正确的日期格式 ! ：pd.DataFrame, yyyymmdd int'
        assert date_utils.valid_date(pd.DataFrame(factor_data.index.values)), '因子预处理：预处理时因子必须带日期索引'
        assert (factor_data.shape == ind_of_stock.shape), '因子预处理：因子和行业因子形状不匹配'

        #只处理tradable的因子，并添加索引
        tradable_factor = pd.DataFrame(np.array(stock_tradable) * np.array(factor_data),
                                       index=[factor_data.index.values], columns=[factor_data.columns.values])
        #调仓日的所有因子值
        tradable_factor_adjdate = tradable_factor.loc[adj_date.iloc[:,0].values,:]
        stock_tradable_adjdate = stock_tradable.loc[adj_date.iloc[:,0].values,:]
        ind_of_stock_adjdate = ind_of_stock.loc[adj_date.iloc[:, 0].values, :]
        #inf的数据用nan替换
        tradable_factor_adjdate=tradable_factor_adjdate.copy()
        tradable_factor_adjdate[np.isinf(tradable_factor_adjdate) |
                                np.isinf(tradable_factor_adjdate*-1)] = np.nan
        #只处理tradable的股票
        tradable_factor_adjdate_filled = tradable_factor_adjdate.apply(
                                             lambda x: Factor.preprocess_onePeriod(x, stock_tradable_adjdate,
                                             ind_of_stock_adjdate, method), axis=1)
        # temp = tradable_factor_adjdate_filled.copy()
        tradable_factor_adjdate_filled.index = tradable_factor_adjdate_filled.index.get_level_values(0).values
        # tradable_factor_adjdate_filled.columns =tradable_factor_adjdate_filled.columns.levels[0].values
        tradable_factor_adjdate_filled.columns = tradable_factor_adjdate_filled.columns.get_level_values(0).values


        return tradable_factor_adjdate_filled

    @staticmethod
    def neutralize(factor_i: pd.DataFrame, ifind: bool, ind_of_stock:pd.DataFrame, ifrisk: bool, risk_data: list) ->pd.DataFrame:
        """
        对因子进行风险中性处理
        :param factor_i: 因子值，索引为交易日
        :param ifind: 是否行业中性
        :param ind_of_stock: 行业数据
        :param ifrisk: 是否风险中性
        :param risk_data: 各个风险因子值(pd.DataFrame)，存在list里
        :return: 处理好的因子
        """
        factor_i = factor_i.copy()
        ind_of_stock = ind_of_stock.copy()
        ind_of_stock = ind_of_stock.replace(np.nan, 0)
        # 检查数据(因子的索引日期必须在行业数据内)
        # assert ([factor_i.name[0]]  in ind_of_stock.index.values) , \
        #         ("因子中性化：因子日期不在行业因子矩阵内", factor_i.name[0])
        risk_data_num = len(risk_data)
        # if risk_data_num>0:
        #     for risk_data_i in risk_data:
        #         assert ([factor_i.name[0]] in list(risk_data_i.index.values)), \
        #             ("因子中性化：因子日期不在行业因子矩阵内", factor_i.name[0])

        #提取调仓日的行业和风险因子
        date_factor_i = factor_i.index.values
        # ind_of_stock_i = ind_of_stock.loc[date_factor_i,:]
        # for i in range(0, risk_data_num):
        #     risk_data[i] = risk_data[i].loc[date_factor_i,:]
        factor_neut = factor_i.copy()
        factor_neut = factor_neut * np.nan

        #中性化
        if ifind & ifrisk:
            print("行业和风险中性...")
            for date_j in date_factor_i:
                # print(date_j)
                if factor_i.loc[date_j,:].notnull().sum()>0:
                    ind_of_stock_j = ind_of_stock.loc[date_j, :]
                    dum_ind = pd.get_dummies(ind_of_stock_j)
                    dum_ind = dum_ind.loc[:,dum_ind.sum()>0]
                    X = dum_ind.copy()
                    for i in range(0, risk_data_num):
                        # X = pd.concat([X, risk_data[i].loc[date_j, :]], axis=1, sort=False)
                        X = pd.merge(X, risk_data[i].loc[date_j, :], left_index=True, right_index=True,suffixes=('', '_'+str(i)))
                    #剔除nan和inf
                    lm_data = pd.merge(factor_i.loc[date_j,:].to_frame(), X, left_index=True, right_index=True,
                                       suffixes=('_y', '_'+str(i))).dropna()
                    model = sm.OLS(lm_data.iloc[:,0].values, lm_data.iloc[:,1:].values)
                    resid =  model.fit().resid
                    factor_neut.loc[date_j, lm_data.index.values] = resid
                else:
                    #本期因子值全为nan
                    pass
        elif ifind:
            print("行业中性:")
            for date_j in date_factor_i:
                # print(date_j)
                if factor_i.loc[date_j, :].notnull().sum() > 0:
                    ind_of_stock_j = ind_of_stock.loc[date_j, :]
                    dum_ind = pd.get_dummies(ind_of_stock_j)
                    dum_ind = dum_ind.loc[:, dum_ind.sum() > 0]
                    X = dum_ind.copy()
                    # 剔除nan和inf
                    lm_data = pd.merge(factor_i.loc[date_j, :].to_frame(), X, left_index=True, right_index=True,
                                       suffixes=('_y', '_x')).dropna()
                    model = sm.OLS(lm_data.iloc[:, 0].values, lm_data.iloc[:, 1:].values)
                    resid = model.fit().resid
                    factor_neut.loc[date_j, lm_data.index.values] = resid
                else:
                    # 本期因子值全为nan
                    pass
        elif ifrisk:
            print("风险中性:")
            for date_j in date_factor_i:
                # print(date_j)
                if factor_i.loc[date_j,:].notnull().sum()>0:
                    X = pd.DataFrame()
                    for i in range(0, risk_data_num):
                        X = pd.concat([X, risk_data[i].loc[date_j, :]], axis=1, sort=False)
                        # X = pd.merge(X, risk_data[i].loc[date_j, :], left_index=True, right_index=True)
                    X = sm.add_constant(X)
                    #剔除nan和inf
                    lm_data = pd.merge(factor_i.loc[date_j,:].to_frame(), X, left_index=True, right_index=True,
                                       suffixes=('_y', '_x')).dropna()

                    model = sm.OLS(lm_data.iloc[:,0].values, lm_data.iloc[:,1:].values)
                    resid =  model.fit().resid
                    factor_neut.loc[date_j, lm_data.index.values] = resid
                else:
                    #本期因子值全为nan
                    pass
        return factor_neut

    # @staticmethod
    # def get_hedge_index(hedge, industry=''):
    #     """
    #     根据输入返回对冲基准的净值曲线
    #     :param hedge:
    #     :param industry:
    #     :return:
    #     """
    #
    #     index_range = ['SZ50', 'HS300', 'ZZ500', 'equal', 'custom']
    #     index_mapping = {'HS300': '000300.SH',
    #                      'ZZ500': '000905.SH',
    #                      'SZ50': '000016.SH'}
    #
    #     index_cp = cls.add_index(cls.get_apidata(('index_daily.h5', 'index_cp')), type='index')
    #
    #
    #     if not hedge in index_range:
    #         raise ValueError("对冲基准不在范围内，范围：{}".format(index_range))
    #     else:
    #         if hedge in ['SZ50', 'HS300', 'ZZ500']:
    #               benchmark = index_cp.loc[:, index_mapping[hedge]]
    #         else:
    #             pass






def get_factor_std(factor_setting, prepro_setting, **kwargs):
    print("导入数据...")
    stklist, trade_dt = Factor.get_axis(type='stock')
    cp = Factor.add_index(Factor.get_apidata(('stk_daily.h5', 'cp')))
    ind_of_stock = Factor.add_index(Factor.get_apidata(('stk_daily.h5', 'id_citic1')))
    factor_data=[]
    #导入因子值"""
    if 'factor_data' in kwargs.keys():
        #已经自定义导入dataframe的情况"""
        # print(kwargs['factor_data'])
        print('注意：本次分析的是自定义因子')
        factor_data=kwargs['factor_data']
    else:
        factor_data = Factor.get_customdata((factor_setting['factor_dir'],factor_setting['name']))

    #检查因子shape是否为标准矩阵, 没有索引的话默认按照标准矩阵索引添加,否则按照标准索引reindex
    # if Factor.valid_shape(factor_data):
    #     factor_data = Factor.add_index(factor_data)
    # else:
    #     raise ValueError('因子形状不一致！')

    #先检查是否有索引，有的话按照标准矩阵reindex, 没有的话先检查形状，一致的话按照标准索引reindex
    if factor_data.columns.dtype == 'int64':
        if Factor.valid_shape(factor_data):
            factor_data = Factor.add_index(factor_data)
        else:
            raise ValueError('因子形状不一致！')
    else:
        #已经有索引的情况下，reindex成标准矩阵
        factor_data = factor_data.reindex(index=trade_dt.iloc[:,0], columns=stklist.iloc[:,0])






    print("选定样本集...")
    # 能交易的为1,否则为nan
    if "sample_index_customdir" in prepro_setting.keys():
        stock_sample = Factor.select_range(prepro_setting["sample_index"],
                                       prepro_setting["sample_ind"], prepro_setting["sample_index_customdir"])
    else:
        stock_sample = Factor.select_range(prepro_setting["sample_index"],
                                           prepro_setting["sample_ind"])
    print("选定可交易股票...")
    #能交易的为1,否则为nan
    stock_tradable_0 = Factor.valid_tradable(prepro_setting['tradable']['ifnoST'], prepro_setting['tradable']['ifnoSusp'],
                                           prepro_setting['tradable']['ifnoUpDownLimit'], prepro_setting['tradable']['ifnoNewStock'],
                                             if_trace = prepro_setting['tradable']['if_trace'] )


    stock_tradable = Factor.add_index(stock_tradable_0 *stock_sample)
    # stock_tradable = Factor.add_index(stock_tradable)
    print("选定选股日...")
    adj_date = date_utils.get_adjustdate(trade_dt, prepro_setting['adjdate_beg'], prepro_setting['adjdate_end'], prepro_setting['adj_mode'])

    print("因子预处理...")
    if (prepro_setting['preprocess_ifrisk'] == 1) & (
            ((prepro_setting['preprocess_prepromethod']['norm'] == '') |
             (prepro_setting['preprocess_prepromethod']['norm'] is None))):
        raise ValueError("进行风险中性化需要设定因子预处理方式：preprocess_prepromethod")
    #风险因子数据导入"""
    if prepro_setting['preprocess_ifrisk'] == 1:
        print("风险因子数据导入...")
        riskfactor_name = []
        for i, dir_i in enumerate(prepro_setting['preprocess_risk_factor']):
            # print(str(i))
            risk_name_i = dir_i[1].split('.')[0]
            if dir_i[0] == 'risk_factor.h5': #提取风险因子
                exec("{}=Factor.get_apidata(dir_i)".format(dir_i[1]))
            else: #提取alpha因子或自定义因子
                exec("{}=Factor.get_customdata(dir_i)".format(risk_name_i))
            #检查风险因子形状，加上索引
            if eval("Factor.valid_shape({})".format(risk_name_i)):
                exec("{} = Factor.add_index({})".format(risk_name_i,risk_name_i))

            else:
                raise ValueError('风险因子形状不一致:' + risk_name_i)
            riskfactor_name.append(risk_name_i)

    #因子数据预处理,得到调仓日的因子值"""
    factor_data_std = Factor.preprocess_factor(adj_date, factor_data, stock_tradable, prepro_setting['preprocess_prepromethod'], ind_of_stock)

    """因子中性化（风险因子中性、行业中性）"""
    if (prepro_setting["preprocess_ifind"] == 1) | (prepro_setting["preprocess_ifrisk"] == 1):
        print("因子中性化处理...")
        #先将风险因子预处理"""
        risk_data=[] #存储风险因子数据
        if prepro_setting["preprocess_ifrisk"] == 1:
            for risk_name_i in riskfactor_name:
                exec("{} = Factor.preprocess_factor(adj_date, {}, stock_tradable, prepro_setting[" \
                         "\'preprocess_prepromethod\'],ind_of_stock)".format(risk_name_i + '_std', risk_name_i))
                exec("risk_data.append({})".format(risk_name_i + '_std'))
        #中性化
        factor_data_std = Factor.neutralize(factor_data_std, prepro_setting["preprocess_ifind"], ind_of_stock,
                                             prepro_setting["preprocess_ifrisk"], risk_data)
        print("因子数据处理完毕!")
    return factor_data_std

def get_multifactor(factor_setting, prepro_setting, **kwargs):
    pass












if __name__ == '__main__':
    #
    # """
    # 测试Factor()
    # """

    # Factor.select_range('HS300', 'all')
    # id_300 = Factor.get_apidata(('stk_daily.h5','id_300'))
    # bp = Factor.get_apidata(('alpha/ep.h5', 'ep'))
    # stock_tradable_0 = Factor.valid_tradable(0, 0, 0, 1, if_trace={'suspend': (20, 1), 'st':(20,1)})
    stklist, trade_dt = Factor.get_axis()

    print('...')

    #初始化
    temp0 = Factor()

    # 更改api路径
    temp0 = Factor(('risk_factor.h5', 'size'), api="E:/Data/temp_data/testdata/")
    temp = temp0.get_apidata(('risk_factor.h5', 'Size'))
    temp0 = Factor.get_apidata(('stk_daily.h5','cp'))

    risk_factor_names = Factor.get_apikeys(apidata_dir = 'risk_factor.h5')
    for i in risk_factor_names:
        print(i)
        temp = Factor.add_index(Factor.get_apidata(('risk_factor.h5', i[1:])))
        temp.to_csv(f"{i[1:]}.csv")


    """
    #提取api下的文件
    #stk_daily.h5下的因子
    temp= Factor.get_apidata(('stk_daily.h5','cp'))
    temp= Factor.get_apidata(('stk_daily.h5','stklist'))
    temp = Factor.get_apidata(('stk_daily.h5', 'trade_dt'))
    #riskfactor.h5下的因子
    temp = Factor.add_index(Factor.get_apidata(('risk_factor.h5', 'Momentum')))
    # alpha因子文件夹下 .h5下的的alpha因子
    bp = Factor.get_apidata(('alpha/ep.h5', 'ep')) #调用类实例方法
    #index_daily.h5下的因子
    temp = Factor.get_apidata(('index_daily.h5', 'indexlist'))


    #提取自定义路径下的csv文件
    temp = Factor.get_customdata(('E:/Data/temp_data/testdata/', 'bp.csv'))
    #提取自定义路径下的.npy文件
    temp = Factor.get_customdata(('E:/Data/temp_data/testdata/', 'bp.npy'))
    # 提取自定义路径下的.h5 .csv文件
    temp = Factor.get_customdata(('E:/Data/temp_data/testdata/test_h5/alpha/bp.h5', 'bp.csv'))

    # 提取stklist和trade_dt
    stklist, trade_dt = Factor.get_axis(type='stock')

    #给因子加标签
    Factor.add_index(bp)


    # np.save('E:/Data/temp_data/testdata/bp.npy', bp)


"""