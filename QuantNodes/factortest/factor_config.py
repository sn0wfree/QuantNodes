# coding = utf-8
"""
Author: DaisyZhou

date: 2019/11/1 17:33
"""

"""
测试配置信息
"""

#输入待测试因子
factor_setting = {
        "name": 'ep',
        "factor_dir": './testdata/test_h5_new/alpha/ep.h5',
        # "factor_dir": 'E:/Data/temp_data/testdata/test_h5_new/alpha/ep.h5',
        # "name": 'mom21.npy',
        # "factor_dir": 'E:/Data/temp_data/testdata/',
        # "name": 'amt.csv',
        # "factor_dir": 'E:/Data/temp_data/testdata/'
        }

"""
预处理方式：
调仓日
样本池：指数、行业
预处理方法:缺失值补充、去极值、标准化、行业、风险中性等
"""
prepro_setting = {
        "adjdate_beg": 20170801,  #数字型
        "adjdate_end": 20171231,  #数字型
        "adj_mode": ('M', 'end'),
                                # ('M', 'end'), ('M', 'begin')
                                # ('D', int),
                                # ('W', 'end'), ('W', 'begin')
                                # ('custom', adj_date_arg)#自定义日期,需要自己输入调仓日pd.DataFrame或者pd.Series类型

        "sample_index": 'all',  #可选'all'、'HS300'、'ZZ500'、'ZZ800'、'SZ50'、'custom'(需要填customdir)
                                # "sample_index": 'custom',
                                # "sample_index_customdir":('E:/Data/temp_data/testdata/', 'id_300.csv'),

        "sample_ind": 'all',    #('id_citic1', '房地产'),

        "tradable": {"ifnoST": 1, "ifnoSusp": 1, "ifnoUpDownLimit": 0, "ifnoNewStock": 360, "if_trace": None},

        "preprocess_prepromethod": {'missing': '', 'extreme': '', 'norm': ''},#填补缺失值、去极值、标准化
                                  #'missing': None, 'ind_avg'
                                  #'extreme': None, 'median', 'pct_shrink',
                                  #'norm': None, 'zscore', 'norm'
        "preprocess_ifind": 0,
        "preprocess_ifrisk": 0,   #做风险中性预处理方法preprocess_prepromethod不能为''
        "preprocess_risk_factor": [('risk_factor.h5','Size'), #支持：风险因子库、alpha因子库、自定义因子
                                   ('risk_factor.h5','Value'),]
                                   # ('E:/Data/temp_data/testdata/test_h5_new/alpha/ep.h5', 'ep')]
                                   # ('E:/Data/temp_data/testdata/', 'bp.npy')]
                                                                              #自定义因子支持csv, npy格式（alpha因子库的需要完整路径:
                                                                              #风险因子库:（'riskfactor.h5', 因子名）
                                                                              #alpha因子库：('E:/Data/temp_data/testdata/test_h5_new/alpha/ep.h5', 'ep')
                                                                              #自定义因子：('E:/Data/temp_data/testdata/', 'bp.npy')
}

"""
执行选项：
有效性分析：IC, IR, 回归检验, 分组检验等
分样本有效性分析
市值行业分组打分
与风格因子相关性
"""
execute_setting = {
        "EffectAnalysis_ifrun": 1,
        "EffectAnalysis_fac_ori": 1,    #1:越大越好, -1越小越好
        "EffectAnalysis_group": 5,
        "EffectAnalysis_floor": 'group', #当期股票数少于组数该期就不选, 'last', #当期股票数少于组数该期就按照上一期的分组
        "EffectAnalysis_hedge": 'equal', #'SZ50','HS300', 'ZZ500', 'equal', 'custom',其他的待因子库完善后补充
                                         # "EffectAnalysis_hedge": 'custom', #当对冲指数为自定义时，需要输入文件路径,支持csv,npy
                                         # "EffectAnalysis_hedgedir" : ('E:/Data/temp_data/testdata/', 'index_cp_000300.npy'),

        "EffectBySample_ifrun": 0,         # 是否逐个样本运行
        "EffectBySample_Sample": 'by_ind', # 分行业进行分析（逐个行业进行一次有效性分析，运行速度慢，慎用）
                                           # "EffectBySample_Sample": 'custom', #自定义样本进行分析, 按照样本所有可能的取值进行分类
                                           # "EffectBySample_Sampledir": ('E:/Data/temp_data/testdata/test_h5_new/stk_daily.h5', 'id_citic1'), #('E:/Data/temp_data/testdata/', 'bp.npy'),('E:/Data/temp_data/testdata/', 'id_300.csv')
        "EffectBySample_fac_ori": 1,
        "EffectBySample_group": 5,
        "EffectBySample_floor": 'group',
        "EffectBySample_hedge": 'equal',
                                        # "EffectBySample_hedge": 'custom',
                                        # "EffectBySample_hedgedir": ('E:/Data/temp_data/testdata/', 'index_cp_000300.npy'),

        "EffectBy_SizeInd_ifrun": 1,    #市值行业分层打分（执行此项不需要进行市值行业中性）

        "RiskAnalysis_ifrun": 1,        #是否运行风险分析
        "RiskAnalysis_factors": 'all',  # 默认所有的风险因子,也可以自定义一个文件夹，计算的是该文件夹下的因子与当前因子的相关性

        "save_filedir": './output/' #输出excel的地址
}
