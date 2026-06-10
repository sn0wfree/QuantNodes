# coding: utf-8
"""常量定义 / Constants"""

# 指数映射
INDEX_MAPPING = {
    'HS300': ('stk_daily.h5', 'id_300'),
    'ZZ500': ('stk_daily.h5', 'id_500'),
    'SZ50': ('stk_daily.h5', 'id_50'),
}

# 指数收盘价映射
INDEX_CP_MAPPING = {
    'HS300': '000300.SH',
    'ZZ500': '000905.SH',
    'SZ50': '000016.SH',
}

# 中信行业映射
INDUSTRY_MAPPING = {
    'id_citic1A': 'ind_name_CITIC_1A',
    'id_citic1': 'ind_name_CITIC_1',
}

# 年化天数
ANNUAL_DAYS = 250
