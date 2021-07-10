# # coding=utf-8
#
# import numpy as np
# import pandas as pd
#
# from QuantNodes.database_node.clickhouse_node import ClickHouseDBPool
#
#
# def get_node(p={'host': '***REMOVED***', 'port': 8123, 'user': 'default', 'password': '***REMOVED***', 'db': 'default'}):
#     r2 = ClickHouseDBPool(settings=p)
#     return r2
#
#
# #
# # def load_sp500(path='/Users/sn0wfree/PycharmProjects/nodes/nodes/test/sp500.xlsx'):
# #     sp500 = pd.read_excel(path).dropna()
# #     sp500['date'] = sp500['date'].astype('int')
# #     return sp500
#
#
# def load_db_obj(
#         p={'host': '***REMOVED***', 'port': 8123, 'user': 'default', 'password': '***REMOVED***', 'db': 'default'}):
#     r2 = get_node(p=p).default.ontime
#     return r2
#
#
# def load_daily_return_data(
#         p={'host': '***REMOVED***', 'port': 8123, 'user': 'default', 'password': '***REMOVED***', 'db': 'default'},
#         sql='select * from paper.sp500_with_daily_return'):
#     obj = load_db_obj(p=p)
#
#     df = obj.query(sql)
#     df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
#
#     return df  # .set_index('date')
#
#
# class BuyAndHold(object):
#
#     @staticmethod
#     def _prepare_data_(df=None, e_list=[1, 2, 5, 10, 20], arg_list=['max', 'min']):
#         df = load_daily_return_data() if df is None else df
#         for e in e_list:
#             for arg in arg_list:
#                 name = f'rolling_{e}_{arg}'
#                 ext_name = f'rolling_{e}_{arg}_diff'
#                 ext_pct_name = f'rolling_{e}_{arg}_diff_pct'
#                 ext_abs_name = f'rolling_{e}_{arg}_diff_abs'
#                 df[name] = df['sp'].rolling(e).apply(lambda x: getattr(np, arg)(x))
#                 df[ext_name] = df['sp'] - df[name]
#                 df[ext_abs_name] = np.abs(df[ext_name])
#                 df[ext_pct_name] = (df['sp'] - df[name]) / df[name]
#         return df
#
#
# class FilterRules(BuyAndHold):
#     """
#
#
#     A filter rule strategy is specified as follows.
#     If the daily closing price (in U.S. dollars) of a foreign currency goes up by x% or more from its most recent low,
#     then the speculator borrows the dollar and uses the proceeds to buy the foreign currency.
#
#     When the closing price of the foreign currency drops by at least y% from a subsequent high,
#     the speculator short sells the foreign currency and uses the proceeds to buy the dollar.
#
#     We define the subsequent high as the highest price over the e most recent days and the subsequent low as the lowest price over the e most recent days.
#     We also consider the case where a given long or short position is held for c days during which time all other signals are ingored
#
#
#
#
#     Basic Filter Rules:
#     When the daily closing price of an asset moves up by over x% from its most recent low, the rule generates a "buy" forecast.
#     When the daily closing price moves down by at least y% from a recent high, the rule generates a "sell" forecast.
#     Otherwise, the forecast is "neutral".
#
#
#
#
#     x: increase in the log return required to generate a "buy"
#     signal x = 0.0005, 0.001, 0.005, 0.01, 0.05, 0.10 (6 values);
#
#     y: decrease in the log return required to generate a "sell"
#     signal y = 0.0005, 0.001, 0.005, 0.01, 0.05 (5 values)
#
#     e: the number of the most recent days needed to define a low (high) based on which the filters are applied to generate a "buy" ("sell") signal
#     e = 1, 2, 5, 10, 20 (5 values)
#
#
#     c: number of days a position is held during which all other signals are ignored c = 1, 5, 10, 25 (4 values)
#
#     Note that y must be less than x, hence there are 15 (x,y) combinations
#     Number of rules in FR class= x×c+x×e+x×y+((x,y) combinations))= 24+30+15 = 69
#     """
#
#
#
#
# class MovingAverageRules(object):
#     """
#
#
#
#     The moving average of a currency price for a given day is computed as the simple average of prices over the previous n days, including the current day.
#     Under a moving average rule, when the short moving average of a foreign currency price is above the long moving average by an amount larger than the band with b%,
#     the speculator borrows the dollar to buy the foreign currency.
#     Similarly, when the short moving average is below the long moving average by b%,
#     the speculator short sells the FX to buy the dollar.
#
#     In addition to this fixed percentage band filter,
#     we also implement the moving average rules with a time delay filter, which requires that the long or short signals remain valid for d days before he takes any action.
#     As in the filter rule case, we also consider the case where a given long or short position is held for c days during which time all other signals are ignored
#
#
#     Basic Moving Average Rule:
#     First, to calculate the (equally-weighted) moving average of an asset prices for a given day t over the n days.
#     When current price is above moving average, it generates a "buy" forecast;
#     when current price is below moving average, it generates a "sell" forecast;
#     otherwise, it generates a "neutral" forecast.
#
#
#     n: number of days in a moving average
#     n = 2, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250 (11 values)
#
#
#     m: number of fast-slow combinations of n $m=c_{11}^{2}$
#
#     b: fixed band multiplicative value
#     b = 0, 0.0005, 0.001, 0.005, 0.01, 0.05 (6 values)
#
#     c: number of days a position is held, ignoring all other signals during that time c = 5, 10, 25 (3 values)
#
#     d: number of days for the time delay filter d = 2, 3, 4, 5 (4 values)
#
#     Number of rules in MA class: = b(n+m)+d(n+m)+c(n+m)= 396+264+198 = 858
#
#     """
#
#     @staticmethod
#     def df_with_ma(n, df=None):
#         df = load_daily_return_data() if df is None else df
#         arg = 'mean'
#         name = f'ma_{n}'
#         df[name] = df['sp'].rolling(n).apply(lambda x: getattr(np, arg)(x))
#         return df
#
#     @classmethod
#     def _parpare_ma_(cls, n_list=[2, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250], df=None):
#         df = load_daily_return_data() if df is None else df
#         for n in n_list:
#             df = cls.df_with_ma(n, df=df)
#         return df
#
#     @classmethod
#     def ma(cls, n, b, c, d, df=None):
#         df = load_daily_return_data() if df is None else df
#         df = cls.df_with_ma(df=df)
#         pass
#
#     pass
# class  TradingRangeBreak(BuyAndHold):
#     """
#
#     Basic Support and Resistance Rule:
#     Under a trading range break rule, when the price of an asset moves above the maximum price (resistance level) over the previous n days by b%, it generates a "buy" forecast.
#     When the price falls below the minimum price over the previous n days by b%, it generates a "sell" forecast.
#     Otherwise, it generates a "neutral" forecast.
#
#
#     n: number of days in the support and resistance range; n = 5, 10, 15, 20, 25, 50, 100 (7 values);
#     e: used for an alternative definition of extrema where a low (high) can be defined as the most recent closing price that is less (greater) than the n previous closing prices;
#     e = 2, 3, 4, 5, 10, 25, 50 (7 values);
#     b: fixed band multiplicative value;
#     b = 0.0005, 0.001, 0.005, 0.01, 0.05 (5 values);
#     c: number of days a position is held, ignoring all other signals during that time c = 1, 5, 10, 25 (4 values);
#     d: number of days for the time delay filter; d = 2, 3, 4, 5 (4 values);
#
#     """
#     @classmethod
#     def sr_single(cls, x, y, e, df=None, return_all_cols=True):
#         if df is None:
#             df = cls._prepare_data_()
#         arg = 'max'  # > y decrease sell
#         arg = 'min'  # > x increase buy
#         # name = f'rolling_{e}_{arg}'
#         # ext_name = f'rolling_{e}_{arg}_diff'
#         x_related = f'rolling_{e}_min_diff_pct'
#         y_related = f'rolling_{e}_max_diff_pct'
#         print(x)
#         # print(int(round(x * 100, 1)))
#         x_col = 'x_signal_{x}_pct_{e}'.format(x=str(x).replace('.', '_'), e=e)
#         y_col = 'y_signal_{y}_pct_{e}'.format(y=str(y).replace('.', '_'), e=e)
#         df[x_col] = (df[x_related] > x) * 1
#         df[y_col] = (df[y_related] < y * -1) * 1
#         if return_all_cols:
#             return df
#         else:
#             cols = ['date', 'sp', x_col, y_col, x_related, y_related]
#             return df[cols]
#
#         pass
#
#     @classmethod
#     def sr(cls, x_list=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.10],
#            y_list=[0.0005, 0.001, 0.005, 0.01, 0.05],
#            e_list=[1, 2, 5, 10, 20]):
#         df = cls._prepare_data_(e_list=e_list)
#         from itertools import product
#         sco = [(x, y) for x, y in product(x_list, y_list) if x > y]
#
#         for (x, y), e in product(sco, e_list):
#             df = cls.fr_single(x, y, e, df=df)
#
#         return df
#
#
# if __name__ == '__main__':
#     x = 0.10
#     y = 0.05
#     e = 20
#     df = FilterRules.fr()
#
#     # sp500 = load_sp500()
#     # SP500['paper.SP500'] = sp500
#
#     print(1)
#     pass
