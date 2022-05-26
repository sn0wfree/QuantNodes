# coding=utf-8
import pandas as pd
import datetime
import numpy as np
from scipy import stats
import statsmodels.api as sm


def detect_and_transform_dt_format(alist):
    first = alist[0]
    if isinstance(first, (int, np.int64, np.int32)):
        return pd.to_datetime(tuple(map(str, alist)), format='%Y%m%d')
    elif isinstance(first, str):
        if first.isnumeric():
            return pd.to_datetime(alist, format='%Y%m%d')
        else:
            return pd.to_datetime(alist)
    elif isinstance(first, datetime.datetime):
        return alist
    else:
        raise TypeError('unknown cik_dts type! only accept str,int32,int64 or datetime')


class STKPrice2Ret(object):
    @staticmethod
    def _same_point_trade(stk_price_matrix, periods: int, cik_dt_list=None, cik_id_list=None):
        if cik_dt_list is None:
            if cik_id_list is None:
                return stk_price_matrix.pct_change(periods=periods)
            else:
                return stk_price_matrix.reindex(columns=cik_id_list).pct_change(periods=periods)
        else:
            if cik_id_list is None:
                return stk_price_matrix.reindex(index=cik_dt_list).pct_change(periods=periods)
            else:
                return stk_price_matrix.reindex(index=cik_dt_list, columns=cik_id_list).pct_change(periods=periods)

    @staticmethod
    def _diff_point_trade(stk_price1_matrix, stk_price2_matrix, periods: int, cik_dt_list=None, cik_id_list=None):
        if cik_dt_list is None:
            if cik_id_list is None:
                return stk_price1_matrix.shift(-1 * periods) / stk_price2_matrix - 1

            else:
                return stk_price1_matrix.reindex(columns=cik_id_list).shift(-1 * periods) / stk_price2_matrix.reindex(
                    columns=cik_id_list) - 1

        else:
            if cik_id_list is None:
                return stk_price1_matrix.reindex(index=cik_dt_list).shift(-1 * periods) / stk_price2_matrix.reindex(
                    index=cik_dt_list) - 1

            else:
                stk_price1_matrix.reindex(index=cik_dt_list,
                                          columns=cik_id_list).shift(-1 * periods) / stk_price2_matrix.reindex(
                    index=cik_dt_list, columns=cik_id_list) - 1

    @classmethod
    def close_2_close(cls, stk_close_price_matrix, periods, cik_dt_list=None, cik_id_list=None):
        return cls._same_point_trade(stk_close_price_matrix, periods, cik_dt_list=cik_dt_list, cik_id_list=cik_id_list)

    @classmethod
    def open_2_open(cls, stk_open_price_matrix, periods, cik_dt_list=None, cik_id_list=None):
        return cls._same_point_trade(stk_open_price_matrix, periods, cik_dt_list=cik_dt_list, cik_id_list=cik_id_list)

    @classmethod
    def open_2_close(cls, stk_open_price_matrix, stk_close_price_matrix, periods, cik_dt_list=None, cik_id_list=None):
        return cls._diff_point_trade(stk_close_price_matrix, stk_open_price_matrix, periods, cik_dt_list=cik_dt_list,
                                     cik_id_list=cik_id_list)

    @classmethod
    def close_2_open(cls, stk_open_price_matrix, stk_close_price_matrix, periods, cik_dt_list=None, cik_id_list=None):
        return cls._diff_point_trade(stk_open_price_matrix, stk_close_price_matrix, periods, cik_dt_list=cik_dt_list,
                                     cik_id_list=cik_id_list)


class FactorTestConfig(object):
    pass


class FactorICTools(object):
    @staticmethod
    def ic_summary(ic_series, mu=0, alternative='two-sided'):
        upbound_count = np.nansum(ic_series >= mu)
        avg = ic_series.mean()
        std = ic_series.std(ddof=1)

        icir = avg / std
        nonan_count = ic_series.notnull().sum()

        c = stats.ttest_1samp(ic_series, mu, alternative=alternative)
        # t = (avg - mu) / (std / np.sqrt(nonan_count))
        t_value = c.statistic
        p_value = c.pvalue
        return pd.Series([avg, std, icir, t_value, p_value, upbound_count, nonan_count],
                         index=['mean', 'std', 'ir', 'tval', 'pval', 'upbound_count', 'nonan_count'])

    @classmethod
    def cal_ic_and_eval(cls, factor_matrix: pd.DataFrame, stk_cp_matrix: pd.DataFrame, cik_dt_list, cik_id_list,
                        shift=1, limit=10,
                        corr_method='pearson', rank_corr_method='spearman',
                        alternative='two-sided', mu=0
                        ):
        """

        :param factor_matrix:
        :param stk_cp_matrix:
        :param cik_dt_list:
        :param cik_id_list:
        :param shift:
        :param limit:
        :param ic_corr_method:
        :param rank_ic_corr_method:
        :param autocorr_method:
        :param alternative:
        :param mu:
        :return:
        """
        print('calculating ic...')
        # create ic series
        res_iter = cls.cal_ic_raw_yield(factor_matrix, stk_cp_matrix, cik_dt_list, cik_id_list, shift=shift,
                                        corr_method=corr_method, rank_corr_method=rank_corr_method, )
        res = pd.DataFrame(res_iter,
                           columns=['dt', 'next_dt', 'ic', 'rank_ic', 'autocorr', 'rank_autocorr', 'shift',
                                    'nonan_count'])
        limit_mask = res['nonan_count'] <= limit  # remove nonan count less {limit} data
        if limit_mask.sum(axis=0) == 0:
            pass
        else:
            res.loc[limit_mask, ['ic', 'rank_ic', 'autocorr', 'rank_autocorr']] = [np.nan, np.nan, np.nan]

        # eval
        ic_summary = cls.ic_summary(res['ic'], mu=mu, alternative=alternative)
        rank_ic_summary = cls.ic_summary(res['rank_ic'], mu=mu, alternative=alternative)
        autocorr_summary = cls.ic_summary(res['autocorr'], mu=mu, alternative=alternative)
        rank_autocorr_summary = cls.ic_summary(res['rank_autocorr'], mu=mu, alternative=alternative)

        c2 = pd.concat([ic_summary, rank_ic_summary, autocorr_summary, rank_autocorr_summary], axis=1)
        c2.columns = ['ic', 'rank_ci', 'autocorr', 'rank_autocorr']
        c2['shift'] = shift
        return dict(ic_result=res, ic_summary=c2)

    @staticmethod
    def cal_ic_raw_yield(factor_matrix: pd.DataFrame, stk_cp_matrix: pd.DataFrame, cik_dt_list, cik_id_list, shift=1,
                         corr_method='pearson', rank_corr_method='spearman',
                         ):
        """
        Information Coefficient
        calculate factor ic, rank ic and others

        use current stk ret and last day factor value

        :param factor_matrix:
        :param stk_cp_matrix:
        :param cik_dt_list:
        :param cik_id_list:
        :param shift:
        :param corr_method:
        :param rank_corr_method:
        :return:
        """
        # print('calculating ic...')
        cik_dt_list = sorted(detect_and_transform_dt_format(cik_dt_list))
        # cp 2 ret
        stk_ret_matrix_shifted_reindex = STKPrice2Ret.close_2_close(stk_cp_matrix, shift, cik_dt_list=cik_dt_list,
                                                                    cik_id_list=cik_id_list)

        # shift factor value to cal corr
        factor_matrix_reindex = factor_matrix.reindex(index=cik_dt_list, columns=cik_id_list).shift(shift)

        # cal current factor rank and next {shift} period corr

        for dt, next_dt in zip(cik_dt_list[:-1 * shift], cik_dt_list[shift:]):
            ret_i = stk_ret_matrix_shifted_reindex.loc[dt, :]
            factor_i = factor_matrix_reindex.loc[dt, :]
            ic = factor_i.corr(ret_i, method=corr_method)
            rank_ic = factor_i.corr(ret_i, method=rank_corr_method)
            autocorr = factor_i.corr(factor_matrix_reindex.loc[next_dt, :], method=corr_method)
            rank_autocorr = factor_i.corr(factor_matrix_reindex.loc[next_dt, :], method=rank_corr_method)
            nonan_count = factor_i.isnull().sum()
            yield dt, next_dt, ic, rank_ic, autocorr, rank_autocorr, shift, nonan_count


class FactorRetTools(object):
    @staticmethod
    def create_reg_data(stk_ret_matrix_shifted_reindex: pd.DataFrame, factor_matrix_reindex_tuple: tuple,
                        factor_name_tuple: tuple, cik_dt_list, shift=1, add_const=True):
        """

        :param stk_ret_matrix_shifted_reindex:
        :param factor_matrix_reindex_tuple:
        :param factor_name_tuple:
        :param cik_dt_list:
        :param shift:
        :param add_const:
        :return:
        """

        for dt, next_dt in zip(cik_dt_list[:-1 * shift], cik_dt_list[shift:]):
            ret_i = stk_ret_matrix_shifted_reindex.loc[dt, :]
            factor_matrix_list = [factor_matrix_reindex.loc[dt, :] for factor_matrix_reindex in
                                  factor_matrix_reindex_tuple]

            temp = pd.concat([ret_i] + list(factor_matrix_list), axis=1).dropna()
            temp.columns = ['ret'] + list(factor_name_tuple)

            formula = f"ret ~ 1 + {'+'.join(factor_name_tuple)}" if add_const else f"ret ~ {'+'.join(factor_name_tuple)}"
            yield formula, temp, dt

    @staticmethod
    def run_ols_result(formula: str, data: pd.DataFrame, dt):
        model = sm.OLS.from_formula(formula, data).fit()
        info = (dt, model.fvalue, model.f_pvalue, model.rsquared, model.rsquared_adj)
        param_df = pd.concat([model.params, model.pvalues, model.tvalues], axis=1)
        param_df.columns = ['params', 'pvalues', 'tvalues']
        param_df['dt'] = dt
        return param_df, info

    @classmethod
    def cal_factor_ret(cls, factor_matrix_dict: dict, stk_cp_matrix: pd.DataFrame, cik_dt_list,
                       cik_id_list,
                        shift=1, add_const=True, ):
        tasks = cls.cal_factor_ret_raw_yield(factor_matrix_dict, stk_cp_matrix, cik_dt_list, cik_id_list, shift=shift,
                                             add_const=add_const)
        params = (param_df for param_df, info in tasks)

        res = pd.concat(params, axis=0)
        params_matrixed = res.reset_index().pivot_table(columns='var_name', index='dt', values='params')  # factor ret
        # pvalues_matrixed = res.pivot_table(columns='var_name', index='dt', values='pvalues') # p-values
        return params_matrixed  # , pvalues_matrixed

    @classmethod
    def cal_factor_ret_raw_yield(cls, factor_matrix_dict: dict, stk_cp_matrix: pd.DataFrame, cik_dt_list,
                                 cik_id_list, shift=1, add_const=True, ):
        """

        :param factor_matrix_dict:
        :param stk_cp_matrix:
        :param cik_dt_list:
        :param cik_id_list:
        :param shift: int,>0
        :param add_const:
        :return:
        """
        print('calculating ...')
        cik_dt_list = sorted(detect_and_transform_dt_format(cik_dt_list))
        # cp 2 ret
        stk_ret_matrix_shifted_reindex = STKPrice2Ret.close_2_close(stk_cp_matrix, shift, cik_dt_list=cik_dt_list,
                                                                    cik_id_list=cik_id_list)
        # stk_ret_matrix_shifted_reindex = stk_cp_matrix.pct_change(periods=shift).reindex(
        #     index=cik_dt_list,
        # columns=cik_id_list)

        factor_name_tuple, factor_matrix_reindex_tuple = zip(
            *[(name, factor_matrix.shift(shift).reindex(index=cik_dt_list, columns=cik_id_list)) for
              name, factor_matrix in
              factor_matrix_dict.items()])

        # cal current cross-section factor ret

        tasks = cls.create_reg_data(stk_ret_matrix_shifted_reindex, factor_matrix_reindex_tuple, factor_name_tuple,
                                    cik_dt_list, shift=shift, add_const=add_const)
        for formula, data, dt in tasks:
            if not data.empty:
                param_df, info = cls.run_ols_result(formula, data, dt)
                param_df.index.name = 'var_name'
                yield param_df, info


class PortfolioIndicatorCreator(object):
    @staticmethod
    def ann_ret_period(net_value, period):
        return net_value.rolling(period).apply(lambda x: np.power(x[-1] / x[0], 250 / period) - 1)

    @staticmethod
    def ann_ret(net_value):
        return np.power(net_value[-1] / net_value[0], 250 / len(net_value)) - 1

    @staticmethod
    def max_drawdown(net_value, return_dt=True):
        max_value = np.maximum.accumulate(net_value)
        i = np.argmax((max_value - net_value) / max_value)  # 最小的净值
        if i == 0:
            return None
        j = np.argmax(net_value[:i])
        if return_dt:
            return net_value.index[i].strftime("%Y%m%d"), net_value.index[j].strftime("%Y%m%d"), (
                    - net_value[j] + net_value[i]) / net_value[j]
        else:
            return i, j, (net_value[i] - net_value[j]) / net_value[j]

    # def cal_indicator_set

    @classmethod
    def cal_indicators(cls, nv_series: pd.Series, daily_ret_series, var_name,
                       period_draft_days: list = [20, 40, 60, 90],
                       rf: (float, np.array, list, tuple) = 1.5 / 100,
                       return_dt=True):
        """

        :param nv_series:
        :param daily_ret_series:
        :param var_name:
        :param period_draft_days:
        :param rf:
        :param return_dt:
        :return:
        """

        # high_beta_nv['daily_yield'] = high_beta_nv.pct_change(1)
        nv_info = {}
        nv_info_rolling = {}

        nv_info[(f"{var_name}", "AnnRet")] = cls.ann_ret(nv_series)
        nv_info[(f'{var_name}', 'Vol')] = daily_ret_series.std() * np.sqrt(250)
        nv_info[(f'{var_name}', 'Sharpe')] = (nv_info[(f"{var_name}", "AnnRet")] - rf) / \
                                             nv_info[(f'{var_name}', 'Vol')]
        end, start, r = cls.max_drawdown(nv_series, return_dt=return_dt)
        nv_info[(f'{var_name}', 'MaxDrawDown_start')] = start
        nv_info[(f'{var_name}', 'MaxDrawDown_end')] = end
        nv_info[(f'{var_name}', 'MaxDrawDown_Rate')] = r

        nv_info = pd.Series(nv_info).reset_index()
        nv_info.columns = ['portfolio', 'indicator', 'value']
        nv_info = nv_info.pivot(index='indicator', columns='portfolio', values='value')

        for num in period_draft_days:
            nv_info_rolling[(f"{var_name}", f"r{num}_ann_ret")] = cls.ann_ret_period(nv_series, num)
            nv_info_rolling[(f"{var_name}", f"r{num}_vol")] = daily_ret_series.rolling(num).std() * np.sqrt(250)
            nv_info_rolling[(f"{var_name}", f"r{num}_sharpe")] = (nv_info_rolling[
                                                                      (f"{var_name}", f"r{num}_ann_ret")] - rf) / \
                                                                 nv_info_rolling[
                                                                     (f"{var_name}", f"r{num}_vol")]

        nv_info_rolling = pd.DataFrame(nv_info_rolling)

        # 月份场景分析，年度场景分析

        return nv_info, nv_info_rolling


class PortfolioStatisticTools(object):
    """


    """

    @classmethod
    def cal_port_ret_info(cls, var_name, port_weight_matrix, stk_ret_matrix, cik_dt_list: list = None,
                          daily_ret_col='daily_ret', unit_net_value_col='unit_net_value',
                          period_draft_days: list = [20, 40, 60, 90, 180],
                          rf: (float, np.array, list, tuple) = 1.5 / 100,
                          ):
        weighted_stk_ret_matrix = cls.cal_port_daily_ret_matrix_method(port_weight_matrix, stk_ret_matrix,
                                                                       cik_dt_list=cik_dt_list,
                                                                       daily_ret_col=daily_ret_col)
        weighted_stk_ret_matrix[unit_net_value_col] = cls.cal_port_net_value(weighted_stk_ret_matrix[daily_ret_col])
        # 计算指标和 多周期指标
        nv_info, nv_info_rolling = PortfolioIndicatorCreator.cal_indicators(weighted_stk_ret_matrix[unit_net_value_col],
                                                                            weighted_stk_ret_matrix[daily_ret_col],
                                                                            var_name,
                                                                            period_draft_days=period_draft_days,
                                                                            rf=rf,
                                                                            )

        return weighted_stk_ret_matrix, nv_info, nv_info_rolling

    @staticmethod
    def cal_port_net_value(daily_ret_series):
        # net_value_series.name = 'unit_net_value'
        return (daily_ret_series + 1).cumprod()

    @staticmethod
    def cal_port_daily_ret_matrix_method(port_weight_matrix, stk_ret_matrix, cik_dt_list: list = None,
                                         daily_ret_col='daily_ret'):
        """

        :param port_weight_matrix:
        :param stk_ret_matrix:
        :param cik_dt_list:
        :param daily_ret_col:
        :return:
        """

        stk_list = port_weight_matrix.columns.tolist()

        cik_dt_list = port_weight_matrix.index.tolist() if cik_dt_list is None else cik_dt_list
        port_weight_matrix_reindex = port_weight_matrix.reindex(index=cik_dt_list)

        stk_ret_matrix_reindex = stk_ret_matrix.reindex(index=cik_dt_list, columns=stk_list)

        weighted_stk_ret_matrix = port_weight_matrix_reindex * stk_ret_matrix_reindex
        # daily ret
        weighted_stk_ret_matrix[daily_ret_col] = weighted_stk_ret_matrix.sum(axis=1, skipna=True)  # will skip na
        return weighted_stk_ret_matrix

        pass


if __name__ == '__main__':
    from data.temp_stk_daily import FakeDataGet, STKDaily, RiskFactor
    from factor_table.core.factortable import FactorTable
    from factor_table.helper.transform_dt_format import transform_dt_format
    import factor_table

    print(factor_table.__version__)

    ft = FactorTable()

    stk = STKDaily()
    print(stk.keys)

    trade_dt = stk.trade_dt
    stklist = stk.stklist
    # amt = stk.get('amt').stack(-1).reset_index()
    # amt.columns = ['trade_dt', 'stklist', 'amt']
    # amt['trade_dt'] = pd.to_datetime(amt['trade_dt'].astype(str), format='%Y%m%d')
    # cp = stk.get('cp').stack(-1).reset_index()
    # cp.columns = ['trade_dt', 'stklist', 'cp']
    # cp['trade_dt'] = pd.to_datetime(cp['trade_dt'].astype(str), format='%Y%m%d')
    # print('data loaded!')

    # ----
    # ft.add_factor('cp', cp, 'trade_dt', 'stklist', factor_names='cp')
    # ft.add_factor('amt', amt, 'trade_dt', 'stklist', factor_names='amt')
    # print('factor added!')
    #
    # import datetime, time
    #
    # t0 = time.time()
    # c = all(map(lambda dt0: isinstance(dt0, datetime.datetime), cp['trade_dt']))
    # t1 = time.time()
    # c1 = transform_dt_format(cp, col='trade_dt')
    # t2 = time.time()
    # print(t1 - t0, t2 - t1)
    #
    # ft.transform_matrix()
    #
    # cp1 = ft.get_transform('cp')
    # amt1 = ft.get_transform('amt')
    amt = stk.get('amt')
    amt.index = pd.to_datetime(amt.index.astype(str), format='%Y%m%d')
    cp = stk.get('cp')
    cp.index = pd.to_datetime(cp.index.astype(str), format='%Y%m%d')
    # res = FactorICTools.cal_ic_and_eval(amt, cp, trade_dt, stklist, shift=1)
    hp = stk.get('hp')
    hp.index = pd.to_datetime(hp.index.astype(str), format='%Y%m%d')
    # res2 = FactorRetTools.cal_factor_ret({'amt': amt}, cp, trade_dt, stklist, shift=1, add_const=True)

    # -------------------
    # create stk_ret
    stk_ret_matrix_shifted_reindex = cp.pct_change(periods=1).shift(-1 * 1)

    weight = []
    for dt, d in ((hp > 1.5) * hp).iterrows():
        ds = d / np.nansum(d)
        ds.name = dt
        weight.append(ds)
    weight = pd.concat(weight, axis=1).T.fillna(0)

    res, nv_info, nv_info_rolling = PortfolioStatisticTools.cal_port_ret_info('test1', weight,
                                                                              stk_ret_matrix_shifted_reindex,
                                                                              daily_ret_col='daily_ret')
    res['unit_net_value'].plot()

    print(1)

    pass
