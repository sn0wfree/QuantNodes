# coding=utf-8
from collections import deque
import pandas as pd
import numpy as np
from QuantNodes.factor_table.Factors import __Factor__, FactorUnit, FactorInfo


class FactorPool(deque):
    def __init__(self, ):
        super(FactorPool, self).__init__()
        self._cik_iids = None
        self._cik_dts = None

    def add_factor(self, *args, **kwargs):
        factor = args[0]
        if isinstance(factor, __Factor__):
            self.append(factor)
        else:
            factor = FactorUnit(*args, **kwargs)
            self.append(factor)

    def merge_df_factor(self):
        res = list(filter(lambda x: x._obj_type == 'DF', self))
        if len(res) == 0:
            yield None
        else:
            for df in res:
                yield df

    def merge_h5_factor(self):
        res = list(filter(lambda x: x._obj_type == 'H5', self))
        if len(res) == 0:
            yield None
        else:
            for h5 in res:
                yield h5

    def merge_sql_factor(self):
        cols = list(FactorInfo._fields[:3]) + list(FactorInfo._fields[-2:])
        factor_name_col = FactorInfo._fields[3]
        alias_col = FactorInfo._fields[4]

        factors = list(filter(lambda x: x._obj_type.startswith(('SQL_', 'db_table')), self))
        if len(factors) == 0:
            yield None
        else:
            factors_info = pd.DataFrame(list(map(lambda x: x.factor_info(), factors)))
            # print(factors_info, factors_info.index)

            # factors.extend(sql_f)  # dataframe have not reduced!
            for (db_table, dts, iid, via, info), df in factors_info.groupby(cols):
                # merge same source data
                cc = df[[factor_name_col, alias_col]].apply(lambda x: ','.join(x))
                origin_factor_names = cc[factor_name_col].split(',')
                alias = cc[alias_col].split(',')
                # use set will disrupt the order
                # we need keep the origin order
                back = list(zip(origin_factor_names, alias))
                disrupted = list(set(back))
                disrupted.sort(key=back.index)

                origin_factor_names_new, alias_new = zip(*disrupted)
                alias_new = list(map(lambda x: x if x != 'None' else None, alias_new))

                # add_factor process have checked
                # db_table_sql: str, query: object, cik_dt: str, cik_id: str, factor_names: str, *args,
                #                  as_alias: (list, tuple, str) = None
                # name, obj, cik_dt: str, cik_id: str, factor_names: (list, tuple, str), *args,
                # as_alias: (list, tuple, str) = None, db_table: str = None, ** kwargs
                idx_mask = df.index[0]
                f = factors[idx_mask]

                res = FactorUnit(db_table, f._obj, dts, iid, origin_factor_names_new, via,
                                 as_alias=alias_new,
                                 db_table=db_table, db_type=f.db_type)
                yield res

    def merge_factors(self, inplace=False):
        # todo unstack factor name to check whether factor exists duplicates!!
        factors = FactorPool()
        sql_factor = list(self.merge_sql_factor())
        if sql_factor[0] is not None:
            factors.extend(sql_factor)
        df_factor = list(self.merge_df_factor())
        if df_factor[0] is not None:
            factors.extend(df_factor)
        h5_factor = list(self.merge_h5_factor())
        if h5_factor[0] is not None:
            factors.extend(h5_factor)

        if inplace:
            self = factors
        else:
            return factors

    def fetch(self, _cik_dts=None, _cik_iids=None, reduced=True, add_limit=False, ):
        """

        :param reduced: whether use reduce form
        :param _cik_dts: set up dts
        :param _cik_iids:  set up iids
        :param add_limit: use force limit columns
        :return:
        """
        if not add_limit:
            if _cik_dts is not None:
                self._cik_dts = _cik_dts
            else:
                if self._cik_dts is None:
                    raise KeyError('cik_dts(either default approach or fetch) both are not setup!')

            if _cik_iids is not None:
                self._cik_iids = _cik_iids
            else:
                if self._cik_iids is None:
                    raise KeyError('cik_iids(either default approach or fetch) both are not setup!')

        if reduced:
            factors = self.merge_factors()
        else:
            factors = self
        fetched = []
        for f in factors:
            res = f.get(self._cik_dts, self._cik_iids).set_index(['cik_dts', 'cik_iid'])
            fetched.append(res)
        result = pd.concat(fetched, axis=1)
        # columns = result.columns.tolist()
        return result


if __name__ == '__main__':
    np.random.seed(1)
    from ClickSQL import BaseSingleFactorTableNode

    src = 'clickhouse://default:***REMOVED***@***REMOVED***:8123/system'
    node = BaseSingleFactorTableNode(src)
    f2_ = FactorPool()

    df = pd.DataFrame(np.random.random(size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v1', 'v2'])
    f1 = FactorUnit('EDGAR_LOG.parsed_edgar_log', node, 'date', 'ip', factor_names=['zone', 'cik'],
                    db_type='ClickHouse')
    f2_.add_factor(f1)
    f2_.add_factor('EDGAR_LOG.parsed_edgar_log', node, 'date', 'ip', factor_names='size,crawler', db_type='ClickHouse')
    ip = ['107.23.85.jfd', '104.198.37.bdc', '104.155.169.jbh']
    f_ip = "','".join(ip)
    sql = f"select date,ip,idx,norefer,noagent from EDGAR_LOG.parsed_edgar_log where  ip in ('{f_ip}')  "
    df = node(sql)
    f2_.add_factor('parsed_edgar_log', df, 'date', 'ip', factor_names='idx,norefer,noagent')

    #
    # f3 = FactorUnit('test.test2', query, 'cik_dts', 'cik_iid', factor_names=['v3'])
    #

    # f2_.add_factor('EDGAR_LOG.parsed_edgar_log', node, 'date', 'ip', 'code')
    # f2_.add_factor('select * from EDGAR_LOG.parsed_edgar_log', node, 'date', 'ip', 'idx,cik,noagent')
    f2_.merge_factors(inplace=True)
    df = f2_.fetch(_cik_dts=['2017-04-30'], _cik_iids=ip)
    pass
