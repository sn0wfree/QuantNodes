# coding=utf-8
from collections import namedtuple, deque, Callable
import pandas as pd
import numpy as np
from QuantNodes.factor_table.Factors import __Factor__, FactorUnit

FactorInfo = namedtuple('FactorInfo',
                        ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'via', 'conditions'))


# class _FactorPool(deque):
#     @classmethod
#     def add_factor(cls, db_table: str, factor_names: (list, tuple, str), cik_dt,
#                    cik_iid, cik_dt_format: str = 'datetime', as_alias: (list, tuple, str) = None,
#                    conds='1'):
#         if isinstance(db_table, str):
#             if db_table.lower().startswith('select'):
#                 via = 'sql'
#                 return cls.add_factor_via_sql(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
#                                               as_alias=as_alias, conds=conds, cik_dt_format=cik_dt_format)
#             else:
#                 via = 'db_table'
#                 return cls.add_factor_via_db_table(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
#                                                    as_alias=as_alias, conds=conds, cik_dt_format=cik_dt_format)
#         elif isinstance(db_table, pd.DataFrame):
#
#             return cls.add_factor_via_df(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
#                                          as_alias=as_alias, conds=conds, cik_dt_format=cik_dt_format)
#
#         elif isinstance(db_table, __MetaFactorTable__):
#             return list(cls.add_factor_via_ft(db_table, factor_names, cik_dt=cik_dt, cik_iid=cik_iid,
#                                               as_alias=as_alias, conds=conds, cik_dt_format=cik_dt_format))
#         else:
#             raise NotImplementedError('not supported')
#
#     @staticmethod
#     def _add_factor_via_obj(db_table: str, factor_names: (list, tuple, str), via: str, cik_dt,
#                             cik_iid, cik_dt_format: str = 'datetime', as_alias: (list, tuple, str) = None,
#                             conds='1', check=False):
#         """
#
#        :param as_alias:
#        :param db_table:
#        :param factor_names:
#        :param cik_dt:
#        :param cik_iid:
#        :param conds:  conds = @test1>1 | @test2<1
#        :return:
#        """
#         factor_names = __MetaFactorTable__.generate_factor_names(factor_names)
#         alias = __MetaFactorTable__.generate_alias(factor_names, as_alias=as_alias)
#         # rename variables
#         f_names_list = [f if (a is None) or (f == a) else f"{f} as {a}" for f, a in zip(factor_names, alias)]
#         cols_str = ','.join(f_names_list)
#
#         # change dt dtype for suitable dtype
#         conditions = '1' if conds == '1' else conds.replace('&', 'and').replace('|', 'or').replace('@', '')
#
#         # convert cik_dt
#         if cik_dt == 'cik_dt':
#             cik_dt_str = cik_dt
#         else:
#             if cik_dt_format == 'str':
#                 cik_dt_str = f"parseDateTimeBestEffort({cik_dt}) as cik_dt"
#             elif cik_dt_format == 'datetime':
#                 cik_dt_str = f"{cik_dt} as cik_dt"
#             elif cik_dt_format == 'int':
#                 cik_dt_str = f"parseDateTimeBestEffort(toString({cik_dt})) as cik_dt"
#             else:
#                 cik_dt_str = f"parseDateTimeBestEffort(toString({cik_dt})) as cik_dt"
#
#         # convert cik_iid
#         cik_iid_str = f"{cik_iid} as cik_iid"
#         if via == 'sql':
#             sql = f'select {cols_str}, {cik_dt_str}, {cik_iid_str}  from ({db_table}) where {conditions}'
#         else:
#             sql = f'select {cols_str}, {cik_dt_str}, {cik_iid_str}  from {db_table} where {conditions}'
#
#         res = FactorInfo(db_table, cik_dt, cik_iid, ','.join(map(str, factor_names)), ','.join(map(str, alias)),
#                          sql, via, conds)  #
#         return res
#         # self.append(res)
#
#     @classmethod
#     def add_factor_via_db_table(cls, db_table: str, factor_names: (list, tuple, str), cik_dt=None, cik_iid=None,
#                                 as_alias: (list, tuple, str) = None, conds='1', cik_dt_format: str = 'datetime', ):
#         return cls._add_factor_via_obj(db_table, factor_names, 'db_table', cik_dt=cik_dt, cik_dt_format=cik_dt_format,
#                                        cik_iid=cik_iid, as_alias=as_alias, conds=conds)
#
#     @classmethod
#     def add_factor_via_sql(cls, sql_ori, factor_names: (list, tuple, str), cik_dt=None, cik_iid=None,
#                            cik_dt_format: str = 'datetime',
#                            as_alias: (list, tuple, str) = None, conds='1'):
#         return cls._add_factor_via_obj(sql_ori, factor_names, 'sql', cik_dt=cik_dt, cik_dt_format=cik_dt_format,
#                                        cik_iid=cik_iid, as_alias=as_alias, conds=conds)
#
#     @classmethod
#     def add_factor_via_df(cls, factor_df: pd.DataFrame, factor_names: (list, tuple), cik_dt='cik_dt',
#                           cik_iid='cik_iid',
#                           as_alias: (list, tuple, str) = None, conds='1', cik_dt_format: str = 'datetime', **kwargs):
#         factor_names = __MetaFactorTable__.generate_factor_names(factor_names)
#         alias = __MetaFactorTable__.generate_alias(factor_names, as_alias=as_alias)
#         if isinstance(factor_df, pd.DataFrame):  # todo add factor via dataframe
#             # check cik_dt or cik_iid exists
#             factor_df = factor_df.reset_index()
#             exists_cols = factor_df.columns.tolist()
#             if cik_dt not in exists_cols:
#                 raise ValueError(f'cannot local {cik_dt} column! please check cik_dt parameter is correct!')
#             if cik_iid not in exists_cols:
#                 raise ValueError(f'cannot local {cik_iid} column! please check cik_iid parameter is correct!')
#
#             for f in factor_names:
#                 if f not in exists_cols:
#                     raise ValueError(f'cannot local {f} column! please check factor_names parameter is correct!')
#
#             sql = ''
#             via = 'pd.DataFrame'
#
#             res = FactorInfo(factor_df, cik_dt, cik_iid, ','.join(map(str, factor_names)),
#                              ','.join(map(str, alias)), sql, via, conds)  #
#             return res
#
#         else:
#
#             raise NotImplementedError(
#                 f'factor_df only accept pd.DataFrame,but got {type(factor_df)}. its type is not supported!')
#         pass
#
#     @classmethod
#     def add_factor_via_ft(cls, factor_table: __MetaFactorTable__, factor_names: (list, tuple), cik_dt=None,
#                           cik_iid=None,
#                           as_alias: (list, tuple, str) = None, conds='1', cik_dt_format: str = 'datetime', ):
#         # todo add factor via factortable
#         if isinstance(factor_table, __MetaFactorTable__):
#             for f in factor_table._factors.show_factors(reduced=True, to_df=False):  # get one factorInfo
#                 origin_factor_names = f.origin_factor_names.split(',')
#                 alias_factor_names = f.alias.split(',')
#                 for o_f, alias_f in zip(origin_factor_names, alias_factor_names):  # mapping factor and its alias
#                     if o_f in factor_names or alias_f in factor_names:
#                         # if required factor name at factor or alias, will yield this factorinfo and break this loop
#                         yield f
#                         # todo optimize the add factor via ft process!!
#                         # todo if get multi-factors, need get required factors only rather than the whole factorinfo
#                         break
#
#             # raise NotImplementedError('not supported')
#         else:
#             raise TypeError('factor_table must be FactorTable or its subclass!')
#
#     def show_factors(self, reduced=False, to_df=True):
#         # todo unstack factor name to check whether factor exists duplicates!!
#         if reduced:
#             # ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql','via', 'conditions')
#             # ['db_table', 'dts', 'iid', 'conditions']
#             cols = list(FactorInfo._fields[:3]) + list(FactorInfo._fields[-2:])
#             no_df_f = list(filter(lambda x: x.via != 'pd.DataFrame', self))
#             df_f = list(filter(lambda x: x.via == 'pd.DataFrame', self))
#             f = pd.DataFrame(no_df_f, columns=FactorInfo._fields)
#             factor_name_col = FactorInfo._fields[3]
#             alias_col = FactorInfo._fields[4]
#
#             # can_merged_index = (fgroupby['sql'].count() > 1).reset_index()
#             # can_merged_index = can_merged_index[can_merged_index['sql']]
#             # can_merged_index = fgroupby.count().index
#             factors = []
#             factors.extend(df_f)  # dataframe have not reduced!
#             # no_df_f = f[no_df_mask]
#             for (db_table, dts, iid, via, conditions), df in f.groupby(cols):
#                 # masks = (f['db_table'] == db_table) & (f['dts'] == dts) & (f['iid'] == iid) & (
#                 #         f['conditions'] == conditions)
#                 cc = df[[factor_name_col, alias_col]].apply(lambda x: ','.join(x))
#
#                 origin_factor_names = cc[factor_name_col].split(',')
#                 alias = cc[alias_col].split(',')
#                 # use set will disrupt the order
#                 # we need keep the origin order
#                 back = list(zip(origin_factor_names, alias))
#                 disrupted = list(set(back))
#                 disrupted.sort(key=back.index)
#
#                 origin_factor_names_new, alias_new = zip(*disrupted)
#                 alias_new = list(map(lambda x: x if x != 'None' else None, alias_new))
#
#                 # cik_dt, cik_iid = self.check_cik_dt(cik_dt=dts, default_cik_dt=self._cik.dts), self.check_cik_iid(
#                 #     cik_iid=iid, default_cik_iid=self._cik.iid)
#                 # add_factor process have checked
#                 res = self._add_factor_via_obj(db_table, origin_factor_names_new, via, cik_dt=dts, cik_iid=iid,
#                                                conds=conditions, as_alias=alias_new)
#                 factors.append(res)
#         else:
#             factors = self
#         if to_df:
#             return pd.DataFrame(factors, columns=FactorInfo._fields)
#         else:
#             return factors
#
#     @staticmethod
#     def _generate_fetch_sql_iter(factors, filter_cond_dts, filter_cond_ids, reduced=True, add_limit=False, **kwargs):
#
#         # sql_list = []
#         if add_limit:
#             limit_str = 'limit 100'
#         else:
#             limit_str = ''
#         # todo 可能存在性能点
#         no_df_factors_list = filter(lambda x: x.via != 'pd.DataFrame', factors)
#
#         for db_table, dts, iid, origin_factor_names, alias, sql, via, conditions in no_df_factors_list:
#             sql2 = f"select * from ({sql}) where {filter_cond_dts} and {filter_cond_ids} {limit_str} "
#             yield sql2
#
#     def _fetch_sql_part(self, query, factors, filter_cond_dts, filter_cond__ids, reduced=True, add_limit=False,
#                         to_sql=False):
#         if not isinstance(query, Callable):
#             raise ValueError('query must database connector with __call__')
#         sql_list_iter = self._generate_fetch_sql_iter(factors, filter_cond_dts, filter_cond__ids, reduced=reduced,
#                                                       add_limit=add_limit)
#         if to_sql:
#             for sql2 in sql_list_iter:
#                 yield sql2
#         else:
#             for sql2 in sql_list_iter:
#                 df = query(sql2)
#                 res = pd.DataFrame(df).set_index(['cik_dt', 'cik_iid'])
#
#                 # ['db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions']
#                 yield res
#
#     @staticmethod
#     def _fetch_df_part(df_factors, filter_cond_dts, filter_cond__ids, reduced=True, add_limit=False,
#                        to_sql=False):
#
#         for db_table, dts, iid, origin_factor_names, alias, sql, via, conditions in df_factors:
#             origin_factor_names = origin_factor_names.split(',')
#             alias = alias.split(',')
#             c = {o: a for o, a in zip(origin_factor_names, alias) if a != 'None'}
#             if len(c) == 0:
#                 pass
#             else:
#                 db_table = db_table.rename(columns=c)
#             yield db_table.set_index(['cik_dt', 'cik_iid'])
#
#     def fetch_iter(self, query, filter_cond_dts, filter_cond__ids, reduced=True, add_limit=False, to_sql=False):
#         if not isinstance(query, Callable):
#             raise ValueError('query must database connector with __call__')
#         factors = self.show_factors(reduced=reduced, to_df=False)
#         sql_factors = list(filter(lambda x: x.via != 'pd.DataFrame', factors))
#         res_iter = self._fetch_sql_part(query, sql_factors, filter_cond_dts, filter_cond__ids, reduced=reduced,
#                                         add_limit=add_limit, to_sql=to_sql)
#
#         # mask = factors['via'] == 'pd.DataFrame'
#         # df_factors = factors[mask]
#         df_factors_list = list(filter(lambda x: x.via == 'pd.DataFrame', factors))
#
#         for s in res_iter:
#             yield s
#         if to_sql:
#             if len(df_factors_list) != 0:
#                 # warning dataframe mode
#                 warnings.warn('factors have at least one pd.DataFrame data,which will ignore at to_sql=True mode!!!')
#         else:
#             if len(df_factors_list) != 0:
#                 df_factor_res = self._fetch_df_part(df_factors_list, filter_cond_dts, filter_cond__ids, reduced=reduced,
#                                                     add_limit=add_limit, to_sql=to_sql)
#                 for df_f in df_factor_res:
#                     yield df_f
#
#     def fetch_all(self, query, filter_cond_dts, filter_cond__ids, reduced=True, add_limit=False, to_sql=False):
#         if not isinstance(query, Callable):
#             raise ValueError('query must database connector with __call__')
#         factors = self.show_factors(reduced=reduced, to_df=False)
#         sql_list_iter = self._generate_fetch_sql_iter(factors, filter_cond_dts, filter_cond__ids, reduced=reduced,
#                                                       add_limit=add_limit)
#
#         from functools import reduce
#
#         def join(sql1, sql2):
#             settings = ' settings joined_subquery_requires_alias=0 '
#             sql = f"select * from ({sql1}) all full join ({sql2}) using (cik_dt,cik_iid)  {settings}"
#             return sql
#
#         s = reduce(lambda x, y: join(x, y), sql_list_iter)
#         if to_sql:
#             yield s
#         else:
#             df = query(s)
#             res = pd.DataFrame(df).set_index(['cik_dt', 'cik_iid'])
#
#             # ['db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'conditions']
#             yield res

class FactorPool(deque):
    def __init__(self, ):
        super(FactorPool, self).__init__()

    def add_factor(self, *args, **kwargs):
        factor = args[0]
        if isinstance(factor, __Factor__):
            self.append(factor)
        else:
            factor = FactorUnit(*args, **kwargs)
            self.append(factor)

    def reduce_sql_type(self):
        cols = list(FactorInfo._fields[:3]) + list(FactorInfo._fields[-2:])
        factor_name_col = FactorInfo._fields[3]
        alias_col = FactorInfo._fields[4]
        sql_f = list(filter(lambda x: (x.via.startswith('SQL_')) or (x.via == 'db_table'), self))
        f = pd.DataFrame(sql_f, columns=FactorInfo._fields)
        for (db_table, dts, iid, via, conditions), df in f.groupby(cols):
            # masks = (f['db_table'] == db_table) & (f['dts'] == dts) & (f['iid'] == iid) & (
            #         f['conditions'] == conditions)
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
            yield (db_table, dts, iid, origin_factor_names_new, alias_new, via, conditions)

    def show_factors(self, reduced=False, to_df=True):

        # todo unstack factor name to check whether factor exists duplicates!!
        if reduced:
            # ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql','via', 'conditions')
            cols = list(FactorInfo._fields[:3]) + list(FactorInfo._fields[-2:])
            factor_name_col = FactorInfo._fields[3]
            alias_col = FactorInfo._fields[4]
            no_df_f = list(filter(lambda x: (x.via == 'SQL') or (x.via == 'db_table'), self))
            h5_f = list(filter(lambda x: x.via == 'H5', self))
            df_f = list(filter(lambda x: x.via == 'DF', self))
            f = pd.DataFrame(no_df_f, columns=FactorInfo._fields)

            factors = []
            factors.extend(df_f)  # dataframe have not reduced!
            # no_df_f = f[no_df_mask]
            for (db_table, dts, iid, via, conditions), df in f.groupby(cols):
                # masks = (f['db_table'] == db_table) & (f['dts'] == dts) & (f['iid'] == iid) & (
                #         f['conditions'] == conditions)
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

                # cik_dt, cik_iid = self.check_cik_dt(cik_dt=dts, default_cik_dt=self._cik.dts), self.check_cik_iid(
                #     cik_iid=iid, default_cik_iid=self._cik.iid)
                # add_factor process have checked
                res = self._add_factor_via_obj(db_table, origin_factor_names_new, via, cik_dt=dts, cik_iid=iid,
                                               conds=conditions, as_alias=alias_new)
                factors.append(res)
        else:
            factors = self
        if to_df:
            return pd.DataFrame(factors, columns=FactorInfo._fields)
        else:
            return factors


if __name__ == '__main__':
    f_ = FactorPool()
    np.random.seed(1)
    df = pd.DataFrame(np.random.random(size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v1', 'v2'])
    f1 = FactorUnit('test', df, 'cik_dts', 'cik_iid', factor_names=['v1', 'v2'])

    df = pd.DataFrame(np.random.random(size=(1000, 3)), columns=['cik_dts', 'cik_iid', 'v3'])
    f2 = FactorUnit('test2', df, 'cik_dts', 'cik_iid', factor_names=['v3'])
    from ClickSQL import BaseSingleFactorTableNode
    node = BaseSingleFactorTableNode('clickhouse://default:123456@***REMOVED***:8123/system')
    query = node.query
    f3 = FactorUnit('test.test2', query, 'cik_dts', 'cik_iid', factor_names=['v3'])

    f_.append(f1)
    f_.append(f2)
    f_.append(f3)
    f_.reduce_sql_type()
    pass
