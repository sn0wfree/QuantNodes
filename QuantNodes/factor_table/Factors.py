# coding=utf-8
from collections import namedtuple,Callable
from functools import partial
from QuantNodes.utils_node.check_file_type import filetype
import pandas as pd

CIK = namedtuple('CoreIndexKeys', ('dts', 'iid'))

FactorInfo = namedtuple('FactorInfo',
                        ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'via', 'conditions'))


class __Factor__(object):
    def __init__(self, name, obj, cik_dt: str, cik_id: str, factor_names: (list, tuple, str), *args,
                 as_alias: (list, tuple, str) = None, db_table: str = None, **kwargs):
        self._obj = obj
        self._cik = CIK(cik_dt, cik_id)
        self._factor_name_list = self.generate_factor_names(factor_names)
        self._alias = self.generate_alias(self._factor_name_list, as_alias)
        self._obj_type = 'Meta'
        self._db_table = db_table
        self._name = name

    __slots__ = ['_obj', '_cik', '_factor_name_list', '_alias', '_obj_type', '_db_table', '_name']

    @staticmethod
    def generate_factor_names(factor_names: (list, tuple, str)):
        if isinstance(factor_names, str):
            if ',' in factor_names:
                factor_names = factor_names.split(',')
            else:
                factor_names = [factor_names]
        elif isinstance(factor_names, (list, tuple)):
            factor_names = list(factor_names)
        else:
            raise ValueError('columns only accept list tuple str!')
        return factor_names

    @staticmethod
    def generate_alias(factor_names: (list,), as_alias: (list, tuple, str) = None):
        if as_alias is None:
            alias = len(factor_names) * [None]
        elif isinstance(as_alias, str):
            if ',' in as_alias:
                alias = as_alias.split(',')
            else:
                alias = [as_alias]

        elif isinstance(as_alias, (list, tuple)):
            if len(as_alias) != len(factor_names):
                raise ValueError('as_alias is not match factor_names')
            else:
                alias = list(as_alias)
        else:
            raise ValueError('alias only accept list tuple str!')
        return alias

    def factor_info(self):
        return FactorInfo(self._obj,  # 'db_table' # df
                          self._cik.dts,  # 'dts'
                          self._cik.iid,  # 'iid'
                          ','.join(self._factor_name_list),  # 'origin_factor_names'
                          ','.join(self._alias),  # 'alias'
                          '',  # sql
                          self._obj_type,  # via
                          ''  # conditions
                          )

    # ('db_table', 'dts', 'iid', 'origin_factor_names', 'alias', 'sql', 'via', 'conditions')

    def _df_get_(self, df, cik_dt_list, cik_id_list):
        dt_mask = df[self._cik.dts].isin(cik_dt_list)
        id_mask = df[self._cik.iid].isin(cik_id_list)
        mask = dt_mask & id_mask
        cols = [self._cik.dts, self._cik.iid] + self._factor_name_list
        return df[mask][cols].rename(columns={self._cik.dts: 'cik_dts', self._cik.iid: 'cik_iid'})


class __FactorDF__(__Factor__):
    __slots__ = ['_obj', '_cik', '_factor_name_list', '_alias', '_obj_type', '_db_table', '_name']

    def __init__(self, name: str, df: pd.DataFrame, cik_dt: str, cik_id: str, factor_names: (list, tuple, str), *args,
                 as_alias: (list, tuple, str) = None, **kwargs):
        super(__FactorDF__, self).__init__(name, df, cik_dt, cik_id, factor_names, *args,
                                           as_alias=as_alias, **kwargs)
        self._obj_type = 'DF'

    def get(self, cik_dt_list, cik_id_list, **kwargs):
        return self._df_get_(self._obj, cik_dt_list, cik_id_list)
        # self.get = partial(self._df_get_, df=self._obj)


class __FactorHDF__(__Factor__):
    __slots__ = ['_obj', '_cik', '_factor_name_list', '_alias', '_obj_type', '_db_table', '_name']

    def __init__(self, key: str, h5_path: str, cik_dt: str, cik_id: str, factor_names: (list, tuple, str), *args,
                 as_alias: (list, tuple, str) = None, **kwargs):
        super(__FactorHDF__, self).__init__(key, h5_path, cik_dt, cik_id, factor_names, *args,
                                            as_alias=as_alias, **kwargs)
        if filetype(h5_path) == 'HDF5':
            self._obj_type = 'H5'
        else:
            raise ValueError(f'{h5_path} is not a HDF5 file !')
        # self.get = partial(self._df_get_, df=self._obj)

    def factor_info(self):
        return FactorInfo(self._obj,  # 'db_table' = H5_path
                          self._cik.dts,  # 'dts'
                          self._cik.iid,  # 'iid'
                          ','.join(self._factor_name_list),  # 'origin_factor_names'
                          ','.join(self._alias),  # 'alias'
                          self._name,  # sql name = key
                          self._obj_type,  # via # H5
                          ''  # conditions
                          )

    def get(self, cik_dt_list, cik_id_list):
        df = pd.read_hdf(self._obj, self._name)  # name =key
        return self._df_get_(df, cik_dt_list, cik_id_list)


class __FactorSQL__(__Factor__):
    def __init__(self, db_table_sql: str, query: object, cik_dt: str, cik_id: str, factor_names: str, *args,
                 as_alias: (list, tuple, str) = None, **kwargs):
        super(__FactorSQL__, self).__init__(db_table_sql, query, cik_dt, cik_id, factor_names, *args,
                                            as_alias=as_alias, **kwargs)

        db_type = kwargs.get('db_type','ClickHouse')

        if db_table_sql.lower().startswith('select '):
            self._obj_type = f'SQL_{db_type}'
        else:
            self._obj_type = 'db_table'
        self._db_table = db_table_sql

    def factor_info(self):
        if self._obj_type.startswith('SQL_'):
            return FactorInfo(self._db_table,  # 'db_table' = db_table_sql
                              self._cik.dts,  # 'dts'
                              self._cik.iid,  # 'iid'
                              ','.join(self._factor_name_list),  # 'origin_factor_names'
                              ','.join(self._alias),  # 'alias'
                              self._db_table,  # sql
                              self._obj_type,  # via # SQL
                              ''  # conditions
                              )
        else:
            return FactorInfo(self._db_table,  # 'db_table' = db_table_sql
                              self._cik.dts,  # 'dts'
                              self._cik.iid,  # 'iid'
                              ','.join(self._factor_name_list),  # 'origin_factor_names'
                              ','.join(self._alias),  # 'alias'
                              '',  # sql name = key
                              self._obj_type,  # via # db_table
                              ''  # conditions
                              )

    def get(self, cik_dt_list, cik_id_list):

        f_names_list = [f if (a is None) or (f == a) else f"{f} as {a}" for f, a in
                        zip(self._factor_name_list, self._alias)]
        cols_str = ','.join(f_names_list)

        conditions = f"cik_dts in ({','.join(cik_dt_list)}) and cik_iid in ({','.join(cik_id_list)})"

        if self._obj_type == 'SQL':
            sql = f'select {cols_str},{self._cik.dts} as cik_dts, {self._cik.iid}  as cik_iid  from ({self._db_table}) where {conditions}'
        else:
            sql = f'select {cols_str},{self._cik.dts} as cik_dts, {self._cik.iid}  as cik_iid  from {self._db_table} where {conditions}'

        return self._obj(sql)


class FactorUnit(type):
    def __new__(cls, name, obj, cik_dt: str, cik_id: str, factor_names: (list, tuple, str), *args,
                as_alias: (list, tuple, str) = None, db_table: str = None, **kwargs):
        if isinstance(obj, pd.DataFrame):
            _obj = __FactorDF__(name, obj, cik_dt, cik_id, factor_names, *args,
                                as_alias, db_table, **kwargs)
        elif isinstance(obj, str):
            _obj = __FactorHDF__(name, obj, cik_dt, cik_id, factor_names, *args,
                                 as_alias, db_table, **kwargs)
        elif isinstance(obj, Callable):
            _obj = __FactorSQL__(name, obj, cik_dt, cik_id, factor_names, *args,
                                 as_alias, db_table, **kwargs)
        else:
            raise ValueError('unknow info')
        _obj.__class__.__name__ = 'FactorUnit'
        return _obj


if __name__ == '__main__':
    import numpy as np
    df = pd.DataFrame(np.random.random(size=(1000, 4)), columns=['cik_dts', 'cik_iid', 'v1', 'v2'])
    f1 = FactorUnit('test', df, 'cik_dts', 'cik_iid', factor_names=['v1', 'v2'])
    print(type(FactorUnit))
    print(isinstance(f1, FactorUnit))

    pass
