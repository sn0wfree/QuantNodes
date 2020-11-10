# coding=utf-8
import re
from collections import namedtuple, ChainMap

from ClickSQL import ClickHouseTableNodeExt

factor_parameters = ('dt', 'code', 'value', 'fid')
ft_node = namedtuple('factortable', factor_parameters)


# from Nodes.utils_node.timer import timer


class FactorBackendCH(object):
    __slots__ = ['_src', 'node', 'db_table']

    def __init__(self, src: str):
        """

        :param src:  sample: clickhouse://test:sysy@199.199.199.199:1234/drre
        """
        # db_settings = parse_rfc1738_args(src)
        self._src = src
        # self._settings = db_settings
        # self._store_type = self._settings['name']

        # self._para = node(self._settings['host'], self._settings['port'], self._settings['user'],
        #                   self._settings['password'], self._settings['database'])  # store connection information
        self.node = ClickHouseTableNodeExt(conn_str=src)
        self.db_table = self.node._para.database
        # db_settings['host'], db_settings['port'], db_settings['user'],
        # db_settings['password'], db_settings['database']

    def __call__(self, sql: str, **kwargs):
        return self.node.query(sql, **kwargs)

    @staticmethod
    def __extend_dict_value__(conditions: (dict, ChainMap)):
        for s in conditions.values():
            if isinstance(s, str):
                yield s
            elif isinstance(s, (tuple, list)):
                for s_sub in s:
                    yield s_sub
            else:
                raise ValueError('filter settings get wrong type! only accept string and tuple of string')

    @staticmethod
    def __obtain_other_filter__(other_filters):
        exits_keys = []
        for k, v in other_filters.items():
            if k in exits_keys:
                raise ValueError(f'found duplicated key: {k}')
            exits_keys.append(k)
            if isinstance(v, dict):
                yield v
            elif isinstance(v, (str, tuple)):
                yield {k: v}
            else:
                raise ValueError('filter settings get wrong type! only accept string and tuple of string')

    @classmethod
    def _get_sql(cls, db_table: str, cols: (tuple, None, list) = None, data_filter: dict = {}, include_filter=True,
                 **other_filters):
        """

        :param data_filter:
        :param cols:
        :param include_filter:
        :param other_filters:
        :return:
        """
        if cols is None:
            cols = factor_parameters
        elif len(cols) == 0:
            cols = ['*']

        conditions = ChainMap(data_filter, *list(cls.__obtain_other_filter__(other_filters)))
        filter_yield = cls.__extend_dict_value__(conditions)
        if include_filter:
            cols = set(list(cols) + list(conditions.keys()))
        else:
            cols = set(cols)

        sql = f"select {','.join(cols)} from {db_table} where {' and '.join(sorted(set(['1'] + list(filter_yield))))} "
        return sql

    def _execute(self, sql: str, **kwargs):

        return self.node.query(sql, **kwargs)
        # self.__execute__ = self.operator.query

    @staticmethod
    def _check_end_with_limit(string, pattern=r'[\s]+limit[\s]+[0-9]+$'):
        m = re.findall(pattern, string)
        if m is None or m == []:
            return False
        else:
            return True


#
# def EXECUTE(obj, sql: str):
#     if hasattr(obj, '__execute__'):
#         return obj.__execute__(sql)
#     else:
#         raise ValueError()


class BaseSingleFactorHelper(object):
    __Name__ = "基础因子库单因子基类"
    __slots__ = ('_operator', 'db', 'table', 'db_table', '_kwargs', '_raw_kwargs', '_status', '_INFO', 'depend_tables')

    def __init__(self, src: str, db_table: (None, str) = None, info=None, **kwargs):
        """

        :type kwargs: object
        :param src: string sample: clickhouse://test:sysy@199.199.199.199:1234/drre
        :param db_table:
        :param info:
        :param kwargs:  data_filter will store operator for some cols
        """

        self._operator = FactorBackendCH(src)

        # self._execute = self._operator._execute

        if db_table is None:
            src_db_table = self._operator.db_table
            if '.' in src_db_table:
                self.db_table = src_db_table
            else:
                raise ValueError('db_table parameter get wrong type!')
        elif isinstance(db_table, str):
            self.db_table = db_table
        else:
            raise ValueError('db_table only accept str!')
        db, table = self.db_table.split('.')
        self.db = db
        self.table = table
        self.depend_tables = [self.db_table]
        self._kwargs = kwargs
        self._raw_kwargs = kwargs
        self._status = 'SQL'
        self._INFO = info

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status: str):
        self._status = status

    def __str__(self):
        return self.__sql__

    @property
    def __sql__(self):
        return self._operator._get_sql(db_table=self.db_table, **self._kwargs)

    @property
    def row_count(self):
        sql = f"select count(1) as row_count from {self.db_table}"
        row_count = self._operator(sql)['row_count'].values[0]
        return row_count

    @property
    def table_engine(self):
        sql = f"select engine from system.tables where database ='{self.db}' and name='{self.table}'"
        engine = self._operator(sql)['engine'].values[0]
        return engine

    @property
    # @timer
    def empty(self):
        if self.row_count == 0:
            return True
        else:
            return False

    def fetch(self, pattern=r'[\s]+limit[\s]+[0-9]+$'):
        """
        fetch first 1000 line
        :return:
        """
        sql = self.__sql__
        end_with_limit = self._operator._check_end_with_limit(sql, pattern=pattern)
        if end_with_limit:
            return self._operator(sql)
        else:
            return self._operator(sql + ' limit 1000')

    def fetch_all(self):
        """
        fetch all data
        :return:
        """

        return self._operator(self.__sql__)

    def update(self, **kwargs):
        self._kwargs.update(kwargs)

    def __call__(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        self.update(**kwargs)
        if self._status == 'SQL':
            return self.__sql__
        elif self._status == 'SQL:fetch':
            return self.fetch()
        elif self._status == 'SQL:fetch_all':
            return self.fetch_all()

    # def __data__(self):
    #     return self.__query__(**self._kwargs)
    @property
    def operator(self):
        return self._operator


class BaseSingleFactorTableNode(BaseSingleFactorHelper):
    pass


"add auto-increment col by materialized bitOr(bitShiftLeft(toUInt64(now64()),24), rowNumberInAllBlocks()) "


class UpdateProcess(object):
    @classmethod
    def update(cls, src_table: BaseSingleFactorTableNode,
               dst_table: BaseSingleFactorTableNode,
               fid_ck: str = 'fid',
               dt_max_1st=True):
        if src_table.empty:
            raise ValueError(f'{src_table.db_table} is empty')
        if src_table.empty:
            update_status = 'full'
            sql = cls.full_update(src_table, dst_table)
        else:
            update_status = ' incremental'
            sql = cls.incremental_update(src_table, dst_table, fid_ck, dt_max_1st=dt_max_1st)
        return sql, update_status

    @staticmethod
    def full_update(src_table: BaseSingleFactorTableNode, dst_table: BaseSingleFactorTableNode):
        dst_db_table = dst_table.db_table
        insert_sql = f"insert into {dst_db_table} {src_table}"
        return insert_sql

    @staticmethod
    def incremental_update(src_table: BaseSingleFactorTableNode, dst_table: BaseSingleFactorTableNode,
                           fid_ck: str, dt_max_1st=True):
        # src_db_table = src_table.db_table
        src_table_type = src_table.table_engine
        dst_db_table = dst_table.db_table
        if dt_max_1st:
            order_asc = ' desc'
        else:
            order_asc = ' asc'
        if src_table_type != 'View':
            sql = f'select max({fid_ck}) as {fid_ck} from {dst_db_table}'
        else:
            sql = f" select distinct {fid_ck} from {dst_db_table} order by {fid_ck} {order_asc} limit 1 "
        fid_ck_values = dst_table.operator(sql).values[0]
        src_table.update(**{f'{fid_ck} as src_{fid_ck}': f' {fid_ck} > {fid_ck_values}'})

        insert_sql = f"insert into {dst_db_table} {src_table}"
        return insert_sql

        # start, end = dt_ck_df.min(), dt_ck_df.max()

        # insert_sql = f"insert into {dst_db_table} {src_table}"
        # return insert_sql


# create table
# update table

# merge table

# group table
if __name__ == '__main__':
    factor = BaseSingleFactorHelper(
        'clickhouse://default:***REMOVED***@***REMOVED***:8123/raw.product_id_info',
        cols=['test1']
    )
    print(factor.empty)
    print(factor.table_engine)
    print(f"{factor}")
    pass
