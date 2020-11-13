# coding=utf-8
import copy
import re
from collections import namedtuple, ChainMap

from ClickSQL import ClickHouseTableNodeExt

factor_parameters = ('dt', 'code', 'value', 'fid')
ft_node = namedtuple('factortable', factor_parameters)


class FactorBackendCH(object):
    __slots__ = ['_src', 'node', 'db_table']

    def __init__(self, src: str):
        """

        :param src:  sample: clickhouse://test:sysy@199.199.199.199:1234/drre
        """
        self._src = src
        self.node = ClickHouseTableNodeExt(conn_str=src)
        self.db_table = self.node._para.database

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
    def _get_sql(cls, db_table: str, cols: (tuple, None, list) = None,
                 order_by_cols: (list, tuple, None) = None,
                 data_filter: dict = {}, include_filter=True,
                 **other_filters):
        """

        :param data_filter:
        :param cols:
        :param include_filter:
        :param other_filters:
        :param order_by_cols: ['test1 asc','test2 desc']
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
        if order_by_cols is None:
            order_by_clause = ''
        else:
            order_by_clause = f" order by ({','.join(order_by_cols)})"
        sql = f"select {','.join(cols)} from {db_table} where {' and '.join(sorted(set(['1'] + list(filter_yield))))} {order_by_clause}"
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


"add auto-increment col by materialized bitOr(bitShiftLeft(toUInt64(now64()),24), rowNumberInAllBlocks()) "


class BaseSingleFactorNode(object):
    __Name__ = "基础因子库单因子基类"
    __slots__ = (
        'operator', 'db', 'table', 'db_table', '_kwargs', '_raw_kwargs', 'status', '_INFO', 'depend_tables',
        '_fid_ck', '_dt_max_1st', '_execute', '_no_self_update'
    )

    def __init__(self, src: str, db_table: (None, str) = None, info=None,
                 fid_ck: str = 'fid',
                 dt_max_1st: bool = True,
                 execute: bool = False,
                 no_self_update: bool = True, **kwargs):
        """

        :type kwargs: object
        :param src: string sample: clickhouse://test:sysy@199.199.199.199:1234/drre
        :param db_table:
        :param info:
        :param kwargs:  data_filter will store operator for some cols
        """

        self.operator = FactorBackendCH(src)
        self._fid_ck = fid_ck
        self._dt_max_1st = dt_max_1st
        self._execute = execute
        self._no_self_update = no_self_update

        # self._execute = self._operator._execute

        if db_table is None:
            src_db_table = self.operator.db_table
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
        self.status = 'SQL'
        self._INFO = info

    def groupby(self, by: (str, list, tuple), apply_func: (list,), having: (list, tuple, None) = None, execute=False):

        if isinstance(by, str):
            by = [by]
            group_by_clause = f"group by {by}"
        elif isinstance(by, (list, tuple)):
            group_by_clause = f"group by ({','.join(by)})"
        else:
            raise ValueError(f'by only accept str list tuple! but get {type(by)}')

        db_table_or_sql = self.__sql__

        if having is None:
            having_clause = ''
        elif isinstance(having, (list, tuple)):
            having_clause = 'having ' + " and ".join(having)
        else:
            raise ValueError(f'having only accept list,tuple,None! but get {type(having)}')

        sql = f"select  {','.join(by + apply_func)}  from ({db_table_or_sql}) {group_by_clause} {having_clause} "
        if execute:
            self.operator(sql)
        else:
            return sql

    # create table
    def create(self, *args, **kwargs):
        return self.operator.node.create(*args, **kwargs)

    def update(self, **kwargs):
        self._kwargs.update(kwargs)

    def __str__(self):
        return self.__sql__

    @property
    def __sql__(self):
        return self.operator._get_sql(db_table=self.db_table, **self._kwargs)

    @property
    def __factor_id__(self):  # add iid function get factor table id
        return hash(self.__sql__)

    @property
    def total_rows(self):
        """
        return row count
        :return:
        """

        # sql = f"-- select count(1) as row_count from {self.db_table}"
        temp = self.__system_tables__
        if not temp.empty:
            return temp['total_rows'].values[0]
        else:
            raise ValueError(f'{self.db_table} is not exists!')

    @property
    def __system_tables__(self):
        sql = f"select total_rows,engine from system.tables where database ='{self.db}' and name='{self.table}'"
        res = self.operator(sql)
        return res

    @property
    def table_exist(self):
        """
        return table exists status
        :return:
        """

        return not self.__system_tables__.empty

    @property
    def table_engine(self):
        """
        return table engine
        :return:
        """

        return self.__system_tables__['engine'].values[0]

    @property
    # @timer
    def empty(self):
        return self.total_rows == 0

    def fetch(self, pattern=r'[\s]+limit[\s]+[0-9]+$'):
        """
        fetch first 1000 line
        :return:
        """
        sql = self.__sql__
        end_with_limit = self.operator._check_end_with_limit(sql, pattern=pattern)
        if end_with_limit:
            return self.operator(sql)
        else:
            return self.operator(sql + ' limit 1000')

    def fetch_all(self):
        """
        fetch all data
        :return:
        """

        return self.operator(self.__sql__)

    # def __call__(self, **kwargs):
    #     """
    #
    #     :param kwargs:
    #     :return:
    #     """
    #     self.update(**kwargs)
    #     if self.status == 'SQL':
    #         return self.__sql__
    #     elif self.status == 'SQL:fetch':
    #         return self.fetch()
    #     elif self.status == 'SQL:fetch_all':
    #         return self.fetch_all()
    #     else:
    #         raise ValueError('status code is not supported!')


class UpdateSQLUtils(object):

    @staticmethod
    def full_update(src_db_table: BaseSingleFactorNode, dst_db_table: BaseSingleFactorNode, **kwargs):
        # dst_db_table = dst_db_table.db_table
        # dst_db, dst_table = dst_db_table.db, dst_db_table.table
        dst_table_type = dst_db_table.table_engine
        dst = dst_db_table.db_table
        if dst_table_type == 'View':
            raise ValueError(f'{dst} is View ! cannot be updated!')
        insert_sql = f"insert into {dst} {src_db_table}"
        return insert_sql

    @staticmethod
    def incremental_update(src_db_table: BaseSingleFactorNode, dst_db_table: BaseSingleFactorNode,
                           fid_ck: str, dt_max_1st=True, inplace=False, **kwargs):
        # src_db_table = src_table.db_table
        # src_table_type = src_db_table.table_engine
        dst_table_type = dst_db_table.table_engine
        dst = dst_db_table.db_table
        if dst_table_type == 'View':
            raise ValueError(f'{dst} is View ! cannot be updated!')
        if dt_max_1st:
            order_asc = ' desc'
        else:
            order_asc = ' asc'
        sql = f" select distinct {fid_ck} from {dst} order by {fid_ck} {order_asc} limit 1 "
        fid_ck_values = src_db_table.operator(sql).values.ravel().tolist()[0]
        if inplace:
            src_db_table.update(**{f'{fid_ck} as src_{fid_ck}': f' {fid_ck} > {fid_ck_values}'})
            insert_sql = f"insert into {dst} {src_db_table}"
        else:
            src_db_table_copy = copy.deepcopy(src_db_table)
            src_db_table_copy.update(**{f'{fid_ck} as src_{fid_ck}': f' {fid_ck} > {fid_ck_values}'})
            insert_sql = f"insert into {dst} {src_db_table_copy}"

        return insert_sql


class BaseSingleFactorTableNode(BaseSingleFactorNode):

    def __call__(self, sql, **kwargs):
        return self.operator(sql, **kwargs)

    # update table
    def __lshift__(self, src_db_table: BaseSingleFactorNode):
        print('lshift')
        fid_ck = self._fid_ck
        dt_max_1st = self._dt_max_1st
        execute = self._execute
        no_self_update = self._no_self_update

        if isinstance(src_db_table, str):
            src_conn = copy.deepcopy(self.operator._src).replace(self.db_table, src_db_table)
            src_db_table = BaseSingleFactorNode(src_conn, cols=['*'])
        elif isinstance(src_db_table, BaseSingleFactorNode):
            pass
        else:
            raise ValueError('src_db_table is not valid! please check!')

        if src_db_table.empty:
            raise ValueError(f'{src_db_table.db_table} is empty')
        # check two table are same
        if no_self_update and self.db_table == src_db_table.db_table and self.__factor_id__ == src_db_table.__factor_id__:
            dst = src_db_table.db_table
            src = self.db_table
            raise ValueError(
                f'Detect self-update process! these operator attempts to update data from {src} to {dst}')

        update_status = 'full' if self.empty else 'incremental'

        func = getattr(UpdateSQLUtils, f'{update_status}_update')
        sql = func(src_db_table, self.db_table, fid_ck, dt_max_1st=dt_max_1st)
        if execute:
            self.operator(sql)
        return sql, update_status

    # update table
    def __rshift__(self, dst_db_table: str):
        """

        UpdateSQLUtils

        :param dst_db_table:
        :param fid_ck:
        :param dt_max_1st:
        :param execute:
        :param no_self_update:
        :return:
        """
        # print('rshift')
        fid_ck = self._fid_ck
        dt_max_1st = self._dt_max_1st
        execute = self._execute
        no_self_update = self._no_self_update
        if self.empty:
            raise ValueError(f'{self.db_table} is empty')

        if isinstance(dst_db_table, str):
            dst_conn = copy.deepcopy(self.operator._src)
            dst_db_table = BaseSingleFactorNode(
                dst_conn.replace(self.db_table, dst_db_table),
                cols=['*']
            )
        elif isinstance(dst_db_table, BaseSingleFactorNode):
            pass
        else:
            raise ValueError('dst_db_table is not valid! please check!')
        # check two table are same
        if no_self_update and self.db_table == dst_db_table.db_table:
            if self.__factor_id__ == dst_db_table.__factor_id__:
                dst = dst_db_table.db_table
                src = self.db_table
                raise ValueError(
                    f'Detect self-update process! these operator attempts to update data from {src} to {dst}')

        update_status = 'full' if dst_db_table.empty else 'incremental'

        func = getattr(UpdateSQLUtils, f'{update_status}_update')
        sql = func(self, dst_db_table, fid_ck, dt_max_1st=dt_max_1st)
        if execute:
            self.operator(sql)
        return sql, update_status


# merge table

# group table
if __name__ == '__main__':
    factor = BaseSingleFactorTableNode(
        'clickhouse://default:***REMOVED***@***REMOVED***:8123/test.test4',
        cols=['test1'], data_filter={'test': 'test > 2'}
    )
    factor2 = BaseSingleFactorTableNode(
        'clickhouse://default:***REMOVED***@***REMOVED***:8123/test.test',
        cols=['test1']
    )
    factor >> 'test.test'
    # print(factor)
    print(factor)

    # c = factor('show tables from raw')
    # c2 = factor.groupby(['test2'], apply_func=['sum(fid)'])
    # print(c2)

    # print(1 >> 2)
    pass
