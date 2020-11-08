# coding=utf-8
from collections import namedtuple, ChainMap

import re
from ClickSQL import ClickHouseTableNode

factor_parameters = ('dt', 'code', 'value')
ft_node = namedtuple('factortable', factor_parameters)


class FactorBackendCH(object):
    __slots__ = ['_src', 'node', 'db']

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
        self.node = ClickHouseTableNode(conn_str=src)
        self.db = self.node._para.database
        # db_settings['host'], db_settings['port'], db_settings['user'],
        # db_settings['password'], db_settings['database']

    def __cal__(self, sql: str, **kwargs):
        return self.node.query(sql, **kwargs)

    @staticmethod
    def __extend_dict_value__(conditions: dict):
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
    def _get(cls, db_table: str, itdv: (tuple, None, list) = None, data_filter: dict = {}, include_filter=True,
             **other_filters):
        """

        :param data_filter:
        :param itdv:
        :param include_filter:
        :param other_filters:
        :return:
        """
        if itdv is None:
            itdv = factor_parameters

        conditions = ChainMap(data_filter, *list(cls.__obtain_other_filter__(other_filters)))
        filter_yield = cls.__extend_dict_value__(conditions)
        if include_filter:
            cols = set(list(itdv) + list(conditions.keys()))
        else:
            cols = set(itdv)

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
    __slots__ = ('_operator', 'db_table', '_kwargs', '_status', '_INFO', '_execute', 'depend_tables')

    def __init__(self, src: str, db_table: (None, str) = None, INFO=None, **kwargs):
        """

        :param src:
        :param db_table:
        :param INFO:
        :param kwargs:  data_filter will store operator for some cols
        """

        self._operator = FactorBackendCH(src)
        self._execute = self._operator._execute

        if db_table is None:
            src_db_table = self._operator.db
            if '.' in src_db_table:
                self.db_table = src_db_table
            else:
                raise ValueError('db_table parameter get wrong type!')

        elif isinstance(db_table, str):
            self.db_table = db_table
        else:
            raise ValueError('db_table only accept str!')
        self.depend_tables = [self.db_table]
        self._kwargs = kwargs
        self._status = 'SQL'
        self._INFO = INFO

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
        return self._operator._get(db_table=self.db_table, **self._kwargs)

    def fetch(self):
        """
        fetch first 1000 line
        :return:
        """
        sql = self.__sql__
        end_with_limit = self._operator._check_end_with_limit(sql)
        if end_with_limit:
            return self._execute(sql)
        else:
            return self._execute(sql + ' limit 1000')

    def fetch_all(self):
        """
        fetch all data
        :return:
        """

        return self._execute(self.__sql__)

    def update(self, **kwargs):
        self._kwargs.update(kwargs)

    def __call__(self, **kwargs):
        self.update(**kwargs)
        if self._status == 'SQL':
            return self.__sql__
        elif self._status == 'SQL:fetch':
            return self.fetch()
        elif self._status == 'SQL:fetch_all':
            return self.fetch_all()

    # def __data__(self):
    #     return self.__query__(**self._kwargs)


if __name__ == '__main__':
    factor = BaseSingleFactorHelper(
        'clickhouse://default:***REMOVED***@***REMOVED***:8123/raw.product_id_info',
    )
    print(factor.__sql__)
    print(str(factor))
    sql = factor()
    print(sql)
    print(str(factor))

    pass
