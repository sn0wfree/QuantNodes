# coding=utf-8
import inspect
from functools import wraps

import pandas as pd

from Nodes.basic.basic_node import BasicNode
from Nodes.conf_node.load_settings_node import ClickHouseSettings
from Nodes.data_node._ConnectionParser import ConnectionParser
from Nodes.utils_node.lazy_load import LazyInit

upload_code = 'ol.p;/'


class _ClickHouseTableBaseNode(BasicNode):
    def __init__(self, table, conn):
        """

        :param conn:
        :param table:
        :param db:
        """
        super(_ClickHouseTableBaseNode, self).__init__(table)
        self.conn = conn
        self.db = conn._para.db
        self.table = table

    def query(self, sql):
        if sql.strip(' \n\t').lower()[:4] in ['sele', 'desc', 'show']:
            return self.conn.get(sql, )
        else:
            return self.conn.insert_query(sql)

    def __call__(self, func):
        if 'conn' in inspect.getfullargspec(func).args:
            pass
        else:
            raise TypeError('conn arguments not exists!')
        conn_decorator = self

        @wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs.get('conn') is None:
                kwargs['conn'] = conn_decorator
            else:
                pass
            return func(*args, **kwargs)
        return wrapper



    # @property
    # def columns(self):
    #     table = self.table
    #     db = self.db
    #     # sql = f'select * from  {db}.{table} limit 1'
    #     sql = f'describe {db}.{table} '
    #     cols = self.query(sql)['name'].to_list()  # .columns.values.tolist()
    #     return cols

    def _get_data(self, item: (list, tuple, str)):

        table = self.table
        db = self.db
        if item != '*':
            columns = ','.join(item)
        else:
            columns = item
        sql = f"select {columns} from {db}.{table}"
        return self.query(sql)

    def __getitem__(self, sql: str):

        return self.query(sql)

    def __setitem__(self, upload_key: str, df: pd.DataFrame):
        """

        :param key:  upload_code
        :param df:
        :return:
        """
        if upload_key != upload_code:
            raise KeyError('upload key invalid! insert method require checksum key for the safety consideration!')
        else:
            self.conn.insert(df, self.table, db=self.db)


class ClickHouseTableNode(_ClickHouseTableBaseNode):
    def __init__(self, table: str, settings):
        """

        :param settings:
        :param table:
        """
        if settings is None:
            settings = ClickHouseSettings().get()
        settings, conn = ConnectionParser.checker_multi_and_create(table, settings, target_db_type='ClickHouse')
        # conn = MysqlConnEnforcePandas('test_clickhouse', **settings)
        # db = settings['db'] if db is None else db
        super(ClickHouseTableNode, self).__init__(table, conn, )
        self._para = settings
        self.table_name = table
        self.db = settings['db']



    # def __run__(self, sql):
    #     return self.query(sql)


class ClickHouseDBPool(LazyInit):  # lazy load to improve loading speed
    def __init__(self, db: str = 'default', settings: (str, dict, object, None) = None):
        super(ClickHouseDBPool, self).__init__()

        if settings is None:
            settings = ClickHouseSettings().get()
        if db is not None:
            settings['db'] = db
        else:
            db = settings['db']
        self.db = db
        settings, conn = ConnectionParser.checker_multi_and_create(db, settings, target_db_type='ClickHouse')
        self._settings = settings
        self._conn = conn
        for table in self.tables:
            if table != 'tables':
                try:
                    setattr(self, table, ClickHouseTableNode(table, self._conn))
                except Exception as e:
                    print(str(e))
                    pass

    @property
    def tables(self):
        tables = [table[0] for table in self._conn.SHOWTABLES()]
        return tables

    def __getitem__(self, table: str):
        return getattr(self, table)

    def __setitem__(self, table: str, table_obj):
        setattr(self, table, table_obj)


if __name__ == '__main__':
    db = 'default'
    ch_test = None
    test_clickhouse = ClickHouseDBPool(db, ch_test)
    print(test_clickhouse.user_test.query('select * from user_test limit 100'))
    pass
