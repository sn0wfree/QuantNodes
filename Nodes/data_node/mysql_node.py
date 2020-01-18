# coding=utf-8
"""
data (table) node require following methods：

__getitem__
__setitem__
query


"""
import pandas as pd

from Nodes.basic.basic_node import BasicNode
from Nodes.conf_node.load_settings_node import MySQLSettings
from Nodes.data_node import ConnectionParser

upload_code = 'ol.p;/'


class _MySQLTableBaseNode(BasicNode):
    def __init__(self, table, conn):
        """

        :param conn:
        :param table:
        :param db:
        """
        super(_MySQLTableBaseNode, self).__init__(table)
        self.conn = conn
        self.db = conn._para.db
        self.table = table

    def query(self, sql):
        return self.conn.sql2data(sql)

    @property
    def columns(self):
        table = self.table
        db = self.db
        # sql = f'select * from  {db}.{table} limit 1'
        sql = f'desc  {db}.{table} '
        cols = self.query(sql)['Field'].to_list()  # .columns.values.tolist()
        return cols

    def _get_data(self, cols: (list, tuple)):
        table = self.table
        db = self.db
        columns = ','.join(cols)
        sql = f"select {columns} from {db}.{table}"
        return self.query(sql)

    def __getitem__(self, sql: (list, tuple, str)):
        # table = self.table
        # db = self.db
        # if item != '*':
        #     columns = ','.join(item)
        # else:
        #     columns = item
        # sql = f"select {columns} from {db}.{table}"
        return self.query(sql)

    def __setitem__(self, upload_key: str, df: pd.DataFrame):
        if upload_key != upload_code:
            raise KeyError('upload key invalid! insert method require checksum key for the safety consideration!')
        else:
            self.conn.df2sql(df, self.table, db=self.db, csv_store_path='/tmp/', auto_incre_col=False, rm_csv=True)


class MySQLTableNode(_MySQLTableBaseNode):
    def __init__(self, table, settings, ):
        """

        :param settings:
        :param table:
        :param db:
        """
        settings, conn = ConnectionParser.checker_multi_and_create(table, settings, target_db_type='MySQL')
        # conn = MysqlConnEnforcePandas('test_clickhouse', **settings)
        # db = settings['db'] if db is None else db
        super(MySQLTableNode, self).__init__(table, conn, )
        self._para = settings
        self.table_name = table
        self.db = settings['db']

    # def __run__(self, sql):
    #     return self.query(sql)


class MySQLDBPool(BasicNode):
    def __init__(self, db: str, settings: (str, dict, object) = None):
        super(MySQLDBPool, self).__init__(db)
        self.db = db
        if settings is None:
            settings = MySQLSettings().get()
        settings, conn = ConnectionParser.checker_multi_and_create(db, settings, target_db_type='MySQL')
        self._settings = settings
        self._conn = conn
        for table in self.tables:
            setattr(self, table, MySQLTableNode(table, self._conn))
        # conn = MysqlConnEnforcePandas('test_clickhouse', **settings)

    @property
    def tables(self):
        tables = [table[0] for table in self._conn.SHOWTABLES()]
        return tables

    # def __run__(self, table, sql):
    #     return getattr(self, table).query(sql)
    def __getitem__(self, table: str):
        # table = self.table
        # db = self.db
        # if item != '*':
        #     columns = ','.join(item)
        # else:
        #     columns = item
        # sql = f"select {columns} from {db}.{table}"
        return getattr(self, table)

    def __setitem__(self, table: str, table_obj):
        setattr(self, table, table_obj)


if __name__ == '__main__':
    db = 'test_clickhouse'
    mysql_test = None  #
    # conn = MysqlConnEnforcePandas('test_clickhouse', **mysql_test)
    # print(conn.DetectConnectStatus())
    test_clickhouse = MySQLDBPool(db, mysql_test)
    print(test_clickhouse.user_test)

    pass
