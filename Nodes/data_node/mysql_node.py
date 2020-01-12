# coding=utf-8

import pandas as pd

from Nodes.data_node import ConnectionParser

upload_code = 'ol.p;/'


class MySQLTableBaseNode(object):
    def __init__(self, conn, table, ):
        """

        :param conn:
        :param table:
        :param db:
        """
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

    # def get_data(self, cols: (list, tuple)):
    #     table = self.table
    #     db = self.db
    #     columns = ','.join(cols)
    #     sql = f"select {columns} from {db}.{table}"
    #     return self.__query__(sql)

    def __getitem__(self, item: (list, tuple, str)):
        table = self.table
        db = self.db
        if item != '*':
            columns = ','.join(item)
        else:
            columns = item
        sql = f"select {columns} from {db}.{table}"
        return self.query(sql)

    def __setitem__(self, key: str, value: pd.DataFrame):
        if key != upload_code:
            raise KeyError('upload key invalid! insert method require checksum key for the safety consideration!')
        else:
            self.conn.df2sql(value, self.table, db=self.db, csv_store_path='/tmp/', auto_incre_col=False, rm_csv=True)


class MySQLTableNode(MySQLTableBaseNode):
    def __init__(self, settings, table):
        """

        :param settings:
        :param table:
        :param db:
        """
        settings, conn = ConnectionParser.checker_multi_and_create(table, settings, target_db_type='MySQL')
        # conn = MysqlConnEnforcePandas('test_clickhouse', **settings)
        # db = settings['db'] if db is None else db
        super(MySQLTableNode, self).__init__(conn, table)
        self._para = settings
        self.table_name = table
        self.db = settings['db']


class MySQLDBPool(object):
    def __init__(self, db, settings):
        self.db = db
        settings, conn = ConnectionParser.checker_multi_and_create(db, settings, target_db_type='MySQL')
        self._settings = settings
        self._conn = conn
        for table in self.tables:
            setattr(self, table, MySQLTableNode(self._conn, table))
        # conn = MysqlConnEnforcePandas('test_clickhouse', **settings)

    @property
    def tables(self):
        tables = self._conn.SHOWTABLES()

        return [table[0] for table in tables]

    pass


if __name__ == '__main__':
    db = 'test_clickhouse'
    mysql_test = dict(host='***REMOVED***', port=3306, user='linlu', passwd='***REMOVED***', db='test_clickhouse')
    # conn = MysqlConnEnforcePandas('test_clickhouse', **mysql_test)
    # print(conn.DetectConnectStatus())
    c1 = MySQLDBPool(db, mysql_test)
    print(c1.columns)

    pass
