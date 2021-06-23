# coding=utf-8
import copy

from QuantNodes.database_node.clickhouse_node import ClickHouseDBNode
from QuantNodes.operator_node.SQLUtils import SQLBuilder



class Node2(BaseNode):
    def __init__(self, table_name, settings, mode='limit'):
        super(Node2, self).__init__(table_name, settings, mode=mode)
        self._sql_str = f'select * from {self.db_table}'
        pass

    @property
    def _sql(self):
        return self._sql_str  # f'select * from {self.db_table}'

    @_sql.setter
    def _sql(self, sql):
        self._sql_str = sql

    def __getitem__(self, item: str):  ## get cols
        if isinstance(item, (str, list)):
            if isinstance(item, str):
                cols_str = item
            else:
                cols_str = ','.join(item)
            db_table = self.db_table
            if self.mode == 'limit':
                limit_clause = 'limit 1000'
            else:
                limit_clause = ''
            sql = f'select {cols_str} from {db_table} {limit_clause}'
        else:
            raise ValueError('items must be str')
        self._sql = sql
        return self  # self.query(sql)

    def compute(self):
        return self.query(self._sql)


if __name__ == '__main__':
    conn_settings = "ClickHouse://default:***REMOVED***@***REMOVED***:8123/default"
    n = Node2('test2', conn_settings)
    d = n.groupby(by=['r2', 'r3'], agg_cols=['count(id)'])
    print(d.query('show tables'))
    pass
