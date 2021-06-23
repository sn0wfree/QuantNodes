#coding=utf-8
import unittest

from QuantNodes.database_node.clickhouse_node import SQLBuilder, ClickHouseDBPool


class MyTestCase(unittest.TestCase):

    @property
    def nodes(self):
        p = {'host': '***REMOVED***', 'port': 8123, 'user': 'default', 'password': '***REMOVED***', 'db': 'default'}

        # sql = 'select * from default.user_test limit 10000 '
        r2 = ClickHouseDBPool(p)
        table_node = r2.default.user_test
        return table_node

    def base_equal_test(self, sql, others):
        nodes = self.nodes
        df = nodes.query(sql)
        sql2 = SQLBuilder.selector(**others)
        df2 = nodes.query(sql2)
        print(sql, '\n', sql2)
        self.assertEqual(True, (df == df2).all().all())

    def test_sql1(self):
        sql = 'select user_name, count(1) from default.test2 group by user_name limit 19'

        others = dict(DB_TABLE='default.test2', cols=['user_name', 'count(1)'], group_by=['user_name'],
                      limit=19)

        self.base_equal_test(sql, others)

    def test_sql2(self):
        sql = 'select user_name* from default.test2 order by user_name  limit 19'

        others = dict(DB_TABLE='default.test2', cols=['user_name'], limit=19,order_by = ['user_name'])

        self.base_equal_test(sql, others)


if __name__ == '__main__':
    unittest.main()
