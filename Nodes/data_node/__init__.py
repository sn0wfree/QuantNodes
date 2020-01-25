# coding=utf-8

from Nodes.data_node.clickhouse_node import ClickHouseDBPool, ClickHouseTableNode, _ClickHouseTableBaseNode
from Nodes.data_node.mysql_node import MySQLDBPool, MySQLTableNode, _MySQLTableBaseNode

ACCEPT_POOL_TYPE = {'MySQL': (MySQLTableNode, MySQLDBPool, _MySQLTableBaseNode, 'user', 'mysql'),
                    'ClickHouse': (
                        ClickHouseTableNode, ClickHouseDBPool, _ClickHouseTableBaseNode, 'columns', 'system')}


class CreateNodes(object):
    @staticmethod
    def _create_table(table, pool_type='MySQL'):
        if pool_type in ACCEPT_POOL_TYPE.keys():
            pass
        else:
            raise ValueError('wrong pool_type: {} , only accept {}'.format(pool_type, ','.join(ACCEPT_POOL_TYPE)))
        create_func = ACCEPT_POOL_TYPE.get(pool_type)[0]
        # default_db = ACCEPT_POOL_TYPE.get(pool_type)[1]

        # table, settings
        # init
        init = create_func(table, settings=None)
        # dff = init.query('show databases').values.ravel().tolist()

        # args = {}
        return init

    @staticmethod
    def _create_db_node(db, pool_type='MySQL'):
        if pool_type in ACCEPT_POOL_TYPE.keys():
            pass
        else:
            raise ValueError('wrong pool_type: {} , only accept {}'.format(pool_type, ','.join(ACCEPT_POOL_TYPE)))
        create_func = ACCEPT_POOL_TYPE.get(pool_type)[1]
        return create_func(db, settings=None)

    @classmethod
    def _create_server_node(cls, name: str, pool_type='MySQL'):
        create_func = ACCEPT_POOL_TYPE.get(pool_type)[0]
        table = ACCEPT_POOL_TYPE.get(pool_type)[3]
        # db = ACCEPT_POOL_TYPE.get(pool_type)[4]

        # table, settings
        # init
        init = create_func(table, settings=None)
        dbs = init.query('show databases').values.ravel().tolist()
        del init
        u = {db: cls._create_db_node(db, pool_type=pool_type) for db in dbs}
        u['databases'] = dbs
        return type(name, (), u)

    @classmethod
    def create(cls, name: str, pool_type='MySQL'):
        """

        :param name:
        :param pool_type:
        :return:
        """
        return cls._create_server_node(name, pool_type)


if __name__ == '__main__':
    import time

    # test_clickhouse = ClickHouseDBPool(db, ch_test)
    # c = ClickHouseDBPool('default', None)
    # df = c.user_test.query('select * from user_test limit 100')
    t = time.time()
    a = CreateNodes.create('mysql', pool_type='MySQL')
    # user_test = a.default.user_test
    # res = user_test.query('show tables')
    # print(res)
    print(time.time() - t)

    print(1)

    pass
