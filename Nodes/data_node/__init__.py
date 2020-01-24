# coding=utf-8

from Nodes.data_node.clickhouse_node import ClickHouseDBPool
from Nodes.data_node.mysql_node import MySQLDBPool

ACCEPT_POOL_TYPE = {'MySQL': (MySQLDBPool,), 'ClickHouse': (ClickHouseDBPool, 'default')}


def create_pool(pool_type='MySQL'):
    if pool_type in ACCEPT_POOL_TYPE.keys():
        pass
    else:
        raise ValueError('wrong pool_type: {} , only accept {}'.format(pool_type, ','.join(ACCEPT_POOL_TYPE)))
    create_func = ACCEPT_POOL_TYPE.get(pool_type)[0]
    default_db = ACCEPT_POOL_TYPE.get(pool_type)[1]

    # init
    init = create_func(db=default_db, settings=None)
    init

    args = {}

    return type(name, base, args)


if __name__ == '__main__':
    pass
