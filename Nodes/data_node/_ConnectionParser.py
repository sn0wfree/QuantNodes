# coding=utf-8
from collections import namedtuple

from Nodes.utils_node.MySQLConn_v004_node import MySQLNode, MySQLNodeName
from Nodes.utils_node.ch2pandas_node import ClickHouseNode, ClickHouseNodeName

conn_info_tuple = namedtuple('db_info', ['db_type', 'db_para_dict'])

connect_func = {'MySQL': (MySQLNode, MySQLNodeName), 'ClickHouse': (ClickHouseNode, ClickHouseNodeName)}


class ConnectionParser(object):
    available_head = ['MySQL://', 'ClickHouse://']

    @classmethod
    def detect_db_type(cls, url_str):
        for head in cls.available_head:
            if url_str.startswith(head):
                return head[:-3], url_str.split(head)[-1]
            else:
                pass
        else:
            raise ValueError('unknown db type {} '.format(url_str.split('://')[0]))

    @classmethod
    def parser(cls, url_str):
        db_type, rest = cls.detect_db_type(url_str)
        user_info, host_info = rest.split('@')
        user, passwd = user_info.split(':')
        host_port, db = host_info.split('/')
        host, port = host_port.split(':')
        db_dict = dict(host=host, port=int(port), db=db, user=user, password=passwd)
        return conn_info_tuple(db_type, db_dict)

    @classmethod
    def checker(cls, url_str, target_db_type):
        conn_info = cls.parser(url_str)
        db_type = conn_info.db_type
        if db_type == target_db_type:
            return db_type, conn_info.db_para_dict
        else:
            raise ValueError(
                'this is {target_db_type} Node, but settings receive no {target_db_type} settings: {db_type}'.format(
                    target_db_type=target_db_type, db_type=db_type))

    @classmethod
    def checker_multi_type(cls, settings: (str, dict), target_db_type: str):
        if isinstance(settings, dict):
            settings = settings
        elif isinstance(settings, str):
            db_type, db_para_dict = cls.checker(settings, target_db_type=target_db_type)
            settings = db_para_dict
        else:
            raise ValueError('unknown settings format, only accept special str or dict format')
        return settings

    @classmethod
    def checker_multi_and_create(cls, name, settings: (str, dict, object), target_db_type: str):
        if isinstance(settings, (str, dict)):
            settings = cls.checker_multi_type(settings, target_db_type)
            create_func, func_name = connect_func[target_db_type]
            conn = create_func(name, **settings)

        elif isinstance(settings, (MySQLNode, ClickHouseNode)):
            conn = settings
            settings = settings._para._asdict()
        else:
            raise ValueError('unsupported settings format! please use str dict or instanced object!')
        return settings, conn


if __name__ == '__main__':
    url_str = "ClickHouse://{user}:{passwd}@{host}:1000/{db}"
    print(ConnectionParser.parser(url_str))
    pass
