# # coding=utf8
#
# from collections import namedtuple
#
# import redis
# from Nodes.basic.basic_node import BasicNode
#
# ## todo uncompletd
# redis_conn_tuple = namedtuple('redis_conn', ['host', 'port', 'user', 'passwd', 'charset', 'db'])
#
#
# # class RedisControlor(object):
# #     def __init__(self, host='120.78.81.81', port=6379, password='eef1ef8031e75ca1849c6a590f10ccb0', db=0):
# #         self._redis_instance = RedisConenctor(host=host, port=port, password=password, db=db)
# #
# #         self.RedisList = RedisList(self._redis_instance)
#
#
# class RedisConnect(redis.Redis):
#     def __init__(self, host='120.78.81.81', port=6379, passwd='eef1ef8031e75ca1849c6a590f10ccb0', db=0):
#         # self.redis = redis.Redis.from_url('redis://{}@{}:{}/{}'.format(password, host, port, db))
#         super(RedisConnect, self).__init__(host=host, port=port, password=passwd, db=db)
#         user = None
#         charset = 'utf-8'
#         self._para = redis_conn_tuple(host, port, user, passwd, charset, db)
#
#
# operators = {'strings': {'GET': 'get', 'SET': 'set'},
#              'lists': {'GET': 'lindex', 'SET': 'lpush'},
#              'sets': {'GET': 'sunion', 'SET': 'sadd'},
#              'hashes': {'GET': 'hget', 'SET': 'hset'}
#              }
#
#
# class RedisBaseNode(BasicNode):
#     def __init__(self, type_name, conn):
#         super(RedisBaseNode, self).__init__(type_name)
#         # settings = dict(host = '120.78.81.81', port = 6379, password = 'XXXX', db = 0)s
#         self._conn = conn
#         self.table_name = type_name
#         self._table_name_lower = self.table_name.lower()
#         self.db = conn._para.db
#         self._get, self._set = self._opt
#
#         # self._para = ConnectionParser
#         pass
#
#     @property
#     def _opt(self):
#         return operators[self.dtype]
#
#     @property
#     def dtype(self):
#         dtypes = list(operators.keys())  # ['strings', 'lists', 'sets', 'hashes']
#         for dty in dtypes:
#             if self._table_name_lower.startswith(dty):
#                 return dty
#             else:
#                 pass
#         else:
#             raise ValueError(
#                 'The given type is not supported yet! currently supported types are: {}'.format(',\n'.join(dtypes)))
#
#     def query(self, cmd, *key, **kwargs):
#         return getattr(self._conn, cmd)(*key, **kwargs)
#
#     def __getitem__(self, *item, **kwargs):
#
#         return self.query(self._get, *item, **kwargs)
#
#     def __setitem__(self, *item, **kwargs):
#
#         return self.query(self._set, *item, **kwargs)
#
#     @staticmethod
#     def byte2str(b):
#         return b.decode('utf8') if isinstance(b, bytes) else b
#
#
# class RedisListNode(RedisBaseNode):
#     def __init__(self, redisinstance):
#         super(RedisListNode, self).__init__()
#         self.redisinstance = redisinstance
#
#     def create(self, listname, *values):
#         self.redisinstance.lpush(listname, *values)
#
#     def add(self, listname, *values):
#         self.redisinstance.lpush(listname, *values)
#
#     def pop(self, listname, method='blpop'):
#         return getattr(self.redisinstance, method)(listname)
#
#     def length(self, listname):
#         return self.redisinstance.llen(listname)
#
#
# # __setitem__
# # query
#
#
# # class RedisConenctor(redis.Redis):
# #     def __init__(self, host='120.78.81.81', port=6379, password='eef1ef8031e75ca1849c6a590f10ccb0', db=0):
# #         # self.redis = redis.Redis.from_url('redis://{}@{}:{}/{}'.format(password, host, port, db))
# #         super(RedisConenctor, self).__init__(host=host, port=port, password=password, db=db)
# #         # self.r = redis.Redis()
# #
# #     def excute_command(self, commd, *key):
# #         return getattr(self, commd)(*key)
# #
# #     def create_list(self, listname, *values):
# #         self.lpush(listname, *values)
# #
# #     def add_value_into_list(self, listname, *values):
# #         self.create_list(listname, *values)
# #
# #     def obtain_a_task_once(self, redis_key, *para):
# #         return self.lpop(redis_key)
# #
# #     def obtain_a_task_else_waiting(self, redis_key, timeout=10):
# #         return self.blpop(redis_key, timeout)
# #
# #     @staticmethod
# #     def byte2str(b):
# #         return b.decode('utf8') if isinstance(b, bytes) else b
#
#
# if __name__ == '__main__':
#     # tasks_df = load_task()
#     rr = RedisBaseNode()
#     # rr.create_list('LagouTaskstest', *tasks_df.values.ravel())
#     rr.add_value_into_list('b', 1)
#     retu = rr.obtain_a_task_else_waiting('b', 1)
#     print(retu, retu is not None)
#     # print(type(retu), type(retu.decode('utf8')))
#     print(isinstance(retu, bytes))
