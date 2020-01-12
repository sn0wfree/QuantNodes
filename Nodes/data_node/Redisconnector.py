# coding=utf8

import redis


class RedisList(object):
    def __init__(self, redisinstance):
        self.redisinstance = redisinstance

    def create(self, listname, *values):
        self.redisinstance.lpush(listname, *values)

    def add(self, listname, *values):
        self.redisinstance.lpush(listname, *values)

    def pop(self, listname, method='blpop'):
        return getattr(self.redisinstance, method)(listname)

    def length(self, listname):
        return self.redisinstance.llen(listname)


class RedisControlor(object):
    def __init__(self, host='120.78.81.81', port=6379, password='eef1ef8031e75ca1849c6a590f10ccb0', db=0):
        self._redis_instance = RedisConenctor(host=host, port=port, password=password, db=db)

        self.RedisList = RedisList(self._redis_instance)


class RedisConenctor(redis.Redis):
    def __init__(self, host='120.78.81.81', port=6379, password='eef1ef8031e75ca1849c6a590f10ccb0', db=0):
        # self.redis = redis.Redis.from_url('redis://{}@{}:{}/{}'.format(password, host, port, db))
        super(RedisConenctor, self).__init__(host=host, port=port, password=password, db=db)
        # self.r = redis.Redis()

    def excute_command(self, commd, *key):
        return getattr(self, commd)(*key)

    def create_list(self, listname, *values):
        self.lpush(listname, *values)

    def add_value_into_list(self, listname, *values):
        self.create_list(listname, *values)

    def obtain_a_task_once(self, redis_key, *para):
        return self.lpop(redis_key)

    def obtain_a_task_else_waiting(self, redis_key, timeout=10):
        return self.blpop(redis_key, timeout)

    @staticmethod
    def byte2str(b):
        return b.decode('utf8') if isinstance(b, bytes) else b


if __name__ == '__main__':
    # tasks_df = load_task()
    rr = RedisConenctor()
    # rr.create_list('LagouTaskstest', *tasks_df.values.ravel())
    rr.add_value_into_list('b', 1)
    retu = rr.obtain_a_task_else_waiting('b', 1)
    print(retu, retu is not None)
    # print(type(retu), type(retu.decode('utf8')))
    print(isinstance(retu, bytes))
