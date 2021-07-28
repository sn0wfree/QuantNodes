# coding=utf-8
from functools import wraps


class TaskSubclass(object):
    # __slots__ = ('run',)

    @staticmethod
    def run():
        raise NotImplementedError('subclasses must implement run')
    pass


class TaskHolder(object):

    def identify(self):
        return 'i am a task'

    def task_transform(self, decorated):
        """将函数转换成类的装饰器"""
        tsc = TaskSubclass()
        tsc.run = wraps(decorated)

        return tsc


if __name__ == '__main__':
    TH = TaskHolder()


    @TH.task_transform
    def foo(x: int):
        return x + 2


    print(foo.run(1), str(foo))

    pass
