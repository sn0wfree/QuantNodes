# coding=utf-8
from abc import abstractmethod, ABCMeta
from collections import Iterable

# import random
# from QuantNodes.test import GOOG

# from QuantNodes.utils_node.file_cache import file_cache
class Strategy(metaclass=ABCMeta):
    """
    提供用户自定义界面
    """

    # def __str__(self):
    #     params = ','.join(f'{i[0]}={i[1]}' for i in zip(self._params.keys(),
    #                                                     map(_as_str, self._params.values())))
    #     if params:
    #         params = '(' + params + ')'
    #     return f'{self.__class__.__name__}{params}'

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    @abstractmethod
    def init(self, *args, **kwargs):
        """
        初始化
        :return:
        """
        pass

    @abstractmethod
    def next(self) -> Iterable:
        """
        iter operate strategy for each day
        :return:
        """
        pass