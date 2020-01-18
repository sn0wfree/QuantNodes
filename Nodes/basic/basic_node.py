# coding=utf-8
import time

from Nodes.utils_node.generate_str_node import random_str
from Nodes.utils_node.lazy_load import LazyInit


class BasicNode(LazyInit):
    def __init__(self, name, random_task_id=False):
        if random_task_id:
            random_task_id = random_str() + "_" + str(int(time.time()))
            name = name + '_' + random_task_id
        else:
            pass

        self._node_name_ = name
        self._node_args = None
        self._node_setup = False

    def __getitem__(self, item):
        pass

    def __setitem__(self, item, value):
        pass

    def __setup__(self, method, *args, **kwargs):
        self._node_args = (method, args, kwargs)
        self._node_setup = True

    def update(self, all_args, replace=False):
        method, args, kwargs = all_args['method'], all_args['args'], all_args['kwargs']
        self.__setup__(method, *args, **kwargs)
        if replace:
            pass
        else:
            return self

    def __detect_mode__(self, raw_obj):
        pass

    def __check_instance_status__(cls, obj1):
        pass

    def __run__(self):
        if self._node_setup:
            method, args, kwargs = self._node_args
            if method.lower() == 'get':
                return self.__getitem__(*args, **kwargs)
            elif method.lower() == 'set':
                return self.__setitem__(*args, **kwargs)
            else:
                raise ValueError('unknown method')
        else:
            raise ValueError('setup incompleted!')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(exc_type, exc_val, exc_tb)
        pass

    def __lshift__(self, *args):  # real signature unknown
        """ Return self<<value. """
        pass


if __name__ == '__main__':
    pass
