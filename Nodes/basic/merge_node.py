# coding=utf-8
test1 = 29


class MergeNode(object):
    def __init__(self, node, init_kwargs):
        self.node = node(**init_kwargs)

    def __start__(self, **kwargs):
        return self.node.__run__(**kwargs)

        pass

    def end_start(self):
        pass


if __name__ == '__main__':
    pass
