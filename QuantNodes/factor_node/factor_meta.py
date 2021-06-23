# coding=utf-8


class FactorDataBase(object):
    __Name__ = "基础因子库基类_database"

    # ------------------------------数据源操作---------------------------------

    def __connection__(self):
        pass

    def __status__(self):
        pass

    def __available__(self):
        pass


class FactorTableBase(FactorDataBase):
    __Name__ = "基础因子库因子表"

    def name(self):
        return self.__Name__


class MetaFactorTable(metaclass=FactorTableBase):
    pass


if __name__ == '__main__':
    pass
