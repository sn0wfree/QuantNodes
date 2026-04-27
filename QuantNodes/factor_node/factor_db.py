# coding=utf-8
"""因子数据库

包含 FactorDB（只读接口）和 WritableFactorDB（可写入接口）
"""
import pandas as pd
from traits.api import Str

from QuantNodes.core.quant_nodes_object import QuantNodesObject as _QN_Object
from QuantNodes.core.base import FactorError


class FactorDB(_QN_Object):
    """因子库（只读接口）

    数据库由若干张因子表组成。
    不支持某个操作时，方法产生错误。
    没有相关数据时，方法返回 None。
    """
    Name = Str("因子库")

    # ------------------------------数据源操作---------------------------------
    def connect(self):
        """链接到数据库"""
        return 0

    def disconnect(self):
        """断开到数据库的链接"""
        return 0

    def isAvailable(self):
        """检查数据库是否可用"""
        return True

    # -------------------------------表的操作---------------------------------
    @property
    def TableNames(self):
        """表名，返回: [表名]"""
        return []

    def getTable(self, table_name, args={}):
        """返回因子表对象"""
        return None

    def getID(self):
        """获取 ID 序列"""
        return []

    def getDateTime(self):
        """获取时间点序列"""
        return []


class WritableFactorDB(FactorDB):
    """可写入的因子数据库"""

    # -------------------------------表的操作---------------------------------
    def renameTable(self, old_table_name, new_table_name):
        """重命名表。必须具体化"""
        return 0

    def deleteTable(self, table_name):
        """删除表。必须具体化"""
        return 0

    def setTableMetaData(self, table_name, key=None, value=None, meta_data=None):
        """设置表的元数据。必须具体化"""
        return 0

    # --------------------------------因子操作-----------------------------------
    def renameFactor(self, table_name, old_factor_name, new_factor_name):
        """对一张表的因子进行重命名。必须具体化"""
        return 0

    def deleteFactor(self, table_name, factor_names):
        """删除一张表中的某些因子。必须具体化"""
        return 0

    def setFactorMetaData(self, table_name, ifactor_name, key=None, value=None, meta_data=None):
        """设置因子的元数据。必须具体化"""
        return 0

    def writeData(self, data, table_name, if_exists="update", data_type={}, **kwargs):
        """写入数据。必须具体化"""
        return 0

    # -------------------------------数据变换------------------------------------
    def offsetDateTime(self, lag, table_name, factor_names, args={}):
        """时间平移，沿着时间轴将所有数据纵向移动 lag 期"""
        if lag == 0:
            return 0
        FT = self.getTable(table_name, args=args)
        Data = FT.readData(
            factor_names=factor_names,
            ids=self.getID(),
            dts=self.getDateTime(),
            args=args,
        )
        if lag > 0:
            Data.iloc[:, lag:, :] = Data.iloc[:, :-lag, :].values
            Data.iloc[:, :lag, :] = None
        elif lag < 0:
            Data.iloc[:, :lag, :] = Data.iloc[:, -lag:, :].values
            Data.iloc[:, :lag, :] = None
        DataType = FT.getFactorMetaData(
            factor_names, key="DataType", args=args
        ).to_dict()
        self.deleteFactor(table_name, factor_names)
        self.writeData(Data, table_name, data_type=DataType)
        return 0

    def changeData(self, table_name, factor_names, ids, dts, args={}):
        """数据变换，通过某种变换函数得到新的时间序列和ID序列"""
        FT = self.getTable(table_name, args=args)
        Data = FT.readData(
            factor_names=factor_names, ids=ids, dts=dts, args=args
        )
        DataType = FT.getFactorMetaData(
            factor_names, key="DataType", args=args
        ).to_dict()
        self.deleteFactor(table_name, factor_names)
        self.writeData(Data, table_name, data_type=DataType)
        return 0

    def fillNA(self, filled_value, table_name, factor_names, ids, dts, args={}):
        """填充缺失值"""
        Data = self.getTable(table_name).readData(
            factor_names=factor_names, ids=ids, dts=dts, args=args
        )
        Data.fillna(filled_value, inplace=True)
        self.writeData(Data, table_name, if_exists="update")
        return 0

    def replaceData(self, old_value, new_value, table_name, factor_names, ids, dts, args={}):
        """替换数据"""
        Data = self.getTable(table_name).readData(
            factor_names=factor_names, ids=ids, dts=dts, args=args
        )
        Data = Data.where(Data != old_value, new_value)
        self.writeData(Data, table_name, if_exists="update")
        return 0

    def optimizeData(self, table_name, factor_names):
        """优化数据"""
        return 0

    def fixData(self, table_name, factor_names):
        """修复数据，依赖具体实现，不保证一定修复"""
        return 0
