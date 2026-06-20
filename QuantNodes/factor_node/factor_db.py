# coding=utf-8
"""因子数据库

包含 FactorDB（只读接口）和 WritableFactorDB（可写入接口）
v2.0: 移除 traits 依赖
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


from QuantNodes.factor_node.quant_nodes_object import QuantNodesObject


@dataclass
class FactorDB(QuantNodesObject):
    """因子库基类（只读接口）

    数据库由若干张因子表组成。
    不支持某个操作时，方法产生错误。
    没有相关数据时，方法返回 None。
    """
    name: str = "FactorDB"

    def connect(self) -> int:
        """连接到数据库

        Returns:
            0 表示成功
        """
        return 0

    def disconnect(self) -> int:
        """断开数据库连接

        Returns:
            0 表示成功
        """
        return 0

    def isAvailable(self) -> bool:
        """检查数据库是否可用

        Returns:
            True 表示可用
        """
        return True

    @property
    def TableNames(self) -> List[str]:
        """获取所有表名

        Returns:
            表名列表
        """
        return []

    def getTable(
        self, table_name: str, args: Optional[Dict[str, Any]] = None
    ) -> Optional["FactorTable"]:  # noqa: F821
        """获取因子表对象

        Args:
            table_name: 表名
            args: 额外参数

        Returns:
            FactorTable 对象，不存在返回 None
        """
        return None

    def getID(self) -> List[str]:
        """获取 ID 序列

        Returns:
            ID 列表
        """
        return []

    def getDateTime(self) -> List[Any]:
        """获取时间点序列

        Returns:
            时间点列表
        """
        return []


class WritableFactorDB(FactorDB):
    """可写入的因子数据库"""

    def renameTable(self, old_table_name: str, new_table_name: str) -> int:
        """重命名表

        Args:
            old_table_name: 旧表名
            new_table_name: 新表名

        Returns:
            0 表示成功
        """
        return 0

    def deleteTable(self, table_name: str) -> int:
        """删除表

        Args:
            table_name: 表名

        Returns:
            0 表示成功
        """
        return 0

    def setTableMetaData(
        self,
        table_name: str,
        key: Optional[str] = None,
        value: Optional[Any] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """设置表的元数据

        Args:
            table_name: 表名
            key: 元数据键
            value: 元数据值
            meta_data: 元数据字典

        Returns:
            0 表示成功
        """
        return 0

    def renameFactor(
        self,
        table_name: str,
        old_factor_name: str,
        new_factor_name: str,
    ) -> int:
        """重命名因子

        Args:
            table_name: 表名
            old_factor_name: 旧因子名
            new_factor_name: 新因子名

        Returns:
            0 表示成功
        """
        return 0

    def deleteFactor(self, table_name: str, factor_names: List[str]) -> int:
        """删除因子

        Args:
            table_name: 表名
            factor_names: 因子名列表

        Returns:
            0 表示成功
        """
        return 0

    def setFactorMetaData(
        self,
        table_name: str,
        ifactor_name: str,
        key: Optional[str] = None,
        value: Optional[Any] = None,
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        """设置因子的元数据

        Args:
            table_name: 表名
            ifactor_name: 因子名称
            key: 元数据键
            value: 元数据值
            meta_data: 元数据字典

        Returns:
            0 表示成功
        """
        return 0

    def writeData(
        self,
        data: Any,
        table_name: str,
        if_exists: str = "update",
        data_type: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> int:
        """写入数据

        Args:
            data: 数据
            table_name: 表名
            if_exists: 如果存在 ("append", "update")
            data_type: 数据类型字典 {因子名: 数据类型}
            **kwargs: 其他参数

        Returns:
            0 表示成功
        """
        return 0

    def offsetDateTime(
        self,
        lag: int,
        table_name: str,
        factor_names: List[str],
        args: Optional[Dict[str, Any]] = None,
    ) -> int:
        """时间平移

        沿着时间轴将所有数据纵向移动 lag 期
        lag > 0 向前移动，lag < 0 向后移动
        空出来的地方填 nan

        Args:
            lag: 平移期数
            table_name: 表名
            factor_names: 因子名列表
            args: 额外参数

        Returns:
            0 表示成功
        """
        if lag == 0:
            return 0
        FT = self.getTable(table_name, args=args)
        if FT is None:
            return -1
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
            Data.iloc[:, lag:, :] = None
        DataType = FT.getFactorMetaData(
            factor_names, key="DataType", args=args
        ).to_dict()
        self.deleteFactor(table_name, factor_names)
        self.writeData(Data, table_name, data_type=DataType)
        return 0

    def _read_transform_write(
        self,
        table_name: str,
        factor_names: List[str],
        ids: List[str],
        dts: List[Any],
        transform_fn,
        args: Optional[Dict[str, Any]] = None,
        if_exists: str = "update",
    ) -> int:
        """通用读取-变换-写入模式

        Args:
            table_name: 表名
            factor_names: 因子名列表
            ids: ID 列表
            dts: 时间点列表
            transform_fn: 变换函数，接收 (Data, FT) 返回变换后的 Data
            args: 额外参数
            if_exists: 写入模式

        Returns:
            0 表示成功, -1 表示表不存在
        """
        FT = self.getTable(table_name, args=args)
        if FT is None:
            return -1
        Data = FT.readData(
            factor_names=factor_names, ids=ids, dts=dts, args=args
        )
        Data = transform_fn(Data, FT)
        if Data is not None:
            self.writeData(Data, table_name, if_exists=if_exists)
        return 0

    def changeData(
        self,
        table_name: str,
        factor_names: List[str],
        ids: List[str],
        dts: List[Any],
        args: Optional[Dict[str, Any]] = None,
    ) -> int:
        """数据变换

        通过某种变换函数得到新的时间序列和ID序列

        Args:
            table_name: 表名
            factor_names: 因子名列表
            ids: ID 列表
            dts: 时间点列表
            args: 额外参数

        Returns:
            0 表示成功
        """
        def _transform(Data, FT):
            DataType = FT.getFactorMetaData(
                factor_names, key="DataType", args=args
            ).to_dict()
            self.deleteFactor(table_name, factor_names)
            self.writeData(Data, table_name, data_type=DataType)
            return None  # Already written
        return self._read_transform_write(table_name, factor_names, ids, dts, _transform, args)

    def fillNA(
        self,
        filled_value: Any,
        table_name: str,
        factor_names: List[str],
        ids: List[str],
        dts: List[Any],
        args: Optional[Dict[str, Any]] = None,
    ) -> int:
        """填充缺失值

        Args:
            filled_value: 填充值
            table_name: 表名
            factor_names: 因子名列表
            ids: ID 列表
            dts: 时间点列表
            args: 额外参数

        Returns:
            0 表示成功
        """
        def _transform(Data, FT):
            Data.fillna(filled_value, inplace=True)
            return Data
        return self._read_transform_write(
            table_name, factor_names, ids, dts, _transform, args
        )

    def replaceData(
        self,
        old_value: Any,
        new_value: Any,
        table_name: str,
        factor_names: List[str],
        ids: List[str],
        dts: List[Any],
        args: Optional[Dict[str, Any]] = None,
    ) -> int:
        """替换数据

        Args:
            old_value: 旧值
            new_value: 新值
            table_name: 表名
            factor_names: 因子名列表
            ids: ID 列表
            dts: 时间点列表
            args: 额外参数

        Returns:
            0 表示成功
        """
        def _transform(Data, FT):
            return Data.where(Data != old_value, new_value)
        return self._read_transform_write(
            table_name, factor_names, ids, dts, _transform, args
        )

    def optimizeData(self, table_name: str, factor_names: List[str]) -> int:
        """优化数据

        Args:
            table_name: 表名
            factor_names: 因子名列表

        Returns:
            0 表示成功
        """
        return 0

    def fixData(self, table_name: str, factor_names: List[str]) -> int:
        """修复数据

        依赖具体实现，不保证一定修复

        Args:
            table_name: 表名
            factor_names: 因子名列表

        Returns:
            0 表示成功
        """
        return 0
