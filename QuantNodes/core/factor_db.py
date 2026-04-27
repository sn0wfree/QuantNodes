# coding=utf-8
"""
因子库基类

替代 QuantStudio.FactorDataBase.FactorDB.FactorDB
"""

from typing import Any, Dict, List, Optional

from QuantNodes.core.quant_nodes_object import QuantNodesObject


class FactorDB(QuantNodesObject):
    """
    因子库基类

    代表一个因子库，由若干张因子表组成

    Attributes:
        Name: 因子库名称
    """

    def __init__(
        self,
        name: str = "因子库",
        sys_args: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        self._Name = name
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)

    @property
    def Name(self) -> str:
        return self._Name

    def connect(self) -> int:
        """
        连接到数据库

        Returns:
            0 表示成功
        """
        return 0

    def disconnect(self) -> int:
        """
        断开数据库连接

        Returns:
            0 表示成功
        """
        return 0

    def isAvailable(self) -> bool:
        """
        检查因子库是否可用

        Returns:
            True 表示可用
        """
        return True

    @property
    def TableNames(self) -> List[str]:
        """
        获取所有表名

        Returns:
            表名列表
        """
        return []

    def getTable(self, table_name: str, args: Optional[Dict[str, Any]] = None) -> Optional["FactorTable"]:
        """
        获取因子表对象

        Args:
            table_name: 表名
            args: 额外参数

        Returns:
            FactorTable 对象，不存在返回 None
        """
        return None

    def offsetDateTime(
        self,
        lag: int,
        table_name: str,
        factor_names: List[str],
        args: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        时间平移

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
        return 0


class WritableFactorDB(FactorDB):
    """可写入的因子数据库"""

    def renameTable(self, old_table_name: str, new_table_name: str) -> int:
        """
        重命名表

        Args:
            old_table_name: 旧表名
            new_table_name: 新表名

        Returns:
            0 表示成功
        """
        return 0

    def deleteTable(self, table_name: str) -> int:
        """
        删除表

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
        """
        设置表的元数据

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
        """
        重命名因子

        Args:
            table_name: 表名
            old_factor_name: 旧因子名
            new_factor_name: 新因子名

        Returns:
            0 表示成功
        """
        return 0

    def deleteFactor(self, table_name: str, factor_names: List[str]) -> int:
        """
        删除因子

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
        """
        设置因子的元数据

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
        """
        写入数据

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
