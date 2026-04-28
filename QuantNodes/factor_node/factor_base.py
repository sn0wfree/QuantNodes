# coding=utf-8
"""
因子基类

替代 QuantStudio.FactorDataBase.FactorDB.Factor
"""

import os
import shelve
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from traits.api import Callable as TraitCallable, Dict as TraitDict, Enum, Instance, List as TraitList, Str

from QuantNodes.core.base import FactorError
from QuantNodes.factor_node.quant_nodes_object import QuantNodesObject


def _default_operator(f, idt, iid, x, args):
    """默认算子，返回 NaN"""
    return np.nan


class Factor(QuantNodesObject):
    """
    因子基类

    代表一个因子，支持遍历模式和运算模式两种数据获取方式

    Attributes:
        Name: 因子名称
        DataType: 数据类型 ("double", "string", "object")
    """

    Name = Str("因子")
    DataType = Enum("double", "string", "object")

    def __init__(
        self,
        name: str,
        ft: Optional["FactorTable"] = None,
        sys_args: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化因子

        Args:
            name: 因子名称
            ft: 因子所属的因子表，None 表示衍生因子
            sys_args: 系统参数字典
            config_file: 配置文件路径
            **kwargs: 其他关键字参数
        """
        self._FactorTable = ft
        self._NameInFT = name
        self.Name = name
        self._isStarted = False
        self._CacheData = None
        self._OperationMode = None
        self._RawDataFile = ""
        self._isCacheDataOK = False
        self.UserData = {}
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)

    @property
    def FactorTable(self) -> Optional["FactorTable"]:
        """获取因子所属的因子表"""
        return self._FactorTable

    @property
    def Descriptors(self) -> List["Factor"]:
        """获取描述子列表，子类应重写"""
        return []

    @property
    def Args(self) -> Dict[str, Any]:
        """获取参数字典"""
        return {}

    def getMetaData(self, key: Optional[str] = None, args: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        获取因子的元数据

        Args:
            key: 元数据键，None 返回全部
            args: 额外参数

        Returns:
            pd.Series 包含元数据
        """
        args = args or {}
        if self._FactorTable is not None:
            return self._FactorTable.getFactorMetaData(
                factor_names=[self._NameInFT], key=key, args=args
            ).loc[self._NameInFT]
        return pd.Series({"DataType": self.DataType})

    def getID(self, idt: Optional[Any] = None) -> List[Any]:
        """
        获取 ID 序列

        Args:
            idt: 时间点，用于过滤

        Returns:
            ID 列表
        """
        if self._OperationMode is not None and self._OperationMode._isStarted:
            return self._OperationMode.IDs
        if self._FactorTable is not None:
            return self._FactorTable.getID(ifactor_name=self._NameInFT, idt=idt, args=self.Args)
        return []

    def getDateTime(
        self,
        iid: Optional[Any] = None,
        start_dt: Optional[Any] = None,
        end_dt: Optional[Any] = None,
    ) -> List[Any]:
        """
        获取时间点序列

        Args:
            iid: ID，用于过滤
            start_dt: 起始时间
            end_dt: 结束时间

        Returns:
            时间点列表
        """
        if self._OperationMode is not None and self._OperationMode._isStarted:
            return self._OperationMode.DateTimes
        if self._FactorTable is not None:
            return self._FactorTable.getDateTime(
                ifactor_name=self._NameInFT,
                iid=iid,
                start_dt=start_dt,
                end_dt=end_dt,
                args=self.Args,
            )
        return []

    def readData(
        self,
        ids: List[Any],
        dts: List[Any],
        **kwargs,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        读取因子数据

        Args:
            ids: ID 列表
            dts: 时间点列表
            **kwargs: 其他参数

        Returns:
            pd.DataFrame or pd.Series
        """
        if not self._isStarted:
            if self._FactorTable is not None:
                return self._FactorTable.readData(
                    factor_names=[self._NameInFT], ids=ids, dts=dts, args=self.Args
                ).loc[self._NameInFT]
            return pd.Series(dtype=object)

        if self._CacheData is None:
            if self._FactorTable is not None:
                self._CacheData = self._FactorTable.readData(
                    factor_names=[self._NameInFT], ids=ids, dts=dts, args=self.Args
                ).loc[self._NameInFT]
            else:
                self._CacheData = pd.Series(dtype=object)
            return self._CacheData

        new_dts = sorted(set(dts).difference(self._CacheData.index))
        if new_dts and self._FactorTable is not None:
            new_cache_data = self._FactorTable.readData(
                factor_names=[self._NameInFT],
                ids=self._CacheData.columns.tolist(),
                dts=new_dts,
                args=self.Args,
            ).loc[self._NameInFT]
            self._CacheData = pd.concat([self._CacheData, new_cache_data]).loc[dts]

        new_ids = sorted(set(ids).difference(self._CacheData.columns))
        if new_ids and self._FactorTable is not None:
            new_cache_data = self._FactorTable.readData(
                factor_names=[self._NameInFT],
                ids=new_ids,
                dts=self._CacheData.index.tolist(),
                args=self.Args,
            ).loc[self._NameInFT]
            self._CacheData = pd.merge(
                self._CacheData,
                new_cache_data,
                left_index=True,
                right_index=True,
                how="outer",
            )

        return self._CacheData.loc[dts, ids]

    def _QN_init_operation(
        self,
        start_dt: Any,
        dt_dict: Dict[str, Any],
        prepare_ids: List[Any],
        id_dict: Dict[str, List[Any]],
    ) -> None:
        """
        初始化运算模式

        Args:
            start_dt: 开始时间
            dt_dict: 时间信息字典
            prepare_ids: 准备数据的 ID
            id_dict: ID 信息字典
        """
        old_start_dt = dt_dict.get(self.Name, start_dt)
        dt_dict[self.Name] = start_dt if start_dt < old_start_dt else old_start_dt
        prepare_ids = id_dict.setdefault(self.Name, prepare_ids)
        if prepare_ids != id_dict.get(self.Name):
            raise FactorError(f"因子 {self.Name} 指定了不同的截面!")

    def __QN_prepare_cache_data__(self, ids: Optional[List[Any]] = None) -> np.ndarray:
        """
        准备缓存数据

        Args:
            ids: ID 列表

        Returns:
            标准化数据数组
        """
        raise NotImplementedError("Factor.__QN_prepare_cache_data__ 子类必须实现")

    def _QN_get_data(
        self,
        dts: List[Any],
        pids: Optional[List[Any]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        获取数据

        Args:
            dts: 时间点列表
            pids: 进程 ID 列表
            **kwargs: 其他参数

        Returns:
            数据数组
        """
        raise NotImplementedError("Factor._QN_get_data 子类必须实现")

    def start(self, dts: List[Any], **kwargs) -> int:
        """
        启动遍历模式

        Args:
            dts: 时间点列表
            **kwargs: 其他参数

        Returns:
            0 表示成功
        """
        self._isStarted = True
        return 0

    def move(self, idt: Any, **kwargs) -> int:
        """
        移动到指定时间点

        Args:
            idt: 时间点
            **kwargs: 其他参数

        Returns:
            0 表示成功
        """
        return 0

    def end(self) -> int:
        """
        结束遍历模式

        Returns:
            0 表示成功
        """
        self._isStarted = False
        return 0

    def __neg__(self) -> "Factor":
        """一元负运算"""
        raise NotImplementedError("Factor.__neg__ 子类应实现")

    def __add__(self, other: Any) -> "Factor":
        """加法运算"""
        raise NotImplementedError("Factor.__add__ 子类应实现")

    def __radd__(self, other: Any) -> "Factor":
        """反向加法运算"""
        raise NotImplementedError("Factor.__radd__ 子类应实现")

    def __sub__(self, other: Any) -> "Factor":
        """减法运算"""
        raise NotImplementedError("Factor.__sub__ 子类应实现")

    def __rsub__(self, other: Any) -> "Factor":
        """反向减法运算"""
        raise NotImplementedError("Factor.__rsub__ 子类应实现")

    def __mul__(self, other: Any) -> "Factor":
        """乘法运算"""
        raise NotImplementedError("Factor.__mul__ 子类应实现")

    def __rmul__(self, other: Any) -> "Factor":
        """反向乘法运算"""
        raise NotImplementedError("Factor.__rmul__ 子类应实现")

    def __truediv__(self, other: Any) -> "Factor":
        """除法运算"""
        raise NotImplementedError("Factor.__truediv__ 子类应实现")

    def __rtruediv__(self, other: Any) -> "Factor":
        """反向除法运算"""
        raise NotImplementedError("Factor.__rtruediv__ 子类应实现")
