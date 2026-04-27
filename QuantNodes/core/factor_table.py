# coding=utf-8
"""
因子表基类

替代 QuantStudio.FactorDataBase.FactorDB.FactorTable
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from traits.api import Instance

from QuantNodes.core.quant_nodes_object import QuantNodesObject


class FactorTable(QuantNodesObject):
    """
    因子表基类

    代表一个因子表，提供数据读取和因子管理功能

    Attributes:
        Name: 因子表名称
        ErgodicMode: 遍历模式对象
        OperationMode: 运算模式对象
    """

    ErgodicMode = Instance("ErgodicMode", allow_none=True)
    OperationMode = Instance("OperationMode", allow_none=True)

    def __init__(
        self,
        name: str,
        fdb: Optional["FactorDB"] = None,
        sys_args: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        self._Name = name
        self._FactorDB = fdb
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)

    @property
    def Name(self) -> str:
        return self._Name

    @property
    def FactorDB(self) -> Optional["FactorDB"]:
        return self._FactorDB

    def __QS_initArgs__(self, sys_args: Optional[Dict[str, Any]] = None) -> None:
        pass

    def getMetaData(self, key: Optional[str] = None, args: Optional[Dict[str, Any]] = None) -> Dict:
        if key is None:
            return {}
        return None

    @property
    def FactorNames(self) -> List[str]:
        return []

    def getFactor(
        self,
        ifactor_name: str,
        args: Optional[Dict[str, Any]] = None,
        new_name: Optional[str] = None,
    ) -> "Factor":
        from QuantNodes.core.factor_base import Factor
        args = args or {}
        i_factor = Factor(name=ifactor_name, ft=self)
        if new_name is not None:
            i_factor.Name = new_name
        return i_factor

    def getFactorMetaData(
        self,
        factor_names: List[str],
        key: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if key is None:
            return pd.DataFrame(index=factor_names, dtype=np.dtype("O"))
        return pd.Series([None] * len(factor_names), index=factor_names, dtype=np.dtype("O"))

    def getID(self, ifactor_name: Optional[str] = None, idt: Optional[Any] = None, args: Optional[Dict[str, Any]] = None) -> List[Any]:
        return []

    def getIDMask(
        self,
        idt: Any,
        ids: Optional[List[Any]] = None,
        id_filter_str: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> pd.Series:
        args = args or {}
        if ids is None:
            ids = self.getID(idt=idt, args=args)
        if not id_filter_str:
            return pd.Series(True, index=ids)
        raise NotImplementedError("ID 过滤字符串功能需要实现 testIDFilterStr")

    def getFilteredID(
        self,
        idt: Any,
        ids: Optional[List[Any]] = None,
        id_filter_str: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        args = args or {}
        if not id_filter_str:
            return self.getID(idt=idt, args=args)
        if ids is None:
            ids = self.getID(idt=idt, args=args)
        raise NotImplementedError("ID 过滤字符串功能需要实现 testIDFilterStr")

    def getDateTime(
        self,
        ifactor_name: Optional[str] = None,
        iid: Optional[Any] = None,
        start_dt: Optional[Any] = None,
        end_dt: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        return []

    def __QS_prepareRawData__(
        self,
        factor_names: List[str],
        ids: List[Any],
        dts: List[Any],
        args: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        return None

    def __QS_calcData__(
        self,
        raw_data: Any,
        factor_names: List[str],
        ids: List[Any],
        dts: List[Any],
        args: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        return None

    def readData(
        self,
        factor_names: List[str],
        ids: List[Any],
        dts: List[Any],
        args: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        args = args or {}
        if self.ErgodicMode is not None and self.ErgodicMode._isStarted:
            return self._readData_ErgodicMode(factor_names=factor_names, ids=ids, dts=dts, args=args)
        return self.__QS_calcData__(
            raw_data=self.__QS_prepareRawData__(factor_names=factor_names, ids=ids, dts=dts, args=args),
            factor_names=factor_names,
            ids=ids,
            dts=dts,
            args=args,
        )

    def _readData_ErgodicMode(
        self,
        factor_names: List[str],
        ids: List[Any],
        dts: List[Any],
        args: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError("遍历模式需要实现 ErgodicMode 相关功能")

    def start(self, dts: List[Any], **kwargs) -> int:
        if self.ErgodicMode is not None:
            self.ErgodicMode._isStarted = True
        return 0

    def move(self, idt: Any, **kwargs) -> int:
        return 0

    def end(self) -> int:
        if self.ErgodicMode is not None:
            self.ErgodicMode._isStarted = False
        return 0

    def __QS_onBackTestMoveEvent__(self, event: Any) -> None:
        pass

    def __QS_onBackTestEndEvent__(self, event: Any) -> None:
        pass

    def __QS_genGroupInfo__(self, factors: List["Factor"], operation_mode: Any) -> List[Any]:
        return []

    def write2FDB(
        self,
        factor_names: List[str],
        ids: List[Any],
        dts: List[Any],
        factor_db: "FactorDB",
        table_name: str,
        if_exists: str = "update",
        **kwargs,
    ) -> int:
        return 0


class CustomFT(FactorTable):
    """自定义因子表"""

    def __init__(
        self,
        name: str = "CustomFactorTable",
        sys_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self._Factors = {}
        self._DateTimes = []
        self._IDs = []
        super().__init__(name=name, fdb=None, sys_args=sys_args, **kwargs)

    def __QS_initArgs__(self, sys_args: Optional[Dict[str, Any]] = None) -> None:
        self._Factors = {}
        self._DateTimes = []
        self._IDs = []

    @property
    def FactorNames(self) -> List[str]:
        return list(self._Factors.keys())

    @property
    def Factors(self) -> Dict[str, "Factor"]:
        return self._Factors

    def getFactor(
        self,
        ifactor_name: str,
        args: Optional[Dict[str, Any]] = None,
        new_name: Optional[str] = None,
    ) -> "Factor":
        from QuantNodes.core.factor_base import Factor
        args = args or {}
        if ifactor_name not in self._Factors:
            raise ValueError(f"因子 '{ifactor_name}' 不存在!")
        i_factor = self._Factors[ifactor_name]
        if new_name is not None:
            i_factor.Name = new_name
        return i_factor

    def getDateTime(
        self,
        ifactor_name: Optional[str] = None,
        iid: Optional[Any] = None,
        start_dt: Optional[Any] = None,
        end_dt: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        return self._DateTimes

    def setDateTime(self, dts: List[Any]) -> None:
        self._DateTimes = dts

    def getID(
        self,
        ifactor_name: Optional[str] = None,
        idt: Optional[Any] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        return self._IDs

    def setID(self, ids: List[Any]) -> None:
        self._IDs = ids

    def IDFilterStr(self) -> str:
        return ""

    def setIDFilter(self, id_filter_str: str) -> None:
        pass

    def addFactors(
        self,
        factor_list: Optional[List["Factor"]] = None,
        factor_table: Optional[FactorTable] = None,
        factor_names: Optional[List[str]] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> int:
        from QuantNodes.core.factor_base import Factor
        args = args or {}
        if factor_table is not None:
            if factor_names is None:
                factor_names = factor_table.FactorNames
            for i_name in factor_names:
                i_factor = factor_table.getFactor(i_name, args=args)
                if i_factor.Name in self._Factors:
                    raise ValueError(f"因子: '{i_factor.Name}' 有重名!")
                self._Factors[i_factor.Name] = i_factor
        elif factor_list is not None:
            for i_factor in factor_list:
                if not isinstance(i_factor, Factor):
                    raise ValueError("添加的必须是 Factor 对象!")
                if i_factor.Name in self._Factors:
                    raise ValueError(f"因子: '{i_factor.Name}' 有重名!")
                self._Factors[i_factor.Name] = i_factor
        return 0

    def deleteFactors(self, factor_names: Optional[List[str]] = None) -> int:
        if factor_names is None:
            self._Factors = {}
        else:
            for i_name in factor_names:
                if i_name not in self._Factors:
                    raise ValueError(f"因子: '{i_name}' 不存在!")
                del self._Factors[i_name]
        return 0

    def renameFactor(self, factor_name: str, new_factor_name: str) -> int:
        if factor_name not in self._Factors:
            raise ValueError(f"因子: '{factor_name}' 不存在!")
        if (new_factor_name != factor_name) and (new_factor_name in self._Factors):
            raise ValueError(f"因子: '{new_factor_name}' 已存在!")
        self._Factors[new_factor_name] = self._Factors.pop(factor_name)
        self._Factors[new_factor_name].Name = new_factor_name
        return 0

    def __QS_calcData__(
        self,
        raw_data: Any,
        factor_names: List[str],
        ids: List[Any],
        dts: List[Any],
        args: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        return None
