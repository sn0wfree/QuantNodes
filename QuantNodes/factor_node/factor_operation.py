# -*- coding: utf-8 -*-
"""
因子运算模块

提供因子运算操作类，包括单点运算、时间序列运算、截面运算和面板运算。
v2.0: 移除 traits 和 multiprocessing 依赖，使用纯 Python/Polars

Classes:
    DerivativeFactor: 因子运算基类
    PointOperation: 单点运算，对描述子进行单点运算
    TimeOperation: 时间序列运算，对描述子进行时间序列运算（滚动/扩展窗口）
    SectionOperation: 截面运算，对描述子进行截面运算
    PanelOperation: 面板运算，结合时间序列和截面运算
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from QuantNodes.core.base import FactorError
from QuantNodes.factor_node.factor import Factor, DataType


def _DefaultOperator(f: Factor, idt: Any, iid: Any, x: List[np.ndarray], args: Dict[str, Any]) -> np.ndarray:
    """默认算子，返回 NaN"""
    return np.nan


class DataOperationType(Enum):
    """数据类型枚举"""
    DOUBLE = "double"
    STRING = "string"
    OBJECT = "object"


@dataclass
class DerivativeFactor(Factor):
    """因子运算基类

    所有因子运算类的基类，提供描述子管理和通用接口。
    
    Attributes:
        Operator: 运算函数，签名为 (f, idt, iid, x, args) -> result
        ModelArgs: 参数字典
        DataType: 数据类型
    """
    operator: Callable = field(default=_DefaultOperator)
    model_args: Dict[str, Any] = field(default_factory=dict)
    data_type: DataOperationType = DataOperationType.DOUBLE
    _descriptors: List[Factor] = field(default_factory=list)

    def __init__(self, name: str = "", descriptors: List[Factor] = None, sys_args: Dict[str, Any] = None, **kwargs):
        self._descriptors = descriptors if descriptors else []
        self.UserData = {}
        self.name = name
        self.model_args = sys_args or {}
        self.operator = kwargs.get('operator', _DefaultOperator)
        
        if descriptors and hasattr(descriptors[0], '_logger'):
            self._logger = descriptors[0]._logger
        
        super().__init__(name=name, ft=None, sys_args=sys_args)

    @property
    def Descriptors(self) -> List[Factor]:
        return self._descriptors

    @property
    def Operator(self) -> Callable:
        return self.operator

    @Operator.setter
    def Operator(self, value: Callable):
        self.operator = value

    def __init__(self, name: str = "", descriptors: List[Factor] = None, sys_args: Dict[str, Any] = None, **kwargs):
        self._Descriptors = descriptors if descriptors else []
        self.UserData = {}
        if descriptors:
            kwargs.setdefault("logger", descriptors[0]._QN_logger)
        return super().__init__(name=name, ft=None, sys_args=sys_args or {}, config_file=None, **kwargs)

    @property
    def Descriptors(self) -> List[Factor]:
        """描述子列表"""
        return self._Descriptors

    def getMetaData(self, key: str = None, args: Dict[str, Any] = None) -> Any:
        """获取元数据

        Args:
            key: 元数据键名，None时返回包含DataType的Series
            args: 参数字典

        Returns:
            元数据值或包含元数据的Series
        """
        DataType = args.get("数据类型", self.DataType) if args else self.DataType
        if key is None:
            return pd.Series({"DataType": DataType})
        elif key == "DataType":
            return DataType
        return None

    def start(self, dts: List[Any], **kwargs) -> int:
        """开始运算前的初始化

        Args:
            dts: 时间点列表
            **kwargs: 其他关键字参数

        Returns:
            成功返回0
        """
        for iDescriptor in self._Descriptors:
            iDescriptor.start(dts=dts, **kwargs)
        return 0

    def end(self) -> int:
        """运算结束后的清理

        Returns:
            成功返回0
        """
        for iDescriptor in self._Descriptors:
            iDescriptor.end()
        return 0


# 单点运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算时点为多时点，则该值为[时点]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为 [ID]
# x: 描述子当期的数据, [单个描述子值 or array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 运算ID参数为单ID, 那么 x 元素为单个描述子值, 返回单个元素
# 如果运算时点参数为单时点, 运算ID参数为多ID, 那么 x 元素为 array(shape=(nID, )), 注意并发时 ID 并不是全截面, 返回 array(shape=(nID,))
# 如果运算时点参数为多时点, 运算ID参数为单ID, 那么 x 元素为 array(shape=(nDT, )), 返回 array(shape=(nID, ))
# 如果运算时点参数为多时点, 运算ID参数为多ID, 那么 x 元素为 array(shape=(nDT, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nDT, nID))
from enum import Enum


class DTModeType(Enum):
    """运算时点模式"""
    SINGLE = "单时点"
    MULTI = "多时点"


class IDModeType(Enum):
    """运算ID模式"""
    SINGLE = "单ID"
    MULTI = "多ID"


@dataclass
class PointOperation(DerivativeFactor):
    """单点运算

    对描述子进行单点运算，即每个时点-ID组合独立计算。
    
    Attributes:
        dt_mode: 运算时点模式
        id_mode: 运算ID模式
    """
    dt_mode: DTModeType = DTModeType.SINGLE
    id_mode: IDModeType = IDModeType.SINGLE

    def readData(self, ids: List[Any], dts: List[Any], **kwargs) -> pd.DataFrame:
        """读取并计算数据

        Args:
            ids: ID列表
            dts: 时间点列表
            **kwargs: 其他关键字参数

        Returns:
            计算结果DataFrame，index为dts，columns为ids
        """
        if len(dts) == 0:
            return create_empty_dataframe(dts, ids, self.DataType)
        StdData = self._calcData(
            ids=ids,
            dts=dts,
            descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs).values
                           for iDescriptor in self._Descriptors]
        )
        return pd.DataFrame(StdData, index=dts, columns=ids)

    def _QN_init_operation(self, start_dt: Any, dt_dict: Dict[str, Any], prepare_ids: List[Any], id_dict: Dict[str, List[Any]]) -> None:
        """初始化运算环境

        Args:
            start_dt: 起始时间点
            dt_dict: 时间点字典
            prepare_ids: 准备计算的ID列表
            id_dict: ID字典
        """
        super()._QN_init_operation(start_dt, dt_dict, prepare_ids, id_dict)
        for i, iDescriptor in enumerate(self._Descriptors):
            iDescriptor._QN_init_operation(dt_dict[self.Name], dt_dict, prepare_ids, id_dict)

    def _calcData(self, ids: List[Any], dts: List[Any], descriptor_data: List[np.ndarray]) -> np.ndarray:
        """计算数据（策略分派入口）

        Args:
            ids: ID列表
            dts: 时间点列表
            descriptor_data: 描述子数据列表

        Returns:
            计算结果数组
        """
        handler_name = self._DT_ID_DISPATCH.get((self.DTMode, self.IDMode))
        if handler_name:
            return getattr(self, handler_name)(ids, dts, descriptor_data)
        return create_std_data(dts, ids, self.DataType)

    def _calcData_multi_time_multi_id(self, ids: List[Any], dts: List[Any], descriptor_data: List[np.ndarray]) -> np.ndarray:
        """多时点-多ID模式计算"""
        return self.Operator(self, dts, ids, descriptor_data, self.ModelArgs)

    def _calcData_single_time_single_id(self, ids: List[Any], dts: List[Any], descriptor_data: List[np.ndarray]) -> np.ndarray:
        """单时点-单ID模式计算

        Args:
            ids: ID列表
            dts: 时间点列表
            descriptor_data: 描述子数据列表

        Returns:
            计算结果数组
        """
        StdData = create_std_data(dts, ids, self.DataType)
        for i, iDT in enumerate(dts):
            for j, jID in enumerate(ids):
                StdData[i, j] = self.Operator(
                    self, iDT, jID,
                    [iData[i, j] for iData in descriptor_data],
                    self.ModelArgs
                )
        return StdData

    def _calcData_multi_time_single_id(self, ids: List[Any], dts: List[Any], descriptor_data: List[np.ndarray]) -> np.ndarray:
        """多时点-单ID模式计算

        Args:
            ids: ID列表
            dts: 时间点列表
            descriptor_data: 描述子数据列表

        Returns:
            计算结果数组
        """
        StdData = create_std_data(dts, ids, self.DataType)
        for j, jID in enumerate(ids):
            StdData[:, j] = self.Operator(
                self, dts, jID,
                [iData[:, j] for iData in descriptor_data],
                self.ModelArgs
            )
        return StdData

    def _calcData_single_time_multi_id(self, ids: List[Any], dts: List[Any], descriptor_data: List[np.ndarray]) -> np.ndarray:
        """单时点-多ID模式计算

        Args:
            ids: ID列表
            dts: 时间点列表
            descriptor_data: 描述子数据列表

        Returns:
            计算结果数组
        """
        StdData = create_std_data(dts, ids, self.DataType)
        for i, iDT in enumerate(dts):
            StdData[i, :] = self.Operator(
                self, iDT, ids,
                [iData[i, :] for iData in descriptor_data],
                self.ModelArgs
            )
        return StdData

    def __QN_prepare_cache_data__(self, ids: Optional[List[Any]] = None) -> pd.DataFrame:
        """准备缓存数据

        Args:
            ids: ID列表

        Returns:
            标准数据DataFrame
        """
        PID = self._OperationMode._iPID
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = (
            self._OperationMode.DTRuler.index(StartDT),
            self._OperationMode.DTRuler.index(EndDT)
        )
        DTs = list(self._OperationMode.DTRuler[StartInd:EndInd + 1])
        IDs = partition_ids_for_pid(self._OperationMode, self._OperationMode._FactorPrepareIDs[self.Name], PID)
        if IDs:
            StdData = self._calcData(
                ids=IDs, dts=DTs,
                descriptor_data=[iDescriptor._QN_get_data(DTs, pids=[PID]).values
                                 for iDescriptor in self._Descriptors]
            )
            StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
        else:
            StdData = create_empty_dataframe(DTs, [], self.DataType)
        write_cache_file(
            self._OperationMode, PID, self.Name,
            self._OperationMode._FactorID[self.Name], StdData, IDs
        )
        self._isCacheDataOK = True
        return StdData


class LookBackMode(Enum):
    """回溯模式"""
    ROLLING = "滚动窗口"
    EXPANDING = "扩张窗口"


@dataclass
class _LookBackOperation(DerivativeFactor):
    """带 LookBack 窗口运算的基类
    
    提取 TimeOperation 和 PanelOperation 中相同的 LookBack 窗口计算逻辑。
    子类需要实现具体的 _calcData 方法或使用策略分派模式。
    
    Attributes:
        look_back: 回溯期数列表，对应每个描述子
        look_back_mode: 回溯模式列表
        i_look_back: 自身回溯期数
        i_look_back_mode: 自身回溯模式
        i_init_data: 自身初始值DataFrame
    """
    look_back: List[int] = field(default_factory=list)
    look_back_mode: List[LookBackMode] = field(default_factory=list)
    i_look_back: int = 0
    i_look_back_mode: LookBackMode = LookBackMode.ROLLING
    i_init_data: Optional[pd.DataFrame] = None

    def __init__(self, name: str = "", descriptors: List[Factor] = None, sys_args: Dict = None, **kwargs):
        super().__init__(name=name, descriptors=descriptors, sys_args=sys_args, **kwargs)
        self._init_lookback()

    def _init_lookback(self) -> None:
        """初始化回溯参数"""
        n = len(self._descriptors)
        self.look_back = [0] * n
        self.look_back_mode = [LookBackMode.ROLLING] * n

    def _prepare_lookback_data(
        self,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray],
        dt_ruler: List[Any]
    ) -> Tuple[np.ndarray, int, List[Any], List[Tuple[int, int]], int, int, List[np.ndarray]]:
        """准备 LookBack 窗口数据
        
        计算窗口参数、处理初始数据、扩展时间标尺。
        
        Args:
            ids: ID列表
            dts: 时间点列表
            descriptor_data: 描述子数据列表
            dt_ruler: 时间标尺列表
        
        Returns:
            tuple: (StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, descriptor_data)
            - StdData: 预分配的结果数组
            - iStartInd: 初始数据起始索引
            - DTRuler: 扩展后的时间标尺
            - StartIndAndLen: 每个描述子的(起始索引, 窗口长度)列表
            - MaxLookBack: 最大回溯期数
            - MaxLen: 最大窗口长度
            - descriptor_data: 处理后的描述子数据
        """
        StdData = create_std_data(dts, ids, self.DataType)
        StartIndAndLen, MaxLookBack, MaxLen = [], 0, 1
        for i in range(len(self._Descriptors)):
            iLookBack = self.look_back[i]
            if self.look_back_mode[i] == "滚动窗口":
                StartIndAndLen.append((iLookBack, iLookBack + 1))
                MaxLen = max(MaxLen, iLookBack + 1)
            else:
                StartIndAndLen.append((iLookBack, np.inf))
                MaxLen = np.inf
            MaxLookBack = max(MaxLookBack, iLookBack)
        iStartInd = 0
        if (self.i_look_back_mode == LookBackMode.EXPANDING) or (self.i_look_back != 0):
            if self.i_init_data is not None:
                iInitData = self.iInitData.loc[self.iInitData.index < dts[0], :]
                if iInitData.shape[0] > 0:
                    if iInitData.columns.intersection(ids).shape[0] > 0:
                        iInitData = iInitData.loc[:, ids].values.astype(StdData.dtype)
                    else:
                        iInitData = np.full(shape=(iInitData.shape[0], len(ids)), dtype=StdData.dtype)
                    iStartInd = min(self.i_look_back, iInitData.shape[0])
                    StdData = np.r_[iInitData[-iStartInd:], StdData]
            if self.i_look_backMode == "扩张窗口":
                StartIndAndLen.insert(0, (iStartInd - 1, np.inf))
                MaxLen = np.inf
            else:
                StartIndAndLen.insert(0, (iStartInd - 1, self.i_look_back))
                MaxLen = max(MaxLen, self.i_look_back + 1)
            MaxLookBack = max(MaxLookBack, self.i_look_back)
            descriptor_data.insert(0, StdData)
        start_ind = dt_ruler.index(dts[0])
        if start_ind >= MaxLookBack:
            DTRuler = dt_ruler[start_ind - MaxLookBack:]
        else:
            DTRuler = [None] * (MaxLookBack - start_ind) + dt_ruler
        return StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, descriptor_data


# 时间序列运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多时点，则该值为 [时点]
# iid: 当前待计算的ID, 如果运算ID为多ID，则该值为 [ID]
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 运算ID参数为单ID, 那么x元素为array(shape=(回溯期数, )), 返回单个元素
# 如果运算时点参数为单时点, 运算ID参数为多ID, 那么x元素为array(shape=(回溯期数, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nID, ))
# 如果运算时点参数为多时点, 运算ID参数为单ID, 那么x元素为array(shape=(回溯期数+nDT, )), 返回 array(shape=(nDate,))
# 如果运算时点参数为多时点, 运算ID参数为多ID, 那么x元素为array(shape=(回溯期数+nDT, nID)), 注意并发时 ID 并不是全截面, 返回 array(shape=(nDT, nID))
class OutputModeType(Enum):
    """输出模式"""
    FULL_SECTION = "全截面"
    SINGLE_ID = "单ID"


@dataclass
class TimeOperation(_LookBackOperation):
    """时间序列运算"""
    dt_mode: DTModeType = DTModeType.SINGLE
    id_mode: IDModeType = IDModeType.SINGLE

    def _QN_init_operation(
        self,
        start_dt: Any,
        dt_dict: Dict[str, Any],
        prepare_ids: List[Any],
        id_dict: Dict[str, List[Any]]
    ) -> None:
        """初始化运算环境

        Args:
            start_dt: 起始时间点
            dt_dict: 时间点字典
            prepare_ids: 准备计算的ID列表
            id_dict: ID字典
        """
        super(_LookBackOperation, self)._QN_init_operation(start_dt, dt_dict, prepare_ids, id_dict)
        if len(self._Descriptors) > len(self.LookBack):
            raise FactorError(
                "时间序列运算因子 : '%s' 的参数'回溯期数'序列长度小于描述子个数!" % self.Name
            )
        StartDT = dt_dict[self.Name]
        StartInd = self._OperationMode.DTRuler.index(StartDT)
        if (self.i_look_backMode == "扩张窗口") and (self.iInitData is not None) and (self.iInitData.shape[0] > 0):
            if self.iInitData.index[-1] not in self._OperationMode.DTRuler:
                self._QN_logger.warning(
                    "注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name,)
                )
            else:
                StartInd = min(StartInd, self._OperationMode.DTRuler.index(self.iInitData.index[-1]) + 1)
        for i, iDescriptor in enumerate(self._Descriptors):
            iStartInd = StartInd - self.look_back[i]
            if iStartInd < 0:
                self._QN_logger.warning(
                    "注意: 对于因子 '%s' 的描述子 '%s', 时点标尺长度不足, 不足的部分将填充 nan!" % (self.Name, iDescriptor.Name)
                )
            iStartDT = self._OperationMode.DTRuler[max(0, iStartInd)]
            iDescriptor._QN_init_operation(iStartDT, dt_dict, prepare_ids, id_dict)

    def readData(self, ids: List[Any], dts: List[Any], **kwargs) -> pd.DataFrame:
        """读取并计算数据

        Args:
            ids: ID列表
            dts: 时间点列表
            **kwargs: 其他关键字参数，支持 dt_ruler 指定时间标尺

        Returns:
            计算结果DataFrame，index为dts，columns为ids
        """
        if len(dts) == 0:
            return create_empty_dataframe(dts, ids, self.DataType)
        DTRuler = kwargs.get("dt_ruler", dts)
        StartInd = (DTRuler.index(dts[0]) if dts[0] in DTRuler else 0)
        if (self.i_look_backMode == "扩张窗口") and (self.iInitData is not None) and (self.iInitData.shape[0] > 0):
            if self.iInitData.index[-1] not in DTRuler:
                self._QN_logger.warning("注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name,))
            else:
                StartInd = min(StartInd, DTRuler.index(self.iInitData.index[-1]) + 1)
        EndInd = (DTRuler.index(dts[-1]) if dts[-1] in DTRuler else len(DTRuler) - 1)
        if StartInd > EndInd:
            return pd.DataFrame(index=dts, columns=ids)
        nID = len(ids)
        DescriptorData = []
        for i, iDescriptor in enumerate(self._Descriptors):
            iDTs = DTRuler[max(StartInd - self.look_back[i], 0):EndInd + 1]
            if iDTs:
                iDescriptorData = iDescriptor.readData(ids=ids, dts=iDTs, **kwargs).values
            else:
                iDescriptorData = np.full((0, nID), np.nan)
            if StartInd < self.look_back[i]:
                iLookBackData = np.full((self.look_back[i] - StartInd, nID), np.nan)
                iDescriptorData = np.r_[iLookBackData, iDescriptorData]
            DescriptorData.append(iDescriptorData)
        StdData = self._calcData(
            ids=ids,
            dts=DTRuler[StartInd:EndInd + 1],
            descriptor_data=DescriptorData,
            dt_ruler=DTRuler
        )
        return pd.DataFrame(StdData, index=DTRuler[StartInd:EndInd + 1], columns=ids).loc[dts, :]

    def _calcData(
        self,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray],
        dt_ruler: List[Any]
    ) -> np.ndarray:
        """计算数据（策略分派入口）

        Args:
            ids: ID列表
            dts: 时间点列表
            descriptor_data: 描述子数据列表
            dt_ruler: 时间标尺

        Returns:
            计算结果数组
        """
        StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, descriptor_data = \
            self._prepare_lookback_data(ids, dts, descriptor_data, dt_ruler)
        handler_name = self._DT_ID_DISPATCH.get((self.DTMode, self.IDMode))
        if handler_name:
            return getattr(self, handler_name)(
                StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen,
                ids, dts, descriptor_data
            )
        return self.Operator(self, DTRuler, ids, descriptor_data, self.ModelArgs)

    def _calcData_single_time_single_id(
        self,
        StdData: np.ndarray,
        iStartInd: int,
        DTRuler: List[Any],
        StartIndAndLen: List[Tuple[int, int]],
        MaxLookBack: int,
        MaxLen: int,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray]
    ) -> np.ndarray:
        """单时点-单ID模式计算"""
        for i, iDT in enumerate(dts):
            iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
            for j, jID in enumerate(ids):
                x = []
                for k, kDescriptorData in enumerate(descriptor_data):
                    kStartInd, kLen = StartIndAndLen[k]
                    x.append(kDescriptorData[max(0, kStartInd + 1 + i - kLen):kStartInd + 1 + i, j])
                StdData[iStartInd + i, j] = self.Operator(self, iDTs, jID, x, self.ModelArgs)
        return StdData[iStartInd:, :]

    def _calcData_single_time_multi_id(
        self,
        StdData: np.ndarray,
        iStartInd: int,
        DTRuler: List[Any],
        StartIndAndLen: List[Tuple[int, int]],
        MaxLookBack: int,
        MaxLen: int,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray]
    ) -> np.ndarray:
        """单时点-多ID模式计算"""
        for i, iDT in enumerate(dts):
            iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
            x = []
            for k, kDescriptorData in enumerate(descriptor_data):
                kStartInd, kLen = StartIndAndLen[k]
                x.append(kDescriptorData[max(0, kStartInd + 1 + i - kLen):kStartInd + 1 + i])
            StdData[iStartInd + i, :] = self.Operator(self, iDTs, ids, x, self.ModelArgs)
        return StdData[iStartInd:, :]

    def _calcData_multi_time_single_id(
        self,
        StdData: np.ndarray,
        iStartInd: int,
        DTRuler: List[Any],
        StartIndAndLen: List[Tuple[int, int]],
        MaxLookBack: int,
        MaxLen: int,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray]
    ) -> np.ndarray:
        """多时点-单ID模式计算"""
        for j, jID in enumerate(ids):
            StdData[iStartInd:, j] = self.Operator(
                self, DTRuler, jID,
                [kDescriptorData[:, j] for kDescriptorData in descriptor_data],
                self.ModelArgs
            )
        return StdData[iStartInd:, :]

    def _calcData_multi_time_multi_id(
        self,
        StdData: np.ndarray,
        iStartInd: int,
        DTRuler: List[Any],
        StartIndAndLen: List[Tuple[int, int]],
        MaxLookBack: int,
        MaxLen: int,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray]
    ) -> np.ndarray:
        """多时点-多ID模式计算"""
        return self.Operator(self, DTRuler, ids, descriptor_data, self.ModelArgs)

    def __QN_prepare_cache_data__(self, ids: Optional[List[Any]] = None) -> pd.DataFrame:
        """准备缓存数据

        Args:
            ids: ID列表

        Returns:
            标准数据DataFrame
        """
        PID = self._OperationMode._iPID
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = (
            self._OperationMode.DTRuler.index(StartDT),
            self._OperationMode.DTRuler.index(EndDT)
        )
        DTs = list(self._OperationMode.DTRuler[StartInd:EndInd + 1])
        IDs = partition_ids_for_pid(self._OperationMode, self._OperationMode._FactorPrepareIDs[self.Name], PID)
        if IDs:
            DescriptorData = []
            for i, iDescriptor in enumerate(self._Descriptors):
                iStartInd = StartInd - self.look_back[i]
                iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
                iDescriptorData = iDescriptor._QN_get_data(iDTs, pids=[PID]).values
                if iStartInd < 0:
                    iDescriptorData = np.r_[
                        np.full(shape=(abs(iStartInd), iDescriptorData.shape[1]), fill_value=np.nan),
                        iDescriptorData
                    ]
                DescriptorData.append(iDescriptorData)
            StdData = self._calcData(
                ids=IDs, dts=DTs, descriptor_data=DescriptorData,
                dt_ruler=self._OperationMode.DTRuler
            )
            StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
        else:
            StdData = create_empty_dataframe(DTs, [], self.DataType)
        write_cache_file(
            self._OperationMode, PID, self.Name,
            self._OperationMode._FactorID[self.Name], StdData, IDs
        )
        self._isCacheDataOK = True
        return StdData


# 截面运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多时点，则该值为 [时点]
# iid: 当前待计算的ID, 如果输出形式为全截面, 则该值为 [ID], 该序列在并发时也是全体截面 ID
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 那么 x 元素为 array(shape=(nID, )), 如果输出形式为全截面返回 array(shape=(nID, )), 否则返回单个值
# 如果运算时点参数为多时点, 那么 x 元素为 array(shape=(nDT, nID)), 如果输出形式为全截面返回 array(shape=(nDT, nID)), 否则返回 array(shape=(nDT, ))
class SectionOperation(DerivativeFactor):
    """截面运算
    
    对描述子进行截面运算，即在同一时点对全截面ID进行计算。
    
    Attributes:
        dt_mode: 运算时点模式
        output_mode: 输出形式
        descriptor_section: 描述子截面列表
    """
    dt_mode: DTModeType = DTModeType.SINGLE
    output_mode: OutputModeType = OutputModeType.FULL_SECTION
    descriptor_section: List = None

    def __init__(self, name: str = "", descriptors: List[Factor] = None, sys_args: Dict = None, **kwargs):
        super().__init__(name=name, descriptors=descriptors, sys_args=sys_args, **kwargs)
        if descriptors:
            self.descriptor_section = [None] * len(descriptors)

    def readData(self, ids: List[Any], dts: List[Any], **kwargs) -> pd.DataFrame:
        """读取并计算数据

        Args:
            ids: ID列表
            dts: 时间点列表
            **kwargs: 其他关键字参数，支持 section_ids 指定截面ID

        Returns:
            计算结果DataFrame，index为dts，columns为ids
        """
        SectionIDs = kwargs.pop("section_ids", ids)
        DescriptorData = []
        for i, iDescriptor in enumerate(self._Descriptors):
            iSectionIDs = self.DescriptorSection[i]
            if iSectionIDs is None:
                iSectionIDs = SectionIDs
            DescriptorData.append(iDescriptor.readData(ids=iSectionIDs, dts=dts, **kwargs).values)
        StdData = self._calcData(ids=SectionIDs, dts=dts, descriptor_data=DescriptorData)
        return pd.DataFrame(StdData, index=dts, columns=SectionIDs).loc[:, ids]

    def _QN_init_operation(
        self,
        start_dt: Any,
        dt_dict: Dict[str, Any],
        prepare_ids: List[Any],
        id_dict: Dict[str, List[Any]]
    ) -> None:
        """初始化运算环境

        Args:
            start_dt: 起始时间点
            dt_dict: 时间点字典
            prepare_ids: 准备计算的ID列表
            id_dict: ID字典
        """
        OldStartDT = dt_dict.get(self.Name, None)
        if (OldStartDT is None) or (start_dt < OldStartDT):
            dt_dict[self.Name] = start_dt
            StartInd, EndInd = (
                self._OperationMode.DTRuler.index(dt_dict[self.Name]),
                self._OperationMode.DTRuler.index(self._OperationMode.DateTimes[-1])
            )
            DTs = self._OperationMode.DTRuler[StartInd:EndInd + 1]
            DTPartition = partition_list(DTs, len(self._OperationMode._PIDs))
            self._PID_DTs = {iPID: DTPartition[i] for i, iPID in enumerate(self._OperationMode._PIDs)}
        PrepareIDs = id_dict.setdefault(self.Name, prepare_ids)
        if prepare_ids != PrepareIDs:
            raise FactorError("因子 %s 指定了不同的截面!" % self.Name)
        for i, iDescriptor in enumerate(self._Descriptors):
            if self.DescriptorSection[i] is None:
                iDescriptor._QN_init_operation(start_dt, dt_dict, prepare_ids, id_dict)
            else:
                iDescriptor._QN_init_operation(start_dt, dt_dict, self.DescriptorSection[i], id_dict)
        if (self._OperationMode.SubProcessNum > 0) and (self.Name not in self._OperationMode._Event):
            self._OperationMode._Event[self.Name] = (Queue(), Event())

    def _calcData(
        self,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray]
    ) -> np.ndarray:
        """计算数据（策略分派入口）

        Args:
            ids: ID列表
            dts: 时间点列表
            descriptor_data: 描述子数据列表

        Returns:
            计算结果数组
        """
        StdData = create_std_data(dts, ids, self.DataType)
        handler_name = self._OUTPUT_DT_DISPATCH.get((self.OutputMode, self.DTMode))
        if handler_name:
            return getattr(self, handler_name)(StdData, ids, dts, descriptor_data)
        return StdData

    def _calcData_full_section_single_time(
        self,
        StdData: np.ndarray,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray]
    ) -> np.ndarray:
        """全截面-单时点模式计算"""
        for i, iDT in enumerate(dts):
            StdData[i, :] = self.Operator(
                self, iDT, ids,
                [kDescriptorData[i] for kDescriptorData in descriptor_data],
                self.ModelArgs
            )
        return StdData

    def _calcData_full_section_multi_time(
        self,
        StdData: np.ndarray,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray]
    ) -> np.ndarray:
        """全截面-多时点模式计算"""
        return self.Operator(self, dts, ids, descriptor_data, self.ModelArgs)

    def _calcData_single_id_single_time(
        self,
        StdData: np.ndarray,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray]
    ) -> np.ndarray:
        """单ID-单时点模式计算"""
        for i, iDT in enumerate(dts):
            x = [kDescriptorData[i] for kDescriptorData in descriptor_data]
            for j, jID in enumerate(ids):
                StdData[i, j] = self.Operator(self, iDT, jID, x, self.ModelArgs)
        return StdData

    def _calcData_single_id_multi_time(
        self,
        StdData: np.ndarray,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray]
    ) -> np.ndarray:
        """单ID-多时点模式计算"""
        for j, jID in enumerate(ids):
            StdData[:, j] = self.Operator(self, dts, jID, descriptor_data, self.ModelArgs)
        return StdData

    def __QN_prepare_cache_data__(self, ids: Optional[List[Any]] = None) -> pd.DataFrame:
        """准备缓存数据

        Args:
            ids: ID列表

        Returns:
            标准数据DataFrame
        """
        DTs = list(self._PID_DTs[self._OperationMode._iPID])
        IDs = self._OperationMode._FactorPrepareIDs[self.Name]
        if IDs is None:
            IDs = list(self._OperationMode.IDs)
        if len(DTs) == 0:
            iDTs = [self._OperationMode.DateTimes[-1]]
            for i, iDescriptor in enumerate(self._Descriptors):
                iDescriptor._QN_get_data(iDTs, pids=None)
            StdData = create_empty_dataframe([], IDs, self.DataType, include_index=False)
        elif IDs:
            StdData = self._calcData(
                ids=IDs, dts=DTs,
                descriptor_data=[iDescriptor._QN_get_data(DTs, pids=None).values
                                 for i, iDescriptor in enumerate(self._Descriptors)]
            )
            StdData = pd.DataFrame(StdData, index=DTs, columns=IDs)
        else:
            StdData = create_empty_dataframe(DTs, [], self.DataType)
        PID_IDs = (
            self._OperationMode._PID_IDs
            if self._OperationMode._FactorPrepareIDs[self.Name] is None
            else {
                self._OperationMode._PIDs[i]: iSubIDs
                for i, iSubIDs in enumerate(partition_list(IDs, len(self._OperationMode._PIDs)))
            }
        )
        write_cache_files_for_all_pids(
            self._OperationMode, PID_IDs, self.Name,
            self._OperationMode._FactorID[self.Name], StdData
        )
        StdData = None
        if self._OperationMode.SubProcessNum > 0:
            Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self._isCacheDataOK = True
        return StdData


# 面板运算
# f: 该算子所属的因子, 因子对象
# idt: 当前待计算的时点, 如果运算日期为多日期，则该值为 [回溯期数]+[时点]
# iid: 当前待计算的 ID, 如果输出形式为全截面, 则该值为 [ID], 该序列在并发时也是全体截面 ID
# x: 描述子当期的数据, [array]
# args: 参数, {参数名:参数值}
# 如果运算时点参数为单时点, 那么 x 元素为 array(shape=(回溯期数, nID)), 如果输出形式为全截面返回 array(shape=(nID, )), 否则返回单个值
# 如果运算时点参数为多时点, 那么 x 元素为 array(shape=(回溯期数+nDT, nID)), 如果输出形式为全截面返回 array(shape=(nDT, nID)), 否则返回 array(shape=(nDT, ))
class PanelOperation(_LookBackOperation):
    """面板运算
    
    结合时间序列和截面运算，对描述子进行面板数据计算。
    
    Attributes:
        dt_mode: 运算时点模式
        output_mode: 输出形式
        descriptor_section: 描述子截面列表
    """
    dt_mode: DTModeType = DTModeType.SINGLE
    output_mode: OutputModeType = OutputModeType.FULL_SECTION
    descriptor_section: List = None

    def __init__(self, name: str = "", descriptors: List[Factor] = None, sys_args: Dict = None, **kwargs):
        super().__init__(name=name, descriptors=descriptors, sys_args=sys_args, **kwargs)
        if descriptors:
            self.descriptor_section = [None] * len(descriptors)

    def _QN_init_operation(self, start_dt, dt_dict, prepare_ids, id_dict):
        if len(self._descriptors) > len(self.look_back): raise FactorError(
            "面板运算因子 : '%s' 的参数'回溯期数'序列长度小于描述子个数!" % self.name)
        OldStartDT = dt_dict.get(self.Name, None)
        DTRuler = self._OperationMode.DTRuler
        if (OldStartDT is None) or (start_dt < OldStartDT):
            StartDT = dt_dict[self.Name] = start_dt
            StartInd, EndInd = DTRuler.index(StartDT), DTRuler.index(self._OperationMode.DateTimes[-1])
            if (self.i_look_backMode == "扩张窗口") and (self.iInitData is not None) and (self.iInitData.shape[0] > 0):
                if self.iInitData.index[-1] not in self._OperationMode.DTRuler:
                    self._QN_logger.warning("注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name,))
                else:
                    StartInd = min(StartInd, self._OperationMode.DTRuler.index(self.iInitData.index[-1]) + 1)
            DTs = DTRuler[StartInd:EndInd + 1]
            if self.i_look_backMode == "扩张窗口":
                DTPartition = [DTs] + [[]] * (len(self._OperationMode._PIDs) - 1)
            else:
                DTPartition = partition_list(DTs, len(self._OperationMode._PIDs))
            self._PID_DTs = {iPID: DTPartition[i] for i, iPID in enumerate(self._OperationMode._PIDs)}
        else:
            StartInd = DTRuler.index(OldStartDT)
        PrepareIDs = id_dict.setdefault(self.Name, prepare_ids)
        if prepare_ids != PrepareIDs: raise FactorError("因子 %s 指定了不同的截面!" % self.Name)
        for i, iDescriptor in enumerate(self._Descriptors):
            iStartInd = StartInd - self.look_back[i]
            if iStartInd < 0: self._QN_logger.warning(
                "注意: 对于因子 '%s' 的描述子 '%s', 时点标尺长度不足!" % (self.Name, iDescriptor.Name))
            iStartDT = DTRuler[max(0, iStartInd)]
            if self.DescriptorSection[i] is None:
                iDescriptor._QN_init_operation(iStartDT, dt_dict, prepare_ids, id_dict)
            else:
                iDescriptor._QN_init_operation(iStartDT, dt_dict, self.DescriptorSection[i], id_dict)
        if (self._OperationMode.SubProcessNum > 0) and (self.Name not in self._OperationMode._Event):
            self._OperationMode._Event[self.Name] = (Queue(), Event())

    def readData(self, ids, dts, **kwargs):
        DTRuler = kwargs.get("dt_ruler", dts)
        SectionIDs = kwargs.pop("section_ids", ids)
        StartInd = (DTRuler.index(dts[0]) if dts[0] in DTRuler else 0)
        if (self.i_look_backMode == "扩张窗口") and (self.iInitData is not None) and (self.iInitData.shape[0] > 0):
            if self.iInitData.index[-1] not in DTRuler:
                self._QN_logger.warning("注意: 因子 '%s' 的初始值不在时点标尺的范围内, 初始值和时点标尺之间的时间间隔将被忽略!" % (self.Name,))
            else:
                StartInd = min(StartInd, DTRuler.index(self.iInitData.index[-1]) + 1)
        EndInd = (DTRuler.index(dts[-1]) if dts[-1] in DTRuler else len(DTRuler) - 1)
        if StartInd > EndInd: return pd.DataFrame(index=dts, columns=ids)
        DescriptorData = []
        for i, iDescriptor in enumerate(self._Descriptors):
            iDTs = DTRuler[max(StartInd - self.look_back[i], 0):EndInd + 1]
            iSectionIDs = self.DescriptorSection[i]
            if iSectionIDs is None: iSectionIDs = SectionIDs
            iIDNum = len(iSectionIDs)
            if iDTs:
                iDescriptorData = iDescriptor.readData(ids=iSectionIDs, dts=iDTs, **kwargs).values
            else:
                iDescriptorData = np.full((0, iIDNum), np.nan)
            if StartInd < self.look_back[i]:
                iLookBackData = np.full((self.look_back[i] - StartInd, iIDNum), np.nan)
                iDescriptorData = np.r_[iLookBackData, iDescriptorData]
            DescriptorData.append(iDescriptorData)
        StdData = self._calcData(ids=SectionIDs, dts=DTRuler[StartInd:EndInd + 1], descriptor_data=DescriptorData,
                                 dt_ruler=DTRuler)
        return pd.DataFrame(StdData, index=DTRuler[StartInd:EndInd + 1], columns=SectionIDs).loc[dts, ids]

    def _calcData(self, ids, dts, descriptor_data, dt_ruler):
        StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, descriptor_data = \
            self._prepare_lookback_data(ids, dts, descriptor_data, dt_ruler)
        handler_name = self._OUTPUT_DT_DISPATCH.get((self.OutputMode, self.DTMode))
        if handler_name:
            return getattr(self, handler_name)(StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, ids, dts, descriptor_data)
        return self.Operator(self, DTRuler, ids, descriptor_data, self.ModelArgs)

    def _calcData_full_section_single_time(self, StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, ids, dts, descriptor_data):
        for i, iDT in enumerate(dts):
            iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
            x = []
            for k, kDescriptorData in enumerate(descriptor_data):
                kStartInd, kLen = StartIndAndLen[k]
                x.append(kDescriptorData[max(0, kStartInd + 1 + i - kLen):kStartInd + 1 + i])
            StdData[iStartInd + i, :] = self.Operator(self, iDTs, ids, x, self.ModelArgs)
        return StdData[iStartInd:, :]

    def _calcData_full_section_multi_time(self, StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, ids, dts, descriptor_data):
        return self.Operator(self, DTRuler, ids, descriptor_data, self.ModelArgs)

    def _calcData_single_id_single_time(self, StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, ids, dts, descriptor_data):
        for i, iDT in enumerate(dts):
            iDTs = DTRuler[max(0, MaxLookBack + i + 1 - MaxLen):i + 1 + MaxLookBack]
            x = []
            for k, kDescriptorData in enumerate(descriptor_data):
                kStartInd, kLen = StartIndAndLen[k]
                x.append(kDescriptorData[max(0, kStartInd + 1 + i - kLen):kStartInd + 1 + i])
            for j, jID in enumerate(ids):
                StdData[iStartInd + i, j] = self.Operator(self, iDTs, jID, x, self.ModelArgs)
        return StdData[iStartInd:, :]

    def _calcData_single_id_multi_time(self, StdData, iStartInd, DTRuler, StartIndAndLen, MaxLookBack, MaxLen, ids, dts, descriptor_data):
        for j, jID in enumerate(ids):
            StdData[iStartInd:, j] = self.Operator(self, DTRuler, jID, descriptor_data, self.ModelArgs)
        return StdData[iStartInd:, :]

    def __QN_prepare_cache_data__(self, ids=None):
        DTs = list(self._PID_DTs[self._OperationMode._iPID])
        IDs = self._OperationMode._FactorPrepareIDs[self.Name]
        if IDs is None:
            IDs = list(self._OperationMode.IDs)
        if len(DTs) == 0:
            iDTs = [self._OperationMode.DateTimes[-1]]
            for i, iDescriptor in enumerate(self._Descriptors):
                iDescriptor._QN_get_data(iDTs, pids=None)
            StdData = create_empty_dataframe([], IDs, self.DataType, include_index=False)
        elif IDs:
            DescriptorData, StartInd = [], self._OperationMode.DTRuler.index(DTs[0])
            for i, iDescriptor in enumerate(self._Descriptors):
                iStartInd = StartInd - self.look_back[i]
                iDTs = list(self._OperationMode.DTRuler[max(0, iStartInd):StartInd]) + DTs
                iDescriptorData = iDescriptor._QN_get_data(iDTs, pids=None).values
                if iStartInd < 0: iDescriptorData = np.r_[
                    np.full(shape=(abs(iStartInd), iDescriptorData.shape[1]), fill_value=np.nan), iDescriptorData]
                DescriptorData.append(iDescriptorData)
            StdData = self._calcData(ids=IDs, dts=DTs, descriptor_data=DescriptorData,
                                     dt_ruler=self._OperationMode.DTRuler)
            DescriptorData, iDescriptorData, StdData = None, None, pd.DataFrame(StdData, index=DTs, columns=IDs)
        else:
            StdData = create_empty_dataframe(DTs, [], self.DataType)
        PID_IDs = self._OperationMode._PID_IDs if self._OperationMode._FactorPrepareIDs[self.Name] is None else \
            {self._OperationMode._PIDs[i]: iSubIDs for i, iSubIDs in
             enumerate(partition_listMovingSampling(IDs, len(self._OperationMode._PIDs)))}
        write_cache_files_for_all_pids(self._OperationMode, PID_IDs, self.Name,
                                       self._OperationMode._FactorID[self.Name], StdData)
        StdData = None
        if self._OperationMode.SubProcessNum > 0:
            Sub2MainQueue, PIDEvent = self._OperationMode._Event[self.Name]
            Sub2MainQueue.put(1)
            PIDEvent.wait()
        self._isCacheDataOK = True
        return StdData
