# coding=utf-8
"""
因子运算操作类

替代 QuantStudio.FactorDataBase.FactorOperation
"""

import os
import shelve
from multiprocessing import Event, Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from traits.api import Enum, Int, List as TraitList

from QuantNodes.core.factor_base import DerivativeFactor


def _default_operator(f, idt, iid, x, args):
    """默认算子，返回 NaN"""
    return np.nan


class PointOperation(DerivativeFactor):
    """
    单点运算

    对描述子进行单点运算

    Attributes:
        DTMode: 运算时点 ("单时点" 或 "多时点")
        IDMode: 运算ID ("单ID" 或 "多ID")
    """

    DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=3)
    IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=4)

    def __init__(
        self,
        name: str = "",
        descriptors: List[DerivativeFactor] = None,
        sys_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(name=name, descriptors=descriptors, sys_args=sys_args, **kwargs)

    def readData(
        self,
        ids: List[Any],
        dts: List[Any],
        **kwargs,
    ) -> pd.DataFrame:
        """读取并计算数据"""
        std_data = self._calcData(
            ids=ids,
            dts=dts,
            descriptor_data=[iDescriptor.readData(ids=ids, dts=dts, **kwargs).values
                           for iDescriptor in self._Descriptors],
        )
        return pd.DataFrame(std_data, index=dts, columns=ids)

    def _QN_init_operation(
        self,
        start_dt: Any,
        dt_dict: Dict[str, Any],
        prepare_ids: List[Any],
        id_dict: Dict[str, List[Any]],
    ) -> None:
        """初始化运算模式"""
        super()._QN_init_operation(start_dt, dt_dict, prepare_ids, id_dict)
        for i, iDescriptor in enumerate(self._Descriptors):
            iDescriptor._QN_init_operation(dt_dict[self.Name], dt_dict, prepare_ids, id_dict)

    def _calcData(
        self,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray],
    ) -> np.ndarray:
        """计算数据"""
        if self.DTMode == "多时点" and self.IDMode == "多ID":
            std_data = self.Operator(self, dts, ids, descriptor_data, self.ModelArgs)
        else:
            if self.DataType == "double":
                std_data = np.full(shape=(len(dts), len(ids)), fill_value=np.nan, dtype="float")
            else:
                std_data = np.full(shape=(len(dts), len(ids)), fill_value=None, dtype="O")

            if self.DTMode == "单时点" and self.IDMode == "单ID":
                for i, iDT in enumerate(dts):
                    for j, jID in enumerate(ids):
                        std_data[i, j] = self.Operator(
                            self, iDT, jID,
                            [iData[i, j] for iData in descriptor_data],
                            self.ModelArgs,
                        )
            elif self.DTMode == "多时点" and self.IDMode == "单ID":
                for j, jID in enumerate(ids):
                    std_data[:, j] = self.Operator(
                        self, dts, jID,
                        [iData[:, j] for iData in descriptor_data],
                        self.ModelArgs,
                    )
            elif self.DTMode == "单时点" and self.IDMode == "多ID":
                for i, iDT in enumerate(dts):
                    std_data[i, :] = self.Operator(
                        self, iDT, ids,
                        [iData[i, :] for iData in descriptor_data],
                        self.ModelArgs,
                    )
        return std_data

    def __QN_prepare_cache_data__(self, ids: Optional[List[Any]] = None) -> np.ndarray:
        """准备缓存数据"""
        pid = self._OperationMode._iPID
        start_dt = self._OperationMode._FactorStartDT[self.Name]
        end_dt = self._OperationMode.DateTimes[-1]
        start_ind = self._OperationMode.DTRuler.index(start_dt)
        end_ind = self._OperationMode.DTRuler.index(end_dt)
        dts = list(self._OperationMode.DTRuler[start_ind:end_ind + 1])
        
        prepare_ids = self._OperationMode._FactorPrepareIDs.get(self.Name)
        if prepare_ids is None:
            prepare_ids = list(self._OperationMode._PID_IDs.get(pid, []))
        
        if prepare_ids:
            from QuantNodes.core.tools import partition_list_moving_sampling
            sampled_ids = partition_list_moving_sampling(
                prepare_ids, len(self._OperationMode._PIDs)
            )[self._OperationMode._PIDs.index(pid)]
            
            std_data = self._calcData(
                ids=sampled_ids,
                dts=dts,
                descriptor_data=[
                    iDescriptor._QN_get_data(dts, pids=[pid]).values
                    for iDescriptor in self._Descriptors
                ],
            )
            std_data = pd.DataFrame(std_data, index=dts, columns=sampled_ids)
        else:
            dtype = "float" if self.DataType == "double" else "O"
            std_data = pd.DataFrame(index=dts, columns=prepare_ids, dtype=dtype)

        with self._OperationMode._PID_Lock.get(pid, type("DummyLock", (), {"__enter__": lambda s: None, "__exit__": lambda s, *a: None})()):
            cache_dir = self._OperationMode._CacheDataDir + os.sep + pid
            os.makedirs(cache_dir, exist_ok=True)
            with shelve.open(cache_dir + os.sep + self.Name + str(self._OperationMode._FactorID[self.Name])) as cache_file:
                cache_file["StdData"] = std_data
                cache_file["_QN_ids"] = sampled_ids if prepare_ids else []
        
        self._isCacheDataOK = True
        return std_data


class TimeOperation(DerivativeFactor):
    """
    时间序列运算

    对描述子进行时间序列运算（滚动/扩展窗口）

    Attributes:
        DTMode: 运算时点 ("单时点" 或 "多时点")
        IDMode: 运算ID ("单ID" 或 "多ID")
        LookBack: 回溯期数列表
        LookBackMode: 回溯模式列表
        iLookBack: 自身回溯期数
        iLookBackMode: 自身回溯模式
    """

    DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=3)
    IDMode = Enum("单ID", "多ID", arg_type="SingleOption", label="运算ID", order=4)
    LookBack = TraitList(arg_type="ArgList", label="回溯期数", order=5)
    LookBackMode = TraitList(Enum("滚动窗口", "扩张窗口"), arg_type="ArgList", label="回溯模式", order=6)
    iLookBack = Int(0, arg_type="Integer", label="自身回溯期数", order=7)
    iLookBackMode = Enum("滚动窗口", "扩张窗口", arg_type="SingleOption", label="自身回溯模式", order=8)

    def __init__(
        self,
        name: str = "",
        descriptors: List[DerivativeFactor] = None,
        sys_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self._LookBack = [0] * len(descriptors) if descriptors else [0]
        self._LookBackMode = ["滚动窗口"] * len(descriptors) if descriptors else ["滚动窗口"]
        self._iInitData = None
        super().__init__(name=name, descriptors=descriptors, sys_args=sys_args, **kwargs)

    def __QN_initArgs__(self, sys_args: Optional[Dict[str, Any]] = None) -> None:
        """初始化参数"""
        super().__QN_initArgs__(sys_args)
        if not self.LookBack:
            self.LookBack = [0] * len(self._Descriptors)
        if not self.LookBackMode:
            self.LookBackMode = ["滚动窗口"] * len(self._Descriptors)
        if self._Descriptors:
            self._LookBack = list(self.LookBack)
            self._LookBackMode = list(self.LookBackMode)

    def readData(
        self,
        ids: List[Any],
        dts: List[Any],
        **kwargs,
    ) -> pd.DataFrame:
        """读取并计算数据"""
        lookback_ids = self._get_lookback_ids(ids)
        lookback_dts = self._get_lookback_dts(dts)
        
        descriptor_data = []
        for i, iDescriptor in enumerate(self._Descriptors):
            iLookBack = self._LookBack[i] if i < len(self._LookBack) else 0
            iLookBackMode = self._LookBackMode[i] if i < len(self._LookBackMode) else "滚动窗口"
            
            iDTS = lookback_dts[i] if i < len(lookback_dts) else dts
            descriptor_data.append(
                iDescriptor.readData(ids=lookback_ids[i], dts=iDTS, **kwargs).values
            )
        
        std_data = self._calcData(ids=ids, dts=dts, descriptor_data=descriptor_data)
        return pd.DataFrame(std_data, index=dts, columns=ids)

    def _get_lookback_ids(self, ids: List[Any]) -> List[List[Any]]:
        """获取回溯ID列表"""
        lookback_ids = []
        for i, iDescriptor in enumerate(self._Descriptors):
            if i < len(self._Descriptors):
                lookback_ids.append(ids)
            else:
                lookback_ids.append(ids)
        return lookback_ids

    def _get_lookback_dts(self, dts: List[Any]) -> List[List[Any]]:
        """获取回溯时间列表"""
        return [dts] * len(self._Descriptors)

    def _calcData(
        self,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray],
    ) -> np.ndarray:
        """计算数据"""
        n_dt = len(dts)
        n_id = len(ids)
        
        if self.DataType == "double":
            std_data = np.full(shape=(n_dt, n_id), fill_value=np.nan, dtype="float")
        else:
            std_data = np.full(shape=(n_dt, n_id), fill_value=None, dtype="O")
        
        for j, j_id in enumerate(ids):
            for i, i_dt in enumerate(dts):
                std_data[i, j] = self.Operator(
                    self, i_dt, j_id,
                    [i_data[i, j] if i_data.shape[1] > j else np.nan for i_data in descriptor_data],
                    self.ModelArgs,
                )
        
        return std_data

    def __QN_prepare_cache_data__(self, ids: Optional[List[Any]] = None) -> np.ndarray:
        """准备缓存数据"""
        raise NotImplementedError("TimeOperation.__QN_prepare_cache_data__ 需要实现")


class SectionOperation(DerivativeFactor):
    """
    截面运算

    对描述子进行截面运算

    Attributes:
        DTMode: 运算时点 ("单时点" 或 "多时点")
        OutputMode: 输出形式 ("全截面" 或 "单ID")
        DescriptorSection: 描述子截面
    """

    DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=3)
    OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=4)
    DescriptorSection = TraitList(arg_type="List", label="描述子截面", order=5)

    def __init__(
        self,
        name: str = "",
        descriptors: List[DerivativeFactor] = None,
        sys_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(name=name, descriptors=descriptors, sys_args=sys_args, **kwargs)

    def __QN_initArgs__(self, sys_args: Optional[Dict[str, Any]] = None) -> None:
        """初始化参数"""
        super().__QN_initArgs__(sys_args)
        if not self.DescriptorSection:
            self.DescriptorSection = [None] * len(self._Descriptors)

    def readData(
        self,
        ids: List[Any],
        dts: List[Any],
        **kwargs,
    ) -> pd.DataFrame:
        """读取并计算数据"""
        section_ids = kwargs.pop("section_ids", ids)
        
        descriptor_data = []
        for i, iDescriptor in enumerate(self._Descriptors):
            i_section_ids = self.DescriptorSection[i] if i < len(self.DescriptorSection) else None
            if i_section_ids is None:
                i_section_ids = section_ids
            descriptor_data.append(
                iDescriptor.readData(ids=i_section_ids, dts=dts, **kwargs).values
            )
        
        std_data = self._calcData(ids=section_ids, dts=dts, descriptor_data=descriptor_data)
        return pd.DataFrame(std_data, index=dts, columns=section_ids).loc[:, ids]

    def _calcData(
        self,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray],
    ) -> np.ndarray:
        """计算数据"""
        n_dt = len(dts)
        n_id = len(ids)
        
        if self.DataType == "double":
            std_data = np.full(shape=(n_dt, n_id), fill_value=np.nan, dtype="float")
        else:
            std_data = np.full(shape=(n_dt, n_id), fill_value=None, dtype="O")

        if self.OutputMode == "全截面":
            if self.DTMode == "单时点":
                for i, iDT in enumerate(dts):
                    std_data[i, :] = self.Operator(
                        self, iDT, ids,
                        [k_desc_data[i] for k_desc_data in descriptor_data],
                        self.ModelArgs,
                    )
            else:
                std_data = self.Operator(self, dts, ids, descriptor_data, self.ModelArgs)
        else:
            if self.DTMode == "单时点":
                for i, iDT in enumerate(dts):
                    x = [k_desc_data[i] for k_desc_data in descriptor_data]
                    for j, jID in enumerate(ids):
                        std_data[i, j] = self.Operator(self, iDT, jID, x, self.ModelArgs)
            else:
                for j, jID in enumerate(ids):
                    std_data[:, j] = self.Operator(
                        self, dts, jID, descriptor_data, self.ModelArgs,
                    )
        
        return std_data

    def __QN_prepare_cache_data__(self, ids: Optional[List[Any]] = None) -> np.ndarray:
        """准备缓存数据"""
        raise NotImplementedError("SectionOperation.__QN_prepare_cache_data__ 需要实现")


class PanelOperation(DerivativeFactor):
    """
    面板运算

    结合时间序列和截面运算

    Attributes:
        DTMode: 运算时点 ("单时点" 或 "多时点")
        LookBack: 回溯期数列表
        OutputMode: 输出形式 ("全截面" 或 "单ID")
    """

    DTMode = Enum("单时点", "多时点", arg_type="SingleOption", label="运算时点", order=3)
    LookBack = TraitList(arg_type="ArgList", label="回溯期数", order=4)
    OutputMode = Enum("全截面", "单ID", arg_type="SingleOption", label="输出形式", order=5)

    def __init__(
        self,
        name: str = "",
        descriptors: List[DerivativeFactor] = None,
        sys_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(name=name, descriptors=descriptors, sys_args=sys_args, **kwargs)

    def __QN_initArgs__(self, sys_args: Optional[Dict[str, Any]] = None) -> None:
        """初始化参数"""
        super().__QN_initArgs__(sys_args)
        if not self.LookBack:
            self.LookBack = [0] * len(self._Descriptors)

    def readData(
        self,
        ids: List[Any],
        dts: List[Any],
        **kwargs,
    ) -> pd.DataFrame:
        """读取并计算数据"""
        raise NotImplementedError("PanelOperation.readData 需要实现")

    def _calcData(
        self,
        ids: List[Any],
        dts: List[Any],
        descriptor_data: List[np.ndarray],
    ) -> np.ndarray:
        """计算数据"""
        raise NotImplementedError("PanelOperation._calcData 需要实现")

    def __QN_prepare_cache_data__(self, ids: Optional[List[Any]] = None) -> np.ndarray:
        """准备缓存数据"""
        raise NotImplementedError("PanelOperation.__QN_prepare_cache_data__ 需要实现")
