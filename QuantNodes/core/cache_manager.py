# coding=utf-8
"""
缓存管理器

替代 QuantStudio 中与缓存和进程管理相关的功能
"""

import gc
import mmap
import os
import pickle
import shelve
from multiprocessing import Lock
from typing import Any, Dict, List, Optional

import pandas as pd

from QuantNodes.core.tools import get_shelve_file_suffix


class ErgodicMode:
    """
    遍历模式参数对象

    管理因子遍历模式下的缓存策略
    """

    def __init__(
        self,
        forward_period: int = 600,
        backward_period: int = 1,
        cache_mode: str = "因子",
        max_factor_cache_num: int = 60,
        max_id_cache_num: int = 10000,
        cache_size: int = 300,
        ergodic_dts: Optional[List] = None,
        ergodic_ids: Optional[List] = None,
    ):
        self.ForwardPeriod = forward_period
        self.BackwardPeriod = backward_period
        self.CacheMode = cache_mode
        self.MaxFactorCacheNum = max_factor_cache_num
        self.MaxIDCacheNum = max_id_cache_num
        self.CacheSize = cache_size
        self.ErgodicDTs = ergodic_dts or []
        self.ErgodicIDs = ergodic_ids or []
        self._isStarted = False
        self._CurDT = None
        self._DateTimes = None
        self._IDs = None
        self._CurInd = None
        self._DTNum = None
        self._CacheDTs = None
        self._CacheData = None
        self._CacheFactorNum = None
        self._CacheIDNum = None
        self._FactorReadNum = None
        self._IDReadNum = None
        self._Queue2SubProcess = None
        self._Queue2MainProcess = None
        self._TagName = None
        self._MMAPCacheData = None
        self._CacheDataProcess = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_CacheDataProcess" in state:
            state["_CacheDataProcess"] = None
        return state


class OperationMode:
    """
    运算模式参数对象

    管理因子运算模式下的多进程调度
    """

    def __init__(
        self,
        ft: Any = None,
        factor_names: Optional[List[str]] = None,
        subprocess_num: int = 0,
    ):
        self._FT = ft
        self._isStarted = False
        self._Factors = []
        self._FactorDict = {}
        self._FactorID = {}
        self._FactorStartDT = {}
        self._FactorPrepareIDs = {}
        self._iPID = "0"
        self._PIDs = []
        self._PID_IDs = {}
        self._PID_Lock = {}
        self._CacheDir = None
        self._RawDataDir = ""
        self._CacheDataDir = ""
        self._Event = {}
        self._DateTimes = []
        self._IDs = []
        self._DTRuler = []
        self.SubProcessNum = subprocess_num
        self.FactorNames = factor_names or []
        self.FileSuffix = get_shelve_file_suffix()
        if self.FileSuffix:
            self.FileSuffix = "." + self.FileSuffix

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._CacheDir is not None:
            state["_CacheDir"] = self._CacheDir.name
        return state


def prepare_mmap_factor_cache_data(ft: Any, mmap_cache: Any) -> int:
    """
    基于 mmap 的因子缓冲数据准备子进程
    """
    CacheData = {}
    CacheDTs = []
    CacheSize = int(ft.ErgodicMode.CacheSize * 2**20)

    if os.name == "nt":
        MMAPCacheData = mmap.mmap(-1, CacheSize, tagname=ft.ErgodicMode._TagName)
    else:
        MMAPCacheData = mmap_cache

    while True:
        Task = ft.ErgodicMode._Queue2SubProcess.get()
        if Task is None:
            break

        if Task[0] is None and Task[1] is None:
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            for i in range(int(DataLen / CacheSize) + 1):
                iStartInd = i * CacheSize
                iEndInd = min((i + 1) * CacheSize, DataLen)
                if iEndInd > iStartInd:
                    MMAPCacheData.seek(0)
                    MMAPCacheData.write(CacheDataByte[iStartInd:iEndInd])
                    ft.ErgodicMode._Queue2MainProcess.put(iEndInd - iStartInd)
                    ft.ErgodicMode._Queue2SubProcess.get()
            ft.ErgodicMode._Queue2MainProcess.put(0)
            del CacheDataByte
            gc.collect()
        elif Task[0] is None:
            NewFactors, PopFactors = Task[1]
            for iFactorName in PopFactors:
                CacheData.pop(iFactorName)
            if NewFactors:
                if CacheDTs:
                    CacheData.update(
                        dict(
                            ft.__QN_calc_data__(
                                raw_data=ft.__QN_prepare_raw_data__(
                                    factor_names=NewFactors,
                                    ids=ft.ErgodicMode._IDs,
                                    dts=CacheDTs,
                                ),
                                factor_names=NewFactors,
                                ids=ft.ErgodicMode._IDs,
                                dts=CacheDTs,
                            )
                        )
                    )
                else:
                    CacheData.update(
                        {
                            iFactorName: pd.DataFrame(index=CacheDTs, columns=ft.ErgodicMode._IDs)
                            for iFactorName in NewFactors
                        }
                    )
        else:
            CurInd = Task[0] + ft.ErgodicMode.ForwardPeriod + 1
            DTNum = len(ft.ErgodicMode._DateTimes)
            if CurInd < DTNum:
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ft.ErgodicMode._DateTimes[
                    max((0, CurInd - ft.ErgodicMode.BackwardPeriod)) : min(
                        (DTNum, CurInd + ft.ErgodicMode.ForwardPeriod + 1)
                    )
                ].tolist()
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    isDisjoint = OldCacheDTs.isdisjoint(CacheDTs)
                    CacheFactorNames = list(CacheData.keys())
                    if NewCacheDTs:
                        NewCacheData = ft.__QN_calc_data__(
                            raw_data=ft.__QN_prepare_raw_data__(
                                factor_names=CacheFactorNames,
                                ids=ft.ErgodicMode._IDs,
                                dts=NewCacheDTs,
                            ),
                            factor_names=CacheFactorNames,
                            ids=ft.ErgodicMode._IDs,
                            dts=NewCacheDTs,
                        )
                    else:
                        NewCacheData = pd.DataFrame(
                            index=CacheFactorNames, columns=CacheDTs
                        )
                    for iFactorName in CacheData:
                        if isDisjoint:
                            CacheData[iFactorName] = NewCacheData[iFactorName]
                        else:
                            CacheData[iFactorName] = CacheData[iFactorName].loc[CacheDTs, :]
                            CacheData[iFactorName].loc[NewCacheDTs, :] = NewCacheData[iFactorName]
                    NewCacheData = None
    return 0


def prepare_mmap_id_cache_data(ft: Any, mmap_cache: Any) -> int:
    """
    基于 mmap 的 ID 缓冲数据准备子进程
    """
    CacheData = {}
    CacheDTs = []
    CacheSize = int(ft.ErgodicMode.CacheSize * 2**20)

    if os.name == "nt":
        MMAPCacheData = mmap.mmap(-1, CacheSize, tagname=ft.ErgodicMode._TagName)
    else:
        MMAPCacheData = mmap_cache

    while True:
        Task = ft.ErgodicMode._Queue2SubProcess.get()
        if Task is None:
            break

        if Task[0] is None and Task[1] is None:
            CacheDataByte = pickle.dumps(CacheData)
            DataLen = len(CacheDataByte)
            for i in range(int(DataLen / CacheSize) + 1):
                iStartInd = i * CacheSize
                iEndInd = min((i + 1) * CacheSize, DataLen)
                if iEndInd > iStartInd:
                    MMAPCacheData.seek(0)
                    MMAPCacheData.write(CacheDataByte[iStartInd:iEndInd])
                    ft.ErgodicMode._Queue2MainProcess.put(iEndInd - iStartInd)
                    ft.ErgodicMode._Queue2SubProcess.get()
            ft.ErgodicMode._Queue2MainProcess.put(0)
            del CacheDataByte
            gc.collect()
        elif Task[0] is None:
            NewID, PopID = Task[1]
            if PopID:
                CacheData.pop(PopID)
            if NewID:
                if CacheDTs:
                    CacheData[NewID] = ft.__QN_calc_data__(
                        raw_data=ft.__QN_prepare_raw_data__(
                            factor_names=ft.FactorNames,
                            ids=[NewID],
                            dts=CacheDTs,
                        ),
                        factor_names=ft.FactorNames,
                        ids=[NewID],
                        dts=CacheDTs,
                    ).iloc[:, :, 0]
                else:
                    CacheData[NewID] = pd.DataFrame(index=CacheDTs, columns=ft.FactorNames)
        else:
            CurInd = Task[0] + ft.ErgodicMode.ForwardPeriod + 1
            DTNum = len(ft.ErgodicMode._DateTimes)
            if CurInd < DTNum:
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ft.ErgodicMode._DateTimes[
                    max((0, CurInd - ft.ErgodicMode.BackwardPeriod)) : min(
                        (DTNum, CurInd + ft.ErgodicMode.ForwardPeriod + 1)
                    )
                ].tolist()
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    isDisjoint = OldCacheDTs.isdisjoint(CacheDTs)
                    CacheIDs = list(CacheData.keys())
                    if NewCacheDTs:
                        NewCacheData = ft.__QN_calc_data__(
                            raw_data=ft.__QN_prepare_raw_data__(
                                factor_names=ft.FactorNames,
                                ids=CacheIDs,
                                dts=NewCacheDTs,
                            ),
                            factor_names=ft.FactorNames,
                            ids=CacheIDs,
                            dts=NewCacheDTs,
                        )
                    else:
                        NewCacheData = pd.DataFrame(
                            index=ft.FactorNames, columns=CacheDTs
                        )
                    for iID in CacheData:
                        if isDisjoint:
                            CacheData[iID] = NewCacheData[iID]
                        else:
                            CacheData[iID] = CacheData[iID].loc[CacheDTs, :]
                            CacheData[iID].loc[NewCacheDTs, :] = NewCacheData[iID]
                    NewCacheData = None
    return 0


def save_raw_data(
    raw_data: Any,
    factor_names: List[str],
    raw_data_dir: str,
    pid_ids: Dict[str, List[str]],
    file_name: str,
    pid_lock: Dict[str, Lock],
) -> int:
    """
    保存原始数据到文件
    """
    if raw_data is None:
        return 0
    if isinstance(raw_data, pd.DataFrame) and ("ID" in raw_data):
        raw_data = raw_data.set_index(["ID"])
        CommonCols = raw_data.columns.difference(factor_names).tolist()
        AllIDs = set(raw_data.index)
        for iPID, iIDs in pid_ids.items():
            with shelve.open(raw_data_dir + os.sep + iPID + os.sep + file_name) as iFile:
                iInterIDs = sorted(AllIDs.intersection(iIDs))
                iData = raw_data.loc[iInterIDs]
                if factor_names:
                    for jFactorName in factor_names:
                        iFile[jFactorName] = iData[CommonCols + [jFactorName]].reset_index()
                else:
                    iFile["RawData"] = iData[CommonCols].reset_index()
                iFile["_QN_IDs"] = iIDs
    else:
        for iPID, iIDs in pid_ids.items():
            with shelve.open(raw_data_dir + os.sep + iPID + os.sep + file_name) as iFile:
                iFile["RawData"] = raw_data
                iFile["_QN_IDs"] = iIDs
    return 0


__all__ = [
    "ErgodicMode",
    "OperationMode",
    "prepare_mmap_factor_cache_data",
    "prepare_mmap_id_cache_data",
    "save_raw_data",
]
