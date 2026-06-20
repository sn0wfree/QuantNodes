# coding=utf-8
"""因子表

包含 FactorTable（因子表接口）和 CustomFT（自定义因子表）
以及相关的遍历模式、运算模式。
v2.0: 移除 traits 和 multiprocessing，使用 dataclass + concurrent.futures
"""
import datetime as dt
import gc
import mmap
import os
import pickle
import shelve
import tempfile
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Queue, Process, Lock
from os import cpu_count
from typing import Dict, List

import numpy as np
import pandas as pd
from progressbar import ProgressBar

from QuantNodes.factor_node.quant_nodes_object import QuantNodesObject
from QuantNodes.core.base import FactorError
from QuantNodes.core.tools import (
    compile_id_filter_str,
    gen_available_name,
    partition_list_moving_sampling,
    start_multi_process,
)


class ErgodicModeType(Enum):
    """遍历模式"""
    FACTOR = "因子"
    ID = "ID"


@dataclass
class _ErgodicMode(QuantNodesObject):
    """遍历模式"""
    forward_period: int = 600
    backward_period: int = 1
    cache_mode: ErgodicModeType = ErgodicModeType.FACTOR
    max_factor_cache_num: int = 60
    max_id_cache_num: int = 10000
    cache_size: int = 300
    ergodic_dts: List = field(default_factory=list)
    ergodic_ids: List = field(default_factory=list)

    def __init__(self, sys_args: Dict = None, **kwargs):
        super().__init__(sys_args=sys_args, **kwargs)
        self._isStarted = False
        self._CurDT = None


def _prepareMMAPFactorCacheData(ft, mmap_cache):
    """基于 mmap 的因子缓冲数据准备子进程"""
    CacheData, CacheDTs, MMAPCacheData, DTNum = {}, [], mmap_cache, len(ft.ErgodicMode._DateTimes)
    CacheSize = int(ft.ErgodicMode.CacheSize * 2 ** 20)
    if os.name == 'nt':
        MMAPCacheData = mmap.mmap(-1, CacheSize, tagname=ft.ErgodicMode._TagName)
    while True:
        Task = ft.ErgodicMode._Queue2SubProcess.get()
        if Task is None:
            break
        if (Task[0] is None) and (Task[1] is None):
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
                    CacheData.update(dict(ft.__QN_calc_data__(
                        raw_data=ft.__QN_prepare_raw_data__(factor_names=NewFactors, ids=ft.ErgodicMode._IDs,
                                                          dts=CacheDTs), factor_names=NewFactors,
                        ids=ft.ErgodicMode._IDs, dts=CacheDTs)))
                else:
                    CacheData.update(
                        {iFactorName: pd.DataFrame(index=CacheDTs, columns=ft.ErgodicMode._IDs) for iFactorName in
                         NewFactors})
        else:
            CurInd = Task[0] + ft.ErgodicMode.ForwardPeriod + 1
            if CurInd < DTNum:
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ft.ErgodicMode._DateTimes[max((0, CurInd - ft.ErgodicMode.BackwardPeriod)):min(
                    (DTNum, CurInd + ft.ErgodicMode.ForwardPeriod + 1))].tolist()
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    isDisjoint = OldCacheDTs.isdisjoint(CacheDTs)
                    CacheFactorNames = list(CacheData.keys())
                    if NewCacheDTs:
                        NewCacheData = ft.__QN_calc_data__(
                            raw_data=ft.__QN_prepare_raw_data__(factor_names=CacheFactorNames, ids=ft.ErgodicMode._IDs,
                                                              dts=NewCacheDTs), factor_names=CacheFactorNames,
                            ids=ft.ErgodicMode._IDs, dts=NewCacheDTs)
                    else:
                        NewCacheData = {name: pd.DataFrame(index=NewCacheDTs, columns=ft.ErgodicMode._IDs)
                                        for name in CacheFactorNames}
                    for iFactorName in CacheData:
                        if isDisjoint:
                            CacheData[iFactorName] = NewCacheData[iFactorName]
                        else:
                            CacheData[iFactorName] = CacheData[iFactorName].loc[CacheDTs, :]
                            CacheData[iFactorName].loc[NewCacheDTs, :] = NewCacheData[iFactorName]
                    NewCacheData = None
    return 0


def _prepareMMAPIDCacheData(ft, mmap_cache):
    """基于 mmap 的 ID 缓冲数据准备子进程"""
    CacheData, CacheDTs, MMAPCacheData, DTNum = {}, [], mmap_cache, len(ft.ErgodicMode._DateTimes)
    CacheSize = int(ft.ErgodicMode.CacheSize * 2 ** 20)
    if os.name == 'nt':
        MMAPCacheData = mmap.mmap(-1, CacheSize, tagname=ft.ErgodicMode._TagName)
    while True:
        Task = ft.ErgodicMode._Queue2SubProcess.get()
        if Task is None:
            break
        if (Task[0] is None) and (Task[1] is None):
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
                        raw_data=ft.__QN_prepare_raw_data__(factor_names=ft.FactorNames, ids=[NewID], dts=CacheDTs),
                        factor_names=ft.FactorNames, ids=[NewID], dts=CacheDTs).iloc[:, :, 0]
                else:
                    CacheData[NewID] = pd.DataFrame(index=CacheDTs, columns=ft.FactorNames)
        else:
            CurInd = Task[0] + ft.ErgodicMode.ForwardPeriod + 1
            if CurInd < DTNum:
                OldCacheDTs = set(CacheDTs)
                CacheDTs = ft.ErgodicMode._DateTimes[max((0, CurInd - ft.ErgodicMode.BackwardPeriod)):min(
                    (DTNum, CurInd + ft.ErgodicMode.ForwardPeriod + 1))].tolist()
                NewCacheDTs = sorted(set(CacheDTs).difference(OldCacheDTs))
                if CacheData:
                    isDisjoint = OldCacheDTs.isdisjoint(CacheDTs)
                    CacheIDs = list(CacheData.keys())
                    if NewCacheDTs:
                        NewCacheData = ft.__QN_calc_data__(
                            raw_data=ft.__QN_prepare_raw_data__(factor_names=ft.FactorNames, ids=CacheIDs,
                                                              dts=NewCacheDTs), factor_names=ft.FactorNames,
                            ids=CacheIDs, dts=NewCacheDTs)
                    else:
                        NewCacheData = {name: pd.DataFrame(index=NewCacheDTs, columns=CacheIDs)
                                        for name in ft.FactorNames}
                    for iID in CacheData:
                        if isDisjoint:
                            CacheData[iID] = NewCacheData.loc[:, :, iID]
                        else:
                            CacheData[iID] = CacheData[iID].loc[CacheDTs, :]
                            CacheData[iID].loc[NewCacheDTs, :] = NewCacheData.loc[:, :, iID]
                    NewCacheData = None
    return 0


@dataclass
class _OperationMode(QuantNodesObject):
    """运算模式"""
    date_times: List = field(default_factory=list)
    ids: List = field(default_factory=list)
    factor_names: List = field(default_factory=list)
    sub_process_num: int = 0
    dt_ruler: List = field(default_factory=list)

    def __init__(self, ft, sys_args: Dict = None, config_file: str = None, **kwargs):
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
        from QuantNodes.core.tools import get_shelve_file_suffix
        self._FileSuffix = get_shelve_file_suffix()
        if self._FileSuffix:
            self._FileSuffix = "." + self._FileSuffix
        super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._CacheDir is not None:
            state["_CacheDir"] = self._CacheDir.name
        return state


def _prepareRawData(args):
    """因子表准备原始数据子进程"""
    nGroup = len(args['GroupInfo'])
    if "Sub2MainQueue" not in args:
        with ProgressBar(max_value=nGroup) as ProgBar:
            for i in range(nGroup):
                iFT, iFactorNames, iRawFactorNames, iDTs, iArgs = args['GroupInfo'][i]
                iPrepareIDs = args["PrepareIDs"][i]
                if iPrepareIDs is None:
                    iPrepareIDs = args["FT"].OperationMode.IDs
                iPID_PrepareIDs = args["PID_PrepareIDs"][i]
                if iPID_PrepareIDs is None:
                    iPID_PrepareIDs = args["FT"].OperationMode._PID_IDs
                iRawData = iFT.__QN_prepare_raw_data__(iRawFactorNames, iPrepareIDs, iDTs, iArgs)
                iFT.__QN_save_raw_data__(iRawData, iRawFactorNames, args["FT"].OperationMode._RawDataDir, iPID_PrepareIDs,
                                       args["RawDataFileNames"][i], args["FT"].OperationMode._PID_Lock)
                ProgBar.update(i + 1)
    else:
        for i in range(nGroup):
            iFT, iFactorNames, iRawFactorNames, iDTs, iArgs = args['GroupInfo'][i]
            iPrepareIDs = args["PrepareIDs"][i]
            if iPrepareIDs is None:
                iPrepareIDs = args["FT"].OperationMode.IDs
            iPID_PrepareIDs = args["PID_PrepareIDs"][i]
            if iPID_PrepareIDs is None:
                iPID_PrepareIDs = args["FT"].OperationMode._PID_IDs
            iRawData = iFT.__QN_prepare_raw_data__(iRawFactorNames, iPrepareIDs, iDTs, iArgs)
            iFT.__QN_save_raw_data__(iRawData, iRawFactorNames, args["FT"].OperationMode._RawDataDir, iPID_PrepareIDs,
                                   args["RawDataFileNames"][i], args["FT"].OperationMode._PID_Lock)
            args['Sub2MainQueue'].put((args["PID"], 1, None))
    return 0


def _build_task_dispatch(FT, TDB, TableName, SpecificTarget):
    """构建任务分发字典"""
    if SpecificTarget:
        TaskDispatched = OrderedDict()
        for iFactorName in FT.OperationMode.FactorNames:
            iDB, iTableName, iTargetFactorName = SpecificTarget.get(iFactorName, (None, None, None))
            if iDB is None:
                iDB = TDB
            if iTableName is None:
                iTableName = TableName
            if iTargetFactorName is None:
                iTargetFactorName = iFactorName
            iDBTable = (id(iDB), iTableName)
            if iDBTable in TaskDispatched:
                TaskDispatched[iDBTable][1].append(FT.OperationMode._FactorDict[iFactorName])
                TaskDispatched[iDBTable][2].append(iTargetFactorName)
            else:
                TaskDispatched[iDBTable] = (iDB, [FT.OperationMode._FactorDict[iFactorName]], [iTargetFactorName])
    else:
        TaskDispatched = {(id(TDB), TableName): (TDB, FT.OperationMode._Factors, list(FT.OperationMode.FactorNames))}
    return TaskDispatched


def _write_factor_data_batch(iDB, iTableName, iFactors, iTargetFactorNames, FT, PID, ProgBar, TaskCount, if_exists):
    """单进程写入因子数据 (writeFactorData 路径)"""
    for j, jFactor in enumerate(iFactors):
        jData = jFactor._QN_get_data(dts=FT.OperationMode.DateTimes, pids=[PID])
        if FT.OperationMode._FactorPrepareIDs[jFactor.Name] is not None:
            jData = jData.loc[:, FT.OperationMode.IDs]
        iDB.writeFactorData(jData, iTableName, iTargetFactorNames[j], if_exists=if_exists,
                            data_type=jFactor.getMetaData(key="DataType"))
        jData = None
        TaskCount += 1
        ProgBar.update(TaskCount)
    return TaskCount


def _write_panel_batch(iDB, iTableName, iFactors, iTargetFactorNames, FT, PID, nDT, ProgBar, TaskCount, if_exists):
    """单进程写入面板数据 (writeData 路径)"""
    iFactoNum = len(iFactors)
    iDTLen = int(np.ceil(nDT / iFactoNum))
    iDataTypes = {iTargetFactorNames[j]: jFactor.getMetaData(key="DataType") for j, jFactor in enumerate(iFactors)}
    for j in range(iFactoNum):
        jDTs = list(FT.OperationMode.DateTimes[j * iDTLen:(j + 1) * iDTLen])
        if jDTs:
            jData = {}
            for k, kFactor in enumerate(iFactors):
                ijkData = kFactor._QN_get_data(dts=jDTs, pids=[PID])
                if FT.OperationMode._FactorPrepareIDs[kFactor.Name] is not None:
                    ijkData = ijkData.loc[:, FT.OperationMode.IDs]
                jData[iTargetFactorNames[k]] = ijkData
                if j == 0:
                    TaskCount += 0.5
                    ProgBar.update(TaskCount)
            jData = {name: jData[name] for name in iTargetFactorNames if name in jData}
            iDB.writeData(jData, iTableName, if_exists=if_exists, data_type=iDataTypes)
            jData = None
        TaskCount += 0.5
        ProgBar.update(TaskCount)
    return TaskCount


def _write_factor_data_single(iDB, iTableName, iFactors, iTargetFactorNames, FT, args):
    """多进程写入因子数据 (writeFactorData 路径)"""
    for j, jFactor in enumerate(iFactors):
        if FT.OperationMode._FactorPrepareIDs[jFactor.Name] is not None:
            jData = jFactor._QN_get_data(dts=FT.OperationMode.DateTimes, pids=None)
            jData = jData.loc[:, FT.OperationMode._PID_IDs[args["PID"]]]
        else:
            jData = jFactor._QN_get_data(dts=FT.OperationMode.DateTimes, pids=[args["PID"]])
        iDB.writeFactorData(jData, iTableName, iTargetFactorNames[j], if_exists=args["if_exists"],
                            data_type=jFactor.getMetaData(key="DataType"))
        jData = None
        args["Sub2MainQueue"].put((args["PID"], 1, None))


def _write_panel_single(iDB, iTableName, iFactors, iTargetFactorNames, FT, args, nDT):
    """多进程写入面板数据 (writeData 路径)"""
    iFactoNum = len(iFactors)
    iDTLen = int(np.ceil(nDT / iFactoNum))
    iDataTypes = {iTargetFactorNames[j]: jFactor.getMetaData(key="DataType") for j, jFactor in enumerate(iFactors)}
    for j in range(iFactoNum):
        jDTs = list(FT.OperationMode.DateTimes[j * iDTLen:(j + 1) * iDTLen])
        if jDTs:
            jData = {}
            for k, kFactor in enumerate(iFactors):
                ijkData = kFactor._QN_get_data(dts=jDTs, pids=[args["PID"]])
                if FT.OperationMode._FactorPrepareIDs[kFactor.Name] is not None:
                    ijkData = ijkData.loc[:, FT.OperationMode.IDs]
                jData[iTargetFactorNames[k]] = ijkData
                if j == 0:
                    args["Sub2MainQueue"].put((args["PID"], 0.5, None))
            jData = {name: jData[name] for name in iTargetFactorNames if name in jData}
            iDB.writeData(jData, iTableName, if_exists=args["if_exists"], data_type=iDataTypes)
            jData = None
        args["Sub2MainQueue"].put((args["PID"], 0.5, None))


def _calculate_single_process(FT, TaskDispatched, TableName, args, nDT):
    """单进程执行因子计算"""
    nTask = len(FT.OperationMode.FactorNames)
    TaskCount = 0
    with ProgressBar(max_value=nTask) as ProgBar:
        for i, iTask in enumerate(TaskDispatched):
            iDB, iFactors, iTargetFactorNames = TaskDispatched[iTask]
            iTableName = iTask[1]
            if hasattr(iDB, "writeFactorData"):
                TaskCount = _write_factor_data_batch(iDB, iTableName, iFactors, iTargetFactorNames,
                                                     FT, args["PID"], ProgBar, TaskCount, args["if_exists"])
            else:
                TaskCount = _write_panel_batch(iDB, iTableName, iFactors, iTargetFactorNames,
                                               FT, args["PID"], nDT, ProgBar, TaskCount, args["if_exists"])


def _calculate_multi_process(FT, TaskDispatched, TableName, args, nDT):
    """多进程执行因子计算"""
    for i, iTask in enumerate(TaskDispatched):
        iDB, iFactors, iTargetFactorNames = TaskDispatched[iTask]
        iTableName = iTask[1]
        if hasattr(iDB, "writeFactorData"):
            _write_factor_data_single(iDB, iTableName, iFactors, iTargetFactorNames, FT, args)
        else:
            _write_panel_single(iDB, iTableName, iFactors, iTargetFactorNames, FT, args, nDT)


def _calculate(args):
    """因子表运算子进程"""
    FT = args["FT"]
    FT.OperationMode._iPID = args["PID"]
    TDB, TableName, SpecificTarget = args["FactorDB"], args["TableName"], args["specific_target"]
    TaskDispatched = _build_task_dispatch(FT, TDB, TableName, SpecificTarget)
    nDT = len(FT.OperationMode.DateTimes)
    if FT.OperationMode.SubProcessNum == 0:
        _calculate_single_process(FT, TaskDispatched, TableName, args, nDT)
    else:
        _calculate_multi_process(FT, TaskDispatched, TableName, args, nDT)
    return 0


class FactorTable(QuantNodesObject):
    """因子表（接口类）

    因子表可看做一个独立的数据集或命名空间，
    可看做 Panel(items=[因子], major_axis=[时间点], minor_axis=[ID])。
    """
    ergodic_mode: _ErgodicMode = None
    operation_mode: _OperationMode = None

    def __init__(self, name, fdb=None, sys_args={}, config_file=None, **kwargs):
        self._Name = name
        self._FactorDB = fdb
        self.ergodic_mode = _ErgodicMode()
        self.operation_mode = _OperationMode(ft=self)
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)

    @property
    def Name(self):
        return self._Name

    @property
    def FactorDB(self):
        return self._FactorDB

    def getMetaData(self, key=None, args={}):
        if key is None:
            return {}
        return None

    @property
    def FactorNames(self):
        return []

    def getFactor(self, ifactor_name, args={}, new_name=None):
        from QuantNodes.factor_node.factor import Factor
        iFactor = Factor(name=ifactor_name, ft=self)
        iFactor.name = new_name or ifactor_name
        return iFactor

    def getFactorMetaData(self, factor_names, key=None, args={}):
        if key is None:
            return pd.DataFrame(index=factor_names, dtype=np.dtype("O"))
        else:
            return pd.Series([None] * len(factor_names), index=factor_names, dtype=np.dtype("O"))

    def getID(self, ifactor_name=None, idt=None, args={}):
        return []

    def getIDMask(self, idt, ids=None, id_filter_str=None, args={}):
        if ids is None:
            ids = self.getID(idt=idt, args=args)
        if not id_filter_str:
            return pd.Series(True, index=ids)
        CompiledIDFilterStr, IDFilterFactors = compile_id_filter_str(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None:
            raise FactorError("过滤条件字符串有误!")
        return eval(CompiledIDFilterStr, {}, {
            "temp": self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]
        })

    def getFilteredID(self, idt, ids=None, id_filter_str=None, args={}):
        if not id_filter_str:
            return self.getID(idt=idt, args=args)
        if ids is None:
            ids = self.getID(idt=idt, args=args)
        CompiledIDFilterStr, IDFilterFactors = compile_id_filter_str(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None:
            raise FactorError("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]  # noqa: F841 (used in eval below)
        return eval("temp[" + CompiledIDFilterStr + "].index.tolist()")

    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        return []

    def __QN_prepare_raw_data__(self, factor_names, ids, dts, args={}):
        return None

    def __QN_calc_data__(self, raw_data, factor_names, ids, dts, args={}):
        return None

    def readData(self, factor_names, ids, dts, args={}):
        if self.ErgodicMode._isStarted:
            return self._readData_ErgodicMode(factor_names=factor_names, ids=ids, dts=dts, args=args)
        return self.__QN_calc_data__(
            raw_data=self.__QN_prepare_raw_data__(factor_names=factor_names, ids=ids, dts=dts, args=args),
            factor_names=factor_names, ids=ids, dts=dts, args=args)

    def _readData_FactorCacheMode(self, factor_names, ids, dts, args={}):
        self.ErgodicMode._FactorReadNum[factor_names] += 1
        if (self.ErgodicMode.MaxFactorCacheNum <= 0) or (not self.ErgodicMode._CacheDTs) or (
                dts[0] < self.ErgodicMode._CacheDTs[0]) or (dts[-1] > self.ErgodicMode._CacheDTs[-1]):
            return self.__QN_calc_data__(
                raw_data=self.__QN_prepare_raw_data__(factor_names=factor_names, ids=ids, dts=dts, args=args),
                factor_names=factor_names, ids=ids, dts=dts, args=args)
        Data = {}
        DataFactorNames = []
        CacheFactorNames = set()
        PopFactorNames = []
        for iFactorName in factor_names:
            iFactorData = self.ErgodicMode._CacheData.get(iFactorName)
            if iFactorData is None:
                if self.ErgodicMode._CacheFactorNum < self.ErgodicMode.MaxFactorCacheNum:
                    self.ErgodicMode._CacheFactorNum += 1
                    CacheFactorNames.add(iFactorName)
                else:
                    CacheFactorReadNum = self.ErgodicMode._FactorReadNum[self.ErgodicMode._CacheData.keys()]
                    MinReadNumInd = CacheFactorReadNum.argmin()
                    if CacheFactorReadNum.loc[MinReadNumInd] < self.ErgodicMode._FactorReadNum[iFactorName]:
                        CacheFactorNames.add(iFactorName)
                        PopFactor = MinReadNumInd
                        self.ErgodicMode._CacheData.pop(PopFactor)
                        PopFactorNames.append(PopFactor)
                    else:
                        DataFactorNames.append(iFactorName)
            else:
                Data[iFactorName] = iFactorData
        CacheFactorNames = list(CacheFactorNames)
        if CacheFactorNames:
            iData = dict(self.__QN_calc_data__(
                raw_data=self.__QN_prepare_raw_data__(factor_names=CacheFactorNames, ids=self.ErgodicMode._IDs,
                                                    dts=self.ErgodicMode._CacheDTs, args=args),
                factor_names=CacheFactorNames, ids=self.ErgodicMode._IDs, dts=self.ErgodicMode._CacheDTs, args=args))
            Data.update(iData)
            self.ErgodicMode._CacheData.update(iData)
        self.ErgodicMode._Queue2SubProcess.put((None, (CacheFactorNames, PopFactorNames)))
        if len(Data) > 0:
            Data = {name: df.loc[dts, ids] for name, df in Data.items() if isinstance(df, pd.DataFrame)}
        if not DataFactorNames:
            return {name: Data[name] for name in factor_names if name in Data}
        return self.__QN_calc_data__(
            raw_data=self.__QN_prepare_raw_data__(factor_names=DataFactorNames, ids=ids, dts=dts, args=args),
            factor_names=DataFactorNames, ids=ids, dts=dts, args=args)

    def _readIDData(self, iid, factor_names, dts, args={}):
        self.ErgodicMode._IDReadNum[iid] = self.ErgodicMode._IDReadNum.get(iid, 0) + 1
        if (self.ErgodicMode.MaxIDCacheNum <= 0) or (not self.ErgodicMode._CacheDTs) or (
                dts[0] < self.ErgodicMode._CacheDTs[0]) or (dts[-1] > self.ErgodicMode._CacheDTs[-1]):
            return self.__QN_calc_data__(
                raw_data=self.__QN_prepare_raw_data__(factor_names=factor_names, ids=[iid], dts=dts, args=args),
                factor_names=factor_names, ids=[iid], dts=dts, args=args).iloc[:, :, 0]
        IDData = self.ErgodicMode._CacheData.get(iid)
        if IDData is None:
            if self.ErgodicMode._CacheIDNum < self.ErgodicMode.MaxIDCacheNum:
                self.ErgodicMode._CacheIDNum += 1
                IDData = self.__QN_calc_data__(
                    raw_data=self.__QN_prepare_raw_data__(factor_names=self.FactorNames, ids=[iid],
                                                        dts=self.ErgodicMode._CacheDTs, args=args),
                    factor_names=self.FactorNames, ids=[iid], dts=self.ErgodicMode._CacheDTs, args=args).iloc[:, :, 0]
                self.ErgodicMode._CacheData[iid] = IDData
                self.ErgodicMode._Queue2SubProcess.put((None, (iid, None)))
            else:
                CacheIDReadNum = self.ErgodicMode._IDReadNum[self.ErgodicMode._CacheData.keys()]
                MinReadNumInd = CacheIDReadNum.argmin()
                if CacheIDReadNum.loc[MinReadNumInd] < self.ErgodicMode._IDReadNum[iid]:
                    IDData = self.__QN_calc_data__(
                        raw_data=self.__QN_prepare_raw_data__(factor_names=self.FactorNames, ids=[iid],
                                                            dts=self.ErgodicMode._CacheDTs, args=args),
                        factor_names=self.FactorNames, ids=[iid], dts=self.ErgodicMode._CacheDTs, args=args).iloc[:, :,
                             0]
                    PopID = MinReadNumInd
                    self.ErgodicMode._CacheData.pop(PopID)
                    self.ErgodicMode._CacheData[iid] = IDData
                    self.ErgodicMode._Queue2SubProcess.put((None, (iid, PopID)))
                else:
                    return self.__QN_calc_data__(
                        raw_data=self.__QN_prepare_raw_data__(factor_names=factor_names, ids=[iid], dts=dts, args=args),
                        factor_names=factor_names, ids=[iid], dts=dts, args=args).iloc[:, :, 0]
        return IDData.loc[dts, factor_names]

    def _readData_ErgodicMode(self, factor_names, ids, dts, args={}):
        if self.ErgodicMode.CacheMode == "因子":
            return self._readData_FactorCacheMode(factor_names=factor_names, ids=ids, dts=dts, args=args)
        # pd.Panel removed - return dict of DataFrames keyed by ID
        return {iID: self._readIDData(iID, factor_names=factor_names, dts=dts, args=args) for iID in ids}

    def start(self, dts, **kwargs):
        if self.ErgodicMode._isStarted:
            return 0
        self.ErgodicMode._DateTimes = np.array(
            (self.getDateTime() if not self.ErgodicMode.ErgodicDTs else self.ErgodicMode.ErgodicDTs), dtype="O")
        if self.ErgodicMode._DateTimes.shape[0] == 0:
            raise FactorError("因子表: '%s' 的默认时间序列为空, 请设置参数 '遍历模式-遍历时点' !" % self.Name)
        self.ErgodicMode._IDs = (self.getID() if not self.ErgodicMode.ErgodicIDs else list(self.ErgodicMode.ErgodicIDs))
        if not self.ErgodicMode._IDs:
            raise FactorError("因子表: '%s' 的默认 ID 序列为空, 请设置参数 '遍历模式-遍历ID' !" % self.Name)
        self.ErgodicMode._CurInd = -1
        self.ErgodicMode._DTNum = self.ErgodicMode._DateTimes.shape[0]
        self.ErgodicMode._CacheDTs = []
        self.ErgodicMode._CacheData = {}
        self.ErgodicMode._CacheFactorNum = 0
        self.ErgodicMode._CacheIDNum = 0
        self.ErgodicMode._FactorReadNum = pd.Series(0, index=self.FactorNames)
        self.ErgodicMode._IDReadNum = pd.Series()
        self.ErgodicMode._Queue2SubProcess = Queue()
        self.ErgodicMode._Queue2MainProcess = Queue()
        if self.ErgodicMode.CacheSize > 0:
            if os.name == "nt":
                self.ErgodicMode._TagName = str(uuid.uuid1())
                self._MMAPCacheData = None
            else:
                self.ErgodicMode._TagName = None
                self._MMAPCacheData = mmap.mmap(-1, int(self.ErgodicMode.CacheSize * 2 ** 20))
            if self.ErgodicMode.CacheMode == "因子":
                self.ErgodicMode._CacheDataProcess = Process(target=_prepareMMAPFactorCacheData,
                                                             args=(self, self._MMAPCacheData), daemon=True)
            else:
                self.ErgodicMode._CacheDataProcess = Process(target=_prepareMMAPIDCacheData,
                                                             args=(self, self._MMAPCacheData), daemon=True)
            self.ErgodicMode._CacheDataProcess.start()
            if os.name == "nt":
                self._MMAPCacheData = mmap.mmap(-1, int(self.ErgodicMode.CacheSize * 2 ** 20),
                                                tagname=self.ErgodicMode._TagName)
        self.ErgodicMode._isStarted = True
        return 0

    def move(self, idt, **kwargs):
        if idt == self.ErgodicMode._CurDT:
            return 0
        self.ErgodicMode._CurDT = idt
        PreInd = self.ErgodicMode._CurInd
        self.ErgodicMode._CurInd = PreInd + np.sum(self.ErgodicMode._DateTimes[PreInd + 1:] <= idt)
        if (self.ErgodicMode.CacheSize > 0) and (self.ErgodicMode._CurInd > -1) and (
                (not self.ErgodicMode._CacheDTs) or (
                self.ErgodicMode._DateTimes[self.ErgodicMode._CurInd] > self.ErgodicMode._CacheDTs[-1])):
            self.ErgodicMode._Queue2SubProcess.put((None, None))
            DataLen = self.ErgodicMode._Queue2MainProcess.get()
            CacheData = b""
            while DataLen > 0:
                self._MMAPCacheData.seek(0)
                CacheData += self._MMAPCacheData.read(DataLen)
                self.ErgodicMode._Queue2SubProcess.put(DataLen)
                DataLen = self.ErgodicMode._Queue2MainProcess.get()
            self.ErgodicMode._CacheData = pickle.loads(CacheData)
            if self.ErgodicMode._CurInd == PreInd + 1:
                self.ErgodicMode._Queue2SubProcess.put((self.ErgodicMode._CurInd, None))
                self.ErgodicMode._CacheDTs = self.ErgodicMode._DateTimes[
                                             max((0, self.ErgodicMode._CurInd - self.ErgodicMode.BackwardPeriod)):min((
                                                 self.ErgodicMode._DTNum,
                                                 self.ErgodicMode._CurInd + self.ErgodicMode.ForwardPeriod + 1))].tolist()
            else:
                LastCacheInd = (self.ErgodicMode._DateTimes.searchsorted(
                    self.ErgodicMode._CacheDTs[-1]) if self.ErgodicMode._CacheDTs else self.ErgodicMode._CurInd - 1)
                self.ErgodicMode._Queue2SubProcess.put((LastCacheInd + 1, None))
                self.ErgodicMode._CacheDTs = self.ErgodicMode._DateTimes[
                                             max((0, LastCacheInd + 1 - self.ErgodicMode.BackwardPeriod)):min((
                                                 self.ErgodicMode._DTNum,
                                                 LastCacheInd + 1 + self.ErgodicMode.ForwardPeriod + 1))].tolist()
        return 0

    def __QN_on_backtest_move_event__(self, event):
        return self.move(**event.Data)

    def end(self):
        if not self.ErgodicMode._isStarted:
            return 0
        self.ErgodicMode._CacheData, self.ErgodicMode._FactorReadNum, self.ErgodicMode._IDReadNum = None, None, None
        if self.ErgodicMode.CacheSize > 0:
            self.ErgodicMode._Queue2SubProcess.put(None)
        self.ErgodicMode._Queue2SubProcess = self.ErgodicMode._Queue2MainProcess = self.ErgodicMode._CacheDataProcess = None
        self.ErgodicMode._isStarted = False
        self.ErgodicMode._CurDT = None
        self._MMAPCacheData = None
        return 0

    def __QN_on_backtest_end_event__(self, event):
        return self.end()

    def __QN_gen_group_info__(self, factors, operation_mode):
        StartDT = dt.datetime.now()
        FactorNames, RawFactorNames = [], set()
        for iFactor in factors:
            FactorNames.append(iFactor.Name)
            RawFactorNames.add(iFactor._NameInFT)
            StartDT = min((StartDT, operation_mode._FactorStartDT[iFactor.Name]))
        EndDT = operation_mode.DateTimes[-1]
        StartInd, EndInd = operation_mode.DTRuler.index(StartDT), operation_mode.DTRuler.index(EndDT)
        return [(self, FactorNames, list(RawFactorNames), operation_mode.DTRuler[StartInd:EndInd + 1], {})]

    def __QN_save_raw_data__(self, raw_data, factor_names, raw_data_dir, pid_ids, file_name, pid_lock, **kwargs):
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

    def _genFactorDict(self, factors, factor_dict={}):
        for iFactor in factors:
            iFactor._OperationMode = self.OperationMode
            if (not isinstance(iFactor.Name, str)) or (iFactor.Name == "") or (
                    iFactor is not factor_dict.get(iFactor.Name, iFactor)):
                iFactor.Name = gen_available_name("TempFactor", factor_dict)
            factor_dict[iFactor.Name] = iFactor
            self.OperationMode._FactorID[iFactor.Name] = len(factor_dict)
            factor_dict.update(self._genFactorDict(iFactor.Descriptors, factor_dict))
        return factor_dict

    def _initOperation(self):
        if not self.OperationMode.DateTimes:
            raise FactorError("运算时点序列不能为空!")
        if not self.OperationMode.IDs:
            raise FactorError("运算 ID 序列不能为空!")
        try:
            DTs = pd.Series(np.arange(0, len(self.OperationMode.DTRuler)), index=list(self.OperationMode.DTRuler)).loc[
                list(self.OperationMode.DateTimes)]
        except (KeyError, IndexError):
            raise FactorError("运算时点序列超出了时点标尺!")
        if pd.isnull(DTs).sum() > 0:
            raise FactorError("运算时点序列超出了时点标尺!")
        elif (DTs.diff().iloc[1:] != 1).sum() > 0:
            raise FactorError("运算时点序列的频率与时点标尺不一致!")
        if not self.OperationMode.FactorNames:
            self.OperationMode.FactorNames = self.FactorNames
        self.OperationMode._Factors = []
        self.OperationMode._FactorDict = {}
        self.OperationMode._FactorID = {}
        for i, iFactorName in enumerate(self.OperationMode.FactorNames):
            iFactor = self.getFactor(iFactorName)
            iFactor._OperationMode = self.OperationMode
            self.OperationMode._Factors.append(iFactor)
            self.OperationMode._FactorDict[iFactorName] = iFactor
            self.OperationMode._FactorID[iFactorName] = i
        self.OperationMode._FactorDict = self._genFactorDict(self.OperationMode._Factors,
                                                             self.OperationMode._FactorDict)
        self.OperationMode._Event = {}
        self.OperationMode._CacheDir = tempfile.TemporaryDirectory()
        self.OperationMode._RawDataDir = self.OperationMode._CacheDir.name + os.sep + "RawData"
        self.OperationMode._CacheDataDir = self.OperationMode._CacheDir.name + os.sep + "CacheData"
        os.mkdir(self.OperationMode._RawDataDir)
        os.mkdir(self.OperationMode._CacheDataDir)
        if self.OperationMode.SubProcessNum == 0:
            self.OperationMode._PIDs = ["0"]
            self.OperationMode._PID_IDs = {"0": list(self.OperationMode.IDs)}
            os.mkdir(self.OperationMode._RawDataDir + os.sep + "0")
            os.mkdir(self.OperationMode._CacheDataDir + os.sep + "0")
            self.OperationMode._PID_Lock = {"0": Lock()}
        else:
            self.OperationMode._PIDs = []
            self.OperationMode._PID_IDs = {}
            nPrcs = min((self.OperationMode.SubProcessNum, len(self.OperationMode.IDs)))
            SubIDs = partition_list_moving_sampling(list(self.OperationMode.IDs), nPrcs)
            self.OperationMode._PID_Lock = {}
            for i in range(nPrcs):
                iPID = "0-" + str(i)
                self.OperationMode._PIDs.append(iPID)
                self.OperationMode._PID_IDs[iPID] = SubIDs[i]
                os.mkdir(self.OperationMode._RawDataDir + os.sep + iPID)
                os.mkdir(self.OperationMode._CacheDataDir + os.sep + iPID)
                self.OperationMode._PID_Lock[iPID] = Lock()
        self.OperationMode._FactorStartDT = {}
        self.OperationMode._FactorPrepareIDs = {}
        for iFactor in self.OperationMode._Factors:
            iFactor._QN_init_operation(self.OperationMode.DateTimes[0], self.OperationMode._FactorStartDT,
                                      self.OperationMode.SectionIDs, self.OperationMode._FactorPrepareIDs)

    def _prepare(self, factor_names, ids, dts):
        self.OperationMode.FactorNames = factor_names
        self.OperationMode.DateTimes = dts
        self.OperationMode.IDs = ids
        self._initOperation()
        InitGroups = {}
        for iFactor in self.OperationMode._FactorDict.values():
            if iFactor.FactorTable is None:
                continue
            iFTID = id(iFactor.FactorTable)
            iPrepareIDs = self.OperationMode._FactorPrepareIDs[iFactor.Name]
            if iFTID not in InitGroups:
                InitGroups[iFTID] = [(iFactor.FactorTable, [iFactor], iPrepareIDs)]
            else:
                iGroups = InitGroups[iFTID]
                for j in range(len(iGroups)):
                    if iPrepareIDs == iGroups[j][2]:
                        iGroups[j][1].append(iFactor)
                        break
                else:
                    iGroups.append((iFactor.FactorTable, [iFactor], iPrepareIDs))
        GroupInfo, RawDataFileNames, PrepareIDs, PID_PrepareIDs = [], [], [], []
        for iFTID, iGroups in InitGroups.items():
            iGroupInfo = []
            jStartInd = 0
            for j in range(len(iGroups)):
                iFT = iGroups[j][0]
                ijGroupInfo = iFT.__QN_gen_group_info__(iGroups[j][1], self.OperationMode)
                iGroupInfo.extend(ijGroupInfo)
                ijGroupNum = len(ijGroupInfo)
                for k in range(ijGroupNum):
                    ijkRawDataFileName = iFT.Name + "-" + str(iFTID) + "-" + str(jStartInd + k)
                    for m in range(len(ijGroupInfo[k][1])):
                        self.OperationMode._FactorDict[ijGroupInfo[k][1][m]]._RawDataFile = ijkRawDataFileName
                    RawDataFileNames.append(ijkRawDataFileName)
                jStartInd += ijGroupNum
                PrepareIDs += [iGroups[j][2]] * ijGroupNum
                if iGroups[j][2] is not None:
                    PID_PrepareIDs += [{self.OperationMode._PIDs[i]: iSubIDs for i, iSubIDs in enumerate(
                        partition_list_moving_sampling(iGroups[j][2], len(self.OperationMode._PIDs)))}] * ijGroupNum
                else:
                    PID_PrepareIDs += [None] * ijGroupNum
            GroupInfo.extend(iGroupInfo)
        args = {"GroupInfo": GroupInfo, "FT": self, "RawDataFileNames": RawDataFileNames, "PrepareIDs": PrepareIDs,
                "PID_PrepareIDs": PID_PrepareIDs}
        if self.OperationMode.SubProcessNum == 0:
            Error = _prepareRawData(args)
        else:
            nPrcs = min((self.OperationMode.SubProcessNum, len(args["GroupInfo"])))
            Procs, Main2SubQueue, Sub2MainQueue = start_multi_process(pid="0", n_prc=nPrcs, target_fun=_prepareRawData,
                                                                    arg=args,
                                                                    partition_arg=["GroupInfo", "RawDataFileNames",
                                                                                   "PrepareIDs", "PID_PrepareIDs"],
                                                                    n_partition_head=0, n_partition_tail=0,
                                                                    main2sub_queue="None", sub2main_queue="Single")
            nGroup = len(GroupInfo)
            with ProgressBar(max_value=nGroup) as ProgBar:
                for i in range(nGroup):
                    iPID, Error, iMsg = Sub2MainQueue.get()
                    if Error != 1:
                        for iPID, iProc in Procs.items():
                            if iProc.is_alive():
                                iProc.terminate()
                        raise FactorError(iMsg)
                    ProgBar.update(i + 1)
            for iPrcs in Procs.values():
                iPrcs.join()
        self.OperationMode._isStarted = True
        return 0

    def _exit(self):
        self.OperationMode._CacheDir = None
        self.OperationMode._isStarted = False
        for iFactorName, iFactor in self.OperationMode._FactorDict.items():
            iFactor._exit()
        return 0

    def write2FDB(self, factor_names, ids, dts, factor_db, table_name, if_exists="update",
                  subprocess_num=cpu_count() - 1, dt_ruler=None, section_ids=None, specific_target={}, **kwargs):
        from QuantNodes.factor_node.factor_db import WritableFactorDB
        if not isinstance(factor_db, WritableFactorDB):
            raise FactorError("因子数据库: %s 不可写入!" % factor_db.Name)
        print("==========因子运算==========", "1. 原始数据准备", sep="\n", end="\n")
        TotalStartT = time.perf_counter()
        self.OperationMode.SubProcessNum = subprocess_num
        self.OperationMode.DTRuler = (dts if dt_ruler is None else dt_ruler)
        self.OperationMode.SectionIDs = section_ids
        self._prepare(factor_names, ids, dts)
        print(("耗时 : %.2f" % (time.perf_counter() - TotalStartT,)), "2. 因子数据计算", end="\n", sep="\n")
        StartT = time.perf_counter()
        Args = {"FT": self, "PID": "0", "FactorDB": factor_db, "TableName": table_name, "if_exists": if_exists,
                "specific_target": specific_target}
        if self.OperationMode.SubProcessNum == 0:
            _calculate(Args)
        else:
            nPrcs = len(self.OperationMode._PIDs)
            nTask = len(self.OperationMode._Factors) * nPrcs
            EventState = {iFactorName: 0 for iFactorName in self.OperationMode._Event}
            Procs, Main2SubQueue, Sub2MainQueue = start_multi_process(pid="0", n_prc=nPrcs, target_fun=_calculate,
                                                                    arg=Args,
                                                                    main2sub_queue="None", sub2main_queue="Single")
            iProg = 0
            with ProgressBar(max_value=nTask) as ProgBar:
                while True:
                    nEvent = len(EventState)
                    if nEvent > 0:
                        FactorNames = tuple(EventState.keys())
                        for iFactorName in FactorNames:
                            iQueue = self.OperationMode._Event[iFactorName][0]
                            while not iQueue.empty():
                                jInc = iQueue.get()
                                EventState[iFactorName] += jInc
                            if EventState[iFactorName] >= nPrcs:
                                self.OperationMode._Event[iFactorName][1].set()
                                EventState.pop(iFactorName)
                    while ((not Sub2MainQueue.empty()) or (nEvent == 0)) and (iProg < nTask):
                        iPID, iSubProg, iMsg = Sub2MainQueue.get()
                        iProg += iSubProg
                        ProgBar.update(iProg)
                    if iProg >= nTask:
                        break
            for iPID, iPrcs in Procs.items():
                iPrcs.join()
        print(("耗时 : %.2f" % (time.perf_counter() - StartT,)), "3. 清理缓存", end="\n", sep="\n")
        StartT = time.perf_counter()
        factor_db.connect()
        self._exit()
        print(('耗时 : %.2f' % (time.perf_counter() - StartT,)), ("总耗时 : %.2f" % (time.perf_counter() - TotalStartT,)),
              "=" * 28, sep="\n", end="\n")
        return 0


class CustomFT(FactorTable):
    """自定义因子表"""

    def __init__(self, name, sys_args={}, config_file=None, **kwargs):
        self._DateTimes = []
        self._IDs = []
        self._Factors = {}
        self._FactorDict = pd.DataFrame(columns=["FTID", "ArgIndex", "NameInFT", "DataType"], dtype=np.dtype("O"))
        self._TableArgDict = {}
        self._IDFilterStr = None
        self._CompiledIDFilter = {}
        self._isStarted = False
        return super().__init__(name=name, fdb=None, sys_args=sys_args, config_file=config_file, **kwargs)

    @property
    def FactorNames(self):
        return sorted(self._Factors)

    def getFactorMetaData(self, factor_names=None, key=None, args={}):
        if factor_names is None:
            factor_names = self.FactorNames
        if key is not None:
            return pd.Series({iFactorName: self._Factors[iFactorName].getMetaData(key) for iFactorName in factor_names})
        else:
            return pd.DataFrame(
                {iFactorName: self._Factors[iFactorName].getMetaData(key) for iFactorName in factor_names}).T

    def getFactor(self, ifactor_name, args={}, new_name=None):
        iFactor = self._Factors[ifactor_name]
        if new_name is not None:
            iFactor.Name = new_name
        return iFactor

    def getDateTime(self, ifactor_name=None, iid=None, start_dt=None, end_dt=None, args={}):
        DateTimes = self._DateTimes
        if (start_dt is not None) or (end_dt is not None):
            DateTimes = np.array(DateTimes, dtype="O")
            if start_dt is not None:
                DateTimes = DateTimes[DateTimes >= start_dt]
            if end_dt is not None:
                DateTimes = DateTimes[DateTimes <= end_dt]
            DateTimes = DateTimes.tolist()
        return DateTimes

    def getID(self, ifactor_name=None, idt=None, args={}):
        return self._IDs

    def getIDMask(self, idt, ids=None, id_filter_str=None, args={}):
        if ids is None:
            ids = self.getID(idt=idt, args=args)
        OldIDFilterStr = self.setIDFilter(id_filter_str)
        if self._IDFilterStr is None:
            self._IDFilterStr = OldIDFilterStr
            return pd.Series(True, index=ids)
        CompiledFilterStr, IDFilterFactors = self._CompiledIDFilter[self._IDFilterStr]
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]  # noqa: F841 (used in eval below)
        self._IDFilterStr = OldIDFilterStr
        return eval(CompiledFilterStr)

    def getFilteredID(self, idt, ids=None, id_filter_str=None, args={}):
        OldIDFilterStr = self.setIDFilter(id_filter_str)
        if ids is None:
            ids = self.getID(idt=idt, args=args)
        if self._IDFilterStr is None:
            self._IDFilterStr = OldIDFilterStr
            return ids
        CompiledFilterStr, IDFilterFactors = self._CompiledIDFilter[self._IDFilterStr]
        if CompiledFilterStr is None:
            raise FactorError("过滤条件字符串有误!")
        temp = self.readData(factor_names=IDFilterFactors, ids=ids, dts=[idt], args=args).loc[:, idt, :]  # noqa: F841 (used in eval below)
        self._IDFilterStr = OldIDFilterStr
        return eval("temp[" + CompiledFilterStr + "].index.tolist()")

    def __QN_calc_data__(self, raw_data, factor_names, ids, dts, args={}):
        return {iFactorName: self._Factors[iFactorName].readData(ids=ids, dts=dts, dt_ruler=self._DateTimes,
                                                                 section_ids=self._IDs) for iFactorName in
                factor_names}

    def write2FDB(self, factor_names, ids, dts, factor_db, table_name, if_exists="update",
                  subprocess_num=cpu_count() - 1, dt_ruler=None, section_ids=None, specific_target={}, **kwargs):
        if dt_ruler is None:
            dt_ruler = self._DateTimes
        if not dt_ruler:
            dt_ruler = None
        if section_ids is None:
            section_ids = self._IDs
        if (not section_ids) or (section_ids == ids):
            section_ids = None
        return super().write2FDB(factor_names, ids, dts, factor_db, table_name, if_exists, subprocess_num,
                                 dt_ruler=dt_ruler, section_ids=section_ids, specific_target=specific_target, **kwargs)

    def addFactors(self, factor_list=[], factor_table=None, factor_names=None, args={}):
        """添加因子"""
        for iFactor in factor_list:
            if iFactor.Name in self._Factors:
                raise FactorError("因子: '%s' 有重名!" % iFactor.Name)
            self._Factors[iFactor.Name] = iFactor
        if factor_table is None:
            return 0
        if factor_names is None:
            factor_names = factor_table.FactorNames
        for iFactorName in factor_names:
            if iFactorName in self._Factors:
                raise FactorError("因子: '%s' 有重名!" % iFactorName)
            iFactor = factor_table.getFactor(iFactorName, args=args)
            self._Factors[iFactor.Name] = iFactor
        return 0

    def deleteFactors(self, factor_names=None):
        """删除因子"""
        if factor_names is None:
            factor_names = self.FactorNames
        for iFactorName in factor_names:
            if iFactorName not in self._Factors:
                continue
            self._Factors.pop(iFactorName, None)
        return 0

    def renameFactor(self, factor_name, new_factor_name):
        """重命名因子"""
        if factor_name not in self._Factors:
            raise FactorError("因子: '%s' 不存在!" % factor_name)
        if (new_factor_name != factor_name) and (new_factor_name in self._Factors):
            raise FactorError("因子: '%s' 有重名!" % new_factor_name)
        self._Factors[new_factor_name] = self._Factors.pop(factor_name)
        return 0

    def setDateTime(self, dts):
        """设置时间点序列"""
        self._DateTimes = sorted(dts)

    def setID(self, ids):
        """设置 ID 序列"""
        self._IDs = sorted(ids)

    @property
    def IDFilterStr(self):
        """ID 过滤条件"""
        return self._IDFilterStr

    def setIDFilter(self, id_filter_str):
        """设置 ID 过滤条件"""
        OldIDFilterStr = self._IDFilterStr
        if not id_filter_str:
            self._IDFilterStr = None
            return OldIDFilterStr
        elif not isinstance(id_filter_str, str):
            raise FactorError("条件字符串必须为字符串或者为 None!")
        CompiledIDFilter = self._CompiledIDFilter.get(id_filter_str, None)
        if CompiledIDFilter is not None:
            self._IDFilterStr = id_filter_str
            return OldIDFilterStr
        CompiledIDFilterStr, IDFilterFactors = compile_id_filter_str(id_filter_str, self.FactorNames)
        if CompiledIDFilterStr is None:
            raise FactorError("条件字符串有误!")
        self._IDFilterStr = id_filter_str
        self._CompiledIDFilter[id_filter_str] = (CompiledIDFilterStr, IDFilterFactors)
        return OldIDFilterStr

    def start(self, dts, **kwargs):
        super().start(dts=dts, **kwargs)
        for iFactor in self._Factors.values():
            iFactor.start(dts=dts, **kwargs)
        return 0

    def end(self):
        super().end()
        for iFactor in self._Factors.values():
            iFactor.end()
        return 0
