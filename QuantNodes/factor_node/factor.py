# coding=utf-8
"""因子类

包含 Factor（因子基类实现）和 DataFactor（数据因子）
以及因子运算符辅助函数。
"""
import gc
import os
import shelve
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from traits.api import Enum, Int, Str

from QuantNodes.factor_node.quant_nodes_object import QuantNodesObject as _QN_Object
from QuantNodes.core.base import FactorError
from QuantNodes.core.tools import (
    partition_list_moving_sampling as partitionListMovingSampling,
    fill_na_by_lookback as fillNaByLookback,
)


def Factorize(factor_object, factor_name, args={}, **kwargs):
    """将运算结果转换成真正的可以存储的因子"""
    factor_object.Name = factor_name
    for iArg, iVal in args.items():
        factor_object[iArg] = iVal
    if "logger" in kwargs:
        factor_object._QN_Logger = kwargs.logger
    return factor_object


def _UnitaryOperator(f, idt, iid, x, args):
    """一元运算符"""
    Fun = args.get("Fun", None)
    if Fun is not None:
        Data = Fun(f, idt, iid, x, args["Arg"])
    else:
        Data = x[0]
    OperatorType = args.get("OperatorType", "neg")
    if OperatorType == "neg":
        return -Data
    elif OperatorType == "abs":
        return np.abs(Data)
    elif OperatorType == "not":
        return (~Data)
    else:
        raise FactorError("尚不支持的单因子运算符: %s" % OperatorType)


def _BinaryOperator(f, idt, iid, x, args):
    """二元运算符"""
    Fun1 = args.get("Fun1", None)
    if Fun1 is not None:
        Data1 = Fun1(f, idt, iid, x[:args["SepInd"]], args["Arg1"])
    else:
        Data1 = args.get("Data1", None)
        if Data1 is None:
            Data1 = x[0]
    Fun2 = args.get("Fun2", None)
    if Fun2 is not None:
        Data2 = Fun2(f, idt, iid, x[args["SepInd"]:], args["Arg2"])
    else:
        Data2 = args.get("Data2", None)
        if Data2 is None:
            Data2 = x[args["SepInd"]]
    OperatorType = args.get("OperatorType", "add")
    if OperatorType == "add":
        return Data1 + Data2
    elif OperatorType == "sub":
        return Data1 - Data2
    elif OperatorType == "mul":
        return Data1 * Data2
    elif OperatorType == "div":
        if np.isscalar(Data2):
            return (Data1 / Data2 if Data2 != 0 else np.empty(Data1.shape) + np.nan)
        Data2[Data2 == 0] = np.nan
        return Data1 / Data2
    elif OperatorType == "floordiv":
        return Data1 // Data2
    elif OperatorType == "mod":
        return Data1 % Data2
    elif OperatorType == "pow":
        if np.isscalar(Data2):
            if Data2 < 0:
                Data1[Data1 == 0] = np.nan
            return Data1 ** Data2
        if np.isscalar(Data1):
            if Data1 == 0:
                Data2[Data2 < 0] = np.nan
            return Data1 ** Data2
        Data1[(Data1 == 0) & (Data2 < 0)] = np.nan
        return Data1 ** Data2
    elif OperatorType == "and":
        return (Data1 & Data2)
    elif OperatorType == "or":
        return (Data1 | Data2)
    elif OperatorType == "xor":
        return (Data1 ^ Data2)
    elif OperatorType == "<":
        return (Data1 < Data2)
    elif OperatorType == "<=":
        return (Data1 <= Data2)
    elif OperatorType == ">":
        return (Data1 > Data2)
    elif OperatorType == ">=":
        return (Data1 >= Data2)
    elif OperatorType == "==":
        return (Data1 == Data2)
    elif OperatorType == "!=":
        return (Data1 != Data2)
    else:
        raise FactorError("尚不支持的多因子运算符: %s" % OperatorType)


class Factor(_QN_Object):
    """因子

    因子可看做一个 DataFrame(index=[时间点], columns=[ID])。
    时间点数据类型是 datetime.datetime，ID 的数据类型是 str。
    """
    Name = Str("因子")

    def __init__(self, name, ft, sys_args={}, config_file=None, **kwargs):
        self._FactorTable = ft
        self._NameInFT = name
        self.Name = name
        self._isStarted = False
        self._CacheData = None
        self._OperationMode = None
        self._RawDataFile = ""
        self._isCacheDataOK = False
        return super().__init__(sys_args=sys_args, config_file=config_file, **kwargs)

    @property
    def FactorTable(self):
        return self._FactorTable

    @property
    def Descriptors(self):
        return []

    def getMetaData(self, key=None, args={}):
        Args = self.Args
        Args.update(args)
        return self._FactorTable.getFactorMetaData(factor_names=[self._NameInFT], key=key, args=Args).loc[
            self._NameInFT]

    def getID(self, idt=None):
        if (self._OperationMode is not None) and (self._OperationMode._isStarted):
            return self._OperationMode.IDs
        if self._FactorTable is not None:
            return self._FactorTable.getID(ifactor_name=self._NameInFT, idt=idt, args=self.Args)
        return []

    def getDateTime(self, iid=None, start_dt=None, end_dt=None):
        if (self._OperationMode is not None) and (self._OperationMode._isStarted):
            return self._OperationMode.DateTimes
        if self._FactorTable is not None:
            return self._FactorTable.getDateTime(ifactor_name=self._NameInFT, iid=iid,
                                                 start_dt=start_dt, end_dt=end_dt, args=self.Args)
        return []

    def readData(self, ids, dts, **kwargs):
        if not self._isStarted:
            return self._FactorTable.readData(
                factor_names=[self._NameInFT], ids=ids, dts=dts, args=self.Args
            ).loc[self._NameInFT]
        if self._CacheData is None:
            self._CacheData = self._FactorTable.readData(
                factor_names=[self._NameInFT], ids=ids, dts=dts, args=self.Args
            ).loc[self._NameInFT]
            return self._CacheData
        NewDTs = sorted(set(dts).difference(self._CacheData.index))
        if NewDTs:
            NewCacheData = self._FactorTable.readData(
                factor_names=[self._NameInFT],
                ids=self._CacheData.columns.tolist(),
                dts=NewDTs,
                args=self.Args,
            ).loc[self._NameInFT]
            self._CacheData = self._CacheData.append(NewCacheData).loc[dts]
        NewIDs = sorted(set(ids).difference(self._CacheData.columns))
        if NewIDs:
            NewCacheData = self._FactorTable.readData(
                factor_names=[self._NameInFT],
                ids=NewIDs,
                dts=self._CacheData.index.tolist(),
                args=self.Args,
            ).loc[self._NameInFT]
            self._CacheData = pd.merge(self._CacheData, NewCacheData, left_index=True, right_index=True)
        return self._CacheData.loc[dts, ids]

    def _QN_init_operation(self, start_dt, dt_dict, prepare_ids, id_dict):
        OldStartDT = dt_dict.get(self.Name, start_dt)
        dt_dict[self.Name] = start_dt if start_dt < OldStartDT else OldStartDT
        PrepareIDs = id_dict.setdefault(self.Name, prepare_ids)
        if prepare_ids != PrepareIDs:
            raise FactorError("因子 %s 指定了不同的截面!" % self.Name)

    def __QN_prepare_cache_data__(self, ids=None):
        StartDT = self._OperationMode._FactorStartDT[self.Name]
        EndDT = self._OperationMode.DateTimes[-1]
        StartInd, EndInd = self._OperationMode.DTRuler.index(StartDT), self._OperationMode.DTRuler.index(EndDT)
        DTs = self._OperationMode.DTRuler[StartInd:EndInd + 1]
        RawDataFilePath = self._OperationMode._RawDataDir + os.sep + self._OperationMode._iPID + os.sep + self._RawDataFile
        if os.path.isfile(RawDataFilePath + self._OperationMode._FileSuffix):
            with shelve.open(RawDataFilePath, "r") as File:
                PrepareIDs = File["_QN_IDs"]
                if self._NameInFT in File:
                    RawData = File[self._NameInFT]
                elif "RawData" in File:
                    RawData = File["RawData"]
                else:
                    RawData = None
            if PrepareIDs is None:
                PrepareIDs = self._OperationMode._PID_IDs[self._OperationMode._iPID]
            if RawData is not None:
                StdData = self._FactorTable.__QN_calc_data__(
                    RawData, factor_names=[self._NameInFT], ids=PrepareIDs, dts=DTs, args=self.Args
                ).iloc[0]
            else:
                StdData = self._FactorTable.readData(
                    factor_names=[self._NameInFT], ids=PrepareIDs, dts=DTs, args=self.Args
                ).iloc[0]
        else:
            PrepareIDs = self._OperationMode._FactorPrepareIDs[self.Name]
            if PrepareIDs is None:
                PrepareIDs = self._OperationMode._PID_IDs[self._OperationMode._iPID]
            else:
                PrepareIDs = partitionListMovingSampling(PrepareIDs, len(self._OperationMode._PID_IDs))[
                    self._OperationMode._PIDs.index(self._OperationMode._iPID)]
            StdData = self._FactorTable.readData(
                factor_names=[self._NameInFT], ids=PrepareIDs, dts=DTs, args=self.Args
            ).iloc[0]
        with self._OperationMode._PID_Lock[self._OperationMode._iPID]:
            with shelve.open(
                self._OperationMode._CacheDataDir + os.sep + self._OperationMode._iPID + os.sep + self.Name + str(
                    self._OperationMode._FactorID[self.Name])) as CacheFile:
                CacheFile["StdData"] = StdData
                CacheFile["_QN_IDs"] = PrepareIDs
        self._isCacheDataOK = True
        return StdData

    def _QN_get_data(self, dts, pids=None, **kwargs):
        if pids is None:
            pids = set(self._OperationMode._PID_IDs)
            AllPID = True
        else:
            pids = set(pids)
            AllPID = False
        if not self._isCacheDataOK:
            StdData = self.__QN_prepare_cache_data__()
            if (StdData is not None) and (self._OperationMode._iPID in pids):
                pids.remove(self._OperationMode._iPID)
            else:
                StdData = None
        else:
            StdData = None
        while len(pids) > 0:
            iPID = pids.pop()
            iFilePath = self._OperationMode._CacheDataDir + os.sep + iPID + os.sep + self.Name + str(
                self._OperationMode._FactorID[self.Name])
            if not os.path.isfile(iFilePath + self._OperationMode._FileSuffix):
                pids.add(iPID)
                continue
            with self._OperationMode._PID_Lock[iPID]:
                with shelve.open(iFilePath, 'r') as CacheFile:
                    iStdData = CacheFile["StdData"]
            if StdData is None:
                StdData = iStdData
            else:
                StdData = pd.merge(StdData, iStdData, how='inner', left_index=True, right_index=True)
        if not AllPID:
            StdData = StdData.loc[list(dts), :]
        elif self._OperationMode._FactorPrepareIDs[self.Name] is None:
            StdData = StdData.loc[list(dts), self._OperationMode.IDs]
        else:
            StdData = StdData.loc[list(dts), self._OperationMode._FactorPrepareIDs[self.Name]]
        gc.collect()
        return StdData

    def _exit(self):
        self._OperationMode = None
        self._RawDataFile = ""
        self._isCacheDataOK = False

    def start(self, dts, **kwargs):
        self._isStarted = True
        return 0

    def end(self):
        self._CacheData = None
        self._isStarted = False
        return 0

    def _genUnitaryOperatorInfo(self):
        if self.Name == "":
            Args = {"Fun": self.Operator, "Arg": self.ModelArgs}
            return (self.Descriptors, Args)
        else:
            return ([self], {})

    def _genBinaryOperatorInfo(self, other):
        if isinstance(other, Factor):
            if (self.Name == "") and (other.Name == ""):
                Args = {"Fun1": self.Operator, "Fun2": other.Operator, "SepInd": len(self.Descriptors),
                        "Arg1": self.ModelArgs, "Arg2": other.ModelArgs}
                return (self.Descriptors + other.Descriptors, Args)
            elif self.Name == "":
                Args = {"Fun1": self.Operator, "SepInd": len(self.Descriptors), "Arg1": self.ModelArgs}
                return (self.Descriptors + [other], Args)
            elif other.Name == "":
                Args = {"Fun2": other.Operator, "SepInd": 1, "Arg2": other.ModelArgs}
                return ([self] + other.Descriptors, Args)
            else:
                Args = {"SepInd": 1}
                return ([self, other], Args)
        elif self.Name == "":
            Args = {"Fun1": self.Operator, "SepInd": len(self.Descriptors), "Data2": other, "Arg1": self.ModelArgs}
            return (self.Descriptors, Args)
        else:
            Args = {"SepInd": 1, "Data2": other}
            return ([self], Args)

    def _genRBinaryOperatorInfo(self, other):
        if self.Name == "":
            Args = {"Fun2": self.Operator, "SepInd": 0, "Data1": other, "Arg2": self.ModelArgs}
            return (self.Descriptors, Args)
        else:
            Args = {"SepInd": 0, "Data1": other}
            return ([self], Args)

    def _binary_op(self, other, op_type, is_reverse=False):
        from QuantNodes.factor_node.factor_operation import PointOperation
        if is_reverse:
            Descriptors, Args = self._genRBinaryOperatorInfo(other)
        else:
            Descriptors, Args = self._genBinaryOperatorInfo(other)
        Args["OperatorType"] = op_type
        return PointOperation("", Descriptors,
                              {"算子": _BinaryOperator, "参数": Args, "运算时点": "多时点", "运算ID": "多ID"},
                              logger=self._QN_Logger)

    def __add__(self, other):
        return self._binary_op(other, "add", is_reverse=False)

    def __radd__(self, other):
        return self._binary_op(other, "add", is_reverse=True)

    def __sub__(self, other):
        return self._binary_op(other, "sub", is_reverse=False)

    def __rsub__(self, other):
        return self._binary_op(other, "sub", is_reverse=True)

    def __mul__(self, other):
        return self._binary_op(other, "mul", is_reverse=False)

    def __rmul__(self, other):
        return self._binary_op(other, "mul", is_reverse=True)

    def __pow__(self, other):
        return self._binary_op(other, "pow", is_reverse=False)

    def __rpow__(self, other):
        return self._binary_op(other, "pow", is_reverse=True)

    def __truediv__(self, other):
        return self._binary_op(other, "div", is_reverse=False)

    def __rtruediv__(self, other):
        return self._binary_op(other, "div", is_reverse=True)

    def __floordiv__(self, other):
        return self._binary_op(other, "floordiv", is_reverse=False)

    def __rfloordiv__(self, other):
        return self._binary_op(other, "floordiv", is_reverse=True)

    def __mod__(self, other):
        return self._binary_op(other, "mod", is_reverse=False)

    def __rmod__(self, other):
        return self._binary_op(other, "mod", is_reverse=True)

    def __and__(self, other):
        return self._binary_op(other, "and", is_reverse=False)

    def __rand__(self, other):
        return self._binary_op(other, "and", is_reverse=True)

    def __or__(self, other):
        return self._binary_op(other, "or", is_reverse=False)

    def __ror__(self, other):
        return self._binary_op(other, "or", is_reverse=True)

    def __xor__(self, other):
        return self._binary_op(other, "xor", is_reverse=False)

    def __rxor__(self, other):
        return self._binary_op(other, "xor", is_reverse=True)

    def __lt__(self, other):
        return self._binary_op(other, "<", is_reverse=False)

    def __le__(self, other):
        return self._binary_op(other, "<=", is_reverse=False)

    def __eq__(self, other):
        return self._binary_op(other, "==", is_reverse=False)

    def __ne__(self, other):
        return self._binary_op(other, "!=", is_reverse=False)

    def __gt__(self, other):
        return self._binary_op(other, ">", is_reverse=False)

    def __ge__(self, other):
        return self._binary_op(other, ">=", is_reverse=False)

    def _unary_op(self, op_type):
        from QuantNodes.factor_node.factor_operation import PointOperation
        Descriptors, Args = self._genUnitaryOperatorInfo()
        Args["OperatorType"] = op_type
        return PointOperation("", Descriptors,
                              {"算子": _UnitaryOperator, "参数": Args, "运算时点": "多时点", "运算ID": "多ID"},
                              logger=self._QN_Logger)

    def __neg__(self):
        return self._unary_op("neg")

    def __pos__(self):
        return self

    def __abs__(self):
        return self._unary_op("abs")

    def __invert__(self):
        return self._unary_op("not")


class DataFactor(Factor):
    """直接赋予数据产生的因子"""
    DataType = Enum("double", "string", "object", arg_type="SingleOption", label="数据类型", order=0)
    LookBack = Int(0, arg_type="Integer", label="回溯天数", order=1)

    def __init__(self, name, data, sys_args={}, config_file=None, **kwargs):
        if isinstance(data, pd.Series):
            if data.index.is_all_dates:
                self._DataContent = "DateTime"
            else:
                self._DataContent = "ID"
            if "数据类型" not in sys_args:
                try:
                    data = data.astype(np.float)
                except:
                    sys_args["数据类型"] = "object"
                else:
                    sys_args["数据类型"] = "double"
        elif isinstance(data, pd.DataFrame):
            self._DataContent = "Factor"
            if "数据类型" not in sys_args:
                try:
                    data = data.astype(np.float)
                except:
                    sys_args["数据类型"] = "object"
                else:
                    sys_args["数据类型"] = "double"
        else:
            self._DataContent = "Value"
            if "数据类型" not in sys_args:
                if isinstance(data, str):
                    sys_args["数据类型"] = "string"
                else:
                    try:
                        data = float(data)
                    except:
                        sys_args["数据类型"] = "object"
                    else:
                        sys_args["数据类型"] = "double"
        self._Data = data
        return super().__init__(name=name, ft=None, sys_args=sys_args, config_file=None, **kwargs)

    def getMetaData(self, key=None, args={}):
        DataType = args.get("数据类型", self.DataType)
        if key is None:
            return pd.Series({"DataType": DataType})
        elif key == "DataType":
            return DataType
        return None

    def getID(self, idt=None):
        if (self._OperationMode is not None) and (self._OperationMode._isStarted):
            return self._OperationMode.IDs
        if self._DataContent == "Factor":
            return self._Data.columns.tolist()
        elif self._DataContent == "ID":
            return self._Data.index.tolist()
        else:
            return []

    def getDateTime(self, iid=None, start_dt=None, end_dt=None):
        if (self._OperationMode is not None) and (self._OperationMode._isStarted):
            return self._OperationMode.DateTimes
        if self._DataContent in ("DateTime", "Factor"):
            return self._Data.index.tolist()
        else:
            return []

    def readData(self, ids, dts, **kwargs):
        if self._DataContent == "Value":
            return pd.DataFrame([(self._Data,) * len(ids)] * len(dts), index=dts, columns=ids)
        elif self._DataContent == "ID":
            Data = pd.DataFrame(self._Data.values.reshape((1, self._Data.shape[0])).repeat(len(dts), axis=0), index=dts,
                                columns=self._Data.index)
        elif self._DataContent == "DateTime":
            Data = pd.DataFrame(self._Data.values.reshape((self._Data.shape[0], 1)).repeat(len(ids), axis=1),
                                index=self._Data.index, columns=ids)
        else:
            Data = self._Data
        if (Data.columns.intersection(ids).shape[0] == 0) or (Data.index.intersection(dts).shape[0] == 0):
            return pd.DataFrame(index=dts, columns=ids, dtype=("O" if self.DataType != "double" else np.float))
        if self.LookBack == 0:
            return Data.loc[dts, ids]
        else:
            return fillNaByLookback(Data.loc[sorted(Data.index.union(dts)), ids],
                                    lookback=self.LookBack * 24.0 * 3600).loc[dts, :]

    def __QN_prepare_cache_data__(self, ids=None):
        return self._Data

    def _QN_get_data(self, dts, pids=None, **kwargs):
        IDs = kwargs.get("ids", None)
        if IDs is None:
            if pids is None:
                IDs = list(self._OperationMode.IDs)
            else:
                IDs = []
                for iPID in pids:
                    IDs.extend(self._OperationMode._PID_IDs[iPID])
        dts = list(dts)
        return self.readData(sorted(IDs), dts)
