# -*- coding: utf-8 -*-
"""缓存工具函数

从 FactorOperation.py 中提取的通用缓存操作函数，消除重复代码。
"""
import os
import shelve
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def create_std_data(
    dts: List[Any],
    ids: List[Any],
    data_type: str,
) -> np.ndarray:
    """创建标准数据数组

    Args:
        dts: 时间点列表
        ids: ID 列表
        data_type: 数据类型 ("double", "string", "object")

    Returns:
        填充了默认值的 numpy 数组
    """
    dtype = "float" if data_type == "double" else "O"
    fill = np.nan if data_type == "double" else None
    return np.full(shape=(len(dts), len(ids)), fill_value=fill, dtype=dtype)


def create_empty_dataframe(
    dts: List[Any],
    ids: List[Any],
    data_type: str,
    include_index: bool = True,
) -> pd.DataFrame:
    """创建空 DataFrame

    Args:
        dts: 时间点列表
        ids: ID 列表
        data_type: 数据类型 ("double", "string", "object")
        include_index: 是否包含索引（当 dts 为空时为 False）

    Returns:
        空的 DataFrame
    """
    dtype = "float" if data_type == "double" else "O"
    if include_index:
        return pd.DataFrame(index=dts, columns=ids, dtype=dtype)
    return pd.DataFrame(columns=ids, dtype=dtype)


def partition_ids_for_pid(
    operation_mode: Any,
    ids: Optional[List[Any]],
    pid: str,
) -> List[Any]:
    """按 PID 分区 ID 列表

    Args:
        operation_mode: 运算模式对象
        ids: 原始 ID 列表，None 表示使用该 PID 的全部 ID
        pid: 进程 ID

    Returns:
        该 PID 对应的 ID 子列表
    """
    from QuantNodes.core.tools import partition_list_moving_sampling

    if ids is None:
        return list(operation_mode._PID_IDs.get(pid, []))
    return partition_list_moving_sampling(ids, len(operation_mode._PIDs))[
        operation_mode._PIDs.index(pid)
    ]


def write_cache_file(
    operation_mode: Any,
    pid: str,
    factor_name: str,
    factor_id: int,
    std_data: pd.DataFrame,
    ids: List[Any],
    append: bool = False,
) -> None:
    """写入 shelve 缓存文件

    Args:
        operation_mode: 运算模式对象
        pid: 进程 ID
        factor_name: 因子名称
        factor_id: 因子唯一 ID
        std_data: 标准数据 DataFrame
        ids: ID 列表
        append: 是否追加模式（用于 SectionOperation/PanelOperation 写入所有 PID）
    """
    cache_dir = operation_mode._CacheDataDir + os.sep + pid
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = cache_dir + os.sep + factor_name + str(factor_id)

    with operation_mode._PID_Lock.get(pid, _DummyLock()):
        with shelve.open(cache_path) as cache_file:
            if append and "StdData" in cache_file:
                cache_file["StdData"] = pd.concat(
                    [cache_file["StdData"], std_data.loc[:, ids]]
                ).sort_index()
            else:
                cache_file["StdData"] = std_data
            cache_file["_QS_IDs"] = ids


def write_cache_files_for_all_pids(
    operation_mode: Any,
    pid_ids: Dict[str, List[Any]],
    factor_name: str,
    factor_id: int,
    std_data: pd.DataFrame,
) -> None:
    """为所有 PID 写入缓存文件（SectionOperation/PanelOperation 使用）

    Args:
        operation_mode: 运算模式对象
        pid_ids: {PID: [ID]} 映射
        factor_name: 因子名称
        factor_id: 因子唯一 ID
        std_data: 标准数据 DataFrame
    """
    for i_pid, i_ids in pid_ids.items():
        write_cache_file(
            operation_mode=operation_mode,
            pid=i_pid,
            factor_name=factor_name,
            factor_id=factor_id,
            std_data=std_data,
            ids=i_ids,
            append=True,
        )


class _DummyLock:
    """空锁，用于单进程模式"""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
