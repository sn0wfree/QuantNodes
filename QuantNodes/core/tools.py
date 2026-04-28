# coding=utf-8
"""
工具函数

替代 QuantStudio.Tools 中的工具函数
"""

import multiprocessing as mp
import os
import tempfile
import uuid
from typing import Any, Callable, Iterator, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd

T = TypeVar("T")


def gen_available_name(base: str = "Temp", used_names: Optional[set] = None) -> str:
    """
    生成唯一可用的名称

    Args:
        base: 名称基础前缀
        used_names: 已使用的名称集合

    Returns:
        唯一的名称字符串
    """
    used = used_names or set()
    while True:
        name = f"{base}_{uuid.uuid4().hex[:8]}"
        if name not in used:
            used.add(name)
            return name


def partition_list(data: List[T], n_parts: int) -> List[List[T]]:
    """
    将列表分割为 n 个大致相等的部分

    Args:
        data: 要分割的列表
        n_parts: 分割数量

    Returns:
        分割后的子列表
    """
    if n_parts <= 0:
        return [data]
    if n_parts >= len(data):
        return [[x] for x in data]
    
    part_size = len(data) // n_parts
    remainder = len(data) % n_parts
    
    result = []
    start = 0
    for i in range(n_parts):
        end = start + part_size + (1 if i < remainder else 0)
        result.append(data[start:end])
        start = end
    
    return result


def partition_list_moving_sampling(
    data: List[Any],
    n_parts: int,
    step: Optional[int] = None,
) -> List[List[Any]]:
    """
    将列表分割为 n 个部分，支持移动采样

    Args:
        data: 要分割的列表
        n_parts: 分割数量
        step: 采样步长（可选）

    Returns:
        分割后的子列表
    """
    if n_parts <= 0:
        return [data]
    if n_parts >= len(data):
        return [[x] for x in data]
    
    if step is None:
        step = max(1, len(data) // n_parts)
    
    result = []
    for i in range(n_parts):
        start = i * step
        end = min(start + step + (len(data) % n_parts if i < len(data) % n_parts else 0), len(data))
        if start < len(data):
            result.append(data[start:end])
        else:
            result.append([])
    
    return result


def start_multi_process(
    func: Callable,
    args_list: List[tuple],
    n_processes: Optional[int] = None,
    daemon: bool = True,
) -> List[Any]:
    """
    启动多进程执行任务

    Args:
        func: 要执行的函数
        args_list: 参数列表，每个元素是一个 tuple
        n_processes: 进程数，None 则使用 CPU 核心数
        daemon: 是否守护进程

    Returns:
        结果列表
    """
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    n_processes = min(n_processes, len(args_list)) if args_list else 1
    
    if n_processes == 1:
        return [func(*args) for args in args_list]
    
    with mp.Pool(processes=n_processes, maxtasksperchild=1) as pool:
        results = pool.starmap(func, args_list)
    
    return results


def fill_na_by_lookback(
    data: Union[pd.DataFrame, pd.Series],
    lookback: int = 1,
    method: str = "ffill",
) -> Union[pd.DataFrame, pd.Series]:
    """
    通过回溯填充 NaN 值

    Args:
        data: 数据 DataFrame 或 Series
        lookback: 回溯期数
        method: 填充方法，"ffill"（前向填充）或 "bfill"（后向填充）

    Returns:
        填充后的数据
    """
    if lookback <= 0:
        return data
    
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for _ in range(lookback):
            result = result.ffill()
        return result
    else:
        result = data.copy()
        for _ in range(lookback):
            result = result.ffill()
        return result


def get_shelve_file_suffix() -> str:
    """
    获取 shelve 数据库文件后缀

    Returns:
        文件后缀字符串
    """
    return ".db"


def test_id_filter_str(
    filter_str: str,
    factor_names: List[str],
) -> tuple:
    """
    测试并编译 ID 过滤字符串

    Args:
        filter_str: 过滤条件字符串
        factor_names: 可用的因子名列表

    Returns:
        (编译后的字符串, 涉及的因子列表) 或 (None, None) 如果失败
    """
    if not filter_str:
        return None, None
    
    try:
        valid_names = set(factor_names)
        if "@" in filter_str:
            factors = []
            parts = filter_str.replace("=", "==").replace(">", " > ").replace("<", " < ").replace("&", " & ").replace("|", " | ").split()
            for part in parts:
                part = part.strip()
                if part.startswith("@") and part[1:] in valid_names:
                    factors.append(part[1:])
            return filter_str, list(set(factors))
        return filter_str, []
    except Exception:
        return None, None


def create_temp_dir(prefix: str = "quantnodes_") -> str:
    """
    创建临时目录

    Args:
        prefix: 目录名前缀

    Returns:
        临时目录路径
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return temp_dir


def merge_data_frames(
    dfs: List[pd.DataFrame],
    how: str = "inner",
    on: Optional[str] = None,
    left_index: bool = False,
    right_index: bool = False,
) -> pd.DataFrame:
    """
    合并多个 DataFrame

    Args:
        dfs: DataFrame 列表
        how: 合并方式 ("inner", "outer", "left", "right")
        on: 合并键
        left_index: 左侧使用索引
        right_index: 右侧使用索引

    Returns:
        合并后的 DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    if len(dfs) == 1:
        return dfs[0]
    
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(result, df, how=how, on=on, left_index=left_index, right_index=right_index)
    
    return result


def chunk_iterable(iterable: Iterator[T], chunk_size: int) -> Iterator[List[T]]:
    """
    将迭代器分块

    Args:
        iterable: 可迭代对象
        chunk_size: 块大小

    Yields:
        块列表
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


__all__ = [
    "gen_available_name",
    "partition_list",
    "partition_list_moving_sampling",
    "start_multi_process",
    "fill_na_by_lookback",
    "get_shelve_file_suffix",
    "test_id_filter_str",
    "create_temp_dir",
    "merge_data_frames",
    "chunk_iterable",
    "timer",
    "retry",
]


def timer(func):
    """计时器装饰器"""
    from functools import wraps
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} spend: {end - start:.4f}s")
        return res

    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """重试装饰器"""
    from functools import wraps
    import time

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator
