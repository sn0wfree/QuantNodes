# coding=utf-8
"""
内置因子运算函数

替代 QuantStudio.FactorDataBase.FactorOperation 中的因子运算函数

重构说明:
- 使用装饰器模式消除 90% 的模板代码
- 内置算子注册表，支持动态发现、文档生成、配置驱动
- 100% 向后兼容
"""

import datetime as dt
import inspect
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import uuid
from functools import wraps
from typing import Callable, Dict, Any, Optional, List

from QuantNodes.core.base import FactorError
from QuantNodes.factor_node.factor import Factor
from QuantNodes.factor_node.factor_operation import PointOperation, TimeOperation, SectionOperation


# ==============================================================================
# 常量定义
# ==============================================================================

class OperatorCategory:
    """算子分类常量"""
    POINT = "point"
    TIME = "time"
    SECTION = "section"
    MULTI_SECTION = "multi_section"


_METADATA = {
    "multi_dt": "多时点",
    "multi_id": "多ID",
    "full_section": "全截面",
}


# ==============================================================================
# 算子注册表
# ==============================================================================

_OPERATOR_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    OperatorCategory.POINT: {},
    OperatorCategory.TIME: {},
    OperatorCategory.SECTION: {},
    OperatorCategory.MULTI_SECTION: {},
}


def _register_operator(category: str, func: Callable, name: Optional[str] = None) -> None:
    """内部注册算子"""
    op_name = name or func.__name__
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    _OPERATOR_REGISTRY[category][op_name] = {
        "name": op_name,
        "category": category,
        "func": func,
        "doc": doc,
        "signature": str(sig),
        "parameters": list(sig.parameters.keys()),
    }


# ==============================================================================
# 注册表 API
# ==============================================================================

def list_operators(category: Optional[str] = None) -> List[str]:
    """
    列出所有算子名称
    
    Args:
        category: 算子分类，可选值: point, time, section, multi_section
    
    Returns:
        算子名称列表
    """
    if category:
        return list(_OPERATOR_REGISTRY.get(category, {}).keys())
    return [name for cat in _OPERATOR_REGISTRY for name in _OPERATOR_REGISTRY[cat]]


def get_operator(name: str, category: Optional[str] = None) -> Optional[Callable]:
    """
    根据名称获取算子函数
    
    Args:
        name: 算子名称
        category: 算子分类（可选，加快查找）
    
    Returns:
        算子函数，找不到返回 None
    """
    if category:
        op_info = _OPERATOR_REGISTRY.get(category, {}).get(name)
        return op_info["func"] if op_info else None
    for cat in _OPERATOR_REGISTRY:
        if name in _OPERATOR_REGISTRY[cat]:
            return _OPERATOR_REGISTRY[cat][name]["func"]
    return None


def operator_info(name: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    获取算子详细信息
    
    Args:
        name: 算子名称
        category: 算子分类（可选）
    
    Returns:
        算子信息字典: name, category, doc, signature, parameters
    """
    if category:
        return _OPERATOR_REGISTRY.get(category, {}).get(name)
    for cat in _OPERATOR_REGISTRY:
        if name in _OPERATOR_REGISTRY[cat]:
            return _OPERATOR_REGISTRY[cat][name]
    return None


def generate_documentation(output_format: str = "markdown") -> str:
    """
    生成算子文档
    
    Args:
        output_format: "markdown" 或 "dict"
    
    Returns:
        完整的算子文档
    """
    if output_format == "dict":
        return _OPERATOR_REGISTRY
    
    doc_lines = ["# 因子算子文档", ""]
    
    category_names = {
        OperatorCategory.POINT: "单点运算",
        OperatorCategory.TIME: "时间序列运算",
        OperatorCategory.SECTION: "单截面运算",
        OperatorCategory.MULTI_SECTION: "多截面运算",
    }
    
    for category in _OPERATOR_REGISTRY:
        if not _OPERATOR_REGISTRY[category]:
            continue
        
        doc_lines.append(f"## {category_names.get(category, category)}")
        doc_lines.append("")
        
        for op_name, op_info in _OPERATOR_REGISTRY[category].items():
            doc_lines.append(f"### `{op_name}`")
            doc_lines.append("")
            doc_lines.append(f"**签名**: `{op_info['signature']}`")
            doc_lines.append("")
            if op_info['doc']:
                doc_lines.append("**说明**:")
                doc_lines.append("")
                for line in op_info['doc'].split('\n'):
                    doc_lines.append(f"    {line}")
                doc_lines.append("")
            doc_lines.append("---")
            doc_lines.append("")
    
    return "\n".join(doc_lines)


# ==============================================================================
# 装饰器实现
# ==============================================================================

def point_operator(data_type: str = "double", multi_factor: bool = False):
    """
    单点运算装饰器
    
    使用方式:
        @point_operator()
        def isnull(f, idt, iid, x, args):
            Data = _genOperatorData(f, idt, iid, x, args)[0]
            return pd.isnull(Data)
        
        # 带参数的算子:
        @point_operator()
        def log(f, idt, iid, x, args, base=np.e):
            Data = _genOperatorData(f, idt, iid, x, args)[0]
            return np.log(Data) / np.log(base)
        
        # 多因子算子:
        @point_operator(multi_factor=True)
        def nansum(f, idt, iid, x, args, all_nan=0):
            Data = _genOperatorData(f, idt, iid, x, args)
            return np.nansum(np.array(Data), axis=0)
    
    Args:
        data_type: 数据类型描述
        multi_factor: 是否接收多个因子作为参数
    """
    def decorator(impl_func: Callable) -> Callable:
        # 获取实现函数的参数列表
        impl_sig = inspect.signature(impl_func)
        impl_params = list(impl_sig.parameters.keys())
        # 内部参数不提取到 OperatorArg
        internal_params = {"f", "idt", "iid", "x", "args"}
        op_arg_params = [p for p in impl_params if p not in internal_params]
        
        @wraps(impl_func)
        def wrapper(*args, **kwargs):
            if multi_factor:
                # 多因子: 所有位置参数都是因子
                factors = args
                args_after = ()
            else:
                # 单因子: 第一个位置参数是因子，其余作为算子参数
                factors = args[:1]
                args_after = args[1:]
            
            Descriptors, Args = _genMultivariateOperatorInfo(*factors)
            
            # 提取算子参数到 OperatorArg
            operator_args = {}
            # 先处理位置参数
            for i, param in enumerate(op_arg_params[:len(args_after)]):
                operator_args[param] = args_after[i]
            # 再处理关键字参数
            for param in op_arg_params[len(args_after):]:
                if param in kwargs:
                    operator_args[param] = kwargs.pop(param)
            
            Args["OperatorArg"] = operator_args
            
            return PointOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "运算时点": _METADATA["multi_dt"],
                    "运算ID": _METADATA["multi_id"],
                    "数据类型": data_type,
                },
                **kwargs
            )
        
        # 自动注册
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.POINT, wrapper, op_name)
        return wrapper
    return decorator


def single_section_operator():
    """
    单截面运算装饰器
    
    统一处理 mask/cat_data/weight_data/dummy_data/X 参数
    消除 8 个算子中 95% 的重复代码
    
    使用方式:
        @single_section_operator()
        def standardizeZScore(f, idt, iid, x, args, avg_statistics="平均值", ...):
            # 只有实际计算逻辑
            return result
    """
    def decorator(impl_func: Callable) -> Callable:
        impl_sig = inspect.signature(impl_func)
        impl_params = list(impl_sig.parameters.keys())
        internal_params = {"f", "idt", "iid", "x", "args"}
        # 辅助因子参数
        factor_params = {"mask", "cat_data", "weight_data", "dummy_data", "X"}
        # 算子自定义参数
        op_params = [p for p in impl_params if p not in internal_params and p not in factor_params]
        
        @wraps(impl_func)
        def wrapper(f, mask=None, cat_data=None, weight_data=None,
                    dummy_data=None, X=None, **kwargs):
            Factors = [f]
            OperatorArg = {}
            
            def add_factor(name, value):
                if value is None:
                    OperatorArg[name] = None
                    return None
                if isinstance(value, Factor):
                    Factors.append(value)
                    OperatorArg[name] = 1
                elif isinstance(value, list):
                    Factors.extend(value)
                    OperatorArg[name] = len(value)
                else:
                    OperatorArg[name] = value
                return OperatorArg[name]
            
            add_factor("mask", mask)
            add_factor("cat_data", cat_data)
            add_factor("weight_data", weight_data)
            add_factor("dummy_data", dummy_data)
            add_factor("X", X)
            
            # 提取算子自定义参数
            for param in op_params:
                if param in kwargs:
                    OperatorArg[param] = kwargs.pop(param)
            
            Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
            Args["OperatorArg"] = OperatorArg
            return SectionOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "运算时点": _METADATA["multi_dt"],
                    "输出形式": _METADATA["full_section"],
                },
                **kwargs
            )
        
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.SECTION, wrapper, op_name)
        return wrapper
    return decorator


def rolling_operator():
    """
    滚动窗口运算装饰器
    
    统一处理 window, min_periods, win_type, weights 等参数
    自动计算回溯期数，返回 TimeOperation
    
    使用方式:
        @rolling_operator()
        def rolling_mean(f, idt, iid, x, args):
            Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
            return Data.rolling(**args["OperatorArg"]).mean().values[args["OperatorArg"]["window"] - 1:]
    """
    def decorator(impl_func: Callable) -> Callable:
        impl_sig = inspect.signature(impl_func)
        impl_params = list(impl_sig.parameters.keys())
        internal_params = {"f", "idt", "iid", "x", "args"}
        op_params = [p for p in impl_params if p not in internal_params]
        
        @wraps(impl_func)
        def wrapper(f, window, min_periods=1, win_type=None, weights=None, **kwargs):
            Descriptors, Args = _genMultivariateOperatorInfo(f)
            OperatorArg = {"window": window, "min_periods": min_periods, "win_type": win_type}
            
            # 处理权重参数（如果有）
            if weights is not None:
                OperatorArg["window"] = len(weights)
                OperatorArg["weights"] = weights
            
            # 提取其他自定义参数
            for param in op_params:
                if param in kwargs:
                    OperatorArg[param] = kwargs.pop(param)
            
            Args["OperatorArg"] = OperatorArg
            
            # 回溯期数
            lookback = OperatorArg["window"] - 1
            
            return TimeOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "回溯期数": [lookback] * len(Descriptors),
                    "运算时点": _METADATA["multi_dt"],
                    "运算ID": "多ID",
                },
                **kwargs
            )
        
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def expanding_operator():
    """
    扩展窗口运算装饰器
    
    统一处理 min_periods 等参数
    自动计算回溯期数（min_periods - 1），返回 TimeOperation
    
    使用方式:
        @expanding_operator()
        def expanding_mean(f, idt, iid, x, args):
            Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
            return Data.expanding(**args["OperatorArg"]).mean().values[args["OperatorArg"]["min_periods"] - 1:]
    """
    def decorator(impl_func: Callable) -> Callable:
        impl_sig = inspect.signature(impl_func)
        impl_params = list(impl_sig.parameters.keys())
        internal_params = {"f", "idt", "iid", "x", "args"}
        op_params = [p for p in impl_params if p not in internal_params]
        
        @wraps(impl_func)
        def wrapper(f, min_periods=1, **kwargs):
            Descriptors, Args = _genMultivariateOperatorInfo(f)
            OperatorArg = {"min_periods": min_periods}
            
            # 提取其他自定义参数
            for param in op_params:
                if param in kwargs:
                    OperatorArg[param] = kwargs.pop(param)
            
            Args["OperatorArg"] = OperatorArg
            
            # 回溯期数
            lookback = min_periods - 1
            
            return TimeOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "回溯期数": [lookback] * len(Descriptors),
                    "运算时点": _METADATA["multi_dt"],
                    "运算ID": "多ID",
                },
                **kwargs
            )

        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def ewm_operator(dual_factor: bool = False, has_sub_args: bool = False):
    """
    指数加权移动平均（EWM）运算装饰器
    
    统一处理 com, span, halflife, alpha 等指数加权参数
    支持单因子和双因子算子，自动处理 SubOperatorArg 嵌套参数
    
    使用方式:
        @ewm_operator()
        def ewm_mean(f, idt, iid, x, args):
            Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
            return Data.ewm(**args["OperatorArg"]).mean().values[args["OperatorArg"]["min_periods"] - 1:]
        
        @ewm_operator(dual_factor=True, has_sub_args=True)
        def ewm_cov(f, idt, iid, x, args):
            Data1, Data2 = _genOperatorData(f, idt, iid, x, args)
            OperatorArg = args["OperatorArg"].copy()
            SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
            return pd.DataFrame(Data1).ewm(**OperatorArg).cov(pd.DataFrame(Data2), **SubOperatorArg).values[
                   args["OperatorArg"]["min_periods"] - 1:]
    """
    def decorator(impl_func: Callable) -> Callable:
        impl_sig = inspect.signature(impl_func)
        impl_params = list(impl_sig.parameters.keys())
        internal_params = {"f", "idt", "iid", "x", "args"}
        op_params = [p for p in impl_params if p not in internal_params]
        
        @wraps(impl_func)
        def wrapper(f1, f2=None, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, bias=False, **kwargs):
            if dual_factor:
                Descriptors, Args = _genMultivariateOperatorInfo(f1, f2)
            else:
                Descriptors, Args = _genMultivariateOperatorInfo(f1)
            
            OperatorArg = {
                "com": com,
                "span": span,
                "halflife": halflife,
                "alpha": alpha,
                "min_periods": min_periods,
                "adjust": adjust,
                "ignore_na": ignore_na,
            }
            
            if has_sub_args:
                OperatorArg["SubOperatorArg"] = {"bias": bias}
            
            for param in op_params:
                if param in kwargs:
                    OperatorArg[param] = kwargs.pop(param)
            
            Args["OperatorArg"] = OperatorArg
            
            lookback = min_periods - 1 if min_periods > 0 else 0
            
            return TimeOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "回溯期数": [lookback] * len(Descriptors),
                    "运算时点": _METADATA["multi_dt"],
                    "运算ID": "多ID",
                },
                **kwargs
            )
        
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def dual_factor_rolling_operator(has_sub_args: bool = False, dynamic_time_mode: bool = False):
    """
    双因子滚动窗口运算装饰器
    
    用于处理需要两个因子输入的滚动运算（如协方差、相关系数）
    支持动态运算时点切换（单时点/多时点）和 SubOperatorArg 嵌套参数
    
    使用方式:
        @dual_factor_rolling_operator(has_sub_args=True)
        def rolling_cov(f, idt, iid, x, args):
            Data1, Data2 = _genOperatorData(f, idt, iid, x, args)
            OperatorArg = args["OperatorArg"].copy()
            SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
            return pd.DataFrame(Data1).rolling(**OperatorArg).cov(pd.DataFrame(Data2), **SubOperatorArg).values[
                   args["OperatorArg"]["window"] - 1:]
    """
    def decorator(impl_func: Callable) -> Callable:
        impl_sig = inspect.signature(impl_func)
        impl_params = list(impl_sig.parameters.keys())
        internal_params = {"f", "idt", "iid", "x", "args"}
        op_params = [p for p in impl_params if p not in internal_params]
        
        @wraps(impl_func)
        def wrapper(f1, f2, window, min_periods=1, win_type=None, ddof=1, method="pearson", **kwargs):
            Descriptors, Args = _genMultivariateOperatorInfo(f1, f2)
            
            OperatorArg = {
                "window": window,
                "min_periods": min_periods,
                "win_type": win_type,
            }
            
            if has_sub_args:
                OperatorArg["SubOperatorArg"] = {"ddof": ddof}
            
            if dynamic_time_mode:
                OperatorArg["method"] = method
            
            for param in op_params:
                if param in kwargs:
                    OperatorArg[param] = kwargs.pop(param)
            
            Args["OperatorArg"] = OperatorArg
            
            lookback = window - 1
            
            time_mode = "多时点"
            if dynamic_time_mode and method != "pearson":
                time_mode = "单时点"
            
            return TimeOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "回溯期数": [lookback] * len(Descriptors),
                    "运算时点": time_mode,
                    "运算ID": "多ID",
                },
                **kwargs
            )
        
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def dual_factor_expanding_operator(has_sub_args: bool = False):
    """
    双因子扩展窗口运算装饰器
    
    用于处理需要两个因子输入的扩展窗口运算（如协方差、相关系数）
    """
    def decorator(impl_func: Callable) -> Callable:
        impl_sig = inspect.signature(impl_func)
        impl_params = list(impl_sig.parameters.keys())
        internal_params = {"f", "idt", "iid", "x", "args"}
        op_params = [p for p in impl_params if p not in internal_params]
        
        @wraps(impl_func)
        def wrapper(f1, f2, min_periods=1, ddof=1, **kwargs):
            Descriptors, Args = _genMultivariateOperatorInfo(f1, f2)
            
            OperatorArg = {"min_periods": min_periods}
            
            if has_sub_args:
                OperatorArg["SubOperatorArg"] = {"ddof": ddof}
            
            for param in op_params:
                if param in kwargs:
                    OperatorArg[param] = kwargs.pop(param)
            
            Args["OperatorArg"] = OperatorArg
            
            lookback = min_periods - 1
            
            return TimeOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "回溯期数": [lookback] * len(Descriptors),
                    "运算时点": _METADATA["multi_dt"],
                    "运算ID": "多ID",
                },
                **kwargs
            )
        
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def time_shift_operator(lookback_param: str = "n"):
    """
    时间位移运算装饰器
    
    用于处理 lag, diff 等需要时间位移的算子
    自动处理回溯期数计算
    
    使用方式:
        @time_shift_operator(lookback_param="window")
        def lag(f, idt, iid, x, args):
            if args["OperatorArg"]['dt_change_fun'] is None:
                return x[0][args["OperatorArg"]['window'] - args["OperatorArg"]['lag_period']:x[0].shape[0] - args["OperatorArg"]['lag_period']]
            ...
    """
    def decorator(impl_func: Callable) -> Callable:
        impl_sig = inspect.signature(impl_func)
        impl_params = list(impl_sig.parameters.keys())
        internal_params = {"f", "idt", "iid", "x", "args"}
        op_params = [p for p in impl_params if p not in internal_params]
        
        @wraps(impl_func)
        def wrapper(f, lag_period=1, window=1, n=1, dt_change_fun=None, **kwargs):
            Descriptors, Args = _genMultivariateOperatorInfo(f)
            
            OperatorArg = {}
            if "lag_period" in impl_params or lag_period != 1:
                OperatorArg["lag_period"] = lag_period
            if "window" in impl_params or window != 1:
                OperatorArg["window"] = window
            if "n" in impl_params or n != 1:
                OperatorArg["n"] = n
            if "dt_change_fun" in impl_params or dt_change_fun is not None:
                OperatorArg["dt_change_fun"] = dt_change_fun
            
            for param in op_params:
                if param in kwargs:
                    OperatorArg[param] = kwargs.pop(param)
            
            Args["OperatorArg"] = OperatorArg
            
            if lookback_param == "window":
                lookback = OperatorArg.get("window", 1)
            else:
                lookback = OperatorArg.get("n", 1)
            
            return TimeOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "回溯期数": [lookback] * len(Descriptors),
                    "运算时点": _METADATA["multi_dt"],
                    "运算ID": "多ID",
                },
                **kwargs
            )
        
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def fillna_operator():
    """
    缺失值填充装饰器
    
    支持两种模式：
    1. value=None: 向前填充，返回 TimeOperation
    2. value!=None: 常值填充，返回 PointOperation
    
    使用方式:
        @fillna_operator()
        def fillna(f, idt, iid, x, args):
            Data = _genOperatorData(f, idt, iid, x, args)[0]
            if args["OperatorArg"]["value"] is None:
                LookBack = args["OperatorArg"]["lookback"]
                return pd.DataFrame(Data).fillna(method="ffill", limit=LookBack).values[LookBack:]
            else:
                Data[pd.isnull(Data)] = args["OperatorArg"]["value"]
                return Data
    """
    def decorator(impl_func: Callable) -> Callable:
        @wraps(impl_func)
        def wrapper(f, value=None, lookback=1, **kwargs):
            Descriptors, Args = _genMultivariateOperatorInfo(f)
            Args["OperatorArg"] = {"lookback": lookback, "value": value}
            
            if value is None:
                return TimeOperation(
                    kwargs.pop("factor_name", str(uuid.uuid1())),
                    Descriptors,
                    {
                        "算子": impl_func,
                        "参数": Args,
                        "回溯期数": [lookback] * len(Descriptors),
                        "运算时点": _METADATA["multi_dt"],
                        "运算ID": "多ID",
                    },
                    **kwargs
                )
            else:
                return PointOperation(
                    kwargs.pop("factor_name", str(uuid.uuid1())),
                    Descriptors,
                    {
                        "算子": impl_func,
                        "参数": Args,
                        "运算时点": _METADATA["multi_dt"],
                        "运算ID": "多ID",
                    },
             **kwargs
         )
         
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def nav_operator():
    """
    净值（NAV）运算装饰器
    
    支持自身回溯模式和初始值设置
    
    使用方式:
        @nav_operator()
        def nav(f, idt, iid, x, args):
            Price = x[0]
            Return, = _genOperatorData(f, idt, iid, x[1:], args)
            ...
    """
    def decorator(impl_func: Callable) -> Callable:
        @wraps(impl_func)
        def wrapper(ret, init=None, **kwargs):
            Descriptors, Args = _genMultivariateOperatorInfo(ret)
            return TimeOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "回溯期数": [0] * len(Descriptors),
                    "自身回溯期数": 1,
                    "自身回溯模式": "扩张窗口",
                    "自身初始值": init,
                    "运算时点": _METADATA["multi_dt"],
                    "运算ID": "多ID",
                },
                **kwargs
            )
        
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def multifactor_rolling_operator():
    """
    多因子滚动回归装饰器
    
    支持 Y + *X 模式的多因子输入，自动设置动态 dtype
    
    使用方式:
        @multifactor_rolling_operator()
        def rolling_regress(f, idt, iid, x, args):
            X = _genOperatorData(f, idt, iid, x, args)
            Y = X[0]
            ...
    """
    def decorator(impl_func: Callable) -> Callable:
        @wraps(impl_func)
        def wrapper(Y, *X, window=20, constant=True, half_life=np.inf, **kwargs):
            Descriptors, Args = _genMultivariateOperatorInfo(*((Y,) + X))
            Args["OperatorArg"] = {"window": window, "constant": constant, "half_life": half_life}
            nX = len(X)

            f = TimeOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "回溯期数": [window - 1] * len(Descriptors),
                    "运算时点": _METADATA["multi_dt"],
                    "运算ID": "多ID",
                    "数据类型": "object",
                },
                **kwargs
            )
            
            if constant:
                DataType = [('alpha', np.float64)] + [('beta' + str(i), np.float64) for i in range(nX)]
                DataType += [('t_alpha', np.float64)] + [('t_beta' + str(i), np.float64) for i in range(nX)]
            else:
                DataType = [('beta' + str(i), np.float64) for i in range(nX)]
                DataType += [('t_beta' + str(i), np.float64) for i in range(nX)]
            DataType += [('fvalue', np.float64), ('rsquared', np.float64), ('rsquared_adj', np.float64)]
            f.TempData["dtype"] = DataType
            return f
        
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def point_rolling_operator():
    """
    点输出滚动算子装饰器
    
    滚动计算但输出是单时点（如 rolling_regress_change）
    """
    def decorator(impl_func: Callable) -> Callable:
        @wraps(impl_func)
        def wrapper(f, window=20, min_periods=2, **kwargs):
            Descriptors, Args = _genMultivariateOperatorInfo(f)
            Args["OperatorArg"] = {"min_periods": min_periods}
            return PointOperation(
                kwargs.pop("factor_name", str(uuid.uuid1())),
                Descriptors,
                {
                    "算子": impl_func,
                    "参数": Args,
                    "运算时点": "单时点",
                    "运算ID": "多ID",
                },
                **kwargs
            )
        
        op_name = impl_func.__name__.lstrip('_')
        _register_operator(OperatorCategory.TIME, wrapper, op_name)
        return wrapper
    return decorator


def _genMultivariateOperatorInfo(*factors):
    Args = {}
    Descriptors = []
    for i, iFactor in enumerate(factors):
        iInd = str(i + 1)
        if isinstance(iFactor, Factor):  # 第i个操作子为因子
            if iFactor.Name == "":  # 第i个因子为中间运算因子
                Args["Fun" + iInd] = iFactor.Operator
                Args["Arg" + iInd] = iFactor.ModelArgs
                Args["SepInd" + iInd] = Args.get("SepInd" + str(i), 0) + len(iFactor.Descriptors)
                Descriptors += iFactor.Descriptors
            else:  # 第i个因子为最终因子
                Args["SepInd" + iInd] = Args.get("SepInd" + str(i), 0) + 1
                Descriptors += [iFactor]
        else:  # 第i个操作子为标量
            Args["Data" + iInd] = iFactor
            Args["SepInd" + iInd] = Args.get("SepInd" + str(i), 0)
    Args["nData"] = len(factors)
    return (Descriptors, Args)


def _genOperatorData(f, idt, iid, x, args):
    Data = []
    for i in range(args["nData"]):
        iInd = str(i + 1)
        iFun = args.get("Fun" + iInd, None)
        if iFun is not None:
            Data.append(iFun(f, idt, iid, x[args.get("SepInd" + str(i), 0):args.get("SepInd" + iInd, args["nData"])],
                             args["Arg" + iInd]))
        else:
            if "Data" + iInd in args:
                Data.append(args["Data" + iInd])
            else:
                Data.append(x[args.get("SepInd" + str(i), 0)])
    return Data


# ----------------------单点运算--------------------------------
@point_operator()
def astype(f, idt, iid, x, args, dtype):
    """类型转换"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return Data.astype(dtype=args["OperatorArg"]["dtype"])


@point_operator()
def log(f, idt, iid, x, args, base=np.e):
    """取对数"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    Data[Data <= 0] = np.nan
    return np.log(Data.astype(float)) / np.log(args["OperatorArg"]["base"])


@point_operator()
def isnull(f, idt, iid, x, args):
    """判断是否为空值"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return pd.isnull(Data)


@point_operator()
def notnull(f, idt, iid, x, args):
    """判断是否为非空值"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return pd.notnull(Data)


@point_operator()
def sign(f, idt, iid, x, args):
    """取符号"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return np.sign(Data.astype(float))


@point_operator()
def ceil(f, idt, iid, x, args):
    """向上取整"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return np.ceil(Data.astype(float))


@point_operator()
def floor(f, idt, iid, x, args):
    """向下取整"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return np.floor(Data.astype(float))


@point_operator()
def fix(f, idt, iid, x, args):
    """向零取整"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return np.fix(Data.astype(float))


@point_operator()
def applymap(f, idt, iid, x, args, func=id):
    """应用自定义函数"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    Func = args["OperatorArg"]["func"]
    return Data.applymap(Func).values


@point_operator()
def fetch(f, idt, iid, x, args, pos=0, dtype="double"):
    """获取指定位置数据"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    if isinstance(args["OperatorArg"]["pos"], str):
        return Data.astype(args["OperatorArg"]["dtype"])[args["OperatorArg"]["pos"]]
    SampleData = Data[0, 0]
    DataType = np.dtype(
        [(str(i), (np.float64 if isinstance(SampleData[i], float) else "O")) for i in range(len(SampleData))])
    return Data.astype(DataType)[str(args["OperatorArg"]["pos"])]


@point_operator(multi_factor=True)
def where(f, idt, iid, x, args):
    """条件选择"""
    Data = _genOperatorData(f, idt, iid, x, args)
    return np.where(Data[1], Data[0], Data[2])


@point_operator()
def replace(f, idt, iid, x, args, value_map=None):
    """值替换"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    ValueMap = args["OperatorArg"]["value_map"]
    if f.DataType == "double":
        Rslt = np.full_like(Data, fill_value=np.nan, dtype="float64")
    else:
        Rslt = np.full_like(Data, fill_value=None, dtype="O")
    for iKey, iVal in ValueMap.items():
        if pd.isnull(iKey):
            Rslt[pd.isnull(Data)] = iVal
        else:
            Rslt[Data == iKey] = iVal
    return Rslt


@point_operator(multi_factor=True)
def clip(f, idt, iid, x, args):
    """截断处理"""
    Data = _genOperatorData(f, idt, iid, x, args)
    return np.clip(Data[0].astype(float), Data[1], Data[2])


@point_operator(multi_factor=True)
def nansum(f, idt, iid, x, args, all_nan=0):
    """忽略空值求和"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    Data = np.array(Data)
    Rslt = np.nansum(Data, axis=0)
    Mask = (np.sum(pd.notnull(Data), axis=0) == 0)
    Rslt[Mask] = args["OperatorArg"]["all_nan"]
    return Rslt


@point_operator(multi_factor=True)
def nanprod(f, idt, iid, x, args, all_nan=1):
    """忽略空值求积"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    Data = np.array(Data)
    Rslt = np.nanprod(Data, axis=0)
    Mask = (np.sum(pd.notnull(Data), axis=0) == 0)
    Rslt[Mask] = args["OperatorArg"]["all_nan"]
    return Rslt


@point_operator(multi_factor=True)
def nanmax(f, idt, iid, x, args):
    """忽略空值求最大值"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    return np.nanmax(np.array(Data), axis=0)


@point_operator(multi_factor=True)
def nanmin(f, idt, iid, x, args):
    """忽略空值求最小值"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    return np.nanmin(np.array(Data), axis=0)


@point_operator(multi_factor=True)
def nanargmax(f, idt, iid, x, args):
    """忽略空值求最大值索引"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    Data = np.array(Data)
    Mask = pd.isnull(Data)
    Data[Mask] = -np.inf
    Rslt = np.nanargmax(Data, axis=0)
    Mask = (np.sum(Mask, axis=0) == Data.shape[0])
    Rslt[Mask] = np.nan
    return Rslt


@point_operator(multi_factor=True)
def nanargmin(f, idt, iid, x, args):
    """忽略空值求最小值索引"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    Data = np.array(Data)
    Mask = pd.isnull(Data)
    Data[Mask] = np.inf
    Rslt = np.nanargmin(Data, axis=0)
    Mask = (np.sum(Mask, axis=0) == Data.shape[0])
    Rslt[Mask] = np.nan
    return Rslt


@point_operator(multi_factor=True)
def nanmean(f, idt, iid, x, args, weights=None, ignore_nan_weight=True):
    """忽略空值求均值"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    Weights = args["OperatorArg"]["weights"]
    if Weights is None:
        if args["OperatorArg"]["ignore_nan_weight"]:
            return np.nanmean(np.array(Data), axis=0)
        Weights = [1] * len(Data)
    Rslt = np.zeros(Data[0].shape)
    WeightArray = np.zeros(Data[0].shape)
    for i, iData in enumerate(Data):
        iMask = pd.notnull(iData)
        WeightArray += iMask * Weights[i]
        iData[~iMask] = 0.0
        Rslt += iData * Weights[i]
    if args["OperatorArg"]["ignore_nan_weight"]:
        WeightArray[WeightArray == 0.0] = np.nan
        return Rslt / WeightArray
    else:
        Rslt[WeightArray == 0.0] = np.nan
        return Rslt / len(Data)


@point_operator(multi_factor=True)
def nanstd(f, idt, iid, x, args, ddof=1):
    """忽略空值求标准差"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    return np.nanstd(np.array(Data), axis=0, ddof=args["OperatorArg"]["ddof"])


@point_operator(multi_factor=True)
def nanvar(f, idt, iid, x, args, ddof=1):
    """忽略空值求方差"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    return np.nanvar(np.array(Data), axis=0, ddof=args["OperatorArg"]["ddof"])


@point_operator(multi_factor=True)
def nanmedian(f, idt, iid, x, args):
    """忽略空值求中位数"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    return np.nanmedian(np.array(Data), axis=0)


@point_operator(multi_factor=True)
def nanquantile(f, idt, iid, x, args, quantile=0.5):
    """忽略空值求分位数"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    return np.nanpercentile(np.array(Data), args["OperatorArg"]["quantile"] * 100, axis=0)


@point_operator(multi_factor=True)
def nancount(f, idt, iid, x, args):
    """统计空值数量"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    return np.nansum(pd.isnull(np.array(Data)), axis=0)


@point_operator(multi_factor=True)
def regress_change_rate(f, idt, iid, x, args):
    """回归变化率"""
    Rslt = np.array(DataPreprocessingFun.regressChangeRate(x), dtype=np.float64)
    Rslt[Rslt == 0.0] = np.nan
    return Rslt


@point_operator(multi_factor=True)
def tolist(f, idt, iid, x, args):
    """转换为列表"""
    Data = [(iData if isinstance(iData, np.ndarray) else np.full(shape=(len(idt), len(iid)), fill_value=iData)) for
            iData in _genOperatorData(f, idt, iid, x, args)]
    return pd.DataFrame(
        np.array([np.array(d).flatten() for d in Data]).T,
        index=pd.MultiIndex.from_product([idt, iid])
    ).apply(lambda s: s.tolist(), axis=1).unstack().values


@point_operator()
def to_json(f, idt, iid, x, args):
    """转换为JSON"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return pd.DataFrame(Data).applymap(lambda v: json.dumps(v, ensure_ascii=False) if pd.notnull(v) else None).values


@point_operator(multi_factor=True)
def single_quarter(f, idt, iid, x, args):
    """单季度处理"""
    ReportPeriod, Last, Prev = _genOperatorData(f, idt, iid, x, args)
    f_vec = np.vectorize(lambda x: x[-4:] == "0331")
    Rslt = Last - Prev
    Mask = f_vec(ReportPeriod)
    Rslt[Mask] = Last[Mask]
    return Rslt


@point_operator()
def strftime(f, idt, iid, x, args, dt_format="%Y%m%d"):
    """日期格式化"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    DTFormat = args["OperatorArg"]["dt_format"]
    return pd.DataFrame(Data).applymap(lambda x: x.strftime(DTFormat) if pd.notnull(x) else None).values


@point_operator()
def strptime(f, idt, iid, x, args, dt_format="%Y%m%d", is_datetime=True):
    """字符串转日期"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    DTFormat = args["OperatorArg"]["dt_format"]
    if args["OperatorArg"]["is_datetime"]:
        return pd.DataFrame(Data).applymap(
            lambda x: dt.datetime.strptime(x, DTFormat) if pd.notnull(x) else None).values
    else:
        return pd.DataFrame(Data).applymap(
            lambda x: dt.datetime.strptime(x, DTFormat).date() if pd.notnull(x) else None).values


# ----------------------时间序列运算--------------------------------
@rolling_operator()
def rolling_mean(f, idt, iid, x, args):
    """滚动窗口均值"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    if "weights" not in args["OperatorArg"]:
        return Data.rolling(**args["OperatorArg"]).mean().values[args["OperatorArg"]["window"] - 1:]
    else:
        weights = np.array(args["OperatorArg"]["weights"])
        return Data.rolling(**args["OperatorArg"]).apply(
            lambda x: np.nansum(x * weights) / np.nansum(pd.notnull(x) * weights), raw=True).values[
               args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_sum(f, idt, iid, x, args):
    """滚动窗口求和"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).sum().values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_prod(f, idt, iid, x, args):
    """滚动窗口求积"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    Rslt = np.nanprod(Data, axis=0)
    Rslt[np.sum(pd.notnull(Data), axis=0) < args["OperatorArg"]["min_periods"]] = np.nan
    return Rslt


@rolling_operator()
def rolling_std(f, idt, iid, x, args, ddof=1):
    """滚动窗口标准差"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = {"ddof": OperatorArg.pop("ddof", 1)}
    return Data.rolling(**OperatorArg).apply(lambda x: np.nanstd(x, **SubOperatorArg), raw=True).values[
           args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_max(f, idt, iid, x, args):
    """滚动窗口最大值"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).max().values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_min(f, idt, iid, x, args):
    """滚动窗口最小值"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).min().values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_argmax(f, idt, iid, x, args):
    """滚动窗口最大值索引"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).apply(np.nanargmax).values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_argmin(f, idt, iid, x, args):
    """滚动窗口最小值索引"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).apply(np.nanargmin).values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_median(f, idt, iid, x, args):
    """滚动窗口中位数"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).median().values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_skew(f, idt, iid, x, args):
    """滚动窗口偏度"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).skew().values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_kurt(f, idt, iid, x, args):
    """滚动窗口峰度"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).kurt().values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_var(f, idt, iid, x, args, ddof=1):
    """滚动窗口方差"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = {"ddof": OperatorArg.pop("ddof", 1)}
    return Data.rolling(**OperatorArg).apply(lambda x: np.nanvar(x, **SubOperatorArg), raw=True).values[
           args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_quantile(f, idt, iid, x, args, quantile=0.5):
    """滚动窗口分位数"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = {"quantile": OperatorArg.pop("quantile", 0.5)}
    return Data.rolling(**OperatorArg).quantile(**SubOperatorArg).values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_count(f, idt, iid, x, args):
    """滚动窗口计数"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).count().values[args["OperatorArg"]["window"] - 1:]


@rolling_operator()
def rolling_change_rate(f, idt, iid, x, args):
    """滚动变化率"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    Numerator = Data[args["OperatorArg"]["window"] - 1:]
    Denominator = Data[:-args["OperatorArg"]["window"] + 1]
    Mask = (Denominator == 0) | np.isnan(Denominator)
    Rslt = np.full_like(Numerator, np.nan)
    ValidMask = ~Mask
    Rslt[ValidMask] = (Numerator[ValidMask] - Denominator[ValidMask]) / np.abs(Denominator[ValidMask])
    Rslt[Mask & (Numerator > 0)] = 1.0
    Rslt[Mask & (Numerator < 0)] = -1.0
    Rslt[Mask & (Numerator == 0)] = 0.0
    return Rslt


@rolling_operator()
def rolling_rank(f, idt, iid, x, args):
    """滚动排名"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).apply(lambda s: np.sort(s).searchsorted(s[-1])).values[
           args["OperatorArg"]["window"] - 1:]


@expanding_operator()
def expanding_mean(f, idt, iid, x, args):
    """扩展窗口均值"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.expanding(**args["OperatorArg"]).mean().values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_sum(f, idt, iid, x, args):
    """扩展窗口求和"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.expanding(**args["OperatorArg"]).sum().values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_std(f, idt, iid, x, args, ddof=1):
    """扩展窗口标准差"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = {"ddof": OperatorArg.pop("ddof", 1)}
    return Data.expanding(**OperatorArg).std(**SubOperatorArg).values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_max(f, idt, iid, x, args):
    """扩展窗口最大值"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.expanding(**args["OperatorArg"]).max().values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_min(f, idt, iid, x, args):
    """扩展窗口最小值"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.expanding(**args["OperatorArg"]).min().values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_median(f, idt, iid, x, args):
    """扩展窗口中位数"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.expanding(**args["OperatorArg"]).median().values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_skew(f, idt, iid, x, args):
    """扩展窗口偏度"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.expanding(**args["OperatorArg"]).skew().values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_kurt(f, idt, iid, x, args):
    """扩展窗口峰度"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.expanding(**args["OperatorArg"]).kurt().values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_var(f, idt, iid, x, args, ddof=1):
    """扩展窗口方差"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = {"ddof": OperatorArg.pop("ddof", 1)}
    return Data.expanding(**OperatorArg).var(**SubOperatorArg).values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_quantile(f, idt, iid, x, args, quantile=0.5):
    """扩展窗口分位数"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = {"quantile": OperatorArg.pop("quantile", 0.5)}
    return Data.expanding(**OperatorArg).quantile(**SubOperatorArg).values[args["OperatorArg"]["min_periods"] - 1:]


@expanding_operator()
def expanding_count(f, idt, iid, x, args):
    """扩展窗口计数"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.expanding(**args["OperatorArg"]).count().values[args["OperatorArg"]["min_periods"] - 1:]


@ewm_operator()
def ewm_mean(f, idt, iid, x, args):
    """指数加权移动平均"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.ewm(**args["OperatorArg"]).mean().values[args["OperatorArg"]["min_periods"] - 1:]


@ewm_operator(has_sub_args=True)
def ewm_std(f, idt, iid, x, args):
    """指数加权移动标准差"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return Data.ewm(**OperatorArg).std(**SubOperatorArg).values[args["OperatorArg"]["min_periods"] - 1:]


@ewm_operator(has_sub_args=True)
def ewm_var(f, idt, iid, x, args):
    """指数加权移动方差"""
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return Data.ewm(**OperatorArg).var(**SubOperatorArg).values[args["OperatorArg"]["min_periods"] - 1:]


@dual_factor_rolling_operator(has_sub_args=True)
def rolling_cov(f, idt, iid, x, args):
    """滚动协方差（双因子）"""
    Data1, Data2 = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return pd.DataFrame(Data1).rolling(**OperatorArg).cov(pd.DataFrame(Data2), **SubOperatorArg).values[
           args["OperatorArg"]["window"] - 1:]


@dual_factor_rolling_operator(dynamic_time_mode=True)
def rolling_corr(f, idt, iid, x, args):
    """滚动相关系数（双因子）"""
    Data1, Data2 = _genOperatorData(f, idt, iid, x, args)
    Method = args["OperatorArg"]["method"]
    if Method == "pearson":
        return pd.DataFrame(Data1).rolling(window=args["OperatorArg"]["window"],
                                           min_periods=args["OperatorArg"]["min_periods"],
                                           win_type=args["OperatorArg"]["win_type"]).corr(pd.DataFrame(Data2)).values[
               args["OperatorArg"]["window"] - 1:]
    Mask = np.sum(pd.notnull(Data1) & pd.notnull(Data2), axis=0)
    Rslt = pd.DataFrame(Data1).corrwith(pd.DataFrame(Data2), axis=0, drop=False, method=Method).values
    Rslt[Mask < args["OperatorArg"]["min_periods"]] = np.nan
    return Rslt


@multifactor_rolling_operator()
def rolling_regress(f, idt, iid, x, args):
    """滚动回归"""
    X = _genOperatorData(f, idt, iid, x, args)
    Y = X[0]
    if args["OperatorArg"]['constant']:
        X = np.array([np.ones(Y.shape)] + X[1:])
    else:
        X = np.array(X[1:])
    Window = args["OperatorArg"]['window']
    Weight = (0.5 ** (1 / args["OperatorArg"]['half_life'])) ** np.arange(Window)
    Weight = Weight[::-1] / np.sum(Weight)
    Rslt = np.empty((Y.shape[0] - Window + 1, Y.shape[1]), dtype="O")
    for i in range(Rslt.shape[0]):
        for j in range(Rslt.shape[1]):
            iY = Y[i:i + Window, j]
            iX = X[:, i:i + Window, j].T
            try:
                iRslt = sm.WLS(iY, iX, missing='drop').fit()
                Rslt[i, j] = tuple(iRslt.params) + tuple(iRslt.tvalues) + (
                    iRslt.fvalue, iRslt.rsquared, iRslt.rsquared_adj)
            except:
                Rslt[i, j] = (np.nan,) * int(X.shape[0] * 2 + 3)
    return Rslt


@point_rolling_operator()
def rolling_regress_change(f, idt, iid, x, args):
    """滚动回归斜率变化"""
    Y = _genOperatorData(f, idt, iid, x, args)[0]
    X = np.arange(Y.shape[0]).astype("float").reshape((Y.shape[0], 1)).repeat(Y.shape[1], axis=1)
    Mask = pd.isnull(Y)
    X[Mask] = np.nan
    X = X - np.nanmean(X, axis=0)
    Y = Y - np.nanmean(Y, axis=0)
    Rslt = np.nansum(X * Y, axis=0) / np.nansum(X ** 2, axis=0)
    Rslt[Y.shape[0] - np.sum(Mask) < args["OperatorArg"]["min_periods"]] = np.nan
    Rslt[np.isinf(Rslt)] = np.nan
    return Rslt


@dual_factor_expanding_operator(has_sub_args=True)
def expanding_cov(f, idt, iid, x, args):
    """扩展窗口协方差（双因子）"""
    Data1, Data2 = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return pd.DataFrame(Data1).expanding(**OperatorArg).cov(pd.DataFrame(Data2), **SubOperatorArg).values[
           args["OperatorArg"]["min_periods"] - 1:]


@dual_factor_expanding_operator()
def expanding_corr(f, idt, iid, x, args):
    """扩展窗口相关系数（双因子）"""
    Data1, Data2 = _genOperatorData(f, idt, iid, x, args)
    return pd.DataFrame(Data1).expanding(**args["OperatorArg"]).corr(pd.DataFrame(Data2)).values[
           args["OperatorArg"]["min_periods"] - 1:]


@ewm_operator(dual_factor=True, has_sub_args=True)
def ewm_cov(f, idt, iid, x, args):
    """指数加权移动协方差（双因子）"""
    Data1, Data2 = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    SubOperatorArg = OperatorArg.pop("SubOperatorArg", {})
    return pd.DataFrame(Data1).ewm(**OperatorArg).cov(pd.DataFrame(Data2), **SubOperatorArg).values[
           args["OperatorArg"]["min_periods"] - 1:]


@ewm_operator(dual_factor=True)
def ewm_corr(f, idt, iid, x, args):
    """指数加权移动相关系数（双因子）"""
    Data1, Data2 = _genOperatorData(f, idt, iid, x, args)
    return pd.DataFrame(Data1).ewm(**args["OperatorArg"]).corr(pd.DataFrame(Data2)).values[
           args["OperatorArg"]["min_periods"] - 1:]


@time_shift_operator(lookback_param="window")
def lag(f, idt, iid, x, args):
    """滞后算子"""
    if args["OperatorArg"]['dt_change_fun'] is None: return x[0][args["OperatorArg"]['window'] - args["OperatorArg"][
        'lag_period']:x[0].shape[0] - args["OperatorArg"]['lag_period']]
    TargetDTs = args["OperatorArg"]['dt_change_fun'](idt)
    Data = pd.DataFrame(x[0], index=idt)
    TargetData = Data.loc[TargetDTs].values
    TargetData[args["OperatorArg"]['lag_period']:] = TargetData[:-args["OperatorArg"]['lag_period']]
    if f.FactorDataType != "double":
        Data = pd.DataFrame(np.empty(Data.shape, dtype="O"), index=Data.index, columns=iid)
    else:
        Data = pd.DataFrame(index=Data.index, columns=iid, dtype="float64")
    Data.loc[TargetDTs] = TargetData
    return Data.fillna(method='ffill').values[args["OperatorArg"]['window']:]


@time_shift_operator(lookback_param="n")
def diff(f, idt, iid, x, args):
    """差分算子"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    return np.diff(Data, n=args["OperatorArg"]['n'], axis=0)


@fillna_operator()
def fillna(f, idt, iid, x, args):
    """缺失值填充"""
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    if args["OperatorArg"]["value"] is None:
        LookBack = args["OperatorArg"]["lookback"]
        return pd.DataFrame(Data).fillna(method="ffill", limit=LookBack).values[LookBack:]
    else:
        Data[pd.isnull(Data)] = args["OperatorArg"]["value"]
        return Data


@nav_operator()
def nav(f, idt, iid, x, args):
    """净值（NAV）计算"""
    Price = x[0]
    Return, = _genOperatorData(f, idt, iid, x[1:], args)
    if Price.shape[0] <= Return.shape[0]:
        NAV = np.nancumprod(Return + 1, axis=0)
    else:
        NAV = Price[-Return.shape[0] - 1, :] * np.nancumprod(Return + 1, axis=0)
    NAV[pd.isnull(Return)] = np.nan
    return NAV


# ----------------------单截面运算--------------------------------
@single_section_operator()
def standardizeZScore(f, idt, iid, x, args, avg_statistics="平均值", dispersion_statistics="标准差",
                      other_handle='填充None'):
    """Z-Score标准化"""
    Data = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd] == 1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData == 1:
        CatData = Data[StartInd]
        StartInd += 1
    elif CatData is not None:
        CatData = Data[StartInd:StartInd + CatData]
        StartInd += len(CatData)
        CatData = np.array(list(zip(*CatData)))
    AvgWeight = OperatorArg.pop("avg_weight")
    if AvgWeight is not None:
        AvgWeight = Data[StartInd]
        StartInd += 1
    DispersionWeight = OperatorArg.pop("dispersion_weight")
    if DispersionWeight is not None:
        DispersionWeight = Data[StartInd]
    Rslt = np.zeros(FactorData.shape) + np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.standardizeZScore(FactorData[i], mask=(Mask[i] if Mask is not None else None),
                                                         cat_data=(CatData[i].T if CatData is not None else None),
                                                         avg_weight=(AvgWeight[i] if AvgWeight is not None else None),
                                                         dispersion_weight=(DispersionWeight[
                                                                                i] if DispersionWeight is not None else None),
                                                         **OperatorArg)
    return Rslt


@single_section_operator()
def standardizeRank(f, idt, iid, x, args, ascending=True, uniformization=True, perturbation=False, offset=0.5,
                    other_handle='填充None'):
    """Rank标准化"""
    Data = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd] == 1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData == 1:
        CatData = Data[StartInd]
    elif CatData is not None:
        CatData = Data[StartInd:StartInd + CatData]
        CatData = np.array(list(zip(*CatData)))
    Rslt = np.zeros(FactorData.shape) + np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.standardizeRank(FactorData[i], mask=(Mask[i] if Mask is not None else None),
                                                       cat_data=(CatData[i].T if CatData is not None else None),
                                                       **OperatorArg)
    return Rslt


@single_section_operator()
def fillNaNByVal(f, idt, iid, x, args, fill_value=0, fill_method="常数"):
    """按值填充NaN"""
    Data = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd] == 1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData == 1:
        CatData = Data[StartInd]
    elif CatData is not None:
        CatData = Data[StartInd:StartInd + CatData]
        CatData = np.array(list(zip(*CatData)))
    Rslt = FactorData.copy()
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.fillNaNByValue(FactorData[i], mask=(Mask[i] if Mask is not None else None),
                                                      cat_data=(CatData[i].T if CatData is not None else None),
                                                      **OperatorArg)
    return Rslt


@single_section_operator()
def winsorize(f, idt, iid, x, args, winsorize_lower=0.01, winsorize_upper=0.01, fill_value=None, fill_method="均值方差",
             boundary_method="边界值", other_handle='填充None'):
    """Winsorize处理"""
    Data = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd] == 1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData == 1:
        CatData = Data[StartInd]
    elif CatData is not None:
        CatData = Data[StartInd:StartInd + CatData]
        CatData = np.array(list(zip(*CatData)))
    Rslt = FactorData.copy()
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.winsorize(FactorData[i], mask=(Mask[i] if Mask is not None else None),
                                                  cat_data=(CatData[i].T if CatData is not None else None),
                                                  **OperatorArg)
    return Rslt


@single_section_operator()
def standardizeQuantile(f, idt, iid, x, args, ascending=True, perturbation=False, other_handle='填充None'):
    """分位数标准化"""
    Data = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd] == 1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData == 1:
        CatData = Data[StartInd]
    elif CatData is not None:
        CatData = Data[StartInd:StartInd + CatData]
        CatData = np.array(list(zip(*CatData)))
    Rslt = np.zeros(FactorData.shape) + np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.standardizeQuantile(FactorData[i], mask=(Mask[i] if Mask is not None else None),
                                                           cat_data=(CatData[i].T if CatData is not None else None),
                                                           **OperatorArg)
    return Rslt


@single_section_operator()
def fillNaNByFun(f, idt, iid, x, args, val_fun="平均值"):
    """按函数值填充NaN"""
    Data = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd] == 1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData == 1:
        CatData = Data[StartInd]
    elif CatData is not None:
        CatData = Data[StartInd:StartInd + CatData]
        CatData = np.array(list(zip(*CatData)))
    ValFun = OperatorArg.pop("val_fun")
    if ValFun == "平均值":
        ValFun = (lambda x, n: np.zeros(n) + np.nanmean(x))
    elif ValFun == "中位数":
        ValFun = (lambda x, n: np.zeros(n) + np.nanmedian(x))
    elif ValFun == "最大值":
        ValFun = (lambda x, n: np.zeros(n) + np.nanmax(x))
    elif ValFun == "最小值":
        ValFun = (lambda x, n: np.zeros(n) + np.nanmin(x))
    elif ValFun == "高斯随机数":
        ValFun = (lambda x, n: np.random.randn(n) * np.nanstd(x) + np.nanmean(x))
    elif ValFun == "均匀随机数":
        ValFun = (lambda x, n: np.random.rand(n) * (np.nanmax(x) - np.nanmin(x)) + np.nanmin(x))
    Rslt = np.zeros(FactorData.shape) + np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.fillNaNByFun(FactorData[i], mask=(Mask[i] if Mask is not None else None),
                                                    cat_data=(CatData[i].T if CatData is not None else None),
                                                    val_fun=ValFun, **OperatorArg)
    return Rslt


@single_section_operator()
def fillNaNByRegress(f, idt, iid, x, args, intercept=True, weight_data=None, dummy_data=None):
    """按回归预测值填充NaN"""
    Data = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd] == 1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData == 1:
        CatData = Data[StartInd]
        StartInd += 1
    elif CatData is not None:
        CatData = Data[StartInd:StartInd + CatData]
        StartInd += len(CatData)
        CatData = np.array(list(zip(*CatData)))
    X = OperatorArg.pop("X")
    if X == 1:
        X = Data[StartInd]
    elif X is not None:
        X = Data[StartInd:StartInd + X]
    Rslt = np.zeros(FactorData.shape) + np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.fillNaNByRegress(FactorData[i], mask=(Mask[i] if Mask is not None else None),
                                                         cat_data=(CatData[i].T if CatData is not None else None),
                                                         X=(X[i].T if X is not None else None), **OperatorArg)
    return Rslt


@single_section_operator()
def orthogonalize(f, idt, iid, x, args, method="gram_schmidt", weight_data=None, dummy_data=None):
    """正交化处理"""
    Data = _genOperatorData(f, idt, iid, x, args)
    OperatorArg = args["OperatorArg"].copy()
    FactorData = Data[0]
    StartInd = 1
    Mask = OperatorArg.pop("mask")
    if Mask is not None:
        Mask = (Data[StartInd] == 1)
        StartInd += 1
    CatData = OperatorArg.pop("cat_data")
    if CatData == 1:
        CatData = Data[StartInd]
    elif CatData is not None:
        CatData = Data[StartInd:StartInd + CatData]
        CatData = np.array(list(zip(*CatData)))
    X = OperatorArg.pop("X")
    if X == 1:
        X = Data[StartInd]
    elif X is not None:
        X = Data[StartInd:StartInd + X]
    Rslt = np.zeros(FactorData.shape) + np.nan
    for i in range(FactorData.shape[0]):
        Rslt[i] = DataPreprocessingFun.orthogonalize(FactorData[i], mask=(Mask[i] if Mask is not None else None),
                                                      cat_data=(CatData[i].T if CatData is not None else None),
                                                      X=(X[i].T if X is not None else None), **OperatorArg)
    return Rslt


# ----------------------多截面运算--------------------------------
def _aggregate(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nID = len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    AggrFun = args["OperatorArg"]["aggr_fun"]
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nID,), fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                Rslt[i] = AggrFun(FactorData[iMask])
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                Rslt[iMask] = AggrFun(FactorData[iMask])
    else:
        Rslt = np.full(shape=(nID,), fill_value=AggrFun(FactorData[Mask]))
    return Rslt


def aggregate(f, aggr_fun=np.nansum, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
    Factors = [f]
    if mask is not None:
        Factors.append(mask)
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    Args["OperatorArg"] = {"aggr_fun": aggr_fun, "Mask": (mask is not None), "CatData": (cat_data is not None),
                           "SectionChged": (descriptor_ids is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors,
                            {"算子": _aggregate, "参数": Args, "运算时点": "单时点", "描述子截面": [descriptor_ids] * len(Descriptors)},
                            **kwargs)


def _disaggregate(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        for i, iID in enumerate(args["OperatorArg"]["aggr_ids"]):
            iMask = (CatData == iID)
            Rslt[iMask] = FactorData[:, [i]].repeat(nID, axis=1)[iMask]
    else:
        Rslt = FactorData.repeat(nID, axis=1)
    return Rslt


def disaggregate(f, aggr_ids, cat_data=None, disaggr_ids=None, **kwargs):  # 将聚合因子分解成为普通因子
    if (len(aggr_ids) > 1) and (cat_data is None): raise FactorError("解聚合算子 disaggregate: 缺少类别因子!")
    Factors = [f]
    if cat_data is not None:
        Factors.append(cat_data)
    Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
    DescriptorIDs = [aggr_ids] * Args.get("SepInd1", 0) + [disaggr_ids] * (len(Descriptors) - Args.get("SepInd1", 0))
    Args["OperatorArg"] = {"aggr_ids": aggr_ids, "CatData": (cat_data is not None)}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors,
                            {"算子": _disaggregate, "参数": Args, "运算时点": "多时点", "描述子截面": DescriptorIDs}, **kwargs)

_register_operator(OperatorCategory.MULTI_SECTION, aggregate, "aggregate")
_register_operator(OperatorCategory.MULTI_SECTION, disaggregate, "disaggregate")


def _make_aggr_func(name, op_func, extra_params=None):
    extra_params = extra_params or {}

    def aggr_func(f, mask=None, cat_data=None, descriptor_ids=None, **kwargs):
        Factors = [f]
        if mask is not None:
            Factors.append(mask)
        if cat_data is not None:
            Factors.append(cat_data)
        for k, v in extra_params.items():
            kwargs.setdefault(k, v)
        Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
        op_arg = {"Mask": (mask is not None), "CatData": (cat_data is not None),
                  "SectionChged": (descriptor_ids is not None)}
        for k, v in extra_params.items():
            if k in kwargs:
                op_arg[k] = kwargs.pop(k)
        Args["OperatorArg"] = op_arg
        FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
        return SectionOperation(FactorName, Descriptors,
                                 {"算子": op_func, "参数": Args, "运算时点": "多时点",
                                  "描述子截面": [descriptor_ids] * len(Descriptors)},
                                 **kwargs)

    aggr_func.__name__ = name
    _register_operator(OperatorCategory.MULTI_SECTION, aggr_func, name)
    return aggr_func


def _aggr_sum(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nansum(iData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            iData = np.full(shape=FactorData.shape, fill_value=np.nan)
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nansum(iData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nansum(FactorData * Mask, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt


aggr_sum = _make_aggr_func("aggr_sum", _aggr_sum)


def _aggr_prod(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanprod(iData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            iData = np.full(shape=FactorData.shape, fill_value=np.nan)
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanprod(iData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanprod(FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt


aggr_prod = _make_aggr_func("aggr_prod", _aggr_prod)


def _aggr_max(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if FactorData.shape[1] == 0: return np.full(shape=(nDT, nID), fill_value=np.nan)
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanmax(iData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanmax(iData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanmax(FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt


aggr_max = _make_aggr_func("aggr_max", _aggr_max)


def _aggr_min(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if FactorData.shape[1] == 0: return np.full(shape=(nDT, nID), fill_value=np.nan)
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanmin(iData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanmin(iData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanmin(FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt


aggr_min = _make_aggr_func("aggr_min", _aggr_min)


def _aggr_mean(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["Weight"]:
        WeightData = Data[2 - args["OperatorArg"]["Mask"]]
    else:
        WeightData = np.ones(FactorData.shape)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                Rslt[:, i] = np.nansum(iMask * WeightData * FactorData, axis=1) / np.nansum(iMask * WeightData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                Rslt[iMask] = \
                    (np.nansum(iMask * WeightData * FactorData, axis=1) / np.nansum(iMask * WeightData,
                                                                                    axis=1)).reshape(
                        (nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        Rslt = (np.nansum(FactorData * WeightData * Mask, axis=1) / np.nansum(WeightData * Mask, axis=1)).reshape(
            (nDT, 1)).repeat(nID, axis=1)
    return Rslt


def _make_aggr_mean_func(name, op_func):
    def aggr_func(f, mask=None, cat_data=None, weight_data=None, descriptor_ids=None, **kwargs):
        Factors = [f]
        if mask is not None:
            Factors.append(mask)
        if weight_data is not None:
            Factors.append(weight_data)
        if cat_data is not None:
            Factors.append(cat_data)
        Descriptors, Args = _genMultivariateOperatorInfo(*Factors)
        Args["OperatorArg"] = {"Mask": (mask is not None), "Weight": (weight_data is not None),
                               "CatData": (cat_data is not None), "SectionChged": (descriptor_ids is not None)}
        FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
        return SectionOperation(FactorName, Descriptors,
                                {"算子": op_func, "参数": Args, "运算时点": "多时点",
                                 "描述子截面": [descriptor_ids] * len(Descriptors)},
                                **kwargs)
    aggr_func.__name__ = name
    _register_operator(OperatorCategory.MULTI_SECTION, aggr_func, name)
    return aggr_func


aggr_mean = _make_aggr_mean_func("aggr_mean", _aggr_mean)


def _aggr_std(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanstd(iData, axis=1, ddof=args["OperatorArg"]["ddof"])
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = \
                    np.nanstd(iData, axis=1, ddof=args["OperatorArg"]["ddof"]).reshape((nDT, 1)).repeat(nID, axis=1)[
                        iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanstd(FactorData, axis=1, ddof=args["OperatorArg"]["ddof"]).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt


aggr_std = _make_aggr_func("aggr_std", _aggr_std, {"ddof": 1})


def _aggr_var(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanvar(iData, axis=1, ddof=args["OperatorArg"]["ddof"])
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = \
                    np.nanvar(iData, axis=1, ddof=args["OperatorArg"]["ddof"]).reshape((nDT, 1)).repeat(nID, axis=1)[
                        iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanvar(FactorData, axis=1, ddof=args["OperatorArg"]["ddof"]).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt


aggr_var = _make_aggr_func("aggr_var", _aggr_var, {"ddof": 1})


def _aggr_median(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanmedian(iData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = np.nanmedian(iData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanmedian(FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt


aggr_median = _make_aggr_func("aggr_median", _aggr_median)


def _aggr_quantile(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = Data[0]
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        iData = np.full(shape=FactorData.shape, fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[:, i] = np.nanpercentile(iData, q=args["OperatorArg"]["quantile"] * 100, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                iData[:] = np.nan
                iData[iMask] = FactorData[iMask]
                Rslt[iMask] = \
                    np.nanpercentile(iData, q=args["OperatorArg"]["quantile"] * 100, axis=1).reshape((nDT, 1)).repeat(
                        nID,
                        axis=1)[
                        iMask]
    else:
        FactorData[~Mask] = np.nan
        Rslt = np.nanpercentile(FactorData, q=args["OperatorArg"]["quantile"] * 100, axis=1).reshape((nDT, 1)).repeat(
            nID, axis=1)
    return Rslt


aggr_quantile = _make_aggr_func("aggr_quantile", _aggr_quantile, {"quantile": 0.5})


def _aggr_count(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    nDT, nID = len(idt), len(iid)
    FactorData = pd.notnull(Data[0])
    if args["OperatorArg"]["Mask"]:
        Mask = (Data[1] == 1)
    else:
        Mask = np.full(FactorData.shape, fill_value=True)
    if args["OperatorArg"]["CatData"]:
        CatData = Data[-1]
        Rslt = np.full(shape=(nDT, nID), fill_value=np.nan)
        if args["OperatorArg"]["SectionChged"]:
            for i, iID in enumerate(iid):
                iMask = ((CatData == iID) & Mask)
                Rslt[:, i] = np.nansum(iMask * FactorData, axis=1)
        else:
            AllCats = pd.unique(CatData.flatten())
            for i, iCat in enumerate(AllCats):
                if pd.isnull(iCat):
                    iMask = (pd.isnull(CatData) & Mask)
                else:
                    iMask = ((CatData == iCat) & Mask)
                Rslt[iMask] = np.nansum(iMask * FactorData, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)[iMask]
    else:
        Rslt = np.nansum(FactorData * Mask, axis=1).reshape((nDT, 1)).repeat(nID, axis=1)
    return Rslt


aggr_count = _make_aggr_func("aggr_count", _aggr_count)


def _merge(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)
    Rslt = np.concatenate(Data, axis=1)
    IDs = []
    [IDs.extend(iIDs) for iIDs in f.DescriptorSection]
    return pd.DataFrame(Rslt, columns=IDs).loc[:, iid].values


def merge(factors, descriptor_ids, **kwargs):
    if len(factors) != len(descriptor_ids): raise FactorError("描述子个数与描述子截面个数不一致!")
    Descriptors, Args = _genMultivariateOperatorInfo(*factors)
    DescriptorIDs = []
    for i in range(len(factors)):
        StartInd, EndInd = Args.get("SepInd" + str(i), 0), Args.get("SepInd" + str(i + 1), 0)
        DescriptorIDs += [descriptor_ids[i]] * (EndInd - StartInd)
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    return SectionOperation(FactorName, Descriptors, {"算子": _merge, "参数": Args, "运算时点": "多时点", "描述子截面": DescriptorIDs},
                            **kwargs)


def _chg_ids(f, idt, iid, x, args):
    Data = _genOperatorData(f, idt, iid, x, args)[0]
    IDMap = args["OperatorArg"]["id_map"]
    OldIDs = f.DescriptorSection[0]
    Rslt = np.full(shape=(len(idt), len(iid)), fill_value=np.nan, dtype=Data.dtype)
    for i, iID in enumerate(iid):
        iOldID = IDMap.get(iID, None)
        if iOldID not in OldIDs: continue
        Rslt[:, i] = Data[:, OldIDs.index(iOldID)]
    return Rslt


def chg_ids(f, old_ids, id_map={}, **kwargs):  # id_map: {新ID:旧ID}
    Descriptors, Args = _genMultivariateOperatorInfo(f)
    Args["OperatorArg"] = {"id_map": id_map}
    FactorName = kwargs.pop("factor_name", str(uuid.uuid1()))
    DataType = f.getMetaData(key="DataType")

    if DataType is None: DataType = "object"
    return SectionOperation(FactorName, Descriptors,
                            {"算子": _chg_ids, "参数": Args, "运算时点": "多时点", "描述子截面": [old_ids] * len(Descriptors),
                             "数据类型": DataType}, **kwargs)
