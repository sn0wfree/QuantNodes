# coding=utf-8
"""
types/constants.py - 共享常量

从 llm/parser.py 提取，供 agent/workflows 和 research.quant_alpha 共同导入。
"""

ALLOWED_OPERATORS: set[str] = {
    # 时序
    "ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min", "ts_median",
    "ts_rank", "ts_zscore", "ts_skew", "ts_kurt",
    "ts_decay_linear", "ts_corr", "ts_cov", "ts_delay",
    # 截面
    "rank", "zscore", "winsorize", "IndNeutralize",
    # 一元
    "abs", "sign", "log", "sqrt", "signedpower",
    # 二元
    "add", "sub", "mul", "div", "greater", "less",
    # 时序位移
    "delta", "delay",
    # 复合（解析器展开）
    "returns",
    # polars 原生语法兼容
    "shift", "Ref",
}


__all__ = ["ALLOWED_OPERATORS"]
