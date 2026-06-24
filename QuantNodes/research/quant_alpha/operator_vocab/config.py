# coding=utf-8
"""
OperatorVocab 配置 - 控制 namespace 构建与算子查询行为
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OperatorVocabConfig:
    """OperatorVocab 配置

    控制：
    - per-date over() 默认行为（cross_sectional）
    - 启用的算子类别
    - TA-Lib 是否自动加载
    - L0/L1/L2 注册表路径
    """

    # 截面算子默认行为：True=per-date over(date)，False=全局
    # 修复旧 12-lambda namespace 的 BUG 2（rank/zscore 全局错误）
    cross_sectional: bool = True

    # 启用的算子类别（默认全部）
    enabled_categories: List[str] = field(default_factory=lambda: [
        "point", "time", "section", "multi_section", "talib"
    ])

    # 是否自动加载 talib_ops（109 算子）
    # 默认 True；如 talib 未安装会自动降级为 False
    talib_enabled: bool = True

    # eval 沙箱的超时（秒），None 表示不限制
    eval_timeout_seconds: Optional[float] = None

    # 公式长度上限（字符），None 表示不限制
    # 防止超长公式 DoS
    max_formula_length: Optional[int] = 5000

    # 公式嵌套深度上限
    # 防止递归嵌套爆炸
    max_formula_depth: int = 20

    def __post_init__(self):
        # 验证 enabled_categories
        valid = {"point", "time", "section", "multi_section", "talib"}
        for cat in self.enabled_categories:
            if cat not in valid:
                raise ValueError(
                    f"Invalid category '{cat}'. Must be one of: {valid}"
                )
