# coding=utf-8
"""
philosophy.py - Alpha 158/360 特征设计哲学 + 4 类模板

vs 直接移植 158/360 公式：本文件**仅做借鉴**，提取：
- 4 类特征的设计哲学（KBAR / Price / Volume / Rolling）
- 每类的公式模板（参数化）
- 类别间的依赖关系

参考：
- Yang, X. et al. (2020). "Qlib." arXiv:2009.11189
- qlib.contrib.data.handler.Alpha158/Alpha360
- WeChat article: 四大量化因子库
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# 默认窗口
DEFAULT_WINDOWS = [5, 10, 20, 30, 60]

# Alpha 360 默认 lookback 范围
ALPHA360_LOOKBACK_RANGE = list(range(60))  # 0-59


@dataclass
class CategoryTemplate:
    """特征类别的设计模板

    Attributes:
        name: 类别名
        total_features: 该类别的特征总数（Alpha 158 子集）
        philosophy: 设计哲学描述
        formula_template: 公式模板（含 {field} / {window} 等占位符）
        parameters: 可调参数
        examples: 几个公式示例
        category_id: 类别 ID
    """
    category_id: str  # KBAR / Price / Volume / Rolling
    name: str
    total_features: int
    philosophy: str
    formula_template: str
    parameters: Dict[str, List[Any]] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 4 类特征模板（Alpha 158）
# ==============================================================================

FEATURE_CATEGORIES: List[CategoryTemplate] = [
    CategoryTemplate(
        category_id="KBAR",
        name="K线形态",
        total_features=9,
        philosophy=(
            "KBAR 特征捕捉单根 K 线的几何信息（实体长度、上下影线、价格重心）。"
            "所有 KBAR 特征都做归一化（除以 open 或 high-low 振幅），"
            "消除价格水平影响。"
        ),
        formula_template=(
            # 单根 K 线的几何特征
            "({expr}) / {denominator}"
        ),
        parameters={
            "denominator": ["open", "$close - $open + 1e-12", "high - low + 1e-12"],
        },
        examples=[
            "KMID = (close - open) / open",  # 实体占比
            "KLEN = (high - low) / open",  # 振幅
            "KMID2 = (close - open) / ((high - low) + 1e-12)",  # 实体占振幅
            "KSFT = (2 * close - high - low) / open",  # 重心偏移
            "KUP = (high - max(open, close)) / open",  # 上影线
            "KLOW = (min(open, close) - low) / open",  # 下影线
        ],
        metadata={"qlib": "KBAR", "factors": 9},
    ),
    CategoryTemplate(
        category_id="Price",
        name="价格时序",
        total_features=20,
        philosophy=(
            "Price 特征是历史价格相对当前价格的归一化值。"
            "形式：Ref($field, d) / $close，field ∈ {OPEN, HIGH, LOW, VWAP}，"
            "d ∈ {0, 1, 2, 3, 4}。4 字段 × 5 延迟 = 20 特征。"
            "无截面算子（pure time-series）。"
        ),
        formula_template="Ref({field}, {delay}) / $close",
        parameters={
            "field": ["open", "high", "low", "vwap"],
            "delay": [0, 1, 2, 3, 4],
        },
        examples=[
            "OPEN0 = open / close",  # 今日开盘 / 今日收盘
            "OPEN1 = open.shift(1) / close",  # 昨日开盘 / 今日收盘
            "HIGH0 = high / close",
            "LOW0 = low / close",
            "VWAP0 = vwap / close",
        ],
        metadata={"qlib": "Price", "factors": 20},
    ),
    CategoryTemplate(
        category_id="Volume",
        name="成交量时序",
        total_features=5,
        philosophy=(
            "Volume 特征是历史成交量相对当前成交量的比值。"
            "形式：Ref($volume, d) / ($volume + 1e-12)，d ∈ {0, 1, 2, 3, 4}。"
            "捕捉量能变化：d=0 是今日成交；d=1-4 是历史成交。"
        ),
        formula_template="Ref({field}, {delay}) / ({field} + 1e-12)",
        parameters={
            "field": ["volume"],
            "delay": [0, 1, 2, 3, 4],
        },
        examples=[
            "VOL0 = volume / (volume + 1e-12)",  # 今日量 / 今日量 ≈ 1
            "VOL1 = volume.shift(1) / (volume + 1e-12)",  # 昨日量 / 今日量
            "VOL2 = volume.shift(2) / (volume + 1e-12)",
        ],
        metadata={"qlib": "Volume", "factors": 5},
    ),
    CategoryTemplate(
        category_id="Rolling",
        name="滚动统计",
        total_features=124,
        philosophy=(
            "Rolling 特征是 25 种统计指标 × 5 个时间窗口（5/10/20/30/60）的笛卡尔积。"
            "统计指标：ROC, MA, STD, BETA(Slope), RSQR, RESI, MAX, MIN, QTLU, QTLD, "
            "RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, SUMP, SUMN, "
            "VMA, VSTD, WVMA, VSUMP, VSUMN。"
            "这些特征可同时喂给 LightGBM/XGBoost/LSTM 等 ML 模型。"
        ),
        formula_template=(
            "rolling_{op}({field}, {window})"
        ),
        parameters={
            "op": [
                "roc",      # 变化率: close / lag(close, w) - 1
                "ma",       # 移动平均
                "std",      # 标准差
                "beta",     # 斜率（回归系数）
                "rsqr",     # R²
                "resi",     # 残差
                "max",      # 最大值
                "min",      # 最小值
                "qtlu",     # 上分位数（0.7）
                "qtld",     # 下分位数（0.3）
                "rank",     # 滚动 rank
                "rsv",      # RSV (类 KDJ 随机指标)
                "imax",     # argmax 位置
                "imin",     # argmin 位置
                "imxd",     # argmax - argmin
                "corr",     # 相关（双字段）
                "cntp",     # count positive
                "cntn",     # count negative
                "sump",     # sum positive
                "sumn",     # sum negative
                "vma",      # 量能 MA
                "vstd",     # 量能 std
                "wvma",     # weighted VMA
                "vsump",    # 量能 sum positive
                "vsumn",    # 量能 sum negative
            ],
            "window": [5, 10, 20, 30, 60],
            "field": ["close", "open", "high", "low", "vwap", "volume"],
        },
        examples=[
            "ROC(close, 5) = close / close.shift(5) - 1",
            "MA(close, 20) = close.rolling(20).mean()",
            "STD(close, 20) = close.rolling(20).std()",
            "BETA(close, volume, 20) = rolling_beta(close, volume, 20)",
            "CORR(close, volume, 20) = rolling_corr(close, volume, 20)",
            "CNTP(close, 20) = count(close > close.shift(1), 20)",
            "IMAX(high, 30) = argmax(high, 30)",
        ],
        metadata={"qlib": "Rolling", "factors": 124, "ops": 25, "windows": 5},
    ),
]


# ==============================================================================
# Alpha 360 模板
# ==============================================================================


@dataclass
class Alpha360Template:
    """Alpha 360 模板（6 字段 × 60 lookback = 360 特征）"""
    fields: List[str] = field(default_factory=lambda: ["close", "open", "high", "low", "vwap", "volume"])
    lookback_range: List[int] = field(default_factory=lambda: list(range(60)))
    total_features: int = 360
    philosophy: str = (
        "Alpha 360 把 6 个原始字段在 60 个 lookback 时间步上的值作为特征。"
        "价格字段除以当日 close 归一化，成交量字段除以当日成交量归一化。"
        "形成 (60, 6) 的二维矩阵，天然适合序列深度学习模型（GRU/LSTM/Transformer）。"
    )

    def formula_template(self) -> str:
        """公式模板"""
        return "Ref({field}, {delay}) / {denominator}"


ALPHA360_TEMPLATE = Alpha360Template()


# ==============================================================================
# 辅助函数
# ==============================================================================


def get_template_by_category(category_id: str) -> Optional[CategoryTemplate]:
    """按 category_id 查模板"""
    for t in FEATURE_CATEGORIES:
        if t.category_id == category_id:
            return t
    return None


def get_template_by_name(name: str) -> Optional[CategoryTemplate]:
    """按 name 查模板"""
    for t in FEATURE_CATEGORIES:
        if t.name == name:
            return t
    return None


def list_categories() -> List[str]:
    """列出所有 category_id"""
    return [t.category_id for t in FEATURE_CATEGORIES]


def total_feature_count() -> int:
    """Alpha 158 总特征数（=158）"""
    return sum(t.total_features for t in FEATURE_CATEGORIES)
