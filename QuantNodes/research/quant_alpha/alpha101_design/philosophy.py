# coding=utf-8
"""
philosophy.py - Alpha 101 设计哲学 + 核心算子

vs 直接移植 101 公式：本文件**仅做借鉴**，提取：
- 8 条设计哲学
- 10-20 个核心算子（含经济意义）
- A 股可移植性矩阵

参考：
- Kakushadze, Z. (2015). "101 Formulaic Alphas." arXiv:1601.00991
- WeChat article: 四大量化因子库
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DesignPrinciple:
    """Alpha 101 设计原则"""
    id: str  # 原则 ID (P1-P8)
    name: str  # 名称
    description: str  # 描述
    examples: List[str] = None  # 公式示例

    def __post_init__(self):
        if self.examples is None:
            self.examples = []


# ==============================================================================
# 8 条设计原则
# ==============================================================================

DESIGN_PHILOSOPHY: List[DesignPrinciple] = [
    DesignPrinciple(
        id="P1",
        name="数学即代码",
        description=(
            "每个 alpha 公式同时是数学表达式和可执行代码。"
            "一旦定义函数、算子、输入数据，公式即可直接运行。"
            "无需额外的解释层或中间表示。"
        ),
        examples=[
            "rank(ts_argmax(SignedPower(close, 2.), 5.)) - 0.5",
            "(close - open) / ((high - low) + 0.001)",
        ],
    ),
    DesignPrinciple(
        id="P2",
        name="动量 vs 反转",
        description=(
            "alpha 公式粗略分两类：动量（趋势跟随）和反转（均值回归）。"
            "动量：信号方向与历史收益方向一致；"
            "反转：信号方向与历史收益方向相反。"
        ),
        examples=[
            # 动量
            "close / ts_lag(close, 5) - 1",  # 5日动量
            # 反转
            "-1 * correlation(open, volume, 10)",  # 量价背离
        ],
    ),
    DesignPrinciple(
        id="P3",
        name="截面 rank",
        description=(
            "几乎所有 alpha 都使用截面 rank() 而非原始值。"
            "rank 把不同价格水平的股票放到同一尺度，"
            "消除市值/价格 bias。"
        ),
        examples=[
            "rank(close)",  # 截面 rank
            "rank(ts_std(returns, 20))",  # 排名波动率
        ],
    ),
    DesignPrinciple(
        id="P4",
        name="ts_argmax/min 提取极值位置",
        description=(
            "ts_argmax(x, d) 和 ts_argmin(x, d) 提取过去 d 天内极值出现的位置。"
            "极值位置常用于捕捉时间模式（如：5 天前是否创新高）。"
        ),
        examples=[
            "ts_argmax(close, 5)",  # 5 天内最高价出现日距今天数
            "ts_argmin(volume, 20)",  # 20 天内最低成交量出现日
        ],
    ),
    DesignPrinciple(
        id="P5",
        name="signedpower 保留符号",
        description=(
            "signedpower(x, a) = sign(x) * |x|^a 保留正负号的同时应用幂函数。"
            "vs 简单 pow(x, a) 会让负数变成 NaN/复杂数。"
            "Alpha 101 #1/#21 等大量使用 signedpower。"
        ),
        examples=[
            "SignedPower(((close < low) ? stddev(returns, 20) : close), 2.)",
            "signedpower(delta(close, 1), 0.5)",  # sqrt with sign
        ],
    ),
    DesignPrinciple(
        id="P6",
        name="decay_linear 加权",
        description=(
            "decay_linear(x, d) 用线性权重 [1,2,...,d]/d 加权求和，"
            "比简单 ts_mean(x, d) 给予近期值更高权重。"
            "捕捉动量/反转的时序模式。"
        ),
        examples=[
            "rank(decay_linear(volume, 9))",  # 9 日线性加权成交量
            "decay_linear(correlation(close, volume, 10), 5)",
        ],
    ),
    DesignPrinciple(
        id="P7",
        name="三元条件 ? : 表达分支逻辑",
        description=(
            "Alpha 101 大量使用 (cond) ? a : b 三元表达式。"
            "在 polars 中用 pl.when(cond).then(a).otherwise(b) 表达。"
            "比 if-else 嵌套更简洁。"
        ),
        examples=[
            "(close < low) ? stddev(returns, 20) : close",  # 异常时用波动率代替
        ],
    ),
    DesignPrinciple(
        id="P8",
        name="行业中性化 IndNeutralize",
        description=(
            "IndNeutralize(x, ind_class) 把因子值在每个行业内去均值，"
            "消除行业 bias。Alpha 101 假设 IndClass 已知；"
            "QuantNodes 中 ind_class = 'citic_1' (中信一级行业)。"
        ),
        examples=[
            "rank(IndNeutralize(close, industry))",
            "IndNeutralize(decay_linear(correlation(close, volume, 5), 5), industry)",
        ],
    ),
]


# ==============================================================================
# 10-20 个核心算子（含经济意义）
# ==============================================================================


@dataclass
class CoreOperator:
    """核心算子 + 经济意义"""
    name: str  # 算子名
    category: str  # 类别：time/section/point/composite
    economic_meaning: str  # 经济意义
    complexity: int = 1  # 1=简单 2=中等 3=复杂
    alpha101_examples: List[str] = None  # Alpha 101 公式示例

    def __post_init__(self):
        if self.alpha101_examples is None:
            self.alpha101_examples = []


CORE_OPERATORS: List[CoreOperator] = [
    # ============ 截面算子（3 个）============
    CoreOperator(
        name="rank",
        category="section",
        economic_meaning=(
            "截面 rank：把不同价格/市值的股票放到同一尺度。"
            "消除市值/价格 bias，捕捉相对位置。"
        ),
        complexity=1,
        alpha101_examples=[
            "rank(close)",
            "rank(ts_mean(volume, 20))",
        ],
    ),
    CoreOperator(
        name="zscore",
        category="section",
        economic_meaning=(
            "截面 z-score：标准化到 (mean, std) 区间。"
            "比 rank 更精确（保留距离信息）。"
        ),
        complexity=1,
        alpha101_examples=[
            "zscore(close)",
        ],
    ),
    CoreOperator(
        name="IndNeutralize",
        category="section",
        economic_meaning=(
            "行业中性化：在每个行业内去均值。"
            "消除行业 beta，捕捉行业内 alpha。"
        ),
        complexity=2,
        alpha101_examples=[
            "rank(IndNeutralize(close, industry))",
        ],
    ),

    # ============ 时序算子（8 个）============
    CoreOperator(
        name="ts_mean",
        category="time",
        economic_meaning=(
            "滚动均值：平滑短期波动，提取趋势。"
        ),
        complexity=1,
        alpha101_examples=[
            "ts_mean(close, 20)",
        ],
    ),
    CoreOperator(
        name="ts_std",
        category="time",
        economic_meaning=(
            "滚动标准差：衡量波动率。"
            "高波动率常预示反转或突破。"
        ),
        complexity=1,
        alpha101_examples=[
            "ts_std(returns, 20)",
        ],
    ),
    CoreOperator(
        name="ts_argmax",
        category="time",
        economic_meaning=(
            "滚动 argmax：过去 d 天内极值出现的位置。"
            "位置信息常用于捕捉时间模式。"
        ),
        complexity=2,
        alpha101_examples=[
            "ts_argmax(close, 5)",
        ],
    ),
    CoreOperator(
        name="ts_argmin",
        category="time",
        economic_meaning=(
            "滚动 argmin：过去 d 天内最低值位置。"
        ),
        complexity=2,
        alpha101_examples=[
            "ts_argmin(volume, 20)",
        ],
    ),
    CoreOperator(
        name="ts_delta",
        category="time",
        economic_meaning=(
            "差分：x - lag(x, d)。"
            "等价于 d 期间的变化量。"
        ),
        complexity=1,
        alpha101_examples=[
            "ts_delta(close, 1)",
            "sign(ts_delta(volume, 1)) * (-1 * ts_delta(close, 1))",
        ],
    ),
    CoreOperator(
        name="ts_rank",
        category="time",
        economic_meaning=(
            "滚动 rank：x 在过去 d 天中的排名位置（0-1）。"
            "比 ts_argmax 更精细（保留相对位置）。"
        ),
        complexity=2,
        alpha101_examples=[
            "ts_rank(delta(close, 7), 5)",
        ],
    ),
    CoreOperator(
        name="ts_corr",
        category="time",
        economic_meaning=(
            "滚动相关系数：x 和 y 在过去 d 天的相关性。"
            "捕捉量价关系、跨资产关系。"
        ),
        complexity=2,
        alpha101_examples=[
            "-1 * correlation(open, volume, 10)",
        ],
    ),
    CoreOperator(
        name="decay_linear",
        category="time",
        economic_meaning=(
            "线性衰减加权：近期权重高，远期权重低。"
            "比 ts_mean 更敏感于最新数据。"
        ),
        complexity=2,
        alpha101_examples=[
            "rank(decay_linear(volume, 9))",
        ],
    ),

    # ============ 点算子（3 个）============
    CoreOperator(
        name="signedpower",
        category="point",
        economic_meaning=(
            "带符号幂：sign(x) * |x|^a。"
            "Alpha 101 关键算子，应用幂变换且保留正负号。"
        ),
        complexity=2,
        alpha101_examples=[
            "SignedPower(((close < low) ? stddev(returns, 20) : close), 2.)",
        ],
    ),
    CoreOperator(
        name="sign",
        category="point",
        economic_meaning=(
            "符号函数：返回 -1/0/1。"
            "用于捕捉方向（涨跌/正负）。"
        ),
        complexity=1,
        alpha101_examples=[
            "sign(ts_delta(volume, 1))",
        ],
    ),
    CoreOperator(
        name="log",
        category="point",
        economic_meaning=(
            "自然对数：压缩大值，保留小值细节。"
            "对价格/收益率常做对数变换。"
        ),
        complexity=1,
        alpha101_examples=[
            "log(volume)",
        ],
    ),

    # ============ 组合算子（2 个）============
    CoreOperator(
        name="correlation",
        category="composite",
        economic_meaning=(
            "等同于 ts_corr：滚动相关系数。"
            "（composite DAG 形式，更易组合）"
        ),
        complexity=2,
        alpha101_examples=[
            "decay_linear(correlation(close, volume, 5), 5)",
        ],
    ),
    CoreOperator(
        name="stddev",
        category="composite",
        economic_meaning=(
            "标准差（同 ts_std）。"
        ),
        complexity=1,
        alpha101_examples=[
            "stddev(returns, 20)",
        ],
    ),
]


# ==============================================================================
# A 股可移植性矩阵
# ==============================================================================


@dataclass
class AShareCompatibility:
    """A 股可移植性记录"""
    alpha_id: str  # Alpha 101 #1-#101
    name: str  # 简短名称
    formula: str  # 公式
    a_share_compatible: bool  # 是否适用 A 股
    reason: str  # 原因（适用/不适用）
    adaptation: Optional[str] = None  # A 股适配建议


A_SHARE_COMPATIBILITY: List[AShareCompatibility] = [
    AShareCompatibility(
        alpha_id="#42",
        name="Delay-0 多空组合",
        formula="(-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)",
        a_share_compatible=False,
        reason="Delay-0 当日交易：A 股 T+1 制度不可行",
        adaptation="改为 Delay-1（次日开盘价交易）",
    ),
    AShareCompatibility(
        alpha_id="#48",
        name="Delay-0 极值比",
        formula="(-1 * (rank(correlation(rank(close), rank(volume), 5)) - rank(close + volume - adv20)))",
        a_share_compatible=False,
        reason="Delay-0 当日交易：A 股 T+1 制度不可行",
        adaptation="改为 Delay-1",
    ),
    AShareCompatibility(
        alpha_id="#53",
        name="Delay-0 价格位置",
        formula="-1 * delta((((close - low) - (high - close)) / (close - low)), 9)",
        a_share_compatible=False,
        reason="Delay-0 当日交易 + 除零风险（涨跌停 close=low）",
        adaptation="加 + 0.001 epsilon + 改为 Delay-1",
    ),
    AShareCompatibility(
        alpha_id="#54",
        name="Delay-0 量价背离",
        formula="(-1 * (low - close) * (open^5) * ((close - high) / (close - low)) * (high - close))",
        a_share_compatible=False,
        reason="Delay-0 + 除零 + 极端值",
        adaptation="需大幅改造，不推荐",
    ),
    AShareCompatibility(
        alpha_id="#1",
        name="复杂反转波动率",
        formula="rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5.)) - 0.5",
        a_share_compatible=True,
        reason="Delay-1，可直接用 close/returns 输入",
        adaptation=None,
    ),
    AShareCompatibility(
        alpha_id="#6",
        name="简单量价相关",
        formula="-1 * correlation(open, volume, 10)",
        a_share_compatible=True,
        reason="经典量价因子，跨市场适用",
        adaptation=None,
    ),
    AShareCompatibility(
        alpha_id="#12",
        name="量价反转",
        formula="sign(delta(volume, 1)) * (-1 * delta(close, 1))",
        a_share_compatible=True,
        reason="量增价跌 = 卖出信号，反之亦然",
        adaptation=None,
    ),
    AShareCompatibility(
        alpha_id="#101",
        name="日内动量",
        formula="(close - open) / ((high - low) + 0.001)",
        a_share_compatible=True,
        reason="标准日内动量因子（已加 0.001 epsilon 防除零）",
        adaptation=None,
    ),
]


# ==============================================================================
# 辅助函数
# ==============================================================================


def get_philosophy_by_id(pid: str) -> Optional[DesignPrinciple]:
    """按 ID 查设计原则"""
    for p in DESIGN_PHILOSOPHY:
        if p.id == pid:
            return p
    return None


def get_operator_by_name(name: str) -> Optional[CoreOperator]:
    """按名称查核心算子"""
    for op in CORE_OPERATORS:
        if op.name == name:
            return op
    return None


def get_a_share_compatible_count() -> int:
    """A 股可移植的 alpha 数量"""
    return sum(1 for x in A_SHARE_COMPATIBILITY if x.a_share_compatible)
