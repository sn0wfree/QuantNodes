# coding=utf-8
"""回测配置数据类.

统一 v1-v7 的回测配置, 消除各版本重复的 Config 定义.

用法:
    from QuantNodes.strategy.momentum_etf_rotation.common.backtest_config import (
        BacktestConfig, CostConfig, VolTargeting, TrendFilter, StopLossConfig,
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostConfig:
    """交易成本配置.

    支持两种模式:
    - 结构化: commission_bp + slippage_bp * impact_factor (v1/v2/v6 用)
    - 平坦: flat_cost_bps (v3/v4 用)
    """
    enabled: bool = False
    commission_bp: float = 5.0       # 佣金 (基点)
    slippage_bp: float = 10.0        # 滑点 (基点)
    impact_factor: float = 0.1       # 冲击成本因子
    flat_cost_bps: float | None = None  # 平坦成本 (v3/v4 用)

    def cost_rate(self) -> float:
        """计算成本率."""
        if self.flat_cost_bps is not None:
            return self.flat_cost_bps / 10000
        return (self.commission_bp + self.slippage_bp * self.impact_factor) / 10000


@dataclass
class VolTargeting:
    """波动率目标配置 (v2/v6 用)."""
    enabled: bool = False
    target_vol: float = 0.15         # 目标年化波动率
    lookback: int = 60               # 波动率计算窗口 (天)
    min_scale: float = 0.3           # 最小缩放比例
    max_scale: float = 2.0           # 最大缩放比例


@dataclass
class TrendFilter:
    """趋势过滤器 (Stage 9-B): 基于基准指数均线的熊市减仓.

    启用时, 当 benchmark (默认沪深300) 跌破 ma_window 日均线时,
    整体仓位按 exposure_bear 缩放 (剩余仓位建议转债券).
    """
    enabled: bool = False
    benchmark_code: str = "510300"   # 沪深300
    ma_window: int = 200             # MA 窗口 (天)
    exposure_bull: float = 1.0       # 多头满仓
    exposure_bear: float = 0.5       # 熊市半仓
    bond_code: str = "511260"        # 国债 ETF (熊市配置)


@dataclass
class StopLossConfig:
    """止损配置 (v7 用, DD-based + cooldown)."""
    enabled: bool = False
    threshold: float = -0.10         # 回撤阈值 (如 -0.10 = -10%)
    cooldown_weeks: int = 5          # 冷却期 (周)


@dataclass
class BacktestConfig:
    """统一回测配置.

    通过不同参数组合, 表达 v1-v7 所有版本的配置:

    v1: rebal_freq="M", top_n=10, cost.enabled=False
    v2: rebal_freq="M", top_n=10, cost.enabled=True, vol_targeting.enabled=True, trend_filter.enabled=True
    v6: rebal_freq="M", top_n=10, cost.enabled=True, vol_targeting.enabled=True, trend_filter.enabled=True, execution_lag=1
    v7: rebal_freq="W", min_history=52, top_n=10, cost.enabled=True, stop_loss.enabled=True
    """
    # 调仓
    rebal_freq: str = "M"            # M=月度, W=周度, Q=季度
    min_history: int = 252           # 预热期 (日频=252, 周频=52)

    # 选择
    top_n: int = 10                  # 选择资产数
    max_weight: float = 0.25         # 最大权重

    # 权重
    weight_method: str = "inverse_vol"  # inverse_vol / equal
    vol_window: int = 60             # 波动率窗口 (天)
    vol_floor: float = 0.01          # 波动率下限

    # 成本
    cost: CostConfig = field(default_factory=CostConfig)

    # 风控
    vol_targeting: VolTargeting = field(default_factory=VolTargeting)
    trend_filter: TrendFilter = field(default_factory=TrendFilter)
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)

    # 执行
    execution_lag: int = 0           # 0=当日, 1=T+1 (v5.1/v6)

    # 输出
    return_detail: bool = False      # True: 返回权重历史等详情
