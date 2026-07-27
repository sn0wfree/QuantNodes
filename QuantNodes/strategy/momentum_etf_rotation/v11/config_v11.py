# coding=utf-8
"""v10 配置中心 — 所有可调参数集中管理.

基于 docs/57-v10_final_design.md 用户决策.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MacroLayerConfig:
    """Layer 1: 宏观择时配置.

    5 宏观因子 z-score + 熵权法合成综合得分 + TV-PR 可选.
    """
    # 启用开关
    enabled: bool = True

    # 5 宏观因子方向 (复用于 v9 citic_macro)
    factor_signs: dict[str, int] = field(default_factory=lambda: {
        '宏观增长因子': +1,        # 增长↑ → 风险资产偏好
        '宏观通胀因子_生活端': 0,   # 通胀对股债方向相反, 不参与
        '信用利差因子': -1,        # 利差↑ → 流动性收紧
        '宏观汇率因子': -1,        # 人民币↑ → 利好股
        '期限利差因子_债': +1,    # 期限走阔 → 利好股
    })

    # 熵权法窗口 (周, 默认 104 = 2 年)
    entropy_window: int = 104

    # 滚动 z-score 窗口 (周, 默认 52)
    zscore_window: int = 52

    # TV-PR (可选, 用户决策 #1: 必加可配置)
    # use_tvpr=True: TV-PR 与熵权混合, 权重 tvpr_weight
    # use_tvpr=False: 仅用熵权 (Layer 1 跳过 TV-PR, 节省计算)
    use_tvpr: bool = True
    tvpr_lambda_tv: float = 1.0
    tvpr_lambda_l1: float = 0.5
    tvpr_min_history: int = 104
    # tvpr_weight: TV-PR vs 熵权 混合权重 (Stage 30 调优最优)
    # - 0.0 = 仅熵权
    # - 1.0 = 仅 TV-PR
    # - 0.5 = 50/50 混合 (跨窗口平均 Sharpe 0.873, 最优)
    tvpr_weight: float = 0.5

    # Regime 阈值
    bull_threshold: float = 0.5
    bear_threshold: float = -0.5


@dataclass
class IndustryLayerConfig:
    """Layer 2A: 行业轮动配置.

    动量 + 反向波动率打分 + regime 条件.
    """
    enabled: bool = True

    # 因子窗口
    momentum_lookback: int = 52    # 动量 12-1 月
    momentum_skip: int = 4
    volatility_lookback: int = 26

    # Top-K (行业类别中, 选几个最优)
    top_k: int = 5

    # Top-K 行业 5x 加权, 其他 0.5x (沿用 v9 中信行业轮动)
    sector_mult: float = 5.0
    sector_floor_mult: float = 0.5

    # Regime 条件
    regime_enabled: bool = True
    bull_offensive_boost: float = 1.5
    bear_defensive_boost: float = 1.5

    # 相关性约束 (Stage 30 实证无效, 默认关闭)
    corr_constraint: bool = False
    corr_threshold: float = 0.7
    corr_window: int = 52


@dataclass
class StyleLayerConfig:
    """Layer 2B: 风格轮动配置 (IC 驱动).

    复用 v4 factor_timing_v4.py 的 IC 驱动逻辑.
    """
    enabled: bool = True

    # IC 窗口 (周)
    ic_lookback: int = 60
    ic_step: int = 5
    ic_base: float = 0.05
    ic_power: float = 2.0
    ic_threshold: float = 0.05

    # 因子特异性 forward_window (来自 v4 factor_timing_v4 Stage 18)
    factor_fw: dict[str, int] = field(default_factory=lambda: {
        'momentum': 120,
        'value': 40,
        'reversal': 60,
        'quality': 252,
        'size': 60,
        'low_vol': 60,
    })

    # 因子特异性 lag 平滑
    factor_smooth_window: dict[str, int] = field(default_factory=lambda: {
        'momentum': 4,
        'value': 4,
        'reversal': 1,
        'quality': 4,
        'size': 4,
        'low_vol': 4,
    })

    # 权重上下限
    min_weight: float = 0.10
    max_weight: float = 0.50

    # Regime 条件
    regime_enabled: bool = True
    bull_momentum_boost: float = 1.3
    bear_value_boost: float = 1.5
    bear_quality_boost: float = 1.5


@dataclass
class FactorLayerConfig:
    """Layer 2C: 因子选股配置 (5 因子 + K=10).

    复用 v9 citic_multifactor.py 的 5 风格因子.
    用户决策 #2: K=10 (不是 v4 Stage 30 的 K=3).
    """
    enabled: bool = True

    # 5 因子窗口 (与 v9 citic_multifactor 一致)
    lookback_mom: int = 52
    skip_mom: int = 4
    lookback_vol: int = 26
    lookback_qual: int = 26
    lookback_size: int = 4
    lookback_value: int = 104

    # 因子权重 (与 v9 中信多因子一致)
    factor_weights: dict[str, float] = field(default_factory=lambda: {
        'momentum': 1.0,
        'volatility': 1.0,
        'quality': 1.0,
        'size': 1.0,
        'value_reversal': 1.0,
    })

    # Top-K (用户决策 #2: K=10)
    top_k: int = 10

    # 候选池权重
    candidate_pool_weight: float = 0.50

    # Softmax temperature
    temperature: float = 1.0

    # 单 ETF 上下限
    floor: float = 0.005
    cap: float = 0.15

    # Regime 条件
    regime_enabled: bool = True
    bull_momentum_boost: float = 1.3
    bear_quality_boost: float = 1.5
    bear_low_vol_boost: float = 1.5


@dataclass
class RiskLayerConfig:
    """Layer 3: 风险控制 (Jump Model) 配置.

    复用 v8 jump_model_periodic_retrain.
    """
    enabled: bool = True

    # Jump Model 参数 (沿用 v8 权益类参数)
    asset_type: str = 'equity'
    jump_penalty: float | None = None    # None = 用 asset_type 默认
    train_window: int | None = None      # 1000 天
    retrain_every: int | None = None     # 30 天
    n_iter: int = 10
    n_restarts: int = 10

    # bear 概率平滑窗口
    bear_prob_window: int = 60

    # 是否使用指数衰减特征
    use_exp_features: bool | None = None


@dataclass
class PositionLayerConfig:
    """Layer 4: 动态仓位配置.

    pos = (0.7 - 0.5 * z_score).clip(0.2, 1.0)
    pos *= (1 - bear_prob × 0.5)  # Jump Model 调整
    """
    enabled: bool = True

    # pos 公式参数 (沿用 v9 银河方案)
    pos_intercept: float = 0.7
    pos_z_coef: float = 0.5
    pos_min: float = 0.2
    pos_max: float = 1.0

    # Jump Model 仓位调整 (用户决策 #4: 需要)
    use_bear_prob_adjustment: bool = True
    bear_prob_adjustment_coef: float = 0.5

    # z_score 合成权重 (Layer 1 macro + Layer 2 sector + Layer 2B style)
    z_score_weights: dict[str, float] = field(default_factory=lambda: {
        'macro': 0.5,
        'sector': 0.3,
        'style': 0.2,
    })


@dataclass
class PortfolioLayerConfig:
    """Layer 5: 组合构建配置."""
    enabled: bool = True

    # 底仓: 风险平价 vs 等权
    base_method: Literal['risk_parity', 'equal_weight'] = 'risk_parity'

    # 风险平价窗口 (周)
    rp_lookback: int = 52

    # 单 ETF 上下限
    floor: float = 0.005
    cap: float = 0.15


@dataclass
class V10Config:
    """v10 完整配置."""
    # 调仓频率 (用户决策 #5: 周+月都测试)
    rebal_freq: Literal['W', 'M'] = 'W'

    # 成本
    cost_bps: float = 5.0

    # 预热期
    warmup_days: int = 252

    # 各层配置
    macro: MacroLayerConfig = field(default_factory=MacroLayerConfig)
    industry: IndustryLayerConfig = field(default_factory=IndustryLayerConfig)
    style: StyleLayerConfig = field(default_factory=StyleLayerConfig)
    factor: FactorLayerConfig = field(default_factory=FactorLayerConfig)
    risk: RiskLayerConfig = field(default_factory=RiskLayerConfig)
    position: PositionLayerConfig = field(default_factory=PositionLayerConfig)
    portfolio: PortfolioLayerConfig = field(default_factory=PortfolioLayerConfig)

    # 输出选项
    save_states: bool = True
