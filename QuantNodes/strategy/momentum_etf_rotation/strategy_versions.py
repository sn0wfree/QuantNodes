# coding=utf-8
"""策略版本锁定 - 用户可显式选择版本.

每个版本返回固定配置的 RotationConfig.
新增版本时:
    1. 添加新函数 (e.g., def v1_1() -> RotationConfig)
    2. 添加到 VERSIONS dict
    3. 更新 STRATEGY_VERSIONS.md

约定:
    - Minor 版本 (v1.x): 保持向后兼容, 默认值不变
    - Major 版本 (v2.x): 可改变接口, 提供迁移指南
"""
from __future__ import annotations

from .portfolio import (
    RotationConfig,
    TrendFilter,
    VolTargeting,
    CostModel,
)


def v0_0_baseline() -> RotationConfig:
    """v0.0 baseline (Stage 8, 无任何增强).

    用途: 学术研究 / 理想化基准对比.
    Calmar 0.78, DD -21%, Ann 16.35%.
    """
    return RotationConfig(lookback=90, top_n=10)


def v0_1_vt_only() -> RotationConfig:
    """v0.1 + Stage 9-C (波动率目标).

    用途: 仅启用波动率目标, 不启用其他.
    Calmar 1.00, DD -6.89%, Ann 6.87%.
    """
    return RotationConfig(
        lookback=90, top_n=10,
        vol_targeting=VolTargeting(
            enabled=True, target_vol=0.15, lookback=60,
            min_scale=0.3, max_scale=1.5,
        ),
    )


def v0_2_tf_only() -> RotationConfig:
    """v0.2 + Stage 9-B (趋势过滤器).

    用途: 仅启用趋势过滤器, 不启用其他.
    Calmar 0.88, DD -17.05%, Ann 14.98%.
    """
    return RotationConfig(
        lookback=90, top_n=10,
        trend_filter=TrendFilter(
            enabled=True, benchmark_code="510300",
            ma_window=200, exposure_bear=0.7,
            bond_code="511260",
        ),
    )


def v0_3_vt_cost() -> RotationConfig:
    """v0.3 + Stage 13 (交易成本, 基于 v0.1).

    用途: v0.1 + 交易成本 (5bp+10bp).
    Calmar 0.98, DD -6.94%, Ann 6.83%.
    """
    return RotationConfig(
        lookback=90, top_n=10,
        vol_targeting=VolTargeting(
            enabled=True, target_vol=0.15, lookback=60,
            min_scale=0.3, max_scale=1.5,
        ),
        cost_model=CostModel(
            enabled=True, commission_bp=5, slippage_bp=10,
            impact_factor=0.1,
        ),
    )


def v0_4_hybrid() -> RotationConfig:
    """v0.4 + Stage 12A (hybrid 动量, 基于 v0.3).

    用途: v0.3 + hybrid 动量, 不含新功能.
    Calmar 1.17, DD -12.72%, Ann 14.84%.
    """
    return RotationConfig(
        lookback=90, top_n=10,
        momentum_type="hybrid",
        momentum_fused_weight=0.5,
        vol_targeting=VolTargeting(
            enabled=True, target_vol=0.15, lookback=60,
            min_scale=0.3, max_scale=1.5,
        ),
        cost_model=CostModel(
            enabled=True, commission_bp=5, slippage_bp=10,
            impact_factor=0.1,
        ),
    )


def v1_0() -> RotationConfig:
    """v1.0 锁定配置 (Stage 12A 完成).

    混合动量 (price + slope×R²) + 波动率目标 + 交易成本.

    指标 (2019-2026):
        Calmar 1.60, DD -3.93%, Ann 6.28%
        OOS Calmar 0.84

    用途: 风险厌恶型实盘部署.
    """
    return RotationConfig(
        lookback=90, top_n=10,
        momentum_type="hybrid",
        momentum_fused_weight=0.5,
        vol_targeting=VolTargeting(
            enabled=True, target_vol=0.15, lookback=60,
            min_scale=0.3, max_scale=1.5,
        ),
        cost_model=CostModel(
            enabled=True, commission_bp=5, slippage_bp=10,
            impact_factor=0.1,
        ),
    )


# 版本注册表
VERSIONS: dict[str, callable] = {
    "1.0": v1_0,
    "0.4": v0_4_hybrid,
    "0.3": v0_3_vt_cost,
    "0.2": v0_2_tf_only,
    "0.1": v0_1_vt_only,
    "0.0": v0_0_baseline,
}

# 最新版本 (默认)
LATEST = "1.0"


def get_version(version: str = LATEST) -> RotationConfig:
    """获取指定版本的配置.

    Args:
        version: 版本号 (默认 LATEST = "1.0")

    Returns:
        对应版本的 RotationConfig
    """
    if version not in VERSIONS:
        available = ", ".join(sorted(VERSIONS.keys(), reverse=True))
        raise ValueError(
            f"未知版本: {version!r}. 可用: {available}"
        )
    return VERSIONS[version]()


__all__ = [
    "v0_0_baseline",
    "v0_1_vt_only",
    "v0_2_tf_only",
    "v0_3_vt_cost",
    "v0_4_hybrid",
    "v1_0",
    "VERSIONS",
    "LATEST",
    "get_version",
]