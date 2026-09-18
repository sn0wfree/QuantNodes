# coding=utf-8
"""品种选择去偏: 板块分散化品种池 + IS→OOS 两阶段验证.

问题: 事后基于业绩筛选品种 (如聚焦 9 品种) 存在数据挖掘偏差,
    历史测试显示 IS→OOS Calmar 比率仅 -0.07, 严重过拟合。

解决:
    1. 板块分散化: 按板块 (贵金属/有色/黑色/能化/农产品/国债/股指)
       各选 Top-N 流动性品种, 不基于业绩筛选
    2. IS→OOS 两阶段: IS (2020-2024) 选品种/调参数, OOS (2025-2026) 验证
    3. 多周期混合: 短/中/长周期 CTA 等权组合, 降低单一策略回撤
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


# ── 板块品种分类 (基于市场惯例 + 流动性) ───────────────────
# 品种按板块分组, 用于板块分散化品种池构建
# 板块内按流动性降序 (基于 2020-2026 实际成交数据)
SECTOR_PRODUCTS: dict[str, list[str]] = {
    "贵金属": ["AU", "AG"],
    "有色金属": ["CU", "AL", "ZN", "NI", "sn", "pb"],
    "黑色": ["RB", "I", "hc", "SF", "SM", "j", "jm"],
    "能化": ["SC", "TA", "MA", "bu", "eg", "eb", "v", "l", "pp"],
    "农产品": ["M", "Y", "P", "SR", "cf", "rm", "oi"],
    "国债": ["T", "TF", "TS"],
    "股指": ["IF", "IC", "IH"],
}

# 各板块的"流动性 Top-N"映射 (无业绩筛选, 仅按市场地位)
LIQUIDITY_TOP_N: dict[str, int] = {
    "贵金属": 1,    # 仅 AU (主选), AG 波动大但流动性稍低
    "有色金属": 3,  # CU/AL/ZN (3 大基本金属)
    "黑色": 2,      # RB (螺纹) / I (铁矿石)
    "能化": 3,      # SC/TA/MA (原油/PTA/甲醇)
    "农产品": 2,    # M (豆粕) / Y (豆油)
    "国债": 1,      # T (10年国债主力)
    "股指": 0,      # IF/IC/IH 默认不纳入 (Phase 7 WF 显示过拟合风险高)
}

# 默认板块分散化品种池 (推荐生产配置)
DEFAULT_DIVERSIFIED_POOL: list[str] = [
    "AU",  # 贵金属 (黄金)
    "CU", "AL", "ZN",  # 有色金属
    "RB", "I",  # 黑色
    "SC", "TA", "MA",  # 能化
    "M", "Y",  # 农产品
    "T",  # 国债
    # IF 暂不加入 (Phase 7 WF 显示过拟合风险高)
]


def sector_diversified_pool(
    top_n_per_sector: dict[str, int] | None = None,
    exclude_sectors: Iterable[str] = (),
    exclude_products: Iterable[str] = (),
) -> list[str]:
    """构建板块分散化品种池.

    原则:
        1. 不基于历史业绩筛选
        2. 按板块均匀分布 (1-3 个/板块)
        3. 板块内按流动性 Top-N 选择

    Args:
        top_n_per_sector: 各板块选 Top-N (默认 LIQUIDITY_TOP_N)
        exclude_sectors: 排除板块
        exclude_products: 排除品种

    Returns:
        list[str]: 板块分散化的品种池
    """
    n_per_sector = top_n_per_sector or LIQUIDITY_TOP_N
    excluded_sectors = set(exclude_sectors)
    excluded_products = set(exclude_products)

    pool: list[str] = []
    for sector, products in SECTOR_PRODUCTS.items():
        if sector in excluded_sectors:
            continue
        n = n_per_sector.get(sector, 1)
        # 按预设顺序取前 n 个 (顺序已按流动性降序)
        for p in products[:n]:
            if p not in excluded_products:
                pool.append(p)
    return pool


def pool_sector_breakdown(pool: list[str]) -> dict[str, list[str]]:
    """返回品种池的板块分布 (用于验证分散度)."""
    breakdown: dict[str, list[str]] = {}
    for sector, products in SECTOR_PRODUCTS.items():
        members = [p for p in pool if p in products]
        if members:
            breakdown[sector] = members
    return breakdown


# ── IS→OOS 两阶段验证 ────────────────────────────────────────

def split_is_oos(
    start: str = "2020-01-01",
    end: str = "2026-09-04",
    is_years: float = 5.0,
) -> tuple[str, str, str, str]:
    """划分 IS / OOS 区间.

    Args:
        start: 起始日期
        end: 结束日期
        is_years: IS 阶段年数 (OOS 阶段取剩余)

    Returns:
        (is_start, is_end, oos_start, oos_end)
    """
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    is_end = s + pd.DateOffset(years=int(is_years)) - pd.Timedelta(days=1)
    oos_start = is_end + pd.Timedelta(days=1)
    return (
        str(s.date()),
        str(is_end.date()),
        str(oos_start.date()),
        str(e.date()),
    )


def is_oos_validation_ratio(
    is_metric: float,
    oos_metric: float,
    threshold: float = 0.5,
) -> dict:
    """评估 IS→OOS 的稳定性.

    Args:
        is_metric: 样本内指标 (Calmar/Sharpe)
        oos_metric: 样本外指标
        threshold: 比率阈值 (默认 0.5, 比率 > 0.5 视为无严重过拟合)

    Returns:
        dict: ratio, is_metric, oos_metric, robust, severity
    """
    if is_metric == 0:
        ratio = float("inf") if oos_metric == 0 else 0.0
    else:
        ratio = oos_metric / is_metric

    if ratio >= threshold:
        severity = "robust"
    elif ratio >= 0.0:
        severity = "mild_overfit"
    elif ratio >= -0.5:
        severity = "moderate_overfit"
    else:
        severity = "severe_overfit"

    return {
        "ratio": round(ratio, 4),
        "is_metric": is_metric,
        "oos_metric": oos_metric,
        "threshold": threshold,
        "robust": ratio >= threshold,
        "severity": severity,
    }


# ── 多周期 CTA 组合 ─────────────────────────────────────────

# 短/中/长周期参数 (基于 confirm_window + ema_slow 区分)
# 短周期: 反应快, 频繁交易, 收益波动大
# 中周期: 默认参数, 平衡
# 长周期: 反应慢, 持仓时间长, 回撤大但趋势收益厚
PERIOD_PRESETS: dict[str, dict] = {
    "short": {
        "confirm_window": 5,
        "ema_slow": 30,
        "max_holding_days": 30,
        "chandelier_window": 15,
        "chandelier_mult_adjust": -0.5,  # 更紧的止损
        "description": "短周期: 高频, 低延迟",
    },
    "medium": {
        "confirm_window": 10,
        "ema_slow": 60,
        "max_holding_days": 60,
        "chandelier_window": 22,
        "chandelier_mult_adjust": 0.0,
        "description": "中周期: 默认参数",
    },
    "long": {
        "confirm_window": 20,
        "ema_slow": 120,
        "max_holding_days": 120,
        "chandelier_window": 35,
        "chandelier_mult_adjust": +1.0,  # 更宽的止损
        "description": "长周期: 慢趋势, 厚收益",
    },
}


def make_period_config(
    base_cfg: "HuataiTimingConfig",  # noqa: F821
    period: str,
) -> "HuataiTimingConfig":  # noqa: F821
    """基于基础配置生成多周期变体.

    Args:
        base_cfg: 基础配置
        period: 'short' / 'medium' / 'long'

    Returns:
        新 HuataiTimingConfig, 参数被周期预设覆盖
    """
    from copy import deepcopy
    # HuataiTimingConfig 是 base_cfg 的类型, deepcopy 后修改即可

    if period not in PERIOD_PRESETS:
        raise ValueError(f"period must be one of {list(PERIOD_PRESETS)}, got {period!r}")

    preset = PERIOD_PRESETS[period]
    cfg = deepcopy(base_cfg)
    cfg.confirm_window = preset["confirm_window"]
    cfg.ema_slow = preset["ema_slow"]
    cfg.max_holding_days = preset["max_holding_days"]
    cfg.chandelier_window = preset["chandelier_window"]

    # 调整 chandelier_mult (基于基础配置的乘数 ± 调整)
    adj = preset["chandelier_mult_adjust"]
    new_mult = {}
    base_mult = base_cfg.chandelier_mult
    for p, m in base_mult.items():
        new_mult[p] = max(1.0, m + adj)
    cfg.chandelier_mult = new_mult
    return cfg


def combine_period_navs(
    navs: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """多周期 NAV 等权/自定义权组合.

    Args:
        navs: {period_name: nav_series}
        weights: {period_name: weight}, 默认等权

    Returns:
        加权组合后的 NAV (pd.Series, 索引为各 nav 的并集)
    """
    if not navs:
        raise ValueError("navs must be non-empty")

    if weights is None:
        # 等权
        weights = {k: 1.0 / len(navs) for k in navs}
    else:
        # 归一化
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("weights must sum to positive")
        weights = {k: v / total for k, v in weights.items()}

    # 对齐到共同索引 (取交集)
    common_idx = None
    for name, nav in navs.items():
        idx = nav.dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError("no common index across navs")

    # 加权组合
    combined = pd.Series(0.0, index=common_idx)
    for name, nav in navs.items():
        aligned = nav.reindex(common_idx).ffill().fillna(1.0)
        # 归一化到起始值=1.0
        normalized = aligned / aligned.iloc[0]
        combined = combined + weights[name] * normalized
    return combined

