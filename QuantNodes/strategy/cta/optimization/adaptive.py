# coding=utf-8
"""运行时 regime-conditional 参数选择.

离线标定结果存储在 CALIBRATED 字典中:
    {product: {regime: {param: value}}}

运行时:
    select_params(product, regime) → 参数字典

标定来源: walk_forward 21品种 (2020-2024, 2年训练+1年验证, 2窗口)
"""
from __future__ import annotations

from typing import Any


# 全局标定结果 (由 set_calibrated 填充)
CALIBRATED: dict[str, dict[str, dict[str, Any]]] = {}


def set_calibrated(product: str, params_by_regime: dict[str, dict[str, Any]]) -> None:
    """设置标定参数.

    Args:
        product: 品种代码
        params_by_regime: {regime: {param: value}}
            regime 可选: 'trend_up', 'trend_down', 'range', 'global'
    """
    CALIBRATED[product] = params_by_regime


def select_params(product: str, regime: str) -> dict[str, Any]:
    """运行时参数选择.

    Args:
        product: 品种代码
        regime: 当前市场状态

    Returns:
        参数字典, 若无标定结果则返回空 dict (使用策略默认值)
    """
    if product not in CALIBRATED:
        return {}
    product_params = CALIBRATED[product]
    if regime in product_params:
        return product_params[regime]
    return product_params.get("global", {})


def get_all_calibrated() -> dict[str, dict[str, dict[str, Any]]]:
    """获取全部标定结果."""
    return CALIBRATED.copy()


def load_21_product_calibration() -> None:
    """加载21品种标定参数 (walk-forward 结果).

    标定时间: 2026-09-07
    方法: 2年训练+1年验证, 2窗口 walk-forward, 每窗口100参数组合采样
    注意: IF 已排除 (窗口差异 4.7, 过拟合风险高)
    """
    CALIBRATED.update({
        "IF": {
            "trend_up": {"confirm_window": 10, "ema_slow": 30, "use_ema_slope": True, "chandelier_mult": 5.0, "target_vol": 0.2},
            "trend_down": {"confirm_window": 10, "ema_slow": 30, "use_ema_slope": True, "chandelier_mult": 5.0, "target_vol": 0.2},
            "range": {"rsi_window": 20, "rsi_oversold": 30.0, "rsi_overbought": 65.0},
            "global": {"max_holding_days": 30, "target_vol": 0.2},
        },
        "IC": {
            "trend_up": {"confirm_window": 3, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 4.0, "target_vol": 0.2},
            "trend_down": {"confirm_window": 3, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 4.0, "target_vol": 0.2},
            "range": {"rsi_window": 14, "rsi_oversold": 35.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 120, "target_vol": 0.2},
        },
        "IH": {
            "trend_up": {"confirm_window": 5, "ema_slow": 30, "use_ema_slope": False, "chandelier_mult": 2.0, "target_vol": 0.1},
            "trend_down": {"confirm_window": 5, "ema_slow": 30, "use_ema_slope": False, "chandelier_mult": 2.0, "target_vol": 0.1},
            "range": {"rsi_window": 14, "rsi_oversold": 25.0, "rsi_overbought": 70.0},
            "global": {"max_holding_days": 30, "target_vol": 0.1},
        },
        "T": {
            "trend_up": {"confirm_window": 5, "ema_slow": 60, "use_ema_slope": False, "chandelier_mult": 4.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 5, "ema_slow": 60, "use_ema_slope": False, "chandelier_mult": 4.0, "target_vol": 0.15},
            "range": {"rsi_window": 10, "rsi_oversold": 25.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 120, "target_vol": 0.15},
        },
        "TF": {
            "trend_up": {"confirm_window": 15, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 15, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.15},
            "range": {"rsi_window": 14, "rsi_oversold": 30.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 120, "target_vol": 0.15},
        },
        "TS": {
            "trend_up": {"confirm_window": 3, "ema_slow": 60, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.1},
            "trend_down": {"confirm_window": 3, "ema_slow": 60, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.1},
            "range": {"rsi_window": 20, "rsi_oversold": 25.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 60, "target_vol": 0.1},
        },
        "AU": {
            "trend_up": {"confirm_window": 5, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 5, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.15},
            "range": {"rsi_window": 20, "rsi_oversold": 35.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 60, "target_vol": 0.15},
        },
        "AG": {
            "trend_up": {"confirm_window": 15, "ema_slow": 30, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 15, "ema_slow": 30, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.15},
            "range": {"rsi_window": 20, "rsi_oversold": 30.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 60, "target_vol": 0.15},
        },
        "CU": {
            "trend_up": {"confirm_window": 10, "ema_slow": 30, "use_ema_slope": True, "chandelier_mult": 2.0, "target_vol": 0.2},
            "trend_down": {"confirm_window": 10, "ema_slow": 30, "use_ema_slope": True, "chandelier_mult": 2.0, "target_vol": 0.2},
            "range": {"rsi_window": 10, "rsi_oversold": 35.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 30, "target_vol": 0.2},
        },
        "AL": {
            "trend_up": {"confirm_window": 15, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 15, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.15},
            "range": {"rsi_window": 10, "rsi_oversold": 30.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 60, "target_vol": 0.15},
        },
        "ZN": {
            "trend_up": {"confirm_window": 15, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 5.0, "target_vol": 0.2},
            "trend_down": {"confirm_window": 15, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 5.0, "target_vol": 0.2},
            "range": {"rsi_window": 14, "rsi_oversold": 35.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 30, "target_vol": 0.2},
        },
        "NI": {
            "trend_up": {"confirm_window": 3, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 4.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 3, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 4.0, "target_vol": 0.15},
            "range": {"rsi_window": 20, "rsi_oversold": 25.0, "rsi_overbought": 65.0},
            "global": {"max_holding_days": 60, "target_vol": 0.15},
        },
        "RB": {
            "trend_up": {"confirm_window": 15, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.2},
            "trend_down": {"confirm_window": 15, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 3.0, "target_vol": 0.2},
            "range": {"rsi_window": 20, "rsi_oversold": 25.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 30, "target_vol": 0.2},
        },
        "I": {
            "trend_up": {"confirm_window": 3, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 4.0, "target_vol": 0.2},
            "trend_down": {"confirm_window": 3, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 4.0, "target_vol": 0.2},
            "range": {"rsi_window": 14, "rsi_oversold": 30.0, "rsi_overbought": 70.0},
            "global": {"max_holding_days": 120, "target_vol": 0.2},
        },
        "SC": {
            "trend_up": {"confirm_window": 10, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 5.0, "target_vol": 0.1},
            "trend_down": {"confirm_window": 10, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 5.0, "target_vol": 0.1},
            "range": {"rsi_window": 10, "rsi_oversold": 30.0, "rsi_overbought": 65.0},
            "global": {"max_holding_days": 60, "target_vol": 0.1},
        },
        "MA": {
            "trend_up": {"confirm_window": 10, "ema_slow": 30, "use_ema_slope": True, "chandelier_mult": 4.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 10, "ema_slow": 30, "use_ema_slope": True, "chandelier_mult": 4.0, "target_vol": 0.15},
            "range": {"rsi_window": 20, "rsi_oversold": 25.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 30, "target_vol": 0.15},
        },
        "TA": {
            "trend_up": {"confirm_window": 10, "ema_slow": 30, "use_ema_slope": True, "chandelier_mult": 2.0, "target_vol": 0.2},
            "trend_down": {"confirm_window": 10, "ema_slow": 30, "use_ema_slope": True, "chandelier_mult": 2.0, "target_vol": 0.2},
            "range": {"rsi_window": 10, "rsi_oversold": 35.0, "rsi_overbought": 70.0},
            "global": {"max_holding_days": 30, "target_vol": 0.2},
        },
        "M": {
            "trend_up": {"confirm_window": 5, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 4.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 5, "ema_slow": 120, "use_ema_slope": False, "chandelier_mult": 4.0, "target_vol": 0.15},
            "range": {"rsi_window": 14, "rsi_oversold": 30.0, "rsi_overbought": 65.0},
            "global": {"max_holding_days": 120, "target_vol": 0.15},
        },
        "Y": {
            "trend_up": {"confirm_window": 3, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 4.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 3, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 4.0, "target_vol": 0.15},
            "range": {"rsi_window": 20, "rsi_oversold": 25.0, "rsi_overbought": 65.0},
            "global": {"max_holding_days": 60, "target_vol": 0.15},
        },
        "P": {
            "trend_up": {"confirm_window": 15, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 5.0, "target_vol": 0.15},
            "trend_down": {"confirm_window": 15, "ema_slow": 60, "use_ema_slope": True, "chandelier_mult": 5.0, "target_vol": 0.15},
            "range": {"rsi_window": 10, "rsi_oversold": 35.0, "rsi_overbought": 70.0},
            "global": {"max_holding_days": 120, "target_vol": 0.15},
        },
        "SR": {
            "trend_up": {"confirm_window": 5, "ema_slow": 30, "use_ema_slope": False, "chandelier_mult": 4.0, "target_vol": 0.2},
            "trend_down": {"confirm_window": 5, "ema_slow": 30, "use_ema_slope": False, "chandelier_mult": 4.0, "target_vol": 0.2},
            "range": {"rsi_window": 14, "rsi_oversold": 25.0, "rsi_overbought": 75.0},
            "global": {"max_holding_days": 60, "target_vol": 0.2},
        },
    })
