# coding=utf-8
"""系统性全局风控.

文章 §"系统性全局风控" 给出四类组合层面工具:
    1. 流动性预警 (依赖宏观数据, 本模块不做)
    2. 资产间相关性监控 (本模块提供 drawdown-based 简化版)
    3. 目标波动率管理 (vol-targeting)
    4. 极端行情压力测试 (本模块提供 nav-based 简化版)
    5. 全账户硬熔断机制

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DrawdownCircuitBreaker:
    """全账户回撤熔断器.

    语义 (研报原文 §"系统性全局风控"):
        累计回撤触及预警线 → 暂停所有新开仓, **进入观察期**;
        观察期 (observation_days) 结束后解除, 以当前 NAV 重置峰值重新计量.
        (recovery_ratio 旧语义 "回撤收窄至 X% 解除" 已废弃:
         ratio=0.5 会造成反复触发/解除的重开循环 — AU 2025 爆仓根因;
         ratio=1.0 则要求回撤完全修复, 高波动品种可能永久锁死.)
    """

    def __init__(
        self,
        threshold: float = -0.10,
        recovery_ratio: float = 1.0,
        observation_days: int = 60,
    ):
        """Args:
            threshold: 预警回撤阈值 (如 -0.10 = 回撤 10% 触发)
            recovery_ratio: 兼容参数, 不再使用
            observation_days: 触发后的观察期天数 (默认 60), 期满解除并重置峰值
        """
        self.threshold = threshold
        self.recovery_ratio = recovery_ratio  # 兼容保留
        self.observation_days = observation_days
        self._active = False
        self._peak = np.nan
        self._lock_count = 0

    def reset(self) -> None:
        self._active = False
        self._peak = np.nan
        self._lock_count = 0

    def update(self, nav: float) -> bool:
        """输入当前 NAV, 返回是否处于熔断状态 (禁止新开仓).

        Args:
            nav: 当前账户净值

        Returns:
            bool: True 表示当前处于熔断状态 (禁止新开仓).
        """
        if np.isnan(self._peak) or nav > self._peak:
            self._peak = nav
        drawdown = nav / self._peak - 1.0 if self._peak > 0 else 0.0

        if self._active:
            self._lock_count -= 1
            if self._lock_count <= 0:
                # 观察期结束: 解除, 峰值重置为当前 NAV
                self._active = False
                self._peak = nav
            return True

        # 用 < 而非 <=, 避免浮点边界比较的不稳定
        if drawdown < self.threshold:
            self._active = True
            self._lock_count = self.observation_days
        return self._active

    @property
    def is_active(self) -> bool:
        return self._active


def target_vol_leverage(
    returns: pd.Series,
    target_vol: float = 0.15,
    vol_lookback: int = 20,
    max_leverage: float = 3.0,
    annualization: int = 252,
) -> pd.Series:
    """目标波动率杠杆: leverage = target_vol / realized_vol.

    Args:
        returns: 策略日收益序列
        target_vol: 目标年化波动率 (默认 0.15)
        vol_lookback: 波动率回看窗口 (默认 20 日)
        max_leverage: 杠杆上限 (默认 3x)
        annualization: 年化因子 (日频 252)

    Returns:
        pd.Series[float], 每日建议杠杆, 范围 [0, max_leverage].
    """
    realized_vol = returns.rolling(vol_lookback).std() * np.sqrt(annualization)
    leverage = (target_vol / realized_vol.replace(0, np.nan)).clip(upper=max_leverage)
    return leverage.fillna(0.0)


def rolling_drawdown(nav: pd.Series) -> pd.Series:
    """累计回撤序列: nav / peak(nav) - 1."""
    peak = nav.cummax()
    return nav / peak.replace(0, np.nan) - 1.0


def stress_test_extreme_days(
    returns: pd.Series,
    quantile: float = 0.99,
    metric: str = "calmar",
) -> dict:
    """极端行情压力测试: 剔除 |ret| 分位数 > quantile 的日期后, 评估策略稳健性.

    Args:
        returns: 策略日收益
        quantile: 极端日阈值分位数 (默认 0.99, 即剔除最极端 1% 日子)
        metric: 评估指标 (calmar / sharpe)

    Returns:
        dict: {'full': 全样本指标, 'excl_extreme': 剔除极端日后的指标,
               'extreme_days': int, 'dependency_ratio': float}.
    """
    threshold = returns.abs().quantile(quantile)
    mask = returns.abs() <= threshold
    excl = returns[mask]
    full = returns

    def _calmar(r: pd.Series) -> float:
        if r.empty:
            return np.nan
        ann = r.mean() * 252
        mdd = rolling_drawdown((1 + r).cumprod()).min()
        return ann / abs(mdd) if mdd < 0 else np.nan

    def _sharpe(r: pd.Series) -> float:
        if r.std() == 0:
            return np.nan
        return r.mean() / r.std() * np.sqrt(252)

    fn = _calmar if metric == "calmar" else _sharpe
    full_m = fn(full)
    excl_m = fn(excl)
    return {
        "full": full_m,
        "excl_extreme": excl_m,
        "extreme_days": int((~mask).sum()),
        "dependency_ratio": excl_m / full_m if full_m else np.nan,
    }
