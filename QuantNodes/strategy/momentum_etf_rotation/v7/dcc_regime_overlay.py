# coding=utf-8
"""DCC Regime Overlay — 用 dcc_zscore_mean 作为 crisis 预警信号.

当 dcc_zscore_mean > 阈值时, 切换到防御模式 (减仓或全仓现金).

设计动机:
  dcc_zscore_mean 衡量当前相关性结构相对于正常水平 (0.3) 的异常程度.
  危机时, 通常不相关的资产开始同向运动, dcc_zscore_mean 急剧上升.
  此时减仓可以避免系统性风险.

用法:
  from v7.dcc_regime_overlay import DCCRegimeOverlay
  overlay = DCCRegimeOverlay(threshold=1.5, defense_mode='reduce', reduce_factor=0.5)
  nav_adjusted = overlay.apply(nav, weights_df, dcc_scores)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class DCCRegimeOverlay:
    """DCC Regime Overlay — 基于相关性异常的防御性调整.

    Parameters:
        threshold: dcc_zscore_mean 触发阈值 (默认 1.5)
        defense_mode: 防御模式 ('reduce' = 减仓, 'cash' = 全仓现金)
        reduce_factor: reduce 模式下的仓位比例 (默认 0.5 = 减半)
        cooldown: 触发后冷却周数 (默认 4)
    """

    def __init__(
        self,
        threshold: float = 1.5,
        defense_mode: str = "reduce",
        reduce_factor: float = 0.5,
        cooldown: int = 4,
    ):
        self.threshold = threshold
        self.defense_mode = defense_mode
        self.reduce_factor = reduce_factor
        self.cooldown = cooldown

    def apply(
        self,
        nav: pd.Series,
        weights_df: pd.DataFrame,
        dcc_scores: pd.Series,
    ) -> pd.Series:
        """应用 DCC regime overlay 到 NAV.

        Parameters:
            nav: 原始周频 NAV
            weights_df: 持仓权重 DataFrame (date, code, weight)
            dcc_scores: dcc_zscore_mean 周频序列 (index 对齐 nav)

        Returns:
            nav_adjusted: 调整后的 NAV
        """
        nav_adj = nav.copy()
        triggered = False
        cooldown_remaining = 0

        for i in range(1, len(nav)):
            date = nav.index[i]

            # 检查 DCC 分数
            if date in dcc_scores.index:
                score = dcc_scores.loc[date]
                if not np.isnan(score) and score > self.threshold and not triggered:
                    triggered = True
                    cooldown_remaining = self.cooldown

            # 冷却递减
            if triggered and cooldown_remaining > 0:
                cooldown_remaining -= 1
                if cooldown_remaining <= 0:
                    triggered = False

            # 调整 NAV
            if triggered:
                if self.defense_mode == "cash":
                    # 全仓现金: NAV 不变
                    nav_adj.iloc[i] = nav_adj.iloc[i - 1]
                elif self.defense_mode == "reduce":
                    # 减仓: 只保留 reduce_factor 的收益
                    ret = nav.iloc[i] / nav.iloc[i - 1] - 1
                    nav_adj.iloc[i] = nav_adj.iloc[i - 1] * (1 + ret * self.reduce_factor)

        return nav_adj

    def get_trigger_periods(
        self,
        dcc_scores: pd.Series,
        nav: pd.Series,
    ) -> list[dict]:
        """获取触发防御的时间段."""
        periods = []
        triggered = False
        start = None
        cooldown_remaining = 0

        for i in range(len(nav)):
            date = nav.index[i]
            if date in dcc_scores.index:
                score = dcc_scores.loc[date]
                if not np.isnan(score) and score > self.threshold and not triggered:
                    triggered = True
                    start = date
                    cooldown_remaining = self.cooldown

            if triggered and cooldown_remaining > 0:
                cooldown_remaining -= 1
                if cooldown_remaining <= 0:
                    periods.append({
                        "start": start,
                        "end": date,
                        "dcc_score": dcc_scores.loc[start] if start in dcc_scores.index else None,
                    })
                    triggered = False

        return periods
