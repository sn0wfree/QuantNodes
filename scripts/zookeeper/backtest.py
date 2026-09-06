# coding: utf-8
"""Zookeeper 因子月度再平衡截面回测 — P4 待实装.

流程 (论文 §4):
  1. 月末 t: 60 商品 × 38 特征 → L-En 预测 r̂_t+1
  2. 横截面排序: 前 25% Q4 / 后 25% Q1
  3. t 收盘等权做多 Q4 / 做空 Q1, 完全保证金
  4. t+1 末重复, 月度再平衡
  5. **严格扣 cost_model.json** (知乎陷阱 §7.1)
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import (TURN_WARN)

LOG = logging.getLogger(__name__)


def cross_section_backtest(predictions: pd.DataFrame,
                           returns: pd.DataFrame,
                           cost_bps: float = 0.0015) -> dict:
    """Zookeeper 因子月度截面回测.

    Args:
        predictions: DataFrame, index=trade_date (MonthEnd), columns=product.
                    值 = L-En 预测下期收益.
        returns: DataFrame, index=trade_date (MonthEnd), columns=product.
                值 = 实际月度收益.
        cost_bps: 单边成本 (基础 15bps + 换月滑点).

    Returns:
        dict: nav / monthly_returns / turnover / metrics
    """
    # TODO P4: 排序 → Q4/Q1 → 多空 → 扣成本 → 输出 NAV / Sharpe / t-stat / 换手
    raise NotImplementedError("P4 待实装")


def apply_costs(nav: pd.Series, turnover: pd.Series,
                cost_bps: float = 0.0015) -> pd.Series:
    """成本扣减 (沿用 §22 cost_model.json 校准)."""
    # TODO P4: nav[t] *= (1 - turnover[t] * cost_bps)
    raise NotImplementedError("P4 待实装")


def compute_metrics(nav: pd.Series, monthly_ret: pd.Series) -> dict:
    """评估指标: 年化 / Sharpe / t-stat / 最大回撤."""
    # TODO P4: sharpe = mean/std * sqrt(12); t = mean/std * sqrt(n)
    raise NotImplementedError("P4 待实装")


def check_turnover_alert(turnover: pd.Series, threshold: float = TURN_WARN) -> dict:
    """换手监控 (知乎陷阱 §7.1 报警)."""
    # TODO P4: if turnover.mean() > threshold: warning
    raise NotImplementedError("P4 待实装")
