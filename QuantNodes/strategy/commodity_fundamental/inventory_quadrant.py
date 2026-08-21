# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/inventory_quadrant.py

§3.1 库存状态四象限识别.

d_inv  = 库存 z-score 的 4 周变化
d_price = 价格 z-score 的 4 周变化

| 状态          | d_inv | d_price | 仓位 |
|--------------|-------|---------|------|
| 被动去库      |   ↓   |    ↑    | 1.0  |
| 主动补库      |   ↑   |    ↑    | 0.7  |
| 主动去库      |   ↓   |    ↓    | 0.4  |
| 被动补库      |   ↑   |    ↓    | 0.1  |
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sign_constrained_lasso import rolling_zscore


def compute_deltas(
    series: pd.Series,
    z_window: int = 104,
    delta_window: int = 4,
) -> pd.Series:
    """z-score 的 delta_window 周变化.

    Δz = z_t - z_{t-delta_window}
    """
    z = rolling_zscore(series, window=z_window)
    return z.diff(delta_window).rename(f"dz_{delta_window}")


def classify_quadrant(
    d_inv: pd.Series,
    d_price: pd.Series,
) -> pd.Series:
    """返回仓位 0.1/0.4/0.7/1.0 的 Series.

    边界处理: NaN 或 d_inv == 0 → 不持仓 (NaN)
    """
    pos = pd.Series(np.nan, index=d_inv.index, name="quadrant_pos")
    inv_down = d_inv < 0
    inv_up = d_inv > 0
    pr_down = d_price < 0
    pr_up = d_price > 0

    pos[inv_down & pr_up] = 1.0   # 被动去库
    pos[inv_up & pr_up] = 0.7     # 主动补库
    pos[inv_down & pr_down] = 0.4 # 主动去库
    pos[inv_up & pr_down] = 0.1   # 被动补库

    return pos


def inventory_state_for_product(
    inv_series: pd.Series,
    price_series: pd.Series,
    z_window: int = 156,
    delta_window: int = 4,
    inv_dir: int = -1,
) -> pd.DataFrame:
    """对单个商品产出 d_inv, d_price, 仓位序列.

    参数:
            inv_dir: -1 表示库存下降为利多 (金属/原油默认), +1 表示库存上升为利多
            z_window: z-score窗口 (默认156周, 减少象限转换频率)
    """
    inv_z = rolling_zscore(inv_series, window=z_window).dropna()
    pr_z = rolling_zscore(price_series, window=z_window).dropna()

    d_inv = inv_z.diff(delta_window) * inv_dir   # 标准化方向: 库存↓ → d_inv↑
    d_pr = pr_z.diff(delta_window)

    common = d_inv.index.intersection(d_pr.index)
    d_inv = d_inv.reindex(common)
    d_pr = d_pr.reindex(common)

    pos = classify_quadrant(d_inv, d_pr)
    return pd.DataFrame({
        "d_inv": d_inv,
        "d_price": d_pr,
        "quadrant_pos": pos,
    })


__all__ = ["compute_deltas", "classify_quadrant", "inventory_state_for_product"]