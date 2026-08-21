# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/kalman_filter.py

§2.1 卡尔曼滤波局部水平模型 — 商品价格中枢.

模型:
    y_t = μ_t + ε_t, ε_t ~ N(0, R)
    μ_t = μ_{t-1} + η_t, η_t ~ N(0, Q)

默认 q = Q/R = 0.015 → 稳态 K ≈ 0.115, 等价 ~5.7 个交易日记忆.
对日度收盘做滤波 → 每周五重采样 → 周频中枢.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_local_level(
    y: pd.Series,
    q: float = 0.015,
    init_state: float | None = None,
    init_var: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """局部水平KF 闭式递推.

    参数:
        y: 观测序列 (pd.Series, index 可任意)
        q: 噪声比 Q/R
        init_state: μ_0 初始; None → y.iloc[0]
        init_var: V_0 初始; None → 1000 (弱先验)

    返回:
        (μ_hat, gain): 滤波后的状态序列 + 各步增益
    """
    y_arr = y.values.astype(float)
    n = len(y_arr)

    if init_state is None:
        mu = y_arr[0] if not np.isnan(y_arr[0]) else 0.0
    else:
        mu = init_state
    if init_var is None:
        V = 1000.0
    else:
        V = init_var

    # 固定观测噪声 R=1 (相对单位), Q = q*R = q
    R = 1.0
    Q = q * R

    mus = np.empty(n)
    gains = np.empty(n)

    for t in range(n):
        # 预测步 (μ_{t|t-1} = μ_{t-1|t-1}, V_{t|t-1} = V_{t-1|t-1} + Q)
        V_pred = V + Q
        # 增益
        K = V_pred / (V_pred + R)
        gains[t] = K
        # 更新
        if not np.isnan(y_arr[t]):
            mu = mu + K * (y_arr[t] - mu)
        mus[t] = mu
        V = (1.0 - K) * V_pred

    return pd.Series(mus, index=y.index, name="mu_hat"), pd.Series(gains, index=y.index, name="gain")


def weekly_kalman_center(
    daily_close: pd.Series,
    q: float = 0.015,
    resample_rule: str = "W-FRI",
) -> pd.Series:
    """日度KF → 周频中枢 (每周五重采样最后一日滤波值)."""
    mu_daily, _ = kalman_local_level(daily_close, q=q)
    mu_weekly = mu_daily.resample(resample_rule).last()
    return mu_weekly.dropna()


if __name__ == "__main__":
    import duckdb

    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    cu = con.execute("SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date").df()
    cu["trade_date"] = pd.to_datetime(cu["trade_date"])
    cu = cu.set_index("trade_date")["close"]
    mu_weekly = weekly_kalman_center(cu)
    print(f"CU 周频中枢: {len(mu_weekly)} 周, {mu_weekly.index.min()} ~ {mu_weekly.index.max()}")
    print(f"中枢价格范围: {mu_weekly.min():.0f} ~ {mu_weekly.max():.0f}")
    print(f"中枢 vs 实际价格相关性: {mu_weekly.corr(cu.resample('W-FRI').last().dropna()):.4f}")