# coding=utf-8
"""copper_layered_ensemble.py — 铜分层集成信号 (借鉴黄金生产模式).

与 copper_combined.py (简单加权) 的区别:
    combined:  先加权 → 回测内dd止损
    layered:   先各自风控 → 再加权 (借鉴 gold_ensemble.compute_layered_ensemble_signal)

三级风控 (shared/commodity_risk_overlay, 周频参数):
    L1 波动率压制: realized_vol > vol_cap → 仓位×(vol_cap/realized_vol)
    L2 软减仓: rolling_dd < soft_cap → 仓位×0.5
    L3 硬止损: rolling_dd < hard_cap OR 周跌幅 < crash_cap → 清仓+冷却期

信号权重 (与 copper_combined 一致):
    breakout 75% + carry 10% + quadrant 15%
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from QuantNodes.strategy.commodity_fundamental.copper.copper_combined import (
    backtest_combined,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_breakout import (
    compute_breakout_signal,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_carry import (
    build_signal_carry,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import (
    compute_inventory_cycle_optimized,
)

from scripts.cmf.shared import compute_risk_overlay

DB = "data/cmf/fundamental.duckdb"

# 周频风控参数 (铜: 波动高于黄金, 回撤更深)
DEFAULT_RISK_PARAMS = dict(
    hard_cap=-0.18,
    soft_cap=-0.12,
    crash_cap=-0.07,
    vol_cap=0.25,
    cooldown_periods=2,
    vol_window=13,
    dd_window=52,
    periods_per_year=52,
    warmup_periods=26,
)


def load_copper_weekly_close() -> pd.Series:
    """加载铜周频收盘价."""
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].resample("W-FRI").last()


def compute_layered_ensemble_signal_cu(
    breakout_weight: float = 0.75,
    carry_weight: float = 0.10,
    quadrant_weight: float = 0.15,
    use_risk_overlay: bool = True,
    breakout_hard_stop: bool = False,
    deadband: float = 0.05,
    **risk_params,
) -> pd.DataFrame:
    """铜分层集成信号 — 各信号独立风控后再加权.

    Args:
        breakout_weight: 突破策略权重
        carry_weight: carry权重
        quadrant_weight: 四象限权重
        use_risk_overlay: 是否对各信号独立应用三级风控
        breakout_hard_stop: breakout是否也应用L3硬止损
            (默认False: breakout自带3%追踪止损, L3与其双重止损冲突,
             仅适用L1波动压制+L2软减仓; carry/quadrant慢信号用全三级)
        deadband: 死带平滑阈值 (仓位变化小于此值不动)
        **risk_params: 风控参数覆盖 (见 DEFAULT_RISK_PARAMS)

    Returns:
        DataFrame with columns:
            - close_w: 周频收盘价
            - breakout_pos: 突破仓位 (风控后)
            - carry_pos: carry仓位 (风控后)
            - quadrant_pos: 四象限仓位 (风控后)
            - position: 加权组合仓位
    """
    params = dict(DEFAULT_RISK_PARAMS)
    params.update(risk_params)

    close_w = load_copper_weekly_close()

    # 三路信号
    bo_pos = compute_breakout_signal()["position"]
    bo_pos = bo_pos.reindex(close_w.index, method="ffill").fillna(0)

    con = duckdb.connect(DB, read_only=True)
    try:
        carry_pos = build_signal_carry(con)["position"]
    finally:
        con.close()
    carry_pos = carry_pos.reindex(close_w.index).fillna(0)

    quad_pos = compute_inventory_cycle_optimized()["position"]
    quad_pos = quad_pos.reindex(close_w.index).fillna(0)

    # timing过滤器 (ex-ante信号质量过滤, 与copper_combined一致, 仅作用于breakout)
    # overlay是ex-post风控, 两者互补
    mom = close_w.pct_change(2)
    vol = close_w.pct_change().rolling(4).std() * np.sqrt(52)
    vol_pct = vol.rolling(52).rank(pct=True)
    timing = ((mom > 0) | (vol_pct < 0.7)).fillna(False)
    bo_pos = bo_pos * timing.astype(float)

    # 再入场确认 (防止chop-out): 沿用同一timing模式
    reentry_ok = timing

    # 各信号独立风控 (layered核心: 先风控 → 再加权)
    if use_risk_overlay:
        bo_params = dict(params)
        if not breakout_hard_stop:
            bo_params["hard_cap"] = None
            bo_params["crash_cap"] = None
        bo_pos = compute_risk_overlay(
            close=close_w, position=bo_pos, reentry_ok=reentry_ok, **bo_params
        )
        carry_pos = compute_risk_overlay(
            close=close_w, position=carry_pos, reentry_ok=reentry_ok, **params
        )
        quad_pos = compute_risk_overlay(
            close=close_w, position=quad_pos, reentry_ok=reentry_ok, **params
        )

    # 加权平均
    position = (
        breakout_weight * bo_pos
        + carry_weight * carry_pos
        + quadrant_weight * quad_pos
    ).clip(0, 1)

    # 死带平滑
    if deadband > 0:
        for i in range(1, len(position)):
            if abs(position.iloc[i] - position.iloc[i - 1]) < deadband:
                position.iloc[i] = position.iloc[i - 1]

    return pd.DataFrame({
        "close_w": close_w,
        "breakout_pos": bo_pos,
        "carry_pos": carry_pos,
        "quadrant_pos": quad_pos,
        "position": position,
    })


def compare_combined_vs_layered() -> pd.DataFrame:
    """对比回测: 旧简单加权 vs 新layered (同窗口 apples-to-apples).

    已知结论 (2026-09-02归档): layered在铜上为负结果 —
    基准combined 10.71%/0.99/-9.99% 优于 layered ~9.5%/0.89/-13.8%.
    原因: copper breakout自带3%追踪止损(内部风控), 外部overlay在周频上
    砍在恢复期, MaxDD反而更深. 黄金layered模式不适用于铜.
    """
    from QuantNodes.strategy.commodity_fundamental.copper.copper_combined import (
        compute_combined_signal,
    )

    old_sig = compute_combined_signal()
    common = old_sig.index

    new_sig = compute_layered_ensemble_signal_cu().reindex(common).fillna(0)

    rows = []
    for name, s in [("combined(简单加权)", old_sig), ("layered(先风控后加权)", new_sig)]:
        res = backtest_combined(s)
        rows.append({
            "mode": name,
            "ann_ret": res["ann_ret"], "sharpe": res["sharpe"],
            "max_dd": res["max_dd"], "win_rate": res["win_rate"],
            "avg_pos": s["position"].mean(),
        })

    return pd.DataFrame(rows)


__all__ = [
    "compute_layered_ensemble_signal_cu",
    "compare_combined_vs_layered",
    "load_copper_weekly_close",
    "DEFAULT_RISK_PARAMS",
]
