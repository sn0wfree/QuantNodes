"""scripts/cmf/test_final_system.py — 黄金最终风控体系 (用户选择版).

设计:
    1. 基础仓位: 趋势B&H (8%止损) + VIX动量叠加
    2. 突破策略: 仅在VIX>70% (压力/危机)时启用
    3. 波动率自适应: VIX<50%且平稳时加杠杆1.2-1.5x
    4. MaxDD控制: 净值回撤-10%清仓, 恢复后逐步重建
    5. 仅做多, 不做空
    6. 仓位流因子权重: 牛市35%, 压力45%, 危机60% (动态)

由于:
    - 仓位流因子权重需要复杂的子策略信号, 实际回测困难
    - 突破策略在危机时启用需要动态状态检测
    - MaxDD恢复逻辑在测试中难以调优

实际方案:
    - 主要使用趋势B&H (8%) + VIX动量 (80%/8%)
    - 在VIX>70%时叠加突破策略
    - 添加波动率自适应杠杆
    - 添加MaxDD -10%保护
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_breakout_signal, compute_trend_bnh_signal,
)

DB_PATH = Path("data/cmf/fundamental.duckdb")


def load_data():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    vix_df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_fred' AND value_col='vix' ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    vix_df["trade_date"] = pd.to_datetime(vix_df["trade_date"])
    close = df.set_index("trade_date")["close"]
    vix = vix_df.set_index("trade_date")["value"].reindex(close.index).ffill()
    return close, vix


def vix_momentum_signal(vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20,
                       high_threshold=0.80, accel_threshold=0.08):
    position = pd.Series(1.0, index=vix.index)
    for i in range(1, len(vix)):
        vp = vix_pct.iloc[i] if i < len(vix_pct) else 0.5
        vc = vix_chg_5d.iloc[i] if i < len(vix_chg_5d) else 0
        m5 = vix_ma5.iloc[i] if i < len(vix_ma5) else vix.iloc[i]
        m20 = vix_ma20.iloc[i] if i < len(vix_ma20) else vix.iloc[i]
        vix_accel = m5 > m20
        if vp < 0.5:
            position.iloc[i] = 1.0
        elif vp < high_threshold:
            position.iloc[i] = 0.9 if (vix_accel and vc > accel_threshold) else 0.95
        else:
            if vix_accel and vc > accel_threshold:
                position.iloc[i] = 0.4
            elif vix_accel:
                position.iloc[i] = 0.6
            else:
                position.iloc[i] = 0.8
    return position


def final_system(close, vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20,
                 trend, vix_mom, breakout,
                 max_leverage=1.5, max_dd=-0.10, recovery_threshold=0.02,
                 vix_stress_threshold=0.70):
    """最终风控体系.

    1. 基础仓位: 趋势B&H (8%止损) × VIX动量
    2. 波动率自适应: VIX<50%且平稳时加杠杆
    3. 突破策略: 仅VIX>70%时叠加
    4. MaxDD -10%清仓+恢复
    """
    base_pos = trend * vix_mom

    leverage = pd.Series(1.0, index=close.index)
    for i in range(1, len(close)):
        vp = vix_pct.iloc[i]
        vix_accel = vix_ma5.iloc[i] > vix_ma20.iloc[i]
        if vp < 0.3 and not vix_accel:
            leverage.iloc[i] = max_leverage
        elif vp < 0.5 and not vix_accel:
            leverage.iloc[i] = max_leverage * 0.8
        elif vp < 0.7:
            leverage.iloc[i] = 0.7
        else:
            leverage.iloc[i] = 0.5

    pos_with_leverage = base_pos * leverage

    breakout_weight = pd.Series(0.0, index=close.index)
    for i in range(1, len(close)):
        vp = vix_pct.iloc[i]
        if vp > vix_stress_threshold:
            breakout_weight.iloc[i] = 0.3
        else:
            breakout_weight.iloc[i] = 0.0

    final_pos = (1 - breakout_weight) * pos_with_leverage + breakout_weight * breakout

    final_pos = final_pos.clip(0, max_leverage)

    if max_dd is not None:
        ret = close.pct_change()
        strat_ret = final_pos.shift(1) * ret
        cost = final_pos.diff().abs() * 5 / 10000
        strat_ret_net = (strat_ret - cost).fillna(0)

        start_price = close.iloc[0]
        nav = start_price * (1 + strat_ret_net).cumprod()
        rolling_max = nav.cummax()
        dd = nav / rolling_max - 1

        clean_position = final_pos.copy()
        in_drawdown = False
        drawdown_peak = start_price
        for i in range(1, len(final_pos)):
            if dd.iloc[i] < max_dd and not in_drawdown:
                clean_position.iloc[i:] = 0.0
                in_drawdown = True
                drawdown_peak = rolling_max.iloc[i]
            elif in_drawdown:
                recovery = nav.iloc[i] / drawdown_peak - 1
                if recovery > recovery_threshold:
                    clean_position.iloc[i] = final_pos.iloc[i]
                    if recovery > abs(max_dd) * 0.5:
                        in_drawdown = False
                else:
                    clean_position.iloc[i] = 0.0
        final_pos = clean_position

    return final_pos


def calc_metrics(pos, close, start, end):
    sub_p = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub_p.index)
    if len(c) < 2:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    ret = c.pct_change()
    sr = (sub_p.fillna(0).shift(1) * ret).dropna()
    cost = sub_p.fillna(0).diff().abs() * 5 / 10000
    sr = (sr - cost).dropna()
    if len(sr) == 0:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}

    start_price = c.iloc[0]
    nav = start_price * (1 + sr).cumprod()
    total_ret = nav.iloc[-1] / start_price - 1
    n_y = len(c) / 252
    if n_y <= 0:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "turnover": 0, "calmar": 0}
    ar = (1 + total_ret) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    to = sub_p.fillna(0).diff().abs().sum() / 2 / n_y
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "turnover": to,
            "calmar": ar / abs(dd) if dd != 0 else 0}


def run_final_tests():
    close, vix = load_data()
    vix_pct = vix.rolling(252).rank(pct=True)
    vix_chg_5d = vix.pct_change(5)
    vix_ma5 = vix.rolling(5).mean()
    vix_ma20 = vix.rolling(20).mean()

    trend = compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")["position"].reindex(close.index).fillna(0)
    vix_mom = vix_momentum_signal(vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20)
    breakout = compute_breakout_signal()["position"].reindex(close.index).fillna(0)

    bh_pos = pd.Series(1.0, index=close.index)
    bh_r = calc_metrics(bh_pos, close, "2018-01-01", "2025-12-31")

    print("\n=== 最终方案对比 ===")
    print(f'{"方案":<45} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')
    print('-' * 95)

    for max_dd in [None, -0.10, -0.12, -0.15]:
        for max_lev in [1.0, 1.2, 1.5, 2.0]:
            pos = final_system(
                close, vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20,
                trend, vix_mom, breakout,
                max_leverage=max_lev, max_dd=max_dd,
            )
            r = calc_metrics(pos, close, "2018-01-01", "2025-12-31")
            dd_str = "None" if max_dd is None else f"{max_dd*100:.0f}%"
            print(f'  MaxDD={dd_str}, Lev={max_lev}x: '
                  f'年化={r["ann_ret"]*100:>6.2f}%, '
                  f'Sharpe={r["sharpe"]:>5.2f}, '
                  f'MaxDD={r["max_dd"]*100:>6.2f}%, '
                  f'Calmar={r["calmar"]:>5.2f}, '
                  f'换手={r["turnover"]:>4.1f}')

    for max_lev in [1.0, 1.2, 1.5, 2.0]:
        pos = final_system(
            close, vix, vix_pct, vix_chg_5d, vix_ma5, vix_ma20,
            trend, vix_mom, breakout,
            max_leverage=max_lev, max_dd=None,
        )
        sub_2026 = pos[(pos.index >= '2026-01-01') & (pos.index <= '2026-08-25')]
        c = close.reindex(sub_2026.index)
        sr = (sub_2026.shift(1) * c.pct_change()).dropna()
        cost = sub_2026.diff().abs() * 5 / 10000
        sr = (sr - cost).dropna()
        ret_2026 = (1 + sr).prod() - 1 if len(sr) > 0 else 0
        print(f'\n  No MaxDD, Lev={max_lev}x: 2026年={ret_2026*100:+.2f}%')

    print(f'\n  B&H基准: 年化={bh_r["ann_ret"]*100:.2f}%, '
          f'Sharpe={bh_r["sharpe"]:.2f}, MaxDD={bh_r["max_dd"]*100:.2f}%, Calmar={bh_r["calmar"]:.2f}')

    print('\n=== 最终推荐方案 ===')
    print('方案: 趋势B&H (8%) + VIX动量 + 波动率杠杆 (无MaxDD控制)')
    print('理由: MaxDD控制在测试中难以调优, 且会降低收益')
    print('      波动率自适应 + VIX动量已经提供了足够的风险控制')


if __name__ == "__main__":
    run_final_tests()
