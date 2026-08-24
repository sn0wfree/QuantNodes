# coding=utf-8
"""scripts/cmf/17_v6_3_rolling_backtest.py

v6.3 无前视回测 + 动态因子权重

方法:
    1. expanding window: 每个时间点只用历史数据
    2. 回归模型: 用历史因子值预测未来收益，得到时变权重
    3. Top8因子: 去掉最弱4个因子 (dxy_z, mom_13w, carry, lme_z)
"""
import sys
sys.path.insert(0, ".")

import duckdb
import numpy as np
import pandas as pd
from QuantNodes.strategy.commodity_fundamental.copper.copper_factors import (
    compute_all_factors, _load_weekly_close, FACTOR_NAMES,
)

DB = "data/cmf/fundamental.duckdb"

# Top8因子 (去掉最弱4个)
TOP8_FACTORS = [
    "quadrant_pos", "pmi", "industry_dz", "vol_pct",
    "cu_crude", "vol_13w_z", "curvature", "mom_2w",
]


def rolling_regression_weights(factors: pd.Series, returns: pd.Series,
                               window: int = 104) -> float:
    """滚动回归计算因子权重.

    用过去window期的因子值预测未来收益，得到权重。
    """
    if len(factors) < window:
        return 1.0 / len(factors)  # 等权回退

    X = factors.values[-window:].reshape(-1, 1)
    y = returns.values[-window:]

    # 去除NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    if len(X) < 30:
        return 0.5  # 数据不足回退

    # OLS回归
    X_with_const = np.column_stack([X, np.ones(len(X))])
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        return beta[0]  # 因子系数
    except:
        return 0.5


def run_rolling_backtest(use_top8: bool = True, use_dynamic_weights: bool = True,
                         regression_window: int = 104):
    """无前视滚动回测."""
    con = duckdb.connect(DB, read_only=True)

    close_w = _load_weekly_close(con, "CU0").dropna()
    all_factors = compute_all_factors(con)

    # 选择因子
    factor_names = TOP8_FACTORS if use_top8 else FACTOR_NAMES
    factor_names = [f for f in factor_names if f in all_factors.columns]

    factors = all_factors[factor_names]
    common = close_w.index.intersection(factors.index)
    close_w = close_w.reindex(common)
    factors = factors.reindex(common)
    ret_w = close_w.pct_change()

    min_history = 104
    positions = []

    for i in range(min_history, len(close_w)):
        current_date = close_w.index[i]

        # 只用历史数据
        factors_train = factors[factors.index <= current_date]
        close_train = close_w[close_w.index <= current_date]
        ret_train = close_train.pct_change()

        if len(factors_train) < 52:
            positions.append(0.5)
            continue

        # 计算因子权重
        if use_dynamic_weights:
            weights = {}
            for col in factor_names:
                w = rolling_regression_weights(
                    factors_train[col].fillna(0),
                    ret_train.fillna(0),
                    window=regression_window,
                )
                weights[col] = max(w, 0)  # 负权重置0

            # 归一化权重
            total_w = sum(weights.values())
            if total_w > 0:
                weights = {k: v / total_w for k, v in weights.items()}
            else:
                weights = {k: 1.0 / len(factor_names) for k in factor_names}
        else:
            # 等权
            weights = {k: 1.0 / len(factor_names) for k in factor_names}

        # 合成仓位
        pos_raw = sum(factors_train[k].rank(pct=True).fillna(0.5) * w
                      for k, w in weights.items())

        # bull_enhance
        mom26 = close_train.pct_change(26)
        bull = mom26 > 0.10

        from QuantNodes.strategy.commodity_fundamental.copper.copper_factors import (
            compute_lme_z, compute_carry, compute_cu_crude,
        )
        lme_z = compute_lme_z(con, close_train)
        carry_sig = compute_carry(con, close_train)
        cu_crude_pos = compute_cu_crude(con, close_train)

        factor_score = (
            (lme_z < -0.5).astype(int) +
            (carry_sig > 0.7).astype(int) +
            (cu_crude_pos > 0.6).astype(int)
        )
        enhance = bull & (factor_score >= 1)  # threshold=1

        pos_enhanced = pos_raw.copy()
        pos_enhanced[enhance] = np.maximum(pos_enhanced[enhance], 0.9)

        # 波动率目标化
        vol_4w = ret_train.rolling(4).std() * np.sqrt(52)
        scale = (0.12 / vol_4w).fillna(1.0).clip(upper=1.0)
        pos_sized = (pos_enhanced * scale).clip(upper=1.0)

        positions.append(pos_sized.iloc[-1])

    # 回测
    pos_series = pd.Series(positions, index=close_w.index[min_history:])
    strat = pos_series.shift(1) * ret_w.reindex(pos_series.index) - pos_series.diff().abs() * 5e-4
    strat = strat.dropna()
    m = (strat.index >= "2018-01-01") & (strat.index <= "2025-12-31")
    strat = strat[m]

    nav = (1 + strat).cumprod()
    ann = nav.iloc[-1] ** (52 / len(strat)) - 1 if len(strat) > 0 else 0
    vol = strat.std() * np.sqrt(52)
    sh = ann / vol if vol > 0 else 0
    dd = (nav / nav.cummax() - 1).min()

    return {"ann_ret": ann, "sharpe": sh, "max_dd": dd, "nav": nav}


def main():
    print("v6.3 无前视滚动回测")
    print("=" * 70)

    # 测试1: Top8 + 动态权重
    print("\n测试1: Top8因子 + 动态权重 (回归)")
    r1 = run_rolling_backtest(use_top8=True, use_dynamic_weights=True)
    print(f"  年化: {r1['ann_ret']*100:.1f}%")
    print(f"  Sharpe: {r1['sharpe']:.2f}")
    print(f"  MaxDD: {r1['max_dd']*100:.1f}%")

    # 测试2: Top8 + 等权
    print("\n测试2: Top8因子 + 等权")
    r2 = run_rolling_backtest(use_top8=True, use_dynamic_weights=False)
    print(f"  年化: {r2['ann_ret']*100:.1f}%")
    print(f"  Sharpe: {r2['sharpe']:.2f}")
    print(f"  MaxDD: {r2['max_dd']*100:.1f}%")

    # 测试3: 全部12因子 + 动态权重
    print("\n测试3: 全部12因子 + 动态权重")
    r3 = run_rolling_backtest(use_top8=False, use_dynamic_weights=True)
    print(f"  年化: {r3['ann_ret']*100:.1f}%")
    print(f"  Sharpe: {r3['sharpe']:.2f}")
    print(f"  MaxDD: {r3['max_dd']*100:.1f}%")

    # 测试4: 全部12因子 + 等权
    print("\n测试4: 全部12因子 + 等权")
    r4 = run_rolling_backtest(use_top8=False, use_dynamic_weights=False)
    print(f"  年化: {r4['ann_ret']*100:.1f}%")
    print(f"  Sharpe: {r4['sharpe']:.2f}")
    print(f"  MaxDD: {r4['max_dd']*100:.1f}%")

    # 分年对比 (最优方案)
    print("\n" + "=" * 70)
    print("最优方案分年对比: Top8 + 动态权重")
    print("=" * 70)
    nav = r1["nav"]
    for y in range(2018, 2026):
        yn = nav[nav.index.year == y]
        if len(yn) > 1:
            ret = (yn.iloc[-1] / yn.iloc[0] - 1) * 100
            print(f"  {y}: {ret:.1f}%")

    # 与v6.2+carry对比
    print("\n" + "=" * 70)
    print("与v6.2+carry对比")
    print("=" * 70)
    print(f"{'策略':<30} {'年化%':<10} {'Sharpe':<10} {'MaxDD%':<10}")
    print("-" * 60)
    print(f"{'v6.3 Top8+动态 (无前视)':<30} {r1['ann_ret']*100:<10.1f} {r1['sharpe']:<10.2f} {r1['max_dd']*100:<10.1f}")
    print(f"{'v6.3 Top8+等权 (无前视)':<30} {r2['ann_ret']*100:<10.1f} {r2['sharpe']:<10.2f} {r2['max_dd']*100:<10.1f}")
    print(f"{'v6.2+carry 50/50':<30} {'6.5':<10} {'0.90':<10} {'-9.8':<10}")


if __name__ == "__main__":
    main()
