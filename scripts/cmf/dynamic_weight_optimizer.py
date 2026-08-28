#!/usr/bin/env python3
"""scripts/cmf/dynamic_weight_optimizer.py

铜策略动态权重优化器

功能:
1. 静态权重优化 (最大Sharpe/最小方差/风险平价)
2. 动态平衡 (根据近期表现调整权重)
3. 杠杆策略 (目标收益/目标波动率)
4. 回撤控制 (渐进式降仓)

策略: v6.2, carry, 突破策略, 四象限
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DB = Path("data/cmf/fundamental.duckdb")


def load_strategy_returns(start="2018-01-01", end="2025-12-31"):
    """加载策略的周频收益率."""
    from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6 import build_signal_v6
    from QuantNodes.strategy.commodity_fundamental.copper.copper_carry import build_signal_carry
    from QuantNodes.strategy.commodity_fundamental.copper.copper_breakout import compute_breakout_signal
    from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import compute_inventory_cycle_optimized

    con = duckdb.connect(str(DB), read_only=True)

    # 加载价格
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    close_w = df_price.set_index("trade_date")["close"].resample("W-FRI").last()

    # 加载各策略信号
    v6_signal = build_signal_v6("CU0", con)
    carry_signal = build_signal_carry(con)
    quad_signal = compute_inventory_cycle_optimized()

    # 突破策略 (日频转周频)
    breakout_signal = compute_breakout_signal()
    breakout_ret = breakout_signal["close"].pct_change()
    breakout_pos = breakout_signal["position"]
    breakout_strat = breakout_pos.shift(1) * breakout_ret
    breakout_weekly = breakout_strat.resample("W-FRI").sum()

    # 对齐
    common = v6_signal.index.intersection(carry_signal.index).intersection(
        quad_signal.index
    ).intersection(close_w.index)

    v6_pos = v6_signal.loc[common, "position"].fillna(0)
    carry_pos = carry_signal.loc[common, "position"].fillna(0)
    quad_pos = quad_signal.loc[common, "position"].fillna(0)
    close_w = close_w.reindex(common)

    # 计算收益
    ret = close_w.pct_change()
    cost_bps = 5

    v6_ret = v6_pos.shift(1) * ret - v6_pos.diff().abs() * cost_bps / 10000
    carry_ret = carry_pos.shift(1) * ret - carry_pos.diff().abs() * cost_bps / 10000
    quad_ret = quad_pos.shift(1) * ret - quad_pos.diff().abs() * cost_bps / 10000

    # 组合
    returns = pd.DataFrame({
        "v6.2": v6_ret,
        "carry": carry_ret,
        "突破": breakout_weekly.reindex(common).fillna(0),
        "四象限": quad_ret,
    }).dropna()

    # 截取时间范围
    returns = returns[(returns.index >= start) & (returns.index <= end)]

    return returns


def calc_metrics(returns, weights, leverage=1.0, cost_bps=5.0):
    """计算组合指标."""
    port_ret = returns @ weights * leverage
    nav = (1 + port_ret).cumprod()

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(port_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = port_ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (port_ret > 0).mean()

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "nav": nav,
    }


def negative_sharpe(w, mean_returns, cov_matrix):
    """负Sharpe (用于最小化)."""
    ret = np.dot(w, mean_returns)
    vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    return -ret / vol if vol > 0 else 0


def risk_parity_objective(w, cov_matrix):
    """风险平价目标."""
    vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    if vol == 0:
        return 0
    marginal = np.dot(cov_matrix, w)
    rc = w * marginal / vol
    rc = np.maximum(rc, 1e-10)
    n = len(rc)
    obj = 0
    for i in range(n):
        for j in range(i + 1, n):
            obj += (np.log(rc[i]) - np.log(rc[j])) ** 2
    return obj


def optimize_static_weights(returns):
    """静态权重优化."""
    mean_returns = returns.mean() * 52
    cov_matrix = returns.cov() * 52
    n_assets = len(returns.columns)
    x0 = np.ones(n_assets) / n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n_assets

    results = {}

    # 最大Sharpe
    res = minimize(negative_sharpe, x0, args=(mean_returns.values, cov_matrix.values),
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-10})
    if res.success:
        results["max_sharpe"] = res.x

    # 最小方差
    res = minimize(lambda w: np.sqrt(np.dot(w, np.dot(cov_matrix.values, w))), x0,
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-10})
    if res.success:
        results["min_variance"] = res.x

    # 风险平价
    res = minimize(risk_parity_objective, x0, args=(cov_matrix.values,),
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000, "ftol": 1e-10})
    if res.success:
        results["risk_parity"] = res.x

    # 等权
    results["equal_weight"] = np.ones(n_assets) / n_assets

    # 用户指定
    if n_assets == 4:
        results["user_specified"] = np.array([0.25, 0.30, 0.30, 0.15])

    return results


def calculate_dynamic_weights(returns, lookback_window=52, rebalance_freq="monthly"):
    """计算动态平衡权重.

    参数:
        returns: 策略收益率DataFrame
        lookback_window: 回看窗口 (周)
        rebalance_freq: 调仓频率 ("weekly", "monthly", "quarterly")

    返回: DataFrame with dynamic weights
    """
    n_assets = len(returns.columns)
    weights = pd.DataFrame(0.0, index=returns.index, columns=range(n_assets))

    # 初始化权重
    current_weights = np.ones(n_assets) / n_assets

    # 确定调仓日期
    if rebalance_freq == "weekly":
        rebalance_mask = returns.index.weekday == 4  # 周五
    elif rebalance_freq == "monthly":
        # 每月第一个交易日
        months = returns.index.to_period("M")
        rebalance_mask = pd.Series(False, index=returns.index)
        for i in range(1, len(months)):
            if months[i] != months[i-1]:
                rebalance_mask.iloc[i] = True
    elif rebalance_freq == "quarterly":
        # 每季度第一个交易日
        quarters = returns.index.to_period("Q")
        rebalance_mask = pd.Series(False, index=returns.index)
        for i in range(1, len(quarters)):
            if quarters[i] != quarters[i-1]:
                rebalance_mask.iloc[i] = True
    else:
        rebalance_mask = pd.Series(False, index=returns.index)

    for i in range(lookback_window, len(returns)):
        # 检查是否需要调仓
        if rebalance_mask.iloc[i]:
            # 计算过去lookback_window周的滚动Sharpe
            window_returns = returns.iloc[i-lookback_window:i]
            rolling_sharpe = window_returns.mean() / window_returns.std()
            rolling_sharpe = rolling_sharpe.fillna(0)

            # 根据Sharpe分配权重 (softmax)
            exp_sharpe = np.exp(rolling_sharpe * 2)  # 放大差异
            new_weights = exp_sharpe / exp_sharpe.sum()

            # 平滑过渡 (避免频繁大幅调仓)
            current_weights = current_weights * 0.7 + new_weights.values * 0.3

        weights.iloc[i] = current_weights

    # 填充前lookback_window周
    weights.iloc[:lookback_window] = 1.0 / n_assets

    return weights


def backtest_dynamic_balance(returns, weights, leverage=1.0, cost_bps=5.0):
    """回测动态平衡组合."""
    # 确保列名一致
    weights.columns = returns.columns

    # 计算组合收益
    port_ret = (returns * weights).sum(axis=1) * leverage

    # 计算换手成本
    turnover = weights.diff().abs().sum(axis=1)
    cost = turnover * cost_bps / 10000
    port_ret = port_ret - cost

    # 计算指标
    nav = (1 + port_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(port_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = port_ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min()
    win_rate = (port_ret > 0).mean()

    return {
        "n_weeks": len(port_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "nav": nav,
        "avg_turnover": turnover.mean(),
    }


def main():
    """主函数."""
    print("=" * 60)
    print("铜策略动态权重优化器")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1] 加载策略收益...")
    returns = load_strategy_returns("2018-01-01", "2025-12-31")
    print(f"  数据范围: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"  周数: {len(returns)}")

    # 2. 策略统计
    print(f"\n[2] 策略统计:")
    for col in returns.columns:
        ann_ret = returns[col].mean() * 52
        ann_vol = returns[col].std() * np.sqrt(52)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        print(f"  {col}: 年化={ann_ret:.1%}, 波动={ann_vol:.1%}, Sharpe={sharpe:.2f}")

    # 3. 静态权重优化
    print(f"\n[3] 静态权重优化:")
    static_results = optimize_static_weights(returns)

    for name, weights in static_results.items():
        m = calc_metrics(returns, weights)
        w_str = ", ".join([f"{n}={v:.0%}" for n, v in zip(returns.columns, weights)])
        print(f"  {name}: {w_str}")
        print(f"    年化={m['ann_ret']:.1%}, Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}")

    # 4. 动态平衡测试
    print(f"\n[4] 动态平衡测试:")

    for freq in ["monthly", "quarterly"]:
        for lookback in [26, 52]:
            weights = calculate_dynamic_weights(returns, lookback_window=lookback, rebalance_freq=freq)
            m = backtest_dynamic_balance(returns, weights)
            print(f"  {freq}, 回看{lookback}周: 年化={m['ann_ret']:.1%}, Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}, 换手={m['avg_turnover']:.2f}")

    # 5. 用户指定组合
    print(f"\n[5] 用户指定组合:")
    user_weights = np.array([0.25, 0.30, 0.30, 0.15])
    m_user = calc_metrics(returns, user_weights)
    print(f"  v6.2=25%, carry=30%, 突破=30%, 四象限=15%")
    print(f"  年化={m_user['ann_ret']:.1%}, Sharpe={m_user['sharpe']:.2f}, MaxDD={m_user['max_dd']:.1%}")

    # 6. 杠杆策略
    print(f"\n[6] 杠杆策略:")
    for lev in [1.0, 1.2, 1.3, 1.4, 1.5]:
        m = calc_metrics(returns, user_weights, lev)
        print(f"  杠杆={lev:.1f}x: 年化={m['ann_ret']:.1%}, Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}")

    # 7. 推荐配置
    print(f"\n[7] 推荐配置:")
    print(f"  静态最优: {', '.join([f'{n}={v:.0%}' for n, v in zip(returns.columns, static_results['max_sharpe'])])}")
    m_opt = calc_metrics(returns, static_results["max_sharpe"])
    print(f"    年化={m_opt['ann_ret']:.1%}, Sharpe={m_opt['sharpe']:.2f}, MaxDD={m_opt['max_dd']:.1%}")

    print(f"\n  用户组合+杠杆:")
    m_user_lev = calc_metrics(returns, user_weights, 1.4)
    print(f"    v6.2=25%, carry=30%, 突破=30%, 四象限=15%, 杠杆1.4x")
    print(f"    年化={m_user_lev['ann_ret']:.1%}, Sharpe={m_user_lev['sharpe']:.2f}, MaxDD={m_user_lev['max_dd']:.1%}")

    print(f"\n  动态平衡:")
    weights_dyn = calculate_dynamic_weights(returns, lookback_window=52, rebalance_freq="monthly")
    m_dyn = backtest_dynamic_balance(returns, weights_dyn)
    print(f"    月度调仓, 52周回看")
    print(f"    年化={m_dyn['ann_ret']:.1%}, Sharpe={m_dyn['sharpe']:.2f}, MaxDD={m_dyn['max_dd']:.1%}")

    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
