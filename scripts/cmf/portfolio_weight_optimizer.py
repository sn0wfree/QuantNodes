#!/usr/bin/env python3
"""scripts/cmf/portfolio_weight_optimizer.py

铜策略权重优化器

优化方法:
    1. 最大Sharpe: max(μᵀw / √(wᵀΣw))
    2. 最小方差: min(wᵀΣw)
    3. 风险平价: 各策略风险贡献相等
    4. 等权: w = [1/3, 1/3, 1/3]

约束:
    - 权重和为1
    - 每个权重在[0, 1]之间

数据范围: 2018-01-01 ~ 2025-12-31
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
    """加载策略的周频收益率.

    返回: DataFrame with columns=[v6.2, 四象限, carry, 动量, AU-CU比, 突破策略], index=weekly dates
    """
    from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6 import build_signal_v6
    from QuantNodes.strategy.commodity_fundamental.copper.copper_inventory_cycle import compute_inventory_cycle_optimized
    from QuantNodes.strategy.commodity_fundamental.copper.copper_carry import build_signal_carry
    from QuantNodes.strategy.commodity_fundamental.copper.copper_momentum import compute_momentum_signal
    from QuantNodes.strategy.commodity_fundamental.copper.copper_breakout import (
        compute_breakout_signal,
    )

    con = duckdb.connect(str(DB), read_only=True)

    # 加载价格
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    close_w = df_price.set_index("trade_date")["close"].resample("W-FRI").last()

    # 加载各策略信号
    v6_signal = build_signal_v6("CU0", con)
    quad_signal = compute_inventory_cycle_optimized()
    carry_signal = build_signal_carry(con)
    mom_signal = compute_momentum_signal(con=con, lookback=26)

    # 突破策略 (日频转周频)
    breakout_signal = compute_breakout_signal()
    breakout_ret = breakout_signal["close"].pct_change()
    breakout_pos = breakout_signal["position"]
    breakout_strat = breakout_pos.shift(1) * breakout_ret
    breakout_weekly = breakout_strat.resample("W-FRI").sum()

    # 对齐
    common = v6_signal.index.intersection(quad_signal.index).intersection(
        carry_signal.index
    ).intersection(mom_signal.index).intersection(close_w.index)

    v6_pos = v6_signal.loc[common, "position"].fillna(0)
    quad_pos = quad_signal.loc[common, "position"].fillna(0)
    carry_pos = carry_signal.loc[common, "position"].fillna(0)
    mom_pos = mom_signal.loc[common, "position"].fillna(0)
    close_w = close_w.reindex(common)

    # 计算收益
    ret = close_w.pct_change()
    cost_bps = 5

    v6_ret = v6_pos.shift(1) * ret - v6_pos.diff().abs() * cost_bps / 10000
    quad_ret = quad_pos.shift(1) * ret - quad_pos.diff().abs() * cost_bps / 10000
    carry_ret = carry_pos.shift(1) * ret - carry_pos.diff().abs() * cost_bps / 10000
    mom_ret = mom_pos.shift(1) * ret - mom_pos.diff().abs() * cost_bps / 10000

    # 组合
    returns = pd.DataFrame({
        "v6.2": v6_ret,
        "四象限": quad_ret,
        "carry": carry_ret,
        "动量": mom_ret,
    }).dropna()

    # 添加突破策略
    breakout_weekly = breakout_weekly.reindex(returns.index).fillna(0)
    returns["突破策略"] = breakout_weekly

    # 截取时间范围
    returns = returns[(returns.index >= start) & (returns.index <= end)]

    return returns


def portfolio_return(w, mean_returns):
    """组合收益."""
    return np.dot(w, mean_returns)


def portfolio_volatility(w, cov_matrix):
    """组合波动率."""
    return np.sqrt(np.dot(w, np.dot(cov_matrix, w)))


def negative_sharpe(w, mean_returns, cov_matrix):
    """负Sharpe (用于最小化)."""
    ret = portfolio_return(w, mean_returns)
    vol = portfolio_volatility(w, cov_matrix)
    if vol == 0:
        return 0
    return -ret / vol


def risk_contribution(w, cov_matrix):
    """各策略风险贡献."""
    vol = portfolio_volatility(w, cov_matrix)
    if vol == 0:
        return np.zeros(len(w))
    marginal = np.dot(cov_matrix, w)
    return w * marginal / vol


def risk_parity_objective(w, cov_matrix):
    """风险平价目标: 各策略风险贡献相等.

    使用对数形式避免数值问题:
    min Σ (ln(rc_i) - ln(rc_j))^2 for all i < j
    """
    rc = risk_contribution(w, cov_matrix)
    # 避免零风险贡献
    rc = np.maximum(rc, 1e-10)
    n = len(rc)
    obj = 0
    for i in range(n):
        for j in range(i + 1, n):
            obj += (np.log(rc[i]) - np.log(rc[j])) ** 2
    return obj


def max_drawdown(returns):
    """计算最大回撤."""
    nav = (1 + returns).cumprod()
    dd = nav / nav.cummax() - 1
    return dd.min()


def calculate_metrics(returns, weights, leverage=1.0, vol_target=None, max_dd_limit=None, max_leverage=1.5):
    """计算组合指标.

    参数:
        returns: 策略收益率DataFrame
        weights: 权重数组
        leverage: 杠杆倍数
        vol_target: 目标波动率 (可选，用于动态杠杆)
        max_dd_limit: 最大回撤限制 (可选，用于风险控制)
        max_leverage: 最大杠杆限制 (默认1.5倍)
    """
    # 限制杠杆
    leverage = min(leverage, max_leverage)

    port_ret = returns @ weights * leverage

    # 动态风控: 波动率目标化
    if vol_target is not None:
        rolling_vol = port_ret.rolling(52).std() * np.sqrt(52)
        dynamic_leverage = (vol_target / rolling_vol).fillna(1.0).clip(upper=max_leverage)
        port_ret = returns @ weights * dynamic_leverage
        leverage_info = f"动态 (目标{vol_target:.0%}, 上限{max_leverage:.1f}x)"
    else:
        leverage_info = f"{leverage:.2f}x"

    nav = (1 + port_ret).cumprod()

    # 动态风控: 回撤保护
    if max_dd_limit is not None:
        drawdown = nav / nav.cummax() - 1
        # 回撤超限时降低仓位 (渐进式风控)
        risk_scale = pd.Series(1.0, index=port_ret.index)
        risk_scale[drawdown < max_dd_limit * 0.5] = 0.7  # 回撤达50%限制时降仓30%
        risk_scale[drawdown < max_dd_limit * 0.8] = 0.5  # 回撤达80%限制时降仓50%
        risk_scale[drawdown < max_dd_limit] = 0.3        # 回撤达限制时降仓70%
        port_ret = port_ret * risk_scale
        nav = (1 + port_ret).cumprod()

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(port_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = port_ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = max_drawdown(port_ret)
    win_rate = (port_ret > 0).mean()

    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "nav": nav,
        "leverage": leverage_info,
    }


def optimize_portfolio(start="2018-01-01", end="2025-12-31"):
    """主优化函数.

    返回: dict with optimization results
    """
    print("=" * 60)
    print("铜策略权重优化器")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1] 加载策略收益 ({start} ~ {end})...")
    returns = load_strategy_returns(start, end)
    print(f"  数据范围: {returns.index[0].date()} ~ {returns.index[-1].date()}")
    print(f"  周数: {len(returns)}")

    # 2. 计算统计量
    mean_returns = returns.mean() * 52  # 年化收益
    cov_matrix = returns.cov() * 52  # 年化协方差
    corr_matrix = returns.corr()

    print(f"\n[2] 策略统计:")
    print(f"\n  年化收益:")
    for col in returns.columns:
        print(f"    {col}: {mean_returns[col]:.1%}")

    print(f"\n  相关性矩阵:")
    print(f"  {'':10} {'v6.2':>8} {'四象限':>8} {'carry':>8}")
    for i, row in enumerate(corr_matrix.index):
        vals = [f"{corr_matrix.iloc[i, j]:.2f}" for j in range(3)]
        print(f"  {row:10} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8}")

    # 3. 定义约束
    n_assets = len(returns.columns)
    x0 = np.ones(n_assets) / n_assets  # 初始权重: 等权
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n_assets

    results = {}

    # 4. 最大Sharpe
    print(f"\n[3] 优化中...")
    res = minimize(
        negative_sharpe, x0, args=(mean_returns.values, cov_matrix.values),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10}
    )
    if res.success:
        results["max_sharpe"] = res.x
        print(f"  最大Sharpe: {res.x}")
    else:
        print(f"  最大Sharpe优化失败: {res.message}")

    # 5. 最小方差
    res = minimize(
        portfolio_volatility, x0, args=(cov_matrix.values,),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10}
    )
    if res.success:
        results["min_variance"] = res.x
        print(f"  最小方差: {res.x}")

    # 6. 风险平价
    res = minimize(
        risk_parity_objective, x0, args=(cov_matrix.values,),
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10}
    )
    if res.success:
        results["risk_parity"] = res.x
        print(f"  风险平价: {res.x}")

    # 7. 等权
    results["equal_weight"] = np.ones(n_assets) / n_assets
    print(f"  等权: {results['equal_weight']}")

    # 8. 最大Sharpe + 杠杆 (目标10%收益)
    print(f"\n[4] 杠杆优化 (目标10%收益, 最大杠杆1.5x)...")
    if "max_sharpe" in results:
        base_weights = results["max_sharpe"]
        base_metrics = calculate_metrics(returns, base_weights)

        # 计算所需杠杆 (限制最大1.5倍)
        target_ret = 0.10
        if base_metrics["ann_ret"] > 0:
            required_leverage = target_ret / base_metrics["ann_ret"]
            required_leverage = min(required_leverage, 1.5)  # 最大1.5倍杠杆
        else:
            required_leverage = 1.0

        # 杠杆 + 回撤控制
        results["leveraged"] = {
            "weights": base_weights,
            "leverage": required_leverage,
            "max_dd_limit": -0.12,  # 最大回撤限制-12%
        }
        print(f"  所需杠杆: {required_leverage:.2f}x")

    # 9. 动态杠杆 (波动率目标)
    if "max_sharpe" in results:
        results["dynamic_leverage"] = {
            "weights": base_weights,
            "leverage": 1.0,
            "vol_target": 0.10,  # 目标10%波动率
            "max_leverage": 1.5,  # 最大1.5倍杠杆
        }
        print(f"  动态杠杆: 目标10%波动率, 最大1.5x")

    return results, returns, corr_matrix


def print_results(results, returns):
    """打印优化结果."""
    print("\n" + "=" * 60)
    print("优化结果")
    print("=" * 60)

    strategy_names = returns.columns.tolist()

    for method, params in results.items():
        if isinstance(params, dict):
            weights = params["weights"]
            leverage = params.get("leverage", 1.0)
            vol_target = params.get("vol_target")
            max_dd_limit = params.get("max_dd_limit")
        else:
            weights = params
            leverage = 1.0
            vol_target = None
            max_dd_limit = None

        metrics = calculate_metrics(returns, weights, leverage, vol_target, max_dd_limit)

        print(f"\n{method}:")
        print(f"  权重: {', '.join([f'{n}={w:.0%}' for n, w in zip(strategy_names, weights)])}")
        print(f"  杠杆: {metrics['leverage']}")
        print(f"  年化: {metrics['ann_ret']:.1%}")
        print(f"  Sharpe: {metrics['sharpe']:.2f}")
        print(f"  MaxDD: {metrics['max_dd']:.1%}")
        print(f"  胜率: {metrics['win_rate']:.1%}")

    # 找出最优 (考虑杠杆后的Sharpe)
    best_method = None
    best_sharpe = -np.inf
    for method, params in results.items():
        if isinstance(params, dict):
            weights = params["weights"]
            leverage = params.get("leverage", 1.0)
            vol_target = params.get("vol_target")
            max_dd_limit = params.get("max_dd_limit")
        else:
            weights = params
            leverage = 1.0
            vol_target = None
            max_dd_limit = None

        metrics = calculate_metrics(returns, weights, leverage, vol_target, max_dd_limit)
        if metrics["sharpe"] > best_sharpe:
            best_sharpe = metrics["sharpe"]
            best_method = method

    if best_method:
        params = results[best_method]
        if isinstance(params, dict):
            weights = params["weights"]
            leverage = params.get("leverage", 1.0)
            vol_target = params.get("vol_target")
            max_dd_limit = params.get("max_dd_limit")
        else:
            weights = params
            leverage = 1.0
            vol_target = None
            max_dd_limit = None

        best_metrics = calculate_metrics(returns, weights, leverage, vol_target, max_dd_limit)

        print("\n" + "=" * 60)
        print(f"推荐: {best_method}")
        print(f"  权重: {', '.join([f'{n}={w:.0%}' for n, w in zip(strategy_names, weights)])}")
        print(f"  杠杆: {best_metrics['leverage']}")
        print(f"  年化: {best_metrics['ann_ret']:.1%}, Sharpe: {best_metrics['sharpe']:.2f}, MaxDD: {best_metrics['max_dd']:.1%}")
        print("=" * 60)

    return best_method, weights, best_metrics


def save_results(results, returns, output_dir="reports/cmf"):
    """保存结果."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    strategy_names = returns.columns.tolist()

    # 保存权重
    weights_data = {}
    for method, params in results.items():
        if isinstance(params, dict):
            weights_data[method] = params["weights"]
        else:
            weights_data[method] = params

    weights_df = pd.DataFrame(weights_data, index=strategy_names)
    weights_path = output_path / "portfolio_weights.csv"
    weights_df.to_csv(weights_path)
    print(f"\n权重已保存: {weights_path}")

    # 保存净值
    nav_df = pd.DataFrame()
    for method, params in results.items():
        if isinstance(params, dict):
            weights = params["weights"]
            leverage = params.get("leverage", 1.0)
            vol_target = params.get("vol_target")
            max_dd_limit = params.get("max_dd_limit")
        else:
            weights = params
            leverage = 1.0
            vol_target = None
            max_dd_limit = None

        metrics = calculate_metrics(returns, weights, leverage, vol_target, max_dd_limit)
        nav_df[method] = metrics["nav"]

    nav_path = output_path / "portfolio_nav.csv"
    nav_df.to_csv(nav_path)
    print(f"净值已保存: {nav_path}")


def plot_results(results, returns, output_dir="reports/cmf"):
    """画图对比."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=100)

    # 图1: 净值曲线
    ax1 = axes[0]
    colors = {"max_sharpe": "#1f77b4", "min_variance": "#ff7f0e",
              "risk_parity": "#2ca02c", "equal_weight": "#7f7f7f",
              "leveraged": "#9467bd", "dynamic_leverage": "#8c564b"}

    for method, params in results.items():
        if isinstance(params, dict):
            weights = params["weights"]
            leverage = params.get("leverage", 1.0)
            vol_target = params.get("vol_target")
            max_dd_limit = params.get("max_dd_limit")
        else:
            weights = params
            leverage = 1.0
            vol_target = None
            max_dd_limit = None

        metrics = calculate_metrics(returns, weights, leverage, vol_target, max_dd_limit)
        ax1.plot(metrics["nav"].index, metrics["nav"].values,
                 label=f"{method} (Sharpe={metrics['sharpe']:.2f})",
                 linewidth=1.5, color=colors.get(method))

    ax1.set_title("铜策略组合净值对比 (2018-2025)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("净值 (归一化)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 图2: 权重对比
    ax2 = axes[1]
    strategy_names = returns.columns.tolist()
    x = np.arange(len(results))
    width = 0.25

    for i, name in enumerate(strategy_names):
        weights = []
        for method, params in results.items():
            if isinstance(params, dict):
                weights.append(params["weights"][i])
            else:
                weights.append(params[i])
        ax2.bar(x + i * width, weights, width, label=name)

    ax2.set_xlabel("优化方法", fontsize=12)
    ax2.set_ylabel("权重", fontsize=12)
    ax2.set_title("各优化方法权重配置", fontsize=14, fontweight="bold")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(results.keys())
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # 保存图片
    plot_path = output_path / "portfolio_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"图片已保存: {plot_path}")

    plt.show()


def main():
    """主函数."""
    # 运行优化
    results, returns, corr_matrix = optimize_portfolio(
        start="2018-01-01", end="2025-12-31"
    )

    # 打印结果
    best_method, best_weights, best_metrics = print_results(results, returns)

    # 保存结果
    save_results(results, returns)

    # 画图
    plot_results(results, returns)

    return results, best_method, best_weights


if __name__ == "__main__":
    main()
