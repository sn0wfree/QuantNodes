# coding=utf-8
"""scripts/cmf/18_auto_weight_allocation.py

三策略自动权重配置: v6.2 + carry + CTA振荡器Alpha

方法:
    1. 反向波动率 (基础)
    2. 趋势自适应 (Scheme G)
    3. 安全约束 (DD保护+换手上限)
"""
import sys
sys.path.insert(0, ".")

import duckdb
import numpy as np
import pandas as pd
from QuantNodes.strategy.cta.strategies.factor_modules import (
    OscillatorAlphaModuleStrategy, ModuleConfig,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_v6_model import V6Model
from QuantNodes.strategy.commodity_fundamental.copper.copper_carry_model import CarryModel

DB = "data/cmf/fundamental.duckdb"


# ============================================================
# 1. 反向波动率权重
# ============================================================
def inverse_volatility_weights(returns: pd.DataFrame, lookback: int = 52) -> pd.DataFrame:
    """反向波动率权重: w_i = (1/σ_i) / Σ(1/σ_j).
    
    参数:
        returns: (T, N) 策略收益率矩阵
        lookback: 波动率计算窗口
    
    返回:
        (T, N) 权重矩阵, 每行和为1
    """
    vol = returns.rolling(lookback, min_periods=20).std() * np.sqrt(52)
    inv_vol = 1.0 / (vol.replace(0, np.nan))
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights.fillna(1.0 / returns.shape[1])


# ============================================================
# 2. 趋势自适应 (Scheme G)
# ============================================================
def trend_adaptive_weights(returns: pd.DataFrame, lookback: int = 52,
                           trend_window: int = 5, trend_threshold: float = 0.02) -> pd.DataFrame:
    """趋势自适应权重.
    
    逻辑:
        - 基础: 反向波动率
        - 趋势强 (5日收益>2%): 增加高收益策略权重
        - 趋势弱: 增加低风险策略权重
    
    参数:
        returns: (T, N) 策略收益率矩阵
        lookback: 波动率计算窗口
        trend_window: 趋势判断窗口
        trend_threshold: 趋势阈值
    """
    # 基础权重: 反向波动率
    base_weights = inverse_volatility_weights(returns, lookback)
    
    # 趋势信号: 组合收益率的5日累计
    portfolio_ret = returns.sum(axis=1) / returns.shape[1]
    trend = portfolio_ret.rolling(trend_window).sum()
    
    # 趋势调整: 趋势强时微调权重
    # 这里简化处理, 实际可以更复杂
    trend_adj = pd.Series(0.0, index=returns.index)
    trend_adj[trend > trend_threshold] = 0.1   # 趋势强: +10%
    trend_adj[trend < -trend_threshold] = -0.1  # 趋势弱: -10%
    
    # 应用调整 (简化: 均匀调整)
    weights = base_weights.copy()
    for col in weights.columns:
        weights[col] = weights[col] + trend_adj / weights.shape[1]
    
    # 归一化
    weights = weights.clip(lower=0.01)
    weights = weights.div(weights.sum(axis=1), axis=0)
    
    return weights


# ============================================================
# 3. 安全约束
# ============================================================
def apply_safety_constraints(weights: pd.DataFrame, nav: pd.Series,
                             max_turnover: float = 0.30,
                             defensive_floor: float = 0.25,
                             dd_threshold: float = -0.05) -> pd.DataFrame:
    """安全约束.
    
    1. 防御下限: 至少25%在防御策略 (v6.2)
    2. DD保护: 回撤>5%时减半风险
    3. 换手上限: 单次调仓不超过30%
    """
    weights = weights.copy()
    
    # 1. 防御下限: v6.2至少25%
    if "v6.2" in weights.columns:
        weights["v6.2"] = np.maximum(weights["v6.2"], defensive_floor)
        weights = weights.div(weights.sum(axis=1), axis=0)
    
    # 2. DD保护
    dd = (nav / nav.cummax() - 1).min()
    if dd < dd_threshold:
        # 回撤超阈值, 减半风险
        weights = weights * 0.5
        weights["v6.2"] = weights["v6.2"] + 0.25  # 加到防御
        weights = weights.div(weights.sum(axis=1), axis=0)
    
    # 3. 换手上限
    weight_change = weights.diff().abs().sum(axis=1)
    excess = weight_change - max_turnover
    mask = excess > 0
    if mask.any():
        scale = 1.0 - excess[mask] / weight_change[mask]
        weights.loc[mask] = weights.loc[mask].mul(scale, axis=0)
        weights = weights.div(weights.sum(axis=1), axis=0)
    
    return weights


# ============================================================
# 主程序
# ============================================================
def main():
    con = duckdb.connect(DB, read_only=True)
    
    # 加载价格
    df = pd.read_sql("SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date", con)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    cw = df.set_index('trade_date')['close']
    ret_w = cw.pct_change()
    
    print("=" * 70)
    print("三策略自动权重配置测试")
    print("=" * 70)
    
    # 1. 计算各策略NAV
    print("\n1. 计算各策略NAV")
    
    # v6.2
    model_v6 = V6Model(version="v6.2", threshold=3, bull_mom_weeks=13)
    sig_v6 = model_v6.build_signal(con)
    p_v6 = sig_v6["position"]
    strat_v6 = p_v6.shift(1) * ret_w - p_v6.diff().abs() * 5e-4
    nav_v6 = (1 + strat_v6.dropna()).cumprod()
    
    # carry
    model_carry = CarryModel(z_window=26, mom_window=4, min_duration=6)
    sig_carry = model_carry.build_signal(con)
    p_carry = sig_carry["position"]
    strat_car = p_carry.shift(1) * ret_w - p_carry.diff().abs() * 5e-4
    nav_car = (1 + strat_car.dropna()).cumprod()
    
    # CTA
    config = ModuleConfig(target_vol=0.20, max_leverage=2.0, aggregation="inv_vol",
        continuous=True, execution_lag=1, position_ema_alpha=0.0,
        min_position_change=0.30, slippage_bps=0.0)
    strat_cta = OscillatorAlphaModuleStrategy(products=["CU"], start="2018-01-01", end="2025-12-31", config=config)
    result_cta = strat_cta.run()
    nav_cta = result_cta.nav
    
    # 构建收益率矩阵
    all_dates = nav_v6.index.intersection(nav_car.index).intersection(nav_cta.index)
    ret_matrix = pd.DataFrame({
        "v6.2": nav_v6.pct_change().reindex(all_dates),
        "carry": nav_car.pct_change().reindex(all_dates),
        "CTA": nav_cta.pct_change().reindex(all_dates),
    }).dropna()
    
    # 2. 测试不同权重方法
    print("\n2. 权重方法对比")
    
    def calc_portfolio_metrics(weights, nav_dict, label):
        """计算组合指标."""
        nav_list = [nav_dict[col].reindex(weights.index) for col in weights.columns]
        nav_array = np.array([n.values for n in nav_list]).T
        portfolio_nav = (weights.values * nav_array).sum(axis=1)
        portfolio_nav = pd.Series(portfolio_nav, index=weights.index)
        
        total_ret = portfolio_nav.iloc[-1] / portfolio_nav.iloc[0] - 1
        n_years = (portfolio_nav.index[-1] - portfolio_nav.index[0]).days / 365.25
        ann = (1 + total_ret) ** (1 / n_years) - 1
        ret = portfolio_nav.pct_change().dropna()
        vol = ret.std() * np.sqrt(252)
        sh = ann / vol if vol > 0 else 0
        dd = (portfolio_nav / portfolio_nav.cummax() - 1).min()
        
        print(f"{label:<35} ann={ann*100:5.1f}%  sh={sh:.2f}  dd={dd*100:5.1f}%")
        return {"ann": ann, "sharpe": sh, "max_dd": dd, "nav": portfolio_nav}
    
    nav_dict = {"v6.2": nav_v6, "carry": nav_car, "CTA": nav_cta}
    
    # 方法1: 等权
    equal_weights = pd.DataFrame(1/3, index=ret_matrix.index, columns=ret_matrix.columns)
    r_equal = calc_portfolio_metrics(equal_weights, nav_dict, "等权 (1/3 each)")
    
    # 方法2: 反向波动率
    inv_vol_weights = inverse_volatility_weights(ret_matrix, lookback=52)
    r_invvol = calc_portfolio_metrics(inv_vol_weights, nav_dict, "反向波动率")
    
    # 方法3: 趋势自适应
    trend_weights = trend_adaptive_weights(ret_matrix, lookback=52, trend_window=5)
    r_trend = calc_portfolio_metrics(trend_weights, nav_dict, "趋势自适应")
    
    # 方法4: 安全约束
    nav_equal = r_equal["nav"]
    safe_weights = apply_safety_constraints(equal_weights, nav_equal)
    r_safe = calc_portfolio_metrics(safe_weights, nav_dict, "安全约束")
    
    # 方法5: 趋势+安全
    nav_trend = r_trend["nav"]
    trend_safe_weights = apply_safety_constraints(trend_weights, nav_trend)
    r_trend_safe = calc_portfolio_metrics(trend_safe_weights, nav_dict, "趋势+安全")
    
    # 3. 固定权重对比
    print("\n3. 固定权重对比")
    fixed_configs = [
        (0.6, 0.2, 0.2, "60/20/20 (v6主导)"),
        (0.5, 0.25, 0.25, "50/25/25 (均衡)"),
        (0.4, 0.3, 0.3, "40/30/30 (CTA增加)"),
    ]
    
    for w1, w2, w3, name in fixed_configs:
        fixed_weights = pd.DataFrame({"v6.2": w1, "carry": w2, "CTA": w3}, index=ret_matrix.index)
        r_fixed = calc_portfolio_metrics(fixed_weights, nav_dict, name)
    
    # 4. Buy&Hold
    print("\n4. Buy&Hold基准")
    nav_bh = (1 + ret_w).cumprod()
    nav_bh = nav_bh.reindex(all_dates)
    r_bh = calc_portfolio_metrics(pd.DataFrame(1.0, index=nav_bh.index, columns=["B&H"]), 
                                  {"B&H": nav_bh}, "Buy&Hold")
    
    # 5. 分年对比
    print("\n5. 分年对比")
    print(f"{'年份':<8} {'等权':<10} {'反向波动':<10} {'趋势+安全':<10} {'B&H':<10}")
    print("-" * 48)
    
    for y in range(2018, 2026):
        y_eq = r_equal["nav"][r_equal["nav"].index.year == y]
        y_iv = r_invvol["nav"][r_invvol["nav"].index.year == y]
        y_ts = r_trend_safe["nav"][r_trend_safe["nav"].index.year == y]
        y_bh = nav_bh[nav_bh.index.year == y]
        
        r_eq = (y_eq.iloc[-1]/y_eq.iloc[0]-1)*100 if len(y_eq)>1 else 0
        r_iv = (y_iv.iloc[-1]/y_iv.iloc[0]-1)*100 if len(y_iv)>1 else 0
        r_ts = (y_ts.iloc[-1]/y_ts.iloc[0]-1)*100 if len(y_ts)>1 else 0
        r_bh = (y_bh.iloc[-1]/y_bh.iloc[0]-1)*100 if len(y_bh)>1 else 0
        
        print(f"{y:<8} {r_eq:<10.1f} {r_iv:<10.1f} {r_ts:<10.1f} {r_bh:<10.1f}")
    
    # 6. 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"""
各方法对比:
  等权:      ann={r_equal['ann']*100:.1f}%, Sharpe={r_equal['sharpe']:.2f}, MaxDD={r_equal['max_dd']*100:.1f}%
  反向波动率: ann={r_invvol['ann']*100:.1f}%, Sharpe={r_invvol['sharpe']:.2f}, MaxDD={r_invvol['max_dd']*100:.1f}%
  趋势自适应: ann={r_trend['ann']*100:.1f}%, Sharpe={r_trend['sharpe']:.2f}, MaxDD={r_trend['max_dd']*100:.1f}%
  安全约束:  ann={r_safe['ann']*100:.1f}%, Sharpe={r_safe['sharpe']:.2f}, MaxDD={r_safe['max_dd']*100:.1f}%
  趋势+安全: ann={r_trend_safe['ann']*100:.1f}%, Sharpe={r_trend_safe['sharpe']:.2f}, MaxDD={r_trend_safe['max_dd']*100:.1f}%
  B&H:      ann={r_bh['ann']*100:.1f}%, Sharpe={r_bh['sharpe']:.2f}, MaxDD={r_bh['max_dd']*100:.1f}%
""")


if __name__ == "__main__":
    main()
