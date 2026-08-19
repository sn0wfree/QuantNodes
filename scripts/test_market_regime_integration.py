"""树模型市场状态检测 — 集成到策略.

将训练好的模型输出 P(震荡) 用于策略权重缩放.
"""
from __future__ import annotations

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig,
)
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

PRODUCTS_10 = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]


def calc_features(prices: pd.DataFrame) -> pd.DataFrame:
    """与训练时相同的特征工程."""
    features = pd.DataFrame(index=prices.index)
    ret = prices.pct_change()
    vol_5 = ret.rolling(5).std()
    vol_10 = ret.rolling(10).std()
    vol_20 = ret.rolling(20).std()
    vol_60 = ret.rolling(60).std()

    features['vol_ratio_5_20'] = (vol_5 / vol_20).mean(axis=1)
    features['vol_ratio_10_60'] = (vol_10 / vol_60).mean(axis=1)
    features['vol_of_vol'] = (vol_20.rolling(20).std() / vol_20).mean(axis=1)
    features['vol_skew'] = ((vol_20.rolling(20).max() - vol_20.rolling(20).min()) / vol_20).mean(axis=1)
    features['vol_breakout'] = (vol_5.iloc[-1] / vol_5.rolling(5).max()).mean() if len(vol_5) > 5 else 1.0
    features['vol_persistence'] = vol_20.rolling(20).apply(
        lambda x: pd.Series(x).autocorr(5) if len(x) > 5 else 0, raw=True
    ).mean(axis=1)

    corr_5 = ret.rolling(5).corr()
    corr_20 = ret.rolling(20).corr()
    corr_60 = ret.rolling(60).corr()
    features['cross_corr_5d'] = corr_5.groupby(level=0).mean().mean(axis=1)
    features['cross_corr_20d'] = corr_20.groupby(level=0).mean().mean(axis=1)
    features['cross_corr_60d'] = corr_60.groupby(level=0).mean().mean(axis=1)
    features['corr_change_5_60'] = (
        (features['cross_corr_5d'] - features['cross_corr_60d']) / features['cross_corr_60d']
    )

    mom_5 = prices.pct_change(5)
    mom_20 = prices.pct_change(20)
    mom_60 = prices.pct_change(60)
    features['mom_dispersion_20d'] = (mom_20.std(axis=1) / mom_20.mean(axis=1).abs())
    features['mom_extreme_pct'] = (mom_20.abs() > 0.1).sum(axis=1) / len(prices.columns)
    features['mom_consistency_5_60'] = (
        (np.sign(mom_5) == np.sign(mom_60)).sum(axis=1) / len(prices.columns)
    )
    features['mom_accel'] = (mom_5 - mom_5.shift(5)).mean(axis=1)
    features['ret_sharpe_20d'] = (ret.rolling(20).mean() / ret.rolling(20).std()).mean(axis=1)

    high = prices.rolling(14).max()
    low = prices.rolling(14).min()
    tr = (high - low).replace(0, np.nan)
    adx_proxy = (prices.diff().abs().rolling(14).mean() / tr).mean(axis=1) * 100
    features['avg_adx_14'] = adx_proxy
    features['adx_change_5_60'] = adx_proxy - adx_proxy.rolling(60).mean()
    features['ma_diff_20'] = (prices / prices.rolling(20).mean() - 1).mean(axis=1)
    features['ma_cross_pct'] = (
        (prices.rolling(5).mean() > prices.rolling(20).mean()).sum(axis=1) / len(prices.columns)
    )
    features['trend_consistency'] = (
        (np.sign(prices - prices.rolling(20).mean()) > 0).sum(axis=1) / len(prices.columns)
    )

    gap = (prices / prices.shift(1) - 1)
    features['gap_freq'] = (gap.abs() > 0.01).sum(axis=1) / len(prices.columns)
    features['gap_avg'] = gap.mean(axis=1).abs()
    intraday_range = (high - low) / prices
    features['intraday_vol'] = intraday_range.mean(axis=1)

    return features


def get_choppy_probability(prices: pd.DataFrame, model_path: str = None) -> pd.Series:
    """获取震荡概率序列.

    使用简单的逻辑回归模型(快速,无需加载pkl).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # 计算特征
    features = calc_features(prices)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.fillna(0))

    # 训练简单模型: 未来20天是否震荡
    lookforward = 20
    label = pd.Series(0, index=prices.index, dtype=int)
    for t in range(len(prices) - lookforward - 1):
        future = prices.iloc[t + 1:t + 1 + lookforward]
        if len(future) < lookforward:
            continue
        avg_ret = future.pct_change().mean().mean()
        max_dd = (future / future.cummax() - 1).min().min()
        if abs(avg_ret) < 0.02 and abs(max_dd) < 0.05:
            label.iloc[t] = 1

    # 滚动训练: 用过去数据预测未来
    choppy_prob = pd.Series(0.5, index=prices.index)

    for t in range(252, len(prices) - lookforward - 1):
        # 用过去252天训练
        train_idx = features.index[max(0, t - 504):t]
        X_train = X_scaled[max(0, t - 504):t]
        y_train = label.loc[train_idx]

        if y_train.nunique() < 2:
            choppy_prob.iloc[t] = 0.5
            continue

        model = LogisticRegression(C=0.1, random_state=42, max_iter=500)
        model.fit(X_train, y_train)

        # 预测当前
        if t < len(X_scaled):
            prob = model.predict_proba(X_scaled[t:t + 1])[:, 1][0]
            choppy_prob.iloc[t] = prob

    # 前252天用固定值0.5（无历史数据训练）
    return choppy_prob


def apply_regime_to_strategy(weights: pd.DataFrame, choppy_prob: pd.Series,
                             threshold_high: float = 0.6,
                             threshold_mid: float = 0.4) -> pd.DataFrame:
    """根据震荡概率调整策略权重.

    - P(震荡) > threshold_high: 权重 × 0.3
    - P(震荡) > threshold_mid: 权重 × 0.6
    - 其他: 权重 × 1.0
    """
    scale = pd.Series(1.0, index=weights.index)
    scale[choppy_prob > threshold_high] = 0.3
    scale[(choppy_prob > threshold_mid) & (choppy_prob <= threshold_high)] = 0.6

    # 对齐索引
    common_idx = weights.index.intersection(scale.index)
    result = weights.copy()
    result.loc[common_idx] = result.loc[common_idx].mul(scale.loc[common_idx], axis=0)
    return result


def test_with_regime():
    """测试加入市场状态检测后的策略表现."""
    print("=" * 80)
    print("测试: 树模型市场状态检测集成")
    print("=" * 80)

    # 加载数据
    cfg = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting='inv_vol', continuous=True, cont_z_window=252, cont_z_clip=3.0,
        margin_call=True, execution_lag=1, aggregation='tvpr',
        commission_rate=0.0002, min_position_change=0.30, slippage_bps=0.5,
        position_ema_alpha=0.15, tail_risk_enabled=True, tail_risk_corr_threshold=0.30,
        enable_cluster_filter=False, execution_price_type='open',
    )

    print("\n[1] 加载数据 + 计算震荡概率...")
    s = FactorModuleComboStrategy(products=PRODUCTS_10, config=cfg, start='2018-01-01', end='2025-12-31')
    s.load_panel()
    prices = s._price_panel.dropna()

    # 计算震荡概率
    import time
    t0 = time.time()
    choppy_prob = get_choppy_probability(prices)
    print(f"  震荡概率计算耗时: {time.time()-t0:.0f}s")
    print(f"  震荡概率范围: {choppy_prob.min():.3f} ~ {choppy_prob.max():.3f}")

    # 统计震荡概率分布
    print(f"\n[2] 震荡概率分布:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pct = (choppy_prob > threshold).mean() * 100
        print(f"  P(震荡)>{threshold}: {pct:.1f}%")

    # 2024年震荡概率
    cp_2024 = choppy_prob[choppy_prob.index.year == 2024]
    print(f"\n[3] 2024年震荡概率:")
    for threshold in [0.3, 0.4, 0.5, 0.6]:
        pct = (cp_2024 > threshold).mean() * 100
        print(f"  P(震荡)>{threshold}: {pct:.1f}%")

    # 测试回测
    # 方案A: 策略权重缩放
    # 方案B: 仓位过滤

    return s, prices, choppy_prob, cfg


def run_backtest_with_filter(s, prices, choppy_prob, cfg, filter_type='weight'):
    """运行带震荡过滤的回测."""
    # Monkey-patch _signal_to_lots方法
    from QuantNodes.strategy.cta.strategies.factor_modules import vol_target_positions
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics
    import numpy as np

    original_signal_to_lots = s._signal_to_lots
    choppy_prob_dict = {'prob': choppy_prob}

    def patched_signal_to_lots(signals, cfg):
        weights = vol_target_positions(
            signals, s._price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
        )
        if s._filter is not None:
            weights = weights * s._filter.coefficients(s._price_panel)
        weights = s._tail_risk_filter(weights)

        # 应用震荡过滤
        prob = choppy_prob_dict['prob']
        common_idx = weights.index.intersection(prob.index)
        scale = pd.Series(1.0, index=weights.index)
        for idx in common_idx:
            p = prob.loc[idx]
            if p > 0.6:
                scale.loc[idx] = 0.3
            elif p > 0.4:
                scale.loc[idx] = 0.6
        weights = weights.mul(scale, axis=0)

        return s._weights_to_lots(weights, cfg)

    s._signal_to_lots = patched_signal_to_lots
    r = s.run()
    m = compute_metrics(r.nav)

    nav_2024 = r.nav[r.nav.index.year == 2024]
    ret_2024 = float(nav_2024.iloc[-1] / nav_2024.iloc[0] - 1) if len(nav_2024) > 10 else 0

    return m, ret_2024


if __name__ == "__main__":
    s, prices, choppy_prob, cfg = test_with_regime()

    print("\n" + "=" * 80)
    print("回测对比")
    print("=" * 80)

    # 基线: 不带震荡过滤
    s_base = FactorModuleComboStrategy(products=PRODUCTS_10, config=cfg, start='2018-01-01', end='2025-12-31')
    s_base.load_panel()
    r_base = s_base.run()
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics
    m_base = compute_metrics(r_base.nav)
    nav_2024_base = r_base.nav[r_base.nav.index.year == 2024]
    ret_2024_base = float(nav_2024_base.iloc[-1] / nav_2024_base.iloc[0] - 1) if len(nav_2024_base) > 10 else 0

    print(f"\n基线 (无regime):")
    print(f"  Sharpe={m_base['Sharpe']:.3f}  AnnRet={m_base['AnnRet']:.1%}  MaxDD={m_base['MaxDD']:.1%}  2024={ret_2024_base:>+.1%}")

    # 测试: 带震荡过滤
    s_test = FactorModuleComboStrategy(products=PRODUCTS_10, config=cfg, start='2018-01-01', end='2025-12-31')
    s_test.load_panel()

    # 预先计算震荡概率
    from QuantNodes.strategy.cta.strategies.factor_modules import vol_target_positions
    import time
    t0 = time.time()
    cp = get_choppy_probability(s_test._price_panel.dropna())
    print(f"\n  震荡概率计算耗时: {time.time()-t0:.0f}s")

    # 注入震荡概率
    choppy_prob_global = {'prob': cp}

    original = s_test._signal_to_lots
    def patched(signals, cfg):
        weights = vol_target_positions(
            signals, s_test._price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
        )
        if s_test._filter is not None:
            weights = weights * s_test._filter.coefficients(s_test._price_panel)
        weights = s_test._tail_risk_filter(weights)

        # 根据震荡概率缩放仓位 (阈值根据概率分布调整)
        prob = choppy_prob_global['prob']
        scale = pd.Series(1.0, index=weights.index)
        for idx in weights.index:
            if idx in prob.index:
                p = prob.loc[idx]
                if p > 0.20:  # 高震荡概率
                    scale.loc[idx] = 0.3
                elif p > 0.10:  # 中等震荡概率
                    scale.loc[idx] = 0.6
        weights = weights.mul(scale, axis=0)

        return s_test._weights_to_lots(weights, cfg)

    s_test._signal_to_lots = patched
    r_test = s_test.run()
    m_test = compute_metrics(r_test.nav)
    nav_2024_test = r_test.nav[r_test.nav.index.year == 2024]
    ret_2024_test = float(nav_2024_test.iloc[-1] / nav_2024_test.iloc[0] - 1) if len(nav_2024_test) > 10 else 0

    print(f"\n+ 树模型震荡过滤 (thresholds: 0.10/0.20):")
    print(f"  Sharpe={m_test['Sharpe']:.3f}  AnnRet={m_test['AnnRet']:.1%}  MaxDD={m_test['MaxDD']:.1%}  2024={ret_2024_test:>+.1%}")

    # 对比
    print(f"\n" + "=" * 80)
    print("对比")
    print("=" * 80)
    print(f"  {'配置':<30} {'Sharpe':>8} {'AnnRet':>8} {'MaxDD':>8} {'2024':>8}")
    print("-" * 65)
    print(f"  {'基线':<30} {m_base['Sharpe']:>8.3f} {m_base['AnnRet']:>7.1%} {m_base['MaxDD']:>7.1%} {ret_2024_base:>+7.1%}")
    print(f"  {'+ 树模型(th=0.10/0.20)':<30} {m_test['Sharpe']:>8.3f} {m_test['AnnRet']:>7.1%} {m_test['MaxDD']:>7.1%} {ret_2024_test:>+7.1%}")

    # 保存震荡概率
    cp.to_frame('choppy_prob').to_parquet('output/market_regime_choppy_prob.parquet')
    print(f"\n震荡概率已保存: output/market_regime_choppy_prob.parquet")
