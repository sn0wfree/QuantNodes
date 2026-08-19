"""树模型市场状态检测 — 完整验证.

目标:
1. 用2018-2025全部数据进行滚动训练/测试
2. 评估每个模型的AUC
3. 调优阈值
4. 输出: 最优配置
"""
from __future__ import annotations

import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig,
)

PRODUCTS_10 = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]


def calc_features(prices: pd.DataFrame) -> pd.DataFrame:
    """计算特征."""
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


def make_label(prices: pd.DataFrame, lookforward: int = 20) -> pd.Series:
    """标签: 未来N天是否震荡."""
    label = pd.Series(0, index=prices.index, dtype=int)
    for t in range(len(prices) - lookforward - 1):
        future = prices.iloc[t + 1:t + 1 + lookforward]
        if len(future) < lookforward:
            continue
        avg_ret = future.pct_change().mean().mean()
        max_dd = (future / future.cummax() - 1).min().min()
        if abs(avg_ret) < 0.02 and abs(max_dd) < 0.05:
            label.iloc[t] = 1
    return label


def rolling_train_predict(prices, features, label, train_window=504, window=20):
    """滚动训练: 每个时间点用过去train_window天训练模型,预测当前."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    models = {
        'LogisticRegression': lambda: LogisticRegression(C=0.1, random_state=42, max_iter=500),
        'RandomForest': lambda: RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_leaf=20, random_state=42),
        'DecisionTree': lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42),
    }

    results = {}
    for mname, make_model in models.items():
        print(f"\n  训练 {mname}...")
        
        all_probs = pd.Series(0.5, index=features.index)
        valid_count = 0
        
        for t in range(train_window, len(features) - window - 1):
            # 训练窗口
            train_idx = features.index[t - train_window:t]
            X_train = features.loc[train_idx].values
            y_train = label.loc[train_idx].values
            
            if len(np.unique(y_train)) < 2:
                all_probs.iloc[t] = 0.5
                continue
            
            # 标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # 训练
            model = make_model()
            model.fit(X_train_scaled, y_train)
            
            # 预测
            X_test = features.iloc[t:t+1].values
            X_test_scaled = scaler.transform(X_test)
            prob = model.predict_proba(X_test_scaled)[:, 1][0]
            all_probs.iloc[t] = prob
            valid_count += 1
        
        # 计算AUC
        valid_mask = (label > 0) | (label == 0)
        common_idx = all_probs.index.intersection(label.index)
        y_true = label.loc[common_idx]
        y_prob = all_probs.loc[common_idx]
        
        # 只用有标签的区间
        mask = (y_true.index >= features.index[train_window]) & (y_true.index < features.index[-window-1])
        y_true_valid = y_true[mask]
        y_prob_valid = y_prob[mask]
        
        if len(np.unique(y_true_valid)) >= 2:
            auc = roc_auc_score(y_true_valid, y_prob_valid)
        else:
            auc = 0.5
        
        print(f"    AUC={auc:.3f}  有效预测={valid_count}")
        results[mname] = {'probs': all_probs, 'auc': auc}
    
    return results


def optimize_thresholds(all_probs, label, lookforward=20):
    """阈值调优: 测试不同阈值组合."""
    results = []
    
    for high_th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        for low_th in [0.05, 0.10, 0.15]:
            if high_th <= low_th:
                continue
            
            # 模拟仓位缩放
            scale = pd.Series(1.0, index=all_probs.index)
            scale[all_probs > high_th] = 0.3
            scale[(all_probs > low_th) & (all_probs <= high_th)] = 0.6
            
            # 计算2024年表现
            mask_2024 = all_probs.index.year == 2024
            scale_2024 = scale[mask_2024].mean()
            
            # 计算震荡期正确率
            mask_test = label.index >= all_probs.index[504]
            true_choppy = label[mask_test]
            pred_choppy = all_probs[mask_test] > high_th
            if (true_choppy.sum() + pred_choppy.sum()) > 0:
                precision = (true_choppy & pred_choppy).sum() / pred_choppy.sum() if pred_choppy.sum() > 0 else 0
            else:
                precision = 0
            
            results.append({
                'high_th': high_th,
                'low_th': low_th,
                'avg_scale_2024': scale_2024,
                'precision': precision,
            })
    
    return results


def main():
    print("=" * 80)
    print("树模型市场状态检测 — 完整验证")
    print("=" * 80)

    # 加载数据
    print("\n[1] 加载数据...")
    cfg = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting='inv_vol', continuous=True, cont_z_window=252, cont_z_clip=3.0,
        margin_call=True, execution_lag=1, aggregation='tvpr',
        commission_rate=0.0002, min_position_change=0.30, slippage_bps=0.5,
        position_ema_alpha=0.15, tail_risk_enabled=True, tail_risk_corr_threshold=0.30,
        enable_cluster_filter=False, execution_price_type='open',
    )
    s = FactorModuleComboStrategy(products=PRODUCTS_10, config=cfg, start='2018-01-01', end='2025-12-31')
    s.load_panel()
    prices = s._price_panel.dropna()
    print(f"  数据: {prices.shape[0]} 天, {prices.shape[1]} 品种")

    # 计算特征和标签
    print("\n[2] 计算特征和标签...")
    features = calc_features(prices)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 多窗口标签
    print("\n  标签分布:")
    for window in [10, 20, 30]:
        label = make_label(prices, lookforward=window)
        print(f"    {window}天窗口: 震荡占比={label.mean():.1%}")

    # 用20天窗口
    label = make_label(prices, lookforward=20)

    # 滚动训练
    print("\n[3] 滚动训练...")
    train_results = rolling_train_predict(prices, features, label, train_window=504, window=20)

    # 阈值调优
    print("\n[4] 阈值调优...")
    best_results = {}
    for mname, res in train_results.items():
        print(f"\n  {mname} 阈值调优:")
        thresholds = optimize_thresholds(res['probs'], label, lookforward=20)
        
        # 按精度排序
        thresholds.sort(key=lambda x: -x['precision'])
        
        print(f"    最佳精度配置:")
        for t in thresholds[:3]:
            print(f"      high={t['high_th']:.2f} low={t['low_th']:.2f}  "
                  f"精度={t['precision']:.3f}  2024平均仓位={t['avg_scale_2024']:.2f}")
        
        best_results[mname] = thresholds[0]

    # 总结
    print("\n" + "=" * 80)
    print("模型对比总结")
    print("=" * 80)
    print(f"  {'模型':<25} {'AUC':>8} {'最佳高阈值':>12} {'最佳低阈值':>12} {'精度':>8}")
    print("-" * 65)
    for mname, res in train_results.items():
        best = best_results[mname]
        print(f"  {mname:<25} {res['auc']:>8.3f} {best['high_th']:>12.2f} {best['low_th']:>12.2f} {best['precision']:>8.3f}")

    # 保存
    with open('output/market_regime_model_results.pkl', 'wb') as f:
        pickle.dump({
            'train_results': train_results,
            'best_results': best_results,
        }, f)
    print(f"\n结果已保存: output/market_regime_model_results.pkl")


if __name__ == "__main__":
    main()
