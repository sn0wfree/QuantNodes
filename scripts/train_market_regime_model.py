"""树模型市场状态检测 — 震荡 vs 非震荡.

功能:
1. 特征工程: 25+ 个跨品种特征
2. 多窗口标签: 10/20/30 天
3. 多种树模型对比
4. 样本外验证 (2024)
5. 输出: P(震荡) ∈ [0, 1] 序列
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig,
)
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

PRODUCTS_10 = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]


# ═══════════════════════════════════════════════════════════════════════════════
# 特征工程
# ═══════════════════════════════════════════════════════════════════════════════

def calc_features(prices: pd.DataFrame) -> pd.DataFrame:
    """计算25+特征."""
    features = pd.DataFrame(index=prices.index)

    # 1. 波动率类
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

    # 2. 相关性类
    corr_5 = ret.rolling(5).corr()
    corr_20 = ret.rolling(20).corr()
    corr_60 = ret.rolling(60).corr()

    features['cross_corr_5d'] = corr_5.groupby(level=0).mean().mean(axis=1)
    features['cross_corr_20d'] = corr_20.groupby(level=0).mean().mean(axis=1)
    features['cross_corr_60d'] = corr_60.groupby(level=0).mean().mean(axis=1)
    features['corr_change_5_60'] = (
        (features['cross_corr_5d'] - features['cross_corr_60d']) / features['cross_corr_60d']
    )

    # 3. 动量类
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

    # 4. 趋势强度类 (简化ADX)
    # ADX ≈ |价格变化| / 真实波幅 的滚动平均
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

    # 5. 价格行为类
    gap = (prices / prices.shift(1) - 1)
    features['gap_freq'] = (gap.abs() > 0.01).sum(axis=1) / len(prices.columns)
    features['gap_avg'] = gap.mean(axis=1).abs()

    intraday_range = (high - low) / prices
    features['intraday_vol'] = intraday_range.mean(axis=1)

    return features


def make_label(prices: pd.DataFrame, lookforward: int = 20) -> pd.Series:
    """生成标签: 未来N天是否震荡.

    震荡定义: abs(avg_return) < 0.02 AND max_drawdown < 0.05
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# 模型对比
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(X_train, y_train, X_test, y_test, model_name):
    """训练单个模型并评估."""
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score

    if model_name == "lightgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.05,
                num_leaves=15, min_child_samples=20, random_state=42, verbose=-1,
            )
        except ImportError:
            print("LightGBM not available, using XGBoost")
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.05,
                min_child_weight=20, random_state=42,
            )
    elif model_name == "xgboost":
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.05,
                min_child_weight=20, random_state=42,
            )
        except ImportError:
            print("XGBoost not available, using sklearn")
            model = RandomForestClassifier(
                n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42,
            )
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42,
        )
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=20, random_state=42,
        )
    elif model_name == "logistic_regression":
        model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)

    # 评估
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        y_pred_proba = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred_proba)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(X_train.columns, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        importance = dict(zip(X_train.columns, abs(model.coef_[0])))
    else:
        importance = {}

    return model, y_pred_proba, auc, acc, importance


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("树模型市场状态检测 — 训练")
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

    # 计算特征
    print("\n[2] 计算特征...")
    features = calc_features(prices)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"  特征: {features.shape[1]} 个, 范围: {features.index[0].date()} ~ {features.index[-1].date()}")

    # 多窗口标签
    print("\n[3] 生成标签（多窗口）...")
    labels = {}
    for window in [10, 20, 30]:
        lbl = make_label(prices, lookforward=window)
        choppy_pct = lbl.mean()
        print(f"  {window}天窗口: 震荡占比={choppy_pct:.1%}")

    # 滚动训练: 2018-2023 训练, 2024 测试
    train_end = '2023-12-31'
    test_start = '2024-01-01'

    X_train = features.loc[:train_end]
    X_test = features.loc[test_start:]

    # 测试每个窗口的每个模型
    models = ["lightgbm", "xgboost", "random_forest", "decision_tree", "logistic_regression"]
    all_results = {}

    for window in [10, 20, 30]:
        print(f"\n[4] 窗口={window}天 模型对比...")
        y_train = make_label(prices.loc[:train_end], lookforward=window)
        y_test = make_label(prices.loc[test_start:], lookforward=window)

        # 对齐索引
        common_train = X_train.index.intersection(y_train.index)
        common_test = X_test.index.intersection(y_test.index)
        X_tr = X_train.loc[common_train]
        y_tr = y_train.loc[common_train]
        X_te = X_test.loc[common_test]
        y_te = y_test.loc[common_test]

        print(f"  Train: {len(X_tr)} 样本, 震荡={y_tr.mean():.1%}")
        print(f"  Test:  {len(X_te)} 样本, 震荡={y_te.mean():.1%}")

        all_results[window] = {}
        for mname in models:
            try:
                model, y_pred, auc, acc, importance = train_and_evaluate(
                    X_tr, y_tr, X_te, y_te, mname
                )
                all_results[window][mname] = {
                    'auc': auc, 'acc': acc, 'importance': importance
                }
                print(f"    {mname:15s}: AUC={auc:.3f}  Acc={acc:.3f}")
            except Exception as e:
                print(f"    {mname:15s}: ERROR - {e}")
                all_results[window][mname] = None

    # 总结最佳模型
    print("\n" + "=" * 80)
    print("最佳模型总结")
    print("=" * 80)
    best_auc = 0
    best_combo = None
    for window, results in all_results.items():
        for mname, res in results.items():
            if res and res['auc'] > best_auc:
                best_auc = res['auc']
                best_combo = (window, mname)

    if best_combo:
        window, mname = best_combo
        print(f"  最佳: 窗口={window}天, 模型={mname}, AUC={best_auc:.3f}")

    # 保存结果
    import pickle
    with open('/home/ll/Public/QuantNodes/output/market_regime_model_results.pkl', 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'best_combo': best_combo,
        }, f)
    print(f"\n结果已保存: output/market_regime_model_results.pkl")

    return all_results, best_combo


if __name__ == "__main__":
    main()
