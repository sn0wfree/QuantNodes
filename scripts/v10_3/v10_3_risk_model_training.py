"""v10.3 Risk Warning System — Phase 2: ML Model Training.

LightGBM/XGBoost binary classifier with walk-forward validation.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / 'reports' / 'momentum_etf_rotation' / 'v10_3'
OUT_DIR.mkdir(exist_ok=True)

OOS_START = '2022-01-01'
OOS_END = '2026-05-29'

log = print


def load_data():
    log('=' * 60)
    log('v10.3 Risk Warning — ML Model Training')
    log('=' * 60)

    log('\n[1] 加载特征矩阵和目标变量...')
    X = pd.read_parquet(OUT_DIR / 'risk_feature_matrix.parquet')
    y = pd.read_parquet(OUT_DIR / 'risk_target.parquet')['y_E']
    log(f'  特征矩阵: {X.shape}')
    log(f'  目标变量: {len(y)} samples, {y.mean():.1%} positive')

    # 对齐
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    log(f'  对齐后: {len(X)} samples')

    return X, y


def select_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    log('\n[2] 特征选择...')

    # 2.1 移除缺失率 > 50% 的特征
    missing_rate = X.isnull().mean()
    keep_cols = missing_rate[missing_rate < 0.50].index.tolist()
    X_filt = X[keep_cols]
    log(f'  移除缺失率 > 50%: {len(X.columns) - len(keep_cols)} dropped')

    # 2.2 缺失值用中位数填充
    X_fill = X_filt.fillna(X_filt.median())

    # 2.3 单信号 AUC 筛选 (保留 AUC > 0.52 的)
    aucs = []
    for col in X_fill.columns:
        try:
            auc = roc_auc_score(y, X_fill[col])
            if auc >= 0.52:
                aucs.append((col, auc))
        except Exception:
            pass

    aucs.sort(key=lambda x: x[1], reverse=True)
    sel_cols = [c for c, _ in aucs]
    log(f'  单信号 AUC > 0.52: {len(sel_cols)} features')

    # 2.4 去重: 去除相关性 > 0.95 的冗余特征
    if len(sel_cols) > 0:
        corr = X_fill[sel_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
        sel_cols = [c for c in sel_cols if c not in to_drop]
        log(f'  去重 (corr > 0.95): {len(to_drop)} dropped')

    log(f'  最终特征数: {len(sel_cols)}')
    return X_fill[sel_cols]


def walk_forward_splits(X: pd.DataFrame, y: pd.Series):
    """生成 walk-forward 折.

    每折: 训练期 4 年, 测试期 1 年, 步长 1 年.
    purge=5 天, embargo=5 天.
    """
    dates = X.index
    splits = []

    train_start = dates[0]
    test_start = pd.Timestamp(OOS_START)
    test_end = pd.Timestamp(OOS_END)

    while test_start < test_end:
        train_end = test_start - pd.Timedelta(days=5 + 5)  # purge + embargo
        if train_end > train_start:
            train_idx = dates[(dates >= train_start) & (dates <= train_end)]
            test_idx = dates[(dates >= test_start) & (dates <= test_start + pd.Timedelta(days=365))]
            if len(train_idx) > 252 and len(test_idx) > 20:
                splits.append((train_idx, test_idx))
        test_start += pd.Timedelta(days=365)

    log(f'\n[3] Walk-forward 折: {len(splits)} folds')
    for i, (tr, te) in enumerate(splits):
        log(f'  Fold {i+1}: train {tr[0].date()} ~ {tr[-1].date()} ({len(tr)} days), '
            f'test {te[0].date()} ~ {te[-1].date()} ({len(te)} days), '
            f'y_pos={y.loc[te].mean():.1%}')

    return splits


def train_optuna(X_train, y_train, n_trials=20):
    import optuna
    import lightgbm as lgb

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'n_jobs': -1,
        }
        dtrain = lgb.Dataset(X_train, y_train)
        cv_result = lgb.cv(
            params, dtrain, num_boost_round=500, nfold=4, stratified=True,
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        return max(cv_result['valid auc-mean'])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log(f'\n  Optuna 最佳 AUC: {study.best_value:.4f}')
    log(f'  最佳参数: {study.best_params}')

    return study.best_params


def train_model(X_train, y_train, params, model_type='lgb'):
    from sklearn.calibration import CalibratedClassifierCV

    if model_type == 'lgb':
        import lightgbm as lgb

        params = params.copy()
        params['verbose'] = -1
        base = lgb.LGBMClassifier(**params, n_estimators=500, n_jobs=-1)
        base.fit(X_train, y_train)
    elif model_type == 'xgb':
        import xgboost as xgb

        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        base = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            n_estimators=500,
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.05),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            reg_alpha=params.get('reg_alpha', 1.0),
            reg_lambda=params.get('reg_lambda', 1.0),
            n_jobs=-1,
            verbosity=0,
        )
        base.fit(X_train, y_train)

    model = CalibratedClassifierCV(base, method='sigmoid', cv=3)
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_prob):
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, f1_score,
        precision_score, recall_score, brier_score_loss,
    )

    y_pred = (y_prob >= 0.5).astype(float)

    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'F1': f1_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Brier': brier_score_loss(y_true, y_prob),
        'PosRate': y_true.mean(),
        'N': len(y_true),
    }


def main():
    X, y = load_data()
    X_sel = select_features(X, y)
    splits = walk_forward_splits(X_sel, y)

    # 训练 + 评估 LightGBM
    log('\n[4] LightGBM 训练 + Walk-forward 评估...')
    all_metrics = []
    all_y_true, all_y_prob = [], []

    for i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_sel.loc[tr_idx], y.loc[tr_idx]
        X_te, y_te = X_sel.loc[te_idx], y.loc[te_idx]

        log(f'\n  Fold {i+1}: training ({len(X_tr)} samples)...')

        # 在第一个 fold 上做 Optuna
        if i == 0:
            best_params = train_optuna(X_tr, y_tr, n_trials=20)
            params = best_params
        else:
            params = best_params

        model = train_model(X_tr, y_tr, params, model_type='lgb')
        y_prob = model.predict_proba(X_te)[:, 1]
        metrics = evaluate(y_te, y_prob)
        metrics['Fold'] = i + 1
        all_metrics.append(metrics)
        all_y_true.extend(y_te.tolist())
        all_y_prob.extend(y_prob.tolist())

        log(f'    AUC={metrics["AUC"]:.4f} PR-AUC={metrics["PR-AUC"]:.4f} '
            f'F1={metrics["F1"]:.4f} Brier={metrics["Brier"]:.4f}')

    # 汇总
    log('\n--- Walk-forward 汇总 ---')
    mf = pd.DataFrame(all_metrics)
    log(f'  AUC: {mf["AUC"].mean():.4f} ± {mf["AUC"].std():.4f}')
    log(f'  PR-AUC: {mf["PR-AUC"].mean():.4f} ± {mf["PR-AUC"].std():.4f}')
    log(f'  F1: {mf["F1"].mean():.4f} ± {mf["F1"].std():.4f}')
    log(f'  Brier: {mf["Brier"].mean():.4f} ± {mf["Brier"].std():.4f}')

    mf.to_csv(OUT_DIR / 'risk_walkforward_metrics.csv', index=False)
    log(f'  保存: {OUT_DIR / "risk_walkforward_metrics.csv"}')

    # 全 OOS 评估
    oos_idx = X_sel.index[(X_sel.index >= OOS_START) & (X_sel.index <= OOS_END)]
    X_oos = X_sel.loc[oos_idx]
    y_oos = y.loc[oos_idx]

    log(f'\n[5] SHAP 分析...')
    import shap
    final_model = train_model(X_oos, y_oos, best_params, model_type='lgb')
    base_est = final_model.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_est)
    shap_values = explainer.shap_values(X_oos)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    shap_imp = pd.DataFrame({
        'feature': X_oos.columns,
        'importance': np.abs(shap_values).mean(axis=0),
    }).sort_values('importance', ascending=False)
    shap_imp.to_csv(OUT_DIR / 'risk_shap_importance.csv', index=False)
    log(f'  Top 15 特征 (SHAP):')
    log(shap_imp.head(15).to_string(index=False))

    # 训练最终模型 (全 OOS 数据)
    log(f'\n[6] 训练最终模型 (全 OOS 数据)...')
    final_model = train_model(X_oos, y_oos, best_params, model_type='lgb')
    y_prob_final = final_model.predict_proba(X_oos)[:, 1]
    final_metrics = evaluate(y_oos, y_prob_final)
    log(f'  OOS AUC={final_metrics["AUC"]:.4f} PR-AUC={final_metrics["PR-AUC"]:.4f} '
        f'Brier={final_metrics["Brier"]:.4f}')

    # 保存模型
    import joblib
    model_path = OUT_DIR / 'risk_alert_model.pkl'
    joblib.dump({'model': final_model, 'features': list(X_oos.columns)}, model_path)
    log(f'  保存模型: {model_path}')

    # 保存 OOS 预测概率
    oos_pred = pd.DataFrame({
        'y_true': y_oos,
        'y_prob': y_prob_final,
    }, index=X_oos.index)
    oos_pred.to_parquet(OUT_DIR / 'risk_oos_predictions.parquet')
    log(f'  保存 OOS 预测: {OUT_DIR / "risk_oos_predictions.parquet"}')

    # XGBoost 对比
    log(f'\n[7] XGBoost 对比...')
    xgb_params = {k: v for k, v in best_params.items()
                  if k in ('max_depth', 'learning_rate', 'subsample',
                           'colsample_bytree', 'reg_alpha', 'reg_lambda')}
    xgb_model = train_model(X_oos, y_oos, xgb_params, model_type='xgb')
    y_prob_xgb = xgb_model.predict_proba(X_oos)[:, 1]
    xgb_metrics = evaluate(y_oos, y_prob_xgb)
    log(f'  XGBoost OOS: AUC={xgb_metrics["AUC"]:.4f} PR-AUC={xgb_metrics["PR-AUC"]:.4f} '
        f'Brier={xgb_metrics["Brier"]:.4f}')
    log(f'  LightGBM OOS: AUC={final_metrics["AUC"]:.4f} PR-AUC={final_metrics["PR-AUC"]:.4f} '
        f'Brier={final_metrics["Brier"]:.4f}')

    log('\n[完成] 模型训练结束')


if __name__ == '__main__':
    main()
