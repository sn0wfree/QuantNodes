"""CA-GCP 训练 v4 — 半监督学习 + 特征降维

改进:
  1. 用 2008-2017 未标注宏观数据训练自编码器 (representation learning)
  2. 自编码器瓶颈特征作为新维度 (有信号的特征压缩)
  3. Pseudo-labeling: 用 v3 最佳模型给未标注数据打伪标签
  4. 自训练: 高置信度伪标签加入训练集
  5. 强正则化: 处理 132 特征 vs 1320 样本
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports" / "momentum_etf_rotation" / "ca_gcp_ew"))  # noqa: E402

warnings.filterwarnings("ignore")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ca_gcp_training_v3 import (  # noqa: E402
    build_macro_features_v3,
    build_y_labels,
    feature_columns_v3,
    load_hs300_returns,
    load_macro_daily,
    train_with_optuna,
    wash_forward_validate,
)

from ca_gcp_ew_eval import (  # noqa: E402
    OUT_DIR,
    load_returns,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

MACRO_PATH = ROOT / "data" / "high_freq_macro" / "v7_6_X_macro_weekly.parquet"

UNLABELED_END = pd.Timestamp("2017-12-31")
LABELED_START = pd.Timestamp("2018-01-01")

AUTOENCODER_EPOCHS = 50
AUTOENCODER_BOTTLENECK = 8
AUTOENCODER_HIDDEN = 32


def build_unlabeled_macro_features() -> pd.DataFrame:
    """构建未标注期 (2008-2017) 的宏观特征"""
    macro_daily = load_macro_daily()
    macro_features = build_macro_features_v3(macro_daily)
    unlabeled = macro_features.loc[:UNLABELED_END]
    print(f"  未标注宏观特征: {unlabeled.shape}, {unlabeled.index[0].date()} ~ {unlabeled.index[-1].date()}")  # noqa: E501
    return unlabeled


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(
    features: pd.DataFrame,
    epochs: int = AUTOENCODER_EPOCHS,
    hidden_dim: int = AUTOENCODER_HIDDEN,
    bottleneck_dim: int = AUTOENCODER_BOTTLENECK,
) -> tuple[np.ndarray, StandardScaler, Autoencoder]:
    """训练自编码器，返回瓶颈特征"""
    X = features.replace([np.inf, -np.inf], 0).fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_t = torch.FloatTensor(X_scaled)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(X.shape[1], hidden_dim, bottleneck_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    batch_size = 128
    n = X_t.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        for i in range(0, n, batch_size):
            batch = X_t[perm[i:i + batch_size]].to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss: {total_loss/n:.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(X_t.to(device)).cpu().numpy()

    print(f"  瓶颈特征 shape: {z.shape}")
    return z, scaler, model


def encode_features(
    features: pd.DataFrame, scaler: StandardScaler, model: Autoencoder
) -> np.ndarray:
    X = features.replace([np.inf, -np.inf], 0).fillna(0).values
    X_scaled = scaler.transform(X)
    X_t = torch.FloatTensor(X_scaled)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = model.encode(X_t.to(device)).cpu().numpy()
    return z


def pseudo_labeling(
    features_labeled: pd.DataFrame,
    features_unlabeled: pd.DataFrame,
    y_labeled: pd.Series,
    best_model_type: str,
    best_params: dict,
    confidence_threshold: float = 0.8,
) -> tuple[pd.DataFrame, pd.Series]:
    """Pseudo-labeling: 在未标注数据上预测，高置信度样本加入训练集"""
    print(f"  [Pseudo-labeling] 用 {best_model_type} 在 {len(features_unlabeled)} 未标注样本上预测...")  # noqa: E501
    pos_weight = float((y_labeled == 0).sum()) / max(float((y_labeled == 1).sum()), 1)

    features_labeled = features_labeled.replace([np.inf, -np.inf], 0).fillna(0)

    features_unlabeled = features_unlabeled.reindex(columns=features_labeled.columns)
    features_unlabeled = features_unlabeled.replace([np.inf, -np.inf], 0).fillna(0)

    if best_model_type == "lgb":
        params = best_params.copy()
        params.update({"objective": "binary", "metric": "binary_logloss",
                       "bagging_freq": 5, "min_data_in_leaf": 30,
                       "scale_pos_weight": pos_weight, "verbose": -1})
        train_data = lgb.Dataset(features_labeled.values, label=y_labeled.values)
        model = lgb.train(params, train_data, num_boost_round=200,
                          callbacks=[lgb.log_evaluation(0)])
        proba = model.predict(features_unlabeled.values)
    else:
        params = best_params.copy()
        params.update({"objective": "binary:logistic", "eval_metric": "logloss",
                       "scale_pos_weight": pos_weight, "verbosity": 0})
        dtrain = xgb.DMatrix(features_labeled.values, label=y_labeled.values)
        dtest = xgb.DMatrix(features_unlabeled.values)
        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=0)
        proba = model.predict(dtest)

    high_pos = features_unlabeled.index[proba > confidence_threshold]
    high_neg = features_unlabeled.index[proba < (1 - confidence_threshold)]
    print(f"  高置信正样本: {len(high_pos)}, 高置信负样本: {len(high_neg)}")

    pseudo_y = pd.Series(1, index=high_pos)
    pseudo_y = pd.concat([
        pseudo_y,
        pd.Series(0, index=high_neg),
    ])

    pseudo_features = features_unlabeled.loc[pseudo_y.index]
    return pseudo_features, pseudo_y


def train_with_strong_regularization(
    X: pd.DataFrame, y: pd.Series, model_type: str, n_trials: int = 30,
    feature_fraction_max: float = 0.5
) -> dict:
    """强正则化训练: 限制树深度 + 高 min_data + L1"""
    pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial):
        if model_type == "lgb":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": trial.suggest_int("num_leaves", 4, 31),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.2, feature_fraction_max),  # noqa: E501
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.8),
                "bagging_freq": 5,
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 150),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10.0, log=True),
                "scale_pos_weight": pos_weight,
                "verbose": -1,
            }
        else:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 0.8),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, feature_fraction_max),  # noqa: E501
                "min_child_weight": trial.suggest_int("min_child_weight", 30, 150),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "scale_pos_weight": pos_weight,
                "verbosity": 0,
            }

        fold_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            if model_type == "lgb":
                train_data = lgb.Dataset(X_tr.values, label=y_tr.values)
                m = lgb.train(params, train_data, num_boost_round=300,
                              callbacks=[lgb.log_evaluation(0)])
                proba = m.predict(X_te.values)
            else:
                dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
                dtest = xgb.DMatrix(X_te.values)
                m = xgb.train(params, dtrain, num_boost_round=300, verbose_eval=0)
                proba = m.predict(dtest)
            best_f1 = 0
            for th in np.arange(0.3, 0.7, 0.05):
                pred = (proba > th).astype(int)
                if pred.sum() < 3:
                    continue
                f1 = f1_score(y_te.values, pred, zero_division=0)
                best_f1 = max(best_f1, f1)
            fold_scores.append(best_f1)
        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    if model_type == "lgb":
        best_params.update({
            "objective": "binary", "metric": "binary_logloss",
            "bagging_freq": 5, "scale_pos_weight": pos_weight,
            "verbose": -1,
        })
    else:
        best_params.update({
            "objective": "binary:logistic", "eval_metric": "logloss",
            "scale_pos_weight": pos_weight, "verbosity": 0,
        })

    oof_proba = np.zeros(len(X))
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        if model_type == "lgb":
            train_data = lgb.Dataset(X_tr.values, label=y_tr.values)
            m = lgb.train(best_params, train_data, num_boost_round=300,
                          callbacks=[lgb.log_evaluation(0)])
            oof_proba[test_idx] = m.predict(X_te.values)
        else:
            dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
            dtest = xgb.DMatrix(X_te.values)
            m = xgb.train(best_params, dtrain, num_boost_round=300, verbose_eval=0)
            oof_proba[test_idx] = m.predict(dtest)

    best_th, best_f1 = 0.5, -1
    for th in np.arange(0.3, 0.7, 0.05):
        pred = (oof_proba > th).astype(int)
        if pred.sum() < 5:
            continue
        f1 = f1_score(y.values, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    auc = roc_auc_score(y.values, oof_proba)
    oof_pred = (oof_proba > best_th).astype(int)
    p = precision_score(y.values, oof_pred, zero_division=0)
    r = recall_score(y.values, oof_pred, zero_division=0)
    f1 = f1_score(y.values, oof_pred, zero_division=0)

    return {
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
        "best_threshold": best_th,
        "P": p, "R": r, "F1": f1, "AUC": auc,
        "best_params": best_params,
    }


def main() -> None:
    print("=" * 70)
    print("CA-GCP 训练 v4 — 半监督学习 + 特征降维")
    print("=" * 70)

    returns = load_returns()

    from ca_gcp_training_v3 import build_full_features
    features = build_full_features(returns)
    hs300 = load_hs300_returns()
    y_labels = build_y_labels(returns, hs300)

    common_idx = features.index.intersection(y_labels.index)
    X_labeled = features.loc[common_idx]
    Y = y_labels.loc[common_idx]

    print(f"\n[标注数据] {X_labeled.shape}, {X_labeled.index[0].date()} ~ {X_labeled.index[-1].date()}")  # noqa: E501
    print("[未标注数据] 2008-01 ~ 2017-12 (10 年宏观数据)")

    print()
    print("=" * 70)
    print("Phase 1: 自编码器训练 (用 2008-2017 未标注宏观数据)")
    print("=" * 70)

    unlabeled_features = build_unlabeled_macro_features()
    all_macro_features_for_ae = pd.concat([unlabeled_features, features]).drop_duplicates()
    print(f"  训练自编码器数据: {all_macro_features_for_ae.shape}")

    z_unlabeled, ae_scaler, ae_model = train_autoencoder(
        all_macro_features_for_ae, epochs=AUTOENCODER_EPOCHS
    )

    z_labeled = encode_features(X_labeled, ae_scaler, ae_model)
    z_labeled_df = pd.DataFrame(
        z_labeled,
        index=X_labeled.index,
        columns=[f"ae_z{i}" for i in range(AUTOENCODER_BOTTLENECK)]
    )

    print()
    print("=" * 70)
    print("Phase 2: Pseudo-labeling (v3 最佳模型打伪标签)")
    print("=" * 70)

    feat_cols_v3 = feature_columns_v3(X_labeled)
    X_v3 = X_labeled[feat_cols_v3].fillna(0)

    v3_results = {}
    for y_name in ["y_A", "y_B", "y_D", "y_E"]:
        y = Y[y_name]
        for mt in ["lgb", "xgb"]:
            key = f"{y_name}_{mt}"
            res = train_with_optuna(X_v3, y, mt, n_trials=20)
            v3_results[key] = res

    best_key = max(v3_results.keys(), key=lambda k: v3_results[k]["F1"])
    print(f"\n最佳 v3 基模型: {best_key} (F1={v3_results[best_key]['F1']:.3f})")

    pseudo_features, pseudo_y = pseudo_labeling(
        X_v3, unlabeled_features, Y["_".join(best_key.split("_")[:2])],
        best_key.split("_")[2], v3_results[best_key]["best_params"],
        confidence_threshold=0.75
    )

    print()
    print("=" * 70)
    print("Phase 3: 增强特征 + 强正则化训练 (132 + 8 维 = 140 维)")
    print("=" * 70)

    X_aug = X_v3.join(z_labeled_df, how="inner")
    X_with_pseudo = pd.concat([X_aug, pseudo_features], axis=0)
    pseudo_y_aligned = pseudo_y.reindex(X_with_pseudo.index)
    Y_with_pseudo = pd.DataFrame(index=X_with_pseudo.index)
    Y_with_pseudo["y_A"] = pseudo_y_aligned
    Y_with_pseudo["y_B"] = pseudo_y_aligned
    Y_with_pseudo["y_C"] = pseudo_y_aligned
    Y_with_pseudo["y_D"] = pseudo_y_aligned
    Y_with_pseudo["y_E"] = pseudo_y_aligned
    Y_with_pseudo.loc[X_aug.index, :] = Y.loc[X_aug.index, :].values

    print(f"  增强后特征: {X_with_pseudo.shape}, 标签: {Y_with_pseudo.shape}")

    v4_results = {}
    for y_name in ["y_A", "y_B", "y_D", "y_E"]:
        y_full = Y_with_pseudo[y_name]
        y_labeled_only = Y[y_name]
        for mt in ["lgb", "xgb"]:
            key = f"{y_name}_{mt}"
            print(f"  [{y_name}/{mt}] 训练中...")
            res = train_with_strong_regularization(
                X_with_pseudo.fillna(0), y_full, mt, n_trials=25
            )
            v4_results[key] = res
            oof_labeled = res["oof_proba"][:len(X_aug)]
            oof_pred_labeled = (oof_labeled > res["best_threshold"]).astype(int)
            f1_labeled = f1_score(y_labeled_only.values, oof_pred_labeled, zero_division=0)
            print(f"    F1={res['F1']:.3f}, AUC={res['AUC']:.3f}, "
                  f"标注F1={f1_labeled:.3f}")

    print()
    print("=" * 70)
    print("Phase 4: Wash-Forward 验证")
    print("=" * 70)

    X_labeled_aug = X_aug.fillna(0)
    top3_v4 = sorted(v4_results.keys(), key=lambda k: v4_results[k]["F1"], reverse=True)[:3]
    wf_results = {}
    for key in top3_v4:
        y_name = "_".join(key.split("_")[:2])
        model_type = key.split("_")[2]
        params = v4_results[key]["best_params"]
        y_labeled_only = Y[y_name]
        print(f"\n--- Wash-Forward: {key} ---")
        wf_res = wash_forward_validate(X_labeled_aug, y_labeled_only, model_type, params)
        wf_results[key] = wf_res
        print(f"  WF F1: {wf_res['mean_F1']:.3f} ± {wf_res['std_F1']:.3f}")

    print()
    print("=" * 70)
    print("Phase 5: 报告")
    print("=" * 70)

    best_v4_key = max(v4_results.keys(), key=lambda k: v4_results[k]["F1"])

    report = [
        "# CA-GCP 训练 v4 — 半监督学习 + 特征降维报告",
        "",
        "## 1. 改进点",
        "",
        "- **未标注数据利用**: 2008-2017 宏观数据 (522 周) 训练自编码器",
        "- **瓶颈特征**: 自编码器 8 维 bottleneck 作为新特征",
        "- **Pseudo-labeling**: v3 最佳模型对未标注数据打伪标签",
        "- **强正则化**: 高 min_data_in_leaf + L1/L2 正则",
        "",
        "## 2. 增强特征矩阵",
        "",
        f"- v3 特征: {len(feat_cols_v3)}",
        f"- 自编码瓶颈: {AUTOENCODER_BOTTLENECK}",
        f"- 总维度: {len(feat_cols_v3) + AUTOENCODER_BOTTLENECK}",
        "",
        "## 3. v4 单模型结果 (强正则化)",
        "",
        "| 标签 | 模型 | P | R | F1 | AUC |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in sorted(v4_results.items()):
        y_name_k = "_".join(k.split("_")[:2])
        model_k = k.split("_")[2]
        report.append(
            f"| {y_name_k} | {model_k} | {v['P']:.3f} | {v['R']:.3f} | {v['F1']:.3f} | {v['AUC']:.3f} |"  # noqa: E501
        )

    report += [
        "",
        f"**最佳 v4 单模型**: {best_v4_key}, F1={v4_results[best_v4_key]['F1']:.3f}",
        "",
        "## 4. Wash-Forward 稳健性 (Top-3)",
        "",
        "| 模型 | WF F1 (mean ± std) | 折数 |",
        "|---|---|---|",
    ]
    for k, v in wf_results.items():
        report.append(f"| {k} | {v['mean_F1']:.3f} ± {v['std_F1']:.3f} | {v['n_folds']} |")

    report += [
        "",
        "## 5. 对比 (v3 vs v4 vs 启发式)",
        "",
        "| 方法 | 最佳 F1 | WF F1 |",
        "|---|---|---|",
        "| 启发式 v3 (and_fired) | 0.500 | - |",
        "| v3 y_E XGB | 0.462 | 0.509 ± 0.218 |",
        f"| v4 强正则化 (best) | {v4_results[best_v4_key]['F1']:.3f} | {wf_results[best_v4_key]['mean_F1']:.3f} ± {wf_results[best_v4_key]['std_F1']:.3f} |",  # noqa: E501
        "",
        "## 6. 推荐",
        "",
    ]
    best_v4_f1 = v4_results[best_v4_key]["F1"]
    best_v4_wf = wf_results[best_v4_key]["mean_F1"]
    if best_v4_f1 > 0.50 or best_v4_wf > 0.50:
        report.append(f"**v4 已超越启发式 v3** (F1 {best_v4_f1:.3f}, WF {best_v4_wf:.3f})")
    elif best_v4_wf >= 0.48:
        report.append(f"**v4 接近启发式 v3** (F1 {best_v4_f1:.3f}, WF {best_v4_wf:.3f})")
    else:
        report.append("**启发式 v3 仍最优**")

    (OUT_DIR / "ca_gcp_training_v4_report.md").write_text("\n".join(report))
    print(f"[报告] {OUT_DIR / 'ca_gcp_training_v4_report.md'}")


if __name__ == "__main__":
    main()
