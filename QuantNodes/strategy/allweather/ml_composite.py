# coding=utf-8
"""AllWeather V6.6 ML 复合因子 (PLAN §15).

测试 Sub-A/B/C 三个子因子能否通过 ML (线性 + 树) 组合成更有效的复合因子,
用于预测风险 (vol spike / vol level) 或转折点 (turning point).

核心组件:
    1. PurgedKFold — 金融专用时序 CV (5 折 + 5d purge + 1% embargo)
    2. build_features — 特征工程 (29 维: 15 sub-factor + 14 asset)
    3. build_targets — 三目标定义 (Y1/Y2/Y3)
    4. train_models — 4 模型训练 (LogReg/Ridge/DT/RF)
    5. to_soft_signal — 输出 Soft 信号 ∈ [-1, +1]
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# ── 1. Purged K-Fold ──────────────────────────────────────────────────
class PurgedKFold:
    """金融专用时序 K-Fold (López de Prado 'Advances in Financial ML' Ch.7).

    关键约束:
      1. 时间顺序: train_index < test_index (无未来)
      2. Purge: train 末尾 5d 与 test 头部 5d 重叠区间剔除 (防标签泄漏)
      3. Embargo: test 后 1% 时段作为 embargo (防 test 信息渗入后续 train)
      4. k=5 默认, 每折 train/test 比例 80/20

    Args:
        n_splits: K 折数, 默认 5.
        purge_gap: purge 间隔天数, 默认 5.
        embargo_pct: embargo 比例 (test 后), 默认 0.01.
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 5,
                 embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, y=None, groups=None):
        """生成 (train_idx, test_idx) tuples.

        Args:
            X: 特征矩阵 (index 为 DatetimeIndex 或 MultiIndex (date, asset)).
            y: 目标 (可选, 仅用于 API 兼容).
            groups: 分组 (可选).

        Yields:
            (train_idx, test_idx) 元组.
        """
        n = len(X)
        test_size = n // (self.n_splits + 1)
        embargo_size = int(n * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n
            test_idx = np.arange(test_start, test_end)

            train_end = test_start - self.purge_gap
            train_idx = np.arange(0, max(train_end, 0))

            if i < self.n_splits - 1:
                next_train_start = test_end + self.purge_gap + embargo_size
                if next_train_start < n:
                    mask = (train_idx >= next_train_start) & (train_idx < n)
                    train_idx = train_idx[~mask]

            yield train_idx, test_idx


# ── 2. 特征工程 ────────────────────────────────────────────────────────
SECTOR_MAP: dict[str, str] = {
    "au": "metal", "ag": "metal", "cu": "metal",
    "al": "metal", "zn": "metal", "ni": "metal",
    "rb": "steel", "hc": "steel", "i": "steel",
    "j": "steel", "jm": "steel",
    "ta": "chem", "ma": "chem", "pp": "chem",
    "l": "chem", "bu": "chem", "sc": "energy",
    "m": "agri", "y": "agri", "p": "agri",
    "cf": "agri", "sr": "agri", "c": "agri",
    "if": "index", "ic": "index", "ih": "index",
    "t": "bond", "tf": "bond", "ts": "bond",
}


def _asset_sector(panel: pd.DataFrame) -> pd.Series:
    """资产 → sector 映射."""
    sectors = []
    for col in panel.columns:
        col_str = str(col).lower()
        if col_str.startswith("fut_") and col_str[4:] in SECTOR_MAP:
            sectors.append(SECTOR_MAP[col_str[4:]])
        elif col_str.startswith("etf_"):
            sectors.append("etf_eq")
        else:
            sectors.append("other")
    return pd.Series(sectors, index=panel.columns, name="sector")


def _asset_vol_class(panel: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """资产 → vol class (高/中/低, 基于历史 vol 三分位)."""
    rets = panel.pct_change(fill_method=None)
    vols = rets.rolling(lookback, min_periods=60).std().mean()
    tertiles = pd.qcut(vols, q=3, labels=["vol_lo", "vol_md", "vol_hi"])
    tertiles = tertiles.fillna("vol_md")
    return tertiles


def _asset_volume_class(panel: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """资产 → volume class (高/中/低, 基于平均成交量三分位)."""
    vol_mean = panel.abs().rolling(lookback, min_periods=60).mean().mean()
    tertiles = pd.qcut(vol_mean, q=3, labels=["vol_lo", "vol_md", "vol_hi"])
    tertiles = tertiles.fillna("vol_md")
    return tertiles


def build_features(panel: pd.DataFrame, sf_cfg=None) -> pd.DataFrame:
    """构建特征矩阵 (T × N × F → 2D: (T × N) × F).

    特征列表 (29 维):
        1-3:   sub_a, sub_b, sub_c
        4-6:   sub_a_lag1, sub_b_lag1, sub_c_lag1
        7-9:   sub_a_lag5, sub_b_lag5, sub_c_lag5
        10-12: sub_a_ma5, sub_b_ma5, sub_c_ma5
        13-15: sub_a_std5, sub_b_std5, sub_c_std5
        16-17: sub_a_x_b, sub_b_x_c
        18-25: sector one-hot × 8
        26-28: vol_class one-hot × 3
        29-31: volume_class one-hot × 3

    输出: pd.DataFrame index=MultiIndex(date, asset), columns=31 features.
    """
    from .subfactors import SubFactorConfig, compute_all_subfactors

    cfg = sf_cfg or SubFactorConfig()
    factors = compute_all_subfactors(
        prices=panel, volumes=panel, open_interest=None, cfg=cfg,
    )

    parts = []
    for asset in panel.columns:
        a = factors["a"][asset]
        b = factors["b"][asset]
        c = factors["c"][asset]
        df = pd.DataFrame({
            "date": a.index,
            "asset": asset,
            "sub_a": a.values,
            "sub_b": b.values,
            "sub_c": c.values,
            "sub_a_lag1": a.shift(1).values,
            "sub_b_lag1": b.shift(1).values,
            "sub_c_lag1": c.shift(1).values,
            "sub_a_lag5": a.shift(5).values,
            "sub_b_lag5": b.shift(5).values,
            "sub_c_lag5": c.shift(5).values,
            "sub_a_ma5": a.rolling(5).mean().values,
            "sub_b_ma5": b.rolling(5).mean().values,
            "sub_c_ma5": c.rolling(5).mean().values,
            "sub_a_std5": a.rolling(5).std().values,
            "sub_b_std5": b.rolling(5).std().values,
            "sub_c_std5": c.rolling(5).std().values,
            "sub_a_x_b": (a * b).values,
            "sub_b_x_c": (b * c).values,
        })
        parts.append(df)
    feat_long = pd.concat(parts, ignore_index=True)
    feat_long["date"] = pd.to_datetime(feat_long["date"])
    feat_long = feat_long.set_index(["date", "asset"])

    sector = _asset_sector(panel)
    vol_class = _asset_vol_class(panel)
    volume_class = _asset_volume_class(panel)
    sector_dummies = pd.get_dummies(sector, prefix="sec").reindex(panel.columns).fillna(0)
    vol_dummies = pd.get_dummies(vol_class, prefix="vclass").reindex(panel.columns).fillna(0)
    volu_dummies = pd.get_dummies(volume_class, prefix="vclass_v").reindex(panel.columns).fillna(0)

    asset_feats = pd.concat([sector_dummies, vol_dummies, volu_dummies], axis=1)
    asset_feats.index = panel.columns
    asset_feats.index.name = "asset"

    feat_reset = feat_long.reset_index()
    feat_reset = feat_reset.merge(
        asset_feats.reset_index(), on="asset", how="left",
    )
    feat_reset = feat_reset.set_index(["date", "asset"])
    return feat_reset


# ── 3. 三目标定义 ──────────────────────────────────────────────────────
def _per_asset_targets(panel: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """对每个 (date, asset) 计算三个目标.

    Y1: vol_spike (binary) — 未来 horizon 日 vol > 历史 90 分位
    Y2: vol_level (regression) — 未来 horizon 日年化 vol
    Y3: turning_point (binary) — 未来 horizon 日趋势反转

    返回: pd.DataFrame index=MultiIndex(date, asset), columns=[y_vol_spike, y_vol_level, y_turning].
    """
    parts = []
    for asset in panel.columns:
        s = panel[asset]
        rets = s.pct_change(fill_method=None)
        fwd_vol = rets.rolling(horizon).std().shift(-horizon) * np.sqrt(252)
        hist_vol = rets.rolling(252, min_periods=60).std() * np.sqrt(252)

        hist_q90 = hist_vol.expanding(min_periods=60).quantile(0.9)

        y1 = (fwd_vol > hist_q90).astype(float)

        y2 = fwd_vol

        fwd_ret = s.pct_change(horizon, fill_method=None).shift(-horizon)
        cur_ret = s.pct_change(horizon, fill_method=None)
        y3 = ((np.sign(fwd_ret) != np.sign(cur_ret)) & (fwd_ret != 0)).astype(float)

        df = pd.DataFrame({
            "date": s.index,
            "asset": asset,
            "y_vol_spike": y1.values,
            "y_vol_level": y2.values,
            "y_turning": y3.values,
        })
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.set_index(["date", "asset"])
    return out


def build_targets(panel: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """构建三目标 DataFrame."""
    return _per_asset_targets(panel, horizon)


# ── 4. 模型训练 ──────────────────────────────────────────────────────
def _get_models(target_type: Literal["binary", "regression"]) -> dict:
    if target_type == "binary":
        return {
            "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "dt": DecisionTreeClassifier(max_depth=5, class_weight="balanced"),
            "rf": RandomForestClassifier(
                n_estimators=100, max_depth=8,
                class_weight="balanced", n_jobs=-1, random_state=42,
            ),
        }
    return {
        "ridge": Ridge(alpha=1.0),
        "dt": DecisionTreeRegressor(max_depth=5, random_state=42),
        "rf": RandomForestRegressor(
            n_estimators=100, max_depth=8, n_jobs=-1, random_state=42,
        ),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_type: Literal["binary", "regression"] = "binary",
    cv: PurgedKFold | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    """训练模型, 返回 (OOF 预测, 特征重要性, 全数据模型).

    Args:
        X_train: 特征矩阵.
        y_train: 目标 Series.
        target_type: 'binary' / 'regression'.
        cv: PurgedKFold 实例, 默认 5 折 + 5d purge + 1% embargo.

    Returns:
        oof_preds: {model_name: OOF predictions array}.
        feature_importances: {model_name: feature importance array}.
        fitted_models: {model_name: 全数据训练的 sklearn 模型}.
    """
    cv = cv or PurgedKFold()
    models = _get_models(target_type)

    Xv = X_train.values
    yv = y_train.values

    oof_preds: dict[str, np.ndarray] = {}
    feature_importances: dict[str, np.ndarray] = {}
    fitted_models: dict[str, object] = {}

    for name, model in models.items():
        oof = np.full(len(Xv), np.nan)
        for train_idx, test_idx in cv.split(X_train):
            tr_X, tr_y = Xv[train_idx], yv[train_idx]
            te_idx_clean = test_idx[test_idx < len(Xv)]
            if len(te_idx_clean) == 0 or len(np.unique(tr_y[~np.isnan(tr_y)])) < 2:
                continue
            tr_mask = ~np.isnan(tr_y)
            model.fit(tr_X[tr_mask], tr_y[tr_mask])
            oof[te_idx_clean] = model.predict(Xv[te_idx_clean])

        oof_preds[name] = oof

        full_mask = ~np.isnan(yv)
        if full_mask.sum() > 0 and len(np.unique(yv[full_mask])) >= 2:
            model.fit(Xv[full_mask], yv[full_mask])
            fitted_models[name] = model
            if hasattr(model, "feature_importances_"):
                feature_importances[name] = model.feature_importances_
            elif hasattr(model, "coef_"):
                feature_importances[name] = np.abs(model.coef_).flatten()
        else:
            feature_importances[name] = np.zeros(Xv.shape[1])

    return oof_preds, feature_importances, fitted_models


# ── 5. Soft 信号 ──────────────────────────────────────────────────────
def to_soft_signal(oof_preds: np.ndarray, target_type: str) -> np.ndarray:
    """ML 输出 → Soft 信号 ∈ [-1, +1].

    Binary: sigmoid(oof * 2) ∈ [0, 1] → 映射 [-1, +1].
    Regression: tanh(y_pred / y_std) 标准化.
    """
    oof = np.asarray(oof_preds, dtype=float)
    oof = np.nan_to_num(oof, nan=0.0)
    if target_type == "binary":
        sig = 1.0 / (1.0 + np.exp(-oof * 2.0))
        return 2.0 * sig - 1.0
    std = oof.std() if oof.std() > 0 else 1.0
    return np.tanh(oof / std)


# ── 6. 评估指标 ──────────────────────────────────────────────────────
def evaluate_classifier(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """评估二分类: AUC + accuracy."""
    from sklearn.metrics import roc_auc_score, accuracy_score
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    mask = ~np.isnan(y_score) & ~np.isnan(y_true)
    y_true_c = y_true[mask]
    y_score_c = y_score[mask]
    if len(np.unique(y_true_c)) < 2 or len(y_true_c) == 0:
        return {"auc": np.nan, "accuracy": np.nan, "n": len(y_true_c)}
    auc = float(roc_auc_score(y_true_c, y_score_c))
    acc = float(accuracy_score(y_true_c, (y_score_c > 0.5).astype(int)))
    return {"auc": auc, "accuracy": acc, "n": len(y_true_c)}


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """评估回归: RMSE + MAE + 相关系数."""
    from scipy.stats import pearsonr
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan, "n": 0}
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
    mae = float(np.abs(yt - yp).mean())
    corr = float(pearsonr(yt, yp)[0]) if yt.std() > 0 and yp.std() > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "corr": corr, "n": len(yt)}


def per_asset_ic(oof_preds: np.ndarray, fwd_returns: np.ndarray,
                 assets: np.ndarray) -> float:
    """按资产分组计算 IC (rank correlation), 然后取平均."""
    from scipy.stats import spearmanr
    if len(oof_preds) == 0:
        return np.nan
    df = pd.DataFrame({
        "pred": oof_preds, "fwd": fwd_returns, "asset": assets,
    }).dropna()
    if df.empty:
        return np.nan
    ics = []
    for _, g in df.groupby("asset"):
        if len(g) < 30:
            continue
        ic = spearmanr(g["pred"], g["fwd"]).correlation
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else np.nan


def rank_feature_importance(importances: dict[str, np.ndarray],
                            feature_names: list[str]) -> pd.DataFrame:
    """汇总多模型特征重要性, 输出排序表.

    输出列: feature, importance_logreg, importance_dt, importance_rf, importance_mean.
    """
    df = pd.DataFrame({"feature": feature_names})
    for name, imp in importances.items():
        col = f"importance_{name}"
        df[col] = imp
    imp_cols = [c for c in df.columns if c.startswith("importance_")]
    df["importance_mean"] = df[imp_cols].mean(axis=1)
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
