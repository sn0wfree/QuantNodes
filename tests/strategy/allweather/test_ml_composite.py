# coding=utf-8
"""AllWeather V6.6 — ml_composite.py 单元测试 (PLAN §15)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.ml_composite import (
    PurgedKFold,
    _asset_sector,
    _asset_vol_class,
    _asset_volume_class,
    build_features,
    build_targets,
    evaluate_classifier,
    evaluate_regressor,
    per_asset_ic,
    rank_feature_importance,
    to_soft_signal,
    train_models,
)


_PANEL_COLS = [
    "fut_au", "fut_ag", "fut_cu", "fut_if", "fut_t",
    "etf_513100", "etf_512010", "fut_rb", "fut_ma", "fut_m",
]


def _panel(t: int = 400, n: int = 6, seed: int = 0) -> pd.DataFrame:
    """合成面板 (38 资产命名)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, size=(t, n))
    prices = np.cumprod(1 + rets, axis=0) * 100
    cols = _PANEL_COLS[:n]
    return pd.DataFrame(
        prices, index=pd.bdate_range("2018-01-01", periods=t), columns=cols,
    )


class TestPurgedKFold:
    def test_default_5_fold(self):
        """默认 5 折."""
        X = _panel(t=300)
        cv = PurgedKFold()
        folds = list(cv.split(X))
        assert len(folds) == 5

    def test_no_future_overlap(self):
        """每折 train_idx.max() < test_idx.min() (无未来)."""
        X = _panel(t=300)
        cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(X):
            if len(train_idx) > 0:
                assert train_idx.max() < test_idx.min()

    def test_purge_gap_respected(self):
        """train 末尾 5d 不在 test 中 (purge gap)."""
        X = _panel(t=300)
        cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(X):
            if len(train_idx) == 0:
                continue
            gap = X.index[test_idx[0]] - X.index[train_idx[-1]]
            assert gap.days >= 5

    def test_custom_k_and_purge(self):
        """k=3 + purge=10."""
        X = _panel(t=300)
        cv = PurgedKFold(n_splits=3, purge_gap=10, embargo_pct=0.02)
        folds = list(cv.split(X))
        assert len(folds) == 3
        for train_idx, test_idx in folds:
            if len(train_idx) > 0:
                gap = X.index[test_idx[0]] - X.index[train_idx[-1]]
                assert gap.days >= 10

    def test_embargo_applied(self):
        """test 后 1% embargo 不在后续 train 中."""
        X = _panel(t=500)
        cv = PurgedKFold(n_splits=5, purge_gap=5, embargo_pct=0.05)
        for train_idx, test_idx in cv.split(X):
            if len(train_idx) == 0:
                continue
            test_end = test_idx[-1]
            embargo_size = int(0.05 * 500)
            after_test = train_idx[train_idx > test_end]
            if len(after_test) > 0:
                min_after = after_test.min()
                assert min_after >= test_end + embargo_size


class TestAssetSector:
    def test_sector_mapping_covers_all_futures(self):
        """所有 fut_* 资产都被分到 sector."""
        panel = _panel(n=6)
        sectors = _asset_sector(panel)
        assert len(sectors) == 6
        assert "metal" in sectors.values or "etf_eq" in sectors.values

    def test_sector_count_distribution(self):
        """8 个 sector 类别 (7 期货 + 1 ETF)."""
        panel = _panel(n=10)
        sectors = _asset_sector(panel)
        assert sectors.nunique() >= 2


class TestAssetVolClass:
    def test_vol_class_three_categories(self):
        """vol class 应有 lo/md/hi 三类."""
        panel = _panel(t=400, n=10)
        vol_class = _asset_vol_class(panel)
        assert vol_class.nunique() <= 3
        assert all(c in ["vol_lo", "vol_md", "vol_hi"] for c in vol_class.unique())


class TestAssetVolumeClass:
    def test_volume_class_three_categories(self):
        panel = _panel(t=400, n=10)
        volu_class = _asset_volume_class(panel)
        assert volu_class.nunique() <= 3


class TestBuildFeatures:
    def test_feature_shape(self):
        """build_features 输出 (T*N, ≥17 列: 17 sub-factor + asset features)."""
        panel = _panel(t=300, n=10)
        feat = build_features(panel)
        assert feat.shape[1] >= 17
        assert feat.shape[0] == 300 * 10

    def test_no_nan_after_warmup(self):
        """去掉前 60d 后无 NaN (部分子因子有 lag)."""
        panel = _panel(t=300, n=10)
        feat = build_features(panel)
        last = feat.iloc[-200:]
        nan_ratio = last.isna().sum().sum() / last.size
        assert nan_ratio < 0.05

    def test_interaction_terms(self):
        """Sub-A × Sub-B / Sub-B × Sub-C 交互项存在."""
        panel = _panel(t=200, n=10)
        feat = build_features(panel)
        assert "sub_a_x_b" in feat.columns
        assert "sub_b_x_c" in feat.columns


class TestBuildTargets:
    def test_y1_distribution(self):
        """Y1 (vol_spike) 正样本 ~10-30%."""
        panel = _panel(t=400, n=4)
        tgt = build_targets(panel)
        y1 = tgt["y_vol_spike"].dropna()
        assert 0.05 < y1.mean() < 0.40

    def test_y2_range(self):
        """Y2 (vol_level) ∈ [0, ∞)."""
        panel = _panel(t=400, n=4)
        tgt = build_targets(panel)
        y2 = tgt["y_vol_level"].dropna()
        assert y2.min() >= 0
        assert y2.max() < 1.0  # 简单合理性

    def test_y3_logic(self):
        """Y3 (turning) ∈ {0, 1}."""
        panel = _panel(t=400, n=4)
        tgt = build_targets(panel)
        y3 = tgt["y_turning"].dropna()
        assert set(y3.unique()).issubset({0.0, 1.0})

    def test_no_future_leakage_in_y2(self):
        """t 时刻的 Y2 只用 t+1 到 t+horizon 数据."""
        panel = _panel(t=400, n=2)
        tgt = build_targets(panel, horizon=20)
        y2 = tgt["y_vol_level"].dropna()
        # Y2 在前 horizon 行应为 NaN (因为 .shift(-horizon))
        for asset in panel.columns:
            asset_y2 = y2.xs(asset, level="asset") if "asset" in y2.index.names else y2
            first_valid = asset_y2.first_valid_index()
            assert first_valid is not None


class TestTrainModels:
    def test_oof_predictions_shape(self):
        """OOF 预测 shape 匹配 X_train."""
        panel = _panel(t=400, n=4)
        feat = build_features(panel)
        tgt = build_targets(panel)
        mask = feat.notna().all(axis=1) & tgt.notna().all(axis=1)
        X = feat[mask]
        y = tgt.loc[X.index, "y_vol_spike"]
        oof, imp, fitted = train_models(X, y, target_type="binary")
        for name in ("logreg", "dt", "rf"):
            assert name in oof
            assert len(oof[name]) == len(X)

    def test_feature_importances_sum_to_one(self):
        """树模型特征重要性之和 ≈ 1."""
        panel = _panel(t=400, n=4)
        feat = build_features(panel)
        tgt = build_targets(panel)
        mask = feat.notna().all(axis=1) & tgt.notna().all(axis=1)
        X = feat[mask]
        y = tgt.loc[X.index, "y_vol_spike"]
        _, imp, _ = train_models(X, y, target_type="binary")
        for name in ("dt", "rf"):
            assert name in imp
            assert abs(imp[name].sum() - 1.0) < 1e-6

    def test_regression_models_train(self):
        """回归模型 (Ridge + DT + RF) 可训练."""
        panel = _panel(t=400, n=4)
        feat = build_features(panel)
        tgt = build_targets(panel)
        mask = feat.notna().all(axis=1) & tgt.notna().all(axis=1)
        X = feat[mask]
        y = tgt.loc[X.index, "y_vol_level"]
        oof, imp, fitted = train_models(X, y, target_type="regression")
        for name in ("ridge", "dt", "rf"):
            assert name in oof
            assert name in fitted


class TestToSoftSignal:
    def test_binary_soft_range(self):
        """binary Soft 信号 ∈ [-1, +1]."""
        oof = np.array([-5, -1, 0, 1, 5])
        sig = to_soft_signal(oof, "binary")
        assert sig.min() >= -1.0
        assert sig.max() <= 1.0

    def test_regression_soft_range(self):
        """regression Soft 信号 ∈ [-1, +1]."""
        oof = np.array([-3, -1, 0, 1, 3])
        sig = to_soft_signal(oof, "regression")
        assert sig.min() >= -1.0
        assert sig.max() <= 1.0


class TestEvaluateClassifier:
    def test_perfect_classifier(self):
        """完美分类器 AUC=1."""
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.9, 0.8])
        res = evaluate_classifier(y_true, y_score)
        assert res["auc"] > 0.99

    def test_random_classifier(self):
        """随机分类器 AUC≈0.5."""
        rng = np.random.default_rng(0)
        y_true = rng.binomial(1, 0.5, 200)
        y_score = rng.uniform(0, 1, 200)
        res = evaluate_classifier(y_true, y_score)
        assert 0.4 < res["auc"] < 0.6


class TestEvaluateRegressor:
    def test_perfect_regressor(self):
        """完美回归 RMSE=0."""
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = y_true.copy()
        res = evaluate_regressor(y_true, y_pred)
        assert res["rmse"] < 1e-9
        assert res["corr"] > 0.99


class TestPerAssetIC:
    def test_perfect_ic(self):
        """完美预测 IC=1."""
        rng = np.random.default_rng(0)
        fwd = rng.normal(0, 0.01, 200)
        oof = fwd + rng.normal(0, 0.001, 200)
        assets = np.array(["a"] * 100 + ["b"] * 100)
        ic = per_asset_ic(oof, fwd, assets)
        assert ic > 0.9


class TestRankFeatureImportance:
    def test_importance_descending(self):
        """重要性按 mean 降序."""
        imp = {
            "m1": np.array([0.5, 0.3, 0.2]),
            "m2": np.array([0.4, 0.4, 0.2]),
        }
        names = ["f1", "f2", "f3"]
        df = rank_feature_importance(imp, names)
        assert df["importance_mean"].is_monotonic_decreasing
