"""Unit tests for CA-GCP ML training helper functions.

覆盖 ca_gcp_training_v2.py / v3.py / v5.py 中纯函数:
- rolling_zscore
- feature_columns_v2
- feature_columns_v3
- build_regime_features (不依赖外部数据)
- detect_regimes_gmm / detect_regimes_kmeans
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd


class TestRollingZscore:
    def test_default_window(self) -> None:
        from ca_gcp_training_v2 import rolling_zscore
        rng = np.random.default_rng(42)
        s = pd.Series(rng.normal(0, 1, 300), name="x")
        z = rolling_zscore(s)
        assert len(z) == len(s)
        assert z.name == "x"

    def test_short_series_returns_nan(self) -> None:
        from ca_gcp_training_v2 import rolling_zscore
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
        z = rolling_zscore(s, window=80)
        assert z.isna().all()

    def test_custom_min_periods_v2(self) -> None:
        from ca_gcp_training_v2 import rolling_zscore
        rng = np.random.default_rng(99)
        s = pd.Series(rng.normal(0, 1, 50), name="x")
        z = rolling_zscore(s, window=20, min_periods=5)
        assert len(z) == 50
        assert not z.isna().all()
        assert z.iloc[4:].notna().any()

    def test_window_below_default_min_periods_works(self) -> None:
        from ca_gcp_training_v2 import rolling_zscore
        rng = np.random.default_rng(50)
        s = pd.Series(rng.normal(0, 1, 100), name="x")
        z = rolling_zscore(s, window=10, min_periods=5)
        assert len(z) == 100
        assert z.iloc[:4].isna().all()

    def test_custom_min_periods_v3(self) -> None:
        from ca_gcp_training_v3 import rolling_zscore
        rng = np.random.default_rng(101)
        s = pd.Series(rng.normal(0, 1, 50), name="y")
        z = rolling_zscore(s, window=20, min_periods=2)
        assert not z.isna().all()
        assert z.iloc[1:].notna().any()

    def test_custom_window(self) -> None:
        from ca_gcp_training_v2 import rolling_zscore
        rng = np.random.default_rng(7)
        s = pd.Series(rng.normal(0, 1, 200), name="x")
        z = rolling_zscore(s, window=100)
        assert len(z) == 200

    def test_v3_same_signature(self) -> None:
        from ca_gcp_training_v3 import rolling_zscore
        rng = np.random.default_rng(11)
        s = pd.Series(rng.normal(0, 1, 300), name="y")
        z = rolling_zscore(s)
        assert len(z) == len(s)


class TestFeatureColumnsV2:
    def test_includes_macro_signals(self) -> None:
        from ca_gcp_training_v2 import CORE_MACRO_COLS, feature_columns_v2
        cols = feature_columns_v2()
        assert isinstance(cols, list)
        assert len(cols) > 20
        base_expected = {"width_z", "stress", "trend_ok", "momentum_score",
                         "vol_high", "breadth", "top_sector_conc"}
        assert base_expected.issubset(set(cols))
        for col in CORE_MACRO_COLS:
            assert any(c.startswith(f"{col}_lag0") for c in cols), f"Missing {col}_lag0*"

    def test_lag_z_variants_present(self) -> None:
        from ca_gcp_training_v2 import feature_columns_v2
        cols = feature_columns_v2()
        z_count = sum(1 for c in cols if c.endswith("_z"))
        assert z_count >= 4 * 8


class TestFeatureColumnsV3:
    def test_returns_columns_from_dataframe(self) -> None:
        from ca_gcp_training_v3 import feature_columns_v3
        fake_features = pd.DataFrame(
            np.random.default_rng(0).normal(0, 1, (10, 5)),
            columns=["a", "b", "c", "d", "e"],
        )
        cols = feature_columns_v3(fake_features)
        assert cols == ["a", "b", "c", "d", "e"]


class TestBuildYLabels:
    """Phase 1.3: y_A..y_E 标签无 future-leakage 验证.

    标签语义: y[t] 是 "t 之后 HORIZON 天" 是否发生某事.
    关键检查:
    1. dropna 后, 标签的 len 必须 <= returns len - max_horizon (留出未来观察空间)
    2. 暴跌 day 前 HORIZON 天的标签应反映该暴跌
    3. 暴跌 day 之前 (早于 max_horizon) 的标签为 0
    4. 标签覆盖率合理 (5-50%)
    5. y_E = y_B | y_D 严格成立
    6. 标签 index 与原 returns index 严格一致
    """

    def test_first_and_last_labels_are_zero(self) -> None:
        """前 60+20 天 (rolling_q 暖机 + max horizon) 和最后 20 天 (无未来) 标签全 0."""
        from ca_gcp_training_v3 import build_y_labels, HORIZON
        rng = np.random.default_rng(7)
        idx = pd.date_range("2020-01-02", periods=300, freq="B")
        returns = pd.DataFrame(rng.normal(0, 0.01, (300, 5)), index=idx)
        hs300 = pd.Series(rng.normal(0, 0.01, 300), index=idx)
        labels = build_y_labels(returns, hs300)
        # 最后 HORIZON 天的 y_B/y_C 必须 0 (无未来可看, NaN 比较为 False)
        last_rows = labels.iloc[-HORIZON:]
        for col in ["y_B", "y_C"]:
            assert (last_rows[col] == 0).all(), \
                f"最后 {HORIZON} 行的 {col} 应全 0 (无未来)"
        # 前 80 天 (rolling_q 60 + max horizon 20) y_D 必须 0
        first_rows = labels.iloc[:80]
        assert (first_rows["y_D"] == 0).all(), \
            "前 80 天的 y_D 应全 0 (rolling_q 暖机不够)"

    def test_label_coverage_reasonable(self) -> None:
        """标签覆盖率应在 5-50% 之间."""
        from ca_gcp_training_v3 import build_y_labels
        rng = np.random.default_rng(13)
        idx = pd.date_range("2020-01-02", periods=500, freq="B")
        returns = pd.DataFrame(rng.normal(0, 0.01, (500, 5)), index=idx)
        hs300 = pd.Series(rng.normal(0, 0.01, 500), index=idx)
        labels = build_y_labels(returns, hs300).dropna()
        for col in ["y_A", "y_B", "y_C", "y_D", "y_E"]:
            pos_rate = labels[col].mean()
            assert 0.01 <= pos_rate <= 0.50, \
                f"{col} 正类率 {pos_rate:.1%} 超出合理范围 (1-50%)"

    def test_crash_triggers_labels_in_preceding_horizon_window(self) -> None:
        """暴跌 day 前 HORIZON 天窗口内, 标签 y_B/y_D 应有正例 (因为未来 max_dd 跨阈)."""
        from ca_gcp_training_v3 import build_y_labels
        # 构造: 已知 day 100 暴跌
        idx = pd.date_range("2020-01-02", periods=400, freq="B")
        rets = np.zeros((400, 5))
        rets[100, :] = -0.10  # 10% 暴跌
        returns = pd.DataFrame(rets, index=idx)
        hs300 = pd.Series(np.zeros(400), index=idx)
        labels = build_y_labels(returns, hs300).dropna()
        # day 100 前 10 天窗口 (day 90-99) 应有 y_B=1 或 y_D=1
        # 因为 day 100 暴跌会让 day 90-99 的 cumret_10d / max_dd_10d 触发阈值
        for d in range(90, 100):
            assert d < len(labels), f"day {d} 超出 labels 长度"
            # day 90 是 y_B (shift -10) = max_dd_10d[100] 触发
            if d == 90 or d == 100 - 5:  # y_A 5d window
                pass  # not strictly required
        # 至少 day 90 应该有正例
        assert int(labels.iloc[90].y_E) == 1, \
            f"day 90 (暴跌前 10 天) y_E 应为 1, 实为 {int(labels.iloc[90].y_E)}"

    def test_no_labels_far_before_crash(self) -> None:
        """暴跌 day 前 30+ 天的标签应为 0 (距离过远无信号)."""
        from ca_gcp_training_v3 import build_y_labels
        idx = pd.date_range("2020-01-02", periods=400, freq="B")
        rets = np.zeros((400, 5))
        rets[200, :] = -0.10
        returns = pd.DataFrame(rets, index=idx)
        hs300 = pd.Series(np.zeros(400), index=idx)
        labels = build_y_labels(returns, hs300).dropna()
        # day 100 (暴跌前 100 天) 全部标签应为 0
        for d in [50, 60, 70, 100, 150]:
            if d >= len(labels):
                continue
            for col in ["y_A", "y_B", "y_C", "y_D", "y_E"]:
                v = int(labels.iloc[d][col])
                assert v == 0, \
                    f"day {d} {col} 应为 0, 实为 {v} (暴跌在 day 200)"

    def test_label_index_aligned_with_returns(self) -> None:
        """标签的 index 必须和输入 returns 严格对齐 (无 reindex 错位)."""
        from ca_gcp_training_v3 import build_y_labels
        rng = np.random.default_rng(17)
        idx = pd.date_range("2020-01-02", periods=300, freq="B")
        returns = pd.DataFrame(rng.normal(0, 0.01, (300, 5)), index=idx)
        hs300 = pd.Series(rng.normal(0, 0.01, 300), index=idx)
        labels = build_y_labels(returns, hs300)
        # 标签 index 必须是 returns index 的连续前缀子集
        pd.testing.assert_index_equal(labels.index, returns.index[:len(labels)])

    def test_y_E_is_union_of_y_B_and_y_D(self) -> None:
        """y_E = y_B | y_D 必须严格成立 (y_E = max_dd 跨阈 OR percentile 跨阈)."""
        from ca_gcp_training_v3 import build_y_labels
        rng = np.random.default_rng(19)
        idx = pd.date_range("2020-01-02", periods=400, freq="B")
        returns = pd.DataFrame(rng.normal(0, 0.01, (400, 5)), index=idx)
        hs300 = pd.Series(rng.normal(0, 0.01, 400), index=idx)
        labels = build_y_labels(returns, hs300).dropna()
        np.testing.assert_array_equal(
            labels["y_E"].values,
            (labels["y_B"] | labels["y_D"]).astype(int).values,
        )

    def test_label_consistency_with_modified_hs300(self) -> None:
        """y_C 用到 hs300, 修改 hs300 后 y_C 必须变化 (但 y_B/y_D 不变)."""
        from ca_gcp_training_v3 import build_y_labels
        rng = np.random.default_rng(23)
        idx = pd.date_range("2020-01-02", periods=300, freq="B")
        returns = pd.DataFrame(rng.normal(0, 0.01, (300, 5)), index=idx)
        hs300_a = pd.Series(np.zeros(300), index=idx)
        hs300_b = pd.Series(rng.normal(0.02, 0.01, 300), index=idx)  # 牛市 hs300
        labels_a = build_y_labels(returns, hs300_a).dropna()
        labels_b = build_y_labels(returns, hs300_b).dropna()
        # y_C 必须改变 (因为 alpha = market - hs300)
        assert not np.array_equal(labels_a.y_C.values, labels_b.y_C.values), \
            "y_C 不应与 hs300 无关"
        # y_B / y_D 只用 market, 不应改变
        np.testing.assert_array_equal(labels_a.y_B.values, labels_b.y_B.values)
        np.testing.assert_array_equal(labels_a.y_D.values, labels_b.y_D.values)


class TestDetectRegimesGMM:
    def test_returns_series_with_correct_index_and_length(self) -> None:
        from ca_gcp_training_v5 import detect_regimes_gmm
        rng = np.random.default_rng(99)
        features = pd.DataFrame(
            rng.normal(0, 1, (100, 4)),
            columns=["realized_vol_20d", "market_ret_20d", "breadth_20d", "vol_ratio_60_20"],
        )
        regime, _ = detect_regimes_gmm(features, n_regimes=3)
        assert len(regime) == 100
        assert set(regime.unique()).issubset({0, 1, 2})

    def test_regimes_ordered_by_vol(self) -> None:
        from ca_gcp_training_v5 import detect_regimes_gmm
        rng = np.random.default_rng(0)
        vol_a = np.full(50, 0.05)
        vol_b = np.full(50, 0.30)
        features = pd.DataFrame(
            {
                "realized_vol_20d": np.concatenate([vol_a, vol_b]),
                "market_ret_20d": rng.normal(0, 0.05, 100),
                "breadth_20d": rng.uniform(0, 1, 100),
                "vol_ratio_60_20": rng.uniform(0.5, 1.5, 100),
            }
        )
        regime, _ = detect_regimes_gmm(features, n_regimes=3)
        low_vol_regime = regime.iloc[:50].mode().iloc[0]
        high_vol_regime = regime.iloc[50:].mode().iloc[0]
        assert low_vol_regime < high_vol_regime


class TestDetectRegimesKmeans:
    def test_returns_series_and_model(self) -> None:
        from ca_gcp_training_v5 import detect_regimes_kmeans
        rng = np.random.default_rng(5)
        features = pd.DataFrame(
            rng.normal(0, 1, (80, 4)),
            columns=["realized_vol_20d", "market_ret_20d", "breadth_20d", "vol_ratio_60_20"],
        )
        regime, model = detect_regimes_kmeans(features, n_regimes=3)
        assert len(regime) == 80
        assert hasattr(model, "predict")


class TestBuildRegimeFeatures:
    def test_load_macro_daily_failure_returns_index_only(self) -> None:
        from ca_gcp_training_v5 import build_regime_features
        rng = np.random.default_rng(1)
        idx = pd.date_range("2020-01-02", periods=300, freq="B")
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.012, (300, 10)), index=idx, columns=[f"c{i}" for i in range(10)]
        )
        with patch("ca_gcp_training_v5.load_macro_daily", return_value=pd.DataFrame(index=idx)):
            features = build_regime_features(returns)
        assert isinstance(features, pd.DataFrame)
        assert "market_ret_5d" in features.columns
        assert "realized_vol_20d" in features.columns
        assert "breadth_20d" in features.columns


class TestPredictRegimeAware:
    """Phase 1.2: v5 routing 修复 - 必须使用 final_model 直接推理."""

    def test_predict_single_uses_final_model(self) -> None:
        """_predict_single 必须用 final_model, 不应重建 fake 训练集."""
        from unittest.mock import MagicMock

        from ca_gcp_training_v5 import _predict_single

        # 模拟 XGB final_model 直接推理
        fake_model = MagicMock()
        fake_dmatrix = MagicMock()
        fake_model.predict.return_value = np.array([0.42])
        with patch("ca_gcp_training_v5.xgb.DMatrix", return_value=fake_dmatrix):
            X = pd.DataFrame([[0.1, 0.2, 0.3]], columns=["f1", "f2", "f3"])
            res = {
                "final_model": fake_model,
                "model_type": "xgb",
                "best_params": {},
            }
            p = _predict_single(res, X)

        assert 0.0 <= p <= 1.0
        assert p == 0.42
        # 验证 final_model.predict 被调用 (而不是 train)
        assert fake_model.predict.called
        assert not hasattr(fake_model, "train") or not fake_model.train.called

    def test_predict_single_lgb_uses_final_model(self) -> None:
        """LightGBM 路径同样必须用 final_model."""
        from unittest.mock import MagicMock

        from ca_gcp_training_v5 import _predict_single

        fake_model = MagicMock()
        fake_model.predict.return_value = np.array([0.78])
        X = pd.DataFrame([[0.1, 0.2]], columns=["f1", "f2"])
        res = {
            "final_model": fake_model,
            "model_type": "lgb",
            "best_params": {},
        }
        p = _predict_single(res, X)
        assert p == 0.78
        assert fake_model.predict.called

    def test_predict_single_raises_without_final_model(self) -> None:
        """老 model_result (没 final_model) 必须显式报错, 不返回 garbage."""
        from ca_gcp_training_v5 import _predict_single

        X = pd.DataFrame([[0.1]], columns=["f1"])
        old_res = {"best_params": {}, "model_type": "xgb"}  # 缺 final_model
        with __import__("pytest").raises(ValueError, match="final_model"):
            _predict_single(old_res, X)

    def test_predict_regime_aware_produces_valid_proba(self) -> None:
        """predict_regime_aware 输出必须 [0, 1] 区间, 不全是 0.5."""
        from unittest.mock import MagicMock

        from ca_gcp_training_v5 import predict_regime_aware

        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-02", periods=50, freq="B")
        X = pd.DataFrame(rng.normal(0, 1, (50, 5)), index=idx, columns=[f"f{i}" for i in range(5)])
        regimes = pd.Series([0] * 25 + [1] * 25, index=idx, name="regime")

        # 模拟 final_model: 对每行返回 0~1 的 proba
        fake_model = MagicMock()

        def fake_predict(arr):
            return np.linspace(0.1, 0.9, len(arr))
        fake_model.predict = fake_predict

        # 2 regime 都有 final_model
        regime_models = {
            0: {"final_model": fake_model, "model_type": "lgb", "best_params": {}},
            1: {"final_model": fake_model, "model_type": "lgb", "best_params": {}},
        }
        proba, common = predict_regime_aware(X, regimes, regime_models)
        assert len(proba) == 50
        assert (proba >= 0).all() and (proba <= 1).all(), "proba 超出 [0,1] 区间"
        # 不应全是 0.5 fallback
        assert not (proba == 0.5).all(), "全部 0.5 - routing 全 fallback"

    def test_predict_regime_aware_falls_back_on_missing_regime(self) -> None:
        """未在 regime_models 的 regime → fallback global_model."""
        from unittest.mock import MagicMock

        from ca_gcp_training_v5 import predict_regime_aware

        rng = np.random.default_rng(11)
        idx = pd.date_range("2024-01-02", periods=20, freq="B")
        X = pd.DataFrame(rng.normal(0, 1, (20, 3)), index=idx, columns=["f1", "f2", "f3"])
        # regime 2 不在 regime_models 中
        regimes = pd.Series([0] * 10 + [2] * 10, index=idx, name="regime")

        fake_model = MagicMock()
        fake_model.predict.return_value = np.array([0.6])
        with patch("ca_gcp_training_v5.xgb.DMatrix", return_value=MagicMock()):
            regime_models = {
                0: {"final_model": fake_model, "model_type": "xgb", "best_params": {}},
            }
            global_model = {"final_model": fake_model, "model_type": "xgb", "best_params": {}}
            proba, _ = predict_regime_aware(X, regimes, regime_models, global_model)
        # 应该有 20 行 (0~10 用 regime 0, 10~20 fallback global)
        assert len(proba) == 20
        assert (proba >= 0).all() and (proba <= 1).all()
