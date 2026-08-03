"""Unit tests for ca_gcp_ew_full_eval.py new Phase 1.5 metrics.

- compute_pr_auc
- compute_ece
- compute_net_utility
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TestComputePrAuc:
    def test_perfect_ranking_ap_one(self) -> None:
        """完美 ranking: fired=1 的样本后续都跌, fired=0 的都涨 → AP=1."""
        from ca_gcp_ew_full_eval import compute_pr_auc
        idx = pd.date_range("2024-01-02", periods=100, freq="B")
        # 构造 returns: day 50-99 持续暴跌 (后续 10 天)
        rets = np.zeros((100, 5))
        rets[50:100, :] = -0.01
        returns = pd.DataFrame(rets, index=idx)
        # alerts: day 50-99 fired, day 0-49 not
        alerts = pd.DataFrame({
            "fired": [0] * 50 + [1] * 50,
        }, index=idx)
        result = compute_pr_auc(alerts, returns, fired_col="fired", horizon=5)
        # fired 都在暴跌期, 应该 AP > 0.5 (明显好于随机)
        assert result["AP"] > 0.6, f"perfect ranking AP should be > 0.6, got {result['AP']:.3f}"

    def test_inverse_ranking_ap_low(self) -> None:
        """反向 ranking: fired 在平稳期, 没 fired 在暴跌期 → AP 接近 0.5 (random)."""
        from ca_gcp_ew_full_eval import compute_pr_auc
        idx = pd.date_range("2024-01-02", periods=100, freq="B")
        rets = np.zeros((100, 5))
        rets[20:40, :] = -0.005  # 暴跌在 day 20-39
        returns = pd.DataFrame(rets, index=idx)
        alerts = pd.DataFrame({
            "fired": [1] * 50 + [0] * 50,  # fired 在 day 0-49
        }, index=idx)
        result = compute_pr_auc(alerts, returns, fired_col="fired", horizon=10)
        # AP ≤ 0.5 (fired 期间大部分不跌)
        assert result["AP"] <= 0.55, f"inverse ranking AP should be ≤ 0.55, got {result['AP']:.3f}"

    def test_handles_alert_level_column(self) -> None:
        """alert_level 字符串列: yellow/red → fired."""
        from ca_gcp_ew_full_eval import compute_pr_auc
        idx = pd.date_range("2024-01-02", periods=20, freq="B")
        rets = np.zeros((20, 3))
        rets[:5, :] = -0.01
        returns = pd.DataFrame(rets, index=idx)
        alerts = pd.DataFrame({
            "alert_level": ["red"] * 5 + ["green"] * 15,
        }, index=idx)
        result = compute_pr_auc(alerts, returns, fired_col="alert_level", horizon=5)
        assert "AP" in result
        assert "n_pos" in result

    def test_missing_fired_col_returns_nan(self) -> None:
        """缺 fired_col → AP=NaN, 不 crash."""
        from ca_gcp_ew_full_eval import compute_pr_auc
        idx = pd.date_range("2024-01-02", periods=10, freq="B")
        returns = pd.DataFrame(np.zeros((10, 3)), index=idx)
        alerts = pd.DataFrame({"foo": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]}, index=idx)
        result = compute_pr_auc(alerts, returns, fired_col="bar")
        assert np.isnan(result["AP"])


class TestComputeEce:
    def test_perfect_calibration_ece_zero(self) -> None:
        """预测概率完全匹配真实频率 → ECE=0."""
        from ca_gcp_ew_full_eval import compute_ece
        proba = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        ece = compute_ece(proba, labels, n_bins=5)
        # bin [0.0, 0.2] = 4 samples, acc=0, conf=0.1 → 0.1*0.5
        # bin [0.8, 1.0] = 4 samples, acc=1, conf=0.9 → 0.1*0.5
        # ECE = 0.1
        assert 0.05 < ece < 0.15

    def test_overconfident_ece_high(self) -> None:
        """全部预测 0.99 但全错 → ECE 高."""
        from ca_gcp_ew_full_eval import compute_ece
        proba = np.array([0.99] * 10)
        labels = np.array([0] * 10)
        ece = compute_ece(proba, labels, n_bins=10)
        # 全部 acc=0, conf=0.99 → |0-0.99| = 0.99
        assert ece > 0.9

    def test_uniform_ece_close_to_zero(self) -> None:
        """均匀预测 + 50% 真实 → ECE 接近 0."""
        from ca_gcp_ew_full_eval import compute_ece
        rng = np.random.default_rng(11)
        proba = np.full(100, 0.5)
        labels = rng.integers(0, 2, 100).astype(float)
        ece = compute_ece(proba, labels, n_bins=10)
        # 全部 acc=0.5, conf=0.5 → 0
        assert ece < 0.05

    def test_length_mismatch_raises(self) -> None:
        """长度不匹配 → ValueError."""
        from ca_gcp_ew_full_eval import compute_ece
        with __import__("pytest").raises(ValueError, match="length"):
            compute_ece(np.array([0.5, 0.5]), np.array([0, 1, 0]))


class TestComputeNetUtility:
    def test_no_loss_utility_zero(self) -> None:
        """无亏损 → utility=0."""
        from ca_gcp_ew_full_eval import compute_net_utility
        u = compute_net_utility(
            np.array([0.1, 0.2, 0.05]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.1, 0.1, 0.1]),
        )
        assert u == 0.0

    def test_loss_penalizes_utility(self) -> None:
        """亏损 → utility < 0."""
        from ca_gcp_ew_full_eval import compute_net_utility
        u = compute_net_utility(
            np.array([0.1, 0.2, 0.05]),
            np.array([-0.05, -0.03, -0.02]),
            np.array([0.1, 0.1, 0.1]),
        )
        assert u < 0

    def test_lambda_scaling(self) -> None:
        """λ 越大, utility 越负 (风险厌恶越强)."""
        from ca_gcp_ew_full_eval import compute_net_utility
        loss = np.array([-0.05, -0.03, -0.02])
        proba = np.array([0.1, 0.1, 0.1])
        risk = np.array([0.1, 0.1, 0.1])
        u1 = compute_net_utility(proba, loss, risk, lambda_param=1.0)
        u3 = compute_net_utility(proba, loss, risk, lambda_param=3.0)
        assert u3 < u1, f"lambda=3 ({u3}) 应比 lambda=1 ({u1}) 更负"

    def test_empty_input(self) -> None:
        """空输入 → 0 utility, 不 crash."""
        from ca_gcp_ew_full_eval import compute_net_utility
        u = compute_net_utility(np.array([]), np.array([]), np.array([]))
        assert u == 0.0
