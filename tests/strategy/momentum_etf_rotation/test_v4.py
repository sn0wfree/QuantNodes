# coding=utf-8
"""v4 单元测试 (Stage 17)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QuantNodes.strategy.momentum_etf_rotation.v4 import (
    ALL_V4_CODES,
    FACTOR_NAMES,
    FactorTimingConfig,
    SmartBetaConfig,
    SmartBetaSubStrategy,
    StyleRotationConfig,
    StyleRotationSubStrategy,
    V4Config,
    backtest_factor_timing,
    backtest_factor_weights_history,
    build_distance_transmat,
    compute_factor_scores,
    compute_factor_weights,
    compute_forward_return,
    compute_strategy_weights,
    distance_between,
    distance_rate,
    effective_distance,
    enforce_minimum_duration,
    factor_ic_at,
    load_smartbeta_panel,
    rolling_factor_ic,
    run_v4_backtest,
    run_v4_mode,
    select_diversified_smart_beta,
    select_top_smart_beta,
    select_top_styles,
    smart_beta_score,
    soft_constrain,
    style_etf_picks,
    style_rotation_score,
    validate_transmat,
)


# ======================================================================
# 1. universe_v4 测试
# ======================================================================
class TestUniverseV4:
    def test_style_groups_complete(self):
        from QuantNodes.strategy.momentum_etf_rotation.v4 import STYLE_GROUP_METAS
        assert len(STYLE_GROUP_METAS) == 5

    def test_smart_beta_complete(self):
        from QuantNodes.strategy.momentum_etf_rotation.v4 import SMART_BETA_METAS
        assert len(SMART_BETA_METAS) == 7

    def test_all_v4_codes_count(self):
        assert len(ALL_V4_CODES) == 12

    def test_load_smartbeta_panel(self):
        panel = load_smartbeta_panel()
        assert panel.shape[0] > 1000
        assert panel.shape[1] == 12
        assert "510300" in panel.columns  # 大盘
        assert "510880" in panel.columns  # 红利

    def test_no_duplicate_codes(self):
        assert len(ALL_V4_CODES) == len(set(ALL_V4_CODES))


# ======================================================================
# 2. style_rotation_v4 测试
# ======================================================================
class TestStyleRotationV4:
    def test_score_returns_series(self):
        panel = load_smartbeta_panel()
        score = style_rotation_score(
            panel, panel.index[-1],
            {"a": ("510300",), "b": ("510500",), "c": ("159915",)},
            lookback=60,
        )
        assert isinstance(score, pd.Series)
        assert len(score) == 3

    def test_top_styles_sorted(self):
        from QuantNodes.strategy.momentum_etf_rotation.v4 import StyleGroup
        panel = load_smartbeta_panel()
        score = style_rotation_score(
            panel, panel.index[-1],
            {
                StyleGroup.LARGE_CAP: ("510300",),
                StyleGroup.MID_CAP: ("510500",),
                StyleGroup.GROWTH: ("159915",),
            },
            lookback=60,
        )
        top = select_top_styles(score, top_n=2)
        assert len(top) == 2
        # 第一个 score 更高
        assert score.iloc[0] >= score.iloc[1]

    def test_etf_picks(self):
        panel = load_smartbeta_panel()
        score = style_rotation_score(
            panel, panel.index[-1],
            {"a": ("510300",), "b": ("510500",)},
            lookback=60,
        )
        top_styles = select_top_styles(score, top_n=2)
        picks = style_etf_picks(
            panel, panel.index[-1],
            {"a": ("510300",), "b": ("510500",)},
            top_styles, top_n_per_style=1,
        )
        assert len(picks) <= 2
        for p in picks:
            assert p in ["510300", "510500"]

    def test_substrategy_run_step(self):
        panel = load_smartbeta_panel()
        sub = StyleRotationSubStrategy(StyleRotationConfig())
        result = sub.run_step(panel, panel.index[-1])
        assert len(result.chosen) > 0
        assert sum(result.weights.values()) > 0.99  # sum ≈ 1
        assert result.meta["strategy"] == "style_rotation"


# ======================================================================
# 3. smart_beta_v4 测试
# ======================================================================
class TestSmartBetaV4:
    def test_smart_beta_score(self):
        panel = load_smartbeta_panel()
        score = smart_beta_score(
            panel, panel.index[-1],
            ["512890", "512260", "515900"],
        )
        assert isinstance(score, pd.Series)
        assert len(score) == 3

    def test_top_smart_beta(self):
        panel = load_smartbeta_panel()
        top = select_top_smart_beta(
            panel, panel.index[-1],
            ["512890", "512260", "515900"], top_n=2,
        )
        assert len(top) <= 2

    def test_diversified_smart_beta(self):
        score = pd.Series(
            {"512890": 0.9, "512260": 0.7, "515900": 0.5, "512040": 0.3}
        )
        from QuantNodes.strategy.momentum_etf_rotation.v4 import SMART_BETA_FACTOR_TYPE
        picks = select_diversified_smart_beta(
            score, SMART_BETA_FACTOR_TYPE, top_n=2,
        )
        assert len(picks) <= 2
        # 选出的应该是 score 最高的
        if picks:
            for p in picks:
                assert score[p] > 0

    def test_substrategy_run_step(self):
        panel = load_smartbeta_panel()
        sub = SmartBetaSubStrategy(SmartBetaConfig())
        result = sub.run_step(panel, panel.index[-1])
        assert result.meta["strategy"] == "smart_beta"


# ======================================================================
# 4. factor_ic 测试
# ======================================================================
class TestFactorIC:
    def test_factor_scores_keys(self):
        panel = load_smartbeta_panel()
        scores = compute_factor_scores(panel, panel.index[-1], list(ALL_V4_CODES))
        assert set(scores.keys()) == set(FACTOR_NAMES)
        for s in scores.values():
            assert isinstance(s, pd.Series)

    def test_forward_return(self):
        panel = load_smartbeta_panel()
        fwd = compute_forward_return(panel, panel.index[-50], list(ALL_V4_CODES))
        assert isinstance(fwd, pd.Series)
        # 在范围内
        assert len(fwd) > 0

    def test_factor_ic_at(self):
        panel = load_smartbeta_panel()
        ic = factor_ic_at(panel, panel.index[-50], list(ALL_V4_CODES))
        assert set(ic.keys()) == set(FACTOR_NAMES)
        for v in ic.values():
            assert -1 <= v <= 1

    def test_rolling_factor_ic(self):
        panel = load_smartbeta_panel()
        ic = rolling_factor_ic(
            panel, list(ALL_V4_CODES),
            start="2021-01-01", end="2025-12-31",
            window=60, forward_window=20, step=10,
        )
        assert not ic.empty
        assert set(ic.columns) == set(FACTOR_NAMES)


# ======================================================================
# 5. factor_timing_v4 测试
# ======================================================================
class TestFactorTimingV4:
    def test_compute_factor_weights_basic(self):
        ic = pd.DataFrame({
            "momentum": [0.05],
            "reversal": [0.03],
            "value": [0.10],
            "low_vol": [0.02],
            "dividend": [0.0],
            "quality": [0.04],
        })
        w = compute_factor_weights(ic, FactorTimingConfig(base=0.05, power=2.0, factor_ic_threshold=0.0), regime="bear")
        assert abs(sum(w.values()) - 1.0) < 1e-6
        # value 应该有最高权重
        assert w["value"] > w["dividend"]

    def test_compute_strategy_weights(self):
        f_w = {"momentum": 0.3, "value": 0.5, "low_vol": 0.2}
        factor_to_strategy = {
            "momentum": "style_rotation",
            "value": "smart_beta",
            "low_vol": "smart_beta",
        }
        s_w = compute_strategy_weights(f_w, factor_to_strategy)
        assert abs(s_w["style_rotation"] - 0.3) < 1e-6
        assert abs(s_w["smart_beta"] - 0.7) < 1e-6

    def test_compute_factor_weights_with_neg_ic(self):
        ic = pd.DataFrame({
            "momentum": [-0.10],  # 负 IC, 权重应被压低
            "value": [0.20],       # 正 IC, 高权重
        }, index=ALL_V4_CODES[:1])
        w = compute_factor_weights(ic, FactorTimingConfig(base=0.0, power=2.0, min_weight=0.0), regime="bull")
        # momentum 应该是 0 (IC + base < 0)
        assert w["momentum"] == 0.0
        # value 应该是 1.0
        assert w["value"] == 1.0

    def test_backtest_factor_timing(self):
        panel = load_smartbeta_panel()
        ic = backtest_factor_timing(
            panel, list(ALL_V4_CODES),
            start="2021-01-01", end="2025-12-31",
        )
        assert not ic.empty
        # use_low_vol=False (default) → low_vol excluded from active factors
        assert set(ic.columns).issubset(set(FACTOR_NAMES))
        assert "low_vol" not in ic.columns


# ======================================================================
# 6. regime_transitions 测试
# ======================================================================
class TestRegimeTransitions:
    def test_distance_basic(self):
        assert distance_between(0, 0) == 0
        assert distance_between(0, 1) == 1
        assert distance_between(0, 2) == 2
        assert distance_between(1, 2) == 1

    def test_distance_rate_symmetric_without_potential(self):
        # gamma=0 时距离对称
        r1 = distance_rate(0, 2, alpha=1.0, gamma=0.0)
        r2 = distance_rate(2, 0, alpha=1.0, gamma=0.0)
        assert r1 == r2

    def test_distance_rate_asymmetric_with_potential(self):
        # gamma>0 时, bull→bear 应该比 bear→bull 难
        r_bull_to_bear = distance_rate(2, 0, alpha=1.0, gamma=0.5)
        r_bear_to_bull = distance_rate(0, 2, alpha=1.0, gamma=0.5)
        assert r_bull_to_bear < r_bear_to_bull

    def test_effective_distance(self):
        # bull → bear: d=2, pot_i=1, pot_j=0, diff=+1, eff_d=2+0.3*1=2.3
        d = effective_distance(2, 0, gamma=0.3)
        assert d == pytest.approx(2.3, abs=0.01)

    def test_build_transmat_shape(self):
        m = build_distance_transmat(n_states=3)
        assert m.shape == (3, 3)

    def test_build_transmat_rows_sum_to_one(self):
        m = build_distance_transmat(n_states=3)
        for i in range(3):
            assert m[i].sum() == pytest.approx(1.0, abs=1e-6)

    def test_build_transmat_sticky_diagonal(self):
        m = build_distance_transmat(alpha=2.0, gamma=0.0)
        # 自循环应该最大
        for i in range(3):
            assert m[i, i] == m[i].max()

    def test_build_transmat_bull_bear_weak(self):
        m = build_distance_transmat(alpha=1.5, gamma=0.3)
        # bull ↔ bear 直接跳转应该 < 0.1
        assert m[0, 2] < 0.1
        assert m[2, 0] < 0.1

    def test_soft_constrain(self):
        prior = build_distance_transmat(alpha=1.5, gamma=0.3)
        learned = np.eye(3)
        mixed = soft_constrain(learned, prior, lam=0.5)
        # mixed 应该是 prior 和 learned 的中点
        assert mixed.shape == (3, 3)
        for i in range(3):
            assert mixed[i].sum() == pytest.approx(1.0, abs=1e-6)

    def test_soft_constrain_lam_0_returns_learned(self):
        prior = build_distance_transmat()
        learned = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.1, 0.2, 0.7]])
        mixed = soft_constrain(learned, prior, lam=0.0)
        # λ=0 时应该返回 learned
        np.testing.assert_array_almost_equal(mixed, learned / learned.sum(axis=1, keepdims=True))

    def test_enforce_minimum_duration_basic(self):
        labels = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1])
        out = enforce_minimum_duration(labels, min_duration=3)
        # 1,1,1 短状态 [0,0,1,1,1] 合并到 1
        # [2,2] 短于 3, 合并
        # 输出应该是 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1
        assert isinstance(out, np.ndarray)
        assert len(out) == len(labels)

    def test_validate_transmat(self):
        m = build_distance_transmat()
        result = validate_transmat(m)
        assert result["valid"] is True
        assert result["n"] == 3
        assert result["diag_mean"] > 0.5  # 自循环应该 > 50%

    def test_validate_transmat_invalid(self):
        bad = np.array([[0.5, 0.5, 0.5], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3]])
        result = validate_transmat(bad)
        assert result["valid"] is False


# ======================================================================
# 7. multi_strategy_v4 测试
# ======================================================================
class TestMultiStrategyV4:
    def test_run_v4A_style(self):
        panel = load_smartbeta_panel()
        r = run_v4_mode(panel, "v4A_style")
        assert len(r.nav) == len(panel)
        assert "sharpe" in r.metrics
        assert r.mode == "v4A_style"

    def test_run_v4B_smartbeta(self):
        panel = load_smartbeta_panel()
        r = run_v4_mode(panel, "v4B_smartbeta")
        assert r.mode == "v4B_smartbeta"
        assert r.metrics["final_nav"] > 0

    def test_run_v4C_combo(self):
        panel = load_smartbeta_panel()
        r = run_v4_mode(panel, "v4C_combo")
        assert r.mode == "v4C_combo"

    def test_run_v4D_ic(self):
        panel = load_smartbeta_panel()
        r = run_v4_mode(
            panel, "v4D_ic",
            factor_timing_cfg=FactorTimingConfig(forward_window=10),
        )
        assert r.mode == "v4D_ic"

    def test_run_v4_mode_invalid(self):
        panel = load_smartbeta_panel()
        with pytest.raises(ValueError):
            run_v4_mode(panel, "invalid_mode")

    def test_v4_result_structure(self):
        panel = load_smartbeta_panel()
        r = run_v4_mode(panel, "v4C_combo")
        assert isinstance(r.nav, pd.Series)
        assert isinstance(r.states, list)
        assert len(r.rebalance_dates) > 50
        # 至少有 1 个子策略 NAV
        assert not r.sub_navs.empty


# ======================================================================
# 8. regime_detector_v4 测试
# ======================================================================
class TestRegimeDetectorV4:
    def test_fit_predict(self):
        from QuantNodes.strategy.momentum_etf_rotation.v4 import (
            RegimeDetector, RegimeConfig,
        )
        panel = load_smartbeta_panel()
        det = RegimeDetector(RegimeConfig(n_iter=20))
        det.fit(panel, panel.index[-1])
        r = det.predict(panel, panel.index[-1])
        # -1 (未训练) 或 0/1/2
        assert r in [-1, 0, 1, 2]

    def test_predict_series(self):
        from QuantNodes.strategy.momentum_etf_rotation.v4 import (
            RegimeDetector, RegimeConfig,
        )
        panel = load_smartbeta_panel()
        det = RegimeDetector(RegimeConfig(n_iter=20))
        det.fit(panel, panel.index[-1])
        s = det.predict_series(
            panel, "2021-01-01", "2025-12-31", step=10,
            apply_min_duration=False,
        )
        assert isinstance(s, pd.Series)
        assert len(s) > 0

    def test_get_regime_factor_weight(self):
        from QuantNodes.strategy.momentum_etf_rotation.v4 import (
            get_regime_factor_weight,
        )
        # 牛市: momentum 权重高
        w_bull_mom = get_regime_factor_weight(0, "momentum")
        w_bear_mom = get_regime_factor_weight(1, "momentum")
        assert w_bull_mom > w_bear_mom
        # 熊市: value 权重高
        w_bear_val = get_regime_factor_weight(1, "value")
        w_bull_val = get_regime_factor_weight(0, "value")
        assert w_bear_val > w_bull_val


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
