# coding=utf-8
"""参数标定与自适应选择测试."""
from __future__ import annotations


from QuantNodes.strategy.cta.optimization import (
    analyze_stability,
    combo_count,
    evaluate_strategy,
    sample_grid,
    select_params,
    set_calibrated,
)


class TestParamGrid:
    def test_combo_count(self):
        assert combo_count("all") > 100
        assert combo_count("trend") > 10
        assert combo_count("range") > 10

    def test_sample_grid(self):
        combos = sample_grid("trend", n_samples=5)
        assert len(combos) == 5
        for c in combos:
            assert "confirm_window" in c
            assert "ema_slow" in c

    def test_sample_grid_all(self):
        combos = sample_grid("all")
        assert len(combos) == combo_count("all")


class TestEvaluator:
    def test_feasible(self):
        r = evaluate_strategy(calmar=0.5, sharpe=0.3, max_dd=-0.15,
                              signal_coverage=0.15, n_trades=80)
        assert r.feasible
        assert r.calmar == 0.5

    def test_infeasible_maxdd(self):
        r = evaluate_strategy(calmar=0.5, max_dd=-0.35, n_trades=80)
        assert not r.feasible
        assert "MaxDD" in r.reject_reason

    def test_infeasible_coverage(self):
        r = evaluate_strategy(calmar=0.5, max_dd=-0.15,
                              signal_coverage=0.03, n_trades=80)
        assert not r.feasible
        assert "Coverage" in r.reject_reason

    def test_infeasible_trades(self):
        r = evaluate_strategy(calmar=0.5, max_dd=-0.15,
                              signal_coverage=0.15, n_trades=5)
        assert not r.feasible
        assert "Trades" in r.reject_reason


class TestStability:
    def test_stable_params(self):
        from QuantNodes.strategy.cta.optimization.walk_forward import WindowResult
        windows = [
            WindowResult(0, "2020", "2021", "2022", "2022",
                         {"confirm_window": 10, "ema_slow": 60}, 0.1, 0.2, 0.1, -0.1, 0.15, 80),
            WindowResult(1, "2021", "2022", "2023", "2023",
                         {"confirm_window": 10, "ema_slow": 120}, 0.3, 0.4, 0.2, -0.15, 0.12, 90),
        ]
        result = analyze_stability(windows, {"confirm_window": [10, 20], "ema_slow": [60, 120]})
        # confirm_window=10 出现在2/2窗口 → stable
        assert result["confirm_window"]["stable"]
        # ema_slow: 60出现1次, 120出现1次 → 不稳定
        assert not result["ema_slow"]["stable"]


class TestAdaptive:
    def test_set_and_select(self):
        set_calibrated("TEST", {
            "trend_up": {"ema_slow": 30, "confirm_window": 5},
            "global": {"target_vol": 0.2},
        })
        params = select_params("TEST", "trend_up")
        assert params["ema_slow"] == 30
        assert params["confirm_window"] == 5

    def test_select_fallback_to_global(self):
        set_calibrated("TEST2", {
            "global": {"target_vol": 0.15},
        })
        params = select_params("TEST2", "unknown_regime")
        assert params["target_vol"] == 0.15

    def test_select_unknown_product(self):
        params = select_params("NONEXIST", "trend_up")
        assert params == {}
