"""生产运行层单测 (§生产化)."""

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.allweather.production import (
    CARRY_WEIGHT, D_PARAMS, D_WEIGHT, ProductionResult, latest_target,
    summarize)


def test_weights_sum_to_one():
    assert CARRY_WEIGHT + D_WEIGHT == pytest.approx(1.0)


def test_d_params_match_candidate_v3():
    assert D_PARAMS["rebalance_freq"] == "ME"
    assert D_PARAMS["neg_trend_preset"] == "aggressive"
    assert D_PARAMS["resonance_k"] == 0.9
    assert D_PARAMS["resonance_dd_floor"] == -0.05
    assert D_PARAMS["correlation_threshold"] == 0.60


def test_summarize_monotonic_nav():
    idx = pd.bdate_range("2020-01-01", periods=500)
    nav = pd.Series(np.linspace(1.0, 1.5, 500), index=idx)
    m = summarize(nav)
    assert m["start"] == idx[0].date() and m["end"] == idx[-1].date()
    assert m["CAGR"] > 0 and m["Sharpe"] > 0
    assert m["MaxDD"] == pytest.approx(0.0, abs=1e-9)
    assert m["CurrDD"] == pytest.approx(0.0, abs=1e-9)


def test_summarize_reports_drawdown():
    idx = pd.bdate_range("2020-01-01", periods=100)
    nav = pd.Series(np.concatenate([np.linspace(1, 1.2, 50),
                                    np.linspace(1.2, 0.9, 50)]), index=idx)
    m = summarize(nav)
    assert m["MaxDD"] < 0
    assert m["CurrDD"] == pytest.approx(0.9 / 1.2 - 1, abs=1e-6)


def test_latest_target_filters_and_sorts():
    idx = pd.bdate_range("2020-01-01", periods=5)
    weights = pd.DataFrame(
        {"a": [0, 0, 0, 0, 0.1], "b": [0, 0, 0, 0, -0.3],
         "c": [0, 0, 0, 0, 0.0], "d": [0, 0, 0, 0, 0.2]}, index=idx)
    res = ProductionResult(nav=pd.Series(1.0, index=idx), weights=weights,
                           d_nav=pd.Series(1.0, index=idx),
                           carry_ret=pd.Series(0.0, index=idx),
                           d_weights=weights, carry_weights=weights)
    tgt = latest_target(res)
    assert list(tgt.index) == ["b", "d", "a"]    # |w| 降序
    assert "c" not in tgt.index                  # 零权重过滤
