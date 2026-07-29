"""Unit tests for ca_gcp_standalone.py — single-file CA-GCP module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_V102 = Path(__file__).resolve().parent.parent
if str(_V102) not in sys.path:
    sys.path.insert(0, str(_V102))

from ca_gcp_standalone import (  # noqa: E402
    CAGCPConfig,
    CAGCPipeline,
    NeighborQuality,
    PrecomputedWeightedQuantile,
    RiskFilterRules,
    SectorCAGCPResult,
    TheoreticalBound,
    _LOSS_REGISTRY,
    apply_modulator,
    apply_scale_to_weights,
    build_knn_graph,
    build_sector_groups,
    build_v10_2_pipeline,
    ca_gcp_risk_filter,
    compare_bound_to_empirical,
    compute_coverage_metrics,
    compute_neighbor_quality,
    compute_systemic_stress,
    detect_warnings,
    estimate_volatility,
    evaluate_alert,
    experimental_rules,
    extract_risk_signals,
    fit_sector_ca_gcp,
    fit_sector_hybrid_ca_gcp,
    load_sector_map,
    predict_sector_ca_gcp,
    quality_dataframe,
    resolve_loss_fn,
    theoretical_coverage_bound,
    total_variation_distance_ecdf,
    width_bps,
    width_stability,
    width_timeseries,
    width_volatility_correlation,
    weighted_quantile,
)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    codes = [f"A{i}" for i in range(8)]
    return pd.DataFrame(rng.normal(0, 0.01, (500, 8)), index=dates, columns=codes)


@pytest.fixture
def sample_split(sample_returns):
    return sample_returns.iloc[:200], sample_returns.iloc[200:350], sample_returns.iloc[350:]


@pytest.fixture
def fitted_pipeline(sample_split):
    train, calib, test = sample_split
    pipe = CAGCPipeline(CAGCPConfig(k=4, sensitivity_eta=0.5, recency_tau=20))
    pipe.fit(train)
    return pipe


# --- Core: build_knn_graph ---


class TestBuildKnnGraph:
    def test_returns_tuple_of_three(self, sample_returns):
        A, nbrs, codes = build_knn_graph(sample_returns.iloc[:200], k=4)
        assert isinstance(A, np.ndarray)
        assert isinstance(nbrs, dict)
        assert isinstance(codes, list)

    def test_adjacency_shape(self, sample_returns):
        A, _, codes = build_knn_graph(sample_returns.iloc[:200], k=4)
        assert A.shape == (len(codes), len(codes))

    def test_adjacency_symmetric(self, sample_returns):
        A, _, _ = build_knn_graph(sample_returns.iloc[:200], k=4)
        assert np.allclose(A, A.T)

    def test_diagonal_positive(self, sample_returns):
        A, _, _ = build_knn_graph(sample_returns.iloc[:200], k=4)
        assert (np.diag(A) > 0).all()

    def test_neighbors_include_self(self, sample_returns):
        _, nbrs, _ = build_knn_graph(sample_returns.iloc[:200], k=4)
        for i, nbr_list in nbrs.items():
            assert i in nbr_list

    def test_random_method(self, sample_returns):
        A, _, _ = build_knn_graph(sample_returns.iloc[:200], k=3, method="random")
        assert A.shape == (8, 8)

    def test_sector_method(self, sample_returns):
        sectors = {f"A{i}": "s1" if i < 4 else "s2" for i in range(8)}
        A, _, _ = build_knn_graph(sample_returns.iloc[:200], k=4, method="sector", sectors=sectors)
        assert A.shape == (8, 8)


# --- Core: volatility ---


class TestVolatility:
    def test_shape_matches_input(self, sample_returns):
        sigma = estimate_volatility(sample_returns.iloc[:200])
        assert sigma.shape == sample_returns.iloc[:200].shape

    def test_positive(self, sample_returns):
        sigma = estimate_volatility(sample_returns.iloc[:200])
        assert (sigma > 0).all().all()


# --- Core: stress modulator ---


class TestStressModulator:
    def test_stress_in_unit_interval(self, sample_returns):
        sigma = estimate_volatility(sample_returns.iloc[:200:10])
        stress = compute_systemic_stress(sample_returns.iloc[:200:10], sigma)
        assert (stress >= 0).all()
        assert (stress <= 1).all()

    def test_modulator_scales_width(self, sample_returns):
        hw = pd.DataFrame(0.01, index=sample_returns.index[:10], columns=sample_returns.columns)
        stress = pd.Series(0.5, index=sample_returns.index[:10])
        result = apply_modulator(hw, stress, eta=1.0)
        assert (result > hw).all().all()


# --- Core: weighted quantile ---


class TestWeightedQuantile:
    def test_uniform_weights(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = weighted_quantile(scores, np.ones(5), level=0.8, pseudo_count_inf=False)
        assert 3.0 <= q <= 5.0

    def test_empty_returns_inf(self):
        q = weighted_quantile(np.array([]), np.array([]), level=0.95, pseudo_count_inf=True)
        assert np.isinf(q)

    def test_precomputed_matches_slow(self):
        rng = np.random.default_rng(42)
        scores = rng.normal(0, 1, 100)
        weights = rng.uniform(0.5, 1.5, 100)
        slow = weighted_quantile(scores, weights, level=0.95, pseudo_count_inf=False)
        fast = PrecomputedWeightedQuantile(scores, weights, pseudo_count_inf=False)
        assert abs(fast.query(0.95) - slow) < 1e-9


# --- Pipeline ---


class TestPipeline:
    def test_fit_returns_self(self, sample_split):
        train, _, _ = sample_split
        pipe = CAGCPipeline(CAGCPConfig(k=4))
        result = pipe.fit(train)
        assert result is pipe

    def test_fit_populates_attrs(self, fitted_pipeline):
        assert len(fitted_pipeline.codes) == 8
        assert len(fitted_pipeline.neighbors) == 8
        assert fitted_pipeline.A_norm is not None
        assert fitted_pipeline.corr_matrix is not None

    def test_predict_fast_shape(self, sample_split, fitted_pipeline):
        _, calib, test = sample_split
        out = fitted_pipeline.predict_fast(calib, test)
        assert out["lower"].shape == test.shape
        assert out["upper"].shape == test.shape
        assert out["half_width"].shape == test.shape
        assert len(out["stress"]) == len(test)

    def test_predict_fast_no_inf(self, sample_split, fitted_pipeline):
        _, calib, test = sample_split
        out = fitted_pipeline.predict_fast(calib, test)
        assert not np.isinf(out["half_width"].values).any()

    def test_predict_matches_predict_fast(self, sample_split):
        train, calib, test = sample_split
        pipe = CAGCPipeline(CAGCPConfig(k=4, sensitivity_eta=0.5, recency_tau=20))
        pipe.fit(train.iloc[:50])
        slow = pipe.predict(calib.iloc[:20], test.iloc[:20])
        fast = pipe.predict_fast(calib.iloc[:20], test.iloc[:20])
        assert np.allclose(slow["half_width"].values, fast["half_width"].values, atol=1e-6)

    def test_empty_test_returns_empty(self, fitted_pipeline):
        _, calib, _ = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        empty_calib = pd.DataFrame(np.random.randn(10, 8), columns=[f"A{i}" for i in range(8)])
        empty_test = pd.DataFrame(columns=[f"A{i}" for i in range(8)])
        out = fitted_pipeline.predict_fast(empty_calib, empty_test)
        assert out["half_width"].shape[1] == 8


# --- Validators ---


class TestValidators:
    def test_coverage_metrics(self, sample_split, fitted_pipeline):
        _, calib, test = sample_split
        out = fitted_pipeline.predict_fast(calib, test)
        m = compute_coverage_metrics(test, out["lower"], out["upper"])
        assert 0 <= m["marginal"] <= 1
        assert m["pa_std"] >= 0

    def test_width_bps(self, sample_split, fitted_pipeline):
        _, calib, test = sample_split
        out = fitted_pipeline.predict_fast(calib, test)
        w = width_bps(out["half_width"])
        assert w > 0

    def test_width_timeseries(self, sample_split, fitted_pipeline):
        _, calib, test = sample_split
        out = fitted_pipeline.predict_fast(calib, test)
        wts = width_timeseries(out["half_width"])
        assert len(wts) == len(test)
        assert (wts > 0).all()

    def test_width_stability(self, sample_split, fitted_pipeline):
        _, calib, test = sample_split
        out = fitted_pipeline.predict_fast(calib, test)
        s = width_stability(out["half_width"])
        assert s >= 0

    def test_detect_warnings(self, sample_split, fitted_pipeline):
        _, calib, test = sample_split
        out = fitted_pipeline.predict_fast(calib, test)
        w = detect_warnings(out["stress"], out["half_width"])
        assert "fired" in w.columns
        assert set(w["fired"].unique()).issubset({0, 1})


# --- Neighbor quality ---


class TestNeighborQuality:
    def test_quality_dataframe(self, fitted_pipeline):
        qdf = quality_dataframe(fitted_pipeline)
        assert len(qdf) == 8
        assert "weighted_corr_sum" in qdf.columns

    def test_compute_neighbor_quality(self, fitted_pipeline):
        nq = compute_neighbor_quality(fitted_pipeline, 0)
        assert isinstance(nq, NeighborQuality)
        assert nq.borrow_recommendation in ("strong", "moderate", "weak")


# --- Theoretical bound ---


class TestTheoreticalBound:
    def test_tv_distance_identical(self):
        scores = np.array([1.0, 2.0, 3.0])
        assert total_variation_distance_ecdf(scores, scores) == pytest.approx(0.0)

    def test_tv_distance_disjoint(self):
        p = np.array([1.0, 2.0])
        q = np.array([10.0, 20.0])
        assert total_variation_distance_ecdf(p, q) == pytest.approx(1.0)

    def test_tv_distance_empty(self):
        assert total_variation_distance_ecdf(np.array([]), np.array([1.0])) == 1.0

    def test_theoretical_bound(self, sample_split, fitted_pipeline):
        _, calib, test = sample_split
        scores = (calib - 0.0).abs() / estimate_volatility(calib)
        bound = theoretical_coverage_bound(fitted_pipeline, scores)
        assert len(bound) == 8
        assert "bound" in bound.columns

    def test_compare_bound(self):
        bounds = pd.DataFrame({"bound": [0.1, 0.2]}, index=["A", "B"])
        gaps = pd.Series([0.05, 0.15], index=["A", "B"])
        result = compare_bound_to_empirical(bounds, gaps)
        assert "ratio" in result.columns
        assert "bound_satisfied" in result.columns


# --- Sector clustering ---


class TestSectorClustering:
    def test_build_sector_groups(self):
        codes = ["A", "B", "C", "D", "E"]
        sector_map = {"A": "s1", "B": "s1", "C": "s1", "D": "s2", "E": "s2"}
        groups = build_sector_groups(codes, sector_map, min_size=2)
        assert "s1" in groups
        assert "s2" in groups
        assert len(groups["s1"]) == 3

    def test_fit_sector_ca_gcp(self, sample_returns):
        sectors = {"s1": [f"A{i}" for i in range(4)], "s2": [f"A{i}" for i in range(4, 8)]}
        pipes = fit_sector_ca_gcp(sample_returns.iloc[:200], sectors, CAGCPConfig(k=2))
        assert len(pipes) == 2
        assert "s1" in pipes
        assert "s2" in pipes

    def test_predict_sector_ca_gcp(self, sample_split, sample_returns):
        train, calib, test = sample_split
        sectors = {"s1": [f"A{i}" for i in range(4)], "s2": [f"A{i}" for i in range(4, 8)]}
        pipes = fit_sector_ca_gcp(train, sectors, CAGCPConfig(k=2))
        result = predict_sector_ca_gcp(pipes, calib, test)
        assert isinstance(result, SectorCAGCPResult)
        assert result.lower.shape[1] > 0


# --- Risk filter ---


class TestRiskFilter:
    def test_global_green(self, sample_split, fitted_pipeline):
        _, calib, test = sample_split
        out = fitted_pipeline.predict_fast(calib, test)
        weights = pd.Series(0.25, index=fitted_pipeline.codes)
        adj, diag = ca_gcp_risk_filter(weights, out)
        assert diag["alert_level"] == "green"
        assert diag["applied_scale"] == 1.0

    def test_global_red_triggers(self):
        weights = pd.Series(0.5, index=["A", "B"])
        hw = pd.DataFrame(0.01, index=pd.date_range("2020-01-01", periods=5), columns=["A", "B"])
        stress = pd.Series([0.99, 0.99, 0.99, 0.99, 0.99])
        intervals = {"half_width": hw, "stress": stress}
        rules = RiskFilterRules(stress_red=0.5)
        adj, diag = ca_gcp_risk_filter(weights, intervals, rules)
        assert diag["alert_level"] == "red"
        assert adj.sum() == pytest.approx(1.0, abs=0.01)

    def test_grouped_mode(self):
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        hw = pd.DataFrame(0.01, index=pd.date_range("2020-01-01", periods=5), columns=["A", "B", "C"])
        stress = pd.Series([0.99, 0.99, 0.99, 0.99, 0.99])
        intervals = {"half_width": hw, "stress": stress}
        rules = RiskFilterRules(
            stress_red=0.5,
            asset_groups={"g1": ["A", "B"], "g2": ["C"]},
            group_rules={"g1": RiskFilterRules(stress_red=0.3)},
        )
        adj, diag = ca_gcp_risk_filter(weights, intervals, rules)
        assert "group_alerts" in diag

    def test_experimental_rules(self):
        r = experimental_rules()
        assert r.stress_yellow == 0.6
        assert r.stress_red == 0.85

    def test_hysteresis_fields_exist(self):
        r = RiskFilterRules()
        assert r.stress_yellow_recovery < r.stress_yellow
        assert r.width_z_yellow_recovery < r.width_z_yellow

    def test_extract_risk_signals_keys(self):
        hw = pd.DataFrame(0.01, index=pd.date_range("2020-01-01", periods=5), columns=["A", "B"])
        stress = pd.Series([0.3, 0.4, 0.5, 0.6, 0.7])
        intervals = {"half_width": hw, "stress": stress}
        sig = extract_risk_signals(intervals)
        assert "width_z_today" in sig
        assert "stress_today" in sig
        assert isinstance(sig["stress_today"], float)

    def test_evaluate_alert_levels(self):
        rules = RiskFilterRules()
        assert evaluate_alert(0.0, 0.5, rules) == ("green", 1.0)
        assert evaluate_alert(2.0, 0.5, rules) == ("green", 1.0)
        assert evaluate_alert(3.5, 0.5, rules) == ("yellow", 0.85)
        assert evaluate_alert(5.0, 0.5, rules) == ("red", 0.6)
        assert evaluate_alert(0.0, 0.95, rules) == ("yellow", 0.85)
        assert evaluate_alert(0.0, 0.99, rules) == ("red", 0.6)

    def test_apply_scale_to_weights_sum(self):
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        adj = apply_scale_to_weights(weights, 1.0)
        assert adj.sum() == pytest.approx(1.0)
        adj_red = apply_scale_to_weights(weights, 0.6)
        assert adj_red.sum() == pytest.approx(1.0)

    def test_apply_scale_to_weights_residual(self):
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        adj = apply_scale_to_weights(weights, 0.6)
        assert adj["C"] == pytest.approx(0.52)
        assert adj["A"] == pytest.approx(0.30)
        assert adj["B"] == pytest.approx(0.18)


# --- Integration: build_v10_2_pipeline ---


class TestBuildPipeline:
    def test_build_v10_2_pipeline(self, sample_returns):
        pipe = build_v10_2_pipeline(sample_returns.iloc[:200], CAGCPConfig(k=4))
        assert len(pipe.codes) == 8


# --- Data splitting ---


class TestSplitData:
    def test_split_by_ratio(self, sample_returns):
        cfg = CAGCPConfig(k=4, train_ratio=0.6, calib_ratio=0.25)
        pipe = CAGCPipeline(cfg)
        train, calib, test = pipe._split_data(sample_returns)
        assert len(train) == 300
        assert len(calib) == 125
        assert len(test) == 75
        assert len(train) + len(calib) + len(test) == 500

    def test_split_by_date(self, sample_returns):
        cfg = CAGCPConfig(k=4, train_end="2020-09-01", calib_end="2020-12-01")
        pipe = CAGCPipeline(cfg)
        train, calib, test = pipe._split_data(sample_returns)
        assert train.index[-1] < pd.Timestamp("2020-09-01")
        assert calib.index[0] >= pd.Timestamp("2020-09-01")
        assert calib.index[-1] <= pd.Timestamp("2020-12-01")
        assert test.index[0] > pd.Timestamp("2020-12-01")
        assert len(test) > 0

    def test_fit_stores_split(self, sample_returns):
        cfg = CAGCPConfig(k=4, train_ratio=0.6, calib_ratio=0.25)
        pipe = CAGCPipeline(cfg)
        pipe.fit(sample_returns)
        assert pipe._train is not None
        assert pipe._calib is not None
        assert pipe._test is not None
        assert len(pipe._train) + len(pipe._calib) + len(pipe._test) == 500


# --- target_codes filtering ---


class TestTargetCodes:
    def test_target_codes_filters_output(self, sample_returns):
        cfg = CAGCPConfig(k=4, target_codes=["A0", "A1", "A2"])
        pipe = CAGCPipeline(cfg)
        pipe.fit(sample_returns)
        intervals = pipe.predict_fast(pipe._calib, pipe._test)
        assert list(intervals["lower"].columns) == ["A0", "A1", "A2"]
        assert list(intervals["upper"].columns) == ["A0", "A1", "A2"]
        assert list(intervals["half_width"].columns) == ["A0", "A1", "A2"]
        assert list(intervals["thresholds"].columns) == ["A0", "A1", "A2"]

    def test_target_codes_none_returns_all(self, sample_returns):
        cfg = CAGCPConfig(k=4, target_codes=None)
        pipe = CAGCPipeline(cfg)
        pipe.fit(sample_returns)
        intervals = pipe.predict_fast(pipe._calib, pipe._test)
        assert list(intervals["lower"].columns) == [f"A{i}" for i in range(8)]

    def test_stress_unfiltered_by_target_codes(self, sample_returns):
        cfg = CAGCPConfig(k=4, target_codes=["A0", "A1"])
        pipe = CAGCPipeline(cfg)
        pipe.fit(sample_returns)
        intervals = pipe.predict_fast(pipe._calib, pipe._test)
        assert len(intervals["stress"]) == len(pipe._test)


# --- sectors input ---


class TestSectorsInput:
    def test_sectors_passed_to_graph(self, sample_returns):
        sectors = {f"A{i}": "sector1" if i < 4 else "sector2" for i in range(8)}
        cfg = CAGCPConfig(k=3, graph_method="sector", sectors=sectors)
        pipe = CAGCPipeline(cfg)
        pipe.fit(sample_returns)
        for i in range(4):
            nbrs = pipe.neighbors[i]
            nbr_codes = [pipe.codes[j] for j in nbrs]
            for nc in nbr_codes:
                assert sectors[nc] == "sector1"
        for i in range(4, 8):
            nbrs = pipe.neighbors[i]
            nbr_codes = [pipe.codes[j] for j in nbrs]
            for nc in nbr_codes:
                assert sectors[nc] == "sector2"

    def test_sectors_none_with_correlation_method(self, sample_returns):
        cfg = CAGCPConfig(k=4, graph_method="correlation", sectors=None)
        pipe = CAGCPipeline(cfg)
        pipe.fit(sample_returns)
        assert len(pipe.neighbors[0]) > 1


# --- Loss function ---


class TestLossFn:
    def test_loss_pareto_default(self):
        cfg = CAGCPConfig()
        assert cfg.loss_fn == "pareto"
        loss = resolve_loss_fn(cfg.loss_fn)
        m = {"extreme": 0.95, "pa_std": 0.05, "marginal": 0.96}
        assert loss(m, 100.0) == pytest.approx(10.0 * 0.95 - 5.0 * 0.05 - 0.1)

    def test_loss_pareto_nan_extreme(self):
        loss = resolve_loss_fn("pareto")
        assert loss({"extreme": float("nan")}, 100.0) == -1e9

    def test_loss_coverage(self):
        loss = resolve_loss_fn("coverage")
        m = {"marginal": 0.96, "extreme": 0.92}
        assert loss(m, 500.0) == pytest.approx(1.88)

    def test_loss_sharpness(self):
        loss = resolve_loss_fn("sharpness")
        assert loss({}, 200.0) == -200.0

    def test_loss_callable(self):
        def custom(m, w):
            return -m["pa_std"]
        loss = resolve_loss_fn(custom)
        assert loss({"pa_std": 0.1}, 100.0) == -0.1

    def test_loss_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_fn"):
            resolve_loss_fn("unknown_loss")

    def test_registry_keys(self):
        assert set(_LOSS_REGISTRY.keys()) == {"pareto", "coverage", "sharpness"}
