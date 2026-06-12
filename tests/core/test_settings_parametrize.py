"""QualityGateSetting + EvolutionSetting + OperatorSetting 全参数 parametrize (~30 tests)。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from QuantNodes.core.evolution.settings import (
    EvolutionSetting,
    OperatorSetting,
)
from QuantNodes.core.quality_gate.settings import (
    ComplexitySetting,
    ConsistencySetting,
    QualityGateSetting,
    RedundancySetting,
)


# ============================================================================
# 1. OperatorSetting (5 tests)
# ============================================================================

class TestOperatorSettingParams:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_enabled(self, enabled):
        s = OperatorSetting(enabled=enabled)
        assert s.enabled is enabled

    @pytest.mark.parametrize("model", ["mock", "deepseek-v3", "gpt-4o", "claude-3-5-sonnet", "qwen", "custom"])
    def test_model(self, model):
        s = OperatorSetting(model=model)
        assert s.model == model

    @pytest.mark.parametrize("attempts", [0, 1, 3, 5, 10])
    def test_max_correction_attempts(self, attempts):
        s = OperatorSetting(max_correction_attempts=attempts)
        assert s.max_correction_attempts == attempts

    @pytest.mark.parametrize("seed", [0, 42, -1, 2**31 - 1, -2**31])
    def test_seed(self, seed):
        s = OperatorSetting(seed=seed)
        assert s.seed == seed

    @pytest.mark.parametrize("bad_value", ["not_int", None, 1.5])
    def test_invalid_max_correction_attempts(self, bad_value):
        """非 int 必抛错。"""
        with pytest.raises(ValidationError):
            OperatorSetting(max_correction_attempts=bad_value)


# ============================================================================
# 2. ComplexitySetting (5 tests)
# ============================================================================

class TestComplexitySetting:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_enabled(self, enabled):
        s = ComplexitySetting(enabled=enabled)
        assert s.enabled is enabled

    @pytest.mark.parametrize("threshold", [0, 50, 200, 1000, 10000])
    def test_symbol_length_threshold(self, threshold):
        s = ComplexitySetting(symbol_length_threshold=threshold)
        assert s.symbol_length_threshold == threshold

    @pytest.mark.parametrize("threshold", [0, 5, 10, 50])
    def test_base_features_threshold(self, threshold):
        s = ComplexitySetting(base_features_threshold=threshold)
        assert s.base_features_threshold == threshold

    @pytest.mark.parametrize("ratio", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_free_args_ratio_threshold(self, ratio):
        s = ComplexitySetting(free_args_ratio_threshold=ratio)
        assert s.free_args_ratio_threshold == ratio

    def test_defaults(self):
        s = ComplexitySetting()
        assert s.enabled is True
        assert s.symbol_length_threshold == 200
        assert s.base_features_threshold == 5
        assert s.free_args_ratio_threshold == 0.5


# ============================================================================
# 3. RedundancySetting (4 tests)
# ============================================================================

class TestRedundancySetting:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_enabled(self, enabled):
        s = RedundancySetting(enabled=enabled)
        assert s.enabled is enabled

    @pytest.mark.parametrize("threshold", [0, 1, 5, 10, 100])
    def test_threshold(self, threshold):
        s = RedundancySetting(threshold=threshold)
        assert s.threshold == threshold

    @pytest.mark.parametrize("path", [None, "/tmp/zoo", "./local_zoo"])
    def test_zoo_path(self, path):
        s = RedundancySetting(zoo_path=path)
        assert s.zoo_path == path


# ============================================================================
# 4. ConsistencySetting (3 tests)
# ============================================================================

class TestConsistencySetting:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_enabled(self, enabled):
        s = ConsistencySetting(enabled=enabled)
        assert s.enabled is enabled

    @pytest.mark.parametrize("model", ["mock", "deepseek-v3", "gpt-4o"])
    def test_model(self, model):
        s = ConsistencySetting(model=model)
        assert s.model == model

    @pytest.mark.parametrize("attempts", [0, 1, 5, 10])
    def test_max_correction_attempts(self, attempts):
        s = ConsistencySetting(max_correction_attempts=attempts)
        assert s.max_correction_attempts == attempts


# ============================================================================
# 5. QualityGateSetting.any_enabled (3 tests)
# ============================================================================

class TestQualityGateSettingAnyEnabled:
    @pytest.mark.parametrize("complexity,redundancy,consistency,expected", [
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, True),
        (True, True, True, True),
        (False, False, False, False),
    ])
    def test_any_enabled_combinations(self, complexity, redundancy, consistency, expected):
        s = QualityGateSetting(
            complexity=ComplexitySetting(enabled=complexity),
            redundancy=RedundancySetting(enabled=redundancy),
            consistency=ConsistencySetting(enabled=consistency),
        )
        assert s.any_enabled() is expected


# ============================================================================
# 6. EvolutionSetting 8 参数 (8 tests)
# ============================================================================

class TestEvolutionSettingParams:
    @pytest.mark.parametrize("enabled", [True, False])
    def test_enabled(self, enabled):
        s = EvolutionSetting(enabled=enabled)
        assert s.enabled is enabled

    @pytest.mark.parametrize("max_rounds", [0, 1, 3, 5, 10, 100])
    def test_max_rounds(self, max_rounds):
        s = EvolutionSetting(max_rounds=max_rounds)
        assert s.max_rounds == max_rounds

    @pytest.mark.parametrize("strategy", [
        "best", "random", "weighted", "weighted_inverse", "top_percent_plus_random",
    ])
    def test_parent_selection_strategy(self, strategy):
        s = EvolutionSetting(parent_selection_strategy=strategy)
        assert s.parent_selection_strategy == strategy

    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.3, 0.5, 1.0])
    def test_top_percent_threshold(self, threshold):
        s = EvolutionSetting(top_percent_threshold=threshold)
        assert s.top_percent_threshold == threshold

    @pytest.mark.parametrize("metric", ["sharpe", "ic_mean", "arr", "calmar"])
    def test_metric(self, metric):
        s = EvolutionSetting(metric=metric)
        assert s.metric == metric

    @pytest.mark.parametrize("patience", [0, 1, 3, 5, 10])
    def test_early_stop_patience(self, patience):
        s = EvolutionSetting(early_stop_patience=patience)
        assert s.early_stop_patience == patience

    @pytest.mark.parametrize("top_n", [0, 1, 3, 5, 10, 50, 100])
    def test_top_n(self, top_n):
        s = EvolutionSetting(top_n=top_n)
        assert s.top_n == top_n

    @pytest.mark.parametrize("parents", [0, 1, 2, 3, 5, 10])
    def test_parents_per_round(self, parents):
        s = EvolutionSetting(parents_per_round=parents)
        assert s.parents_per_round == parents

    @pytest.mark.parametrize("h,m,c,any_enabled", [
        (True, True, True, True),
        (False, False, False, False),
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, True),
    ])
    def test_any_operator_enabled(self, h, m, c, any_enabled):
        s = EvolutionSetting(
            hypothesizer=OperatorSetting(enabled=h),
            mutator=OperatorSetting(enabled=m),
            crosser=OperatorSetting(enabled=c),
        )
        assert s.any_operator_enabled() is any_enabled

    def test_defaults(self):
        s = EvolutionSetting()
        assert s.enabled is False
        assert s.max_rounds == 3
        assert s.parents_per_round == 1
        assert s.parent_selection_strategy == "top_percent_plus_random"
        assert s.metric == "sharpe"
        assert s.pool_dir is None
        assert s.early_stop_patience == 0
