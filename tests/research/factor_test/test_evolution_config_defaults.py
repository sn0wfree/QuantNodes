# coding: utf-8
"""Unit tests for EvolutionConfig defaults and validation."""

import pytest
from pydantic import ValidationError

from QuantNodes.research.factor_test.config import EvolutionConfig


def test_evolution_config_defaults():
    cfg = EvolutionConfig()
    assert cfg.enabled is False
    assert cfg.max_rounds == 3
    assert cfg.parents_per_round == 1
    assert cfg.parent_selection_strategy == "top_percent_plus_random"
    assert cfg.top_percent_threshold == 0.3
    assert cfg.metric == "sharpe"
    assert cfg.pool_dir is None
    assert cfg.early_stop_patience == 0


def test_evolution_config_custom_values():
    cfg = EvolutionConfig(
        enabled=True,
        max_rounds=5,
        parents_per_round=2,
        parent_selection_strategy="best",
        top_percent_threshold=0.1,
        metric="ic",
        pool_dir="/tmp/pool",
        early_stop_patience=2,
    )
    assert cfg.enabled is True
    assert cfg.max_rounds == 5
    assert cfg.parents_per_round == 2
    assert cfg.parent_selection_strategy == "best"
    assert cfg.top_percent_threshold == 0.1
    assert cfg.metric == "ic"
    assert cfg.pool_dir == "/tmp/pool"
    assert cfg.early_stop_patience == 2


def test_evolution_config_rejects_wrong_types():
    with pytest.raises(ValidationError):
        EvolutionConfig(max_rounds="not-int")
    with pytest.raises(ValidationError):
        EvolutionConfig(top_percent_threshold="abc")
    with pytest.raises(ValidationError):
        EvolutionConfig(enabled="maybe")


def test_evolution_config_round_trip_dump():
    cfg = EvolutionConfig(enabled=True, max_rounds=4)
    d = cfg.model_dump()
    assert d["enabled"] is True
    assert d["max_rounds"] == 4
    again = EvolutionConfig(**d)
    assert again == cfg
