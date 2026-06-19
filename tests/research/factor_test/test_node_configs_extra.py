# coding: utf-8
"""Extra unit tests for node configs (NODE_CONFIG_SCHEMAS)."""

import pytest
from pydantic import ValidationError

from QuantNodes.research.factor_test.nodes.configs import (
    AdjustDateNodeConfig,
    GroupAnalyzerNodeConfig,
    ICAnalyzerNodeConfig,
    LoadDataNodeConfig,
    LongShortNodeConfig,
    NeutralizeNodeConfig,
    NODE_CONFIG_SCHEMAS,
    PreprocessNodeConfig,
    ReportNodeConfig,
    RiskCorrelationNodeConfig,
    SamplePoolNodeConfig,
    ScoreNodeConfig,
    TradabilityNodeConfig,
)


@pytest.mark.parametrize(
    "cls",
    [
        SamplePoolNodeConfig,
        TradabilityNodeConfig,
        AdjustDateNodeConfig,
        PreprocessNodeConfig,
        NeutralizeNodeConfig,
        ICAnalyzerNodeConfig,
        GroupAnalyzerNodeConfig,
        LongShortNodeConfig,
        ScoreNodeConfig,
        RiskCorrelationNodeConfig,
        ReportNodeConfig,
    ],
)
def test_extra_forbid_universally(cls):
    with pytest.raises(ValidationError):
        cls(unknown_field=1)


def test_extra_forbid_load_data():
    with pytest.raises(ValidationError):
        LoadDataNodeConfig(data_path="/tmp/", unknown=True)


def test_load_data_requires_path():
    with pytest.raises(ValidationError):
        LoadDataNodeConfig()


def test_load_data_default_keys_have_tradability():
    cfg = LoadDataNodeConfig(data_path="/tmp/x/")
    for k in ("st", "suspend", "ud_limit", "ipo_days"):
        assert k in cfg.load_keys


def test_sample_pool_index_mapping_typed():
    cfg = SamplePoolNodeConfig(index_mapping={"X": ("a.h5", "b")})
    assert cfg.index_mapping == {"X": ("a.h5", "b")}


def test_sample_pool_invalid_index_mapping_value():
    with pytest.raises(ValidationError):
        SamplePoolNodeConfig(index_mapping={"X": ("only-one",)})


def test_adjust_date_defaults():
    cfg = AdjustDateNodeConfig()
    assert cfg.adj_date_beg is None
    assert cfg.adj_date_end is None
    assert cfg.adj_mode == ["M", "end"]


def test_preprocess_defaults_round_trip():
    cfg = PreprocessNodeConfig()
    assert cfg.mad_n == 5.0
    assert cfg.pct_low == 0.025
    assert cfg.pct_high == 0.975
    rebuilt = PreprocessNodeConfig.model_validate(cfg.model_dump())
    assert rebuilt == cfg


def test_neutralize_defaults():
    cfg = NeutralizeNodeConfig()
    assert cfg.industry_neutral is False
    assert cfg.risk_neutral is False
    assert cfg.risk_factors == []


def test_group_defaults():
    cfg = GroupAnalyzerNodeConfig()
    assert cfg.groups == 5
    assert cfg.factor_direction == 1
    assert cfg.floor_mode == "group"
    assert cfg.hedge == "equal"
    assert cfg.hedge_path is None


def test_score_defaults():
    cfg = ScoreNodeConfig()
    assert cfg.enabled is True
    assert cfg.n_industries == 29
    assert cfg.n_size_groups == 3
    assert cfg.n_quantile_groups == 5


def test_report_defaults():
    cfg = ReportNodeConfig()
    assert cfg.dir == "./output/"
    assert cfg.format == ["parquet", "json"]


def test_node_config_schemas_route_table_complete():
    expected = {
        "LoadData": LoadDataNodeConfig,
        "SamplePoolFilter": SamplePoolNodeConfig,
        "TradabilityFilter": TradabilityNodeConfig,
        "AdjustDate": AdjustDateNodeConfig,
        "FactorPreprocess": PreprocessNodeConfig,
        "FactorNeutralize": NeutralizeNodeConfig,
        "ICAnalyzer": ICAnalyzerNodeConfig,
        "GroupAnalyzer": GroupAnalyzerNodeConfig,
        "LongShort": LongShortNodeConfig,
        "FactorScore": ScoreNodeConfig,
        "RiskCorrelation": RiskCorrelationNodeConfig,
        "FactorTestReport": ReportNodeConfig,
    }
    assert NODE_CONFIG_SCHEMAS == expected
