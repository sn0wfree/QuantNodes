# coding: utf-8
"""Unit tests for Pydantic settings in factor_test.config."""

import pytest
from pydantic import ValidationError

from QuantNodes.research.factor_test.config import (
    AnalysisSetting,
    FactorSetting,
    FeedbackSetting,
    GroupSetting,
    ICSetting,
    LongShortSetting,
    OutputSetting,
    PreprocessSetting,
    QualityGateConfig,
    RiskCorrelationSetting,
    ScoreSetting,
    SingleFactorTestConfig,
    TradableSetting,
)


class TestFactorSetting:
    def test_required_fields(self):
        with pytest.raises(ValidationError):
            FactorSetting()
        with pytest.raises(ValidationError):
            FactorSetting(name="x")

    def test_minimal(self):
        s = FactorSetting(name="alpha", factor_dir="x.h5")
        assert s.name == "alpha"
        assert s.factor_dir == "x.h5"
        assert s.factor_key == ""
        assert s.format == "h5"
        assert s.hypothesis == ""
        assert s.description == ""
        assert s.expression == ""


class TestTradableSetting:
    def test_defaults(self):
        s = TradableSetting()
        assert s.no_st is True
        assert s.no_suspended is True
        assert s.no_up_down_limit is False
        assert s.min_ipo_days == 360
        assert s.trace is None

    def test_trace_dict(self):
        s = TradableSetting(trace={"suspend": (25, 1)})
        assert s.trace == {"suspend": (25, 1)}

    def test_min_ipo_days_negative_allowed_unrestricted(self):
        s = TradableSetting(min_ipo_days=-5)
        assert s.min_ipo_days == -5


class TestICSetting:
    def test_default(self):
        assert ICSetting().min_group_size == 5

    def test_custom(self):
        assert ICSetting(min_group_size=10).min_group_size == 10


class TestGroupSetting:
    def test_defaults(self):
        s = GroupSetting()
        assert s.groups == 5
        assert s.factor_direction == 1
        assert s.floor_mode == "group"
        assert s.hedge == "equal"
        assert s.hedge_path is None

    def test_custom(self):
        s = GroupSetting(groups=10, factor_direction=-1, floor_mode="last",
                         hedge="custom", hedge_path="/p")
        assert s.groups == 10
        assert s.factor_direction == -1
        assert s.floor_mode == "last"
        assert s.hedge == "custom"
        assert s.hedge_path == "/p"


class TestLongShortSetting:
    def test_default(self):
        assert LongShortSetting().factor_direction == 1

    def test_negative(self):
        assert LongShortSetting(factor_direction=-1).factor_direction == -1


class TestRiskCorrelationSetting:
    def test_default(self):
        assert RiskCorrelationSetting().factors == "all"

    def test_custom_string(self):
        assert RiskCorrelationSetting(factors="none").factors == "none"


class TestScoreSettingExtras:
    def test_disabled(self):
        s = ScoreSetting(enabled=False)
        assert s.enabled is False
        assert s.n_industries == 29


class TestOutputSetting:
    def test_defaults(self):
        s = OutputSetting()
        assert s.dir == "./output/"
        assert s.format == ["parquet", "json"]

    def test_custom(self):
        s = OutputSetting(dir="/tmp/o/", format=["json"])
        assert s.dir == "/tmp/o/"
        assert s.format == ["json"]


class TestFeedbackSetting:
    def test_defaults(self):
        s = FeedbackSetting()
        assert s.enabled is False
        assert s.output_dir is None
        assert s.judge_enabled is False
        assert s.judge_model == "mock"
        assert s.judge_max_attempts == 3

    def test_custom(self):
        s = FeedbackSetting(enabled=True, output_dir="/tmp/fb",
                            judge_enabled=True, judge_model="deepseek-v3",
                            judge_max_attempts=5)
        assert s.enabled is True
        assert s.output_dir == "/tmp/fb"
        assert s.judge_enabled is True
        assert s.judge_model == "deepseek-v3"
        assert s.judge_max_attempts == 5


class TestQualityGateConfig:
    def test_defaults(self):
        s = QualityGateConfig()
        assert s.enabled is False
        assert s.zoo_path is None

    def test_custom(self):
        s = QualityGateConfig(enabled=True, zoo_path="/zoo")
        assert s.enabled is True
        assert s.zoo_path == "/zoo"


class TestPreprocessExtras:
    def test_defaults_full_set(self):
        s = PreprocessSetting()
        assert s.adj_mode == ["M", "end"]
        assert s.sample_index == "all"
        assert s.sample_index_customdir is None
        assert s.sample_industry == "all"
        assert isinstance(s.tradable, TradableSetting)
        assert s.missing == ""
        assert s.extreme == ""
        assert s.norm == ""
        assert s.industry_neutral is False
        assert s.risk_neutral is False
        assert s.risk_factors == []
        assert s.i18n_name_map is None

    def test_i18n_name_map_validation(self):
        s = PreprocessSetting(i18n_name_map={"id_x": "name_x"})
        assert s.i18n_name_map == {"id_x": "name_x"}

    def test_sample_index_customdir_tuple(self):
        s = PreprocessSetting(sample_index_customdir=("a.h5", "k"))
        assert s.sample_index_customdir == ("a.h5", "k")


class TestAnalysisSetting:
    def test_defaults_factories(self):
        a = AnalysisSetting()
        assert isinstance(a.ic, ICSetting)
        assert isinstance(a.group, GroupSetting)
        assert isinstance(a.longshort, LongShortSetting)
        assert isinstance(a.score, ScoreSetting)
        assert isinstance(a.risk_corr, RiskCorrelationSetting)

    def test_nested_override(self):
        a = AnalysisSetting(ic=ICSetting(min_group_size=20))
        assert a.ic.min_group_size == 20
        assert a.group.groups == 5


class TestSingleFactorTestConfig:
    def _factor(self):
        return FactorSetting(name="x", factor_dir="x.h5")

    def _preprocess(self):
        return PreprocessSetting(adj_date_beg=20240101, adj_date_end=20241231)

    def test_minimal_required(self):
        cfg = SingleFactorTestConfig(factor=self._factor(), preprocess=self._preprocess())
        assert cfg.data_path == "./testdata/test_h5_new/"
        assert isinstance(cfg.analysis, AnalysisSetting)
        assert isinstance(cfg.output, OutputSetting)
        assert isinstance(cfg.feedback, FeedbackSetting)
        assert isinstance(cfg.quality_gate, QualityGateConfig)
        # load_keys default has tradability keys
        for k in ("st", "suspend", "ud_limit", "ipo_days"):
            assert k in cfg.load_keys

    def test_missing_required_raises(self):
        with pytest.raises(ValidationError):
            SingleFactorTestConfig(preprocess=self._preprocess())
        with pytest.raises(ValidationError):
            SingleFactorTestConfig(factor=self._factor())

    def test_round_trip_dump(self):
        cfg = SingleFactorTestConfig(factor=self._factor(), preprocess=self._preprocess(),
                                     data_path="/tmp/data/")
        d = cfg.model_dump()
        again = SingleFactorTestConfig.model_validate(d)
        assert again.data_path == "/tmp/data/"
        assert again.factor.name == "x"
