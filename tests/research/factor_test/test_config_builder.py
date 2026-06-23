# coding: utf-8
"""SingleFactorTestConfigBuilder 测试 (Phase 3.4)。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from QuantNodes.research.factor_test.config import SingleFactorTestConfig
from QuantNodes.research.factor_test.config_builder import (
    SingleFactorTestConfigBuilder,
)


class TestBuildBasics:
    def test_minimal_build(self):
        cfg = SingleFactorTestConfigBuilder().factor("f", "f.h5").build()
        assert isinstance(cfg, SingleFactorTestConfig)
        assert cfg.factor.name == "f"
        assert cfg.factor.factor_dir == "f.h5"

    def test_missing_factor_raises(self):
        with pytest.raises(ValueError, match="factor is required"):
            SingleFactorTestConfigBuilder().dates(20240101, 20241231).build()

    def test_chaining_returns_self(self):
        b = SingleFactorTestConfigBuilder()
        assert b.factor("f", "f.h5") is b
        assert b.dates(20240101, 20241231) is b
        assert b.groups(5) is b

    def test_defaults_match_setting_defaults(self):
        """未设置的字段沿用 pydantic 默认 (单一真值源)。"""
        cfg = SingleFactorTestConfigBuilder().factor("f", "f.h5").build()
        assert cfg.analysis.group.groups == 5
        assert cfg.preprocess.mad_n == 5.0
        assert cfg.output.dir == "./output/"
        assert cfg.data_path == "./testdata/test_h5_new/"


class TestSetters:
    def test_factor_optional_fields(self):
        cfg = (
            SingleFactorTestConfigBuilder()
            .factor(
                "f", "f.parquet",
                fmt="parquet", hypothesis="momentum",
                description="d", expression="x>0", factor_key="data",
            )
            .build()
        )
        assert cfg.factor.format == "parquet"
        assert cfg.factor.hypothesis == "momentum"
        assert cfg.factor.factor_key == "data"

    def test_dates_and_adj_mode(self):
        cfg = (
            SingleFactorTestConfigBuilder()
            .factor("f", "f.h5")
            .dates(20240101, 20241231)
            .adj_mode(["W", "start"])
            .build()
        )
        assert cfg.preprocess.adj_date_beg == 20240101
        assert cfg.preprocess.adj_date_end == 20241231
        assert cfg.preprocess.adj_mode == ["W", "start"]

    def test_sample(self):
        cfg = (
            SingleFactorTestConfigBuilder()
            .factor("f", "f.h5")
            .sample("HS300", "银行")
            .build()
        )
        assert cfg.preprocess.sample_index == "HS300"
        assert cfg.preprocess.sample_industry == "银行"

    def test_preprocess(self):
        cfg = (
            SingleFactorTestConfigBuilder()
            .factor("f", "f.h5")
            .preprocess(missing="ind_avg", extreme="median", norm="zscore", mad_n=3.0)
            .build()
        )
        assert cfg.preprocess.missing == "ind_avg"
        assert cfg.preprocess.extreme == "median"
        assert cfg.preprocess.norm == "zscore"
        assert cfg.preprocess.mad_n == 3.0

    def test_neutralize(self):
        cfg = (
            SingleFactorTestConfigBuilder()
            .factor("f", "f.h5")
            .neutralize(industry=True, risk=True, risk_factors=[("r.h5", "k")])
            .build()
        )
        assert cfg.preprocess.industry_neutral is True
        assert cfg.preprocess.risk_neutral is True
        assert cfg.preprocess.risk_factors == [("r.h5", "k")]

    def test_tradable(self):
        cfg = (
            SingleFactorTestConfigBuilder()
            .factor("f", "f.h5")
            .tradable(no_st=False, min_ipo_days=120)
            .build()
        )
        assert cfg.preprocess.tradable.no_st is False
        assert cfg.preprocess.tradable.min_ipo_days == 120

    def test_analysis_setters(self):
        cfg = (
            SingleFactorTestConfigBuilder()
            .factor("f", "f.h5")
            .ic(min_group_size=10)
            .groups(10, direction=-1, floor_mode="last", hedge="HS300")
            .longshort(direction=-1)
            .score(enabled=False, n_industries=30)
            .risk_corr("all")
            .build()
        )
        assert cfg.analysis.ic.min_group_size == 10
        assert cfg.analysis.group.groups == 10
        assert cfg.analysis.group.factor_direction == -1
        assert cfg.analysis.group.floor_mode == "last"
        assert cfg.analysis.longshort.factor_direction == -1
        assert cfg.analysis.score.enabled is False
        assert cfg.analysis.score.n_industries == 30
        assert cfg.analysis.risk_corr.factors == "all"

    def test_output_feedback_qg_evolution(self):
        cfg = (
            SingleFactorTestConfigBuilder()
            .factor("f", "f.h5")
            .output("./out/", fmt=["json"])
            .feedback(enabled=True)
            .quality_gate(enabled=True)
            .evolution(enabled=True, max_rounds=5)
            .build()
        )
        assert cfg.output.dir == "./out/"
        assert cfg.output.format == ["json"]
        assert cfg.feedback.enabled is True
        assert cfg.quality_gate.enabled is True
        assert cfg.evolution.enabled is True
        assert cfg.evolution.max_rounds == 5

    def test_top_level(self):
        cfg = (
            SingleFactorTestConfigBuilder()
            .factor("f", "f.h5")
            .data_path("/data/")
            .load_keys(["cp", "st"])
            .build()
        )
        assert cfg.data_path == "/data/"
        assert cfg.load_keys == ["cp", "st"]


class TestValidation:
    def test_invalid_field_raises(self):
        """pydantic extra/type 校验在 build() 时触发。"""
        with pytest.raises(ValidationError):
            (
                SingleFactorTestConfigBuilder()
                .factor("f", "f.h5")
                .ic(min_group_size="not_an_int")
                .build()
            )


class TestEquivalenceWithDirect:
    def test_builder_matches_direct_construction(self):
        """builder 输出与等价直接构造 model_dump bitwise 一致。"""
        from QuantNodes.research.factor_test.config import (
            FactorSetting,
            PreprocessSetting,
            AnalysisSetting,
            OutputSetting,
        )

        direct = SingleFactorTestConfig(
            factor=FactorSetting(name="f", factor_dir="f.h5", hypothesis="momentum"),
            preprocess=PreprocessSetting(
                adj_date_beg=20240101, adj_date_end=20241231,
                missing="", extreme="median", norm="zscore",
            ),
            analysis=AnalysisSetting(
                ic={"min_group_size": 5},
                group={"groups": 5, "factor_direction": 1},
            ),
            output=OutputSetting(dir="./out/", format=["json"]),
            data_path="./data/",
        )
        built = (
            SingleFactorTestConfigBuilder()
            .factor("f", "f.h5", hypothesis="momentum")
            .dates(20240101, 20241231)
            .preprocess(missing="", extreme="median", norm="zscore")
            .ic(min_group_size=5)
            .groups(5, direction=1)
            .output("./out/", fmt=["json"])
            .data_path("./data/")
            .build()
        )
        assert built.model_dump() == direct.model_dump()
