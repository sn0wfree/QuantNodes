# coding=utf-8
"""Tests for PR-QN-4: YAML dual-whitelist strict validation."""
from __future__ import annotations

import pytest

from QuantNodes.operators.composite_dag import (
    _COMPOSITE_REGISTRY,
    _COMPOSITE_REGISTRY_PANDAS,
    _compile_template_string,
    load_composites_from_yaml,
)


class TestCompileTemplateStringEngine:
    def test_polars_whitelist_allows_rolling_mean(self):
        fn = _compile_template_string("x.rolling_mean(window_size=10)", engine="polars")
        assert callable(fn)

    def test_polars_whitelist_rejects_groupby(self):
        with pytest.raises(ValueError, match="不允许"):
            _compile_template_string("x.groupby('a').mean()", engine="polars")

    def test_pandas_whitelist_allows_groupby(self):
        fn = _compile_template_string("x.groupby('a').mean()", engine="pandas")
        assert callable(fn)

    def test_pandas_whitelist_rejects_rolling_mean(self):
        with pytest.raises(ValueError, match="不允许"):
            _compile_template_string("x.rolling_mean(window_size=10)", engine="pandas")

    def test_pandas_whitelist_allows_rolling(self):
        fn = _compile_template_string("x.rolling(10).mean()", engine="pandas")
        assert callable(fn)

    def test_polars_whitelist_allows_over(self):
        fn = _compile_template_string("x.mean().over(g)", engine="polars")
        assert callable(fn)

    def test_pandas_whitelist_rejects_over(self):
        with pytest.raises(ValueError, match="不允许"):
            _compile_template_string("x.mean().over(g)", engine="pandas")

    def test_pandas_whitelist_allows_transform(self):
        fn = _compile_template_string("x.transform('mean')", engine="pandas")
        assert callable(fn)

    def test_polars_whitelist_rejects_transform(self):
        with pytest.raises(ValueError, match="不允许"):
            _compile_template_string("x.transform('mean')", engine="polars")

    def test_default_engine_is_polars(self):
        fn = _compile_template_string("x.rolling_mean(window_size=10)")
        assert callable(fn)


class TestLoadYamlEngineField:
    def test_polars_yaml_register_to_polars(self, tmp_path):
        yaml_content = """
composites:
  - name: test_polars_yaml
    engine: polars
    doc: "test"
    params:
      x: {type: expr, required: true}
    template: "x.rolling_mean(window_size=5)"
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        count = load_composites_from_yaml(str(yaml_file))
        assert count == 1
        assert "test_polars_yaml" in _COMPOSITE_REGISTRY.list()
        assert "test_polars_yaml" not in _COMPOSITE_REGISTRY_PANDAS.list()

    def test_pandas_yaml_register_to_pandas(self, tmp_path):
        yaml_content = """
composites:
  - name: test_pandas_yaml
    engine: pandas
    doc: "test"
    params:
      x: {type: dataframe, required: true}
    template: "x.groupby('a').mean()"
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        count = load_composites_from_yaml(str(yaml_file))
        assert count == 1
        assert "test_pandas_yaml" in _COMPOSITE_REGISTRY_PANDAS.list()
        assert "test_pandas_yaml" not in _COMPOSITE_REGISTRY.list()

    def test_default_engine_is_polars(self, tmp_path):
        # No engine field → defaults to polars
        yaml_content = """
composites:
  - name: test_default_yaml
    doc: "test"
    params:
      x: {type: expr, required: true}
    template: "x.rolling_mean(window_size=5)"
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        count = load_composites_from_yaml(str(yaml_file))
        assert count == 1
        assert "test_default_yaml" in _COMPOSITE_REGISTRY.list()

    def test_pandas_yaml_rejects_polars_only_func(self, tmp_path):
        yaml_content = """
composites:
  - name: bad_yaml
    engine: pandas
    doc: "test"
    params:
      x: {type: dataframe, required: true}
    template: "x.rolling_mean(window_size=5)"
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="不允许"):
            load_composites_from_yaml(str(yaml_file))

    def test_polars_yaml_rejects_pandas_only_func(self, tmp_path):
        yaml_content = """
composites:
  - name: bad_yaml
    engine: polars
    doc: "test"
    params:
      x: {type: expr, required: true}
    template: "x.groupby('a').mean()"
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)
        with pytest.raises(ValueError, match="不允许"):
            load_composites_from_yaml(str(yaml_file))
