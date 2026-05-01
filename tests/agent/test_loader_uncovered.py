# coding=utf-8
"""
ConfigLoader 未覆盖方法单元测试

覆盖:
- load() 文件不存在
- load() 空 YAML
- get_config()
- load_config() 便捷函数
- check_coverage() 未注册算子
"""

import pytest
import yaml
import tempfile
import os

from QuantNodes.agent.config.loader import ConfigLoader, load_config
from QuantNodes.agent.config.types import StrategyConfig, OperationConfig


class TestLoaderMissingFile:
    """load() 文件不存在"""

    def test_missing_file_raises(self):
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/config.yaml")


class TestLoaderEmptyYAML:
    """load() 空 YAML"""

    def test_empty_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("")
            tmp_path = f.name
        try:
            loader = ConfigLoader()
            config = loader.load(tmp_path)
            assert isinstance(config, StrategyConfig)
            assert config.name == ""
            assert config.factors == []
        finally:
            os.unlink(tmp_path)

    def test_null_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("---\n")
            tmp_path = f.name
        try:
            loader = ConfigLoader()
            config = loader.load(tmp_path)
            assert isinstance(config, StrategyConfig)
        finally:
            os.unlink(tmp_path)


class TestGetConfig:
    """get_config() 测试"""

    def test_get_config_returns_loaded(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump({"name": "test_config", "factors": []}, f)
            tmp_path = f.name
        try:
            loader = ConfigLoader()
            loader.load(tmp_path)
            config = loader.get_config()
            assert config is not None
            assert config.name == "test_config"
        finally:
            os.unlink(tmp_path)

    def test_get_config_before_load(self):
        loader = ConfigLoader()
        assert loader.get_config() is None


class TestLoadConfigConvenience:
    """load_config() 便捷函数"""

    def test_load_config_function(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            yaml.dump({
                "name": "convenience_test",
                "factors": [{"name": "ma5", "expr": "ts_mean(close, 5)"}],
            }, f)
            tmp_path = f.name
        try:
            config = load_config(tmp_path)
            assert config.name == "convenience_test"
            assert len(config.factors) == 1
        finally:
            os.unlink(tmp_path)


class TestCheckCoverageUnknownOp:
    """check_coverage() 未注册算子"""

    def test_unknown_operator_category(self):
        loader = ConfigLoader()
        config = StrategyConfig(
            name="test",
            operations=[OperationConfig(
                type="section",
                name="test_op",
                category="nonexistent_operator_xyz",
                inputs=["col"],
            )],
        )
        report = loader.check_coverage(config)
        assert not report.is_complete
        assert any("nonexistent_operator_xyz" in u for u in report.unresolved)
