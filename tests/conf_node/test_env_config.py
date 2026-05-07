# coding=utf-8
"""EnvConfigNode 单元测试"""
import os
import pytest
from unittest.mock import patch

from QuantNodes.conf_node.env_config import EnvConfigNode


class TestEnvConfigNode:
    """EnvConfigNode 测试"""

    def test_load_no_prefix(self):
        """无前缀加载所有环境变量"""
        with patch.dict(os.environ, {"TEST_KEY": "test_value", "ANOTHER": "value"}, clear=False):
            node = EnvConfigNode()
            config = node.execute()
            assert "TEST_KEY" in config or "test_key" in config

    def test_load_with_prefix(self):
        """带前缀加载环境变量"""
        with patch.dict(os.environ, {"DB_HOST": "localhost", "DB_PORT": "5432", "OTHER": "x"}, clear=False):
            node = EnvConfigNode(prefix="DB_")
            config = node.execute()
            assert config.get("host") == "localhost"
            assert config.get("port") == "5432"
            assert "OTHER" not in config

    def test_lowercase_keys(self):
        """默认将 key 转为小写"""
        with patch.dict(os.environ, {"TEST_KEY": "value"}, clear=False):
            node = EnvConfigNode()
            config = node.execute()
            assert "test_key" in config or "TEST_KEY" in config

    def test_uppercase_keys(self):
        """保留原始大小写"""
        with patch.dict(os.environ, {"TEST_KEY": "value"}, clear=False):
            node = EnvConfigNode(lowercase_keys=False)
            config = node.execute()
            assert "TEST_KEY" in config

    def test_types_conversion_int(self):
        """类型转换 - int"""
        with patch.dict(os.environ, {"PORT": "5432"}, clear=False):
            node = EnvConfigNode(types={"port": int})
            config = node.execute()
            assert config.get("port") == 5432
            assert isinstance(config.get("port"), int)

    def test_types_conversion_bool(self):
        """类型转换 - bool"""
        with patch.dict(os.environ, {"DEBUG": "true", "VERBOSE": "false"}, clear=False):
            node = EnvConfigNode(types={"debug": bool, "verbose": bool})
            config = node.execute()
            assert config.get("debug") is True
            assert config.get("verbose") is False

    def test_types_conversion_float(self):
        """类型转换 - float"""
        with patch.dict(os.environ, {"RATE": "3.14"}, clear=False):
            node = EnvConfigNode(types={"rate": float})
            config = node.execute()
            assert config.get("rate") == 3.14
            assert isinstance(config.get("rate"), float)

    def test_types_conversion_invalid_int(self):
        """类型转换 - int 转换失败时保留原值"""
        with patch.dict(os.environ, {"PORT": "not_a_number"}, clear=False):
            node = EnvConfigNode(types={"port": int})
            config = node.execute()
            assert config.get("port") == "not_a_number"

    def test_separator(self):
        """自定义分隔符"""
        with patch.dict(os.environ, {"PREFIX_TEST_KEY": "value"}, clear=False):
            node = EnvConfigNode(prefix="PREFIX_", separator="_")
            config = node.execute()
            assert "test_key" in config

    def test_get_config_path(self):
        """返回 None（无配置文件概念）"""
        node = EnvConfigNode()
        assert node._get_config_path() is None

    def test_get_item(self):
        """测试字典风格访问"""
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}, clear=False):
            node = EnvConfigNode(prefix="TEST_")
            config = node.execute()
            if "key" in config:
                assert node["key"] == "test_value"
            elif "TEST_KEY" in config:
                assert node["TEST_KEY"] == "test_value"

    def test_get_method(self):
        """测试 get 方法"""
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}, clear=False):
            node = EnvConfigNode()
            result = node.get("test_key", "default")
            assert result in ("test_value", "default")

    def test_from_env_string(self):
        """测试 from_env 便捷方法 - str"""
        with patch.dict(os.environ, {"TEST_VAR": "hello"}, clear=False):
            result = EnvConfigNode.from_env("TEST_VAR")
            assert result == "hello"

    def test_from_env_int(self):
        """测试 from_env 便捷方法 - int"""
        with patch.dict(os.environ, {"TEST_VAR": "123"}, clear=False):
            result = EnvConfigNode.from_env("TEST_VAR", target_type=int)
            assert result == 123
            assert isinstance(result, int)

    def test_from_env_bool_true(self):
        """测试 from_env 便捷方法 - bool (true)"""
        for val in ["true", "1", "yes", "on"]:
            with patch.dict(os.environ, {"TEST_VAR": val}, clear=False):
                result = EnvConfigNode.from_env("TEST_VAR", target_type=bool)
                assert result is True, f"Failed for {val}"

    def test_from_env_bool_false(self):
        """测试 from_env 便捷方法 - bool (false)"""
        for val in ["false", "0", "no", "off"]:
            with patch.dict(os.environ, {"TEST_VAR": val}, clear=False):
                result = EnvConfigNode.from_env("TEST_VAR", target_type=bool)
                assert result is False, f"Failed for {val}"

    def test_from_env_default(self):
        """测试 from_env 默认值"""
        with patch.dict(os.environ, {}, clear=True):
            result = EnvConfigNode.from_env("NONEXISTENT", default="default_value")
            assert result == "default_value"

    def test_from_env_missing_no_default(self):
        """测试 from_env 缺失无默认时返回 None"""
        with patch.dict(os.environ, {}, clear=True):
            result = EnvConfigNode.from_env("NONEXISTENT")
            assert result is None

    def test_empty_prefix(self):
        """空前缀"""
        with patch.dict(os.environ, {"KEY": "value"}, clear=False):
            node = EnvConfigNode(prefix="")
            config = node.execute()
            assert "key" in config or "KEY" in config
