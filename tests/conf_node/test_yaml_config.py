# coding=utf-8
"""YAMLConfigNode 单元测试"""
import pytest
from pathlib import Path
from QuantNodes.core.node import NodeExecutionError

from QuantNodes.conf_node.yaml_config import YamlConfigNode


class TestYamlConfigNode:
    """YamlConfigNode 测试"""

    def test_load_basic(self, temp_yaml_file):
        """加载基本配置"""
        node = YamlConfigNode(file_path=temp_yaml_file)
        config = node.execute()
        assert isinstance(config, dict)

    def test_load_with_key(self, temp_yaml_file):
        """加载指定 key 下的配置"""
        node = YamlConfigNode(file_path=temp_yaml_file, key="database")
        config = node.execute()
        assert config["host"] == "localhost"
        assert config["port"] == 5432

    def test_file_not_found(self):
        """文件不存在时抛出异常"""
        node = YamlConfigNode(file_path="/nonexistent/config.yaml")
        with pytest.raises(NodeExecutionError):
            node.execute()

    def test_key_not_found(self, temp_yaml_file):
        """指定的 key 不存在时抛出异常"""
        node = YamlConfigNode(file_path=temp_yaml_file, key="nonexistent")
        with pytest.raises(NodeExecutionError):
            node.execute()

    def test_empty_yaml(self, tmp_path):
        """空 YAML 文件返回空字典"""
        filepath = tmp_path / "empty.yaml"
        filepath.write_text("", encoding="utf-8")
        node = YamlConfigNode(file_path=filepath)
        config = node.execute()
        assert config == {}

    def test_get_config_path(self, temp_yaml_file):
        """返回配置文件路径"""
        node = YamlConfigNode(file_path=temp_yaml_file)
        assert node._get_config_path() == Path(temp_yaml_file)

    def test_execute_uses_cache(self, temp_yaml_file):
        """验证 execute 使用缓存"""
        node = YamlConfigNode(file_path=temp_yaml_file)
        result1 = node.execute()
        result2 = node.execute()
        assert result1 is result2

    def test_reload(self, temp_yaml_file):
        """验证 reload 强制重新加载"""
        node = YamlConfigNode(file_path=temp_yaml_file)
        result1 = node.execute()
        result2 = node.reload()
        assert result1 == result2

    def test_get_item(self, temp_yaml_file):
        """测试字典风格访问"""
        node = YamlConfigNode(file_path=temp_yaml_file, key="database")
        assert node["host"] == "localhost"

    def test_get_method(self, temp_yaml_file):
        """测试 get 方法"""
        node = YamlConfigNode(file_path=temp_yaml_file, key="database")
        assert node.get("host") == "localhost"
        assert node.get("nonexistent", "default") == "default"

    def test_contains(self, temp_yaml_file):
        """测试 in 操作符"""
        node = YamlConfigNode(file_path=temp_yaml_file, key="database")
        assert "host" in node
        assert "nonexistent" not in node

    def test_encoding(self, tmp_path):
        """测试编码支持"""
        content = "key: 中文值\n"
        filepath = tmp_path / "unicode.yaml"
        filepath.write_text(content, encoding="gbk")
        node = YamlConfigNode(file_path=filepath, encoding="gbk")
        config = node.execute()
        assert config["key"] == "中文值"

    def test_with_slash_separator(self, tmp_path):
        """测试嵌套 key 访问"""
        content = """
a:
  b:
    c: value
"""
        filepath = tmp_path / "nested.yaml"
        filepath.write_text(content, encoding="utf-8")
        node = YamlConfigNode(file_path=filepath, key="a")
        config = node.execute()
        assert config["b"]["c"] == "value"
