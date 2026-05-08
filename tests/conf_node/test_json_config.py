# coding=utf-8
"""JSONConfigNode 单元测试"""
import pytest
from pathlib import Path
from QuantNodes.core.node import NodeExecutionError

from QuantNodes.conf_node.json_config import JSONConfigNode


class TestJSONConfigNode:
    """JSONConfigNode 测试"""

    def test_load_basic(self, temp_json_file):
        """加载基本配置"""
        node = JSONConfigNode(file_path=temp_json_file)
        config = node.execute()
        assert isinstance(config, dict)

    def test_load_with_key(self, temp_json_file):
        """加载指定 key 下的配置"""
        node = JSONConfigNode(file_path=temp_json_file, key="database")
        config = node.execute()
        assert config["host"] == "localhost"
        assert config["port"] == 5432

    def test_file_not_found(self):
        """文件不存在时抛出异常"""
        node = JSONConfigNode(file_path="/nonexistent/config.json")
        with pytest.raises(NodeExecutionError):
            node.execute()

    def test_key_not_found(self, temp_json_file):
        """指定的 key 不存在时抛出异常"""
        node = JSONConfigNode(file_path=temp_json_file, key="nonexistent")
        with pytest.raises(NodeExecutionError):
            node.execute()

    def test_get_config_path(self, temp_json_file):
        """返回配置文件路径"""
        node = JSONConfigNode(file_path=temp_json_file)
        assert node._get_config_path() == Path(temp_json_file)

    def test_execute_uses_cache(self, temp_json_file):
        """验证 execute 使用缓存"""
        node = JSONConfigNode(file_path=temp_json_file)
        result1 = node.execute()
        result2 = node.execute()
        assert result1 is result2

    def test_reload(self, temp_json_file):
        """验证 reload 强制重新加载"""
        node = JSONConfigNode(file_path=temp_json_file)
        result1 = node.execute()
        result2 = node.reload()
        assert result1 == result2

    def test_get_item(self, temp_json_file):
        """测试字典风格访问"""
        node = JSONConfigNode(file_path=temp_json_file, key="database")
        assert node["host"] == "localhost"

    def test_get_method(self, temp_json_file):
        """测试 get 方法"""
        node = JSONConfigNode(file_path=temp_json_file, key="database")
        assert node.get("host") == "localhost"
        assert node.get("nonexistent", "default") == "default"

    def test_contains(self, temp_json_file):
        """测试 in 操作符"""
        node = JSONConfigNode(file_path=temp_json_file, key="database")
        assert "host" in node
        assert "nonexistent" not in node

    def test_nested_json(self, tmp_path):
        """测试嵌套 JSON"""
        content = '{"a": {"b": {"c": "value"}}}'
        filepath = tmp_path / "nested.json"
        filepath.write_text(content, encoding="utf-8")
        node = JSONConfigNode(file_path=filepath, key="a")
        config = node.execute()
        assert config["b"]["c"] == "value"

    def test_array_json(self, tmp_path):
        """测试 JSON 数组（应抛出 NodeExecutionError）"""
        content = '[1, 2, 3]'
        filepath = tmp_path / "array.json"
        filepath.write_text(content, encoding="utf-8")
        node = JSONConfigNode(file_path=filepath)
        with pytest.raises(NodeExecutionError):
            node.execute()

    def test_unicode_json(self, tmp_path):
        """测试 Unicode 支持"""
        content = '{"key": "中文值"}'
        filepath = tmp_path / "unicode.json"
        filepath.write_text(content, encoding="utf-8")
        node = JSONConfigNode(file_path=filepath)
        config = node.execute()
        assert config["key"] == "中文值"
