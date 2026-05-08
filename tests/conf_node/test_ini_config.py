# coding=utf-8
"""IniConfigNode 单元测试"""
import pytest
from pathlib import Path
from QuantNodes.core.node import NodeExecutionError

from QuantNodes.conf_node.ini_config import IniConfigNode


class TestIniConfigNode:
    """IniConfigNode 测试"""

    def test_load_all_sections(self, temp_ini_file):
        """加载所有 section"""
        node = IniConfigNode(file_path=temp_ini_file)
        config = node.execute()
        assert isinstance(config, dict)
        assert "database" in config
        assert "cache" in config

    def test_load_single_section(self, temp_ini_file):
        """加载单个 section"""
        node = IniConfigNode(file_path=temp_ini_file, section="database")
        config = node.execute()
        assert config["host"] == "localhost"
        assert config["port"] == "5432"

    def test_file_not_found(self):
        """文件不存在时抛出异常"""
        node = IniConfigNode(file_path="/nonexistent/config.ini")
        with pytest.raises(NodeExecutionError):
            node.execute()

    def test_section_not_found(self, temp_ini_file):
        """指定的 section 不存在时抛出异常"""
        node = IniConfigNode(file_path=temp_ini_file, section="nonexistent")
        with pytest.raises(NodeExecutionError):
            node.execute()

    def test_get_config_path(self, temp_ini_file):
        """返回配置文件路径"""
        node = IniConfigNode(file_path=temp_ini_file)
        assert node._get_config_path() == Path(temp_ini_file)

    def test_execute_uses_cache(self, temp_ini_file):
        """验证 execute 使用缓存"""
        node = IniConfigNode(file_path=temp_ini_file)
        result1 = node.execute()
        result2 = node.execute()
        assert result1 is result2

    def test_reload(self, temp_ini_file):
        """验证 reload 强制重新加载"""
        node = IniConfigNode(file_path=temp_ini_file)
        result1 = node.execute()
        result2 = node.reload()
        assert result1 == result2

    def test_get_item(self, temp_ini_file):
        """测试字典风格访问"""
        node = IniConfigNode(file_path=temp_ini_file, section="database")
        assert node["host"] == "localhost"

    def test_get_method(self, temp_ini_file):
        """测试 get 方法"""
        node = IniConfigNode(file_path=temp_ini_file, section="database")
        assert node.get("host") == "localhost"
        assert node.get("nonexistent", "default") == "default"

    def test_contains(self, temp_ini_file):
        """测试 in 操作符"""
        node = IniConfigNode(file_path=temp_ini_file, section="database")
        assert "host" in node
        assert "nonexistent" not in node

    def test_encoding(self, tmp_path):
        """测试编码支持"""
        content = """[section]
key = value
"""
        filepath = tmp_path / "unicode.ini"
        filepath.write_text(content, encoding="utf-8")
        node = IniConfigNode(file_path=filepath)
        config = node.execute()
        assert "section" in config

    def test_no_section_file(self, tmp_path):
        """无 section 的 INI 文件"""
        content = """[section1]
key1 = value1
key2 = value2
"""
        filepath = tmp_path / "nosection.ini"
        filepath.write_text(content, encoding="utf-8")
        node = IniConfigNode(file_path=filepath)
        config = node.execute()
        assert "section1" in config

    def test_boolean_values(self, tmp_path):
        """测试布尔值解析"""
        content = """[section]
enabled = true
disabled = false
"""
        filepath = tmp_path / "bool.ini"
        filepath.write_text(content, encoding="utf-8")
        node = IniConfigNode(file_path=filepath, section="section")
        config = node.execute()
        assert config["enabled"] == "true"
        assert config["disabled"] == "false"

    def test_numeric_values(self, tmp_path):
        """测试数值解析"""
        content = """[section]
int_val = 123
float_val = 3.14
"""
        filepath = tmp_path / "numeric.ini"
        filepath.write_text(content, encoding="utf-8")
        node = IniConfigNode(file_path=filepath, section="section")
        config = node.execute()
        assert config["int_val"] == "123"
        assert config["float_val"] == "3.14"
