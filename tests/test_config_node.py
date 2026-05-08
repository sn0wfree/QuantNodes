# -*- coding: utf-8 -*-
"""ConfigNode unit tests"""
import pytest

from QuantNodes.conf_node import (
    IniConfigNode,
    YamlConfigNode,
    EnvConfigNode,
    JSONConfigNode,
)
from QuantNodes.core.node import NodeExecutionError


class TestIniConfigNode:
    """Tests for IniConfigNode"""

    def test_ini_load_single_section(self, tmp_path):
        """测试读取单个 section"""
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[database]
host = localhost
port = 5432
username = admin

[cache]
host = redis.local
port = 6379
""")

        node = IniConfigNode(file_path=ini_file, section="database")
        config = node.execute()

        assert config['host'] == 'localhost'
        assert config['port'] == '5432'
        assert config['username'] == 'admin'

    def test_ini_load_all_sections(self, tmp_path):
        """测试读取所有 section"""
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[section1]
key1 = value1

[section2]
key2 = value2
""")

        node = IniConfigNode(file_path=ini_file)
        config = node.execute()

        assert 'section1' in config
        assert 'section2' in config
        assert config['section1']['key1'] == 'value1'

    def test_ini_file_not_found(self, tmp_path):
        """测试文件不存在"""
        node = IniConfigNode(file_path=tmp_path / "nonexistent.ini")
        with pytest.raises(NodeExecutionError, match="INI file not found"):
            node.execute()

    def test_ini_section_not_found(self, tmp_path):
        """测试 section 不存在"""
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[existing]
key = value
""")

        node = IniConfigNode(file_path=ini_file, section="nonexistent")
        with pytest.raises(NodeExecutionError, match="Section 'nonexistent' not found"):
            node.execute()

    def test_ini_dict_access(self, tmp_path):
        """测试字典式访问"""
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[app]
debug = true
""")

        node = IniConfigNode(file_path=ini_file, section="app")
        node.execute()
        assert node['debug'] == 'true'

    def test_ini_get_method(self, tmp_path):
        """测试 get 方法"""
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[app]
name = test
""")

        node = IniConfigNode(file_path=ini_file, section="app")
        node.execute()
        assert node.get('name') == 'test'
        assert node.get('nonexistent', 'default') == 'default'


class TestYamlConfigNode:
    """Tests for YamlConfigNode"""

    def test_yaml_load(self, tmp_path):
        """测试基本加载"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("""
database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret

app:
  debug: true
  port: 8080
""")

        node = YamlConfigNode(file_path=yaml_file)
        config = node.execute()

        assert config['database']['host'] == 'localhost'
        assert config['app']['debug'] is True

    def test_yaml_load_with_key(self, tmp_path):
        """测试加载指定 key"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("""
database:
  host: localhost
  port: 5432

cache:
  host: redis.local
""")

        node = YamlConfigNode(file_path=yaml_file, key="database")
        config = node.execute()

        assert config['host'] == 'localhost'
        assert config['port'] == 5432

    def test_yaml_file_not_found(self, tmp_path):
        """测试文件不存在"""
        node = YamlConfigNode(file_path=tmp_path / "nonexistent.yaml")
        with pytest.raises(NodeExecutionError, match="YAML file not found"):
            node.execute()

    def test_yaml_key_not_found(self, tmp_path):
        """测试 key 不存在"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("existing: value\n")

        node = YamlConfigNode(file_path=yaml_file, key="nonexistent")
        with pytest.raises(NodeExecutionError, match="Key 'nonexistent' not found"):
            node.execute()

class TestEnvConfigNode:
    """Tests for EnvConfigNode"""

    def test_env_load_all(self, monkeypatch):
        """测试加载所有环境变量"""
        monkeypatch.setenv("TEST_VAR1", "value1")
        monkeypatch.setenv("TEST_VAR2", "value2")
        monkeypatch.setenv("OTHER_VAR", "other")

        node = EnvConfigNode(prefix="TEST_")
        config = node.reload()

        assert 'var1' in config
        assert 'var2' in config
        assert config['var1'] == 'value1'

    def test_env_load_lowercase_keys(self, monkeypatch):
        """测试 key 小写化"""
        monkeypatch.setenv("TEST_MYVAR", "myvalue")

        node = EnvConfigNode(prefix="TEST_", lowercase_keys=True)
        config = node.reload()

        assert 'myvar' in config
        assert 'MYVAR' not in config

    def test_env_type_conversion_int(self, monkeypatch):
        """测试 int 类型转换"""
        monkeypatch.setenv("TEST_PORT", "5432")

        node = EnvConfigNode(prefix="TEST_", types={'port': int})
        config = node.reload()

        assert config['port'] == 5432
        assert isinstance(config['port'], int)

    def test_env_type_conversion_bool(self, monkeypatch):
        """测试 bool 类型转换"""
        monkeypatch.setenv("TEST_DEBUG", "true")
        monkeypatch.setenv("TEST_ENABLED", "1")
        monkeypatch.setenv("TEST_DISABLED", "false")

        node = EnvConfigNode(prefix="TEST_", types={'debug': bool, 'enabled': bool, 'disabled': bool})
        config = node.reload()

        assert config['debug'] is True
        assert config['enabled'] is True
        assert config['disabled'] is False

    def test_env_no_prefix(self, monkeypatch):
        """测试不带前缀"""
        monkeypatch.setenv("MYAPP_VAR", "value")

        node = EnvConfigNode()
        config = node.reload()

        assert 'myapp_var' in config

    def test_env_empty_prefix(self, monkeypatch):
        """测试空前缀"""
        monkeypatch.setenv("TEST_VAR", "value")

        node = EnvConfigNode(prefix="")
        config = node.reload()

        assert 'test_var' in config

    def test_env_from_env_classmethod(self, monkeypatch):
        """测试 from_env 类方法"""
        monkeypatch.setenv("TEST_PORT", "8080")
        monkeypatch.setenv("TEST_DEBUG", "true")

        assert EnvConfigNode.from_env("TEST_PORT", target_type=int) == 8080
        assert EnvConfigNode.from_env("TEST_DEBUG", target_type=bool) is True
        assert EnvConfigNode.from_env("NONEXISTENT", default="default") == "default"


class TestJSONConfigNode:
    """Tests for JSONConfigNode"""

    def test_json_load(self, tmp_path):
        """测试基本加载"""
        json_file = tmp_path / "test.json"
        json_content = """{
    "database": {
        "host": "localhost",
        "port": 5432
    },
    "app": {
        "debug": true,
        "name": "testapp"
    }
}"""
        json_file.write_text(json_content)

        node = JSONConfigNode(file_path=json_file)
        config = node.execute()

        assert config['database']['host'] == 'localhost'
        assert config['app']['debug'] is True

    def test_json_load_with_key(self, tmp_path):
        """测试加载指定 key"""
        json_file = tmp_path / "test.json"
        json_content = """{
    "database": {
        "host": "localhost",
        "port": 5432
    }
}"""
        json_file.write_text(json_content)

        node = JSONConfigNode(file_path=json_file, key="database")
        config = node.execute()

        assert config['host'] == 'localhost'
        assert config['port'] == 5432

    def test_json_file_not_found(self, tmp_path):
        """测试文件不存在"""
        node = JSONConfigNode(file_path=tmp_path / "nonexistent.json")
        with pytest.raises(NodeExecutionError, match="JSON file not found"):
            node.execute()

    def test_json_key_not_found(self, tmp_path):
        """测试 key 不存在"""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"existing": "value"}')

        node = JSONConfigNode(file_path=json_file, key="nonexistent")
        with pytest.raises(NodeExecutionError, match="Key 'nonexistent' not found"):
            node.execute()


class TestConfigNodeCaching:
    """Tests for ConfigNode caching behavior"""

    def test_cache_by_default(self, tmp_path, monkeypatch):
        """测试默认使用缓存"""
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[app]
value = original
""")

        node = IniConfigNode(file_path=ini_file, section="app")
        config1 = node.execute()
        assert config1['value'] == 'original'

        ini_file.write_text("""
[app]
value = modified
""")

        config2 = node.execute()
        assert config2['value'] == 'original'

    def test_reload_bypasses_cache(self, tmp_path):
        """测试 reload 绕过缓存"""
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[app]
value = original
""")

        node = IniConfigNode(file_path=ini_file, section="app")
        config1 = node.execute()
        assert config1['value'] == 'original'

        ini_file.write_text("""
[app]
value = modified
""")

        config2 = node.reload()
        assert config2['value'] == 'modified'

    def test_execute_without_cache(self, tmp_path):
        """测试不使用缓存"""
        ini_file = tmp_path / "test.ini"
        ini_file.write_text("""
[app]
value = original
""")

        node = IniConfigNode(file_path=ini_file, section="app")
        config1 = node.execute(use_cache=False)
        assert config1['value'] == 'original'

        ini_file.write_text("""
[app]
value = modified
""")

        config2 = node.execute(use_cache=False)
        assert config2['value'] == 'modified'
