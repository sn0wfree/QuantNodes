# -*- coding: utf-8 -*-
"""QuantNodes.core.config 单元测试"""
import os
from unittest.mock import patch


class TestDatabaseConfig:
    def test_default_values(self):
        from QuantNodes.core.config import DatabaseConfig
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 3306
        assert config.database == "quant"

    def test_custom_values(self):
        from QuantNodes.core.config import DatabaseConfig
        config = DatabaseConfig(
            host="db.example.com",
            port=3307,
            user="admin",
            password="secret",
            database="mydb",
        )
        assert config.host == "db.example.com"
        assert config.port == 3307
        assert config.user == "admin"


class TestClickHouseConfig:
    def test_default_values(self):
        from QuantNodes.core.config import ClickHouseConfig
        with patch.dict(os.environ, {}, clear=True):
            config = ClickHouseConfig()
            assert config.host == "localhost"
            assert config.port == 8123

    def test_custom_values(self):
        from QuantNodes.core.config import ClickHouseConfig
        config = ClickHouseConfig(
            host="ch.example.com",
            port=8443,
            user="admin",
            password="ch_pass",
            database="analytics",
            secure=True,
        )
        assert config.host == "ch.example.com"
        assert config.port == 8443


class TestDuckDBConfig:
    def test_default_values(self):
        from QuantNodes.core.config import DuckDBConfig
        with patch.dict(os.environ, {}, clear=True):
            config = DuckDBConfig()
            assert config.path == ":memory:"
            assert config.read_only is False

    def test_custom_values(self):
        from QuantNodes.core.config import DuckDBConfig
        config = DuckDBConfig(path="/tmp/mydb.duckdb", read_only=True)
        assert config.path == "/tmp/mydb.duckdb"
        assert config.read_only is True


class TestLLMConfig:
    def test_default_values(self):
        from QuantNodes.core.config import LLMConfig
        config = LLMConfig()
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model == "gpt-4"

    def test_custom_values(self):
        from QuantNodes.core.config import LLMConfig
        config = LLMConfig(
            api_key="sk-test-key",
            base_url="https://api.example.com/v1",
            model="gpt-3.5-turbo",
            timeout=120,
            max_retries=5,
        )
        assert config.api_key == "sk-test-key"
        assert config.model == "gpt-3.5-turbo"


class TestSettings:
    def test_default_values(self):
        from QuantNodes.core.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.project_name == "QuantNodes"
            assert settings.debug is True

    def test_nested_configs(self):
        from QuantNodes.core.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            assert settings.mysql is not None
            assert settings.clickhouse is not None
            assert settings.duckdb is not None
            assert settings.llm is not None

    def test_to_dict_hides_passwords(self):
        from QuantNodes.core.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            settings.mysql.password = "real_password"
            data = settings.to_dict()
            assert data['mysql']['password'] == '***'

    def test_to_dict_hides_llm_api_key(self):
        from QuantNodes.core.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            settings.llm.api_key = "sk-real-key"
            data = settings.to_dict()
            assert data['llm']['api_key'] == '***'


class TestGetSettings:
    def test_returns_singleton(self):
        from QuantNodes.core.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cached_result(self):
        from QuantNodes.core.config import get_settings
        result = get_settings()
        assert result is not None
