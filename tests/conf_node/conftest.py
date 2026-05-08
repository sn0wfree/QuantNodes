# coding=utf-8
"""Conf node test fixtures"""
import pytest


@pytest.fixture
def temp_yaml_file(tmp_path):
    """Temporary YAML config file"""
    content = """
database:
  host: localhost
  port: 5432
  user: admin
  password: secret

cache:
  type: redis
  ttl: 3600
"""
    filepath = tmp_path / "config.yaml"
    filepath.write_text(content, encoding="utf-8")
    return filepath


@pytest.fixture
def temp_json_file(tmp_path):
    """Temporary JSON config file"""
    content = '{"database": {"host": "localhost", "port": 5432}, "cache": {"type": "redis"}}'
    filepath = tmp_path / "config.json"
    filepath.write_text(content, encoding="utf-8")
    return filepath


@pytest.fixture
def temp_ini_file(tmp_path):
    """Temporary INI config file"""
    content = """[database]
host = localhost
port = 5432
user = admin

[cache]
type = redis
ttl = 3600
"""
    filepath = tmp_path / "config.ini"
    filepath.write_text(content, encoding="utf-8")
    return filepath
