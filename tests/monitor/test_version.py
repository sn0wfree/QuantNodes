# coding=utf-8
"""版本管理测试"""

import pytest
import tempfile
import os
from pathlib import Path

from QuantNodes.monitor.storage.repository import DatabaseManager, VersionRepository
from QuantNodes.monitor.version.version_manager import VersionManager
from QuantNodes.monitor.version.diff import ConfigDiffer


@pytest.fixture
def version_manager():
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    with tempfile.TemporaryDirectory() as strategies_dir:
        dm = DatabaseManager(db_path)
        dm.connect()
        vm = VersionManager(VersionRepository(dm), strategies_dir)
        yield vm
        dm.close()
    os.unlink(db_path)


class TestVersionManager:

    def test_save_version(self, version_manager, tmp_path):
        config = tmp_path / "test.yaml"
        config.write_text("name: test_strategy\nfactors:\n  - name: ret\n    expr: close / open - 1\n")

        sv = version_manager.save_version("test_strategy", str(config), "initial version")
        assert sv.version == "v1"
        assert sv.commit_hash is not None
        assert sv.config_snapshot.startswith("name:")

    def test_list_versions(self, version_manager, tmp_path):
        for i in range(3):
            config = tmp_path / f"test_{i}.yaml"
            config.write_text(f"name: test_{i}\nversion: {i}\n")
            version_manager.save_version("s1", str(config), f"version {i}")

        versions = version_manager.list_versions("s1")
        assert len(versions) == 3
        assert versions[0].version == "v3"  # 最新在前

    def test_get_current_version(self, version_manager, tmp_path):
        config = tmp_path / "test.yaml"
        config.write_text("name: test\n")
        version_manager.save_version("s1", str(config))
        version_manager.save_version("s1", str(config))

        ver = version_manager.get_current_version("s1")
        assert ver == "v2"

    def test_diff_versions(self, version_manager, tmp_path):
        config1 = tmp_path / "v1.yaml"
        config1.write_text("name: test\nfactors:\n  - name: a\n    expr: close\n")
        config2 = tmp_path / "v2.yaml"
        config2.write_text("name: test\nfactors:\n  - name: a\n    expr: close * 2\n")

        version_manager.save_version("s1", str(config1), "v1")
        version_manager.save_version("s1", str(config2), "v2")

        diff = version_manager.diff_versions("s1", "v1", "v2")
        assert diff != "无差异"


class TestConfigDiffer:

    def test_diff_configs_text(self):
        differ = ConfigDiffer()
        text1 = "name: test\nvalue: 1\n"
        text2 = "name: test\nvalue: 2\n"
        diff = differ.diff_configs_text(text1, text2)
        assert len(diff) > 0

    def test_diff_configs_text_no_diff(self):
        differ = ConfigDiffer()
        text = "name: test\nvalue: 1\n"
        diff = differ.diff_configs_text(text, text)
        assert len(diff) == 0

    def test_diff_configs_dict(self):
        differ = ConfigDiffer()
        d1 = {"name": "a", "value": 1}
        d2 = {"name": "a", "value": 2, "new_key": "x"}
        result = differ.diff_configs(d1, d2)
        assert len(result["changed"]) == 1
        assert len(result["added"]) == 1

    def test_format_diff_no_diff(self):
        differ = ConfigDiffer()
        assert differ.format_diff([]) == "无差异"

    def test_validate_rollback_safe(self):
        differ = ConfigDiffer()
        diff = {
            "added": [],
            "removed": [{"key": "data.source", "value": "csv"}],
            "changed": [],
        }
        safe, risks = differ.validate_rollback_safe(diff)
        assert not safe
        assert len(risks) == 1
