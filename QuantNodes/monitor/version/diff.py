# coding=utf-8
"""YAML配置差异对比"""

from __future__ import annotations

import difflib
from typing import Dict, Any, List, Tuple


class ConfigDiffer:
    """YAML配置差异对比"""

    def diff_configs_text(self, config1_text: str, config2_text: str) -> List[str]:
        """对比两个YAML文本的差异

        Returns:
            unified diff 行列表
        """
        lines1 = config1_text.splitlines(keepends=True)
        lines2 = config2_text.splitlines(keepends=True)
        return list(difflib.unified_diff(
            lines1, lines2,
            fromfile="version_a", tofile="version_b",
            lineterm="",
        ))

    def diff_configs(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """对比两个配置字典的差异

        Returns:
            结构化差异: {"added": [...], "removed": [...], "changed": [...]}
        """
        added = []
        removed = []
        changed = []

        all_keys = set(list(config1.keys()) + list(config2.keys()))
        for key in sorted(all_keys):
            if key not in config1:
                added.append({"key": "data." + key, "value": config2[key]})
            elif key not in config2:
                removed.append({"key": "data." + key, "value": config1[key]})
            elif config1[key] != config2[key]:
                changed.append({
                    "key": "data." + key,
                    "old": config1[key],
                    "new": config2[key],
                })

        return {"added": added, "removed": removed, "changed": changed}

    def format_diff(self, diff_lines: List[str]) -> str:
        """格式化差异为可读文本"""
        if not diff_lines:
            return "无差异"
        return "\n".join(diff_lines)

    def validate_rollback_safe(self, diff_result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查回滚是否安全

        不安全条件:
        - 删除了关键配置 (data.source, data.table)
        - 修改了数据源

        Returns:
            (是否安全, 风险说明列表)
        """
        risks = []
        for item in diff_result.get("removed", []):
            key = item["key"]
            if key in ("data.source", "data.table", "data.conn_ini"):
                risks.append(f"删除了关键配置: {key}")

        for item in diff_result.get("changed", []):
            key = item["key"]
            if key in ("data.source", "data.table"):
                risks.append(f"修改了数据源配置: {key} ({item['old']} → {item['new']})")

        return (len(risks) == 0, risks)
