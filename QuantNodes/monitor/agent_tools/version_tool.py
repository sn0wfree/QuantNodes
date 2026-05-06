# coding=utf-8
"""版本管理Agent工具"""

from __future__ import annotations

from typing import Any, Dict

from QuantNodes.agent.tools.base import Tool
from ..storage.repository import DatabaseManager, VersionRepository
from ..version.version_manager import VersionManager


class VersionTool(Tool):
    """策略版本管理工具"""

    def __init__(self, db_path: str = "~/.quantnodes/monitor.db"):
        self._db_path = db_path
        self._version_manager = None

    def _get_version_manager(self) -> VersionManager:
        if self._version_manager is None:
            db = DatabaseManager(self._db_path)
            db.connect()
            self._version_manager = VersionManager(VersionRepository(db))
        return self._version_manager

    @property
    def name(self) -> str:
        return "strategy_version"

    @property
    def description(self) -> str:
        return "管理策略版本，支持保存、查看历史、对比差异、回滚"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "list", "diff", "rollback", "current"],
                    "description": "操作类型",
                },
                "strategy_name": {
                    "type": "string",
                    "description": "策略名称",
                },
                "version": {
                    "type": "string",
                    "description": "版本号 (如 'v1', 'v2')",
                },
                "config_path": {
                    "type": "string",
                    "description": "YAML配置文件路径 (save操作)",
                },
                "description": {
                    "type": "string",
                    "description": "版本描述 (save操作)",
                },
            },
            "required": ["action", "strategy_name"],
        }

    @property
    def read_only(self) -> bool:
        return False

    async def execute(self, action: str, strategy_name: str,
                      version: str = None, config_path: str = None,
                      description: str = "", **kwargs) -> Any:
        vm = self._get_version_manager()

        if action == "save":
            if not config_path:
                return {"error": "config_path required for save"}
            sv = vm.save_version(strategy_name, config_path, description)
            return {
                "status": "saved",
                "version": sv.version,
                "commit_hash": sv.commit_hash,
            }

        elif action == "list":
            versions = vm.list_versions(strategy_name)
            return {
                "versions": [
                    {
                        "version": v.version,
                        "commit_hash": v.commit_hash,
                        "description": v.description,
                        "created_at": str(v.created_at) if v.created_at else None,
                    }
                    for v in versions
                ]
            }

        elif action == "diff":
            if not version:
                return {"error": "version required for diff (compare with previous)"}
            versions = vm.list_versions(strategy_name)
            if len(versions) < 2:
                return {"error": "need at least 2 versions to diff"}
            # 找到目标版本的前一个版本
            target_idx = next(
                (i for i, v in enumerate(versions) if v.version == version), None
            )
            if target_idx is None:
                return {"error": f"version {version} not found"}
            if target_idx >= len(versions) - 1:
                prev_version = versions[target_idx + 1].version
            else:
                prev_version = versions[0].version
            diff_text = vm.diff_versions(strategy_name, version, prev_version)
            return {"diff": diff_text}

        elif action == "rollback":
            if not version:
                return {"error": "version required for rollback"}
            sv = vm.rollback(strategy_name, version)
            if sv:
                return {
                    "status": "rolled_back",
                    "new_version": sv.version,
                    "commit_hash": sv.commit_hash,
                }
            return {"error": f"version {version} not found"}

        elif action == "current":
            ver = vm.get_current_version(strategy_name)
            return {"current_version": ver}

        return {"error": f"Unknown action: {action}"}
