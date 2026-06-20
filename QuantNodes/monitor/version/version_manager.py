# coding=utf-8
"""基于Git的策略版本管理"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from ..storage.models import StrategyVersion
from ..storage.repository import VersionRepository
from .diff import ConfigDiffer


class VersionManager:
    """基于Git的策略版本管理

    每个策略对应Git仓库中的一个目录:
    ~/.quantnodes/strategies/{strategy_name}/
    """

    def __init__(
        self,
        version_repo: VersionRepository,
        strategies_dir: str = "~/.quantnodes/strategies",
    ):
        self.repo = version_repo
        self.strategies_dir = Path(strategies_dir).expanduser()
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        self.differ = ConfigDiffer()
        self._init_repo()

    def _init_repo(self):
        """初始化Git仓库 (如不存在)"""
        git_dir = self.strategies_dir / ".git"
        if not git_dir.exists():
            import subprocess
            subprocess.run(
                ["git", "init"],
                cwd=str(self.strategies_dir),
                capture_output=True,
                check=False,
            )
            # 配置 git user
            subprocess.run(
                ["git", "config", "user.email", "quantnodes@local"],
                cwd=str(self.strategies_dir),
                capture_output=True,
                check=False,
            )
            subprocess.run(
                ["git", "config", "user.name", "QuantNodes"],
                cwd=str(self.strategies_dir),
                capture_output=True,
                check=False,
            )
            # 创建 .gitignore
            gitignore = self.strategies_dir / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text("__pycache__/\n*.pyc\n")

    def save_version(
        self,
        strategy_name: str,
        config_path: str,
        description: str = "",
    ) -> StrategyVersion:
        """保存策略版本

        1. 复制YAML到 strategies/{name}/
        2. git add + commit
        3. 保存到数据库
        """
        strategy_dir = self.strategies_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # 确定版本号
        existing = self.repo.list_versions(strategy_name)
        version_num = len(existing) + 1
        version_str = f"v{version_num}"

        # 复制YAML配置
        src = Path(config_path).expanduser()
        dst = strategy_dir / f"{strategy_name}_{version_str}.yaml"
        shutil.copy2(str(src), str(dst))

        # Git commit
        commit_msg = f"version {version_str}: {description or strategy_name}"
        commit_hash = self._git_commit(strategy_dir, commit_msg)

        # 同时保存为 current
        current_dst = strategy_dir / f"{strategy_name}_current.yaml"
        shutil.copy2(str(src), str(current_dst))
        self._git_commit(strategy_dir, f"update current: {strategy_name}")

        # 保存到数据库
        config_content = src.read_text(encoding="utf-8")
        sv = StrategyVersion(
            strategy_name=strategy_name,
            version=version_str,
            commit_hash=commit_hash,
            config_snapshot=config_content,
            description=description,
        )
        sv.id = self.repo.save_version(sv)
        return sv

    def list_versions(self, strategy_name: str) -> List[StrategyVersion]:
        """列出策略所有版本"""
        return self.repo.list_versions(strategy_name)

    def get_version(self, strategy_name: str, version: str) -> Optional[StrategyVersion]:
        """获取指定版本的配置"""
        return self.repo.get_version(strategy_name, version)

    def diff_versions(self, strategy_name: str, v1: str, v2: str) -> str:
        """对比两个版本的差异"""
        ver1 = self.repo.get_version(strategy_name, v1)
        ver2 = self.repo.get_version(strategy_name, v2)
        if not ver1 or not ver2:
            return f"版本不存在: {v1} 或 {v2}"
        return self.differ.format_diff(
            self.differ.diff_configs_text(ver1.config_snapshot, ver2.config_snapshot)
        )

    def rollback(self, strategy_name: str, target_version: str) -> Optional[StrategyVersion]:
        """回滚到指定版本"""
        target = self.repo.get_version(strategy_name, target_version)
        if not target:
            return None

        # 写入 current.yaml
        strategy_dir = self.strategies_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        current_path = strategy_dir / f"{strategy_name}_current.yaml"
        current_path.write_text(target.config_snapshot, encoding="utf-8")

        # Git commit
        self._git_commit(strategy_dir, f"rollback to {target_version}")

        # 创建新版本记录
        new_ver_num = len(self.repo.list_versions(strategy_name)) + 1
        new_version = f"v{new_ver_num}"
        sv = StrategyVersion(
            strategy_name=strategy_name,
            version=new_version,
            commit_hash=self._git_head(strategy_dir),
            config_snapshot=target.config_snapshot,
            description=f"rollback to {target_version}",
        )
        sv.id = self.repo.save_version(sv)
        return sv

    def get_current_version(self, strategy_name: str) -> Optional[str]:
        """获取当前最新版本号"""
        latest = self.repo.get_latest(strategy_name)
        return latest.version if latest else None

    def _git_commit(self, repo_dir: Path, message: str) -> str:
        """执行 git add + commit，返回 commit hash"""
        import subprocess

        subprocess.run(
            ["git", "add", "-A"],
            cwd=str(repo_dir), capture_output=True, check=False,
        )
        subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=str(repo_dir), capture_output=True, check=False,
        )
        return self._git_head(repo_dir)

    @staticmethod
    def _git_head(repo_dir: Path) -> str:
        """获取当前 HEAD commit hash"""
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir), capture_output=True, text=True, check=False,
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
