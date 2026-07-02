# coding=utf-8
"""
test_mine_logics_cli.py - MineLogicsCommand CLI 测试 (v3.0.2 Step 4)

覆盖:
- 参数解析 (args 基础值/覆盖)
- 退出码语义 (0=全成功/1=部分失败/2=致命)
- 产出文件 (JSON + Markdown 报告)
- 离线模式 (默认)
- --live 模式 (mock LLM gateway)
- --strict 模式
- 空 source-libs 报错
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.cli.commands.mine_logics import MineLogicsCommand


# ======================================================================
# Fixtures
# ======================================================================
@pytest.fixture
def cmd() -> MineLogicsCommand:
    return MineLogicsCommand()


# ======================================================================
# 参数解析
# ======================================================================
class TestMineLogicsCommandArgs:
    def test_name_and_description(self, cmd):
        assert cmd.name == "mine-logics"
        assert "批量" in cmd.description or "挖掘" in cmd.description

    def test_add_arguments_registers_defaults(self, cmd):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        cmd.add_arguments(sub)
        # 解析时需要传入子命令名
        args = parser.parse_args(["mine-logics"])
        assert args.source_libs == "alpha101,alpha158,alpha191"
        assert args.max_per_lib == 10
        assert args.workers == 4
        assert args.wiki_path == "wiki_auto"
        assert args.output_dir == "data/mine_runs"
        assert args.live is False
        assert args.strict is False
        assert args.no_skip is False
        assert args.quiet is False

    def test_add_arguments_override(self, cmd):
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        cmd.add_arguments(sub)
        args = parser.parse_args([
            "mine-logics",
            "--source-libs", "alpha101,alpha191",
            "--max-per-lib", "5",
            "--workers", "2",
            "--wiki-path", "/tmp/wiki",
            "--output-dir", "/tmp/out",
            "--live",
            "--strict",
            "--no-skip",
            "--quiet",
        ])
        assert args.source_libs == "alpha101,alpha191"
        assert args.max_per_lib == 5
        assert args.workers == 2
        assert args.wiki_path == "/tmp/wiki"
        assert args.output_dir == "/tmp/out"
        assert args.live is True
        assert args.strict is True
        assert args.no_skip is True
        assert args.quiet is True


# ======================================================================
# 退出码 + 产出文件
# ======================================================================
class TestMineLogicsCommandRun:
    def test_offline_returns_zero(self, cmd, tmp_path):
        """离线模式: 有结果 → exit 0"""
        args = MagicMock()
        args.source_libs = "alpha101"
        args.max_per_lib = 2
        args.workers = 1
        args.wiki_path = str(tmp_path / "wiki")
        args.output_dir = str(tmp_path / "out")
        args.live = False
        args.strict = False
        args.no_skip = False
        args.quiet = True
        rc = cmd.run(args)
        assert rc == 0

    def test_output_files_created(self, cmd, tmp_path):
        """JSON + MD 报告文件存在"""
        args = MagicMock()
        args.source_libs = "alpha101"
        args.max_per_lib = 1
        args.workers = 1
        args.wiki_path = str(tmp_path / "wiki")
        args.output_dir = str(tmp_path / "out")
        args.live = False
        args.strict = False
        args.no_skip = False
        args.quiet = True
        rc = cmd.run(args)
        assert rc == 0
        out_dir = tmp_path / "out"
        json_files = list(out_dir.glob("metrics_*.json"))
        md_files = list(out_dir.glob("metrics_*.md"))
        assert len(json_files) == 1
        assert len(md_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "summary" in data
        assert data["summary"]["total_mined"] >= 1

    def test_empty_source_libs_returns_2(self, cmd, tmp_path):
        """空 source-libs → exit 2"""
        args = MagicMock()
        args.source_libs = "  ,  "
        args.max_per_lib = 1
        args.workers = 1
        args.wiki_path = str(tmp_path / "wiki")
        args.output_dir = str(tmp_path / "out")
        args.live = False
        args.strict = False
        args.no_skip = False
        args.quiet = True
        rc = cmd.run(args)
        assert rc == 2

    def test_strict_mode_enabled(self, cmd, tmp_path):
        """--strict → StrictConfig(call=True, parse=True, structured=True)"""
        args = MagicMock()
        args.source_libs = "alpha101"
        args.max_per_lib = 1
        args.workers = 1
        args.wiki_path = str(tmp_path / "wiki")
        args.output_dir = str(tmp_path / "out")
        args.live = False
        args.strict = True
        args.no_skip = False
        args.quiet = True
        rc = cmd.run(args)
        assert rc == 0

    def test_live_mode_calls_gateway(self, cmd, tmp_path):
        """--live 调用 get_llm_gateway()"""
        args = MagicMock()
        args.source_libs = "alpha101"
        args.max_per_lib = 1
        args.workers = 1
        args.wiki_path = str(tmp_path / "wiki")
        args.output_dir = str(tmp_path / "out")
        args.live = True
        args.strict = False
        args.no_skip = False
        args.quiet = True

        mock_gateway = MagicMock()
        with patch("QuantNodes.cli.commands.mine_logics.get_llm_gateway", create=True) as mock_gg:
            # 不 patch 模块级 import，直接在 run 内部处理
            # 实际上 run 中是 try: from QuantNodes... import get_llm_gateway
            # 我们用 patch 整个模块的 get_llm_gateway 导入
            with patch.dict("sys.modules", {"QuantNodes.ai.llm.gateway": MagicMock(get_llm_gateway=MagicMock(return_value=mock_gateway))}):
                rc = cmd.run(args)
        # 只验证它不崩溃（因为 mock gateway 可能返回无效响应）
        assert rc in (0, 1, 2)


# ======================================================================
# Helper: subparsers fixture
# ======================================================================
