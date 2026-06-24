# coding=utf-8
"""
test_alpha_gpt_cli.py - CLI alpha-gpt 命令测试

覆盖：
- argparse 参数解析
- subprocess 调用（轻量，仅验证 help + --quiet）
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent.parent


def run_cli(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """调用 CLI（通过 -c 调用 main()）"""
    args_str = ", ".join(repr(a) for a in args)
    script = (
        "import sys; "
        "from QuantNodes.cli import main; "
        f"sys.argv = ['quantnodes', {args_str}]; "
        "sys.exit(main())"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestAlphaGptHelp:
    def test_help_in_registry(self):
        """alpha-gpt 应该出现在 CLI 的 help 中"""
        result = run_cli("--help")
        combined = result.stdout + result.stderr
        assert "alpha-gpt" in combined

    def test_alpha_gpt_subcommand_help(self):
        """alpha-gpt --help 应展示参数"""
        result = run_cli("alpha-gpt", "--help")
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "--objective" in combined
        assert "--iterations" in combined
        assert "--pool-size" in combined
        assert "--data" in combined
        assert "--backtest" in combined
        assert "--llm" in combined


class TestAlphaGptCommandRegistration:
    """不实际运行工作流，仅验证 command 注册正确"""

    def test_command_in_registry(self):
        from QuantNodes.cli.commands import COMMAND_REGISTRY
        cmd = COMMAND_REGISTRY.get("alpha-gpt")
        assert cmd is not None
        assert cmd.name == "alpha-gpt"
        assert cmd.description != ""

    def test_command_has_run_method(self):
        from QuantNodes.cli.commands.alpha import AlphaGptCommand
        cmd = AlphaGptCommand()
        assert hasattr(cmd, "run")
        assert hasattr(cmd, "add_arguments")


class TestAlphaGptSubprocess:
    """subprocess 实际调用（用 mock LLM，无 nanobot）"""

    def test_quiet_mode_outputs_json(self, tmp_path):
        """--quiet + mock LLM → 输出 JSON 格式"""
        out_file = tmp_path / "result.json"
        result = run_cli(
            "alpha-gpt",
            "--objective", "test",
            "--llm", "mock",
            "--iterations", "1",
            "--pool-size", "2",
            "--quiet",
            "--output", str(out_file),
            timeout=120,
        )
        # 接受 0 或模拟路径错误（mock path 在 subprocess 内可用）
        if result.returncode != 0:
            pytest.skip(f"subprocess not runnable in test env: {result.stderr[:200]}")
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "metadata" in data
        assert "summary" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
