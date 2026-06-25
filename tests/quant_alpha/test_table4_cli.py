# coding=utf-8
"""
test_table4_cli.py - reproduce_table4_mock.py CLI 脚本测试
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
CLI = ROOT / "scripts" / "reproduce_table4_mock.py"


class TestCLIInvocation:
    def test_help(self):
        """--help 不抛错"""
        result = subprocess.run(
            [sys.executable, str(CLI), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Stage 1 mock Table 4" in result.stdout

    def test_quick_mode(self, tmp_path):
        """--quick 模式跑通"""
        result = subprocess.run(
            [
                sys.executable, str(CLI),
                "--quick",
                "--output-dir", str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Stage 1 Mock Table 4" in result.stdout
        assert (tmp_path / "out" / "table4_report.json").exists()
        assert (tmp_path / "out" / "table4_report.md").exists()

    def test_custom_n_stocks(self, tmp_path):
        """--n-stocks 自定义参数"""
        result = subprocess.run(
            [
                sys.executable, str(CLI),
                "--n-stocks", "10",
                "--n-days", "20",
                "--g1-n", "3",
                "--g2-n", "2",
                "--g3-n", "2",
                "--output-dir", str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "G1_Handcrafted" in result.stdout
        assert "G2_LlmOnly" in result.stdout
        assert "G3_AlphaGpt" in result.stdout

    def test_verbose_flag(self, tmp_path):
        """--verbose 不影响退出码"""
        result = subprocess.run(
            [
                sys.executable, str(CLI),
                "--n-stocks", "5",
                "--n-days", "10",
                "--g1-n", "2",
                "--g2-n", "2",
                "--g3-n", "1",
                "--output-dir", str(tmp_path / "out"),
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0

    def test_output_dir_created(self, tmp_path):
        """--output-dir 嵌套目录自动创建"""
        nested = tmp_path / "a" / "b" / "c" / "out"
        result = subprocess.run(
            [
                sys.executable, str(CLI),
                "--n-stocks", "5",
                "--n-days", "10",
                "--g1-n", "2",
                "--g2-n", "1",
                "--g3-n", "1",
                "--output-dir", str(nested),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        assert nested.exists()
        assert (nested / "table4_report.json").exists()


class TestCLIReportContent:
    def test_json_contains_groups(self, tmp_path):
        """生成的 JSON 包含 3 个 group"""
        result = subprocess.run(
            [
                sys.executable, str(CLI),
                "--n-stocks", "5",
                "--n-days", "10",
                "--g1-n", "2",
                "--g2-n", "1",
                "--g3-n", "1",
                "--output-dir", str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        import json
        with open(tmp_path / "out" / "table4_report.json") as f:
            data = json.load(f)
        assert "groups" in data
        assert len(data["groups"]) == 3
        group_names = {g["group"] for g in data["groups"]}
        assert group_names == {"G1_Handcrafted", "G2_LlmOnly", "G3_AlphaGpt"}
        assert data["stage"] == "mock"

    def test_markdown_has_summary(self, tmp_path):
        """生成的 Markdown 包含汇总表"""
        result = subprocess.run(
            [
                sys.executable, str(CLI),
                "--n-stocks", "5",
                "--n-days", "10",
                "--g1-n", "2",
                "--g2-n", "1",
                "--g3-n", "1",
                "--output-dir", str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0
        md = (tmp_path / "out" / "table4_report.md").read_text(encoding="utf-8")
        assert "# Table 4 复现报告" in md
        assert "## 汇总" in md
        assert "## 按 avg_IR 排名" in md