# coding: utf-8
"""P-3: CLI factor-data-fetch --universe 默认值改为 'all' (替代 '沪深300')"""
from __future__ import annotations

import sys
from unittest.mock import patch



class TestFetchUniverseDefault:
    """P-3: --universe 默认 'all', 与 iFinD API 兼容"""

    def _parse_args(self, argv):
        """Helper: 通过 main() 解析 argv"""
        from QuantNodes.cli import main
        backup_argv = sys.argv
        sys.argv = ["quantnodes"] + argv
        try:
            with patch.object(sys, "argv", sys.argv):
                try:
                    main()
                except SystemExit:
                    pass
        finally:
            sys.argv = backup_argv

    def test_default_universe_is_all(self, capsys):
        """默认 universe='all' (不再 '沪深300')"""
        self._parse_args([
            "factor-data-fetch", "--output-dir", "/tmp/test/",
            "--date-beg", "20260101",
        ])
        captured = capsys.readouterr()
        # 输出应包含 "universe: all" (P-3 默认)
        assert "universe: all" in captured.out
        # 不应包含 "universe: 沪深300"
        assert "universe: 沪深300" not in captured.out

    def test_custom_universe_accepted(self, capsys):
        """--universe 沪深300 显式覆盖"""
        self._parse_args([
            "factor-data-fetch", "--output-dir", "/tmp/test/",
            "--date-beg", "20260101", "--universe", "沪深300",
        ])
        captured = capsys.readouterr()
        assert "universe: 沪深300" in captured.out
