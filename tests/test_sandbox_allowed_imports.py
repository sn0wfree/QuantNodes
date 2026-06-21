"""PR-QN-1: CodeSandbox 实例级可配置白/黑名单测试

锁定 PR-QN-1 (2026-06-21) 行为:
- 默认参数行为与 PR 之前完全一致 (向后兼容)
- allowed_imports 追加到白名单
- blocked_imports 追加到黑名单
- 支持正则通配符
"""
from __future__ import annotations

from QuantNodes.ai.sandbox import CodeSandbox


class TestSandboxDefaultUnchanged:
    """默认参数行为不变 (向后兼容)."""

    def test_default_blocks_os(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("import os")
        assert not result.is_safe
        assert any("Dangerous" in e or "os" in e for e in result.errors + result.warnings)

    def test_default_blocks_subprocess(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("import subprocess")
        assert not result.is_safe

    def test_default_blocks_pickle(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("import pickle")
        assert not result.is_safe

    def test_default_allows_pandas(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("import pandas")
        assert result.is_safe or not result.errors

    def test_default_allows_quantnodes(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("import QuantNodes")
        assert result.is_safe or not result.errors

    def test_instance_attrs_initialized(self):
        """默认参数应拷贝类级别到 self._allowed_patterns / self._blocked_imports."""
        sandbox = CodeSandbox()
        assert sandbox._allowed_patterns == list(CodeSandbox.ALLOWED_PATTERNS)
        assert sandbox._blocked_imports == set(CodeSandbox.DANGEROUS_IMPORTS)


class TestAllowedImportsExtra:
    """allowed_imports 参数追加白名单."""

    def test_extra_scipy_allowed(self):
        """allowed_imports=[r'^scipy\\..*'] 应允许 import scipy.stats."""
        sandbox = CodeSandbox(allowed_imports=[r"^scipy\..*"])
        result = sandbox.validate("import scipy.stats")
        assert result.is_safe, f"应通过, 实际 errors={result.errors}"

    def test_extra_scipy_blocks_other(self):
        """仅允许 scipy 不影响默认黑名单 (os 仍被禁)."""
        sandbox = CodeSandbox(allowed_imports=[r"^scipy\..*"])
        result = sandbox.validate("import os")
        assert not result.is_safe

    def test_wildcard_pattern(self):
        """支持通配符 (用户自定义包)."""
        sandbox = CodeSandbox(allowed_imports=[r"^my_pkg\..*"])
        result = sandbox.validate("from my_pkg.deep.module import x")
        assert result.is_safe, f"应通过, 实际 errors={result.errors}"

    def test_multiple_allowed(self):
        """多个 pattern 同时生效."""
        sandbox = CodeSandbox(allowed_imports=[
            r"^scipy\..*", r"^statsmodels\..*",
        ])
        r1 = sandbox.validate("import scipy.optimize")
        r2 = sandbox.validate("import statsmodels.api")
        assert r1.is_safe
        assert r2.is_safe

    def test_empty_list_noop(self):
        """空 list 等价于不传 (默认行为)."""
        sandbox = CodeSandbox(allowed_imports=[])
        result = sandbox.validate("import os")
        assert not result.is_safe  # os 默认黑名单, 应仍被禁


class TestBlockedImportsExtra:
    """blocked_imports 参数追加黑名单."""

    def test_extra_json_blocked(self):
        """json 默认允许, blocked_imports=['json'] 后被禁."""
        sandbox = CodeSandbox(blocked_imports=["json"])
        result = sandbox.validate("import json")
        assert not result.is_safe, "json 应被用户黑名单拦截"

    def test_default_dangerous_still_blocked(self):
        """追加黑名单不影响默认黑名单."""
        sandbox = CodeSandbox(blocked_imports=["json"])
        result = sandbox.validate("import os")
        assert not result.is_safe

    def test_unknown_module_not_blocked(self):
        """未列出的模块不受 blocked_imports 影响."""
        sandbox = CodeSandbox(blocked_imports=["json"])
        result = sandbox.validate("import pandas")
        assert result.is_safe


class TestWildcardAndCombined:
    """组合与边界用例."""

    def test_allowed_and_blocked_combined(self):
        """同时使用 allowed + blocked, 互不影响."""
        sandbox = CodeSandbox(
            allowed_imports=[r"^scipy\..*"],
            blocked_imports=["json"],
        )
        assert sandbox.validate("import scipy.stats").is_safe
        assert not sandbox.validate("import json").is_safe
        assert not sandbox.validate("import os").is_safe

    def test_pattern_extends_existing(self):
        """追加 pattern 不替换默认 — os 仍被禁."""
        sandbox = CodeSandbox(allowed_imports=[r"^scipy\..*"])
        result = sandbox.validate("import os")
        assert not result.is_safe
        result = sandbox.validate("import sys")
        assert not result.is_safe

    def test_existing_kwargs_compat(self):
        """旧 **kwargs 仍可接收 (向后兼容)."""
        sandbox = CodeSandbox(future_param="ignored")
        result = sandbox.validate("import pandas")
        assert result.is_safe or not result.errors
