# -*- coding: utf-8 -*-
"""QuantNodes.ai.sandbox 单元测试"""

from QuantNodes.ai.sandbox import CodeSandbox, CodeValidationResult


class TestCodeSandboxInit:
    def test_default_init(self):
        sandbox = CodeSandbox()
        assert isinstance(sandbox.DANGEROUS_IMPORTS, set)
        assert len(sandbox.DANGEROUS_IMPORTS) > 0
        assert isinstance(sandbox.DANGEROUS_PATTERNS, list)
        assert len(sandbox.DANGEROUS_PATTERNS) > 0


class TestCodeSandboxValidateSafe:
    def test_simple_arithmetic(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("x = 1 + 2")
        assert result.is_safe is True
        assert len(result.errors) == 0

    def test_import_polars(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("import polars as pl")
        assert result.is_safe is True

    def test_polars_dataframe_creation(self):
        sandbox = CodeSandbox()
        code = """
import polars as pl
df = pl.DataFrame({"a": [1, 2, 3]})
result = df.select(pl.col("a") * 2)
"""
        result = sandbox.validate(code)
        assert result.is_safe is True

    def test_function_definition(self):
        sandbox = CodeSandbox()
        code = """
def calculate(x, y):
    return x + y
result = calculate(1, 2)
"""
        result = sandbox.validate(code)
        assert result.is_safe is True


class TestCodeSandboxValidateDangerous:
    def test_os_system(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("import os\nos.system('ls')")
        assert result.is_safe is False
        assert len(result.errors) > 0

    def test_eval(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("eval('1 + 2')")
        assert result.is_safe is False

    def test_exec(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("exec('print(1)')")
        assert result.is_safe is False

    def test_subprocess(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("import subprocess\nsubprocess.run(['ls'])")
        assert result.is_safe is False

    def test_open_builtin(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("open('/etc/passwd')")
        assert result.is_safe is False

    def test_import_socket(self):
        sandbox = CodeSandbox()
        result = sandbox.validate("import socket")
        assert result.is_safe is False


class TestCodeValidationResult:
    def test_dataclass_fields(self):
        result = CodeValidationResult(is_safe=True)
        assert result.is_safe is True
        assert result.errors == []
        assert result.warnings == []
        assert result.warnings_only is False

    def test_dataclass_with_errors(self):
        result = CodeValidationResult(is_safe=False, errors=["error1", "error2"])
        assert result.is_safe is False
        assert len(result.errors) == 2
