# coding=utf-8
"""Tests for PR-QN-4: sandbox pandas bridge (auto-detect + context injection)."""
from __future__ import annotations

import pandas as pd

from QuantNodes.ai.sandbox import CodeSandbox
from QuantNodes.ai.sandbox_pandas_bridge import (
    detect_engine_from_code,
    detect_and_inject_context,
    patch_sandbox_with_bridge,
)


class TestDetectEngineFromCode:
    def test_polars_import(self):
        code = "import polars as pl"
        assert detect_engine_from_code(code) == "polars"

    def test_pandas_import(self):
        code = "import pandas as pd"
        assert detect_engine_from_code(code) == "pandas"

    def test_polars_from_import(self):
        code = "from polars import col"
        assert detect_engine_from_code(code) == "polars"

    def test_pandas_from_import(self):
        code = "from pandas import DataFrame"
        assert detect_engine_from_code(code) == "pandas"

    def test_mixed_prefers_polars(self):
        code = "import polars as pl\nimport pandas as pd"
        assert detect_engine_from_code(code) == "polars"

    def test_empty_returns_polars(self):
        assert detect_engine_from_code("") == "polars"

    def test_syntax_error_returns_polars(self):
        assert detect_engine_from_code("def (") == "polars"


class TestDetectAndInjectContext:
    def test_polars_injects_pl(self):
        code = "import polars as pl"
        ctx = detect_and_inject_context(code)
        assert "pl" in ctx
        assert ctx["__engine__"] == "polars"

    def test_pandas_injects_pd(self):
        code = "import pandas as pd"
        ctx = detect_and_inject_context(code)
        assert "pd" in ctx
        assert ctx["__engine__"] == "pandas"

    def test_injects_quantnodes(self):
        code = "import polars as pl"
        ctx = detect_and_inject_context(code)
        assert "QuantNodes" in ctx

    def test_df_injection_polars(self):
        import polars as pl
        df = pl.DataFrame({"a": [1, 2]})
        code = "import polars as pl"
        ctx = detect_and_inject_context(code, df=df)
        assert ctx["df"] is df  # same object

    def test_df_injection_pandas(self):
        df = pd.DataFrame({"a": [1, 2]})
        code = "import pandas as pd"
        ctx = detect_and_inject_context(code, df=df)
        assert isinstance(ctx["df"], pd.DataFrame)

    def test_df_auto_convert_polars_to_pandas(self):
        import polars as pl
        df = pl.DataFrame({"a": [1, 2]})
        code = "import pandas as pd"
        ctx = detect_and_inject_context(code, df=df)
        assert isinstance(ctx["df"], pd.DataFrame)
        assert list(ctx["df"].columns) == ["a"]

    def test_extra_context(self):
        code = "import polars as pl"
        ctx = detect_and_inject_context(code, foo="bar", x=42)
        assert ctx["foo"] == "bar"
        assert ctx["x"] == 42

    def test_both_libs_always_injected(self):
        code = "import polars as pl"
        ctx = detect_and_inject_context(code)
        assert "pl" in ctx
        assert "pd" in ctx


class TestSandboxDefaultEngine:
    def test_default_engine_polars(self):
        sb = CodeSandbox()
        assert sb.default_engine == "polars"

    def test_default_engine_pandas(self):
        sb = CodeSandbox(default_engine="pandas")
        assert sb.default_engine == "pandas"

    def test_default_engine_auto(self):
        sb = CodeSandbox(default_engine="auto")
        assert sb.default_engine == "auto"


class TestSandboxDetectEngine:
    def test_detect_polars(self):
        sb = CodeSandbox(default_engine="auto")
        code = "import polars as pl\nx = pl.col('a')"
        assert sb._detect_engine(code) == "polars"

    def test_detect_pandas(self):
        sb = CodeSandbox(default_engine="auto")
        code = "import pandas as pd\ndf = pd.DataFrame()"
        assert sb._detect_engine(code) == "pandas"

    def test_default_engine_not_auto(self):
        sb = CodeSandbox(default_engine="polars")
        code = "import pandas as pd"
        assert sb._detect_engine(code) == "polars"  # uses default, not detect


class TestSandboxExecuteEngine:
    def test_execute_pandas_code_returns_engine(self):
        sb = CodeSandbox(default_engine="auto")
        code = "import pandas as pd\nresult = pd.DataFrame({'a': [1]})"
        result = sb.validate_and_execute(code)
        assert result["__engine__"] == "pandas"

    def test_execute_polars_code_returns_engine(self):
        sb = CodeSandbox(default_engine="auto")
        code = "import polars as pl\nresult = pl.DataFrame({'a': [1]})"
        result = sb.validate_and_execute(code)
        assert result["__engine__"] == "polars"

    def test_execute_with_explicit_engine(self):
        sb = CodeSandbox(default_engine="polars")
        code = "x = 1"
        result = sb.validate_and_execute(code, engine="pandas")
        assert result["__engine__"] == "pandas"


class TestPatchSandboxWithBridge:
    def test_patch_adds_bridge(self):
        sb = CodeSandbox()
        patch_sandbox_with_bridge(sb, df=pd.DataFrame({"a": [1]}))
        assert hasattr(sb, "_original_validate") or callable(sb.validate_and_execute)

    def test_patched_execute_injects_context(self):
        sb = CodeSandbox()
        df = pd.DataFrame({"a": [1, 2, 3]})
        patch_sandbox_with_bridge(sb, df=df)
        code = "import pandas as pd\nresult = df['a'].mean()"
        result = sb.validate_and_execute(code)
        assert "df" in result
        assert result["df"]["a"].mean() == 2.0
