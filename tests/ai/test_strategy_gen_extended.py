# coding=utf-8
"""Tests for ai/strategy_gen.py — StrategyGenerator and code extraction.

Covers: creation, code extraction, generation (mocked LLM), result structure.
"""

from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.ai.strategy_gen import StrategyGenerator, GenerationResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_gateway():
    gw = MagicMock()
    gw.chat = MagicMock(return_value=MagicMock(
        content="```python\ndef strategy(data):\n    return data\n```"
    ))
    return gw


@pytest.fixture
def generator(mock_gateway):
    return StrategyGenerator(llm_client=mock_gateway)


# ============================================================================
# Creation
# ============================================================================

class TestStrategyGeneratorCreation:
    def test_creation_with_client(self, mock_gateway):
        gen = StrategyGenerator(llm_client=mock_gateway)
        assert gen is not None

    def test_creation_default_params(self, mock_gateway):
        gen = StrategyGenerator(llm_client=mock_gateway)
        assert gen is not None

    def test_creation_with_custom_temperature(self, mock_gateway):
        gen = StrategyGenerator(llm_client=mock_gateway, temperature=0.3)
        assert gen is not None


# ============================================================================
# Code Extraction
# ============================================================================

class TestCodeExtraction:
    def test_extract_from_code_block(self, generator):
        content = "Here is the code:\n```python\ndef hello():\n    pass\n```\nDone."
        code = generator._extract_code(content)
        assert "def hello" in code

    def test_extract_no_code_block(self, generator):
        content = "def standalone():\n    return 42"
        code = generator._extract_code(content)
        assert "def standalone" in code

    def test_extract_empty_content(self, generator):
        code = generator._extract_code("")
        assert code == ""

    def test_extract_multiline_code_block(self, generator):
        content = """
Some text before.

```python
import pandas as pd

def strategy(data):
    data["signal"] = 0
    return data
```

Some text after.
"""
        code = generator._extract_code(content)
        assert "import pandas" in code
        assert "def strategy" in code

    def test_extract_plain_code_without_block(self, generator):
        content = "x = 1\ny = 2\nreturn x + y"
        code = generator._extract_code(content)
        assert "x = 1" in code


# ============================================================================
# Generation (Mocked LLM)
# ============================================================================

class TestStrategyGeneration:
    def test_generate_basic(self, generator):
        result = generator.generate("Create a momentum strategy")
        assert isinstance(result, GenerationResult)
        assert result.code is not None

    def test_generate_returns_code(self, generator):
        result = generator.generate("simple strategy")
        assert "def" in result.code or "strategy" in result.code.lower()

    def test_generate_with_context(self, generator):
        result = generator.generate(
            "strategy",
            context={"data_columns": ["close", "volume"]},
        )
        assert isinstance(result, GenerationResult)

    def test_generate_handles_llm_error(self, mock_gateway):
        mock_gateway.chat = MagicMock(side_effect=Exception("LLM unavailable"))
        gen = StrategyGenerator(llm_client=mock_gateway)
        result = gen.generate("strategy")
        assert result.is_valid is False
        assert result.error_message is not None

    def test_generate_empty_response(self, mock_gateway):
        mock_gateway.chat = MagicMock(return_value=MagicMock(content=""))
        gen = StrategyGenerator(llm_client=mock_gateway)
        result = gen.generate("strategy")
        # Empty response should produce empty or invalid code
        assert result.code == "" or result.is_valid is False


# ============================================================================
# GenerationResult
# ============================================================================

class TestGenerationResult:
    def test_result_fields(self):
        result = GenerationResult(
            code="def f(): pass",
            is_valid=True,
            validation_result=None,
            error_message=None,
            warnings=[],
        )
        assert result.code == "def f(): pass"
        assert result.is_valid is True
        assert result.warnings == []

    def test_result_with_errors(self):
        result = GenerationResult(
            code="",
            is_valid=False,
            validation_result=None,
            error_message="Syntax error",
            warnings=["warning 1"],
        )
        assert result.is_valid is False
        assert result.error_message == "Syntax error"


# ============================================================================
# Code Block Pattern
# ============================================================================

class TestCodeBlockPattern:
    def test_pattern_matches_python_block(self, generator):
        import re
        text = "```python\nprint('hello')\n```"
        match = re.search(generator.CODE_BLOCK_PATTERN, text)
        assert match is not None

    def test_pattern_matches_generic_block(self, generator):
        import re
        text = "```\nprint('hello')\n```"
        match = re.search(generator.CODE_BLOCK_PATTERN, text)
        assert match is not None
