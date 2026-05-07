# -*- coding: utf-8 -*-
"""QuantNodes.ai.strategy_gen 单元测试"""

from QuantNodes.ai.strategy_gen import StrategyGenerator, GenerationResult
from QuantNodes.ai.sandbox import CodeSandbox


class TestStrategyGeneratorInit:
    def test_default_init(self, mock_llm_client):
        generator = StrategyGenerator(llm_client=mock_llm_client)
        assert generator.llm is mock_llm_client
        assert isinstance(generator.sandbox, CodeSandbox)
        assert generator.temperature == 0.7

    def test_custom_temperature(self, mock_llm_client):
        generator = StrategyGenerator(llm_client=mock_llm_client, temperature=0.5)
        assert generator.temperature == 0.5

    def test_custom_max_tokens(self, mock_llm_client):
        generator = StrategyGenerator(llm_client=mock_llm_client, max_tokens=1000)
        assert generator.max_tokens == 1000

    def test_custom_sandbox(self, mock_llm_client):
        sandbox = CodeSandbox()
        generator = StrategyGenerator(llm_client=mock_llm_client, code_sandbox=sandbox)
        assert generator.sandbox is sandbox


class TestGenerationResult:
    def test_dataclass_fields(self):
        result = GenerationResult(code="x = 1", is_valid=True)
        assert result.code == "x = 1"
        assert result.is_valid is True
        assert result.validation_result is None
        assert result.error_message is None
        assert result.warnings == []

    def test_dataclass_with_error(self):
        result = GenerationResult(
            code="dangerous",
            is_valid=False,
            error_message="Dangerous code",
        )
        assert result.is_valid is False
        assert "Dangerous" in result.error_message


class TestStrategyGeneratorCodeExtraction:
    def test_extract_code_from_markdown(self, mock_llm_client):
        generator = StrategyGenerator(llm_client=mock_llm_client)

        code = generator._extract_code("Some text\n```python\nx = 1\n```\nMore text")
        assert "x = 1" in code

    def test_extract_code_no_code_block(self, mock_llm_client):
        generator = StrategyGenerator(llm_client=mock_llm_client)

        code = generator._extract_code("No code here")
        assert "No code here" in code


class TestStrategyGeneratorValidation:
    def test_validation_safe_code(self, mock_llm_client):
        generator = StrategyGenerator(llm_client=mock_llm_client)

        safe_code = "x = 1 + 2"
        validation = generator.sandbox.validate(safe_code)
        assert validation.is_safe is True

    def test_validation_dangerous_code(self, mock_llm_client):
        generator = StrategyGenerator(llm_client=mock_llm_client)

        dangerous_code = "import os\nos.system('ls')"
        validation = generator.sandbox.validate(dangerous_code)
        assert validation.is_safe is False
        assert len(validation.errors) > 0
