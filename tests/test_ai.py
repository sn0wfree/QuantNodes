# coding=utf-8
"""
AI 模块单元测试

测试 LLM、代码沙箱、策略生成和优化功能。
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import numpy as np
import pandas as pd

from QuantNodes.ai.llm.base import (
    LLMClientBase,
    LLMError,
    RateLimitError,
    AuthenticationError,
    APIError,
    Message,
    MessageRole,
    ChatCompletion,
    ChatCompletionChunk,
)
from QuantNodes.ai.llm.openai import OpenAIClient
from QuantNodes.ai.prompts import PromptLibrary, PromptTemplate, PromptBuilder
from QuantNodes.ai.sandbox import CodeSandbox, CodeValidationResult, DangerousCodeError
from QuantNodes.ai.strategy_gen import StrategyGenerator, GenerationResult
from QuantNodes.ai.optimizer import PipelineOptimizer, OptimizationResult, PipelineAnalyzer


class MockLLMClient(LLMClientBase):
    """模拟 LLM 客户端"""

    def __init__(self, response_content: str = "", **kwargs):
        super().__init__(**kwargs)
        self.response_content = response_content
        self.call_count = 0

    def _call_api(self, messages, model=None, **kwargs):
        self.call_count += 1
        return ChatCompletion(content=self.response_content)


class TestLLMBase(unittest.TestCase):
    """LLM 基类测试"""

    def test_message_creation(self):
        """测试消息创建"""
        msg = Message(role=MessageRole.USER, content="Hello")
        self.assertEqual(msg.role, MessageRole.USER)
        self.assertEqual(msg.content, "Hello")

    def test_chat_completion(self):
        """测试聊天补全"""
        completion = ChatCompletion(content="Test response")
        self.assertEqual(completion.content, "Test response")
        self.assertEqual(completion.role, MessageRole.ASSISTANT)

    def test_message_normalization(self):
        """测试消息规范化"""
        client = MockLLMClient()

        dict_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        normalized = client._normalize_messages(dict_messages)

        self.assertEqual(len(normalized), 2)
        self.assertIsInstance(normalized[0], Message)
        self.assertEqual(normalized[0].role, MessageRole.USER)

    def test_token_counting(self):
        """测试 token 计数"""
        client = MockLLMClient()
        text = "Hello World"
        count = client.count_tokens(text)
        self.assertGreaterEqual(count, 1)


class TestOpenAIClient(unittest.TestCase):
    """OpenAI 客户端测试"""

    def test_client_creation(self):
        """测试客户端创建"""
        client = OpenAIClient(api_key="test-key")
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.model, "gpt-4")

    def test_headers(self):
        """测试请求头"""
        client = OpenAIClient(api_key="test-key")
        headers = client._get_headers()
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["Content-Type"], "application/json")

    def test_message_to_dict(self):
        """测试消息转换"""
        client = OpenAIClient(api_key="test-key")
        msg = Message(role=MessageRole.USER, content="Hello", name="test")
        d = client._message_to_dict(msg)
        self.assertEqual(d["role"], "user")
        self.assertEqual(d["content"], "Hello")
        self.assertEqual(d["name"], "test")


class TestPromptLibrary(unittest.TestCase):
    """提示词库测试"""

    def test_get_template(self):
        """测试获取模板"""
        template = PromptLibrary.get("strategy_generation")
        self.assertIsInstance(template, PromptTemplate)
        self.assertIn("QuantNodes", template.system)

    def test_list_templates(self):
        """测试列出模板"""
        templates = PromptLibrary.list_templates()
        self.assertIn("strategy_generation", templates)
        self.assertIn("code_review", templates)
        self.assertIn("optimization", templates)

    def test_format_template(self):
        """测试格式化模板"""
        system, user = PromptLibrary.format(
            "strategy_generation",
            trading_description="Create a momentum strategy"
        )
        self.assertIn("QuantNodes", system)
        self.assertIn("momentum strategy", user)

    def test_register_template(self):
        """测试注册模板"""
        new_template = PromptTemplate(
            system="Test system",
            user="Test user",
            description="Test template"
        )
        PromptLibrary.register("test_template", new_template)
        retrieved = PromptLibrary.get("test_template")
        self.assertEqual(retrieved.system, "Test system")

    def test_get_system_prompt(self):
        """测试获取系统提示词"""
        system_prompt = PromptLibrary.get_system_prompt()
        self.assertIn("QuantNodes", system_prompt)


class TestPromptBuilder(unittest.TestCase):
    """提示词构建器测试"""

    def test_builder_add_system(self):
        """测试添加系统提示"""
        builder = PromptBuilder()
        builder.add_system("System message")
        system, user = builder.build()
        self.assertIn("System message", system)

    def test_builder_add_user(self):
        """测试添加用户提示"""
        builder = PromptBuilder()
        builder.add_user("User message")
        system, user = builder.build()
        self.assertIn("User message", user)

    def test_builder_full(self):
        """测试完整构建"""
        builder = PromptBuilder()
        builder.add_system("System").add_user("User").add_example("Q", "A")
        system, user = builder.build()
        self.assertIn("System", system)
        self.assertIn("User", user)
        self.assertIn("Q", user)
        self.assertIn("A", user)


class TestCodeSandbox(unittest.TestCase):
    """代码沙箱测试"""

    def setUp(self):
        self.sandbox = CodeSandbox()

    def test_safe_code(self):
        """测试安全代码"""
        code = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})"
        result = self.sandbox.validate(code)
        self.assertTrue(result.is_safe)

    def test_dangerous_import(self):
        """测试危险导入"""
        code = "import os\nos.system('ls')"
        result = self.sandbox.validate(code)
        self.assertFalse(result.is_safe)
        self.assertTrue(any("os" in err.lower() or "dangerous" in err.lower() for err in result.errors))

    def test_dangerous_pattern(self):
        """测试危险模式"""
        code = "eval('print(1)')"
        result = self.sandbox.validate(code)
        self.assertFalse(result.is_safe)

    def test_exec_pattern(self):
        """测试 exec 模式"""
        code = "exec('print(1)')"
        result = self.sandbox.validate(code)
        self.assertFalse(result.is_safe)

    def test_safe_quantnodes_code(self):
        """测试安全的 QuantNodes 代码"""
        code = """
from QuantNodes.factor_node import factor_functions as ff

result = ff.rolling_mean("close", 20)
"""
        result = self.sandbox.validate(code)
        self.assertTrue(result.is_safe)

    def test_extract_imports(self):
        """测试提取导入"""
        code = """
import pandas as pd
import numpy as np
from QuantNodes.core import BaseNode
"""
        imports = self.sandbox.extract_imports(code)
        self.assertIn("pandas", imports['standard'])
        self.assertIn("numpy", imports['standard'])

    def test_extract_quantnodes_usage(self):
        """测试提取 QuantNodes 使用"""
        code = """
from QuantNodes.factor_node import factor_functions as ff
from QuantNodes.backtest import BacktestNode

node = BacktestNode()
result = ff.rolling_mean("close", 20)
"""
        usage = self.sandbox.extract_quantnodes_usage(code)
        usage_types = [u.split(":")[0] for u in usage]
        self.assertIn("module", usage_types)
        self.assertIn("node_class", usage_types)

    def test_validate_and_execute_safe(self):
        """测试验证并执行"""
        code = """
import pandas as pd
result = pd.DataFrame({'a': [1, 2, 3]})
"""
        context = {'pd': __import__('pandas')}
        result = self.sandbox.validate_and_execute(code, context)
        self.assertIn('result', result)


class TestStrategyGenerator(unittest.TestCase):
    """策略生成器测试"""

    def setUp(self):
        self.mock_llm = MockLLMClient(response_content="```python\nimport pandas as pd\nprint('test')\n```")
        self.generator = StrategyGenerator(self.mock_llm)

    def test_generate_extracts_code(self):
        """测试生成提取代码"""
        result = self.generator.generate("Create a strategy", validate=False)
        self.assertIn("print", result.code)
        self.assertTrue(result.is_valid)

    def test_generate_empty_response(self):
        """测试空响应"""
        mock_llm = MockLLMClient(response_content="")
        generator = StrategyGenerator(mock_llm)
        result = generator.generate("Test")
        self.assertFalse(result.is_valid)

    def test_extract_code_from_markdown(self):
        """测试从 markdown 提取代码"""
        content = """
Here is the code:

```python
import pandas as pd
df = pd.DataFrame()
```

Hope this helps!
"""
        code = self.generator._extract_code(content)
        self.assertIn("import pandas", code)
        self.assertIn("pd.DataFrame", code)

    def test_validation_result(self):
        """测试验证结果"""
        result = self.generator.generate("Test", validate=True)
        self.assertIsNotNone(result.validation_result)


class TestPipelineAnalyzer(unittest.TestCase):
    """Pipeline 分析器测试"""

    def test_analyze_simple_code(self):
        """测试分析简单代码"""
        analyzer = PipelineAnalyzer()
        code = """
from QuantNodes.factor_node import factor_functions as ff

factor = ff.rolling_mean("close", 20)
"""
        analyses = analyzer.analyze(code)
        self.assertIsInstance(analyses, list)

    def test_analyze_syntax_error(self):
        """测试分析语法错误"""
        analyzer = PipelineAnalyzer()
        code = "this is not valid python (:"
        analyses = analyzer.analyze(code)
        self.assertEqual(len(analyses), 0)


class TestPipelineOptimizer(unittest.TestCase):
    """Pipeline 优化器测试"""

    def test_optimizer_creation(self):
        """测试优化器创建"""
        optimizer = PipelineOptimizer()
        self.assertIsNotNone(optimizer.analyzer)

    def test_optimize_without_llm(self):
        """测试无 LLM 优化"""
        optimizer = PipelineOptimizer(enable_ai_optimization=False)
        code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
"""
        result = optimizer.optimize(code, validate=False)
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.original_code, code)

    def test_suggest_optimizations(self):
        """测试建议优化"""
        optimizer = PipelineOptimizer()
        code = "DatabaseNode()"
        suggestions = optimizer.suggest_optimizations(code)
        self.assertIsInstance(suggestions, list)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_pipeline_imports(self):
        """测试完整导入"""
        from QuantNodes.ai import (
            LLMClientBase,
            OpenAIClient,
            PromptLibrary,
            CodeSandbox,
            StrategyGenerator,
            PipelineOptimizer,
        )
        self.assertIsNotNone(LLMClientBase)
        self.assertIsNotNone(OpenAIClient)
        self.assertIsNotNone(PromptLibrary)
        self.assertIsNotNone(CodeSandbox)
        self.assertIsNotNone(StrategyGenerator)
        self.assertIsNotNone(PipelineOptimizer)

    def test_mock_llm_integration(self):
        """测试模拟 LLM 集成"""
        mock_response = """
```python
from QuantNodes.core import BaseNode

class MyNode(BaseNode):
    pass
```
"""
        llm = MockLLMClient(response_content=mock_response)
        generator = StrategyGenerator(llm)

        result = generator.generate("Create a node", validate=True)
        self.assertTrue(result.is_valid)
        self.assertIn("BaseNode", result.code)


if __name__ == '__main__':
    unittest.main()