# coding=utf-8
"""
提示词模板库

提供用于生成和优化 Pipeline 的提示词模板。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """提示词模板"""
    system: str
    user: str
    description: str = ""

    def format(self, **kwargs) -> tuple[str, str]:
        """格式化模板"""
        return (
            self.system.format(**kwargs),
            self.user.format(**kwargs)
        )


SYSTEM_PROMPT = """You are a quantitative trading strategy expert specializing in designing and implementing trading systems using the QuantNodes framework.

QuantNodes Framework Overview:
- BaseNode: Unified node base class with execute() method
- Pipeline: Chain nodes together using >> operator
- DatabaseNode: Query data from various databases (SQLite, DuckDB, MySQL, ClickHouse)
- factor_functions: 140+ Polars-based operators (rolling_mean, zscore, rank, etc.)
- BacktestNode: Run backtesting simulations
- OperatorNode: SQL operations and data transformations

Key Conventions:
- All nodes inherit from BaseNode
- Use >> to chain nodes in a pipeline
- Configuration passed via config dict in constructor
- Input/output data is typically pandas DataFrame
- Factor computation uses factor_functions (ff) module

Your task is to generate valid Python code that creates QuantNodes pipelines based on user requests.
"""

STRATEGY_GENERATION_PROMPT = PromptTemplate(
    system=SYSTEM_PROMPT,
    user="""Generate a QuantNodes trading strategy pipeline based on the following description:

{trading_description}

Requirements:
- Use Python code
- Create appropriate nodes (DatabaseNode, BacktestNode, etc.) and use factor_functions for factor computation
- Use the >> operator to chain nodes
- Include necessary configuration

Available Data:
The database contains stock OHLCV data with columns: date, code, open, high, low, close, volume

Generate the Python code:""",
    description="Generate trading strategy pipeline from natural language"
)

CODE_REVIEW_PROMPT = PromptTemplate(
    system=SYSTEM_PROMPT,
    user="""Review the following QuantNodes pipeline code for issues:

```python
{code}
```

Check for:
1. Correct node usage and configuration
2. Proper pipeline chaining
3. Data flow correctness
4. Performance concerns
5. Potential bugs

Provide your review:""",
    description="Review generated code for issues"
)

OPTIMIZATION_PROMPT = PromptTemplate(
    system=SYSTEM_PROMPT,
    user="""Optimize the following QuantNodes pipeline for better performance:

```python
{code}
```

Optimization goals:
- Reduce execution time
- Minimize memory usage
- Improve readability

Provide the optimized code:""",
    description="Optimize pipeline for performance"
)

FACTOR_EXPLANATION_PROMPT = PromptTemplate(
    system="You are a financial factor research expert.",
    user="""Explain the following trading factor in detail:

{factor_description}

Include:
1. What the factor measures
2. How to calculate it
3. Practical usage tips
4. Potential issues or limitations

Provide a comprehensive explanation:""",
    description="Explain trading factors"
)

BACKTEST_CONFIG_PROMPT = PromptTemplate(
    system=SYSTEM_PROMPT,
    user="""Suggest appropriate backtest configuration for the following strategy:

```python
{strategy_code}
```

Consider:
- Initial capital
- Commission rates
- Position sizing
- Risk management

Provide configuration dict:""",
    description="Suggest backtest configuration"
)


class PromptLibrary:
    """
    提示词模板库

    提供各种任务场景的提示词模板。
    """

    _templates: Dict[str, PromptTemplate] = {
        "strategy_generation": STRATEGY_GENERATION_PROMPT,
        "code_review": CODE_REVIEW_PROMPT,
        "optimization": OPTIMIZATION_PROMPT,
        "factor_explanation": FACTOR_EXPLANATION_PROMPT,
        "backtest_config": BACKTEST_CONFIG_PROMPT,
    }

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        """获取模板"""
        if name not in cls._templates:
            raise ValueError(f"Unknown template: {name}. Available: {list(cls._templates.keys())}")
        return cls._templates[name]

    @classmethod
    def list_templates(cls) -> List[str]:
        """列出所有模板"""
        return list(cls._templates.keys())

    @classmethod
    def register(cls, name: str, template: PromptTemplate) -> None:
        """注册新模板"""
        cls._templates[name] = template

    @classmethod
    def format(
        cls,
        name: str,
        **kwargs
    ) -> tuple[str, str]:
        """格式化模板"""
        template = cls.get(name)
        return template.format(**kwargs)

    @classmethod
    def get_system_prompt(cls) -> str:
        """获取系统提示词"""
        return SYSTEM_PROMPT


class PromptBuilder:
    """提示词构建器"""

    def __init__(self):
        self._parts: List[str] = []

    def add_system(self, text: str) -> 'PromptBuilder':
        """添加系统提示"""
        self._parts.append(f"[SYSTEM] {text}")
        return self

    def add_user(self, text: str) -> 'PromptBuilder':
        """添加用户提示"""
        self._parts.append(f"[USER] {text}")
        return self

    def add_assistant(self, text: str) -> 'PromptBuilder':
        """添加助手提示"""
        self._parts.append(f"[ASSISTANT] {text}")
        return self

    def add_example(self, user: str, assistant: str) -> 'PromptBuilder':
        """添加示例"""
        self._parts.append(f"[EXAMPLE]\nUser: {user}\nAssistant: {assistant}")
        return self

    def build(self) -> tuple[str, str]:
        """构建最终提示词"""
        system_parts = []
        user_parts = []

        for part in self._parts:
            if part.startswith("[SYSTEM]"):
                system_parts.append(part[9:])
            elif part.startswith("[USER]"):
                user_parts.append(part[7:])
            elif part.startswith("[ASSISTANT]"):
                system_parts.append(f"Assistant: {part[11:]}")
            elif part.startswith("[EXAMPLE]"):
                user_parts.append(part[10:])

        return "\n".join(system_parts), "\n".join(user_parts)