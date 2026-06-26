# coding=utf-8
"""
AI 模块

提供 AI 生成和优化 Pipeline 的功能。
"""

from QuantNodes.ai.llm import (
    LLMClientBase,
    LLMError,
    RateLimitError,
    AuthenticationError,
    APIError,
    Message,
    MessageRole,
    ChatCompletion,
    ChatCompletionChunk,
    OpenAIClient,
    AzureOpenAIClient,
    LLMGateway,
    ToolCallResponse,
    get_llm_gateway,
    create_llm_gateway,
    reset_llm_gateway,
)

from QuantNodes.ai.prompts import (
    PromptLibrary,
    PromptTemplate,
    PromptBuilder,
)

from QuantNodes.ai.sandbox import (
    CodeSandbox,
    CodeValidationResult,
    DangerousCodeError,
)

from QuantNodes.ai.strategy_gen import (
    StrategyGenerator,
    GenerationResult,
    NaturalLanguageToPipeline,
)

from QuantNodes.ai.optimizer import (
    PipelineOptimizer,
    OptimizationResult,
    PipelineAnalyzer,
    NodeAnalysis,
)

__all__ = [
    # LLM
    'LLMClientBase',
    'LLMError',
    'RateLimitError',
    'AuthenticationError',
    'APIError',
    'Message',
    'MessageRole',
    'ChatCompletion',
    'ChatCompletionChunk',
    'OpenAIClient',
    'AzureOpenAIClient',

    # Gateway
    'LLMGateway',
    'ToolCallResponse',
    'get_llm_gateway',
    'create_llm_gateway',
    'reset_llm_gateway',

    # Prompts
    'PromptLibrary',
    'PromptTemplate',
    'PromptBuilder',

    # Sandbox
    'CodeSandbox',
    'CodeValidationResult',
    'DangerousCodeError',

    # Strategy Generation
    'StrategyGenerator',
    'GenerationResult',
    'NaturalLanguageToPipeline',

    # Optimization
    'PipelineOptimizer',
    'OptimizationResult',
    'PipelineAnalyzer',
    'NodeAnalysis',
]
