# coding=utf-8
"""
策略生成器

提供从自然语言生成 QuantNodes Pipeline 的功能。
"""
from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from QuantNodes.ai.llm.base import LLMClientBase, Message, MessageRole
from QuantNodes.ai.prompts import PromptLibrary
from QuantNodes.ai.sandbox import CodeSandbox, CodeValidationResult


@dataclass
class GenerationResult:
    """生成结果"""
    code: str
    is_valid: bool
    validation_result: Optional[CodeValidationResult] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class StrategyGenerator:
    """
    策略生成器

    将自然语言描述转换为 QuantNodes Pipeline 代码。

    Examples:
        >>> generator = StrategyGenerator(llm_client)
        >>> result = generator.generate("Create a momentum strategy")
        >>> print(result.code)
    """

    CODE_BLOCK_PATTERN = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)

    def __init__(
        self,
        llm_client: LLMClientBase,
        code_sandbox: Optional[CodeSandbox] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        初始化策略生成器

        Args:
            llm_client: LLM 客户端
            code_sandbox: 代码沙箱（用于验证）
            temperature: 生成温度
            max_tokens: 最大 token 数
        """
        self.llm = llm_client
        self.sandbox = code_sandbox or CodeSandbox()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_config = kwargs
        self.logger = logging.getLogger(f"strategy.{self.__class__.__name__}")

    def generate(
        self,
        description: str,
        validate: bool = True,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> GenerationResult:
        """
        生成策略代码

        Args:
            description: 策略描述
            validate: 是否验证代码
            context: 额外上下文
            **kwargs: 额外参数

        Returns:
            GenerationResult 生成结果
        """
        try:
            system_prompt, user_prompt = PromptLibrary.format(
                "strategy_generation",
                trading_description=description
            )

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_prompt),
            ]

            response = self.llm.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )

            if not response:
                return GenerationResult(
                    code="",
                    is_valid=False,
                    error_message="No response from LLM"
                )

            code = self._extract_code(response.content)

            if not code:
                return GenerationResult(
                    code="",
                    is_valid=False,
                    error_message="No code found in response"
                )

            validation_result = None
            if validate:
                validation_result = self.sandbox.validate(code)
                if not validation_result.is_safe:
                    return GenerationResult(
                        code=code,
                        is_valid=False,
                        validation_result=validation_result,
                        error_message=f"Code validation failed: {validation_result.errors}"
                    )

            return GenerationResult(
                code=code,
                is_valid=True,
                validation_result=validation_result,
                warnings=validation_result.warnings if validation_result else []
            )

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return GenerationResult(
                code="",
                is_valid=False,
                error_message=str(e)
            )

    def _extract_code(self, content: str) -> str:
        """从响应中提取代码"""
        matches = self.CODE_BLOCK_PATTERN.findall(content)
        if matches:
            return matches[0].strip()

        if '```' in content:
            return ""

        lines = content.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            if line.startswith('```'):
                in_code = not in_code
                continue
            if in_code or (line.startswith('import ') or line.startswith('from ') or line.startswith('#')):
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines)

        return content.strip()

    def generate_with_refinement(
        self,
        description: str,
        max_iterations: int = 3,
        **kwargs
    ) -> GenerationResult:
        """
        带精化的生成（自动修复问题）

        Args:
            description: 策略描述
            max_iterations: 最大迭代次数
            **kwargs: 额外参数

        Returns:
            GenerationResult 生成结果
        """
        result = self.generate(description, validate=True, **kwargs)

        for iteration in range(max_iterations):
            if result.is_valid:
                return result

            if not result.validation_result:
                break

            if result.validation_result.warnings_only:
                refinement_prompt = f"Refine the following code to address warnings:\n\n{result.code}\n\nWarnings: {result.validation_result.warnings}"
                result = self._refine_code(refinement_prompt, **kwargs)
            else:
                refinement_prompt = f"Fix the following code errors:\n\n{result.code}\n\nErrors: {result.validation_result.errors}"
                result = self._refine_code(refinement_prompt, **kwargs)

        return result

    def _refine_code(self, refinement_prompt: str, **kwargs) -> GenerationResult:
        """精化代码"""
        messages = [
            Message(role=MessageRole.SYSTEM, content=PromptLibrary.get_system_prompt()),
            Message(role=MessageRole.USER, content=refinement_prompt),
        ]

        try:
            response = self.llm.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )

            if not response:
                return GenerationResult(
                    code="",
                    is_valid=False,
                    error_message="No response from LLM"
                )

            code = self._extract_code(response.content)
            validation_result = self.sandbox.validate(code)

            return GenerationResult(
                code=code,
                is_valid=validation_result.is_safe,
                validation_result=validation_result,
                error_message=None if validation_result.is_safe else f"Validation failed: {validation_result.errors}",
                warnings=validation_result.warnings
            )

        except Exception as e:
            return GenerationResult(
                code="",
                is_valid=False,
                error_message=str(e)
            )

    def review(
        self,
        code: str,
        **kwargs
    ) -> str:
        """
        审查代码

        Args:
            code: 待审查的代码
            **kwargs: 额外参数

        Returns:
            str 审查意见
        """
        messages = [
            Message(role=MessageRole.SYSTEM, content=PromptLibrary.get_system_prompt()),
            Message(role=MessageRole.USER, content=f"Review the following QuantNodes code:\n\n{code}"),
        ]

        try:
            response = self.llm.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=self.max_tokens,
                **kwargs
            )

            return response.content if response else "No response"

        except Exception as e:
            self.logger.error(f"Review failed: {e}")
            return f"Review failed: {e}"

    def explain_strategy(self, code: str, **kwargs) -> str:
        """
        解释策略代码

        Args:
            code: 策略代码
            **kwargs: 额外参数

        Returns:
            str 解释
        """
        prompt = f"Explain what this QuantNodes strategy does:\n\n{code}\n\nProvide a clear explanation of the pipeline and its components."

        messages = [
            Message(role=MessageRole.SYSTEM, content=PromptLibrary.get_system_prompt()),
            Message(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = self.llm.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=self.max_tokens,
                **kwargs
            )

            return response.content if response else "No response"

        except Exception as e:
            self.logger.error(f"Explanation failed: {e}")
            return f"Explanation failed: {e}"


class NaturalLanguageToPipeline:
    """
    自然语言转 Pipeline 工具

    提供更高级的 Pipeline 生成接口。
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        code_sandbox: Optional[CodeSandbox] = None,
        **kwargs
    ):
        """
        初始化转换器

        Args:
            llm_client: LLM 客户端
            code_sandbox: 代码沙箱
        """
        self.generator = StrategyGenerator(llm_client, code_sandbox, **kwargs)

    def convert(
        self,
        description: str,
        return_pipeline: bool = False,
        **kwargs
    ) -> Union[GenerationResult, Any]:
        """
        转换自然语言为 Pipeline

        Args:
            description: 策略描述
            return_pipeline: 是否返回可执行的 Pipeline 对象
            **kwargs: 额外参数

        Returns:
            GenerationResult 或 Pipeline 对象
        """
        result = self.generator.generate_with_refinement(description, **kwargs)

        if return_pipeline and result.is_valid:
            try:
                return self._execute_code(result.code)
            except Exception as e:
                result.warnings.append(f"Failed to execute code: {e}")
                return result

        return result

    def _execute_code(self, code: str):
        """执行代码并返回 Pipeline"""
        context = {
            'QuantNodes': __import__('QuantNodes'),
        }
        exec_globals = self.generator.sandbox.validate_and_execute(code, context)
        return exec_globals