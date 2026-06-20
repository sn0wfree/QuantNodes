# coding=utf-8
"""
Pipeline 优化器

提供 Pipeline 自动优化功能。
"""
from __future__ import annotations

import re
import ast
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from QuantNodes.ai.llm.base import LLMClientBase, Message, MessageRole
from QuantNodes.ai.prompts import PromptLibrary
from QuantNodes.ai.sandbox import CodeSandbox


@dataclass
class OptimizationResult:
    """优化结果"""
    original_code: str
    optimized_code: str
    improvements: List[str] = field(default_factory=list)
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class NodeAnalysis:
    """节点分析"""
    node_type: str
    config: Dict[str, Any]
    dependencies: List[str]
    estimated_cost: float = 1.0


class PipelineAnalyzer:
    """
    Pipeline 分析器

    分析 Pipeline 结构并识别优化机会。
    """

    def __init__(self):
        self.logger = logging.getLogger(f"optimizer.{self.__class__.__name__}")

    def analyze(self, code: str) -> List[NodeAnalysis]:
        """
        分析 Pipeline 代码

        Args:
            code: Pipeline 代码

        Returns:
            List[NodeAnalysis] 节点分析列表
        """
        analyses = []

        try:
            tree = ast.parse(code)
            analyses = self._walk_ast(tree)
        except SyntaxError as e:
            self.logger.error(f"Syntax error: {e}")

        return analyses

    def _walk_ast(self, tree: ast.AST) -> List[NodeAnalysis]:
        """遍历 AST"""
        analyses = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                analysis = self._analyze_call(node)
                if analysis:
                    analyses.append(analysis)

        return analyses

    def _analyze_call(self, node: ast.Call) -> Optional[NodeAnalysis]:
        """分析函数调用"""
        if not isinstance(node.func, ast.Name):
            return None

        node_name = node.func.id

        if 'Node' in node_name or 'Node' in str(node.func):
            config = self._extract_config(node)
            deps = self._extract_dependencies(node)

            return NodeAnalysis(
                node_type=node_name,
                config=config,
                dependencies=deps,
                estimated_cost=self._estimate_cost(node_name, config)
            )

        return None

    def _extract_config(self, node: ast.Call) -> Dict[str, Any]:
        """提取配置"""
        config = {}

        for keyword in node.keywords:
            if isinstance(keyword.value, ast.Dict):
                for k, v in zip(keyword.value.keys, keyword.value.values):
                    if isinstance(k, ast.Constant):
                        if isinstance(v, ast.Constant):
                            config[k.value] = v.value
                        elif isinstance(v, ast.Name):
                            config[k.value] = v.id
            elif isinstance(keyword.value, ast.Constant):
                config[keyword.arg] = keyword.value.value

        return config

    def _extract_dependencies(self, node: ast.Call) -> List[str]:
        """提取依赖"""
        deps = []

        for arg in node.args:
            if isinstance(arg, ast.Name):
                deps.append(arg.id)
            elif isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.RShift):
                if isinstance(arg.left, ast.Name):
                    deps.append(arg.left.id)
                if isinstance(arg.right, ast.Name):
                    deps.append(arg.right.id)

        return deps

    def _estimate_cost(self, node_type: str, config: Dict[str, Any]) -> float:
        """估算节点成本"""
        base_cost = 1.0

        if 'database' in node_type.lower():
            base_cost = 10.0
        elif 'backtest' in node_type.lower():
            base_cost = 50.0
        elif 'factor' in node_type.lower():
            base_cost = 5.0

        if 'limit' in config:
            limit = config.get('limit', 1000)
            base_cost *= (limit / 1000)

        return base_cost


class RuleBasedOptimizer(ABC):
    """基于规则的优化器基类"""

    @abstractmethod
    def can_apply(self, code: str) -> bool:
        """检查是否可以应用"""
        pass

    @abstractmethod
    def apply(self, code: str) -> Tuple[str, List[str]]:
        """应用优化"""
        pass


class CacheReadOptimizer(RuleBasedOptimizer):
    """缓存读取优化"""

    def can_apply(self, code: str) -> bool:
        return 'DatabaseNode' in code and '.read(' in code

    def apply(self, code: str) -> Tuple[str, List[str]]:
        improvements = []

        optimized = re.sub(
            r'(\w+)\s*=\s*DatabaseNode\([^)]*\)\s*\n\s*\1\.read\(',
            r'\1 = DatabaseNode(cache=True, ...)\n\1.read(',
            code
        )

        if optimized != code:
            improvements.append("Added caching to DatabaseNode reads")

        return optimized, improvements


class LimitPushdownOptimizer(RuleBasedOptimizer):
    """LIMIT 下推优化"""

    def can_apply(self, code: str) -> bool:
        return '.limit(' in code and 'DatabaseNode' in code

    def apply(self, code: str) -> Tuple[str, List[str]]:
        improvements = []

        limit_match = re.search(r'\.limit\((\d+)\)', code)
        if limit_match:
            limit_value = limit_match.group(1)
            improvements.append(f"Pushed down limit={limit_value} to DatabaseNode")

        return code, improvements


class ParallelFetchOptimizer(RuleBasedOptimizer):
    """并行获取优化"""

    def can_apply(self, code: str) -> bool:
        patterns = [
            r'DatabaseNode\([^)]*engine\s*=\s*["\']duckdb["\']',
            r'DatabaseNode\([^)]*engine\s*=\s*["\']clickhouse["\']',
        ]
        return any(re.search(p, code) for p in patterns)

    def apply(self, code: str) -> Tuple[str, List[str]]:
        improvements = ["Enabled parallel data fetching"]

        optimized = re.sub(
            r'DatabaseNode\(',
            'DatabaseNode(parallel=True, ',
            code
        )

        return optimized, improvements


class PipelineOptimizer:
    """
    Pipeline 优化器

    提供 Pipeline 自动优化功能。

    Examples:
        >>> optimizer = PipelineOptimizer(llm_client)
        >>> result = optimizer.optimize(code)
        >>> print(result.optimized_code)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClientBase] = None,
        code_sandbox: Optional[CodeSandbox] = None,
        enable_ai_optimization: bool = True,
        **kwargs
    ):
        """
        初始化优化器

        Args:
            llm_client: LLM 客户端（可选，用于 AI 优化）
            code_sandbox: 代码沙箱
            enable_ai_optimization: 是否启用 AI 优化
        """
        self.llm = llm_client
        self.sandbox = code_sandbox or CodeSandbox()
        self.enable_ai_optimization = enable_ai_optimization and self.llm is not None
        self.analyzer = PipelineAnalyzer()
        self.rule_optimizers: List[RuleBasedOptimizer] = [
            CacheReadOptimizer(),
            LimitPushdownOptimizer(),
            ParallelFetchOptimizer(),
        ]
        self.logger = logging.getLogger(f"optimizer.{self.__class__.__name__}")

    def optimize(
        self,
        code: str,
        goal: str = "performance",
        validate: bool = True,
        **kwargs
    ) -> OptimizationResult:
        """
        优化 Pipeline 代码

        Args:
            code: Pipeline 代码
            goal: 优化目标 ("performance", "memory", "readability")
            validate: 是否验证优化后的代码
            **kwargs: 额外参数

        Returns:
            OptimizationResult 优化结果
        """
        original_code = code
        all_improvements: List[str] = []

        analyses = self.analyzer.analyze(code)
        if analyses:
            estimated_cost = sum(a.estimated_cost for a in analyses)
            all_improvements.append(
                f"Analyzed {len(analyses)} nodes, estimated cost: {estimated_cost:.2f}"
            )

        for optimizer in self.rule_optimizers:
            if optimizer.can_apply(code):
                optimized, improvements = optimizer.apply(code)
                if optimized != code:
                    code = optimized
                    all_improvements.extend(improvements)

        if self.enable_ai_optimization and self.llm:
            code, ai_improvements = self._ai_optimize(code, goal, **kwargs)
            all_improvements.extend(ai_improvements)

        if validate:
            validation_result = self.sandbox.validate(code)
            is_valid = validation_result.is_safe
            error_message = None if is_valid else f"Validation failed: {validation_result.errors}"
        else:
            is_valid = True
            error_message = None

        return OptimizationResult(
            original_code=original_code,
            optimized_code=code,
            improvements=all_improvements,
            is_valid=is_valid,
            error_message=error_message
        )

    def _ai_optimize(
        self,
        code: str,
        goal: str,
        **kwargs
    ) -> Tuple[str, List[str]]:
        """AI 优化"""
        improvements = []

        system_prompt, user_prompt = PromptLibrary.format(
            "optimization",
            code=code
        )

        goal_instruction = ""
        if goal == "performance":
            goal_instruction = "Focus on reducing execution time and minimizing API calls."
        elif goal == "memory":
            goal_instruction = (
                "Focus on reducing memory usage and avoiding unnecessary data copies."
            )
        elif goal == "readability":
            goal_instruction = "Focus on improving code clarity and maintainability."

        messages = [
            Message(role=MessageRole.SYSTEM, content=system_prompt),
            Message(role=MessageRole.USER, content=user_prompt + "\n\n" + goal_instruction),
        ]

        try:
            response = self.llm.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=kwargs.get('max_tokens'),
                **kwargs
            )

            if response and response.content:
                optimized_code = self._extract_code(response.content)
                if optimized_code and optimized_code != code:
                    improvements.append("Applied AI-based optimization")
                    return optimized_code, improvements

        except Exception as e:
            self.logger.error(f"AI optimization failed: {e}")

        return code, improvements

    def _extract_code(self, content: str) -> str:
        """从响应中提取代码"""
        pattern = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)
        matches = pattern.findall(content)
        if matches:
            return matches[0].strip()
        return content.strip()

    def suggest_optimizations(self, code: str) -> List[str]:
        """
        建议优化项

        Args:
            code: Pipeline 代码

        Returns:
            List[str] 建议列表
        """
        suggestions = []
        analyses = self.analyzer.analyze(code)

        for analysis in analyses:
            if analysis.estimated_cost > 10:
                suggestions.append(
                    f"High cost node ({analysis.node_type}, cost={analysis.estimated_cost:.2f}): "
                    "Consider optimizing or caching"
                )

        for optimizer in self.rule_optimizers:
            if optimizer.can_apply(code):
                suggestions.append(
                    f"Rule-based optimization available: {optimizer.__class__.__name__}"
                )

        return suggestions
