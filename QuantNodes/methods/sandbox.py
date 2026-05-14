# coding=utf-8
"""
Sandbox Method

validate_code(code) -> ValidationResult
execute_code(code, **kwargs) -> ExecutionResult

Provides code security validation for external agents.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ValidationResult:
    is_safe: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    warnings_only: bool = False


@dataclass
class ExecutionResult:
    status: str
    result: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_code(
    code: str,
    allow_warnings: bool = False,
    max_code_length: int = 10000,
    **kwargs
) -> ValidationResult:
    """Validate Python code安全性.

    Args:
        code: 待验证的Python代码
        allow_warnings: 是否允许警告
        max_code_length: 最大代码长度

    Returns:
        ValidationResult with is_safe, errors, warnings
    """
    from QuantNodes.ai.sandbox import CodeSandbox

    sandbox = CodeSandbox(
        allow_warnings=allow_warnings,
        max_code_length=max_code_length
    )

    result = sandbox.validate(code)

    return ValidationResult(
        is_safe=result.is_safe,
        errors=result.errors,
        warnings=result.warnings,
        warnings_only=result.warnings_only
    )


def execute_code(
    code: str,
    context: Dict[str, Any] = None,
    **kwargs
) -> ExecutionResult:
    """Execute code in sandbox environment.

    Args:
        code: 待执行的Python代码
        context: 执行上下文（可选）

    Returns:
        ExecutionResult with status, result, errors
    """
    from QuantNodes.ai.sandbox import CodeSandbox

    sandbox = CodeSandbox()
    validation = sandbox.validate(code)

    if not validation.is_safe:
        return ExecutionResult(
            status="error",
            errors=validation.errors,
            warnings=validation.warnings
        )

    try:
        namespace = sandbox.validate_and_execute(code, context or {})
        return ExecutionResult(
            status="success",
            result={k: v for k, v in namespace.items() if not k.startswith('_')},
            warnings=validation.warnings
        )
    except Exception as e:
        return ExecutionResult(
            status="error",
            errors=[str(e)],
            warnings=validation.warnings
        )