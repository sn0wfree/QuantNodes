# coding=utf-8
"""
Pipeline Method

validate_pipeline(code) -> PipelineValidationResult

Validates QuantNodes Pipeline code for external agents.
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    nodes: List[str] = field(default_factory=list)
    security_status: str = "unknown"


CODE_BLOCK_PATTERN = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)


def validate_pipeline(code: str, **kwargs) -> PipelineValidationResult:
    """Validate QuantNodes Pipeline code.

    Checks:
    - Syntax validity
    - Security (via CodeSandbox)
    - Node extraction

    Args:
        code: QuantNodes Pipeline Python代码

    Returns:
        PipelineValidationResult with validation status and extracted nodes
    """
    result = PipelineValidationResult(is_valid=True)

    extracted_code = _extract_code(code)
    if not extracted_code:
        result.is_valid = False
        result.errors.append("No valid code found")
        return result

    try:
        compile(extracted_code, '<string>', 'exec')
    except SyntaxError as e:
        result.is_valid = False
        result.errors.append("Syntax error: %s" % str(e))
        return result

    try:
        from QuantNodes.ai.sandbox import CodeSandbox

        sandbox = CodeSandbox()
        validation = sandbox.validate(extracted_code)

        result.security_status = "safe" if validation.is_safe else "unsafe"

        if not validation.is_safe:
            result.is_valid = False
            result.errors.extend(validation.errors)

        if validation.warnings:
            result.warnings.extend(validation.warnings)

    except ImportError:
        result.security_status = "skipped"
        result.warnings.append("CodeSandbox not available, security check skipped")

    result.nodes = _extract_nodes(extracted_code)

    return result


def _extract_code(code: str) -> str:
    """Extract code from markdown code blocks if present."""
    match = CODE_BLOCK_PATTERN.search(code)
    if match:
        return match.group(1).strip()
    return code.strip()


def _extract_nodes(code: str) -> List[str]:
    """Extract node types from code."""
    nodes = []
    patterns = [
        r'(\w+Node)\s*\(',
        r'FactorPipeline\s*\(',
        r'Pipeline\s*\(',
        r'BacktestPipeline\s*\(',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, code)
        nodes.extend(matches)
    return list(set(nodes))
