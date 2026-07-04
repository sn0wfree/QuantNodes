"""research/codegen — LLM-driven code generation primitives (vendor).

来源: llmwikify v1.3.0 kernel/codegen/ + 部分 kernel/agent/。

依赖: polars + QuantNodes (third-party)。

包含:
  - code_tools.py: extract_python / validate_syntax / validate_safety /
                   build_execute_namespace / execute_code / _PYTHON_FENCE_RE
  - prompts.py:    SYSTEM_PROMPT_CODE
  - json_extract.py: extract_json_from_response(text)
  - feedback_templates.py: OBSERVE_FEEDBACK_TEMPLATE
  - unified_hook.py: UnifiedHook (from kernel/agent/hook.py)
  - agent/: vendored kernel/agent/ 子包（generate_factor_code_sync 等）
"""
from .code_tools import (
    _PYTHON_FENCE_RE,
    build_execute_namespace,
    execute_code,
    extract_python,
    validate_safety,
    validate_syntax,
)
from .feedback_templates import OBSERVE_FEEDBACK_TEMPLATE
from .json_extract import extract_json_from_response
from .prompts import SYSTEM_PROMPT_CODE
from .unified_hook import UnifiedHook

__all__ = [
    "extract_python",
    "validate_syntax",
    "validate_safety",
    "build_execute_namespace",
    "execute_code",
    "_PYTHON_FENCE_RE",
    "OBSERVE_FEEDBACK_TEMPLATE",
    "extract_json_from_response",
    "SYSTEM_PROMPT_CODE",
    "UnifiedHook",
]
