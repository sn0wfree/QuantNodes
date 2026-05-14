# coding=utf-8
"""
Code Router

API endpoints for code validation and execution.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from ..deps import verify_api_key
from QuantNodes.methods.sandbox import validate_code, execute_code
from QuantNodes.methods.pipeline import validate_pipeline


router = APIRouter()


class CodeValidateRequest(BaseModel):
    code: str
    allow_warnings: bool = False
    max_code_length: int = 10000


class CodeExecuteRequest(BaseModel):
    code: str
    context: Optional[Dict[str, Any]] = None


class PipelineValidateRequest(BaseModel):
    code: str


@router.post("/code/validate")
async def validate_code_endpoint(
    request: CodeValidateRequest,
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Validate Python code安全性.

    Args:
        request.code: 待验证的Python代码
        request.allow_warnings: 是否允许警告 (默认 False)
        request.max_code_length: 最大代码长度 (默认 10000)

    Returns:
        Validation result with is_safe, errors, warnings
    """
    result = validate_code(
        code=request.code,
        allow_warnings=request.allow_warnings,
        max_code_length=request.max_code_length,
    )

    return {
        "is_safe": result.is_safe,
        "errors": result.errors,
        "warnings": result.warnings,
        "warnings_only": result.warnings_only,
    }


@router.post("/code/execute")
async def execute_code_endpoint(
    request: CodeExecuteRequest,
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Execute code in sandbox environment.

    IMPORTANT: Code is validated before execution.

    Args:
        request.code: 待执行的Python代码
        request.context: 执行上下文 (可选)

    Returns:
        Execution result with status, result, errors
    """
    validation = validate_code(request.code)
    if not validation.is_safe:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Code validation failed",
                "errors": validation.errors,
                "warnings": validation.warnings,
            }
        )

    result = execute_code(code=request.code, context=request.context)

    return {
        "status": result.status,
        "result": result.result,
        "errors": result.errors,
        "warnings": result.warnings,
    }


@router.post("/pipeline/validate")
async def validate_pipeline_endpoint(
    request: PipelineValidateRequest,
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Validate QuantNodes Pipeline code.

    Args:
        request.code: Pipeline Python代码

    Returns:
        Validation result with is_valid, errors, nodes, security_status
    """
    result = validate_pipeline(code=request.code)

    return {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "nodes": result.nodes,
        "security_status": result.security_status,
    }