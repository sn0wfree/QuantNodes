# coding=utf-8
"""
Prompts Router

API endpoints for retrieving prompts.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from ..deps import verify_api_key
from QuantNodes.prompts.strategy import (
    MOMENTUM_PROMPT,
    MEAN_REVERSION_PROMPT,
    TREND_FOLLOWING_PROMPT,
    PAIRS_TRADING_PROMPT,
    MARKET_NEUTRAL_PROMPT,
)
from QuantNodes.prompts.backtest import (
    STANDARD_BACKTEST_PROMPT,
    FACTOR_BACKTEST_PROMPT,
)
from QuantNodes.prompts.factor import (
    IC_ANALYSIS_PROMPT,
    GROUP_BACKTEST_PROMPT,
    CORRELATION_PROMPT,
)


router = APIRouter()


STRATEGY_PROMPTS = {
    "momentum": MOMENTUM_PROMPT,
    "mean_reversion": MEAN_REVERSION_PROMPT,
    "trend_following": TREND_FOLLOWING_PROMPT,
    "pairs_trading": PAIRS_TRADING_PROMPT,
    "market_neutral": MARKET_NEUTRAL_PROMPT,
}

BACKTEST_PROMPTS = {
    "standard": STANDARD_BACKTEST_PROMPT,
    "factor_based": FACTOR_BACKTEST_PROMPT,
}

FACTOR_PROMPTS = {
    "ic_analysis": IC_ANALYSIS_PROMPT,
    "group_backtest": GROUP_BACKTEST_PROMPT,
    "correlation": CORRELATION_PROMPT,
}


def prompt_to_dict(prompt) -> Dict[str, Any]:
    """Convert a StrategyPrompt to dict for JSON response."""
    return {
        "version": prompt.version,
        "name": prompt.name,
        "description": prompt.description,
        "created_at": prompt.created_at,
        "updated_at": prompt.updated_at,
        "prompt": prompt.prompt,
        "required_params": prompt.required_params,
        "output_format": prompt.output_format,
        "validation_rules": prompt.validation_rules,
        "example_code": prompt.example_code,
    }


@router.get("/prompts/strategy/{prompt_type}")
async def get_strategy_prompt(
    prompt_type: str,
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get strategy prompt by type.

    Args:
        prompt_type: One of: momentum, mean_reversion, trend_following, pairs_trading, market_neutral

    Returns:
        Complete prompt with parameters and example code
    """
    if prompt_type not in STRATEGY_PROMPTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown prompt type: {prompt_type}. Available: {list(STRATEGY_PROMPTS.keys())}"
        )

    return prompt_to_dict(STRATEGY_PROMPTS[prompt_type])


@router.get("/prompts/strategy")
async def list_strategy_prompts(
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """List all available strategy prompts."""
    return {
        "prompts": {
            name: {
                "name": p.name,
                "description": p.description,
                "version": p.version,
                "required_params": p.required_params,
            }
            for name, p in STRATEGY_PROMPTS.items()
        }
    }


@router.get("/prompts/backtest/{prompt_type}")
async def get_backtest_prompt(
    prompt_type: str,
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get backtest prompt by type.

    Args:
        prompt_type: One of: standard, factor_based

    Returns:
        Complete prompt with parameters and example code
    """
    if prompt_type not in BACKTEST_PROMPTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown prompt type: {prompt_type}. Available: {list(BACKTEST_PROMPTS.keys())}"
        )

    return prompt_to_dict(BACKTEST_PROMPTS[prompt_type])


@router.get("/prompts/backtest")
async def list_backtest_prompts(
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """List all available backtest prompts."""
    return {
        "prompts": {
            name: {
                "name": p.name,
                "description": p.description,
                "version": p.version,
                "required_params": p.required_params,
            }
            for name, p in BACKTEST_PROMPTS.items()
        }
    }


@router.get("/prompts/factor/{prompt_type}")
async def get_factor_prompt(
    prompt_type: str,
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Get factor prompt by type.

    Args:
        prompt_type: One of: ic_analysis, group_backtest, correlation

    Returns:
        Complete prompt with parameters and example code
    """
    if prompt_type not in FACTOR_PROMPTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown prompt type: {prompt_type}. Available: {list(FACTOR_PROMPTS.keys())}"
        )

    return prompt_to_dict(FACTOR_PROMPTS[prompt_type])


@router.get("/prompts/factor")
async def list_factor_prompts(
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """List all available factor prompts."""
    return {
        "prompts": {
            name: {
                "name": p.name,
                "description": p.description,
                "version": p.version,
                "required_params": p.required_params,
            }
            for name, p in FACTOR_PROMPTS.items()
        }
    }


@router.get("/prompts")
async def list_all_prompts(
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """List all available prompts by category."""
    return {
        "strategy": list(STRATEGY_PROMPTS.keys()),
        "backtest": list(BACKTEST_PROMPTS.keys()),
        "factor": list(FACTOR_PROMPTS.keys()),
    }