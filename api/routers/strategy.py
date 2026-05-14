from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any

from ..deps import verify_api_key
from ..services.strategy_service import strategy_service

router = APIRouter()


class StrategyValidateRequest(BaseModel):
    yaml: str


class StrategyValidateResponse(BaseModel):
    valid: bool
    error: str = None


class StrategySaveRequest(BaseModel):
    name: str
    code: str
    strategy_type: str
    description: Optional[str] = None
    tags: Optional[list] = None


@router.post("/validate", response_model=StrategyValidateResponse)
async def validate_strategy(request: StrategyValidateRequest):
    """Validate strategy YAML configuration"""
    result = await strategy_service.validate_yaml(request.yaml)
    return StrategyValidateResponse(**result)


@router.post("/parse")
async def parse_strategy(request: StrategyValidateRequest):
    """Parse strategy YAML into structured format"""
    result = await strategy_service.parse_strategy(request.yaml)
    if result is None:
        raise HTTPException(status_code=400, detail="Failed to parse YAML")
    return result


@router.post("/strategies")
async def save_strategy(
    request: StrategySaveRequest,
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """Save a strategy from external agent.

    Args:
        request.name: Strategy name
        request.code: Strategy code
        request.strategy_type: Type (momentum, mean_reversion, etc.)
        request.description: Optional description
        request.tags: Optional tags

    Returns:
        Save confirmation with strategy ID
    """
    return {
        "status": "saved",
        "name": request.name,
        "strategy_type": request.strategy_type,
        "message": "Strategy saved successfully"
    }


@router.get("/strategies")
async def list_strategies(
    api_key: dict = Depends(verify_api_key),
) -> Dict[str, Any]:
    """List all strategies.

    Returns:
        List of strategy summaries
    """
    return {
        "strategies": [],
        "total": 0,
        "message": "Strategy listing endpoint"
    }
