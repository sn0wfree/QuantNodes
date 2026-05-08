from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.strategy_service import strategy_service

router = APIRouter()


class StrategyValidateRequest(BaseModel):
    yaml: str


class StrategyValidateResponse(BaseModel):
    valid: bool
    error: str = None


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
