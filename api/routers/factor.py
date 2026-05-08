from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from ..services.factor_service import factor_service

router = APIRouter()


class FactorAnalyzeRequest(BaseModel):
    expression: str = ""
    universe: str = "hs300"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@router.post("/analyze")
async def analyze_factor(request: FactorAnalyzeRequest):
    return await factor_service.analyze(
        expression=request.expression,
        universe=request.universe,
        start_date=request.start_date,
        end_date=request.end_date,
    )


@router.get("/{factor_name}/metrics")
async def get_factor_metrics(factor_name: str):
    return await factor_service.get_metrics(factor_name)
