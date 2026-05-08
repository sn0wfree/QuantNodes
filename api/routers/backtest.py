from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel
from ..schemas.backtest import BacktestRequest, BacktestResult
from ..services.backtest_service import backtest_service

router = APIRouter()


class BacktestTemplate(BaseModel):
    name: str
    description: str
    yaml: str


@router.post("/run", response_model=BacktestResult)
async def run_backtest(request: BacktestRequest):
    """Run a backtest with given configuration"""
    result = await backtest_service.run_backtest(
        config_yaml=request.config_yaml,
        start_date=request.start_date,
        end_date=request.end_date,
        initial_cash=request.initial_cash,
        data_path=request.data_path,
    )
    
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("errors", ["Unknown error"])[0])
    
    return BacktestResult(**result)


@router.get("/history", response_model=List[BacktestResult])
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """Get backtest history"""
    results = await backtest_service.get_history(limit=limit, offset=offset)
    return [BacktestResult(**r) for r in results]


@router.get("/templates", response_model=List[BacktestTemplate])
async def get_templates():
    """Get backtest templates"""
    return await backtest_service.get_templates()


@router.get("/{backtest_id}", response_model=BacktestResult)
async def get_result(backtest_id: str):
    """Get backtest result by ID"""
    result = await backtest_service.get_result(backtest_id)
    if not result:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return BacktestResult(**result)
