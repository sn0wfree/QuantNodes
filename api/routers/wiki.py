from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from ..schemas.wiki import FactorInfo, StrategyInfo
from ..services.wiki_service import wiki_service

router = APIRouter()


@router.get("/factors", response_model=List[FactorInfo])
async def get_factors(
    category: Optional[str] = Query(None, description="Filter by category"),
    source: Optional[str] = Query(None, description="Filter by source"),
    sort: str = Query("updated", description="Sort field"),
    limit: int = Query(50, ge=1, le=500),
):
    """Get list of factors with optional filtering"""
    factors = await wiki_service.get_factors(
        category=category,
        source=source,
        sort=sort,
        limit=limit,
    )
    return factors


@router.get("/factors/search")
async def search_factors(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search factors"""
    results = await wiki_service.search(query=q, type="factor", limit=limit)
    return results


@router.get("/factors/{name}")
async def get_factor(name: str):
    """Get factor details by name"""
    factor = await wiki_service.get_factor(name)
    if not factor:
        raise HTTPException(status_code=404, detail=f"Factor '{name}' not found")
    return factor


@router.post("/factors", status_code=201)
async def create_factor(factor: FactorInfo):
    """Create a new factor"""
    result = await wiki_service.create_factor(factor.model_dump(exclude_unset=True))
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.put("/factors/{name}")
async def update_factor(name: str, factor: FactorInfo):
    """Update an existing factor"""
    result = await wiki_service.update_factor(name, factor.model_dump(exclude_unset=True))
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.delete("/factors/{name}")
async def delete_factor(name: str):
    """Delete a factor"""
    result = await wiki_service.delete_factor(name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/strategies", response_model=List[StrategyInfo])
async def get_strategies(
    category: Optional[str] = Query(None, description="Filter by category"),
    sort: str = Query("updated", description="Sort field"),
    limit: int = Query(50, ge=1, le=500),
):
    """Get list of strategies with optional filtering"""
    strategies = await wiki_service.get_strategies(
        category=category,
        sort=sort,
        limit=limit,
    )
    return strategies


@router.get("/strategies/search")
async def search_strategies(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search strategies"""
    results = await wiki_service.search(query=q, type="strategy", limit=limit)
    return results


@router.get("/strategies/{name}")
async def get_strategy(name: str):
    """Get strategy details by name"""
    strategy = await wiki_service.get_strategy(name)
    if not strategy:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")
    return strategy


@router.post("/strategies", status_code=201)
async def create_strategy(strategy: StrategyInfo):
    """Create a new strategy"""
    result = await wiki_service.create_strategy(strategy.model_dump(exclude_unset=True))
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/search")
async def search_wiki(
    q: str = Query(..., description="Search query"),
    type: str = Query("all", description="Search type: all, factor, strategy"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search across all wiki content"""
    results = await wiki_service.search(query=q, type=type, limit=limit)
    return results


@router.get("/status")
async def get_wiki_status():
    """Get wiki system status"""
    return await wiki_service.get_status()
