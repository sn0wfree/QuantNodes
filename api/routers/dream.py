from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from pydantic import BaseModel
from ..services.dream_service import dream_service

router = APIRouter()


class DreamInsight(BaseModel):
    id: str
    title: str
    content: str
    type: str
    category: str
    confidence: float
    created_at: str
    tags: List[str] = []
    insights: List[str] = []
    source: str = ""


class DreamStats(BaseModel):
    total_insights: int
    by_type: dict = {}
    by_category: dict = {}
    avg_confidence: float
    recent_trend: List[dict] = []
    top_tags: List[dict] = []


class DreamGenerateRequest(BaseModel):
    type: str
    content: str
    source: str = ""
    confidence: float = 0.8
    tags: List[str] = []


@router.get("/", response_model=List[DreamInsight])
async def list_dreams(
    limit: int = Query(20, ge=1, le=100),
    type: Optional[str] = Query(None, description="Filter by dream type"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
):
    """Get list of insights"""
    return await dream_service.list_dreams(
        limit=limit,
        dream_type=type,
        min_confidence=min_confidence,
    )


@router.get("/stats", response_model=DreamStats)
async def get_dream_stats():
    """Get insight statistics"""
    return await dream_service.get_stats()


@router.get("/{dream_id}", response_model=DreamInsight)
async def get_dream(dream_id: str):
    """Get insight by ID"""
    dream = await dream_service.get_dream(dream_id)
    if not dream:
        raise HTTPException(status_code=404, detail="Insight not found")
    return dream


@router.post("/", response_model=DreamInsight)
async def generate_dream(request: DreamGenerateRequest):
    """Generate a new insight"""
    result = await dream_service.generate_insight(
        dream_type=request.type,
        content=request.content,
        source=request.source,
        confidence=request.confidence,
        tags=request.tags,
    )
    return result
