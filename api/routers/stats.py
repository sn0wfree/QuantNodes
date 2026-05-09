from fastapi import APIRouter
from ..services.stats_service import stats_service

router = APIRouter()


@router.get("")
async def get_stats():
    return await stats_service.get_stats()


@router.get("/activity")
async def get_activity(limit: int = 10):
    return await stats_service.get_recent_activity(limit=limit)
