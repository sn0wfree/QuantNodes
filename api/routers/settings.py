from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from ..services.settings_service import settings_service

router = APIRouter()


class SettingsUpdateRequest(BaseModel):
    settings: Dict[str, Any]


class APIKeyUpdateRequest(BaseModel):
    provider: str
    api_key: str


class ImportRequest(BaseModel):
    json_data: str


@router.get("/")
async def get_settings():
    """Get all settings"""
    return await settings_service.get_settings()


@router.put("/")
async def update_settings(request: SettingsUpdateRequest):
    """Update settings"""
    return await settings_service.update_settings(request.settings)


@router.post("/reset")
async def reset_settings():
    """Reset settings to defaults"""
    return await settings_service.reset_settings()


@router.get("/export")
async def export_settings():
    """Export settings as JSON"""
    return await settings_service.export_settings()


@router.post("/import")
async def import_settings(request: ImportRequest):
    """Import settings from JSON"""
    try:
        return await settings_service.import_settings(request.json_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api-keys")
async def get_api_keys():
    """Get API keys (masked)"""
    return await settings_service.get_api_keys()


@router.put("/api-keys")
async def update_api_key(request: APIKeyUpdateRequest):
    """Update API key"""
    return await settings_service.update_api_key(request.provider, request.api_key)


@router.get("/{section}")
async def get_section(section: str):
    """Get settings section"""
    return await settings_service.get_section(section)


@router.put("/{section}")
async def update_section(section: str, data: Dict[str, Any]):
    """Update settings section"""
    return await settings_service.update_section(section, data)
