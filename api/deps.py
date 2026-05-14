# coding=utf-8
"""
API Dependencies

API Key authentication and settings.
"""

from fastapi import Header, HTTPException, Security, Depends
from typing import Optional

from .config import settings


API_KEYS = {
    "qn_live_xxxxxxxxxxxxxxxxxxxxxxxx": {"name": "opencode", "rate_limit": 1000},
    "qn_live_yyyyyyyyyyyyyyyyyyyyyyyyy": {"name": "openclaw", "rate_limit": 1000},
}


async def get_settings():
    return settings


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> dict:
    """Verify API Key from request headers.

    Supports two formats:
    - X-API-Key: <api_key>
    - Authorization: Bearer <api_key>

    Returns the API key info if valid.

    Raises HTTPException 401 if invalid.
    """
    key = x_api_key

    if not key and authorization:
        if authorization.startswith("Bearer "):
            key = authorization[7:]
        else:
            key = authorization

    if not key:
        if settings.DEBUG:
            return {"name": "debug", "rate_limit": 100}
        raise HTTPException(
            status_code=401,
            detail="Missing API Key. Provide X-API-Key header or Authorization: Bearer <key>"
        )

    if key not in API_KEYS:
        if settings.DEBUG:
            return {"name": "debug", "rate_limit": 100}
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )

    return API_KEYS[key]


async def optional_api_key(
    x_api_key: Optional[str] = Header(None),
) -> Optional[dict]:
    """Optional API key verification (doesn't raise on missing)."""
    if not x_api_key:
        return None

    if x_api_key in API_KEYS:
        return API_KEYS[x_api_key]

    return None