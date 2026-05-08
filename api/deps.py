from fastapi import Depends, HTTPException
from .config import settings


async def get_settings():
    return settings


async def verify_api_key():
    # Placeholder for API key verification
    pass
