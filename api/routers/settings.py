from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import requests as http_requests
from ..services.settings_service import settings_service
from ..services.agent_service import agent_service

MAINSTREAM_PAID_PROVIDERS = {"qwen", "anthropic"}

router = APIRouter()


class SettingsUpdateRequest(BaseModel):
    settings: Dict[str, Any]


class APIKeyUpdateRequest(BaseModel):
    provider: str
    api_key: str


class ImportRequest(BaseModel):
    json_data: str


class ProviderCreateRequest(BaseModel):
    name: str
    config: Dict[str, Any]


class ProviderUpdateRequest(BaseModel):
    config: Dict[str, Any]


@router.get("")
async def get_settings():
    """Get all settings"""
    return await settings_service.get_settings()


@router.put("")
async def update_settings(request: SettingsUpdateRequest):
    """Update settings"""
    result = await settings_service.update_settings(request.settings)
    settings_service.sync_core_config()
    agent_service.reload_agent()
    return result


@router.post("/reset")
async def reset_settings():
    """Reset settings to defaults"""
    result = await settings_service.reset_settings()
    settings_service.sync_core_config()
    agent_service.reload_agent()
    return result


@router.get("/export")
async def export_settings():
    """Export settings as JSON"""
    return await settings_service.export_settings()


@router.post("/import")
async def import_settings(request: ImportRequest):
    """Import settings from JSON"""
    try:
        result = await settings_service.import_settings(request.json_data)
        settings_service.sync_core_config()
        agent_service.reload_agent()
        return result
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


def _normalize_base_url(api_base: str) -> str:
    """Extract base URL from api_base, stripping path suffixes."""
    url = api_base.rstrip("/")
    for suffix in ("/chat/completions", "/v1/chat/completions", "/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            break
    return url


def _fetch_models_from_provider(api_base: str, api_key: str) -> List[Dict[str, Any]]:
    """Call provider's /models endpoint and return raw model list."""
    base = _normalize_base_url(api_base)
    url = f"{base}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = http_requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json().get("data", [])


def _filter_and_format_models(raw_models: list) -> List[Dict[str, Any]]:
    """Filter to free + mainstream paid, format for frontend."""
    result = []
    for m in raw_models:
        model_id = m.get("id", "")
        provider = model_id.split("/")[0] if "/" in model_id else ""
        pricing = m.get("pricing", {})
        prompt_price = float(pricing.get("prompt", "1") or "1")

        is_free = prompt_price == 0 or model_id.endswith(":free")
        is_mainstream_paid = provider in MAINSTREAM_PAID_PROVIDERS and not is_free

        if not is_free and not is_mainstream_paid:
            continue

        tags = []
        if is_free:
            tags.append("free")
        supported = m.get("supported_parameters", [])
        if "tools" in supported:
            tags.append("tools")

        ctx = m.get("context_length", 0)
        result.append({
            "id": model_id,
            "name": m.get("name", model_id),
            "provider": provider,
            "contextWindow": ctx,
            "priceIn": prompt_price * 1_000_000,
            "priceOut": float(pricing.get("completion", "0") or "0") * 1_000_000,
            "tags": tags,
            "modality": m.get("architecture", {}).get("modality", "text->text"),
        })

    result.sort(key=lambda x: (not x["tags"].__contains__("free"), x["provider"], x["name"]))
    return result


FALLBACK_MODELS: List[Dict[str, Any]] = [
    {
        "id": "minimax/minimax-m2.5:free",
        "name": "MiniMax M2.5 (Free)",
        "provider": "MiniMax",
        "contextWindow": 1000000,
        "priceIn": 0,
        "priceOut": 0,
        "tags": ["free"],
        "modality": "text->text",
    },
    {
        "id": "qwen/qwen3-235b-a22b:free",
        "name": "Qwen3 235B A22B (Free)",
        "provider": "qwen",
        "contextWindow": 131072,
        "priceIn": 0,
        "priceOut": 0,
        "tags": ["free", "tools"],
        "modality": "text->text",
    },
    {
        "id": "anthropic/claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "provider": "anthropic",
        "contextWindow": 200000,
        "priceIn": 3.0,
        "priceOut": 15.0,
        "tags": ["tools"],
        "modality": "text->text",
    },
]


@router.get("/models")
async def list_available_models():
    """Fetch available models from the configured API provider."""
    settings = await settings_service.get_settings()
    agent_cfg = settings.get("agent", {})
    api_base = agent_cfg.get("api_base", "")
    api_key = agent_cfg.get("api_key", "")

    if not api_base:
        return {"models": FALLBACK_MODELS, "source": "fallback", "cached": False}

    try:
        raw = await _fetch_models_from_provider_async(api_base, api_key)
        models = _filter_and_format_models(raw)
        if not models:
            return {"models": FALLBACK_MODELS, "source": "fallback", "cached": False}
        return {"models": models, "source": "provider", "cached": False}
    except Exception:
        return {"models": FALLBACK_MODELS, "source": "fallback", "cached": False}


async def _fetch_models_from_provider_async(api_base: str, api_key: str) -> list:
    """Async wrapper for _fetch_models_from_provider."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _fetch_models_from_provider(api_base, api_key))


# ── Provider CRUD Endpoints ──────────────────────────────────────────


@router.get("/providers")
async def get_providers():
    """Get all configured providers"""
    return await settings_service.get_providers()


@router.post("/providers")
async def add_provider(request: ProviderCreateRequest):
    """Add a new provider"""
    result = await settings_service.add_provider(request.name, request.config)
    settings_service.sync_core_config()
    agent_service.reload_agent()
    return {"name": request.name, **result}


@router.put("/providers/{name}")
async def update_provider(name: str, request: ProviderUpdateRequest):
    """Update an existing provider"""
    try:
        result = await settings_service.update_provider(name, request.config)
        settings_service.sync_core_config()
        agent_service.reload_agent()
        return {"name": name, **result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/providers/{name}")
async def delete_provider(name: str):
    """Delete a provider"""
    deleted = await settings_service.delete_provider(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")
    settings_service.sync_core_config()
    agent_service.reload_agent()
    return {"status": "deleted", "name": name}


@router.post("/providers/{name}/test")
async def test_provider(name: str):
    """Test provider connectivity"""
    return await settings_service.test_provider(name)


@router.get("/providers/models/all")
async def get_all_provider_models():
    """Fetch models from all configured providers"""
    return await settings_service.get_provider_models()


@router.get("/{section}")
async def get_section(section: str):
    """Get settings section"""
    return await settings_service.get_section(section)


@router.put("/{section}")
async def update_section(section: str, data: Dict[str, Any]):
    """Update settings section"""
    return await settings_service.update_section(section, data)
