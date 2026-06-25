"""
Settings Service - User preferences and configuration management
"""

import json
from typing import Optional, Dict, Any
from pathlib import Path

from QuantNodes.constants import DEFAULT_API_PORT, DEFAULT_LLM_MODEL


class SettingsService:
    """Settings service for API layer"""

    def __init__(self, data_dir: str = ".agent"):
        self.data_dir = Path(data_dir)
        self.settings_file = self.data_dir / "settings.json"
        self._settings: Optional[Dict[str, Any]] = None

    async def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        if self._settings is not None:
            return self._settings

        default_settings = {
            "appearance": {
                "theme": "light",
                "language": "en",
                "sidebar_collapsed": False,
                "compact_mode": False,
            },
            "api": {
                "base_url": f"http://localhost:{DEFAULT_API_PORT}",
                "ws_url": f"ws://localhost:{DEFAULT_API_PORT}",
                "timeout": 30000,
            },
            "agent": {
                "provider": "openai",
                "model": DEFAULT_LLM_MODEL,
                "api_key": "",
                "api_base": "",
                "max_iterations": 5,
                "temperature": 0.7,
                "llm_timeout": 60,
                "llm_max_retries": 3,
                "default_mode": "build",
                "mode_models": {
                    "build": {"model": "", "max_tokens": 102400},
                    "plan": {"model": "", "max_tokens": 16000},
                },
            },
            "editor": {
                "font_size": 14,
                "tab_size": 2,
                "word_wrap": True,
                "minimap": True,
                "auto_save": True,
            },
            "backtest": {
                "default_initial_cash": 100000,
                "default_commission": 0.001,
                "auto_save_results": True,
            },
            "notifications": {
                "enabled": True,
                "sound": True,
                "desktop": False,
            },
        }

        if self.settings_file.exists():
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                content = await loop.run_in_executor(
                    None, lambda: self.settings_file.read_text(encoding="utf-8")
                )
                saved = json.loads(content)
                self._settings = self._deep_merge(default_settings, saved)
            except Exception:
                self._settings = default_settings
        else:
            self._settings = default_settings

        return self._settings

    async def _save_settings(self) -> None:
        """Save settings to file"""
        if self._settings is None:
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        import asyncio
        loop = asyncio.get_event_loop()
        content = json.dumps(self._settings, indent=2, ensure_ascii=False)
        await loop.run_in_executor(
            None, lambda: self.settings_file.write_text(content, encoding="utf-8")
        )

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    async def get_settings(self) -> Dict[str, Any]:
        """Get all settings"""
        return await self._load_settings()

    async def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings"""
        current = await self._load_settings()
        self._settings = self._deep_merge(current, settings)
        await self._save_settings()
        return self._settings

    async def get_section(self, section: str) -> Dict[str, Any]:
        """Get settings section"""
        settings = await self._load_settings()
        return settings.get(section, {})

    async def update_section(self, section: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings section"""
        settings = await self._load_settings()
        settings[section] = self._deep_merge(settings.get(section, {}), data)
        self._settings = settings
        await self._save_settings()
        return settings[section]

    async def reset_settings(self) -> Dict[str, Any]:
        """Reset all settings to defaults"""
        self._settings = None
        return await self.get_settings()

    async def export_settings(self) -> str:
        """Export settings as JSON string"""
        settings = await self.get_settings()
        return json.dumps(settings, indent=2, ensure_ascii=False)

    async def import_settings(self, json_str: str) -> Dict[str, Any]:
        """Import settings from JSON string"""
        try:
            imported = json.loads(json_str)
            return await self.update_settings(imported)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    async def get_api_keys(self) -> Dict[str, str]:
        """Get API keys (masked)"""
        settings = await self._load_settings()
        agent = settings.get("agent", {})
        api_key = agent.get("api_key", "")
        
        # Mask API key
        if api_key and len(api_key) > 8:
            masked = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
        else:
            masked = "****" if api_key else ""
        
        return {
            "openai_api_key": masked,
            "has_key": bool(api_key),
        }

    async def update_api_key(self, provider: str, api_key: str) -> Dict[str, Any]:
        """Update API key"""
        settings = await self._load_settings()
        if "agent" not in settings:
            settings["agent"] = {}
        settings["agent"]["api_key"] = api_key
        settings["agent"]["provider"] = provider
        self._settings = settings
        await self._save_settings()
        return {"status": "updated", "provider": provider}

    async def get_providers(self) -> Dict[str, Any]:
        """Get all configured providers"""
        settings = await self._load_settings()
        return settings.get("agent", {}).get("providers", {})

    async def add_provider(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new provider"""
        settings = await self._load_settings()
        if "agent" not in settings:
            settings["agent"] = {}
        if "providers" not in settings["agent"]:
            settings["agent"]["providers"] = {}
        settings["agent"]["providers"][name] = config
        self._settings = settings
        await self._save_settings()
        return settings["agent"]["providers"][name]

    async def update_provider(self, name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing provider"""
        settings = await self._load_settings()
        providers = settings.get("agent", {}).get("providers", {})
        if name not in providers:
            raise ValueError(f"Provider '{name}' not found")
        providers[name] = {**providers[name], **config}
        self._settings = settings
        await self._save_settings()
        return providers[name]

    async def delete_provider(self, name: str) -> bool:
        """Delete a provider"""
        settings = await self._load_settings()
        providers = settings.get("agent", {}).get("providers", {})
        if name not in providers:
            return False
        del providers[name]
        self._settings = settings
        await self._save_settings()
        return True

    async def test_provider(self, name: str) -> Dict[str, Any]:
        """Test provider connectivity by fetching /models"""
        settings = await self._load_settings()
        providers = settings.get("agent", {}).get("providers", {})
        if name not in providers:
            return {"ok": False, "error": f"Provider '{name}' not found"}
        p = providers[name]
        api_base = p.get("api_base", "")
        api_key = p.get("api_key", "")
        if not api_base:
            return {"ok": False, "error": "No api_base configured"}
        try:
            url = f"{api_base.rstrip('/')}/models"
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            import asyncio
            loop = asyncio.get_event_loop()
            import requests as http_requests
            resp = await loop.run_in_executor(
                None, lambda: http_requests.get(url, headers=headers, timeout=15)
            )
            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get("data", []))
                return {"ok": True, "model_count": count}
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def get_provider_models(self) -> Dict[str, Any]:
        """Fetch models from all configured providers"""
        settings = await self._load_settings()
        providers = settings.get("agent", {}).get("providers", {})
        result = {}
        for name, p in providers.items():
            api_base = p.get("api_base", "")
            api_key = p.get("api_key", "")
            if not api_base:
                continue
            try:
                import asyncio
                import requests as http_requests
                loop = asyncio.get_event_loop()
                url = f"{api_base.rstrip('/')}/models"
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                resp = await loop.run_in_executor(
                    None, lambda: http_requests.get(url, headers=headers, timeout=15)
                )
                if resp.status_code == 200:
                    raw = resp.json().get("data", [])
                    result[name] = [
                        {"id": m.get("id", ""), "name": m.get("name", m.get("id", ""))}
                        for m in raw
                    ]
                else:
                    result[name] = []
            except Exception:
                result[name] = []
        return result

    def sync_core_config(self) -> None:
        """Sync settings.json values into QuantNodes.core.config.settings in-memory singleton"""
        if self._settings is None:
            return
        try:
            from QuantNodes.core.config import settings as core_settings
            agent = self._settings.get("agent", {})
            if "api_key" in agent and agent["api_key"]:
                core_settings.llm.api_key = agent["api_key"]
            if "api_base" in agent and agent["api_base"]:
                core_settings.llm.base_url = agent["api_base"]
            if "model" in agent:
                core_settings.llm.model = agent["model"]
            if "llm_timeout" in agent:
                core_settings.llm.timeout = agent["llm_timeout"]
            if "llm_max_retries" in agent:
                core_settings.llm.max_retries = agent["llm_max_retries"]
            if "max_tokens" in agent:
                core_settings.llm.max_tokens = agent["max_tokens"]
            if "providers" in agent:
                core_settings.llm.providers = agent["providers"]
            backtest = self._settings.get("backtest", {})
            if "default_commission" in backtest:
                core_settings.default_commission = backtest["default_commission"]
        except Exception:
            pass


# Singleton instance
settings_service = SettingsService()
