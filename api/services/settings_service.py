"""
Settings Service - User preferences and configuration management
"""

import json
from typing import Optional, Dict, Any
from pathlib import Path


class SettingsService:
    """Settings service for API layer"""

    def __init__(self, data_dir: str = ".quant_agent"):
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
                "base_url": "http://localhost:8000",
                "ws_url": "ws://localhost:8000",
                "timeout": 30000,
            },
            "agent": {
                "provider": "openai",
                "model": "gpt-4",
                "api_key": "",
                "api_base": "",
                "max_iterations": 5,
                "temperature": 0.7,
                "llm_timeout": 60,
                "llm_max_retries": 3,
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
