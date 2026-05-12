# coding=utf-8
"""
Provider 注册表

管理多 LLM Provider 配置，按 model 名路由到正确的 provider。
支持优先级排序和 fallback 机制。
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """单个 Provider 配置"""
    name: str
    api_key: str
    api_base: str
    models: List[str] = field(default_factory=list)
    extra_headers: Dict[str, str] = field(default_factory=dict)
    priority: int = 1
    timeout: int = 60
    max_retries: int = 3


class ProviderRegistry:
    """Provider 注册表 — 管理多 Provider 配置，按 model 路由"""

    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}
        self._default_provider: Optional[str] = None

    @classmethod
    def from_settings(cls, agent_config: dict) -> "ProviderRegistry":
        """从 settings.json 的 agent 节点加载

        支持两种模式：
        1. 新模式：agent.providers 字典配置多 provider
        2. 旧模式：agent.api_key + agent.api_base 单 provider（向后兼容）
        """
        registry = cls()
        providers_data = agent_config.get("providers", {})

        for name, pconfig in providers_data.items():
            registry.register(ProviderConfig(
                name=name,
                api_key=pconfig.get("api_key", ""),
                api_base=pconfig.get("api_base", ""),
                models=pconfig.get("models", []),
                extra_headers=pconfig.get("extra_headers", {}),
                priority=pconfig.get("priority", 1),
                timeout=pconfig.get("timeout", 60),
                max_retries=pconfig.get("max_retries", 3),
            ))

        registry._default_provider = agent_config.get("provider")

        # 向后兼容：无 providers 配置时，用顶层字段创建单 provider
        if not providers_data and agent_config.get("api_key"):
            default_name = registry._default_provider or "default"
            model = agent_config.get("model", "")
            registry.register(ProviderConfig(
                name=default_name,
                api_key=agent_config.get("api_key", ""),
                api_base=agent_config.get("api_base", ""),
                models=[model] if model else [],
                priority=1,
                timeout=agent_config.get("llm_timeout", 60),
                max_retries=agent_config.get("llm_max_retries", 3),
            ))
            registry._default_provider = default_name

        return registry

    def register(self, config: ProviderConfig) -> None:
        """注册一个 Provider"""
        self._providers[config.name] = config

    def get(self, name: str) -> Optional[ProviderConfig]:
        """获取指定名称的 Provider"""
        return self._providers.get(name)

    def resolve(self, model: Optional[str] = None) -> Optional[ProviderConfig]:
        """根据 model 名找到最优 provider

        路由逻辑：
        1. model=None → 返回默认 provider
        2. 默认 provider 的 models 中匹配 → 优先返回
        3. 所有 provider 中匹配 → 按 priority 排序（数字小优先）
        4. 都找不到 → 返回默认 provider 兜底
        """
        if not self._providers:
            return None

        # 1. model=None → 默认 provider
        if model is None:
            return self._get_default_or_first()

        # 2. 默认 provider 优先
        default = self._providers.get(self._default_provider)
        if default and model in default.models:
            return default

        # 3. 所有 provider 中按 priority 排序
        candidates = [
            p for p in self._providers.values()
            if model in p.models
        ]
        if candidates:
            candidates.sort(key=lambda p: p.priority)
            return candidates[0]

        # 4. 兜底：返回默认 provider
        return self._get_default_or_first()

    def _get_default_or_first(self) -> Optional[ProviderConfig]:
        """获取默认 provider，无默认则返回第一个"""
        default = self._providers.get(self._default_provider)
        if default:
            return default
        if self._providers:
            return next(iter(self._providers.values()))
        return None

    def list_providers(self) -> List[ProviderConfig]:
        """列出所有已注册的 Provider"""
        return list(self._providers.values())

    def get_models_map(self) -> Dict[str, List[str]]:
        """获取 provider → models 映射"""
        return {
            name: config.models
            for name, config in self._providers.items()
        }

    def get_client(self, config: ProviderConfig):
        """为指定 provider 创建 OpenAIClient"""
        from QuantNodes.ai.llm.openai import OpenAIClient
        return OpenAIClient(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout,
            max_retries=config.max_retries,
            extra_headers=config.extra_headers,
        )

    def get_default_client(self):
        """获取默认 provider 的 client"""
        config = self._get_default_or_first()
        if config:
            return self.get_client(config)
        return None

    @property
    def default_provider_name(self) -> Optional[str]:
        return self._default_provider

    def __repr__(self) -> str:
        providers_str = ", ".join(self._providers.keys())
        return f"ProviderRegistry(default={self._default_provider}, providers=[{providers_str}])"
