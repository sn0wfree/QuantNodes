# coding=utf-8
"""
测试 Provider 注册表

覆盖：配置加载、model路由、优先级排序、fallback、向后兼容。
"""

from unittest.mock import Mock

from QuantNodes.agent.providers.registry import ProviderConfig, ProviderRegistry


class TestProviderConfig:
    def test_defaults(self):
        config = ProviderConfig(name="test", api_key="k", api_base="http://x")
        assert config.models == []
        assert config.extra_headers == {}
        assert config.priority == 1
        assert config.timeout == 60
        assert config.max_retries == 3

    def test_custom_values(self):
        config = ProviderConfig(
            name="deepseek",
            api_key="sk-xxx",
            api_base="https://api.deepseek.com/v1",
            models=["deepseek-chat", "deepseek-reasoner"],
            extra_headers={"X-Custom": "val"},
            priority=2,
            timeout=30,
            max_retries=5,
        )
        assert config.models == ["deepseek-chat", "deepseek-reasoner"]
        assert config.extra_headers == {"X-Custom": "val"}
        assert config.priority == 2


class TestProviderRegistryFromSettings:
    def test_legacy_single_provider(self):
        """无 providers 字段时，回退到单 provider 模式"""
        agent_config = {
            "provider": "openai",
            "api_key": "sk-test",
            "api_base": "https://api.openai.com/v1",
            "model": "gpt-4",
        }
        registry = ProviderRegistry.from_settings(agent_config)

        assert registry.default_provider_name == "openai"
        assert len(registry.list_providers()) == 1
        p = registry.list_providers()[0]
        assert p.name == "openai"
        assert p.api_key == "sk-test"
        assert p.api_base == "https://api.openai.com/v1"
        assert p.models == ["gpt-4"]

    def test_legacy_no_api_key(self):
        """无 api_key 且无 providers 时，注册表为空"""
        agent_config = {"provider": "openai", "model": "gpt-4"}
        registry = ProviderRegistry.from_settings(agent_config)
        assert len(registry.list_providers()) == 0

    def test_multi_providers(self):
        """多 provider 配置"""
        agent_config = {
            "provider": "dashscope",
            "providers": {
                "deepseek": {
                    "api_key": "sk-ds",
                    "api_base": "https://api.deepseek.com/v1",
                    "models": ["deepseek-chat"],
                    "priority": 1,
                },
                "dashscope": {
                    "api_key": "sk-ds2",
                    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "models": ["deepseek-v4-pro", "qwen3.6-plus"],
                    "priority": 1,
                },
                "openrouter": {
                    "api_key": "sk-or",
                    "api_base": "https://openrouter.ai/api/v1",
                    "models": ["baidu/cobuddy:free"],
                    "extra_headers": {"X-OpenRouter-Title": "QN"},
                    "priority": 3,
                },
            },
        }
        registry = ProviderRegistry.from_settings(agent_config)

        assert registry.default_provider_name == "dashscope"
        assert len(registry.list_providers()) == 3

        p_deepseek = registry.get("deepseek")
        assert p_deepseek.api_key == "sk-ds"
        assert p_deepseek.priority == 1

        p_or = registry.get("openrouter")
        assert p_or.extra_headers == {"X-OpenRouter-Title": "QN"}
        assert p_or.priority == 3


class TestProviderRegistryResolve:
    def _make_registry(self):
        registry = ProviderRegistry()
        registry.register(ProviderConfig(
            name="default", api_key="k1", api_base="http://a",
            models=["gpt-4", "gpt-4o"], priority=1,
        ))
        registry.register(ProviderConfig(
            name="deepseek", api_key="k2", api_base="http://b",
            models=["deepseek-chat", "deepseek-reasoner"], priority=1,
        ))
        registry.register(ProviderConfig(
            name="openrouter", api_key="k3", api_base="http://c",
            models=["gpt-4", "deepseek-chat"], priority=3,
        ))
        registry._default_provider = "default"
        return registry

    def test_resolve_none_returns_default(self):
        registry = self._make_registry()
        result = registry.resolve(None)
        assert result.name == "default"

    def test_resolve_default_provider_preferred(self):
        """model 在默认 provider 中匹配时，优先返回默认 provider"""
        registry = self._make_registry()
        result = registry.resolve("gpt-4")
        assert result.name == "default"

    def test_resolve_non_default_provider(self):
        """model 仅在非默认 provider 中匹配"""
        registry = self._make_registry()
        result = registry.resolve("deepseek-chat")
        # deepseek 和 openrouter 都有 deepseek-chat，deepseek priority=1 更优
        assert result.name == "deepseek"

    def test_resolve_priority_sorting(self):
        """同 model 多 provider 时按 priority 排序"""
        registry = ProviderRegistry()
        registry.register(ProviderConfig(
            name="slow", api_key="k1", api_base="http://a",
            models=["m1"], priority=5,
        ))
        registry.register(ProviderConfig(
            name="fast", api_key="k2", api_base="http://b",
            models=["m1"], priority=1,
        ))
        registry.register(ProviderConfig(
            name="medium", api_key="k3", api_base="http://c",
            models=["m1"], priority=3,
        ))
        registry._default_provider = "slow"

        result = registry.resolve("m1")
        # 默认 provider slow 有 m1，优先返回
        assert result.name == "slow"

    def test_resolve_unknown_model_falls_back(self):
        """未知 model 返回默认 provider"""
        registry = self._make_registry()
        result = registry.resolve("unknown-model")
        assert result.name == "default"

    def test_resolve_empty_registry(self):
        registry = ProviderRegistry()
        assert registry.resolve("anything") is None


class TestProviderRegistryClient:
    def test_get_client(self):
        registry = ProviderRegistry()
        config = ProviderConfig(
            name="test", api_key="sk-123", api_base="http://localhost/v1",
            extra_headers={"X-Test": "val"}, timeout=30, max_retries=5,
        )
        registry.register(config)

        client = registry.get_client(config)
        assert client.api_key == "sk-123"
        assert client.base_url == "http://localhost/v1"
        assert client.extra_headers == {"X-Test": "val"}
        assert client.timeout == 30
        assert client.max_retries == 5

    def test_get_default_client(self):
        registry = ProviderRegistry()
        registry.register(ProviderConfig(
            name="default", api_key="k", api_base="http://x", models=["m"],
        ))
        registry._default_provider = "default"

        client = registry.get_default_client()
        assert client is not None
        assert client.api_key == "k"

    def test_get_default_client_empty(self):
        registry = ProviderRegistry()
        assert registry.get_default_client() is None


class TestProviderRegistryQuery:
    def test_list_providers(self):
        registry = ProviderRegistry()
        registry.register(ProviderConfig(name="a", api_key="k", api_base="http://a"))
        registry.register(ProviderConfig(name="b", api_key="k", api_base="http://b"))
        assert len(registry.list_providers()) == 2

    def test_get_models_map(self):
        registry = ProviderRegistry()
        registry.register(ProviderConfig(
            name="ds", api_key="k", api_base="http://a",
            models=["deepseek-chat", "deepseek-reasoner"],
        ))
        registry.register(ProviderConfig(
            name="qwen", api_key="k", api_base="http://b",
            models=["qwen-plus"],
        ))
        m = registry.get_models_map()
        assert m["ds"] == ["deepseek-chat", "deepseek-reasoner"]
        assert m["qwen"] == ["qwen-plus"]


class TestQuantNodesLLMProviderWithRegistry:
    """测试 QuantNodesLLMProvider 使用 ProviderRegistry"""

    def test_get_client_for_model_old_mode(self):
        """旧模式：无 registry，返回绑定的 client"""
        from QuantNodes.agent.providers.quantnodes import QuantNodesLLMProvider

        mock_client = Mock()
        provider = QuantNodesLLMProvider(client=mock_client, default_model="gpt-4")
        client, model = provider._get_client_for_model(None)
        assert client == mock_client
        assert model == "gpt-4"

    def test_get_client_for_model_new_mode(self):
        """新模式：有 registry，按 model 路由"""
        from QuantNodes.agent.providers.quantnodes import QuantNodesLLMProvider

        registry = ProviderRegistry()
        registry.register(ProviderConfig(
            name="ds", api_key="sk-ds", api_base="http://ds",
            models=["deepseek-chat"],
        ))
        registry.register(ProviderConfig(
            name="qwen", api_key="sk-qw", api_base="http://qw",
            models=["qwen-plus"],
        ))
        registry._default_provider = "ds"

        provider = QuantNodesLLMProvider(registry=registry, default_model="gpt-4")

        client, model = provider._get_client_for_model("qwen-plus")
        assert client.api_key == "sk-qw"
        assert model == "qwen-plus"

    def test_get_client_for_model_fallback_to_default(self):
        """model 不在任何 provider 中时，返回默认 provider"""
        from QuantNodes.agent.providers.quantnodes import QuantNodesLLMProvider

        registry = ProviderRegistry()
        registry.register(ProviderConfig(
            name="ds", api_key="sk-ds", api_base="http://ds",
            models=["deepseek-chat"],
        ))
        registry._default_provider = "ds"

        provider = QuantNodesLLMProvider(registry=registry, default_model="gpt-4")

        client, model = provider._get_client_for_model("unknown-model")
        assert client.api_key == "sk-ds"
        assert model == "unknown-model"
