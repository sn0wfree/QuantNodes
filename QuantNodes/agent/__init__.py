# coding=utf-8
"""
QuantNodes Agent 系统

基于nanobot架构的量化研究智能体。

Usage:
    from QuantNodes.agent import Agent
    
    agent = Agent(workspace="./workspace", config={"model": "gpt-4o"})
    response = await agent.run("帮我生成一个动量策略")
"""

__version__ = "2.5.0"


class Agent:
    """QuantNodes 量化研究Agent
    
    Doc 14 规定的对外API门面。
    内部组合 AgentLoop + ToolRegistry + LLMProvider。
    
    Examples:
        >>> agent = Agent(workspace="./workspace", config={"model": "gpt-4o"})
        >>> response = await agent.run("生成一个动量因子策略")
    """

    def __init__(self, workspace: str, config: dict = None):
        """初始化Agent
        
        Args:
            workspace: 工作目录路径
            config: 配置字典
                - model: LLM模型名称
                - api_key: API密钥
                - api_base: API基础URL
        """
        from pathlib import Path
        from .core.loop import AgentLoop
        from .bus.queue import MessageBus
        from .tools.registry import ToolRegistry
        from .tools.echo import EchoTool
        from .tools.sandbox import SandboxTool
        from .tools.pipeline import PipelineTool
        from .tools.strategy import StrategyTool
        from .tools.backtest import BacktestTool
        from .tools.factor import FactorTool
        from .tools.config_backtest import ConfigBacktestTool
        from .tools.wiki import WikiTool
        from .tools.file_ops import FileOpsTool
        from .tools.code_search import CodeSearchTool
        from .tools.git_ops import GitOpsTool
        from .tools.web_fetch import WebFetchTool
        from .tools.web_search import WebSearchTool
        from .tools.task import TaskTool
        from .skills.registry import SkillRegistry
        from .skills.loader import SkillLoader
        from .skills.bridge import SkillToolBridge

        config = config or {}
        workspace_path = Path(workspace)
        workspace_path.mkdir(parents=True, exist_ok=True)

        self._max_tokens = config.get("max_tokens", 102400)

        # Build/Plan dual mode support
        self._mode_models = config.get("mode_models", {})
        self._default_mode = config.get("default_mode", "build")
        # Fallback: if mode_models is empty, derive from single model field
        if not self._mode_models:
            model = config.get("model", "")
            self._mode_models = {
                "build": {"model": model, "max_tokens": self._max_tokens},
                "plan": {"model": model, "max_tokens": 16000},
            }

        bus = MessageBus()
        tool_registry = ToolRegistry()

        tool_registry.register(EchoTool())
        tool_registry.register(SandboxTool())
        tool_registry.register(PipelineTool())
        tool_registry.register(StrategyTool())
        tool_registry.register(BacktestTool())
        tool_registry.register(FactorTool())
        tool_registry.register(ConfigBacktestTool())
        tool_registry.register(WikiTool(wiki_path=str(workspace_path / "wiki")))
        tool_registry.register(FileOpsTool(workspace=workspace_path))
        tool_registry.register(CodeSearchTool(workspace=workspace_path))
        tool_registry.register(GitOpsTool(workspace=workspace_path))
        tool_registry.register(WebFetchTool())
        tool_registry.register(WebSearchTool())
        tool_registry.register(TaskTool(workspace=workspace_path))

        skill_registry = SkillRegistry()
        self._skill_loader = SkillLoader(skill_registry)
        self._skill_bridge = SkillToolBridge(skill_registry, tool_registry)

        provider = self._create_provider(config)

        self._loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=workspace_path,
            tool_registry=tool_registry,
            model=config.get("model"),
            max_tokens=self._max_tokens,
            mode_models=self._mode_models,
        )

    def _create_provider(self, config: dict):
        """根据配置创建 LLM Provider

        支持两种模式：
        1. 多Provider模式：config中包含providers字典，使用ProviderRegistry动态路由
        2. 单Provider模式（向后兼容）：仅api_key + api_base，绑定单个client
        """
        try:
            from .providers.quantnodes import QuantNodesLLMProvider
            from .providers.registry import ProviderRegistry

            providers_data = config.get("providers", {})
            model = config.get("model", "gpt-4o")
            max_tokens = config.get("max_tokens", 102400)
            fallback = config.get("fallback_providers", [])

            if providers_data:
                # 多Provider模式：使用ProviderRegistry
                registry = ProviderRegistry.from_settings(config)
                return QuantNodesLLMProvider(
                    registry=registry,
                    default_model=model,
                    default_max_tokens=max_tokens,
                    fallback_providers=fallback,
                )

            # 单Provider模式（向后兼容）
            from QuantNodes.ai.llm.openai import OpenAIClient, AzureOpenAIClient

            provider_type = config.get("provider", "openai")
            api_key = config.get("api_key")
            api_base = config.get("api_base")
            use_litellm = config.get("use_litellm", True)
            rate_limit_rps = config.get("rate_limit_rps", 0.5)
            timeout = config.get("llm_timeout", 60)
            max_retries = config.get("llm_max_retries", 3)

            if provider_type == "azure":
                client = AzureOpenAIClient(
                    api_key=api_key,
                    azure_endpoint=api_base,
                    timeout=timeout,
                    max_retries=max_retries,
                )
            else:
                base_url = api_base or None
                client = OpenAIClient(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    max_retries=max_retries,
                )

            litellm_base_url = api_base

            return QuantNodesLLMProvider(
                api_key=api_key,
                api_base=litellm_base_url,
                client=client,
                default_model=model,
                default_max_tokens=max_tokens,
                use_litellm=use_litellm,
                rate_limit_rps=rate_limit_rps,
                max_retries=max_retries,
                timeout=timeout,
            )
        except (ImportError, Exception) as e:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to create LLM provider: %s. "
                "Chat will not be available until configured.", e
            )
            return None

    @property
    def loop(self):
        """获取底层 AgentLoop"""
        return self._loop

    async def run(self, prompt: str, session_id: str = "default") -> str:
        """运行一次对话
        
        Args:
            prompt: 用户输入
            session_id: 会话ID
            
        Returns:
            Agent回复
        """
        return await self._loop.chat(prompt, session_id=session_id)

    async def chat(self, message: str, session_id: str = "default", model: str | None = None, max_tokens: int | None = None, mode: str | None = None):
        """流式对话（生成器）
        
        Args:
            message: 用户输入
            session_id: 会话ID
            model: 可选，覆盖本次对话使用的模型
            max_tokens: 可选，覆盖本次对话的最大token数
            mode: 可选，'build' 或 'plan'，从 mode_models 中解析模型
            
        Yields:
            dict: 事件字典
                - {"type": "token", "content": str} - 流式文本token
                - {"type": "tool_call", "id": str, "name": str, "arguments": dict}
                - {"type": "tool_result", "id": str, "name": str, "content": str, "success": bool}
                - {"type": "done", "content": str, "tools_used": list, "stop_reason": str}
                - {"type": "error", "content": str}
        """
        if self._loop.provider is None:
            yield {"type": "error", "content": "LLM provider not configured. Set QUANTNODES__LLM__API_KEY in .env"}
            return

        # Resolve model from mode if provided
        resolved_model = model
        resolved_max_tokens = max_tokens or getattr(self, '_max_tokens', 102400)
        mode_models = getattr(self, '_mode_models', {})
        if mode and mode in mode_models:
            mode_config = mode_models[mode]
            if not model:
                resolved_model = mode_config.get("model") or model
            if not max_tokens:
                resolved_max_tokens = mode_config.get("max_tokens", resolved_max_tokens)

        async for event in self._loop.chat_stream(message, session_id=session_id, model=resolved_model, max_tokens=resolved_max_tokens):
            yield event


from .core.loop import AgentLoop
from .core.memory import MemoryStore, MemoryManager, DreamStore
from .core.dream import DreamEngine
from .core.autocompact import truncate_history, microcompact

__all__ = [
    "__version__",
    "Agent",
    "AgentLoop",
    "MemoryStore",
    "MemoryManager",
    "DreamStore",
    "DreamEngine",
    "truncate_history",
    "microcompact",
]
