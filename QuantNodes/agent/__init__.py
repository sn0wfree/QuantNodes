# coding=utf-8
"""
QuantNodes Agent 系统

基于nanobot架构的量化研究智能体。

Usage:
    from QuantNodes.agent import Agent
    
    agent = Agent(workspace="./workspace", config={"model": "gpt-4o"})
    response = await agent.run("帮我生成一个动量策略")
"""

__version__ = "0.1.0"


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
        )

    def _create_provider(self, config: dict):
        """根据配置创建 LLM Provider"""
        try:
            from .providers.quantnodes import QuantNodesLLMProvider
            from QuantNodes.ai.llm.openai import OpenAIClient, AzureOpenAIClient

            provider_type = config.get("provider", "openai")
            api_key = config.get("api_key")
            api_base = config.get("api_base")
            model = config.get("model", "gpt-4o")
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
                # openai, anthropic (via compatible proxy), local (ollama), custom
                base_url = api_base or None
                # Normalize: OpenAIClient appends /chat/completions internally
                if base_url:
                    base_url = base_url.rstrip("/")
                    for suffix in ("/chat/completions", "/v1/chat/completions", "/v1"):
                        if base_url.endswith(suffix):
                            base_url = base_url[: -len(suffix)]
                            break
                client = OpenAIClient(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                    max_retries=max_retries,
                )

            return QuantNodesLLMProvider(
                client,
                default_model=model,
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

    async def chat(self, message: str, session_id: str = "default"):
        """流式对话（生成器）
        
        Args:
            message: 用户输入
            session_id: 会话ID
            
        Yields:
            回复片段
        """
        if self._loop.provider is None:
            yield "Error: LLM provider not configured. Set QUANTNODES__LLM__API_KEY in .env"
            return
        result = await self._loop.chat(message, session_id=session_id)
        yield result


from .core.loop import AgentLoop
from .core.memory import MemoryStore
from .core.autocompact import truncate_history, microcompact

__all__ = [
    "__version__",
    "Agent",
    "AgentLoop",
    "MemoryStore",
    "truncate_history",
    "microcompact",
]
