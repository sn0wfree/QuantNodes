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
            from QuantNodes.ai.llm.openai import OpenAIClient

            client = OpenAIClient(
                api_key=config.get("api_key"),
                api_base=config.get("api_base"),
            )
            return QuantNodesLLMProvider(
                client,
                default_model=config.get("model", "gpt-4o"),
            )
        except (ImportError, Exception) as e:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to create LLM provider: %s. "
                "Tools requiring LLM will not be available.", e
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
